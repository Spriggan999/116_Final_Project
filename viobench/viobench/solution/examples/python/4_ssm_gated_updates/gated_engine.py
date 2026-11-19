"""Gated training, adapted from the existing supervised student orchestrator."""
import argparse
from contextlib import ExitStack,contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
from shutil import copyfile

import numpy as np
import torch
from training_policy import apply_learning_rate, read_lr_schedule, physical_selection_score, load_model_weights

import legacy
from gated_training import ground_truth_targets, validation_inputs, train_student, validate_student
from gated_recovery import save_latest, restore_latest, embedded_recovery
from distill import curriculum_window, distillation_batch_size
from gated_curriculum import DEFAULT_GATED_WINDOWS, HORIZON_POLICY, VALIDATION_POLICY, accuracy_targets
HERE=Path(__file__).resolve().parent


@contextmanager
def training_batches(pipeline,data,length,active,prepare,update,batch_limits,tuned,
                     tracking,run,out,auto=True,curriculum=None,epoch=0):
    """Recover memory pressure before yielding the first production update.

    Probes restore optimizer/model state. Once any real update can have run,
    errors propagate rather than replaying a partially updated minibatch.
    """
    from distillation_pipeline import ScheduleMemoryError
    from itertools import chain
    import gc
    key=(length,tuple(active))
    while True:
        loader=data.ppo_loader(length,batch_size=batch_limits.get(length))
        if curriculum is not None:loader=curriculum.training_loader(loader,data.loader_options,epoch)
        if pipeline.device.type=='cuda' and loader.num_workers:loader.multiprocessing_context='spawn'
        started=False;prepared=None
        try:
            with pipeline.batches(loader,prepare) as batches:
                prepared=next(batches)
                if auto and pipeline.parallel and (key not in tuned or getattr(pipeline,'tuning_key',None)!=key):
                    tracking.phase('tuning_schedule')
                    result=pipeline.autotune(active,prepared,update)
                    result.update(window=length,batch_size=loader.batch_size,target_prefetch=pipeline.target_prefetch)
                    (out/'scheduling.json').write_text(json.dumps(pipeline.tuning,indent=2)+'\n')
                    print('Selected distillation schedule:',json.dumps(result),flush=True)
                    namespace=getattr(tracking,'namespace','distillation')
                    run.summary[namespace+'/scheduling']=pipeline.tuning
                    run.summary[namespace+'/student_schedule_mode']=result['mode']
                    run.summary[namespace+'/student_groups']=result['groups']
                    tuned.add(key)
                    pipeline.tuning_key=key
                tracking.phase('training')
                pipeline.training_batch_size=loader.batch_size
                started=True
                yield chain((prepared,),batches),len(loader)
                return
        except (ScheduleMemoryError,torch.OutOfMemoryError) as error:
            if started or pipeline.device.type!='cuda':raise
            recovery=dict(window=length,batch_size=loader.batch_size,target_prefetch=pipeline.target_prefetch,
                          error=str(error),candidates=getattr(error,'candidates',[]))
            if loader.batch_size>1:
                batch_limits[length]=max(1,loader.batch_size//2)
                recovery.update(action='reduce_batch',next_batch_size=batch_limits[length])
            elif pipeline.parallel and pipeline.target_prefetch:
                pipeline.target_prefetch=False
                recovery.update(action='disable_target_prefetch',next_batch_size=1)
            else:
                raise RuntimeError('Distillation cannot fit batch size 1 without teacher-target prefetch. '
                                   'Reduce --collect-instances or --visual-chunk / --frame-chunk; '
                                   'saved student checkpoints are preserved.') from error
            with (out/'memory_recovery.jsonl').open('a') as stream:stream.write(json.dumps(recovery)+'\n')
            run.summary[getattr(tracking,'namespace','distillation')+'/memory_recovery']=recovery
            tracking.phase('reducing_memory')
            print('Distillation memory recovery:',json.dumps(recovery),flush=True)
        finally:
            if curriculum is not None:data._close_loader(loader)
        # The producer has joined and the exception traceback is now released.
        prepared=None;gc.collect()
        torch.cuda.synchronize(pipeline.device);torch.cuda.empty_cache()


def main(argv=None):
    from gated_models import GatedModel as BudgetStudent, PROFILES, ALL_PROFILES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path,nargs='?',default=HERE/'data')
    parser.add_argument('--teacher',type=Path,help='teacher run root; unused in ground-truth-only mode')
    parser.add_argument('--ground-truth-only',action='store_true',help='direct supervised training; no teacher loading or inference')
    parser.add_argument('--config',type=Path,help='host config JSON for ground-truth-only training')
    parser.add_argument('--curriculum-order',choices=['joint','staged'],help='default staged for ground-truth-only, joint for distillation')
    parser.add_argument('--plateau-patience',type=int,default=3,help='fresh checks for the plateau diagnostic; does not control promotion')
    parser.add_argument('--plateau-tolerance',type=float,default=.05,help='near-best tolerance for the plateau diagnostic')
    parser.add_argument('--plateau-min-delta',type=float,default=.005,help='relative improvement that resets diagnostic plateau patience')
    parser.add_argument('--curriculum-max-stage-checks',type=int,default=18,
                        help='advance after this many fresh validation checks even if accuracy targets are unmet')
    parser.add_argument('--curriculum-accuracy-patience',type=int,default=3,
                        help='consecutive fresh checks meeting every physical accuracy target')
    parser.add_argument('--curriculum-accuracy-targets',type=Path,
                        help='JSON overrides: metric name -> [base tolerance, tolerance per horizon second]')
    parser.add_argument('--collection-config',type=Path,help='JSON overrides for collection settings; teacher architecture is unchanged')
    parser.add_argument('--profiles',nargs='+',choices=tuple(ALL_PROFILES),help='default all current profiles; warm-start defaults to source profiles')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--warm-start',type=Path,help='load best student weights/settings from an earlier distillation run; reset optimizers and patience')
    parser.add_argument('--warm-start-stage',action='store_true',
        help='with --warm-start, load rolling stage-best weights and matching optimizer instead of global-best weights')
    parser.add_argument('--start-epoch',type=int,help='completed epochs; stage recovery infers this from saved progress, otherwise defaults to zero')
    parser.add_argument('--epochs',type=int,default=500,help='safety ceiling; each profile stops independently')
    parser.add_argument('--patience',type=int,default=8,help='non-improving fresh batches; epochs for fixed data')
    parser.add_argument('--window',type=int,help='fixed IMU sample count including the initial state; minimum 2')
    parser.add_argument('--curriculum-windows',type=int,nargs='+',default=list(DEFAULT_GATED_WINDOWS),
                        help='IMU sample counts including the initial state; defaults to 1,2,4,...,8192,12000 intervals plus one')
    parser.add_argument('--stage-epochs',type=int,default=6,
        help='minimum fresh checks before hybrid promotion; epochs per horizon without difficulty curriculum')
    parser.add_argument('--difficulty-curriculum',action=argparse.BooleanOptionalAction,default=True,
        help='adapt motion, appearance, IMU and window difficulty on improvement or a healthy plateau')
    parser.add_argument('--curriculum-min-batches',type=int,default=3,
        help='minimum fresh checks; effective minimum is max(this value, --stage-epochs)')
    parser.add_argument('--curriculum-probe-batches',type=int,default=2,help='fixed held-out minibatches per unfinished difficulty axis')
    parser.add_argument('--min-delta',type=float,default=1e-4,help='absolute validation ATE improvement in metres')
    parser.add_argument('--batch',type=int,help='state minibatch ceiling; default CUDA 8, CPU config value')
    parser.add_argument('--amp',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--frame-chunk',type=int)
    parser.add_argument('--visual-chunk',type=int)
    parser.add_argument('--embedding-chunk',type=int)
    parser.add_argument('--collect-instances',type=int)
    parser.add_argument('--loader-workers',type=int)
    parser.add_argument('--persistent-workers',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--fused-adamw',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--parallel',action=argparse.BooleanOptionalAction,default=True,
        help='overlap teacher prefetch and student tiers on CUDA streams (CPU stays serial)')
    parser.add_argument('--parallel-mode',choices=['auto','serial','streams'],default='auto')
    parser.add_argument('--target-prefetch',action=argparse.BooleanOptionalAction,default=True,
        help='overlap the next teacher batch; auto mode can disable it if batch size 1 lacks memory headroom')
    parser.add_argument('--parallel-teacher-physics',action=argparse.BooleanOptionalAction,default=True,
        help='parallel teacher integration in eval mode; does not enable dropout or AMP')
    parser.add_argument('--max-concurrent',type=int,help='explicit stream group size; disables automatic tuning')
    parser.add_argument('--architecture-config',type=Path,help='JSON mapping profile names to previously fuel-verified settings')
    parser.add_argument('--calibrate-architectures',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--wandb-mode',choices=['online','offline','disabled'],default='online')
    parser.add_argument('--training-policy', choices=['legacy','corrected_v1'])
    parser.add_argument('--architecture', choices=['legacy','persistent_visual_v2'])
    parser.add_argument('--allow-architecture-transfer', action='store_true')
    parser.add_argument('--velocity-update-bound', type=float, help='persistent_visual_v2; default 0.5 m/s or the warm-start value')
    parser.add_argument('--orientation-update-bound', type=float, help='persistent_visual_v2; default 0.02 rad or the warm-start value')
    parser.add_argument('--resume-lr', type=float)
    parser.add_argument('--lr-schedule', type=Path)
    parser.add_argument('--validation-manifest', type=Path)
    parser.add_argument('--curriculum-budget-action', choices=['advance','stop'])
    parser.add_argument('--fixed-horizon', action='store_true', help='hold --window at motion/noise level zero and probe without promotion')
    parser.add_argument('--max-fresh-checks', type=int, help='experiment check budget, preserved on exact resume')
    parser.add_argument('--init-weights', choices=['global','latest','stage'], default='global', help='weight-only initialization; optimizer moments reset')
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--lr-config',type=Path,help='JSON mapping student profiles to fresh-run learning rates')
    parser.add_argument('--kd-weight',type=float,default=1.)
    parser.add_argument('--teacher-state-weight',type=float,default=.5)
    parser.add_argument('--ground-truth-weight',type=float,default=1.)
    parser.add_argument('--embedding-weight',type=float,default=.1,help='JEPA auxiliary loss weight; zero disables it')
    parser.add_argument('--embedding-ema',type=float,default=.99,help='EMA target encoder decay in [0,1)')
    parser.add_argument('--alignment-weight',type=float,default=.1,help='Sim(3)-aligned training-horizon RMSE auxiliary weight; zero disables it')
    parser.add_argument('--device',choices=['cpu','cuda'])
    parser.add_argument('--stream-data',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--max-data-mib',type=float,help='recording budget; default 8192 MiB or --collection-config value')
    parser.add_argument('--max-passes-per-batch',type=int,default=3)
    parser.add_argument('--stream-dir',type=Path)
    parser.add_argument('--collector',type=Path)
    parser.add_argument('--skip-export',action='store_true',help='save student checkpoints; export later with python -m students.export')
    parser.add_argument('--verify-frames',type=int,default=1200)
    parser.add_argument('--wasi-sdk',type=Path)
    parser.add_argument('--seed',type=int,default=17)
    parser.set_defaults(ground_truth_only=True, skip_export=True, calibrate_architectures=False,
                        curriculum_order='staged', curriculum_probe_batches=12, epochs=1500,
                        stream_data=True, collection_config=HERE/'collection_60s.json',
                        collect_instances=1,loader_workers=0,persistent_workers=False)
    args=parser.parse_args(argv)
    previous = json.loads((args.warm_start/'distillation.json').read_text()) if args.warm_start else {}
    args.training_policy = args.training_policy or previous.get('training_policy','legacy')
    args.architecture = args.architecture or previous.get('architecture','legacy')
    args.curriculum_budget_action = args.curriculum_budget_action or previous.get('curriculum_budget_action',
        'stop' if args.training_policy=='corrected_v1' else 'advance')
    inherited_schedule=None
    if args.warm_start_stage and args.warm_start:
        # An exact resume continues the same experiment: keep its opt-in
        # settings unless explicitly overridden on the command line.
        if previous.get('fixed_horizon') and not args.fixed_horizon:
            args.fixed_horizon=True
            if args.window is None:args.window=previous['curriculum_windows'][0]
        for key in ('max_fresh_checks','resume_lr'):
            if getattr(args,key) is None:setattr(args,key,previous.get(key))
        if args.lr_schedule is None:inherited_schedule=previous.get('lr_schedule')
        if args.validation_manifest is None:
            # Validation collection is deterministic; reuse the parent's
            # recordings while they exist instead of collecting a copy.
            from gated_data import validation_files
            candidate=args.warm_start/'procthor_data'/'manifest.json'
            try:
                validation_files(candidate);args.validation_manifest=candidate
                print(f'Reusing validation recordings from {candidate}',flush=True)
            except (OSError,ValueError,KeyError):pass
    if args.warm_start_stage and (args.training_policy != previous.get('training_policy','legacy') or
                                args.architecture != previous.get('architecture','legacy')):
        parser.error('Exact resume requires matching architecture and supervision; use weight-only initialization for adaptation')
    if args.allow_architecture_transfer and (not args.warm_start or args.warm_start_stage):
        parser.error('Architecture transfer requires weight-only --warm-start')
    if args.architecture != previous.get('architecture','legacy') and args.warm_start and not args.allow_architecture_transfer:
        parser.error('Architecture changed; specify --allow-architecture-transfer for legacy to v2 initialization')
    if args.fixed_horizon and (args.window is None or not args.difficulty_curriculum):
        parser.error('--fixed-horizon requires --window and difficulty probes')
    if args.max_fresh_checks is not None and args.max_fresh_checks<1:parser.error('--max-fresh-checks must be positive')
    if args.max_fresh_checks is not None and not args.difficulty_curriculum:
        parser.error('--max-fresh-checks counts curriculum probes; it requires --difficulty-curriculum')
    if args.resume_lr is not None and (not np.isfinite(args.resume_lr) or args.resume_lr<=0):parser.error('Invalid --resume-lr')
    if not all(np.isfinite(v) and v>0 for v in (args.velocity_update_bound,args.orientation_update_bound) if v is not None):
        parser.error('Visual correction bounds must be positive and finite')
    try:
        lr_schedule=read_lr_schedule(args.lr_schedule) if args.lr_schedule else inherited_schedule
        if args.validation_manifest:
            from gated_data import validation_files
            validation_files(args.validation_manifest)
    except (OSError,ValueError,KeyError) as error:parser.error(str(error))
    try:
        target_settings=accuracy_targets(json.loads(args.curriculum_accuracy_targets.read_text())
                                         if args.curriculum_accuracy_targets else None)
    except (OSError,ValueError) as error:parser.error(str(error))
    if not 1<=args.curriculum_accuracy_patience<=max(args.stage_epochs,args.curriculum_min_batches)<=args.curriculum_max_stage_checks:
        parser.error('Require 1 <= accuracy patience <= minimum stage checks <= maximum stage checks')
    if not args.ground_truth_only or not args.skip_export or args.calibrate_architectures:
        parser.error('This experiment supports supervised GPU checkpoints only')
    if args.architecture_config:
        parser.error('Architecture overrides are not supported in this controlled ablation')
    if args.curriculum_order != 'staged':
        parser.error('Use the staged student curriculum')
    if args.warm_start and args.profiles is None:
        try:
            source=json.loads((args.warm_start/'distillation.json').read_text())
            selected=source.get('profiles') or [name for name in ALL_PROFILES if (args.warm_start/name/'student.json').is_file()]
        except (OSError,ValueError) as error:parser.error(str(error))
        if not selected or any(name not in ALL_PROFILES for name in selected):
            parser.error('Warm-start metadata has no supported student profiles')
        args.profiles=list(dict.fromkeys(selected))
    if args.profiles is None:args.profiles=list(PROFILES)
    if not args.ground_truth_only and args.teacher is None:parser.error('--teacher is required for distillation')
    if args.config and not args.ground_truth_only:parser.error('--config is only for ground-truth-only training')
    learning_rates={}
    if args.lr_config:
        try:
            learning_rates=json.loads(args.lr_config.read_text())
            if not isinstance(learning_rates,dict) or any(name not in ALL_PROFILES for name in learning_rates):
                raise ValueError('LR config must map known student profiles to rates')
            if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0 for value in learning_rates.values()):
                raise ValueError('Per-profile learning rates must be finite and positive')
        except (OSError,ValueError) as error:parser.error(str(error))
    args.curriculum_order=args.curriculum_order or ('staged' if args.ground_truth_only else 'joint')
    if args.ground_truth_only:
        args.kd_weight=args.teacher_state_weight=0.
    if args.plateau_patience<2 or not 0<args.plateau_tolerance<1 or not 0<args.plateau_min_delta<1:
        parser.error('plateau patience must be >=2 and tolerance/min-delta must be between 0 and 1')
    recovery=experiment_resume=None
    if args.warm_start_stage:
        if not args.warm_start:parser.error('--warm-start-stage requires --warm-start')
        from distillation_recovery import load_recovery
        try:
            # Newer runs embed recovery metadata in latest.pt (one atomic write).
            recovery,experiment_resume=embedded_recovery(args.warm_start)
            if recovery is None:recovery=load_recovery(args.warm_start,args.profiles)
            elif set(args.profiles)-set(recovery['students']):
                raise ValueError('Recovery metadata is missing selected students')
        except (OSError,ValueError) as error:parser.error(str(error))
        if args.start_epoch is not None and args.start_epoch!=recovery['completed_epochs']:
            parser.error('--start-epoch differs from saved recovery progress; omit it to continue automatically')
    if args.start_epoch is None:args.start_epoch=recovery['completed_epochs'] if recovery else 0
    args.resume_collection_index=recovery.get('collection_next_index',0) if recovery else 0
    windows=[args.window] if args.window is not None else args.curriculum_windows
    if (not windows or windows!=sorted(set(windows)) or min(windows)<2 or
        min(args.epochs,args.patience,args.batch or 1,args.max_passes_per_batch,args.verify_frames,args.stage_epochs,args.max_concurrent or 1)<1):
        parser.error('counts must be positive and --window must include at least two samples (one IMU interval)')
    if not np.isfinite(args.min_delta) or args.min_delta<0:parser.error('--min-delta must be finite and nonnegative')
    if args.max_concurrent is not None and args.max_concurrent<1:parser.error('--max-concurrent must be positive')
    if min(args.curriculum_min_batches,args.curriculum_probe_batches)<1:
        parser.error('curriculum counts must be positive')
    if args.difficulty_curriculum and args.stage_epochs<args.curriculum_min_batches:
        parser.error('--stage-epochs must be at least --curriculum-min-batches')
    if args.start_epoch<0 or args.start_epoch>=args.epochs or (args.start_epoch and not args.warm_start):
        parser.error('--start-epoch must be below --epochs, nonnegative, and requires --warm-start when nonzero')
    if args.warm_start_stage and not args.warm_start:parser.error('--warm-start-stage requires --warm-start')
    if args.warm_start:
        args.warm_start=args.warm_start.resolve()
        previous=json.loads((args.warm_start/'distillation.json').read_text())
        if previous.get('experiment') != '4_ssm_gated_updates':
            parser.error('Recovery source is not a gated-updates experiment')
        if args.warm_start_stage and previous.get('horizon_policy') != HORIZON_POLICY:
            parser.error('The horizon policy changed; start a fresh run rather than resuming the old curriculum')
        if previous.get('ground_truth_only',False)!=args.ground_truth_only:
            parser.error('warm-start training mode differs; use fresh checkpoints for a new objective')
        if previous.get('curriculum_order','joint')!=args.curriculum_order:
            parser.error('warm-start curriculum order differs')
        for filename,key in ([] if args.ground_truth_only else [('state_student.pt','state_sha256'),('frame_student.pt','frame_sha256')]):
            if previous.get(key)!=hashlib.sha256((args.teacher/filename).read_bytes()).hexdigest():
                parser.error('--warm-start teacher weights differ from --teacher')
        for name in args.profiles:
            weights_file='stage_best.pt' if args.warm_start_stage or args.init_weights=='stage' else ('latest.pt' if args.init_weights=='latest' else 'student.pt')
            weight_path=args.warm_start/weights_file if weights_file=='latest.pt' else args.warm_start/name/weights_file
            if not weight_path.is_file() or not (args.warm_start/name/'student.json').is_file():
                parser.error(f'--warm-start is missing {name}/{weights_file} or student.json')
    if not np.isfinite([args.embedding_weight,args.embedding_ema]).all() or args.embedding_weight<0 or not 0<=args.embedding_ema<1:
        parser.error('embedding weight must be finite/nonnegative and EMA decay in [0,1)')
    if not np.isfinite(args.alignment_weight) or args.alignment_weight<0:
        parser.error('alignment weight must be finite and nonnegative')
    weights=(args.kd_weight,args.teacher_state_weight,args.ground_truth_weight)
    if args.lr<=0 or not np.isfinite([args.lr,*weights]).all() or min(weights)<0 or sum(weights)==0:
        parser.error('learning rate must be positive and finite; loss weights must be finite/nonnegative with positive sum')
    if not args.ground_truth_only and not (args.teacher/'training_config.json').is_file():
        parser.error('--teacher needs a complete teacher checkpoint set, including training_config.json')
    if args.teacher:args.teacher=args.teacher.resolve()
    args.out=(args.out or HERE/'checkpoints'/('distill-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))).resolve()
    if args.out.exists():parser.error('choose a new --out directory to preserve existing models')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but no GPU is accessible in this process')
    import config as cfg
    if args.ground_truth_only:
        frame=teacher=None;settings={}
    else:
        frame,teacher,settings=load_teacher(args.teacher,device)
    from performance import PERFORMANCE_SETTINGS,configure,EpochTimer,record_epoch,adamw
    from streaming import COLLECTION_SETTINGS
    from gated_schedule import MeasuredPipeline as DistillationPipeline
    from functools import partial
    saved=(json.loads(args.config.read_text()) if args.config else {}) if args.ground_truth_only else json.loads((args.teacher/'training_config.json').read_text())
    if args.ground_truth_only and (not isinstance(saved,dict) or any(not key.isupper() or not hasattr(cfg,key) for key in saved)):
        parser.error('--config must contain known uppercase config settings')
    for key,value in saved.items():
        if key.isupper() and hasattr(cfg,key) and (args.ground_truth_only or key not in PERFORMANCE_SETTINGS|COLLECTION_SETTINGS):setattr(cfg,key,value)
    if args.collection_config:
        overrides=json.loads(args.collection_config.read_text())
        if not isinstance(overrides,dict) or set(overrides)-COLLECTION_SETTINGS:
            parser.error('--collection-config accepts only collection settings: '+', '.join(sorted(COLLECTION_SETTINGS)))
        for key,value in overrides.items():setattr(cfg,key,value)
    if not args.ground_truth_only:
        cfg.H,cfg.W=settings['height'],settings['width']
        cfg.USE_GENERATED_FRAMES_FOR_STATE=settings['use_generated_frames']
        cfg.VISUAL_PREDICTION=settings['visual_prediction']
    if args.curriculum_order=='staged':
        # Never sample a harder augmentation before its stage is introduced.
        cfg.CURRICULUM_LEVELS=[i/9 for i in range(10)]
        cfg.CURRICULUM_MIX=[.75,.25,0.]
        cfg.CURRICULUM_ALL_EASIER=True
        cfg.CURRICULUM_RANDOM_ERROR_STRENGTH=True
    cfg.CACHE_GENERATED_WINDOWS=False
    cfg.DEVICE=device;cfg.WORKDIR=args.data.resolve();cfg.WINDOW=windows[0]
    cfg.USE_CURRICULUM=False  # This distiller owns its curriculum independently of teacher training.
    cfg.CURRICULUM_BATCH_TOKENS=3208
    if args.batch is None and device=='cuda':cfg.BATCH=8
    if args.stream_data:cfg.COLLECT_STEPS=max(cfg.COLLECT_STEPS,windows[-1])
    cfg.SEED=args.seed;cfg.STREAM_DATA=args.stream_data
    # Twelve held-out 20-second flights can exceed a 1 GiB validation slot.
    # Keep the larger bounded recording ring off the quota-limited home disk.
    if args.stream_data:
        import tempfile,getpass,os
        user=getpass.getuser()
        if args.stream_dir is None:
            parser.error('Start through train.py to configure a disk-backed recording directory')
        if not os.environ.get('CONTA_CACHE'):
            cache_root=Path('/var/tmp') if Path('/var/tmp').is_dir() else Path(tempfile.gettempdir())
            os.environ['CONTA_CACHE']=str(cache_root/('hyperdrone-'+user)/'scene-assets')
            legacy_cache=str(Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'conta')
            searches=os.environ.get('CONTA_SEARCH_PATHS','').split(os.pathsep)
            os.environ['CONTA_SEARCH_PATHS']=os.pathsep.join(dict.fromkeys([path for path in searches if path]+[legacy_cache]))
        if args.max_data_mib is None and not (args.collection_config and 'STREAM_MAX_MIB' in overrides):
            cfg.STREAM_MAX_MIB=8192
    if args.max_data_mib is not None:cfg.STREAM_MAX_MIB=args.max_data_mib
    cfg.STREAM_MAX_PASSES_PER_BATCH=args.max_passes_per_batch
    for option,key in [('batch','BATCH'),('amp','USE_AMP'),('frame_chunk','FRAME_GENERATION_CHUNK'),
                       ('visual_chunk','VISUAL_CHUNK'),('collect_instances','COLLECT_INSTANCES'),('loader_workers','LOADER_WORKERS'),
                       ('persistent_workers','PERSISTENT_WORKERS'),('fused_adamw','FUSED_ADAMW'),('embedding_chunk','EMBEDDING_PREDICTION_CHUNK')]:
        if getattr(args,option) is not None:setattr(cfg,key,getattr(args,option))
    configure(cfg)
    args.out.mkdir(parents=True)
    provenance=dict(teacher=str(args.teacher) if not args.ground_truth_only else None,teacher_settings=settings,
        ground_truth_only=args.ground_truth_only,curriculum_order=args.curriculum_order,
        plateau_patience=args.plateau_patience,plateau_tolerance=args.plateau_tolerance,plateau_min_delta=args.plateau_min_delta,
        state_sha256=hashlib.sha256((args.teacher/'state_student.pt').read_bytes()).hexdigest() if not args.ground_truth_only else None,
        frame_sha256=hashlib.sha256((args.teacher/'frame_student.pt').read_bytes()).hexdigest() if not args.ground_truth_only else None,
        lr=args.lr,learning_rates={name:learning_rates.get(name,args.lr) for name in args.profiles},
        alignment_weight=args.alignment_weight,alignment_scope='current_training_horizon_complete_validation_flights',
        alignment_fit='detached_umeyama_sim3_camera_positions',
        embedding_weight=args.embedding_weight,embedding_ema=args.embedding_ema,
        embedding_std_target=.1,embedding_variance_weight=1.,embedding_covariance_weight=.01,
        epochs=args.epochs,curriculum_windows=windows,stage_epochs=args.stage_epochs,kd_weight=args.kd_weight,
        parallel=bool(args.parallel and args.parallel_mode!='serial' and device=='cuda'),
        target_prefetch_batches=int(args.parallel and args.parallel_mode!='serial' and device=='cuda' and args.target_prefetch),
        parallel_mode=args.parallel_mode,max_concurrent=args.max_concurrent,patience=args.patience,min_delta=args.min_delta,
        parallel_teacher_physics=args.parallel_teacher_physics,
        max_data_mib=cfg.STREAM_MAX_MIB,stream_dir=str(args.stream_dir) if args.stream_dir else None,
        collection={key:getattr(cfg,key,None) for key in sorted(COLLECTION_SETTINGS)},
        scene_cache=(os.environ.get('CONTA_CACHE') if args.stream_data else None),
        performance={key:getattr(cfg,key) for key in PERFORMANCE_SETTINGS},
        teacher_state_weight=args.teacher_state_weight,ground_truth_weight=args.ground_truth_weight)
    provenance.update(warm_start=str(args.warm_start) if args.warm_start else None,start_epoch=args.start_epoch)
    provenance['warm_start_stage']=args.warm_start_stage
    provenance['preserve_curriculum_progress']=bool(recovery)
    provenance['collection_start_index']=args.resume_collection_index
    provenance.update(difficulty_curriculum=args.difficulty_curriculum,
                      curriculum_progression='all_students_competent_or_stage_budget_exhausted',
                      curriculum_max_stage_checks=args.curriculum_max_stage_checks,
                      curriculum_accuracy_patience=args.curriculum_accuracy_patience,
                      curriculum_accuracy_targets=target_settings,curriculum_pause='none',
                      curriculum_transition='restore_stage_best_model_and_optimizer',
                      curriculum_min_batches=args.curriculum_min_batches,curriculum_probe_batches=args.curriculum_probe_batches,
                      curriculum_settings={key:getattr(cfg,key) for key in vars(cfg) if key.startswith('CURRICULUM_')})
    provenance.update(training_policy=args.training_policy, architecture=args.architecture,
        parent_training_policy=previous.get('training_policy','legacy') if previous else None,
        selection_policy='physical_stage_unaligned_global_v1' if args.training_policy=='corrected_v1' else 'sim3',
        resume_lr=args.resume_lr,lr_schedule=lr_schedule,validation_manifest=str(args.validation_manifest) if args.validation_manifest else None,
        curriculum_budget_action=args.curriculum_budget_action,fixed_horizon=args.fixed_horizon,
        max_fresh_checks=args.max_fresh_checks,init_weights=args.init_weights)
    provenance['experiment']='4_ssm_gated_updates' 
    import resource_limits
    from dataclasses import asdict
    if resource_limits.ACTIVE is not None:
        provenance['resource_limits']=asdict(resource_limits.ACTIVE.limits)
        provenance['resource_directory']=str(resource_limits.ACTIVE.root)
    provenance['stored_camera_hz']=20
    provenance['horizon_policy']=HORIZON_POLICY
    provenance['curriculum_validation_policy']=VALIDATION_POLICY
    provenance['horizon_steps']=[length-1 for length in windows]
    provenance['horizon_seconds']=[(length-1)/200. for length in windows]
    provenance['image_height']=cfg.H
    provenance['image_width']=cfg.W
    provenance['profiles']=list(dict.fromkeys(args.profiles))
    (args.out/'distillation.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (args.out/'training_config.json').write_text(json.dumps({key: value for key,value in vars(cfg).items()
        if key.isupper() and isinstance(value,(str,int,float,bool,list,type(None)))},indent=2)+'\n')
    from gated_data import GatedTrainingData as TrainingData
    from streaming import FreshBatchPatience
    from gated_recovery import save_weights
    from leaderboard_metrics import LeaderboardATE
    overrides=json.loads(args.architecture_config.read_text()) if args.architecture_config else {}
    import wandb
    run=wandb.init(project=cfg.WANDB_PROJECT,dir=str(args.out),job_type='gated_supervised',
        mode=args.wandb_mode,config=provenance,name=args.out.name)
    from gated_tracking import GatedLogger as DistillationLogger
    print(f'W&B: project={cfg.WANDB_PROJECT}, mode={args.wandb_mode}, URL={run.url}',flush=True)
    metric_namespace='supervised' if args.ground_truth_only else 'distillation'
    with DistillationLogger(run,args.profiles,namespace=metric_namespace) as tracking:
        students={};recovery_sources={}
        for name in dict.fromkeys(args.profiles):
            torch.manual_seed(args.seed)
            directory=args.out/name;directory.mkdir()
            prior_spec=json.loads((args.warm_start/name/'student.json').read_text()) if args.warm_start else None
            spec=dict(prior_spec or overrides.get(name,ALL_PROFILES[name]),profile=name)
            spec.update(architecture=args.architecture, training_policy=args.training_policy)
            for key,default in (('velocity_update_bound',.5),('orientation_update_bound',.02)):
                value=getattr(args,key)
                spec[key]=value if value is not None else (prior_spec or {}).get(key,default)
            if spec['fuel']!=ALL_PROFILES[name]['fuel']:raise ValueError(f'{name} architecture changes its fuel cap')
            if not args.warm_start and name not in overrides and spec.get('kind')=='mamba2' and args.calibrate_architectures:
                spec=select_architecture(name,directory,args.wasi_sdk,args.verify_frames)
            if args.embedding_weight and spec.get('teacher_style'):spec['jepa_target']=True
            model=BudgetStudent(spec).to(device,non_blocking=True)
            stage_source=None
            if args.warm_start:
                if args.warm_start_stage:
                    stage_source=torch.load(args.warm_start/name/'stage_best.pt',map_location='cpu',weights_only=True)
                    model.load_state_dict(stage_source['model'])
                    recovery_sources[name]={key:stage_source[key] for key in ('stage','stage_index','epoch','validation_mean_ate_m')}
                    copyfile(args.warm_start/name/'stage_best.pt',directory/'stage_best.pt')
                else:
                    if args.init_weights=='latest':
                        source_state=torch.load(args.warm_start/'latest.pt',map_location='cpu',weights_only=False)['models'][name]['model']
                    elif args.init_weights=='stage':
                        source_state=torch.load(args.warm_start/name/'stage_best.pt',map_location='cpu',weights_only=True)['model']
                    else:source_state=torch.load(args.warm_start/name/'student.pt',map_location='cpu',weights_only=True)
                    load_model_weights(model,source_state,transfer=args.allow_architecture_transfer)
                if recovery and (args.warm_start/name/'student.pt').is_file():
                    copyfile(args.warm_start/name/'student.pt',directory/'student.pt')
                else:save_weights(model.state_dict(),directory/'student.pt')
            (directory/'student.json').write_text(json.dumps(spec,indent=2)+'\n')
            optimizer=adamw(model.parameters(),lr=learning_rates.get(name,args.lr))
            if stage_source is not None:optimizer.load_state_dict(stage_source['optimizer'])
            stage_source=None
            students[name]=dict(model=model,optimizer=optimizer,embedding_weight=args.embedding_weight,alignment_weight=args.alignment_weight,
                best=float('inf'),patience=FreshBatchPatience(),active=True,directory=directory,completed_epoch=0)
            if recovery:
                previous_student=recovery['students'][name]
                row=students[name]
                row['active']=previous_student['active']
                row['completed_epoch']=previous_student.get('completed_epoch',args.start_epoch)
                row['best']=(previous_student['best_validation_mean_ate_m'] if (args.warm_start/name/'student.pt').is_file()
                             else recovery_sources[name]['validation_mean_ate_m'])
                vars(row['patience']).update(previous_student['patience'])
                run.summary[name+'/completed_epochs']=row['completed_epoch']
                run.summary[name+'/best_validation_mean_ate_m']=row['best']
                run.summary[name+'/non_improving_data_cycles']=row['patience'].bad_cycles
                run.summary[name+'/active']=row['active']
            print(f'{name}: {sum(p.numel() for p in model.parameters()):,} parameters, GPU experiment',flush=True)
        print('Student runs:',args.out,flush=True)
        if args.warm_start:
            mode='stage-best weights/optimizers and saved progression loaded' if args.warm_start_stage else 'best weights loaded, optimizers/patience reset'
            print(f'Warm start from {args.warm_start}; {mode}; '
                  f'continuing curriculum at epoch {args.start_epoch+1}.',flush=True)
        run.config.update({'student_architectures':{name:row['model'].settings for name,row in students.items()},
                           'student_parameters':{name:sum(p.numel() for p in row['model'].parameters()) for name,row in students.items()}})
        with ExitStack() as stack:
            tracking.phase('collecting_data' if args.stream_data else 'loading_data')
            from gpu_collection import GpuCollectionGate
            args.gpu_collection_gate=GpuCollectionGate(device) if args.stream_data else None
            curriculum=None
            if args.difficulty_curriculum:
                from distillation_curriculum import DistillationCurriculum
                curriculum_class=DistillationCurriculum
                extra={}
                if args.curriculum_order=='staged':
                    from gated_curriculum import GatedStudentCurriculum
                    curriculum_class=GatedStudentCurriculum
                    extra=dict(plateau_patience=args.plateau_patience,tolerance=args.plateau_tolerance,
                               improvement=args.plateau_min_delta,max_stage_checks=args.curriculum_max_stage_checks,
                               accuracy_patience=args.curriculum_accuracy_patience,targets=target_settings,
                               budget_action=args.curriculum_budget_action,fixed_horizon=args.fixed_horizon)
                curriculum=curriculum_class(cfg,windows,args.start_epoch,args.stage_epochs,
                    args.curriculum_min_batches,args.curriculum_probe_batches,**extra)
                if (recovery and recovery['curriculum'] is not None) or (args.warm_start and not args.fixed_horizon and args.training_policy==previous.get('training_policy','legacy') and (args.warm_start/'curriculum_state.json').is_file()):
                    state=recovery['curriculum'] if recovery else json.loads((args.warm_start/'curriculum_state.json').read_text())
                    if state is not None:
                        curriculum.load_state_dict(state,preserve_progress=bool(recovery),
                            students=[name for name,row in students.items() if row['active']])
                        if getattr(curriculum,'horizon_migration',None):
                            print('Curriculum horizon migration:',json.dumps(curriculum.horizon_migration),flush=True)
                            run.config.update({'curriculum_horizon_migration':curriculum.horizon_migration})
                        if getattr(curriculum,'validation_migration',None):
                            print('Curriculum validation migration:',json.dumps(curriculum.validation_migration),flush=True)
                            run.config.update({'curriculum_validation_migration':curriculum.validation_migration})
                        (args.out/'curriculum_state.json').write_text(json.dumps(curriculum.state_dict(),indent=2)+'\n')
                    if recovery and (state is None or 'student_gates' not in state):
                        print('Legacy recovery: saved difficulty restored; no per-student readiness counters were available.',flush=True)
                run.config.update({'effective_hybrid_curriculum':dict(
                    minimum_stage_checks=curriculum.minimum,maximum_stage_checks=curriculum.max_stage_checks,
                    accuracy_patience=curriculum.accuracy_patience,accuracy_targets=curriculum.accuracy_target_settings)})
            for name,row in students.items():
                if recovery:run.summary[name+'/status']=('early_stopped' if not row['active'] else
                    'paused_waiting_for_curriculum' if curriculum and name in curriculum.paused else 'training')
            if recovery:
                restore_latest(args.warm_start, students)
            apply_learning_rate(students,args.resume_lr,lr_schedule,(curriculum.length-1)/200. if curriculum else (windows[0]-1)/200.)
            data=TrainingData(args,cfg,stack)
            if recovery:data.epoch=args.start_epoch
            # Fail early for a dataset too short for the requested final horizon.
            data.ppo_loader(windows[-1],validation=True)
            parallel=args.parallel and args.parallel_mode!='serial'
            pipeline=stack.enter_context(DistillationPipeline(device,students,parallel,args.max_concurrent,args.target_prefetch,
                                                             gpu_gate=args.gpu_collection_gate))
            from gated_recovery import SafeStageBestCheckpoints as StageBestCheckpoints
            stage_checkpoints=StageBestCheckpoints()
            if recovery:
                stage_checkpoints.recover(recovery_sources)
                restored_progress=dict(completed_epochs=args.start_epoch,stage_index=stage_checkpoints.index,
                    window=curriculum.length if curriculum else curriculum_window(args.start_epoch,windows,args.stage_epochs),
                    levels=curriculum.schedule.snapshot() if curriculum else {},
                    paused_students=sorted(curriculum.paused) if curriculum else [],
                    collection_next_index=args.resume_collection_index,legacy_migration=recovery.get('legacy_migration',False))
                print('Recovered curriculum progression:',json.dumps(restored_progress),flush=True)
                run.summary[metric_namespace+'/recovered_progress']=restored_progress
                run.log({metric_namespace+'/epoch':args.start_epoch,'curriculum/stage_index':stage_checkpoints.index,
                         'curriculum/recovered_progress':True})
                if curriculum:tracking.curriculum(args.start_epoch,curriculum,None)
            fresh_checks=0
            if recovery and experiment_resume:fresh_checks=experiment_resume.get('fresh_checks',0)
            elif recovery and (args.warm_start/'experiment_state.json').exists():
                fresh_checks=json.loads((args.warm_start/'experiment_state.json').read_text())['fresh_checks']
            stop_reason=None
            tuned=set();batch_limits={length:distillation_batch_size(cfg.BATCH,length) for length in windows}
            mode_label='Supervised' if args.ground_truth_only else 'Distillation'
            print(f'{mode_label} scheduling: {"parallel CUDA streams" if pipeline.parallel else "serial"}; '
                  f'batch prefetch={pipeline.parallel and pipeline.target_prefetch}',flush=True)
            for epoch in range(args.start_epoch,args.epochs):
                if not any(row['active'] for row in students.values()):break
                timer=EpochTimer(device)
                tracking.phase('waiting_for_data')
                _,loader=data.next_epoch()
                length=curriculum.length if curriculum else curriculum_window(epoch,windows,args.stage_epochs)
                # Avoid forking a CUDA process with live scheduling threads. Dataset
                # objects contain CPU tensors and are safe to send to spawn workers.
                if pipeline.parallel:
                    for state_loader in (data.val_episode_loader,):
                        if state_loader.num_workers:state_loader.multiprocessing_context='spawn'
                active={name:row for name,row in students.items() if row['active']}
                stage=dict(window=length,levels=curriculum.schedule.snapshot() if curriculum else {})
                with pipeline.gpu_work():restored=stage_checkpoints.begin(stage,active,completed_epochs=epoch)
                if restored:
                    print(f'{mode_label} stage restore:',json.dumps(restored),flush=True)
                    with (args.out/'stage_restores.jsonl').open('a') as stream:stream.write(json.dumps(restored)+'\n')
                    tracking.stage_restore(epoch+1,restored)
                apply_learning_rate(students,args.resume_lr,lr_schedule,(length-1)/200.)
                training=curriculum.training_students(active) if curriculum else active
                paused=tuple(name for name in active if name not in training)
                with_images=any(row['model'].settings.get('kind')!='imu_affine' for row in active.values())
                train_terms={name:{} for name in active}
                validation_terms={name:{} for name in active}
                current_stage_validation={}
                current_stage_accuracy={}
                train_losses={name:[0.,0] for name in active};validation={name:[0.,0] for name in active}
                prepare=(partial(ground_truth_targets,device=device,with_images=with_images) if args.ground_truth_only else
                         partial(frame_targets,teacher_frame=frame,teacher_state=teacher,device=device,with_images=with_images,
                                 parallel_teacher_physics=args.parallel_teacher_physics))
                update=partial(train_student,device=device,weights=weights,
                               embedding_weight=args.embedding_weight,embedding_ema=args.embedding_ema,alignment_weight=args.alignment_weight)
                tracking.phase('training')
                auto=args.parallel_mode=='auto' and args.max_concurrent is None
                with training_batches(pipeline,data,length,training,prepare,update,batch_limits,tuned,
                                      tracking,run,args.out,auto=auto,curriculum=curriculum,epoch=epoch) as (batches,total_batches):
                    for batch_index,prepared in enumerate(batches,1):
                        count=len(prepared[0][0]['imu'])
                        batch_losses=pipeline.run(training,prepared,update)
                        for name,loss in batch_losses.items():
                            train_losses[name][0]+=loss*count;train_losses[name][1]+=count
                            for key,value in training[name].get('loss_terms',{}).items():
                                train_terms[name][key]=train_terms[name].get(key,0)+value*count
                        timer.add(count)
                        tracking.progress(epoch+1,batch_index,total_batches,length,batch_losses,pipeline.max_concurrent,
                                          paused_students=paused)
                for row in active.values():row['validation_metric']=LeaderboardATE()
                tracking.phase('validating')
                prepare=partial(validation_inputs,device=device,with_images=with_images)
                with pipeline.batches(data.val_episode_loader,prepare) as batches:
                    for prepared in batches:
                        count=len(prepared[0][0]['imu'])
                        for name,loss in pipeline.run(active,prepared,partial(validate_student,device=device)).items():
                            validation[name][0]+=loss*count;validation[name][1]+=count
                            for key,value in active[name].get('validation_loss_terms',{}).items():
                                validation_terms[name][key]=validation_terms[name].get(key,0)+value*count
                ate_scores={name:row['validation_metric'].result()['mean_ate_m'] for name,row in active.items()}
                if args.training_policy=='legacy':
                    with pipeline.gpu_work():stage_checkpoints.observe(active,ate_scores,epoch+1)
                curriculum_event=None
                if curriculum:
                    cycle_id=f'{args.out}:{data.batch_id}'
                    if not args.stream_data:cycle_id+=f':epoch-{epoch}'
                    fresh_cycle=curriculum.fresh(cycle_id)
                    probe_scores={}
                    if fresh_cycle:
                        tracking.phase('probing_curriculum')
                        probe_scores={}
                        probe_prepare=lambda batch:validation_inputs(batch,device,with_images)[:2]
                        for axis in curriculum.probe_axes:
                            totals={name:[0.,0] for name in active}
                            squared_errors={name:{} for name in active}
                            probe_loader=curriculum.probe_loader(data.val_data,axis,pipeline.training_batch_size)
                            with pipeline.batches(probe_loader,probe_prepare) as batches:
                                for prepared in batches:
                                    count=len(prepared[0][0]['imu'])
                                    for name,loss in pipeline.run(active,prepared,partial(validate_student,device=device)).items():
                                        totals[name][0]+=loss*count;totals[name][1]+=count
                                        for key,value in active[name]['curriculum_squared_errors'].items():
                                            squared_errors[name][key]=squared_errors[name].get(key,0.)+value*count
                            probe_scores[axis]={name:value[0]/value[1] for name,value in totals.items()}
                            current_stage_validation.update(probe_scores[axis])
                            current_stage_accuracy.update({name:{key:(value/totals[name][1])**.5
                                for key,value in errors.items()} for name,errors in squared_errors.items()})
                        if args.training_policy=='corrected_v1':
                            scores={name:physical_selection_score(value,curriculum.current_accuracy_targets()) for name,value in current_stage_accuracy.items()}
                            with pipeline.gpu_work():stage_checkpoints.observe(active,scores,epoch+1)
                        curriculum_event=curriculum.update(cycle_id,probe_scores,ate_scores,current_stage_accuracy)
                        fresh_checks+=1
                        if curriculum_event.get('stop_requested'):stop_reason='stage_budget_exhausted'
                        if args.max_fresh_checks and fresh_checks>=args.max_fresh_checks:stop_reason='fresh_check_limit' 
                    if hasattr(curriculum,'finish_epoch'):
                        forced=curriculum.finish_epoch(epoch+1,active,paused,fresh_batch=fresh_cycle)
                        if forced:
                            if curriculum_event:forced['probe_scores']=curriculum_event['probe_scores']
                            curriculum_event=forced
                    for name in curriculum.paused:active[name]['optimizer'].zero_grad(set_to_none=True)
                    (args.out/'curriculum_state.json').write_text(json.dumps(curriculum.state_dict(),indent=2)+'\n')
                    if curriculum_event:
                        curriculum_event.update(epoch=epoch+1,recording_batch=data.batch_id,student_probe_losses=probe_scores)
                        with (args.out/'curriculum_metrics.jsonl').open('a') as stream:stream.write(json.dumps(curriculum_event)+'\n')
                        print(f'{mode_label} curriculum:',json.dumps(curriculum_event),flush=True)
                    tracking.curriculum(epoch+1,curriculum,curriculum_event)
                if curriculum is None and args.training_policy=='corrected_v1':
                    scores={name:row['validation_metric'].result()['unaligned_ate_rmse_m'] for name,row in active.items()}
                    with pipeline.gpu_work():stage_checkpoints.observe(active,scores,epoch+1)
                measurement=timer.result()
                measurement.update(active_profiles=list(active),parallel=pipeline.parallel,
                                   collector_oom_retries=data.rolling.oom_retries if data.rolling else 0,
                                   collector_oom_recoveries=data.rolling.oom_recoveries if data.rolling else 0,
                                   collector_optix_compile_retries=data.rolling.optix_compile_retries if data.rolling else 0,
                                   collector_optix_compile_recoveries=data.rolling.optix_compile_recoveries if data.rolling else 0,
                                   training_profiles=list(training),paused_profiles=list(paused),
                                   student_updates=timer.samples*len(training),window=length,max_concurrent=pipeline.max_concurrent,
                                   training_batch_size=pipeline.training_batch_size,
                                   target_prefetch_batches=int(pipeline.parallel and pipeline.target_prefetch))
                measurement.update(horizon_imu_steps=length-1,horizon_seconds=(length-1)/200.)
                record_epoch(args.out,metric_namespace,epoch+1,measurement,data)
                print(f'{mode_label} throughput: {measurement["samples_per_second"]:.2f} windows/s '
                      f'({len(training)} training, {len(paused)} paused profiles; includes validation/data wait)',flush=True)
                epoch_records={}
                for name,row in active.items():
                    metric_result=row['validation_metric'].result()
                    score=metric_result['unaligned_ate_rmse_m' if args.training_policy=='corrected_v1' else 'mean_ate_m']
                    if not np.isfinite(score):raise ValueError(f'Nonfinite {name} validation loss')
                    improved=score<row['best']-args.min_delta
                    if improved:
                        row['best']=score;save_weights(row['model'].state_dict(),row['directory']/'student.pt')
                    final_stage=curriculum.early_stopping_eligible if curriculum else length==windows[-1]
                    promoted=bool(curriculum_event and (curriculum_event['promoted'] or curriculum_event['window_promoted']))
                    bad=row['patience'].update(improved or not final_stage,data.batch_id,promoted=promoted,
                        adaptation_cycles=cfg.CURRICULUM_EARLY_STOP_GRACE if promoted else 0)
                    row['active']=bad<args.patience
                    row['completed_epoch']=epoch+1
                    record=dict(epoch=epoch+1,train_loss=(train_losses[name][0]/train_losses[name][1] if train_losses[name][1] else None),
                        curriculum_stage=stage_checkpoints.index,
                        # Empty until this stage's first observation (corrected
                        # policy observes on fresh checks only).
                        stage_best_epoch=stage_checkpoints.best.get(name,{}).get('epoch'),
                        stage_best_validation_mean_ate_m=stage_checkpoints.best.get(name,{}).get('validation_mean_ate_m'),
                        training_samples=train_losses[name][1],training_paused=name in paused,
                        paused_waiting_for_curriculum=bool(curriculum and name in curriculum.paused),
                        validation_state_loss=validation[name][0]/validation[name][1],
                        validation_unaligned_mte_m=row['validation_metric'].result()['unaligned_mte_m'],
                        validation_unaligned_ate_rmse_m=row['validation_metric'].result()['unaligned_ate_rmse_m'],
                        validation_mean_ate_m=metric_result['mean_ate_m'],best_validation_mean_ate_m=row['best'],window=length,
                        learning_rate=row['optimizer'].param_groups[0]['lr'],
                        non_improving_data_cycles=bad,recording_batch=data.batch_id,active=row['active'])
                    record.update({'train_'+key:float(value/train_losses[name][1])
                                   for key,value in train_terms[name].items()} if train_losses[name][1] else {})
                    record.update({'validation_'+key:float(value/validation[name][1])
                                   for key,value in validation_terms[name].items()})
                    record.update({'procthor_val/validation_loss': validation[name][0]/validation[name][1],
                                   'procthor_val/sim3_ate_rmse_m': metric_result['mean_ate_m'],
                                   'procthor_val/unaligned_ate_rmse_m': row['validation_metric'].result()['unaligned_ate_rmse_m']})
                    record['checkpoint_selection_score']=score
                    record['fresh_checks']=fresh_checks
                    record['procthor_curriculum_val/fresh_data_check']=name in current_stage_validation
                    if name in current_stage_validation:
                        prefix='procthor_curriculum_val/'
                        record.update({prefix+'validation_loss':current_stage_validation[name],
                                       prefix+'horizon_imu_steps':length-1,prefix+'horizon_seconds':(length-1)/200.})
                        record.update({prefix+axis+'_level':level for axis,level in stage['levels'].items()})
                        record.update({prefix+key:value for key,value in current_stage_accuracy[name].items()})
                        record.update({prefix+'target_'+key:value for key,value in curriculum_event['accuracy_targets'].items()})
                        record[prefix+'stage_checks']=curriculum_event['stage_checks']
                        record[prefix+'competent']=curriculum_event['students'][name]['competent']
                        record[prefix+'promotion_reason']=curriculum_event['readiness'][curriculum_event['phase']][name]
                    with (row['directory']/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                    epoch_records[name]=record
                    training_label=f'{record["train_loss"]:.6g}' if record['train_loss'] is not None else 'paused'
                    status=('early_stopped' if not row['active'] else
                            'paused_waiting_for_group' if record['paused_waiting_for_curriculum'] else 'training')
                    print(f'{mode_label} {name} {epoch+1}/{args.epochs}: train_loss={training_label}, '
                          f'val_loss={record["validation_state_loss"]:.6g}, '
                          f'jepa_loss={record.get("train_embedding_loss", 0.):.6g}, '
                          f'alignment_loss={record.get("train_alignment_raw", 0.):.6g}, '
                          f'Selection ATE={score:.6g} m, stale cycles={bad}, status={status}',flush=True)
                from distillation_recovery import recovery_state, write_recovery
                state=recovery_state(epoch+1,curriculum,students,stage_checkpoints,data.rolling)
                experiment=dict(fresh_checks=fresh_checks,stop_reason=stop_reason)
                save_latest(args.out,students,recovery=state,experiment=experiment)
                # Human-readable mirrors; latest.pt is authoritative on resume.
                write_recovery(args.out,state)
                temporary=args.out/'experiment_state.json.tmp'
                temporary.write_text(json.dumps(experiment));temporary.replace(args.out/'experiment_state.json')
                tracking.epoch(epoch+1,epoch_records,measurement,data.rolling.last_wait_seconds if data.rolling else 0.)
                if stop_reason or not any(row['active'] for row in students.values()):break
        for name,row in students.items():
            tracking.phase('saving_checkpoint' if args.skip_export else 'exporting:'+name)
            row['model'].load_state_dict(torch.load(row['directory']/'student.pt',map_location=device,weights_only=True))
            run.summary[name+'/checkpoint']=str(row['directory']/'student.pt')
            run.summary[name+'/stop_reason']=stop_reason or ('epoch_limit' if row['active'] else 'early_stopping')
            if not args.skip_export:
                from students.export import export_student
                module=export_student(row['model'].eval(),row['directory']/'native',args.wasi_sdk,args.verify_frames)
                report=json.loads((module.parent/'limits.json').read_text())
                run.summary[name+'/wasm']=str(module)
                run.summary[name+'/fuel_step_max']=report['fuel_step_max']
                run.summary[name+'/status']='exported'
            else:run.summary[name+'/status']='checkpoint_saved'
        print(f'{mode_label} complete; checkpoints:',args.out,flush=True)


if __name__=='__main__':
    main()
