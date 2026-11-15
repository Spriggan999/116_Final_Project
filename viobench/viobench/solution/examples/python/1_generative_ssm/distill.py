"""Shared-teacher distillation into seven independently stopped native students."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[4]))


def load_teacher(directory,device):
    from export import make_models,normalize_settings
    settings=normalize_settings(json.loads((directory/'student_config.json').read_text()))
    if settings['backbone']=='mamba2_hybrid':
        from models import require_official_mamba
        require_official_mamba(device)
    frame,state=make_models(settings)
    frame.load_state_dict(torch.load(directory/'frame_student.pt',map_location='cpu',weights_only=True))
    state.load_state_dict(torch.load(directory/'state_student.pt',map_location='cpu',weights_only=True))
    for model in (frame,state):
        model.to(device,non_blocking=True).eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
    return frame,state,settings


def frame_inputs(batch,device,with_images=True):
    """Causal frame-rate observations and body-frame ground truth."""
    with torch.no_grad():
        ticks=list(range(0,batch['imu'].shape[1],10))
        if len(ticks)<2:raise ValueError('Distillation needs a window of at least 11 IMU ticks')
        images=None
        if with_images:
            from data import camera_values
            anchors=batch['anchor_img_idx'][:,ticks].long().to(device,non_blocking=True)
            cams=camera_values(batch['cam_images'],device)
            images=cams[torch.arange(cams.shape[0],device=device)[:,None],anchors]
            from students.model import camera_frames
            images=camera_frames(images)
        inputs=dict(images=images,imu=batch['imu'].to(device,non_blocking=True),dt=batch['dt'].to(device,non_blocking=True),
                    gravity=batch['gravity'].to(device,non_blocking=True),ticks=ticks)
        truth={key:batch[key][:,ticks].to(device,non_blocking=True).clone() for key in ('pos','vel','rot','omega')}
        truth['omega'][:,0]=0
    return inputs,truth


def frame_targets(batch,teacher_frame,teacher_state,device,with_images=True,parallel_teacher_physics=True):
    """Run teacher once; all budget tiers share its frozen labels in RAM."""
    from data import state_input_frames
    # Share one transfer between student preprocessing, image generation and
    # teacher inference. Keep observed on CPU for the missing-frame planner.
    batch={key:value.to(device,non_blocking=True) if isinstance(value,torch.Tensor) and key!='observed' else value
           for key,value in batch.items()}
    inputs,truth=frame_inputs(batch,device,with_images)
    ticks=inputs['ticks']
    with torch.no_grad():
        generated=state_input_frames(teacher_frame,batch,device)
        from models import HybridStateStudent
        options={'parallel_physics':parallel_teacher_physics} if isinstance(teacher_state,HybridStateStudent) else {}
        prediction=teacher_state(generated,batch['imu'].to(device,non_blocking=True),batch['observed'].to(device,non_blocking=True),
            batch['dt'].to(device,non_blocking=True),batch['gravity'].to(device,non_blocking=True),return_corrections=True,**options)
        target={key:prediction[key][:,ticks].clone() for key in truth}
        target['omega'][:,0]=0
        correction=torch.stack([prediction['corrections'][:,lo+1:hi+1].mean(1)
                                for lo,hi in zip(ticks[:-1],ticks[1:])],1)
    return inputs,truth,target,correction


def state_loss(prediction,target,device):
    from losses import losses
    return losses(prediction,target,device,metrics=False)[0]


def train_student(name,row,payload,device,weights):
    inputs,truth,target,correction=payload
    model=row['model'];model.train()
    prediction=model(**inputs)
    if model.settings.get('kind') in ('imu_affine','visual_imu_affine'):
        intervals=torch.tensor(np.diff(inputs['ticks']),device=device)*inputs['dt'][:,None]
        acceleration=(target['vel'][:,1:]-target['vel'][:,:-1])/intervals[:,:,None]
        kd=F.smooth_l1_loss(prediction['accelerations'],acceleration,beta=.05)
    else:
        kd=F.smooth_l1_loss(prediction['corrections'][:,1:],correction,beta=.05)
    loss=(weights[0]*kd+weights[1]*state_loss(prediction,target,device)
          +weights[2]*state_loss(prediction,truth,device))
    if not torch.isfinite(loss):raise ValueError(f'Nonfinite {name} distillation loss')
    row['optimizer'].zero_grad(set_to_none=True);loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    row['optimizer'].step()
    return loss


def validate_student(name,row,payload,device):
    from performance import amp_disabled
    # Thread-local precision and grad contexts must be set inside each worker.
    with torch.no_grad(),amp_disabled():
        inputs,truth=payload[:2]
        row['model'].eval()
        prediction=row['model'](**inputs)
        if len(payload)>2:row['validation_metric'].update(prediction,payload[2])
        return state_loss(prediction,truth,device)


def validation_inputs(batch,device,with_images=True):
    inputs,truth=frame_inputs(batch,device,with_images)
    metadata=dict(pos=truth['pos'],rot=truth['rot'],
        observed=torch.ones(truth['pos'].shape[:2],dtype=torch.bool),dt=batch['dt']*10)
    for key in ('episode_id','episode_complete','camera_position_in_body','camera_calibration_available'):
        if key in batch:metadata[key]=batch[key]
    return inputs,truth,metadata


def curriculum_window(epoch,windows,stage_epochs):
    return windows[min(epoch//stage_epochs,len(windows)-1)]


def select_architecture(profile,directory,sdk=None,frames=1200):
    """Measure official layouts before training; preserve reports, discard probes."""
    import shutil
    from students.model import architecture_candidates,BudgetStudent
    from students.export import build
    from verify_limits import verify
    from viobench.tools.cli import run_with_large_stack
    reports=[]
    for index,spec in enumerate(architecture_candidates(profile)):
        path=directory/f'candidate-{index}'
        module=build(BudgetStudent(spec),path,sdk)
        result=run_with_large_stack(lambda:verify(module,frames,step_fuel=spec['fuel']))
        result['settings']=spec
        reports.append(result)
        shutil.rmtree(path)
        (directory/'architecture_search.json').write_text(json.dumps(reports,indent=2)+'\n')
        if result['passed'] and result['fuel_step_max']<=.95*spec['fuel']:return spec
    raise RuntimeError(f'No {profile} architecture passed with 5% fuel headroom; see {directory}')


def main():
    from students.model import BudgetStudent,PROFILES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path,nargs='?',default=HERE/'data')
    parser.add_argument('--teacher',type=Path,required=True,help='teacher run root (PPO-selected) or run/post_sl')
    parser.add_argument('--collection-config',type=Path,help='JSON overrides for collection settings; teacher architecture is unchanged')
    parser.add_argument('--profiles',nargs='+',choices=tuple(PROFILES),default=list(PROFILES))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--epochs',type=int,default=500,help='safety ceiling; each profile stops independently')
    parser.add_argument('--patience',type=int,default=8,help='non-improving fresh batches; epochs for fixed data')
    parser.add_argument('--window',type=int,help='fixed IMU window, overriding the default 2–20 second curriculum')
    parser.add_argument('--curriculum-windows',type=int,nargs='+',default=[401,801,1601,2401,4001])
    parser.add_argument('--stage-epochs',type=int,default=6)
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
    parser.add_argument('--parallel-teacher-physics',action=argparse.BooleanOptionalAction,default=True,
        help='parallel teacher integration in eval mode; does not enable dropout or AMP')
    parser.add_argument('--max-concurrent',type=int,help='explicit stream group size; disables automatic tuning')
    parser.add_argument('--architecture-config',type=Path,help='JSON mapping profile names to previously fuel-verified settings')
    parser.add_argument('--calibrate-architectures',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--wandb-mode',choices=['online','offline','disabled'],default='online')
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--kd-weight',type=float,default=1.)
    parser.add_argument('--teacher-state-weight',type=float,default=.5)
    parser.add_argument('--ground-truth-weight',type=float,default=1.)
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
    args=parser.parse_args()
    windows=[args.window] if args.window is not None else args.curriculum_windows
    if (not windows or windows!=sorted(set(windows)) or min(windows)<11 or
        min(args.epochs,args.patience,args.batch or 1,args.max_passes_per_batch,args.verify_frames,args.stage_epochs,args.max_concurrent or 1)<1):
        parser.error('counts must be positive and --window must be at least 11')
    if not np.isfinite(args.min_delta) or args.min_delta<0:parser.error('--min-delta must be finite and nonnegative')
    if args.max_concurrent is not None and args.max_concurrent<1:parser.error('--max-concurrent must be positive')
    weights=(args.kd_weight,args.teacher_state_weight,args.ground_truth_weight)
    if args.lr<=0 or not np.isfinite([args.lr,*weights]).all() or min(weights)<0 or sum(weights)==0:
        parser.error('learning rate must be positive and finite; loss weights must be finite/nonnegative with positive sum')
    if not (args.teacher/'training_config.json').is_file():
        parser.error('--teacher needs a complete teacher checkpoint set, including training_config.json')
    args.teacher=args.teacher.resolve()
    args.out=(args.out or HERE/'checkpoints'/('distill-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))).resolve()
    if args.out.exists():parser.error('choose a new --out directory to preserve existing models')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    frame,teacher,settings=load_teacher(args.teacher,device)
    import config as cfg
    from performance import PERFORMANCE_SETTINGS,configure,EpochTimer,record_epoch,adamw
    from streaming import COLLECTION_SETTINGS
    from distillation_pipeline import DistillationPipeline
    from functools import partial
    saved=json.loads((args.teacher/'training_config.json').read_text())
    for key,value in saved.items():
        if key.isupper() and hasattr(cfg,key) and key not in PERFORMANCE_SETTINGS|COLLECTION_SETTINGS:setattr(cfg,key,value)
    if args.collection_config:
        overrides=json.loads(args.collection_config.read_text())
        if not isinstance(overrides,dict) or set(overrides)-COLLECTION_SETTINGS:
            parser.error('--collection-config accepts only collection settings: '+', '.join(sorted(COLLECTION_SETTINGS)))
        for key,value in overrides.items():setattr(cfg,key,value)
    cfg.H,cfg.W=settings['height'],settings['width']
    cfg.USE_GENERATED_FRAMES_FOR_STATE=settings['use_generated_frames']
    cfg.VISUAL_PREDICTION=settings['visual_prediction']
    cfg.CACHE_GENERATED_WINDOWS=False
    cfg.DEVICE=device;cfg.WORKDIR=args.data.resolve();cfg.WINDOW=windows[0]
    cfg.USE_CURRICULUM=False  # This distiller owns its shared deterministic horizon schedule.
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
            args.stream_dir=Path(tempfile.gettempdir())/('hyperdrone-distill-'+user)
        if not os.environ.get('CONTA_CACHE'):
            cache_root=Path('/var/tmp') if Path('/var/tmp').is_dir() else Path(tempfile.gettempdir())
            os.environ['CONTA_CACHE']=str(cache_root/('hyperdrone-'+user)/'scene-assets')
            legacy=str(Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'conta')
            searches=os.environ.get('CONTA_SEARCH_PATHS','').split(os.pathsep)
            os.environ['CONTA_SEARCH_PATHS']=os.pathsep.join(dict.fromkeys([path for path in searches if path]+[legacy]))
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
    provenance=dict(teacher=str(args.teacher),teacher_settings=settings,
        state_sha256=hashlib.sha256((args.teacher/'state_student.pt').read_bytes()).hexdigest(),
        frame_sha256=hashlib.sha256((args.teacher/'frame_student.pt').read_bytes()).hexdigest(),
        epochs=args.epochs,curriculum_windows=windows,stage_epochs=args.stage_epochs,kd_weight=args.kd_weight,
        parallel=bool(args.parallel and args.parallel_mode!='serial' and device=='cuda'),
        target_prefetch_batches=int(args.parallel and args.parallel_mode!='serial' and device=='cuda'),
        parallel_mode=args.parallel_mode,max_concurrent=args.max_concurrent,patience=args.patience,min_delta=args.min_delta,
        parallel_teacher_physics=args.parallel_teacher_physics,
        max_data_mib=cfg.STREAM_MAX_MIB,stream_dir=str(args.stream_dir) if args.stream_dir else None,
        scene_cache=(os.environ.get('CONTA_CACHE') if args.stream_data else None),
        performance={key:getattr(cfg,key) for key in PERFORMANCE_SETTINGS},
        teacher_state_weight=args.teacher_state_weight,ground_truth_weight=args.ground_truth_weight)
    provenance['profiles']=list(dict.fromkeys(args.profiles))
    (args.out/'distillation.json').write_text(json.dumps(provenance,indent=2)+'\n')
    from streaming import TrainingData,FreshBatchPatience
    from train import save_weights
    from leaderboard_metrics import LeaderboardATE
    overrides=json.loads(args.architecture_config.read_text()) if args.architecture_config else {}
    import wandb
    run=wandb.init(project=cfg.WANDB_PROJECT,dir=str(args.out),job_type='seven_student_distillation',
        mode=args.wandb_mode,config=provenance,name=args.out.name)
    from distillation_tracking import DistillationLogger
    print(f'W&B: project={cfg.WANDB_PROJECT}, mode={args.wandb_mode}, URL={run.url}',flush=True)
    with DistillationLogger(run,args.profiles) as tracking:
        students={}
        for name in dict.fromkeys(args.profiles):
            directory=args.out/name;directory.mkdir()
            spec=dict(overrides.get(name,PROFILES[name]),profile=name)
            if spec['fuel']!=PROFILES[name]['fuel']:raise ValueError(f'{name} architecture changes its fuel cap')
            if name not in overrides and spec.get('kind')=='mamba2' and args.calibrate_architectures:
                spec=select_architecture(name,directory,args.wasi_sdk,args.verify_frames)
            model=BudgetStudent(spec).to(device,non_blocking=True)
            (directory/'student.json').write_text(json.dumps(spec,indent=2)+'\n')
            students[name]=dict(model=model,optimizer=adamw(model.parameters(),lr=args.lr),
                best=float('inf'),patience=FreshBatchPatience(),active=True,directory=directory)
            print(f'{name}: {sum(p.numel() for p in model.parameters()):,} parameters, cap {spec["fuel"]:,} fuel/frame',flush=True)
        print('Student runs:',args.out,flush=True)
        run.config.update({'student_architectures':{name:row['model'].settings for name,row in students.items()},
                           'student_parameters':{name:sum(p.numel() for p in row['model'].parameters()) for name,row in students.items()}})
        with ExitStack() as stack:
            tracking.phase('collecting_data' if args.stream_data else 'loading_data')
            data=TrainingData(args,cfg,stack)
            # Fail early for a dataset too short for the requested final horizon.
            data.ppo_loader(windows[-1],validation=True)
            parallel=args.parallel and args.parallel_mode!='serial'
            pipeline=stack.enter_context(DistillationPipeline(device,students,parallel,args.max_concurrent))
            tuned=set()
            print(f'Distillation scheduling: {"parallel CUDA streams; one target batch prefetched" if pipeline.parallel else "serial"}',flush=True)
            for epoch in range(args.epochs):
                timer=EpochTimer(device)
                tracking.phase('waiting_for_data')
                _,loader=data.next_epoch()
                length=curriculum_window(epoch,windows,args.stage_epochs)
                loader=data.ppo_loader(length)
                # Avoid forking a CUDA process with live scheduling threads. Dataset
                # objects contain CPU tensors and are safe to send to spawn workers.
                if pipeline.parallel:
                    for state_loader in (loader,data.val_episode_loader):
                        if state_loader.num_workers:state_loader.multiprocessing_context='spawn'
                active={name:row for name,row in students.items() if row['active']}
                with_images=any(row['model'].settings.get('kind')!='imu_affine' for row in active.values())
                train_losses={name:[0.,0] for name in active};validation={name:[0.,0] for name in active}
                prepare=partial(frame_targets,teacher_frame=frame,teacher_state=teacher,device=device,with_images=with_images,
                                parallel_teacher_physics=args.parallel_teacher_physics)
                update=partial(train_student,device=device,weights=weights)
                tracking.phase('training')
                with pipeline.batches(loader,prepare) as batches:
                    for batch_index,prepared in enumerate(batches,1):
                        tuning_key=(length,tuple(active))
                        if pipeline.parallel and args.parallel_mode=='auto' and args.max_concurrent is None and tuning_key not in tuned:
                            tracking.phase('tuning_schedule')
                            result=pipeline.autotune(active,prepared,update)
                            result['window']=length
                            (args.out/'scheduling.json').write_text(json.dumps(pipeline.tuning,indent=2)+'\n')
                            print('Selected distillation schedule:',json.dumps(result),flush=True)
                            run.summary['distillation/scheduling']=pipeline.tuning
                            run.summary['distillation/student_schedule_mode']=result['mode']
                            run.summary['distillation/student_groups']=result['groups']
                            tracking.phase('training')
                            tuned.add(tuning_key)
                        count=len(prepared[0][0]['imu'])
                        batch_losses=pipeline.run(active,prepared,update)
                        for name,loss in batch_losses.items():
                            train_losses[name][0]+=loss*count;train_losses[name][1]+=count
                        timer.add(count)
                        tracking.progress(epoch+1,batch_index,len(loader),length,batch_losses,pipeline.max_concurrent)
                for row in active.values():row['validation_metric']=LeaderboardATE()
                tracking.phase('validating')
                prepare=partial(validation_inputs,device=device,with_images=with_images)
                with pipeline.batches(data.val_episode_loader,prepare) as batches:
                    for prepared in batches:
                        count=len(prepared[0][0]['imu'])
                        for name,loss in pipeline.run(active,prepared,partial(validate_student,device=device)).items():
                            validation[name][0]+=loss*count;validation[name][1]+=count
                measurement=timer.result()
                measurement.update(active_profiles=list(active),parallel=pipeline.parallel,
                                   student_updates=timer.samples*len(active),window=length,max_concurrent=pipeline.max_concurrent)
                record_epoch(args.out,'distillation',epoch+1,measurement,data)
                print(f'Distillation throughput: {measurement["samples_per_second"]:.2f} windows/s '
                      f'({len(active)} active profiles; includes validation/data wait)',flush=True)
                epoch_records={}
                for name,row in active.items():
                    score=row['validation_metric'].result()['mean_ate_m']
                    if not np.isfinite(score):raise ValueError(f'Nonfinite {name} validation loss')
                    improved=score<row['best']-args.min_delta
                    if improved:
                        row['best']=score;save_weights(row['model'].state_dict(),row['directory']/'student.pt')
                    final_stage=length==windows[-1]
                    bad=row['patience'].update(improved or not final_stage,data.batch_id)
                    row['active']=bad<args.patience
                    record=dict(epoch=epoch+1,train_loss=train_losses[name][0]/train_losses[name][1],
                        validation_state_loss=validation[name][0]/validation[name][1],
                        validation_mean_ate_m=score,best_validation_mean_ate_m=row['best'],window=length,
                        learning_rate=row['optimizer'].param_groups[0]['lr'],
                        non_improving_data_cycles=bad,recording_batch=data.batch_id,active=row['active'])
                    with (row['directory']/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                    epoch_records[name]=record
                    print(f'Distill {name} {epoch+1}/{args.epochs}: train={record["train_loss"]:.6g}, ATE={score:.6g} m, stale cycles={bad}',flush=True)
                tracking.epoch(epoch+1,epoch_records,measurement,data.rolling.last_wait_seconds if data.rolling else 0.)
                if not any(row['active'] for row in students.values()):break
        for name,row in students.items():
            tracking.phase('saving_checkpoint' if args.skip_export else 'exporting:'+name)
            row['model'].load_state_dict(torch.load(row['directory']/'student.pt',map_location=device,weights_only=True))
            run.summary[name+'/checkpoint']=str(row['directory']/'student.pt')
            run.summary[name+'/stop_reason']='epoch_limit' if row['active'] else 'early_stopping'
            if not args.skip_export:
                from students.export import export_student
                module=export_student(row['model'].eval(),row['directory']/'native',args.wasi_sdk,args.verify_frames)
                report=json.loads((module.parent/'limits.json').read_text())
                run.summary[name+'/wasm']=str(module)
                run.summary[name+'/fuel_step_max']=report['fuel_step_max']
                run.summary[name+'/status']='exported'
            else:run.summary[name+'/status']='checkpoint_saved'
        print('Distillation complete; checkpoints:',args.out,flush=True)


if __name__=='__main__':
    main()
