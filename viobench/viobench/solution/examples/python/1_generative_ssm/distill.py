"""Distill an SL/PPO teacher checkpoint into native 100/10K/1M/100M fuel students."""
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


def frame_targets(batch,teacher_frame,teacher_state,device,with_images=True):
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
        prediction=teacher_state(generated,batch['imu'].to(device,non_blocking=True),batch['observed'].to(device,non_blocking=True),
            batch['dt'].to(device,non_blocking=True),batch['gravity'].to(device,non_blocking=True),return_corrections=True)
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
    if model.settings.get('kind')=='imu_affine':
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
        inputs,truth=payload
        row['model'].eval()
        return state_loss(row['model'](**inputs),truth,device)


def main():
    from students.model import BudgetStudent,PROFILES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path,nargs='?',default=HERE/'data')
    parser.add_argument('--teacher',type=Path,required=True,help='teacher run root (PPO-selected) or run/post_sl')
    parser.add_argument('--collection-config',type=Path,help='JSON overrides for collection settings; teacher architecture is unchanged')
    parser.add_argument('--profiles',nargs='+',choices=tuple(PROFILES),default=list(PROFILES))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=8,help='non-improving fresh batches; epochs for fixed data')
    parser.add_argument('--window',type=int,default=101)
    parser.add_argument('--batch',type=int,help='state minibatch; default 4')
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
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--kd-weight',type=float,default=1.)
    parser.add_argument('--teacher-state-weight',type=float,default=.5)
    parser.add_argument('--ground-truth-weight',type=float,default=1.)
    parser.add_argument('--device',choices=['cpu','cuda'])
    parser.add_argument('--stream-data',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--max-data-mib',type=float,help='recording budget; default 3072 MiB or --collection-config value')
    parser.add_argument('--max-passes-per-batch',type=int,default=3)
    parser.add_argument('--stream-dir',type=Path)
    parser.add_argument('--collector',type=Path)
    parser.add_argument('--skip-export',action='store_true',help='save student checkpoints; export later with python -m students.export')
    parser.add_argument('--verify-frames',type=int,default=1200)
    parser.add_argument('--wasi-sdk',type=Path)
    parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args()
    if min(args.epochs,args.patience,args.batch or 1,args.max_passes_per_batch,args.verify_frames)<1 or args.window<11:
        parser.error('counts must be positive and --window must be at least 11')
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
    cfg.DEVICE=device;cfg.WORKDIR=args.data.resolve();cfg.WINDOW=args.window
    cfg.SEED=args.seed;cfg.STREAM_DATA=args.stream_data
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
        epochs=args.epochs,window=args.window,kd_weight=args.kd_weight,
        parallel=bool(args.parallel and device=='cuda'),target_prefetch_batches=int(args.parallel and device=='cuda'),
        performance={key:getattr(cfg,key) for key in PERFORMANCE_SETTINGS},
        teacher_state_weight=args.teacher_state_weight,ground_truth_weight=args.ground_truth_weight)
    (args.out/'distillation.json').write_text(json.dumps(provenance,indent=2)+'\n')
    from streaming import TrainingData,FreshBatchPatience
    from train import save_weights
    students={}
    for name in dict.fromkeys(args.profiles):
        spec=dict(PROFILES[name],profile=name)
        model=BudgetStudent(spec).to(device,non_blocking=True)
        directory=args.out/name;directory.mkdir()
        (directory/'student.json').write_text(json.dumps(spec,indent=2)+'\n')
        students[name]=dict(model=model,optimizer=adamw(model.parameters(),lr=args.lr),
            best=float('inf'),patience=FreshBatchPatience(),active=True,directory=directory)
        print(f'{name}: {sum(p.numel() for p in model.parameters()):,} parameters, cap {spec["fuel"]:,} fuel/frame',flush=True)
    print('Student runs:',args.out,flush=True)
    with ExitStack() as stack:
        data=TrainingData(args,cfg,stack)
        pipeline=stack.enter_context(DistillationPipeline(device,students,args.parallel))
        print(f'Distillation scheduling: {"parallel CUDA streams; one target batch prefetched" if pipeline.parallel else "serial"}',flush=True)
        for epoch in range(args.epochs):
            timer=EpochTimer(device)
            _,loader=data.next_epoch()
            # Avoid forking a CUDA process with live scheduling threads. Dataset
            # objects contain CPU tensors and are safe to send to spawn workers.
            if pipeline.parallel:
                for state_loader in (loader,data.val_state_loader):
                    if state_loader.num_workers:state_loader.multiprocessing_context='spawn'
            active={name:row for name,row in students.items() if row['active']}
            with_images=any(row['model'].settings.get('kind')!='imu_affine' for row in active.values())
            train_losses={name:[0.,0] for name in active};validation={name:[0.,0] for name in active}
            prepare=partial(frame_targets,teacher_frame=frame,teacher_state=teacher,device=device,with_images=with_images)
            update=partial(train_student,device=device,weights=weights)
            with pipeline.batches(loader,prepare) as batches:
                for prepared in batches:
                    count=len(prepared[0][0]['imu'])
                    for name,loss in pipeline.run(active,prepared,update).items():
                        train_losses[name][0]+=loss*count;train_losses[name][1]+=count
                    timer.add(count)
            prepare=partial(frame_inputs,device=device,with_images=with_images)
            with pipeline.batches(data.val_state_loader,prepare) as batches:
                for prepared in batches:
                    count=len(prepared[0][0]['imu'])
                    for name,loss in pipeline.run(active,prepared,partial(validate_student,device=device)).items():
                        validation[name][0]+=loss*count;validation[name][1]+=count
            measurement=timer.result()
            measurement.update(active_profiles=list(active),parallel=pipeline.parallel,
                               student_updates=timer.samples*len(active))
            record_epoch(args.out,'distillation',epoch+1,measurement,data)
            print(f'Distillation throughput: {measurement["samples_per_second"]:.2f} windows/s '
                  f'({len(active)} active profiles; includes validation/data wait)',flush=True)
            for name,row in active.items():
                score=validation[name][0]/validation[name][1]
                if not np.isfinite(score):raise ValueError(f'Nonfinite {name} validation loss')
                improved=score<row['best']-1e-5
                if improved:
                    row['best']=score;save_weights(row['model'].state_dict(),row['directory']/'student.pt')
                bad=row['patience'].update(improved,data.batch_id)
                row['active']=bad<args.patience
                record=dict(epoch=epoch+1,train_loss=train_losses[name][0]/train_losses[name][1],
                    validation_state_loss=score,best_validation_state_loss=row['best'],
                    non_improving_data_cycles=bad,recording_batch=data.batch_id,active=row['active'])
                with (row['directory']/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                print(f'Distill {name} {epoch+1}/{args.epochs}: train={record["train_loss"]:.6g}, val={score:.6g}, stale cycles={bad}',flush=True)
            if not any(row['active'] for row in students.values()):break
    for row in students.values():
        row['model'].load_state_dict(torch.load(row['directory']/'student.pt',map_location=device,weights_only=True))
        if not args.skip_export:
            from students.export import export_student
            export_student(row['model'].eval(),row['directory']/'native',args.wasi_sdk,args.verify_frames)
    print('Distillation complete; checkpoints:',args.out,flush=True)


if __name__=='__main__':
    main()
