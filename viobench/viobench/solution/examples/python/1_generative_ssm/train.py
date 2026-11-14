"""Train the notebook's frame generator and state student on recorded rollouts."""
import argparse
import json
import random
import os
import shutil
import tempfile
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path


def save_weights(weights, path):
    """Replace a checkpoint atomically so interruption preserves the old file."""
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(weights, temporary)
    temporary.replace(path)


def load_weights(model, path, device, allow_new_patch_stems=False):
    """Preserve old checkpoints; SL may initialize a newly shaped patch stem."""
    import torch
    weights = torch.load(path, map_location=device, weights_only=True)
    expected = model.state_dict()
    for key in ('enc1.0.weight','visual.cnn.0.weight','encoder.cnn.0.weight','target_encoder.cnn.0.weight'):
        if key not in weights or key not in expected or weights[key].shape == expected[key].shape:
            continue
        if weights[key].shape[:2] != expected[key].shape[:2]:
            continue  # A changed width needs a fresh model, diagnosed below.
        if not allow_new_patch_stems:
            raise ValueError(f'{path}: patch stem {key} changed shape; run supervised training before PPO-only')
        print(f'Warm start: initializing new learned patch stem {key}; other weights retained.', flush=True)
        weights[key] = expected[key]
    changed = [key for key in weights if key in expected and weights[key].shape != expected[key].shape]
    if changed:
        raise ValueError(f'{path}: model width/architecture differs at {changed[0]}; '
                         'start a fresh training run without --from-checkpoints for a new model width')
    model.load_state_dict(weights, strict=True)


def save_post_sl(args, cfg, student):
    """Publish an exportable SL snapshot before any PPO updates can start."""
    from export import notebook_settings
    baseline=args.out/'state_student_sl.pt'
    save_weights(student.state_dict(),baseline)
    target=args.out/'post_sl'
    if target.exists():
        raise FileExistsError(f'{target} already exists; choose a new --out to preserve its SL weights')
    # Weight writes use atomic replacement, so hard links preserve the old SL
    # inode when the selected root checkpoint is replaced after PPO. Fall back
    # to copying on filesystems that do not support links.
    with tempfile.TemporaryDirectory(prefix='.post-sl-',dir=args.out) as temp:
        staged=Path(temp)/'checkpoint'; staged.mkdir()
        for source,name in ((Path(cfg.FRAME_CHECKPOINT),'frame_student.pt'),
                            (baseline,'state_student.pt')):
            try:
                os.link(source,staged/name)
            except OSError:
                shutil.copy2(source,staged/name)
        (staged/'student_config.json').write_text(json.dumps(notebook_settings(cfg),indent=2)+'\n')
        shutil.copy2(args.out/'training_config.json',staged/'training_config.json')
        (staged/'stage.json').write_text(json.dumps(dict(stage='post_sl',source_run=str(args.out)),indent=2)+'\n')
        staged.rename(target)
    print(f'Post-SL checkpoints ready for export: {target}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path, nargs='?', default=Path(__file__).resolve().parent/'data', help='directory of rollout_s*_e*.npz files')
    parser.add_argument('--config', type=Path, help='JSON overrides for config.py uppercase settings')
    parser.add_argument('--out', type=Path, help='checkpoint directory; default: checkpoints/<run timestamp>')
    parser.add_argument('--solution-out', type=Path, default=Path(__file__).resolve().parent / 'solution')
    parser.add_argument('--skip-export', action='store_true', help='save teacher checkpoints without ONNX/guest export')
    parser.add_argument('--verify-limits', action='store_true', help='bundle the trained model and verify server resource caps')
    parser.add_argument('--verify-frames', type=int, default=1200, help='resource-test frames; default 1200 = 60 simulated seconds')
    parser.add_argument('--epochs-frame', type=int)
    parser.add_argument('--epochs-state', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--amp',action=argparse.BooleanOptionalAction,default=None,help='BF16 neural layers on CUDA; physics and validation stay FP32')
    parser.add_argument('--visual-prediction',choices=['embedding','rgb'],help='missing-tick visual representation; new runs default to embedding')
    parser.add_argument('--embedding-chunk',type=int,help='missing embeddings predicted per microbatch; default 1024')
    parser.add_argument('--batch',type=int,help='state training sequences per minibatch')
    parser.add_argument('--frame-pair-batch',type=int)
    parser.add_argument('--frame-chunk',type=int,help='missing-frame generation microbatch')
    parser.add_argument('--visual-chunk',type=int)
    parser.add_argument('--collect-instances',type=int,help='parallel simulator flights in each training scene')
    parser.add_argument('--loader-workers',type=int,help='CPU workers for CUDA state windows; default 2; frame pairs use FRAME_LOADER_WORKERS=0')
    parser.add_argument('--persistent-workers',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--fused-adamw',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--wandb-mode', choices=['disabled', 'offline', 'online'],
                        help='W&B mode; defaults to online unless overridden in --config')
    parser.add_argument('--stream-data', action=argparse.BooleanOptionalAction, default=None,
                        help='collect fresh batches in the background, pruning managed old batches')
    parser.add_argument('--stream-dir', type=Path, help='temporary managed data root; checkpoints stay in --out')
    parser.add_argument('--max-data-mib', type=float, help='total rolling-data budget, including validation')
    parser.add_argument('--max-passes-per-batch', type=int,
                        help='maximum SL epochs/PPO iterations on one recording batch; default 3, including first use')
    parser.add_argument('--collector', type=Path, help='alternate collector script with collect.py CLI')
    parser.add_argument('--ppo', action=argparse.BooleanOptionalAction, default=None,
                        help='enable/disable correction-head PPO; enabled by default')
    parser.add_argument('--sl', action=argparse.BooleanOptionalAction, default=None,
                        help='enable/skip SL; default on for fresh runs, off for --from-checkpoints with PPO')
    parser.add_argument('--ppo-only', action='store_true', help='skip SL and refine --from-checkpoints with PPO')
    parser.add_argument('--skip-embedding-training', action='store_true',
                        help='freeze the saved embedding model and start at state SL; requires --from-checkpoints')
    parser.add_argument('--from-checkpoints', type=Path,
                        help='warm-start saved weights/settings; epoch caps use current defaults unless explicitly overridden')
    args = parser.parse_args()
    if args.max_passes_per_batch is not None and args.max_passes_per_batch < 1:
        parser.error('--max-passes-per-batch must be positive')
    if args.verify_limits and (args.skip_export or args.verify_frames < 1):
        parser.error('--verify-limits requires export and a positive --verify-frames')
    if args.out is None:
        args.out = Path(__file__).resolve().parent / 'checkpoints' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    args.out = args.out.resolve()
    import numpy as np
    import torch
    import config as cfg
    from performance import PERFORMANCE_SETTINGS,configure
    from models import RETIRED_MAMBA_SETTINGS,require_official_mamba
    from streaming import COLLECTION_SETTINGS
    retired_settings=RETIRED_MAMBA_SETTINGS|{'COLLECT','EPISODES_PER_SCENE'}
    overrides = {}
    source_settings = {}
    if args.from_checkpoints:
        source_config = args.from_checkpoints/'training_config.json'
        if not source_config.is_file():
            parser.error('--from-checkpoints needs a train.py run with training_config.json')
        source_settings = json.loads(source_config.read_text())
        # Retain the parameter shapes of checkpoints written before patch stems.
        source_settings.setdefault('FRAME_INPUT_STRIDE',1)
        source_settings.setdefault('VISUAL_INPUT_STRIDE',2)
        source_settings.setdefault('VISUAL_PREDICTION','rgb')
        # Changing fresh-run defaults must preserve legacy checkpoint backbones.
        if 'USE_HYBRID_SSM_TRANSFORMER' not in source_settings:
            metadata=args.from_checkpoints/'student_config.json'
            saved_model=json.loads(metadata.read_text()) if metadata.is_file() else {}
            source_settings['USE_HYBRID_SSM_TRANSFORMER']=(
                saved_model.get('backbone',saved_model.get('student_backbone','gru'))=='mamba2_hybrid')
        # Current defaults choose collection coverage, epoch budgets, PPO and W&B;
        # explicit experiment JSON and CLI settings can still override them.
        overrides.update({key:value for key,value in source_settings.items()
                          if key not in PERFORMANCE_SETTINGS|COLLECTION_SETTINGS|retired_settings|{'EPOCHS_FRAME','EPOCHS_STUDENT','USE_PPO','WANDB_MODE'}})
    if args.config:
        overrides.update(json.loads(args.config.read_text()))
    if overrides:
        for key, value in overrides.items():
            if key in RETIRED_MAMBA_SETTINGS:
                parser.error(f'{key} was removed: official Mamba-2 is the only Mamba implementation')
            if not key.isupper() or not hasattr(cfg, key):
                parser.error(f'unknown training setting: {key}')
            setattr(cfg, key, value)
    if args.visual_prediction is not None:cfg.VISUAL_PREDICTION=args.visual_prediction
    if cfg.VISUAL_PREDICTION not in ('embedding','rgb'):parser.error('VISUAL_PREDICTION must be embedding or rgb')
    if args.from_checkpoints and cfg.VISUAL_PREDICTION!=source_settings['VISUAL_PREDICTION']:
        parser.error('changing visual prediction mode requires a fresh run without --from-checkpoints')
    if type(cfg.EMBEDDING_DIM) is not int or cfg.EMBEDDING_DIM<1:
        parser.error('EMBEDDING_DIM must be a positive integer')
    if cfg.EMBEDDING_DIM!=48 and (cfg.VISUAL_PREDICTION!='embedding' or not cfg.USE_LEARNED_VISUAL_ENCODER):
        parser.error('Non-48 embeddings require embedding mode and a learned visual encoder')
    if args.from_checkpoints and cfg.EMBEDDING_DIM!=source_settings.get('EMBEDDING_DIM',48):
        parser.error('changing embedding dimension requires a fresh run without --from-checkpoints')
    if (cfg.EMBEDDING_HIDDEN<1 or not 0<=cfg.EMBEDDING_EMA<1
            or not np.isfinite([cfg.EMBEDDING_EMA,cfg.EMBEDDING_STD_TARGET,cfg.EMBEDDING_VARIANCE_WEIGHT,cfg.EMBEDDING_COVARIANCE_WEIGHT]).all()
            or min(cfg.EMBEDDING_STD_TARGET,cfg.EMBEDDING_VARIANCE_WEIGHT,cfg.EMBEDDING_COVARIANCE_WEIGHT)<0):
        parser.error('embedding hidden width must be positive; EMA must be in [0,1); regularization settings must be finite/nonnegative')
    if cfg.VISUAL_PREDICTION=='embedding':cfg.CACHE_GENERATED_WINDOWS=False
    if 'STUDENT_HIDDEN' not in overrides:
        cfg.STUDENT_HIDDEN = (cfg.HYBRID_STUDENT_HIDDEN if cfg.USE_HYBRID_SSM_TRANSFORMER else
                              cfg.LARGE_STATE_HIDDEN if cfg.USE_LARGER_STATE_STUDENT else 32)
    if 'FRAME_MAX_FLOW_PIXELS' not in overrides:
        cfg.FRAME_MAX_FLOW_PIXELS = 24.0 * cfg.W / 96.0
    if args.ppo_only and (args.sl is True or args.ppo is False):
        parser.error('--ppo-only cannot use --sl or --no-ppo')
    if args.skip_embedding_training:
        if args.ppo_only or args.sl is False:
            parser.error('--skip-embedding-training starts at state SL; cannot use --ppo-only or --no-sl')
        if args.from_checkpoints is None or not (args.from_checkpoints/'frame_student.pt').is_file():
            parser.error('--skip-embedding-training requires --from-checkpoints with frame_student.pt')
        if cfg.VISUAL_PREDICTION!='embedding':
            parser.error('--skip-embedding-training requires an embedding-mode checkpoint')
    if args.from_checkpoints and args.out == args.from_checkpoints.resolve():
        parser.error('choose a new --out directory to preserve the source checkpoints')
    if args.stream_data is not None: cfg.STREAM_DATA = args.stream_data
    if args.ppo is not None: cfg.USE_PPO = args.ppo
    if args.ppo_only: cfg.USE_PPO = True
    use_sl=args.sl if args.sl is not None else (args.skip_embedding_training or args.from_checkpoints is None or not cfg.USE_PPO)
    if args.ppo_only: use_sl=False
    args.ppo_only=not use_sl
    if args.ppo_only:
        if args.from_checkpoints is None or not cfg.USE_PPO:
            parser.error('skipping SL requires --from-checkpoints and PPO enabled')
        missing=[name for name in ('frame_student.pt','state_student.pt')
                 if not (args.from_checkpoints/name).is_file()]
        if missing:
            parser.error('PPO-only needs both trained components; missing '+', '.join(missing)+
                         '. Use --sl to train missing components before PPO.')
    elif (args.out/'post_sl').exists():
        parser.error('choose a new --out directory to preserve its existing post_sl snapshot')
    if args.max_data_mib is not None: cfg.STREAM_MAX_MIB = args.max_data_mib
    if args.max_passes_per_batch is not None: cfg.STREAM_MAX_PASSES_PER_BATCH = args.max_passes_per_batch
    if cfg.STREAM_DATA: cfg.CACHE_GENERATED_WINDOWS = False
    cfg.WORKDIR = args.data.resolve()
    cfg.DEVICE = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.USE_HYBRID_SSM_TRANSFORMER:
        try:require_official_mamba(cfg.DEVICE)
        except (ImportError,ValueError) as error:parser.error(str(error))
    for option,key in [('amp','USE_AMP'),('batch','BATCH'),('frame_pair_batch','FRAME_PAIR_BATCH'),
                       ('frame_chunk','FRAME_GENERATION_CHUNK'),('visual_chunk','VISUAL_CHUNK'),
                       ('collect_instances','COLLECT_INSTANCES'),('loader_workers','LOADER_WORKERS'),
                       ('persistent_workers','PERSISTENT_WORKERS'),('fused_adamw','FUSED_ADAMW'),('embedding_chunk','EMBEDDING_PREDICTION_CHUNK')]:
        if getattr(args,option) is not None:setattr(cfg,key,getattr(args,option))
    configure(cfg)
    from curriculum import validate_config
    try:validate_config(cfg)
    except ValueError as error:parser.error(str(error))
    if cfg.USE_CURRICULUM:cfg.CACHE_GENERATED_WINDOWS=False
    if args.wandb_mode is not None:cfg.WANDB_MODE = args.wandb_mode
    if cfg.WANDB_MODE not in ('disabled','offline','online'):parser.error('WANDB_MODE must be online, offline or disabled')
    if args.epochs_frame is not None: cfg.EPOCHS_FRAME = args.epochs_frame
    if args.epochs_state is not None: cfg.EPOCHS_STUDENT = args.epochs_state
    if min(cfg.EPOCHS_FRAME, cfg.EPOCHS_STUDENT) < 1:
        parser.error('frame and state epoch counts must be positive')
    cfg.BACKBONE_NAME = 'mamba2_hybrid' if cfg.USE_HYBRID_SSM_TRANSFORMER else 'gru'
    cfg.FRAME_MODEL_NAME = 'embedding_predictor' if cfg.VISUAL_PREDICTION=='embedding' else ('flow_generative' if cfg.USE_GENERATIVE_FRAME_MODEL else 'affine')
    cfg.TEACHER_BACKBONE = 'mamba_hybrid' if cfg.USE_HYBRID_SSM_TRANSFORMER else 'gru'
    if cfg.USE_HYBRID_SSM_TRANSFORMER:
        from export import notebook_settings,normalize_settings
        try:normalize_settings(notebook_settings(cfg))
        except ValueError as error:parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    cfg.FRAME_CHECKPOINT = str(args.out / 'frame_student.pt')
    cfg.STATE_CHECKPOINT = str(args.out / 'state_student.pt')
    cfg.TEACHER_CHECKPOINT = str(args.out / 'teacher.pt')
    from export import notebook_settings
    (args.out / 'student_config.json').write_text(json.dumps(notebook_settings(cfg), indent=2) + '\n')
    (args.out / 'training_config.json').write_text(json.dumps(
        {key: str(value) if isinstance(value, Path) else value
         for key, value in vars(cfg).items() if key.isupper()}, indent=2) + '\n')
    print(f'Checkpoints and configuration: {args.out}', flush=True)
    print(f'Visual prediction: {cfg.VISUAL_PREDICTION}',flush=True)
    print('Training stages: '+('SL + PPO' if use_sl and cfg.USE_PPO else 'SL only' if use_sl else 'PPO only'),flush=True)
    if args.skip_embedding_training:
        print('Embedding SL skipped: saved embedding weights stay frozen; starting at state SL.',flush=True)
    print(f'SL epoch limits: frame={cfg.EPOCHS_FRAME}, state={cfg.EPOCHS_STUDENT}; '
          'validation early stopping enabled.', flush=True)
    random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    with ExitStack() as stack:
        run_training(args, cfg, stack)


def run_training(args, cfg, stack):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import wandb
    from models import GenerativeFlowPredictor, MotionWarp, StateStudent, HybridStateStudent, StateTeacher,EmbeddingPredictor
    from data import state_input_frames,frame_pair_images
    from losses import losses, mean_ate
    from streaming import FreshBatchPatience, TrainingData
    from performance import amp_disabled,EpochTimer,record_epoch,adamw,timed_batches,pipeline_log
    data = TrainingData(args, cfg, stack)
    val_pair_loader, val_state_loader = data.val_pair_loader, data.val_state_loader
    print(f'Training windows: {len(data.train_windows)}; fixed validation windows: {len(data.val_windows)}', flush=True)


    frame_student=(EmbeddingPredictor(hidden=cfg.EMBEDDING_HIDDEN,learned_visual=cfg.USE_LEARNED_VISUAL_ENCODER,
                                         visual_input_stride=cfg.VISUAL_INPUT_STRIDE,ema=cfg.EMBEDDING_EMA,
                                         embedding_dim=cfg.EMBEDDING_DIM)
                   if cfg.VISUAL_PREDICTION=='embedding' else
                   GenerativeFlowPredictor(base=cfg.FRAME_GENERATOR_BASE, max_flow_pixels=cfg.FRAME_MAX_FLOW_PIXELS,
                                         input_stride=cfg.FRAME_INPUT_STRIDE)
                   if cfg.USE_GENERATIVE_FRAME_MODEL else MotionWarp(hidden=24)).to(cfg.DEVICE,non_blocking=True)
    print('Frame architecture:',cfg.FRAME_MODEL_NAME,
          'parameters:',sum(p.numel() for p in frame_student.parameters()))
    if args.from_checkpoints and (args.from_checkpoints/'frame_student.pt').is_file():
        load_weights(frame_student, args.from_checkpoints/'frame_student.pt', cfg.DEVICE,
                     allow_new_patch_stems=not (args.ppo_only or args.skip_embedding_training))
        save_weights(frame_student.state_dict(), cfg.FRAME_CHECKPOINT)
    elif args.from_checkpoints:
        print('Frame checkpoint missing; initializing the frame model for supervised training.',flush=True)
    if args.skip_embedding_training:
        frame_student.eval()
        print('Using the trained embedding predictor and encoders without further embedding updates.',flush=True)
    elif cfg.VISUAL_PREDICTION=='embedding':
        from embedding_training import train_embeddings
        train_embeddings(frame_student,data,args,cfg,save_weights)
    else:
        frame_opt=adamw(frame_student.parameters(),lr=cfg.LR_FRAME)
        frame_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            frame_opt, mode='min', factor=cfg.LR_SCHEDULER_FACTOR, patience=cfg.LR_SCHEDULER_PATIENCE)

        run=wandb.init(project=cfg.WANDB_PROJECT,dir=str(args.out),job_type='real_frame_student',mode=cfg.WANDB_MODE,
            config=dict(source='hyperdrone-real',resolution=[cfg.H,cfg.W],max_epochs=cfg.EPOCHS_FRAME,
                lr=cfg.LR_FRAME,patience=cfg.PATIENCE_FRAME,native_capture='actual',
                student=cfg.FRAME_MODEL_NAME,generator_base=cfg.FRAME_GENERATOR_BASE if cfg.USE_GENERATIVE_FRAME_MODEL else 0,
                training_labels='observed native camera frames (not 200Hz GT)' ))
        best_frame=float('inf');bad_frame_epochs=0;best_frame_epoch=0
        frame_patience=FreshBatchPatience()
        data.begin_phase('frame')
        for epoch in range(0 if args.ppo_only else cfg.EPOCHS_FRAME):
            train_pair_loader, train_state_loader = data.next_epoch()
            timer=EpochTimer(cfg.DEVICE)
            frame_student.train();values=[];total_losses=[]
            for b in timed_batches(train_pair_loader,timer):
                previous,anc,tgt=frame_pair_images(b,cfg.DEVICE)
                generator_kwargs = dict(
                    previous=previous, anchor=anc,
                    imu_mean=b['imu'].to(cfg.DEVICE,non_blocking=True), elapsed=b['elapsed'].to(cfg.DEVICE,non_blocking=True),
                    prev_elapsed=b['prev_elapsed'].to(cfg.DEVICE,non_blocking=True))
                if cfg.USE_GENERATIVE_FRAME_MODEL and cfg.USE_FLOW_SMOOTHNESS_LOSS:
                    pred, aux = frame_student(**generator_kwargs, return_aux=True)
                else:
                    pred = frame_student(**generator_kwargs)
                l1 = F.l1_loss(pred, tgt)
                data.observe_errors(b,(pred.detach()-tgt).abs().mean((1,2,3)))
                loss = l1
                if cfg.USE_FRAME_EDGE_LOSS:
                    dx = (pred[:,:,:,1:] - pred[:,:,:,:-1]) - (tgt[:,:,:,1:] - tgt[:,:,:,:-1])
                    dy = (pred[:,:,1:,:] - pred[:,:,:-1,:]) - (tgt[:,:,1:,:] - tgt[:,:,:-1,:])
                    loss = loss + cfg.FRAME_EDGE_LOSS_WEIGHT*(dx.abs().mean()+dy.abs().mean())
                if cfg.USE_GENERATIVE_FRAME_MODEL and cfg.USE_FLOW_SMOOTHNESS_LOSS:
                    flow = aux['flow_pixels']
                    smooth_x = (flow[:,:,:,1:]-flow[:,:,:,:-1]).abs().mean()
                    smooth_y = (flow[:,:,1:,:]-flow[:,:,:-1,:]).abs().mean()
                    loss = loss + cfg.FRAME_FLOW_SMOOTHNESS_WEIGHT*(smooth_x+smooth_y)
                frame_opt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(frame_student.parameters(),1.)
                frame_opt.step()
                n=len(anc);values.append(l1.detach()*n);total_losses.append(loss.detach()*n);timer.add(n)

            frame_student.eval();vals=[];val_mse=[];validation_count=0
            performance=timer.result()
            performance=record_epoch(args.out,'frame',epoch+1,performance,data,loader=train_pair_loader)
            with torch.no_grad(),amp_disabled():
                for b in val_pair_loader:
                    previous,anchor,tgt=frame_pair_images(b,cfg.DEVICE)
                    p=frame_student(previous,anchor,
                                    b['imu'].to(cfg.DEVICE,non_blocking=True),b['elapsed'].to(cfg.DEVICE,non_blocking=True),
                                    b['prev_elapsed'].to(cfg.DEVICE,non_blocking=True))
                    n=tgt.shape[0]
                    vals.append(F.l1_loss(p,tgt).item()*n)
                    val_mse.append(F.mse_loss(p,tgt).item()*n)
                    validation_count+=n
            train_l1=float(torch.stack(values).sum()/performance['samples'])
            val_l1=float(sum(vals)/validation_count)
            mean_mse=float(sum(val_mse)/validation_count)
            val_psnr=float(-10*np.log10(max(mean_mse,1e-12)))  # RGB values in [0,1]
            frame_scheduler.step(val_l1)
            def frame_probe(loader):
                total,count=0.,0
                with torch.no_grad(),amp_disabled():
                    for batch in loader:
                        previous,anchor,target=frame_pair_images(batch,cfg.DEVICE)
                        prediction=frame_student(previous,
                            anchor,batch['imu'].to(cfg.DEVICE,non_blocking=True),
                            batch['elapsed'].to(cfg.DEVICE,non_blocking=True),batch['prev_elapsed'].to(cfg.DEVICE,non_blocking=True))
                        n=len(target);total+=float(F.l1_loss(prediction,target))*n;count+=n
                return total/count
            promoted=data.finish_curriculum_epoch(val_l1,frame_probe)
            improved=(val_l1 < best_frame)  # retain the true lowest validation L1
            if improved:
                best_frame=val_l1;bad_frame_epochs=0;best_frame_epoch=epoch+1
                save_weights(frame_student.state_dict(),cfg.FRAME_CHECKPOINT)
            bad_frame_epochs=data.update_patience(frame_patience,improved,promoted)
            run.log({'frame/train_l1':train_l1,'frame/train_total':float(torch.stack(total_losses).sum()/performance['samples']),'frame/val_l1':val_l1,
                     'frame/train_pairs_per_second':performance['samples_per_second'],
                     'frame/val_psnr_db':val_psnr,'frame/val_mse':mean_mse,
                     'frame/lr':frame_opt.param_groups[0]['lr'],'frame/epoch':epoch+1,
                     'frame/best_val_l1':best_frame,'frame/non_improving_data_cycles':bad_frame_epochs,
                     'frame/early_stopping_grace_remaining':frame_patience.grace_remaining,
                     'frame/train_seconds':performance['seconds'],
                     'frame/loader_wait_seconds':performance['loader_wait_seconds']}|data.curriculum_log()|pipeline_log(performance))
            print(f'Frame {epoch+1}/{cfg.EPOCHS_FRAME}: train L1={train_l1:.5f}, '
                  f'val L1={val_l1:.5f}, PSNR={val_psnr:.2f} dB, '
                  f'LR={frame_opt.param_groups[0]["lr"]:.2e}')
            print(f'Frame throughput: {performance["samples_per_second"]:.2f} pairs/s',flush=True)
            if bad_frame_epochs>=cfg.PATIENCE_FRAME:
                print(f'Frame model early stopped at epoch {epoch+1}; best epoch={best_frame_epoch}; '
                      f'non-improving data cycles={bad_frame_epochs}')
                break

        # Use the actual BEST checkpoint for the W&B preview and for all later training.
        frame_student.load_state_dict(torch.load(cfg.FRAME_CHECKPOINT,map_location=cfg.DEVICE,weights_only=True))
        frame_student.eval()
        if cfg.WANDB_MODE != 'disabled' and not args.ppo_only:
            with torch.no_grad(),amp_disabled():
                preview_batch=next(iter(val_pair_loader))
                previous,anchor,target=(image[:1] for image in frame_pair_images(preview_batch,cfg.DEVICE))
                prediction=frame_student(previous,anchor,
                                         preview_batch['imu'][:1].to(cfg.DEVICE,non_blocking=True),
                                         preview_batch['elapsed'][:1].to(cfg.DEVICE,non_blocking=True),
                                         preview_batch['prev_elapsed'][:1].to(cfg.DEVICE,non_blocking=True))
                montage=torch.cat([anchor[0],prediction[0],target[0]],dim=-1)
                run.log({'frame/example_anchor_pred_gt':wandb.Image(montage.cpu())})
                if cfg.USE_GENERATIVE_FRAME_MODEL:
                    _,aux=frame_student(previous,anchor,
                                        preview_batch['imu'][:1].to(cfg.DEVICE,non_blocking=True),
                                        preview_batch['elapsed'][:1].to(cfg.DEVICE,non_blocking=True),
                                        preview_batch['prev_elapsed'][:1].to(cfg.DEVICE,non_blocking=True),return_aux=True)
                    flow = aux['flow_pixels'][0].norm(dim=0).detach().cpu()
                    # Normalize flow for visualization
                    flow = flow / flow.max().clamp(min=1e-8)
                    # Convert (H, W) to (1, H, W)
                    flow = flow.unsqueeze(0)
                    mask = aux['mask'][0].detach().cpu()
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0)
                    run.log({
                        'frame/flow_magnitude_pixels': wandb.Image(flow),
                        'frame/refinement_mask': wandb.Image(mask)
                    })
        for p in frame_student.parameters():p.requires_grad_(False)
        if cfg.WANDB_MODE != 'disabled':
            frame_artifact=wandb.Artifact('frame-'+cfg.FRAME_MODEL_NAME,type='model')
            frame_artifact.add_file(cfg.FRAME_CHECKPOINT)
            run.log_artifact(frame_artifact)
        run.summary['best_epoch']=best_frame_epoch
        run.summary['best_val_l1']=best_frame
        run.finish()

    for parameter in frame_student.parameters():parameter.requires_grad_(False)


    @torch.no_grad()
    def build_window_frame_cache(dataset, split, model, device):
        dataset.cached_frames=None  # force fresh generation, never read prior cache
        count=len(dataset)
        size_gib=count*dataset.length*3*cfg.H*cfg.W/(1024**3)
        path=cfg.WORKDIR/f'frame_cache_{split}_{cfg.FRAME_MODEL_NAME}_{cfg.H}x{cfg.W}.npy'
        print(f'Creating {split} RGB cache: {count} windows, about {size_gib:.2f} GiB at {path}')
        array=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,
                                        shape=(count,dataset.length,3,cfg.H,cfg.W))
        loader=DataLoader(dataset,batch_size=cfg.FRAME_CACHE_BATCH,shuffle=False,num_workers=0)
        k=0
        model.eval()
        for batch_idx,b in enumerate(loader):
            frames=state_input_frames(model,b,device)
            frames=(frames.clamp(0,1)*255.).round().to(torch.uint8).cpu().numpy()
            n=len(frames)
            array[k:k+n]=frames
            k+=n
            if (batch_idx+1)%50==0:
                array.flush()
                print(f'  {split}: {k}/{count} windows cached')
        assert k==count
        array.flush()
        del array
        dataset.cached_frames=np.load(path,mmap_mode='r')
        print('Ready:',path)

    if cfg.CACHE_GENERATED_WINDOWS and cfg.USE_GENERATED_FRAMES_FOR_STATE and not args.ppo_only:
        build_window_frame_cache(data.train_windows,'train',frame_student,cfg.DEVICE)
        with amp_disabled():
            build_window_frame_cache(data.val_windows,'val',frame_student,cfg.DEVICE)
    else:
        print('Skipping frame cache; downstream state model will use the configured frame-input mode directly.')


    student=(HybridStateStudent(hidden=cfg.STUDENT_HIDDEN,
                   n_mamba=cfg.HYBRID_NUM_MAMBA_LAYERS,
                   attn_window=cfg.HYBRID_ATTENTION_WINDOW,
                   d_state=cfg.HYBRID_D_STATE,headdim=cfg.HYBRID_HEAD_DIM,
                   learned_visual=cfg.USE_LEARNED_VISUAL_ENCODER,visual_input_stride=cfg.VISUAL_INPUT_STRIDE,
                   visual_embeddings=cfg.VISUAL_PREDICTION=='embedding',embedding_dim=cfg.EMBEDDING_DIM)
              if cfg.USE_HYBRID_SSM_TRANSFORMER
              else StateStudent(hidden=cfg.STUDENT_HIDDEN,learned_visual=cfg.USE_LEARNED_VISUAL_ENCODER,
                                visual_input_stride=cfg.VISUAL_INPUT_STRIDE,
                                visual_embeddings=cfg.VISUAL_PREDICTION=='embedding',embedding_dim=cfg.EMBEDDING_DIM)).to(cfg.DEVICE,non_blocking=True)
    print('Selected student:',type(student).__name__,'/', cfg.BACKBONE_NAME,
          'learned visual:',cfg.USE_LEARNED_VISUAL_ENCODER)
    print('Frame weights:',sum(p.numel() for p in frame_student.parameters()))
    print('State weights:',sum(p.numel() for p in student.parameters()))
    print('Total FP32 MiB:',4*(sum(p.numel() for p in frame_student.parameters())+
                                   sum(p.numel() for p in student.parameters()))/2**20)
    if args.from_checkpoints and (args.from_checkpoints/'state_student.pt').is_file():
        load_weights(student, args.from_checkpoints/'state_student.pt', cfg.DEVICE,
                     allow_new_patch_stems=not args.ppo_only)
        save_weights(student.state_dict(), cfg.STATE_CHECKPOINT)
    elif args.from_checkpoints:
        print('State checkpoint missing; initializing the state model for supervised training.',flush=True)
    state_opt=adamw(student.parameters(),lr=cfg.LR_STUDENT,weight_decay=1e-3)
    state_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        state_opt, mode='min', factor=cfg.LR_SCHEDULER_FACTOR, patience=cfg.LR_SCHEDULER_PATIENCE)
    run=wandb.init(project=cfg.WANDB_PROJECT,dir=str(args.out),job_type='real_vio_student',mode=cfg.WANDB_MODE,
        config=dict(source='hyperdrone-real',
                    embedding_training_skipped=args.skip_embedding_training,
                    checkpoint_source=str(args.from_checkpoints.resolve()) if args.from_checkpoints else None,
                    frame_source=cfg.VISUAL_PREDICTION if cfg.USE_GENERATED_FRAMES_FOR_STATE else 'repeat_real',
                    frame_model=cfg.FRAME_MODEL_NAME,hidden=cfg.STUDENT_HIDDEN,
                    backbone=cfg.BACKBONE_NAME,learned_visual=cfg.USE_LEARNED_VISUAL_ENCODER,
                    visual_feature_dropout=cfg.STATE_VISUAL_FEATURE_DROPOUT,
                    n_mamba=cfg.HYBRID_NUM_MAMBA_LAYERS if cfg.USE_HYBRID_SSM_TRANSFORMER else 0,
                    attention_window=cfg.HYBRID_ATTENTION_WINDOW if cfg.USE_HYBRID_SSM_TRANSFORMER else 0,
                    window=cfg.WINDOW,max_epochs=cfg.EPOCHS_STUDENT,
                    lr=cfg.LR_STUDENT,patience=cfg.PATIENCE_STUDENT,
                    rgb_hz='approximately 20 from causal native captures',imu_hz=200,
                    velocity_target='change relative to window initial velocity'))
    best_state=float('inf');bad_state_epochs=0;best_state_epoch=0
    state_patience=FreshBatchPatience()
    data.begin_phase('state')
    val_state_loader=data.state_validation_loader()
    for epoch in range(0 if args.ppo_only else cfg.EPOCHS_STUDENT):
        train_pair_loader, train_state_loader = data.next_epoch()
        timer=EpochTimer(cfg.DEVICE)
        student.train();tr=[]
        for b in timed_batches(train_state_loader,timer):
            frames=state_input_frames(frame_student,b,cfg.DEVICE)
            pred=student(frames,b['imu'].to(cfg.DEVICE,non_blocking=True),b['observed'].to(cfg.DEVICE,non_blocking=True),
                         b['dt'].to(cfg.DEVICE,non_blocking=True),b['gravity'].to(cfg.DEVICE,non_blocking=True))
            loss,_=losses(pred,b,cfg.DEVICE,metrics=False)
            data.observe_prediction(b,pred)
            state_opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(),1.)
            state_opt.step()
            tr.append(loss.detach()*len(frames));timer.add(len(frames))
        performance=timer.result()
        performance=record_epoch(args.out,'state',epoch+1,performance,data,loader=train_state_loader)
        student.eval();ev=[];metrics=[];validation_count=0
        with torch.no_grad(),amp_disabled():
            for b in val_state_loader:
                frames=state_input_frames(frame_student,b,cfg.DEVICE)
                p=student(frames,b['imu'].to(cfg.DEVICE,non_blocking=True),b['observed'].to(cfg.DEVICE,non_blocking=True),
                          b['dt'].to(cfg.DEVICE,non_blocking=True),b['gravity'].to(cfg.DEVICE,non_blocking=True))
                loss,m=losses(p,b,cfg.DEVICE)
                n=len(frames);ev.append(loss.detach()*n);validation_count+=n;metrics.append((m,n))
        train_loss=float(torch.stack(tr).sum()/performance['samples']);val_loss=float(torch.stack(ev).sum()/validation_count)
        from ppo import validation_loss as curriculum_validation
        promoted=data.finish_curriculum_epoch(val_loss,
            lambda loader:curriculum_validation(frame_student,student,loader,cfg.DEVICE))
        data.step_scheduler(state_scheduler,val_loss)
        improved=val_loss<best_state-1e-5 or best_state_epoch==0
        if improved:
            best_state=val_loss;bad_state_epochs=0;best_state_epoch=epoch+1
            save_weights(student.state_dict(),cfg.STATE_CHECKPOINT)
        bad_state_epochs=data.update_patience(state_patience,improved,promoted)
        logs={'student/train_loss':train_loss,'student/val_loss':val_loss,
              'student/epoch':epoch+1,'student/lr':state_opt.param_groups[0]['lr'],
              'student/best_val_loss':best_state,
              'student/backbone_is_hybrid':int(cfg.USE_HYBRID_SSM_TRANSFORMER),
              'student/non_improving_data_cycles':bad_state_epochs,
              'student/early_stopping_grace_remaining':state_patience.grace_remaining,
              'student/train_windows_per_second':performance['samples_per_second']}
        logs.update({'student/train_seconds':performance['seconds'],
                     'student/loader_wait_seconds':performance['loader_wait_seconds']})
        logs.update(data.curriculum_log())
        logs.update(pipeline_log(performance))
        for k in metrics[0][0]:
            value=sum((m[k]**2 if k.endswith('rmse') else m[k])*n for m,n in metrics)/validation_count
            logs['student/val_'+k]=float(np.sqrt(value) if k.endswith('rmse') else value)
        run.log(logs)
        print(f'Student {epoch+1}/{cfg.EPOCHS_STUDENT}: train={train_loss:.5f} '
              f'val={val_loss:.5f}, position RMSE={logs["student/val_pos_rmse"]:.4f} m, '
              f'LR={state_opt.param_groups[0]["lr"]:.2e}')
        print(f'State throughput: {performance["samples_per_second"]:.2f} windows/s',flush=True)
        if bad_state_epochs>=cfg.PATIENCE_STUDENT:
            print(f'Student early stopped at epoch {epoch+1}; best epoch={best_state_epoch}; '
                  f'non-improving data cycles={bad_state_epochs}')
            break

    student.load_state_dict(torch.load(cfg.STATE_CHECKPOINT,map_location=cfg.DEVICE,weights_only=True))
    student.eval()
    if cfg.WANDB_MODE != 'disabled':
        student_artifact=wandb.Artifact('state-'+cfg.BACKBONE_NAME+'-'+cfg.FRAME_MODEL_NAME,type='model')
        student_artifact.add_file(cfg.STATE_CHECKPOINT)
        run.log_artifact(student_artifact)
    run.summary['best_epoch']=best_state_epoch
    run.summary['best_val_loss']=best_state
    run.finish()


    if cfg.USE_TEACHER and not args.ppo_only:
        teacher=StateTeacher(hidden=cfg.TEACHER_HIDDEN,backbone=cfg.TEACHER_BACKBONE,
                             learned_visual=cfg.USE_LEARNED_VISUAL_ENCODER,
                             visual_input_stride=cfg.VISUAL_INPUT_STRIDE,
                             visual_embeddings=cfg.VISUAL_PREDICTION=='embedding',embedding_dim=cfg.EMBEDDING_DIM).to(cfg.DEVICE,non_blocking=True)
        opt=adamw(teacher.parameters(),lr=cfg.LR_TEACHER)
        teacher_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,mode='min',factor=cfg.LR_SCHEDULER_FACTOR,patience=cfg.LR_SCHEDULER_PATIENCE)
        run=wandb.init(project=cfg.WANDB_PROJECT,dir=str(args.out),job_type='teacher_training',mode=cfg.WANDB_MODE,
                       config={'teacher_backbone':cfg.TEACHER_BACKBONE,'student_backbone':cfg.BACKBONE_NAME,
                               'learned_visual':cfg.USE_LEARNED_VISUAL_ENCODER,'hidden':cfg.TEACHER_HIDDEN,'frame_model':cfg.FRAME_MODEL_NAME,'lr':cfg.LR_TEACHER,
                               'max_epochs':cfg.EPOCHS_TEACHER})
        best=float('inf');bad_teacher_epochs=0;best_teacher_epoch=0
        teacher_patience=FreshBatchPatience()
        data.begin_phase('teacher')
        val_state_loader=data.state_validation_loader()
        for epoch in range(cfg.EPOCHS_TEACHER):
            train_pair_loader, train_state_loader = data.next_epoch()
            timer=EpochTimer(cfg.DEVICE)
            teacher.train();tl=[];train_count=0
            for b in timed_batches(train_state_loader,timer):
                frames=state_input_frames(frame_student,b,cfg.DEVICE)
                pred=teacher(frames,b['imu'].to(cfg.DEVICE,non_blocking=True),b['observed'].to(cfg.DEVICE,non_blocking=True),
                             b['dt'].to(cfg.DEVICE,non_blocking=True),b['gravity'].to(cfg.DEVICE,non_blocking=True))
                loss,_=losses(pred,b,cfg.DEVICE,metrics=False)
                data.observe_prediction(b,pred)
                opt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(teacher.parameters(),1.)
                opt.step();n=len(frames);tl.append(loss.detach()*n);train_count+=n;timer.add(n)
            performance=record_epoch(args.out,'teacher',epoch+1,timer.result(),data,loader=train_state_loader)
            teacher.eval();vl=[];val_ate=[];val_count=0
            with torch.no_grad(),amp_disabled():
                for b in val_state_loader:
                    f=state_input_frames(frame_student,b,cfg.DEVICE)
                    p=teacher(f,b['imu'].to(cfg.DEVICE,non_blocking=True),b['observed'].to(cfg.DEVICE,non_blocking=True),
                              b['dt'].to(cfg.DEVICE,non_blocking=True),b['gravity'].to(cfg.DEVICE,non_blocking=True))
                    n=len(b['imu']);vl.append(losses(p,b,cfg.DEVICE,metrics=False)[0].detach()*n);val_count+=n
                    val_ate.append(mean_ate(p['pos'],b['pos'].to(cfg.DEVICE,non_blocking=True))*n)
            mean_train=float(torch.stack(tl).sum()/train_count);mean_val=float(torch.stack(vl).sum()/val_count)
            promoted=data.finish_curriculum_epoch(mean_val,
                lambda loader:curriculum_validation(frame_student,teacher,loader,cfg.DEVICE))
            data.step_scheduler(teacher_scheduler,mean_val)
            improved=mean_val<best-1e-5 or best_teacher_epoch==0
            if improved:
                best=mean_val;bad_teacher_epochs=0;best_teacher_epoch=epoch+1
                save_weights(teacher.state_dict(),cfg.TEACHER_CHECKPOINT)
            bad_teacher_epochs=data.update_patience(teacher_patience,improved,promoted)
            run.log({'teacher/train_loss':mean_train,'teacher/val_loss':mean_val,
                     'teacher/val_mean_ate_m':float(torch.stack(val_ate).sum()/val_count),
                     'teacher/epoch':epoch+1,'teacher/lr':opt.param_groups[0]['lr'],
                     'teacher/best_val_loss':best,'teacher/non_improving_data_cycles':bad_teacher_epochs,
                     'teacher/early_stopping_grace_remaining':teacher_patience.grace_remaining}|data.curriculum_log()|pipeline_log(performance))
            print(f'Teacher {epoch+1}/{cfg.EPOCHS_TEACHER}: {mean_train:.5f}/{mean_val:.5f}')
            if bad_teacher_epochs>=cfg.PATIENCE_TEACHER:
                print('Teacher early stopped; best epoch:',best_teacher_epoch)
                break
        run.summary['best_epoch']=best_teacher_epoch
        run.summary['best_val_loss']=best
        run.finish()
        teacher.load_state_dict(torch.load(cfg.TEACHER_CHECKPOINT,map_location=cfg.DEVICE,weights_only=True))
        teacher.eval()
        for p in teacher.parameters():p.requires_grad_(False)

        # Supervised + teacher correction distillation, with held-out validation.
        opt=adamw(student.parameters(),lr=cfg.LR_DISTILL,weight_decay=1e-3)
        distill_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,mode='min',factor=cfg.LR_SCHEDULER_FACTOR,patience=cfg.LR_SCHEDULER_PATIENCE)
        run=wandb.init(project=cfg.WANDB_PROJECT,dir=str(args.out),job_type='student_distillation',mode=cfg.WANDB_MODE,
                       config={'teacher_backbone':cfg.TEACHER_BACKBONE,'student_backbone':cfg.BACKBONE_NAME,
                               'lr':cfg.LR_DISTILL,'max_epochs':cfg.EPOCHS_DISTILL})
        best_distill=float('inf');bad_distill_epochs=0;best_distill_epoch=0
        distill_patience=FreshBatchPatience()
        data.begin_phase('distill')
        val_state_loader=data.state_validation_loader()
        for epoch in range(cfg.EPOCHS_DISTILL):
            train_pair_loader, train_state_loader = data.next_epoch()
            timer=EpochTimer(cfg.DEVICE)
            student.train();tr=[];train_count=0
            for b in timed_batches(train_state_loader,timer):
                frames=state_input_frames(frame_student,b,cfg.DEVICE)
                imu=b['imu'].to(cfg.DEVICE,non_blocking=True); obs=b['observed'].to(cfg.DEVICE,non_blocking=True)
                dt=b['dt'].to(cfg.DEVICE,non_blocking=True);grav=b['gravity'].to(cfg.DEVICE,non_blocking=True)
                with torch.no_grad(),amp_disabled():target=teacher(frames,imu,obs,dt,grav)
                pred=student(frames,imu,obs,dt,grav,return_corrections=True)
                gt_loss,_=losses(pred,b,cfg.DEVICE,metrics=False)
                data.observe_prediction(b,pred)
                kd=F.smooth_l1_loss(pred['corrections'],target['corrections'])
                loss=gt_loss+0.2*kd
                opt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(),1.)
                opt.step();n=len(frames);tr.append(loss.detach()*n);train_count+=n;timer.add(n)
            performance=record_epoch(args.out,'distill',epoch+1,timer.result(),data,loader=train_state_loader)
            student.eval();vl=[];val_ate=[];val_count=0
            with torch.no_grad(),amp_disabled():
                for b in val_state_loader:
                    frames=state_input_frames(frame_student,b,cfg.DEVICE)
                    p=student(frames,b['imu'].to(cfg.DEVICE,non_blocking=True),b['observed'].to(cfg.DEVICE,non_blocking=True),
                              b['dt'].to(cfg.DEVICE,non_blocking=True),b['gravity'].to(cfg.DEVICE,non_blocking=True))
                    n=len(b['imu']);vl.append(losses(p,b,cfg.DEVICE,metrics=False)[0].detach()*n);val_count+=n
                    val_ate.append(mean_ate(p['pos'],b['pos'].to(cfg.DEVICE,non_blocking=True))*n)
            val=float(torch.stack(vl).sum()/val_count)
            promoted=data.finish_curriculum_epoch(val,
                lambda loader:curriculum_validation(frame_student,student,loader,cfg.DEVICE))
            data.step_scheduler(distill_scheduler,val)
            improved=val<best_distill-1e-5 or best_distill_epoch==0
            if improved:
                best_distill=val;best_distill_epoch=epoch+1;bad_distill_epochs=0
                save_weights(student.state_dict(),str(args.out / 'state_student_distilled.pt'))
            bad_distill_epochs=data.update_patience(distill_patience,improved,promoted)
            run.log({'distill/train_loss':float(torch.stack(tr).sum()/train_count),'distill/val_loss':val,
                     'distill/val_mean_ate_m':float(torch.stack(val_ate).sum()/val_count),
                     'distill/lr':opt.param_groups[0]['lr'],'distill/epoch':epoch+1,
                     'distill/best_val_loss':best_distill,
                     'distill/non_improving_data_cycles':bad_distill_epochs,
                     'distill/early_stopping_grace_remaining':distill_patience.grace_remaining}|data.curriculum_log()|pipeline_log(performance))
            print(f'Distill {epoch+1}/{cfg.EPOCHS_DISTILL}: train={float(torch.stack(tr).sum()/train_count):.5f}, val={val:.5f}')
            if bad_distill_epochs>=cfg.PATIENCE_DISTILL:
                print('Distillation early stopped; best epoch:',best_distill_epoch)
                break
        student.load_state_dict(torch.load(str(args.out / 'state_student_distilled.pt'),map_location=cfg.DEVICE,weights_only=True))
        student.eval()
        if cfg.WANDB_MODE != 'disabled':
            distilled_artifact=wandb.Artifact('state-student-distilled',type='model')
            distilled_artifact.add_file(str(args.out / 'state_student_distilled.pt'))
            run.log_artifact(distilled_artifact)
        run.summary['best_epoch']=best_distill_epoch
        run.summary['best_val_loss']=best_distill
        run.finish()
    else:
        print('Skipping optional teacher and distillation (USE_TEACHER=False).')

    if cfg.USE_TEACHER and not args.ppo_only:
        save_weights(student.state_dict(), cfg.STATE_CHECKPOINT)

    if not args.ppo_only:
        save_post_sl(args,cfg,student)

    if cfg.USE_PPO:
        from ppo import refine_ppo
        if args.ppo_only:
            # This is a pre-PPO baseline, which can already contain RL updates.
            save_weights(student.state_dict(), args.out/'state_student_sl.pt')
        refine_ppo(frame_student, student, data, cfg, args.out, save_weights)
        save_weights(student.state_dict(), cfg.STATE_CHECKPOINT)

    from export import export_models, notebook_settings
    settings = notebook_settings(cfg)
    (args.out / 'student_config.json').write_text(json.dumps(settings, indent=2) + '\n')
    if args.skip_export:
        print(f'Teacher checkpoints saved to {args.out}; export skipped.', flush=True)
    else:
        export_models(frame_student, student, settings, args.solution_out)
        if args.verify_limits:
            from viobench.solution.runtime.python.bundler.bundle import bundle
            from viobench.tools.cli import run_with_large_stack
            from verify_limits import verify
            module = args.out/'teacher.wasm'
            bundle(args.solution_out, module)
            result = run_with_large_stack(lambda: verify(module, args.verify_frames, args.out/'teacher-limits.json'))
            if not result['passed']:
                raise RuntimeError('Teacher failed resource verification; see teacher-limits.json')


if __name__ == '__main__':
    main()
