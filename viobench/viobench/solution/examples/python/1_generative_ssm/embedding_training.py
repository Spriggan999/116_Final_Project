"""Joint embedding prediction on observed camera pairs, with no RGB decoder."""
import torch
import torch.nn.functional as F

from performance import EpochTimer,adamw,amp_disabled,record_epoch,timed_batches,pipeline_log
from data import frame_pair_images
from streaming import FreshBatchPatience


def embedding_objective(model,batch,config,return_errors=False):
    previous,anchor,future=frame_pair_images(batch,config.DEVICE)
    online=model.encoder(torch.cat([previous,anchor,future],0))
    previous_z,anchor_z,future_z=online.chunk(3,0)
    predicted=model.predict_embeddings(previous_z,anchor_z,
        batch['imu'].to(config.DEVICE,non_blocking=True),batch['elapsed'].to(config.DEVICE,non_blocking=True),
        batch['prev_elapsed'].to(config.DEVICE,non_blocking=True))
    with torch.no_grad():target=model.target_encoder(future)
    prediction=F.smooth_l1_loss(predicted,target,beta=.1)
    content=F.smooth_l1_loss(future_z,target,beta=.1)
    # Batch content regularization prevents a constant embedding from solving
    # the prediction objective. Population variance is finite even at batch one.
    centered=online-online.mean(0)
    variance=F.relu(config.EMBEDDING_STD_TARGET-torch.sqrt(centered.square().mean(0)+1e-4)).mean()
    covariance=centered.T@centered/max(1,len(online)-1)
    off_diagonal=covariance-torch.diag_embed(covariance.diagonal())
    covariance=off_diagonal.square().sum()/online.shape[-1]
    total=(prediction+content+config.EMBEDDING_VARIANCE_WEIGHT*variance
           +config.EMBEDDING_COVARIANCE_WEIGHT*covariance)
    metrics=dict(prediction=prediction,content=content,variance=variance,covariance=covariance)
    if return_errors:
        return total,metrics,(predicted.detach()-target).square().mean(-1)
    return total,metrics


def embedding_validation(model,loader,config):
    total,count=0.,0
    model.eval()
    with torch.no_grad(),amp_disabled():
        for batch in loader:
            loss,_=embedding_objective(model,batch,config)
            n=len(batch['anchor']);total+=float(loss)*n;count+=n
    if not count:raise ValueError('No embedding curriculum validation pairs')
    return total/count


def train_embeddings(model,data,args,config,save_weights):
    import wandb
    optimizer=adamw((p for p in model.parameters() if p.requires_grad),lr=config.LR_FRAME)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',
        factor=config.LR_SCHEDULER_FACTOR,patience=config.LR_SCHEDULER_PATIENCE)
    run=wandb.init(project=config.WANDB_PROJECT,dir=str(args.out),job_type='embedding_prediction',mode=config.WANDB_MODE,
        config=dict(source='hyperdrone-real',visual_prediction='embedding',embedding_dim=model.embedding_dim,
                    ema=config.EMBEDDING_EMA,hidden=config.EMBEDDING_HIDDEN))
    best=float('inf');best_epoch=0;patience=FreshBatchPatience()
    data.begin_phase('frame')
    try:
        for epoch in range(0 if args.ppo_only else config.EPOCHS_FRAME):
            loader,_=data.next_epoch()
            timer=EpochTimer(config.DEVICE)
            model.train();train=[]
            for batch in timed_batches(loader,timer):
                loss,_,errors=embedding_objective(model,batch,config,return_errors=True)
                data.observe_errors(batch,errors)
                if not torch.isfinite(loss):raise ValueError('Nonfinite embedding training loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();model.update_target()
                count=len(batch['anchor']);train.append(loss.detach()*count);timer.add(count)
            measurement=timer.result()
            measurement=record_epoch(args.out,'embedding',epoch+1,measurement,data,loader=loader)
            model.eval();validation=[];counts=0;terms={}
            with torch.no_grad(),amp_disabled():
                for batch in data.val_pair_loader:
                    loss,metrics=embedding_objective(model,batch,config)
                    count=len(batch['anchor']);counts+=count;validation.append(loss*count)
                    for key,value in metrics.items():terms[key]=terms.get(key,0)+value*count
            score=float(torch.stack(validation).sum()/counts)
            if not torch.isfinite(torch.tensor(score)):raise ValueError('Nonfinite embedding validation loss')
            scheduler.step(score)
            promoted=data.finish_curriculum_epoch(score,lambda loader:embedding_validation(model,loader,config))
            improved=score<best-1e-5 or best_epoch==0
            if improved:
                best=score;best_epoch=epoch+1
                save_weights(model.state_dict(),config.FRAME_CHECKPOINT)
            bad=data.update_patience(patience,improved,promoted)
            record=dict(epoch=epoch+1,train_loss=float(torch.stack(train).sum()/timer.samples),
                        validation_loss=score,best_validation_loss=best,non_improving_data_cycles=bad,
                        early_stopping_grace_remaining=patience.grace_remaining,
                        train_seconds=measurement['seconds'],loader_wait_seconds=measurement['loader_wait_seconds'],
                        pairs_per_second=measurement['samples_per_second'],
                        **{'validation_'+key:float(value/counts) for key,value in terms.items()})
            import json
            with (args.out/'embedding_metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            run.log({'embedding/'+key:value for key,value in record.items()}|data.curriculum_log()|pipeline_log(measurement))
            print(f'Embedding {epoch+1}/{config.EPOCHS_FRAME}: train={record["train_loss"]:.6g}, '
                  f'val={score:.6g}; {measurement["samples_per_second"]:.2f} pairs/s',flush=True)
            if bad>=config.PATIENCE_FRAME:break
        model.load_state_dict(torch.load(config.FRAME_CHECKPOINT,map_location=config.DEVICE,weights_only=True))
        model.eval()
        run.summary['best_epoch']=best_epoch
        if best_epoch:run.summary['best_validation_loss']=best
        if config.WANDB_MODE!='disabled':
            artifact=wandb.Artifact('embedding-predictor',type='model')
            artifact.add_file(config.FRAME_CHECKPOINT);run.log_artifact(artifact)
    finally:
        run.finish()
