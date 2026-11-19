"""Fixed-data fresh-checkpoint LR screening for eight students and the VIO teacher.

Candidates share initialization, sample order, objectives and validation flights.
This is a bounded pilot, not proof of globally optimal learning rates.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import default_collate

import config as cfg
from data import EpisodeStore,StateWindows,StateEpisodes,FramePairs,state_input_frames
from distill import train_student
from embedding_training import embedding_objective
from export import normalize_settings
from leaderboard_metrics import LeaderboardATE
from losses import losses
from performance import configure,adamw,amp_disabled
from students.model import BudgetStudent,PROFILES
from train import save_weights


def write_json(path,value):
    temporary=Path(str(path)+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def state_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sparse_gray(cameras,width,height):
    """Exact uint8 640x480 -> rounded 320x240 ABI pixels -> sparse gray grids."""
    yy=(2*torch.arange(height)+1)*240//(2*height)
    xx=(2*torch.arange(width)+1)*320//(2*width)
    if cameras.shape[-2:]==(480,640):
        value=sum(cameras[:,:,2*yy[:,None]+dy,2*xx[None,:]+dx].to(torch.int32)
                  for dy in range(2) for dx in range(2))
        value=((value+2)//4).float()
    elif cameras.shape[-2:]==(240,320):value=cameras[:,:,yy[:,None],xx[None,:]].float()
    else:raise ValueError('Sweep requires full-resolution or ABI-resolution camera captures')
    return (value.mean(1)/255.).flatten(1)


class CachedStudent(BudgetStudent):
    def image_features(self,images):
        return images if images.ndim==3 else super().image_features(images)


def historical_analysis(directory,out):
    result=dict(source=str(directory),models={},interpretation=[
        'All students used one constant 3e-4 LR; these observations do not identify a causal LR effect.',
        'Early minima versus later validation regressions justify testing lower LRs, alongside the baseline and an upper control.',
        'Curriculum never left 2 seconds; whole-flight validation and longer-horizon confirmation are needed.'])
    for path in sorted(directory.glob('*/metrics.jsonl')):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        best=min(rows,key=lambda row:row['validation_mean_ate_m']);last=rows[-1]
        result['models'][path.parent.name]=dict(epochs=last['epoch'],lr=last['learning_rate'],
            best_ate=best['validation_mean_ate_m'],best_epoch=best['epoch'],final_ate=last['validation_mean_ate_m'],
            final_state_loss=last['validation_state_loss'],final_train_loss=last['train_loss'],
            tail_ate_std=float(np.std([r['validation_mean_ate_m'] for r in rows[-100:]])),
            final_vs_best_pct=100*(last['validation_mean_ate_m']/best['validation_mean_ate_m']-1))
    write_json(out/'existing_run_analysis.json',result)
    return result


def to_gpu(tree):
    if torch.is_tensor(tree):return tree.cuda(non_blocking=True)
    if isinstance(tree,dict):return {key:to_gpu(value) for key,value in tree.items()}
    return tree


def subset(tree,indices):
    return {key:(value[indices] if torch.is_tensor(value) else value) for key,value in tree.items()}


@torch.no_grad()
def build_cache(store,length,frame=None,training=True,per_flight=4):
    dataset=StateWindows(store,length=length,compact_cameras=True) if training else StateEpisodes(store)
    items=[]
    if training:
        for index,entry in enumerate(store.entries):
            maximum=(len(entry['state'])-length)//entry['stride']
            for start in sorted(set(np.linspace(0,maximum,per_flight).round().astype(int).tolist())):
                items.append(dataset.window((index,start*entry['stride']),length))
    else:items=[dataset[i] for i in range(len(dataset))]
    base=[];student_truth=[];teacher_truth=[];teacher_features=[]
    grids={tuple((p['grid_width'],p['grid_height'])):[] for p in PROFILES.values()}
    for i,item in enumerate(items):
        ticks=list(range(0,len(item['imu']),10));images=item['cam_images'][item['anchor_img_idx'][ticks]]
        for (w,h),values in grids.items():values.append(sparse_gray(images,w,h))
        base.append({key:item[key] for key in ['imu','dt','gravity','observed','camera_position_in_body','camera_calibration_available']})
        student_truth.append({key:item[key][ticks].clone() for key in ['pos','vel','rot','omega']}|
                             {'camera_position_in_body':item['camera_position_in_body']})
        student_truth[-1]['omega'][0]=0
        teacher_truth.append({key:item[key] for key in ['pos','vel','rot','omega','camera_position_in_body']})
        if frame is not None:
            with amp_disabled():features=state_input_frames(frame,default_collate([item]),'cuda')
            teacher_features.append(features[0].float().cpu())
        if (i+1)%8==0:print('CACHE',length,'train' if training else 'validation',i+1,'/',len(items),flush=True)
    result=dict(base=to_gpu(default_collate(base)),student_truth=to_gpu(default_collate(student_truth)),
                teacher_truth=to_gpu(default_collate(teacher_truth)),grids={key:to_gpu(torch.stack(value)) for key,value in grids.items()},
                ticks=list(range(0,len(items[0]['imu']),10)),count=len(items))
    if teacher_features:result['teacher_features']=to_gpu(torch.stack(teacher_features))
    return result


def student_inputs(cache,name,indices):
    base=subset(cache['base'],indices);spec=PROFILES[name]
    return dict(images=cache['grids'][(spec['grid_width'],spec['grid_height'])][indices],
                imu=base['imu'],dt=base['dt'],gravity=base['gravity'],ticks=cache['ticks'])


def student_parameter_groups(model,state_lr,embedding_lr):
    """Partition trainable parameters without duplicating the online encoder."""
    visual=list(model.vision.parameters())+list(model.embedding_predictor.parameters())
    visual_ids={id(value) for value in visual}
    state=[value for value in model.parameters() if id(value) not in visual_ids]
    return [dict(params=state,lr=state_lr,name='state'),
            dict(params=visual,lr=embedding_lr,name='embedding')]


def teacher_prediction(model,cache,indices):
    b=subset(cache['base'],indices)
    return model(cache['teacher_features'][indices],b['imu'],b['observed'],b['dt'],b['gravity'],parallel_physics=True)


@torch.no_grad()
def evaluate(model,cache,name):
    model.eval();metric=LeaderboardATE();state=[]
    with amp_disabled():
        for i in range(cache['count']):
            indices=torch.tensor([i],device='cuda');base=subset(cache['base'],indices)
            if name=='teacher':
                prediction=teacher_prediction(model,cache,indices);truth=subset(cache['teacher_truth'],indices)
                observed=base['observed'];dt=base['dt']
            else:
                prediction=model(**student_inputs(cache,name,indices));truth=subset(cache['student_truth'],indices)
                observed=torch.ones_like(prediction['pos'][...,0],dtype=torch.bool);dt=base['dt']*10
            value=losses(prediction,truth,'cuda',metrics=False)[0]
            if not torch.isfinite(value):raise ValueError('Nonfinite validation state loss')
            state.append(value)
            metadata=dict(pos=truth['pos'],rot=truth['rot'],observed=observed,dt=dt,
                episode_id=[str(i)],episode_complete=[True],camera_position_in_body=base['camera_position_in_body'],
                camera_calibration_available=base['camera_calibration_available'])
            metric.update(prediction,metadata)
    result=metric.result();result['state_loss']=float(torch.stack(state).mean())
    return result


class Sweep:
    def __init__(self,args):
        self.args=args;self.out=args.out;self.out.mkdir(parents=True,exist_ok=False)
        self.scratch=args.scratch;self.scratch.mkdir(parents=True,exist_ok=True)
        self.records=[];self.initial_hashes={};self.global_step=0
        self.analysis=historical_analysis(args.previous,self.out)
        self.settings=normalize_settings(json.loads((args.teacher_config/'student_config.json').read_text()))
        write_json(self.out/'protocol.json',dict(candidates=args.rates,seed=args.seed,screen_steps=args.screen_steps,
            confirmation_steps=args.confirm_steps,train=str(args.train),validation=str(args.validation),
            primary_selection='mean of last two validation-state-loss checkpoints; lower is better',
            checkpoint_selection='lowest held-out state loss within each trial',
            training_horizons_seconds=[2,10,20],validation_horizon_seconds=20,
            fresh_initialization=True,teacher_settings=self.settings,loss_weights=dict(state=1.,embedding=.1,alignment=.1),
            limitations=['Fixed eight-scene training subset, not the full 192-scene streaming curriculum.',
                         'Bounded LR screening; no global optimum or full convergence claim.',
                         'Teacher state trials share a frozen encoder trained from fresh initialization in this sweep.',
                         'Teacher PPO is outside this supervised-learning sweep.']))
        import wandb
        self.wandb=wandb
        try:
            previous_run=wandb.Api(timeout=30).run('ysaxena_personal/hyperdrone-vio-fuel-aware/f3i9stlc')
            remote=dict(url=previous_run.url,state=previous_run.state,
                        completed_epochs=previous_run.summary.get('supervised/epoch'),
                        best_ate={name:previous_run.summary.get(name+'/best_validation_mean_ate_m') for name in PROFILES})
            write_json(self.out/'existing_wandb_summary.json',remote)
        except Exception as error:
            write_json(self.out/'existing_wandb_summary.json',dict(error=str(error),local_history_used=True))
        self.run=wandb.init(project=cfg.WANDB_PROJECT,dir=str(self.out),mode=args.wandb_mode,
            job_type='learning_rate_sweep',name=self.out.name,config=json.loads((self.out/'protocol.json').read_text()))
        self.run.summary['existing_run_analysis']=self.analysis
        print('W&B',self.run.url,flush=True)
        write_json(self.out/'wandb.json',dict(url=self.run.url,id=self.run.id))

    def log(self,record):
        self.global_step+=1
        with (self.out/'history.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        scalars={key:value for key,value in record.items() if isinstance(value,(str,int,float,bool))}
        namespace=record.get('model','sweep')
        namespaced={namespace+'/'+key:value for key,value in scalars.items() if key!='model'}
        self.run.log({'sweep/index':self.global_step,**scalars,**namespaced})
        print(json.dumps(record),flush=True)

    def trial(self,name,rate,cache,val,steps,phase,seed,embedding_rate=None):
        if name=='teacher' and embedding_rate is not None:
            raise ValueError('Teacher embeddings are pretrained separately, not a state-model parameter group')
        tag=f'{name}-{phase}-{rate:g}-s{seed}'
        if embedding_rate is not None:tag+=f'-embedding{embedding_rate:g}'
        completed=self.scratch/(tag+'.json')
        if completed.exists():
            record=json.loads(completed.read_text());self.records.append(record);return record
        seed_all(seed)
        if name=='teacher':
            from models import HybridStateStudent
            s=self.settings
            model=HybridStateStudent(hidden=s['hidden'],n_mamba=s['mamba_layers'],attn_window=s['attention_window'],
                d_state=s['hybrid_d_state'],headdim=s['hybrid_head_dim'],heads=s['hybrid_attention_heads'],
                learned_visual=s['learned_visual'],visual_input_stride=s['visual_input_stride'],visual_embeddings=True,
                embedding_dim=s['embedding_dim']).cuda()
            optimizer=adamw(model.parameters(),lr=rate,weight_decay=.001)
            batch=max(1,min(4,1604//cache['base']['imu'].shape[1]))
        else:
            model=CachedStudent(dict(PROFILES[name],jepa_target=True)).cuda()
            parameters=model.parameters() if embedding_rate is None else student_parameter_groups(model,rate,embedding_rate)
            optimizer=adamw(parameters,lr=rate)
            batch=max(1,min(8,3208//cache['base']['imu'].shape[1]))
        digest=state_hash(model);key=(name,seed)
        if key in self.initial_hashes:assert self.initial_hashes[key]==digest,'Unmatched trial initialization'
        self.initial_hashes[key]=digest
        generator=torch.Generator().manual_seed(seed+991);order=[];row=dict(model=model,optimizer=optimizer)
        history=[];best=float('inf');best_metrics=None;best_step=0;norms=[];start=time.perf_counter()
        status='complete';checkpoint=self.scratch/(tag+'.pt');last_training_loss=None
        try:
            for step in range(steps+1):
                if step==0 or step%max(1,steps//4)==0 or step==steps:
                    metrics=evaluate(model,val,name)
                    logged=dict(model=name,phase=phase,lr=rate,seed=seed,updates=step,batch=batch,
                                seconds=time.perf_counter()-start,**{'validation_'+k:v for k,v in metrics.items()})
                    if embedding_rate is not None:
                        logged['embedding_lr']=embedding_rate
                        from student_embedding import objective
                        totals=[];parts={}
                        with torch.no_grad(),amp_disabled():
                            for index in range(val['count']):
                                value,terms=objective(model,student_inputs(val,name,torch.tensor([index],device='cuda')))
                                totals.append(value)
                                for key,item in terms.items():parts.setdefault(key,[]).append(item)
                        metrics['embedding_loss']=float(torch.stack(totals).mean())
                        if not np.isfinite(metrics['embedding_loss']):raise ValueError('Nonfinite embedding validation loss')
                        logged.update({'validation_'+key:float(torch.stack(values).mean()) for key,values in parts.items()})
                    history.append(metrics)
                    if last_training_loss is not None:
                        logged['training_loss']=float(last_training_loss)
                        logged.update({'training_'+key:float(value) for key,value in row.get('loss_terms',{}).items()})
                        if 'gradient_norm_preclip' in row:logged['gradient_norm_preclip']=float(row['gradient_norm_preclip'])
                    self.log(logged)
                    if metrics['state_loss']<best:
                        best=metrics['state_loss'];best_metrics=metrics;best_step=step
                        save_weights(model.state_dict(),checkpoint)
                if step==steps:break
                if len(order)<batch:order+=torch.randperm(cache['count'],generator=generator).tolist()
                indices=torch.tensor(order[:batch],device='cuda');order=order[batch:]
                model.train()
                if name=='teacher':
                    prediction=teacher_prediction(model,cache,indices)
                    loss,parts=losses(prediction,subset(cache['teacher_truth'],indices),'cuda',metrics=False,components=True)
                    row['loss_terms']=parts
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('Nonfinite update')
                    optimizer.step();norms.append(float(norm));row['gradient_norm_preclip']=norm.detach()
                else:
                    loss=train_student(name,row,(student_inputs(cache,name,indices),subset(cache['student_truth'],indices),None,None),
                        'cuda',(0.,0.,1.),embedding_weight=.1,embedding_ema=.99,alignment_weight=.1)
                    if not torch.isfinite(row['gradient_norm_preclip']):raise ValueError('Nonfinite gradient norm')
                if not torch.isfinite(loss):raise ValueError('Nonfinite update')
                last_training_loss=loss.detach()
        except (ValueError,torch.OutOfMemoryError) as error:
            status='failed';self.log(dict(model=name,phase=phase,lr=rate,error=str(error),updates=step))
        record=dict(model=name,phase=phase,lr=rate,seed=seed,status=status,steps=step,batch=batch,initial_sha256=digest,
            checkpoint=str(checkpoint),best_step=best_step,best_validation=best_metrics,
            score=float(np.mean([v['state_loss'] for v in history[-2:]])) if status=='complete' else None,
            initial_validation=history[0] if history else None,
            final_validation=history[-1] if history else None,seconds=time.perf_counter()-start,
            train_gradient_norm_mean=float(np.mean(norms)) if norms else None)
        if embedding_rate is not None:
            record.update(embedding_lr=embedding_rate,
                embedding_score=float(np.mean([value['embedding_loss'] for value in history[-2:]])) if status=='complete' else None)
        write_json(completed,record);self.records.append(record);write_json(self.out/'trials.json',self.records)
        del model,optimizer,row;gc.collect();torch.cuda.empty_cache()
        return record

    def embedding_trials(self,train_store,val_store):
        pairs=FramePairs(train_store,max_samples=256,seed=self.args.seed,raw_images=True)
        vp=FramePairs(val_store,max_samples=64,seed=self.args.seed+1,raw_images=True)
        train=default_collate([pairs[i] for i in range(len(pairs))]);val=default_collate([vp[i] for i in range(len(vp))])
        candidates=[]
        for rate in self.args.rates:
            seed_all(self.args.seed)
            from models import EmbeddingPredictor
            model=EmbeddingPredictor(hidden=self.settings['embedding_hidden'],learned_visual=True,
                visual_input_stride=self.settings['visual_input_stride'],ema=.99,embedding_dim=self.settings['embedding_dim']).cuda()
            optimizer=adamw((p for p in model.parameters() if p.requires_grad),lr=rate)
            generator=torch.Generator().manual_seed(self.args.seed+991);scores=[];best=float('inf')
            checkpoint=self.scratch/f'teacher-embedding-{rate:g}.pt'
            for step in range(self.args.embedding_steps+1):
                if step%max(1,self.args.embedding_steps//4)==0 or step==self.args.embedding_steps:
                    model.eval();values=[]
                    with torch.no_grad(),amp_disabled():
                        for start in range(0,len(vp),8):
                            b={key:value[start:start+8] for key,value in val.items()}
                            values.append(embedding_objective(model,b,cfg)[0])
                    score=float(torch.stack(values).mean());scores.append(score)
                    self.log(dict(model='teacher_embedding',phase='screen',lr=rate,updates=step,validation_embedding_loss=score))
                    if score<best:best=score;save_weights(model.state_dict(),checkpoint)
                if step==self.args.embedding_steps:break
                idx=torch.randperm(len(pairs),generator=generator)[:8]
                b={key:value[idx] for key,value in train.items()};model.train()
                loss,_=embedding_objective(model,b,cfg)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();model.update_target()
            candidates.append(dict(lr=rate,score=float(np.mean(scores[-2:])),best_validation=best,checkpoint=str(checkpoint)))
            del model,optimizer;gc.collect();torch.cuda.empty_cache()
        winner=min(candidates,key=lambda row:row['score'])
        write_json(self.out/'teacher_embedding_trials.json',candidates)
        from models import EmbeddingPredictor
        frame=EmbeddingPredictor(hidden=self.settings['embedding_hidden'],learned_visual=True,
            visual_input_stride=self.settings['visual_input_stride'],ema=.99,embedding_dim=self.settings['embedding_dim']).cuda()
        frame.load_state_dict(torch.load(winner['checkpoint'],map_location='cuda',weights_only=True));frame.eval().requires_grad_(False)
        directory=self.out/'selected'/'teacher';directory.mkdir(parents=True)
        shutil.copy2(winner['checkpoint'],directory/'frame_student.pt')
        self.run.summary['teacher_embedding/selected_lr']=winner['lr']
        return frame,winner

    def execute(self):
        train_files=sorted(self.args.train.glob('rollout*.npz'));val_files=sorted(self.args.validation.glob('rollout*.npz'))
        train=EpisodeStore(train_files,480,640);val=EpisodeStore(val_files,480,640)
        assert not {r['scene_id'] for r in train.entries}&{r['scene_id'] for r in val.entries}
        assert all(112<=r['scene_id']<120 for r in train.entries) and all(192<=r['scene_id']<200 for r in val.entries)
        manifest=dict(train=[dict(file=r['file'],scene_id=r['scene_id']) for r in train.entries],
                      validation=[dict(file=r['file'],scene_id=r['scene_id']) for r in val.entries])
        write_json(self.out/'dataset.json',manifest)
        frame,embedding=self.embedding_trials(train,val)
        val_cache=build_cache(val,4001,frame,training=False)
        screening=build_cache(train,401,frame,training=True,per_flight=4)
        finalists={}
        for name in [*PROFILES,'teacher']:
            trials=[self.trial(name,rate,screening,val_cache,self.args.screen_steps,'screen-2s',self.args.seed) for rate in self.args.rates]
            successful=[r for r in trials if r['status']=='complete']
            finalists[name]=[r['lr'] for r in sorted(successful,key=lambda r:r['score'])[:2]]
            write_json(self.out/'finalists.json',finalists)
        del screening;gc.collect();torch.cuda.empty_cache()
        for horizon in [2001,4001]:
            cache=build_cache(train,horizon,frame,training=True,per_flight=2 if horizon<4001 else 1)
            for name in [*PROFILES,'teacher']:
                for rate in finalists[name]:
                    self.trial(name,rate,cache,val_cache,self.args.confirm_steps,f'confirm-{(horizon-1)//200}s',self.args.seed)
            del cache;gc.collect();torch.cuda.empty_cache()
        recommendations={}
        for name in [*PROFILES,'teacher']:
            scores={}
            for phase in ['screen-2s','confirm-10s','confirm-20s']:
                group=[r for r in self.records if r['model']==name and r['phase']==phase and r['status']=='complete' and r['lr'] in finalists[name]]
                best=min(r['score'] for r in group)
                for row in group:scores[row['lr']]=scores.get(row['lr'],0)+row['score']/max(best,1e-12)
            rate=min(scores,key=scores.get);choices=[r for r in self.records if r['model']==name and r['lr']==rate and r['status']=='complete']
            selected=next(r for r in choices if r['phase']=='confirm-20s')
            directory=self.out/'selected'/name;directory.mkdir(parents=True,exist_ok=True)
            shutil.copy2(selected['checkpoint'],directory/('state_student.pt' if name=='teacher' else 'student.pt'))
            spec=self.settings if name=='teacher' else dict(PROFILES[name],jepa_target=True,profile=name)
            write_json(directory/'student_config.json' if name=='teacher' else directory/'student.json',spec)
            recommendations[name]=dict(lr=rate,baseline_lr=.0001 if name=='teacher' else .0003,
                normalized_score=scores[rate]/3,checkpoint=str(directory),trials=choices,
                provisional=True,selection='mean normalized tail-validation state loss across 2/10/20-second fresh trials')
            self.run.summary[name+'/selected_lr']=rate
        recommendations['teacher_embedding']=embedding
        write_json(self.out/'selected'/'teacher'/'training_config.json',
                   {key:value for key,value in vars(cfg).items() if key.isupper() and isinstance(value,(int,float,str,bool,list,type(None)))})
        write_json(self.out/'recommendations.json',recommendations)
        table=self.wandb.Table(columns=['model','phase','lr','score','ate_m','unaligned_mte_m','seconds'],data=[
            [r['model'],r['phase'],r['lr'],r['score'],r['final_validation']['mean_ate_m'],r['final_validation']['unaligned_mte_m'],r['seconds']]
            for r in self.records if r['status']=='complete'])
        self.run.log({'sweep/results':table});self.run.summary['status']='completed';self.run.finish()
        print('COMPLETE',self.out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,required=True);parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--previous',type=Path,default=Path('checkpoints/sl-parallel-20261115-024206-439181'))
    parser.add_argument('--teacher-config',type=Path,default=Path('checkpoints/20261113-192121-curriculum-tujmpo0u'))
    parser.add_argument('--rates',type=float,nargs='+',default=[1e-5,3e-5,1e-4,3e-4,1e-3])
    parser.add_argument('--screen-steps',type=int,default=512);parser.add_argument('--confirm-steps',type=int,default=512)
    parser.add_argument('--embedding-steps',type=int,default=256);parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--wandb-mode',choices=['online','offline','disabled'],default='online')
    args=parser.parse_args()
    if not torch.cuda.is_available():parser.error('CUDA required')
    if not all(np.isfinite(v) and v>0 for v in args.rates) or min(args.screen_steps,args.confirm_steps,args.embedding_steps)<4:parser.error('invalid sweep budget')
    cfg.DEVICE='cuda';cfg.USE_AMP=True;cfg.TORCH_CPU_THREADS=2;configure(cfg)
    sweep=Sweep(args)
    try:sweep.execute()
    except BaseException:
        sweep.run.finish(exit_code=1);raise


if __name__=='__main__':main()
