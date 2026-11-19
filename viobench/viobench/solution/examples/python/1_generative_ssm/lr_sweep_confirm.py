"""Expand edge winners and confirm LR candidates plus the original baseline."""
import argparse
import gc
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch

import config as cfg
from data import EpisodeStore
from lr_sweep import Sweep,build_cache,write_json
from models import EmbeddingPredictor
from performance import configure
from students.model import PROFILES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1024)
    args=parser.parse_args()
    parent=args.screen
    records=json.loads((parent/'trials.json').read_text())
    options=SimpleNamespace(out=parent/'confirmation',scratch=args.data_root/'confirmation-checkpoints',
        previous=Path('checkpoints/sl-parallel-20261115-024206-439181'),
        teacher_config=Path('checkpoints/20261113-192121-curriculum-tujmpo0u'),
        rates=[1e-5,3e-5,1e-4,3e-4,1e-3],seed=29,screen_steps=512,confirm_steps=args.steps,
        embedding_steps=256,train=args.data_root/'train',validation=args.data_root/'validation',wandb_mode='online')
    cfg.DEVICE='cuda';cfg.USE_AMP=True;cfg.TORCH_CPU_THREADS=2;configure(cfg)
    sweep=Sweep(options);sweep.run.summary['parent_sweep']=json.loads((parent/'wandb.json').read_text())['url']
    sweep.run.config.update({'purpose':'edge expansion, mandatory baseline, second state initialization',
                             'teacher_encoder':'fixed freshly trained encoder from parent sweep'})
    for record in records:sweep.initial_hashes[(record['model'],record['seed'])]=record['initial_sha256']
    frame=EmbeddingPredictor(hidden=sweep.settings['embedding_hidden'],learned_visual=True,
        visual_input_stride=sweep.settings['visual_input_stride'],ema=.99,embedding_dim=sweep.settings['embedding_dim']).cuda()
    frame.load_state_dict(torch.load(parent/'selected/teacher/frame_student.pt',map_location='cuda',weights_only=True))
    frame.eval().requires_grad_(False)
    train=EpisodeStore(sorted(options.train.glob('rollout*.npz')),480,640)
    validation=EpisodeStore(sorted(options.validation.glob('rollout*.npz')),480,640)
    val=build_cache(validation,4001,frame,training=False)
    cache=build_cache(train,401,frame,per_flight=4)
    finalists={}
    for name in [*PROFILES,'teacher']:
        group=[r for r in records if r['model']==name and r['phase']=='screen-2s' and r['status']=='complete']
        winner=min(group,key=lambda r:r['score'])
        extra=[1e-6,3e-6] if winner['lr']==min(options.rates) else [3e-3,1e-2] if winner['lr']==max(options.rates) else []
        for rate in extra:
            trial=sweep.trial(name,rate,cache,val,512,'edge-2s',17);group.append(trial)
        successful=[r for r in group if r['status']=='complete']
        rates=[r['lr'] for r in sorted(successful,key=lambda r:r['score'])[:2]]
        baseline=1e-4 if name=='teacher' else 3e-4
        finalists[name]=sorted(set([*rates,baseline]))
        for rate in finalists[name]:sweep.trial(name,rate,cache,val,args.steps,'seed29-2s',29)
        write_json(sweep.out/'finalists.json',finalists)
    del cache;gc.collect();torch.cuda.empty_cache()
    for horizon in [2001,4001]:
        cache=build_cache(train,horizon,frame,per_flight=2 if horizon<4001 else 1)
        seconds=(horizon-1)//200
        for name,rates in finalists.items():
            for rate in rates:
                existing=[r for r in records if r['model']==name and r['lr']==rate and r['phase']==f'confirm-{seconds}s' and r['status']=='complete']
                if not existing:sweep.trial(name,rate,cache,val,512,f'edge-{seconds}s',17)
                sweep.trial(name,rate,cache,val,args.steps,f'seed29-{seconds}s',29)
        del cache;gc.collect();torch.cuda.empty_cache()
    combined=records+sweep.records
    result={}
    phases=[('screen-2s','edge-2s'),('confirm-10s','edge-10s'),('confirm-20s','edge-20s'),
            ('seed29-2s',),('seed29-10s',),('seed29-20s',)]
    for name,rates in finalists.items():
        scores={rate:[] for rate in rates}
        for accepted in phases:
            group=[r for r in combined if r['model']==name and r['phase'] in accepted and r['lr'] in rates and r['status']=='complete']
            minimum=min(r['score'] for r in group)
            for record in group:scores[record['lr']].append(record['score']/max(minimum,1e-12))
        valid={rate:sum(values)/len(phases) for rate,values in scores.items() if len(values)==len(phases)}
        selected=min(valid,key=valid.get);baseline=1e-4 if name=='teacher' else 3e-4
        trial=next(r for r in combined if r['model']==name and r['lr']==selected and r['phase']=='seed29-20s')
        directory=sweep.out/'selected'/name;directory.mkdir(parents=True,exist_ok=True)
        shutil.copy2(trial['checkpoint'],directory/('state_student.pt' if name=='teacher' else 'student.pt'))
        spec=sweep.settings if name=='teacher' else dict(PROFILES[name],jepa_target=True,profile=name)
        write_json(directory/('student_config.json' if name=='teacher' else 'student.json'),spec)
        if name=='teacher':shutil.copy2(parent/'selected/teacher/frame_student.pt',directory/'frame_student.pt')
        per_horizon={}
        for horizon,accepted in [(2,('screen-2s','edge-2s','seed29-2s')),(10,('confirm-10s','edge-10s','seed29-10s')),(20,('confirm-20s','edge-20s','seed29-20s'))]:
            horizon_scores={}
            for seed in [17,29]:
                group=[r for r in combined if r['model']==name and r['phase'] in accepted and r['seed']==seed and r['lr'] in rates and r['status']=='complete']
                minimum=min(r['score'] for r in group)
                for r in group:horizon_scores[r['lr']]=horizon_scores.get(r['lr'],0)+r['score']/max(minimum,1e-12)
            per_horizon[str(horizon)]=min(horizon_scores,key=horizon_scores.get)
        result[name]=dict(lr=selected,baseline_lr=baseline,best_lr_by_horizon=per_horizon,normalized_validation_score=valid[selected],
            baseline_normalized_validation_score=valid.get(baseline),candidate_scores=valid,
            relative_score_reduction_pct=100*(1-valid[selected]/valid[baseline]) if baseline in valid else None,
            checkpoint=str(directory),initialization_seeds=[17,29],horizons_seconds=[2,10,20],
            provisional=True,trial=trial)
        sweep.run.summary[name+'/selected_lr']=selected
        sweep.run.summary[name+'/baseline_relative_score_reduction_pct']=result[name]['relative_score_reduction_pct']
    write_json(sweep.out/'recommendations.json',result)
    write_json(sweep.out/'student_lrs.json',{name:row['lr'] for name,row in result.items() if name!='teacher'})
    write_json(sweep.out/'all_trials.json',combined)
    table=sweep.wandb.Table(columns=['model','baseline_lr','recommended_lr','relative_score_reduction_pct'],
        data=[[name,row['baseline_lr'],row['lr'],row['relative_score_reduction_pct']] for name,row in result.items()])
    sweep.run.log({'sweep/recommendations':table});sweep.run.summary['status']='completed';sweep.run.finish()
    print('CONFIRMATION COMPLETE',sweep.out,flush=True)


if __name__=='__main__':main()
