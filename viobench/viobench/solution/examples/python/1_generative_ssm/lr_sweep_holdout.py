"""One baseline-versus-selected comparison on additional unseen ProcTHOR scenes."""
import argparse
import gc
import json
from pathlib import Path

import torch

import config as cfg
from data import EpisodeStore
from export import normalize_settings
from lr_sweep import CachedStudent,build_cache,evaluate,write_json
from models import EmbeddingPredictor,HybridStateStudent
from performance import configure
from students.model import PROFILES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep',type=Path,required=True);parser.add_argument('--test',type=Path,required=True)
    args=parser.parse_args();parent=args.sweep;confirmation=parent/'confirmation'
    recommendations=json.loads((confirmation/'recommendations.json').read_text())
    trials=json.loads((confirmation/'all_trials.json').read_text())
    settings=normalize_settings(json.loads((parent/'selected/teacher/student_config.json').read_text()))
    cfg.DEVICE='cuda';cfg.USE_AMP=True;cfg.TORCH_CPU_THREADS=2;configure(cfg)
    frame=EmbeddingPredictor(hidden=settings['embedding_hidden'],learned_visual=True,
        visual_input_stride=settings['visual_input_stride'],ema=.99,embedding_dim=settings['embedding_dim']).cuda()
    frame.load_state_dict(torch.load(parent/'selected/teacher/frame_student.pt',map_location='cuda',weights_only=True));frame.eval().requires_grad_(False)
    store=EpisodeStore(sorted(args.test.glob('rollout*.npz')),480,640)
    assert all(160<=row['scene_id']<168 for row in store.entries)
    cache=build_cache(store,4001,frame,training=False)
    result={}
    import wandb
    run=wandb.init(project=cfg.WANDB_PROJECT,dir=str(confirmation),job_type='lr_sweep_independent_check',
        name=parent.name+'-independent-check',mode='online',config=dict(
            parent=str(parent),test=str(args.test),scenes=list(range(160,168)),selection_used_these_scenes=False))
    write_json(confirmation/'holdout_wandb.json',dict(url=run.url,id=run.id))
    for name,recommendation in recommendations.items():
        result[name]={}
        for label,rate in [('baseline',recommendation['baseline_lr']),('selected',recommendation['lr'])]:
            record=next(row for row in trials if row['model']==name and row['phase']=='seed29-20s' and row['lr']==rate and row['status']=='complete')
            if name=='teacher':
                model=HybridStateStudent(hidden=settings['hidden'],n_mamba=settings['mamba_layers'],
                    attn_window=settings['attention_window'],d_state=settings['hybrid_d_state'],headdim=settings['hybrid_head_dim'],
                    heads=settings['hybrid_attention_heads'],learned_visual=True,visual_input_stride=settings['visual_input_stride'],
                    visual_embeddings=True,embedding_dim=settings['embedding_dim']).cuda()
            else:model=CachedStudent(dict(PROFILES[name],jepa_target=True)).cuda()
            model.load_state_dict(torch.load(record['checkpoint'],map_location='cuda',weights_only=True))
            metrics=evaluate(model,cache,name);result[name][label]=dict(lr=rate,best_step=record['best_step'],**metrics)
            run.log({name+'/'+label+'/'+key:value for key,value in metrics.items()})
            print(name,label,rate,json.dumps(metrics),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
        before=result[name]['baseline'];after=result[name]['selected']
        result[name]['state_loss_change_pct']=100*(after['state_loss']/before['state_loss']-1)
        result[name]['ate_change_pct']=100*(after['mean_ate_m']/before['mean_ate_m']-1)
        run.summary[name+'/state_loss_change_pct']=result[name]['state_loss_change_pct']
        run.summary[name+'/ate_change_pct']=result[name]['ate_change_pct']
        write_json(confirmation/'independent_check.json',result)
    run.log({'independent/comparison':wandb.Table(
        columns=['model','baseline_lr','selected_lr','baseline_state_loss','selected_state_loss',
                 'baseline_sim3_ate_m','selected_sim3_ate_m','state_loss_change_pct','ate_change_pct'],
        data=[[name,row['baseline']['lr'],row['selected']['lr'],row['baseline']['state_loss'],
               row['selected']['state_loss'],row['baseline']['mean_ate_m'],row['selected']['mean_ate_m'],
               row['state_loss_change_pct'],row['ate_change_pct']] for name,row in result.items()])})
    run.save(str(confirmation/'independent_check.json'),base_path=str(confirmation),policy='now')
    run.summary['status']='completed'
    run.finish();print('Independent comparison complete',flush=True)


if __name__=='__main__':main()
