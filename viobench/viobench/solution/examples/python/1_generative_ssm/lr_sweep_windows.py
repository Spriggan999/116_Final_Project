"""Complete the curriculum-window LR study, reusing finished state trials.

Student embedding trials change the online encoder/forecaster parameter-group
LR under the normal joint SL objective, with the state LR held fixed. The teacher
embedding stage uses frame pairs, has no state horizon, and is reused unchanged.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import fcntl
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import queue
from types import SimpleNamespace

import numpy as np
import torch

import config as cfg
from data import EpisodeStore
from export import normalize_settings
from lr_sweep import Sweep,build_cache,write_json
from models import EmbeddingPredictor
from performance import configure
from students.model import PROFILES

WINDOWS=list(range(2,21,2))
SEEDS=(17,29)
BASE_RATES=(1e-5,3e-5,1e-4,3e-4,1e-3)
BOUNDARY_RATES=(1e-7,3e-7,1e-6,3e-6,*BASE_RATES,3e-3,1e-2)
ORDER=('1k','10k','100k','1m_ssm','1m','1m_hybrid','100m','100m_hybrid','teacher')
WORKER=None


def horizon(record):
    if 'window_seconds' in record:return record['window_seconds']
    match=re.search(r'(?:^|-)(\d+)s(?:-|$)',record['phase'])
    if not match:raise ValueError(f'No window in phase {record["phase"]}')
    return int(match[1])


def successful(records):
    return [row for row in records if row['status']=='complete' and row.get('score') is not None
            and np.isfinite(row['score'])]


def common_seed_ranking(records,key='lr',metric='score'):
    """Exclude candidates missing a successful matched comparison in any seed."""
    groups={seed:[row for row in successful(records) if row['seed']==seed
                  and row.get(metric) is not None and np.isfinite(row[metric])] for seed in SEEDS}
    candidates=set.intersection(*[{row[key] for row in group} for group in groups.values()])
    if not candidates:raise ValueError('No rate has complete matched-seed results')
    scores={rate:0. for rate in candidates}
    for group in groups.values():
        by_rate={row[key]:row[metric] for row in group if row[key] in candidates}
        if len(by_rate)!=sum(row[key] in candidates for row in group):
            raise ValueError('Duplicate model/window/seed/rate trial')
        denominator=max(min(by_rate.values()),1e-12)
        for rate,value in by_rate.items():scores[rate]+=value/denominator/len(SEEDS)
    return sorted(scores,key=lambda rate:(scores[rate],rate)),scores


def embedding_boundary_rates(records):
    rows=successful(records)
    if not rows:return []
    core=sorted({row['embedding_lr'] for row in rows})
    winners=[min(rows,key=lambda row:(row[key],row['embedding_lr']))['embedding_lr']
             for key in ('score','embedding_score')]
    extra=[]
    if core[0] in winners:extra.extend([rate for rate in BOUNDARY_RATES if rate<core[0]][-2:])
    if core[-1] in winners:extra.extend([rate for rate in BOUNDARY_RATES if rate>core[-1]][:2])
    return sorted(set(extra))


def tensor_tree(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,dict):return {key:tensor_tree(item,device) for key,item in value.items()}
    if isinstance(value,list):return [tensor_tree(item,device) for item in value]
    return value


def source_fingerprint(parent,data_root):
    digest=hashlib.sha256()
    for name in ['selected/teacher/frame_student.pt','selected/teacher/student_config.json',
                 'confirmation/all_trials.json']:
        path=parent/name
        digest.update(name.encode());digest.update(path.read_bytes())
    for split in ['train','validation']:
        for path in sorted((data_root/split).glob('rollout*.npz')):
            info=path.stat();digest.update(f'{path.resolve()}:{info.st_size}:{info.st_mtime_ns}'.encode())
    digest.update(json.dumps(PROFILES,sort_keys=True).encode())
    return digest.hexdigest()


class TrialWorker(Sweep):
    """GPU trial process; only the parent owns the W&B run and aggregate files."""
    def __init__(self,scratch,settings,initial_hashes,events,source_digest):
        self.scratch=Path(scratch);self.settings=settings;self.initial_hashes=dict(initial_hashes)
        self.out=self.scratch/f'worker-{os.getpid()}';self.out.mkdir(exist_ok=True)
        self.events=events;self.source_digest=source_digest;self.cached_seconds=None
        self.cache=None;self.val=None;self.records=[]

    def log(self,record):self.events.put(record)

    def execute_job(self,job):
        if self.val is None:
            self.val=torch.load(self.scratch/f'cache-validation-20s-{self.source_digest[:16]}.pt',
                                map_location='cuda',weights_only=True)
        if self.cached_seconds!=job['seconds']:
            self.cache=None;gc.collect();torch.cuda.empty_cache()
            self.cache=torch.load(self.scratch/f'cache-train-{job["seconds"]}s-{self.source_digest[:16]}.pt',
                                  map_location='cuda',weights_only=True)
            self.cached_seconds=job['seconds']
        self.records=[]
        return self.trial(job['name'],job['rate'],self.cache,self.val,512 if job['seed']==17 else 1024,
                          f'window-{job["seconds"]}s',job['seed'],embedding_rate=job['embedding_rate'])


def initialize_worker(scratch,settings,initial_hashes,events,source_digest):
    global WORKER
    cfg.DEVICE='cuda';cfg.USE_AMP=True;cfg.TORCH_CPU_THREADS=2;configure(cfg)
    WORKER=TrialWorker(scratch,settings,initial_hashes,events,source_digest)


def worker_trial(job):return WORKER.execute_job(job)


class WindowSweep(Sweep):
    def __init__(self,args):
        self.args=args;self.out=args.out;self.out.mkdir(parents=True,exist_ok=True)
        self.lock=(self.out/'sweep.lock').open('a')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.scratch=args.scratch;self.scratch.mkdir(parents=True,exist_ok=True)
        self.parent=args.screen
        self.teacher_embedding=json.loads((self.parent/'recommendations.json').read_text())['teacher_embedding']
        self.settings=normalize_settings(json.loads((self.parent/'selected/teacher/student_config.json').read_text()))
        self.previous=json.loads((self.parent/'confirmation/all_trials.json').read_text())
        self.records=json.loads((self.out/'trials.json').read_text()) if (self.out/'trials.json').exists() else []
        self.initial_hashes={}
        for row in self.previous+self.records:
            key=(row['model'],row['seed']);digest=row['initial_sha256']
            if key in self.initial_hashes and self.initial_hashes[key]!=digest:
                raise ValueError('Source trial initializations disagree')
            self.initial_hashes[key]=digest
        self.source_digest=source_fingerprint(self.parent,args.data_root)
        protocol=dict(parent=str(self.parent.resolve()),data_root=str(args.data_root.resolve()),
            source_fingerprint=self.source_digest,windows_seconds=WINDOWS,seeds=list(SEEDS),
            state_screen_rates=list(BASE_RATES),embedding_screen_rates=list(BASE_RATES),
            state_screen_steps=512,state_confirmation_steps=1024,
            validation_seconds=20,student_embedding_objective='unchanged joint SL: state + 0.1 JEPA + 0.1 alignment',
            embedding_parameter_group=['vision','embedding_predictor'],
            embedding_state_lr='fixed to the state sweep winner for that model/window',
            teacher_embedding='reused completed frame-pair sweep; no trajectory-window parameter exists',
            state_selection='mean normalized tail state loss across matched seeds',
            embedding_selection='report state-loss and JEPA-loss winners separately',
            fresh_initialization=True,production_defaults_changed=False,
            limitations=['Fixed eight-scene training subset, not full curriculum training.',
                         'Mid-flight window position/velocity target inconsistency left unchanged.',
                         'Only default motion/noise/vision levels are tested.',
                         'Second seed gets 1024 updates; first seed gets 512, matched within each seed.',
                         'Original 2/10/20-second state trials are reused with their original candidate pools.',
                         'Embedding confirmation uses a shortlist; rates remain pilot candidates.'])
        path=self.out/'protocol.json'
        if path.exists() and json.loads(path.read_text())!=protocol:
            raise ValueError('Resume protocol/source changed; use a new output directory')
        write_json(path,protocol)
        import wandb
        self.wandb=wandb
        previous=json.loads((self.out/'wandb.json').read_text()) if (self.out/'wandb.json').exists() else None
        self.run=wandb.init(project=cfg.WANDB_PROJECT,dir=str(self.out),mode=args.wandb_mode,
            id=previous['id'] if previous else None,resume='allow' if previous else None,
            job_type='curriculum_window_lr_sweep',name=self.out.name,config=protocol)
        write_json(self.out/'wandb.json',dict(id=self.run.id,url=self.run.url))
        self.global_step=sum(1 for _ in (self.out/'history.jsonl').open()) if (self.out/'history.jsonl').exists() else 0
        self.run.summary['status']='running';self.progress('starting')
        self.run.config.update({'concurrent_trial_workers':args.workers},allow_val_change=True)
        self.executor=None;self.events=None
        print('W&B',self.run.url,flush=True)

    def progress(self,phase,**extra):
        record=dict(phase=phase,completed_new_trials=len(self.records),**extra)
        write_json(self.out/'progress.json',record)
        self.run.summary['progress']=record
        print('PROGRESS',json.dumps(record),flush=True)

    def log(self,record):
        record=dict(record)
        record['window_seconds']=horizon(record)
        record['component']='embedding_group' if 'embedding_lr' in record else 'state'
        super().log(record)

    def group(self,name,seconds,embedding=False):
        return [row for row in self.previous+self.records if row['model']==name and horizon(row)==seconds
                and ('embedding_lr' in row)==embedding]

    def get_trial(self,name,seconds,seed,rate,cache,val,embedding_rate=None):
        pool=self.group(name,seconds,embedding_rate is not None)
        match=[row for row in pool if row['seed']==seed and row['lr']==rate
               and (embedding_rate is None or row.get('embedding_lr')==embedding_rate)]
        if match:
            if len(match)!=1:raise ValueError('Ambiguous existing trial')
            return match[0]
        steps=512 if seed==17 else 1024
        phase=f'window-{seconds}s'
        self.progress('embedding' if embedding_rate is not None else 'state',model=name,window_seconds=seconds,
                      seed=seed,state_lr=rate,embedding_lr=embedding_rate)
        result=self.trial(name,rate,cache,val,steps,phase,seed,embedding_rate=embedding_rate)
        return result

    def run_trials(self,name,seconds,seed,pairs,cache,val):
        if getattr(self,'executor',None) is None:
            return [self.get_trial(name,seconds,seed,rate,cache,val,embedding_rate=embedding_rate)
                    for rate,embedding_rate in pairs]
        futures={};completed=[]
        for rate,embedding_rate in pairs:
            match=[row for row in self.group(name,seconds,embedding_rate is not None)
                   if row['seed']==seed and row['lr']==rate
                   and (embedding_rate is None or row.get('embedding_lr')==embedding_rate)]
            if len(match)>1:raise ValueError('Ambiguous existing trial')
            if match:completed.append(match[0]);continue
            job=dict(name=name,seconds=seconds,seed=seed,rate=rate,embedding_rate=embedding_rate)
            futures[self.executor.submit(worker_trial,job)]=job
        if futures:
            self.progress('parallel_embedding' if pairs[0][1] is not None else 'parallel_state',
                          model=name,window_seconds=seconds,seed=seed,pending_trials=len(futures),workers=self.args.workers)
        def drain():
            while True:
                try:record=self.events.get_nowait()
                except queue.Empty:break
                self.log(record)
        while futures:
            drain()
            done,_=wait(futures,timeout=.2,return_when=FIRST_COMPLETED)
            for future in done:
                job=futures.pop(future);record=future.result()
                self.records.append(record);completed.append(record)
                write_json(self.out/'trials.json',self.records)
                self.progress('trial_complete',model=name,window_seconds=seconds,seed=seed,
                              state_lr=job['rate'],embedding_lr=job['embedding_rate'],
                              pending_trials=len(futures),status=record['status'])
        drain()
        return completed

    def cache(self,store,seconds,frame,training):
        split='train' if training else 'validation'
        path=self.scratch/f'cache-{split}-{seconds}s-{self.source_digest[:16]}.pt'
        if path.exists():return torch.load(path,map_location='cuda',weights_only=True)
        per_flight=4 if seconds==2 else 1 if seconds==20 else 2
        result=build_cache(store,seconds*200+1,frame,training=training,per_flight=per_flight)
        temporary=path.with_suffix('.tmp')
        torch.save(tensor_tree(result,'cpu'),temporary);temporary.replace(path)
        return result

    def state_window(self,name,seconds,cache,val):
        if seconds not in (2,10,20):
            old=json.loads((self.parent/'confirmation/recommendations.json').read_text())[name]
            rates=sorted(set([*BASE_RATES,*old['best_lr_by_horizon'].values()]))
            self.run_trials(name,seconds,17,[(rate,None) for rate in rates],cache,val)
            screen=sorted(successful([row for row in self.group(name,seconds) if row['seed']==17]),key=lambda row:(row['score'],row['lr']))
            if not screen:raise ValueError(f'All state candidates failed for {name}/{seconds}s')
            baseline=1e-4 if name=='teacher' else 3e-4
            shortlist=sorted({baseline,*[row['lr'] for row in screen[:2]]})
            self.run_trials(name,seconds,29,[(rate,None) for rate in shortlist],cache,val)
        ranking,scores=common_seed_ranking(self.group(name,seconds))
        return dict(lr=ranking[0],candidate_scores=scores,baseline_lr=1e-4 if name=='teacher' else 3e-4,
                    reused_state_window=seconds in (2,10,20),provisional=True)

    def embedding_window(self,name,seconds,state_lr,cache,val):
        rates=sorted(set([*BASE_RATES,state_lr]))
        self.run_trials(name,seconds,17,[(state_lr,rate) for rate in rates],cache,val)
        screen=successful([row for row in self.group(name,seconds,True) if row['seed']==17 and row['lr']==state_lr])
        if not screen:raise ValueError(f'All embedding candidates failed for {name}/{seconds}s')
        state_short=sorted(screen,key=lambda row:(row['score'],row['embedding_lr']))[:2]
        embedding_short=sorted(screen,key=lambda row:(row['embedding_score'],row['embedding_lr']))[:1]
        shortlist=sorted({state_lr,*[row['embedding_lr'] for row in state_short+embedding_short]})
        self.run_trials(name,seconds,29,[(state_lr,rate) for rate in shortlist],cache,val)
        group=[row for row in self.group(name,seconds,True) if row['lr']==state_lr]
        state_rank,state_scores=common_seed_ranking(group,'embedding_lr','score')
        embedding_rank,embedding_scores=common_seed_ranking(group,'embedding_lr','embedding_score')
        return dict(state_lr=state_lr,embedding_lr=state_rank[0],embedding_loss_winner_lr=embedding_rank[0],
                    candidate_state_scores=state_scores,candidate_embedding_scores=embedding_scores,
                    jointly_trained=True,provisional=True)

    def report(self,results,finished=False):
        write_json(self.out/'window_results.json',results)
        all_trials=self.previous+self.records
        write_json(self.out/'all_trials.json',all_trials)
        lines=['# Curriculum-window LR pilot','',
            'Experimental candidates only. State rates use shared model/embedding LR; embedding-group trials then',
            'hold the selected state LR fixed and vary the encoder + embedding predictor LR under joint SL.',
            'The state and JEPA columns can choose different embedding rates. All validation uses complete 20-second flights.',
            'The original target equations are unchanged; the known mid-flight position/velocity issue remains.',
            'Two fresh initializations; 512 updates for seed 17 and 1024 for seed 29. Production defaults are unchanged.',
            '',f'Status: {"complete" if finished else "running"}. New trials: {len(self.records)}.',
            '', '## State LR by training window','',
            '| Model | '+' | '.join(f'{window} s' for window in WINDOWS)+' |',
            '|---|'+'---:|'*len(WINDOWS)]
        for name in ORDER:
            values=[results.get(str(window),{}).get('state',{}).get(name,{}).get('lr') for window in WINDOWS]
            lines.append('| '+name+' | '+' | '.join(f'{value:g}' if value else 'pending' for value in values)+' |')
        for title,field in [('Embedding-group LR ranked by state loss','embedding_lr'),
                            ('Embedding-group LR ranked by JEPA loss','embedding_loss_winner_lr')]:
            lines+=['', '## '+title,'',
                '| Model | '+' | '.join(f'{window} s' for window in WINDOWS)+' |',
                '|---|'+'---:|'*len(WINDOWS)]
            for name in ORDER[:-1]:
                values=[results.get(str(window),{}).get('embedding',{}).get(name,{}).get(field) for window in WINDOWS]
                lines.append('| '+name+' | '+' | '.join(f'{value:g}' if value else 'pending' for value in values)+' |')
        lines+=['', f'Teacher embedding: {self.teacher_embedding["lr"]:g} from the completed frame-pair sweep. Its objective has no trajectory window.',
                'Student embedding-group results are conditional on the state LR; this is not an exhaustive Cartesian two-LR search.',
                'Selection uses only the tuning-validation split. No independent-test results are used for LR selection.',
                'See protocol.json, all_trials.json and the prior study for candidate pools and further limits.',
                '',f'[W&B]({self.run.url})','']
        if (self.out/'embedding_boundary_protocol.json').exists():
            lines+=['Embedding edge candidates were extended once within the bounds 1e-7–1e-2.',
                    'See embedding_boundary_protocol.json; JSON results flag winners still at a tested edge.',
                    'A boundary winner is not an established optimum.','']
        (self.out/'comparison.md').write_text('\n'.join(lines))
        table=[]
        for window,groups in results.items():
            for name,row in groups['state'].items():
                embedding=groups['embedding'].get(name,{})
                table.append([name,int(window),row['lr'],embedding.get('embedding_lr'),embedding.get('embedding_loss_winner_lr')])
        if finished:
            self.run.log({'windows/recommendations':self.wandb.Table(
                columns=['model','window_seconds','state_lr','embedding_lr_state_selected','embedding_lr_jepa_selected'],data=table)})
            self.run.log({'windows/trials':self.wandb.Table(
                columns=['model','window_seconds','seed','state_lr','embedding_lr','status','tail_state_loss','tail_embedding_loss','seconds'],
                data=[[row['model'],horizon(row),row['seed'],row['lr'],row.get('embedding_lr'),row['status'],
                       row.get('score'),row.get('embedding_score'),row['seconds']] for row in all_trials])})
            self.run.save(str(self.out/'window_results.json'),base_path=str(self.out),policy='now')
            self.run.save(str(self.out/'comparison.md'),base_path=str(self.out),policy='now')

    def execute(self):
        train=EpisodeStore(sorted((self.args.data_root/'train').glob('rollout*.npz')),480,640)
        validation=EpisodeStore(sorted((self.args.data_root/'validation').glob('rollout*.npz')),480,640)
        train_scenes={entry['scene_id'] for entry in train.entries}
        val_scenes={entry['scene_id'] for entry in validation.entries}
        if train_scenes!=set(range(112,120)) or val_scenes!=set(range(192,200)):
            raise ValueError('Expected original scene-disjoint pilot splits')
        frame=EmbeddingPredictor(hidden=self.settings['embedding_hidden'],learned_visual=True,
            visual_input_stride=self.settings['visual_input_stride'],ema=.99,embedding_dim=self.settings['embedding_dim']).cuda()
        frame.load_state_dict(torch.load(self.parent/'selected/teacher/frame_student.pt',map_location='cuda',weights_only=True))
        frame.eval().requires_grad_(False)
        self.progress('validation_cache')
        val=self.cache(validation,20,frame,False)
        del validation;gc.collect()
        if self.args.workers>1:
            context=multiprocessing.get_context('spawn');self.events=context.Queue()
            self.executor=ProcessPoolExecutor(max_workers=self.args.workers,mp_context=context,
                initializer=initialize_worker,initargs=(str(self.scratch),self.settings,self.initial_hashes,
                                                       self.events,self.source_digest))
        results={}
        for seconds in WINDOWS:
            self.progress('training_cache',window_seconds=seconds)
            cache=self.cache(train,seconds,frame,True)
            results[str(seconds)]=dict(state={},embedding={})
            for name in ORDER:
                results[str(seconds)]['state'][name]=self.state_window(name,seconds,cache,val)
                self.report(results)
            for name in ORDER[:-1]:
                state_lr=results[str(seconds)]['state'][name]['lr']
                results[str(seconds)]['embedding'][name]=self.embedding_window(name,seconds,state_lr,cache,val)
                self.report(results)
            self.progress('window_complete',window_seconds=seconds)
            del cache;gc.collect();torch.cuda.empty_cache()
        if self.executor is not None:self.executor.shutdown();self.executor=None
        self.report(results,finished=True)
        counts=collections.Counter(row['status'] for row in self.records)
        self.run.summary['trial_counts']=dict(counts)
        self.run.summary['completed_windows']=len(results)
        self.run.summary['status']='completed';self.progress('complete',windows=len(results))
        self.run.finish();print('ALL WINDOWS COMPLETE',self.out,flush=True)


class EmbeddingBoundarySweep(WindowSweep):
    """Supplement completed window trials with bounded embedding edge checks."""
    def execute(self):
        results=json.loads((self.out/'window_results.json').read_text())
        if any(str(window) not in results or len(results[str(window)]['state'])!=9
               or len(results[str(window)]['embedding'])!=8 for window in WINDOWS):
            raise ValueError('Complete the main window sweep before extending embedding boundaries')
        extension=dict(purpose='one bounded extension beyond initial embedding-group edge winners',
            bounds=[min(BOUNDARY_RATES),max(BOUNDARY_RATES)],
            primary_state_rates_unchanged=True,seed_budgets={'17':512,'29':1024},
            next_rates='two lower/upper neighbors when either state-loss or JEPA-loss screen winner is an edge',
            repeat_initial_window_trials=False)
        path=self.out/'embedding_boundary_protocol.json'
        if path.exists() and json.loads(path.read_text())!=extension:raise ValueError('Boundary protocol changed')
        write_json(path,extension);self.run.config.update({'embedding_boundary_extension':extension},allow_val_change=True)
        context=multiprocessing.get_context('spawn');self.events=context.Queue()
        self.executor=ProcessPoolExecutor(max_workers=self.args.workers,mp_context=context,
            initializer=initialize_worker,initargs=(str(self.scratch),self.settings,self.initial_hashes,
                                                   self.events,self.source_digest))
        val_path=self.scratch/f'cache-validation-20s-{self.source_digest[:16]}.pt'
        if not val_path.exists():raise ValueError('Missing fixed validation cache; recover main sweep first')
        val=torch.load(val_path,map_location='cuda',weights_only=True)
        for seconds in WINDOWS:
            train_path=self.scratch/f'cache-train-{seconds}s-{self.source_digest[:16]}.pt'
            if not train_path.exists():raise ValueError('Missing fixed training cache; recover main sweep first')
            cache=torch.load(train_path,map_location='cuda',weights_only=True)
            for name in ORDER[:-1]:
                state_lr=results[str(seconds)]['state'][name]['lr']
                # Always derive the extension from the original screen, so a
                # recovery cannot recursively widen the search after each stop.
                original=sorted({*BASE_RATES,state_lr})
                screen=[row for row in self.group(name,seconds,True) if row['seed']==17
                        and row['lr']==state_lr and row['embedding_lr'] in original]
                extra=embedding_boundary_rates(screen)
                self.run_trials(name,seconds,17,[(state_lr,rate) for rate in extra],cache,val)
                # The ordinary method reuses every completed core trial, then
                # includes successful new edge rates when making the shortlist.
                row=self.embedding_window(name,seconds,state_lr,cache,val)
                row['boundary_extension_rates']=extra
                group=[item for item in successful(self.group(name,seconds,True)) if item['lr']==state_lr]
                tested=sorted({item['embedding_lr'] for item in group})
                row['state_winner_at_tested_edge']=row['embedding_lr'] in (tested[0],tested[-1])
                row['jepa_winner_at_tested_edge']=row['embedding_loss_winner_lr'] in (tested[0],tested[-1])
                results[str(seconds)]['embedding'][name]=row;self.report(results)
            del cache;gc.collect();torch.cuda.empty_cache()
        self.executor.shutdown();self.executor=None
        self.report(results,finished=True)
        self.run.summary['status']='completed';self.run.summary['embedding_boundary_extension_complete']=True
        self.run.summary['trial_counts']=dict(collections.Counter(row['status'] for row in self.records))
        self.progress('complete',windows=len(results),embedding_boundary_extension=True)
        self.run.finish();print('EMBEDDING BOUNDARY CHECKS COMPLETE',self.out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3,help='isolated concurrent GPU trial processes; 1 runs serially')
    parser.add_argument('--embedding-boundaries',action='store_true',help='after main sweep, extend initial embedding edge winners once')
    parser.add_argument('--wandb-mode',choices=['online','offline','disabled'],default='online')
    args=parser.parse_args()
    if not torch.cuda.is_available():parser.error('CUDA required')
    if not 1<=args.workers<=3:parser.error('use 1–3 workers to retain VRAM headroom')
    cfg.DEVICE='cuda';cfg.USE_AMP=True;cfg.TORCH_CPU_THREADS=2;configure(cfg)
    cls=EmbeddingBoundarySweep if args.embedding_boundaries else WindowSweep
    sweep=cls(SimpleNamespace(**vars(args)))
    try:sweep.execute()
    except BaseException:
        if sweep.executor is not None:sweep.executor.shutdown(cancel_futures=True)
        sweep.run.finish(exit_code=1);raise


if __name__=='__main__':main()
