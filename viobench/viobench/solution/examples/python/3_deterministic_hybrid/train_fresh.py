"""Generate one fixed, varied GPU dataset and train each hybrid to a validation plateau.

Run in myenv on the NVIDIA machine. The simulator uses the GPU; the tiny neural
networks train on CPU. Public scene 1 and held-out scenes 192+ are never sampled.
"""
import argparse
from dataclasses import asdict
import ctypes as ct
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import torch

from prepare import HERE, NAMES, generate
from train import ABI, fit, sha

FULL_HERE=HERE
FULL_GENERATE=generate
FULL_FIT=fit
CURRENT_PROFILE="full"


def atomic_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str));temporary.replace(path)


def check_scenes(training,validation):
    if not training or not validation:raise ValueError('Both scene splits must be nonempty')
    if len(set(training))!=len(training) or len(set(validation))!=len(validation):raise ValueError('Duplicate scenes')
    if set(training)&set(validation):raise ValueError('Training and validation scenes must be disjoint')
    if any(i<0 or i>=192 or i==1 for i in training+validation):
        raise ValueError('Use ProcTHOR indices 0..191 except public scene 1; 192+ are held out')


class TraceCollector:
    def __init__(self, directory,prepare_sources=True):
        if prepare_sources:generate()
        library=directory/'collector.so'
        subprocess.run(['g++','-shared','-fPIC','-O2','-std=c++17','-I'+str(ABI),
                        str(HERE/'common/collect.cpp'),'-o',str(library)],check=True)
        self.lib=ct.CDLL(str(library.resolve()))
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.init.argtypes=[ct.c_char_p,ct.c_uint32]
        self.lib.step.argtypes=[ct.c_double,ct.c_void_p,ptr,ct.c_int32,ptr]
        self.lib.hybrid_patches.argtypes=[ct.c_void_p]
        self.lib.hybrid_window.argtypes=[ct.c_void_p]

    def collect(self, episode):
        config,frames=episode.adapt((320,240))
        raw=json.dumps(config).encode()
        if self.lib.init(raw,len(raw))!=0:raise RuntimeError('Baseline collector initialization failed')
        # Ground truth is used ONLY for labels, never passed to the estimator.
        truth=episode.ground_truth_state()[:,:3]
        times=episode.frame_times
        patch=[];window=[]
        for t,frame,imu in episode.steps(frames):
            frame=np.ascontiguousarray(frame,dtype=np.uint8)
            imu=np.ascontiguousarray(imu,dtype=np.float64)
            output=np.empty(13,np.float64)
            if self.lib.step(float(t),frame.ctypes.data,imu,len(imu),output)!=0:
                raise RuntimeError(f'Baseline collector failed at t={t}')
            records=np.empty((20,138),np.float32)
            count=self.lib.hybrid_patches(records.ctypes.data)
            if count:patch.extend(records[:count].copy())
            record=np.empty(18,np.float64)
            if self.lib.hybrid_window(record.ctypes.data):
                t0,t1=record[12:14]
                p0=np.array([np.interp(t0,times,truth[:,k]) for k in range(3)])
                p1=np.array([np.interp(t1,times,truth[:,k]) for k in range(3)])
                window.append(np.r_[record,p1-p0])
        return (np.asarray(patch,dtype=np.float32).reshape(-1,138),
                np.asarray(window,dtype=np.float64).reshape(-1,21))


def generate_split(args,collector,split,round_index):
    """Cache one atomic trace shard per scene/round; resume completed shards.

    Full RGB is spooled temporarily by VIOBench and discarded after collection.
    Only small patch pairs, statistics, labels and provenance are retained.
    """
    from viobench.tools.datasets.generate.simulation import SimConfig,simulate
    from hyperlux.datasets.procthor import Train200
    scene_ids=args.validation_scenes if split=='validation' else args.train_scenes
    sim=SimConfig(duration_seconds=args.duration,instances=args.instances,
                  frame_rate_hz=20.,width=640,height=480,waypoints=48,
                  fov=120.,fov_range=15.,hold_seconds=1.,spool=str(args.out/'spool'))
    references=Train200().references
    helper_path=FULL_HERE.parent/'1_generative_ssm/scene_assets.py'
    spec=importlib.util.spec_from_file_location('hybrid_scene_assets',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    directory=args.out/'traces'/split;directory.mkdir(parents=True,exist_ok=True)
    for scene in scene_ids:
        path=directory/f'round_{round_index:04d}_scene_{scene:03d}.npz'
        meta_path=path.with_suffix('.json')
        if path.exists() and meta_path.exists():
            print(f'Reusing {split} traces: {path.name}',flush=True);continue
        used=sum(p.stat().st_size for p in (args.out/'traces').rglob('*.npz'))
        if used>=args.max_trace_gib*2**30:
            raise RuntimeError('Trace disk limit reached; increase --max-trace-gib and resume')
        # Separate namespaces and non-overlapping seed ranges per scene/round.
        seed=args.seed+(10_000_000 if split=='validation' else 0)+round_index*100_000+scene*100
        scene_name=f'ProcTHOR-Train-{scene}'
        patches=[];windows=[];pgroups=[];wgroups=[];metadata=[]
        scratch=tempfile.TemporaryDirectory(prefix='hybrid-scene-',dir=args.out)
        scene_path,removed=helper.prepare_scene(references[scene],scratch.name)
        if removed:print(f'Scene {scene}: removed {removed} empty geometry primitives',flush=True)
        generator=simulate(sim,{scene_name:scene_path},episodes=args.episodes_per_scene,
                           seed=seed,name=f'hybrid_{split}_{round_index:04d}')
        try:
            for group,ep in enumerate(generator):
                p,w=collector.collect(ep)
                patches.append(p);windows.append(w)
                pgroups.append(np.full(len(p),group,np.int32));wgroups.append(np.full(len(w),group,np.int32))
                metadata.append(dict(name=ep.name,frames=ep.num_frames,duration_s=ep.duration,
                                     patches=len(p),windows=len(w),meta=ep.meta,
                                     horizontal_fov_degrees=ep.config['camera'].get('horizontal_fov_degrees')))
                print(f'{split} round {round_index} scene {scene} episode {group}: '
                      f'{len(p)} patches, {len(w)} windows',flush=True)
                del ep
        finally:
            generator.close();scratch.cleanup()
        if len(metadata)!=args.episodes_per_scene:raise RuntimeError('Incomplete simulation batch')
        temporary=path.with_suffix('.npz.tmp')
        with temporary.open('wb') as f:
            np.savez_compressed(f,patch=np.concatenate(patches),window=np.concatenate(windows),
                                patch_group=np.concatenate(pgroups),window_group=np.concatenate(wgroups),
                                scene=np.int32(scene))
        temporary.replace(path)
        atomic_json(meta_path,dict(split=split,round=round_index,scene_id=scene,seed=seed,
                                   sim=asdict(sim),episodes=metadata,sha256=sha(path)))


def _stop_collectors(active):
    """Stop only the scene subprocess groups owned by this scheduler."""
    for process,log,_,_ in active.values():
        if process.poll() is None:
            try:os.killpg(process.pid,signal.SIGINT)
            except ProcessLookupError:pass
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and any(p.poll() is None for p,_,_,_ in active.values()):
        time.sleep(.1)
    for process,log,_,_ in active.values():
        if process.poll() is None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        process.wait();log.close()


def generate_parallel(args):
    """Bounded independent collectors, using subprocess isolation as streaming.py does.

    Keep the requested two flights per scene; concurrency changes scheduling,
    rather than discarding extra flights from a larger simulator batch.
    """
    tasks=[]
    for split,scenes in [('validation',args.validation_scenes),('train',args.train_scenes)]:
        for scene in scenes:
            path=args.out/'traces'/split/f'round_0000_scene_{scene:03d}.npz'
            if not path.exists() or not path.with_suffix('.json').exists():tasks.append((split,scene))
    if not tasks:return
    jobs=args.out/'generation_jobs';jobs.mkdir(exist_ok=True)
    logs=args.out/'generation_logs';logs.mkdir(exist_ok=True)
    active={};timings=[];started=time.monotonic();warmed=False
    # Let one collector finish first to populate the common HyperDrone JIT cache
    # before multiple processes request the same configuration.
    try:
        while tasks or active:
            limit=args.generation_workers if warmed else 1
            while tasks and len(active)<limit:
                split,scene=tasks.pop(0);key=f'{split}_{scene:03d}'
                payload=dict(vars(args),out=str(args.out),worker_task=None)
                payload.update(split=split,scene=scene)
                task_path=jobs/(key+'.json');atomic_json(task_path,payload)
                log_path=logs/(key+'.log');log=log_path.open('w')
                env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED=str(getattr(args,'seed',0)))
                process=subprocess.Popen([sys.executable,'-u',str(Path(getattr(args,'worker_entry',__file__)).resolve()),
                    '--worker-task',str(task_path)],stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,env=env)
                active[key]=(process,log,log_path,time.monotonic())
                print(f'Collecting {split} scene {scene}: PID {process.pid}, log {log_path}',flush=True)
            for key,(process,log,log_path,begin) in list(active.items()):
                code=process.poll()
                if code is None:continue
                log.close();del active[key]
                if code:
                    tail='\n'.join(log_path.read_text(errors='replace').splitlines()[-15:])
                    raise RuntimeError(f'Collector {key} failed (exit {code}); completed shards remain cached.\n{tail}')
                elapsed=time.monotonic()-begin;timings.append(dict(scene=key,seconds=elapsed));warmed=True
                print(f'Completed {key} in {elapsed:.1f} s',flush=True)
            if active:time.sleep(.2)
    finally:
        _stop_collectors(active)
        atomic_json(args.out/'generation_timing.json',dict(workers=args.generation_workers,
                    elapsed_s=time.monotonic()-started,completed=timings))


def run_worker(task_path):
    args=argparse.Namespace(**json.loads(task_path.read_text()));args.out=Path(args.out)
    # Generation-only children never train or publish model weights.
    configure_profile(getattr(args,'profile','full'))
    args.train_scenes=[args.scene];args.validation_scenes=[args.scene]
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    directory=args.out/'generation_jobs'/f'worker_{os.getpid()}'
    directory.mkdir(parents=True,exist_ok=True)
    collector=TraceCollector(directory,prepare_sources=False)
    generate_split(args,collector,args.split,0)


def load_split(directory, max_patches,max_windows,seed,round_limit=None,scenes=None):
    """Priority reservoir bounds RAM independently of total generated trace size.

    Window transitions are constructed BEFORE sampling so they never join
    different episodes or jump over discarded windows.
    """
    rng=np.random.default_rng(seed)
    pools={};keys={};counts={}
    def add(name,rows,scene,limit):
        if not len(rows):return
        values=np.column_stack([np.full(len(rows),scene),rows])
        priority=rng.random(len(rows))
        if name in pools:
            values=np.concatenate([pools[name],values]);priority=np.concatenate([keys[name],priority])
        counts[name]=counts.get(name,0)+len(rows)
        if len(values)>limit:
            keep=np.argpartition(priority,limit-1)[:limit]
            values=values[keep];priority=priority[keep]
        pools[name]=values;keys[name]=priority
    paths=sorted(directory.glob('*.npz'))
    if round_limit is not None:
        paths=[p for p in paths if int(p.stem.split('_')[1])<=round_limit]
    for path in paths:
        with np.load(path) as d:
            scene=int(d['scene'])
            if scenes is not None and scene not in scenes:continue
            p=d['patch'];w=d['window'];groups=d['window_group']
            add('patch',p,scene,max_patches);add('window',w,scene,max_windows)
            controls=[]
            for group in np.unique(groups):
                indices=np.flatnonzero(groups==group)
                for previous,current in zip(indices[:-1],indices[1:]):
                    r=w[current,:8]
                    action=0 if r[2]>.85 and r[1]<.25 and r[6]<.2 and r[0]<1 else 1
                    if r[2]<.65 or r[1]>.6 or r[6]>.6:action=2
                    controls.append(np.r_[w[previous,:8],action])
            add('control',np.asarray(controls).reshape(-1,9),scene,max_windows)
    if any(name not in pools for name in ['patch','window','control']):
        raise RuntimeError(f'Not enough usable traces in {directory}; generate more episodes')
    return pools,counts


def make_tasks(train,val):
    """Build objectives while keeping fixed validation samples separate."""
    p=np.concatenate([train['patch'],val['patch']]);ps=p[:,0].astype(int);p=p[:,1:]
    w=np.concatenate([train['window'],val['window']]);ws=w[:,0].astype(int);w=w[:,1:]
    c=np.concatenate([train['control'],val['control']]);cs=c[:,0].astype(int);c=c[:,1:]
    valid_patch=p[:,6]>.5
    direction=w[:,8:11];distance=w[:,11];truth=w[:,18:21]
    length=np.linalg.norm(truth,axis=1);along=(truth*direction).sum(1)
    perpendicular=np.linalg.norm(truth-direction*along[:,None],axis=1)
    angle=np.arctan2(perpendicular,np.maximum(along,1e-6))
    error2=np.column_stack([(angle/np.maximum(w[:,14],.02))**2,
                            ((along-distance)/np.maximum(w[:,15],.01))**2]).clip(0,100)
    consistent=(along>.005)&(distance>.005)&(length>.01)&(along/np.maximum(length,1e-9)>.7)
    scale=np.log(np.maximum(along[consistent],1e-6)/distance[consistent]).clip(-np.log(2),np.log(2))[:,None]
    return dict(patch_quality=(p[:,:6],p[:,6:7],ps,'binary'),
                patch_tracker=(p[valid_patch,10:],p[valid_patch,7:9],ps[valid_patch],'regression'),
                measurement_confidence=(w[:,:8],error2,ws,'variance'),
                compute_controller=(c[:,:8],c[:,8].astype(int),cs,'class'),
                translation_scale=(w[consistent,:8],scale,ws[consistent],'regression'))


TASK_FACTORY=make_tasks


def configure_profile(profile):
    global HERE,generate,fit,CURRENT_PROFILE,TASK_FACTORY
    if profile==CURRENT_PROFILE:return
    if profile=='lite':
        from lite_profile import HERE as lite_home,generate as lite_generate,fit_lite,make_tasks_lite
        HERE=lite_home;generate=lite_generate;fit=fit_lite;TASK_FACTORY=make_tasks_lite
    else:
        HERE=FULL_HERE;generate=FULL_GENERATE;fit=FULL_FIT;TASK_FACTORY=make_tasks
    CURRENT_PROFILE=profile


def publish(source,destination):
    for name in NAMES:
        for file in ['weights.h','checkpoint.pt','training.json']:
            target=destination/name/file;temp=target.with_suffix(target.suffix+'.tmp')
            shutil.copyfile(source/name/file,temp);temp.replace(target)


def parser():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--profile',choices=['full','lite'],default='full')
    p.add_argument('--out',type=Path,help='Run directory; lite defaults to local scratch')
    p.add_argument('--train-scenes',nargs='+',type=int,default=list(range(2,18)))
    p.add_argument('--validation-scenes',nargs='+',type=int,default=[180,181,182,183])
    p.add_argument('--episodes-per-scene',type=int,default=2)
    p.add_argument('--duration',type=float,default=60.)
    p.add_argument('--generation-workers',type=int,default=1,help='Independent scene collectors on the GPU; try 2, preserving cached data')
    p.add_argument('--worker-task',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--instances',type=int,default=2,help='Parallel GPU flights; reduce to 1 if GPU memory is tight')
    p.add_argument('--epochs',type=int,default=1000,help='Maximum epochs per model on the fixed dataset')
    p.add_argument('--epoch-patience',type=int,default=40)
    p.add_argument('--min-delta',type=float,default=1e-4,help='Absolute validation-loss improvement threshold')
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--max-train-patches',type=int,default=100_000)
    p.add_argument('--max-train-windows',type=int,default=50_000)
    p.add_argument('--max-trace-gib',type=float,default=8.,help='Persistent trace disk limit; temporary frames/assets need additional space')
    p.add_argument('--seed',type=int,default=1162026)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--reuse-traces',action='store_true',help='Train from cached traces only; never start GPU generation')
    p.add_argument('--no-publish',action='store_true',help='Keep candidates under --out/models without replacing active models')
    p.add_argument('--plan',action='store_true',help='Print configuration without generating or training')
    return p


def main():
    p=parser();args=p.parse_args()
    if args.worker_task:
        run_worker(args.worker_task);return
    configure_profile(args.profile)
    default_out=Path(tempfile.gettempdir())/f'viobench-{os.getuid()}'/'lite_training' if args.profile=='lite' else HERE/'fresh_training'
    args.out=(args.out or default_out).resolve()
    check_scenes(args.train_scenes,args.validation_scenes)
    for key in ['episodes_per_scene','instances','generation_workers','epochs','epoch_patience',
                'max_train_patches','max_train_windows','cpu_threads']:
        if getattr(args,key)<1:p.error(f'{key} must be positive')
    if args.duration<1 or args.min_delta<=0 or args.learning_rate<=0 or args.max_trace_gib<=0:
        p.error('duration, min-delta, learning-rate and max-trace-gib must be positive (duration >= 1)')
    config=dict(train_scenes=args.train_scenes,validation_scenes=args.validation_scenes,
                episodes_per_scene=args.episodes_per_scene,duration=args.duration,instances=args.instances,
                seed=args.seed,fov=120.,fov_range=15.,waypoints=48,hold_seconds=1.,
                base_sha256=sha(HERE/'base/solution.cpp'),
                collector_hooks_sha256=sha(FULL_HERE/('lite_profile.py' if args.profile=='lite' else 'prepare.py')),trace_schema=1)
    if args.profile=='lite':config.update(profile='lite',trace_schema=2)
    if args.plan:
        print(json.dumps(dict(generation=config,training=vars(args)),indent=2,default=str));return
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.cpu_threads)
    config_path=args.out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Generation settings differ from existing run. Use a new --out directory.')
    atomic_json(config_path,config)
    # Preserve a reproducible starting point, including original weights/metadata.
    initial=args.out/'initial_models';candidate=args.out/'models'
    if not initial.exists():shutil.copytree(HERE/'models',initial)
    if not candidate.exists():shutil.copytree(initial,candidate)
    state_path=args.out/'fixed_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(
        strategy='fixed_dataset',models={},stop_reason='running')
    generate()
    # Only dataset index zero is used, even if an older multi-round run exists.
    expected={split:[args.out/'traces'/split/f'round_0000_scene_{scene:03d}.npz' for scene in scenes]
              for split,scenes in [('train',args.train_scenes),('validation',args.validation_scenes)]}
    missing=[path for paths in expected.values() for path in paths
             if not path.exists() or not path.with_suffix('.json').exists()]
    if args.reuse_traces and missing:
        raise ValueError(f'Incomplete cached dataset: {missing[0]}. Run without --reuse-traces to finish generation.')
    if missing:
        if args.generation_workers>1:
            generate_parallel(args)
        else:
            collector=TraceCollector(args.out)
            generate_split(args,collector,'validation',0)
            generate_split(args,collector,'train',0)
    else:print('Reusing the complete fixed dataset; no GPU generation needed.',flush=True)
    manifest=[]
    for paths in expected.values():
        for path in paths:
            meta=json.loads(path.with_suffix('.json').read_text())
            if sha(path)!=meta['sha256']:raise ValueError(f'Trace checksum mismatch: {path}')
            manifest.append(meta)
    sampling=dict(max_train_patches=args.max_train_patches,max_train_windows=args.max_train_windows,seed=args.seed)
    dataset_hash=hashlib.sha256(json.dumps(dict(
        shards=[(m['split'],m['scene_id'],m['sha256']) for m in manifest],sampling=sampling)).encode()).hexdigest()
    if state.get('dataset_sha256',dataset_hash)!=dataset_hash:
        raise ValueError('Fixed dataset changed after training began; use a new --out directory')
    state['dataset_sha256']=dataset_hash
    atomic_json(args.out/'dataset.json',dict(strategy='fixed_dataset',sha256=dataset_hash,
        generation=config,shards=manifest,sampling=sampling))
    validation,vc=load_split(args.out/'traces/validation',100_000,50_000,args.seed,round_limit=0,scenes=args.validation_scenes)
    training,tc=load_split(args.out/'traces/train',args.max_train_patches,args.max_train_windows,
                           args.seed,round_limit=0,scenes=args.train_scenes)
    tasks=TASK_FACTORY(training,validation)
    state.update(train_trace_counts=tc,validation_trace_counts=vc)
    print('Fixed dataset ready. Training each model independently until validation plateaus.',flush=True)
    criterion=dict(patience=args.epoch_patience,min_delta=args.min_delta)
    same_criterion=state.get('criterion')==criterion
    for name in NAMES:
        previous=state['models'].get(name)
        if previous and previous['stop_reason']=='validation_plateau' and same_criterion:
            print(f'{name}: already reached validation plateau; retaining its best checkpoint.',flush=True)
            continue
        x,y,scene,task=tasks[name]
        meta=fit(name,x,y,scene,task,args.epochs,args.seed,
                 validation_scenes=args.validation_scenes,patience=args.epoch_patience,
                 min_delta=args.min_delta,learning_rate=args.learning_rate,
                 model_root=candidate,warm_start=True,augment=True)
        meta.update(base_sha256=config['base_sha256'],public_episodes_used=False,profile=args.profile,
                    data=manifest,generation=config,training_strategy='fixed_dataset',
                    training_run=str(args.out),training_data_sha256=dataset_hash)
        atomic_json(candidate/name/'training.json',meta)
        state['models'][name]={key:meta[key] for key in
            ['epochs','best_epoch','stop_reason','initial_validation_loss','validation_loss',
             'train_samples','validation_samples']}
        state['stop_reason']='running';atomic_json(state_path,state)
    # Every candidate is complete before active model weights are replaced.
    if not args.no_publish:publish(candidate,HERE/'models')
    state['criterion']=criterion
    state['stop_reason']='validation_plateau' if all(
        m['stop_reason']=='validation_plateau' for m in state['models'].values()) else 'max_epochs'
    atomic_json(state_path,state)
    print(f"Stopped: {state['stop_reason']}. Fixed dataset generated once; no fresh-data rounds.",flush=True)
    if state['stop_reason']=='max_epochs':
        print('Some models reached the epoch cap. Increase --epochs and rerun to continue those models.')
    print('Validation plateaus concern component losses; they do not prove a global optimum or lower trajectory ATE.')
    print(f'Weights: {candidate if args.no_publish else HERE/"models"}; curves and checkpoints: {args.out}')

if __name__=='__main__':main()
