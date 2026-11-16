"""Compare baseline + five hybrids on identical, full 60-second public episodes.

Defaults to all 10 public archives. --smoke-frames is explicitly a short check,
not a 60-second performance result. All inference, including networks, is metered.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
import tempfile
from pathlib import Path
import subprocess

import numpy as np
from viobench.episodes import public
from viobench.tools.datasets.episode import FrameSequence
from viobench.tools.datasets.sources import episodes
from viobench.tools.runner.engine import Engine
from viobench.tools.runner.metrics import fill_missing_states
from viobench.tools.runner.run import EpisodeReport, run_episode, summarize
from viobench.tools.runner.sandbox import Limits
from viobench.tools.runner.score import write_estimates
from prepare import HERE, NAMES, generate

BASELINE='2_deterministic'
MODELS=[BASELINE]+NAMES
FULL_HERE=HERE
FULL_GENERATE=generate
BASELINE_FILE=FULL_HERE.parent/'2_deterministic/solution.cpp'
CURRENT_PROFILE='full'

def configure_profile(profile):
    global HERE,BASELINE,BASELINE_FILE,generate,CURRENT_PROFILE,MODELS
    if profile=='lite':
        from lite_profile import HERE as lite_home,generate as lite_generate
        HERE=lite_home;generate=lite_generate;BASELINE='solution_lite'
        BASELINE_FILE=FULL_HERE.parent/'2_deterministic/solution_lite.cpp'
    else:
        HERE=FULL_HERE;generate=FULL_GENERATE;BASELINE='2_deterministic'
        BASELINE_FILE=FULL_HERE.parent/'2_deterministic/solution.cpp'
    CURRENT_PROFILE=profile;MODELS=[BASELINE]+NAMES

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def build(names, directory):
    generate();directory.mkdir(parents=True,exist_ok=True)
    workspace=FULL_HERE.parents[2]
    original=BASELINE_FILE
    if digest(original)!=digest(HERE/'base/solution.cpp'):
        raise RuntimeError('Baseline has changed since the hybrid snapshot; reconcile base/solution.cpp before comparing')
    binaries={}
    for name in names:
        source=original if name==BASELINE else HERE/'models'/name/'solution.cpp'
        if name!=BASELINE:
            weights=source.with_name('weights.h');meta=source.with_name('training.json')
            if not weights.exists() or not meta.exists():raise RuntimeError(f'Train {name} before benchmarking')
            info=json.loads(meta.read_text())
            if info['base_sha256']!=digest(original) or info['public_episodes_used']:
                raise RuntimeError(f'Invalid training provenance for {name}')
        path=directory/(name+'.wasm')
        subprocess.run([str(workspace/'build/toolchain/wasi-sdk/bin/clang++'),'-O2','-msimd128',
                        '-fno-exceptions','-I'+str(workspace/'runtime/native'),str(source),
                        '-mexec-model=reactor','-Wl,--initial-memory=67108864','-o',str(path)],check=True)
        binaries[name]=path
        print(f'Built {name}: {path.stat().st_size:,} bytes',flush=True)
    return binaries

def prefix(ep,count):
    if count<3 or count>ep.num_frames:raise ValueError('Invalid smoke frame count')
    config=deepcopy(ep.config);config['episode']['num_frames']=count
    return replace(ep,config=config,frame_times=ep.frame_times[:count],
                   frames=FrameSequence(lambda i:ep.frames[i],count,ep.frames.shape[1:],ep.frames.dtype))

def write_comparison(results, path):
    lines=['| Model | Completed | Mean unaligned error (m) | Mean unaligned RMSE (m) | Mean Sim(3) ATE (m) | Max fuel/frame | Mean fuel/frame |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,result in results.items():
        s=result['summary'];means=result['means']
        label=result.get('display_name',name)
        lines.append(f"| {label} | {s['completed']}/{result.get('expected_episodes') or s['episodes']} | {means['unaligned_mean_error_m']:.5f} | {means['unaligned_ate_rmse_m']:.5f} | {s['sim3_ate_rmse_mean']:.5f} | {s['fuel_per_frame_max']:,} | {s['fuel_per_frame_mean']:,.1f} |")
    path.write_text('\n'.join(lines)+'\n');return '\n'.join(lines)

def configure_cache(cache_dir=None):
    """Write new downloads to local scratch; search existing caches read-only."""
    previous=os.environ.get('CONTA_CACHE') or str(Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'conta')
    target=Path(cache_dir) if cache_dir else Path(tempfile.gettempdir())/f'viobench-{os.getuid()}'/'conta'
    target.mkdir(parents=True,exist_ok=True)
    search=[previous,*os.environ.get('CONTA_SEARCH_PATHS','').split(os.pathsep)]
    os.environ['CONTA_CACHE']=str(target)
    os.environ['CONTA_SEARCH_PATHS']=os.pathsep.join(dict.fromkeys(p for p in search if p and Path(p)!=target))
    print(f'Download cache: {target}; reusing existing caches: {os.environ["CONTA_SEARCH_PATHS"]}',flush=True)


def validate_resume(previous,current):
    if previous.get('profile','full')!=current.get('profile','full'):raise ValueError('Cannot resume: estimator profile changed')
    for key in ['sources','smoke_frames']:
        if previous.get(key)!=current.get(key):raise ValueError(f'Cannot resume: {key} changed')
    before={name:entry['sha256'] for name,entry in previous['binaries'].items()}
    after={name:entry['sha256'] for name,entry in current['binaries'].items()}
    if before!=after:raise ValueError('Cannot resume: model binaries changed; use a new output directory')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--profile',choices=['full','lite'],default='full')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--cache-dir',type=Path,help='Download cache; default local /tmp scratch, reusing existing caches')
    parser.add_argument('--resume',action='store_true',help='Reuse saved episode reports after verifying model hashes')
    parser.add_argument('--models',nargs='+',choices=['2_deterministic','solution_lite']+NAMES)
    parser.add_argument('--episodes',nargs='+',help='Optional local archive paths/hashes; default all 10 public episodes')
    parser.add_argument('--build-only',action='store_true')
    parser.add_argument('--smoke-frames',type=int,help='Short verification only; never a full benchmark score')
    args=parser.parse_args()
    configure_profile(args.profile)
    args.models=args.models or MODELS
    if any(name not in MODELS for name in args.models):parser.error('Selected baseline does not match --profile')
    if len(set(args.models))!=len(args.models):parser.error('Duplicate model names')
    default_out=(Path(tempfile.gettempdir())/f'viobench-{os.getuid()}'/('lite_smoke' if args.smoke_frames else 'lite_benchmark_60s')) if args.profile=='lite' else HERE/('benchmark_smoke' if args.smoke_frames else 'benchmark_60s')
    out=args.out or default_out
    out.mkdir(parents=True,exist_ok=True)
    binaries=build(args.models,out/'build')
    if args.build_only:return
    sources=args.episodes or public
    if not args.episodes and len(public)!=10:raise RuntimeError('Expected 10 public episodes')
    manifest=dict(profile=args.profile,sources=sources,smoke_frames=args.smoke_frames,
        duration='full public 60-second episodes' if not args.smoke_frames else 'SMOKE TEST',
        base_sha256=digest(HERE/'base/solution.cpp'),
        binaries={name:dict(path=str(p),sha256=digest(p)) for name,p in binaries.items()},
        training={name:json.loads((HERE/'models'/name/'training.json').read_text()) for name in args.models if name!=BASELINE})
    if args.resume:
        if not (out/'manifest.json').exists():raise ValueError('No saved manifest to resume')
        validate_resume(json.loads((out/'manifest.json').read_text()),manifest)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    configure_cache(args.cache_dir)
    engine=Engine();limits=Limits()
    reports={name:[] for name in args.models};fuels={name:[] for name in args.models};rows={name:[] for name in args.models}
    if args.resume:
        for name in args.models:
            result_path=out/name/'results.json'
            if result_path.exists() and json.loads(result_path.read_text())['limits']!=asdict(limits):
                raise ValueError('Cannot resume: sandbox limits changed')
    seen=set()
    for episode_index,ep in enumerate(episodes(sources),1):
        if ep.key in seen:raise ValueError('Duplicate episode')
        seen.add(ep.key)
        if args.smoke_frames:ep=prefix(ep,args.smoke_frames)
        elif not args.episodes:
            rate=float(ep.config['episode']['frame_rate_hz'])
            if ep.num_frames!=round(60*rate):raise RuntimeError(f'{ep.key} is not a full 60-second episode')
        truth=ep.ground_truth_state()
        for name in args.models:
            log=out/name/'logs'/ep.key;log.mkdir(parents=True,exist_ok=True)
            print(f'episode {episode_index}: {name}, {ep.num_frames} frames',flush=True)
            cached=args.resume and all((log/file).exists() for file in ['report.json','trajectory.npz','fuel.npy'])
            if cached:
                report=EpisodeReport(**json.loads((log/'report.json').read_text()))
                with np.load(log/'trajectory.npz') as saved:
                    estimate=saved['estimates']
                    np.testing.assert_allclose(saved['times'],ep.frame_times,rtol=0,atol=1e-9)
                fuel=np.load(log/'fuel.npy',allow_pickle=False)
                if report.hash!=ep.hash or report.frames!=ep.num_frames or estimate.shape!=truth.shape:
                    raise ValueError(f'Cached episode mismatch: {log}')
                print(f'  Reused completed evaluation for {name}',flush=True)
            else:
                report,estimate,fuel=run_episode(binaries[name],ep,engine,limits,log)
            reports[name].append(report);fuels[name].append(fuel)
            filled=fill_missing_states(estimate)
            row=dict(key=ep.key,hash=ep.hash,frames=ep.num_frames,duration_s=float(ep.duration),
                     unaligned_mean_error_m=float(np.linalg.norm(filled[:,:3]-truth[:,:3],axis=1).mean()),
                     unaligned_ate_rmse_m=report.metrics['ate_rmse'],sim3_ate_rmse_m=report.metrics['sim3_ate_rmse'],
                     completed=report.completed,failure=report.failure,
                     fuel_per_frame_max=report.fuel_per_frame_max,fuel_per_frame_mean=report.fuel_per_frame_mean,
                     fuel_init=report.fuel_init,fuel_total=report.fuel_total)
            rows[name].append(row)
            if not cached:
                write_estimates(log,ep.frame_times,estimate)
                np.save(log/'fuel.npy',fuel,allow_pickle=False)
                np.savez(log/'trajectory.npz',times=ep.frame_times,truth=truth,estimates=estimate)
                (log/'report.json').write_text(json.dumps(report.as_dict(),indent=2))
            print(json.dumps(row),flush=True)
        # Save complete comparisons after each episode so partial progress is inspectable.
        results={}
        for name in args.models:
            result=dict(model=name,profile=args.profile,display_name=("lite_"+name if args.profile=="lite" and name!=BASELINE else name),smoke_test=bool(args.smoke_frames),limits=asdict(limits),
                        expected_episodes=10 if not args.episodes else None,
                        full_public_set=(not args.episodes and not args.smoke_frames and len(reports[name])==10),
                        summary=summarize(reports[name],fuels[name]),episodes=rows[name],
                        means={key:float(np.mean([r[key] for r in rows[name]])) for key in
                               ('unaligned_mean_error_m','unaligned_ate_rmse_m','sim3_ate_rmse_m')})
            result['summary']['fuel_init_total']=sum(r.fuel_init for r in reports[name])
            results[name]=result
            (out/name/'results.json').write_text(json.dumps(result,indent=2))
        (out/'comparison.json').write_text(json.dumps(results,indent=2))
        write_comparison(results,out/'comparison.md')
    if not seen:raise RuntimeError('No episodes evaluated')
    if not args.episodes and len(seen)!=10:raise RuntimeError('Incomplete public episode set')
    print(('SMOKE TEST ONLY\n' if args.smoke_frames else '')+write_comparison(results,out/'comparison.md'))

if __name__=='__main__':main()
