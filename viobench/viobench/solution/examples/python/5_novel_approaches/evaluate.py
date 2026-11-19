"""Build estimator variants and score them on public and held-out 60-second episodes.

A variant is a C++ source plus compile-time overrides (see VARIANTS in variants.py).
Primary metric: mean unaligned ATE RMSE (the course metric after the presentations).
Sim(3) ATE and fuel are reported alongside. Per-episode trajectories are saved so
analysis/oracle.py can decompose the remaining error.

    python 5_novel_approaches/evaluate.py --variants base imu_scale --sets public heldout
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
DEFAULT_OUT = Path('/var/tmp')/f'novel-approaches-{os.getuid()}'/'evaluation'


def episode_sources(name):
    if name == 'public':
        from viobench.episodes import public
        return list(public)
    folder = Path('/var/tmp')/f'novel-approaches-{os.getuid()}'/'episodes'/name
    files = sorted(str(p) for p in folder.glob('*.zip'))
    if not files:
        raise FileNotFoundError(f'no episodes in {folder}; run data/simulate.py --split {name}')
    return files


def build(name, spec, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'{name}.wasm'
    flags = [f'-D{k}={v}' for k, v in spec.get('defines', {}).items()]
    command = [str(WORKSPACE/'build/toolchain/wasi-sdk/bin/clang++'), '-O2', '-msimd128', '-fno-exceptions',
               '-I'+str(WORKSPACE/'runtime/native'), *flags, str(HERE/spec['source']),
               '-mexec-model=reactor', '-Wl,--initial-memory=67108864', '-o', str(path)]
    subprocess.run(command, check=True)
    return path


def run_one(job):
    """Worker: one variant on one episode, cached on disk."""
    binary, source, log = job
    log = Path(log)
    if (log/'row.json').exists():
        return json.loads((log/'row.json').read_text())
    from viobench.tools.datasets.sources import episodes
    from viobench.tools.runner.engine import Engine
    from viobench.tools.runner.metrics import fill_missing_states
    from viobench.tools.runner.run import run_episode
    from viobench.tools.runner.sandbox import Limits
    ep = next(iter(episodes([source])))
    truth = ep.ground_truth_state()
    log.mkdir(parents=True, exist_ok=True)
    report, estimate, fuel = run_episode(Path(binary), ep, Engine(), Limits(), log)
    np.savez(log/'trajectory.npz', times=ep.frame_times, truth=truth, estimates=estimate)
    filled = fill_missing_states(estimate)
    row = dict(key=ep.key, completed=report.completed, failure=report.failure,
               unaligned_ate_rmse_m=report.metrics['ate_rmse'], sim3_ate_rmse_m=report.metrics['sim3_ate_rmse'],
               unaligned_mean_error_m=float(np.linalg.norm(filled[:, :3]-truth[:, :3], axis=1).mean()),
               fuel_per_frame_max=report.fuel_per_frame_max, fuel_per_frame_mean=report.fuel_per_frame_mean)
    (log/'row.json').write_text(json.dumps(row))
    return row


def main():
    from variants import VARIANTS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument('--sets', nargs='+', default=['public', 'heldout'])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--rebuild', action='store_true', help='discard cached results of the selected variants')
    args = parser.parse_args()
    if args.rebuild:
        import shutil
        for name in args.variants:
            shutil.rmtree(args.out/name, ignore_errors=True)
    # New downloads go to scratch (home quota is small); existing caches are reused read-only.
    previous = os.environ.get('CONTA_CACHE') or str(Path.home()/'.cache'/'conta')
    cache = args.out.parent/'conta'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['CONTA_SEARCH_PATHS'] = os.pathsep.join(
        dict.fromkeys(p for p in [previous, *os.environ.get('CONTA_SEARCH_PATHS', '').split(os.pathsep)] if p))
    os.environ['CONTA_CACHE'] = str(cache)
    binaries = {name: build(name, VARIANTS[name], args.out/name) for name in args.variants}
    jobs, keys = [], []
    for split in args.sets:
        for source in episode_sources(split):
            for name in args.variants:
                tag = source['hash'][:12] if isinstance(source, dict) else Path(source).stem
                jobs.append((str(binaries[name]), source, str(args.out/name/split/tag)))
                keys.append((name, split))
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_one, jobs))
    table = {}
    for (name, split), row in zip(keys, rows):
        table.setdefault(name, {}).setdefault(split, []).append(row)
    summary = {name: {split: dict(episodes=len(r), completed=sum(x['completed'] for x in r),
                                  unaligned_ate_rmse_m=float(np.mean([x['unaligned_ate_rmse_m'] for x in r])),
                                  sim3_ate_rmse_m=float(np.mean([x['sim3_ate_rmse_m'] for x in r])),
                                  fuel_per_frame_max=int(max(x['fuel_per_frame_max'] for x in r)))
                      for split, r in splits.items()} for name, splits in table.items()}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(f"{'variant':<24}" + ''.join(f'{s+" unaligned":>20}{s+" sim3":>14}' for s in args.sets) + f"{'max fuel':>12}")
    for name in args.variants:
        s = summary[name]
        print(f'{name:<24}' + ''.join(f"{s[k]['unaligned_ate_rmse_m']:20.3f}{s[k]['sim3_ate_rmse_m']:14.3f}" for k in args.sets)
              + f"{max(s[k]['fuel_per_frame_max'] for k in args.sets):12,d}")


if __name__ == '__main__':
    main()
