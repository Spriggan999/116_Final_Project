"""Official-sandbox check (fuel + unaligned ATE) of a wasm on episode zips, in parallel.

    python sandbox_eval.py WASM EPISODE.zip [...] [--workers 8] [--out DIR]
Prints per-episode unaligned ATE, max/mean fuel per frame, and the means.
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np


def run_one(job):
    binary, source, log = job
    from viobench.tools.datasets.sources import episodes
    from viobench.tools.runner.engine import Engine
    from viobench.tools.runner.run import run_episode
    from viobench.tools.runner.sandbox import Limits
    ep = next(iter(episodes([source])))
    Path(log).mkdir(parents=True, exist_ok=True)
    report, estimate, fuel = run_episode(Path(binary), ep, Engine(), Limits(), Path(log))
    np.save(Path(log) / 'fuel.npy', fuel)
    return dict(key=Path(source).stem, completed=report.completed, ate=report.metrics['ate_rmse'],
                fuel_max=report.fuel_per_frame_max, fuel_mean=report.fuel_per_frame_mean,
                p99=float(np.percentile(fuel, 99)), argmax=int(np.argmax(fuel)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wasm'); ap.add_argument('episodes', nargs='+')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default=f'/var/tmp/novel-approaches-{os.getuid()}/sandbox-msckf')
    a = ap.parse_args()
    tag = Path(a.wasm).stem
    jobs = [(a.wasm, e, f'{a.out}/{tag}/{Path(e).stem}') for e in a.episodes]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(run_one, jobs))
    for r in rows:
        print(f"{r['key']:34s} ate {r['ate']:.3f}  fuel max {r['fuel_max']:,} (frame {r['argmax']})  p99 {r['p99']:,.0f}  mean {r['fuel_mean']:,.0f}  {'ok' if r['completed'] else 'FAILED'}")
    print(f"MEAN ate {np.mean([r['ate'] for r in rows]):.3f}  max fuel {max(r['fuel_max'] for r in rows):,}  mean fuel {np.mean([r['fuel_mean'] for r in rows]):,.0f}")
    Path(a.out, tag + '.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
