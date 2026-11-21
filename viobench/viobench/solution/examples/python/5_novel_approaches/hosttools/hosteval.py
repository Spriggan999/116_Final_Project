"""Host replay evaluation (no fuel): run host binaries over feed sets and compare paired per flight.

A host binary = solution.cpp + learned/tools/host_run.cpp built with g++ (prints "rmse <unaligned ATE> frames <n>").
Feeds live in /tmp/learned-<uid>/feeds/<set>/<flight> (made by learned/tools/make_feeds.py).

    python hosteval.py BIN [BIN2 ...] --sets eval        # eval = the 112 flights learned_lin never saw
    python hosteval.py BIN --sets tune                   # tune = learned_lin's training sets (our tuning sets)
    python hosteval.py BIN --ref /tmp/learned-<uid>/bin/host_learned_lin  (default reference)
Results are cached per (binary hash, flight) in /tmp/learned-<uid>/hosteval.
"""
import argparse, hashlib, json, os, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(f'/tmp/learned-{os.getuid()}')
FEEDS = ROOT / 'feeds'
CACHE = ROOT / 'hosteval'
GROUPS = {
    'eval': ['agile-hard-dev', 'agile-xfast-dev', 'agile-hard-test', 'agile-xfast-test', 'fresh-hard', 'fresh-xfast', 'heldout'],
    'tune': ['train-hard', 'train-med', 'train-xfast', 't2-hard', 't2-xfast', 't2-xxfast', 't2-hssd'],
    'quick': ['train-hard', 't2-xfast'],
    'normal': ['normal-dev'],
    'dev': ['agile-hard-dev', 'agile-xfast-dev', 'normal-dev'],                       # selection (round 2)
    'test': ['agile-hard-test', 'agile-xfast-test', 'fresh-hard', 'fresh-xfast', 'heldout', 'test2-hard', 'test2-xfast'],   # report once (round 2; test2 = houses 80-95)
}


def sha(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


def run(binary, h, feed, env):
    key = CACHE / h / (feed.parent.name + '__' + feed.name + '.json')
    if key.exists():
        return json.loads(key.read_text())
    r = subprocess.run([str(binary), str(feed)], capture_output=True, text=True, env=env, timeout=600)
    try:
        parts = r.stdout.split()
        row = dict(rmse=float(parts[parts.index('rmse') + 1]), frames=int(parts[parts.index('frames') + 1]))
    except (ValueError, IndexError):
        row = dict(rmse=float('nan'), frames=0, err=(r.stderr or '')[-300:])
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps(row))
    return row


def evaluate(binary, flights, workers):
    h = sha(binary) + os.environ.get('HOSTEVAL_TAG', '')
    env = dict(os.environ)
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda f: run(binary, h, f, env), flights))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('binaries', nargs='+')
    ap.add_argument('--sets', nargs='+', default=['eval'])
    ap.add_argument('--ref', default=str(ROOT / 'bin' / 'host_learned_lin'))
    ap.add_argument('--workers', type=int, default=18)
    ap.add_argument('--per-flight', action='store_true')
    a = ap.parse_args()
    sets = [s for name in a.sets for s in GROUPS.get(name, [name])]
    flights = {s: sorted(d for d in (FEEDS / s).iterdir() if (d / 'frames.g8').exists()) for s in sets}
    allf = [f for s in sets for f in flights[s]]
    ref = np.array([r['rmse'] for r in evaluate(a.ref, allf, a.workers)])
    for b in a.binaries:
        res = evaluate(b, allf, a.workers)
        v = np.array([r['rmse'] for r in res])
        fails = sum(r['frames'] < 1200 for r in res)
        print(f'== {Path(b).name}  (ref {Path(a.ref).name})  incomplete flights: {fails}')
        i = 0
        for s in sets:
            n = len(flights[s]); x, y = v[i:i + n], ref[i:i + n]; d = x - y
            print(f'  {s:18s} n={n:3d}  ref {y.mean():6.3f}  this {x.mean():6.3f}  diff {d.mean():+.3f} (se {d.std(ddof=1)/np.sqrt(n):.3f})  better {int((d < 0).sum())}/{n}  median {np.median(x):.3f}')
            if a.per_flight:
                for f, xx, yy in zip(flights[s], x, y):
                    print(f'      {f.name:34s} {yy:7.3f} {xx:7.3f}')
            i += n
        d = v - ref
        print(f'  {"ALL":18s} n={len(v):3d}  ref {ref.mean():6.3f}  this {v.mean():6.3f}  diff {d.mean():+.3f} (se {d.std(ddof=1)/np.sqrt(len(d)):.3f})  better {int((d < 0).sum())}/{len(d)}  median {np.median(v):.3f}')


if __name__ == '__main__':
    main()
