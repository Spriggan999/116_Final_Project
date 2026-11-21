"""Black-box policy search (the "RL" approach) and host evaluation for the learned noise multipliers.

The estimator is v9_adapted with a tiny network that, once per 6-frame cycle, maps 9 window features to 7 log
multipliers (rotation / direction / speed / velocity-prior / small-motion row noise, gyro-bias and accelerometer
process noise). The reward is minus the mean unaligned position RMSE over whole training flights, replayed on the
host (bin/host_rt reads the weights from env LEARN_WFILE). CMA-ES needs no gradients and sees the delayed effect of
every multiplier on the trajectory.

    python search.py static  --train FEEDS... --out static.txt            # headroom: 7 constant multipliers
    python search.py policy  --train FEEDS... --init static.txt --out rl.txt   # full network (109 weights)
    python search.py eval    --weights rl.txt --sets DIR...                # per-set mean, paired vs no multipliers
    python search.py export  --weights rl.txt --header ../policy_rl.h      # weights -> C header for the wasm build
    python search.py policy --lin --harm 1 --train ... --val ... --init static.txt --out lin.txt
        # speed-aware linear model (21 numbers: per output a constant, a speed slope and a parallax slope),
        # objective = mean ATE + harm * mean(how much each flight got worse than v9_adapted), early stopping on --val
"""
import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

LF, LH, LO = 9, 6, 7
LW_N = LH * LF + LH + LO * LH + LO
LIN_N = 3 * LO                       # LEARN_LIN: [constant, speed slope, parallax slope] per output
B2 = LH * LF + LH + LO * LH          # offset of the output biases
CLASSES = ['rot', 'dir', 'spd', 'pri', 'small', 'q_bg', 'q_acc']
BIN = Path(os.environ.get('LEARN_BIN', f'/tmp/learned-{os.getuid()}/bin'))
POOL = ThreadPoolExecutor(int(os.environ.get('LEARN_WORKERS', '16')))


def feeds(paths):
    out = []
    for p in paths:
        p = Path(p)
        out += sorted(d for d in p.iterdir() if (d / 'frames.g8').exists()) if not (p / 'frames.g8').exists() else [p]
    return out


def run_one(binary, feed, wfile):
    env = dict(os.environ)
    if wfile:
        env['LEARN_WFILE'] = wfile
    r = subprocess.run([str(binary), str(feed)], capture_output=True, text=True, env=env)
    try:
        return float(r.stdout.split()[1])
    except (IndexError, ValueError):
        return float('nan')


def rmse_all(w, flights, binary=None):
    """Per-flight RMSE with weights w (None = the unmodified estimator)."""
    if w is None:
        return np.array(list(POOL.map(lambda f: run_one(BIN / 'host_base', f, None), flights)))
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, dir=f'/tmp/learned-{os.getuid()}') as fh:
        fh.write(' '.join(f'{v:.10g}' for v in w))
        name = fh.name
    try:
        binary = binary or BIN / ('host_lin' if len(w) == LIN_N else 'host_rt')
        return np.array(list(POOL.map(lambda f: run_one(binary, f, name), flights)))
    finally:
        os.remove(name)


def cmaes(f, x0, sigma0, iters, popsize, log, seed=0, on_iter=None):
    """(mu/mu_w, lambda)-CMA-ES (Hansen's tutorial defaults). f maps a vector to a scalar to minimize."""
    rng = np.random.default_rng(seed)
    n = len(x0)
    lam = popsize
    mu = lam // 2
    wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    wts /= wts.sum()
    mueff = 1 / (wts ** 2).sum()
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chin = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    m, sigma = np.array(x0, float), sigma0
    pc, ps, C = np.zeros(n), np.zeros(n), np.eye(n)
    best_x, best_f = m.copy(), f(m)
    log({'iter': 0, 'best': best_f, 'mean_f': best_f, 'sigma': sigma})
    for it in range(1, iters + 1):
        Dsq, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(Dsq, 1e-20))
        z = rng.standard_normal((lam, n))
        y = z @ (B * D).T
        X = m + sigma * y
        F = np.array([f(x) for x in X])
        order = np.argsort(F)
        if F[order[0]] < best_f:
            best_f, best_x = F[order[0]], X[order[0]].copy()
        ysel = y[order[:mu]]
        yw = wts @ ysel
        m = m + sigma * yw
        Cinvsqrt = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinvsqrt @ yw)
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * it)) / chin < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * yw
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) \
            + cmu * (ysel.T * wts) @ ysel
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chin - 1))
        rec = {'iter': it, 'best': best_f, 'gen_best': float(F[order[0]]), 'mean_f': float(F.mean()), 'sigma': sigma}
        if on_iter:
            rec.update(on_iter(it, m) or {})
        log(rec)
    return best_x, best_f


def load_w(path):
    return np.array([float(v) for v in Path(path).read_text().split()])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['static', 'policy', 'eval', 'export'])
    ap.add_argument('--train', nargs='+', default=[])
    ap.add_argument('--sets', nargs='+', default=[])
    ap.add_argument('--init')
    ap.add_argument('--weights')
    ap.add_argument('--out')
    ap.add_argument('--header')
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--popsize', type=int, default=16)
    ap.add_argument('--sigma', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lin', action='store_true', help='speed-aware linear model (21 numbers) instead of the network')
    ap.add_argument('--harm', type=float, default=0.0, help='objective weight on per-flight harm vs v9_adapted')
    ap.add_argument('--val', nargs='+', default=[], help='validation flights: keep the CMA mean that scores best here')
    ap.add_argument('--val-every', type=int, default=5)
    args = ap.parse_args()

    if args.mode in ('static', 'policy'):
        flights = feeds(args.train)
        base = rmse_all(None, flights)
        print(f'{len(flights)} training flights, unmodified mean RMSE {base.mean():.4f}', flush=True)
        x0 = load_w(args.init) if args.init else np.zeros(LW_N)
        if args.lin:                # constants from a network file's output biases (e.g. the static optimum), slopes 0
            x0 = x0 if len(x0) == LIN_N else np.concatenate([[b, 0.0, 0.0] for b in (x0[B2:] if len(x0) == LW_N else np.zeros(LO))])
        if args.mode == 'static' and not args.lin:   # only the 7 output biases: constant multipliers
            def full(b):
                w = x0.copy(); w[B2:] = b; return w
            start, n = x0[B2:], LO
        else:                       # every weight, starting from the given network (e.g. the static optimum)
            def full(v):
                return v
            start, n = x0, len(x0)
        hist = []

        def objective(r, b):
            return float(np.mean(r) + args.harm * np.mean(np.maximum(0.0, r - b)))

        def f(v):
            return objective(rmse_all(full(v), flights), base)
        val = feeds(args.val) if args.val else []
        vbase = rmse_all(None, val) if val else None
        vbest = {'f': objective(rmse_all(full(start), val), vbase) if val else None, 'w': full(start).copy(), 'it': 0}
        if val:
            print(f'{len(val)} validation flights, unmodified mean {vbase.mean():.4f}, start objective {vbest["f"]:.4f}', flush=True)

        def on_iter(it, m):
            if not val or it % args.val_every:
                return None
            vf = objective(rmse_all(full(m), val), vbase)
            if vf < vbest['f']:
                vbest.update(f=vf, w=full(m).copy(), it=it)
            return {'val': vf, 'val_best': vbest['f'], 'val_best_it': vbest['it']}

        def log(rec):
            hist.append(rec)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        best, best_f = cmaes(f, start, args.sigma, args.iters, args.popsize, log, args.seed, on_iter)
        w = vbest['w'] if val else full(best)   # with validation flights: the best CMA mean on them (early stopping)
        if val:
            Path(args.out).with_suffix('.train_best.txt').write_text(' '.join(f'{v:.10g}' for v in full(best)))
        Path(args.out).write_text(' '.join(f'{v:.10g}' for v in w))
        Path(args.out).with_suffix('.json').write_text(json.dumps(dict(
            mode=args.mode, lin=args.lin, harm=args.harm, train=[str(p) for p in args.train], flights=len(flights),
            base=float(base.mean()), best=best_f, val_flights=len(val), val_best=vbest['f'], val_best_iter=vbest['it'],
            multipliers=dict(zip(CLASSES, np.exp(np.clip(w[0::3] if args.lin else w[B2:], -2, 2)).round(3).tolist())),
            history=hist), indent=1))
        print(f'done: train objective {objective(base, base):.4f} -> {best_f:.4f}; kept iteration {vbest["it"] if val else "final"};',
              'constant part of the multipliers:', dict(zip(CLASSES, np.exp(np.clip(w[0::3] if args.lin else w[B2:], -2, 2)).round(3).tolist())))
    elif args.mode == 'eval':
        w = load_w(args.weights) if args.weights else None
        for s in args.sets:
            flights = feeds([s])
            a = rmse_all(None, flights)
            b = rmse_all(w, flights) if w is not None else a
            d = b - a
            print(f'{Path(s).name:18s} n={len(flights):3d}  base {a.mean():.4f}  learned {b.mean():.4f}  '
                  f'diff {d.mean():+.4f} (se {d.std(ddof=1) / np.sqrt(len(d)):.4f})  better on {(d < 0).sum()}/{len(d)}')
    else:
        w = load_w(args.weights)
        assert len(w) in (LW_N, LIN_N)
        rows = ',\n    '.join(', '.join(f'{v:.10g}' for v in w[i:i + 9]) for i in range(0, len(w), 9))
        what = ('linear model: [constant, speed slope, parallax slope] per output (build with LEARN_LIN=1)' if len(w) == LIN_N
                else 'network: W1 [6][9], b1 [6], W2 [7][6], b2 [7]')
        Path(args.header).write_text(f'// Learned noise multipliers ({Path(args.weights).name}), {what}\n'
                                     f'static const double LW[{len(w)}] = {{\n    {rows}}};\n')
        print('wrote', args.header)


if __name__ == '__main__':
    main()
