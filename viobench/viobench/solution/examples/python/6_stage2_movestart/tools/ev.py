"""ev.py SET wasm1 [wasm2 ...]   SET = glob of episode zips (quoted). Prints unaligned ATE (stage-2 metric), Sim3 ATE, max fuel.
Results cached per (wasm hash, episode). Env PROCS (default 2), DETAIL=1 for per-episode rows."""
import sys, glob, os, json, hashlib, time, numpy as np
from multiprocessing import Pool
sys.path.insert(0, __import__('os').environ.get('VIOBENCH_ROOT') or (glob.glob('/sessions/*/mnt/Final_project/viobench') + [''])[0])   # VIOBENCH_ROOT = the folder holding the viobench package
from viobench.tools.datasets import sources
from viobench.tools.runner.sandbox import Sandbox
from viobench.tools.runner.metrics import evaluate_trajectory
C = os.path.expanduser('~/ev/cache'); os.makedirs(C, exist_ok=True)
def one(a):
    wasm, z, key = a
    f = f'{C}/{key}_{os.path.basename(z)}.json'
    if os.path.exists(f): return json.load(open(f))
    ep = next(iter(sources.episodes([z]))); cfg, frames = ep.adapt((320, 240)); gt = ep.ground_truth_state()
    sb = Sandbox(wasm); sb.init(cfg); E = []; F = []; ok = True
    try:
        for i, fr in enumerate(frames):
            r = sb.step(ep.frame_times[i], np.ascontiguousarray(np.array(fr, dtype=np.uint8)), np.array(ep.imu_window(i)))
            E.append(np.r_[r.pose, r.twist]); F.append(r.fuel)
    except Exception as e:
        ok = False
    n = len(E)
    if n < len(gt): E += [np.full(13, np.nan)] * (len(gt) - n)
    m = evaluate_trajectory(np.array(gt), np.array(E), np.array(ep.frame_times))
    gp = np.array(gt)[:, :3]
    r = dict(ate=m.ate_rmse, sim3=m.sim3_ate_rmse, scale=m.sim3_scale, fuel=int(max(F or [0])), mean_fuel=float(np.mean(F or [0])), ok=ok,
             zero=float(np.sqrt((gp ** 2).sum(1).mean())), path=float(np.linalg.norm(np.diff(gp, axis=0), axis=1).sum()), argmax=int(np.argmax(F or [0])))
    json.dump(r, open(f, 'w')); return r
if __name__ == '__main__':
    zs = sorted(sum([glob.glob(g) for g in sys.argv[1].split()], [])); ws = sys.argv[2:]
    keys = {w: hashlib.md5(open(w, 'rb').read()).hexdigest()[:12] for w in ws}
    jobs = [(w, z, keys[w]) for w in ws for z in zs]
    with Pool(int(os.environ.get('PROCS', '2'))) as p: res = p.map(one, jobs, chunksize=1)
    R = {(j[0], j[1]): r for j, r in zip(jobs, res)}
    base = np.array([R[(ws[0], z)]['ate'] for z in zs])
    print(f"{len(zs)} episodes, zero-output ATE {np.mean([R[(ws[0], z)]['zero'] for z in zs]):.2f} m, mean path {np.mean([R[(ws[0], z)]['path'] for z in zs]):.0f} m")
    for w in ws:
        a = np.array([R[(w, z)]['ate'] for z in zs]); s = np.array([R[(w, z)]['sim3'] for z in zs]); sc = np.array([R[(w, z)]['scale'] for z in zs])
        fu = max(R[(w, z)]['fuel'] for z in zs); bad = sum(not R[(w, z)]['ok'] for z in zs)
        line = f"{os.path.basename(w)[:-5]:28s} ATE {a.mean():6.3f} (med {np.median(a):.3f} max {a.max():.2f}) | Sim3 {s.mean():.3f} | scale med {np.median(sc):.2f} | max fuel {fu:>7,}"
        if w != ws[0]: line += f" | d {np.mean(a - base):+.3f} wins {(a < base).sum()}/{len(a)}"
        if bad: line += f" | FAILED {bad}"
        print(line)
        if os.environ.get('DETAIL'):
            for z in zs: r = R[(w, z)]; print(f"    {os.path.basename(z)[4:-4]:26s} ate {r['ate']:.2f} sim3 {r['sim3']:.2f} scale {r['scale']:.2f} fuel {r['fuel']}")
