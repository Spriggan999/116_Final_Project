"""evmid.py K0 SET wasm1 [wasm2 ...]  : start each episode K0 frames in (no hover at the start; truth re-referenced to the new first frame).
Prints unaligned ATE, zero-output ATE, max fuel. PROCS env (default 2)."""
import sys, glob, os, json, hashlib, numpy as np
from multiprocessing import Pool
sys.path.insert(0, __import__('os').environ.get('VIOBENCH_ROOT') or (glob.glob('/sessions/*/mnt/Final_project/viobench') + [''])[0])   # VIOBENCH_ROOT = the folder holding the viobench package
from viobench.tools.datasets import sources
from viobench.tools.runner.sandbox import Sandbox
from viobench.tools.runner.metrics import evaluate_trajectory
from viobench.tools.geometry import quaternion_to_matrix, matrix_to_quaternion
C = os.path.expanduser('~/ev/cache'); os.makedirs(C, exist_ok=True)
def one(a):
    wasm, z, key, k0, sk, nz, bz = a
    f = f'{C}/mid3{k0}' + (f'x{sk}' if sk > 1 else '') + (f'n{nz}b{bz}' + os.environ.get('BIAS_ONLY', '') if (nz != 1 or bz) else '') + f'_{key}_{os.path.basename(z)}.json'
    import zlib; rng = np.random.default_rng(zlib.crc32(os.path.basename(z).encode())); ex = np.sqrt(max(nz * nz - 1, 0))
    bg = rng.normal(size=3); bg *= 0.02 * bz / np.linalg.norm(bg); ba = rng.normal(size=3); ba *= 0.2 * bz / np.linalg.norm(ba)
    bo = os.environ.get('BIAS_ONLY', '')
    if bo == 'g': ba = ba * 0
    if bo == 'a': bg = bg * 0
    if os.path.exists(f): return json.load(open(f))
    ep = next(iter(sources.episodes([z]))); cfg, frames = ep.adapt((320, 240)); gt = np.array(ep.ground_truth_state())
    R0 = quaternion_to_matrix(gt[k0, 3:7]); p0 = gt[k0, :3].copy(); G = []
    idx = list(range(k0, len(gt), sk))
    for g in gt[idx]:
        G.append(np.r_[R0.T @ (g[:3] - p0), matrix_to_quaternion(R0.T @ quaternion_to_matrix(g[3:7])), g[7:]])
    G = np.array(G)
    sb = Sandbox(wasm); sb.init(cfg); E = []; F = []; ok = True
    try:
        for i, fr in enumerate(frames):
            if i < k0 or (i - k0) % sk: continue
            im = np.concatenate([np.array(ep.imu_window(j)) for j in range(max(0, i - sk + 1) if i > k0 else i, i + 1)])
            if nz != 1 or bz:
                im = im.copy(); im[:, 1:4] += ba + rng.normal(0, 0.28 * ex, (len(im), 3)); im[:, 4:7] += bg + rng.normal(0, 0.07 * ex, (len(im), 3))
            r = sb.step(ep.frame_times[i], np.ascontiguousarray(np.array(fr, dtype=np.uint8)), im)
            E.append(np.r_[r.pose, r.twist]); F.append(r.fuel)
    except Exception as e:
        ok = False
    n = len(E)
    if n < len(G): E += [np.full(13, np.nan)] * (len(G) - n)
    m = evaluate_trajectory(G, np.array(E), np.array(ep.frame_times)[idx])
    r = dict(ate=m.ate_rmse, sim3=m.sim3_ate_rmse, scale=m.sim3_scale, rpe=m.rpe_rotation_rmse, fuel=int(max(F or [0])), ok=ok, zero=float(np.sqrt((G[:, :3] ** 2).sum(1).mean())),
             v0=float(np.linalg.norm(gt[k0, 7:10])), tilt0=float(np.degrees(np.arccos(np.clip((quaternion_to_matrix(gt[k0,3:7]))[1,1],-1,1)))))
    json.dump(r, open(f, 'w')); return r
if __name__ == '__main__':
    import re as _re; _m = _re.match(r'(\d+)(?:x(\d+))?(?:n([\d.]+))?(?:b([\d.]+))?$', sys.argv[1]); k0 = int(_m.group(1)); sk = int(_m.group(2) or 1); nz = float(_m.group(3) or 1); bz = float(_m.group(4) or 0); zs = sorted(sum([glob.glob(g) for g in sys.argv[2].split()], [])); ws = sys.argv[3:]
    keys = {w: hashlib.md5(open(w, 'rb').read()).hexdigest()[:12] for w in ws}
    jobs = [(w, z, keys[w], k0, sk, nz, bz) for w in ws for z in zs]
    with Pool(int(os.environ.get('PROCS', '2'))) as p: res = p.map(one, jobs, chunksize=1)
    R = {(j[0], j[1]): r for j, r in zip(jobs, res)}
    print(f"start at frame {k0} skip {sk}: {len(zs)} episodes, zero-output ATE {np.mean([R[(ws[0], z)]['zero'] for z in zs]):.2f} m, start speed {np.mean([R[(ws[0], z)]['v0'] for z in zs]):.2f} m/s, start tilt {np.mean([R[(ws[0], z)]['tilt0'] for z in zs]):.1f} deg")
    for w in ws:
        a = np.array([R[(w, z)]['ate'] for z in zs]); s = np.array([R[(w, z)]['sim3'] for z in zs])
        print(f"  {os.path.basename(w)[:-5]:28s} ATE {a.mean():6.3f} (med {np.median(a):.3f} max {a.max():.2f}) | Sim3 {s.mean():.3f} | scale med {np.median([R[(w, z)][chr(115)+chr(99)+chr(97)+chr(108)+chr(101)] for z in zs]):.2f} | RPErot {np.mean([R[(w, z)][chr(114)+chr(112)+chr(101)] for z in zs]):.3f} | fuel {max(R[(w, z)]['fuel'] for z in zs):,} | failed {sum(not R[(w, z)]['ok'] for z in zs)}")
        for z in zs: r = R[(w, z)]; print(f"      {os.path.basename(z)[4:-4]:26s} ate {r['ate']:.2f} zero {r['zero']:.2f} sim3 {r['sim3']:.2f} rpe {r['rpe']:.3f}")
