"""full.py SET wasm : server-style per-episode table (ATE, Sim3, RPE rot, drift/m) + distance, mean speed, max yaw rate."""
import sys, glob, os, json, hashlib, numpy as np
from multiprocessing import Pool
sys.path.insert(0, __import__('os').environ.get('VIOBENCH_ROOT') or (glob.glob('/sessions/*/mnt/Final_project/viobench') + [''])[0])   # VIOBENCH_ROOT = the folder holding the viobench package
from viobench.tools.datasets import sources
from viobench.tools.runner.sandbox import Sandbox
from viobench.tools.runner.metrics import evaluate_trajectory
C = os.path.expanduser('~/ev/cache')
def one(a):
    wasm, z, key = a
    f = f'{C}/full_{key}_{os.path.basename(z)}.json'
    if os.path.exists(f): return json.load(open(f))
    ep = next(iter(sources.episodes([z]))); cfg, frames = ep.adapt((320, 240)); gt = np.array(ep.ground_truth_state())
    sb = Sandbox(wasm); sb.init(cfg); E = []
    for i, fr in enumerate(frames):
        r = sb.step(ep.frame_times[i], np.ascontiguousarray(np.array(fr, dtype=np.uint8)), np.array(ep.imu_window(i))); E.append(np.r_[r.pose, r.twist])
    E = np.array(E); m = evaluate_trajectory(gt, E, np.array(ep.frame_times))
    w = np.array([np.abs(np.array(ep.imu_window(i))[:, 4:7]).max(0) for i in range(len(gt))])
    r = dict(ate=m.ate_rmse, sim3=m.sim3_ate_rmse, rpe=m.rpe_rotation_rmse, rpet=m.rpe_translation_rmse, dpm=m.drift_per_metre, dist=m.distance_travelled,
             rot=m.rotation_rmse, scale=m.sim3_scale, wmax=float(w.max()), w95=float(np.percentile(np.linalg.norm(w, axis=1), 95)),
             zero=float(np.sqrt((gt[:, :3] ** 2).sum(1).mean())), ext=float(np.ptp(gt[:, :3], axis=0).max()))
    json.dump(r, open(f, 'w')); return r
if __name__ == '__main__':
    zs = sorted(sum([glob.glob(g) for g in sys.argv[1].split()], [])); w = sys.argv[2]
    key = hashlib.md5(open(w, 'rb').read()).hexdigest()[:12]
    with Pool(int(os.environ.get('PROCS', '2'))) as p: res = p.map(one, [(w, z, key) for z in zs], chunksize=1)
    print(f"{'episode':30s}  ATE   Sim3  RPErot  drift/m  dist  rotRMSE scale  zero  extent  gyro95")
    for z, r in zip(zs, res):
        print(f"{os.path.basename(z)[4:-4]:30s} {r['ate']:5.2f} {r['sim3']:5.2f} {r['rpe']:6.3f} {r['dpm']:7.3f} {r['dist']:5.0f} {r['rot']:6.3f} {r['scale']:5.2f} {r['zero']:5.2f} {r['ext']:5.1f} {r['w95']:5.2f}")
    a = lambda k: np.mean([r[k] for r in res])
    print(f"{'MEAN':30s} {a('ate'):5.2f} {a('sim3'):5.2f} {a('rpe'):6.3f} {a('dpm'):7.3f} {a('dist'):5.0f} {a('rot'):6.3f} {a('scale'):5.2f} {a('zero'):5.2f} {a('ext'):5.1f} {a('w95'):5.2f}")
