"""orbias.py SET wasm MODE : ATE when the gyro stream is corrected with the true bias (MODE z = yaw axis only, xyz = all, none)."""
import sys, glob, os, json, hashlib, numpy as np
from multiprocessing import Pool
sys.path.insert(0, __import__('os').environ.get('VIOBENCH_ROOT') or (glob.glob('/sessions/*/mnt/Final_project/viobench') + [''])[0])   # VIOBENCH_ROOT = the folder holding the viobench package
from viobench.tools.datasets import sources
from viobench.tools.runner.sandbox import Sandbox
from viobench.tools.runner.metrics import evaluate_trajectory
C = os.path.expanduser('~/ev/cache')
def one(a):
    w, z, key, mode = a
    f = f'{C}/orb{mode}_{key}_{os.path.basename(z)}.json'
    if os.path.exists(f): return json.load(open(f))
    ep = next(iter(sources.episodes([z]))); cfg, frames = ep.adapt((320, 240)); gt = np.array(ep.ground_truth_state())
    R_ci = np.array(cfg['T_cam_imu'])[:3, :3]
    IM = [np.array(ep.imu_window(i)) for i in range(len(gt))]
    G = np.array([im[:, 4:7].mean(0) for im in IM]); b = (G - (R_ci.T @ gt[:, 10:13].T).T).mean(0)
    m = {'z': np.array([0, 0, 1.0]), 'xyz': np.ones(3), 'none': np.zeros(3)}[mode]
    sb = Sandbox(w); sb.init(cfg); E = []
    for i, fr in enumerate(frames):
        im = IM[i].copy(); im[:, 4:7] -= b * m
        r = sb.step(ep.frame_times[i], np.ascontiguousarray(np.array(fr, dtype=np.uint8)), im); E.append(np.r_[r.pose, r.twist])
    mt = evaluate_trajectory(gt, np.array(E), np.array(ep.frame_times))
    r = dict(ate=mt.ate_rmse, sim3=mt.sim3_ate_rmse, rot=mt.rotation_rmse); json.dump(r, open(f, 'w')); return r
if __name__ == '__main__':
    zs = sorted(sum([glob.glob(g) for g in sys.argv[1].split()], [])); w = sys.argv[2]
    key = hashlib.md5(open(w, 'rb').read()).hexdigest()[:12]
    for mode in sys.argv[3].split(','):
        with Pool(2) as p: R = p.map(one, [(w, z, key, mode) for z in zs], chunksize=1)
        print(f"{mode:5s} ATE {np.mean([r['ate'] for r in R]):.3f}  Sim3 {np.mean([r['sim3'] for r in R]):.3f}  rotRMSE {np.mean([r['rot'] for r in R]):.3f}")
