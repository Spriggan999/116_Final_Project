"""roterr.py K0 zip wasm : rotation error over time (tilt and yaw parts, deg) and 1 s RPE rotation per 5 s block."""
import sys, glob, numpy as np
sys.path.insert(0, __import__('os').environ.get('VIOBENCH_ROOT') or (glob.glob('/sessions/*/mnt/Final_project/viobench') + [''])[0])   # VIOBENCH_ROOT = the folder holding the viobench package
from viobench.tools.datasets import sources
from viobench.tools.runner.sandbox import Sandbox
from viobench.tools.geometry import quaternion_to_matrix as q2m
k0 = int(sys.argv[1]); ep = next(iter(sources.episodes([sys.argv[2]]))); cfg, frames = ep.adapt((320, 240)); gt = np.array(ep.ground_truth_state())
sb = Sandbox(sys.argv[3]); sb.init(cfg); Rs = []
T = np.array(cfg['T_cam_imu']) if 'T_cam_imu' in cfg else None
for i, fr in enumerate(frames):
    if i < k0: continue
    r = sb.step(ep.frame_times[i], np.ascontiguousarray(np.array(fr, dtype=np.uint8)), np.array(ep.imu_window(i))); Rs.append(q2m(r.pose[3:7]))
R0 = q2m(gt[k0, 3:7]); G = [R0.T @ q2m(g[3:7]) for g in gt[k0:]]
def ang(R): return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
# gravity direction in c0 camera frame: use world z expressed via gt: camera frames; world 'up' in c0 coords
up_c0 = R0.T @ np.array([0, 0, 1.0]) if False else None
n = len(Rs)
print("  t(s)  rotErr  RPE1s(mean over block)")
for s in range(0, n - 20, 100):
    rp = [ang((G[i].T @ G[i + 20]).T @ (Rs[i].T @ Rs[i + 20])) for i in range(s, min(s + 100, n - 20))]
    print(f"{s / 20:5.0f} {ang(G[s].T @ Rs[s]):7.2f} {np.mean(rp):7.2f}")
