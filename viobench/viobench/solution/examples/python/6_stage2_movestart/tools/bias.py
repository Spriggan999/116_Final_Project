"""bias.py zip wasm : true gyro bias (from ground-truth angular rate) vs the filter's estimate over time; rotation-row usage."""
import sys, glob, numpy as np
sys.path.insert(0, __import__('os').environ.get('VIOBENCH_ROOT') or (glob.glob('/sessions/*/mnt/Final_project/viobench') + [''])[0])   # VIOBENCH_ROOT = the folder holding the viobench package
from viobench.tools.datasets import sources
from viobench.tools.runner.sandbox import Sandbox
ep = next(iter(sources.episodes([sys.argv[1]]))); cfg, frames = ep.adapt((320, 240)); gt = np.array(ep.ground_truth_state())
T = np.array(cfg['T_cam_imu']); R_ci = T[:3, :3]   # cam <- imu
sb = Sandbox(sys.argv[2]); sb.init(cfg); E = []; G = []
for i, fr in enumerate(frames):
    im = np.array(ep.imu_window(i)); G.append(im[:, 4:7].mean(0))
    E.append(sb.step(ep.frame_times[i], np.ascontiguousarray(np.array(fr, dtype=np.uint8)), im).twist)
E = np.array(E); G = np.array(G)
w_body = (R_ci.T @ gt[:, 10:13].T).T          # true body rate
b_true = (G - w_body)
for s in range(0, len(E), 200):
    bt = b_true[max(0, s - 100):s + 100].mean(0)
    rows = E[s, 3]; win = rows // 1e4; rows = rows % 1e4
    print(f"t {s / 20:4.0f}s  true bias {np.round(bt, 4)}  est {np.round(E[s, :3], 4)}  windows {win:.0f} rows {rows:.0f} accepted {E[s, 4]:.0f} kept {E[s, 5]:.0f}")
