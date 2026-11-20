import sys, glob, numpy as np
sys.path.insert(0, __import__('os').environ.get('VIOBENCH_ROOT') or (glob.glob('/sessions/*/mnt/Final_project/viobench') + [''])[0])   # VIOBENCH_ROOT = the folder holding the viobench package
from viobench.tools.datasets import sources
from viobench.tools.runner.sandbox import Sandbox
ep = next(iter(sources.episodes([sys.argv[1]]))); cfg, frames = ep.adapt((320, 240))
sb = Sandbox(sys.argv[2]); sb.init(cfg); E = []
for i, fr in enumerate(frames):
    E.append(sb.step(ep.frame_times[i], np.ascontiguousarray(np.array(fr, dtype=np.uint8)), np.array(ep.imu_window(i))).twist)
E = np.array(E)
for s in [0, 20, 60, 100, 200, 400, 800, 1199]: print(f"t {s/20:5.1f}s  sigma_bg {np.round(E[s,:3],4)}  rot-row sigma (world x,y,z) {np.round(E[s,3:],4)}")
