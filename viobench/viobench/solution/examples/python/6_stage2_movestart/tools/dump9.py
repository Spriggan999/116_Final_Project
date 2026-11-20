import sys, glob, os, numpy as np
sys.path.insert(0, __import__('os').environ.get('VIOBENCH_ROOT') or (glob.glob('/sessions/*/mnt/Final_project/viobench') + [''])[0])   # VIOBENCH_ROOT = the folder holding the viobench package
from viobench.tools.datasets import sources
from viobench.tools.runner.sandbox import Sandbox
from multiprocessing import Pool
wasm = sys.argv[1]; tag = sys.argv[3]; D = os.path.expanduser('~/dump9'); os.makedirs(D, exist_ok=True)
def one(z):
    out = f'{D}/{tag}_{os.path.basename(os.path.dirname(z))}_{os.path.basename(z)[4:-4]}.npz'
    if os.path.exists(out): return out
    ep = next(iter(sources.episodes([z]))); cfg, frames = ep.adapt((320, 240)); gt = np.array(ep.ground_truth_state())
    sb = Sandbox(wasm); sb.init(cfg); E = []
    for i, fr in enumerate(frames):
        r = sb.step(ep.frame_times[i], np.ascontiguousarray(np.array(fr, dtype=np.uint8)), np.array(ep.imu_window(i))); E.append(np.r_[r.pose, r.twist])
    np.savez(out + '.tmp.npz', E=np.array(E), gt=gt, t=np.array(ep.frame_times), fx=cfg['fx']); os.rename(out + '.tmp.npz', out); return out
if __name__ == '__main__':
    zs = sorted(sum([glob.glob(g) for g in sys.argv[2].split()], []))
    zs = [z for z in zs if not os.path.exists(f'{D}/{tag}_{os.path.basename(os.path.dirname(z))}_{os.path.basename(z)[4:-4]}.npz')]
    with Pool(2) as p: p.map(one, zs, chunksize=1)
