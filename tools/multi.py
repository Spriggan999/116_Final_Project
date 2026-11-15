"""Score wasm variants on all real episode pickles in parallel: mean Sim3 ATE, per-episode, scale, velocity rmse, max fuel."""
import sys, glob, pickle, numpy as np
from multiprocessing import Pool
from viobench.tools.runner.sandbox import Sandbox
from viobench.tools.runner.metrics import evaluate_trajectory
def one(args):
    wasm, pk = args
    cfg, data = pickle.load(open(pk, "rb"))
    sb = Sandbox(wasm); sb.init(cfg)
    E=[];G=[];T=[];F=0
    try:
        for t, f, imu, gt in data:
            r = sb.step(t, np.ascontiguousarray(f), imu); E.append(np.r_[r.pose, r.twist]); G.append(gt); T.append(t); F=max(F,r.fuel)
    except Exception as e:
        return (wasm, pk, 99, 0, 99, F, 999)
    m = evaluate_trajectory(np.array(G), np.array(E), np.array(T))
    return (wasm, pk, m.sim3_ate_rmse, m.sim3_scale, m.velocity_rmse, F, np.degrees(m.rotation_rmse))
if __name__ == "__main__":
    pks = sorted(glob.glob("/home/claude/tests/real_*.pkl"))
    jobs = [(w, p) for w in sys.argv[1:] for p in pks]
    with Pool(8) as pool: res = pool.map(one, jobs)
    for w in sys.argv[1:]:
        r = [x for x in res if x[0] == w]
        a = np.array([x[2] for x in r]); s = np.array([x[3] for x in r]); v = np.array([x[4] for x in r])
        print(f"{w.split('/')[-1]:10s} MEAN {a.mean():.3f}  eps {' '.join(f'{x:.2f}' for x in a)}  scale med {np.median(s):.2f}  vel {np.median(v):.2f}  fuel {max(x[5] for x in r):,}  rot {np.mean([x[6] for x in r]):.1f}deg")
