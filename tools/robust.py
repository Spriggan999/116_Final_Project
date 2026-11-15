"""Robust scorer: every wasm on the 10 public episodes AND stress variants built on the fly (no extra disk):
  base   as recorded
  mid    starts 10 s into the flight (no hover to calibrate on; truth re-referenced to the new first frame)
  imu    extra accel bias (0.15 m/s^2) and gyro bias (0.012 rad/s) in random directions + 1.5x white noise
  light  darker, lower-contrast frames with a slow exposure drift
  fov    narrower lens: centre 80% crop resized back to 320x240 (intrinsics adjusted)
Prints mean Sim3 ATE per variant, split halves A/B of the episodes, worst episode, and max fuel."""
import sys, glob, pickle, numpy as np, cv2
from multiprocessing import Pool
from viobench.tools.runner.sandbox import Sandbox
from viobench.tools.runner.metrics import evaluate_trajectory
from viobench.tools.geometry import quaternion_to_matrix, matrix_to_quaternion
VARIANTS = ["base", "mid", "imu", "light", "fov"]

def transform(cfg, data, v, seed):
    cfg = dict(cfg)
    if v == "base": return cfg, data
    rng = np.random.default_rng(seed)
    if v == "mid":
        k0 = 200
        R0 = quaternion_to_matrix(data[k0][3][3:7]); p0 = data[k0][3][:3]
        out = []
        for t, f, imu, g in data[k0:]:
            R = quaternion_to_matrix(g[3:7])
            g2 = np.r_[R0.T @ (g[:3] - p0), matrix_to_quaternion(R0.T @ R), g[7:]]
            out.append((t, f, imu, g2))
        return cfg, out
    if v == "imu":
        ba = rng.normal(size=3); ba *= 0.15 / np.linalg.norm(ba)
        bgv = rng.normal(size=3); bgv *= 0.012 / np.linalg.norm(bgv)
        out = []
        for t, f, imu, g in data:
            imu = imu.copy()
            imu[:, 1:4] += ba + rng.normal(0, 0.14, (len(imu), 3))
            imu[:, 4:7] += bgv + rng.normal(0, 0.035, (len(imu), 3))
            out.append((t, f, imu, g))
        return cfg, out
    if v == "light":
        out = []
        for i, (t, f, imu, g) in enumerate(data):
            gain = 0.55 * (1 + 0.25 * np.sin(2 * np.pi * t / 17.0))
            f2 = np.clip(f.astype(np.float32) * gain + 18, 0, 255).astype(np.uint8)
            out.append((t, f2, imu, g))
        return cfg, out
    if v == "fov":
        s = 0.8; W, H = 320, 240; cw, ch = int(W * s), int(H * s); x0, y0 = (W - cw) // 2, (H - ch) // 2
        cfg["fx"] = cfg["fx"] / s; cfg["fy"] = cfg["fy"] / s
        cfg["cx"] = (cfg["cx"] - x0 + 0.5) / s - 0.5; cfg["cy"] = (cfg["cy"] - y0 + 0.5) / s - 0.5
        out = [(t, cv2.resize(np.ascontiguousarray(f[y0:y0 + ch, x0:x0 + cw]), (W, H), interpolation=cv2.INTER_LINEAR), imu, g)
               for t, f, imu, g in data]
        return cfg, out

def one(args):
    wasm, pk, v, idx = args
    cfg, data = pickle.load(open(pk, "rb"))
    cfg, data = transform(cfg, data, v, 1000 + idx)
    sb = Sandbox(wasm); sb.init(cfg)
    E = []; G = []; T = []; F = 0
    try:
        for t, f, imu, g in data:
            r = sb.step(t, np.ascontiguousarray(f), imu); E.append(np.r_[r.pose, r.twist]); G.append(g); T.append(t); F = max(F, r.fuel)
    except Exception:
        return (wasm, v, idx, 9.99, F)
    m = evaluate_trajectory(np.array(G), np.array(E), np.array(T))
    return (wasm, v, idx, m.sim3_ate_rmse, F)

if __name__ == "__main__":
    args = sys.argv[1:]
    variants = VARIANTS
    if args and args[0].startswith("--v="): variants = args[0][4:].split(","); args = args[1:]
    pks = sorted(glob.glob("/home/claude/tests/real_*.pkl"))
    jobs = [(w, p, v, i) for w in args for v in variants for i, p in enumerate(pks)]
    with Pool(2) as pool: res = pool.map(one, jobs, chunksize=1)
    for w in args:
        line = f"{w.split('/')[-1]:10s}"
        allv = []; F = 0
        for v in variants:
            a = np.array([x[3] for x in res if x[0] == w and x[1] == v]); F = max(F, max(x[4] for x in res if x[0] == w))
            allv.append(a.mean())
            line += f" | {v} {a.mean():.3f} (A {a[::2].mean():.2f} B {a[1::2].mean():.2f} worst {a.max():.2f})"
        line += f" || ALL {np.mean(allv):.3f}  fuel {F:,}"
        print(line, flush=True)
