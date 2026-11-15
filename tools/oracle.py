"""Which component limits ATE? Rebuild our trajectory from window displacements with oracle speed or oracle direction."""
import sys, glob, pickle, numpy as np
from multiprocessing import Pool
from viobench.tools.runner.sandbox import Sandbox
from viobench.tools.runner.metrics import evaluate_trajectory
W = 9
def ate(G, P, T):
    E = G.copy(); E[:, :3] = P
    return evaluate_trajectory(G, E, T).sim3_ate_rmse
def one(a):
    w, pk = a
    cfg, data = pickle.load(open(pk, "rb")); sb = Sandbox(w); sb.init(cfg)
    E = []; G = []; T = []
    for t, f, imu, gt in data:
        r = sb.step(t, np.ascontiguousarray(f), imu); E.append(np.r_[r.pose, r.twist]); G.append(gt); T.append(t)
    E = np.array(E); G = np.array(G); T = np.array(T)
    pe, pg = E[:, :3], G[:, :3]
    idx = np.arange(0, len(T), W); idx = np.r_[idx, len(T) - 1] if idx[-1] != len(T) - 1 else idx
    de = np.diff(pe[idx], axis=0); dg = np.diff(pg[idx], axis=0)
    ne = np.linalg.norm(de, axis=1) + 1e-9; ng = np.linalg.norm(dg, axis=1) + 1e-9
    # global scale between ours and GT (Sim3 forgives it)
    s = np.sum(ne * ng) / np.sum(ne * ne)
    def build(D):
        P = np.zeros_like(pe); acc = pe[0].copy(); k = 0
        for j in range(len(idx) - 1):
            a, b = idx[j], idx[j + 1]
            for i in range(a, b): P[i] = acc + D[j] * (i - a) / (b - a)
            acc = acc + D[j]
        P[-1] = acc; return P
    win = ate(G, build(de), T)                                     # our windows, re-integrated (sanity)
    spd = ate(G, build(de / ne[:, None] * ng[:, None] / s), T)      # our direction, true speed (kept in our scale)
    dirn = ate(G, build(dg / ng[:, None] * ne[:, None]), T)         # true direction, our speed
    # direction in OUR world frame vs GT frame differ by a rotation: align with the Sim3 rotation first
    from scipy.spatial.transform import Rotation as Rot
    Re = Rot.from_quat(E[:, 3:7]); Rg = Rot.from_quat(G[:, 3:7])
    # our window displacement in our camera frame at the window end, mapped to the world with the TRUE attitude
    da = np.array([ (Rg[idx[j+1]] * Re[idx[j+1]].inv()).apply(de[j]) for j in range(len(de)) ])
    att = ate(G, build(da), T)
    err = (Rg * Re.inv()).magnitude() * 180 / np.pi
    m = evaluate_trajectory(G, E, T)
    return m.sim3_ate_rmse, spd, dirn, att, np.sqrt(np.mean(err**2)), err[-1]
if __name__ == "__main__":
    pks = sorted(glob.glob("/home/claude/tests/real_*.pkl"))
    with Pool(10) as P: R = np.array(P.map(one, [(sys.argv[1], p) for p in pks]))
    print("ep   ours  true-speed  true-dir  true-attitude  rot-rms-deg  rot-final-deg")
    for p, r in zip(pks, R): print(p.split('_')[-1][:6], np.round(r, 3))
    print("mean", np.round(R.mean(0), 3))
