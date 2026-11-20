"""Decode DIAG9 dumps into (features, dmeas, true window displacement) records."""
import numpy as np, glob, os
def records(f, shift=0):
    d = np.load(f); E = d['E']; P = d['gt'][:, :3]; n = len(E)
    A = {}; B = {}
    for k in range(n):
        if E[k, 0] == 0: A[k] = E[k, [1, 2, 7, 8, 9, 10, 11, 12]]
        else: B[k] = E[k, [1, 2, 7, 8, 9]]
    out = []
    prevn = 0
    for k in sorted(B):
        b = B[k]; nn = int(round(b[3]))
        if nn == prevn or nn == 0: continue
        prevn = nn
        kk = k
        if (k - 1) in A and (k - 3) in A and not np.allclose(A[k - 1], A[k - 3]): kk = k - 1
        a = A.get(kk if kk != k else k + 1)
        if a is None or b[4] < 0.5: continue
        k0 = kk + shift
        if k0 - 8 < 0 or k0 - 1 >= n: continue
        w1 = b[2]
        dt = (1 - w1) * (P[k0 - 2] - P[k0 - 8]) + w1 * (P[k0 - 1] - P[k0 - 7])
        feat = np.r_[a[1:8], b[0], b[1]]
        out.append(np.r_[feat, a[0], np.linalg.norm(dt), kk])
    return np.array(out).reshape(-1, 12)
def load(pattern, shift=0):
    R = []; G = []
    for i, f in enumerate(sorted(glob.glob(os.path.expanduser(pattern)))):
        r = records(f, shift); R.append(r); G.append(np.full(len(r), i))
    return np.vstack(R), np.concatenate(G)
if __name__ == '__main__':
    import sys
    for sh in (-2, -1, 0, 1, 2):
        R, _ = load('~/dump9/t_train_med_*.npz', sh)
        ok = (R[:, 10] > 0.03) & (R[:, 9] > 1e-3)
        y = np.log(R[ok, 10] / R[ok, 9])
        print(f"shift {sh:+d}: n {ok.sum()}  mean log(true/meas) {y.mean():+.3f}  std {y.std():.3f}  MAE about median {np.median(np.abs(y - np.median(y))):.3f}")
