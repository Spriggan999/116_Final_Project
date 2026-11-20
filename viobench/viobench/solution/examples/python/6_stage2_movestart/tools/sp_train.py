"""Train the tiny speed-correction MLP: y = log(true |d| / dmeas) from 9 window features. Numpy only.
Usage: sp_train.py H [epochs]  (H hidden units, 0 = linear). Prints house-split CV and writes ~/speednn_H.npz (fit on all train data)."""
import numpy as np, sys, os, re, glob
sys.path.insert(0, os.path.expanduser('~/ev')); from sp_data import records
def load(pat):
    X = []; Y = []; H = []
    for f in sorted(glob.glob(os.path.expanduser(pat))):
        r = records(f, -1); ok = (r[:, 10] > 0.03) & (r[:, 9] > 1e-3)
        r = r[ok]; X.append(r[:, :9]); Y.append(np.log(r[:, 10] / r[:, 9])); H += [int(re.search(r'Train-(\d+)', f).group(1))] * len(r)
    return np.vstack(X), np.concatenate(Y), np.array(H)
def fit(X, Y, Hn, ep=300, lr=3e-3, l2=1e-3, seed=0, delta=0.3):
    rng = np.random.default_rng(seed); d = X.shape[1]
    mu = X.mean(0); sd = X.std(0) + 1e-6; Z = (X - mu) / sd
    if Hn == 0: W1 = None; W = [rng.normal(0, .01, (d, 1)), np.zeros(1)]
    else: W = [rng.normal(0, 1 / np.sqrt(d), (d, Hn)), np.zeros(Hn), rng.normal(0, .1, (Hn, 1)), np.zeros(1)]
    m = [np.zeros_like(w) for w in W]; v = [np.zeros_like(w) for w in W]; t = 0; n = len(Z); bs = 256
    for e in range(ep):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            b = idx[s:s + bs]; z = Z[b]; y = Y[b]
            if Hn == 0:
                out = (z @ W[0])[:, 0] + W[1][0]; r = out - y; g = np.clip(r, -delta, delta) / len(b)
                G = [z.T @ g[:, None] + l2 * W[0], np.array([g.sum()])]
            else:
                h = np.tanh(z @ W[0] + W[1]); out = (h @ W[2])[:, 0] + W[3][0]; r = out - y; g = np.clip(r, -delta, delta) / len(b)
                gh = (g[:, None] @ W[2].T) * (1 - h * h)
                G = [z.T @ gh + l2 * W[0], gh.sum(0), h.T @ g[:, None] + l2 * W[2], np.array([g.sum()])]
            t += 1
            for i in range(len(W)):
                m[i] = .9 * m[i] + .1 * G[i]; v[i] = .999 * v[i] + .001 * G[i] ** 2
                W[i] -= lr * (m[i] / (1 - .9 ** t)) / (np.sqrt(v[i] / (1 - .999 ** t)) + 1e-8)
    return mu, sd, W
def pred(model, X, Hn):
    mu, sd, W = model; z = (X - mu) / sd
    if Hn == 0: return (z @ W[0])[:, 0] + W[1][0]
    return (np.tanh(z @ W[0] + W[1]) @ W[2])[:, 0] + W[3][0]
if __name__ == '__main__':
    Hn = int(sys.argv[1]); ep = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    X, Y, H = load('~/dump9/t_*.npz'); hs = np.unique(H)
    print(f"{len(Y)} windows from {len(hs)} houses; target mean {Y.mean():+.3f} MAE-about-mean {np.abs(Y - Y.mean()).mean():.3f}")
    folds = np.array_split(np.random.default_rng(1).permutation(hs), 4); E0 = []; E1 = []
    for f in folds:
        te = np.isin(H, f); mdl = fit(X[~te], Y[~te], Hn, ep)
        p = np.clip(pred(mdl, X[te], Hn), -0.7, 0.7)
        E0.append(np.abs(Y[te] - Y[~te].mean())); E1.append(np.abs(Y[te] - p))
    E0 = np.concatenate(E0); E1 = np.concatenate(E1)
    print(f"H={Hn}: CV MAE constant-offset {E0.mean():.3f} -> model {E1.mean():.3f}  (RMS {np.sqrt((E0**2).mean()):.3f} -> {np.sqrt((E1**2).mean()):.3f})")
    mdl = fit(X, Y, Hn, ep); mu, sd, W = mdl
    np.savez(os.path.expanduser(f'~/speednn_{Hn}.npz'), mu=mu, sd=sd, *W)
