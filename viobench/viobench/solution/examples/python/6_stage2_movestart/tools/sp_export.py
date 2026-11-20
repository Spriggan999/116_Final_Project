"""sp_export.py H EPOCHS DROP OUTDIR : fit on all train windows (features in DROP=comma list are ignored) and write OUTDIR/speednn.h"""
import numpy as np, sys, os
sys.path.insert(0, os.path.expanduser('~/ev')); from sp_train import load, fit, pred
Hn = int(sys.argv[1]); ep = int(sys.argv[2]); drop = [int(c) for c in sys.argv[3].split(',') if c != '']; out = sys.argv[4]
X, Y, H = load('~/dump9/t_*.npz')
X = X.copy(); X[:, drop] = 0.0
mu, sd, W = fit(X, Y, Hn, ep)
sd = sd.copy(); sd[drop] = 1.0
if Hn == 0: W1 = W[0] / sd[:, None]; b1 = None; b = W[1][0] - (mu / sd) @ W[0][:, 0]
else:
    W1 = W[0] / sd[:, None]; b1 = W[1] - (mu / sd) @ W[0]; W2 = W[2][:, 0]; b2 = W[3][0]
W1[drop, :] = 0.0
p = np.clip(pred((mu, np.where(np.isin(np.arange(9), drop), 1.0, sd), W), X, Hn), -.7, .7)
f = lambda a: ', '.join(f'{x:.9g}' for x in np.ravel(a))
with open(os.path.join(out, 'speednn.h'), 'w') as o:
    o.write(f"// tiny speed-correction model ({'linear' if Hn == 0 else f'9-{Hn}-1 tanh MLP'}), trained on {len(Y)} windows from ProcTHOR houses 20-43 (ignored features: {drop})\n")
    o.write("// output: log(true window displacement / assumed-depth measurement); input normalisation folded into the first layer\n#pragma once\n")
    if Hn == 0:
        o.write(f"static const double SNN_W[9] = {{{f(W1[:, 0])}}};\nstatic const double SNN_B = {b:.9g};\n")
        o.write("static inline double speednn_eval(const double* x) { double y = SNN_B; for (int i = 0; i < 9; i++) y += SNN_W[i] * x[i]; return y; }\n")
    else:
        o.write(f"#define SNN_H {Hn}\nstatic const double SNN_W1[9 * SNN_H] = {{{f(W1)}}};\nstatic const double SNN_B1[SNN_H] = {{{f(b1)}}};\n")
        o.write(f"static const double SNN_W2[SNN_H] = {{{f(W2)}}};\nstatic const double SNN_B2 = {b2:.9g};\n")
        o.write("static inline double speednn_eval(const double* x) {\n    double y = SNN_B2;\n    for (int j = 0; j < SNN_H; j++) {\n        double a = SNN_B1[j];\n        for (int i = 0; i < 9; i++) a += SNN_W1[i * SNN_H + j] * x[i];\n        y += SNN_W2[j] * std::tanh(a);\n    }\n    return y;\n}\n")
print(f"wrote {out}/speednn.h: train MAE {np.abs(Y - p).mean():.3f}")
