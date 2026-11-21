"""Supervised ("SL") version: learn each row class's noise from its actual error against ground truth.

1. Replay the training flights on the host with logging (bin/host_log): every direction, speed, small-motion and
   velocity-prior row is written with its innovation y, its noise R, the state part it measures, the estimate it
   predicted with, the time span it covers and the 9 window features.
2. Rebuild the truth in the estimator's world frame (the log's first line holds the world <- first-camera transform
   and the camera-IMU extrinsics) and compute each row's true error r = y - a . (truth - estimate).
3. Fit the same 9-6-7 network as the policy search by Gaussian likelihood: for a row of class c with z = r^2 / R,
   minimise 0.5 * (mu_c + z * exp(-mu_c)). exp(mu_c) is then the multiplier that makes R match the real error.
   Rotation rows and the two process-noise outputs have no direct target here and stay at 1.

    python sl_train.py --train FEEDS... --out sl.txt
"""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch

from search import BIN, LF, LH, LO, LW_N, CLASSES, feeds

DIR, SPD, PRI, SMALL = 1, 2, 3, 4


def quat_to_R(q):   # x y z w
    x, y, z, w = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def log_flight(feed, logdir):
    out = logdir / (feed.name + '.log')
    if not out.exists():
        env = dict(os.environ, LEARN_LFILE=str(out))
        env.pop('LEARN_WFILE', None)
        subprocess.run([str(BIN / 'host_log'), str(feed)], env=env, capture_output=True, check=True)
    return out


def rows_of(feed, logfile):
    """(features, class, z) for every logged row of one flight."""
    lines = logfile.read_text().split('\n')
    init = [l for l in lines if l.startswith('I ')]
    if not init:
        return None
    v = np.array(init[0].split()[1:], float)
    R_wc0, p_wc0, R_ic, r_ci = v[:9].reshape(3, 3), v[9:12], v[12:21].reshape(3, 3), v[21:24]
    t_f = np.fromfile(feed / 'frame_times.f64')
    gt = np.fromfile(feed / 'gt.f64').reshape(len(t_f), 7)
    R_wb = R_wc0 @ quat_to_R(gt[:, 3:7]) @ R_ic.T
    p_imu = gt[:, :3] @ R_wc0.T + p_wc0 - R_wb @ r_ci          # IMU position in the estimator's world
    v_imu = np.gradient(p_imu, t_f, axis=0)

    def P(t):
        return np.stack([np.interp(t, t_f, p_imu[:, i]) for i in range(3)], -1)

    def V(t):
        return np.stack([np.interp(t, t_f, v_imu[:, i]) for i in range(3)], -1)
    R = np.array([l.split()[1:] for l in lines if l.startswith('R ')], float)
    if len(R) == 0:
        return None
    cls = R[:, 0].astype(int)
    t, tk0, td0, tk1, td1, w1 = R[:, 1], R[:, 2], R[:, 3], R[:, 4], R[:, 5], R[:, 6]
    y, Rn, a, est, x = R[:, 8], R[:, 9], R[:, 10:13], R[:, 13:16], R[:, 16:16 + LF]
    d_true = (1 - w1)[:, None] * (P(td0) - P(tk0)) + w1[:, None] * (P(td1) - P(tk1))
    truth = np.where((cls == PRI)[:, None], V(t), d_true)
    r = y - np.einsum('ij,ij->i', a, truth - est)
    z = r * r / np.maximum(Rn, 1e-12)
    keep = np.isfinite(z) & (Rn > 0)
    return x[keep], cls[keep], z[keep]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2 = torch.nn.Linear(LF, LH), torch.nn.Linear(LH, LO)
        torch.nn.init.zeros_(self.l2.weight); torch.nn.init.zeros_(self.l2.bias)

    def forward(self, x):
        return torch.clamp(self.l2(torch.tanh(self.l1(x))), -2.0, 2.0)


def nll(net, x, c, z):
    mu = net(x).gather(1, c[:, None])[:, 0]
    return (0.5 * (mu + z * torch.exp(-mu))).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--zcap', type=float, default=25.0, help='cap on r^2/R (outliers the filter would gate)')
    ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--classes', nargs='+', default=['dir', 'spd', 'pri', 'small'],
                    help='row classes to train on; the others keep multiplier 1')
    args = ap.parse_args()
    use = [CLASSES.index(c) for c in args.classes]
    flights = feeds(args.train)
    logdir = Path(f'/tmp/learned-{os.getuid()}/sl_logs'); logdir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as pool:
        logs = list(pool.map(lambda f: log_flight(f, logdir), flights))
    per = [rows_of(f, l) for f, l in zip(flights, logs)]
    idx = [i for i, p in enumerate(per) if p is not None]
    rng = np.random.default_rng(args.seed)
    val_f = set(rng.choice(idx, size=max(1, len(idx) // 5), replace=False).tolist())   # flight-level split
    def cat(sel):
        X = np.concatenate([per[i][0] for i in sel]); C = np.concatenate([per[i][1] for i in sel])
        Z = np.minimum(np.concatenate([per[i][2] for i in sel]), args.zcap)
        m = np.isin(C, use)
        X, C, Z = X[m], C[m], Z[m]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(C), torch.tensor(Z, dtype=torch.float32)
    tr, va = cat([i for i in idx if i not in val_f]), cat(sorted(val_f))
    stats = {CLASSES[k]: dict(rows=int((tr[1] == k).sum()), mean_z=float(tr[2][tr[1] == k].mean()),
                              median_z=float(tr[2][tr[1] == k].median())) for k in (DIR, SPD, PRI, SMALL) if (tr[1] == k).any()}
    print('rows per class (train):', json.dumps(stats))
    torch.manual_seed(args.seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    best, best_state, bad = 1e9, None, 0
    for ep in range(args.epochs):
        opt.zero_grad(); loss = nll(net, *tr); loss.backward(); opt.step()
        with torch.no_grad():
            vl = float(nll(net, *va))
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 250 == 0:
            print(f'epoch {ep} train {float(loss):.4f} val {vl:.4f}', flush=True)
        if bad > 300:
            break
    net.load_state_dict(best_state)
    with torch.no_grad():   # classes without targets keep multiplier 1
        for k in [k for k in range(LO) if k not in use]:
            net.l2.weight[k].zero_(); net.l2.bias[k].zero_()
        const = {CLASSES[k]: float(torch.exp(net(va[0][va[1] == k])[:, k]).mean()) for k in use if (va[1] == k).any()}
    w = np.concatenate([net.l1.weight.detach().numpy().ravel(), net.l1.bias.detach().numpy(),
                        net.l2.weight.detach().numpy().ravel(), net.l2.bias.detach().numpy()]).astype(float)
    assert len(w) == LW_N
    Path(args.out).write_text(' '.join(f'{v:.10g}' for v in w))
    Path(args.out).with_suffix('.json').write_text(json.dumps(dict(train=args.train, classes=args.classes, flights=len(idx), val_flights=len(val_f),
                                                                   best_val_nll=best, class_stats=stats, mean_multiplier_val=const), indent=1))
    print(f'best val NLL {best:.4f}; mean learned multiplier on validation rows: {json.dumps(const)}')


if __name__ == '__main__':
    main()
