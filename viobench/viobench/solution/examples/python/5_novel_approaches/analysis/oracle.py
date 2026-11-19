"""Where does an estimator's unaligned ATE come from? Swap one component for truth.

Input: benchmark log directories containing trajectory.npz (times, truth, estimates;
(N, 13) camera states relative to the first camera: position, quaternion xyzw, twist).
Each frame-to-frame step of the estimate is decomposed and one part is replaced by
ground truth before re-chaining the path from the origin:

  as_built          the estimate itself (official unaligned ATE RMSE)
  true_speed        estimated step direction, true step length
  true_direction    true world-frame step direction, estimated length
  true_rotation     estimated step in the estimated camera frame, rotated by true orientation
  speed+rotation    both of the above
  global_scale      one scale factor for all episodes (least squares on the steps)
  episode_scale     one scale factor per episode (oracle; shows how much is a per-flight bias)

Diagnostic only: these use ground truth and are not achievable scores.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from viobench.tools.geometry import quaternion_to_matrix


def rmse(path, truth):
    return float(np.sqrt(np.mean(np.sum((path - truth)**2, axis=1))))


def chain(steps):
    return np.vstack((np.zeros(3), np.cumsum(steps, axis=0)))


def variants(truth, est, global_scale=None, stride=20):
    # Decompose over `stride`-frame steps: per-frame estimates jitter, so per-frame
    # step lengths overcount distance even when net displacement is right.
    truth, est = truth[::stride], est[::stride]
    tp, ep = truth[:, :3], est[:, :3]
    dt, de = np.diff(tp, axis=0), np.diff(ep, axis=0)
    lt, le = np.linalg.norm(dt, axis=1), np.linalg.norm(de, axis=1)
    ut = dt/np.maximum(lt, 1e-12)[:, None]
    ue = de/np.maximum(le, 1e-12)[:, None]
    Rt = quaternion_to_matrix(truth[:-1, 3:7])     # camera k -> camera 0 (xyzw)
    Re = quaternion_to_matrix(est[:-1, 3:7])
    body = np.einsum('nji,nj->ni', Re, de)          # estimated step in the estimated camera frame
    rotated = np.einsum('nij,nj->ni', Rt, body)     # ...expressed with the true orientation
    lr = np.linalg.norm(rotated, axis=1)
    out = dict(as_built=rmse(ep, tp),
               true_speed=rmse(chain(ue*lt[:, None]), tp),
               true_direction=rmse(chain(ut*le[:, None]), tp),
               true_rotation=rmse(chain(rotated), tp),
               **{'speed+rotation': rmse(chain(rotated/np.maximum(lr, 1e-12)[:, None]*lt[:, None]), tp)})
    s = float(lt @ le/max(le @ le, 1e-12))
    out['episode_scale'] = rmse(chain(de*s), tp)
    out['episode_scale_factor'] = s
    if global_scale is not None:
        out['global_scale'] = rmse(chain(de*global_scale), tp)
    out['distance_m'] = float(lt.sum())
    out['estimated_distance_m'] = float(le.sum())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logs', type=Path, nargs='+', help='directories containing <episode>/trajectory.npz')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--stride', type=int, default=20, help='frames per decomposed step (20 = 1 s)')
    args = parser.parse_args()
    files = sorted(f for d in args.logs for f in d.glob('*/trajectory.npz'))
    if not files:
        parser.error('no trajectory.npz files found')
    data = [np.load(f) for f in files]
    k = args.stride
    lt = np.concatenate([np.linalg.norm(np.diff(z['truth'][::k, :3], axis=0), axis=1) for z in data])
    le = np.concatenate([np.linalg.norm(np.diff(z['estimates'][::k, :3], axis=0), axis=1) for z in data])
    g = float(lt @ le/max(le @ le, 1e-12))
    rows = {f.parent.name[:8]: variants(z['truth'], z['estimates'], g, k) for f, z in zip(files, data)}
    keys = ['as_built', 'true_speed', 'true_direction', 'true_rotation', 'speed+rotation', 'global_scale', 'episode_scale']
    print(f'{len(rows)} episodes; {k}-frame steps; ATE on those frames; global step-length scale {g:.3f}')
    print(f"{'episode':<10}" + ''.join(f'{k:>15}' for k in keys) + f"{'ep. scale':>11}{'dist m':>8}")
    for name, r in rows.items():
        print(f'{name:<10}' + ''.join(f'{r[k]:15.3f}' for k in keys) + f"{r['episode_scale_factor']:11.2f}{r['distance_m']:8.1f}")
    mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in keys}
    print(f"{'MEAN':<10}" + ''.join(f'{mean[k]:15.3f}' for k in keys))
    if args.out:
        args.out.write_text(json.dumps(dict(global_scale=g, mean=mean, episodes=rows), indent=2))


if __name__ == '__main__':
    main()
