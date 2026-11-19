"""Flight statistics of an episode set: how hard are its trajectories?

Statistics only: no estimator is run and nothing is fitted. Used to generate
look-alike training/dev data for a harder evaluation set (with the user's OK to
measure the server's `private` set this way; it is never used for tuning).

Per episode: duration, start hover, speed, acceleration, yaw rate, climb,
episode config (FOV, exposure, hover), and IMU white noise / bias against truth.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from episode_imu import ideal_imu


def stats(ep):
    gt = ep.ground_truth
    t, R, v, a, w, dt = ideal_imu(gt)
    speed = np.linalg.norm(v, axis=1)
    k = max(1, int(0.25/dt))
    sm = lambda x: np.convolve(x, np.ones(k)/k, 'same')            # 0.25 s smoothing for derivatives
    vs = np.stack([sm(v[:, i]) for i in range(3)], 1)
    acc = np.linalg.norm(np.gradient(vs, dt, axis=0), axis=1)
    move = np.flatnonzero(speed > 0.05)
    vb = np.einsum('nji,nj->ni', R, v)
    fast = speed > 0.1
    imu = ep.imu
    n = min(len(imu), len(t)-1)
    f = np.einsum('nji,nj->ni', R[1:n+1], a[1:n+1]+np.array([0, 0, 9.81]))
    err = np.hstack((imu[:n, 1:4]-f, imu[:n, 4:7]-w[1:n+1]))[10:-10]
    cfg = {k: v for k, v in ep.config.get('episode', {}).items() if k != 'name'}
    cam = ep.config.get('camera', {})
    return dict(
        duration_s=float(t[-1]-t[0]), frames=int(ep.num_frames), config=cfg,
        camera={k: cam[k] for k in cam if k in ('fov_deg', 'horizontal_fov', 'width', 'height', 'exposure_seconds', 'rolling_shutter_seconds')},
        first_motion_s=float(move[0]*dt) if len(move) else None, start_speed=float(speed[:20].mean()),
        speed_mean=float(speed.mean()), speed_p95=float(np.percentile(speed, 95)), speed_max=float(speed.max()),
        accel_p95=float(np.percentile(acc, 95)), yaw_rate_p99=float(np.percentile(np.abs(w[:, 2]), 99)),
        climb_p95=float(np.percentile(np.abs(v[:, 2]), 95)), path_m=float(speed.sum()*dt),
        net_extent_m=float(np.linalg.norm(gt[:, 1:4]-gt[0, 1:4], axis=1).max()),
        forward_fraction=float((vb[fast, 0]/speed[fast]).mean()) if fast.any() else None,
        lateral_speed_rms=float(np.sqrt((vb[fast, 1]**2).mean())) if fast.any() else None,
        imu_noise=(err-err.mean(0)).std(0).round(4).tolist(), imu_bias=err.mean(0).round(4).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('set', help="'public', 'private', or a folder name under episodes/")
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = Path('/var/tmp')/f'novel-approaches-{os.getuid()}'
    os.environ['CONTA_CACHE'] = str(root/'conta')
    os.environ['CONTA_SEARCH_PATHS'] = os.path.expanduser('~/.cache/conta')
    from viobench.tools.datasets.sources import episodes
    if args.set in ('public', 'private'):
        import viobench.episodes as sets
        sources = list(getattr(sets, args.set))
    else:
        sources = sorted(glob.glob(str(root/'episodes'/args.set/'*.zip')))
    rows = [stats(ep) for ep in episodes(sources)]
    out = args.out or root/f'flight_statistics_{args.set}.json'
    out.write_text(json.dumps(rows, indent=1))
    keys = ['duration_s', 'first_motion_s', 'start_speed', 'speed_mean', 'speed_p95', 'speed_max', 'accel_p95',
            'yaw_rate_p99', 'climb_p95', 'path_m', 'net_extent_m', 'forward_fraction', 'lateral_speed_rms']
    print(f'{args.set}: {len(rows)} episodes')
    for k in keys:
        vals = np.array([r[k] for r in rows if r[k] is not None], dtype=float)
        print(f'  {k:18s} min {vals.min():8.3f}  median {np.median(vals):8.3f}  max {vals.max():8.3f}')
    noise = np.array([r['imu_noise'] for r in rows]); bias = np.abs(np.array([r['imu_bias'] for r in rows]))
    print('  imu noise median   ', np.median(noise, 0).round(3), '\n  |imu bias| median  ', np.median(bias, 0).round(3))
    print('  configs            ', sorted({json.dumps(r['config'], sort_keys=True) for r in rows})[:4])
    print('  cameras            ', sorted({json.dumps(r['camera'], sort_keys=True) for r in rows})[:4])


if __name__ == '__main__':
    main()
