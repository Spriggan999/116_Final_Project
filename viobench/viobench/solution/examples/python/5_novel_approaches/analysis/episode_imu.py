"""Sensor errors and start-of-flight behaviour in benchmark-format episodes.

Uses the episodes' 200 Hz ground truth (t, position, quaternion xyzw) to rebuild the
ideal IMU and compares it with the delivered IMU. Answers:
  - what happens during the 1 s start hover (is the drone truly still? when does it move?)
  - per-flight accelerometer / gyro bias, how much it drifts, white noise
  - how well a hover-window estimate recovers the whole-flight bias, vs window length
Analysis only: run it on held-out/training episodes; public episodes are fine to
inspect but nothing here fits parameters to them.
"""
import argparse
import glob
import os
import numpy as np


def ideal_imu(gt):
    from viobench.tools.geometry import quaternion_to_matrix
    t = gt[:, 0]
    dt = float(np.median(np.diff(t)))
    R = quaternion_to_matrix(gt[:, 4:8])
    v = np.gradient(gt[:, 1:4], dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    dR = np.einsum('nji,njk->nik', R[:-1], R[1:])       # R_k^T R_{k+1}
    w = np.stack([dR[:, 2, 1]-dR[:, 1, 2], dR[:, 0, 2]-dR[:, 2, 0], dR[:, 1, 0]-dR[:, 0, 1]], 1)/(2*dt)
    w = np.vstack((w, w[-1:]))
    return t, R, v, a, w, dt


def analyze(ep):
    gt = ep.ground_truth
    imu = ep.imu
    t, R, v, a, w, dt = ideal_imu(gt)
    n = min(len(imu), len(t)-1)
    meas = imu[:n, 1:]
    errs = {}
    for g in (+9.81, -9.81):            # pick the gravity sign that explains the data
        f = np.einsum('nji,nj->ni', R[1:n+1], a[1:n+1]+np.array([0, 0, g]))
        e = np.hstack((meas[:, :3]-f, meas[:, 3:]-w[1:n+1]))
        errs[g] = e
    g = min(errs, key=lambda k: np.abs(errs[k][200:-200, 2]).mean())
    err = errs[g][10:-10]
    speed = np.linalg.norm(v, axis=1)
    move = np.flatnonzero(speed > 0.02)
    start_move = move[0]*dt if len(move) else None
    bias = err.mean(0)
    noise = (err-bias).std(0)
    hover = {}
    for seconds in (0.25, 0.5, 0.9):
        k = int(seconds/dt)
        hover[seconds] = np.abs(err[20:20+k].mean(0)-bias)
    win = int(10/dt)
    chunks = np.array([err[i:i+win].mean(0) for i in range(0, len(err)-win+1, win)])
    return dict(gravity_sign=g, start_move_s=start_move, start_accel=meas[:5, 2].round(2).tolist(),
                hover_max_speed=float(speed[:int(1.0/dt)].max()), bias=bias, noise=noise,
                hover_error=hover, drift=chunks.max(0)-chunks.min(0))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', default='heldout', help="'public' or a folder name under episodes/")
    args = parser.parse_args()
    from viobench.tools.datasets.sources import episodes
    os.environ['CONTA_SEARCH_PATHS'] = os.path.expanduser('~/.cache/conta')
    if args.set == 'public':
        from viobench.episodes import public
        sources = list(public)
    else:
        sources = sorted(glob.glob(f'/var/tmp/novel-approaches-{os.getuid()}/episodes/{args.set}/*.zip'))
    rows = [analyze(ep) for ep in episodes(sources)]
    np.set_printoptions(precision=4, suppress=True)
    A = lambda k: np.array([r[k] for r in rows])
    print(f'{len(rows)} episodes ({args.set}); gravity sign {set(A("gravity_sign"))}')
    print('first motion (speed > 2 cm/s) at s:', np.array([r['start_move_s'] for r in rows], dtype=float))
    print('max true speed during first 1 s :', A('hover_max_speed').max().round(4))
    print('accel z, first 5 samples (ep 0)  :', rows[0]['start_accel'])
    print('                       accel x,y,z (m/s^2)           gyro x,y,z (rad/s)')
    print('|bias| median         ', np.median(np.abs(A('bias')), 0))
    print('|bias| max            ', np.abs(A('bias')).max(0))
    print('white noise std       ', np.median(A('noise'), 0))
    print('bias drift over 10 s  ', np.median(A('drift'), 0))
    for s in (0.25, 0.5, 0.9):
        print(f'hover {s:.2f} s estimate error', np.median([r['hover_error'][s] for r in rows], 0))


if __name__ == '__main__':
    main()
