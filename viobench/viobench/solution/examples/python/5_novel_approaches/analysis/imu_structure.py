"""What the IMU and the flight pattern give away, from 200 Hz collector recordings.

Sensor error = measured IMU - ideal IMU reconstructed from true state
(specific force R^T (dv/dt - g), body rate omega). Reports per flight:
  hover            how long the drone stays still at the start, and whether it is truly still
  bias             mean sensor error over the whole flight vs over the hover only
                   (does a hover estimate calibrate the entire flight?)
  bias drift       change of the 10 s mean error across the flight (random walk?)
  noise            white-noise std per axis after removing the bias
  flight habits    speed, body-frame velocity direction (nose-first?), yaw rate, climb
"""
import argparse
import glob
import json
import numpy as np

DT = 0.005


def analyze(path):
    with np.load(path) as z:
        imu = z['imu'][1:].astype(float)
        state = z['state'][1:].astype(float)
    R = state[:, 3:12].reshape(-1, 3, 3)
    v = state[:, 12:15]
    acc = np.gradient(v, DT, axis=0)
    ideal_f = np.einsum('nji,nj->ni', R, acc+np.array([0, 0, 9.81]))
    ideal_w = state[:, 15:18]
    err = np.hstack((imu[:, :3]-ideal_f, imu[:, 3:]-ideal_w))[5:-5]   # drop gradient edge samples
    speed = np.linalg.norm(v, axis=1)
    moving = np.flatnonzero(speed > 0.02)
    hover_s = (moving[0] if len(moving) else len(speed))*DT
    h = max(20, int(min(hover_s, 1.0)/DT)-10)
    hover_bias = err[:h].mean(0)
    full_bias = err.mean(0)
    win = int(10/DT)
    chunks = np.array([err[i:i+win].mean(0) for i in range(0, len(err)-win+1, win)])
    noise = (err-full_bias).std(0)
    vb = np.einsum('nji,nj->ni', R, v)
    fast = speed > 0.1
    fwd = vb[fast]/speed[fast, None]
    yaw_rate = ideal_w[:, 2]
    return dict(hover_s=hover_s, hover_speed_max=float(speed[:int(hover_s/DT)].max()) if hover_s > 0 else 0.0,
                hover_bias=hover_bias.tolist(), full_bias=full_bias.tolist(),
                hover_bias_error=(hover_bias-full_bias).tolist(),
                bias_drift_10s=(chunks.max(0)-chunks.min(0)).tolist(), noise=noise.tolist(),
                speed_mean=float(speed.mean()), speed_p95=float(np.percentile(speed, 95)),
                body_velocity_direction_mean=fwd.mean(0).tolist(),
                yaw_rate_p99=float(np.percentile(np.abs(yaw_rate), 99)),
                climb_rate_p95=float(np.percentile(np.abs(v[:, 2]), 95)),
                frac_still=float((speed < 0.05).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default='/var/tmp/novel-approaches-94445')
    parser.add_argument('--out')
    args = parser.parse_args()
    files = sorted(glob.glob(f'{args.root}/flights/*/rollout_s*_e*.npz'))
    rows = {f.split('/')[-1]: analyze(f) for f in files}
    A = lambda k: np.array([r[k] for r in rows.values()])
    np.set_printoptions(precision=4, suppress=True)
    print(f'{len(rows)} flights')
    print('hover duration s      ', np.percentile(A('hover_s'), [0, 50, 100]), ' max speed in hover', A('hover_speed_max').max().round(4))
    print('                 accel x,y,z (m/s^2)        gyro x,y,z (rad/s)')
    print('|full bias| median  ', np.median(np.abs(A('full_bias')), 0))
    print('|full bias| max     ', np.abs(A('full_bias')).max(0))
    print('|hover-full| median ', np.median(np.abs(A('hover_bias_error')), 0))
    print('bias drift (10 s)   ', np.median(A('bias_drift_10s'), 0))
    print('white noise std     ', np.median(A('noise'), 0))
    print('speed mean / p95    ', np.median(A('speed_mean')).round(3), np.median(A('speed_p95')).round(3))
    print('v_body direction    ', A('body_velocity_direction_mean').mean(0), '(unit vectors, averaged; |x| large = nose-first)')
    print('yaw rate p99        ', np.median(A('yaw_rate_p99')).round(3), 'rad/s;  climb p95', np.median(A('climb_rate_p95')).round(3), 'm/s;  still fraction', np.median(A('frac_still')).round(3))
    if args.out:
        open(args.out, 'w').write(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
