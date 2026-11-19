"""Causal IMU-history features -> body-frame velocity, from collector recordings.

Features at a frame time t (every 50 ms, after the 1 s hover calibration):
  the last HISTORY seconds of IMU as BIN-second means (accel/9.81, gyro), and the
  gravity direction in the body frame (the estimator knows tilt, not yaw).
Target: velocity in the body frame (state rows: pos, R body->world, vel world, omega).
"""
import glob
import numpy as np

BIN = 10          # IMU samples per bin (50 ms)
HISTORY = 40      # bins (2 s)


def flight_features(path, stride=10):
    with np.load(path) as z:
        imu = z['imu'][1:].astype(np.float64)
        state = z['state'][1:].astype(np.float64)
    n = len(imu)//BIN
    bins = imu[:n*BIN].reshape(n, BIN, 6).mean(1)
    bins[:, :3] /= 9.81
    R = state[BIN-1::BIN][:n, 3:12].reshape(-1, 3, 3)
    v = state[BIN-1::BIN][:n, 12:15]
    vb = np.einsum('nji,nj->ni', R, v)
    gb = np.einsum('nji,j->ni', R, np.array([0., 0., -1.]))
    X, Y = [], []
    for k in range(max(HISTORY, 20), n, max(1, stride//BIN)):
        X.append(np.concatenate([bins[k-HISTORY+1:k+1].ravel(), gb[k]]))
        Y.append(vb[k])
    return np.array(X), np.array(Y)


def load(paths):
    parts = [flight_features(p) for p in paths]
    return [p[0] for p in parts], [p[1] for p in parts]


def recordings(root):
    return sorted(glob.glob(f'{root}/flights/*/rollout_s*_e*.npz'))
