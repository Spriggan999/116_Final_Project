# lqg: steer the estimated trajectory with a controller

The estimate is treated as a **virtual drone** that a controller keeps on top of the real one.
The IMU drives it; a controller adds corrections to the IMU so its trajectory stays on track;
the pose and twist are read off that controlled trajectory.

There are two models, built from one source file (`solution.cpp`, a copy of v8; `-DLQG=0` gives v8 exactly):

| Model | Folder | Binary | Error signal (what tells the controller it is off track) |
|---|---|---|---|
| **LQG + vision** | `vision/` | `../wasm/lqg_vision.wasm` | v8's corner tracker (motion direction, parallax speed, gyro check) + velocity prior |
| **LQG, IMU only** | `imu/` | `../wasm/lqg_imu.wasm` | start-hover calibration + velocity prior (no image access) |

Build: `viobench bundle 5_novel_approaches/lqg/vision -o lqg_vision.wasm` (same for `imu`).

## How it works

LQG = an estimator (LQE, a Kalman filter) plus a controller (LQR).

**1. Plant.** The estimated trajectory, integrated from the corrected IMU:
attitude from `gyro - b_g + u_w`, velocity and position from `accel - b_a + u_a`.

**2. Estimator (LQE): fixed gains, designed once.**
- At start-up, the Riccati recursion of v8's 18-state error model runs over 30 s of a nominal
  flight (level, 0.25 m/s, motion cycling forward / up / sideways / diagonal).
- The converged covariance before each kind of update is stored: gyro-check rows, direction rows,
  velocity prior.
- At run time each update uses that fixed covariance, rotated to the current heading and tilt:
  `K = P Hᵀ (H P Hᵀ + R)⁻¹`. No covariance maths per frame (v8 spends ~22k fuel on it).

**3. Controller (LQR): corrections become IMU inputs.**
- The estimated position, velocity and attitude errors are not jumped into the state.
- They are drained over the next frames by an accelerometer correction `u_a` and a gyro correction `u_w`.
- Gains come from a discrete Riccati equation with a 0.1 s closed-loop time constant.
- Bias errors are the integral states: they stay in the IMU correction from then on.

**4. Error signals.**
- **Vision:** the v8 tracker's rows, the nose-first cruise prior, and the vision stall detector
  (from `vio_robust`).
- **IMU only:**
  - the hover calibration over the whole 1 s start hover;
  - the cruise prior, one body axis per frame, on 3 frames of each window;
  - a **wind check**. The simulator has no drag, so a steady horizontal accelerometer reading during
    the hover means an external force (wind). The prior's expected forward speed shrinks with it:
    full below 0.8 m/s², zero above 2.6 m/s². In the episodes this force is 0.04–0.31 m/s² in calm
    air and 1.0–3.6 m/s² in wind, and above ~2.7 m/s² the drone does not move at all.

## Results

Mean unaligned ATE (m). Variants were chosen on dev and windy-dev only. Held-out sets were run once, at the end.

### Held-out (scenes 192–195, 16 episodes per set)

| Model | Normal | Windy | Hard | Extreme | Max fuel/frame |
|---|---:|---:|---:|---:|---:|
| Stay at start (outputs the first pose) | 2.532 | 1.562 | 2.733 | 2.234 | — |
| v8 (`base`) | 1.341 | 1.159 | 2.342 | 2.576 | 97.8k |
| `vio_robust` (EKF) | 1.193 | **1.071** | **2.003** | 2.581 | 97.9k |
| `vio_robust_lite` | 1.382 | 1.097 | 2.312 | 3.258 | 38.7k |
| `ultra` | 1.840 | 1.387 | 2.609 | 2.712 | 24.0k |
| **`lqg_vision`** | **1.196** | 1.438 | 4.369 | 12.06 | 84.0k |
| **`lqg_imu`** | 1.420 | 1.667 | 61.9 | 218.9 | **12.6k** |

### Dev (selection)

| Model | Dev | Windy-dev |
|---|---:|---:|
| Stay at start | 2.820 | 1.572 |
| v8 | 1.619 | 1.185 |
| `vio_robust` | 1.157 | 0.952 |
| `ultra` | 1.916 | 1.288 |
| `imu_cruise` (earlier IMU-only PI loop) | 14.74 | — |
| **`lqg_vision`** | 1.158 | 1.638 |
| **`lqg_imu`** | 1.382 | 1.555 |

### What this shows
- **Normal flights:** `lqg_vision` matches the EKF (1.196 vs 1.193 m) at 14% less fuel.
  `lqg_imu` beats ultra (1.42 vs 1.84 m) at half the fuel without looking at the camera.
  It is 10× better than the earlier IMU-only attempt, because the fixed Kalman gains also correct
  tilt and gyro bias from the prior.
- **Wind:** both are worse than the EKF. `lqg_imu` is about as good as staying at the start.
- **Hard / extreme (2–3× IMU noise and bias, faster autopilot):** both fail.
  - The gains were designed for the nominal noise and do not adapt.
  - `lqg_imu` diverges to hundreds of metres on 7 of 16 extreme flights; it has no divergence guard.
  - **Do not use these on flights with wind or several times the default IMU noise.**
- **Corner tracker vs full optical flow:** with the same tracker, the LQG matches the EKF on normal
  flights. The tracker is not what limits accuracy, so a dense-flow version was not built.

## Design findings (dev, unaligned ATE)

| Change | Result | Kept? |
|---|---|---|
| Gains designed for nose-first motion only | 3.24 m, some flights 600 m: wrong gains when the drone climbs | no: mixed directions (1.37 m) |
| Vision corrects only position / velocity (no tilt, bias) | 1.62 vs 1.37 m | no |
| LQR corrections vs jumping the correction in at once | 1.37 vs 1.41 m (within noise) | LQR |
| Thrust-axis accelerometer only (no drag) | fine in calm air, windy-dev 12.8 m: wind is a force the accelerometer measures | no |
| Full-hover calibration (0.97 s, accel bias) | vision 1.37 → 1.18 m | yes |
| IMU-only velocity prior on one axis per frame | max fuel 16.5k → 10.1k, same accuracy | yes |
| Wind-scaled prior | IMU-only windy-dev 5.16 → 1.62 m, vision 1.84 → 1.70 m | yes |
| Tilt-aware covariance mapping | neutral | yes (correct for tilted hover) |
| Gain scheduling (hover / moving designs, `LQG_SCHED=1`) | dev 1.145, windy-dev 1.80 (mixed) | no, left as a switch |
| LQG with the lite tracker | 1.89 m @ 37k (EKF lite: 1.51 m @ 38.7k) | no |

## Switches (all in `solution.cpp`)

| Switch | Effect |
|---|---|
| `LQG` | 0 = v8 EKF |
| `LQG_VISION` | 0 = IMU only |
| `LQR_TAU` | controller time constant (0 = apply corrections at once) |
| `LQG_MIX_DIRS` | design over mixed motion directions |
| `LQG_TILT_ROT` | map body blocks with the tilt |
| `LQG_WIND_HI` / `LQG_WIND_LO` | wind-scaled prior |
| `LQG_SCHED` | gain scheduling |
| `LQG_PRIOR_SPLIT` | one prior axis per frame (default on without vision) |
| `LQG_DIR_STATES` | translation-only vision gains |
| `THRUST_ONLY` | thrust-axis accelerometer only |
| `LQG_DESIGN_S`, `LQG_NOM_*` | nominal design flight |
| `LQG_SCHED_WIN` | start-up gain schedule: design covariances of the first N windows, starting from the initial priors |
| `LQG_ADAPT` | in-flight covariance scale per update type (prior 1), updated from normalized innovations |
| `LQG_STILL_CAL` | motion-gated calibration: hold the pose and average gyro bias / gravity until the first motion |
| `VPRIOR_REL` | zero-velocity prior whose sigma grows with speed |

The four adaptive switches (variants `lqga_*`) did not stop the fixed-gain filter from diverging on faster
flights; the EKF with the same settings (`ekf_move`) did not diverge. Fixed gains do not suit fast, changing motion.

Every tested configuration is a variant in `../variants.py` (`lqg_*`).
Reproduce the tables from `examples/python`:

```bash
$PY 5_novel_approaches/evaluate.py --variants base robust_soft lite_robust_soft ultra lqg_vision_final lqg_imu_final \
    --sets dev windy-dev heldout windy-heldout hard-heldout extreme-heldout
```

## Next steps, if pursued
1. **Noise-adaptive gains.** Scale the design covariance by a running innovation statistic, or keep
   designs for 1×/2×/3× noise and schedule between them. This targets the hard / extreme failures.
2. **Divergence guard for `lqg_imu`.** `VMAX_RESET=5`, as in `vio_robust`.

Both need selection on hard-dev / extreme-dev, not the held-out sets above.
