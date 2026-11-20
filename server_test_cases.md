# What we know about the stage-2 server test flights

EECS 116 Project 3 (viobench), notes as of Fri 11/20. This is everything we have learned about the hidden server set from the course announcement, our three uploads, and local experiments built to imitate it. Statements are marked **known** (stated by the course or read directly off the server), **measured** (from our own local experiments), or **inferred** (our best reading of the evidence).

## 1. What the course told us (known)

- 12 episodes, 1200 frames each (60 s at 20 Hz), the same 12 every upload. Each episode's ID on the results page is the first 12 characters of the SHA-1 of its archive. We cannot download them or recover them from the ID.
- "Same (visual) distribution, just more agile and (hopefully) less predictable." The generator code is not released.
- The metric is **mean unaligned ATE** (positions only, no Sim(3) alignment). Sim(3) ATE, median ATE and RPE are shown as diagnostics only.
- One upload per day per team; the score is the claimed area of the best single submission on a log-log plot of max fuel per frame against ATE.

## 2. Our server results (known)

Three uploads. Per-flight unaligned ATE (m), with Sim(3) ATE and 1 s rotation RPE (rad) in brackets.

| Episode | v2 tracker (Thu 00:15) | movestart_nn (Thu) | stage2_adapt (Fri) |
|---|---|---|---|
| 004c5894ac96 | 3.90 (2.22, 0.067) | 4.39 (3.13, 0.052) | 4.39 (3.13, 0.052) |
| 1c6e73f3323e | 1.24 (1.06, 0.020) | 2.56 (2.02, 0.027) | 2.56 (2.02, 0.027) |
| 21494468052b | 6.03 (3.28, 0.058) | 6.06 (3.16, 0.038) | 6.06 (3.16, 0.038) |
| 726458d9bc31 | 2.26 (0.65, 0.049) | 1.14 (0.83, 0.033) | 1.14 (0.83, 0.033) |
| 78ed493f5b51 | 4.23 (1.97, 0.085) | 5.82 (2.10, 0.061) | 5.81 (2.09, 0.061) |
| 8950399d97f5 | 1.43 (0.63, 0.039) | 1.52 (0.73, 0.027) | 1.52 (0.73, 0.027) |
| 89f8c99e0a08 | 3.92 (2.58, 0.105) | 2.65 (1.83, 0.053) | 2.65 (1.83, 0.053) |
| a61ad08d7b50 | 3.73 (2.26, 0.052) | 4.24 (2.48, 0.055) | 4.33 (2.23, 0.050) |
| cd6dc704267a | 4.44 (1.90, 0.044) | 4.59 (2.08, 0.053) | 4.58 (2.08, 0.053) |
| dd7a737e6813 | 4.95 (2.92, 0.045) | 3.02 (2.49, 0.046) | 3.04 (2.50, 0.046) |
| e7aa5aee7fb7 | 11.60 (2.37, 0.096) | 2.51 (2.18, 0.038) | 2.51 (2.18, 0.038) |
| f7c2546ff0f2 | 3.96 (1.36, 0.042) | 2.76 (1.70, 0.034) | 2.77 (1.79, 0.037) |
| **Mean** | **4.31** (1.93, 0.058) | **3.44** (2.06, 0.043) | **3.45** (2.05, 0.042) |
| Max fuel/frame | 86.5k | 88k | 87.8k |

Other team uploads (from `5_novel_approaches/README.md`): ultra-lite 5.25 m @ 24k, `vio_cruise` 5.88 m @ 97k, `vio_cruise_lite` 14.6 m @ 39k. On the stage-1 public set these models scored about 1.1-1.8 m, so every model loses a factor of 3 or more on the server.

## 3. What the server flights look like

### 3.1 They do not start from a calm hover (inferred, strong evidence)

- The default generator hovers for 1 s before the route starts (`viobench simulate --hover`, default 1.0). v2 calibrated its gyro bias and level attitude over that first second.
- v2's server RPE (0.058) is 4x our local hover-start value (0.015). Starting the same model 10 s into local flights reproduces 0.058 exactly; starting right after the hover gives 0.037.
- After we removed the start calibration (movestart_nn), the worst flight e7aa went from 11.6 m to 2.5 m and RPE dropped to 0.043.
- Only 1c6e looks like a normal hover start (v2 RPE 0.020, and it was v2's best flight). It got worse once we removed the calibration, which is the expected cost of that change on a hover start.

### 3.2 They are longer and faster than ours (inferred)

- Locally, final drift is about 1.2x the ATE. Using that ratio with the server's drift-per-metre column, the server flights cover roughly **40-140 m per minute** (median about 70 m). Our default and agile local flights cover 26-39 m.
- The server also reports **Velocity 0.761 m/s** for the Fri upload. This is most likely our velocity RMSE rather than the flights' speed.
- Rule of thumb: our error per metre on the server is similar to locally; the flights are longer, so absolute errors are larger.
- Being fast does not break the tracker on its own: 1c6e covers an estimated ~140 m and v2 scored 1.24 m on it.

### 3.3 The IMU white noise is normal (inferred from the Fri upload)

- stage2_adapt measures the IMU noise level online from second differences of consecutive samples (accurate to <1% locally) and only changes behaviour when the noise is above the default (0.07 rad/s gyro, 0.28 m/s² accel per sample).
- On the server it scored the same as movestart_nn on every flight (10 of 12 within 0.02 m). So it never saw elevated noise, and **the server IMU noise is at the default level**.

### 3.4 Rotation error is still about 2x ours, so the gyro bias is the main suspect (inferred, not yet confirmed)

- With the start fixed, server RPE is 0.042-0.043; on local moving-start flights it is about 0.02.
- Injecting errors into clean local flights (default noise): an extra **gyro bias of 2x** (0.04 rad/s) raises our ATE from 1.71 to 2.55 m and RPE to about 0.031. 2x accelerometer bias or 2x white noise does almost nothing.
- Our filter learns the tilt-axis gyro biases but barely learns the yaw-axis bias: it stays near zero for 30-40 s even on clean flights, because the camera rotation measurement is weak (0.01-0.1 rad per window against about 0.006 rad of bias effect per window).
- So the likeliest remaining difference is a **larger gyro bias** (and perhaps a different bias model), with normal white noise.

### 3.5 Wind and other effects (open)

- We have no direct evidence of wind. A world-frame horizontal acceleration state (to absorb a constant wind force) gave no gain on our windy local set.
- Not tested: camera effects (blur, exposure, rolling shutter), different scene catalogs, longer episodes (the server's are 60 s like ours).

## 4. Local sets that imitate the server (measured)

All generated with `data/agile_sim2.py` (autopilot constants, IMU noise/bias scale and wind injected into the simulator template, the same edit as `5_novel_approaches/data/simulate_hard.py`) and `--hover 0`.

| Set | Settings | movestart_nn | v2 | Zero output |
|---|---|---|---|---|
| s2_hard_noisy (32) | fast autopilot, IMU noise x2, bias x2, wind 1 N | 1.75 m | about 2.2 m | 2.0 m |
| s2_xfast_extreme (32) | fastest autopilot, IMU x3, bias x3, wind 2 N | 2.84 m | n/a | 2.2 m |
| s2_hssd_noisy (32) | larger HSSD houses, as hard_noisy | 2.06 m | n/a | 3.1 m |

On the first 24 flights of these sets, the ratio movestart_nn / v2 was 0.82, against 0.80 on the server, and the RPE matched (0.043). But since the server IMU noise turned out to be normal, these sets are not an exact match: their noise is too high. The next test set should use **default noise with 2-3x gyro bias**.

Autopilot tiers we used (CRUISE_SPEED_MIN, CRUISE_SPEED_MAX, MAX_ACCELERATION, MAX_JERK, MAX_YAW_RATE, PHASE_ACCELERATION, PHASE_JERK[, TRACKING_MARGIN]):

- default: 0.45 0.8 0.7 2.0 0.8 0.2 0.5
- med: 1.0 2.0 2.0 6.0 1.6 0.6 1.5
- hard: 2.0 3.5 4.0 12.0 2.5 1.2 3.0
- xfast: 2.5 4.0 6.0 20.0 3.0 2.0 5.0 0.5

Even the xfast tier only covers about 39 m per minute in ProcTHOR houses, about half the server's estimated distance.

## 5. What has been ruled out or did not transfer

- **Assuming a start hover** (any calibration window): very costly on the server.
- **Assuming the default autopilot's habits** (e.g. a 0.24 m/s nose-first cruise prior as in `vio_cruise` / `vio_robust`): it hurts fast flights badly (locally 1.50 -> 2.28 m on fast flights).
- **Elevated IMU white noise:** not present on the server (section 3.3).
- **Speed measurement bias:** our learned speed correction (`speednn.h`) is unbiased on the server-like sets.
- **Heading as the limit on noisy-IMU flights:** with the true gyro bias fed in (oracle), ATE on s2_hard_noisy did not improve, although the heading error halved. This may differ on the server, where noise is normal and the bias is likely larger.

## 6. Recommendations

1. Never assume a hover, a level start or a known speed. Initialise from the first frame with wide priors on velocity, tilt and gyro bias.
2. Make the gyro bias (especially yaw) learnable within a few seconds. The planned approach is to measure rotation from persistent feature tracks over about 2 s instead of 0.3 s, since the bias effect grows with time while the measurement error does not.
3. Judge every change on a local set with no start hover, a fast autopilot and an enlarged gyro bias, and check it does not regress on clean sets. Do not tune on the public set; it is one small, easy house.
4. Treat the per-flight server table as a paired comparison: the 12 flights are fixed, so a change is real only if most flights move the same way.

## 7. Where things are

- Our candidates: `data/dev/submit/` (stage2_tracker_v2, stage2_movestart_nn, stage2_adapt). Sources: `data/dev/stage2_movestart_nn/`, `data/dev/stage2_adapt/`.
- Generators: `data/agile_sim.py` / `agile.sh` (autopilot only), `data/agile_sim2.py`, `gen_s2like.sh` (test set), `gen_s2train.sh` (training set).
