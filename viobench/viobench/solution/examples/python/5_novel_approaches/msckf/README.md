# msckf: an OpenVINS-style MSCKF for viobench

A from-scratch C++ Multi-State Constraint Kalman Filter, the estimator at the core of OpenVINS, made compute-aware for
the fuel limit. It does not share code with v8/v9 or learned_lin. It gets metric scale from the IMU and multi-view
geometry instead of an assumed room depth. Binary: `../wasm/msckf.wasm`.

## Results (unaligned ATE, mean RMSE in m, paired per flight against learned_lin)

Evaluation flights: the 112 flights learned_lin never saw (its dev/test/fresh agile sets and the normal held-out set).
Nothing here was tuned on them. Tuning used learned_lin's own training flights (`tune` in `../hosttools/hosteval.py`)
and 20 normal-difficulty dev flights (ProcTHOR 60-79).

| Set | Flights | learned_lin | msckf | Difference (se) | msckf better on |
|---|---:|---:|---:|---:|---:|
| agile-hard-dev | 16 | 1.831 | **0.768** | -1.06 (0.28) | 15/16 |
| agile-xfast-dev | 16 | 1.903 | **0.602** | -1.30 (0.23) | 15/16 |
| agile-hard-test | 16 | 1.872 | **0.749** | -1.12 (0.25) | 15/16 |
| agile-xfast-test | 16 | 2.078 | **0.661** | -1.42 (0.29) | 15/16 |
| fresh-hard | 16 | 2.156 | **0.820** | -1.34 (0.25) | 15/16 |
| fresh-xfast | 16 | 1.760 | **1.230** | -0.53 (0.32) | 13/16 |
| heldout (normal, start hover) | 16 | **1.234** | 1.775 | +0.54 (0.19) | 4/16 |
| **All agile** | 96 | 1.933 | **0.805** | **-1.13 (0.11)** | 88/96 |
| **All 112** | 112 | 1.833 | **0.943** | **-0.89 (0.11)** | 92/112 |
| normal-dev (tuning) | 20 | 1.371 | 1.277 | -0.09 (0.17) | 11/20 |

Medians: 0.54 m vs 1.64 m over all 112 flights. 8 of the 112 flights are more than 1 m worse than learned_lin.

Official sandbox (fuel metered) on agile-hard-test, agile-xfast-test and heldout reproduces the host numbers exactly.

| Sandbox, 16 flights each | agile-hard-test | agile-xfast-test | heldout |
|---|---:|---:|---:|
| msckf unaligned ATE | 0.749 | 0.661 | 1.775 |
| msckf max fuel per frame | 6,235,141 | 5,983,256 | 5,834,899 |
| msckf mean fuel per frame | 2.40M | 2.42M | 2.08M |

learned_lin's max fuel is 88.6k per frame, so msckf costs about 70 times more per frame.

**Weak spots:**
- Slow, normal-difficulty flights that start with a hover. Slow motion gives little IMU excitation and short
  baselines, and the hover-to-flight transition is hard for the zero-velocity logic. msckf is worse than learned_lin
  on the held-out set (+0.54 m) and about equal on the dev set (-0.09 m).
- A robustness check on the older harder dev levels (all start with a hover; checked once, not tuned on), unaligned
  ATE in m, 16 flights each:

  | Level | learned_lin | msckf |
  |---|---:|---:|
  | noisy (x3 IMU noise) | 1.463 | 2.135 |
  | hard | 1.673 | 1.766 |
  | windy | 0.905 | 1.764 (se of difference 0.59) |

  The filter assumes the default IMU noise and has no wind model.

## How it works

**State.** Error-state EKF in the first IMU frame, so the output needs no alignment.
- IMU part (18 states): rotation, position, velocity, gyro bias, accelerometer bias, and the gravity vector (3 states,
  magnitude renormalized). Initial gravity comes from the first accelerometer readings, with a 0.15 rad tilt sigma.
- 11 pose clones (6 states each), one per frame, sliding window of 0.55 s. Up to 84 states in total.

**Propagation.** The 10 IMU samples per frame are integrated, and the transition matrix is accumulated in closed form
per block. The covariance is propagated using Phi's block structure (identity plus a few 3x3 blocks), about 10x
cheaper than a dense multiply and exactly the same result.

**Front end.**
- Up to 32 KLT tracks on the green channel, on an 8x6 grid with at most one feature per cell.
- Templates: 9x9, sub-pixel bilinear. LK translation plus brightness offset, up to 6 iterations, starting from a
  gyro-derotated prediction plus the track's last motion.
- Detection: Shi-Tomasi minimum eigenvalue on a 4-px lattice in empty cells, with gradient products computed once per
  cell. Blank cells are skipped by a cheap sparse pre-test and rested for one frame.

**MSCKF update (as in OpenVINS).**
- A track is used once: when it is lost, or when it spans the whole window.
- Up to 5 evenly spaced observations per track.
- Anchored inverse-depth Gauss-Newton triangulation. The track is rejected if it has under 0.02 rad of parallax or
  any reprojection error above 2 px.
- Reprojection Jacobians are projected onto the left null space of the feature Jacobian (Householder), so landmarks
  never enter the state.
- Rows of all tracks are stacked into one update, capped at 24 rows per frame. Tracks that do not fit wait in a queue
  for the next frame and are re-linearized there, so per-frame cost stays flat without discarding information.
- The update is written in row (axpy) form (HP = H P, G = L^-1 HP, P -= G^T G), which clang vectorizes to wasm SIMD.

**Additions needed for these flights** (each is a compile flag, see the top of `solution.cpp`):
- *Zero-velocity update* (OpenVINS has one too). Applied when the median feature disparity over the whole window is
  below 1.5 px (rotation-compensated with the filter's clone rotations), with sigma 0.15 m/s. Per-frame flow cannot
  tell hover wobble from slow flight; the window disparity can.
- *Weak velocity prior*: zero velocity with sigma 0.8 m/s, once every 10 frames. It bounds IMU drift when features are
  scarce and was the largest single gain (tuning-set mean 1.01 -> 0.65 m). Fast flights still gained.
- *Divergence recovery*: if almost no tracks triangulate consistently for 10 frames (low-parallax failures do not
  count), or the speed exceeds 3 m/s, the window is dropped. Velocity restarts from a visual guess: direction from
  the epipolar normals of the derotated flow, magnitude from the flow at the average triangulated depth.

## Files and commands

| File | Use |
|---|---|
| `solution.cpp` | the whole estimator (single file, ~1200 lines, every parameter overridable with `-D`) |
| `build_wasm.sh OUT.wasm [-D...]` | wasm build (wasi-sdk of `../evaluate.py`, `-O3 -msimd128`) |
| `build_host.sh NAME [-D...]` | host replay build `/tmp/learned-<uid>/bin/host_msckf_NAME` (no fuel; same results as the wasm) |
| `sandbox_eval.py WASM EP.zip ...` | official runner: unaligned ATE, max/p99/mean fuel per frame, per-frame fuel saved |
| `CMakeLists.txt` | for `viobench bundle msckf -o msckf.wasm` |
| `../hosttools/hosteval.py BIN... --sets eval/tune/normal-dev` | paired host evaluation vs learned_lin, cached |

Host debug output (host builds only): `MSCKF_STATS=1` (track/triangulation/reset counts), `MSCKF_DEBUG=1`
(`MSCKF_EVERY=n`), `MSCKF_RESETS=1`.

## What did not help

- Integer-pixel templates; half-resolution detection; detecting in only a few cells per frame; long cell cooldowns.
  All lose accuracy: the filter needs good, well-spread features.
- A per-feature chi-square gate never rejected anything, because tracks are triangulated with the same poses. It is
  off (`CHI2_ON`).
- Fewer clones (8-9), or 16-20 rows per frame: worse.
- Allowing zero-velocity updates only at the start: worse; the default autopilot does slow down in flight.
- IMU noise settings x0.5 / x2: within noise.

## SLAM landmarks in the state (tried, not adopted)

`-DSLAM_N=4` keeps up to 4 long-lived features in the EKF state (3 states each, between the IMU states and the clones),
as OpenVINS does. A feature is promoted after `SLAM_MINOBS` observations using the delayed initialization of
OpenVINS (the 3 Householder rows give its covariance and cross-covariance), then updated with 2 rows per frame while
its KLT track lives. Default `SLAM_N=0` is bit-identical to router_lite2.

Paired on 84 selection flights against router_lite2's MSCKF part (0.643 m):

| Variant | Difference |
|---|---:|
| Promote any landmark, update every frame | +1.20 m |
| Update every 5 / 10 frames | +0.82 / +0.69 m |
| Pixel sigma 5 px | +1.12 m |
| Promote only, never update | +0.04 m |
| Promote only if position sigma < 0.3 m | +0.23 m |
| Promote only if position sigma < 0.15 m | -0.008 m (almost nothing promoted) |
| Anchored templates (needed for long tracks) alone / with SLAM | +0.08 / +1.25 m |

Residuals stay sub-pixel and pass the chi-square gate, so the damage is not outliers. Landmarks enter with 2-3 m
position sigma, are linearized at poor positions, and long KLT tracks drift slowly in a correlated way that the
filter reads as precise information. In 0.55 s windows indoors, features rarely live long enough to add anything
the MSCKF rows do not already give.
