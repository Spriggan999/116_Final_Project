# 5_novel_approaches: estimators for unaligned ATE

After the stage-1 presentations, the course metric changed to **unaligned ATE**.
Sim(3) alignment no longer hides scale and heading errors. This directory builds on
the C++ estimator in `../2_deterministic/solution.cpp` (v8, plus its lite build).
That source is copied to `estimator/solution.cpp`, and every new behaviour sits
behind a compile-time flag whose default reproduces v8 exactly. Inference is C++
compiled to WASM. Python is used only offline, for data generation, analysis and
training.

## New models (2026-11-20): MSCKF, Gaussian-splat map, router

Final comparison on 144 test flights; unaligned ATE in m, host replay. The official sandbox gives the same numbers
on every set it was run on.

- **Agile (128):** agile-hard/xfast-test, fresh-hard/xfast (houses 120-137), and test2-hard/xfast (houses 80-95,
  generated after all model choices, untouched).
- **Normal held-out (16):** start hover, slow flights.
- **Selection** used only the dev and training flights.

| Model | Agile 128 | Normal held-out 16 | All 144 (mean / median) | Better than v9_adapted on | Max fuel/frame |
|---|---:|---:|---:|---:|---:|
| v9_adapted | 2.072 | 1.246 | 1.980 / 1.830 | | 88.6k |
| learned_lin | 1.858 | **1.234** | 1.788 / 1.614 | | 88.6k |
| learned_lin + splat (`splat.wasm`) | 1.580 | 1.207 | 1.539 / 1.398 | | 274k |
| OpenVINS-style MSCKF core only | 77.99 (a few flights diverge) | 2.512 | 69.6 / 0.638 | | ~5.4M |
| + divergence recovery | 1.481 | 2.174 | 1.558 / 0.686 | | ~5.4M |
| + weak velocity prior (`msckf_v2.wasm`) | 0.820 | 1.775 | 0.926 / 0.533 | | 5.37M |
| + Gaussian-splat map (`msckf_splat_v2.wasm`) | 0.735 | 1.625 | 0.834 / 0.434 | | 5.72M |
| **router** (`router.wasm`: msckf_splat + learned_lin by speed regime) | **0.735** | 1.452 | **0.815 / 0.434** | **127/144** | 5.77M |
| router_lite (`router_lite.wasm`: 7x7 patches, 20 rows, 16 detection cells) | 0.730 | 1.639 | 0.831 / 0.528 | 128/144 | 3.89M |

Paired comparisons for the router:

| Compared with | Difference over all 144 | se | Better on | Normal held-out | Agile flights |
|---|---:|---:|---:|---:|---|
| v9_adapted | -1.165 m | 0.097 | 127/144 | +0.21 m (8/16 better) | -1.1 to -1.7 m on every set |
| msckf_splat | -0.019 m | 0.006 | | -0.173 m (se 0.040, 15/16 better) | identical |

Fuel is the official-sandbox maximum per frame over 48 episodes (agile-hard-test, agile-xfast-test, heldout). The
MSCKF models are about 60 times more expensive per frame than learned_lin and v9_adapted.

Details: `msckf/README.md`, `msckf_splat/README.md`, `splat/README.md`, `router/README.md`. Paired host evaluation:
`hosttools/hosteval.py`.

Tried and not adopted:
- First-estimate Jacobians; motion keyframes; a two-rate window; anchored KLT; growing the splat map; rotation-only
  rows from low-parallax tracks; a percentile stillness test; a speed-adaptive velocity prior.
- IMU input filters (trapezoidal, low-pass) and image smoothing before tracking: neutral or worse.

## What to submit

| Binary (`wasm/`) | Use | Max fuel/frame |
|---|---|---:|
| **`vio_robust.wasm`** | Recommended full point: beats or ties v8 on every held-out difficulty level | 97.9k |
| **`vio_robust_lite.wasm`** | Recommended lite point: better than v8 lite except under ×3 IMU noise | 38.7k |
| `vio_cruise.wasm`, `vio_cruise_lite.wasm` | Earlier version. Best on calm flights, but **fails when wind stalls the drone**. Do not submit against the harder set | 97.8k / 38.7k |

Each file is the exact binary that produced the numbers below (verified by
SHA-256 against the evaluation build). To copy them to a laptop:

```bash
scp "eecs116-aaq@c204-17.eecs.berkeley.edu:/home/cc/eecs116/fa26/class/eecs116-aaq/Documents/project_3/viobench/viobench/solution/examples/python/5_novel_approaches/wasm/vio_robust*.wasm" ~/Downloads/
```

## Server results so far

Course server, harder evaluation set (12 episodes):

| Submitted | Model | Mean ATE | Max fuel/frame |
|---|---|---:|---:|
| 11/15 | "trained on unseen data" (ultra-lite, `2_deterministic/solution_ultra.cpp`) | 5.25 m | 24.0k |
| 11/18 | `vio_cruise` | 5.88 m | 96.8k |
| 11/18 | `vio_cruise_lite` | 14.6 m | 38.7k |

The cruise models were tuned for the default generator and lost on this set. The
robust models are the response; they have not been submitted yet.

## How models are evaluated

Every score here is mean unaligned ATE RMSE over complete 60-second episodes,
produced by `evaluate.py` with the official runner and sandbox (fuel metered).
To avoid the overfitting seen in other teams' stage-1 work:

- **Public (10):** the published flights, which v8 was originally tuned on. Reported, never used to choose.
- **Dev:** ProcTHOR scenes 60–79 (normal difficulty) and 60–63 (harder levels). All variant selection happens here.
- **Held-out:** ProcTHOR scenes 192–199 (normal) and 192–195 (harder levels). Final numbers only; nothing was trained on or chosen with them.
- **Training flights for the velocity prior:** collector recordings, scenes 20–59.

Normal-difficulty episodes come from `viobench simulate` with the public-set
settings. Harder levels come from `data/simulate_hard.py`, described below. The
server's `private` episodes are listed in `viobench/episodes` but cannot be
downloaded (no public index lists their hashes).

## Results

### Normal difficulty

| Model | Held-out (16) | Dev (15) | Public (10) | Held-out Sim(3) | Max fuel |
|---|---:|---:|---:|---:|---:|
| v8 (`2_deterministic`) | 1.341 m | 1.647 m | 1.356 m | 0.720 m | 97.8k |
| `vio_cruise` | **1.077 m** | **1.002 m** | 1.326 m | **0.550 m** | 97.8k |
| `vio_robust` | 1.193 m | 1.157 m | — | 0.591 m | 97.9k |
| v8 lite | 1.704 m | 2.051 m | 2.159 m | 0.972 m | 38.7k |
| `vio_cruise_lite` | 1.242 m | 1.021 m | 1.766 m | 0.642 m | 38.7k |
| `vio_robust_lite` | 1.382 m | 1.514 m | — | 0.683 m | 38.7k |
| ultra (`solution_ultra.cpp`) | 1.840 m | 1.916 m | — | 0.959 m | 24.0k |

v8 does worse on scenes it was not tuned on (dev 1.65 m vs public 1.36 m), so the
public set alone understates every improvement.

### Harder flights: dev levels (selection)

Unaligned ATE (m) on the dev scenes; level definitions are below.

| Model | Normal | Windy | Extreme | Hard | HSSD | HSSD hard | Noisy | Fast | Mean | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v8 | 1.619 | 1.185 | 2.406 | 2.249 | 1.003 | 1.579 | 2.088 | 2.032 | 1.770 | 2.406 |
| `vio_cruise` | 1.048 | 3.067 | 2.313 | 1.640 | — | — | 1.248 | 1.286 | — | 3.067 |
| **`vio_robust`** | 1.157 | 0.952 | 1.931 | 1.878 | 0.942 | 1.489 | 1.381 | 1.600 | **1.416** | **1.931** |
| v8 lite | 1.973 | 1.458 | 2.312 | 2.791 | 1.281 | 1.641 | 2.265 | 2.192 | 1.989 | 2.791 |
| **`vio_robust_lite`** | 1.514 | 1.229 | 2.193 | 2.296 | 1.025 | 1.583 | 2.019 | 2.064 | **1.740** | **2.296** |
| ultra | 1.916 | 1.288 | 2.560 | 2.492 | 1.536 | 1.830 | 2.145 | 2.317 | 2.010 | 2.560 |

### Harder flights: held-out levels (final)

Scenes 192–195, a fresh seed; never used for any choice.

| Model | Normal | Windy | Extreme | Hard | Mean | Max fuel |
|---|---:|---:|---:|---:|---:|---:|
| v8 | 1.341 | 1.159 | 2.576 | 2.342 | 1.855 | 97.8k |
| `vio_cruise` | **1.077** | 2.375 | 2.638 | 2.023 | 2.028 | 97.8k |
| **`vio_robust`** | 1.193 | **1.071** | 2.581 | **2.003** | **1.712 (−8%)** | 97.9k |
| v8 lite | 1.704 | 1.327 | 2.660 | 2.747 | 2.110 | 38.7k |
| **`vio_robust_lite`** | 1.382 | 1.097 | 3.258 | 2.312 | 2.012 (−5%) | 38.7k |
| ultra | 1.840 | 1.387 | 2.712 | 2.609 | 2.137 | 24.0k |

- **`vio_robust`** beats or ties v8 on every held-out level. It fixes the wind failure of `vio_cruise` (2.38 → 1.07 m) at a modest cost on calm flights (1.08 → 1.19 m).
- **`vio_robust_lite`** is better than v8 lite on three of four levels. On `extreme` (×3 IMU noise and bias), 2 of 16 episodes go wrong (8.6 m and 7.7 m).
- **Gap to the server:** the hardest synthetic level gives about 2.6 m, against 5–15 m on the server. Something about the server set is harder than anything reproduced here.

## What the models do

Both recommended models are v8 with three additions. All are compile flags in `estimator/solution.cpp`.

### 1. A nose-first cruise velocity prior (`VEL_PRIOR=2`, `VP_ATTITUDE=1`)

v8 applies a weak zero-velocity prior (0 ± 0.5 m/s) once per ~0.45 s window. That
prior is replaced with the autopilot's habit, stated in the **body frame**:

- **Mean:** 0.241 m/s forward, nothing sideways, no climb.
- **Spreads:** 0.17, 0.05 and 0.17 m/s, measured on training flights (scenes 20–49).
- **Inflation ×1.5,** chosen on dev. The prior is reapplied every window, but its error persists for seconds, so without inflation the filter over-trusts it.
- **Attitude Jacobian:** each row measures body-frame velocity, which depends on attitude as well as velocity: d(axis·v)/dθ = axis × v. Without that term, a heading error drags the velocity the wrong way (public 1.231 → 1.156 m when added).

### 2. A sustained-stall detector (`VP_STALL_PX=0.5`, `VP_STALL_TAU=8`, `VP_STALL_START=0`, `VP_STILL_SIG=0.25`)

Wind can stall the autopilot: the drone hovers near its start for the whole flight.
The detector keeps a running mean of the vision update's translational parallax over
about 8 windows (~3.5 s); only windows with enough tracks update it.

- The cruise prior applies only after vision has shown sustained translation.
- Otherwise a still prior applies, 0 ± 0.25 m/s on each axis.
- It starts in "still" mode, which is correct for the 1 s start hover.

### 3. A realistic divergence guard (`VMAX_RESET=5`)

The filter restarts when velocity exceeds 5 m/s; v8 waited for 50 m/s. Real flights
stay under ~1.5 m/s.

Fuel is unchanged, since all three replace or extend existing per-window steps with
a few multiply-adds.

## Findings behind the changes

### Where v8's error comes from (`analysis/oracle.py`)

The analysis swaps one ground-truth component into each 1-second step of v8's own
public trajectories and re-chains the path:

- **Orientation is already good:** true orientation only takes 1.43 → 1.36 m.
- **Translation carries the error:** true speed gives 1.08 m and true direction 1.04 m.
- **It is not a constant scale bias:** neither a global nor a per-episode scale factor helps.
- **Speed is per-window noise:** v8 gets speed from parallax with a fixed 8 m depth. Its error is a factor of ~1.8 and decorrelates within 1–2 s, so it is not slow drift, and an online depth-scale state would not help.

### What the IMU and flights reveal (`analysis/imu_structure.py`, `analysis/episode_imu.py`)

- **No rotor drag in the simulator.** Horizontal specific force is uncorrelated with body velocity, so the IMU cannot measure velocity directly; any IMU speed model is learning autopilot habits.
- **Strong habits at default settings:** nose-first flight (body-x carries 0.83 of the velocity direction), sideways speed ~0.03–0.05 m/s, mean speed ~0.26–0.30 m/s, top yaw rate 0.665 rad/s.
- **Fixed sensor model:** white noise 0.28 m/s² and 0.070 rad/s in every episode. Gyro bias ~0.01–0.02 rad/s, nearly constant per flight; accelerometer bias ~0.1 m/s².
- **The 1 s start hover sags and wobbles.** The accelerometer reads ~7.7 m/s² at first and the drone drifts up to 8 cm/s. Averaging the gyro over 0–0.5 s (v8) leaves a bias error about as large as the bias (0.019 vs 0.021 rad/s); 0–0.9 s halves it. In the full estimator this did not help robustly (see below). The hover length is not passed to solutions.

### Harder flights (`data/simulate_hard.py`, `analysis/flight_statistics.py`)

| Level | Cruise (m/s) | Accel / jerk / yaw | IMU noise and bias | Wind (N, std) | Measured mean speed | Sideways speed |
|---|---|---|---|---|---:|---:|
| normal | 0.45–0.8 | 0.7 / 2 / 0.8 | ×1 | 0 | 0.26 m/s | 0.025 m/s |
| `fast` | 0.8–1.5 | 1.5 / 4 / 1.2 | ×1 | 0 | — | — |
| `faster` | 1.5–3.0 | 3 / 8 / 1.5 | ×1 | 0 | 0.52 m/s | 0.23 m/s |
| `noisy` | default | default | ×2 | 1 | — | — |
| `windy` | default | default | ×1 | 3 | 0.10 m/s | 0.03 m/s |
| `hard` | 1.0–2.0 | 2 / 5 / 1.5 | ×2 | 1 | 0.45 m/s | 0.16 m/s |
| `extreme` | 1.5–3.0 | 3 / 8 / 1.5 | ×3 | 2 | — | — |
| HSSD (default / hard) | as above | | | | | |

- **Speed settings barely change flights:** indoor routes and autopilot pacing cap actual speed (even `faster` averages 0.52 m/s). Fast flight is not what breaks the estimators, and a wider patch search does not help.
- **Wind is the failure mode.** In 7 of 16 windy dev flights the drone travels under 0.5 m in 60 s. The ungated cruise prior invents ~14 m of motion (windy 3.07 m vs 1.19 m for v8), and the lite build fails worse. This matches the server pattern.
- **HSSD's larger houses are not the problem:** every model scores about as on ProcTHOR (~1.0 m).

## Tried and not kept

All of these are in `variants.py`.

| Idea | Result | Why dropped |
|---|---|---|
| Accelerometer sets scale (`imu_scale*`) | Public 3.1–4.5 m | IMU integration alone cannot hold velocity |
| Learned velocity prior: MLP on 2 s of IMU, C++ inference (`VEL_PRIOR=1`) | Offline velocity RMSE 0.18 vs 0.24 m/s, but worse ATE (dev 1.06 vs 0.90 m) and 359k fuel | Better offline fit did not help in the filter; 3.7× fuel |
| No-sideslip-only prior | Worse than v8 | The forward-speed term carries the gain |
| Full-hover calibration (`CALIB_SECONDS=0.9`, `CALIB_ACCEL_BIAS=1`) | Wash across sets | Not robust |
| IMU-only cruise estimator (`imu_cruise/`, ~6k fuel) | 5.5–47 m | Without vision, errors accumulate |
| Per-window motion gate (`VP_GATE_PX`) | Fixes wind, loses most of the calm-flight gain | Too twitchy |
| Stall detector starting "moving", or treating track loss as still | Invents motion at the start, or hurts calm flights | Superseded by the still-start detector |
| Innovation-adaptive prior spread (`VP_ADAPT_R`) | No change | The prior was never detectably inconsistent |
| Adaptive cruise speed (`VP_ADAPT_MEAN`) | Slightly worse | — |
| Wider LK search (`CROP_MARGIN=6`, `MAX_CORR_PX=6`) | Helps calm dev, not harder levels; +28% fuel | Not robust |
| Shorter vision window (`KF_LEN=4`), doubled filter noise | Worse | — |

## Files

| Path | Contents |
|---|---|
| `estimator/solution.cpp` | v8 copy with all new flags; defaults reproduce v8 |
| `estimator/velocity_prior_weights.{h,json}` | Learned-prior weights (experimental `VEL_PRIOR=1` only) |
| `imu_cruise/solution.cpp` | IMU-only experiment |
| `lqg/` | LQG observer + LQR IMU-correction controller: `lqg/vision` (with v8's tracker) and `lqg/imu` (IMU only); see `lqg/README.md` |
| `variants.py` | Every variant (source plus `-D` flags); recommended names `vio_robust`, `vio_robust_lite` at the end |
| `evaluate.py` | Builds variants, runs episodes in parallel through the official runner, caches per episode, prints tables |
| `data/generate.py` | Collector training flights (raw 200 Hz IMU and state) for the velocity prior |
| `data/simulate.py` | Benchmark-format episodes from `viobench simulate` (held-out and dev splits) |
| `data/simulate_hard.py` | Harder levels via a modified world template (viobench untouched) |
| `analysis/oracle.py` | Swap-in-truth error decomposition |
| `analysis/imu_structure.py`, `analysis/episode_imu.py` | IMU bias, noise, hover and flight-habit analysis |
| `analysis/flight_statistics.py` | Per-set flight statistics (speed, wind drift, travel, noise) |
| `velocity_prior/` | Dataset and training/export for the learned prior |
| `wasm/` | Submission binaries |

## Reproduce

Run from `examples/python` with `myenv`. Data and results go to
`/var/tmp/novel-approaches-<uid>` (about 20 GB), because the home quota is small.

```bash
PY=/home/cc/eecs116/fa26/class/eecs116-aaq/.conda/envs/myenv/bin/python
# data (GPU)
$PY 5_novel_approaches/data/generate.py --scenes 40                       # collector flights, scenes 20-59
$PY 5_novel_approaches/data/simulate.py --split heldout                    # normal held-out, scenes 192-199
$PY 5_novel_approaches/data/simulate.py --split dev --episodes 1 --seed 909
for L in hard fast faster noisy windy extreme; do $PY 5_novel_approaches/data/simulate_hard.py --level $L --split dev; done
for L in windy extreme hard; do $PY 5_novel_approaches/data/simulate_hard.py --level $L --split heldout --seed 2626; done
$PY 5_novel_approaches/data/simulate_hard.py --level default --split hssd-dev
# score (cached per episode; sets are folder names under episodes/, or 'public')
$PY 5_novel_approaches/evaluate.py --variants base vio_robust lite_base vio_robust_lite ultra \
    --sets heldout windy-heldout extreme-heldout hard-heldout
# analyses
$PY 5_novel_approaches/analysis/oracle.py /var/tmp/novel-approaches-$(id -u)/evaluation/base/public
$PY 5_novel_approaches/analysis/flight_statistics.py windy-dev
$PY 5_novel_approaches/analysis/episode_imu.py --set heldout
# learned prior (experimental)
(cd 5_novel_approaches/velocity_prior && $PY train.py)
```

## Operational notes

- **Generation needs the GPU** and the OptiX renderer build under `/var/tmp/gated-updates-<uid>/hyperlux`. The scripts reuse it and put the conda `bin` on `PATH`; without that, hyperlux tries to build WEBGPU and fails.
- **Running long jobs:** use `systemd-run --user --scope tmux -L persistent new-session -s <name>`. Even that server was killed once when the last login closed (`Linger=no`). After reconnecting, check `tmux -L persistent ls` and rerun any missing level; finished episodes are kept.
- **Workers:** `evaluate.py --workers N` runs N episodes in parallel. Evaluations slow down concurrent generation, so let generation finish first.

## Caveats and open questions

- **The cruise prior encodes the generator's autopilot:** forward speed, nose-first flight, no sideslip. If the hidden set uses a different controller or pacing, re-measure the three means and spreads from new flights.
- **The server set is still harder than anything reproduced here** (5–15 m vs ≤ 2.6 m). Plausible untested differences include stronger or gusting wind, starts in flight, longer episodes, camera effects (blur, exposure, rolling shutter) and other scene catalogs. The robust models should be submitted to find out.
- **Sample size:** 16 episodes per level. Differences under ~5% between close variants are within noise.
