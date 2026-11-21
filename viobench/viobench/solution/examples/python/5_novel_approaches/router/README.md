# router: MSCKF + splat map and learned_lin, blended by flight regime

The models are good in different regimes:

- **MSCKF + Gaussian-splat map** (`../msckf`, `-DSPLAT=1`): far better on agile flights.
- **learned_lin**: competitive on slow, normal-difficulty flights that start with a hover.

On slow flights the two models' errors are fairly independent, so averaging them beats either one. The router runs
both every frame. learned_lin costs about 0.09M fuel per frame, against about 5.5M for the MSCKF.

## How it decides

- **Regime signal:** the MSCKF's own running speed, as an exponential mean with a 10 s time constant. It starts at
  0.7 m/s, so every flight begins in MSCKF mode.
- **Weight on the MSCKF:**
  - speed >= 0.45 m/s: w = 1, the MSCKF alone;
  - speed <= 0.30 m/s: w = 0.5, an equal average;
  - in between: linear.
- **Output:** each frame, position = previous output + w x (MSCKF step) + (1 - w) x (learned_lin step). Increments
  are blended, never positions, so the output cannot jump when the regime changes. Attitude and twist are the
  MSCKF's.

Parameters: `RT_LO`, `RT_HI`, `RT_TAU`, `RT_V0`, `RT_WMIN` (compile flags).

## Selection results (host replay, paired against MSCKF + splat alone)

| Set | Flights | MSCKF + splat | router | Difference (se) | Better on |
|---|---:|---:|---:|---:|---:|
| normal-dev | 20 | 1.250 | 1.165 | -0.085 (0.029) | 17/20 |
| normal-dev2 (new houses 112-119, 164-179) | 16 | 1.058 | 0.969 | -0.089 (0.042) | 9/16 |
| agile dev and training sets | 108 | | identical | 0.000 | |

On agile flights the router never leaves MSCKF mode. learned_lin on normal-dev scores 1.371, so the router also beats
it on the normal flights.

How the rule was chosen: offline, from saved trajectories of both models on the selection flights. Picking the better
model for each flight with hindsight would give 0.644 m overall, against 0.728 m for MSCKF + splat; the router reaches
0.706 m. A fixed 50/50 blend gives 1.08 m: averaging hurts agile flights. Using learned_lin + splat as the partner
was no better (0.708 m), for three times the extra fuel.

## Build

```bash
./build_host.sh NAME [-DRT_...]      # /tmp/learned-<uid>/bin/host_msckf_NAME (host replay, no fuel)
./build_wasm.sh OUT.wasm [-DRT_...]  # three translation units (msckf_part, lin_part, router) linked into one reactor
```

Each model is compiled from its own unchanged source in its own translation unit, with its entry points renamed
(`msckf_init`/`msckf_step`, `lin_init`/`lin_step`). `router.cpp` exports the real `init`/`step`.

Sandbox check (official runner, 32 flights: agile-hard-test and heldout): mean ATE 1.078, the same as the host
replay; max fuel 5,563,214 per frame.

## router_lite: the same router at about 2/3 of the fuel (`../wasm/router_lite.wasm`)

Build: `./build_wasm.sh OUT.wasm -DPR=3 -DMAX_ROWS=20 -DDETECT_CELLS=16`. Changes to the MSCKF part:
- 7x7 KLT patches instead of 9x9;
- at most 20 EKF rows per frame instead of 24;
- at most 16 detection cells scored per frame.

| | router | router_lite |
|---|---:|---:|
| Max fuel per frame, sandbox, 48 episodes | 5.77M | **3.89M** (-33%) |
| Mean fuel per frame | 2.2M | 1.7M |
| Test, 144 flights | 0.815 m | 0.831 m |
| Normal held-out | 1.452 m | 1.639 m |

On the 144 test flights the difference is +0.016 m (se 0.053), better on 64/144. Against v9_adapted, router_lite is
-1.149 m (se 0.092), better on 128/144.

Where the fuel goes, at the router's 20 heaviest frames:

| Component | Fuel |
|---|---:|
| Corner detection | ~1.7M |
| EKF update | ~1.4M |
| KLT tracking | ~0.75M |
| Templates | ~0.23M |

Heavy frames are ones where many tracks ended at once: a full update plus many cells to score.

Tried for fuel, measured paired on 144 selection flights against the router:

| Change | Accuracy | Fuel | Kept? |
|---|---:|---|---|
| Cap detection at 24 / 16 / 12 cells per frame | +0.11 / +0.03 / +0.12 m | | 16 only, inside router_lite |
| Cap rows at 20 / 16 per frame | +0.07 / +0.20 m | | 20 only, inside router_lite |
| Load balancing: detect less on frames with a heavy update | +0.05 to +0.06 m, consistently | | no |
| Earlier LK stop (0.05 / 0.1 px) | neutral | mean -1 to -5%, max unchanged | no |
| Single-precision update intermediates | +0.05 m | no gain | no |
| Fewer LK iterations | +0.09 m | | no |

The compiler already vectorizes most of the update, LK and detection loops, so hand-written SIMD would gain little.

### Making router_lite smaller (tried, not adopted)

Each change was added on top of router_lite. Score = an estimate of claimed area on the log-log fuel/ATE reference
rectangle: (log10 of the right edge of the claimed region - log10 max fuel) x log10(reference ATE / ATE), divided by
the rectangle's area. It uses selection ATE on 144 flights and max fuel over 6 sandbox episodes. Only the relative
values between rows matter.

| Change | Selection ATE | Max fuel (6 episodes) |
|---|---:|---:|
| router_lite | 0.888 | 3.59M |
| 10 / 9 clones | 0.898 / 1.047 | 3.36M / 3.32M |
| 28 / 24 features | 0.876 / 0.900 | 3.37M / 3.17M |
| 12 detection cells | 0.895 | 3.26M |
| 16 rows | 0.918 | 3.28M |
| 4 observations per track | 0.871 | 3.48M |
| no splat map | 0.926 | 3.22M |
| 5 LK iterations | 0.889 | 3.34M |
| combinations of 3-5 of these | 0.94-1.06 | 2.9-3.4M |
| 12 detection cells + 5 LK iterations | 0.885 | 3.20M |

On the area estimate, every single change trades about evenly against router_lite, and combinations are clearly
worse because the accuracy losses compound.

The three best candidates were then checked on the 144 test flights and on 48 sandbox episodes:

| Change | Test ATE | Difference vs router_lite (se) | Max fuel (48 episodes) |
|---|---:|---:|---:|
| router_lite | 0.831 | | 3.89M |
| 12 detection cells + 5 LK iterations | 0.893 | +0.062 (0.051) | 3.61M |
| 28 features | 0.849 | +0.018 (0.038) | 3.71M |
| 4 observations per track | 0.858 | +0.027 (0.055) | 4.02M |

None claims more area than router_lite, which stays the recommended build. Moving the score substantially would need
a much larger structural fuel cut, not another parameter.

## router_lite2: same accuracy, 16% less fuel (`../wasm/router_lite2.wasm`)

Build:

```bash
./build_wasm.sh OUT.wasm -DPR=3 -DMAX_ROWS=20 -DDET_SPARSE=1 -DSPL_SKIP_ROWS=16 -DVMAX=6.0 \
    -DVP_ADAPT=1 -DVP_K=1.2 -DVP_MIN=0.8 -DVPRIOR_SIG=3.0 -DSPL_GROW=1 -DSPL_SMAX_LATE=2.0 -DRT_WMAX=0.95
```

Changes against router_lite:

- **Sparse corner scoring (`DET_SPARSE`).** The Shi-Tomasi score comes from 9 gradient samples 2 px apart instead of a
  dense 5x5 window, with no per-pixel product pass. Detection is about 4x cheaper per cell, so the 16-cell cap is
  dropped. On its own: selection -0.006 m (se 0.047), max fuel 3.89M to 3.45M.
- **No splat search on heavy frames (`SPL_SKIP_ROWS=16`).** The worst frames are those where a full update, many
  detection cells, all tracks and a splat search coincide. The splat search is skipped when the update used more than
  16 rows, and retried later. On its own: selection -0.015 m (se 0.013), max fuel 3.45M to 3.20M.
- **Speed-aware safeguards.** The divergence limit goes from 3 to 6 m/s; this is bit-identical locally, because no
  local flight reaches 3 m/s. The weak velocity prior's sigma becomes 1.2 x the recent RMS speed, with a floor of
  0.8 m/s: unchanged for slow flights, looser for fast ones. Selection -0.001 m overall, -0.13 m on 3x-gyro-bias
  flights.
- **Growing splat map (`SPL_GROW=1`).** New splats are added where few splats project into view, and the stalest slot
  is recycled. It helps most on the faster flights.
- **5% learned_lin on fast flights too (`RT_WMAX=0.95`).** The two estimators' errors are partly independent, so even
  a small share helps: selection -0.012 m (se 0.004, 113/176 better), test -0.016 m (se 0.005, 82/144 better). A plain
  3% shrink of the MSCKF's steps is worse (+0.008 m), so the gain is averaging, not shrinkage. 10% is about the same;
  20% is worse.

Results, paired against router_lite:

| | Flights | Difference (se) | Better on |
|---|---:|---:|---:|
| Selection (dev, training, normal-dev2, 3x-gyro-bias sets) | 176 | -0.039 m (0.041) | 90/176 |
| Fast flights with 3x gyro bias only | 32 | -0.143 m (0.073) | 20/32 |
| Test, scored once (without the 5% blend) | 144 | +0.004 m (0.060) | 74/144 |
| Test, final build with the 5% blend | 144 | -0.011 m (0.058), mean 0.820 m | 71/144 |

Max fuel (sandbox, 48 episodes): 3.27M, against 3.89M for router_lite.

Tried and rejected in this round:

| Change | Result |
|---|---|
| Coarse LK level for fast flights | +0.21 m on the fast flights |
| Load balancing detection on heavy frames | +0.05 m |
| Single-precision update | +0.05 m, no fuel gain |
| Manhattan-world heading measurement (`MANHATTAN=1`) | the dominant-axis angle from per-pixel gradients is too noisy on agile flights: +0.18 m |
| LK start from the filter's predicted translation at the mean depth (`LK_TPRED`) | +0.31 m: velocity errors feed back into tracking |
| Pixel noise 1.25 / 1.5 px; corner threshold 6 / 15; LK residual threshold 10 / 18 | +0.07 to +0.16 m |
| Cap the MSCKF weight for 20-40 frames after a divergence reset | -0.003 to +0.05 m |
| Soft reset (inflate velocity covariance only); stricter or more eager reset triggers | +0.04 to +0.19 m |
| Lower corner threshold only when fewer than 16-20 tracks are active | +0.06 to +0.13 m |

### Scheduling work so it does not coincide (tried, not adopted)

A linear cost model fits router_lite2's per-frame fuel well (R^2 0.96): about 0.33M + 67k per update row + 44k per
detection cell scored + 18k per track + 166k per splat search. The heaviest frames have about 16 rows, 15 cells and 31
tracks at once. They are frames where many tracks were lost: the lost tracks become update rows, and their cells need
refilling, in the same frame.

Measured paired on 187 selection flights against router_lite2; max fuel from the sandbox (48 episodes; router_lite2:
3.27M):

| Change | Accuracy | Max fuel |
|---|---:|---:|
| Budget scheduler, detection capped by the budget (`SCHED=1`, 2.0-2.6M) | +0.08 to +0.17 m | |
| Budget scheduler, update rows deferred, detection reserved (`SCHED=2`, 2.4M) | +0.02 m | 3.34M: the reserve underestimated refills |
| The same with detection also capped | +0.11 m | |
| Larger detection reserve, so more rows deferred (`SCHED=3`) | +0.12 to +0.26 m | |
| Staggered track lengths 10-11 frames (`STAGGER=1`); 8-11 frames | +0.03 m; +0.14 to +0.16 m | 3.29M |
| Coarser corner lattice only on big refill frames (`DET_REFILL_N=8`) | +0.07 m | 3.12M |
| Lost tracks' rows one frame later (`LOST_DEFER=1`) | +0.05 m | 3.21M |
| Shared gradient grid for sparse detection (`DET_SPARSE=2`) | identical | 3.81M (slower in wasm) |

Every way of delaying work, whether new features, update rows or shorter tracks, costs accuracy roughly in
proportion to how often it delays. The few-percent fuel savings never pay for that on the area score.

## router_dp: router_lite2 plus a depth-based scale prior on slow flights (`../wasm/router_dp.wasm`)

Build:

```bash
./build_wasm.sh OUT.wasm -DPR=3 -DMAX_ROWS=20 -DDET_SPARSE=1 -DSPL_SKIP_ROWS=16 -DVMAX=6.0 \
    -DVP_ADAPT=1 -DVP_K=1.2 -DVP_MIN=0.8 -DVPRIOR_SIG=3.0 -DSPL_GROW=1 -DSPL_SMAX_LATE=2.0 -DRT_WMAX=0.95 \
    -DDP=1 -DDP_Z0=1.8 -DDP_SIG=0.6 -DDP_RUN=0.40
```

**Why.** On slow flights most of the error is scale, not heading. In slow cruise the accelerations are too small for
the IMU to fix the MSCKF's metric scale: its cruise speed is right on average but ranges from 0.63x to 1.5x of the
truth across flights. Indoor rooms have a consistent depth, so the depth of triangulated features carries scale
information. The median true depth of freshly triangulated features is 1.8 m on the normal flights and 2.2 m on the
agile ones; per flight it ranges from 1.2 to 2.9 m (window to window, log-sd about 0.4).

**How.** Every 10 frames, the median anchor depth D of the features triangulated since the last row gives one EKF
row on the speed along the current velocity: speed x 1.8 / D, with a relative sigma of 0.6. It needs at least 6
triangulated features and a speed over 0.1 m/s. It is only applied while the flight's running speed (the 10 s mean
the velocity prior already keeps) is below 0.40 m/s, so agile flights are unchanged. Ungated, the prior helps slow
flights as much but hurts agile flights by +0.08 to +0.28 m: there the IMU already gives good scale.

| Paired against router_lite2 | Flights | Difference (se) | Better on |
|---|---:|---:|---:|
| Selection (dev, training, normal-dev2, 3x-gyro-bias sets) | 132 | -0.045 m (0.029) | 41/132 (the rest unchanged) |
| Selection, normal flights only | 68 | -0.130 / -0.069 m (normal-dev / normal-dev2) | 41/68 |
| Neighbouring settings (sigma 0.5 / 0.8, depth 1.6 m, gate 0.45 m/s, every 5 frames) | 132 | -0.027 to -0.066 m | |
| Test, scored once | 144 | -0.016 m (0.019), mean 0.804 m | 14/144 |
| Test, normal held-out | 16 | -0.135 m (0.176) | 11/16 |

Max fuel (sandbox, 48 episodes): 3.27M, the same as router_lite2. Sandbox ATE equals the host replay. On the area
estimate it is about 2% better than router_lite2.

### router_dp2: learned_lin blended up to 0.65 m/s (`../wasm/router_dp2.wasm`, add `-DRT_HI=0.65`)

With the depth prior in place, blending in learned_lin over a wider speed range helps. The MSCKF weight still goes
from 0.5 at 0.30 m/s, but now reaches 0.95 at 0.65 m/s instead of 0.45 m/s.

| Upper threshold (paired against router_dp, 132 selection flights) | Difference (se) | Better on |
|---|---:|---:|
| 0.40 / 0.35 (lower thresholds 0.25 / 0.20) | +0.008 / +0.010 m | |
| 0.50 / 0.55 | -0.014 / -0.023 m (0.006) | 46 / 59 |
| **0.65** | **-0.034 m (0.010)** | **82** |
| 0.80 | -0.024 m: agile flights +0.01 to +0.07 m | 66 |

Test, scored once: 0.797 m, -0.007 m (se 0.006) against router_dp and -0.023 m (se 0.019) against router_lite2.
Normal held-out 1.164 m (router_lite2 1.407), but the agile test sets are +0.01 to +0.03 m worse. Max fuel 3.27M.
It leans more on learned_lin at medium speeds, so it is the riskier choice on flights unlike the local ones.

Tried in the same round, not adopted:

| Change | Result |
|---|---|
| IMU bias noise matched to the simulator (gyro bias walk 6e-4, accel walk 1e-3, accel turn-on 0.2) | +0.02 to +0.17 m |
| Rotation-only rows for low-parallax tracks during hover (`LOWPAR`, `LP_BASE`, `LP_ZV`) | fixes the gyro-bias estimate at takeoff and halves heading error at 15-30 s, but ATE -0.02 to +0.10 m: heading error barely affects position |
| Stricter or softer zero-velocity tests (`ZV_WIN` 1.0 / 1.2, `ZV_SOFT`, `ZW_MIN_SPAN` 10) | +0.02 to +0.09 m, although about 20 zero-velocity updates per slow flight fire at over 0.1 m/s |
| Depth prior gated by its own running speed starting at 0 | hurts agile flights again (+0.04 to +0.32 m on agile sets) |
| Router step cap at 3 / 4 / 5 m/s (`RT_VCAP`), or learned_lin's step when the MSCKF jumps (`RT_CAPLIN`) | +0.02 to +0.19 m: large MSCKF jumps are mostly corrections toward the truth |
| Router slow-flight weight 0.4 / 0.6 / 0.7 with the depth prior | within 0.005 m |
| Depth prior also below 0.1 m/s filter speed (`DP_VMIN` 0.03 / 0.05), or only above 0.15 | -0.002 to +0.04 m |
| Veto the zero-velocity update when the IMU-only velocity change over 0.5-1 s is large (`ZV_ACC`) | +0.11 to +0.22 m: too noisy to tell takeoff from hover |
| More clones (13, 15), 7 observations per track, 24 rows, 40 features, 9x9 patches | +0.09 to +0.37 m on selection |

## router_dpn / router_dp2n: robustness additions (`../wasm/router_dpn.wasm`, `../wasm/router_dp2n.wasm`)

Build: router_dp's (or router_dp2's) flags plus `-DIMU_EST=1 -DIMU_EST_TH=1.5 -DIMU_EST_BIAS=1 -DRESET_VCAP_K=2`.

**Online IMU noise level (`IMU_EST`, `IMU_EST_TH`, `IMU_EST_BIAS`).** With 2-3x the nominal IMU noise the MSCKF
trusts the IMU too much and degrades badly. At 200 Hz the true signal barely changes between samples, so the variance
of successive-sample differences is twice the white-noise variance; on the local flights the estimate is exact
(accelerometer 0.278 vs 0.28 m/s^2, gyro 0.070 vs 0.07 rad/s). It starts from the nominal value with a 200-sample
prior weight, is clamped to 0.5-5x nominal, and is only used when it exceeds 1.5x nominal: without that threshold the
tiny estimation differences perturbed nominal flights chaotically (+0.05 m selection, +0.16 m test). After 100
samples, if the IMU is k times noisier, the initial bias sigmas are widened by k too (in these levels bias and noise
scale together). On nominal flights all of this is bit-identical to router_dp.

**Cap on the velocity guess after a divergence reset (`RESET_VCAP_K=2`).** The reset restarts the velocity from a
visual guess. On one fast flight that guess was 5.4 m/s against a true 0.2-0.5 m/s, under the 6 m/s divergence limit,
and the position ran away by 17 m. The guess is now capped at max(1 m/s, 2 x the flight's RMS speed). It changes
only a few flights: selection -0.015 m (5 of 176 flights changed), test 8 flights better and 1 worse (a flight that
diverges early either way).

**Results.** Official sandbox ATE (m); 16 flights per level; the hard levels use 2-3x IMU noise and bias, wind and
faster flight (see `../data/simulate_hard.py`):

| | router_lite2 | router_dp | router_dpn | router_dp2n |
|---|---:|---:|---:|---:|
| Test (144 flights, host replay) | 0.820 | 0.804 | **0.772** | **0.765** |
| Selection (176 flights) | 0.814 | 0.771 | 0.756 | 0.739 |
| noisy level (dev) | 2.424 | 2.483 | 1.369 | 1.319 |
| fast level (dev) | 0.708 | 0.859 | 0.700 | 0.725 |
| faster level (dev) | 0.532 | | 0.527 | 0.557 |
| **hard level, held-out** | 2.481 | | 1.951 | 1.906 |
| **extreme level, held-out** | 5.623 | | 1.924 | 1.923 |
| **windy level, held-out** | 1.539 | | 1.476 | 1.423 |
| Max fuel, 48 standard episodes | 3.27M | 3.27M | 3.27M | 3.27M |

The adaptation settings were chosen on the dev levels and the selection flights; the held-out levels and the test
flights were run once per candidate. Estimated claimed area: router_dpn about 3.05%, router_dp2n about 3.08%, against
2.85% for router_lite2. router_dp2n leans more on learned_lin at medium speeds (slightly worse on the fast levels and
the agile test sets, better on slow and windy flights).

Also tried in this round, not adopted:

| Change | Result |
|---|---|
| Vision every second frame only (`VIS_EVERY=2`; first step to splitting work over two frames) | 0.77 -> 1.70 m |
| Tighter splat gate (chi^2 9.21 / 5.99) or a cap on how far one splat match may move the position (`SPL_DXMAX`) | -0.00 to +0.01 m: fixes one failure, causes another |
| Average of several near-identical variants (ensemble) | at most -0.006 m, for 2x fuel |
| Cap the MSCKF weight while it has few tracks (`RT_LOWTR`) | -0.002 to +0.05 m |
| IMU noise estimate without the threshold | see above |
| Depth prior also off above 0.6 / 0.8 m/s instantaneous speed (`DP_VMAX`) | did not fix the fast-level regression (the reset cap did) |
