# vins: sliding-window VIO after VINS-Mono

A tightly coupled sliding-window optimizer in the style of VINS-Mono (Qin, Li, Shen, T-RO 2018), written from scratch
in one file (`solution.cpp`, plus the relocalization map `vmap.inc`). The tracking front end (KLT tracks, sparse
corner detection) is the one of `../msckf`. Binary: `../wasm/vins.wasm` (the default build).

Every setting is a `-D` flag at the top of `solution.cpp` / `vmap.inc`; the defaults are the chosen configuration.
Three builds, one per fuel level (`PRESET`; single flags still override):

| Binary | Build | Window, tracks | Max fuel/frame | Test ATE (144) | Est. area alone |
|---|---|---|---:|---:|---:|
| `vins.wasm` | default | 11 frames, 32 tracks, 3 map searches, at most 7 observations per feature | 6.11M | 0.387 | 4.86% |
| `vins_lite.wasm` | `-DPRESET=1` | 6 frames, 24 tracks, `KF_PAR=0.07` | 3.20M | 0.461 | 4.77% |
| `vins_ultra.wasm` | `-DPRESET=2` | 5 frames, 20 tracks, 16 detection cells, 1 map search | 2.03M | 0.532 | 4.58% |
| `vins_pipe.wasm` | `-DPIPE=2 -DPIPE_SCHED=2 -DPIPE_MAX=3 -DPIPE_DET=1 -DPIPE_CAP=2.6e6` | as vins, solve every second frame, scheduled | 2.70M | 0.359 | 5.73% |
| `vins_pipe_fixed.wasm` | `-DPIPE=2` | as vins, solve every second frame, fixed half/half split | 3.77M | 0.364 | 5.43% |
| `vins_edge.wasm` | `-DEDGE=1 -DEDGE_WHEN=16` | vins + edgelets (line features) when corners are scarce | 6.46M | 0.327 | 5.32% |
| `vins_pipe_edge.wasm` | vins_pipe flags + `-DEDGE=1 -DEDGE_WHEN=16` | vins_pipe + edgelets | 2.79M | 0.336 | 5.94% |

router_dpn: 3.24M, 0.772, 3.05%. The three VINS points together: about 5.6% (union of the rectangles).
The lite and ultra settings were chosen on the selection flights (`KF_PAR`, tracks, detection cells, map searches).

The estimator is chaotic: a change that only reorders floating-point sums (the SIMD LK, for example, agrees with the
scalar LK to 1e-5 px on 99% of patches) moves the 144-flight mean by up to about 0.02 m either way. Differences of that
size between builds are not real.

## What it does (VINS-Mono parts)

- **Window**: up to `WIN` = 11 frames (keyframes + the newest frame). Each frame has attitude, position and velocity
  (9 error states). Gravity direction (2) and one gyro and one accel bias for the whole window (6) come first: 107 states.
- **IMU preintegration** between window frames, with bias Jacobians and covariance. When a frame is dropped, the two
  preintegrations are composed (exact to first order, no re-integration).
- **Features**: inverse depth anchored at the first observation in the window, Huber reprojection cost. Each feature's
  depth is eliminated (Schur complement) before the dense solve.
- **Solver**: one Gauss-Newton step per frame (`NIT`), dense Cholesky of the 107 x 107 system.
- **Two-way marginalization**: if the second-newest frame has little parallax to the previous keyframe (`KF_PAR`), it is
  dropped; otherwise the oldest frame is marginalized into a dense prior (Schur complement).
- **Relocalization** (`vmap.inc`, the role of VINS-Mono's loop closure): features that leave the window while their
  track lives become map points (3D position + 7x7 template). Each frame up to 3 visible map points are searched (NCC on
  a 2-px grid, then LK); a match is a reprojection factor of the fixed point on the newest frame.

## Changes from the paper

| Change | Why |
|---|---|
| No SfM initialization: priors on the first frame (velocity 1 m/s, tilt 0.15 rad, biases) | flights start at any state; the priors are enough |
| No relative Levenberg damping (`LM_REL=0`) | damping of 1e-4 x diagonal froze the weakly observed scale direction: the solution stayed in a wrong minimum with 30 m errors |
| Log-depth prior on every feature, ln(depth / 2.2 m) ~ N(0, 1) (`DPF_*`); untriangulated features start at 2.2 m | fixes scale on slow, steady flights (VINS-Mono's fixed INIT_DEPTH, made a weak prior) |
| Shared window biases with an information floor (`SHARED_BIAS`, `BIAS_FLOOR_*`) | same accuracy as per-frame biases, 107 instead of 167 states |
| Features marginalized once with the oldest frame, then restarted (`MARG_KEEP=0`) | VINS-Mono keeps their later observations too (double counting): +0.28 m |
| Strict outliers: a feature with mean reprojection error > 1.5 px is dropped and its track ended; Huber 1 sigma, sigma 0.7 px | looser settings were much worse |
| Variable projection: each depth refined (3 GN steps, poses fixed) before the joint step (`VARPRO`) | -0.05 m |
| Zero-velocity factor when the 90th percentile of the 10-frame track disparity is < 1.5 px and the gyro is quiet; such frames are kept as keyframes | the median disparity flagged slow flight along the optical axis as still (-0.03 m); a dropped frame would lose the factor |
| Weak velocity prior (sigma 1.2 x running RMS speed, in [0.8, 3] m/s, per 0.5 s of window); running speed updated only when >= 10 features were used (`V2_MINF`) | from `../msckf`; the gate stops a starved flight's drift from loosening its own prior |
| Divergence reset (speed > 6 m/s or a failed solve) with a visual velocity guess, capped at 2 x RMS speed | from `../msckf` |
| Map used only while the running RMS speed > 0.45 m/s (`MAP_VRUN`) | the map helps every agile set but hurt slow flights (+0.09 m) |

## Fuel

Every wasm instruction counts, including `local.get/set`, so a double multiply-add costs about 8 instructions even with
SIMD. The main fuel work:

- Cholesky with 4 rows per pass and constant-offset loads (`CHOL4`): ~4.7 instructions per multiply-add (2.4M -> 1.4M).
- Builders fill only the lower triangle; per-feature Hessian kept as anchor/diagonal blocks; compact 9-row IMU factor.
- The prior is stored as a lower triangle; the solver and prior buffers are swapped after marginalization (no copy).
- The worst frames were the first marginalizations (all initial features at once, plus map-point creation) and
  marginalization frames that also searched the map: at most 8 features (12 in lite/ultra) are marginalized per frame
  (`MARG_FMAX`), 6 map points made per frame, and the keyframe decision is taken before the solve so that a
  marginalizing frame skips the map search (`KF_PRE`, `MAP_SKIP_MARG`).
- Block-structured covariance step of the preintegration (`COV_FAST`, about 330 instead of 1460 multiply-adds per IMU
  sample) and SIMD LK (`LK_FAST`: the 8x9 green patch is extracted with one shuffle per row and the residuals run four
  columns at a time; the host build computes the same lanes in the same order, so replay stays bit-identical).
- Corner candidates on a 6-px lattice (`DET_STEP=6`; lite keeps 4).
- At most 7 observations per feature in the solve (`OBS_MAX`; anchor, newest and evenly spaced between). The worst frame
  was the first marginalization, where every feature had been tracked since the start (about 10 observations each; the
  feature cost grows with the square of that): 6.75M -> 6.11M, windy level 7.34M -> 6.40M. Lite and ultra windows are
  shorter than 7 frames, so they are unaffected.

Max fuel per frame, sandbox, 48 standard episodes (agile-hard-test, agile-xfast-test, heldout): vins 6.11M (mean 4.2M),
vins_lite 3.20M, vins_ultra 2.03M. The same flights cost router_dpn 3.24M.

## Pipelined solve (`PIPE`, 2026-11-21)

In the default build every frame runs the whole solve (build ~1.2M, dense 107x107 Cholesky ~1.9M, marginalization),
so the mean frame (4.9M) is close to the worst one (6.1M). `PIPE=2` runs the optimizer on every second frame and
spreads one solve over two frames; KLT tracking still runs on every frame:

- frame A: a new window frame (observations, stillness, triangulation, keyframe decision, map search), build, the
  first part of the Cholesky (left-looking, so it splits by columns);
- frame B: the rest of the Cholesky, the step, drop or marginalization, detection (no window observation; new tracks
  join the window on the next frame A), map points. Frame B outputs the solved frame A propagated with the IMU.

`PIPE_SCHED=2` replaces the fixed split by a scheduler. The cycle's work runs in small units: the prior, each IMU
factor, each feature, each Cholesky column, the step, and the drop/marginalization build and Schur rows in chunks.
Each unit has a cost estimate in fuel, fitted by regressing sandbox fuel on host-logged unit counts (`VINS_SCLOG`,
within ~0.1M per frame). Each frame first does its fixed work (tracking, detection, map search), measures it with
the same model, then runs the chain up to a common per-frame level; the last frame finishes the cycle. A cycle whose
level would exceed `PIPE_CAP` gets one more frame (`PIPE_MAX`). With `PIPE_DET=1` detection runs on frame A, where its
(variable) cost is known before the cycle is planned.

| Build | Worst frame (hard / windy / noisy / extreme dev) | Selection (187) | Test (144) | hard | windy | noisy | extreme dev / heldout |
|---|---|---:|---:|---:|---:|---:|---:|
| vins (default) | 6.11 / 6.40 / 6.15 / 6.22M | 0.390 | 0.387 | 0.606 | 0.392 | 0.790 | 0.799 / 0.819 |
| vins_pipe_fixed | 3.77 / 3.58 / 3.67 / 3.72M | 0.364 | 0.364 | 0.559 | 0.447 | 0.545 | 0.794 / 1.080 |
| vins_pipe | 2.70 / 2.67 / 2.69 / 2.70M | 0.402 | 0.359 | 0.567 | 0.449 | 0.611 | 1.075 / 1.229 |
| 3-frame cycle (`PIPE=3`, scheduled, `PIPE_MAX=4`) | 2.14 / 2.14 / 2.09 / 2.13M | 0.386 | 0.435 | 0.695 | 0.565 | 0.880 | 1.563 / - |

Estimated area alone: on the 144 test flights vins 4.82% (6.40M), vins_pipe_fixed 5.43%, vins_pipe 5.73%; on the hard
level 3.48 / 4.03 / 4.17%; on the extreme level (32 flights) 2.61 / 2.35 / 1.76%. Pipelining costs accuracy on the
fastest flights: at 10 Hz a short track lands in fewer window frames. The 3-frame cycle loses too much there.

The refactor for the scheduler (split Schur/marginalization functions, column-range Cholesky) leaves the default
build's results unchanged and makes it about 1% cheaper (rebuild of `vins.wasm`: windy 6.40M -> 6.34M).

## Edgelets: line features from PL-VINS / MLINE-VINS (`EDGE`, 2026-11-21)

Diagnosis: on the 144 test flights the quarter of flights that spend the most frames with fewer than 15 features in
the solve (31% of frames) have mean ATE 0.718, the other three quarters 0.24. Those frames look at rendered walls:
uniform grey with 4-level, noise-free, one-pixel steps at wall/ceiling/furniture edges and no corners. The detector's
pretest (6 levels) and corner threshold never consider them.

PL-VINS adds LSD line segments with Plücker-coordinate factors; MLINE-VINS tracks lines by optical flow instead of
detecting them every frame. `EDGE=1` takes the flow-tracked idea in its cheapest form, SVO-style edgelets: a point on
an edge (largest structure-tensor eigenvalue at least `EDGE_MIN`, ratio below `EDGE_RATIO`), detected in cells without
a corner (pretest down to `EDGE_PRETEST`), tracked by a 1-D LK along the edge normal (shift + brightness offset), with
the normal re-estimated from each new template. Its residual is the reprojection error projected on the observed
normal (one row; the Jacobian rows are projected the same way), so the inverse-depth Schur step, outlier test and
marginalization are unchanged. Edgelets start at the depth prior (no two-ray triangulation) and are left out of the
stillness test, visual velocity, keyframe parallax and map points. `EDGE_WHEN=16`: detected only while fewer than 16
corner tracks are live, so the busiest frames (32 corners) get no extra tracks. Starved frames: 8.8% -> 4.1%.

| Build | Test 144 | hard | windy | noisy | extreme dev / heldout | Worst frame |
|---|---:|---:|---:|---:|---:|---:|
| vins | 0.387 | 0.606 | 0.392 | 0.790 | 0.799 / 0.819 | 6.40M |
| vins_edge | 0.327 | 0.594 | 0.301 | 0.510 | 0.679 / - | 6.46M |
| vins_pipe | 0.359 | 0.567 | 0.449 | 0.611 | 1.075 / 1.229 | 2.74M |
| vins_pipe_edge | 0.336 | 0.528 | 0.287 | - | 0.881 / 0.924 | 2.79M |

Estimated area alone (test / hard / extreme flights): vins 4.82 / 3.48 / 2.61%, vins_edge 5.32 / 3.53 / 3.13%,
vins_pipe 5.73 / 4.17 / 1.76%, vins_pipe_edge 5.94 / 4.41 / 2.58%. Selection (187): vins_edge 0.363 vs 0.390 (EDGE_WHEN
chosen there: all edgelets 0.386, `EDGE_WHEN=24 EDGE_MAX=16` 0.394, `EDGE_MIN=0.6` 0.381).

Against the routers (router numbers from `../router/README.md`, same episodes; sandbox ATE, m):

| | router_lite2 | router_dpn | router_dp2n | vins_edge | vins_pipe_edge |
|---|---:|---:|---:|---:|---:|
| Test (144, host replay) | 0.820 | 0.772 | 0.765 | 0.327 | 0.336 |
| hard level, held-out | 2.481 | 1.951 | 1.906 | 0.571 | 0.586 |
| extreme level, held-out | 5.623 | 1.924 | 1.923 | 0.750 | 0.924 |
| windy level, held-out | 1.539 | 1.476 | 1.423 | 0.519 | 0.340 |
| noisy level, dev | 2.424 | 1.369 | 1.319 | 0.510 | 0.502 |
| Max fuel (standard / levels) | 3.27M / - | 3.24 / 3.46M | 3.24 / 3.46M | 6.46M | 2.79M |
| Est. area, test flights | 2.85% | 3.05% | 3.08% | 5.32% | 5.94% |

Other ideas from the requested papers:
- MLINE-VINS / StructVIO Manhattan constraints: rotation drift is small here (median 3 deg after 60 s, ~0.07 m of a
  0.5 m error); the Manhattan heading factor was already neutral (`MANHATTAN`).
- SuperVINS (SuperPoint + LightGlue): a CNN per frame is far beyond the fuel budget.
- Dynamic-VINS / DynaVINS: the scenes are static; the IMU-consistency idea as a pre-solve gate on new observations
  (`OBS_GATE` 3/4/8 px) was noise (+0.033 / -0.019 / +0.009).
- VINS-Mah (VINS-Fusion + a Mahalanobis gate on features before the solve, chi-square 5.99, with the pose covariance
  recovered from the optimization): `MAH=1`. Here the newest frame is the last block of the solve, so its marginal
  covariance is the inverse of L_NN L_NN^T (last 9x9 block of the Cholesky factor), propagated with the IMU to the new
  frame, plus inverse-depth and pixel noise. The gate rejects only ~0.1% of observations (tracker and the post-solve
  outlier test already remove the rest); selection +0.024 on vins_edge (`MAH_SIG` 1 or 2 px), +0.019 on vins_pipe_edge.
- A router between vins_edge (extreme flights) and vins_pipe_edge: the score's fuel is the worst frame over all
  flights, so any routed flight costs the router 6.46M; even a perfect router (5.32 / 3.88 / 3.13% on test / hard /
  extreme) is below the two builds submitted as separate points (6.02 / 4.41 / 3.44%).
- Faint corners when tracks are few (`MS_LOW_N=20 MIN_SCORE_LOW=0.3 PRETEST_LOW=2`): -0.02 (noise); starved frames
  8.8% -> 6.2% only, since the walls have edges rather than corners.

## ROVIO, VI-DSO and DM-VIO ideas (2026-11-21, all rejected)

Base: vins_pipe_edge; selection = 187 flights (dev, normal-dev2, train-*, bias3-xfast/xxfast, hssd3), host replay,
paired. Base selection mean 0.381 / median 0.283; one start-up blow-up (agile-hard-dev Train-64-0001, 7.4 m) carries
most mean differences, so medians and the other 171 flights decide. Every flag defaults off, and the four shipped
builds (vins, vins_lite, vins_edge, vins_pipe_edge) compile to byte-identical wasm.

Diagnosis first (`VINS_TRK=1`: track deaths at exit; `VINS_TRK2`: per-track residual log):
- Tracks live 0.5-1 s (11-20 frames). Agile flights: ~45% of deaths are LK residual > `MAX_RES`, ~20% outlier kills,
  ~10% leave the image. Gyro-bias flights: outlier kills dominate (1100-1400 per flight).
- Surviving tracks keep the same residual at every age (~3.4 grey levels from age 2 to 32): templates do not drift.
  65% of residual deaths are on the first step after detection (residual ~22). Those features are untrackable even
  from a good start (below), most likely silhouette corners and aliased one-pixel edges.
- Error builds up as drift (median 0.08 m at 5 s, 0.14 at 10 s, 0.37 at 60 s), not mostly at start-up.

| Idea (paper) | Flag | Selection mean / median (base 0.381 / 0.283) | Verdict |
|---|---|---:|---|
| Motion-aware LK start from neighbours' flow (DSO coarse tracking, ROVIO multilevel) | `NB_FLOW=1` / `2` (+retry) | 0.346 / 0.265, 0.348 / 0.269 | first-step deaths only -15%; gain is the one blow-up flight (other 171: -0.005) |
| Structure-tensor weighting of corners (ROVIO photometric innovation) | `ST_W=1`, `ST_G0` 8000 / 3000 | 0.351 / 0.292, 0.371 / 0.297 | worse without the blow-up (88/187 better) |
| Scale as an explicit state (VI-DSO): depth-prior centre = window state s | `SCALE_ST=1`, `S_SIG0`/`S_FLOOR` 0.5/0.1, 1.0/0.3, 0.3/0.05 | 0.420, 0.474, 0.375 / 0.284 | the freer s, the worse; the fixed room-depth prior carries scale the IMU cannot replace |
| Dynamic photometric weight (DM-VIO) | `VW_DYN` 0.7 / 1.0 / 1.5 | 0.458, 0.397, 0.386 / 0.310 | worse everywhere, most on gyro bias |
| Patch warping (ROVIO) | - | not built | no template drift to fix (see diagnosis) |
| Delayed marginalization for IMU init (DM-VIO) | - | not built | start-up is a small share of the error; SCALE_ST showed the IMU scale is weak |

The blow-up flight: velocity is ~3x too small during the first forward acceleration (0.4 vs 1.1 m/s at 3 s) while
the IMU cost climbs to 30. `DPF_Z0=4` fixes it (1.11 m) and `DPF_Z0=1.5` makes it 10.5 m: the room-depth prior sets
the scale there. Making that centre a state did not help (it moved the wrong way at the start).

Combined with router_dp's speed gate (`SCALE_ST=1 S_V0 S_V1 S_PIN`): ungated, the free scale state cost slow flights
(+0.14 m below 0.35 m/s) and helped fast ones (-0.05 to -0.08 m above 0.9 m/s), the same split router_dp found for
its depth prior. `S_V0=0.4 S_V1=0.5` pins s (sigma `S_PIN` 0.02, solves only) below 0.40 m/s running speed and frees
it above 0.50: selection -0.014 (median 0.281 vs 0.283), dev levels (sandbox, 64 flights) -0.038 (se 0.024, 36/64;
extreme -0.062, hard -0.080, windy -0.007, noisy -0.002), worst frame unchanged (2.76M). Scored once on the test
group: +0.010 (se 0.013, 77/144), so not adopted. Gate 0.45-0.60 and `NB_FLOW` on top were neutral on selection.
The slow flights' scale bias is small (median x1.04; per-flight spread x1.00-1.12), so a speed-dependent depth centre
would not help either.

## Results (unaligned ATE, m)

Selection (171 flights: dev, normal-dev2, train-*, 3x gyro bias, hssd3), host replay:

| Build | Mean | Median |
|---|---:|---:|
| router_dpn | 0.732 | 0.575 |
| vins | 0.386 | 0.322 |
| vins_lite | 0.414 | 0.324 |
| vins_ultra | 0.482 | 0.391 |

Test (144 flights, scored once per candidate; "better" = flights where vins beats router_dpn):

| Set | router_dpn | vins | vins_lite | vins_ultra | vins better on |
|---|---:|---:|---:|---:|---:|
| agile-hard-test | 1.069 | 0.294 | 0.430 | 0.514 | 14/16 |
| agile-xfast-test | 0.599 | 0.285 | 0.292 | 0.427 | 15/16 |
| fresh-hard | 0.874 | 0.344 | 0.335 | 0.478 | 14/16 |
| fresh-xfast | 0.655 | 0.299 | 0.614 | 0.528 | 15/16 |
| heldout (slow) | 1.272 | 0.604 | 0.666 | 0.706 | 14/16 |
| test2-hard | 0.778 | 0.598 | 0.626 | 0.723 | 26/32 |
| test2-xfast | 0.461 | 0.228 | 0.280 | 0.344 | 29/32 |
| **all 144** | **0.772** | **0.387** | **0.461** | **0.532** | **127/144** |

Earlier full builds scored 0.356 (7.87M) and 0.371 (6.75M) on the same flights; the differences are within the chaotic
spread described above.

Simulator difficulty levels (dev, 16 flights each, sandbox, final binaries; router_dpn from the same episodes):

| Level | router_dpn | vins | vins_lite | vins_ultra |
|---|---:|---:|---:|---:|
| hard | 1.455 | 0.606 | 0.661 | 0.710 |
| noisy | 1.369 | 0.790 | 0.570 | 0.740 |
| extreme | | 0.799 | 0.811 | 0.913 |
| windy | | 0.392 | 0.402 | 0.471 |
| max fuel on these levels | 3.46M (hard, noisy) | 6.40M | 3.35M | 2.23M |

The windy level has the heaviest frames; with these maxima the area estimates drop by about
0.05 points each (union 5.5%).

**Fresh-house check (run once, after all choices).** 44 new flights in ProcTHOR houses no set or training run had used
(normal 34-37, agile-hard 44-47, hard level 54-55, noisy level 56-57; 4 flights per house, 2 for houses 47 and 56 because
of disk space), official sandbox. Binaries from before the last fuel work (same accuracy in expectation):

| Flights | router_dpn | vins | vins_lite | vins_ultra | vins better on |
|---|---:|---:|---:|---:|---:|
| normal, 16 | 1.087 | 0.451 | 0.484 | 0.560 | 16/16 |
| agile-hard, 14 | 0.657 | 0.191 | 0.281 | 0.363 | 14/14 |
| hard level, 8 | 1.198 | 0.461 | 0.657 | 0.628 | 7/8 |
| noisy level, 6 | 1.369 | 0.587 | 0.778 | 0.716 | 6/6 |
| **all 44** | **1.009** | **0.389** | **0.491** | **0.531** | **43/44** |

Estimated claimed area (same log-log rectangle as `../router/README.md`): vins 4.9%, vins_lite 4.8%, vins_ultra 4.6%,
all three together 5.6%, against 3.05% for router_dpn.

## Build and evaluate

```bash
./build_wasm.sh $PWD/../wasm/vins.wasm [-D...]          # absolute output path (the script cd's here)
./build_host.sh NAME [-D...]                            # /tmp/learned-<uid>/bin/host_vins_NAME
python3 ../hosttools/hosteval.py /tmp/learned-$(id -u)/bin/host_vins_NAME --ref ... --sets dev normal-dev2
python ../msckf/sandbox_eval.py ../wasm/vins.wasm EP.zip ...   # fuel
```

Host debug switches (environment): `VINS_DEBUG=N` (state every N frames), `VINS_RESETS`, `VINS_FS` (per-frame window
events), `VINS_MAPSTATS`, `VINS_GCHECK=frame` (numerical gradient check; `VINS_JAC` for the IMU Jacobian),
`VINS_SCLOG` (`PIPE_SCHED`: per-frame fixed-work counts and estimated units per chain step, for refitting `CU_*`).
`VINS_TRK` (track deaths by cause at exit), `VINS_TRK2` (per-track LK residuals), `VINS_ST` (template structure
tensors), `VINS_SST` (`SCALE_ST` state), `VINS_VW` (RMS reprojection error per solve).

## Tried and rejected

| Idea | Result |
|---|---|
| Blend with learned_lin by speed (`../router_vins`) | slow flights worse; the gentlest blend is neutral |
| Online IMU noise estimate (`IMU_EST`, from `../msckf`) | hard 0.57 -> 0.89, noisy 0.76 -> 1.09 |
| Depth-prior centre adapted from well-triangulated features (`DPF_ADAPT`) | +0.03 to +0.08 (circular) |
| Chord iterations (reuse the factorization, `CHORD`) | diverge |
| Single-precision Cholesky + refinement (`CHOL_F32`) | diverges (ill-conditioned) |
| Two iterations on frames that do not marginalize (`NIT2_LIGHT`) | +0.016 |
| Observation sigma from the LK residual (`LK_R0`) | +0.01 to +0.07 |
| Cap on features per solve (`OPT_FMAX`), 4-6 observations per feature, `VARPRO` 0/1 | worse, no fuel gain at the worst frame |
| Keep an outlier's track (`OUT_KEEP_TRACK`) | +0.14 |
| Window 8/9/10 or 14/17 | smaller is worse (WIN 9: +0.025), larger no better |
| Map without the speed gate, or gated at 0.3 m/s | slow flights +0.05 to +0.09 |
| Fewer detection cells, 4 LK iterations, 24 tracks, other front-end settings | worse or neutral |
| Marginalization reusing the solve's feature blocks (`MARG_CACHE`) | higher max fuel (the cache is filled on the same frame) |
| Marginalize only ended tracks (`MARG_LIVE=0`) | +0.08 |
| Map points only from well-triangulated features (`MAP_MIN_PAR`) | slow flights still worse with the map |
| No detection on marginalizing frames (`DEFER_MARG`) | starves detection in steady flight (every frame marginalizes) |
| Two iterations for lite/ultra; smaller bias floors; other IMU noise levels | neutral or worse |
| Manhattan-world heading factor on the last keyframe, slow flights only (`MANHATTAN`, from `../msckf`) | +0.026 (either sign worse; weaker sigma neutral) |
| Fewer detection cells on marginalizing frames (`DET_MARG_CELLS`) | +0.06 to +0.10 (starves tracking in steady flight) |
| Lower corner threshold or looser outlier gate when few features (`MS_LOW_N`, `OUT_LOW_N`) | neutral or worse |
| Pipelined solve over 3 frames (`PIPE=3`, fixed split or scheduled) | 2.1M worst frame but test 0.435, extreme 1.56 |
| `PIPE` with `KF_MAXGAP 20`, detection on frame A without the scheduler, no map search on marginalizing cycles | +0.01 to +0.04 (noise level) |
| Scheduler with detection spread over the cycle (`PIPE_DET=3`) or on the last frame | higher worst frame (2.6M): detection cost not known when planning |
| Lower `PIPE_CAP` (1.4-1.7M) or `PIPE_MAX` 5 | worst frame no lower (model error), more slow cycles |
| IMU-consistency gate on new observations (`OBS_GATE`, DynaVINS idea) | noise |
| Faint corners when few tracks (`PRETEST_LOW`, `MIN_SCORE_LOW` 0.3-1) | -0.01 to -0.02 (noise) |
| VINS-Mah Mahalanobis gate before the solve (`MAH`) | +0.02 (rejects ~0.1% of observations) |
| VINS-Mah gate doing the work of the post-solve test (`MAH=1` with `OUT_PX` 2.5-3, `MAX_RES` 20) | +0.11 to +0.18 (looser `OUT_PX` alone: +0.08) |
| LK started at the IMU-predicted projection of the triangulated point (`MAH_PRED=1`) | +0.28: wrong predictions feed back (agile-hard 2.6 m) |
| Second LK start at that prediction, keep the better fit (`MAH_PRED=2`) | +0.02 to +0.03 (median better, mean not) |
