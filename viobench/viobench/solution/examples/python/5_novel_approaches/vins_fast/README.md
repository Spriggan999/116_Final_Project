# vins_fast: vins_pipe_edge with ~38% less worst-frame fuel

**Server (uploaded Sat 11/21): 12/12 flights, mean ATE 0.300 m, Sim(3) 0.166 m, max fuel 1.7M (p95 1.5M).**

Built on Yash's `../vins` (vins_pipe_edge); his folder is unchanged. Same estimator; the changes are fuel engineering and scheduling.

## Results (local, unaligned ATE / max fuel of any frame)
| Build | 32 fast flights | 38 normal flights | all 70 |
|---|---|---|---|
| vins_pipe_edge (original) | 0.436 m @ 2.77M | 0.659 m @ 2.73M | 0.557 m @ 2.77M |
| exact-only build (same outputs as original) | 0.436 m @ 1.89M | 0.659 m @ 1.93M | 0.557 m @ 1.93M |
| **vins_fast_x155.wasm** (this folder, uploaded) | **0.385 m @ 1.71M** | **0.618 m @ 1.71M** | **0.511 m @ 1.71M** |
ATE differences of about ±0.03 between configurations are noise (chaotic trajectories); the fuel cut is real.

## Key fact about fuel
Fuel = 1 per wasm instruction, *including* local.get/set, i32.const and address adds; bulk memory ops (memory.fill/copy)
cost about 1 per byte. So the wins come from fewer stack-machine instructions per useful operation.

## Changes (all `#define` switches in solution.cpp; defaults = this build)
Bit-identical (same outputs as the original):
- Compiler flags: `-mllvm -disable-lsr` (loads use constant offsets), `-mllvm -unroll-threshold=3000`, `-mno-bulk-memory -mno-bulk-memory-opt`.
  Alone: -17% mean fuel.
- MEM_OWN: own SIMD memset/memcpy/memmove (bulk memory ops cost ~1 fuel/byte).
- CHOL8: Cholesky columns 8 rows per pass, 8 elements per iteration (~3.1 fuel per multiply-add, was ~4.3).
- BASE_FUSE, NO_UPPER0: prior copy + gradient in one SIMD pass; no zeroing of the unused upper triangle.
- FEAT_FAST: feature 6x6 blocks in SIMD. SCHUR_FAST / SCATTER_FAST (earlier), SIMD Schur compaction.
- DET_FAST: corner pretest and scores from exact integer SIMD sums (load8_lane gathers).
- TPL_FAST 2: SIMD template stores; LK reuses the converted patch while the integer position is unchanged.
Same maths, not bit-identical:
- VARPRO_FAST: inverse-depth refinement with the camera point affine in 1/lam (precomputed per observation).
- Cost model recalibrated to the new code (CU_* in build_wasm.sh; fit with a per-group regression), PIPE_CAP 1.55M,
  PIPE_MAX 4 (rare heavy solves may take 4 frames).
Tried and not kept: skyline Cholesky (frames-first or theta-p / v / shared order: -2% only), f32 template sums,
PIPE=3 (-16% fuel but +20% ATE), lower caps (1.45-1.50M: ATE worse).

## z145/ (not uploaded)
x155 + bit-identical SIMD map search (`vmap.inc` with `MAP_SIMD`), skyline Cholesky (`SKYLINE=1`, theta-p / v / shared order),
float template sums (`TPL_FAST=3`), recalibrated costs, cap 1.45M. Local: 0.544 m @ 1.64M (fast 0.433, normal 0.638).

## Build
`./build_wasm.sh $PWD/out.wasm` (defaults: `WS=../../../..`, the wasi-sdk of `../evaluate.py`; or set `WS`, `CLANG`)
(needs a clang that knows `-mno-bulk-memory-opt`, LLVM 20+; wasi-sdk with clang 23 was used.)

## Learned depth prior (NN_DEPTH, 2026-11-22, rejected)
Idea: vins sets scale on slow flights from a fixed room-depth prior (every feature: ln depth ~ N(ln 2.2 m, 1)); on the
start-up blow-up flight (agile-hard-dev Train-64-0001: 7.56 m for x155) the prior centre alone decides between 1.1 m
and 10 m. A small MLP predicts each corner's log depth (centre of its prior) from its elevation below the horizon
(gravity estimate), camera pitch, image position, template texture and contrast, FOV (and, `NN_WSET=0`, speed and height).
- Data: host log of each new window feature (`VINS_DLOG=path`) on 108 training flights (train-*, normal-dev2,
  bias3-xfast), triangulated with ground-truth poses (0.52 px median reprojection); 71k training corners.
- Fit (numpy, Gaussian NLL, 14 -> 16 tanh -> mean, log sigma): held-out log-depth error 0.436 vs 0.524 for the best
  constant (linear 0.441); without speed and height (`NN_WSET=3`, 11 inputs) 0.457.
- Closed loop (`-DNN_DEPTH=2 -DNN_OFF=0.2`), paired against x155 with the same runners:

| | x155 | + learned prior | Paired difference |
|---|---:|---:|---|
| Selection (187, host) | 0.392 / median 0.291 | 0.357 / 0.265 | -0.035; without the blow-up flight -0.009 (se 0.014) |
| Test (144, host) | 0.332 | 0.320 | -0.013 (se 0.013), 70/144 better |
| Held-out hard / extreme / windy (sandbox) | 0.514 / 0.983 / 0.344 | 0.541 / 1.130 / 0.391 | +0.074 (se 0.043), 17/48 better |
| Geometry-only net, dev levels (sandbox, 64) | | | +0.075 (se 0.039), 29/64 better |
| mini + learned prior: test / held-out | 0.510 / - | 0.488 / - | -0.022 (se 0.018) / +0.065 (se 0.117), 22/48 |

Speed was the strongest depth predictor (corr +0.28: in the normal flights faster flights are in larger rooms), which
does not hold on the difficulty levels; without it the gain on selection is gone and the dev levels are still worse
(the elevation input depends on the gravity estimate, which degrades with wind and IMU noise). Worst-frame fuel is
unchanged (the net runs once per new feature). All `NN_*` flags default off; the default build's code is identical
to the committed x155 (only the wasm name section differs).

## Scheduler fuel fixes (2026-11-22): x155 1.745M -> 1.624M, mini 0.604M -> 0.558M worst frame, same accuracy
Diagnosis from sandbox fuel joined with host logs of the planner (`VINS_PLOG`, bit-identical replay):
- `PIPE_LA=0` (look-ahead): `sc_work` started a unit whenever the estimate was below the frame's target, so a large
  indivisible unit (system build ~220k, solve ~175k, marginalization build ~256k / prep, drop) could end far above it;
  x155's worst frame was the middle frame of the 3-frame cycle of the first marginalization (frame ~50, 1.745M). On
  frames before the cycle's last, such a unit now waits when it would end above the target. Same chain, same order:
  ATE changes only at the 0.001 m level (a solve can land one frame later). x155 dev flights 1.745M -> 1.678M.
- `PIPE_LX=1 PIPE_CAPX=1.50` (learned cycle length): the remaining worst frames are the last frame of 2-frame cycles
  (planned level fits the cap, the remainder does not). A linear model of that frame's overshoot from what the
  planner knows (chain estimate, fixed work, tracks, cells, map tries, marginalization, window size) is fitted on
  sandbox fuel of agile-hard-dev (validation agile-xfast-dev: error sd 49k -> 22k); a 2-frame cycle predicted above
  1.50M gets 3 frames (~25% of 2-frame cycles).
- `PIPE_FXP=1` (mini): at start-up mini's 4-frame cycle planned its later frames' fixed work from the previous cycle
  (fewer tracks: ~104k assumed vs ~200k), so frame 12 inherited a 413k remainder (0.60M on every flight). The planner
  now uses at least CU_MID / CU_PHL + CU_TRACK x live tracks.

Paired against the originals, same runners (test = 144 flights host replay; levels = official sandbox, 16 per level):

| | Worst frame (all flights run) | Test ATE | Held-out hard / extreme / windy |
|---|---:|---:|---|
| x155 | 1.745M | 0.332 | 0.514 / 0.983 / 0.344 |
| x155 + `PIPE_LA=0 PIPE_LX=1 PIPE_CAPX=1.50` | **1.624M** | 0.323 (-0.009, se 0.013) | 0.595 / 0.786 / 0.389 (-0.024, se 0.048, 24/48) |
| mini m1 | 0.604M | 0.510 | 1.202 / 1.700 / 0.487 |
| mini + `PIPE_LA=0 PIPE_FXP=1` | **0.558M** | 0.500 (-0.010, se 0.019) | 1.219 / 2.041 / 0.540 (+0.138, se 0.147, 20/48; mini flights swing +-2-3 m) |

Area estimate (our conversion): x155 + mini 7.06% -> 7.23%. Build: `./build_wasm.sh OUT -DPIPE_LA=0 -DPIPE_LX=1
-DPIPE_CAPX=1.50`; mini: its build script with `-DPIPE_LA=0 -DPIPE_FXP=1`. The look-ahead alone does not change mini's
worst frame (start-up remainder); `PIPE_LX` is fitted to x155's costs (not used for mini, whose cycles are mostly at
PIPE_MAX=4).

## Host replay (bit-identical to the sandbox, ~0.4 s per flight)
`hostsimd/wasm_simd128.h` emulates the wasm SIMD intrinsics lane by lane; `./build_host.sh NAME [-D..]` builds
/tmp/learned-<uid>/bin/host_vfast_NAME with x155's settings, `mini/build_host.sh NAME` the m1 settings
(host_vmini_NAME). Checked: same ATE as the sandbox on 4 + 2 flights. Use with `../hosttools/hosteval.py`.

## CMA-ES noise weights (LEARN_P, LP_GATE, 2026-11-22, not adopted)
9 log multipliers, each a constant + slope on running speed (pixel noise, depth-prior sigma and centre, accel and gyro
noise, velocity prior, zero-velocity sigma, outlier threshold, bias floor; `lp_weights.h`). CMA-ES (learned/tools/search.py,
pop 14, 25 generations) on host replay of 108 training flights, each nominal and with IMU noise and bias x2.5
(`VINS_AUG`, host only); objective = mean ATE (capped 3 m) + mean harm against x155; checkpoint chosen on 79 choice
flights (gen 17). It learned: pixel noise x0.6, tighter depth prior (more so when fast), tighter velocity prior when
fast, looser outlier test when fast. `LP_GATE=1` applies the weights only when the accelerometer noise measured online
(successive differences) exceeds 1.3-1.8x nominal; on nominal flights it is identical to x155.

| Paired against x155 | Selection (187) | Dev levels (64, sandbox) | Test (144) | Held-out levels (48, sandbox) |
|---|---|---|---|---|
| CMA weights | -0.007, median 0.310 vs 0.291 | -0.080 (se 0.030), extreme -0.17, noisy -0.11 | - | - |
| CMA weights + fuel fixes | | | +0.017 (se 0.015), slow 'heldout' set +0.13 | -0.013 (se 0.065) |
| gated CMA weights + fuel fixes | = fuel fixes (gate off) | -0.059 (se 0.038), median -0.001 | = fuel fixes | -0.024 (se 0.057) |
| fuel fixes only | -0.012 | | -0.009 (se 0.013) | -0.024 (se 0.048) |

The dev-level gain does not carry over to the held-out houses (the extreme -0.18 / hard +0.08 swings there appear for
the fuel-only build too: chaos). The weights hurt normal flights (test 'heldout' set). Not adopted; all flags default off.
