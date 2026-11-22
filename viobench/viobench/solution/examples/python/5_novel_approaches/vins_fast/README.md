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
