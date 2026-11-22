#!/usr/bin/env bash
# build_wasm.sh OUT.wasm [-DFLAG=..]   Put this folder next to the original vins/ folder (same depth) or set WS / CLANG.
# Fuel-tuned flags (bit-identical maths): -mllvm -disable-lsr (constant load offsets), -mllvm -unroll-threshold=3000,
# -mno-bulk-memory -mno-bulk-memory-opt (memset/memcpy become the SIMD loops in solution.cpp: bulk ops cost ~1 fuel/byte).
set -e
cd "$(dirname "$0")"
out=$1; shift
WS=${WS:-../../../..}
CLANG=${CLANG:-$WS/build/toolchain/wasi-sdk/bin/clang++}
$CLANG -O3 -msimd128 -fno-exceptions -mllvm -disable-lsr -mllvm -unroll-threshold=3000 -mno-bulk-memory -mno-bulk-memory-opt \
    -I$WS/runtime/native \
    -DPIPE=2 -DPIPE_SCHED=2 -DPIPE_DET=1 -DEDGE=1 -DEDGE_WHEN=16 -DPIPE_CAP=1.55e6 -DPIPE_MAX=4 \
    -DCU_N2=19.13 -DCU_IMU=37180 -DCU_F0=2150 -DCU_FM=605 -DCU_CH=3.77 -DCU_SOLVE=14.23 -DCU_SOLC=11860 -DCU_DAMPC=1186 -DCU_SR=4.14 -DCU_MPK=5.87 -DCU_MPC=4140 -DCU_MFK=0.207 -DCU_MFC=1725 -DCU_DRC=13800 -DCU_TRACK=7290 -DCU_CELL=1894 -DCU_CFULL=9106 -DCU_SEARCH=141750 -DCU_PH0=0 -DCU_PHL=36470 -DCU_MID=42600 \
    "$@" solution.cpp -mexec-model=reactor -Wl,--initial-memory=67108864 -o "$out"
