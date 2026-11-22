#!/usr/bin/env bash
# vins_mini (m1): VINS-ultra window (5 frames), 16 corner tracks + 6 edgelets, solve every 2nd frame, cost model fitted
# to this size, cap 0.42M; same fuel engineering as ../ (flags, SIMD kernels, SIMD map search in vmap.inc)
set -e
cd "$(dirname "$0")"
out=$1; shift
WS=${WS:-../../../../..}
CLANG=${CLANG:-$WS/build/toolchain/wasi-sdk/bin/clang++}
$CLANG -O3 -msimd128 -fno-exceptions -mllvm -disable-lsr -mllvm -unroll-threshold=3000 -mno-bulk-memory -mno-bulk-memory-opt \
    -I$WS/runtime/native \
    -DPRESET=2 -DPIPE=2 -DPIPE_SCHED=2 -DPIPE_DET=1 -DEDGE=1 -DEDGE_WHEN=16 -DPIPE_CAP=0.42e6 -DPIPE_MAX=4 -DMAX_FEATS=16 -DEDGE_MAX=6 \
    -DCU_N2=17.68 -DCU_IMU=36810 -DCU_F0=2612 -DCU_FM=735 -DCU_CH=4.31 -DCU_SOLVE=15.1 -DCU_SOLC=12600 -DCU_DAMPC=1260 -DCU_SR=5.88 -DCU_MPK=8.34 -DCU_MPC=5880 -DCU_MFK=0.294 -DCU_MFC=2450 -DCU_DRC=19610 -DCU_TRACK=7045 -DCU_CELL=1367 -DCU_CFULL=10420 -DCU_SEARCH=55260 -DCU_PH0=59170 -DCU_PHL=25570 -DCU_MID=55060 \
    "$@" solution.cpp -mexec-model=reactor -Wl,--initial-memory=67108864 -o "$out"
