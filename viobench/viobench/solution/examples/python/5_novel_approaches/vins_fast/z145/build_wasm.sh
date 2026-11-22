#!/usr/bin/env bash
# z145: x155 + SIMD map search (vmap.inc) + skyline Cholesky + float template sums, recalibrated, cap 1.45M
set -e
cd "$(dirname "$0")"
out=$1; shift
WS=${WS:-../../../../..}
CLANG=${CLANG:-$WS/build/toolchain/wasi-sdk/bin/clang++}
$CLANG -O3 -msimd128 -fno-exceptions -mllvm -disable-lsr -mllvm -unroll-threshold=3000 -mno-bulk-memory -mno-bulk-memory-opt \
    -I$WS/runtime/native \
    -DPIPE=2 -DPIPE_SCHED=2 -DPIPE_DET=1 -DEDGE=1 -DEDGE_WHEN=16 -DPIPE_CAP=1.45e6 -DPIPE_MAX=4 -DSKYLINE=1 -DTPL_FAST=3 \
    -DCU_N2=18.38 -DCU_IMU=34470 -DCU_F0=2150 -DCU_FM=605 -DCU_CH=3.08 -DCU_SOLVE=21.3 -DCU_SKYB=4.5 -DCU_SOLC=17800 -DCU_DAMPC=1780 -DCU_SR=4.14 -DCU_MPK=5.87 -DCU_MPC=4140 -DCU_MFK=0.207 -DCU_MFC=1725 -DCU_DRC=13800 -DCU_TRACK=6624 -DCU_CELL=1894 -DCU_CFULL=9106 -DCU_SEARCH=60450 -DCU_PH0=0 -DCU_PHL=36470 -DCU_MID=42600 \
    "$@" solution.cpp -mexec-model=reactor -Wl,--initial-memory=67108864 -o "$out"
