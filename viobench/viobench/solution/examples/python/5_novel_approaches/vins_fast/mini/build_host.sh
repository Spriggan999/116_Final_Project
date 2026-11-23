#!/usr/bin/env bash
# build_host.sh NAME [-DFLAG=..] -> /tmp/learned-<uid>/bin/host_vmini_NAME : host replay binary of vins_mini (m1 settings),
# from ../solution.cpp with this folder's vmap.inc (SIMD map search), wasm SIMD emulated lane by lane (../hostsimd)
set -e
cd "$(dirname "$0")"
name=$1; shift
T=$(mktemp -d); cp ../solution.cpp ../nn_depth_weights*.h "$T"/; cp vmap.inc "$T"/
M1="-DPRESET=2 -DPIPE=2 -DPIPE_SCHED=2 -DPIPE_DET=1 -DEDGE=1 -DEDGE_WHEN=16 -DPIPE_CAP=0.42e6 -DPIPE_MAX=4 -DMAX_FEATS=16 -DEDGE_MAX=6 -DCU_N2=17.68 -DCU_IMU=36810 -DCU_F0=2612 -DCU_FM=735 -DCU_CH=4.31 -DCU_SOLVE=15.1 -DCU_SOLC=12600 -DCU_DAMPC=1260 -DCU_SR=5.88 -DCU_MPK=8.34 -DCU_MPC=5880 -DCU_MFK=0.294 -DCU_MFC=2450 -DCU_DRC=19610 -DCU_TRACK=7045 -DCU_CELL=1367 -DCU_CFULL=10420 -DCU_SEARCH=55260 -DCU_PH0=59170 -DCU_PHL=25570 -DCU_MID=55060"
g++ -O2 -std=c++17 -ffp-contract=off -D__wasm_simd128__ -D__wasm_bulk_memory__ -I"$(pwd)/../hostsimd" -I"$(pwd)/../../../../../runtime/native" $M1 "$@" "$T"/solution.cpp "$(pwd)/../../learned/tools/host_run.cpp" -o /tmp/learned-$(id -u)/bin/host_vmini_$name
rm -rf "$T"
