#!/usr/bin/env bash
# build_host.sh NAME [-DFLAG=..]  -> /tmp/learned-<uid>/bin/host_vfast_NAME  (host replay binary with x155's settings)
set -e
cd "$(dirname "$0")"
name=$1; shift
X155="-DPIPE=2 -DPIPE_SCHED=2 -DPIPE_DET=1 -DEDGE=1 -DEDGE_WHEN=16 -DPIPE_CAP=1.55e6 -DPIPE_MAX=4 -DCU_N2=19.13 -DCU_IMU=37180 -DCU_F0=2150 -DCU_FM=605 -DCU_CH=3.77 -DCU_SOLVE=14.23 -DCU_SOLC=11860 -DCU_DAMPC=1186 -DCU_SR=4.14 -DCU_MPK=5.87 -DCU_MPC=4140 -DCU_MFK=0.207 -DCU_MFC=1725 -DCU_DRC=13800 -DCU_TRACK=7290 -DCU_CELL=1894 -DCU_CFULL=9106 -DCU_SEARCH=141750 -DCU_PH0=0 -DCU_PHL=36470 -DCU_MID=42600"
g++ -O2 -std=c++17 -ffp-contract=off -D__wasm_simd128__ -D__wasm_bulk_memory__ -Ihostsimd -I../../../../runtime/native $X155 "$@" solution.cpp ../learned/tools/host_run.cpp -o /tmp/learned-$(id -u)/bin/host_vfast_$name
