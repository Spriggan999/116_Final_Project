#!/usr/bin/env bash
# build_wasm.sh OUT.wasm [-DFLAG=..]   (wasi-sdk toolchain of ../evaluate.py, -O3)
set -e
cd "$(dirname "$0")"
out=$1; shift
WS=../../../..
$WS/build/toolchain/wasi-sdk/bin/clang++ -O3 -msimd128 -fno-exceptions -I$WS/runtime/native "$@" solution.cpp \
    -mexec-model=reactor -Wl,--initial-memory=67108864 -o "$out"
