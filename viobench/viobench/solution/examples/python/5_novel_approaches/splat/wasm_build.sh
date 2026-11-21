#!/usr/bin/env bash
# wasm build: bash wasm_build.sh OUT.wasm "-DSPLAT=1 ..."  (learned_lin configuration + the given flags)
cd "$(dirname "$0")"
WS=$(cd ../../../.. && pwd)
$WS/build/toolchain/wasi-sdk/bin/clang++ -O2 -msimd128 -fno-exceptions -I$WS/runtime/native $(../learned/tools/flags.sh) \
    -DLEARN=1 -DLEARN_LIN=1 -DLEARN_HEADER='"policy_lin.h"' $2 solution.cpp -mexec-model=reactor -Wl,--initial-memory=67108864 -o "$1" 2>&1 | grep -v warning | grep -E "error" 
exit 0
