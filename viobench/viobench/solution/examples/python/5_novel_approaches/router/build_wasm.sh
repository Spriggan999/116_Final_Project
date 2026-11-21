#!/usr/bin/env bash
# build_wasm.sh OUT.wasm [-D...]   router wasm (wasi-sdk of ../evaluate.py, -O3 -msimd128); MSCKF built with SPLAT=1
set -e; cd "$(dirname "$0")"; out=$1; shift
WS=../../../..; CXX=$WS/build/toolchain/wasi-sdk/bin/clang++; N=$WS/runtime/native
O=$(mktemp -d /tmp/learned-$(id -u)/rwasm.XXXX)
F="-O3 -msimd128 -fno-exceptions -I$N"
$CXX $F -DSPLAT=1 "$@" -c msckf_part.cpp -o $O/m.o
$CXX $F -I../learned $(../learned/tools/flags.sh) -c lin_part.cpp -o $O/l.o
$CXX $F "$@" -c router.cpp -o $O/r.o
$CXX $F $O/m.o $O/l.o $O/r.o -mexec-model=reactor -Wl,--initial-memory=67108864 -o "$out"
rm -rf $O
