#!/usr/bin/env bash
# host build: bash build.sh NAME "-DSPLAT=1 ..."  -> /tmp/learned-<uid>/bin/host_splat_NAME
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -I../../../../runtime/native $(../learned/tools/flags.sh) -DLEARN=1 -DLEARN_LIN=1 -DLEARN_HEADER='"policy_lin.h"' $2 solution.cpp ../learned/tools/host_run.cpp -o /tmp/learned-$(id -u)/bin/host_splat_$1
