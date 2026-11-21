#!/usr/bin/env bash
# build_host.sh NAME [-DFLAG=..]  -> /tmp/learned-<uid>/bin/host_vins_NAME  (host replay binary, no fuel)
set -e
cd "$(dirname "$0")"
name=$1; shift
g++ -O2 -std=c++17 -I../../../../runtime/native "$@" solution.cpp ../learned/tools/host_run.cpp -o /tmp/learned-$(id -u)/bin/host_vins_$name
