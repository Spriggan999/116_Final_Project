#!/usr/bin/env bash
# build_host.sh NAME [-D...]  -> /tmp/learned-<uid>/bin/host_msckf_NAME  (router flags RT_*; MSCKF flags apply to msckf_part)
set -e; cd "$(dirname "$0")"; name=$1; shift
N=../../../../runtime/native; O=/tmp/learned-$(id -u)/router-$name; mkdir -p $O
g++ -O2 -std=c++17 -I$N -DSPLAT=1 "$@" -c msckf_part.cpp -o $O/m.o
g++ -O2 -std=c++17 -I$N -I../learned $(../learned/tools/flags.sh) -c lin_part.cpp -o $O/l.o
g++ -O2 -std=c++17 -I$N "$@" -c router.cpp -o $O/r.o
g++ -O2 -std=c++17 -I$N $O/m.o $O/l.o $O/r.o ../learned/tools/host_run.cpp -o /tmp/learned-$(id -u)/bin/host_msckf_$name
