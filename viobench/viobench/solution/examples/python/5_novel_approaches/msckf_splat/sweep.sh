#!/usr/bin/env bash
# sweep.sh SETS "name|flags" ...   build host variants (SPLAT=1 + flags) in parallel, paired eval vs msckf final3
cd "$(dirname "$0")"
sets=$1; shift
for spec in "$@"; do n=${spec%%|*}; f=${spec#*|}; ./build_host.sh $n -DSPLAT=1 $f 2>&1 | grep -m3 error & done; wait
for spec in "$@"; do n=${spec%%|*}
  echo -n "$n: "; python3 ../hosttools/hosteval.py /tmp/learned-94445/bin/host_msplat_$n --sets $sets --ref /tmp/learned-94445/bin/host_msckf_final3 2>&1 | grep -v "^ *sim" | grep -v "==" | awk '{printf "%s %s(%s) %s | ", $1, $9, $11, $13}'; echo
done
