#!/usr/bin/env bash
# bash sweep.sh SET "name|flags" ...   (host builds + hosteval, paired vs learned_lin)
cd "$(dirname "$0")"
SET=$1; shift
for v in "$@"; do n=${v%%|*}; bash build.sh $n "${v#*|}" & done; wait
for v in "$@"; do n=${v%%|*}; echo "$n: $(cd ../hosttools && python3 hosteval.py /tmp/learned-$(id -u)/bin/host_splat_$n --sets $SET --workers 8 | grep ALL)"; done
