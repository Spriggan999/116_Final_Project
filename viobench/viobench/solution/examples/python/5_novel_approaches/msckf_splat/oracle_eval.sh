#!/usr/bin/env bash
# oracle_eval.sh NAME SETS...  run host_msplat_NAME with SPL_GT set per flight; print mean RMSE per set vs msckf final3
n=$1; shift
for s in "$@"; do
  for F in /tmp/learned-94445/feeds/$s/sim-*; do echo "echo $s \$(SPL_GT=$F/gt.f64 timeout 120 /tmp/learned-94445/bin/host_msplat_$n $F 2>/dev/null | grep rmse) \$(timeout 120 /tmp/learned-94445/bin/host_msckf_final3 $F 2>/dev/null)"; done
done | xargs -P 18 -I{} sh -c "{}" | python3 -c "
import sys,collections
d=collections.defaultdict(list)
for l in sys.stdin:
    p=l.split(); d[p[0]].append((float(p[2]), float(p[6])))
import numpy as np
tot=[]
for k,v in d.items():
    a=np.array(v); tot+=v; print(f'{k} {a[:,0].mean():.3f} vs {a[:,1].mean():.3f}', end=' | ')
a=np.array(tot); print(f'ALL {a[:,0].mean():.3f} vs {a[:,1].mean():.3f}')"
