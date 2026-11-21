#!/usr/bin/env bash
# diag.sh NAME SET...   aggregate ground-truth splat diagnostics of host binary host_msplat_NAME over feed sets
b=/tmp/learned-94445/bin/host_msplat_$1; shift
for s in "$@"; do for F in /tmp/learned-94445/feeds/$s/sim-*; do echo "SPL_GT=$F/gt.f64 timeout 120 $b $F 2>&1 | grep '^GT'"; done; done | xargs -P 18 -I{} sh -c "{}" | python3 -c "
import sys,re
t=v=w=a=a2=0; me=[]
for l in sys.stdin:
    m=re.search(r'tries (\d+) valid (\d+) inwin ([\d.]+) accepted (\d+) within2px ([\d.]+)',l)
    if not m: continue
    T,V,I,A,A2=int(m[1]),int(m[2]),float(m[3]),int(m[4]),float(m[5]); t+=T; v+=V; w+=I*V; a+=A; a2+=A2*A
    mm=re.search(r'match_err_px n \d+ median ([\d.-]+)',l); me.append(float(mm[1]))
print(f'tries {t} inwin {w/max(v,1):.2f} accepted {a} ({a/max(t,1):.2f}) within2px {a2/max(a,1):.2f} good/flight {a2/max(len(me),1):.1f} bad/flight {(a-a2)/max(len(me),1):.1f}')
"
