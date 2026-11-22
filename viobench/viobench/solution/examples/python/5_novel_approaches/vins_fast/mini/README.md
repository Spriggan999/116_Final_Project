# vins_mini (m1): low-fuel VINS point

Same code as `../` (vins_fast), VINS-ultra window: 5 frames, 16 corner tracks + 6 edgelets, map search 1 per frame,
solve every second frame, cost model refitted for this size (`calm`), cap 0.42M. SIMD map search (`vmap.inc`, bit-identical).

Local (unaligned ATE / max fuel):

| Set | x155 (`../`) | m1 |
|---|---|---|
| 32 fast flights | 0.385 m @ 1.71M | 1.046 m @ 0.59M |
| 38 normal flights | 0.618 m @ 1.71M | 0.890 m @ 0.59M |

Scaled by x155's server/local ratio, the expected server ATE is about 0.56 m at 0.59M. That adds about 0.7% area to the union
(the region left of x155 and below the 88k tracker's 3.44 m).

Lower-fuel variants tried (fast set): 14 tracks without map 1.13 m @ 0.52M; solve every 3rd frame 1.40 m @ 0.51M;
VARPRO 1 or WIN 4: 1.3-1.7 m @ 0.47-0.63M (worse area). Full ultra preset (20 tracks + 10 edgelets): 0.82 m @ 0.84M.
