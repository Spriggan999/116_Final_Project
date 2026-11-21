# msckf_schmidt: Schmidt-EKF splat map on router_lite2

A copy of `../msckf` in which the Gaussian-splat map can be used as a **Schmidt-EKF map** (`-DSCHMIDT=1`). The router
build lives in `../router_schmidt`. Binary: `../wasm/router_schmidt.wasm`.

With `SCHMIDT=0` (the default) the code is bit-identical to router_lite2 (host replay, paired, diff 0.000), and its
fuel is the same (within 0.01%).

## Idea

In router_lite2, a splat match is treated as independent of the filter state: the splat's own covariance is
inflated (x4) and added to the measurement noise. A splat is triangulated from the filter's poses, so its error is
correlated with the state, and repeated matches of the same splat are counted as new information each time.

The Schmidt-EKF (Geneva et al., ICRA 2019) keeps the **cross-covariance between the state and the map points**,
but never updates the map points themselves. Cost stays linear in the map size, and repeated matches are handled
consistently.

## Implementation (all behind `SCHMIDT`)

- **Active set.** Up to `SCH_N` splats (default 12) hold a slot. Each slot has 3 columns in `Xc` (state x map,
  `NMAX x 3*SCH_N`) and a block in `Pmm` (map x map). Other splats keep the old independent model.
- **Creation.** The splat's position depends on the anchor clone: p_f = p_a + R_a q. That gives
  `J = [-[p_f - p_a]x , I]` on the clone's (attitude, position). So `Xc = P[:, anchor] J^T` and
  `Pmm = J P_aa J^T + C_tri`. `C_tri` is the triangulation covariance only (the pose term is now in `J`). Cross
  blocks with other active splats are `J Xc[anchor, other]`.
- **Slots.** If the set is full, the active splat that was matched longest ago is evicted, but only after
  `SCH_EVICT` frames (default 200). Its slot is released by moving the last slot into it. A recycled splat gives
  its slot back first.
- **Propagation.** The IMU block of Phi is applied to the `Xc` columns, with the same structured 3x3-block transform
  as P (`xf` lambda).
- **Clones.** Augmentation copies the IMU rows of `Xc`. Marginalization removes the clone's rows. A divergence reset
  zeroes the velocity rows.
- **Ordinary MSCKF updates.** These have no map columns, so `Xc -= G^T L^-1 H Xc` (G = L^-1 H P, already
  computed). The loops are blocked 4 rows / 4 nonzeros at a time. With `SCH_F32=1`, `Xc` is single precision.
- **Splat match of an active splat.** The innovation covariance uses the joint covariance,
  `S = Hx Pxx Hx^T + Hx Pxm Hf^T + Hf Pmx Hx^T + SCH_INF * Hf Pmm Hf^T + sigma^2`. It is used for the search window
  and the gate. The update is then applied separately, after the frame's normal update. It is re-linearized at the
  current state and re-gated, with K = (P H^T)_x S^-1. The state is updated; `P -= K (P H^T)^T` and
  `Xc -= K H [Pxm; Pmm]`. `Pmm` and the splat do not change. Splat refinement is skipped for active splats.
- **Host checks.** `SCH_CHECK=1` checks that the joint 9x9 [attitude, position, map point] covariance is positive
  definite at every Schmidt update. It always was. `SCH_PROF=1/2` is for fuel profiling only, and its results are
  wrong.

| Flag | Default | Meaning |
|---|---:|---|
| `SCHMIDT` | 0 | Schmidt map on/off |
| `SCH_N` | 12 | active splats |
| `SCH_INF` | 1.0 | inflation of the map covariance in S |
| `SCH_EVICT` | 200 | frames without a match before an active splat can be evicted |
| `SCH_PREF` | 1 | search active splats first (0 = old round robin) |
| `SCH_F32` | 0 | single-precision cross-covariance |

Final build: `$(cat B3 flags) -DSCHMIDT=1 -DSCH_INF=2 -DSCH_PREF=0 -DSCH_F32=1`
(B3 = router_lite2: `-DPR=3 -DMAX_ROWS=20 -DDET_SPARSE=1 -DSPL_SKIP_ROWS=16 -DVMAX=6.0 -DVP_ADAPT=1 -DVP_K=1.2
-DVP_MIN=0.8 -DVPRIOR_SIG=3.0 -DSPL_GROW=1 -DSPL_SMAX_LATE=2.0 -DRT_WMAX=0.95`).

```bash
cd ../router_schmidt
./build_wasm.sh OUT.wasm $B3 -DSCHMIDT=1 -DSCH_INF=2 -DSCH_PREF=0 -DSCH_F32=1
./build_host.sh NAME $B3 -DSCHMIDT=1 ...      # /tmp/learned-<uid>/bin/host_schmidt_NAME
```

## Results (unaligned ATE, paired against router_lite2)

Selection (187 flights: dev, training, normal-dev2, 3x gyro bias, hssd3):

| Variant | Difference (se) | Better on |
|---|---:|---:|
| SCH_INF 1 (12 slots, active first) | +0.037 (0.019) | 90/187 |
| SCH_INF 2 / 4 / 8 | -0.000 / +0.002 / +0.009 (0.011-0.012) | 85-89 |
| SCH_INF 4, round robin | -0.006 (0.014) | 92 |
| SCH_INF 2, round robin (**chosen**) | -0.011 (0.010) | 100 |
| SCH_INF 1 / 3, round robin | +0.022 / +0.002 | 86 / 96 |
| 6 / 16 / 24 slots | +0.007 / +0.005 / +0.011 | |
| 16 slots, SCH_INF 4, round robin | -0.016 (0.014) | 91 |
| eviction after 0 / 40 / 100 frames | +0.013 to +0.031 | |
| chosen, single precision | -0.001 (0.010) | 96 |
| chosen, single precision, 4 / 6 slots | +0.005 / +0.001 | 69 / 82 |

Test (144 flights, each candidate scored once):

| Build | Mean | Difference (se) | Better on |
|---|---:|---:|---:|
| router_lite2 | 0.820 | | |
| Schmidt, double | 0.817 | -0.003 (0.016) | 76/144 |
| Schmidt, single precision (`router_schmidt.wasm`) | 0.822 | +0.002 (0.016) | 73/144 |

Fuel (official sandbox, 48 episodes; sandbox ATE equals the host replay):

| Build | Max fuel per frame | Mean |
|---|---:|---:|
| router_lite2 | 3.27M | 1.69M |
| Schmidt, double, unblocked | 4.36M | 2.24M |
| Schmidt, single precision, blocked | 3.61M | 1.87M |

Claimed-area estimate (log-log reference rectangle; only differences between rows matter):

| Build | Area |
|---|---:|
| router_lite2 | 2.85% |
| Schmidt, single precision | 2.80% |
| Schmidt, double | 2.74% |

## Conclusion

The Schmidt map is consistent: the joint covariance stayed positive definite. It is **neutral for accuracy**: every
variant is within about 1-2 standard errors of router_lite2. The fuel cost is real. Keeping `Xc` up to date costs
about 170k fuel per frame on average (12 slots, single precision). The worst frame is 0.35M higher.
**It claims less area than router_lite2, which stays the recommended build.**

Why it does not help:
- Matches are rare: SPL_PF=1 search per frame, 5-35 Schmidt updates per 60-s flight.
- The map is mostly re-observed soon after it was built. The pose-conditional map variance collapses after a few
  matches (0.005 to 0.0017 m^2 on the checked flight), so later matches carry little information. This is correct
  behaviour, but it gives no extra correction. router_lite2's x4-inflated independent model already gets a
  similar effective weight.
