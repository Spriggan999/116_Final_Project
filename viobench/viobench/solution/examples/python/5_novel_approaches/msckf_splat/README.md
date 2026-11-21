# msckf_splat: the MSCKF plus a Gaussian-splat map

A copy of `../msckf` (the OpenVINS-style MSCKF, "final3") with a persistent map of 3D Gaussian splats. Everything new
is in `splat_map.inc`, plus a few hooks in `solution.cpp`. With `SPLAT=0` (the default of `solution.cpp`) the build
is exactly msckf final3 (host replay identical on all selection flights). `model/solution.cpp` sets `SPLAT=1`.
Binary: `../wasm/msckf_splat.wasm`.

## Results (unaligned ATE, m; paired per flight)

Test flights, scored once after all choices were made:

| Set | Flights | learned_lin | msckf | **msckf_splat** | vs msckf: difference (se) | Better than msckf |
|---|---:|---:|---:|---:|---:|---:|
| agile-hard-test | 16 | 1.872 | 0.749 | **0.703** | -0.046 (0.048) | 10/16 |
| agile-xfast-test | 16 | 2.078 | 0.661 | **0.436** | -0.225 (0.121) | 12/16 |
| heldout (normal, start hover) | 16 | **1.234** | 1.775 | 1.625 | -0.150 (0.093) | 10/16 |
| **All** | 48 | 1.728 | 1.062 | **0.921** | **-0.140 (0.053)** | 32/48 |

Against learned_lin: -0.81 m (se 0.18), better on 36/48. It is still worse than learned_lin on the normal held-out
flights (+0.39 m), though less so than msckf (+0.54 m). The fresh-* test sets were not yet regenerated after the
reboot, so they are not included.

**Selection** used 76 flights: agile-hard-dev, agile-xfast-dev, normal-dev, and the regenerated train-hard and
train-xfast. The chosen configuration scores -0.136 m (se 0.035) versus msckf there, better on 55/76.

**Official sandbox** (`sandbox_eval.py`, fuel metered) gives the same ATE as the host replay:

| | agile-hard-test | heldout |
|---|---:|---:|
| msckf_splat ATE | 0.703 | 1.625 |
| msckf_splat max fuel per frame | 6,377,128 | 5,857,813 |
| msckf max fuel per frame | 6,235,141 | 5,834,899 |
| msckf_splat mean fuel per frame | 2.60M | 2.25M |

The map adds about 0.14M to the worst frame and about 0.2M to the mean.

## How it works

- **Splats.** Each splat has:
  - a mean in the filter's world frame (the first IMU frame);
  - a 3×3 covariance: the triangulation information of the MSCKF track (anchored inverse depth, Gauss-Newton
    Hessian, mapped to xyz) plus the anchor clone's position and attitude covariance;
  - a 9×9 NCC signature of 2×2 box means on a 2 px grid;
  - 9×9 LK templates at stride 2 and 1;
  - the creation viewing direction and distance.
- **Creation.** A track becomes a splat when it spans the MSCKF window, is accepted by the MSCKF update, and is
  created in the first 20 s. It must have a position sigma under 0.5 m, a depth of 0.4–8 m, and no other splat
  within 12 px in the current view. The map holds up to 96 splats and is frozen after 20 s.
- **Search.** Each frame, one visible splat is searched for: the one tried least recently, at least 4 s old, viewed
  within a 2× distance change and 60° of its creation direction, and not tried in the last 3 frames.
  1. Its innovation covariance S sets a 3σ search window, capped at ±30 px. S is the pose covariance plus the splat
     covariance, projected with the EWA/splatting Jacobian.
  2. Coarse NCC: a 5×5 sub-signature on a 4 px grid.
  3. The best 3 distinct peaks are refined with the full signature on the 2 px grid.
  4. The match must have NCC ≥ 0.85 and beat the runner-up by 0.05.
  5. LK coarse, then fine; the result must stay within 3 px of the NCC peak.
- **Update.** The match becomes 2 EKF rows on the current attitude and position. The noise is (projected splat
  covariance × 4) plus 1.5 px, whitened.

## What the evidence showed

Host-only ground-truth diagnostics (`SPL_GT=<feed>/gt.f64`, printed with the host build). The true point is
re-triangulated from the creation track with true poses; 82–97% of these are self-consistent within 1 px.
- **Window:** the true location is inside the search window 84–92% of the time (it was 43% for the learned_lin
  splats). MSCKF's metric scale fixes the prediction.
- **Match accuracy is the weak part:** about 57% of accepted matches are within 2 px and about 15% are off by more
  than 5 px. The error barely depends on scale change, NCC score or margin.
- **Map error:** splats are about 1.5–2× the filter's own position error at creation. That is 0.26–0.45 m on agile
  flights, and 0.9–3 m on slow normal flights.
- **Oracles** on the 68 dev + train-hard flights (msckf 0.835 m):

  | Variant | Mean ATE |
  |---|---:|
  | Real map, real matching (this model) | ~0.72 m |
  | Perfect matching only | 0.669 m |
  | Perfect map only | 0.678 m |
  | Both perfect | 0.41 m |

  Both parts limit the gain, and fixing both together would compound.

Tried and rejected, each on the selection flights:

| Variant | Difference vs msckf on selection flights |
|---|---|
| Consensus: 3–8 searches per frame, use only matches whose offsets agree | Precision 54% -> 76%, but 2.5× fewer matches; -0.07 instead of -0.11 m |
| Soft consensus (non-agreeing matches with 4× noise) | -0.09 m |
| More searches per frame without consensus (2, 4, 8) | Equal or worse; 8 harms normal flights |
| Refining splat positions with each match (per-splat Kalman update) | -0.03 m |
| Refreshing the appearance after confident matches | -0.04 m |
| Tighter innovation gates (2–2.5σ), stricter NCC (0.9) or margin (0.08) | Worse |
| Information gate (search only if pose variance > κ × splat variance) | Not better than the age rule |
| Splats used after 1 s | +0.03 m (worse than msckf) |

The age rule (splats used after 4 s: -0.136 m) was the largest single gain. A young splat repeats what the MSCKF
window already knows, with correlated error. Map size (64/96/160), mapping period (10/20/30 s), window cap (24–36 px),
noise inflation (3–6) and sigma (1–2.5 px) were all within ±0.02 m.

## Files

| File | Use |
|---|---|
| `solution.cpp` | copy of msckf final3 + hooks (`SPLAT=0` = msckf exactly) |
| `splat_map.inc` | the splat map; every parameter is a `-D` switch (defaults = the chosen model) |
| `model/` | `SPLAT=1` wrapper + CMakeLists for `viobench bundle model -o msckf_splat.wasm` |
| `build_host.sh NAME [-D..]` | host replay binary `/tmp/learned-<uid>/bin/host_msplat_NAME` |
| `build_wasm.sh OUT.wasm [-D..]` | wasm (-O3 -msimd128); the shipped binary is `model/solution.cpp` built the same way |
| `sweep.sh SETS "name\|flags" ...` | parallel builds + paired evaluation vs msckf final3 |
| `diag.sh NAME SETS`, `oracle_eval.sh NAME SETS` | ground-truth match diagnostics; oracle runs (`-DSPL_ORACLE_MAP=1`, `-DSPL_ORACLE_MATCH=1`) |
| `sandbox_eval.py` | official runner with fuel |

## Growing map (tried 2026-11-20, not adopted)

The splat code is also merged into `../msckf/solution.cpp` (`-DSPLAT=1`; `splat_map.inc` there). Instead of freezing the
map after 20 s, `SPL_GROW` keeps creating splats later in the flight, recycling the longest-unseen slot when all 96 are
used:

- `SPL_GROW=1`: only where fewer than `SPL_VIS_MIN` splats project into the view.
- `SPL_GROW=2`: only where fewer than `SPL_HIT_MIN` splats were matched in the last `SPL_RECENT` frames.
  Projection ignores walls, so this is the better "new area" test.
- `SPL_SMAX_LATE` and `SPL_POSE_K` control how uncertain a late splat may be, and how much of the drone's pose
  uncertainty at creation it carries.

Paired against the frozen map on 92 selection flights:

| Variant | Difference | Better on |
|---|---:|---:|
| Grow where few splats project into view, `SPL_SMAX_LATE=2` | -0.011 m (se 0.016) | 50/92 |
| Grow where few matches recently, `SPL_RECENT=80`, `SPL_SMAX_LATE=2` | +0.014 m (se 0.015) | 32/92 |
| Grow where few matches recently, `SPL_RECENT=40` | +0.019 m (se 0.010) | 28/92 |

With `SPL_GROW=2` the map really grows: on a normal flight there were 124 late splats and 71 late matches. It still did
not help, for two reasons:

- A splat made at time t carries the pose error the filter had at t. Matching it later mostly confirms the current,
  already drifted, estimate. The MSCKF already keeps short-term motion consistent.
- The early splats are valuable because the pose error is smallest at the start of the flight. In these 60 s flights
  the first 20 s already see most of the house.
