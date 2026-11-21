# splat: learned_lin + a Gaussian-splat map

`solution.cpp`, `params.h`, `speednn.h`, `policy_lin.h` are a copy of `../learned` (learned_lin). With `SPLAT=0` the
build is learned_lin exactly (host replay: identical on all 276 flights). Everything new is in `splat.inc`, behind
compile-time switches.

| Model | Configuration | Binary | Max fuel / frame |
|---|---|---|---:|
| **splat** | `model/solution.cpp` | `../wasm/splat.wasm` | 274,165 |
| splat_lite | `lite/solution.cpp` | `../wasm/splat_lite.wasm` | 216,634 |
| splat_lite_rt | `lite_rt/solution.cpp` (splat_lite + retuned multipliers) | `../wasm/splat_lite_rt.wasm` | 218,851 |
| learned_lin (reference) | `../learned/lin` | `../wasm/learned_lin.wasm` | 88,551 |

## Idea

The estimator keeps a map of 3D Gaussians ("splats"): a mean in the world frame, a 3×3 covariance, an 8×8 template
and (splat only) a 16×16 context template of the green channel.

1. **Making splats.** During the first 20 s of a flight, each corner the tracker follows long enough is
   triangulated from two rays (the feature's first keyframe and now). Its covariance is the triangulation error
   plus the filter's pose uncertainty at that moment. Early in the flight that uncertainty is small, so the early
   map is nearly free of drift. The map is then frozen (up to 96 splats).
2. **Finding them again.** On every frame, one map splat that should be in view is projected into the image. The
   EWA splatting Jacobian turns its 3D covariance plus the current pose covariance into a 2D ellipse, which sets
   the search window (capped at ±30 px at 160×120; ±20 px in splat_lite). Inside it:
   - coarse ZNCC on a 2 px grid (16 samples of the template, SIMD, 4 positions at a time);
   - the full 8×8 template at ±1 px around the best score;
   - Lucas–Kanade refinement.

   A match is kept only if the peak is ≥ 0.8 and unique: it must beat the best score ≥ 4 px away by 0.05.
3. **Using them.** A match gives two EKF reprojection rows on position and attitude. The splat's covariance goes
   into the noise, and the χ² gate applies. Re-finding an early splat later in the flight works like a small loop
   closure: it pulls the drifted pose back towards where the map says the camera is.

Cost is flat: one search per frame on top of learned_lin's work.

## Results

Unaligned ATE (m), host replay, which matches the official sandbox exactly. Each flight is compared with
learned_lin; a negative difference is better.

- **Selection** used only learned_lin's own training sets (`tune`, 164 flights). splat: −0.268 m (se 0.038),
  better on 119/164. splat_lite: −0.223 m (se 0.034), better on 119/164.
- **Evaluation** used the 112 flights that neither learned_lin nor these models saw, run once per model:

| Set | n | learned_lin | splat | Difference (se) | Better on | splat_lite | Difference (se) | Better on |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| agile-hard-dev | 16 | 1.831 | 1.630 | −0.201 (0.080) | 12/16 | 1.666 | −0.165 (0.074) | 11/16 |
| agile-xfast-dev | 16 | 1.903 | 1.487 | −0.416 (0.128) | 13/16 | 1.535 | −0.368 (0.145) | 12/16 |
| agile-hard-test | 16 | 1.872 | 1.640 | −0.232 (0.057) | 15/16 | 1.762 | −0.110 (0.109) | 12/16 |
| agile-xfast-test | 16 | 2.078 | 1.909 | −0.169 (0.138) | 12/16 | 1.753 | −0.324 (0.166) | 10/16 |
| fresh-hard | 16 | 2.156 | 1.853 | −0.304 (0.141) | 13/16 | 2.000 | −0.156 (0.160) | 10/16 |
| fresh-xfast | 16 | 1.760 | 1.537 | −0.222 (0.148) | 13/16 | 1.494 | −0.265 (0.143) | 14/16 |
| heldout (normal) | 16 | 1.234 | 1.207 | −0.027 (0.101) | 8/16 | 1.198 | −0.036 (0.052) | 8/16 |
| **All** | **112** | **1.833** | **1.609** | **−0.224 (0.045)** | **86/112** | **1.630** | **−0.204 (0.048)** | **77/112** |

The gain is mostly on agile flights. On normal-speed held-out flights it is within noise.

**splat_lite_rt.** learned_lin's 21 noise multipliers retuned with splat_lite switched on, using learned_lin's own
procedure: CMA-ES on 136 training flights, objective = mean ATE + mean harm versus learned_lin, best mean on the
28 HSSD validation flights kept (iteration 30, validation 1.266 → 1.197; `lite_rt_cma.json`). Evaluation (112
flights, run once): 1.599 m, −0.235 m (se 0.050) versus learned_lin, better on 79/112. That is about 0.03 m better
than splat_lite, which is within noise. The same weights did not help the full splat model on validation, so splat
keeps learned_lin's multipliers.

**Official sandbox** (`../evaluate.py`, fuel metered, 80 episodes per model). The unaligned values equal the host
replay.

| Set | learned_lin unaligned / Sim(3) | splat | splat_lite |
|---|---:|---:|---:|
| agile-hard-test | 1.872 / 1.017 | 1.640 / 0.955 | 1.762 / 1.058 |
| agile-xfast-test | 2.080 / 1.112 | 1.908 / 1.090 | 1.753 / 0.996 |
| agile-hard-dev | 1.831 / 1.114 | 1.630 / 1.015 | 1.666 / 1.034 |
| agile-xfast-dev | 1.903 / 1.184 | 1.487 / 0.942 | 1.535 / 0.924 |
| heldout (normal) | 1.234 / 0.609 | 1.207 / 0.664 | 1.198 / 0.615 |
| **Max fuel per frame** | **88,551** | **274,165** | **216,634** |

splat_lite_rt in the sandbox: 1.574 / 1.773 / 1.663 / 1.454 / 1.135 m on the same five sets; max fuel 218,851.

**Harder held-out levels** (sandbox, 16 episodes each, from `../data/simulate_hard.py`; never used for selection):

| Level | learned_lin | splat | splat_lite |
|---|---:|---:|---:|
| hard | 1.927 | 1.753 | 1.689 |
| extreme (×3 IMU noise) | 1.961 | 1.788 | 1.751 |
| windy | 1.148 | 0.965 | 1.008 |

## What did not work, and why

| Variant | Tune Δ (m) | Eval Δ (m) |
|---|---:|---:|
| Splats measured continuously by the live tracker (map outside the filter) | +0.22 | — |
| Same, with the splats in the EKF state and full cross-covariance (EKF-SLAM, `SPLAT=2`) | +0.47 (48 flights) | +0.42 |
| Narrow re-acquisition only (±3–8 px LK around the projection) | −0.01 | +0.004 |
| Wide relocalization, map from the first 5 s, no pose covariance in the splats | −0.07 | — |
| Wide relocalization of the 20 s map, pose covariance in the splats (splat_lite) | −0.22 | −0.20 |
| + 16×16 context signature, ±30 px window (splat) | −0.27 | −0.22 |

- **Continuous splat rows collapse the scale.** learned_lin's local displacement is about 0.65× the true one (median
  over 1 s on the tune flights). Splats triangulated from that baseline come out too close (median depth 0.69×
  true). Anchoring the pose to them shrinks the motion further (speed ratio 0.83 → 0.75), and new splats get closer
  still. The splat geometry itself is fine: under ground-truth poses the tracked splats reproject to 0.5 px
  (median).
- **The map is the bottleneck, not the machinery.**
  - Ground-truth splat positions with the same rows: held-out 1.234 → 0.637 m, agile-hard-test 1.872 → 0.784 m.
  - Perfect matching against the filter's own 5 s map: 1.809 → 1.324 m (48 flights).

  The real matcher gets about 40% of that last gain.
- **Matching is the limit now.** Against ground truth, the true location is inside the search window in only 43%
  of searches, mostly because the drift is larger than the window cap. About 40% of accepted matches are within
  2 px (44% with the 16×16 signature). Wider windows, stricter thresholds and an extra 2D gate did not help.
- **Heading drift** (gyro-bias error, about 0.6°/s, 44° median at 60 s) is large. Fixing it alone (true bias)
  improves learned_lin only from 1.81 to 1.63 m: most of the error is translational.
- Retuning learned_lin's 21 noise multipliers with continuous splat rows (CMA-ES) did not recover the loss. Neither
  did stronger speed rows, a weaker velocity prior, a depth gain or a delayed start. Tracking a re-found splat for a
  few more frames was no better either.

## Switches (`splat.inc`; defaults are off / learned_lin)

| Switch | splat (lite) | Meaning |
|---|---|---|
| `SPLAT` | 1 | 1: map of splats outside the filter; 2: splats in the filter state (EKF-SLAM, `SPL_K` of them) |
| `SPL_RELOC`, `SPL_RELOC_PF`, `SPL_RELOC_R` | 1, 1, 30 (20) | wide relocalization, searches per frame, window cap (px) |
| `SPL_BIG` | 1 (0) | 16×16 context template as the coarse signature |
| `SPL_NCC`, `SPL_NCC_MARGIN`, `SPL_FINE_H` | 0.8, 0.05, 1 | match threshold, uniqueness margin, fine-search half width |
| `SPL_VIEW_COS`, `SPL_DRATIO` | 0.7, 2.0 | allowed change of viewing direction and distance since creation |
| `SPL_T_MAP`, `SPL_N`, `SPL_FREEZE`, `SPL_POSE_COV` | 20, 96, 1, 1 | mapping period (s), map size, frozen map, pose covariance in splats |
| `SPL_LINKED` | 0 | rows from live tracked features (harmful, see above) |
| `SPL_RINF` | 2 | noise inflation of splat rows |
| debug only | | `SPL_DEBUG` (counters), `SPL_GTLOG`, `SPL_GTCHK`, `SPL_GTMAP`, `SPL_ORACLE_MATCH`, `ORACLE_BG`: ground truth from env `SPL_GT` (host only) |

## Build and test

```bash
# wasm, same compiler and flags as ../evaluate.py (identical result: viobench bundle model -o splat.wasm)
WS=../../../..   # the solution/ directory
$WS/build/toolchain/wasi-sdk/bin/clang++ -O2 -msimd128 -fno-exceptions -I$WS/runtime/native $(../learned/tools/flags.sh) \
    model/solution.cpp -mexec-model=reactor -Wl,--initial-memory=67108864 -o ../wasm/splat.wasm
# host replay build and paired evaluation against learned_lin (feeds from ../learned/tools/make_feeds.py)
g++ -O2 -std=c++17 -I$WS/runtime/native $(../learned/tools/flags.sh) model/solution.cpp ../learned/tools/host_run.cpp -o host_splat
python3 ../hosttools/hosteval.py host_splat --sets eval
# experiments: bash build.sh NAME "-DSPLAT=1 ..." (host), bash wasm_build.sh OUT.wasm "...", bash sweep.sh tune "name|flags" ...
```
