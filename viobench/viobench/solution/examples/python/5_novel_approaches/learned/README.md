# learned: learned noise multipliers on a copy of v9_adapted

`solution.cpp`, `params.h`, `speednn.h` and `CMakeLists.txt` are a copy of `../../6_stage2_movestart` (v9_adapted,
`stage2_adapt.wasm`). With `LEARN=0` (the default) the copy behaves exactly like v9_adapted: same ATE, same fuel.
`6_stage2_movestart` itself is not changed.

## What is learned

Once per 6-frame cycle, on the update frame, a tiny network maps 9 features of the last window to 7 noise
multipliers. The multipliers scale the filter's own noise settings: they are the tunable values, updated in flight.

- **Network:** 9 inputs → 6 tanh units → 7 outputs. Each output is a log multiplier, capped at ±2 (×0.14 to ×7.4).
  109 weights; about 700 fuel per cycle.
- **Inputs:** parallax, fraction of tracks found, direction-fit quality, speed, rotation rate, yaw gyro-bias
  sigma, velocity sigma, time since start, measured gyro-noise ratio.
- **Outputs:** noise of the rotation, direction, speed, velocity-prior and small-motion rows; gyro-bias and
  accelerometer process noise.

Two ways to train it, plus a constant baseline:

| Build | How the weights are found |
|---|---|
| `static/` | CMA-ES over the 7 output biases only (constant multipliers): the headroom check |
| `rl/` | CMA-ES over all 109 weights, started from `static`. Reward = −mean unaligned ATE of whole training flights (black-box policy search) |
| `sl/` | Supervised. Each logged row's true error against ground truth; Gaussian likelihood per row class. The velocity-prior output is switched off, because its error (the drone's velocity) persists over seconds and is not white noise. Rotation and process-noise outputs have no direct target and stay ×1 |

Binaries: `../wasm/learned_static.wasm`, `learned_rl.wasm`, `learned_sl.wasm`.
Weights: `policy_static.h`, `policy_rl.h`, `policy_sl.h`.

## Results

Mean unaligned ATE (m), official sandbox, paired per flight against v9_adapted. Training flights come from houses
100–111. Dev and test are local agile flights in other houses: dev in houses 64–67 (used to choose), test in
houses 196–199 (run once). 16 flights per set.

| Build | Hard dev | Xfast dev | Hard test | Xfast test | Normal held-out | Max fuel |
|---|---:|---:|---:|---:|---:|---:|
| v9_adapted | 1.957 | 2.181 | 2.172 | 2.097 | 1.246 | 88.6k |
| static | 1.745 | 2.014 | 1.779 | 2.117 | 1.281 | 88.6k |
| rl | 1.973 | 1.849 | **1.723** | **2.068** | 1.410 | 88.6k |
| sl | 1.900 | 2.045 | 2.139 | 2.109 | **1.128** | 88.6k |

Paired over the 32 test flights: rl −0.24 m (se 0.11, better on 22/32), static −0.19 m (se 0.12, 19/32),
sl −0.01 m (se 0.06, 16/32).

### Later additions, and a fresh check

- **`sl2/`:** SL retrained on direction, speed and small-motion rows only (no prior rows in training).
- **`combo/`:** sl2 for those rows, plus the static rotation (×0.14) and accelerometer (×0.37) multipliers.

Both were then checked, together with static and rl, on 32 fresh flights in 8 more unseen houses (120–130),
scored once. Values are differences from v9_adapted in m; negative is better.

| Build | Fresh flights (v9_adapted 2.144 m) | Better on | All 96 agile flights (dev + test + fresh) | Better on | Flights worse by > 1 m |
|---|---:|---:|---:|---:|---:|
| static | +0.02 (se 0.17) | 18/32 | −0.12 (se 0.08) | 56/96 | 3 |
| rl | +0.02 (se 0.15) | 16/32 | −0.13 (se 0.07) | 56/96 | 4 |
| sl2 | +0.04 (se 0.07) | 18/32 | −0.05 (se 0.04) | 56/96 | 0 |
| combo | −0.02 (se 0.11) | 20/32 | −0.10 (se 0.08) | 60/96 | 7 |

static and rl were tuned only on the training houses, so their 96-flight numbers are fair. sl2 and combo were
shaped partly by dev / test results, so for them only the fresh column is clean.

### Speed-aware model (`lin/`, `LEARN_LIN=1`): the current best

Each log multiplier is a constant plus a slope on estimated speed and a slope on parallax (21 numbers,
`policy_lin.h`).
- **Training:** found by CMA-ES on 132 training flights (the 40 above plus 92 in 24 more houses: ProcTHOR 140–163
  and HSSD). The objective is mean ATE + 1 × the mean amount by which a flight gets worse than v9_adapted, so
  wrecking a flight costs double.
- **Early stopping:** on 32 flights in 8 held-out training houses. Iteration 15 was kept: training error kept
  falling afterwards, the held-out error did not.
- **Results** on all 112 evaluation flights (none used for training or stopping), differences from v9_adapted:

| Flights | Difference (m) | Better on |
|---|---:|---:|
| fast (dev + test + fresh, 96) | −0.18 (se 0.06) | — |
| fresh only (32) | −0.19 (se 0.11) | — |
| normal held-out (16) | −0.01 (se 0.08) | — |
| all 112 | — | 69/112 (5 flights worse by > 1 m) |

  The gain grows with speed: +0.02 / −0.19 / −0.18 / −0.24 m in the speed bins < 0.4 / 0.4–0.6 / 0.6–0.75 /
  > 0.75 m/s.
- **Sandbox** (`../wasm/learned_lin.wasm`): hard test 1.872 vs 2.172, xfast test 2.080 vs 2.097, normal held-out
  1.234 vs 1.246. Max fuel 88,551, the same as v9_adapted.

What it learned:
- rotation rows ×0.14 and direction rows ×7.4 (both at the ±2 cap);
- the velocity prior tightens with speed (×0.76 at rest, ×0.34 at 1 m/s);
- gyro-bias and accelerometer process noise lower (×0.6 and ×0.3), rising slightly with speed.

A faster autopilot tier did not make flights faster: ProcTHOR and HSSD room layouts cap the mean speed near 0.9 m/s.

**Conclusion for the earlier versions** (before the speed-aware model). None of them was a clear improvement on
v9_adapted:
- On the fresh flights all four are within ±0.04 m.
- Pooled, static and rl gain about 0.12 m (≈1.7 standard errors), with occasional flights that get much worse.
- The supervised version is the safest (never > 1 m worse) but gains the least.
- Larger sets (100+ flights per comparison) would be needed to settle a 0.1 m effect.

- **Dev and test disagree.** sl was best on dev but did nothing on test; rl did nothing on hard dev but was best on
  hard test. With 16 flights per set, single-set differences of ±0.2 m are within noise.
- **The constant multipliers are the most consistent gain on agile flights:** −0.19 m on dev and −0.19 m on test,
  neutral on normal flights. What they encode:
  - trust the camera's rotation measurement about 7× more (gyro bias learned faster);
  - trust the direction rows about 7× less;
  - accelerometer noise ×0.37.

  rl starts from the same constants and adds input-dependent changes. Those help on agile flights and hurt on
  normal ones (+0.16 m).
- Max fuel is unchanged. The network runs on the update frame, which is not the most expensive frame.

## Tools (`tools/`)

| File | Use |
|---|---|
| `make_feeds.py` | episode zips → green-channel replay feeds (the estimator reads only green) |
| `host_run.cpp` | replays a feed through `init` / `step` on the host. Matches the sandbox exactly; ~0.1 s per 60 s flight |
| `flags.sh` | the `CMakeLists.txt` configuration as `-D` flags for host builds |
| `search.py` | CMA-ES (`static`, `policy`), paired evaluation (`eval`), header export (`export`) |
| `sl_train.py` | logs rows on the host, computes true errors, fits the supervised version |

Host builds (from this folder):

```bash
g++ -O2 -std=c++17 -I../../../runtime/native $(tools/flags.sh) -DLEARN=1 -DLEARN_RUNTIME=1 solution.cpp tools/host_run.cpp -o host_rt
```

Add `-DLEARN_LOG=1` for `host_log`; with no `LEARN` flags you get `host_base`.

The copy also fixes the host (non-wasm) build. The scalar path of `rank_slice` did not fill the score array that
`select_top` reads, so host builds of v9 tracked nothing. The wasm path is unchanged.
