# 6_stage2_movestart: stage-2 solution (unaligned ATE, hidden agile flights)

Our stage-2 uploads, their source, the flight generators and the evaluation tools. The findings about the server set are in `server_test_cases.md` at the repo root.

## Uploads

| Binary (`../wasm/`) | Server mean ATE | Max fuel | What it is |
|---|---|---|---|
| `stage2_tracker_v2.wasm` | 4.31 m | 86.5k | persistent two-group tracker, rotation clone, thrust-only IMU model; calibrates during a start hover |
| `stage2_movestart_nn.wasm` | 3.44 m | 88k | + starts on the first frame with no hover assumption (MOVE_START) + tiny speed-correction MLP (SPEED_NN) |
| `stage2_adapt.wasm` | 3.45 m | 87.8k | + accelerometer model error scaled by the IMU noise measured online (ADAPT_IMU=2); identical on the server because its IMU noise is normal |

## Source

- `solution.cpp`, `params.h`: the v8 estimator with every switch in `params.h`. The defaults reproduce v8; `CMakeLists.txt` sets the configuration of `stage2_adapt` (drop `ADAPT_IMU=2` for `stage2_movestart_nn`).
- `speednn.h`: the speed model, a 9-8-1 tanh MLP (89 weights) that multiplies the assumed-depth speed measurement by exp(out), clamped to x0.5-x2. Trained on 19.5k windows from 192 flights in ProcTHOR houses 20-43.
- Build: `viobench bundle viobench/solution/examples/python/6_stage2_movestart -o out.wasm`.

Main switches added in stage 2 (all documented in `params.h`):

| Switch | Effect |
|---|---|
| `TRK=2`, `TRK_SCHED=3`, `TRK_DISP=1` | features persist across windows; two staggered groups keep the per-frame fuel flat |
| `ROT_CLONE=1`, `ROT_REFIT=0` | attitude clone at the keyframe; vision measures relative rotation (corrects attitude and gyro bias) |
| `THRUST_ONLY=1` | the simulator has no drag: specific force is along body z only |
| `MOVE_START=1`, `CALIB_SECONDS=0.0`, `REST_GYRO_MEAN=0.0`, `HOVER_INIT=0` | no start-hover calibration: filter starts on frame 0 with wide priors |
| `SPEED_NN=1` | speed-correction MLP (`speednn.h`) |
| `ADAPT_IMU=2` | isotropic accel model error follows the measured accel noise ratio (capped x3) |
| `WIND_STATE`, `ADAPT_IMU=1`, `ROTC_MIN_KEPT` | tried, no gain (off) |
| `DIAG9`, `DIAG10`, `DIAG11` | diagnostics written into the outputs (speed-model features, gyro bias, rotation-row sigmas); never submit |

## Generating flights (`generation/`, run inside WSL with the OptiX setup from CLAUDE.md)

- `agile_sim.py` / `agile.sh`: `viobench simulate` with a faster autopilot (env `AGILE` = 7 or 8 autopilot constants; tiers med, hard, xfast in `agile.sh`).
- `agile_sim2.py`: also IMU noise scale (`IMU`), bias scale (`BIAS`) and wind force std in N (`WIND`), the same template edit as `5_novel_approaches/data/simulate_hard.py`.
- `gen_train.sh`: 192 training flights (houses 20-43). `gen_test.sh`: hover-start test flights (houses 120-161 and 1).
- `gen_s2like.sh`: server-like test flights, no start hover (`--hover 0`), houses 50-57 and 4 HSSD houses. Checking only.
- `gen_s2train.sh`: server-like training flights for the speed model (houses 80-95, 4 mixed settings).
- Each new setting recompiles the simulator once (5-7 min); a hanging scene is skipped after a timeout. ProcTHOR scenes above 199 do not exist.

## Tools (`tools/`)

Written for a Linux shell with the `viobench` package importable; set `VIOBENCH_ROOT` to the folder containing the `viobench` package if it is not found automatically. Results are cached per (wasm hash, episode) in `~/ev/cache`.

| Script | Use |
|---|---|
| `ev.py "SET" a.wasm b.wasm` | unaligned ATE, Sim(3), scale, max fuel; paired difference and wins against the first wasm |
| `evmid.py K0[xS][nN][bB] "SET" wasms` | start each flight K0 frames in (no hover), feed every S-th frame, IMU noise xN, bias xB (env `BIAS_ONLY=g` or `a`) |
| `full.py "SET" wasm` | server-style per-flight table (ATE, Sim(3), RPE, drift/m, distance) |
| `bias.py zip wasm` | true gyro bias (from ground truth) against the filter's estimate over time (`DIAG10` build) |
| `orbias.py "SET" wasm none,z,xyz` | oracle: ATE when the gyro stream is corrected with the true bias |
| `roterr.py K0 zip wasm` | rotation error and 1 s RPE over time |
| `dump9.py`, `sp_data.py`, `sp_train.py`, `sp_export.py` | dump speed-model windows (`DIAG9` build), train the MLP (numpy, house-split CV), export `speednn.h` |
| `imuonly.py` | IMU-only estimator experiment (worse than outputting zeros) |
| `d11.py zip wasm` | gyro-bias sigma and rotation-row sigma over time (`DIAG11` build) |
