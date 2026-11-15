# CLAUDE.md — EECS 116 Project 3: Compute-Aware VIO (viobench)

Team: Ninad (+2). Goal this week: a **simple, hyper-efficient** VIO that claims as much Pareto area as possible.
Repo: https://github.com/jonas-eschmann/viobench · Server: https://viobench.rl.tools (1 upload/day, resets midnight PT).

## 1. The task in one screen
- Input per `step`: RGB frame (320×240 default, or 640×480 via `frame_width()`), timestamp `t`, and exactly 10 IMU rows
  `[t, ax, ay, az, gx, gy, gz]` (200 Hz, specific force incl. gravity, rad/s, IMU frame). Frames 20 Hz.
- Config (JSON, at `init`): `width, height, fx, fy, cx, cy, T_cam_imu` (4×4). **Intrinsics vary per episode.** Pinhole, no distortion.
  Camera axes: x right, y down, z forward.
- Output `out[13]` = `[x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]`:
  pose `T_c0_ck` (current camera in first-camera frame, quaternion **scalar last**), twist **in current camera frame**.
  Write all 13 every step. NaN/inf/zero-quaternion/trap ⇒ episode fails, rest filled with zeros.
- Not given: gravity direction, whether episode starts at rest, episode length, IMU noise model. Sim episodes hover ~1 s at start
  (`hold_seconds=1.0`) — but the private set may not; never rely on it blindly.
- Causal: estimate final once `step` returns.

## 2. Scoring — what actually matters
- One point per submission: **x = MAX fuel of any single step** (all episodes), **y = mean Sim(3) ATE** (equal weight per episode).
- **Sim(3) ATE aligns rotation, translation AND uniform scale** over the whole trajectory ⇒
  - global metric scale does NOT matter, only *scale consistency* over time;
  - global yaw/gravity alignment does NOT matter for the headline metric;
  - what matters is the *shape* of the trajectory.
- Constant output (cpp_minimal, 113 fuel) scores ≈ 2.69 m (RMS spread of GT). That is the "do nothing" bar.
- Area score is **log–log**: fuel 1 → 24 G, ATE 0.001 → 3.23 m. A 10× fuel cut is worth as much as a 10× ATE cut.
  ⇒ Big wins live in the **low-fuel, decent-ATE** corner, where baselines are empty.
- Fuel = Wasmtime fuel ≈ **1 unit per executed wasm instruction** (nop/block/loop/drop free). Deterministic.
  - Native C with `-O2 -msimd128` (already in `viobench_reactor.cmake`): a v128 op processes 16 bytes for 1 fuel ⇒ SIMD/vectorizable loops are ~16× cheaper per pixel.
  - Python runtime costs ~0.09 M fuel per step just to exist; numpy has no SIMD/BLAS. **Use native C/C++ for efficiency.**
  - Max-per-step counts, not mean: **keep per-frame cost flat** (no periodic feature re-detection spikes, no growing state, no first-frame heavy work in `step` — do setup in `init`, which has its own 2e11 budget).

### Current frontier (read off the plot, approx.)
| fuel/frame | Sim(3) ATE | likely |
|---|---|---|
| 113 | 2.69 m | cpp_minimal (identity) |
| ~2e4 | ~2.6 m | ? |
| ~4 M | ~1.9 m | python dead_reckoning (4.1 M) |
| ~15 M | ~0.7 m | python tiny_cnn (14 M) |
| ~1.5 G | ~0.55 m | heavy (openvins/klt) |
| ~3 G | ~0.53 m | heavy |
Our first submission: ~3 G fuel, ~2.0 m ATE → dominated, 0% area. Don't repeat that direction.

### Limits
step fuel 2e10, init fuel 2e11, memory 2 GiB, wall 60 s/step, submission ≤128 MiB.

## 3. Strategy (research-backed)
**Core idea: gyro for rotation + accelerometer-propagated velocity, corrected by cheap optical-flow epipolar constraints. Native C, no OpenCV, fixed cost.**

Why:
- Rotation from the gyro is nearly free and accurate over 1-minute episodes (bias from startup).
- Accelerometer double integration drifts in seconds (dead_reckoning ≈1.9 m). Vision only needs to **bound velocity drift**.
- Translational optical flow at bearing `x_i` (after subtracting gyro-predicted rotational flow) satisfies the continuous
  epipolar / subspace constraint `v · (x_i × u_i) = 0` — **linear in velocity and independent of depth** (Heeger & Jepson '92;
  used for MAVs in Weiss et al. "Inertial Optical Flow for Throw-and-Go MAVs"). ⇒ A plain linear Kalman update on velocity, no depth/landmark states.
- When derotated flow ≈ 0 everywhere → zero-velocity update (hover). Huge drift killer, very cheap.
- Speed along the motion direction is observable from the accelerometer once direction changes (drone flies waypoint routes).
- Cheap flow references: EdgeFlow / Edge-FS (McGuire et al., RA-L 2017) — edge histograms + block matching, ran at 20 Hz on a 168 MHz STM32F4.
  ROVIO (Bloesch et al., IROS 2015) — direct photometric patches in an EKF, ~50 patches, no feature descriptors. PX4Flow-style block matching.
- Sim(3) scoring forgives global scale error, so even imperfect metric scale is fine as long as it's consistent.

### Pipeline per step (target ≤ 100–300 k fuel)
1. **IMU (~2–5 k fuel):** integrate gyro (minus bias) → `R_wb`; propagate `v`, `p` with `R_wb·a + g`. 10 samples, use timestamps for dt.
2. **Image (~10–30 k):** gray + downsample to 80×60 (or 40×30) via box sum on a strided subset of rows; vectorizable loop. Keep prev small image.
3. **Flow (~30–100 k):** fixed grid of N≈9–16 patches (e.g. 8×8 at 80×60), 1–2 level pyramid, Lucas–Kanade 2–3 iterations,
   **initialized with the gyro-predicted rotational flow**. Reject patches with low texture (min-eigen of structure tensor) or high residual.
4. **Derotate:** translational flow `u_i = flow_i − rotflow_i(ω, x_i)`; build normals `n_i = x_i × u_i` (normalized bearing coords).
5. **KF update (~5–10 k):** state `[v(3), accel bias(3)]` (+ optional gyro bias). Measurements `n_iᵀ v = 0` with noise ∝ 1/|n_i|; Huber/χ² gate.
   If median |u| < threshold → ZUPT `v = 0`.
6. **Output:** pose from `R_wb`, `p` mapped through `T_cam_imu` relative to first camera; twist = `R_cbᵀ`-rotated v with lever arm `ω × r`, ω from gyro.

### Init
- First ~0.5 s: mean accel → gravity direction (and accel bias along it); if still (low std), mean gyro → gyro bias. If moving at start,
  bias = 0 and rely on KF. Output identity + zero twist until initialized (never NaN).

### Roadmap
- v0 (day 1): native C IMU dead-reckoning — validates conventions, ABI, fuel tooling. Expect ~1.9 m at ~5 k fuel (already new frontier point).
- v1: + grid LK flow + epipolar velocity KF + ZUPT. Target < 1.0 m at < 300 k fuel.
- v2: tuning on `viobench simulate` data; bias states; outlier gating; twist quality.
- Later options: native port of a tiny CNN (tiny_cnn gets 0.7 m at 14 M in Python; native+SIMD and a smaller input could be ~10–100× cheaper);
  submit variants at different fuel levels — each own point can claim area.

### Contraction theory angle (Sasha Davydov, Rice ME; Bullo group, UCSB)
- Design the velocity/attitude estimator as a **contracting observer**. Error dynamics shrink exponentially from any initial
  condition, so starting in flight with wrong gravity, bias or velocity is forgotten automatically. This matters on hidden episodes,
  and any divergence or NaN fills the rest of the episode with zeros.
- Concretely: a cascade like Lopez 2023. First a complementary or contracting attitude observer (gyro + accel/flow) with gyro bias.
  Then a linear velocity/accel-bias observer driven by the flow epipolar rows `n_iᵀ v = 0` and ZUPT. Check that `A − LC` has a
  negative matrix measure (fixed gains are cheaper than a KF covariance update and can be tuned offline).
- If we learn anything recurrent (bias predictor, flow→velocity correction), constrain it to be contracting
  (Davydov et al.'s non-Euclidean contraction conditions for RNNs / implicit nets; RENs). That gives guaranteed stability, a tiny
  model and no blow-ups.
- Full link list: `references.md`.

### Local GPU (RTX 3060, 12 GB) — offline only
Inference runs in wasm on the CPU and is metered, so the GPU never helps the score directly. Use it for:
1. `viobench simulate` (HyperDrone) for training/validation episodes without waiting for lab days. Needs CUDA in WSL2; test a small run first.
2. Automated tuning: sweep observer gains, patch grid and image size on many sim episodes, then plot the ATE-vs-fuel curve.
3. Training tiny learned parts (tiny_cnn-style or contracting RNN), then porting the weights to native C with int8/fixed point.

### Status (2026-11-15)
- Teammate's distilled-ML submission `sub_500_mff`: 150 fuel, 1.93 m. At 150 fuel it can't read the IMU, so it's essentially
  a learned trajectory shape. Sim(3) alignment means a "typical flight shape" alone gets ~1.9 m. IMU-only (v0) is dominated by it.
- **v1** (`v1/solution.cpp` + `params.h`): 18-state error-state EKF [p v θ ba bg p_kf]. Every KF_LEN=8 frames, 16 LK patches
  (80×60) are tracked from the keyframe, started at the gyro prediction, and aggregated into one motion direction
  (smallest eigenvector of Σ n nᵀ). Two constraints e·d = 0 with H_θ = e × d. Max fuel ≈1.25 M (keyframe frames), median ≈67 k.
  Synthetic noisy scenes (Sim3 ATE vs constant baseline): fast 0.39 vs 1.68, room 0.62 vs 0.80, hover 0.80 vs 1.11.
- Lessons: (1) per-frame flow at 80×60 is only ~0.3 px of translation, too small, so a longer keyframe baseline is needed;
  (2) many noisy epipolar rows pull velocity toward zero, so aggregate into one direction first; (3) direction noise must
  scale with 1/parallax, and low-parallax updates are skipped; (4) the rest test must check |mean gyro| and | |a| − g |,
  not just variance; (5) tilt and gyro bias must be in the filter.
- **v2** (`v2/`): v1 plus (a) 160×120 images and a denser 14×10 candidate grid (real ProcTHOR rooms are low-texture);
  (b) cheirality sign for the motion direction; (c) small-parallax "displacement ≈ 0" update (hover);
  (d) **parallax speed proxy** |d| ≈ parallax·PAR_SPEED/f along u (speed scale from accel alone collapsed on real data;
  Sim(3) only needs a consistent scale); (e) weak velocity prior σ=0.5 m/s; (f) patch selection deferred one frame to flatten
  max fuel. Public set (10 eps): **mean Sim3 ATE 0.630 m, max fuel 966k** (eps 0.29–0.82). Identity = 3.24, v0 = 2.15.
  Caveat: tuned on the public set, and the proxy assumes typical indoor depth (fast synthetic scene got worse, 0.39 → 0.84).
- **v3** (`v3/`): v2 plus a vision rotation check. The tracks also estimate the rotation error δ of the gyro-predicted
  keyframe→current rotation (pure-rotation fit at low parallax, one alternating epipolar step otherwise). δ_body = +T·δb_g
  updates the gyro bias, with noise growing with parallax because of rotation/translation ambiguity. It is NOT fed back into the
  direction fit (that hurt). Yaw drift was gyro z-bias ~0.015–0.02 rad/s. Public: **0.597 m, max fuel 1,000,404**,
  rotation RMSE 14.1° → 7.6°. Deterministic: same binary and same numbers on rebuild and rerun.
- **v4** (`v4/`): same estimator, rebuilt for fuel. Public: **0.563 m, max fuel 156,535** (median 32k/frame). Area calc
  (tests/area.py replicating the server, calibrated on the teammate's 1.77%): v3 2.26% → **v4 3.98%**, plus teammate 5.18%.
  Fuel changes: covariance propagated once per window with block-sparse row ops (f64x2 SIMD); IMU loop without sin/cos;
  no full-image resize (pixels fetched on demand from the raw RGB frame); candidates scored from a sparse 4×4 raw sample
  grid; 8×8 patches tracked with wasm SIMD LK (constant bilinear weights per iteration, one-pass brightness offset);
  templates/tracking split over SPLIT=3 frames; candidate ranking spread over the keyframe frame.
  Tried and rejected: interleaved keyframe pipelines (more updates → velocity shrinkage, worse ATE); stale ranking on an
  idle frame (worse); direction-corrected speed proxy (better velocity, worse ATE); 80×60 tracking (worse).
  50% area is not reachable: even 5 cm at 1e3 fuel ≈ 29%. Accuracy at low fuel is worth more than fuel cuts.
- **v5** (`v5/`): v4 + robustness, chosen on a stress suite (tools/robust.py): public 10 + on-the-fly variants
  mid (start 10 s in), imu (extra biases, 1.5x noise), light (dim, drifting exposure), fov (narrower lens), each reported
  per half A/B. Private episodes are not downloadable. Null perturbations change nothing, so ~0.03 m gaps on the suite are real.
  Changes: (1) texture threshold scaled by the frame's gradient energy (CONTRAST_REF 2000; free, uses the ranking tensors) —
  dim images 0.99 -> 0.55 m; (2) SIG_ROT_PAR 0.05 (trust the vision rotation check less while translating; a mid-turn start
  had driven the gyro z-bias wrong, 82 deg heading error) — mid worst 1.47 -> 0.89; (3) PAR_SPEED 8 (middle of a flat 6–10 optimum).
  Suite ALL: v4 0.675 -> v5 0.555. Public: **0.545 m, max fuel 157,823**. Area: v5 4.13%, +lite (N_TRACK=12: 0.650 m, 136k)
  4.24%, +teammate 5.33%. Rejected on the suite: lower MIN_EIG (helps dim, hurts base), PROP_ONCE, fewer grid cells,
  KF_LEN 5/7, stronger velocity prior (hurts mid start).
- **v6** (`v6/`, overnight): v5 + process-noise retune on the stress suite: SIG_ACC 0.15 -> 0.3 (clear trend: 0.08 worse,
  0.25–0.4 best, 0.9 worse) and DEPTH_PRIOR 4 -> 6 (flat 6–9). Public **0.500 m, max fuel 157,823**; suite ALL 0.525 (v5 0.555).
  Area: v6 4.65%, +teammate 5.85%. Flat/no gain: SIG_BA, SIG_DIR_PX/FLOOR, MIN_PAR_PX, SMALL_FLOW_PX, init sigmas, N_TRACK 24.
  Cheaper ladder: N_TRACK 12 = 0.660 m @136k (adds only +0.1%), N_TRACK 8 = 0.77 m @119k.
- **Fixed-gain / LTI filter (Ninad's idea), tried and rejected** (`FIXED_GAIN=1` in v7 code). Constant-gain observer, no covariance:
  displacement residual split along the motion direction (speed gain) / perpendicular (direction gain), velocity leak,
  gyro bias from the rotation check. Lessons: without a tilt loop it is useless (~2.0 m: 0.01 rad tilt leaks 0.1 m/s^2);
  a Mahony-style loop (horizontal low-frequency residual -> tilt, integral -> gyro bias) brings it to 0.72 m public,
  suite ALL 0.768 at 135k. Gain scheduling by window quality made it worse (tilt loop needs steady high gain).
  Dominated by the EKF lite variant, so not useful. The EKF's cross-covariances (tilt <-> velocity <-> bias) are what matter.
- **v7** (`v7/`): v6 estimator, per-frame work rebalanced. Public **0.512 m, max fuel 99,040** (v6 158k); suite ALL 0.528
  (v6 0.525, within noise). Area v7 4.91% (v6 4.65%), +teammate 6.03%. Exact speed-ups (identical results):
  SIMD scoring of 4 candidate cells per pass + histogram prefilter for the top-K (ranking 95k -> ~54k);
  SIMD pixel fetch (green bytes 6 apart via one i8x16 shuffle). Behaviour-changing but neutral on the suite:
  position clone + propagation moved to frame 1 (off the ranking frame); covariance propagated once per window on the
  idle frame, ahead over the 3 tracking frames (PROP_MODE 3; propagating only up to the idle frame hurt: 0.66);
  corrections injected once per frame; velocity prior on the frame after the direction update.
  Rejected: uneven subset sizes, RANK_SPLIT 2 (ranking part on frame 1 hurts ~0.015), velocity prior on frame 1
  (public 0.476 but suite 0.540 -> overfit), direction update one frame late. Lite (N_TRACK 12) 0.658 @ 87k: little area.
  Fuel costs (exact, measured by running a piece twice): one scalar EKF row ~3.6k, inject ~5k, propagation 22k,
  tracking ~7k per patch, 4-cell SIMD score ~730. Wasm fuel is ~2-3x a naive instruction estimate (local.get/set).
  Per-phase max now: keyframe ~98k, last tracking frame ~99k; other frames <= 70k.
- **EKF add-ons tried after v7 (all off by default; build byte-identical to v7):** adaptive measurement noise from
  normalized innovations (`ADAPT_R`; two-way 0.542-0.551 worse, inflate-only 0.530 neutral), Huber weighting
  (`HUBER_C` 0.528 neutral, chi2 gate already handles outliers), log-scale state of the speed proxy (`SCALE_STATE`, 19 states:
  0.527-0.534 neutral, +8.5k fuel; without a pull to s=0 it collapses to 1.26 like v2's accel-only scale). Scale drift across
  rooms of different depth is untestable on our suite (fixed images). Conclusion: remaining error is not mis-set noise or scale.
- **v8** (`v8/`): v7 + generator prior on the motion sign. Public **0.467 m, max fuel 99,263** (official evaluate);
  suite ALL 0.500 (v7 0.528; every variant better). Area 5.49%, +teammate 6.61%.
  Diagnosis (tools/oracle.py, tools/dirstat.py): rebuilding each episode from per-window displacements with the TRUE
  direction and our speed gives 0.378 m; true speed + our direction 0.527; true attitude 0.488 -> per-window direction is
  the bottleneck. Its error is mostly camera-x (sideways), random per window (lag-1 autocorr ~0.1), plus ~1.4% sign flips.
  The generator's FPV autopilot flies nose-first (all 10 eps: camera-frame motion 90-98% forward), so when
  |u . z_cam| > SIGN_FWD=0.7 the sign is taken as forward (flat 0.6-0.85). Risk: backward flight along the axis.
  Rejected: sideways-displacement prior (0.67-0.94: reinforces flipped estimates, flips 1.4% -> 4.7%),
  sideways-velocity prior NHC_SIG (0.504 alone, no gain on the suite with the sign rule, +5k fuel),
  full-res tracking DS=1 (0.71-1.18), LK_ITERS 4, median parallax, retuned PAR_SPEED_REL/VPRIOR/MAX_Q/CHI2/SIG_ROT_PAR.
- Tools: tools/oracle.py (oracle direction/speed/attitude rebuild), tools/dirstat.py (per-window direction errors),
  tools/robust.py (stress suite), tools/area.py, tools/phaseall.py (per-phase max fuel over episodes).
- Synthetic test harness lives in Claude's sandbox (tests/harness.py on viobench's tests/vio_scene.py renderer).

## 4. Commands
```bash
# Windows: everything inside WSL2 (Ubuntu), files in the WSL filesystem (~/...), NOT /mnt/c/...
sudo apt install build-essential cmake curl git python3-venv xz-utils
git clone https://github.com/jonas-eschmann/viobench && python3 -m venv .venv && source .venv/bin/activate
pip install -e ./viobench                      # lab GPUs only if training: "./viobench[train]"
viobench init native ws && cd ws               # C/C++ workspace, solution/ = cpp_minimal
viobench bundle solution -o solution.wasm      # first time fetches wasi-sdk
viobench check solution.wasm                   # synthetic frames: errors + fuel/step
viobench evaluate solution.wasm -o report      # public set (10 episodes, ~200 MB each, cached in ~/.cache/conta)
viobench simulate -o data/train --episodes 64 --seed 1   # needs GPU (lab RTX 5080)
cmake -S solution -B build-host && cmake --build build-host   # host debug build (gdb) — docs/DEBUGGING.md
curl -H "Authorization: Bearer $TOKEN" https://viobench.rl.tools/config > config.json   # match server limits: --limits
```
Read `report/report.json`: check `max fuel`, `p99 fuel`, `sim3_ate`, `filled_frames`, `scale`.

## 5. Gotchas
- Native: no `exit/abort`/uncaught exceptions (they trap) — return -1. Build is `-O2 -msimd128`.
- Copy anything you keep from `frame`/`imu` — buffers are overwritten. Config bytes valid only during `init`.
- Normalize quaternions; guard divisions; never output NaN.
- Lever arm: camera centre ≠ IMU centre — use `T_cam_imu`.
- Fuel is deterministic: seed any RNG; avoid data-dependent loops (RANSAC iteration counts) that create max-fuel spikes.
- Private test episodes are hidden and may differ (start in flight, other scenes/FOV). Prefer robust over tuned.
- AI policy: anything goes, but must be reproducible; highlight simplicity/robustness in the report.
- Timeline: 11/16, 11/23 progress presentations (frontier frozen each week), 11/30 final.

## 6. User preferences
- Give whole updated scripts, not diffs. Python plot scripts: no directory paths (run from the data folder).
- Never open a browser tab without permission.
- Keep explanations brief.
