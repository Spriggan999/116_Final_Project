# CLAUDE.md: EECS 116 Project 3, compute-aware VIO (viobench)

Handoff notes for any agent helping Ninad. Written 2026-11-18 from the full history of the project.
Course framework: viobench (https://github.com/jonas-eschmann/viobench). Team repo: https://github.com/Spriggan999/116_Final_Project.
Leaderboard server: https://viobench.rl.tools (**1 upload per day**, resets midnight Pacific). Presentations 11/16 and 11/23, final 11/30.

## 0. User preferences (follow these)
- Keep explanations brief and easy to understand; no filler.
- When updating code, give the **whole updated script**, not diffs.
- Python plotting scripts: no directory paths; he runs them from the folder holding the data files.
- **Never open a browser tab without permission.**
- He wants a fundamental/classical approach, robust rather than overfit to the public set.
- He pushes with **Git Bash on Windows** (agents in a VM usually cannot push). Commit/push only when asked.
- Never put the team's server token or his email in files or requests.

## 1. The task and the score (one screen)
- Each `step` gets one RGB frame (320×240, 20 Hz), its timestamp and exactly 10 IMU rows `[t, ax, ay, az, gx, gy, gz]` (200 Hz).
  `init` gets JSON: `width, height, fx, fy, cx, cy, T_cam_imu`. **Intrinsics differ per episode** (FOV ~110–135°).
- Output 13 numbers every step: pose `T_c0_ck` `[x y z qx qy qz qw]` (quaternion scalar LAST) + twist in the current camera frame.
  NaN/inf/trap = episode fails and the rest is zero-filled.
- Compiled to WebAssembly; **fuel ≈ 1 unit per wasm instruction**, deterministic. SIMD (`-msimd128`) ops count once for 16 bytes.
- Score: one point per submission: x = **max fuel of any single frame** (over all episodes), y = **mean Sim(3) ATE**.
  Claimed **area on a log-log plot** (fuel 1..2.4e10, ATE 0.001..3.23 m) beyond the course baselines. Several submissions = several points.
  Sim(3) alignment removes global scale/rotation/offset, so only the trajectory *shape* and *consistent scale* matter.
- Because only the worst frame counts, **per-frame cost must be flat**: spread work over frames.
- Limits: step fuel 2e10, init fuel 2e11, memory 2 GiB, submission ≤128 MiB.

## 2. What is submitted / current best (as of 2026-11-18)
| Point | Source in repo | Built wasm | Public (10 eps, house Train-1) | Unseen houses 192–194 (24 eps) |
|---|---|---|---|---|
| **Normal: translation_scale_normal_new** (hybrid, 81-weight net) | `3_deterministic_hybrid/models/translation_scale/solution.cpp` + `common/estimator.inc` | `examples/python/wasm/translation_scale_normal_new.wasm` | **0.400 m @ 97,838** | **1.004 m @ 96,156** |
| **Cheap: ultra (uf2)** (classical) | `examples/python/solution_ultra.cpp` | `examples/python/wasm/ultra_uf2.wasm` | **0.734 m @ 24,040** | **1.181 m @ 24,040** |
| optional 3rd: confidence_lite_new (hybrid, 46 weights) | `3_deterministic_hybrid/lite/models/measurement_confidence/` + `lite/common/estimator.inc` | `wasm/confidence_lite_new.wasm` | 0.605 m @ 38,735 | 1.200 m @ 38,735 |
| classical normal: v8_new | `examples/python/solution.cpp` | `wasm/v8_new.wasm` | 0.459 m @ 97,483 | 1.057 m @ 95,793 |
| classical lite: v8lite_new | `examples/python/solution_lite.cpp` | `wasm/v8lite_new.wasm` | 0.662 m @ 38,735 | 1.208 m @ 38,735 |

- Teammate's earlier submission `sub_500_mff` (distilled ML, `1_generative_ssm/`): **150 fuel, 1.93 m** (a learned trajectory shape).
- Area estimates (tools/area.py; baselines read off the plot, approximate): with teammate, translation_scale_new + ultra =
  **8.33% public / 3.82% unseen**. Best 2-point pick on both sets. + confidence_lite adds ~0.1% on public only.
  ultra dominates both lite models on unseen houses (more accurate at 38% less fuel).
- **Unseen houses are ~2× harder than the public house** for every model. Public numbers flatter everything.

## 3. The machine and environment
- Windows laptop ("DESKTOP-K5JJK4M"), user `ninad`, NVIDIA **RTX 3060 Laptop GPU, 6 GB**, Windows driver 592.00 (WSL reports 590.62).
  Ninad also runs SolidWorks on it; keep background jobs polite unless he asks for speed.
- **WSL2 Ubuntu 24.04** is where everything runs. Python venv: `~/vio/.venv` (Python 3.12.3, `viobench` + `hyperdrone 0.1.10` installed).
  viobench native workspace: `~/vio/ws` (from `viobench init native ws`; has `runtime/native/`, needed by builds).
  Windows folder from WSL: `/mnt/c/Users/ninad/Desktop/EECS116/Final_project` (writes to /mnt/c are slow; fine for results).
- **GPU episode rendering works in WSL via OptiX, with a workaround** (NVIDIA's WSL driver ships no Linux OptiX):
  - CUDA toolkit 12.9 installed in WSL at `/usr/local/cuda` (`nvcc` on PATH when you `export PATH=/usr/local/cuda/bin:$PATH`).
  - Linux driver 590.48.01 package unpacked in `~/optix_build/`; its `libnvoptix.so.1`, `libnvidia-rtcore.so.590.48.01`,
    `libnvidia-gpucomp.so.590.48.01`, `nvoptix.bin` copied to **`~/optix_wsl/`**. Nothing system-wide changed. Undo: `rm -rf ~/optix_wsl ~/optix_build`.
  - Run with `LD_LIBRARY_PATH=$HOME/optix_wsl:/usr/lib/wsl/lib:$LD_LIBRARY_PATH HYPERLUX_BACKEND=OPTIX`.
  - HyperLux build caches: `~/.cache/hyperlux/renderer-optix-a6f791f0` (good) and `environment-optix-aa1689b1`.
    **The renderer cache key includes the interpreter path**: run via `~/vio/.venv/bin/python3` (what `viobench` uses). Calling
    `.venv/bin/python` triggers a 30+ min full rebuild (`renderer-optix-6dd7f74b`, a half-built leftover, safe to delete).
  - Each new `PARALLEL_FLIGHTS` value triggers a one-time ~5–7 min environment compile (silent; watch `ps` for cmake/nvcc).
  - Scenes download once into `~/.cache/conta` (Hugging Face `conta-shards/procthor`).
  - GPU use shows ~15–30% in nvidia-smi: the GPU is not the bottleneck (CPU physics, PNG encoding, /mnt/c writes are). In WSL,
    `nvidia-smi` often lists no processes; check GPU-Util/memory. Task Manager needs the "Cuda"/"Compute_0" graph.
  - Throughput: 8 flights × 60 s per ProcTHOR house in ~5 min (≈5× the stock 3-at-a-time + video, ≈20× CPU rendering).
- Lab computers (native Linux, NVIDIA GPUs) run stock `viobench simulate` with OptiX directly.

## 4. Filesystem map (Windows: `C:\Users\ninad\Desktop\EECS116\Final_project`)
This folder is the **git repo** (origin `Spriggan999/116_Final_Project`, branch `main`). Old local branches
(`final-v8`, `ninad-vio-v8`, `ninad-vio-v8-clean`, `vio-condensed`) are stale.
```
Final_project/
├─ CLAUDE.md                     this file (untracked unless someone adds it)
├─ README.md, .gitignore         ignores data/, *.wasm, report-*/, build dirs, *.pt …
├─ 116_Final_Project/            OLD nested clone (Windows venv, branch ninad-vio-v8). Stale; ignore.
├─ reports_fresh/                first fresh-set reports of v7/v8/v8lite (old builds)
├─ viobench/                     course framework + all team solutions
│  ├─ docs/ (TASK.md, ABI.md, STUDENT_GUIDE.md, DEBUGGING.md, RUNTIME.md)   read TASK.md + ABI.md first
│  └─ viobench/
│     ├─ solution/runtime/native/viobench_abi.h, viobench_reactor.cmake      native ABI + build rules
│     ├─ tools/runner/sandbox.py, metrics.py, run.py                         evaluation (fuel + Sim(3))
│     ├─ tools/datasets/generate/simulation.py                               HyperDrone episode generator
│     └─ solution/examples/python/                                           ← THE SOLUTIONS
│        ├─ solution.cpp          v8_new  (single file; `#define VIO_LITE 0`)
│        ├─ solution_lite.cpp     v8lite_new (identical except `#define VIO_LITE 1`)
│        ├─ solution_ultra.cpp    ultra (uf2), single self-contained file
│        ├─ wasm/                 built: translation_scale_normal_new, ultra_uf2, confidence_lite_new, v8_new, v8lite_new
│        │                        (force-added with `git add -f`; *.wasm is gitignored)
│        ├─ 1_generative_ssm/     teammate's distilled-ML approach (sub_500_mff); many READMEs (README.md, MODELS_README.md …)
│        ├─ 2_deterministic/      teammates' unmodified copy of an older v8 (used as the hybrid's baseline)
│        ├─ 3_deterministic_hybrid/
│        │  ├─ common/estimator.inc            full-size estimator with HYBRID_MODE switch (+ fuel fixes)
│        │  ├─ common/network.h                tiny MLP inference
│        │  ├─ models/translation_scale/       solution.cpp (#define HYBRID_MODE 5 + includes), weights.h, checkpoint.pt, training.json
│        │  ├─ lite/common/estimator.inc       lite estimator (+ fuel fixes)
│        │  ├─ lite/models/measurement_confidence/   HYBRID_MODE 3, weights.h, …
│        │  ├─ train.py, train_fresh.py        training (scenes 2–191 allowed; 1 = public; 192+ reserved held-out)
│        │  ├─ benchmark.py, PRESERVED_MODELS.md
│        └─ dead_reckoning/, klt_imu/, tiny_cnn/, minimal/   course examples
└─ data/                         (gitignored) episodes, models, reports, scripts
   ├─ ep0000.zip + 9 hash-named *.zip      the 10 PUBLIC episodes (house ProcTHOR-Train-1), ~200 MB each
   ├─ fresh/      28 eps: Train-1 ×1 (seed 7), Train-2 ×3, Train-3/4/5 ×8 (seed 11)   = DEVELOPMENT set
   ├─ heldout/    24 eps: Train-192/193/194 ×8 (seed 23)                               = HELD-OUT set (never tune on it)
   ├─ models/         v8, v8lite, translation_scale_normal, confidence_lite: old and *_new .wasm
   ├─ models_ultra/   ultra_uf2.wasm (chosen), ultra_ue1.wasm (runner-up)
   ├─ candidates/     copies of the four fuel-fixed sources + solution_ultra.cpp (already in the repo)
   ├─ hyb/            shallow copies of hybrid sources (Claude's bridge cannot stage paths >7 folders deep)
   ├─ reports_fresh/ reports_heldout/ reports_ultra/   evaluate reports: <model>/<house>/report-<model>/report.json
   ├─ eval_fresh.sh, eval_heldout.sh (+ *_log.txt)     parallel evaluation scripts
   ├─ gpu_fast.sh, fast_sim.py (+ gpu_fast_log.txt)    GPU episode generation
   ├─ optix_test/, report-v1/, .spool/                 leftovers / temp frames
   └─ dev/                        experiment kit (see §6)
      ├─ experimental/            full-switch source: solution.cpp + params.h + CMakeLists.txt
      ├─ tools/                   devset.py, build_variant.sh, make_pickles.sh, load_real.py, area.py, oracle.py, robust.py, dirstat.py, phaseall.py
      └─ archive/                 v0 … v8, v8lite sources (+ wasm), references.md, solution_ultra_single_file.cpp
```
Gone from the laptop (folder was replaced by the clone): the old `v1`–`v8`, `tools`, `final`, `gpu` folders. Their content is in `data/dev/archive`.

## 5. Everyday commands (WSL, `source ~/vio/.venv/bin/activate`)
```bash
F=/mnt/c/Users/ninad/Desktop/EECS116/Final_project
P=$F/viobench/viobench/solution/examples/python
# build a single-file solution (copy into a folder with CMakeLists from ~/vio/ws/solution, or use dev/tools/build_variant.sh)
viobench bundle <solution dir> -o out.wasm
viobench check out.wasm                          # synthetic smoke test
viobench evaluate out.wasm                       # public set (downloads/caches in ~/.cache/conta); writes ./report-out/report.json
viobench evaluate out.wasm $F/data/heldout/*.zip # any episode zips
# all models in data/models (or MODELS=models_ultra) on held-out houses, 1 job per (model, house), in parallel:
MODELS=models REPORTS=reports_heldout bash $F/data/eval_heldout.sh      # JOBS=6 to leave CPU free
bash $F/data/eval_fresh.sh                                              # same for data/fresh
# generate episodes on the GPU (8 flights at once, no videos):
bash $F/data/gpu_fast.sh <outdir under data> <episodes per scene> <seed> "ProcTHOR-Train-200 ProcTHOR-Train-201"
#   VIO_PAR=12 for more flights at once (one-time recompile; watch 6 GB VRAM). Avoid scene 1 (public) and 2–17 (hybrid training).
```
Git (Ninad, Git Bash): `cd /c/Users/ninad/Desktop/EECS116/Final_project && git add <files> && git commit -m "…" && git push`.
From a Linux view of this Windows checkout, `git status` shows ~368 files "modified": CRLF line endings only.
Use `git -c core.filemode=false diff --ignore-cr-at-eol --stat` to see real changes; never `git add .` from there.

## 6. Experiment kit (`data/dev`)
- `experimental/` = v8 estimator with **every switch** in `params.h` (defaults = v8_new). Verified: default build reproduces v8_new,
  and these presets reproduce the shipped models exactly in accuracy (fuel +~300 for normal/lite from switch overhead):
  - lite: `-DN_TRACK=11 -DGRID_X=12 -DGRID_Y=8 -DN_EXTRA=4 -DMIN_TRACKS=5 -DSPLIT=4 -DKF_LEN=6 -DDIR_OWN=1 -DVPRIOR_IDLE=1 -DP_FLOAT=1 -DSUB_CAP=3`
  - ultra: `-DULTRA=1 -DULTRA_SPLIT=1 -DROT_FROM_VISION=0 -DRANK_SPLIT=4 -DSPLIT=3 -DKF_LEN=5 -DN_TRACK=9 -DMIN_TRACKS=4 -DGRID_X=12 -DGRID_Y=8 -DN_EXTRA=4 -DP_FLOAT=1 -DSUB_CAP=3`
  - profiling switches: `PROF_SKIP_TRACK/TEMPLATES/DIR/PROP=1`, `PROF_REP=<n>` (run one component twice to measure its fuel).
- Workflow (from `data/dev`):
  1. `bash tools/make_pickles.sh ../fresh pkl/dev` and `bash tools/make_pickles.sh .. pkl/public` (decode once; ~40 s/episode).
     Dev set used so far = Train-1, -3, -5 (17 eps); put only those in pkl/dev to compare with numbers here.
  2. `bash tools/build_variant.sh NAME "-DFOO=1 …"` → `builds/NAME.wasm` (needs `~/vio/ws/runtime/native`; `VIO_WS` to override).
  3. `python tools/devset.py builds/base.wasm builds/NAME.wasm` → mean ATE, max fuel, mean difference and wins vs the first; cached.
  4. Only promising changes: check on `data/heldout` with eval_heldout.sh (official `viobench evaluate`). **Do not tune on held-out.**
- `tools/area.py`: claimed-area estimate, `claimed([(fuel, ate), …])`. `oracle.py`/`dirstat.py`: diagnose direction vs speed error
  (`VIO_PKL` glob). `robust.py`: stress suite (late start, extra IMU noise, dim light, narrow FOV). `phaseall.py`: per-phase max fuel.

## 7. How the estimator works (classical core, shared by all our models)
- **18-state error-state EKF**: p, v, θ (attitude), b_a, b_g, p_kf (position clone at the keyframe). IMU predicts every frame
  (10 samples). Covariance propagated once per window (block-sparse, ~22k fuel); can be float (`P_FLOAT`).
- **Image**: never resized; a virtual 160×120 green image (2×2 average) is fetched on demand (SIMD shuffle).
- **Window (v8: 9 frames)**: frame 0 keyframe: score all candidate cells (v8 16×11 = 176; lite/ultra 12×8 = 96), each from
  **4×4 raw samples** → 9 gradient pairs → 2×2 structure tensor → **λ_min** score (corner-ness); contrast-scaled threshold;
  keep top N (v8 20, lite 11, ultra 9) + spares. Frames 1–3: clone + cut 8×8 templates, one subset per frame (`SPLIT`).
  `KF_LEN` frames later (v8 6, ultra 5) track each subset: gyro-predicted start + **3 Lucas–Kanade steps** (SIMD, bilinear,
  brightness offset), reject if >3 px from the prediction (`MAX_CORR_PX`).
- **Direction**: derotate with the gyro; normal nᵢ = (R fᵢ) × f′ᵢ per track; direction u = smallest eigenvector of Σ nᵢnᵢᵀ;
  robust refit dropping |nᵢ·u| > 3× median; reject if eigenvalue ratio q (smallest/middle) > MAX_Q or too few tracks;
  sign by cheirality + **forward prior** (`SIGN_FWD 0.7`: the generator's autopilot flies nose-first).
- **Speed**: parallax (derotated flow size) → |d| ≈ parallax · PAR_SPEED / f (assumes a typical room depth). Only scale
  *consistency* matters for Sim(3).
- **Updates**: 3 scalar rows (two e·d = 0 perpendicular to u, one uᵀd = |d|), ~3.6k fuel each; χ² gate; vision rotation check
  updates b_g (`ROT_FROM_VISION`, off in ultra); weak zero-velocity prior σ = 0.5 m/s once per window; small-parallax "hover" update.
- **Hybrid models** (teammates): same estimator; at each direction update an MLP sees 8 window features
  [parallax/f·100, √q·3, tracks found/N, kept/N, |p−p_kf|, |v|/5, |ω|, window time/0.3 s]:
  - `translation_scale` (8→8→1, HYBRID_MODE 5): multiplies the speed estimate by exp(out), clamped ×0.5–×2.
  - `measurement_confidence` (8→4→2, HYBRID_MODE 3, lite): scales measurement noise R of direction and displacement rows
    (lite clamp ×0.5–×4). Trained on ProcTHOR houses 2–17 (so fresh houses 2–5 are NOT a fair test for them; 192+ are).

### Fuel engineering that matters
- **Fuel fixes (all "_new" models)**: `SUB_CAP` caps templates cut per subset (normal 7, lite/ultra 3); the last subset used to
  "catch up" after texture failures on the same frame as the direction update → 104–115k spikes on unseen houses.
  `RANK_FAST`: top-K sorts only cells above the histogram threshold bin (bounded cost on low-texture frames).
  Result on unseen houses: max fuel 115k → 96k (normal), 51k → 39k (lite), accuracy −0.01 m (≈1%, within noise).
- **ULTRA schedule** (18 frames, 0.9 s): 0–3 rank ¼ of cells each; 4–5 propagate (F·P on 4, ·Fᵀ on 5: exact) + clone;
  6–8 cut 3 templates each; 9–10 propagate ahead (split); 11–13 track one subset each; 14 direction maths (rows deferred);
  15–16 apply rows (half each); 17 velocity prior. Gyro-bias vision check removed (it alone pushed tracking frames to 38k).
  Cost: vision update every 0.9 s instead of ~0.55 s; slightly stale ranking/rows. Fine for these indoor flights.
- Per-frame floor ≈ 5.5k (IMU + output). Measured costs: one EKF row ~3.6k, inject ~5k, propagation ~22k, direction maths ~15k,
  LK ~6–7k per patch, ranking 96 cells ~29k (16 cells ~12k), 4-cell SIMD score ~730.

## 8. Results history (public set unless noted)
| Version | Mean ATE | Max fuel | Note |
|---|---|---|---|
| v1 | — | 1.25 M | first EKF + LK |
| v2 | 0.630 | 966k | 160×120, parallax speed, cheirality |
| v3 | 0.597 | 1.0 M | vision gyro-bias check |
| v4 | 0.563 | 157k | fuel rebuild (SIMD LK, sparse scoring, split) |
| v5 | 0.545 | 158k | contrast-scaled threshold, robustness |
| v6 | 0.500 | 158k | process-noise retune |
| v7 | 0.512 | 99k | per-frame rebalancing |
| v8 | 0.467 | 99k | forward sign prior |
| v8-lite | 0.682 | 45k | cheap point |
| translation_scale (hybrid) | 0.406 | 99.6k | teammates |
| confidence_lite (hybrid) | 0.620 | 44.8k | teammates |
| **_new (fuel fixes)** | see §2 | 97.5k / 38.7k | |
| **ultra uf2** | 0.734 | 24.0k | |

Generalization (all mean Sim(3) ATE):
- Fresh dev set (Train-1,3,5; 17 eps): translation_scale_new 0.621, v8_new 0.689, confidence_lite_new 0.852, ultra_uf2 0.871, v8lite_new 0.884, ultra_ue1 0.943.
- Held-out (192–194; 24 eps): see §2. Old versions there: translation_scale 0.997 @115k, v8 1.043 @115k, confidence_lite 1.188 @51k, v8lite 1.198 @51k.
- Paired, held-out: translation_scale beats v8 on 20/24 flights (−0.046 m, se 0.019): **real gain**. confidence_lite vs v8lite: within noise.
  v8 vs v7 on fresh: within noise (the forward-sign gain was mostly public-set luck). Differences < ~0.02 m on 24 flights are noise.
- Speed sensitivity (synthetic): 0.46 m at ~0.7 m/s → 1.75 m at 1.9 m/s (±3 px LK search). All test flights are 0.3–0.7 m/s.

## 9. Tried and rejected (don't repeat without a new reason)
- Fixed-gain/LTI observer (needs a tilt loop; best 0.72 m, dominated: EKF cross-covariances matter). Comb filter idea: not useful.
- Adaptive R from innovations, Huber weighting, log-scale state: neutral or worse. Sideways displacement/velocity priors: worse/neutral.
- Long-baseline re-tracking (extra rows worse; blended neutral at +25k). Interleaved keyframe pipelines: worse (velocity shrinkage).
- Full-res tracking, 80×60 tracking, LK_ITERS 4, median parallax, fewer cells, smaller grids for lite: worse.
- Ranking on an earlier/idle frame (stale cells, even rotation-compensated): +0.04–0.15 m. Direction update one frame late: +0.04 m (normal).
- Parameter sweeps on fresh data (PAR_SPEED_REL, VPRIOR_SIG, KF_LEN, MAX_CORR_PX, MIN_EIG, N_TRACK, NN output gains): no gain on
  both dev and public; current values are near-optimal off the public set too.
- WSL GPU: Dozen/Vulkan not available on Ubuntu 24.04 Mesa; plain OptiX fails 7804/7805 without the `~/optix_wsl` workaround.

## 10. Open ideas (most promising first)
1. **Cheaper filter maths** (the floor of ultra): Schur-complement/state reduction (fewer states), or slower bias states.
2. **Stratified / coarse-to-fine cell scoring** for the normal models (frame-0 ranking is one of their heavy frames).
3. **IMU-predicted translation in the LK start** (fix fast-flight failures; cheap).
4. 2-point RANSAC for the direction (u = n₁ × n₂ hypotheses) against rare sign flips; costs fuel.
5. Retrain translation_scale with held-out-safe houses and evaluate on new held-out houses (e.g. 195+).

## 11. Gotchas
- Evaluate on **held-out houses 192+** before believing any gain; tune on fresh/dev only. Public set = one easy house.
- Claude's device bridge cannot stage files >7 folders below `Final_project` (copy to `data/` first); deletes in connected folders need permission.
- Never write shared scripts while a run is reading them (bash re-reads the file: "unexpected EOF").
- Build output `.wasm` files are gitignored; force-add (`git add -f`) only the ones meant for GitHub.
- Native code: no exceptions/abort (they trap); never output NaN; normalize quaternions; copy frame/IMU buffers if kept.
