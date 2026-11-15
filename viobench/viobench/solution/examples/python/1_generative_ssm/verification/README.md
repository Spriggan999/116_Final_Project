# Teacher and student resource profiles

The [ten-horizon review](student_ten_horizon_review.json) records the default
2/4/6/8/10/12/14/16/18/20-second schedule and recovery from older five-horizon
checkpoints without resetting current difficulty, readiness or holdout counters.
Forty-five checks passed; five GPU-only cases were skipped. Motion still resets
at each horizon extension; the sole-holdout exception remains active.

The [sole-holdout timeout review](sole_holdout_timeout_review.json) records the
five-completed-epoch exception to the shared staged student barrier. Seven new
checks cover exact timing, reused recordings, regression/identity/group guards,
recovery, horizon resets, stage-best restoration and W&B reporting. Sixteen
supervised regressions and four tracking checks passed; the offline W&B check
requires a host local socket. Ordinary plateau criteria and stage-best recovery
remain active. See the [supervised guide](../SUPERVISED_STUDENTS_README.md).

The [student alignment review](student_alignment_review.json) records the new
0.1-weight Sim(3) camera-trajectory auxiliary, stable detached transform fitting,
finite-difference/official-scorer agreement, degeneracy safeguards, CUDA updates
for all eight profiles and serial/concurrent equivalence. Training follows the
current horizon through complete 20-second flights; validation stays held out.
The [supervised guide](../SUPERVISED_STUDENTS_README.md) includes the existing
auto-parallel scheduler's tmux command. No long training run was launched.

The [unaligned MTE review](unaligned_mte_review.json) records complete-episode
mean translation error and the separate unaligned ATE RMSE diagnostic, using
camera timestamps, lever arms and official invalid-state filling. Both reach
student/teacher/PPO W&B logs. Eighteen metric/recovery checks and one streaming
SL+PPO/continued-PPO check passed. Selection/stopping criteria are unchanged.

The [loss component logging review](loss_component_logging_review.json) records
raw values, coefficients and weighted contributions for student/teacher state
losses, JEPA, correction KD, RGB/edge/flow and PPO. Reconstruction and gradient
invariance checks passed, along with eight-profile CUDA updates, checkpoint
recovery and embedding/RGB SL+PPO end-to-end checks. Telemetry uses detached
sample-weighted means. See the [metric guide](../LOSSES_README.md#component-logging-and-balancing-weights)
for W&B keys and local files. Existing weights and selection criteria are unchanged.

The [student JEPA review](student_jepa_review.json) records the default auxiliary
embedding objective with stop-gradient EMA targets, content/variance/covariance
regularization, checkpoint recovery, W&B components and CUDA checks for all eight
profiles. EMA weights are training-only buffers and are excluded from deployment
weights/offsets. Nineteen CPU checks, two CUDA checks and twelve checkpoint/tiny
native/WASM regressions passed. See [all loss functions](../LOSSES_README.md).
No new long training run or accuracy evaluation was launched.

The [1K nine-correction review](student_1k_nine_corrections_review.json) records
the current 169-parameter model: the same bounded gyro/body-acceleration/initial-
frame-acceleration outputs and generic integration path as larger students. It
measured **5,195/6,000 fuel/frame** over 1,200 WASM frames. Native/WASM agreement,
all nine correction semantics, gradients, legacy/CLI recovery and 20-second CUDA
agreement passed. The name `1k` remains for CLI compatibility; its export cap is
now 6,000. Earlier six-output orientation results below are historical.

The [1K orientation review](student_1k_orientation_review.json) records the new
82-parameter learned acceleration/gyro/vision heads and normalized quaternion
integration. The 1,200-frame WASM check measured **1,031/1,100 fuel/frame**.
Native/WASM parity, camera extrinsics/lever arm, changing attitude, learned gyro
gradients, legacy compatibility, CLI recovery and CUDA parallel 20-second
integration checks passed. Earlier 64-parameter 1K results below are historical.
The new model needs fresh training; no accuracy evaluation was performed.

The [100K student review](student_100k_review.json) records the new 4,417-parameter
CNN/embedding-forecaster/RNN student: **84,184/100,000 fuel per frame** over a
1,200-frame WASM resource check. All eight default profiles are integrated into
the sequential launcher and CUDA scheduling checks. The focused 100K, CLI recovery
and launcher suite passed seven tests. These are correctness/resource checks;
no new long training run or accuracy evaluation was launched.

The [1K CNN review](student_1k_cnn_review.json) records the replacement of the
default scalar 500 tier with a real spatial CNN: 4×4 gray grid, two shared 2×2
filters, 2D embedding and an 11→2→2 forecaster. Its 64 parameters completed a
1,200-frame WASM check at **750/1,000 fuel**, native/PyTorch parity and BF16 CUDA
training. Older 500 checkpoints remain compatible with their saved caps.

The [learned student frontend review](student_frontend_review.json) records all
seven new encoder/embedding-forecaster/state pipelines, native/WASM parity and
CUDA/BF16 ground-truth training checks. Every new model passed a 1,200-frame
resource check; the 500 tier uses a scalar substitute because the tested general
full-pose CNN alternative exceeds its cap. See the
[architecture comparison](../STUDENT_ARCHITECTURES_README.md) for shapes,
parameters, cadence differences, measured fuel and the new sequential order.

The [ten-level curriculum review](supervised_ten_level_review.json) records the
current supervised policy: ten motion/IMU/vision levels, 75% current and 25%
all-earlier-category training replay, and randomized IMU amplitudes and blur.
Validation has no easier-category replay. The latest supervised suite passed
all 16 checks; the broader legacy curriculum checks and host worker reruns are
recorded separately in that review.

The [supervised student review](student_supervised_review.json) records the
ground-truth-only path, plateau-only staged curriculum, 41 passing regression
tests and one real CUDA training/validation epoch across all seven architectures.
This is a bounded correctness check, not an accuracy/convergence measurement.
Use [the supervised guide](../SUPERVISED_STUDENTS_README.md) to train the eight
models sequentially from fresh weights.

The [two-scene six-second screening](../checkpoints/heldout-6s-eval-epoch64/README.md)
evaluates saved student deployment checkpoints and the exported teacher on the
same held-out ProcTHOR/HSSD episode pair. The snapshot was taken at completed
epoch 67 (the output folder name predates the pause). Training resumed afterward.
See its [comparison table](../checkpoints/heldout-6s-eval-epoch64/results/comparison.md)
and [chart](../checkpoints/heldout-6s-eval-epoch64/results/pareto.png).
The area calculation uses the website implementation, but comparison against
the supplied baseline table is approximate because the episodes differ.
`evaluate_heldout_snapshot.py` reproduces the snapshot/evaluation workflow;
pass `--teacher PATH` to include a teacher WASM and use a fresh output directory.

The [latest training and capacity review](training_optimization_review.md) covers
CUDA input-memory reductions, parallel training physics, vectorized PPO, HSSD
empty-mesh recovery and the full resource check for a 96-dimensional candidate.
Its scoped measurements are separate from the historical reports below.

Fresh training and `full_resolution.json` now default to the verified 896-wide
official Mamba-2/local-attention hybrid with direct embeddings. `gru.json`
selects the previous GRU alternative. Historical reports below retain their
measured architectures; GRU throughput figures do not measure the new hybrid.

## Expanded collection defaults

For the subsequent W&B run analysis, CPU/GPU utilization plots, early curriculum
stopping diagnosis and revised training preset, see
[2026-11-13 W&B curriculum review](wandb_curriculum_20261113/README.md).
The review includes raw read-only exports, a reproducible summary script, and a
CPU input-only benchmark; it does not claim an end-to-end GPU speedup.
The review also includes a [recording I/O benchmark](wandb_curriculum_20261113/README.md#collection-write-optimization-for-future-runs)
for the faster lossless compression enabled in future streaming runs. Its NPZ
write-path measurement excludes simulation, downloads and model training.

Fresh training and checkpoint restarts use a pool of 192 training scenes and
eight fixed held-out scenes from Train200 (indices 0–191 and 192–199). Each
rolling training batch requests four scenes and two flights per scene; two
flights run simultaneously within one scene. Validation requests one flight
per held-out scene. The full native episode limit is 2,000 samples at 200 Hz,
about 10 seconds, replacing 401-sample/two-second recordings. Native termination
can still shorten a flight. Collection is capped at 3,072 MiB across current,
prefetched and validation slots; validation fails clearly if its budget cuts
off held-out scenes. Explicit JSON/CLI overrides remain available.

`expanded_collection.json` records a full collection and CUDA workflow probe
on the RTX 5080 / 64 GiB host. All eight training flights in scenes 0–3 and
all eight held-out flights in scenes 192–199 reached 2,000 samples and 500
camera captures at 640x480. The training recordings took 48.65 seconds and
708.6 MiB; validation took 57.28 seconds and 296.9 MiB. Each directory fits its
roughly 1 GiB slot. Two full training directories plus validation would use
about 1.67 GiB of the 3 GiB budget for these measured scenes.

RGB is retained as uint8 in `EpisodeStore`, then normalized when frame pairs
or camera windows are sampled. These 16 flights occupy 6.87 GiB of image
storage instead of 27.47 GiB as float32. The training process peaked at about
10.02 GiB RSS; this is not a sum of worker/renderer RSS. Initial dataset aliases
no longer retain the first training batch across the whole run. Legacy resized
recordings retain their float interpolation behavior.

A real embedding forward/backward/AdamW update and an official 896-wide
hybrid state update both passed with finite loss/gradients. PyTorch peak
allocated CUDA memory was 230.6 MiB and 802.2 MiB respectively; these allocation
figures exclude the simulator, driver context and allocator reservations.
The state check used the default batch four, 101-tick windows and two spawned
loader workers. Longer recordings provide more windows; they do not change
model width or sequence length. This establishes workflow/resource compatibility,
not an accuracy improvement or exhaustive profiling of all 200 environments.

Tests additionally cover exact uint8/float sampled-pixel agreement, embedding
and distillation preprocessing, training/validation scene separation over a
complete rotation, partial-validation rejection and restoring current collection
defaults from older checkpoints.

From `1_generative_ssm`, after collecting the two recording directories:

```bash
python verification/benchmark_collection.py --train TRAIN_DIR --validation VALIDATION_DIR --output /tmp/expanded-collection.json
```

`distill.py --collection-config FILE.json` accepts collection-setting overrides
without restoring the teacher's old small scene pool or flight lengths. Its
`--max-data-mib` flag takes precedence over the collection JSON budget. Manual
collection has the same 192-scene/2,000-sample/3,072-MiB defaults; the output
budget can stop a one-off collection before every scene has been visited.

## Official Mamba-2 parity and hybrid limits (2026-11-12)

The custom PyTorch Mamba-2 and backend/export toggles have been removed. The
GRU/hybrid architecture toggle remains; `hybrid.json` selects the verified
896-wide embedding hybrid. Only official `mamba_ssm.Mamba2` constructs host
Mamba layers. The WASM guest retains its necessary NumPy recurrence and loads
those exact weights; CUDA kernels are not converted into WASM.

`mamba_cuda_parity.json` and `mamba_cuda_parity_512.json` record RTX 5080 checks
with torch 2.15.0+cu132, Triton 3.9, official Mamba 2.3.2.post1 and
causal-conv1d 1.7.0. FP32 checks disable PyTorch TF32 and use Triton IEEE dot
products, with predeclared `atol=5e-5, rtol=5e-4`. Batch two and lengths 11,
101, 257 and 513 cover short sequences, nonmultiples of the scan chunk and
cross-chunk state. Hybrid checks use three Mamba blocks, four attention heads,
state size 16, head dimension 32, a bounded 16-token attention history, and
nonzero output heads. A real backward/AdamW update checks finite gradients and
repeats the deployment comparison after updating the weights.

| Comparison | Largest measured absolute error |
|---|---:|
| Official FP32 scan vs official recurrent step | 9.54e-7 |
| Official 896-wide hybrid tokens vs NumPy deployment | 8.94e-6 |
| Official 896-wide corrections vs NumPy deployment | 3.87e-7 |
| Official 896-wide tokens vs deployment after optimizer update | 8.34e-6 |
| Retired PyTorch block vs identical official weights before removal | 9.83e-7 |
| Real WASM vs official streaming, 512-wide, 100 camera frames | 4.39e-5 |
| Real WASM vs official streaming, 896-wide, 100 camera frames | 1.59e-4 |

The archived fallback comparison is in `mamba_retired_parity.json`. Actual
WASM results in `mamba_wasm_parity.json` cover the real-image CNN, direct
embedding prediction, ten IMU ticks per frame, recurrent state, local attention
and integrated 13D camera pose/twist, with `atol=rtol=5e-4`. Quaternion signs
are aligned before comparison. Floating-point integration rounding can
accumulate over longer trajectories; these 100-frame comparisons are not an
accuracy evaluation or a guarantee for every trajectory/kernel version.

An initial scan check with Triton's default reduced-precision FP32 dots failed
four elements at length 101 (maximum difference 9.69e-5). PyTorch's TF32 flags
alone did not control these Triton dots. Setting `TRITON_F32_DEFAULT=ieee`
brought scan/step error below 1e-6 without loosening the tolerance. The hybrid
factory now sets that default before importing Mamba; BF16 neural training
remains available. Explicit environment overrides require their own validation.

`hybrid_resources.json` records **1,200 actual WASM camera frames** (60 simulated
seconds) under every supplied cap, including table, init/step fuel, memory,
wall-clock and upload limits. Image size is 640x480; the learned CNN and direct
embedding predictor use the current defaults. Models use seed 1701, with
nonzero randomly initialized output heads; checkpoint weights strict-load into
the official CUDA modules used by the camera/IMU parity checks.

| Hybrid width | Upload | Step fuel max | Memory max | Init fuel | Step wall max | Episode elapsed |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 71.87 MiB | 3.003 G | 141.2 MiB | 2.041 G | 0.237 s | 249.9 s |
| 896 | 118.01 MiB | 8.620 G | 325.6 MiB | 4.377 G | 0.746 s | 672.4 s |
| 1,792 | FP32 weights alone: 274.42 MiB | — | — | — | — | Rejected before export |

A final export through the official-only factory passed the host guest shim.
All 40 state arrays and 24 embedding-predictor arrays exactly matched the
resource-tested export, and the visual ONNX graph was identical. Archive
metadata and the ONNX producer-version string changed across exporter versions.

The 896-wide profile leaves about 10 MiB upload margin and uses 43.1% of the
20 G step allowance. It is a verified practical profile, not a proof of the
largest possible model or an accuracy improvement over GRU. A longer evaluation
episode must be timed separately against the 24-minute cap. Retrained weights,
architecture changes and runtime changes need fresh verification.

From `1_generative_ssm`, with the official CUDA dependencies installed:

```bash
python verification/verify_mamba_cuda.py --hidden 896 --output /tmp/mamba-parity.json
python train.py --config hybrid.json --device cuda --stream-data --verify-limits
# For a trained checkpoint and its already bundled WASM:
python verification/verify_mamba_wasm.py --checkpoints CHECKPOINT_DIR --module MODEL.wasm --output /tmp/mamba-wasm.json
python verify_limits.py MODEL.wasm --frames 1200 --report /tmp/hybrid-limits.json
```

Regression validation also passed 105 CPU tests (12 CUDA/dependency skips).
The four affected CUDA test modules passed 55 cases initially; after adapting
two test assumptions to the official cache API and supported hidden widths,
both failed cases and the GRU PPO companion passed in a three-case rerun.
This covers 57 distinct CUDA-environment cases, including synthetic supervised
training/checkpoint re-export, embedding scan/stream agreement, PPO updates,
legacy-setting restoration and rejection of removed toggles before collection.

The verified wheels and their dependencies were installed into the existing `myenv`
training environment; a direct CUDA forward/backward passed without temporary
`PYTHONPATH` additions. PyTorch and CUDA versions were preserved.

The build used pristine official package sources. This host's PyTorch 2.15
headers require C++20 while upstream Mamba's setup supplied `-std=c++17`.
Compiler launchers replaced that flag with `-std=c++20`, and an NVCC launcher
kept the RTX 5080 `compute_120` target to avoid building unrelated GPU targets.
No model equations or CUDA/Triton kernel sources were changed. Built wheel
SHA256 values were `465b622ab648a7643bbec64e0d84190b4a252782a90614408ac07ff705844d65`
(Mamba) and `5f281f91500c13e21e9f34838a35220c2df112a4d752279e8ec6666cb8339b30`
(causal-conv1d). Dependency installation is optional for GRU; ordinary package
installation needs a compatible PyTorch/CUDA/C++ toolchain. The official
[source/build instructions](https://github.com/state-spaces/mamba) describe the
host CUDA package; `requirements-mamba.txt` pins the versions checked here.

## Capacity headroom review (2026-11-12)

`capacity_headroom_review.json` records larger architecture candidates; these
are resource experiments, and the training defaults are unchanged. CUDA
training improvements alone do not change WASM caps. The embedding architecture
created deployment savings, leaving the FP32 Python teacher primarily limited
by upload size: the current export is 123.49 MiB against 128 MiB, despite only
3.97 G step fuel in its short check.

The working teacher candidate is GRU 1,824 / predictor hidden 256, which keeps
more upload margin than the near-boundary 1,840-wide variant. Larger student
candidates are a 6x3 grid for 10K, hidden 216 for 1M, and hidden 2,048 for 100M;
the 100-fuel model already uses 99 fuel. These choices are candidates for
retraining and held-out accuracy comparison, rather than promoted defaults.
The [getting-started guide](../README.md#choose-model-size-against-deployment-limits)
shows a fresh-run teacher configuration; the [model guide](../MODELS_README.md#embedding-teacher-and-student-capacity-planning)
explains the binding limits and student profile settings.

| Candidate | Binding resource measured | Scope |
|---|---:|---|
| Teacher GRU 1,824, embedding hidden 256 | 126.16 MiB upload; 4.107 G step fuel | 3 frames |
| Teacher GRU 1,840, embedding hidden 256 | 127.50 MiB upload; 4.176 G step fuel | 3 frames |
| Teacher GRU 1,848 | 128.18 MiB: rejected | Upload check |
| Teacher GRU 1,824, embedding hidden 2,048 | 127.53 MiB upload; 4.173 G step fuel | 3 frames |
| 10K student, 6x3 visual grid | 9,705 / 10,000 fuel (97.05%) | 1,200 frames |
| 1M student, hidden 216 | 967,184 / 1,000,000 fuel (96.72%) | 1,200 frames |
| 100M student, hidden 2,048 | 98,058,684 / 100,000,000 fuel (98.06%) | 1,200 frames |

The 100-fuel kernel already uses 99 fuel and has no useful enlargement margin.
A 5x4-grid 10K candidate passed at exactly 10,000 fuel, leaving zero margin;
the 6x3-grid candidate is a more practical experiment. A hidden-2,080 100M
candidate exceeded its fuel tier. The 1,200-frame student probes enforced
upload, init/step fuel, memory, table and wall-clock limits as well.

Teacher checks used the smoke-trained encoder plus random new state/predictor
weights; student checks used random weights with seed 1701. None establish
accuracy improvements. Compare retrained candidates on held-out trajectories,
and run full verification on the selected trained teacher/student exports.
Larger upload utilization is not itself an accuracy objective. If the teacher
is used exclusively offline for distillation, submission caps apply to the
deployed student rather than that offline teacher.

## Direct embedding prediction default (2026-11-12)

`embedding_prediction.json` records the new `--visual-prediction embedding`
default and its retained `--visual-prediction rgb` alternative. The direct path
uses an IMU-conditioned 48D predictor, an online real-image encoder and a
stop-gradient EMA target with content/variance/covariance regularization. The
model guide describes its scope relative to [MC-JEPA](https://arxiv.org/abs/2307.12698).

On the RTX 5080, the frozen full-size GRU teacher inference probe reached
20.17 windows/s with RGB generation and 91.19 with embeddings (4.52x), using
batch four and 101 ticks. Peak allocated CUDA memory fell from 2,533.6 to
445.5 MiB. The visual sequence shrank from `[4,101,3,480,640]` / 1,420.3 MiB
to `[4,101,48]` / 0.074 MiB. Real-image encodes per step fell from 404 to 44.
These measurements include visual prediction/encoding, state inference and host
transfers; they exclude backward, optimizer, simulator, loaders and accuracy.

The matched four-student distillation probe reached 17.39 windows/s for the
RGB teacher and 26.02 for embeddings (1.50x), with parallel scheduling enabled
in both cases. Peak allocated CUDA memory fell from 2,761.6 to 877.8 MiB. That
probe includes shared teacher targets, student forward/backward and fused AdamW,
but excludes simulator/loaders/validation/export. Each comparison uses two
warmup minibatches and three repetitions of eight measured minibatches.

Real Hyperdrone/CUDA smoke checks also completed two embedding-pretraining
epochs, two state SL epochs, one PPO iteration and parallel distillation into
all four tiers. SL, PPO-selected and hold-embedding exports passed the official
guest shim. Teacher checkpoint hashes were preserved by distillation, all
checked weights were finite FP32, and managed recordings were cleaned up.
These tiny runs establish workflow compatibility, not learned VIO accuracy.

The selected full-width embedding teacher's WASM passed a **three-frame** check:
123.49 MiB upload, 3.970 G maximum step fuel, and 347.6 MiB observed linear
memory. This is a short smoke check, not the default 1,200-frame resource check.
Verify each trained model with `--verify-limits` and evaluate real trajectories.

From `1_generative_ssm`, reproduce the synthetic comparisons on CUDA:

```bash
python verification/benchmark_visual_prediction.py --output /tmp/visual-prediction.json
python verification/benchmark_distillation.py --visual-prediction embedding --output /tmp/distill-embedding.json
python verification/benchmark_distillation.py --visual-prediction rgb --output /tmp/distill-rgb.json
```

The following older reports measure the optional RGB architecture.

## Further GPU pipeline improvements (2026-11-12)

`gpu_pipeline_followup.json` records the next round of measurements on the RTX
5080. Fused CUDA AdamW increased the frame-training probe from 863.67 to 975.18
pairs/s (1.13x). Both used batch eight, BF16, 640x480 inputs, twenty warmup steps
and 1,000 measured forward/backward/optimizer steps. Peak allocated CUDA memory
remained about 680 MiB. The shorter state-training comparison changed little.
These synthetic inputs and random starting weights provide no accuracy result.

In matched three-epoch runs with actual Hyperdrone background collection,
persistent loader workers reduced the mean of epochs two and three from 4.88
to 3.08 seconds (37% less time, 1.58x throughput). Each epoch contained only two
training windows and one validation window, using batch one, 101 ticks, the
full-size frozen teacher, all four students, two loader workers and a 512 MiB
recording budget. The timings include validation/data wait and exclude initial
collection setup/checkpoint/export. Tiny epochs emphasize startup; this is not
a claim of the same gain for longer epochs. Each run loaded fresh training
recordings at every epoch, preserved teacher hashes and cleaned up owned data.

The producer now fetches/collates its next minibatch as well as generating
targets. Shared inputs transfer once, and student BF16 weight casts are reused
across camera ticks within each forward pass. Compute-only distillation stayed
around 17 shared windows/s in the short follow-up probes. The clearer benefit
comes from reducing gaps between epochs and fusing frame optimizer updates.

Reproduce the optimizer comparison from `1_generative_ssm`:

```bash
python verification/benchmark_training.py --stage frame --pairs 8 --batch 4 --chunk 16 --amp --warmup 20 --steps 1000 --no-fused-adamw --output /tmp/frame-ordinary.json
python verification/benchmark_training.py --stage frame --pairs 8 --batch 4 --chunk 16 --amp --warmup 20 --steps 1000 --output /tmp/frame-fused.json
```

Both `train.py` and `distill.py` accept `--no-fused-adamw` and
`--no-persistent-workers`. CPU keeps ordinary AdamW and zero loader workers.
Regression checks cover optimizer parity, delayed loader overlap, bounded
prefetch, worker retirement on recording replacement, run cleanup, CUDA
precision boundaries and native export parity. Fused/BF16 arithmetic and worker
RNG progression can change rounding or training order; the objective, model
architecture and export ABI are preserved.

## Parallel distillation throughput (2026-11-12)

`distillation_performance.json` compares serial and parallel scheduling on the
RTX 5080 with the full-size flow/GRU teacher and all four student tiers. Both
use BF16 neural layers, batch four and 101-tick synthetic windows. After two
warmup minibatches, three repetitions of eight measured minibatches reached
11.44 shared windows/s serial and 17.02 parallel (1.49x), with peak allocated
CUDA memory of 2,704 and 2,802 MiB. Targets are computed once per minibatch.
The comparison includes transfers, teacher target generation and student
forward/backward/AdamW. It excludes collection, loaders, validation and export;
random weights/stationary synthetic inputs provide no accuracy evidence.
Allocated GPU memory excludes renderer/context and allocator reservations.

Reproduce on a CUDA host from `1_generative_ssm`:

```bash
python verification/benchmark_distillation.py --visual-prediction rgb --output /tmp/distillation-performance.json
```

The regression suite additionally checks bounded prefetch, failure cleanup,
serial/parallel update equivalence, independent profile stopping and neural
BF16 versus physics/validation FP32 on CUDA.

The report also records two complete CLI epochs with the full-size frozen
teacher, all four students, real 640x480 Hyperdrone recordings, two collector
flights and two spawn loader workers. Both epochs had fresh recordings ready
at the boundary. Teacher hashes were preserved, checkpoints were finite FP32
and owned recordings were cleaned up. That tiny run has only two training
windows per epoch, so loader startup and validation dominate its inclusive
throughput; it is a lifecycle check rather than a speedup measurement.

## Native student resource probes

`student_100_resources.json` records the image-free, six-coefficient affine IMU
model: all 1,200 frames passed the literal 100-fuel cap, with a peak of 99. It
uses nontrivial test coefficients, fixed 20 Hz/ten-sample packets, fixed attitude,
and the last accelerometer sample. It ignores RGB and gyro. This is a highly
restricted sensor-based motion model; there is no accuracy result here.

`student_10k_resources.json`, `student_1m_resources.json` and
`student_100m_resources.json` record complete 1,200-frame native student probes
using untrained PyTorch weights (seed 1701). Maximum fuel per frame was 7,886,
803,632 and 80,073,240 respectively, each enforced against its own 10,000 /
1,000,000 / 100,000,000 cap. All other supplied limits were enforced too.
These demonstrate the native architectures' resource behavior; they do not
measure retained teacher accuracy. Distillation/export verifies trained weights
again and saves a separate `limits.json` alongside each native student.

`student_distillation_smoke.json` additionally records a complete one-epoch
distillation and native export of all four tiers, using a tiny untrained teacher
and synthetic recordings. Each trained export passed 1,200 frames against its
own cap. This checks the end-to-end workflow, not useful model accuracy.

## Teacher capacity search

`teacher_capacity_search.json` records ten historical RGB architecture probes using
640x480 RGB, learned 8x8 projections, different flow widths and GRU sizes.
The selected RGB preset is flow base 33 / GRU hidden 1,792: 20,428,876
parameters, 126.26 MiB bundled, and 19.514 G maximum step fuel in the short check.
Flow base 34 left less than 2% fuel headroom; bases 35 and 36 exceeded 20 G.

These probes measure untrained architectures, using random weights and nonzero
output heads with seed 1701. They are not accuracy measurements. The upload and
fuel margins are intentionally small; verify each trained export against the
same limits, and repeat verification when changing the runtime or configuration.

## Selected large teacher: full episode

`large_teacher_resources.json` records the selected preset's complete
1,200-frame synthetic check. It passed in 994.20 seconds (16 min 34 sec), with
maximum step fuel 19.514 G, observed linear memory 410 MiB, and a 126.26 MiB
module. Initialization used 4.865 G and 0.361 seconds; maximum step time was
1.123 seconds. The table cap was enforced. The module and runtime hashes are
recorded. These are untrained test weights, not an accuracy result.

## Earlier smaller baseline

`full_resolution_resources.json` records a 1,200-frame synthetic episode
(60 simulated seconds at 20 Hz), measured locally on c204-17 using the existing
VIOBench WASI runtime and Wasmtime 50.0.1. The 640x480 teacher used learned 8x8
input projections, the base-24 flow generator and a 96-unit GRU.

Weights were randomly initialized with nonzero flow/correction output heads
(seed 1701). This tests architecture resource use; it is not a trained model
or a trajectory-accuracy result. The temporary `.wasm` path and its SHA-256 are
included for traceability. Only this small report is saved in the repository.

All 1,200 steps passed under the stated fuel, memory, table and wall-clock caps.
The run took 560.02 seconds; maximum step fuel was 12.378 G, maximum observed
linear memory was 168.625 MiB, and the module was 50.67 MiB.

Run `train.py --verify-limits` to profile the actual selected checkpoint after
training. It saves `teacher.wasm` and `teacher-limits.json` in that run directory.
Validate on real evaluation episodes as well. Server submission count/reset
policies and model accuracy are outside this synthetic resource check.

## Training throughput optimization probes (2026-11-12)

`training_performance.json` records RTX 5080 measurements of the full-size teacher.
Synthetic training throughput changed from 293.60 to 940.99 frame pairs/s (3.20x)
and 3.26 to 12.24 state windows/s (3.75x). The selected defaults use state batch
four, pair batch eight, generation/visual chunks 16 and selective BF16. Peak
allocated CUDA memory was 680 MiB in the frame probe and 2,682 MiB in the state
probe. Batch eight reached 15.52 windows/s using 4,259 MiB. These allocations
exclude the renderer, GPU context and allocator reservations.

The probe synchronizes CUDA, warms up once, and measures three optimizer steps.
Frame inputs/targets are identical random images and the state probe uses a
synthetic stationary IMU sequence. It exercises forward/backward/AdamW and online
frame generation; it does not measure learned accuracy or complete epochs.
Actual loader probes include startup/transfers: state workers two reached
10.51 windows/s versus 7.50 with zero workers; frame workers zero were faster
for their simple slices and remain the default. These short measurements depend
on cold startup and are not guarantees for longer runs.

Warm one-scene collection produced approximately 430 captures/s with one flight
and 427 captures/s with two; batched collection did not improve rendering
throughput on that scene. Both settings produced four independent 401-tick
episodes. Renderer setup and NPZ compression are excluded from these rates.

A complete one-epoch GPU SL smoke run on real simulator recordings passed,
including validation and post-SL checkpoint saving. Separate CUDA checks verified
BF16 neural outputs, finite gradients, FP32 geometry/validation/checkpoints, and
FP32 PPO features. A one-iteration GPU PPO-only warm start and one-epoch GPU
distillation into all four student tiers also completed. PPO correctly retained
the SL baseline when its update worsened held-out validation. The CPU regression
suite additionally covers causality, flight
termination, rolling-data/reuse policies, PPO, distillation, ONNX/guest conversion
and native student math. These are correctness checks, not ATE evaluations.

From the model folder, reproduce the synthetic default probe:

```bash
python verification/benchmark_training.py --source . --output /tmp/hyperdrone-performance.json --batch 4 --pairs 8 --chunk 16 --amp
```

Compare an earlier checkout with `--batch 1 --pairs 1 --chunk 1` (omit `--amp`).
Using those flags on the optimized source is a current FP32 ablation, not a
reproduction of the previous implementation.
