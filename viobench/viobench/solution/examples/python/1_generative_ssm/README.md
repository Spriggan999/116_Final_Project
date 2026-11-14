# HyperDrone: commands and getting started

This guide explains how to train the full-resolution teacher with SL followed by
PPO, produce `.wasm` submissions for both stages, and evaluate them.

For architecture, losses, coordinate systems, and implementation details, read
[MODELS_README.md](MODELS_README.md).

For SSH access from your personal computer, persistent training sessions, and
remote monitoring or stopping runs, read [SSH_README.md](SSH_README.md).

The latest [training optimization and capacity review](verification/training_optimization_review.md)
covers lower-memory camera processing, parallel training physics, vectorized PPO,
the HSSD empty-mesh repair, and a resource-checked optional 96-dimensional
embedding model. Defaults retain 48-dimensional checkpoint compatibility.

## Start here

The usual workflow is:

1. Activate the Python environment and enter this folder.
2. Train with background data generation.
3. Export, bundle, and check the trained model.
4. Evaluate accuracy on actual benchmark episodes.
5. Optionally [distill the teacher into 100/10K/1M/100M fuel students](#14-distill-teachers-into-fuel-budgeted-students).

**Run commands one line at a time.** Do not copy the terminal prompt or add `\`
to the end of a command. Wait for a command to succeed before continuing.

## 1. Activate the environment and enter the correct folder

On this lab computer, an environment named `myenv` already exists:

```bash
eval "$(conda shell.bash hook)"
conda activate /home/cc/eecs116/fa26/class/eecs116-aaq/.conda/envs/myenv
cd /home/cc/eecs116/fa26/class/eecs116-aaq/Documents/project_3/viobench/viobench/solution/examples/python/1_generative_ssm
pwd
python --version
```

`pwd` should end with **`viobench/solution/examples/python/1_generative_ssm`**.
All later commands in this guide assume you are in that folder, unless stated
otherwise. Keep `myenv` activated so the native build uses its compiler/CUDA settings.
On another computer, substitute the paths and environment name for that installation.

To use the existing interpreter explicitly:

```bash
HD_PYTHON=/home/cc/eecs116/fa26/class/eecs116-aaq/.conda/envs/myenv/bin/python
"$HD_PYTHON" train.py --help
```

You can replace `python` with `"$HD_PYTHON"` in later commands. Still activate the
environment for native simulator/runtime builds. The variable lasts only in the
terminal where you set it.

## 2. Install dependencies once

Skip this section if the environment is already set up. These commands temporarily
switch to the repository root for installation, then return to the model folder:

```bash
cd /home/cc/eecs116/fa26/class/eecs116-aaq/Documents/project_3/viobench
python -m pip install --no-cache-dir -e '.[train]'
python -m pip install --no-cache-dir -r viobench/solution/examples/python/1_generative_ssm/requirements-train.txt
cd viobench/solution/examples/python/1_generative_ssm
python -m pip install --no-build-isolation -r requirements-mamba.txt
```

`--no-cache-dir` avoids keeping another copy of downloaded installation packages.
The simulator needs a working GPU/render environment and native build tools.
The first simulation and first WebAssembly runtime build may spend substantial
time compiling dependencies.

Check GPU access from your activated terminal:

```bash
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

The default official Mamba-2 hybrid requires CUDA and its Mamba dependencies,
already installed in this machine's `myenv`. Training selects CUDA automatically
when available. For GRU, including CPU training, pass `--config gru.json`.

## 3. Recommended: train, export, bundle, and verify

Fresh training defaults to **direct visual embedding prediction**, with full
**640x480** real camera inputs and the **896-wide hybrid Transformer–Mamba-2**. The image encoder runs
on real captures; an IMU-conditioned predictor supplies 48-value embeddings
between captures. The state estimator consumes those features directly.
[`full_resolution.json`](full_resolution.json) pins this default. The command
below verifies the actual trained export; the older flow/RGB resource figures
in the model guide describe the optional RGB architecture.

```bash
python train.py --stream-data --max-data-mib 3072 --verify-limits
```

This command:

- Collects initial training and fixed validation recordings at **640x480**.
- Generates the next training batch in a background process while learning.
- Replaces old generated batches after they are consumed.
- Trains the embedding predictor/encoder, then the recurrent state estimator.
- Saves a complete post-SL checkpoint set in `checkpoints/RUN/post_sl/`.
- Refines the correction head with PPO, preserving the post-SL checkpoint set.
- Saves the best models and configuration locally.
- Exports deployment assets into `solution/`.
- Bundles `teacher.wasm` and runs a 1,200-frame synthetic resource check.

New collection defaults use **313 training scenes** (192 ProcTHOR Train200 + 121
HSSD Train125), with **12 separate held-out scenes** (8 ProcTHOR + 4 HSSD)
for validation. Each rolling training batch requests four scenes and two
flights per scene, with two flights simulated together. Flights now use up to
**2,000 IMU samples / approximately 10 seconds** by default. The simulator's native
limit is informational; larger `COLLECT_STEPS` values allow longer flights, while
crashed slots still stop independently. Validation requests one flight in each of
the 12 held-out scenes. The 3 GiB recording budget is split between current data,
one prefetched batch and fixed validation. Loaded RGB stays uint8 until sampled,
so longer recordings do not expand into full float32 image stores in host RAM.

These are defaults for fresh runs and checkpoint restarts; explicit JSON/CLI
collection overrides still win. Longer recordings increase available training
windows; the recurrent window remains 101 ticks. The scene pool is rotated over
successive batches, rather than loading all 313 environments simultaneously.
The final batch in a scene-pool cycle may contain fewer than four scenes.
See the [expanded-collection hardware check](verification/README.md#expanded-collection-defaults)
for measured recording sizes, host RAM and CUDA workflow validation.

Fresh runs default to **SL + PPO**. Runs with `--from-checkpoints` default to
**PPO-only**, including runs where both components have only partially completed
SL. Use `--sl` to train with SL again, or `--no-ppo` for SL-only.
If embedding training has already completed, use `--from-checkpoints RUN
--skip-embedding-training` to start at state SL and then run PPO with the saved
embedding model frozen.
The fixed validation scenes are separate from the training scene pool. The 3 GiB limit covers managed recording batches, including
validation; checkpoints, existing `data/`, compiler caches, and RAM are separate.

### Choose direct embeddings or RGB prediction

The default is an MC-JEPA-inspired, IMU-conditioned joint-embedding adaptation:
an online encoder/predictor learns against a frozen-gradient EMA target encoder,
with content consistency and variance/covariance penalties to discourage collapse.
There is no RGB decoder in this path. See the [architecture and objective](MODELS_README.md#model-1-default-direct-embedding-prediction).

```bash
python train.py --stream-data --visual-prediction embedding --max-data-mib 3072 --verify-limits
python train.py --config gru.json --stream-data --visual-prediction rgb --max-data-mib 3072 --verify-limits
```

Omitting `--visual-prediction` selects embeddings for new runs. RGB mode retains
the original flow/affine generator options. `--from-checkpoints` preserves the
saved representation; older checkpoints without this setting keep RGB. Start a
fresh run to switch representation. Export, PPO and native-student distillation
read the mode from the checkpoint metadata automatically.

`--epochs-frame` controls the first stage in either mode. Direct mode writes
`embedding_metrics.jsonl` and keeps the historical `frame_student.pt` filename
for its predictor, online encoder and EMA target. It disables RGB window caches.
`--embedding-chunk 1024` controls the number of missing features predicted at
once; `--frame-chunk` controls the optional RGB path. Existing GPU batching,
background collection and parallel distillation settings continue to apply.
In the matched synthetic four-student RTX 5080 probe, direct mode reached
26.02 versus 17.39 shared windows/s. See [the measured protocol and smoke checks](verification/README.md#direct-embedding-prediction-default-2026-11-12).

### Choose model size against deployment limits

GPU training optimizations make larger experiments cheaper. Deployment capacity
depends on the exported WASM artifact. Direct embeddings reduced inference cost,
and the default 896-wide hybrid passed a 1,200-frame check at **118.01/128 MiB**
upload and **8.62/20 G** maximum step fuel. Its measured memory was 325.6 MiB
and total runtime 672.4 seconds. The earlier GRU embedding export used
123.49 MiB and 3.97 G in a three-frame probe.
Each model must satisfy all the upload, init/step fuel, memory, table, wall-clock
and total-episode limits listed in the [model guide](MODELS_README.md#resource-requirements-and-measured-profile).

These are candidates for retraining and accuracy comparison, not new defaults:

| Model | Comparison baseline | Larger candidate | Candidate resource use | Check length |
|---|---|---|---:|---:|
| Optional GRU teacher | GRU 1,792; predictor hidden 256 | GRU 1,824; predictor hidden 256 | 126.16/128 MiB upload | 3 frames |
| `100` student | Six affine coefficients | Keep current size | 99/100 fuel | 1,200 frames |
| `10k` student | 4x3 visual grid | 6x3 visual grid | 9,705/10,000 fuel | 1,200 frames |
| `1m` student | Hidden 192 | Hidden 216 | 967,184/1,000,000 fuel | 1,200 frames |
| `100m` student | Hidden 1,792 | Hidden 2,048 | 98,058,684/100,000,000 fuel | 1,200 frames |

Teacher width 1,840 fit at 127.50 MiB; width 1,848 exceeded the cap at 128.18 MiB.
The 1,824 candidate retains more upload margin. A larger embedding predictor
is another allocation to compare: GRU 1,824 / predictor hidden 2,048 fit at
127.53 MiB in a three-frame check. Neither allocation has an accuracy result.

To try the more conservative teacher candidate, save this as
`teacher-capacity.json` and start a fresh run:

```json
{"USE_HYBRID_SSM_TRANSFORMER": false, "VISUAL_PREDICTION": "embedding", "STUDENT_HIDDEN": 1824, "EMBEDDING_HIDDEN": 256}
```

```bash
python train.py --config teacher-capacity.json --stream-data --max-data-mib 3072 --verify-limits
```

Retain a candidate only after retraining, comparing held-out trajectory accuracy
and verifying the selected trained export. The teacher probes above cover only
three frames; they do not replace the 1,200-frame check. Student probes used test
weights, so trained student exports must pass their own exact tiers again.
See [the full capacity review](verification/README.md#capacity-headroom-review-2026-11-12).

If a teacher is used exclusively offline for distillation, submission caps apply
to its deployed students. That teacher can instead be sized around GPU memory
and training cost; `--skip-export` saves its checkpoints without deployment.

### Recording reuse and early stopping

If collection is slower than training, the current recording batch can be used
for **up to three passes total** (the first pass plus two extra passes). A ready
replacement is adopted at the next epoch boundary, even before the limit is
reached. Once three passes have been used, training waits for fresh recordings.
The old files are removed only after their replacement is complete.

The limit is configurable:

```bash
python train.py --stream-data --max-data-mib 3072 --max-passes-per-batch 3 --verify-limits
```

Use `--max-passes-per-batch 1` to wait for fresh data after every pass. The JSON
setting is `STREAM_MAX_PASSES_PER_BATCH`. Logs show each batch name and pass count.
The limit is shared across training stages that consume that recording batch.

Validation runs every epoch on the same held-out recordings. In streaming mode,
early-stopping patience advances at most once per recording batch: a
non-improving first epoch on a new batch counts once, while extra epochs on that
batch do not consume additional patience. A qualifying improvement on any epoch,
including an extra pass, resets patience and saves the checkpoint.

With the curriculum enabled, a difficulty or window-length promotion also resets
patience. `CURRICULUM_EARLY_STOP_GRACE=3` protects the next three fresh recording
batches from non-improvement charges; fixed-data training counts three epochs.
The promotion epoch does not consume grace, and reused batches do not consume it
again. Further promotions restart the grace period rather than adding to it.
Validation and best-checkpoint selection continue during adaptation. Set grace
to zero for a promotion reset without an adaptation period. Epoch caps stay fixed.

| Stage | Patience in streaming mode | Default maximum epochs | Curriculum preset maximum epochs |
|---|---:|---:|---:|
| Embedding predictor / optional RGB generator | 8 non-improving fresh batches | 200 | 400 |
| State estimator | 6 non-improving fresh batches | 300 | 600 |

Epoch caps include recording reuse. The curriculum preset also raises optional
distillation from 25 to 100 epochs; its teacher cap remains 60. Early stopping
still applies. The launcher has no automatic time cutoff.

The embedding criterion is the validation joint-embedding objective improving
by more than `0.00001`. RGB mode uses strictly lower validation image L1; the state criterion is
validation state loss improving by more than `0.00001`. Learning-rate scheduling
still runs every epoch. With `--no-stream-data`, patience is counted per epoch.

New runs default to 200 frame epochs and 300 state epochs; `full_resolution.json`
also pins those limits. Warm starts use these current epoch defaults while
restoring the source run's architecture and other settings. Saved 40/50 limits
are not automatically restored. Explicit `--config` or `--epochs-frame` /
`--epochs-state` options take precedence. The resolved limits print at startup.

The command prints a path such as:

```text
Checkpoints and configuration: .../1_generative_ssm/checkpoints/20261112-160000-123456
```

Keep that path. Later examples use **`checkpoints/RUN`**: replace `RUN` with the
actual folder name printed by training. Do not type `RUN` literally.

To see your saved run folders:

```bash
ls -dt checkpoints/*/
```

Default run folders have unique timestamps. `--out` lets you choose a name, but
reusing an existing output folder can replace its files. Use a new name for each run.

### Train without the full resource check

```bash
python train.py --stream-data --max-data-mib 3072
```

This still exports `solution/`, but does not automatically create a `.wasm` or run
resource verification. Follow the manual export/build section below afterward.

### Save checkpoints only

```bash
python train.py --stream-data --max-data-mib 3072 --skip-export
```

This skips ONNX export, bundling and resource checks. It is useful when you only
need teacher weights. SL and PPO still run, and `post_sl/` is still saved. Add
`--no-ppo` to save SL-only weights. Do not combine `--skip-export` with `--verify-limits`.

### Try fewer training epochs

```bash
python train.py --stream-data --max-data-mib 3072 --epochs-frame 1 --epochs-state 1 --no-ppo --skip-export
```

This checks the training workflow with less optimization. Initial simulator setup
and data collection still take time; a short run is not an accuracy benchmark.

### GPU throughput settings

The defaults now overlap a single background collector with the active training
stage and use the following settings on CUDA:

| Setting | Default | Change it with |
|---|---:|---|
| Flights simulated together per training scene | 2 | `--collect-instances 2` |
| State windows per optimizer minibatch | 4 | `--batch 4` |
| Frame pairs per optimizer minibatch | 8 | `--frame-pair-batch 8` |
| Missing frames synthesized per chunk | 16 | `--frame-chunk 16` |
| Frames encoded per visual CNN chunk | 16 | `--visual-chunk 16` |
| State data-loader workers | 2 | `--loader-workers 2` |
| Reuse loader workers between epochs | Enabled | `--persistent-workers` / `--no-persistent-workers` |
| Fused CUDA AdamW updates | Enabled | `--fused-adamw` / `--no-fused-adamw` |
| Neural-layer precision | BF16 on CUDA | `--amp` / `--no-amp` |

Motion integration, losses, validation, and saved weights stay FP32. CPU runs
use FP32 and no data-loader workers. Frame-pair loading stays in the main process
(`FRAME_LOADER_WORKERS=0`): the measured cost of passing these large, simple
slices between processes exceeded its benefit. CUDA loaders pin memory; state
workers prefetch one minibatch each. Transfers request asynchronous copies.
Workers persist across epochs on the same dataset; replacing a rolling training
batch shuts down its old workers first. Fixed validation workers stay warm until
the run exits. This reduces CPU startup gaps without retaining old recording
batches. CUDA AdamW combines parameter updates into fused kernels; CPU retains
ordinary AdamW. Both controls also apply to `distill.py`.
For the hybrid, neural sequence training now honors AMP as well. CUDA training
uses differentiable parallel FP32 physics with `PARALLEL_TRAINING_PHYSICS=true`;
validation and deployment keep the original sequential recurrence. This changes
FP32 reduction order slightly; disable the flag for the former training path.
`HYBRID_ATTENTION_CHUNK=512` bounds local-attention masks, and
`COMPACT_STATE_CAMERAS=true` avoids carrying unused native captures through state
augmentation and IPC. The original selected camera times and targets are retained.
Observed images are selected first, then transferred, normalized and encoded in
bounded chunks. PPO batches action/reward/value/GAE work with
`VECTORIZED_PPO_ROLLOUT=true`, generating new current-policy actions each iteration.
All these controls are recorded in `training_config.json`; full benchmark scopes
and numerical checks are in the optimization review linked above.
The follow-up RTX 5080 probes measured 13% more frame pairs/s with fused AdamW
and 37% shorter warm epochs in a tiny streaming distillation run with worker
reuse. See [the protocol and limits of those measurements](verification/README.md#further-gpu-pipeline-improvements-2026-11-12).

No extra arguments are needed for the usual fresh-teacher command:

```bash
python train.py --stream-data --max-data-mib 3072 --verify-limits
```

For more state-training throughput on this RTX 5080, try batch eight:

```bash
python train.py --stream-data --batch 8 --max-data-mib 3072 --verify-limits
```

This needs more VRAM. Start a new run directory when changing settings. An already
running process keeps its loaded settings. Warm starts retain model architecture
but use the current performance defaults; CLI or explicit JSON settings override
those defaults.

If GPU memory is tight, use smaller minibatches/chunks and one simulator flight:

```bash
python train.py --stream-data --batch 1 --frame-pair-batch 1 --frame-chunk 1 --visual-chunk 1 --collect-instances 1 --loader-workers 0 --max-data-mib 3072 --verify-limits
```

Add `--no-amp` if you need FP32 neural training. This can increase GPU memory use.
For older GPUs without BF16 support, this flag is required. Distillation accepts
`--batch`, `--amp` / `--no-amp`, `--frame-chunk`, `--visual-chunk`,
`--collect-instances`, and `--loader-workers` as well. Its frozen teacher uses
selective BF16 while the native students retain FP32 arithmetic.

Flight count is capped by episodes requested per scene (two by default in the
rolling pipeline). Standalone collection can request more explicitly:

```bash
python collect.py --out data --scenes 313 --episodes 2 --steps 2000 --instances 2
```

Each flight writes its own NPZ; a crashed flight stops without ending its healthy
neighbors. The managed recording budget is 3 GiB, and the three-pass
reuse limit and fresh-data early-stopping policy are unchanged. Fixed validation
collection uses one simulator flight at a time for reproducibility.

Each teacher run writes `performance.jsonl`: stage, epoch, samples/second, elapsed
training time, peak allocated CUDA memory, recording pass, and seconds blocked
waiting for fresh data. The epoch timer includes loader startup/transfers and
excludes validation. Collection directories contain `collection_stats.json`,
including build time and rollout throughput. Temporary collection directories
are deleted by the existing rolling cleanup. Check live GPU use in another terminal:

```bash
watch -n 1 nvidia-smi
```

The [performance measurements](verification/training_performance.json) include
synthetic GPU probes and actual simulator/loader checks. They are throughput
checks, not accuracy results or a guarantee of continuous GPU saturation.

## 4. Continue from saved weights or recover after interruption

To refine an existing model with PPO only (the default for saved weights):

```bash
python train.py --from-checkpoints checkpoints/RUN --stream-data --verify-limits
```

This starts fresh PPO optimizer/critic state; it does **not** resume the exact
previous training iteration. Source checkpoints are preserved, and a new output
folder is created. PPO needs both `frame_student.pt` and `state_student.pt`,
even if they came from an interrupted SL run. A missing component is rejected
before data collection begins.

To continue SL first and then run PPO, including recovery when a component has
not yet saved its first checkpoint:

```bash
python train.py --from-checkpoints checkpoints/RUN --sl --stream-data --verify-limits
```

Available component weights are loaded; missing components start supervised
training from initialization. Both SL stages run with fresh optimizers and
early stopping. Add `--no-ppo` for SL-only continuation. New SL runs create their
own `post_sl/` snapshots before PPO begins.

To keep an already trained embedding model fixed and start at state SL:

```bash
python train.py --from-checkpoints checkpoints/RUN --skip-embedding-training --stream-data --config curriculum.json --max-data-mib 6144 --verify-limits
```

`--skip-embedding-training` enables state SL without requiring `--sl`. It requires
an embedding-mode source with `frame_student.pt`; it cannot be combined with
`--ppo-only` or `--no-sl`. The predictor, online encoder and EMA target remain
unchanged and in evaluation mode, and no embedding training run is opened in W&B.
The state model loads its source weights if present, or initializes if absent.
State curriculum/optimizer and PPO optimizer/critic start fresh. Add `--no-ppo`
for state SL alone. Source files are preserved, and new output includes the fixed
embedding weights and an exportable `post_sl/` snapshot. A changed embedding
patch stem cannot be initialized while skipping its training.

To train the new large teacher explicitly, start a fresh run with the preset:

```bash
python train.py --config full_resolution.json --stream-data --max-data-mib 3072 --verify-limits
```

The default hybrid architecture cannot load GRU weights. Omit
`--from-checkpoints` when switching to this preset. Continuing an old run with
`--from-checkpoints` alone preserves that run's architecture and resolution;
it does not upgrade it to the new teacher. A same-width change to learned patch
stems can retain compatible weights during SL, but requires SL before PPO-only.

If training finished but export failed, **do not retrain just to retry export**.
Use `export.py` as shown in section 6.

Press **Ctrl-C once** to stop training. Normal cleanup stops the collector and
removes this run's temporary recording batches. Previously saved checkpoints
remain available. A forced kill or power loss may leave a managed data directory;
check that its training process has stopped before removing it manually.

## 5. PPO refinement and stage controls

SL means supervised learning from simulator ground truth. PPO refines the existing
motion-correction head and is enabled by default. `--no-ppo` disables it.

| Input/options | Training stages |
|---|---|
| Fresh run, no stage flags | SL, then PPO |
| `--from-checkpoints checkpoints/RUN` | PPO-only |
| `--from-checkpoints checkpoints/RUN --sl` | More SL, then PPO |
| `--from-checkpoints checkpoints/RUN --skip-embedding-training` | State SL, then PPO; embedding weights fixed |
| `--no-ppo` (fresh or saved run) | SL-only |
| `--ppo-only --from-checkpoints checkpoints/RUN` | Explicit PPO-only alias |

Train with SL and then PPO:

```bash
python train.py --stream-data --max-data-mib 3072 --ppo --verify-limits
```

Run only PPO later, using an existing compatible checkpoint:

```bash
python train.py --from-checkpoints checkpoints/RUN --ppo-only --stream-data --max-data-mib 3072 --verify-limits
```

`--ppo-only` requires `--from-checkpoints`. It cannot be combined with `--no-ppo`.
`--no-ppo` explicitly disables PPO even if a configuration enables it.
`--no-sl` also requests PPO-only and needs saved weights. `--ppo-only` cannot
be combined with `--sl`. Explicit JSON `USE_PPO` settings can override the PPO
default; a source checkpoint's old `USE_PPO=False` is not automatically restored.

The encoder and recurrent backbone stay frozen during PPO. The SL weights are
preserved, and PPO weights are selected only when they improve the fixed validation
loss. See [the model guide](MODELS_README.md#optional-ppo-refinement) for the objective
and what this selection criterion means.

PPO shares the recording-batch reuse limit. Here, one pass is one PPO iteration.
Each iteration samples new actions from the current policy and recomputes the
estimator trajectory, rewards, log probabilities and advantages. One rollout is
optimized for up to `PPO_UPDATE_EPOCHS=4` passes, with an earlier KL cutoff, then
discarded. Reusing recordings does not reuse old policy rollouts across iterations.
PPO defaults to `PPO_ITERATIONS=10` iterations; it has no separate
whole-run patience stop. Its KL cutoff ends only the current rollout's updates.

## 6. Export saved models and build `.wasm` manually

Use this route after checkpoint-only training, an export failure, or when you want
to bundle a particular saved run.

### Mixed scene collection and PPO curriculum

`SCENE_CATALOG="mixed"` is the default for streaming. The scene list contains
192 ProcTHOR and 121 HSSD training scenes interleaved proportionally, then 8 ProcTHOR and 4 HSSD
validation scenes. `SCENE_OFFSET` indexes that list. Stable recording IDs are
0..199 for ProcTHOR and 200..324 for HSSD; manifests also record catalog and native
index. Existing ProcTHOR scene IDs are preserved.

For ProcTHOR-only streaming, set `SCENE_CATALOG="procthor"`, `NUM_SCENES=192`,
and `STREAM_VALIDATION_SCENES=8` in your JSON. For HSSD-only, use `"hssd"`, 121,
and 4. These settings also apply to student distillation's rolling collection.
Manual examples:

```bash
python collect.py --out data-hssd --scene-catalog hssd --scenes 125 --episodes 2
python collect.py --out data-all --scene-catalog mixed --scenes 325 --episodes 2
```

The mixed streaming split reserves validation scenes; manually collecting all
325 does not assign that split to fixed-data training, which holds out only its
highest recorded scene ID. Keep separate output folders for separate collections.

Enable the full curriculum during **SL and PPO**:

```bash
python train.py --stream-data --config curriculum.json --max-data-mib 6144 --verify-limits
```

Continue SL from an existing checkpoint, then refine with PPO:

```bash
python train.py --from-checkpoints checkpoints/RUN --sl --stream-data --config curriculum.json --max-data-mib 6144 --verify-limits
```

For PPO-only refinement, omit `--sl`:

```bash
python train.py --from-checkpoints checkpoints/RUN --stream-data --config curriculum.json --max-data-mib 6144 --verify-limits
```

`curriculum.json` enables `USE_CURRICULUM`, requests 4,001-sample recordings,
and gives state SL windows of **2 → 4 → 8 → 12 → 20 seconds**. PPO uses
**2 → 5 → 10 → 20 seconds**. Sample counts include the initial state. It uses batch
size up to eight, 60 PPO iterations, and a proposed longer-horizon discount of 0.9995.
The 3,208-token budget caps SL state batches at **8/4/2/1/1** across its horizons,
and PPO recording-window batches at **8/3/1/1**. The PPO optimizer minibatch size
is unchanged. Frame/embedding training uses batch **64**, two loader workers, uint8 camera transfers,
on-device photometric transforms, and two CUDA host Torch threads.
`FRAME_VAL_PAIR_BATCH=8` preserves the embedding validation objective's batch
statistics. Larger training batches do not change that validation setting.
`EMBEDDING_DIM=48` retains existing model shapes. For a fresh capacity experiment,
the equivalent `curriculum_embedding96.json` preset selects 96 features end to end.
It passed all listed submission caps on a 1,200-frame synthetic WASM episode
(118.95 MiB upload, 8.65 G maximum step fuel, 328 MiB memory, 11.16 minutes total).
Accuracy has not been established; compare held-out trajectories before promoting
it. Width changes require fresh SL and cannot reuse 48-dimensional checkpoints.
The preset allows up to **400 frame/embedding epochs, 600 state epochs, and 100
optional distillation epochs**; optional teacher training retains its 60-epoch
cap. These values apply to fresh runs and checkpoint continuations selecting
`--config curriculum.json`, including the timed launcher. CLI overrides such as
`--batch`, `--frame-pair-batch`, `--epochs-frame`, and `--epochs-state` still win.
At 200 Hz, the default discount of 0.995 has an approximately one-second
exponential decay time; 0.9995 extends that to approximately ten seconds.
Discount and GAE settings should be tuned along with the window length.
These are starting settings for experiments, not measured accuracy improvements.
Collection retains one background collector with two parallel flight instances.
Future streaming runs use **`COLLECT_COMPRESSION_LEVEL=1`** for faster lossless
NPZ writes. It changes compression effort, preserving every camera pixel, IMU
sample and ground-truth value. A bounded real-recording I/O benchmark measured
**5.44× faster writes with 21.5% larger files** than level 6; this excludes scene
downloads, simulation and training. The recording byte cap remains enforced
before publishing a file, so larger files can reduce flights fitting a batch.
Validation still requires coverage of every held-out scene. Set the level to
6 in an experiment JSON to restore the former compression, or use
`collect.py --compression-level 1` for a manual collection. Standalone collection
without this flag retains level 6, so the existing active run's collector
commands retain their original compression. See the
[collection I/O benchmark](verification/wandb_curriculum_20261113/README.md#collection-write-optimization-for-future-runs).
Higher training throughput and GPU saturation have not been verified for the
larger batches. `performance.jsonl` and W&B `pipeline/*` metrics record the
effective `training_batch_size` (the loader limit, excluding a smaller final
partial batch), `window_samples` for state/PPO, `samples_per_second`, loader and
fresh-data waits, and peak allocated CUDA memory when available. Collection
reports include `collection_compression_seconds`, `collection_write_seconds`,
`collection_compression_level`, and `collection_recording_bytes`. PPO throughput
counts recording windows through rollout collection and optimizer updates.
On a GPU, compare the former 32-pair / 4-sequence / 1,604-token settings against
the new preset at the same horizon and seed, with collection active. Measure at
least three fresh training batches after warmup and check loss finiteness,
memory headroom, collection waits, and validation improvement per hour.
Longer recordings and windows increase recording storage, host RAM, GPU memory,
and runtime; adjust the recording budget for your scenes. Collection rejects
incomplete validation coverage and training fails clearly if no sufficiently long
windows survive crashes.

State SL and PPO each retain one fixed validation horizon: the longest window in
their respective lists for **checkpoint selection**. Promotion and state LR
scheduling use fixed probes at the **current training horizon**. Probe references
reset when that horizon changes; checkpoint comparisons retain the global best
20-second score. Promotion needs >2% improvement over two consecutive evaluations
after at least three updates. Promotion also allows a plateau after six updates
if the current score is at most 10% above the stage's best. For example, a best
loss of 10 permits plateau advancement at 10.5 or 11, but not at 11.1. Sustained
improvement can still advance earlier. The same tolerance applies to SL and PPO.

With curriculum enabled, SL early stopping waits until all motion/visual/IMU axes
reach their final level and state training reaches its longest window. It then
protects three fresh recording batches trained at that final stage; ordinary
patience begins on the following fresh batch. Promotion epochs and reuse of the
promotion batch do not count as final-stage exposure. For fixed data, each epoch
counts as a fresh cycle. This replaces the preset's former 24/36-epoch minimums.
Promotions also retain their adaptation grace. Best checkpoints always use the
fixed, unaugmented validation set, independently of promotion and stopping.
Hard epoch caps are 400/600 in the preset (200/300 without it). The launcher has
no default wall-clock limit; an explicit `--seconds` limit remains optional. Large regressions
do not force advancement, so a run can hit its hard cap before completing the
curriculum. W&B logs completion, final-stage exposure, and stopping eligibility.

`CURRICULUM_REQUIRE_COMPLETION=false` restores stopping without requiring full
exposure; `CURRICULUM_FINAL_STAGE_CYCLES` controls the final exposure (default 3).
Optional `CURRICULUM_MIN_FRAME_EPOCHS` / `CURRICULUM_MIN_STATE_EPOCHS` remain
available as extra minimums (default 0). `CURRICULUM_MAX_STAGE_EPOCHS=0` disables
plateau advancement. See the [W&B analysis](verification/wandb_curriculum_20261113/README.md)
for measurements and the reason these settings changed.

Motion, visual, and IMU difficulty have **independent progress gates**:

| Axis | Training control | Fixed probe used for promotion |
|---|---|---|
| Motion | Sample segments by speed, angular rate, and acceleration; use motion quantiles within the current recording batch | Upper third of held-out motion scores |
| Visual | Shared brightness, contrast, and blur changes across the sample's images; no geometric warps | Fixed photometric stress at maximum configured strength |
| IMU | Extra white noise, per-flight bias, and slowly varying bias drift | Fixed extra sensor errors at maximum configured strength |

The main validation set stays unaugmented at the normal sensor settings. Probes
are separate, use fixed samples/seeds within each training horizon, and default
to two batches per axis. A new horizon establishes new probe references.
Increase `CURRICULUM_PROBE_BATCHES` for more representative gates. Native camera
timing and the approximately 20 Hz observation schedule remain unchanged.
Small probe loaders run in the main process, so they do not add persistent workers.

`CURRICULUM_LEVELS=[0,0.5,1]` controls severity. The preset's `CURRICULUM_MIX=[0.6,0.2,0.2]`
mixes current, easier, and harder levels; endpoint levels merge unavailable
neighbors. Motion mixtures operate over sample buckets. Visual/IMU levels are
sampled per flight, so proportions are approximate in a small batch. Optional
`CURRICULUM_PRIORITIZE` adds bounded priorities for examples whose error is
decreasing, rather than favoring large error alone. Priorities compare repeated
examples from the same recording at the same horizon; fresh recordings retain
the base mixture. The table's motion control changes sampling, not autopilot routes.
Priority transfers to CPU are deferred until epoch end to avoid per-batch GPU
synchronization. Model-device photometric work preserves geometry and targets.

The simulator's existing IMU noise is the baseline. Extra errors start at zero
and rise to the configured maxima; this does not reconstruct clean IMU from
noisy recordings. Bias/drift are generated once per flight per epoch, shared by
overlapping windows and frame-pair means. Gravity initialization, visual feature
prediction, and physics integration receive the same augmented IMU stream.
Ground-truth motion stays unchanged. Image augmentation preserves camera geometry
and transforms frame-pair inputs and targets consistently.

`curriculum_metrics.jsonl` records axis probes, levels, SL windows, promotions,
and early-stopping reset/grace events. Embedding metrics and W&B stage logs include
`early_stopping_grace_remaining`. PPO retains its iteration cap without adding
an early-stopping rule. `ppo_metrics.jsonl` records PPO windows.
PPO also gets its own W&B run; SL history now includes input-wait timing and
`curriculum/*` levels, current-horizon probe losses and window progression.
Future runs also log explicit `curriculum/promoted` and per-axis
`curriculum/{motion,visual,imu,window}_promoted` markers on the epoch/iteration
that selects a harder next stage. Per-axis `*_promotion_reason` records
`improved`, `plateau`, or `none`; numeric `*_promoted_improved` and
`*_promoted_plateau` markers can be plotted alongside loss. Current and next
levels/windows distinguish the difficulty just trained from the next update,
including PPO horizon promotions.

State validation logs Mean ATE in metres as `student/val_mean_ate_m`,
`teacher/val_mean_ate_m`, `distill/val_mean_ate_m`, and `ppo/val_mean_ate_m`.
This is the mean Euclidean distance between predicted and ground-truth positions
over timestamps and validation windows, weighted by window count across batches.
Both trajectories already share a local starting frame; no rigid/scale alignment
is fitted. It differs from coordinate-wise `student/val_pos_rmse` and ATE RMSE.
Metrics use the fixed, unaugmented validation horizon, so curriculum changes do
not change their evaluation difficulty. PPO records the SL baseline and the
loss-selected checkpoint's Mean ATE too, in W&B and `ppo_metrics.jsonl`.
These diagnostics reuse validation predictions and do not change training loss
or checkpoint selection. Visual-only training has no estimated position trajectory
and therefore does not report ATE. Existing W&B histories are not backfilled.

`curriculum_state.json` stores the latest
gate state for inspection. Gates restart for each training phase; PPO inherits
the final SL difficulty levels within the same invocation. Checkpoint warm starts
restart the schedules and PPO optimizer/critic rather than resuming exact state.
The optional auxiliary teacher and its in-process SL distillation also use these
controls. The separate native-student `distill.py` workflow retains its existing
training policy.

Set `USE_CURRICULUM=false` to disable motion/visual/IMU controls and SL scheduling.
The standalone PPO window schedule remains available; clear
`PPO_CURRICULUM_WINDOWS` as well to restore ordinary fixed-window training.
Generated-window disk caches are disabled while the full curriculum is active.

This trains the **VIO estimator**, while HyperDrone's autopilot flies the drone.
PPO corrections cannot improve collision avoidance. Long recordings are collected
from the start; PPO progressively uses longer windows within them. Crashes end
recording for that slot, and windows never cross those ends. PPO reserves the
last token to bootstrap the critic at time cutoffs.

Recovery disturbances to internal estimator state are not implemented: the current
correction actor does not observe integrated position/velocity, so hidden state
offsets need a separate observability design. See
[automatic curriculum learning research](https://www.ijcai.org/proceedings/2020/0671.pdf)
and [time-limit bootstrapping](https://gymnasium.farama.org/introduction/migration_guide/).

### Export both the post-SL and post-PPO models

The complete `post_sl/` snapshot is published **before PPO starts**, so it remains
exportable if PPO is interrupted and can be exported from another terminal while
PPO is running. Replace `RUN` with your actual run folder.

```bash
python export.py --checkpoints checkpoints/RUN/post_sl --out checkpoints/RUN/solution_sl
python -m viobench bundle checkpoints/RUN/solution_sl -o checkpoints/RUN/teacher-sl.wasm
```

After PPO finishes, export the final selected model separately:

```bash
python export.py --checkpoints checkpoints/RUN --out checkpoints/RUN/solution_ppo
python -m viobench bundle checkpoints/RUN/solution_ppo -o checkpoints/RUN/teacher-ppo.wasm
```

The post-PPO selection can retain the SL baseline if validation did not improve.
`train.py --verify-limits` automatically bundles and checks the final selected
model as `checkpoints/RUN/teacher.wasm`. To check the separately built SL model:

```bash
python verify_limits.py checkpoints/RUN/teacher-sl.wasm --report checkpoints/RUN/teacher-sl-limits.json
```

PPO-only runs preserve their input model in `state_student_sl.pt` (a historical
filename for the pre-PPO baseline). Their original source run's `post_sl/`
snapshot is preserved; they do not create a new post-SL snapshot.

### Export weights to the guest format

```bash
python export.py --checkpoints checkpoints/RUN
python validate.py solution
```

Export writes model files into `solution/`. Validation runs the official Python
guest interface on your host computer; it does not run WebAssembly or measure fuel.

### Build the Python WebAssembly runtime once

```bash
python -m viobench build-runtime
```

The first build downloads and compiles CPython, NumPy, OpenCV and the WASI toolchain.
It may take 10–15 minutes or longer and needs storage beyond the recording budget.
If the runtime already exists, you can skip this command. `bundle` also builds a
missing runtime automatically.

### Bundle the solution and check it

```bash
python -m viobench bundle solution -o teacher.wasm
python -m viobench check teacher.wasm
```

**Bundle `solution/`, not `1_generative_ssm/`.** The entry point
`solution/solution.py` must be at the root of the directory passed to `bundle`.
Keep all its exported model files together.

After a successful check, the output ends with `check: ok (5 steps)`.
That confirms the module runs on five synthetic frames; it does not establish
trajectory accuracy or full-episode runtime.

### Run the full resource check

```bash
python verify_limits.py teacher.wasm --report teacher-limits.json
```

This runs 1,200 frames, representing 60 simulated seconds at 20 Hz. It checks
upload size, per-call fuel/deadlines, enforced memory/table limits, and total time
under 24 minutes. Allow roughly 17 minutes for the large teacher on this machine;
your actual run time can differ. The report applies to the exact module tested.

For a shorter diagnostic check:

```bash
python verify_limits.py teacher.wasm --frames 5 --report teacher-limits-short.json
```

A five-frame check cannot establish full-episode runtime compliance.

### Advanced export commands

Export to a separate directory without replacing `solution/`:

```bash
python export.py --checkpoints checkpoints/RUN --out teacher_solution
python -m viobench bundle teacher_solution -o teacher.wasm
```

Export the repeat-observed-frame ablation and evaluate it separately:

```bash
python export.py --checkpoints checkpoints/RUN --frame-mode repeat --out solution_repeat
python -m viobench bundle solution_repeat -o teacher-repeat.wasm
```

Export a notebook weights folder containing `state_student.npz`,
`frame_student.npz` and `student_config.json`:

```bash
python export.py --assets /path/to/vio_student_export --out imported_solution
```

Use a prebuilt runtime if you already have one:

```bash
python -m viobench bundle solution --runtime /path/to/runtime.wasm -o teacher.wasm
```

Paths starting `/path/to/` are placeholders; replace them with existing paths.
The exporter's `--frame-width` overrides the ABI request, not the model's trained
image resolution. Current full-resolution models request width 640 automatically.

## 7. Evaluate accuracy and read the results

If section 3 created the module in the checkpoint directory:

```bash
python -m viobench evaluate checkpoints/RUN/teacher.wasm --limits submission_limits.json -o report-teacher
```

If you manually created `teacher.wasm` in this folder:

```bash
python -m viobench evaluate teacher.wasm --limits submission_limits.json -o report-teacher
```

Choose one command for the module you want to evaluate. With no dataset argument,
evaluation uses the published public set and downloads missing episodes. These
downloads are separate from the 3 GiB training-data budget. The final summary
prints **`Sim(3) ATE mean`**, measured in meters; lower is better.

Start with one public episode for a shorter evaluation:

```bash
python -m viobench evaluate teacher.wasm --max-episodes 1 --limits submission_limits.json -o report-one-episode
```

Use a local VIOBench-format episode or dataset:

```bash
python -m viobench evaluate teacher.wasm /path/to/viobench-dataset --limits submission_limits.json -o report-local
```

The collector's `rollout_s*_e*.npz` files are training recordings, not VIOBench
evaluation episodes. Passing the training `data/` folder to `evaluate` does not
convert those recordings.

View a saved report or compare multiple evaluations:

```bash
python -m json.tool report-teacher/report.json
python -m viobench leaderboard report-teacher report-one-episode
```

For host-side debugging on a VIOBench-format dataset:

```bash
python -m viobench debug solution /path/to/viobench-dataset
```

Host debugging does not measure WASM fuel. Use a new report directory name when
comparing different models. `submission_limits.json` specifies the per-call caps;
the separate synthetic verifier checks the 24-minute total episode budget.

## 8. Collect a fixed dataset instead of background generation

This is an alternative to `--stream-data`. It retains all recordings you collect.

```bash
python collect.py --out data --scenes 313 --episodes 2 --steps 2000 --width 640 --height 480
python train.py data --no-stream-data --verify-limits
```

Run training only after collection succeeds. Full-resolution training requires
640x480 recordings. Old 160x120 recordings are rejected rather than upsampled.
Fixed-data training holds out the last scene ID for validation and requires at
least two distinct scenes.

To cap the collector's output directory:

```bash
python collect.py --out data_new --scenes 3 --episodes 2 --steps 2000 --width 640 --height 480 --max-output-mib 512
```

A cap may stop collection before all requested scenes are reached. Check that the
result contains at least two scene IDs before using it for fixed-data training.

## 9. Change settings or choose GRU

Defaults live in [config.py](config.py). For reproducible experiments, create a
JSON file of uppercase overrides, then pass it with `--config`.

For example, save this as `experiment.json`:

```json
{
  "USE_HYBRID_SSM_TRANSFORMER": true,
  "STUDENT_HIDDEN": 896,
  "USE_TEACHER": false,
  "PPO_ITERATIONS": 5
}
```

Then run:

```bash
python train.py --config experiment.json --stream-data --no-ppo --verify-limits
```

The hybrid uses **official `mamba_ssm.Mamba2` CUDA kernels only**. Install its
host dependencies after CUDA PyTorch (required for the default hybrid):

```bash
python -m pip install --no-build-isolation -r requirements-mamba.txt
python train.py --config hybrid.json --device cuda --stream-data --verify-limits
```

The hybrid is the default. The GRU/hybrid architecture toggle remains; set
`USE_HYBRID_SSM_TRANSFORMER=false` or pass `--config gru.json` to select GRU.
Fresh hybrid runs choose width 896
unless `STUDENT_HIDDEN` is explicit; GRU runs keep width 1,792. Width 896, three
Mamba blocks and a 16-token attention window passed all supplied limits over
1,200 synthetic frames: **118.01 MiB upload, 8.62 G maximum step fuel, 325.6 MiB
memory, and 672.4 seconds total**, including initialization. Width 512 also
passed with more headroom (71.87 MiB, 3.00 G, 141.2 MiB). A width-1,792 hybrid
cannot fit: FP32 weights alone are 274.42 MiB. These seeded resource probes do
not establish VIO accuracy or cover episodes longer than the tested 60 simulated
seconds; verify the selected trained export and actual evaluation trajectories.

CUDA training weights are exported into the NumPy recurrence inside WASM; CUDA
kernels are not embedded in WASM. Direct comparisons cover official scan/step,
nonzero hybrid outputs, an optimizer update, and actual WASM camera/IMU outputs.
Triton FP32 dot products default to IEEE precision for agreement with deployment;
BF16 training remains available. The custom PyTorch Mamba and its backend/export
toggles have been removed. Old saved toggle fields are ignored on warm start;
new experiment files must omit them. Hybrid training, PPO and distillation need
CUDA; export only constructs the official module on CPU to serialize its weights.
GRU workflows do not require the Mamba package. Saved runs preserve their
backbone on resume; switching architectures requires a fresh run.

The unused `COLLECT` and `EPISODES_PER_SCENE` settings were also removed; old
saved values are ignored on resume. Use `--stream-data` and
`STREAM_EPISODES_PER_SCENE` for rolling collection, or `collect.py --episodes`
for manual collection.

See [verification results and reproduction](verification/README.md#official-mamba-2-parity-and-hybrid-limits-2026-11-12)
for numerical tolerances, toolchain details and measured limitations.
[MODELS_README.md](MODELS_README.md#configuration-reference) describes the settings.
`USE_TEACHER` is the legacy optional auxiliary-teacher experiment; it is not required
to train the current full-resolution model as your teacher.

W&B defaults to `online` for future `train.py` runs, using project
`hyperdrone-vio-fuel-aware` and your configured W&B account. Add
`--wandb-mode offline` to save a local W&B run or `--wandb-mode disabled` to
disable it. Explicit experiment JSON settings are supported; the CLI flag wins.
Warm starts use the current mode default rather than restoring an old saved mode.
An already-running process keeps the mode it started with.

## 10. Where files are saved

| Location/file | Contents |
|---|---|
| `checkpoints/<timestamp>/` | A separate directory for each default training run |
| `frame_student.pt` | Best embedding predictor/encoder/EMA weights; RGB generator weights in RGB mode |
| `embedding_metrics.jsonl` | Embedding prediction, content, variance/covariance and total validation losses |
| `state_student.pt` | Selected state estimator weights; the historical filename also names the current teacher |
| `student_config.json` | Architecture and export settings |
| `training_config.json` | Resolved training settings, saved before optimization |
| `post_sl/` | Complete best SL frame/state weights and configuration, saved before PPO and exportable independently |
| `teacher.wasm` | Bundled selected model, created by `--verify-limits` |
| `teacher-limits.json` | Resource result for that module, created by `--verify-limits` |
| `state_student_sl.pt`, `state_student_ppo.pt` | Pre-PPO baseline and selected PPO/baseline weights |
| `ppo_training.pt`, `ppo_metrics.jsonl` | Latest PPO training state and iteration metrics |
| `solution/` | Current exported guest code/model assets; a later export can replace them |
| `report-*/` | Evaluation summaries, estimates, logs and requested videos |
| `data/` | Manually collected fixed recordings; never automatically pruned |
| `checkpoints/<timestamp>/stream-data/run-*/` | Managed rolling recordings, normally removed when training exits |

The `.pt` files need PyTorch to reload. The `.wasm` file is the self-contained
submission artifact. Deployment uses `.onnx` and uncompressed `.npz` files inside
its embedded `solution/` bundle.

Checkpoint replacement is atomic. Completed checkpoints survive an interrupted
training run, but a model that has not yet reached a checkpoint save has no weights
to recover. Generated data and checkpoints are ignored by Git; back them up
separately. If training happens in Colab, copy checkpoints to this computer before
the Colab runtime is discarded.

## 11. Keep disk usage bounded

For a fresh curriculum run without an automatic time limit, with scene assets
and recordings on local disk, primary checkpoints in the project, and an automatic
local mirror when training exits:

```bash
python -u run_until_11.py --local-root /var/tmp/hyperdrone-eecs116-aaq
```

Run this inside tmux with `myenv` activated. See the
[launcher and checkpoint-copy details](SSH_README.md#fresh-curriculum-run-with-automatic-checkpoint-copying).
The historical `run_until_11.py` filename is retained; it now runs until training
completes or you stop it. Add `--seconds 3600` only if you want a one-hour limit.
Use `--config curriculum_embedding96.json` for a fresh 96-dimensional trial, or
`--from-checkpoints checkpoints/RUN` to continue SL then PPO in a new output
directory, preserving the source. A missing state checkpoint is initialized;
existing frame weights are reused. Optimizer/curriculum counters restart.
For a completed embedding stage, add `--skip-embedding-training` to the launcher:

```bash
python -u run_until_11.py --from-checkpoints checkpoints/20261113-180203-195540 --skip-embedding-training --config curriculum.json --local-root /var/tmp/hyperdrone-eecs116-aaq --max-data-mib 6144
```

This runs state SL and PPO with the source embedding weights fixed. It uses a new
run directory, retains the final mirror, and has no automatic time cutoff.
`/var/tmp` is disk-backed on this lab machine, while `/tmp` is RAM-backed. The
launcher leaves model checkpoints on persistent project storage throughout the run.

To place only temporary rolling recordings in `/tmp`:

```bash
python train.py --stream-data --stream-dir /tmp/hyperdrone-training-data --max-data-mib 3072 --verify-limits
```

Checkpoints still save under `checkpoints/`. `/tmp` is temporary and RAM-backed on
this computer, so it is suitable for disposable recordings, not persistent weights.
Cleanup owns only uniquely created run directories. Existing fixed recordings and
other runs are kept. Large synthetic-window disk caches are disabled by default.

Inspect storage and your personal quota:

```bash
quota -s
df -h . /tmp /scratch
du -sh data checkpoints solution
python -m pip cache info
```

The free space of a shared filesystem is not your account's storage allowance.
To remove regenerable pip downloads while retaining installed packages:

```bash
python -m pip cache purge
```

Before manually deleting an old run, verify that no training job is using it and
that you have backed up its needed weights. The rolling-data budget does not cap
checkpoints, compiler caches, fixed datasets or evaluation downloads.

## 12. Common errors and confusing output

| Message/symptom | What to do |
|---|---|
| `can't open file .../collect.py` or `train.py` | Run `pwd` and enter the model folder from section 1. |
| Native Assimp `Mesh ... has no faces` / collector SIGABRT | Empty GLB primitives are now removed from a temporary scene copy before loading. Cached assets and nonempty geometry are preserved; no scenes are excluded. The HSSD scene-14 fix passed real OptiX flight checks. |
| `no solution.py or solution/__init__.py` while bundling | Pass `solution`, not the parent `1_generative_ssm` folder. |
| Missing `deployment.json`, `.npz` or `.onnx` | Finish training/export, or retry `export.py --checkpoints checkpoints/RUN`. |
| `full-resolution training requires 640x480 recordings` | Use `--stream-data` or recollect at `--width 640 --height 480`. |
| `Disk quota exceeded` | Check `quota -s`, clear disposable package cache if appropriate, or move rolling recordings to `/tmp`. |
| `Need at least two scenes` | For fixed-data training, collect enough distinct scene IDs; rolling mode creates separate validation automatically. |
| Collector returns no usable batch | Inspect the collector error; a single recording may exceed its data slot. Increase `--max-data-mib` or reduce `COLLECT_STEPS`. |
| CUDA unavailable / simulator initialization failure | Activate `myenv`, check `nvidia-smi` and PyTorch CUDA, and inspect the native build error. |
| A changed patch stem requires supervised training | Add `--sl` for compatible-width changes; the new large preset needs a fresh run. |
| PPO-only is missing a component checkpoint | Use `--from-checkpoints checkpoints/RUN --sl` to train missing components before PPO. |
| PPO validation never improves | Export retains the selected SL baseline or earlier best candidate; inspect `ppo_metrics.jsonl`. |
| ONNX/OpenCV compatibility failure | Keep the checkpoint and diagnose export; the exporter does not silently substitute a different model. |
| `out of fuel`, memory error, timeout, or `limits: FAIL` | Read the report and stderr; the selected model has not passed that resource test. |

Native compilation can print many warnings while it continues. Watch the original
command's final result. For OpenCV build progress, in a second terminal in this folder:

```bash
tail -f ../../../build/runtime/logs/opencv-build.log
```

Press Ctrl-C in that **second** terminal to stop viewing the log. The build in the
first terminal continues. The intermediate OpenCV shared-library link can fail in
this runtime recipe, which uses the compiled objects for static linking; success
is established by the final bundle and WASM checks, not by one warning line.

## 13. Help and tests

```bash
python train.py --help
python collect.py --help
python export.py --help
python validate.py --help
python verify_limits.py --help
python distill.py --help
python -m students.export --help
python -m viobench --help
```

To run the project's model, data-lifecycle, PPO and export regression tests:

```bash
python -m pip install --no-cache-dir pytest
python -m pytest tests -q
```

Tests use temporary data and preserve your real checkpoints and exported solution.
The prebuilt-runtime integration test skips when the runtime is absent.
For model internals and the limits of the measured resource profile, continue to
[MODELS_README.md](MODELS_README.md).

## 14. Distill teachers into fuel-budgeted students

Use a complete teacher checkpoint set: `checkpoints/RUN/post_sl` for its SL
version, or `checkpoints/RUN` for the final PPO-selected version. Distillation
freezes both teacher components and trains four separate compact students.
The teacher's checkpoints are preserved.

With `myenv` active and your terminal in `1_generative_ssm`, train all four
students with background recording generation:

```bash
python distill.py --teacher checkpoints/RUN --stream-data --max-data-mib 3072
```

To use the post-SL teacher instead:

```bash
python distill.py --teacher checkpoints/RUN/post_sl --stream-data --max-data-mib 3072
```
Distillation also uses the current collection defaults. To override scene counts,
flight lengths or collection budgets, supply `--collection-config FILE.json`
with the uppercase collection settings from `config.py`.


The command prints a new directory such as `checkpoints/distill-20261112-...`.
Below, replace `DISTILL_RUN` with that directory's actual name. All profiles share
one teacher forward pass per training minibatch; targets stay in RAM. The same
three-slot recording budget and bounded reuse policy apply. Validation recordings
stay fixed. Defaults are 100 maximum epochs and patience of 8 non-improving fresh
batches (8 epochs for fixed data), independently for each student.

On CUDA, distillation runs each active student on its own CUDA stream and host
worker. A separate stream prepares the next minibatch's shared teacher targets
while those students train. Hyperdrone continues collecting the next recording
batch in its existing background process. Target prefetch is limited to one
minibatch ahead; every student finishes its current update before advancing.
Validation also runs independent tiers concurrently, using FP32 and no teacher
forward pass. Neural layers use the existing BF16 setting during training;
student integration and losses stay FP32. CUDA loader workers use `spawn` to
avoid forking with live CUDA scheduling threads. CPU runs stay serial.
The producer also fetches/collates the next minibatch, so that work can overlap
current updates. Teacher and students share one input transfer, and each
student caches BF16 weight casts across its camera ticks within one forward
pass. The cache ends before optimizer updates. Persistent loader workers and
fused CUDA AdamW are enabled by default; use `--no-persistent-workers` or
`--no-fused-adamw` to compare them independently.

The run's `performance.jsonl` records windows/second, active profiles, student
updates, fresh-recording wait and CUDA peak allocated memory. Windows/second
includes validation and data wait, and counts each shared window once. Compare
against `--no-parallel` on the same GPU/data to measure the scheduling benefit:

```bash
python distill.py data --teacher checkpoints/RUN --device cuda --skip-export
python distill.py data --teacher checkpoints/RUN --device cuda --skip-export --no-parallel
```

Concurrent student activations and teacher inference need more GPU memory than
serial scheduling. Reduce `--batch` or select fewer `--profiles` if needed;
`--no-parallel` disables both CUDA overlap and teacher-target prefetch. Actual
speedup depends on GPU capacity and workload; continuous saturation is not
guaranteed. The recording disk limit does not cap GPU activation memory.

The local RTX 5080 compute probe measured **11.44 → 17.02 shared windows/s
(1.49x)** for the full-size teacher and all tiers, using batch four and 101-tick
windows. It excludes collection, loaders and validation. A separate two-epoch
CUDA smoke run verified real background Hyperdrone collection and loader workers.
See [the measured results and reproduction command](verification/README.md#parallel-distillation-throughput-2026-11-12).

| Profile | Maximum fuel/frame | Student architecture | Parameters |
|---|---:|---|---:|
| `100` | 100 | IMU-only affine acceleration model; fixed attitude and 20 Hz integration | 6 |
| `10k` | 10,000 | Linear correction head, 12 sparse grayscale samples, frame-level IMU integration | 288 |
| `1m` | 1,000,000 | 192-unit recurrent network, 48 samples, per-IMU-sample integration | 58,569 |
| `100m` | 100,000,000 | 1,792-unit recurrent network, 768 samples, per-IMU-sample integration | 5,994,249 |

These are the current default profiles. The [larger sizing candidates](#choose-model-size-against-deployment-limits)
were measured separately. `--profiles` selects existing tiers; it does not
activate the larger architectures. Changing a student's architecture requires
matching model settings before retraining; editing a trained checkpoint's
`student.json` alone cannot resize its weights.

These students request **320x240** ABI images and run as native C/WASM. They
predict motion corrections directly; the teacher's embedding predictor or
optional RGB generator stays on the host. The 10K target cannot use the Python teacher runtime: even
the Python identity solution measured about 87K–214K fuel/frame in a local probe.

After training, each profile is compiled and checked against its exact cap for
1,200 synthetic frames. A passing module is saved as:

```text
checkpoints/DISTILL_RUN/100/native/student.wasm
checkpoints/DISTILL_RUN/10k/native/student.wasm
checkpoints/DISTILL_RUN/1m/native/student.wasm
checkpoints/DISTILL_RUN/100m/native/student.wasm
```

### Literal 100-fuel IMU-only tier

```bash
python distill.py --teacher checkpoints/RUN --stream-data --profiles 100
```

This is a deliberately restricted model with six learned coefficients: one gain
and one offset for each accelerometer axis. It ignores the RGB buffer entirely,
uses only the last accelerometer sample in each packet, and ignores gyro. It
keeps orientation fixed at identity and integrates position/velocity at the ABI's
fixed 20 Hz using semi-implicit Euler. Camera calibration is folded into the
coefficients during initialization. It requires the ABI's standard ten-row IMU
packets; irregular rates/packet sizes are unsupported at this tier.

The teacher still uses images when creating distillation targets. The 100-fuel
student learns to match teacher acceleration and trajectory targets; it is an
IMU-only motion prior, with substantial limitations during rotations and changes
in gravity direction. The resource probe passed 1,200 frames at **99 fuel/frame**.
There is only one fuel unit of margin with the tested compiler, so every export
must pass the exact cap; compiler/code changes can affect whether it fits.

Each `native/` directory also contains `limits.json` and
`submission_limits.json`. Over-budget candidates fail verification and are not
promoted to `student.wasm`. A short check does not establish complete-episode
behavior; the default uses 1,200 frames. Actual accuracy needs benchmark evaluation.

Native export uses the existing `viobench/solution/build/toolchain/wasi-sdk`.
If it is missing, install the repository's native toolchain once:

```bash
bash ../../../toolchain/download_dependencies.sh native
```

Alternatively, pass `--wasi-sdk /path/to/wasi-sdk` to distillation or student
export. The code uses C23 binary embedding, supported by the repository's
wasi-sdk 34. No Python runtime build or `viobench bundle` is needed for students;
native export already produces a standalone `.wasm`.

### Select tiers, change training length, or save checkpoints only

```bash
python distill.py --teacher checkpoints/RUN --stream-data --profiles 10k 1m --epochs 200 --patience 12 --skip-export
```

`--profiles` accepts any subset of `100 10k 1m 100m` (all four by default). Each profile saves its own
`student.pt`, `student.json` and `metrics.jsonl`. The run root records the teacher
path, model settings and checkpoint hashes in `distillation.json`.

Train from existing collector recordings instead of background generation:

```bash
python distill.py data --teacher checkpoints/RUN --no-stream-data
```

Recordings must match the teacher's required image resolution and contain at
least two scene IDs for fixed-data training.

### Export saved student checkpoints later

```bash
python -m students.export --checkpoints checkpoints/DISTILL_RUN/100
python -m students.export --checkpoints checkpoints/DISTILL_RUN/10k
python -m students.export --checkpoints checkpoints/DISTILL_RUN/1m
python -m students.export --checkpoints checkpoints/DISTILL_RUN/100m
```

These commands rebuild and enforce the saved tier's fuel cap before publishing
`native/student.wasm`. `--out` chooses a different native export directory.

### Evaluate students using their own fuel limits

```bash
python -m viobench evaluate checkpoints/DISTILL_RUN/10k/native/student.wasm --limits checkpoints/DISTILL_RUN/10k/native/submission_limits.json -o report-student-10k
```

Substitute `100`, `1m` or `100m` in both paths to test another tier. Use separate report
directories for the SL-teacher and PPO-teacher students. Compare Mean Sim(3) ATE
on held-out episodes; budget compliance alone does not establish retained accuracy.

Recheck a 10K module's resources directly:

```bash
python verify_limits.py checkpoints/DISTILL_RUN/10k/native/student.wasm --step-fuel 10000 --report student-10k-limits.json
```

Distillation loss weights are configurable: `--kd-weight 1`,
`--teacher-state-weight 0.5` and `--ground-truth-weight 1`. See the technical guide
for their meaning. The [saved resource probes](verification/README.md) use untrained
test weights; each trained export is checked again.
