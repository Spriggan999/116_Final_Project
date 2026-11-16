# Four GRU–vision JEPA experiments

These are GPU research models, trained from ground truth without a teacher
checkpoint. The existing `1_generative_ssm` experiments are unchanged.

Every model uses a 256-unit, one-layer GRU on six-axis IMU at 200 Hz, the
teacher's learned RGB encoder with a 48D output on real observations near 20 Hz,
and a `304 → 256 → 9` fusion MLP. Missing images supply exact zero vectors.
The nine bounded corrections feed the teacher's IMU integration equations to
produce position, orientation, velocity, and angular velocity at every IMU tick.
The GRU receives only IMU; visual corrections do not change its hidden state.

| Profile | Visual JEPA | Training-only GRU JEPA | Image-gated innovation correction |
|---|---|---|---|
| `baseline` | yes | no | no |
| `aux_jepa` | yes | yes | no |
| `innovation` | yes | no | yes |
| `combined` | yes | yes | yes |

The common predictor uses two preceding real embeddings, their difference,
mean interval IMU, elapsed time, and previous capture spacing (`152 → 256 → 48`).
Its prediction is formed before access to the target image. The auxiliary head
predicts the current visual EMA target from the IMU GRU state (`256 → 256 → 48`).
Innovation models compare the real embedding with the common visual prediction;
a learned MLP adds nine residuals before the output bound, only on valid real
captures. Its output layer starts at zero. Initial bootstrap has no innovation.

The joint objective is:

```text
3 position + rotation + velocity + 0.1 angular_velocity
+ 0.1 Sim(3) alignment
+ 0.1 (JEPA prediction + content + variance + 0.01 covariance)
+ 0.1 GRU JEPA prediction   # auxiliary variants only
```

State losses run at the full IMU rate; alignment uses camera timestamps.
Rotation uses squared geodesic error, other state terms use Smooth L1.
The EMA target (decay 0.99) has no gradients. Encoder and state branches train
jointly. Shared components start from the same seeded weights across variants.
Stored parameter counts including the EMA encoder are approximately 507K, 585K,
587K, and 666K respectively.

## Run on the GPU computer

Activate `myenv` and run from `solution/examples/python`:

```bash
conda activate /home/cc/eecs116/fa26/class/eecs116-aaq/.conda/envs/myenv
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
bash 4_ssm_gated_updates/run.sh
```

The launcher uses the interpreter in `myenv` directly; override `GATED_PYTHON`
on another computer. An explicit run directory is useful for later evaluation:

```bash
bash 4_ssm_gated_updates/run.sh --out 4_ssm_gated_updates/checkpoints/experiment1
```

This starts all four models in one process with separate weights/optimizers,
shared recordings and augmentation, and online W&B in
`hyperdrone-vio-fuel-aware`. Outputs go to a fresh run directory. The launcher starts a fresh run; it never resumes an existing run implicitly.

Fresh collection requires HyperDrone, scene assets and a working CUDA device.
The `collection_60s.json` preset retains training scenes 0–191 and eight
held-out scenes 192–199, and collects 12,001 IMU samples per flight. The horizons
are **1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1,024, 2,048, 4,096, 8,192,
then 12,000 IMU intervals**. At 200 Hz these range from 0.005 to 60 seconds.
Each training window includes its initial state, so sample counts are the listed
values plus one (`2, 3, 5, ..., 8193, 12001`). These are IMU steps, not 20 Hz
camera frames; a public 60-second flight still has 1,200 camera frames.

Ten motion levels at each horizon, then ten IMU and ten visual-noise levels use
the existing staged student gates,
75/25 replay, plateau qualification and five-epoch sole-holdout exception.
Early stopping is protected until the final curriculum stage. The safety ceiling
is 1,500 epochs. `--help` exposes inherited host-training options; teacher
distillation, architecture calibration and WASM export are disabled here.

Useful overrides:

```bash
bash 4_ssm_gated_updates/run.sh --wandb-mode offline --no-public60
bash 4_ssm_gated_updates/run.sh --profiles baseline combined --batch 4
```

For existing recordings, use `--no-stream-data` and a positional NPZ directory.
Its `scene_id` fields determine the split; filenames do not. The directory must
contain training scenes in 0–191 and validation scenes in 192–199, with enough
samples for the selected maximum horizon. Default production input is 640×480.
The longer recordings use a 24 GiB upper bound for the three-slot scratch ring
(current batch, prefetched batch, fixed validation), configurable with
`--max-data-mib`. Captures remain uint8 in GPU buffers; normalization and
checkpointed CNN work happen in small chunks instead of retaining full float RGB
flights. The separate JEPA prediction/auxiliary prediction losses are zero when a
window contains only the initial image; there is no future visual target yet.
Alignment is zero until at least three camera observations are available.

The new schedule applies only to this experiment. The student schedule in
`1_generative_ssm` stays unchanged. Old staged checkpoints cannot resume under
the changed horizon policy; use fresh weights. W&B/performance records include
`horizon_imu_steps` and `horizon_seconds` as well as sample counts.

## Parallel versus sequential: measure on the target GPU

`--parallel-mode auto` measures serial host updates, serialized streams, pairs,
three-way and four-way CUDA stream groups, and alternative groupings. Each
candidate updates every active model on the same prepared batch: forward,
state/JEPA/alignment losses, backward, clipping, AdamW and EMA. There are two
warmup updates followed by three repeats of five timed updates. Probes restore
model/EMA/optimizer/RNG state and leave no training updates behind.

Choose the fastest schedule with at least 15% device-memory headroom. If
candidates are within 3%, prefer lower concurrency. Retune when horizon or active
model membership changes. OOM recovery can reduce batch size or disable input
prefetch before any production update. Record candidate timings and the chosen
groups in `scheduling.json` and W&B. Shared data preparation and background
collection are overlapped during production; the candidate timings isolate the
model updates on an identical prepared input. Epoch throughput includes data
wait and validation, and should be used to assess overall performance.

This is concurrent/serial **updates on shared batches**, not four unrelated
training processes. It preserves shared curriculum progression. CUDA smoke checks at both 1 and 12,000 intervals passed on the RTX 5080 using
`myenv`, including real forward/backward updates with 640×480 captures. Schedule
selection is still measured again for each production horizon and active group;
these smoke checks are not a throughput or accuracy benchmark.

An optional standalone benchmark uses recorded data at 1-, 1,024- and 12,000-interval
horizons without retaining optimizer updates:

```bash
python 4_ssm_gated_updates/benchmark.py PATH_TO_RECORDINGS \
  --out 4_ssm_gated_updates/scheduling_benchmark.json --batch 1
```

For controlled comparisons, `--parallel-mode serial` forces serial execution;
`--parallel-mode streams --max-concurrent 2` forces paired streams.

## Separate validation suites

ProcTHOR validation runs on complete held-out flights every epoch and alone
controls checkpoint selection, curriculum and stopping. The exact recordings
are retained in the run's `procthor_data` directory for standalone replay.

After training, the selected best checkpoints are evaluated on all ten
`viobench.episodes.public` flights, downloaded through the normal conta cache.
Every flight uses all 1,200 images (nominally 60 seconds; timestamps span 59.95s),
with intervening IMU measurements and one persistent recurrent/integrator state.
Images are decoded lazily. Camera-frame transformations include the camera lever
arm and its angular contribution to velocity. Scoring uses VIOBench's scorer.
Public results never control training or model selection. Disable automatic
public evaluation with `--no-public60`, then run it separately later.

Each model has six distinct W&B headline keys:

| Metric | ProcTHOR | Public |
|---|---|---|
| Validation loss | `MODEL/procthor_val/validation_loss` | `MODEL/public60/validation_loss` |
| Sim(3) ATE | `MODEL/procthor_val/sim3_ate_rmse_m` | `MODEL/public60/sim3_ate_rmse_m` |
| Unaligned ATE | `MODEL/procthor_val/unaligned_ate_rmse_m` | `MODEL/public60/unaligned_ate_rmse_m` |

ATE values are equal-per-episode means of trajectory position RMSEs; unaligned
ATE fits no transform. Validation loss contains the weighted state loss plus
alignment, while JEPA and full training-objective terms are logged separately.
ProcTHOR state loss is sampled at IMU ticks; public state loss is sampled at
camera timestamps in benchmark camera coordinates. Do not pool these losses.
Full-flight metrics, raw losses, coefficients, weighted contributions, gradients,
innovation magnitude, curriculum events, memory and throughput are also logged.

Standalone evaluations create distinct `procthor/` and `public60/` reports and
separate W&B validation jobs; each includes the source training-run path.
They save manifests, per-episode results, trajectories and `summary.json`.

```bash
python 4_ssm_gated_updates/evaluate.py --run PATH_TO_RUN \
  --suite procthor --device cuda --wandb-mode online
python 4_ssm_gated_updates/evaluate.py --run PATH_TO_RUN \
  --suite public60 --device cuda --wandb-mode online
```

The public catalog names ProcTHOR scene 1, which is also inside the inherited
training scene range. These public episodes are separate recordings, not a
scene-disjoint holdout. The ProcTHOR 192–199 set remains scene-disjoint.

## Checkpoints and recovery

Each model directory contains `student.json`, global-best `student.pt`,
stage-best weights plus matching optimizer in `stage_best.pt`, and `metrics.jsonl`.
The root contains `latest.pt`, recovery/curriculum metadata, saved host settings,
performance records and scheduling measurements. EMA weights travel with model
checkpoints. Existing outputs are never overwritten by a new training invocation.

Continue an interrupted run into a fresh directory, preserving latest weights,
optimizer/EMA/RNG state, epoch, curriculum gates and stopping counters:

```bash
bash 4_ssm_gated_updates/run.sh \
  --warm-start PATH_TO_RUN --warm-start-stage \
  --out 4_ssm_gated_updates/checkpoints/recovered
```

Repeat any custom host/collection options from the original invocation.
Without `--warm-start-stage`, warm-start loads selected best weights and starts
fresh optimization. Checkpoint files are local trusted experiment artifacts.

## Verification

```bash
python -m unittest discover -s 4_ssm_gated_updates/tests -v
```

The tests require only the training environment, without pytest. They check all
four models' causality, sequence/streaming agreement, missing-image zeros,
JEPA/EMA gradient paths, camera transforms, recovery, the inherited horizons,
the new geometric horizon schedule, shortest-window losses, compact capture equivalence,
and a synthetic uninterrupted 1,200-frame camera flight. CUDA scheduler rollback/update
checks run when a GPU is accessible and otherwise report a skip. CPU training,
curriculum, resume and standalone evaluation smoke results are recorded in
`verification.json`. Synthetic checks do not establish learned accuracy on the
real public or held-out recordings. WASM export and fuel limits are outside scope.
