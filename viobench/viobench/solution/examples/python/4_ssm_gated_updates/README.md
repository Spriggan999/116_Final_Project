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
75/25 training replay and hybrid competence/budget promotion. Every model
continues training while its peers qualify. Promotion checks use the current held-out curriculum distribution,
as described below.
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
The recording ring is limited to 12 GiB across current data, one prefetched
batch and fixed validation (`--max-data-mib`). The default batch contains two
scenes with one flight each, collected one instance at a time. The collector
stores exactly 20 Hz images and every 200 Hz IMU sample. This reduces memory and
disk use without shortening the 60-second horizon. Compression streams directly
to a bounded temporary file, rather than buffering an entire NPZ in RAM. Captures remain uint8 in GPU buffers; normalization and
checkpointed CNN work happen in small chunks instead of retaining full float RGB
flights. The separate JEPA prediction/auxiliary prediction losses are zero when a
window contains only the initial image; there is no future visual target yet.
Alignment is zero until at least three camera observations are available.

The new schedule applies only to this experiment. The student schedule in
`1_generative_ssm` stays unchanged. Old staged checkpoints cannot resume under
the changed horizon policy; use fresh weights. W&B/performance records include
`horizon_imu_steps` and `horizon_seconds` as well as sample counts.

## Storage and memory protection

The previous `/tmp` recording path is unsafe on hosts where `/tmp` is tmpfs:
those files consume RAM. The default scratch/cache root is now
`/var/tmp/gated-updates-UID`, verified to be disk-backed. RAM-backed
`--resource-dir` and recording `--stream-dir` paths are rejected.

The launcher installs a one-second resource watchdog for the trainer and its
child processes. Defaults are:

| Resource | Default |
|---|---:|
| Minimum filesystem free space | 10 GiB |
| Minimum remaining user quota, when available | 1 GiB |
| Minimum available system RAM | 8 GiB |
| Maximum combined trainer/child RSS | 24 GiB |
| Combined application scratch and cache budget | 16 GiB |
| Per-run checkpoints and log budget | 1 GiB |
| Recording-ring budget within scratch | 12 GiB |

Quota checks use `quota -w` when supported; `df` alone is insufficient on the
instructional home filesystem. Scratch/cache and run-directory sizes are checked
periodically, with additional checks before checkpoint writes. These are
application headroom thresholds, not an OS-level reservation against unrelated
programs or instantaneous external allocations.

Under pressure the watchdog cancels the collector and interrupts training so
normal cleanup can retire that run's temporary/current/prefetched recordings.
Completed model checkpoints stay intact. Interrupted atomic checkpoint writes
remove their temporary file and retain the previous checkpoint. The code never
automatically deletes another run, user datasets, or old checkpoints.

Validation uses one retained set per run on scratch, referenced by its manifest.
Multiple retained validation sets and caches count toward the common 16 GiB
budget. If it fills, free unneeded data explicitly or choose a larger disk-backed
resource directory; the launcher stops rather than silently growing storage.
Old sets are not automatically evicted, because their manifests may still be
needed to reproduce results. `/var/tmp` is scratch storage, not a permanent
backup; model checkpoints remain in the project unless `--out` changes that.

Hyperlux, CUDA, Torch, W&B staging, scene assets and temporary build caches are
redirected into the watched scratch root. Existing scene caches remain usable
as read-only search sources. Native build parallelism is capped at two jobs.
Data-loader workers default to zero to avoid duplicating large in-memory
recordings across worker processes. CUDA model scheduling remains automatic.

The same safeguards wrap standalone `evaluate.py`. Flags use MiB, for example
`--resource-dir /DISK/PATH --max-scratch-mib 16384 --min-disk-free-mib 10240`.
The remaining flags are `--min-quota-free-mib`, `--min-ram-free-mib`,
`--max-host-memory-mib`, and `--max-run-mib`. Lowering headroom weakens the
protection; defaults are intended for the current 62 GiB RAM / 16 GiB GPU host.
`train.py --help` lists resource and training controls. No full training run is
started automatically after a code update. Restart through `run.sh` to use the
new guards; previously running Python processes do not reload them.

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

Curriculum promotion uses up to 12 fixed ProcTHOR held-out windows, matching
**the current horizon, motion bin, IMU noise and visual noise**. Validation has
no easier-level replay or harder-level exploration, and uses fixed examples and
augmentation seeds. Motion filtering remains active during IMU and visual-noise
phases too; an unavailable motion bin uses the nearest available lower bin.
The supervised state-plus-alignment loss (excluding JEPA) remains logged as
`MODEL/procthor_curriculum_val/validation_loss`. Promotion instead uses physical
errors after the initialized sample: position-vector RMSE, velocity-vector RMSE,
orientation geodesic RMSE in degrees, and angular-velocity-vector RMSE. Squared
errors are aggregated across all probe samples before taking their square root,
so minibatch size does not change the criteria.

The hybrid rule is:

1. Train for at least **six fresh recording checks** per stage (the maximum of
   `--stage-epochs` and `--curriculum-min-batches`). Reused epochs do not count.
2. Advance when **all four physical errors meet their targets for three
   consecutive checks**, for every active model. Improvement does not reset this
   competence history. Qualified models keep training; regression removes readiness.
3. At **18 fresh checks**, advance even if targets are unmet, explicitly logging
   `budget_exhausted` and `forced_promotion=true`. This is a bounded fallback,
   not a claim of mastery. It also works when all models are struggling.

Provisional target defaults are `(base + rate * horizon_seconds) * difficulty_scale`:

| Metric | Base | Per second |
|---|---:|---:|
| Position RMSE (m) | 0.02 | 0.02 |
| Velocity RMSE (m/s) | 0.10 | 0.01 |
| Orientation RMSE (degrees) | 2.0 | 0.10 |
| Angular velocity RMSE (rad/s) | 0.05 | 0.001 |

`difficulty_scale = 1 + 0.5*motion_level + imu_level + visual_level`, using levels
normalized to 0–1. These targets need empirical calibration; they are not measured
accuracy guarantees. Alignment still contributes to training and the logged stage
loss; it is not a competence threshold because short windows may have fewer than
three real images. Full-flight ATE never gates promotion. The former near-best
plateau remains a diagnostic only. Public validation and the original curriculum
in `1_generative_ssm` are unchanged.

Use `--curriculum-max-stage-checks` and `--curriculum-accuracy-patience` to change
the budgets. `--curriculum-accuracy-targets FILE.json` accepts overrides mapping
`position_rmse_m`, `velocity_rmse_mps`, `orientation_rmse_deg`, and
`angular_velocity_rmse_radps` to `[base, per_second]`. Omitted keys use defaults.
For example: `{"position_rmse_m": [0.03, 0.02]}`. The accuracy patience must be
no greater than the minimum budget, which must not exceed the maximum budget.
The overall epoch ceiling still applies and can stop training before curriculum
completion.

W&B and local records retain the stage loss, horizon, and difficulty levels under
`procthor_curriculum_val/*`, with physical RMSEs, their effective targets,
competence, and stage-check count. W&B `curriculum/*` also logs readiness reasons,
error-to-target ratios, and forced advances. Detailed promotion events are saved
in `curriculum_metrics.jsonl`. New-policy resumes retain saved budgets, targets,
and check history; W&B records the effective restored settings.

Full-flight, unaugmented ProcTHOR validation still runs every epoch for
separate diagnostics, checkpoint selection and final early stopping. `procthor_data/manifest.json` points to the exact retained validation recordings
on disk-backed scratch for standalone replay; the files are not copied into the
home-directory quota. Keep those referenced files while you need reevaluation.

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

The public suite remains unchanged: all ten complete 60-second flights with
its existing separate metrics and no influence on promotion. Complete ProcTHOR
metrics keep their `procthor_val/*` names; stage probes have the separate
`procthor_curriculum_val/*` namespace and are recorded only on fresh-data checks.

When resuming a run from either former promotion policy, weights, optimizer,
horizon and difficulty are retained. Incompatible plateau/readiness history is
reset, including paused peers and holdout counters. The new minimum budget starts
from zero to collect physical accuracy measurements. Hybrid-policy checkpoints
retain their check history on subsequent resumes. Restart/resume is needed to
apply this code to a running process; editing files does not stop the live run.

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

Repeat any custom host/collection options from the original invocation. These
options carry over automatically unless you override them: training policy,
architecture, visual-update bounds, `--fixed-horizon`/`--window`,
`--max-fresh-checks`, `--resume-lr` and `--lr-schedule`.
Validation collection is deterministic, so an exact resume reuses the parent's
recordings (`procthor_data/manifest.json`) while they exist instead of collecting
another ~0.9 GB copy. `latest.pt` embeds the recovery and fresh-check state in
the same atomic write as the weights. An interruption can no longer leave a run
unresumable. `recovery_state.json` and `experiment_state.json` are readable mirrors.
Without `--warm-start-stage`, warm-start loads selected best weights and starts
fresh optimization. Checkpoint files are local trusted experiment artifacts.

Numerical recovery checks gradients before every optimizer/EMA update. A nonfinite
loss or gradient norm triggers one full-precision retry of the same minibatch.
If the retried loss is finite but gradients remain invalid, that update is skipped
without changing weights, Adam moments, or EMA. Per-epoch logs and W&B expose
`MODEL/train_numerical_retry` and `MODEL/train_optimizer_update_skipped` as
sample-weighted fractions. Persistent nonfinite forward losses still stop with
component diagnostics. Resume from the last completed checkpoint to use this fix.

## Corrected training, LR sweep and persistent-vision pilot

Every option below is opt-in. Without them, training and exact resumes behave as
before (`legacy` policy and architecture).

**`--training-policy corrected_v1`** changes supervision and model selection:

- Velocity targets are absolute (`R0ᵀ v(t)`), consistent with the position target
  `R0ᵀ[p(t)−p(0)]`. Legacy targets subtracted the initial velocity from velocity
  but not from position: about 4 m disagreement at 20.48 s on held-out flights.
- The initialized sample is excluded from state losses.
- Gravity is the mean of up to ten accelerometer samples ending at the window
  start, the same rule as public-flight inference. Recordings store a zero IMU
  placeholder at tick 0; a window starting there uses the first real sample.
  Legacy windows averaged the ten samples *after* the start.
- Stage-best checkpoints are chosen on fresh checks by the mean squared
  error-to-target ratio of the four current-stage physical RMSEs. Global-best
  `student.pt` and early stopping use full-flight **unaligned** ATE RMSE.
  Sim(3) ATE is still logged.
- `--curriculum-budget-action` defaults to `stop` (otherwise `advance`): when the
  stage check budget runs out without competence, the run checkpoints and stops
  with `stop_reason=stage_budget_exhausted` instead of forcing promotion.

Exact resume (`--warm-start-stage`) requires the same policy and architecture.
To move a legacy checkpoint to the corrected policy, use a weight-only warm start
(`--init-weights global|latest|stage`, which resets optimizer moments).

**Learning rate.** `--resume-lr RATE` is applied after every optimizer restore,
including stage transitions; previously, restored optimizer state overwrote the
rate. `--lr-schedule FILE` takes `[{"horizon_seconds": 0.16, "lr": 3e-4}, ...]` and
applies the last entry at or below the current horizon. `--resume-lr` wins over
the schedule. `MODEL/learning_rate` records the effective rate.

**Fixed-horizon experiments.** `--fixed-horizon --window N` trains at one horizon
and motion level 0 without promotion. `--max-fresh-checks N` stops after N fresh
checks (`stop_reason=fresh_check_limit`), and the count carries across exact
resumes. `--validation-manifest FILE` reuses fixed held-out recordings. Any run's
`procthor_data/manifest.json` works, if its `/var/tmp` recordings still exist.

**`--architecture persistent_visual_v2`.** At each real capture after the first,
a visual GRU cell updates the IMU GRU state with the current and previous
embeddings, their difference and elapsed time. Zero-initialized heads add a
bounded velocity update (`--velocity-update-bound`, default 0.5 m/s) and a
right-multiplied orientation update (`--orientation-update-bound`, 0.02 rad).
Position never jumps. Batched training uses the same event order as
`stream_step`; the tests check parity. Start from legacy weights with a weight-only
warm start plus `--allow-architecture-transfer`.

Added telemetry: per-module gradient norms, `gradient_clipped`,
`correction_saturation_fraction`, `rotation_orthogonality_rmse`, and
`visual_{velocity,orientation}_update_rms`.

### Diagnostics (never used for selection)

```bash
python 4_ssm_gated_updates/diagnostics.py --run RUN --weights latest \
  --horizon 20.48 --max-windows 48 --reset-seconds 1 5
```

This re-integrates each model's own corrections on held-out windows under five
modes: deployable `normal`, `oracle_velocity`, `oracle_gravity`, `oracle_both`,
and ground-truth state resets every N seconds (a teacher-forcing diagnostic).
`replay_max_deviation` must stay small, because it checks that the re-integration
reproduces the model's output. Results go to `RUN/diagnostics/`.

### Representative-horizon LR sweep

`gated_lr_sweep.py` runs Baseline at 0.16, 2.56, 10.24 and 20.48 s with
`1e-5, 3e-5, 1e-4, 3e-4`:

1. Each candidate gets 30 fresh checks.
2. The best two continue to 100 checks, and Combined confirms them.
3. Ranking uses the last-ten-check mean unaligned ATE. A candidate is rejected if
   any physical RMSE is more than 10% worse than `3e-4` or more than 1% of updates
   were skipped. ATE results within 2% are tie-broken by fewer skipped updates,
   then the lower LR.
4. If adjacent horizons pick different winners, `refine` adds 0.64 s or 5.12 s.

Each horizon starts every candidate from the same checkpoint: the saved run
nearest on a log scale, with ties going to the more-trained run. Override with
`--source 0.16=PATH`. These are adaptation experiments, not historical snapshots.

Run it in a tmux server that survives SSH logout on this host
(`KillUserProcesses=yes`):

```bash
cd .../solution/examples/python
SWEEP=4_ssm_gated_updates/checkpoints/lr-sweep
OUT=/var/tmp/gated-lr-sweep-$(id -u)    # run outputs: the home quota is only 4 GB
PY=/home/cc/eecs116/fa26/class/eecs116-aaq/.conda/envs/myenv/bin/python
$PY 4_ssm_gated_updates/gated_lr_sweep.py plan --plan $SWEEP/plan.json --out $OUT \
  --validation-manifest 4_ssm_gated_updates/checkpoints/resumed-20261117-220152/procthor_data/manifest.json
# Unattended: run, refine and repeat until every horizon has a decision,
# then write $SWEEP/lr_schedule.json. Safe to rerun after an interruption.
tmux -L persistent new-session -d -s lr-sweep -c "$PWD" \
  "PYTHONUNBUFFERED=1 $PY 4_ssm_gated_updates/gated_lr_sweep.py auto --plan $SWEEP/plan.json 2>&1 | tee -a $OUT/driver.log"
$PY 4_ssm_gated_updates/gated_lr_sweep.py report --plan $SWEEP/plan.json   # any time
# Afterwards, queue and run the 20.48 s redesign pilot:
$PY 4_ssm_gated_updates/gated_lr_sweep.py pilot --plan $SWEEP/plan.json
$PY 4_ssm_gated_updates/gated_lr_sweep.py run --plan $SWEEP/plan.json --limit 100
$PY 4_ssm_gated_updates/gated_lr_sweep.py report --plan $SWEEP/plan.json
```

(Start the `persistent` tmux server once with `systemd-run --user --scope tmux -L
persistent new-session -s vio`. New sessions on that server outlive SSH logout.)

Tasks run one at a time, each logging to `<run>.log`, and completed tasks are
skipped. After an interruption, rerunning continues the unfinished task from its
last checkpoint into a new `-rN` directory, keeping its fresh-check count. If no
epoch had finished, it restarts the task. Old directories are listed under the
task's `attempts` and never reused or deleted. A task that fails three launches
in a row (for example, an LR that diverges) is marked `failed` and ranked as
numerically invalid. If the 3e-4 reference diverges, the other candidates are
compared by raw ATE instead of the ratio to it. The pilot adopts
`persistent_visual_v2` only if, for both Baseline and Combined, unaligned ATE
improves by at least 20% with no physical RMSE more than 10% worse.

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
