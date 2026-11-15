# Fresh supervised student training

The preferred fresh-training entry point is `train_students.py`. It trains the
eight fuel-budgeted student profiles directly from recorded ground
truth. No teacher weights are loaded, teacher inference is skipped, and neither
teacher corrections nor teacher trajectories contribute to the loss. The state
loss is `3 * position + velocity + rotation + 0.1 * angular_velocity`, with the
existing Smooth L1/geodesic components. Fresh runs also add a default
`0.1 * JEPA_embedding_loss`, using a training-only EMA encoder (decay 0.99)
and prediction/content/variance/covariance terms. A default
`0.1 * Sim3_alignment_loss` additionally supervises trajectory shape over the
current horizon (full 20-second flights at the maximum horizon). Set
`--alignment-weight 0` to disable alignment. Disable JEPA with
`--embedding-weight 0`. All components, including raw values, effective coefficients and weighted
contributions for position/velocity/rotation/angular velocity and each JEPA
term, go to W&B and each profile's `metrics.jsonl`; validation state loss and
Sim(3) ATE retain their existing selection/curriculum roles. Unaligned mean
translation error (`validation_unaligned_mte_m`) and unaligned ATE RMSE are
also logged to W&B and `metrics.jsonl` as diagnostics. See
[all loss functions](LOSSES_README.md). Temporal backbone families and fuel caps
are preserved, with new learned visual/embedding frontends described in
[the architecture comparison](STUDENT_ARCHITECTURES_README.md). This does not establish that supervised training will outperform
distillation; compare exported WASMs on representative full-length episodes.

## Train all eight with automatic GPU overlap

No new scheduler is needed: `train_students.py` already supports multiple
independent student models in one process, with memory-aware scheduling and
shared fresh recordings. Create a tmux session:

```bash
tmux new -s student-sl bash
```

Inside it:

```bash
eval "$(conda shell.bash hook)"
conda activate /home/cc/eecs116/fa26/class/eecs116-aaq/.conda/envs/myenv
cd /home/cc/eecs116/fa26/class/eecs116-aaq/Documents/project_3/viobench/viobench/solution/examples/python/1_generative_ssm
HD_SL_RUN="checkpoints/sl-parallel-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p checkpoints
set -o pipefail
python -u train_students.py \
  --profiles 1k 10k 100k 1m_ssm 1m 1m_hybrid 100m 100m_hybrid \
  --out "$HD_SL_RUN" --device cuda --stream-data --parallel-mode auto \
  --epochs 1500 --wandb-mode online --no-calibrate-architectures \
  --embedding-weight 0.1 --embedding-ema 0.99 --alignment-weight 0.1 \
  2>&1 | tee "$HD_SL_RUN.log"
```

Fresh random checkpoints; no pretrained VIO teacher. The scheduler overlaps
models when memory/measurements support it and can choose serial execution.
Collection is overlapped with training. Each model has separate weights,
optimizer and stopping state, but this multi-profile invocation shares curriculum
progression: qualified students pause until all active students qualify, with
the five-epoch sole-holdout timeout described below. The
existing memory safeguards can reduce batches after safe scheduling probes.
Exports remain metered against saved caps. W&B includes every loss component,
unaligned MTE and aligned ATE; the project is `hyperdrone-vio-fuel-aware`.
Detach with Ctrl-b then d; reconnect with `tmux attach -t student-sl`.
Checkpoints/WASMs are under `$HD_SL_RUN/PROFILE/` (WASMs in `native/`).
The sequential launcher below instead gives every model an independent curriculum.

## Run all eight, back to back

```bash
tmux new -A -s student-sl
eval "$(conda shell.bash hook)"
conda activate /home/cc/eecs116/fa26/class/eecs116-aaq/.conda/envs/myenv
cd /home/cc/eecs116/fa26/class/eecs116-aaq/Documents/project_3/viobench/viobench/solution/examples/python/1_generative_ssm
bash train_students_sequential.sh
```

The order is **1K, 10K, 100K, 1M SSM, 1M RNN, 1M Hybrid, 100M RNN, 100M Hybrid**,
ascending by measured deployed fuel/frame (approximately 5.2K, 7.1K, 84.2K, 535K, 694K,
798K, 47.2M, and 53.1M for the learned-frontends resource checks). These measurements order
the architectures; every newly trained export is checked against its own cap.
The smallest profile is now `1k`: a real two-filter 2D CNN on a 4×4 gray grid,
2D embedding and learned embedding forecaster, with a 6,000-fuel export cap. Its
169 parameters include a nine-correction state head, matching the output
convention of larger students. The name `1k` is retained for CLI compatibility;
it now measures 5,195 fuel/frame. Previous six-output/fixed-attitude `1k` and `500`
checkpoints remain loadable/exportable using their saved architecture settings.
The new `100k` profile uses an 8×6 gray grid, a 2→4-channel CNN, an 8D
embedding, a 32→8→8 forecaster and a 48-unit tanh RNN state predictor. Its 4,417
parameters measured 84,184/100,000 fuel/frame over a 1,200-frame WASM check.
Each model starts with random weights and a fresh optimizer in a separate Python
process, with its own curriculum, checkpoints, early stopping, log and W&B run.
The next starts when the previous finishes training and export. A failure stops
the launcher and retains completed runs. The launcher rejects warm-start flags
and existing output roots. It does not launch or delete any prior training run.

Defaults are CUDA, streamed collection, online W&B in
`hyperdrone-vio-fuel-aware`, automatic memory-aware scheduling, three recording
passes at most, and a **1,500-epoch safety ceiling per model**. Early stopping
can finish earlier. Override the ceiling, learning rate or plateau settings by
appending normal `train_students.py` options, for example `--epochs 1000`.
This ceiling does not force curriculum progression: an unstable phase can
remain blocked until the ceiling is reached, which is reported as `epoch_limit`.

The supervised entry point and launcher default to `supervised_procthor.json`:
**ProcTHOR Train200 only**, with scenes **0–191 for training** and **192–199 for
validation**. HSSD is excluded from both sets. Each validation scene contributes
one complete 20-second flight. The full-HSSD holdout was canceled; no persistent
125-scene validation cache is collected. An explicit `--collection-config` can
override this preset for a separate experiment. The selected collection settings
are saved in run metadata and W&B.

Outputs use a new `checkpoints/supervised-TIMESTAMP-PID/` root:

- `runs/PROFILE/PROFILE/student.pt`: independently selected best validation weights.
- `runs/PROFILE/PROFILE/stage_best.pt`: best weights and matching Adam state within the current phase.
- `runs/PROFILE.log`: terminal output for that model.
- `students/PROFILE`: link to the corresponding profile directory after it finishes successfully.
- `students/PROFILE/native/student.wasm`: exported model, checked against its fuel cap.

Choose a known fresh root with `--out-root checkpoints/MY_NEW_SUPERVISED_RUN`.
The same `evaluate_students.py --students ROOT/students ...` workflow evaluates
all completed exports. Use matching full-length episodes for every model and
baseline; the earlier two-scene six-second screening was not representative of
the website's aggregate scores.

## Curriculum order

Motion, IMU noise, and vision each have **10 difficulty levels**, numbered 0–9,
with normalized strengths `0, 1/9, 2/9, ..., 1` (approximately
`0, .111, .222, .333, .444, .556, .667, .778, .889, 1`).
At each of ten horizons **2, 4, 6, 8, 10, 12, 14, 16, 18, then 20 seconds**:

1. Train at base motion difficulty (0), waiting for a plateau.
2. Increase motion through all ten levels, plateauing at each level.
3. After plateauing at motion 1.0, extend the horizon and reset motion to 0.

Visual and IMU augmentation remain at 0 throughout those horizon cycles. At
20 seconds and motion 1.0, progress IMU noise through its ten levels,
then visual augmentation through its ten levels. Vision remains at level 0 until
IMU noise reaches 1.0. Each transition again requires a
plateau. Existing native recorded sensor noise remains present at level 0;
these settings control additional augmentation. Motion difficulty selects motion
complexity bins from the available recordings, rather than changing native
flight dynamics. Motion uses ten quantile bins. If a small recording batch has
an empty current bin, use its closest available lower bin rather than sampling
an unintroduced harder bin.

Training samples **75% from the current category and 25% from any earlier
category**, uniformly across earlier available categories, for each axis.
At level 0 there are no easier categories, so all samples use the base level.
There is no harder-level exploration. Validation probes use only the current
category, with fixed examples and fixed augmentation seeds; they never replay
earlier categories. Full-trajectory validation ATE remains unaugmented across
all eight held-out scenes.

Additional errors are random within each level's strength envelope. IMU white
noise standard deviations, bias limits, and random-walk coefficients each get
independent per-flight/per-channel factors uniformly drawn from **0.5–1.0**,
on top of the random Gaussian innovations and signed bias draws. For example,
at the maximum level, added accelerometer noise standard deviation ranges from
0.07–0.14 m/s² and gyro noise standard deviation from 0.0175–0.035 rad/s.
Blur blend is uniform from **0 to 0.5 × level strength**, rather than a fixed
blend. Brightness and contrast remain random in `1 ± 0.4 × level strength`.
Error parameters stay consistent within a flight; training seeds change each
epoch, while held-out probe seeds remain fixed. Level 0 adds no errors.

Models run sequentially and therefore never wait on another model's gate. If
`train_students.py` is called with multiple profiles, all selected profiles
normally qualify before the shared phase progresses; qualified profiles pause
while waiting. **Sole-holdout exception:** if exactly one active student remains
unqualified and every other active student stays frozen for five consecutive
completed epochs, advance the entire group one curriculum step anyway. The epoch
that first freezes the last peer does not count. Reused-recording epochs count
for this timeout, while ordinary plateau gates still require fresh recordings.
The same holdout and peer group must persist; regression/thawing, a holdout/group
change or an epoch gap resets the waiting streak. Single-model runs and cases
with two or more holdouts do not use this exception. No forced promotion occurs
after maximum difficulty. Stage-best model/optimizer/EMA restoration remains
unchanged for forced transitions. Waiting state survives stage recovery; older
checkpoints without this metadata start its counter at zero. W&B logs
`curriculum/sole_holdout_student`, `curriculum/sole_holdout_wait_epochs`,
`curriculum/forced_promotion`, and `curriculum/forced_student`;
`curriculum_metrics.jsonl` marks the reason as `sole_holdout_timeout`.
A regression revokes qualification and resumes training.

## Plateau qualification

Ordinary promotion requires a near-best plateau; the sole-holdout timeout is
the exception. For both the current-phase
held-out ground-truth state loss and complete unaugmented validation Sim(3) ATE:

- At least **six fresh recording checks** in the phase (`--stage-epochs 6`).
- No meaningful improvement for **three fresh checks** (`--plateau-patience 3`).
- Meaningful improvement means a cumulative reduction of more than **0.5%**
  relative to the last improvement reference (`--plateau-min-delta .005`).
- The last three scores must all be within **5% of the best observed phase score**,
  and their range must be at most 5% of that best (`--plateau-tolerance .05`).

This is an observed stable plateau near the best score, not a guarantee of a
global optimum. A lower validation loss is better; we cannot demand a score
below the unknown true optimum. Phase probes use up to **12 fixed windows**
(`--curriculum-probe-batches 12` is a sample budget in this staged mode), with
fixed augmentation seeds. Motion probes use the current motion-complexity bin.
Memory-driven batch reductions do not change probe examples. On short fixed
datasets fewer probe windows may be available.

Reused passes on a recording batch do not advance the counters. When a phase
changes, each student restores its best completed-epoch weights and matching
optimizer from the previous phase, then starts new plateau gates. That best is
selected by full-trajectory validation ATE. Validation and checkpoint selection
use all eight complete held-out ProcTHOR 20-second flights in the supervised preset.
At full difficulty, three fresh cycles are required before each model's normal
eight-cycle non-improvement early stopping becomes eligible. Data collection
continues in the background while the one active student trains; the coordinated
collector OOM / OptiX compilation retry remains enabled.

## Single-model runs and recovery

```bash
python -u train_students.py --profiles 1m_ssm --device cuda --stream-data \
  --out checkpoints/NEW_1M_SSM_SL --wandb-mode online
```

`--config HOST.json` supplies ordinary host defaults without a teacher config;
`--collection-config COLLECT.json` replaces the default ProcTHOR-only collection
preset and overrides only collection settings. The
sequential launcher uses fresh runs only. To recover an interrupted supervised
model, call the entry point directly with a new output directory:

```bash
python -u train_students.py --profiles 1m_ssm --device cuda --stream-data \
  --warm-start checkpoints/PRIOR_ROOT/runs/1m_ssm --warm-start-stage \
  --out checkpoints/RECOVERED_1M_SSM_SL --wandb-mode online
```

Supply any original custom host/collection settings again. When `--profiles` is
omitted, recovery selects the profiles saved in that source run, so adding the
100K tier does not make older runs require a nonexistent 100K checkpoint.
Recovery preserves
stage-best weights, matching optimizer state, curriculum order, per-model
plateau history, epoch/scene position and stopping counters. Recovery refuses
to mix distillation checkpoints or a different curriculum policy into these
supervised runs. Ten-level runs also reject recovery from the older three-level
policy; use a fresh run for the changed curriculum. The shared engine keeps the metadata filename
`distillation.json` for compatibility, but it explicitly records
`ground_truth_only: true`, zero teacher-loss weights and `teacher: null`.

Terminal output includes training loss, validation state loss, and held-out
Mean Sim(3)-aligned ATE. W&B uses `supervised/*`, per-model metrics, and
`curriculum/*`, including phase, levels, plateau wait and best probe scores.
The legacy `distill.py --teacher ...` path remains available for controlled
distillation comparisons and retains its original joint curriculum by default.


## Recovery from the earlier five-horizon schedule

The default schedule now contains ten horizons, from 2 through 20 seconds in
two-second increments. Existing tmux commands use it automatically for fresh
runs. On staged recovery from the earlier 2/4/8/12/20-second schedule, the
current horizon and difficulty are retained, along with plateau history, frozen
peers and any sole-holdout wait counter. The current horizon is remapped to its
new index; remaining intermediate horizons are inserted ahead. Previously
skipped shorter horizons are not replayed. The migration is recorded in metadata
and printed to the terminal. Custom incompatible schedules still fail validation.
An already-running Python process needs recovery/restart to load the new schedule.
