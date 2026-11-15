# Budget-student distillation

For the new default workflow of fresh **ground-truth-only supervised training**,
run `bash train_students_sequential.sh` and see
[the supervised student guide](SUPERVISED_STUDENTS_README.md). Each student trains
independently, back to back, with a staged plateau-only curriculum. This document
retains the teacher-distillation workflow for comparisons and older run recovery.

Use the CUDA Conda environment and run commands from `1_generative_ssm`.
Default profiles are `1k 10k 100k 1m 100m 1m_ssm 1m_hybrid 100m_hybrid`.
The teacher remains frozen. New SSM students require official Mamba-2/CUDA for
training; their native WASM inference requires neither Python nor CUDA.
The 10K default samples a 6×3 grid. Legacy RNN checkpoints remain compatible.
New defaults add learned visual encoders and causal embedding forecasters to
every profile; see [the architecture comparison](STUDENT_ARCHITECTURES_README.md).
Fresh runs also use a default 0.1-weight JEPA-style auxiliary loss with EMA
targets. Set `--embedding-weight 0` to disable it. See
[all loss functions](LOSSES_README.md) for the complete objectives and defaults.

The smallest default profile retains the name `1k`, with a **6,000 fuel/frame**
export cap. It uses a real two-filter 2×2 CNN on a 4×4 gray grid, a learned 2D
embedding, a 14→2→2 causal forecaster and a 12→9 linear state head. Its
**169 parameters** produce the same gyro/body-acceleration/initial-frame-
acceleration corrections as larger students. The 20 Hz packet-mean integrator
emits changing orientation, position and velocities. The resource check measured
**5,195/6,000 fuel**; it is no longer a literal 1K-fuel submission. Older
six-output/fixed-attitude 1K, 26-parameter scalar
500 and 18-parameter raw-pixel 500 models remain compatible with their original
caps and layouts. Use `--profiles 1k` to train the new CNN tier alone.
Old six-parameter 100-fuel checkpoint
files remain exportable with their saved settings, but need fresh training to
become the current 1K CNN model. Historical timing tables below predate the
learned frontends and do not project their training times.

## Distill all eight

```bash
python distill.py --teacher checkpoints/RUN --device cuda \
  --stream-data --max-data-mib 8192 --parallel-mode auto
```

For a persistent tmux run on the lab machine:

```bash
tmux new -s student-distill
```

Inside tmux:

```bash
eval "$(conda shell.bash hook)"
conda activate /home/cc/eecs116/fa26/class/eecs116-aaq/.conda/envs/myenv
cd /home/cc/eecs116/fa26/class/eecs116-aaq/Documents/project_3/viobench/viobench/solution/examples/python/1_generative_ssm
bash distill_students.sh
```

The launcher defaults to the completed teacher
`checkpoints/20261113-192121-curriculum-tujmpo0u`, all eight students and W&B
online. It prints a fresh checkpoint directory and adjacent `.log` file; both
stdout and stderr are recorded. Python creates the run directory itself.
It returns the training exit code, including errors, and refuses existing
outputs. Override with `bash distill_students.sh --teacher checkpoints/RUN`;
other `distill.py` flags can follow. `HD_DISTILL_PYTHON` optionally selects a
Python executable; activate the environment so its renderer toolchain is on PATH.

Detach with **Ctrl+B**, then **D**. Reattach using
`tmux attach -t student-distill`. **Ctrl+C** inside the session stops the run;
best completed-epoch checkpoints remain saved. This starts fresh student
training, not an optimizer-state resume of an interrupted student run.

To warm-start a future interrupted run with six completed epochs and saved best
weights still present (replace `PREVIOUS_RUN` with its directory):

```bash
bash distill_students.sh \
  --warm-start checkpoints/PREVIOUS_RUN \
  --start-epoch 6
```

This creates a new output directory and online W&B run, loads each student's
saved best weights and fuel-verified architecture, and begins at curriculum
epoch 7. Adam optimizer state and early-stopping counters start fresh; this is
not an exact continuation of the interrupted optimizer state. The original run
is preserved, and teacher checkpoint hashes must match. `--epochs` remains the
total epoch ceiling, including `--start-epoch`.
The partial student weights from the failed `distill-20261114-191710-319824` run
have been removed; its logs/metrics remain for diagnosis. Use the fresh launcher
above for the next run.

To recover the continued run that completed 15 epochs before an OptiX module
compilation failure, keep its checkpoints and run:

```bash
bash distill_students.sh \
  --warm-start checkpoints/distill-20261114-214629-357261 \
  --warm-start-stage
```

`--warm-start-stage` loads each student's rolling `stage_best.pt` weights **and
matching optimizer state**, preserving adaptation to the last stage rather than
loading an all-run best from an earlier stage. It also preserves shared levels
and stage index, every student's gate ages/references/improvement streaks,
readiness and pause/plateau flags, final-difficulty exposure, early-stopping
counters and adaptation grace. Already stopped students stay stopped. The
completed epoch is inferred, so this run continues at epoch 16 in a new
output/W&B run. An explicit `--start-epoch` must agree with saved progress.
The next scene-collection index is retained instead of restarting at scene zero.
Because students choose different best epochs,
this is a stage-best continuation, not an exact replay of their last updates.
Normal validation regression checks can still revoke a restored qualification.
Without `--warm-start-stage`, ordinary warm starts retain fresh optimizers and
readiness checks.

Replace `RUN` with a completed teacher run, or use `RUN/post_sl`. The run root
contains independently selected checkpoints under each profile name and exports
under `PROFILE/native/student.wasm`. Each new architecture is fuel-calibrated
before training and rechecked after training. Initial and final resource checks
are not accuracy measurements.

The 500-epoch maximum is a ceiling. The full difficulty curriculum is enabled
by default, with independent motion, appearance, IMU and window progression.
Horizon lengths are 401/801/1201/1601/2001/2401/2801/3201/3601/4001 IMU ticks
(ten stages, 2–20 seconds in two-second steps). Difficulty can advance after
at least three fresh batches and a sustained 2% relative improvement (two checks),
or after six fresh checks at a healthy plateau within 10% of the stage's best
probe score. Substantial regression holds that axis. `--curriculum-min-batches`
sets minimum exposure; `--stage-epochs` sets the plateau age in full mode.
Motion sampling mixes current/easier/harder windows; appearance adds consistent
brightness, contrast and blur; IMU augmentation adds per-flight white noise,
bias and random walk. Appearance transforms run in bounded GPU chunks.

The teacher and all students use the **same augmented observations** for each
shared minibatch; teacher correction labels are computed from that noisy IMU.
Ground-truth trajectories stay unchanged. This preserves one teacher pass per
shared training batch. Each axis uses fixed held-out probes and individual
relative-error gates. **Every active student must qualify before that axis
advances**; a group statistic cannot advance a slower student's difficulty.
All selected active students form one curriculum group, independently of CUDA
execution groups. Finished axes stop consuming probe inference. Probes run once per
fresh batch and never contribute to the main validation ATE.

A student pauses optimizer updates once it is ready on every unfinished axis
and its complete-flight validation ATE reaches a healthy plateau. Its weights
and optimizer state stay fixed while peers continue training; it still receives
ordinary validation and fixed-axis probe evaluations. A regression revokes
readiness and resumes it. Any shared difficulty change resumes all students and
resets their readiness gates so they can adapt before qualifying again.
At full curriculum completion every student trains again under the existing
independent early-stopping policy. The slowest unready model can hold an axis;
the 500-epoch ceiling remains enforced.

Before the first training update at a new difficulty or horizon, each student
restores its **best completed validation epoch from the stage it just finished**,
including the matching Adam optimizer state (step counters, moments and parameter
groups). A stage is the joint motion/visual/IMU levels and horizon. Selection uses
the lowest complete-flight Mean Sim(3)-aligned ATE within that stage, independently
for each student. Selection resets at every transition, so a best checkpoint from
an older, easier stage cannot override a checkpoint learned in the current one.

`PROFILE/stage_best.pt` is an atomic rolling model/optimizer snapshot with source
epoch, stage and ATE metadata. There is one slot per profile, overwritten on
improvement; snapshots do not accumulate per epoch or stage. CUDA copies finish
before worker streams resume. `stage_restores.jsonl`, terminal messages and W&B
report the source epochs and ATEs restored at transitions. Per-epoch metrics also
include `curriculum_stage`, `stage_best_epoch` and `stage_best_validation_mean_ate_m`.
This also applies to window changes with `--no-difficulty-curriculum`.

The all-run `PROFILE/student.pt` remains the separate best checkpoint used for
final WASM export. Explicit `--warm-start` still loads best weights with fresh
optimizers and readiness checks unless `--warm-start-stage` is supplied.
Stage recovery retains existing per-student progression.

Terminal summaries show `train_loss=paused` when no updates ran and
`status=paused_waiting_for_group` when the next epoch will wait. JSON stores
`train_loss: null`, zero `training_samples`, `training_paused` and
`paused_waiting_for_curriculum`; W&B reports each student's readiness, pause
state and validation without inventing a zero training loss. CUDA scheduling
retunes when the training set changes, including when paused students resume.

Early stopping counts eight non-improving **fresh recording batches** after all
axes and the final 20 s horizon are complete and three fresh final-difficulty
batches have been exercised. Promotions reset patience and allow adaptation.
Reused batches do not consume extra patience; fixed-data runs count epochs.
`--no-difficulty-curriculum` restores the previous window-only schedule with six
epochs per introductory stage. `--window N` fixes the horizon while retaining
the other axes unless the full curriculum is explicitly disabled.
`curriculum_metrics.jsonl`, `curriculum_state.json`, terminal summaries and W&B
`curriculum/*` metrics expose levels, probe losses, promotion reasons and stopping
eligibility. `recovery_state.json` atomically records all curriculum and stopping
metadata at every completed epoch. Stage recovery reloads it without resetting
individual progression. Older runs, including the failed nine-epoch run, migrate
their saved `curriculum_state.json` gates and per-student metrics; adaptation
grace is reconstructed from promotion and recording-batch history. Missing legacy
readiness information is reported rather than invented. Ordinary weight-only
warm starts re-establish readiness and final exposure. Legacy warm starts
without curriculum state initialize new axes at easy difficulty and
select the initial horizon from `--start-epoch` using the old six-epoch schedule.
Checkpoint selection uses full validation recordings, not independently reset
windows. `--min-delta` defaults to 0.0001 m ATE. Short recordings cannot be
silently substituted for the final horizon.
The rolling recording ring defaults to 8 GiB in `/tmp/hyperdrone-distill-USER`,
with separate current/next/validation slots. A detailed 20-second HSSD flight
can exceed 250 MiB; the previous 3 GiB ring could truncate validation coverage.
Set `--stream-dir` and `--max-data-mib` to override. Only owned run subdirectories
are cleaned; checkpoints stay in the persistent `--out` directory.
Scene assets use an explicit `CONTA_CACHE` if set; otherwise they go to
`/var/tmp/hyperdrone-USER/scene-assets`. The old home cache is searched read-only.
This keeps downloads for the larger scene pool away from the home cache quota.
The recording cap describes host storage, separately from the WASM memory cap.

Large OptiX scenes can temporarily need more GPU memory than a minibatch measured
on a smaller scene. A collector OOM or the specific `optixModuleCreate` /
`Module compilation failed` signature triggers one coordinated recovery:
new teacher/student GPU launches pause, outstanding work synchronizes, unused
PyTorch cache is released, and the failed unpublished recording batch is retried
with one flight at a time. Model/optimizer state and current/validation recordings
are retained. Normal parallel collection/training resumes after the retry.
An OptiX compilation failure is not assumed to be an OOM; the retry tests whether
a fresh process with less GPU contention resolves it. Unrelated errors and
persistent failures still propagate, with a bounded native diagnostic tail and
retained failure signatures included in the exception. Terminal/W&B console
logs show recovery; `collector_oom_retries` and `collector_oom_recoveries` are
included in epoch performance and W&B metrics, alongside
`collector_optix_compile_retries` / `collector_optix_compile_recoveries`.
Other SL/RL callers without the
coordinator retain ordinary failure propagation.

Teacher targets are generated once per shared minibatch. Background collection,
CPU loading, target prefetch and independent student updates overlap. Automatic
scheduling measures serial student updates, groups of one/two/three/four/all,
cost-balanced pairs and expensive-student isolation, then chooses the fastest
measured option with 15% estimated VRAM headroom. It repeats at each new horizon
and when early stopping changes the active student set. If none of the plans
fits, it halves the minibatch and retries before any production update. At batch
size one it can disable teacher-target prefetch while retaining student streams.
`--no-target-prefetch` disables that overlap explicitly. PyTorch allocator cache
is excluded from the estimate of external GPU allocations, avoiding double
counting of cached blocks. The 15% headroom requirement remains enforced.
Teacher-target prefetch normally remains enabled even if student updates are
selected to run serially. Concurrent
streams do not imply sevenfold speedup or guaranteed 100% utilization.
The CUDA minibatch ceiling defaults to eight; the 3,208-token budget reduces
actual batches to 8/4/2/2/1/1/1/1/1/1 over the ten-stage default horizon schedule. `--batch N`
overrides the ceiling. This limit is applied by the distiller itself, independently
of the teacher curriculum toggle; previously that toggle incorrectly disabled
the token limit during distillation. Sensor packet means, input projections and output heads
are batched; only the baseline RNN hidden recurrence remains sequential.
Official Triton launchers receive a host-side lock around their mutable
autotuner state, preventing concurrent first-use races. The lock releases after
kernel launch and does not synchronize GPU streams or replace official kernels.
Frozen hybrid-teacher targets use its existing associative physical integration
without enabling teacher training mode, dropout or BF16. Neural corrections are
unchanged; FP32 integration reduction order causes small trajectory differences.
The real 20-second parity probe measured under 1 mm maximum position difference.
Use `--no-parallel-teacher-physics` for the original serial integration order.
`scheduling.json` records the decision and actual batch/prefetch settings;
`memory_recovery.jsonl` records failed probes and fallback actions, also reported
in the W&B summary. `performance.jsonl` includes validation and collection waits
alongside actual batch and prefetch settings. `--parallel-mode streams --max-concurrent 7` forces seven
streams; `--parallel-mode serial` disables GPU overlap. Probe updates are rolled
back; production OOMs fail rather than silently repeat partial optimizer steps.

W&B defaults online in `hyperdrone-vio-fuel-aware`, job type
`student_distillation`. All selected profiles are logged in one run.
The terminal prints that run's URL. Live `progress/*` metrics show the current
epoch/batch, percentage, window length, active students and batch losses about
every 10 seconds during training (plus the first/last batch). Each epoch logs
all active students' loss, full-trajectory ATE, best ATE, learning rate and
early-stopping counters together, alongside throughput, GPU memory, concurrency
and fresh-data wait. Batch and epoch charts have separate step axes.
Each student's terminal epoch summary prints `train_loss`, `val_loss` and
`Mean Sim(3)-aligned ATE` in metres. Training loss combines the distillation and
ground-truth objectives; validation loss is the ground-truth state loss. ATE
uses VIOBench's official trajectory scorer, averaged equally over complete
held-out flights; these training-validation episodes differ from website episodes.
The run status shows collection/loading, training, validation and export phases.
Per-student summaries retain checkpoint paths, stop reasons and verified export
fuel. W&B stays open until exports finish and marks failures/interruption
instead of recording an unsuccessful run as completed. Use `--wandb-mode
offline` or `disabled` only when intentionally opting out of online updates.

## Measure runtimes before starting a long run

```bash
python benchmark_students.py --teacher checkpoints/RUN \
  --recording /path/to/training/rollout_s000_e000.npz \
  --validation-recording /path/to/heldout/rollout_s313_e000.npz \
  --out /tmp/seven-student-pilot --training-recordings 8 \
  --validation-recordings 12
```

Recordings must support the requested horizons. The default pilot bounds timing
work to approximately 20 minutes between stages; one-time compilation, resource
probes and final validation are separate. It performs only a small number of
updates per model, logs no W&B run and launches no collector or full training.
Its native exports contain **initial probe weights**, while `pilot_student.pt`
contains **provisional pilot weights**. Neither represents converged students.

`timing.json` records per-student solo times, shared serial/grouped/stream
throughput, teacher cost, validation time, GPU memory, resource checks and
conditional 100/300/500-epoch projections. The earliest stale-stop forecasts
allow eight fresh batches, or 24 epochs with three-pass recording reuse.
They are not predictions of when learning will converge. The joint forecast
assumes all students remain active; early stopping usually reduces that cost.
One recording's window count may not represent shorter collision-truncated
flights. Set `--collection-seconds-per-batch` from a measured collector run to
include a conservative collection allowance. Default zero explicitly excludes
that allowance. Do not reuse old four-student synthetic throughput as a new ETA.
For an overlap-aware estimate, pass `--collection-stats PATH/collection_stats.json`
from a representative collector run. The forecast amortizes generation across
at most three passes, while patience still counts fresh batches. Initial
validation collection, dataset decoding and renderer/training GPU contention
remain additional costs. Faster models may wait for fresh recordings after
three passes; increasing reuse merely to fill the GPU can worsen overfitting.

The [measured seven-student timing table](verification/seven_student_timing.md)
contains historical window-only per-profile estimates and all-seven projections.
It predates the full adaptive curriculum, its per-axis probes and the observed
64–105 second epochs in the interrupted run; it is not an ETA for current runs.
On the sampled
detailed scene, the collection-aware projections are about **1.24 / 3.45 / 5.66
hours** at 100 / 300 / 500 epochs, assuming all remain active. An immediate
final-stage plateau would stop after roughly **40 minutes plus setup**, but
actual convergence cannot be predicted from a timing pilot. The compute-only
budgets are shorter: about 24 / 29 / 34 minutes, excluding CPU loading and
validation preprocessing. These are distinct estimates.
See [raw timing measurements](verification/seven_student_timing.json) and
[resource/parity verification](verification/seven_student_review.json).

The full 500-epoch run has not been launched. The implementation was checked
with bounded one/two-epoch pilots and a 20-frame eight-model evaluation smoke;
those checkpoints and scores are not a final model comparison.

Reuse the pilot's chosen layouts in the real run:

```bash
python distill.py --teacher checkpoints/RUN --stream-data --device cuda \
  --architecture-config /tmp/seven-student-pilot/architectures.json
```

## Evaluate students plus teacher

```bash
python evaluate_students.py --students checkpoints/DISTILL_RUN \
  --teacher checkpoints/RUN/teacher-ppo.wasm --out reports/eight-models --jobs 2
```

This runs native/WASM inference using VIOBench's evaluator, with identical public
episodes for all eight. Pass `--episodes HASH_OR_PATH ...` for another identical
selection and `--limits limits.json` to match the server's saved evaluation limits.
Each student keeps its stricter tier cap. Two bounded CPU processes can evaluate
models concurrently. Reports retain completion counts and official failed-frame
fallback scoring. Use `--reports-only` to recompute comparisons from saved reports.

Outputs include `comparison.json`, `comparison.md`, and per-model reports/logs.
If optional `matplotlib` is installed, `pareto.png` shows the log–log frontier
and claimed area; otherwise tables still work. `--no-plot` disables it explicitly.
Lower equal-per-episode Mean Sim(3) ATE and lower measured maximum fuel/frame
are better. Model caps and parameter counts are not the plot coordinates.

The eight website baseline values supplied on 2026-11-14 are preserved in
`website_baselines.json`. Local comparisons against that snapshot are explicitly
**approximate** until episode hashes and limits match. The website currently
known from prior results includes private episodes; public-only local scores
cannot reproduce those scores exactly.

For an exact website comparison, save the authenticated website's
`GET /comparison` response to a local file, then run:

```bash
python evaluate_students.py --server-comparison comparison-from-website.json \
  --out reports/website
```

This reads the server's already evaluated own models and shared baselines; it
does not upload anything or need your token. The exact server scoring function
uses log–log dominated area, floors of 1 fuel and 0.001 m, and reference bounds
of `1.2 × step_fuel` and `1.2 × worst eligible baseline ATE`. For the supplied
snapshot and 20G step cap these are 24G fuel and approximately 3.234292 m.

The report ranks standalone improvement over baselines and shows each model's
unique contribution—the area lost when removing it from the full portfolio.
Standalone gains overlap and cannot be summed. A model can add no area despite
being accurate when a cheaper baseline already performs better. No winner is
claimed before trained models have comparable full-episode results.

Use the website's upload form after local validation; check its current quota
before uploading. Prioritize the artifact with the highest predicted incremental
area, then complementary frontier points. Runtime estimates, synthetic fuel
checks and provisional pilot checkpoints do not establish website accuracy.
