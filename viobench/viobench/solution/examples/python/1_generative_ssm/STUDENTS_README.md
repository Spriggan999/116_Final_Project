# Seven-student distillation

Use the CUDA Conda environment and run commands from `1_generative_ssm`.
Profiles are `500 10k 1m 100m 1m_ssm 1m_hybrid 100m_hybrid`.
The teacher remains frozen. New SSM students require official Mamba-2/CUDA for
training; their native WASM inference requires neither Python nor CUDA.
The 10K default samples a 6×3 grid. Legacy RNN checkpoints remain compatible.

The smallest profile is now **500 fuel/frame**, replacing the 100-fuel default.
It uses two grayscale pixels (left/right image centres), their temporal
differences and the final three accelerometer readings. Its 18 learned
coefficients predict acceleration, followed by fixed-20 Hz position/velocity
integration. It keeps attitude fixed and ignores gyro; this is a minimal visual
motion prior. Visual weights start at zero and learn through acceleration KD and
trajectory supervision. The focused 1,200-frame check measured **150/500 fuel**.
Use `--profiles 500` to train it alone. Old six-parameter 100-fuel checkpoint
files remain exportable with their saved settings, but need fresh training to
become the visual 500 model. Historical timing tables below predate this swap;
the 500 profile has not been runtime-benchmarked.

## Distill all seven

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
`checkpoints/20261113-192121-curriculum-tujmpo0u`, all seven students and W&B
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

Replace `RUN` with a completed teacher run, or use `RUN/post_sl`. The run root
contains independently selected checkpoints under each profile name and exports
under `PROFILE/native/student.wasm`. Each new architecture is fuel-calibrated
before training and rechecked after training. Initial and final resource checks
are not accuracy measurements.

The 500-epoch maximum is a ceiling. Horizon lengths are 401/801/1601/2401/4001
IMU ticks, six epochs at each introductory stage. Early stopping counts eight
non-improving **fresh recording batches** after reaching the final 20 s horizon.
Reused batches do not consume extra patience; fixed-data runs count epochs.
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

Teacher targets are generated once per shared minibatch. Background collection,
CPU loading, target prefetch and independent student updates overlap. Automatic
scheduling measures serial student updates, groups of one/two/three/four/all,
cost-balanced pairs and expensive-student isolation, then chooses the fastest
measured option with 15% estimated VRAM headroom. It repeats at each new horizon
and when early stopping changes the active student set. Teacher-target prefetch
remains enabled even if student updates are selected to run serially. Concurrent
streams do not imply sevenfold speedup or guaranteed 100% utilization.
The CUDA minibatch ceiling defaults to eight; the 3,208-token budget reduces
actual batches to 8/4/2/1/1 over the default horizon schedule. `--batch N`
overrides the ceiling. Sensor packet means, input projections and output heads
are batched; only the baseline RNN hidden recurrence remains sequential.
Official Triton launchers receive a host-side lock around their mutable
autotuner state, preventing concurrent first-use races. The lock releases after
kernel launch and does not synchronize GPU streams or replace official kernels.
Frozen hybrid-teacher targets use its existing associative physical integration
without enabling teacher training mode, dropout or BF16. Neural corrections are
unchanged; FP32 integration reduction order causes small trajectory differences.
The real 20-second parity probe measured under 1 mm maximum position difference.
Use `--no-parallel-teacher-physics` for the original serial integration order.
`scheduling.json` records the decision; `performance.jsonl` includes validation
and collection waits. `--parallel-mode streams --max-concurrent 7` forces seven
streams; `--parallel-mode serial` disables GPU overlap. Probe updates are rolled
back; production OOMs fail rather than silently repeat partial optimizer steps.

W&B defaults online in `hyperdrone-vio-fuel-aware`, job type
`seven_student_distillation`. All seven profiles are logged in one run.
The terminal prints that run's URL. Live `progress/*` metrics show the current
epoch/batch, percentage, window length, active students and batch losses about
every 10 seconds during training (plus the first/last batch). Each epoch logs
all active students' loss, full-trajectory ATE, best ATE, learning rate and
early-stopping counters together, alongside throughput, GPU memory, concurrency
and fresh-data wait. Batch and epoch charts have separate step axes.
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
contains per-profile estimates and all-seven projections. On the sampled
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

## Evaluate seven students plus teacher

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
