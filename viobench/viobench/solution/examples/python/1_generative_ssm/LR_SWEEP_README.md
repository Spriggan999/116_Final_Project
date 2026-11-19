# Direct learning-rate experiments

`lr_sweep.py` analyzes the stopped run's local metrics and reads its W&B summary,
then compares fresh initializations on a fixed, scene-disjoint ProcTHOR dataset.
No weights from the stopped students or old teacher are loaded. The old teacher
JSON provides architecture settings only.

The initial experiment uses rates 1e-5, 3e-5, 1e-4, 3e-4 and 1e-3:

- Teacher embedding pretraining: five fresh 256-update trials, same sampled pairs.
- Eight students plus teacher state SL: five fresh 512-update trials per model
  at a 2-second training horizon.
- Top two rates per model: fresh 512-update trials at 10 and 20 seconds.
- Validation always covers all eight complete 20-second validation flights.

`lr_sweep_confirm.py` then expands a winning boundary (down to 1e-6/3e-6 or up to
3e-3/1e-2), retains the original LR as a mandatory baseline, and compares the
finalists at 2/10/20 seconds using a second fresh initialization and 1,024 updates.
Missing first-seed longer-horizon trials for newly introduced candidates are
also run. Every within-seed/horizon comparison uses identical initialization,
minibatch order and update budget. SHA-256 checks enforce identical initial
weights. The teacher state trials share the frozen encoder trained from fresh
initialization in the first experiment; the second seed varies state weights,
not this encoder. Teacher PPO is not part of these SL experiments.

At 2 seconds, student batches contain eight windows and teacher-state batches
contain four; both use one window at 10/20 seconds. Teacher embedding uses eight
pairs per update. The older teacher used embedding batch 64 and configured state
batch 8, so these pilot LRs should not be treated as batch-independent optima.
The generated fresh-teacher configuration uses the pilot batch sizes.

The experiments use fixed recordings, no augmentation or curriculum advancement,
and the existing objectives: state + 0.1 JEPA + 0.1 alignment for students;
standard embedding training and state SL for the teacher. Original optimizer,
weight decay and clipping choices are retained. Each model has its own LR.

Camera preprocessing is cached before trials. Sparse gray caches match the
rounded ABI images; tests compare both predictions and gradients with ordinary
student preprocessing. Teacher features are precomputed using the same causal
embedding generator. Original 200 Hz IMU and physical state integration remain
active. Complete validation uses camera lever arms and records aligned ATE,
unaligned MTE, unaligned ATE RMSE and state loss.

## Interpretation

Rates are ranked by the mean of the last two validation-state-loss checkpoints,
not the luckiest single minimum. Confirmation averages normalized scores across
horizons and state seeds; the baseline is present in every group. Aligned ATE and
unaligned MTE are reported alongside this state-loss ranking. The per-horizon
winner is also saved. Do not interpret a normalized-score reduction as the same
percentage improvement in website ATE or area.

These are provisional, equal-update pilots on eight training scenes (16 flights)
and eight validation scenes, not proof of a global optimum or full convergence
on 192 training scenes. The validation scenes are used for hyperparameter
selection and are not an untouched final test set. Selected checkpoints are
pilot-trained models, not full curriculum training results. The lowest-state-
loss checkpoint within a trial is retained separately from the tail-score LR
ranking. `best_step=0` indicates a trial did not improve on its initialization.

## Artifacts

The sweep directory contains protocol/dataset metadata, existing-run analysis,
W&B URLs, `history.jsonl`, `trials.json`, and recommendations. Confirmation adds
expanded candidates, baseline comparisons, per-horizon winners and a
`student_lrs.json` mapping. Selected student weights/settings and teacher
encoder/state weights are stored in `selected/`. Trial checkpoint files and raw
fixed recordings live in the specified scratch directory; move those before a
temporary-storage cleanup if you want to preserve every candidate.

Example (after preparing the fixed train/validation recording directories):

```bash
python -u lr_sweep.py --train /path/train --validation /path/validation \
  --out checkpoints/lr-study --scratch /tmp/lr-study-trials
python -u lr_sweep_confirm.py --screen checkpoints/lr-study \
  --data-root /path/to/directory/containing/train-and-validation
```

W&B stays online by default in `hyperdrone-vio-fuel-aware`. The confirmation
also logs raw/weighted training components and pre-clipping gradient norms.

To apply the selected per-student rates to a **fresh** normal training run:

```bash
python -u train_students.py --device cuda --stream-data --parallel-mode auto \
  --profiles 1k 10k 100k 1m_ssm 1m 1m_hybrid 100m 100m_hybrid \
  --lr-config checkpoints/lr-study/confirmation/student_lrs.json \
  --out checkpoints/NEW_FRESH_RUN --epochs 1500
```

Unlisted profiles fall back to `--lr` (default 3e-4). Stage recovery restores the
saved optimizer LR, consistent with its existing behavior; use fresh runs for
controlled LR changes. The teacher state LR is applied separately through its
`LR_STUDENT` config; teacher embedding training uses `LR_FRAME`.

## Window-target consistency audit

The study also records `window_target_audit.json`. Existing `StateWindows`
resets integrated velocity to zero and targets `R0.T @ (v(t)-v(0))`, while its
position target is `R0.T @ (p(t)-p(0))`. For a mid-flight window with nonzero
initial velocity, these are not a single consistent kinematic position/velocity
state. In the sweep's selected 2-second training windows, median initial speed
was about 0.21 m/s and maximum about 0.72 m/s; all complete flight resets began
at zero speed. No target equation was changed during the experiment. Treat
short-horizon LR results as conditional on this existing behavior, and investigate
it before interpreting a low LR as a cure for the stalled curriculum.

An optional independent comparison uses additional ProcTHOR scenes 160–167,
never used for this study's training or LR selection. It compares baseline and
selected checkpoints from matched seed-29 20-second trials. These selected models
remain pilots, not final full-curriculum submissions.

## Complete curriculum-window and student embedding-LR sweeps

`lr_sweep_windows.py` covers every 2-second interval from 2 through 20 seconds
for all eight students and teacher state SL. It reuses completed 2/10/20-second
state trials from the parent study. Missing windows screen the five original
rates plus prior per-window winners outside that grid, then confirm the top two
and original baseline with the second initialization. Update budgets remain
512 for seed 17 and 1,024 for seed 29. Batch ceilings match the original study's
token budgets. Fixed validation always uses full 20-second flights.

The student **embedding group** consists of the online visual encoder and causal
embedding predictor. The old state sweeps trained these with the same LR as the
state model. The new embedding sweeps hold the chosen state LR fixed and vary
this group's LR while retaining the full state + 0.1 JEPA + 0.1 alignment
objective. They run at **all ten windows**, since no separate student embedding
LR was tested before. Each uses fresh complete-model weights; this does not
continue a state-trained checkpoint. Five original LR candidates plus the
selected state LR are screened, then the top two by state loss, top one by JEPA
loss and tied-LR baseline are confirmed with the second seed. Both state-loss
and JEPA-loss winners are reported. These are conditional group-LR results,
not an exhaustive Cartesian search over both rates.

The teacher embedding stage is different: it trains on short camera triplets
with at most an 80 ms prediction gap, not state trajectory windows. Its completed
fresh-checkpoint sweep is reused once and linked to every window; repeating
identical teacher frame-pair trials ten times would not test a new setting.

Example:

```bash
python -u lr_sweep_windows.py \
  --screen checkpoints/lr-sweep-20261115-200326 \
  --data-root /tmp/hyperdrone-lr-sweep-20261115-195524 \
  --out checkpoints/NEW_WINDOW_STUDY \
  --scratch /tmp/NEW_WINDOW_STUDY --workers 3
```

Independent rate trials run in three spawned GPU processes by default. Each has
its own RNG, fresh model and optimizer; only the parent writes aggregate results
and W&B history. Workers load the small preprocessed caches rather than three
copies of the raw image dataset. Use `--workers 1` for serial execution. The
optimizer helper supports per-module parameter groups with fused CUDA updates.

Run that command again with **the same arguments** to recover after an
interruption. Completed trials are skipped; an incomplete trial restarts from
its matched fresh initialization. Cached inputs are reused only when the source
fingerprint matches. Changed source recordings, teacher encoder, architecture
or protocol require a new output directory. A file lock prevents two writers
from running the same experiment concurrently.

The output directory contains `progress.json`, `trials.json`, `all_trials.json`,
`window_results.json`, and an incrementally updated `comparison.md` with three
tables: state rates, embedding rates ranked by state loss, and embedding rates
ranked by JEPA loss. `wandb.json` contains the online run URL. W&B logs trial
rates, windows, seeds, state/embedding loss components, aligned ATE, unaligned
MTE, gradient norms and the final result tables. Full training remains stopped;
these runs do not change the production LR defaults or target equations.

After the main sweep completes, repeat its command with
`--embedding-boundaries` to check up to two neighboring rates when an initial
embedding screen winner is at a search edge. This supplemental pass keeps state
LRs fixed, reuses all completed trials, and stays within 1e-7–1e-2. Its exact
protocol is saved in `embedding_boundary_protocol.json`; results flag winners
still at an edge. This is one bounded extension, not an unlimited search.

Run `python summarize_window_sweep.py checkpoints/NEW_WINDOW_STUDY` after all
passes finish to verify all 90 state/model-window and 80 student embedding/window
results, export `window_rates.csv`, and optionally draw an LR heatmap (requires
matplotlib). Verification also checks matched initial weights and that reported
winners have complete comparisons across both seeds.
