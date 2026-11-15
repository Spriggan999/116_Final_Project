# Training objectives and validation metrics

## Current eight students: supervised state plus JEPA-style embeddings

The default `train_students.py` objective is

```
L_total = L_state(prediction, ground_truth) + 0.1 * L_embedding + 0.1 * L_alignment
L_state = 3 * L_position + L_velocity + L_rotation + 0.1 * L_angular_velocity
```

Position, velocity and angular velocity use mean Smooth L1 with beta 0.2.
For a scalar residual r, `H_beta(r) = 0.5*r*r/beta` when `abs(r)<beta`,
otherwise `abs(r)-0.5*beta`. Each tensor loss averages all entries.
Rotation uses mean squared geodesic angle:

```
angle = acos(clamp((trace(R_truth.T @ R_prediction)-1)/2,
                   -1+1e-5, 1-1e-5))
L_rotation = mean(angle**2)
```

The clamp keeps gradients finite; even identical rotations have a small positive
numerical loss floor. Units are metres, m/s, radians and rad/s respectively.
There is no direct ground-truth label/loss on the nine correction values:
state supervision backpropagates through IMU integration into their heads.
The encoder and embedding predictor also receive these state gradients.

### Embedding prediction and target encoder

Let `z_t = E_online(image_t)` and `y_t = stop_gradient(E_target(image_t))`.
The forecaster receives previous online embeddings, IMU and timing and predicts
`z_hat_t`. It never receives the current target image/embedding as input.
The current image is used separately by the encoder and the state predictor.
The first capture has no prediction history and is excluded from prediction and
content losses. The two previous embeddings can coincide during warmup.

```
L_prediction = mean(H_0.1(z_hat_t - y_t))       # t >= 1
L_content    = mean(H_0.1(z_t - y_t))           # t >= 1

X = all online embeddings across batch and time    # N x E, includes first capture
C = X - mean(X, axis=0)
std_j = sqrt(mean(C[:,j]**2) + 1e-4)
L_variance = mean_j(max(0, 0.1 - std_j))
Cov = C.T @ C / max(1, N-1)
L_covariance = sum_{i != j}(Cov[i,j]**2) / E

L_embedding = L_prediction + L_content + L_variance + 0.01 * L_covariance
```

Prediction matches a target embedding; content keeps the online encoder close to
its moving target; variance discourages constant features; covariance discourages
redundant feature dimensions. These penalties help resist collapse but do not
guarantee useful features or improved VIO accuracy.

After every successful optimizer update:

```
target_encoder = 0.99 * target_encoder + 0.01 * online_encoder
```

Targets get no gradients. EMA weights are persistent training buffers, excluded
from optimizers and native/WASM weight blobs. Deployment parameter counts, fuel,
inputs and outputs are unchanged. They survive ordinary recovery, stage-best
restoration and reversible scheduling probes. Older checkpoints without targets
initialize them from their saved online encoder; a new JEPA loss starts then.
No pretrained VIO teacher is required.

CLI controls (also accepted by `train_students_sequential.sh`):

```bash
bash train_students_sequential.sh --embedding-weight 0.1 --embedding-ema 0.99
# Disable the auxiliary objective:
bash train_students_sequential.sh --embedding-weight 0
```

The first command equals the new default. Legacy architectures without a learned
encoder/forecaster cannot use this auxiliary loss and retain their old objective.
W&B and `PROFILE/metrics.jsonl` record total training loss, `train_state_loss`,
`train_embedding_loss`, and the prediction/content/variance/covariance components.
Equivalent `validation_embedding_*` diagnostics are recorded without EMA updates.
The terminal prints total training loss, state validation loss, JEPA training loss
and validation mean Sim(3)-aligned ATE.

**Curriculum gates, checkpoint selection and early stopping still use state
validation/probes and complete-trajectory Sim(3) ATE.** Auxiliary validation loss
is diagnostic; it does not change these criteria. Total training loss and state
validation loss therefore have different definitions.

## Optional budget-student distillation (`distill.py`)

When a pretrained teacher is explicitly selected instead of supervised mode:

```
L_total = 1.0 * L_correction_KD
        + 0.5 * L_state(student, frozen_teacher)
        + 1.0 * L_state(student, ground_truth)
        + 0.1 * L_embedding + 0.1 * L_alignment
L_correction_KD = mean(H_0.05(student_corrections - teacher_corrections))
```

Teacher corrections are averaged over the corresponding IMU interval. CLI
`--kd-weight`, `--teacher-state-weight`, `--ground-truth-weight` and
`--embedding-weight` set the four coefficients. The older affine tiny models
instead match acceleration inferred from teacher velocity differences using the
same Smooth L1 beta 0.05. Supervised mode forces the first two weights to zero.

## Teacher embedding training (`embedding_training.py`)

The teacher already uses the same prediction/content/variance/covariance formula,
with beta 0.1, standard-deviation target 0.1, variance weight 1, covariance weight
0.01 and EMA decay 0.99 by default. Its embedding training phase minimizes
`L_embedding` directly. The subsequent teacher state SL phase minimizes `L_state`.
Student JEPA training is joint with state supervision, rather than a separate
embedding pretraining phase.

## Optional teacher RGB prediction (`train.py`)

The alternative RGB path uses:

```
L_RGB = mean(abs(predicted_RGB - target_RGB))
      + 0.05 * (mean(abs(delta_x_error)) + mean(abs(delta_y_error)))
      + 0.0001 * (mean(abs(delta_x_flow)) + mean(abs(delta_y_flow)))
```

The edge and flow terms apply only when their config toggles are enabled; the
flow term additionally requires the flow-generative model. The default current
students use embedding prediction and have no RGB reconstruction loss.

## Optional teacher PPO (`ppo.py`)

PPO is separate from the current supervised student launcher. Its reward is
negative per-step state error:

```
r = -(||position_error||^2 + 0.1*||velocity_error||^2
      + 0.05*||rotation_matrix_error||_F^2 + 0.01*||angular_velocity_error||^2)
L_actor = -mean(min(ratio*A, clip(ratio, 0.8, 1.2)*A))
L_value = 0.5 * mean((V - return)^2)
L_anchor = 0.5 * mean(sum(((policy_mean-reference_mean)/0.1)^2))
L_PPO = L_actor + L_value + 0.01 * L_anchor
```

`A` and returns use generalized advantage estimation. PPO stops updates if its
approximate KL exceeds the configured threshold; that threshold is not an added
loss. The anchor references the frozen pre-PPO head. There is no explicit entropy
term in this implementation. Config settings can override these defaults.

## Legacy internal teacher-to-student state distillation (`train.py`)

The older internal distillation phase uses `L_state + 0.2 * correction_KD`, with
Smooth L1's default beta 1 for this KD term. This is distinct from the budget
student `distill.py` objective above.

## Website metric

Mean Sim(3)-aligned ATE is computed after trajectory alignment and averaged over
complete validation episodes. It remains the selection/reporting metric. The
new alignment auxiliary uses the corresponding position RMSE over each training
horizon, with a small numerical smoothing term. No loss directly optimizes the
website's fuel–ATE area score.

## Component logging and balancing weights

Every logged loss component exposes three fields:

- `*_raw`: its unweighted mean loss.
- `*_weight`: its effective coefficient in the objective being described.
- `*_weighted`: raw loss multiplied by that coefficient.

Budget students record them in W&B under each profile and in
`PROFILE/metrics.jsonl`. Examples:

```
1k/train_state_position_raw
1k/train_state_position_weight       # 3 with default GT weight 1
1k/train_state_position_weighted
1k/train_state_velocity_raw
1k/train_state_rotation_raw
1k/train_state_angular_velocity_raw
1k/train_embedding_prediction_raw
1k/train_embedding_covariance_weight # 0.1 * 0.01 = 0.001 by default
1k/train_embedding_covariance_weighted
1k/validation_state_velocity_raw
1k/validation_embedding_prediction_weighted
```

Student embedding coefficients include the outer `--embedding-weight`; state
training coefficients include `--ground-truth-weight`. Teacher-trajectory
supervision is separately named `teacher_state_*`. Correction KD exposes its
aggregate and `kd_gyro_*`, `kd_body_acceleration_*`, and
`kd_world_acceleration_*` groups. Each group's effective coefficient is one
third of the aggregate KD coefficient, because the aggregate averages nine axes.
Validation state components describe the standalone state validation loss;
validation embedding components are diagnostics and are not added to that score.
Teacher/KD targets are not newly generated for validation solely for logging.

Teacher phases write W&B metrics and append structured records to the run root's
`loss_components.jsonl`, including phase, epoch, training means and validation
means. W&B prefixes are `embedding`, `frame`, `student`, `teacher`, and `distill`.
The historical `student` phase in `train.py` is the deployable main teacher;
`teacher` denotes its optional auxiliary training-time teacher.

The RGB path logs RGB L1, horizontal/vertical edge errors and horizontal/vertical
flow smoothness separately. A zero coefficient marks disabled terms; flow values
are zero placeholders when no flow field is generated. RGB validation components
are diagnostic; selection still uses validation RGB L1.

PPO appends actor, value and anchor raw/weighted terms to `ppo_metrics.jsonl` and
W&B. It additionally records the underlying value MSE, reference divergence,
clipped/unclipped surrogate diagnostics, clip fraction and `objective_total`.
Its validation state components use the `validation_state_*` prefix.

All means are weighted by actual samples/windows (or PPO minibatch sizes), rather
than treating a short final batch as a full batch. Metrics detach from autograd
and are converted for logging after aggregation. Telemetry does not change the
losses, gradients, selection criteria or deployment artifacts.

**Do not sum aggregates and their children together.** For example,
`state_loss_weighted` already contains the four state contributions;
`embedding_loss_weighted` contains the four embedding contributions. Likewise,
PPO's `value_weighted` and `detail_value_mse_weighted` describe the same term at
two levels. Sum either top-level terms or their non-overlapping leaves.

Raw magnitudes and weighted contributions help diagnose dominance, but do not
measure gradient influence or determine optimal weights by themselves. Compare
candidate weights on held-out trajectory metrics; equalizing numerical losses
is not necessarily optimal. These metrics appear in newly started/restarted
processes; existing history cannot be reconstructed retroactively.

## Unaligned MTE and unaligned ATE RMSE

Validation now additionally records **unaligned MTE** (mean translation error):

```
MTE_episode = mean_camera_frames(norm(predicted_position - true_position))
unaligned_mte_m = mean_episodes(MTE_episode)
```

No translation, rotation or scale is fitted. Predictions and truth already use
the recording's initial frame. The camera/IMU lever arm is included, camera
samples include the initial frame, and invalid estimates use the official
scorer's origin/identity fallback. Each complete episode has equal weight,
independent of its duration. No new model forward pass is needed for this metric.

The separate `unaligned_ate_rmse_m` metric is the mean of per-episode position
RMSE values from the official scorer. MTE is an arithmetic mean of Euclidean
errors; it is not RMSE and does not replace Sim(3)-aligned ATE.

W&B keys:

- Budget students: `PROFILE/validation_unaligned_mte_m` and
  `PROFILE/validation_unaligned_ate_rmse_m`; also in `PROFILE/metrics.jsonl`.
- Main teacher state phase: `student/val_unaligned_mte_m` and
  `student/val_unaligned_ate_rmse_m` (the historical phase name is `student`).
- Optional auxiliary teacher/internal distillation: `teacher/val_*` and
  `distill/val_*` with the same metric names.
- PPO: `ppo/val_unaligned_mte_m` and `ppo/val_unaligned_ate_rmse_m`, also in
  `ppo_metrics.jsonl`.

These are diagnostic validation metrics. Curriculum progression, early stopping,
checkpoint selection and the website scoring metric retain their existing rules.


## Student trajectory alignment auxiliary

Fresh student runs default to `--alignment-weight 0.1`; zero disables this
objective. For each trajectory in the training batch, fit Umeyama Sim(3)
`(scale, rotation, translation)` and compute:

```
aligned_i = scale * rotation @ predicted_camera_position_i + translation
L_alignment_episode = sqrt(mean_i(norm(aligned_i - true_camera_position_i)**2) + 1e-12) - 1e-6
L_alignment = mean_episodes(L_alignment_episode)
```

The coefficient is independent of the state and JEPA coefficients. Existing
unaligned state supervision remains active to constrain physical scale and
orientation. This loss uses camera positions including the camera/IMU lever
arm. It fits over the whole current training horizon: initially 2 seconds,
then 4/6/8/10/12/14/16/18/20 seconds as the curriculum progresses. At the 20-second horizon
it covers the full generated 20-second flight. It does not extend short
training windows or train on validation scenes. Validation diagnostics use
complete held-out flights.

The transform is fitted in float64 under stop-gradient; aligned residuals
retain gradients into predicted positions and, through lever arms, rotations.
For the minimizing fit, the envelope theorem supplies the objective gradient
without differentiating SVD singular vectors. Tests compare against numerical
gradients and the official scorer. Near-constant predictions (position
variance <= 1e-12 m²) use identity rotation, scale one and centroid translation.
Windows with fewer than three camera samples contribute zero alignment loss and
report an insufficient-samples diagnostic.

W&B and each profile's `metrics.jsonl` add `train_alignment_raw`,
`train_alignment_weight`, `train_alignment_weighted`, and corresponding
`validation_alignment_*` metrics. Scale mean and batch min/max (averaged across batches), collapsed-fit fraction
and insufficient-sample fraction are logged. Validation alignment contributions
are diagnostics; state loss/Sim(3) ATE keep their existing checkpoint and
curriculum roles. This is a training-only change with no deployment fuel cost.


## Sole-holdout curriculum timeout

Ordinary progression still requires current-phase state-probe loss and complete
validation Sim(3) ATE to plateau. The shared staged curriculum now has one
exception: exactly one unqualified student may be advanced after every other
active student has been frozen for five consecutive completed epochs. This
counts actual epochs, including recording reuse, rather than five fresh-data
cycles. No counter increment occurs in the epoch that first freezes the peers.
A peer thawing, changed holdout/group, or missed epoch resets the streak. The
counter is checkpointed and the forced reason is logged to W&B, terminal and
`curriculum_metrics.jsonl`. Normal stage-best weights, optimizer and EMA targets
are restored at the transition. JEPA/alignment losses and all coefficients are
unchanged; the exception applies to staged multi-student progression only.
