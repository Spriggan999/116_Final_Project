# HyperDrone: models and training design

For environment setup and runnable commands, use [README.md](README.md).
This document explains the current architecture and training implementation.
For the seven budget students' learned encoders, embedding forecasters, retained
temporal backbones and measured fuel limits, see
[the student architecture comparison](STUDENT_ARCHITECTURES_README.md).
The project originated as `hyperdrone_vio_official_viobench_colab.ipynb` and follows
VIOBench's `train.py` plus bundleable `solution/` structure.

## System overview

The model estimates camera motion from RGB images and IMU measurements. It has two
learned components:

1. An image encoder and causal IMU-conditioned predictor supply 48D visual
   embeddings between real camera observations, without rendering RGB.
2. A recurrent state estimator consumes these features and predicts bounded
   corrections to an IMU motion integrator.

Defaults use **640x480 real RGB**, a 48-feature learned visual encoder, a
256-unit embedding predictor, and an **896-wide hybrid Transformer–Mamba-2**
with three official Mamba-2 layers and bounded local attention. There are
18,268,593 stored parameters including the frozen EMA encoder. CUDA and the
official Mamba dependencies are required for default training.
`EMBEDDING_DIM` defaults to 48; the opt-in `curriculum_embedding96.json` selects
96-dimensional learned features through prediction, state fusion and deployment.
It requires fresh SL. See the [capacity and optimization review](verification/training_optimization_review.md)
for the full resource check and accuracy limitations.

The optional GRU uses width 1,792 and has 19,706,585 stored parameters in
embedding mode. Select it with `USE_HYBRID_SSM_TRANSFORMER=false` or `gru.json`.
The earlier RGB/base-33/GRU-1792 model has 20,428,876 parameters; its historical
fuel and size reports below are separate from the default hybrid measurements.

The full-resolution model is intended to serve as a teacher for later experiments.
Its main weights retain the historical filenames `frame_student.pt` and
`state_student.pt`. The separately implemented `USE_TEACHER` experiment trains an
auxiliary state model and distills into this main estimator. The main teacher
can be distilled independently of that auxiliary experiment.
[`distill.py`](distill.py) trains separate native students from a saved SL or
PPO-selected main teacher, as described below.

## Inputs, cadence and outputs

| Quantity | Representation |
|---|---|
| Camera input | RGB, uint8, 640x480 for the current teacher |
| Guest camera rate | 20 Hz |
| IMU rate | Approximately 200 Hz; timestamps determine integration intervals |
| IMU values | Specific force including gravity, then angular velocity, in IMU axes |
| Guest calibration | Image dimensions/intrinsics and `T_cam_imu` |
| Returned state | 13 values: position, xyzw quaternion, linear velocity, angular velocity |

The notebook collector's native capture cadence can be 50 Hz. `StateWindows`
selects approximately 20 Hz observations causally: the default stride-four captures
map desired ticks 0,10,20,30,... onto available ticks 0,8,20,28,... . This makes
training observation gaps variable; it is not an exact reproduction of the
benchmark's fixed 20 Hz cadence.

The guest supplies a pose relative to the first camera frame and a twist expressed
in the current camera's axes. The first state is identity with zero twist.

## Model 1: default direct embedding prediction

`VISUAL_PREDICTION='embedding'` / `--visual-prediction embedding` is the default
for fresh runs. `EmbeddingPredictor` combines the existing full-image encoder
with a `152 -> 256 -> 48` tanh MLP. Its inputs are the previous real embedding,
latest real embedding, their difference, mean six-axis IMU divided by ten,
elapsed time since the latest capture and the preceding capture interval.
The output is `anchor + 0.25*tanh(MLP(inputs))*clamp(elapsed/0.05,0,2)`.
Observed ticks use the real encoder output directly.

Only selected real captures are encoded. All missing ticks are predicted in
feature microbatches of up to 1,024 rows by default. The state window is
`[B,T,48]`; no `[B,T,3,H,W]` generated image sequence or second visual CNN is
needed. The guest encodes each incoming real image once and keeps two past
48D anchors plus the state estimator's bounded recurrence. Neither future
images nor future IMU enter a missing-tick prediction.

The first training stage uses observed `(previous, anchor, future)` camera
triples. An online encoder and predictor learn future embeddings against a
stop-gradient target encoder, updated after optimizer steps with EMA momentum
0.99. Target features never receive gradients. The default objective is:

```text
SmoothL1(predicted_future, EMA_encoded_future; beta=0.1)
+ SmoothL1(online_encoded_future, EMA_encoded_future; beta=0.1)
+ 1.0 * mean(relu(0.1 - sqrt(population_variance(online_features) + 1e-4)))
+ 0.01 * sum(off_diagonal(feature_covariance)^2) / 48
```

Regularization pools online previous/anchor/future features. Population variance
and a bounded covariance denominator keep batch-one objectives finite. These
penalties discourage collapsed content features; they do not establish learned
accuracy. All context/target encoder feature dropout is disabled so state and
deployment use consistent features. The selected predictor and encoder are
frozen for state SL/PPO and distillation. State-stage visual dropout is therefore
an RGB-mode setting; the embedding encoder is not fine-tuned by state loss.

This is an IMU-conditioned adaptation inspired by [MC-JEPA](https://arxiv.org/abs/2307.12698).
The published method jointly learns optical flow and content features; this
project uses direct global feature prediction and EMA targets rather than its
full optical-flow/self-supervised training pipeline. Hyperparameters are defaults
for this implementation and have no accuracy or convergence result attached.

`--visual-prediction rgb` restores the RGB path below. New runs record the mode
and embedding architecture in both JSON configurations. Warm starts without a
mode field retain RGB, and switching representations requires fresh training.
`USE_GENERATED_FRAMES_FOR_STATE=False` holds the last real embedding in direct
mode, or RGB in RGB mode. `export.py --frame-mode repeat` supports that hold
ablation while retaining the checkpoint's representation. RGB caches are
disabled for direct embeddings. GRU, Mamba, auxiliary teacher, PPO, checkpoint
export and fuel-budgeted student distillation share this representation setting.

## Optional model 1: generative flow predictor

`GenerativeFlowPredictor` receives:

- The previous and most recent real RGB images.
- Their RGB difference and normalized XY coordinates: 11 spatial channels total.
- Mean IMU values since the latest anchor, divided by 10, plus prediction horizon
  and previous real-frame spacing: an eight-value condition vector.

The default learned stem is a convolution with **kernel 8 and stride 8**, consuming
every pixel in each non-overlapping patch. All inputs arrive at full resolution;
there is no preliminary fixed resize to the old 96x72 resolution.

At 640x480, the U-Net-like path is:

| Stage | Height x width | Channels |
|---|---:|---:|
| Input | 480x640 | 11 |
| Learned patch stem / `enc1` | 60x80 | 33 |
| `enc2` | 30x40 | 66 |
| `enc3` and bottleneck | 15x20 | 132 |
| `dec2`, with skip connection | 30x40 | 66 |
| `dec1`, with skip connection | 60x80 | 33 |
| Raw-field head | 60x80 | 6 |
| Bilinearly restored fields | 480x640 | 6 |

Blocks use convolution, GroupNorm and SiLU. An IMU MLP conditions the bottleneck.
The six output fields represent XY flow, a blend mask, and three RGB residuals.

Flow is bounded by `FRAME_MAX_FLOW_PIXELS` times `tanh`, scaled by
`clamp(elapsed / 0.05, 0, 2)`. The current bound is 160 pixels, preserving the
relative displacement range of 24 pixels at the old width of 96. The most recent
real image is warped at full resolution, then blended with an RGB refinement of
that anchor. The mask is `sigmoid(raw_mask - 2)`; the residual bound is 0.25 in
normalized RGB units. Both image inputs precede the target time.

PyTorch training uses `grid_sample`. Deployment exports the raw-field network to
ONNX and performs the warp with OpenCV `remap`. OpenCV interpolation is not
bit-identical to PyTorch's warp; raw ONNX fields are numerically compared at export.

Alternatives retained from the notebook are the small `MotionWarp` affine model
and an ablation that repeats observed frames instead of generating new ones.

## Model 2: visual encoding and recurrent estimation

### Visual encoder

In direct mode this encoder belongs to `EmbeddingPredictor` and runs only on
real captures. The state model uses an identity visual front end. In RGB mode
the state model owns the encoder and applies it to real/generated RGB.

The learned encoder takes one RGB image at 640x480 and applies:

| Stage | Output shape, excluding batch |
|---|---|
| Learned 8x8/stride-8 projection + ReLU | 16x60x80 |
| 3x3/stride-2 convolution + ReLU | 32x30x40 |
| 3x3/stride-2 convolution + ReLU | 48x15x20 |
| 3x3/stride-2 convolution + ReLU | 64x8x10 |
| Adaptive average pool | 64x3x4 |
| Flatten, linear, tanh | 48 features |

RGB-mode training-only feature dropout defaults to 0.05. A low-cost alternative averages
RGB to grayscale and adaptively pools to 6x8, retaining the 48-feature interface.

### Shared state inputs and GRU alternative

At every IMU tick, the recurrent input contains:

| Component | Values |
|---|---:|
| Current visual features | 48 |
| Change from previous tick's features | 48 |
| IMU divided by 10 | 6 |
| Time since last real image | 1 |
| Real-image observation flag | 1 |
| Total | 104 |

A linear/tanh layer maps 104 values to the hidden width. With
`USE_HYBRID_SSM_TRANSFORMER=false`, a GRUCell updates the 1,792-dimensional
hidden state. A linear head produces nine values, bounded by
`0.25 * tanh`: angular-rate corrections, body specific-force corrections, and
acceleration corrections in the integrator's initial reference frame.

### Default Mamba-2 plus local attention

`USE_HYBRID_SSM_TRANSFORMER=True` is the default: three Mamba-2 blocks followed
by local causal attention. Each Mamba block has a residual
connection and LayerNorm. Defaults are state dimension 16, head dimension 32,
expansion factor two, depthwise convolution width four, four attention heads, and
a 16-token attention window.

Mamba caches have constant size; attention keeps a bounded token history.
The only host Mamba implementation is official `mamba_ssm.Mamba2`; hybrid
training, evaluation, PPO and distillation require CUDA. Export constructs its
modules on CPU and serializes the same parameters; the WASM guest uses NumPy
for the recurrence, gated RMS normalization and local attention. No CUDA code
runs inside WASM. The custom PyTorch fallback and backend toggles are removed.

Fresh hybrid training chooses hidden width 896 (explicit `STUDENT_HIDDEN`
overrides it), rather than the GRU width 1,792. The [hybrid preset](hybrid.json)
passed a 1,200-frame synthetic WASM check: 118.01 MiB upload, 8.62 G step fuel,
325.6 MiB linear memory, 0.746 s maximum step wall time and 672.4 s total.
Width 512 also fits. Width 1,792 does not fit the upload cap. Resource results
use seeded weights with nonzero output heads, not an accuracy-trained model.

The verification compares official CUDA parallel scan and recurrent step at
lengths 11, 101, 257 and 513, the NumPy deployment with identical weights, a
real optimizer update, and actual WASM camera/IMU outputs. FP32 checks use
`atol=5e-5, rtol=5e-4` for neural outputs and `atol=rtol=5e-4` for integrated
physical state. PyTorch TF32 flags alone do not control Triton's scan dot
products; `TRITON_F32_DEFAULT` defaults to `ieee` before importing Mamba. BF16
training remains available and has its usual reduced precision. The supported
fused layout requires an inner head count divisible by eight and state size
divisible by four; with head dimension 32, choose hidden multiples of 128.

See [verification](verification/README.md#official-mamba-2-parity-and-hybrid-limits-2026-11-12)
for exact versions, errors and scope; arbitrary upstream Mamba configurations
and future package/kernel versions are not covered by these checks.

## Physical integration and coordinate frames

The estimator retains the notebook's body/IMU-axis correction convention.
For each interval, it adds the learned rate correction to gyro measurements,
rotates corrected specific force into the initial IMU reference, adds estimated
gravity and the learned reference-frame acceleration correction, then integrates
velocity and position. Rotation advances using the SO(3) exponential map.

Training windows express position and rotation relative to their first body frame.
Velocity targets are changes relative to the window's initial velocity. The model
starts with zero velocity and estimates gravity from early accelerometer samples,
normalized to magnitude 9.81. This relies on a near-static-start approximation.

The guest converts the integrated state using `T_cam_imu`, including the camera's
lever arm. Orientation uses scalar-last, normalized xyzw quaternions. Linear and
angular velocities are returned at the camera center in current camera axes.
Calibration does not currently condition the learned encoder/recurrent feature
vector; camera-distribution changes can therefore affect learned corrections.

## Supervised training

The visual-prediction and state stages run sequentially. The state estimator learns from the
selected frozen embedding predictor/encoder, or the RGB generator's outputs. Training that predictor concurrently
would change the state estimator's input distribution and require a different
training schedule. Background recording collection already overlaps with the
active stage. Performance settings batch independent flights and training inputs;
frame and state optimization still run as separate stages.

### Optional RGB frame stage

Native future camera captures supervise the frame generator. The primary loss is
RGB L1. Optional edge and flow-smoothness losses are disabled by default. AdamW,
gradient clipping, validation-driven learning-rate reduction and early stopping
are used. The best validation-L1 checkpoint is retained and then frozen for state
training.

### State stage

Default windows contain 101 IMU ticks, approximately half a second. Visual embeddings
between selected real captures are predicted causally; RGB prediction and hold
ablations are selectable.
The recurrent model is trained through the physical integrator.

`losses.py` computes:

```text
L = 3 * position_SmoothL1
    + velocity_SmoothL1
    + mean(rotation_geodesic_angle_squared)
    + 0.1 * angular_velocity_SmoothL1
```

SmoothL1 uses beta 0.2. The best validation checkpoint is selected. This window
loss is not the same quantity as full-episode Sim(3) ATE.

### Validation and patience with reused recordings

Both stages validate and checkpoint after every actual epoch. Streaming-mode
`FreshBatchPatience` charges a non-improving first epoch on a new recording batch
once; later epochs on that batch do not increase the counter. Any qualifying
improvement resets it, including improvements during reuse. Each stage has its
own counter. Fixed-data mode supplies a new epoch ID each time, preserving
ordinary per-epoch patience. The optional auxiliary-teacher and distillation
stages use the same accounting.

With `USE_CURRICULUM`, any motion/visual/IMU or SL window promotion resets the
counter and starts `CURRICULUM_EARLY_STOP_GRACE=3` protected fresh data cycles.
The promotion evaluation itself does not consume grace. Subsequent fresh batches
each consume one cycle; reused passes remain protected without extra charges.
Fixed-data cycles are epochs. A further promotion restarts the configured period,
and zero grace still resets patience. Checkpoint comparisons remain against the
global best fixed validation score throughout, including adaptation. Epoch caps
remain unchanged. With the full curriculum, state LR scheduling uses the current
horizon's progress probes and restarts plateau tracking when that horizon changes.
PPO retains its iteration cap.

Embedding pretraining selects a validation-objective decrease greater than `1e-5`.
RGB frame training requires any strictly lower validation L1; either has patience
of 8 non-improving fresh batches and a default cap of 200 epochs. The state
estimator requires a validation-loss decrease greater than `1e-5`, with patience
6 and default cap 300. The curriculum preset raises caps to 400/600; stopping
waits for full curriculum exposure and adaptation grace. Extra passes count
toward the epoch caps.
`ReduceLROnPlateau` still receives every epoch's validation score. These criteria
continue to use supervised validation losses; full-trajectory Sim(3) ATE-based
selection is a separate future change.

### Optional auxiliary teacher and distillation

With `USE_TEACHER=True`, `StateTeacher` uses a separately configured hidden width
(256 by default), a two-layer GRU or Mamba stack, and causal attention. It trains after the main
estimator. Distillation combines the same supervised state loss with 0.2 times
SmoothL1 between the models' nine correction outputs. Its best validation
checkpoint is written as `state_student_distilled.pt` and becomes the selected
main estimator. Both use the configured image resolution in this implementation.
Its default width is smaller than the new main teacher's width; configure it
separately for auxiliary-model experiments. This option is distinct from a future
experiment that trains a smaller-resolution student to imitate the current
full-resolution teacher.

## Background data generation and storage

`RollingBatches` owns a unique managed directory and runs collection in a separate
subprocess. It keeps at most one active training batch, one prefetched batch, and
one fixed validation batch. The recorded-data budget is split among these slots,
with metadata reserves. A completed next batch is consumed at an epoch boundary;
only then is the old batch removed and a new collection job launched.

If collection is still running, `STREAM_MAX_PASSES_PER_BATCH=3` permits up to
three uses of the current batch, including its initial pass. The ready replacement
takes priority at every boundary; after the pass limit, the caller waits for
collection. A completed failed producer raises its error immediately at the next
boundary, even if reuse passes remain. Reuse retains the loaded data and loaders;
their shuffled iterations provide new minibatch order without reloading NPZs.
The limit is shared across stages; one use is an SL epoch or a PPO iteration.
Setting the limit to one restores waiting after every pass. Validation is fixed.

Training scenes rotate within `NUM_SCENES` (313) starting at `SCENE_OFFSET` (0).
Four scenes and two flights per scene are requested in each training batch.
Validation uses `STREAM_VALIDATION_SCENES` (12) consecutive held-out scenes,
starting at `SCENE_OFFSET + NUM_SCENES`, with one flight per scene. These defaults
partition all 325 available Train200 + Train125 environments into training and
held-out pools. `SCENE_CATALOG='mixed'` interleaves 192 ProcTHOR + 121 HSSD training
scenes before 8 ProcTHOR + 4 HSSD held-out scenes. Recording IDs stay stable:
ProcTHOR 0..199, HSSD 200..324. Catalog-specific collection is also available.
The final training batch of a rotation can be shorter, so remainder scenes are
visited without entering validation. Batch seeds
change; the fixed validation set keeps checkpoint comparisons consistent. In
fixed-data mode, the highest recorded scene ID is held out instead.

The collector compresses an episode in memory and checks its output budget before
writing. `EpisodeStore` loads/closes NPZ files before loaders use them and retains
matching-resolution RGB as uint8. Frame pairs and sampled camera windows are
converted to normalized float32 only when used. Legacy resized inputs retain
their original float interpolation behavior. This saves 4x image-storage RAM
without changing sampled pixel values. Cleanup
does not remove files still being read. Collection failure, timeout and normal
interruption propagate through the training context and stop the producer before
cleanup. Existing datasets and other run directories are outside that ownership.

These are episode batches, separate from optimizer minibatches. Background
collection and training share GPU resources. They may contend rather than obtain
an ideal sum of their standalone throughput.

## Optional PPO refinement

Fresh runs default to SL followed by PPO. `--from-checkpoints` defaults to
PPO-only when both component weights are available, including partially
completed SL runs. `--sl` explicitly requests more SL before PPO; `--no-ppo`
selects SL-only. If a component checkpoint is missing, PPO-only fails before
collection. `--sl` loads available components and initializes missing ones for
supervised training. `--ppo-only` remains an explicit alias for PPO-only. PPO
optimizes only the existing nine-output correction head; the visual and recurrent
backbone is frozen. The exported architecture does not gain a critic or new inputs.

Camera/IMU recordings are external observations. Each rollout samples **fresh
correction actions** under the current actor and re-integrates the estimator;
those sampled actions determine the estimator trajectory. This refines an
estimator rather than commanding the simulator's autonomous flight controller.
No old action/log-probability buffer is reused across PPO iterations.
The streaming recording scheduler can reuse the camera/IMU batch for up to the
same three-pass limit while waiting for new recordings. This is valid here
because the estimator's correction actions do not affect the simulator flight.
`collect_rollout` still samples from the current head and re-integrates estimator
state on every iteration. PPO logs include recording batch and pass number.

An optional `PPO_CURRICULUM_WINDOWS` list advances estimator windows from short
to long, gated by sustained validation improvement. With `USE_CURRICULUM`,
`SL_CURRICULUM_WINDOWS` also trains the backbone on increasing horizons during SL.
Each phase's checkpoint validation uses its longest horizon throughout, while
progress gates use fixed current-horizon probes. New horizons reset probe and
LR scheduler references. A stable-plateau fallback accepts loss up to 10% above
the stage best after six updates. SL early stopping waits for all axes and the
longest window, then three fresh final-stage batches before ordinary patience.
Hard epoch caps still apply; the launcher has no time limit unless `--seconds` is supplied.
The revised preset starts state training at 2 seconds and interleaves HSSD from
the first batch. See the [W&B evidence](verification/wandb_curriculum_20261113/README.md).
Recordings must
already support that horizon, and each validation recording must supply at least
one full window. The default empty list retains a fixed horizon. The example
[`curriculum.json`](curriculum.json) enables motion, visual, IMU and length controls
for frame/embedding SL, state SL and PPO; see the
[commands and promotion rules](README.md#mixed-scene-collection-and-ppo-curriculum).
This does not train the autonomous flight controller or make collision avoidance
depend on estimator PPO performance.

Actions are Gaussian in the head's pre-tanh coordinates, with fixed standard
deviation 0.1, then bounded by `0.25 * tanh`. The fixed transform's Jacobian cancels
in probability ratios, so PPO uses latent-action log probabilities. The reward is
the negative of:

```text
squared_position_error
+ 0.1 * squared_velocity_error
+ 0.05 * squared_rotation_matrix_Frobenius_error
+ 0.01 * squared_angular_rate_error
```

A training-only critic receives the frozen recurrent feature vector, current
estimated position/velocity/rotation, and privileged target position/velocity/
rotation. Ground truth enters the critic and reward, not the actor. The critic is
a 64-unit tanh MLP with scalar output.

Updates use GAE with critic bootstrapping at time-window boundaries (the final
token is reserved for the next-state value), clipped likelihood ratios, gradient clipping, an approximate-KL
early stop and a penalty anchoring the actor to the pre-PPO head. Deterministic
validation selects a candidate only if the fixed SL loss improves; otherwise the
previous best candidate or SL baseline is retained. This does not guarantee
improvement in full-episode ATE or on unseen trajectory families.

The selected actor, pre-PPO baseline and latest training state are saved separately.
Before PPO in a fresh or explicit SL run, `save_post_sl` atomically publishes
`post_sl/` with frame/state weights, export/training settings and a stage marker.
The selected SL state is also saved as `state_student_sl.pt`. Snapshot weights
use hard links where supported (copies otherwise); all checkpoint weight updates
use atomic replacement, preserving the snapshot's original inodes. PPO changes
the selected root state weights, leaving the post-SL set exportable even after
interruption. Export `post_sl/` for the SL model and the run root for the final
PPO-selected model. The latter can equal the SL baseline. PPO-only runs retain
their pre-PPO baseline but do not label it as a newly completed SL stage.

`ppo_training.pt` is for inspection; the CLI does not provide exact optimizer-state
resumption from it. Current refinement does not implement prioritized scenario
sampling or prioritized experience replay.

## Native student distillation

`distill.py --teacher DIR` accepts the main teacher's complete checkpoint set:
use `RUN/post_sl` for the SL version or the completed run root for the PPO-selected
version. Both the causal frame generator and state model are loaded, frozen, and
used only on the host. Seven independent student models share each training
minibatch's teacher forward pass. Teacher targets are computed online in RAM;
there is no growing disk cache or copy of the teacher weights in student runs.
The teacher path and both checkpoint hashes are recorded for provenance.

CUDA distillation defaults to one host worker and CUDA stream per student plus
a teacher-target producer stream. CUDA events make all students wait for their
shared labels; tensor stream records protect labels until consumers finish.
The producer prepares only one following minibatch while current student
updates run. Each student retains its own optimizer, checkpoint selection and
early stopping, with no overlapping updates to the same model. Validation
uses the same concurrent student streams with thread-local no-grad/FP32
contexts and ground-truth-only targets. CUDA data-loader workers use `spawn`
so no worker forks a process containing active scheduling threads.

The producer fetches/collates the next CPU minibatch as well as generating its
labels. Inputs transfer once for both teacher and student preprocessing.
Student neural recurrence runs in one autocast scope to reuse BF16 weight casts
across camera ticks; integration follows outside that scope in FP32. No cast
cache survives a forward pass or optimizer update.

Student linear layers use the configured BF16 neural autocast during training;
their corrections/hidden state are converted back to FP32 before integration.
CPU and `--no-parallel` runs use serial scheduling, with no target prefetch.
`--no-amp` controls neural precision independently. Prefetch is bounded in RAM,
but simultaneous student activations and teacher inference increase peak GPU
memory compared with serial execution; tune batch size/profile selection on the
actual GPU. `performance.jsonl` records full-epoch windows/second (including
validation/data wait), active profiles, student updates, collection wait and
CUDA peak allocated MiB. Compare serial/parallel runs on the same fixed dataset
to measure benefit; scheduling alone does not establish full GPU saturation.

### Student architectures and cadence

All students request 320x240 RGB through the ABI. The 500 tier uses two grayscale
pixels and their temporal differences plus the last accelerometer sample. The larger tiers sample grayscale values at
fixed pixel centres. Training matches the ABI's uint8 2x2 area-average downsample
from 640x480. Features concatenate current samples, sample differences, mean
six-axis IMU divided by 10, and the frame interval times 20. The input width is
`2 * grid_size + 7`. Intrinsics do not currently condition the neural features.

| Tier | Visual samples | Recurrent state | Trainable parameters | Fuel cap |
|---|---:|---:|---:|---:|
| `500` | 2x1 | Position/velocity + previous two lumas; affine visual/accelerometer model | 18 | 500 |
| `10k` | 6x3 | None; linear correction head | 396 | 10,000 |
| `1m` | 8x6 | 192 | 58,569 | 1,000,000 |
| `100m` | 32x24 | 1,792 | 5,994,249 | 100,000,000 |
| `1m_ssm` | 8x6 | Official Mamba-2, width 64, one block | 34,353 (starting candidate) | 1,000,000 |
| `1m_hybrid` | 8x6 | Official Mamba-2 + bounded attention | 51,121 (starting candidate) | 1,000,000 |
| `100m_hybrid` | 32x24 | Official Mamba-2 + bounded attention, width 384 starting candidate | 3,924,033 (starting candidate) | 100,000,000 |

The new temporal models have distinct profile names and checkpoints. Native
deployment uses the official Mamba-2 equations ported into C, including gated
RMS normalization, residual LayerNorm, and cached attention projections.
Their ordered architecture search enforces 5% fuel headroom before training;
final trained exports are checked again. Existing RNN architectures are retained.
CUDA integration uses differentiable associative quaternion prefix products;
its equations and gradients match the original normalized Euler integrator.

The standalone distiller now defaults to a 500-epoch safety ceiling, a shared
2–20-second horizon ramp and per-profile early stopping after eight stale fresh
batches at the final stage. Full validation episodes select checkpoints by
Mean Sim(3) ATE. Automatic stream grouping is measured at each new horizon and
requires 15% estimated GPU memory headroom. See [STUDENTS_README.md](STUDENTS_README.md).
The frozen hybrid teacher can explicitly use parallel physical integration for
distillation while staying in FP32 evaluation mode. Its neural corrections match
the serial path; the sampled real 20-second target differed by under 1 mm in
position from FP32 reduction order. The original inference default is preserved.
Per-kernel host locks protect Triton's shared autotuner state without serializing
whole models or synchronizing CUDA streams.

The recurrent students use `h = tanh(W*x + U*h + b)` at camera rate, approximately
20 Hz. Each head emits the same nine IMU/reference-frame corrections as the
teacher, bounded by `0.25*tanh`. They do not generate intervening images or run
a CNN at every IMU tick. This compresses the teacher's overall estimation behavior
through output/trajectory targets rather than distilling its pixels or copying
its architecture.

The 10K model aggregates one packet's IMU values and integrates once per camera
frame. The larger two apply the frame's predicted correction to each timestamped
IMU sample. Orientation uses a normalized quaternion Euler update; training uses
the same update as C inference. This approximates the teacher's exponential-map
rotation update. Native physics uses doubles and NN weights/activations use
float32. Output conversion includes the supplied IMU-to-camera rotation and lever
arm. The existing zero-velocity/near-static-gravity initialization assumptions
remain, and the smallest model has very limited visual information and capacity.

The 500-fuel tier uses a separate SIMD kernel and 18 trainable coefficients:
`a = gain * last_accelerometer + bias + W_visual * [left, right, delta_left, delta_right]`.
Gains start at one, bias at `[0,0,-9.81]`, and visual weights at zero. The two
RGB pixel means are normalized to [0,1]. The frame update is `v += a*0.05`,
`p += v*0.05`. Camera rotation and grayscale scaling are folded into coefficient
columns during init. Attitude remains fixed; gyro is omitted. It supports an
empty first IMU packet, followed by ten-row packets at 20 Hz. It replaces the
100-fuel default; legacy six-parameter checkpoint export is retained. The new
profile measured 150/500 fuel over 1,200 frames with nonzero visual weights.

### Distillation objective and checkpoint selection

Each training window supplies camera-rate images causally from its anchor indices.
The teacher processes its predicted 200 Hz embedding/RGB sequence. Its nine corrections
are averaged over each camera interval. Default training loss is:

```text
1.0 * SmoothL1(student_interval_corrections, teacher_interval_mean_corrections)
+ 0.5 * state_loss(student_trajectory, teacher_trajectory_at_camera_ticks)
+ 1.0 * state_loss(student_trajectory, simulator_truth_at_camera_ticks)
```

Correction SmoothL1 uses beta 0.05. `state_loss` is the existing weighted
position/velocity/geodesic-rotation/angular-velocity objective. The weights are
CLI overrides, not experimentally tuned constants. Validation selects each
student using complete held-out trajectory Mean Sim(3) ATE, independently of the
other tiers. Ground-truth state loss is also logged. The expensive teacher is
not run during ground-truth-only validation.

For `500`, the imitation term instead matches the student's visual/IMU acceleration
to finite differences of teacher velocity at camera ticks, divided by their
time interval. It retains the same trajectory/ground-truth terms. Student-only
camera preprocessing is retained for its two visual samples. The frozen
teacher also consumes images to produce its targets. These student runs use
supervised knowledge distillation, without an additional PPO stage.

Default maximum training length is 500 actual epochs, with patience of 8
non-improving fresh recording batches (epochs in fixed-data mode). Streaming uses
the same bounded producer, fixed validation, three-pass reuse cap and fresh-batch
patience helper as teacher SL. Models that early-stop retain their own best
`student.pt`; the other tiers can continue. `--profiles` selects a subset.

### Native export and measured fuel

`python -m students.export` embeds float32 weights with C23 `#embed` and compiles
the ABI reactor with wasi-sdk 34 and SIMD: `-O3` for the 500 tier and legacy 100 export, `-O2` for the
larger tiers. Generated C and weights stay
beside the artifact for inspection. The standalone module has no Python/OpenCV
runtime. A local Python identity solution used roughly 87K–214K fuel/frame, which
already exceeds the smallest tier.

Every export meters the compiled candidate with its tier's exact step cap,
plus the teacher's other supplied resource limits. The default is 1,200 frames.
Only passing candidates are promoted to `native/student.wasm`. `limits.json`
records the module/weight hashes and measurements; `submission_limits.json`
sets the corresponding cap for real benchmark evaluation. A failure preserves
the trained checkpoint so export can be diagnosed or retried separately.

The following four-tier probes are historical and predate the new 500 profile.
Its current [500-fuel resource check](verification/student_500_resources.json)
passed 1,200 frames at 150 fuel/frame. No timing benchmark was repeated.

Historical resource probes passed all 1,200 frames, using random weights (seed 1701) for
the neural tiers and explicit nonzero affine coefficients for the 100 tier:

| Tier/report | Peak fuel/frame | Module size | Episode test time |
|---|---:|---:|---:|
| [100](verification/student_100_resources.json) | 99 | 0.02 MiB | 1.6 s |
| [10K](verification/student_10k_resources.json) | 7,886 | 0.02 MiB | 1.55 s |
| [1M](verification/student_1m_resources.json) | 803,632 | 0.25 MiB | 1.61 s |
| [100M](verification/student_100m_resources.json) | 80,073,240 | 22.89 MiB | 7.74 s |

Observed linear memory was 64 MiB for each. These are architecture resource tests,
not distilled-model accuracy results. The 100-tier probe uses nontrivial affine
test coefficients; the other probes use random initialization. The 100 tier has
only one fuel unit of margin with the measured compiler and is especially
sensitive to code/compiler changes. Verify the actual trained weights and
evaluate Mean Sim(3) ATE on complete held-out episodes using the student's own
limits file. Changing compiler, architecture or inference code requires a new
resource check. In particular, expect substantial accuracy tradeoffs at 100 and 10K.

## Export and VIOBench execution

Training runs with PyTorch and simulator dependencies on the host. The guest uses
NumPy and OpenCV provided by VIOBench's Python/WASI runtime.

Export records architecture, resolution, learned patch strides, flow bound and
normalization conventions. It checks static ONNX graphs with OpenCV against
PyTorch, using nonzero image/condition inputs, then runs the official Python guest
shim. Adaptive pooling is lowered to an equivalent fixed grouped convolution so
the encoder can use OpenCV DNN. Weight NPZ files are uncompressed because the WASI
Python runtime does not include zlib.

GRU/Mamba recurrence uses NumPy; the image networks use ONNX/OpenCV. The guest
returns the ABI's 13-value state for every camera frame. `viobench bundle` embeds
the source and assets into the Python runtime, producing one `.wasm` file.

For direct embeddings, `visual.onnx` contains the predictor's real-image
encoder (when learned), `frame_student.npz` stores the predictor/encoder state,
and the guest evaluates the small prediction MLP in NumPy. No `frame_raw.onnx`
or RGB warp is used. The EMA encoder is training-only. Export keeps the mode
in `deployment.json`; the ABI, physics and native student kernels are unchanged.

Legacy checkpoints without patch-stride metadata retain their original stems:
stride one for flow and stride two for the visual CNN. Applying the current preset
may change kernel shapes; SL initializes the changed stem and retains compatible
remaining weights. PPO-only rejects such a shape change because its backbone is
frozen. Old weights or an old `.wasm` do not become a retrained full-resolution
teacher merely by changing the ABI frame-width request.

## Memory and compute choices

Default sequence minibatches contain four windows; frame minibatches contain
eight pairs. Embedding prediction chunks contain up to 1,024 feature vectors;
optional RGB generation and visual CNN chunks contain up to 16 frames. Visual CNNs use
non-reentrant activation checkpointing during backpropagation. These settings
also appear in `full_resolution.json`; changing the minibatch changes optimizer
updates per epoch, so compare held-out trajectory accuracy after retraining.

RGB frame synthesis plans observed/missing indices once on CPU, copies actual captures
directly, and sends only missing frames through the generator in chunks. It gathers
causal anchors and writes one preallocated FP32 output sequence. There is no
CUDA-boolean read inside its chunk loop. Training loss logging retains detached
GPU scalars until the epoch boundary; unused per-minibatch RMSE conversions are
disabled. Epoch loss averages weight partial final minibatches by sample count.
Synthetic-window disk caches remain off.

On supported CUDA GPUs, convolutional layers and the optional GRU's linear/recurrent
layers use BF16 autocast. One autocast scope spans recurrent ticks to retain cached
trainable weight conversions. Frame warps, flow restoration, SO(3)/quaternion
updates, pose/velocity integration, and supervised losses use FP32. Validation
explicitly disables neural autocast, matching FP32 export arithmetic. Parameters,
optimizer state and checkpoint files remain FP32. CPU and export use FP32. The
default official Mamba hybrid uses FP32 recurrence with IEEE Triton dots; its visual
CNN can use BF16. `--no-amp` disables this optimization without changing shapes.

CUDA state loaders use two CPU workers with one prefetched minibatch each and
pinned buffers. Frame pairs use pinned buffers in the main process: measured
multiprocessing overhead outweighed the benefit for their simple slice operations.
Copies request `non_blocking=True`; no unbounded loading queue or extra disk cache
is introduced. CPU runs use zero workers. `LOADER_WORKERS`,
`FRAME_LOADER_WORKERS`, `LOADER_PREFETCH`, and `PIN_MEMORY` are configurable.

`PERSISTENT_WORKERS=True` keeps workers warm between epochs on the same data.
Training workers are explicitly shut down before a rolling batch is replaced;
validation workers persist until run cleanup. This keeps the worker/data count
bounded and prevents stale recordings. `FUSED_ADAMW=True` enables fused CUDA
AdamW for frame/state SL, auxiliary teacher/distillation, and native student
distillation. CPU uses ordinary AdamW. Both training CLIs accept
`--no-persistent-workers` and `--no-fused-adamw` for independent comparisons.

The existing single producer now runs two independent flights in one scene/environment
at a time. Observations retain their instance axis, only the first camera of each
flight is selected, and terminated slots stop recording while other flights continue.
Episode tails, per-flight IMU/state/RGB and budget checks stay separate. It resets
all slots between rounds and truncates the final round to the requested episode
count. Fixed validation remains single-flight; custom collectors keep their own
parallelism policy and their original CLI contract.

`performance.jsonl` records stage/epoch throughput including loader startup, peak
allocated CUDA memory and fresh-data wait time. Teacher stage timing excludes validation;
distillation timing includes validation and fresh-data wait. In both cases,
allocator memory does not include renderer/driver allocations. Compiler startup,
validation, NPZ compression and waiting for fresh data can still leave GPU gaps.
The [measurement record](verification/training_performance.json) gives local
probe settings and results; these measurements do not establish ATE improvements.
The optional RGB export architecture and native student fuel kernels are preserved;
direct embeddings use the new feature predictor described above.

Default recordings contain up to 2,000 IMU samples (approximately 10 seconds).
The raw simulator's native step limit is informational; `COLLECT_STEPS` can exceed
it, and crashed slots still stop independently. The recording budget is 3 GiB,
split among the active training batch, one prefetched batch and fixed validation.
The larger scene pool is streamed; scenes are built sequentially with two flights
per training scene, not all resident on the GPU. Recorded windows remain 101 ticks.
Compiler caches, checkpoint history, evaluation downloads and RAM are outside
the recording limit. The [expanded-collection hardware probe](verification/README.md#expanded-collection-defaults)
records earlier ProcTHOR-only collection and CUDA forward/backward checks. These
measurements do not validate HSSD throughput or long-window training. Manual
collection defaults to 313 scenes, two flights
per scene, 2,000 samples and a 3,072 MiB output cap; that cap can stop a manual
collection before it visits every requested scene. Rolling validation rejects
incomplete held-out scene coverage instead of silently accepting a truncated split.

The learned patch stems make default resource use practical while consuming all
input pixels. Their features are spatially reduced; the flow field's restored
640x480 shape does not mean every expensive U-Net block runs at that resolution.

## Resource requirements and measured profile

The embedding capacity review below is separate from the historical RGB
capacity and full-episode reports. Verify the selected trained artifact with
`--verify-limits`; host checks alone do not establish WASM fuel or full-episode
behavior.

| Requirement | Cap |
|---|---:|
| Upload | 128 MiB |
| Total time per episode | 24 minutes |
| Initialization fuel | 200 G |
| Fuel per step | 20 G |
| Linear memory | 2 GiB |
| Table elements | 1,048,576 |
| Initialization wall-clock deadline | 300 seconds |
| Step wall-clock deadline | 60 seconds |

The account's two daily submissions reset at midnight in America/Los_Angeles.
Those are server policies, separate from model execution.

### Embedding teacher and student capacity planning

The default hybrid's 1,200-frame WASM check uses 118.01 MiB upload, 8.62 G
maximum step fuel, 325.6 MiB memory and 672.4 seconds total. Its full results
are in [hybrid_resources.json](verification/hybrid_resources.json).

The earlier GRU embedding teacher's short smoke export uses 123.49 MiB of the
128 MiB upload cap, 3.970 G of the 20 G step-fuel cap, and 347.6 MiB of the
2 GiB memory cap. Upload size limits ordinary FP32 width increases first.
Unused fuel, memory or table capacity does not imply that another wider model
will fit the upload allowance. CUDA training throughput and exported WASM
resource use are separate measurements.

The [capacity headroom review](verification/capacity_headroom_review.json)
compares these candidates without changing the defaults:

| Model | Comparison baseline | Candidate | Candidate binding resource |
|---|---|---|---:|
| Optional GRU teacher | GRU 1,792 / predictor 256 | GRU 1,824 / predictor 256 | 126.16 MiB upload |
| `100` student | Six-coefficient affine prior | Retain current model | 99/100 fuel |
| `10k` student | 4x3 visual grid | 6x3 visual grid | 9,705/10,000 fuel |
| `1m` student | Hidden 192 | Hidden 216 | 967,184/1,000,000 fuel |
| `100m` student | Hidden 1,792 | Hidden 2,048 | 98,058,684/100,000,000 fuel |

These GRU candidates require `USE_HYBRID_SSM_TRANSFORMER=false`.
Teacher GRU width 1,840 fit at 127.50 MiB; 1,848 failed at 128.18 MiB.
GRU 1,824 with predictor hidden 2,048 fit at 127.53 MiB, providing another
capacity allocation to compare. These teacher probes used only three frames.
The listed student configurations passed 1,200 synthetic frames with test
weights, enforcing all supplied caps and their stricter fuel tiers. A 5x4-grid
10K variant used exactly its entire fuel allowance; the 6x3 variant retains
margin. The 100-fuel prior has only one fuel unit left.

Choose larger defaults using retrained held-out trajectory accuracy, including
full-episode ATE, and fresh resource verification of the selected weights.
Resource utilization alone is not an accuracy objective. Teacher size can be
configured with `STUDENT_HIDDEN` and `EMBEDDING_HIDDEN` in a fresh run. Student
architecture defaults remain in `students/model.py::PROFILES`; `--profiles`
selects tiers, not alternate widths or grids. Saved checkpoints carry their own
architecture settings, which must match their weights.

An offline distillation teacher does not need to satisfy submission limits
unless it will also be submitted. Its practical limits are GPU memory and
training cost; the deployed student's own artifact must pass submission checks.

### Teacher capacity selection

For the historical RGB path, ten configurations were exported and bundled using the same VIOBench runtime.
The [capacity search](verification/teacher_capacity_search.json) records their
short resource checks with random weights and nonzero output heads (seed 1701):

| Flow base | GRU width | Upload | Maximum step fuel | Short check |
|---|---:|---:|---:|---|
| 24 | 1,792 | 124.74 MiB | 16.193 G | Pass |
| 32 | 1,792 | 126.07 MiB | 18.837 G | Pass |
| **33** | **1,792** | **126.26 MiB** | **19.514 G** | **Selected** |
| 34 | 1,792 | 126.46 MiB | 19.757 G | Pass, less fuel headroom |
| 35 | 1,792 | 126.66 MiB | Exceeded 20 G | Fail |
| 36 | 1,792 | 126.87 MiB | Exceeded 20 G | Fail |

The selected preset uses about 98.6% of the upload allowance and 97.6% of the
step-fuel allowance. Those are the binding constraints; allocating unused memory
or table entries would not add useful model capacity. The remaining upload margin
is about 1.74 MiB; the measured step-fuel margin is about 0.486 G. Changes to the
runtime, exported code, cadence or configuration require fresh verification.
Mamba has its own verified hybrid preset; other widths/backbones do not inherit the GRU measurements.

Select `gru.json` with RGB mode to retain this base-33/GRU-1,792 preset. New
training and [`full_resolution.json`](full_resolution.json) default to direct
embeddings with hybrid width 896 and predictor hidden width 256. Changing widths
requires fresh weights: warm-starting an old run preserves its saved widths,
and forcing the new widths onto those weights is rejected with a clear error.

### Large teacher full-episode verification

The [large-teacher report](verification/large_teacher_resources.json) records
all 1,200 frames of a synthetic 60-second episode using the selected preset,
random weights and nonzero output heads. Every enforced resource cap passed:

| Measurement | Result | Limit |
|---|---:|---:|
| File size | 126.26 MiB | 128 MiB |
| Initialization fuel | 4.865 G | 200 G |
| Maximum step fuel | 19.514 G | 20 G |
| Maximum observed linear memory | 410 MiB | 2 GiB |
| Initialization wall time | 0.361 seconds | 300 seconds |
| Maximum step wall time | 1.123 seconds | 60 seconds |
| Total measured time | 994.20 seconds (16 min 34 sec) | 1,440 seconds |
| Table limit | Enforced; passed | 1,048,576 elements |

This is a resource profile of test weights, not a trained teacher or an accuracy
measurement. Actual trained weights must be checked again. The report records
both module and runtime SHA-256 hashes to identify the measured artifacts.

### Earlier smaller baseline

The [earlier resource profile](verification/full_resolution_resources.json) used the
640x480/base-24/GRU-96 architecture with randomly initialized, nonzero
output heads. It processed a full synthetic episode, not a benchmark trajectory:

| Measurement | Result |
|---|---:|
| Frames | 1,200, representing 60 simulated seconds |
| File size | 50.67 MiB |
| Initialization fuel | 1.04 G |
| Maximum step fuel | 12.378 G |
| Maximum observed linear memory | 168.625 MiB |
| Maximum step wall time | 0.629 seconds |
| Total measured time | 560.02 seconds |

Wasmtime enforced memory/table/fuel/deadline caps. Memory sampling occurs between
steps; transient allocations remain subject to the enforced cap. This profile
establishes local resource behavior of that architecture/test configuration,
not trained-model accuracy or an unconditional guarantee on other hardware,
configurations, or episode lengths. Verify the selected trained checkpoint and
run real benchmark evaluation before submission.

`submission_limits.json` holds per-call limits for local evaluation.
`verify_limits.py` additionally checks file size and the synthetic episode's total
wall-clock budget. Module SHA-256 in its report ties the measurements to a file.

## Configuration reference

Use [config.py](config.py) for defaults and `--config` for uppercase JSON overrides.
Explicit CLI switches such as `--no-ppo` take precedence over corresponding saved
settings. `--from-checkpoints` retains the source architecture and restores its
remaining settings, while collection coverage/budget, performance controls, frame/state epoch caps (ordinary defaults 200/300),
`USE_PPO` (true), and `WANDB_MODE` (`online`) use current defaults. Explicit JSON
and CLI overrides apply afterward. This prevents old budgets or performance
settings from returning silently.
Explicit JSON epoch settings and `--epochs-frame` / `--epochs-state` still win.
The curriculum JSON selects batches of 64 frame pairs and up to eight state
sequences, frame/state caps of 400/600, teacher cap 60, and distillation cap 100.
Its 3,208-token state budget produces SL batches 8/4/2/1/1 and PPO window batches
8/3/1/1 across their respective horizons. It also applies on checkpoint continuation
and through the timed launcher; explicit CLI options take precedence.
The unused `COLLECT` and `EPISODES_PER_SCENE` settings are removed and ignored
when restoring old training configurations. Collection uses `STREAM_DATA`,
`STREAM_EPISODES_PER_SCENE` and `STREAM_VALIDATION_EPISODES`, or the manual
collector's `--episodes` argument.

| Settings | Meaning/default |
|---|---|
| `WANDB_PROJECT`, `WANDB_MODE` | `hyperdrone-vio-fuel-aware`, `online`; override with `--wandb-mode online/offline/disabled` |
| `H`, `W`, `SOURCE_HEIGHT`, `SOURCE_WIDTH` | Model and collection dimensions; 480x640 |
| `VISUAL_PREDICTION` | `embedding` default; `rgb` restores image prediction |
| `EMBEDDING_HIDDEN`, `EMBEDDING_EMA` | Predictor hidden width 256; target encoder EMA 0.99 |
| `EMBEDDING_STD_TARGET`, `EMBEDDING_VARIANCE_WEIGHT`, `EMBEDDING_COVARIANCE_WEIGHT` | 0.1, 1.0, 0.01; feature regularization |
| `EMBEDDING_PREDICTION_CHUNK` | Up to 1024 missing feature vectors per predictor call; `--embedding-chunk` |
| `FRAME_GENERATOR_BASE` | Flow U-Net base width; 33 |
| `FRAME_INPUT_STRIDE`, `VISUAL_INPUT_STRIDE` | Learned patch stems; both 8 |
| `FRAME_MAX_FLOW_PIXELS` | Full-resolution motion bound; 160 |
| `USE_GENERATIVE_FRAME_MODEL` | In RGB mode: flow generator when true; affine when false |
| `USE_GENERATED_FRAMES_FOR_STATE` | Predict intervening embeddings/RGB when true; hold last real observation when false |
| `USE_LEARNED_VISUAL_ENCODER` | Learned CNN when true; grayscale pooling when false |
| `STUDENT_HIDDEN` | Main recurrent width; fresh GRU 1,792, hybrid 896; explicit override wins |
| `USE_LARGER_STATE_STUDENT`, `LARGE_STATE_HIDDEN` | GRU width selection: true uses 1,792, false uses 32; explicit `STUDENT_HIDDEN` wins; hybrid uses `HYBRID_STUDENT_HIDDEN` |
| `USE_HYBRID_SSM_TRANSFORMER` | True by default: official Mamba-2/local attention; false selects GRU |
| `HYBRID_STUDENT_HIDDEN` | Fresh hybrid width 896 when `STUDENT_HIDDEN` is not explicit; official Mamba-2 only |
| `HYBRID_NUM_MAMBA_LAYERS`, `HYBRID_D_STATE`, `HYBRID_HEAD_DIM`, `HYBRID_ATTENTION_WINDOW` | 3, 16, 32 and 16 |
| `HYBRID_ATTENTION_CHUNK` | 512; bounded query/key blocks with the same causal local receptive field; zero restores dense masking |
| `EMBEDDING_DIM` | 48; optional fresh-SL preset 96; saved architecture metadata preserves the width through export |
| `PARALLEL_TRAINING_PHYSICS` | True for CUDA training; differentiable FP32 prefix calculations; validation/deployment retain sequential recurrence; false restores the training loop |
| `COMPACT_STATE_CAMERAS` | True; only the original selected approximately 20 Hz observations enter state augmentation/IPC |
| `VECTORIZED_PPO_ROLLOUT` | True; fresh current-policy actions, batched physical replay/critic and parallel GAE; false restores the per-tick path |
| `WINDOW`, `BATCH`, `FRAME_PAIR_BATCH` | 101 ticks, four sequences, eight frame pairs |
| `FRAME_GENERATION_CHUNK`, `VISUAL_CHUNK`, `FRAME_CACHE_BATCH` | 16, 16 and 1 |
| `COLLECT_INSTANCES` | Two training flights per environment; validation uses one |
| `USE_AMP`, `AMP_DTYPE` | True, BF16 neural layers on CUDA; CPU/physics/validation/checkpoints FP32 |
| `LOADER_WORKERS`, `FRAME_LOADER_WORKERS`, `LOADER_PREFETCH`, `PIN_MEMORY` | CUDA state workers 2, frame workers 0, one queued minibatch/worker, pinned buffers |
| `PERSISTENT_WORKERS`, `FUSED_ADAMW` | True; reuse same-dataset loader workers, fuse CUDA AdamW; CPU optimizer stays ordinary |
| `CACHE_GENERATED_WINDOWS` | False; rolling mode disables it regardless |
| `EPOCHS_FRAME`, `EPOCHS_STUDENT` | Ordinary maximum 200/300; curriculum preset 400/600; early stopping can shorten them |
| `LR_FRAME`, `LR_STUDENT` | 3e-4 and 1e-4 |
| `STATE_VISUAL_FEATURE_DROPOUT` | RGB-state training visual dropout; 0.05; embedding encoder dropout stays zero |
| `USE_FRAME_EDGE_LOSS`, `USE_FLOW_SMOOTHNESS_LOSS` | Optional generator regularization; false |
| `USE_TEACHER`, `TEACHER_HIDDEN` | Auxiliary teacher/distillation; false, width 256 |
| `STREAM_DATA`, `STREAM_MAX_MIB` | Background collection off by default; 3,072 MiB cap when enabled |
| `STREAM_MAX_PASSES_PER_BATCH` | At most 3 SL epochs/PPO iterations per recording batch while waiting; 1 disables reuse |
| `PATIENCE_FRAME`, `PATIENCE_STUDENT` | 8 and 6 non-improving fresh batches in streaming mode; epochs in fixed-data mode |
| `SCENE_CATALOG`, `NUM_SCENES`, `SCENE_OFFSET`, `STREAM_SCENES` | Mixed ProcTHOR/HSSD, 313 training scenes, offset zero, four scenes per batch |
| `STREAM_EPISODES_PER_SCENE`, `STREAM_VALIDATION_SCENES`, `STREAM_VALIDATION_EPISODES` | Two flights per training scene; 12 held-out scenes, one validation flight each |
| `COLLECT_STEPS`, `STREAM_COLLECT_TIMEOUT` | Up to 2,000 samples / 10 seconds; collection timeout 1,800 seconds |
| `COLLECT_COMPRESSION_LEVEL` | Streaming default/preset 1 for fast lossless NPZ; 6 restores former compression. Standalone `collect.py` retains 6 unless `--compression-level` is supplied. Recording caps still apply. |
| `USE_PPO`, `PPO_ITERATIONS`, `PPO_ROLLOUT_BATCHES` | True, ten iterations, up to four minibatches per rollout collection |
| `PPO_UPDATE_EPOCHS`, `PPO_MINIBATCH`, `PPO_LR` | Four, 256 transitions, 3e-5 |
| `PPO_STD`, `PPO_GAMMA`, `PPO_GAE_LAMBDA` | 0.1, 0.995, 0.95 |
| `PPO_CLIP`, `PPO_TARGET_KL`, `PPO_ANCHOR_COEF` | 0.2, 0.02, 0.01 |
| `PPO_CURRICULUM_WINDOWS` | Empty by default; increasing sample counts enable PPO-only window curriculum; fixed validation uses the longest |
| `PPO_CURRICULUM_MIN_ITERATIONS`, `PPO_CURRICULUM_PATIENCE`, `PPO_CURRICULUM_REL_IMPROVEMENT` | 3 iterations per stage, 2 consecutive qualifying evaluations, >2% improvement from stage entry |
| `USE_CURRICULUM`, `SL_CURRICULUM_WINDOWS` | False and empty by default; full example in `curriculum.json`; main validation stays unaugmented |
| `CURRICULUM_LEVELS`, `CURRICULUM_MIX` | `[0,.5,1]` severity; `[.7,.2,.1]` current/easier/harder mixture |
| `CURRICULUM_MIN_EPOCHS`, `CURRICULUM_PATIENCE`, `CURRICULUM_REL_IMPROVEMENT` | 3 updates, 2 qualifying evaluations, >2% improvement; independent fixed probes for motion/visual/IMU |
| `CURRICULUM_EARLY_STOP_GRACE` | 3 fresh recording batches after each SL promotion; epochs for fixed data; zero keeps the reset without a grace period |
| `CURRICULUM_MAX_STAGE_EPOCHS`, `CURRICULUM_PLATEAU_TOLERANCE` | 6/.1; plateau advancement accepts loss up to 10% above stage best; zero maximum disables this fallback |
| `CURRICULUM_REQUIRE_COMPLETION`, `CURRICULUM_FINAL_STAGE_CYCLES` | True/3; SL patience starts after all difficulty axes and the longest window have been trained, followed by three fresh final-stage batches |
| `CURRICULUM_MIN_FRAME_EPOCHS`, `CURRICULUM_MIN_STATE_EPOCHS` | Optional extra minimum epochs; default 0/0; completion and final-stage exposure replace the preset's former 24/36 guards |
| `CURRICULUM_BATCH_TOKENS` | Default 0; preset 3208 caps effective state batches as horizon increases; SL 8/4/2/1/1, PPO windows 8/3/1/1 |
| `FRAME_PAIRS_UINT8`, `TORCH_CPU_THREADS`, `CURRICULUM_DEVICE_PHOTOMETRIC` | Uint8 sampled-frame transfers, 2 CUDA host threads, photometric transforms on model device |
| `FRAME_VAL_PAIR_BATCH` | 8; hold validation batch statistics fixed when tuning frame training batch size |
| `CURRICULUM_PROBE_BATCHES`, `CURRICULUM_PRIORITIZE` | Two fixed held-out batches per axis; bounded learning-progress priorities enabled |
| `CURRICULUM_ACCEL_NOISE`, `CURRICULUM_GYRO_NOISE` | Extra per-sample standard deviations .14 m/s² and .035 rad/s above native noise |
| `CURRICULUM_ACCEL_BIAS`, `CURRICULUM_GYRO_BIAS` | Extra per-flight bias bounds .1 m/s² and .01 rad/s |
| `CURRICULUM_ACCEL_WALK`, `CURRICULUM_GYRO_WALK` | Extra bias random-walk coefficients .0005 and .0003 per sqrt(second) |
| `CURRICULUM_BRIGHTNESS`, `CURRICULUM_CONTRAST`, `CURRICULUM_BLUR` | Maximum brightness/contrast variation .4; blend toward 3x3 blur .5 |

Changing model width, stem stride, backbone, resolution or synthetic-frame behavior
can change resource use and accuracy. Re-run verification and real evaluation.

## Source-file map

| File | Responsibility |
|---|---|
| `collect.py` | Batched native simulator recording and bounded episode writes |
| `performance.py` | Selective precision, bounded loader settings and epoch throughput logging |
| `data.py` | Recorded datasets, causal selection and embedding/RGB prediction |
| `models.py` | Embedding predictor/EMA encoder, optional RGB generators, GRU, Mamba, auxiliary teacher |
| `embedding_training.py` | Joint-embedding objective, EMA updates and first-stage selection |
| `losses.py` | Supervised state losses |
| `train.py` | SL stages, warm starts, checkpoints, optional PPO, export/verification |
| `streaming.py` | Producer lifecycle, bounded data ownership, fixed validation |
| `curriculum.py` | Independent progress gates, photometric/IMU augmentation, motion mixtures and learning-progress sampling |
| `ppo.py` | Correction-head PPO and deterministic checkpoint selection |
| `export.py` | Reconstruct models, static ONNX export and compatibility checks |
| `validate.py` | Host execution through the official Python guest shim |
| `solution/solution.py` | WASI-compatible inference and camera-frame output |
| `verify_limits.py` | Resource verifier and JSON report |
| `distill.py` | Frozen teacher targets, joint multi-tier student training and per-tier selection |
| `distillation_pipeline.py` | Bounded target prefetch and independent student CUDA streams |
| `students/model.py` | Compact student networks, camera features and differentiable quaternion integration |
| `students/native.c`, `students/export.py` | Native ABI inference, weight embedding, compilation and exact-tier resource gates |
| `students/native_500.c` | Minimal 18-parameter visual/IMU SIMD motion model for the 500-fuel tier |
| `students/native_100.c` | Legacy six-parameter checkpoint export only |
| `tests/` | Numerical, data-lifecycle, training/export and resource-check regressions |

## Current limitations and evaluation needs

The gravity and velocity initialization assumptions can fail for moving starts.
Half-second training windows and two-second recordings do not cover all long-run
drift behaviors. Appearance, calibration and sensor distributions can differ
between the collected data and VIOBench. Sim(3) alignment can also hide scale errors,
so inspect the other pose, twist and failure metrics alongside mean aligned ATE.

No meaningful mean Sim(3) ATE is implied by a resource test or `check: ok`.
Test held-out scenes, motion/sensor conditions and longer trajectories. PPO's
window-validation selection does not replace that evaluation.
