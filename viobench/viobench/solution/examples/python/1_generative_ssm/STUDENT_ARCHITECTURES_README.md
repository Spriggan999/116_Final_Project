# Teacher-style budget student architectures

New student defaults now have three learned components: a visual encoder, a
causal IMU-conditioned embedding forecaster, and their original state-prediction
backbone. All eight fit their fuel caps in 1,200-frame (60-second) actual WASM
resource checks with nonzero perturbed weights. These are resource/correctness
checks, not trained accuracy results. The new models need fresh training.

The teacher and students differ in cadence. The teacher supplies embeddings and
updates its hybrid at 200 Hz, predicting between observed captures. The students
retain their existing **20 Hz neural updates** to preserve their fuel budgets.
Their forecaster predicts the current camera-time embedding from the previous
two encoded captures, the IMU packet ending now, and elapsed times. The state
model receives the observed embedding, its innovation relative to that forecast,
IMU and timing features. Neither encoder nor forecaster reads future captures.
Prediction is entirely in embedding space; no RGB reconstruction is performed.

## Comparison

| Model | Visual input and encoder | Embedding | Forecaster MLP | State predictor | Deployed parameters | Measured max fuel/frame / cap |
|---|---|---:|---|---|---:|---:|
| Teacher | 640×480 RGB; CNN channels 16→32→48→64; adaptive 3×4 pooling | 48 | 152→256→48, tanh | 3 official Mamba-2 blocks, width 896, state 16, head dimension 32; 4-head local attention, window 16 | 18,182,369 | 8,619,513,401 / 20,000,000,000 |
| 1K | 4×4 gray grid; 2×2 stride-2 convolution with 2 filters, leaky ReLU, mean pool + projection | 2 | 14→2→2, tanh | Linear 9D correction head; normalized quaternion/IMU integration; no SSM | 169 | 5,195 / 6,000 |
| 10K | 6×3 gray grid; 3×3 stride-2 convolution, channels 1→2; 1×1 convolution, channels 2→4; mean pool + projection | 4 | 20→4→4, tanh | Linear 9D correction head; no recurrent/SSM block | 309 | 7,102 / 10,000 |
| 100K | 8×6 gray grid; two 3×3 stride-2 convolutions, channels 1→2→4; mean pool + projection | 8 | 32→8→8, tanh | Single tanh RNN, hidden 48; full-IMU physical integration | 4,417 | 84,184 / 100,000 |
| 1M SSM | 8×6 gray grid; two 3×3 stride-2 convolutions, channels 1→4→8; mean pool + projection | 16 | 56→32→16, tanh | 1 official Mamba-2 block, width 64, state 8, head dimension 16; no attention | 33,153 | 535,267 / 1,000,000 |
| 1M RNN | Same 8×6 / 4→8 encoder | 16 | 56→32→16, tanh | Original single tanh RNN, hidden 192 | 49,305 | 694,090 / 1,000,000 |
| 1M Hybrid | Same 8×6 / 4→8 encoder | 16 | 56→32→16, tanh | 1 official Mamba-2 block, width 64, state 8, head dimension 16; 2-head attention, window 8 | 49,921 | 797,719 / 1,000,000 |
| 100M RNN | 32×24 gray grid; two 3×3 stride-2 convolutions, channels 1→16→32; mean pool + projection | 32 | 104→128→32, tanh | Original single tanh RNN, hidden 1,792 | 3,381,641 | 47,236,437 / 100,000,000 |
| 100M Hybrid | Same 32×24 / 16→32 encoder | 32 | 104→128→32, tanh | 3 official Mamba-2 blocks, width 384, state 16, head dimension 32; 4-head attention, window 16 | 3,382,593 | 53,128,559 / 100,000,000 |

The 1K CNN uses leaky ReLU after its convolution and embedding projection.
Larger student CNNs use ReLU between convolutions and tanh after the embedding
projection. Gray grids are sparse samples at pixel centres from the ABI's
320×240 RGB frame; the incoming 640×480 training captures use the same area
downsampling/rounding first. These are small learned image encoders, not full
resolution CNNs. Temporal backbone types, widths, layer counts, SSM state sizes
and attention windows were preserved. A larger width in one plain RNN does not
make it larger than the teacher's three-block hybrid: compare total parameters.

The teacher parameter count excludes its training-only target/EMA encoder. Its
fuel figure comes from the existing exported teacher's 1,200-frame check; student
figures come from the new frontend resource checks. Actual trained exports are
metered again before promotion. All student checks completed inside the 128 MiB
upload, 2 GiB memory, table, init/step fuel, and wall-time caps. The largest new
student WASMs are about 12.94 MiB, with 64 MiB observed linear memory in these
checks. Those observations are not theoretical maxima for arbitrary inputs.

## Budget limitations

All three blocks fit their current export budgets in the reduced forms above.
The smallest tier retains the profile name `1k`, but its nine-correction version
now has a **6,000 fuel/frame export cap**. Its 169 parameters include the same
12→9 linear state head pattern and bounded corrections used by larger students.
The nine outputs are gyro corrections (3), body-frame acceleration corrections
(3), and initial-frame acceleration corrections (3). It uses the six-channel
IMU packet mean and the same quaternion/physical integration equations as the
10K model. The 1,200-frame resource check measured **5,195 fuel/frame**; this
version is no longer a literal 1K-fuel submission. The previous 82-parameter
six-output orientation model, 64-parameter fixed-attitude 1K model and 26-parameter
scalar 500 model retain their saved architectures, weights and export caps.
Resource checks do not establish learned accuracy.
The 10K model has a small CNN/forecaster but retains its nonrecurrent state head.
The RNN tiers remain RNNs; only the existing Mamba tiers use official Mamba-2.

The 1K and 10K profiles integrate the packet mean at 20 Hz. Other students integrate each 200 Hz IMU row using the correction
held from that frame's state update. Reproducing the teacher's complete 200 Hz
neural architecture is outside this implementation; these camera-rate designs
are the fuel-verified alternatives.

## Training, exports and compatibility

Fresh runs train end to end with ground-truth state supervision plus a default
**0.1-weight JEPA-style embedding objective** and **0.1-weight Sim(3) trajectory
alignment loss** over the current horizon. A stop-gradient EMA target encoder
provides prediction/content targets; variance and covariance penalties discourage
collapsed/redundant features. EMA targets are training-only buffers, saved for
recovery/stage restoration and omitted from deployment weights and fuel counts.
No pretrained VIO teacher is required. Set `--embedding-weight 0` to disable the
auxiliary objective. See [all loss functions and coefficients](LOSSES_README.md).

The scaled forecaster uses the teacher's residual form
`anchor + 0.25 * bounded(MLP(inputs)) * clamp(elapsed / 0.05, 0, 2)`.
The current 1K tier uses tanh for its forecaster and nine-correction state head. Camera-time innovation is
`observed_embedding - forecast_embedding`. Other state models consume
`[observed_embedding, innovation, normalized_IMU, elapsed, observed_flag]`.
The first-frame IMU feature is zero to match the recorder's initial placeholder.

`students/model.py` defines eight default `PROFILES`, starting with `1k`.
`COMPAT_PROFILES['500']` retains the older 26-parameter scalar pipeline and
`LEGACY_PROFILES` preserves pre-frontend layouts for compatibility tests.
Saved older `student.json` settings
without `teacher_style: true` still construct and export their original model.
New frontends do not silently accept old weights into new shapes. Use the normal
fresh launcher to train them:

```bash
bash train_students_sequential.sh
```

The measured order is now 1K → 10K → 100K → 1M SSM → 1M RNN → 1M Hybrid → 100M RNN →
100M Hybrid. The ProcTHOR-only split and ten-level plateau curriculum are retained.
W&B records the full architecture settings and parameter counts per run.

The native frontend is in `students/native_vision.h`; the current 1K uses the
same `students/native.c` nine-correction kernel as larger students. The earlier
six-output attitude model uses `students/native_1k_orientation.c`; fixed-attitude
1K and compatible 500 pipelines use `students/native_500_frontend.c`.
Older 18-parameter 500 checkpoints still use `native_500.c`.
The official Mamba-2 native recurrence and attention implementation is unchanged.
Checks compare complete PyTorch trajectories with both host-native C and actual
WASM, with nontrivial camera extrinsics/lever arms and cache-reset replay. CUDA
checks include official Mamba-2, full-pipeline gradient flow, causal prefix
invariance, and serial/parallel optimizer agreement. See
`verification/student_frontend_review.json` for source hashes and measurements.

## Inputs and outputs

Each WASM call receives a **320×240 RGB frame**, a timestamp, and an IMU packet
containing ten rows at 200 Hz: `[timestamp, ax, ay, az, gx, gy, gz]`. The 20 Hz
camera frames are converted to grayscale and sampled on the grids above.
All current student neural state models consume the six-channel packet mean.
The 1K and 10K integrators also use that mean;
100K and larger models integrate every individual IMU row with the frame's
predicted correction. Neural updates remain at 20 Hz for all student tiers.

The learned encoder outputs E values (the table's embedding dimension). The
embedding forecaster consumes two previous embeddings, their difference, IMU
conditioning and two elapsed times; it predicts E values, never an RGB image.
Its input is `3E+8` for every current student. State features are current embedding, embedding innovation,
six normalized IMU channels and two timing/observation fields: `2E+8` values.
The 1K uses the same state-feature layout and nine-correction output convention.

| Tier | Encoder output E | Forecaster input → output | State input → internal output | Exported output |
|---|---:|---|---|---|
| 1K | 2 | 14 → 2 | 12 → 9 corrections | 13D pose/twist with changing orientation |
| 10K | 4 | 20 → 4 | 16 → 9 IMU/acceleration corrections | 13D pose/twist |
| 100K | 8 | 32 → 8 | 24 → RNN 48 → 9 corrections | 13D pose/twist |
| 1M SSM | 16 | 56 → 16 | 40 → Mamba-2 64 → 9 corrections | 13D pose/twist |
| 1M RNN | 16 | 56 → 16 | 40 → RNN 192 → 9 corrections | 13D pose/twist |
| 1M Hybrid | 16 | 56 → 16 | 40 → Mamba-2 64 + attention → 9 corrections | 13D pose/twist |
| 100M RNN | 32 | 104 → 32 | 72 → RNN 1,792 → 9 corrections | 13D pose/twist |
| 100M Hybrid | 32 | 104 → 32 | 72 → 3× Mamba-2 384 + attention → 9 corrections | 13D pose/twist |

The common 13D ABI result is position (3, metres), orientation quaternion
`qx,qy,qz,qw` (4), linear velocity (3, m/s), and angular velocity (3, rad/s).
Pose is relative to the initial camera frame; twists use the current camera's
axes. Training returns corresponding position/velocity/rotation-matrix/angular
velocity tensors, with conversion/calibration handled by the native exporter.
