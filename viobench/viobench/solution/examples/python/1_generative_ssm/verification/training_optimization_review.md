# Training and capacity review — 2026-11-13

The changes below optimize future runs. Submission limits apply to deployed WASM;
host training uses the RTX 5080 and is not limited to a 2 GiB CUDA allocation.
Compute probes are not accuracy evaluations or whole-epoch speedup claims.

## Run status and collector repair

Run `20261113-170040-curriculum-ysc92q9_` completed 25 embedding SL epochs, then
stopped because HSSD scene 14 (`103997940_171031257.scene_instance.v7.glb`) made
the native Assimp loader abort. A mesh named `default_shd` contained a triangle
primitive whose index accessor had count zero. State SL and PPO had not started.

`scene_assets.prepare_scene` removes empty primitives from a temporary GLB copy
before native loading, remaps mesh references and preserves binary buffers,
textures, node transforms and all nonempty geometry. Valid assets use their
original path. No scenes are excluded from the mixed catalog, and the cached
source asset is unchanged. Collection logs and manifests record repairs.

[Native repair check](hssd_scene_repair.json): the formerly aborting scene loaded
under OptiX and produced two 41-tick / 11-camera-frame flights. Unit tests verify
BIN preservation and node/mesh remapping.

The saved embedding checkpoint and final local mirror have matching SHA-256:
`982eef1d6dabaa88a025238eebb9af41f70c8a30c4bc9ddaf7052eb7bab965e2`.
The launcher supports continuation into a new directory with `--from-checkpoints`;
it explicitly requests SL before PPO, allowing a missing state checkpoint to be
initialized. This restores model weights; optimizer/curriculum progress restarts.

## Input pipeline and GPU execution

- `COMPACT_STATE_CAMERAS=true` makes training/validation/PPO windows carry only
  their original approximately 20 Hz selected observations. Raw recordings retain
  every native capture. Anchor indices are remapped; observation times, IMU,
  targets and physical spans are unchanged. This reduces state image augmentation,
  worker IPC and pinning work. Direct `StateWindows` callers retain the former
  behavior unless they select `compact_cameras=True`.
- `observed_camera_features` selects observed frames before transfer and
  normalization, encoding at most `VISUAL_CHUNK` images at a time. A 4,001-tick
  [CUDA input probe](state_input_optimizations_cuda.json) used **118.86 MiB** peak
  allocation versus **7,041.63 MiB** before; encoding/transfer time fell from
  **0.2657 s to 0.1101 s**, with maximum embedding difference **zero**. This uses
  synthetic expanded uint8 images and excludes loaders, rendering and state compute.
- Hybrid state training now honors BF16 AMP for neural layers. Validation stays
  FP32. Local attention uses bounded query/key blocks controlled by
  `HYBRID_ATTENTION_CHUNK=512`; zero restores the full dense mask. Observation ages
  use a vectorized clock rather than one GPU launch per tick. Projection weights
  and the 16-token causal receptive field are unchanged.
- `PARALLEL_TRAINING_PHYSICS=true` uses differentiable FP32 prefix products for
  rotations and cumulative sums for velocity/position. This applies to CUDA
  training; validation and deployed inference retain the original sequential
  recurrence. Different FP32 reduction order produces small numerical differences.
  [CUDA parity checks](parallel_physics_parity_cuda.json) measured at most
  **9.14 mm position**, **1.71 mm/s velocity** and **2.48e-5 rotation-entry** error
  in the tested 20-second trajectories; the short gradient check differed by at
  most **9.31e-11**. The long CPU stress test allows 1 cm / 2 mm/s over 24 seconds.
  Set this flag false to restore sequential training integration.

[Synthetic hybrid CUDA probes](parallel_physics_cuda.json), hidden 896, with
precomputed embeddings and forward/backward/AdamW:

| Window | Batch | FP32 sequential physics + dense attention | BF16 neural + bounded attention + parallel FP32 physics |
|---|---:|---:|---:|
| 401 ticks / 2 s | 2 | 0.4120 s/step | 0.01125 s/step |
| 1,601 ticks / 8 s | 2 | 1.5805 s/step | 0.01480 s/step |
| 4,001 ticks / 20 s | 1 | 4.1211 s/step | 0.01579 s/step |

The corresponding isolated compute ratios are 36.6×, 106.8× and 261.0×. Full
epochs also include augmentation, transfers, fixed long-horizon validation,
curriculum probes and fresh-data waits; those costs were excluded here.

## PPO and batch decisions

`VECTORIZED_PPO_ROLLOUT=true` samples new current-policy actions across a window,
replays them with parallel FP32 physics, batches pointwise critic evaluations and
uses a reverse prefix calculation for GAE. The actor depends only on frozen
causal tokens, so this batching does not introduce action/observation feedback or
privileged truth into actor inputs. The final token remains reserved for
bootstrapping. Hybrid frozen-token extraction bypasses unused pose integration.
Setting the flag false restores the per-tick rollout/GAE path.

Tests replay the recorded actions through the original physical step, compare
pre-action critic inputs/rewards/bootstrap/returns, verify actor independence
from privileged truth, and check policy changes on reused recordings.
[CUDA rollout probe](ppo_rollout_optimizations_cuda.json), using precomputed
896-wide tokens and excluding image encoding, backbone inference and optimization:

| Window | Per-tick rollout / GAE | Vectorized rollout / GAE |
|---|---:|---:|
| 401 ticks | 0.3193 s | 0.00236 s |
| 4,001 ticks | 8.6061 s | 0.00689 s |

Keep the curriculum's embedding batch **64** and state batches **8/4/2/1/1**.
The [batch/capacity GPU sweep](training_optimizations_cuda.json) measured only a
2.3% embedding-compute throughput increase at batch 128 versus 64, while peak
allocation doubled from about 4,060 to 8,102 MiB. No larger default batch was
selected based on this small compute-only gain. Validation pair batches remain 8.

## Wider embeddings and submission limits

`EMBEDDING_DIM` is now configurable through training, EMA prediction, state and
teacher fusion, checkpoint metadata, ONNX and the NumPy/WASM guest. Defaults and
legacy metadata retain **48**. Non-48 widths require the learned visual encoder;
changing width requires a fresh SL run rather than loading incompatible weights.
`curriculum_embedding96.json` selects the opt-in **96-dimensional** candidate with
the same backbone and larger-batch curriculum settings. Its predictor compute
cost was close to the 48-dimensional baseline in the isolated GPU probe.

[96-dimensional resource report](embedding96_resources.json): untrained seed-1701
weights, 896-wide three-layer hybrid, full 640x480 input, **1,200-frame synthetic
episode / 60 simulated seconds**, official WASM sandbox:

| Resource | Measured | User limit |
|---|---:|---:|
| Upload | 118.95 MiB | 128 MiB |
| Total episode wall time, including initialization | 669.57 s / 11.16 min | 1,440 s / 24 min |
| Init fuel | 4.425 G | 200 G |
| Maximum step fuel | 8.646 G | 20 G |
| Maximum observed linear memory | 328.125 MiB | 2,048 MiB |
| Init wall time | 0.323 s | 300 s |
| Maximum step wall time | 0.565 s | 60 s |
| Table elements | Sandbox cap enforced; passed | 1,048,576 |

The report includes the module SHA-256 and exact cap values. The deployed Python
source matches the verified artifact; training-only execution changes do not add
inference weights or change deployed integration. This establishes resource
feasibility for the tested episode, not an accuracy gain or a guarantee for every
possible episode. Verify each final trained submission too. Compare widths using
held-out trajectory metrics and validation improvement per hour; embedding losses
alone do not establish that a wider representation is better.

## Running and reproducing

From an activated environment in the model directory, inside tmux, reuse the saved
48-dimensional embedding checkpoint and train SL then PPO into a new run:

```bash
python -u run_until_11.py --local-root /var/tmp/hyperdrone-eecs116-aaq --from-checkpoints checkpoints/20261113-170040-curriculum-ysc92q9_
```

For a fresh 96-dimensional trial:

```bash
python -u run_until_11.py --local-root /var/tmp/hyperdrone-eecs116-aaq --config curriculum_embedding96.json
```

Neither command has a default time cutoff. The launcher retains checkpoints and
mirrors completed files on exit. A continuation was briefly launched in tmux
session `hyperdrone-optimized`, creating run `20261113-185559-curriculum-t820hs_g`.
At the user's request it was interrupted at 18:58 Pacific on November 13, 2026,
during initial sample collection, before any training update. The launcher
completed its backup of logs/configs; the original embedding checkpoint remains
unchanged. Training is stopped and must remain stopped until the user requests
a launch.

Validation completed with **241 passed / 14 skipped** in the full sandbox suite
and **77 passed** in the affected CUDA suite. These suites overlap and should
not be added as independent test counts. Coverage includes 48/96-dimensional SL,
teacher/distillation/PPO flows, native export, frozen-backbone PPO updates,
action replay, bootstrapping, actor truth isolation, causal masks and gradients,
camera/target equivalence, scene repair and launcher continuation/mirroring.
The preserved real embedding checkpoint loaded strictly into the updated model.
A configuration-only CUDA continuation check selected SL + PPO, width 48,
batch 64/8, all new performance controls and compression level 1, without starting
training or collection.

The GPU probes and numerical check are reproducible with
`verification/benchmark_training_optimizations.py`, `benchmark_state_inputs.py`,
`benchmark_ppo_rollouts.py` and `verify_parallel_physics.py`. Each accepts an
`--output` path. `review_embedding_capacity.py --work-dir NEW_DIR --report JSON`
exports and verifies a capacity candidate; its default is dimension 96 and
1,200 frames. `check_scene_repair.py SCENE --report JSON` checks the scene fix;
use the activated environment and `HYPERLUX_BACKEND=OPTIX` on this GPU machine.
