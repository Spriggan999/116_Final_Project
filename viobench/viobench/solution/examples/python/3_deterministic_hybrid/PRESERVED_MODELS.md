# Preserved lite confidence and normal translation-scale models

These are the original trained checkpoints, prior to the later diverse/streaming
training experiment. That experiment's candidate checkpoints were removed.

| Model | Checkpoint | Architecture | Parameters |
|---|---|---|---:|
| Lite confidence | `lite/models/measurement_confidence/checkpoint.pt` | 8 → 4 ReLU → 2 | 46 |
| Normal translation scale | `models/translation_scale/checkpoint.pt` | 8 → 8 ReLU → 1 | 81 |

Each directory includes its PyTorch state dictionary, normalization statistics,
exported `weights.h`, WASM C++ wrapper, and `training.json` provenance. The C++
wrappers use the corresponding shared estimator and framework-free inference
header. Baseline snapshots retain the exact sources used in training.
`train.py` defines the PyTorch architectures and export utilities.

From `viobench/viobench/solution/examples/python`, in the benchmark environment:

```bash
python -u 3_deterministic_hybrid/benchmark.py --profile lite \
  --models solution_lite measurement_confidence \
  --out /tmp/lite_confidence_public

python -u 3_deterministic_hybrid/benchmark.py --profile full \
  --models 2_deterministic translation_scale \
  --out /tmp/normal_scale_public
```

These commands compile the selected model and its unmodified deterministic
baseline and evaluate all ten full public episodes. Downloads default to local
scratch and existing home-cache files are reused. Results include unaligned mean
position error, unaligned ATE RMSE, Mean Sim(3) ATE and maximum/mean fuel per frame.
Use explicit model selections: the other experimental model weights are not
part of this two-model checkpoint commit. `prepare.py` and `lite_profile.py`
generate wrappers for the broader experiment, but evaluation of the selected
models needs only their included weight headers.

The measured public-set results for these preserved checkpoints were:

| Estimator | Mean unaligned error (m) | Mean Sim(3) ATE (m) | Max fuel/frame |
|---|---:|---:|---:|
| Original lite | 1.88980 | 0.68198 | 44,800 |
| Lite confidence | 1.71438 | 0.62005 | 44,800 |
| Original normal | 1.18886 | 0.46748 | 99,263 |
| Normal translation scale | 1.72164 | 0.40641 | 99,572 |

The scale model improves aligned ATE while worsening unaligned error. These
measurements do not establish performance on unseen scenes.
