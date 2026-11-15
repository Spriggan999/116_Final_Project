> Historical timing snapshot: the old 100-fuel profile has since been replaced by the 500-fuel visual/IMU student. Its runtime was not benchmarked.

# Conditional timing projections

Measured with a real frozen teacher and recordings. Epoch budgets are not predictions of convergence.
Compute projections exclude CPU loading, validation preprocessing and initial collection/setup.
Assumptions: 8 training recordings, 12 validation recordings per epoch; collection allowance 0 s/epoch.

| Profile | Measured max fuel/frame | Solo 100 epochs (h) | Solo 300 epochs (h) | Solo 500 epochs (h) |
|---|---:|---:|---:|---:|
| 100 | 94 | 0.17 | 0.19 | 0.22 |
| 10k | 9,705 | 0.18 | 0.21 | 0.24 |
| 1m | 803,982 | 0.23 | 0.28 | 0.33 |
| 100m | 80,074,943 | 0.23 | 0.28 | 0.33 |
| 1m_ssm | 526,898 | 0.20 | 0.23 | 0.26 |
| 1m_hybrid | 789,305 | 0.20 | 0.23 | 0.26 |
| 100m_hybrid | 56,390,451 | 0.21 | 0.25 | 0.29 |

Shared-teacher portfolio: 0.40 / 0.48 / 0.57 hours for 100 / 300 / 500 epochs with all seven active.
Earliest stale-stop scenario after the curriculum: 0.37–0.38 h, assuming no further qualifying improvements.

Two concurrent full 20-second HSSD flights took **29.82 s** at compression level 1, producing 540.1 MiB. Scaling that scene to eight recordings gives **119.3 s per fresh batch**. With at most three passes, ideal-overlap streaming projects **1.24 / 3.45 / 5.66 h** for 100 / 300 / 500 epochs.

The no-further-improvement scenario reaches eight stale fresh batches after about **39.9–39.9 minutes**, plus initial validation collection, dataset decoding and other setup. This is conditional, not a convergence prediction. Mixed scenes may generate faster or slower. Rendering and training on the same GPU can interfere.

The production-loop smoke test processed all seven students at **30.01 shared windows/s** in its second 401-tick epoch, including CPU loaders and one complete validation recording. Its first epoch included compilation/autotuning startup and was slower. Do not substitute this short-window throughput for long-window timing.

Raw recordings default to an 8 GiB bounded ring under `/tmp`; the cap is configurable. Initial native probes and one-epoch trained exports all passed their fuel tiers. Pilot checkpoints are provisional. The eight-model evaluation smoke used only 20 camera frames and is not an accuracy comparison.
