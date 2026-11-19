# Recovered LR sweep results — stopped by user

**No GPU was used for recovery. No sweep or training process was restarted.**
The saved per-study launchers and queued-boundary launcher have been disabled.

## Recovered coverage

- 477 completed trials from the interrupted extension; all have complete numerical records.
- 190 earlier state trials plus five teacher embedding trials: 672 completed trials across both studies.
- State LR results: 2, 4, 6, 8, 10 and 20 seconds for all nine state models (54/90 model-window settings).
- Separate student embedding-group results: 2, 4, 6 and 8 seconds for all eight students (32/80 settings).
- The 10/20-second state results were recovered from the earlier study; they were not rerun.
- No results exist for 12/14/16/18 seconds. The new 10-second embedding sweep never started.
- The planned higher/lower embedding-rate boundary checks never started.

An em dash means **not tested**, not an interpolated or recommended rate.
All values below are best tested pilot candidates, not proven optimal learning rates.

## State learning rates

| Model | 2 s | 4 s | 6 s | 8 s | 10 s | 12 s | 14 s | 16 s | 18 s | 20 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 0.0003 | 0.0003 | 0.001 | 0.001 | 0.0001 | — | — | — | — | 0.0003 |
| 10k | 0.0003 | 0.0003 | 0.001 | 0.001 | 1e-05 | — | — | — | — | 1e-05 |
| 100k | 3e-05 | 0.001 | 0.0003 | 0.0003 | 3e-05 | — | — | — | — | 0.0003 |
| 1m | 1e-05 | 3e-05 | 1e-05 | 1e-05 | 3e-06 | — | — | — | — | 0.0003 |
| 1m_ssm | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | — | — | — | — | 0.0003 |
| 1m_hybrid | 3e-06 | 3e-06 | 3e-06 | 3e-06 | 3e-06 | — | — | — | — | 0.0003 |
| 100m | 0.001 | 0.0003 | 1e-05 | 0.0003 | 0.0003 | — | — | — | — | 0.001 |
| 100m_hybrid | 0.0001 | 1e-05 | 0.001 | 1e-05 | 0.0003 | — | — | — | — | 0.001 |
| teacher | 1e-06 | 1e-06 | 3e-05 | 1e-05 | 1e-06 | — | — | — | — | 0.0001 |

## Embedding-group LR ranked by state loss

The group includes the online visual encoder and causal embedding predictor. State LR is held at the value above.

| Model | 2 s | 4 s | 6 s | 8 s | 10 s | 12 s | 14 s | 16 s | 18 s | 20 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 0.001 | 3e-05 | 0.0003 | 0.0001 | — | — | — | — | — | — |
| 10k | 0.001 | 1e-05 | 0.0001 | 1e-05 | — | — | — | — | — | — |
| 100k | 3e-05 | 0.001 | 1e-05 | 0.0003 | — | — | — | — | — | — |
| 1m | 1e-05 | 1e-05 | 1e-05 | 1e-05 | — | — | — | — | — | — |
| 1m_ssm | 0.001 | 1e-05 | 1e-05 | 1e-05 | — | — | — | — | — | — |
| 1m_hybrid | 3e-06 | 3e-06 | 3e-06 | 3e-06 | — | — | — | — | — | — |
| 100m | 0.0001 | 0.0001 | 1e-05 | 1e-05 | — | — | — | — | — | — |
| 100m_hybrid | 0.001 | 3e-05 | 0.001 | 1e-05 | — | — | — | — | — | — |

## Embedding-group LR ranked by JEPA loss

| Model | 2 s | 4 s | 6 s | 8 s | 10 s | 12 s | 14 s | 16 s | 18 s | 20 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 0.001 | 0.001 | 0.001 | 0.001 | — | — | — | — | — | — |
| 10k | 0.001 | 0.001 | 0.001 | 0.001 | — | — | — | — | — | — |
| 100k | 0.001 | 0.001 | 0.0003 | 1e-05 | — | — | — | — | — | — |
| 1m | 0.001 | 0.001 | 3e-05 | 0.0001 | — | — | — | — | — | — |
| 1m_ssm | 0.001 | 0.001 | 0.0003 | 3e-05 | — | — | — | — | — | — |
| 1m_hybrid | 0.001 | 3e-06 | 3e-06 | 3e-06 | — | — | — | — | — | — |
| 100m | 0.001 | 0.0001 | 0.001 | 3e-05 | — | — | — | — | — | — |
| 100m_hybrid | 0.001 | 3e-05 | 0.0003 | 0.001 | — | — | — | — | — | — |

**Teacher embedding LR: 3e-4**, from its earlier frame-pair sweep. That stage has no trajectory-window parameter.

## Method and limitations

- Rates rank the average normalized tail validation loss across fresh seeds 17 and 29.
- Seed 17 used 512 updates; seed 29 used 1,024, matched within each comparison. Teacher embedding used a separate 256-update pilot.
- Validation used eight complete 20-second flights; training used 16 flights from eight different ProcTHOR scenes.
- Existing 2/10/20-second candidate pools differ from the added 4/6/8-second screens. These are shortlisted, bounded searches.
- Student embedding sweeps retain state + 0.1 JEPA + 0.1 alignment, holding state LR fixed. This is not an exhaustive two-LR grid.
- State and JEPA losses can prefer different rates. Several embedding winners are at the highest tested rate (1e-3), without the planned extension.
- The known mid-flight position/velocity target inconsistency was not fixed. Production defaults were not changed.
- These are tuning-validation results, not new independent-test or website-score improvements.

## Stop and storage incident

The last durable progress record was preparation of the 10-second training cache. The shell recorded exit code 137;
that alone cannot distinguish a manual kill from an OOM kill. The user reports force-stopping the process after a freeze.
`/tmp` is a 32 GiB RAM-backed filesystem here, so recordings, caches and trial checkpoints there also consumed host memory.
At the start of recovery `/tmp` was already almost empty (32 GiB free), and about 58 GiB of RAM was available.
No remaining sweep/worker/queued-boundary processes or tmux sessions were found.
Future execution needs bounded storage/RAM usage and disk-backed scratch; the sweep was not restarted or repaired during recovery.

## What survived

- [Recovered structured results](recovered_lr_results.json).
- [Recovered LR table CSV](recovered_lr_by_window.csv), with explicit tested/not-run statuses.
- [All 672 trial summaries](recovered_lr_trials.csv), including the five teacher-embedding trials and retained checkpoint locations.
- [Original combined trial records](checkpoints/lr-windows-20261115-211044/all_trials.json).
- [Teacher embedding trial records](checkpoints/lr-sweep-20261115-200326/teacher_embedding_trials.json).
- [Earlier independent-scene comparison](checkpoints/lr-sweep-20261115-200326/confirmation/independent_check.json).
- [Retained confirmed selected pilot checkpoints](checkpoints/lr-sweep-20261115-200326/confirmation/selected/).
- New trial weights and raw recordings in `/tmp` were already gone; the scalar results survive, but those missing weights cannot be recovered from logs.

## Cleanup

- Removed/deduplicated preliminary checkpoints and compacted verified logs: 147.7 MiB logical space reclaimed.
- Filesystem-reported allocated space reclaimed: 113.6 MiB.
- Confirmed selected pilot checkpoints and their architecture JSON files were retained. Main production checkpoints were untouched.
- Compressed logs have `.gz` suffixes and were decompressed/hash-verified before their originals were removed.
- Original per-study launchers were archived as `.disabled-original.gz`; active copies now exit without starting work.
- [Exact cleanup manifest](lr_sweep_cleanup_manifest.json).

## Existing W&B runs

- [Interrupted window study](https://forge.coreweave.com/wandb/ysaxena_personal/hyperdrone-vio-fuel-aware/runs/86bc54pq).
- [Earlier confirmation study](https://forge.coreweave.com/wandb/ysaxena_personal/hyperdrone-vio-fuel-aware/runs/xoa55tlu).
Recovery used local records only; no W&B run was restarted or remotely modified.
