# CLAUDE.md: 5_novel_approaches

Notes for agents working in this folder. Read `README.md` here for results; this file is how to work here.

## What is here

| Folder | What it is |
|---|---|
| `vins/` | VINS-Mono-style sliding-window optimizer (keyframe window, preintegration, Schur'ed inverse-depth features, two-way marginalization) + relocalization map (`vmap.inc`). Defaults = the chosen build (`wasm/vins.wasm`). `build_host.sh` makes `host_vins_NAME`. |
| `router_vins/` | Router with vins in place of the MSCKF (experiment; not better than vins alone). |
| `msckf/` | From-scratch OpenVINS-style MSCKF in one file (`solution.cpp`), optional Gaussian-splat map (`-DSPLAT=1`, `splat_map.inc`). Every setting is a `-D` flag at the top of the file; defaults reproduce the `msckf_v2` build. |
| `router/` | Runs the MSCKF (+ splat map) and learned_lin side by side and blends their position steps by the MSCKF's running speed. `router_lite` = `-DPR=3 -DMAX_ROWS=20 -DDETECT_CELLS=16`. The current recommended submission. |
| `learned/` | learned_lin source (v9_adapted + 21 learned noise multipliers). The router compiles it as its second estimator, so it must stay. `LEARN=0` reproduces v9_adapted exactly. |
| `msckf_splat/` | Development copy and write-up of the splat map on the MSCKF. |
| `splat/` | learned_lin + Gaussian-splat map (separate, cheaper model). |
| `lqg/` | Fixed-gain LQG observer experiments (not competitive). |
| `hosttools/hosteval.py` | Paired host-replay evaluation against a reference binary, cached per binary hash. |
| `wasm/` | Built binaries (gitignored; force-add with `git add -f`). |

## Build and evaluate

```bash
cd router && ./build_wasm.sh OUT.wasm -DPR=3 -DMAX_ROWS=20 -DDETECT_CELLS=16   # router_lite
./build_host.sh NAME [-D...]           # host binary /tmp/learned-<uid>/bin/host_msckf_NAME (MSCKF flags apply)
cd ../msckf && ./build_host.sh NAME [-D...]                                     # MSCKF alone
python3 ../hosttools/hosteval.py /tmp/learned-$(id -u)/bin/host_msckf_NAME --ref /tmp/learned-$(id -u)/bin/host_msckf_REF --sets dev
python ../msckf/sandbox_eval.py OUT.wasm EP1.zip EP2.zip ...                    # official runner: ATE + fuel per frame
```

- Host replay gives exactly the sandbox's ATE (about 0.5 s per flight). Use it for accuracy; use the sandbox only for fuel.
- The sandbox's Python is the conda env `/home/cc/eecs116/fa26/class/eecs116-aaq/.conda/envs/myenv/bin/python`.
- Wasm toolchain: `../../../build/toolchain/wasi-sdk/bin/clang++` (the build scripts find it); `-O3 -msimd128`.

## Data

- Replay feeds (green channel only): `/tmp/learned-<uid>/feeds/<set>/<flight>/`. `/tmp` is a RAM disk: wiped on reboot, about 30 GB.
  Fills up; delete sets you no longer need before generating more.
- Episode zips that survive reboots: `/var/tmp/novel-approaches-<uid>/episodes/<set>/`. Make feeds with
  `learned/tools/make_feeds.py OUT_DIR episode.zip ...`.
- Flight generators live outside the repo (`/var/tmp/novel-approaches-<uid>/private/`); one house per simulate call,
  run under `systemd-run --user` so a closed session does not kill them.
- `hosteval.py` groups: `dev` (agile-hard/xfast-dev, normal-dev) and the `train-*`, `normal-dev2`, `bias3-*` sets for
  selection; `test` (agile-*-test, fresh-*, heldout, test2-*) for reporting.

## Rules that matter

- Choose settings on selection sets only; score the `test` group once per candidate. Differences under about 0.05 m
  on 144 flights are noise, and the best of several variants usually regresses toward the mean on new flights.
- Compare paired, flight by flight (`--ref`), never just means.
- Judge fuel by the worst frame (the score uses the max per frame over all episodes), measured in the sandbox.
- Keep notes about the course's hidden test set and leaderboard comparisons out of tracked files.
- Commit or push only when asked; stage only the files of the task at hand.

## Gotchas

- New behaviour goes behind a `-D` flag whose default keeps old results; check with a paired eval that the default is
  bit-identical.
- Sandbox output folders are named by episode stem, and the hard and xfast test sets reuse stems: use separate `--out`
  folders per set when you need per-episode fuel files.
- Never edit a shell script that is running; bash keeps reading the old file. Stop and restart it.
- The router links the two models as separate translation units with renamed entry points (`router/msckf_part.cpp`,
  `router/lin_part.cpp`); learned_lin needs explicit `extern "C"` declarations there.
- Ideas already tested and rejected are listed in `msckf/README.md`, `router/README.md` and `README.md`; check
  before repeating one.
