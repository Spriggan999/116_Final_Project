# Remote access and training monitoring with SSH

Use SSH to control the lab computer from your personal computer, and `tmux`
to keep training running when your SSH connection closes.

The lab computer is **`c204-17.eecs.berkeley.edu`** and the instructional
account is **`eecs116-aaq`**. SSH was confirmed listening and `tmux` was
installed on this machine on November 12, 2026. Authentication and connectivity
from your personal computer have not been tested here.

## Connect from your personal computer

Connect to Berkeley's **bSecure VPN** before connecting from off campus.
See the official [EECS instructional connection instructions](https://inst.eecs.berkeley.edu/connecting.html).

In Terminal or PowerShell on your personal computer:

```bash
ssh eecs116-aaq@c204-17.eecs.berkeley.edu
```

Use your instructional-account credentials. Connect to this specific lab
computer so you reach its GPU and existing `tmux` sessions.

## Start or reconnect to a persistent training session

After SSH login, attach to the overnight session or create it if needed:

```bash
tmux new -A -s teacher-overnight
```

In a new session, activate the environment and enter the model directory:

```bash
eval "$(conda shell.bash hook)"
conda activate /home/cc/eecs116/fa26/class/eecs116-aaq/.conda/envs/myenv
cd /home/cc/eecs116/fa26/class/eecs116-aaq/Documents/project_3/viobench/viobench/solution/examples/python/1_generative_ssm
```

Start your [training command](README.md#3-recommended-train-export-bundle-and-verify)
inside this session. The same applies to a training command wrapped in
`timeout` for a scheduled cutoff. Start the run inside `tmux` from the beginning;
creating a session later does not move an existing process into it.

## Fresh curriculum run with automatic checkpoint copying

Inside tmux, after activating `myenv` and entering the model directory:

```bash
python -u run_until_11.py --local-root /var/tmp/hyperdrone-eecs116-aaq
```

This starts fresh SL + PPO with `curriculum.json` and a 6 GiB rolling recording
budget. The launcher prints a unique primary run directory under the project's
`checkpoints/`. Models save there during validation, and `train.log` captures
console output. Scene downloads, recordings, and runtime caches use local disk;
the existing Hyperlux build cache is retained. The home scene cache is not used.

There is **no automatic time limit**, despite the historical script filename.
Training runs until its stages finish, early stopping or epoch caps apply, or you
stop it. On Ctrl+C, SIGTERM, or SIGHUP, the launcher sends an interrupt and allows
up to 120 seconds for cleanup. After a normal exit, training error, or handled
stop, it mirrors completed checkpoint files and logs to
`<local-root>/checkpoint-backups/<run-name>/`. It excludes
unfinished `*.tmp` files and unpublished `.post-sl-*` staging directories.
An optional `--seconds` timeout returns success when completed model files and the mirror
exist; `run_status.json` records whether training finished normally, was stopped,
or required forced termination. Errors and missing model files return nonzero.

Primary model files are already on persistent project storage before that final
copy. A failed mirror is reported and leaves those files intact. Local storage is
temporary; it is a second copy, not a replacement for the primary checkpoints.
No launcher can run a final copy after its own SIGKILL, a system crash, or a power
loss. An unfinished epoch's in-memory updates are not checkpointed by the launcher.

The launcher skips export so time is spent training. Export selected weights
afterward with the commands in the main README. Optional arguments include
`--seconds 3600` for a one-hour run, `--stop-grace 120`, `--max-data-mib 6144`, and
`--checkpoint-root /path/to/persistent/checkpoints`.

Use `--config curriculum_embedding96.json` for the optional fresh 96-dimensional
experiment. To continue a saved run through SL and PPO with the optimized
defaults, add `--from-checkpoints checkpoints/RUN`. The launcher selects a new
output directory and explicitly requests SL: saved components are loaded and
missing components are initialized. Optimizer and curriculum progress restarts.
Keep the same embedding width when continuing an existing checkpoint.
If embedding training is already complete, add `--skip-embedding-training`:

```bash
python -u run_until_11.py \
  --from-checkpoints checkpoints/20261113-180203-195540 \
  --skip-embedding-training \
  --config curriculum.json \
  --local-root /var/tmp/hyperdrone-eecs116-aaq \
  --max-data-mib 6144
```

The saved embedding predictor/encoders remain fixed while state SL and PPO train.
No state checkpoint is required to start; absent state weights initialize normally.
The new primary run, `post_sl/` snapshot, W&B state/PPO metrics, and final local
mirror work as usual. There is no automatic time cutoff.
The [optimization review](verification/training_optimization_review.md) includes
the preserved checkpoint from the HSSD scene-14 collector abort and its resume command.

## Disconnect and reconnect

To leave training running, press **Ctrl+B**, release both keys, then press
**D**. This detaches the session. After detaching, type `exit` to close SSH.
Your personal computer can disconnect or sleep while the lab computer keeps
running. The lab computer must remain running for training to continue.

Reconnect from your personal computer:

```bash
ssh eecs116-aaq@c204-17.eecs.berkeley.edu
tmux attach -t teacher-overnight
```

List available sessions after SSH login with:

```bash
tmux ls
```

## Monitor progress and GPU usage

Attach to the training session to see its live output. For GPU usage, open
a second SSH connection to the same machine and run:

```bash
nvidia-smi -l 5
```

If you launched training with the overnight command's `tee` logging, you can
also follow the log from the model directory. Replace `YOUR_RUN` with the
actual checkpoint directory name:

```bash
tail -n 50 -f checkpoints/YOUR_RUN/train.log
```

Runs also write `embedding_metrics.jsonl`, `performance.jsonl`, and, after
PPO starts, `ppo_metrics.jsonl` in their checkpoint directory. Press **Ctrl+C**
in the monitoring terminal to stop `tail` or `nvidia-smi`; this does not stop
training in the separate session.

## Stop or modify a run

To stop training, attach to its session and press **Ctrl+C**. Allow cleanup
to finish so the background collector and data-loader workers can stop.

Edit an experiment JSON or source file over SSH using an editor such as
`nano`. Editing settings does not change an already-running process: stop
and restart to apply them. Architecture or model-width changes require a
fresh run with compatible weights.

To restart from saved weights, see the [checkpoint recovery guide](README.md#4-continue-from-saved-weights-or-recover-after-interruption).
`--from-checkpoints` restores saved weights but starts fresh optimizer state;
add `--sl --ppo` to request supervised training followed by PPO again. Use
`--config` for experiment overrides and a new output directory to preserve
the source run.

Completed SL checkpoints are preserved in `post_sl/`. If interrupted during
PPO, the best selected model is `state_student_ppo.pt`; the root
`state_student.pt` is updated only after PPO finishes normally. Export can
select the PPO file explicitly with `--state-checkpoint`.

If SSH times out or the hostname does not resolve, check the VPN connection
and hostname. If login is denied, check the instructional-account credentials.
If a `tmux` session is missing, check that you connected to `c204-17` and that
the session was started there.

If activation says `Run 'conda init' before 'conda activate'`, run
`eval "$(conda shell.bash hook)"` in Bash before activating. This initializes
the current shell without changing your shell startup files.

Home storage and the shadow cache have separate quotas. Check both with
`quota -s`; `df -h` alone does not show your account quota. Old generated
recordings can be removed separately from checkpoints. To put new cache writes
on local temporary storage, use `XDG_CACHE_HOME`, `HYPERLUX_CACHE_DIR`,
`CONTA_CACHE`, `TRITON_CACHE_DIR` and `CUDA_CACHE_PATH`. Hyperlux needs its own
override even when `XDG_CACHE_HOME` is set. `CONTA_SEARCH_PATHS` can reuse the
old scene-asset cache as a read-only source. Local `/tmp` storage is temporary;
keep checkpoints in the project directory and copy temporary results elsewhere
before a reboot or cleanup.
