#!/usr/bin/env bash
# Runs inside WSL. Fast GPU episode generation: more flights at once on the GPU, no MP4 videos, fast lossless PNGs,
# episodes and temporary frames written straight to the Windows folder (nothing stored in WSL). Normal CPU priority.
# Usage: bash gpu_fast.sh EPISODES_PER_SCENE SEED "SCENE1 SCENE2 ..."      (env VIO_PAR = flights at once, default 8)
F=/mnt/c/Users/ninad/Desktop/EECS116/Final_project
LOG=$F/gpu/gpu_fast_log.txt
EPISODES=${1:-8}
SEED=${2:-11}
SCENES=${3:-"ProcTHOR-Train-2 ProcTHOR-Train-3 ProcTHOR-Train-4"}
export VIO_PAR=${VIO_PAR:-8}
exec > >(tee -a "$LOG") 2>&1
echo "=== start $(date)   $EPISODES per scene, seed $SEED, $VIO_PAR flights at once, scenes: $SCENES"
source ~/vio/.venv/bin/activate || { echo "FAIL: venv"; exit 1; }
export PATH=/usr/local/cuda/bin:$PATH
if [ -f ~/optix_wsl/libnvoptix.so.1 ] && [ ! -L ~/optix_wsl/libnvoptix.so.1 ]; then
  export LD_LIBRARY_PATH=$HOME/optix_wsl:/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
  export HYPERLUX_BACKEND=OPTIX
else
  echo "FAIL: ~/optix_wsl missing; run gpu_optix_linuxlibs.sh first"; exit 1
fi
echo "renderer backend: $HYPERLUX_BACKEND"
OUT=$F/data/fresh; SPOOL=$F/data/.spool
mkdir -p "$OUT" "$SPOOL" "$F/data/fresh"
cd ~
# python3 (not python): the renderer cache is keyed on the interpreter path, and viobench uses .venv/bin/python3
time "$VIRTUAL_ENV/bin/python3" "$F/gpu/fast_sim.py" -o "$OUT" --no-video --spool "$SPOOL" \
     --episodes "$EPISODES" --seed "$SEED" --scene-names $SCENES
status=$?
echo "=== simulate exit code $status"
ls -la "$F/data/fresh"
echo "=== done $(date)"
