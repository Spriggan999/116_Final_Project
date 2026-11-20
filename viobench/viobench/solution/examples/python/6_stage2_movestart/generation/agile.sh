#!/usr/bin/env bash
# Runs inside WSL. Agile (stage-2 style) episodes on the GPU. Self-contained: activates the venv itself.
# Usage: bash agile.sh OUTDIR EPISODES_PER_SCENE SEED "SCENE1 SCENE2 ..." TIER      TIER = med | hard | "7 numbers" (see agile_sim.py)
D=/mnt/c/Users/ninad/Desktop/EECS116/Final_project/data
OUT=$D/${1:-agile_dev_med}; EPISODES=${2:-8}; SEED=${3:-31}; SCENES=${4:-"ProcTHOR-Train-195"}; TIER=${5:-med}
case "$TIER" in
  med)  export AGILE="1.0 2.0 2.0 6.0 1.6 0.6 1.5";  WP=120 ;;
  hard) export AGILE="2.0 3.5 4.0 12.0 2.5 1.2 3.0"; WP=200 ;;
  xfast) export AGILE="2.5 4.0 6.0 20.0 3.0 2.0 5.0 0.5"; WP=300 ;;
  *)    export AGILE="$TIER"; WP=${WP:-120} ;;
esac
export VIO_PAR=${VIO_PAR:-8}
exec > >(tee -a "$D/agile_log.txt") 2>&1
echo "=== start $(date)  -> $OUT  tier $TIER ($AGILE)  $EPISODES per scene, seed $SEED, scenes: $SCENES"
source ~/vio/.venv/bin/activate || exit 1
export PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=$HOME/optix_wsl:/usr/lib/wsl/lib:${LD_LIBRARY_PATH} HYPERLUX_BACKEND=OPTIX
mkdir -p "$OUT" "$D/.spool"; cd ~
time "$VIRTUAL_ENV/bin/python3" "$D/agile_sim.py" -o "$OUT" --no-video --spool "$D/.spool" --episodes "$EPISODES" --seed "$SEED" --waypoints "$WP" --scene-names $SCENES
echo "=== exit $?  $(date)"; ls "$OUT"
