#!/usr/bin/env bash
# Runs inside WSL. TRAINING flights for the speed model under server-like conditions, varied so the model cannot
# latch onto one setting: 4 settings x 4 ProcTHOR houses (80-95) x 8 flights = 128 flights, ~17 GB.
# Each new setting recompiles the simulator once (~5-7 min), so houses are grouped 4 per setting.
# Houses 80-95 are used nowhere else. A hanging run is skipped after 90 min.
D=/mnt/c/Users/ninad/Desktop/EECS116/Final_project/data
source ~/vio/.venv/bin/activate || exit 1
export PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=$HOME/optix_wsl:/usr/lib/wsl/lib:${LD_LIBRARY_PATH} HYPERLUX_BACKEND=OPTIX VIO_PAR=8
exec > >(tee -a "$D/agile_log.txt") 2>&1
mkdir -p "$D/.spool" "$D/s2train"; cd ~
MED="1.0 2.0 2.0 6.0 1.6 0.6 1.5"; HARD="2.0 3.5 4.0 12.0 2.5 1.2 3.0"; XFAST="2.5 4.0 6.0 20.0 3.0 2.0 5.0 0.5"
run() {   # SEED AUTOPILOT IMU BIAS WIND HOVER HOUSES...
  local seed=$1 ap=$2 imu=$3 bias=$4 wind=$5 hover=$6; shift 6
  local sc=""; for h in "$@"; do sc="$sc ProcTHOR-Train-$h"; done
  echo "=== s2train $(date) houses $* | autopilot '$ap' imu $imu bias $bias wind $wind hover $hover"
  AGILE="$ap" IMU=$imu BIAS=$bias WIND=$wind timeout -k 10 5400 "$VIRTUAL_ENV/bin/python3" "$D/agile_sim2.py" -o "$D/s2train" --no-video \
     --spool "$D/.spool" --episodes 8 --seed $seed --waypoints 200 --hover $hover --scene-names $sc || echo "=== SKIPPED houses $*"
}
run 181 "$HARD"  2   2   1   0  80 81 82 83
run 182 "$XFAST" 3   3   2   0  84 85 86 87
run 183 "$MED"   1.5 1.5 0.5 1  88 89 90 91
run 184 "$HARD"  2.5 2.5 3   0  92 93 94 95
echo "=== gen_s2train done $(date)"
