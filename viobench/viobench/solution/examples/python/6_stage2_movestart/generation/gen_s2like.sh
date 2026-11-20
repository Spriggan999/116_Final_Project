#!/usr/bin/env bash
# Runs inside WSL. Server-like test flights: no start hover, agile autopilot, noisier IMU with larger biases, wind.
# 3 levels x 4 scenes x 8 flights = 96 flights (~13 GB). Checking only (never train or tune on these).
# Scenes: ProcTHOR 50-57 (unused elsewhere) and 4 HSSD houses (larger real-world layouts). A hanging scene is skipped after 25 min.
D=/mnt/c/Users/ninad/Desktop/EECS116/Final_project/data
source ~/vio/.venv/bin/activate || exit 1
export PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=$HOME/optix_wsl:/usr/lib/wsl/lib:${LD_LIBRARY_PATH} HYPERLUX_BACKEND=OPTIX VIO_PAR=8
exec > >(tee -a "$D/agile_log.txt") 2>&1
mkdir -p "$D/.spool"; cd ~
one() {   # OUT SEED SCENE AGILE IMU BIAS WIND
  mkdir -p "$D/$1"; echo "=== s2like $(date) -> $1 scene $3 agile '$4' imu $5 bias $6 wind $7"
  AGILE="$4" IMU=$5 BIAS=$6 WIND=$7 timeout -k 10 1500 "$VIRTUAL_ENV/bin/python3" "$D/agile_sim2.py" -o "$D/$1" --no-video --spool "$D/.spool" \
     --episodes 8 --seed $2 --waypoints 200 --hover 0 --scene-names $3 || echo "=== SKIPPED $3 ($1)"
}
HARD="2.0 3.5 4.0 12.0 2.5 1.2 3.0"; XFAST="2.5 4.0 6.0 20.0 3.0 2.0 5.0 0.5"
for h in 50 51 52 53; do one s2_hard_noisy 71 ProcTHOR-Train-$h "$HARD" 2 2 1; done
for h in 54 55 56 57; do one s2_xfast_extreme 73 ProcTHOR-Train-$h "$XFAST" 3 3 2; done
SC=$("$VIRTUAL_ENV/bin/python3" -c "from viobench.tools.datasets.generate.simulation import hssd_scenes; print(' '.join(list(hssd_scenes('train'))[60:64]))")
for s in $SC; do one s2_hssd_noisy 75 $s "$HARD" 2 2 1; done
echo "=== gen_s2like done $(date)"; ls $D/s2_* | head -50
