#!/usr/bin/env bash
# Runs inside WSL. Evaluates v7, v8 and v8-lite on the fresh episodes in parallel: one job per (model, house),
# JOBS at once (default: number of CPU threads - 2). Reports: reports_fresh/<model>/<house>/report-<model>/report.json
F=/mnt/c/Users/ninad/Desktop/EECS116/Final_project
LOG=$F/gpu/eval_fresh_log.txt
JOBS=${JOBS:-$(( $(nproc) > 3 ? $(nproc) - 2 : 1 ))}
exec > >(tee "$LOG") 2>&1
source ~/vio/.venv/bin/activate
echo "=== start $(date)   $JOBS jobs at once"
run() {  # model wasm house
  d=$F/reports_fresh/$1/$3; mkdir -p "$d"; cd "$d"
  out=$(viobench evaluate "$2" "$F"/data/fresh/sim-$3-*.zip 2>&1 | grep "^summary")
  echo "$1 $3: $out"
}
export -f run; export F
for m in "v7 $F/v7/v7.wasm" "v8 $F/final/v8/v8.wasm" "v8lite $F/final/v8lite/v8lite.wasm"; do
  for h in ProcTHOR-Train-1 ProcTHOR-Train-2 ProcTHOR-Train-3 ProcTHOR-Train-4 ProcTHOR-Train-5; do echo "$m $h"; done
done | time xargs -P "$JOBS" -L 1 bash -c 'run "$0" "$1" "$2"'
echo "=== done $(date)"
