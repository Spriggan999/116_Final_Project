#!/usr/bin/env bash
# Runs inside WSL. Fixes "optixInit failed 7805" (OptiX entry symbol not found): in WSL the OptiX entry point lives in
# libnvoptix_loader.so.1, so we point a private libnvoptix.so.1 alias at it (no system files touched) and retry.
F=/mnt/c/Users/ninad/Desktop/EECS116/Final_project
LOG=$F/gpu/optix_fix2_log.txt
exec > >(tee "$LOG") 2>&1
echo "=== start $(date)"
echo "=== 1. which library exports optixQueryFunctionTable?"
python3 - <<'PY'
import ctypes
for p in ["/usr/lib/wsl/lib/libnvoptix.so.1", "/usr/lib/wsl/lib/libnvoptix_loader.so.1"]:
    try:
        l = ctypes.CDLL(p); print(p, "->", "HAS optixQueryFunctionTable" if hasattr(l, "optixQueryFunctionTable") else "no entry symbol")
    except OSError as e:
        print(p, "-> load failed:", e)
PY
echo "=== 2. private alias libnvoptix.so.1 -> libnvoptix_loader.so.1"
mkdir -p ~/optix_wsl
ln -sf /usr/lib/wsl/lib/libnvoptix_loader.so.1 ~/optix_wsl/libnvoptix.so.1
ls -la ~/optix_wsl
echo "=== 3. retry the 10 s GPU test flight"
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/optix_wsl:/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
export HYPERLUX_BACKEND=OPTIX
source ~/vio/.venv/bin/activate
mkdir -p "$F/data/optix_test"
cd ~
time viobench simulate -o "$F/data/optix_test" --episodes 1 --seed 3 --duration 10
echo "=== exit code $?"
ls -la "$F/data/optix_test"
echo "=== done $(date)"
