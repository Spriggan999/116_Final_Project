#!/usr/bin/env bash
# Runs inside WSL. The real OptiX engine is nvoptix.bin in the driver store (libnvoptix.so.1 there is only a dxcore loader).
# If nvoptix.bin is a Linux library that exports optixQueryFunctionTable, alias libnvoptix.so.1 to it and retry.
F=/mnt/c/Users/ninad/Desktop/EECS116/Final_project
LOG=$F/gpu/optix_fix3_log.txt
exec > >(tee "$LOG") 2>&1
echo "=== start $(date)"
BIN=$(find /usr/lib/wsl/drivers -name nvoptix.bin 2>/dev/null | head -1)
echo "nvoptix.bin: $BIN"
file -L "$BIN"
echo "=== exports"
nm -D --defined-only "$BIN" 2>/dev/null | awk '{print $NF}' | grep -i -E "optix|query" | head
echo "=== dependencies"
ldd "$BIN" 2>&1 | head -20
python3 - "$BIN" <<'PY'
import ctypes, sys
try:
    l = ctypes.CDLL(sys.argv[1]); print("load OK;", "HAS optixQueryFunctionTable" if hasattr(l, "optixQueryFunctionTable") else "no optixQueryFunctionTable")
except OSError as e:
    print("load FAILED:", e)
PY
echo "=== alias libnvoptix.so.1 -> nvoptix.bin, retry the 10 s GPU test"
mkdir -p ~/optix_wsl
ln -sf "$BIN" ~/optix_wsl/libnvoptix.so.1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/optix_wsl:/usr/lib/wsl/lib:$(dirname "$BIN"):${LD_LIBRARY_PATH}
export HYPERLUX_BACKEND=OPTIX
source ~/vio/.venv/bin/activate
cd ~
time viobench simulate -o "$F/data/optix_test" --episodes 1 --seed 3 --duration 10
echo "=== exit code $?"
ls -la "$F/data/optix_test"
echo "=== OptiX headers used by the build (version)"
grep -rh "define OPTIX_VERSION " ~/.cache/hyperlux ~/.cache/hyperdrone 2>/dev/null | sort -u | head -3
echo "=== done $(date)"
