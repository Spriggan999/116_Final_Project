#!/usr/bin/env bash
# Runs inside WSL. Diagnoses "optixInit failed 7804" (OptiX library not found) and retries the 10 s GPU test flight.
F=/mnt/c/Users/ninad/Desktop/EECS116/Final_project
LOG=$F/gpu/optix_fix_log.txt
exec > >(tee "$LOG") 2>&1
echo "=== start $(date)"
echo "=== 1. is the WSL driver directory in the loader path?"
cat /etc/ld.so.conf.d/*wsl* 2>&1
ldconfig -p | grep -i -E "nvoptix|libcuda.so.1" || echo "(not in ldconfig cache)"
echo "=== 2. can the OptiX library be loaded?"
python3 - <<'PY'
import ctypes
for name in ["libnvoptix.so.1", "/usr/lib/wsl/lib/libnvoptix.so.1", "/usr/lib/wsl/lib/libnvoptix_loader.so.1"]:
    try:
        ctypes.CDLL(name); print("OK  ", name)
    except OSError as e:
        print("FAIL", name, "->", e)
PY
echo "=== 3. retry the test flight with the WSL driver directory on the library path"
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
export HYPERLUX_BACKEND=OPTIX
source ~/vio/.venv/bin/activate
mkdir -p "$F/data/optix_test"
cd ~
time viobench simulate -o "$F/data/optix_test" --episodes 1 --seed 3 --duration 10
echo "=== exit code $?"
ls -la "$F/data/optix_test"
echo "=== done $(date)"
