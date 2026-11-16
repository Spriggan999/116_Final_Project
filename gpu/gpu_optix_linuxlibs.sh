#!/usr/bin/env bash
# Runs inside WSL. Unofficial NVIDIA-forum workaround for OptiX under WSL: take the OptiX libraries from the Linux driver of
# the same branch (590) and load them from a PRIVATE folder (~/optix_wsl) put first on LD_LIBRARY_PATH.
# No system files are changed; to undo, delete ~/optix_wsl. No sudo needed.
F=/mnt/c/Users/ninad/Desktop/EECS116/Final_project
LOG=$F/gpu/optix_linuxlibs_log.txt
VER=${1:-590.48.01}
exec > >(tee "$LOG") 2>&1
echo "=== start $(date)   Linux driver $VER"
mkdir -p ~/optix_build && cd ~/optix_build
RUN=NVIDIA-Linux-x86_64-$VER.run
echo "=== 1. download the Linux driver package (~400 MB; extracted only, never installed)"
[ -f "$RUN" ] || wget -q --show-progress "https://download.nvidia.com/XFree86/Linux-x86_64/$VER/$RUN" || { echo "FAIL: download"; exit 1; }
[ -d "NVIDIA-Linux-x86_64-$VER" ] || sh "$RUN" -x > /dev/null || { echo "FAIL: extract"; exit 1; }
D=~/optix_build/NVIDIA-Linux-x86_64-$VER
echo "=== 2. copy the four OptiX pieces into ~/optix_wsl"
rm -rf ~/optix_wsl && mkdir -p ~/optix_wsl
cp "$D/libnvoptix.so.$VER"        ~/optix_wsl/libnvoptix.so.1
cp "$D/libnvidia-rtcore.so.$VER"  ~/optix_wsl/
cp "$D/libnvidia-gpucomp.so.$VER" ~/optix_wsl/
cp "$D/nvoptix.bin"               ~/optix_wsl/ 2>/dev/null || echo "(no nvoptix.bin in package; only needed for the denoiser)"
ls -la ~/optix_wsl
export LD_LIBRARY_PATH=$HOME/optix_wsl:/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
echo "=== 3. does it load, with the entry function?"
python3 - <<'PY'
import ctypes, os
try:
    l = ctypes.CDLL(os.path.expanduser("~/optix_wsl/libnvoptix.so.1"))
    print("load OK;", "HAS optixQueryFunctionTable" if hasattr(l, "optixQueryFunctionTable") else "no entry symbol")
except OSError as e:
    print("load FAILED:", e)
PY
echo "=== 4. 10 s GPU test flight (OptiX)"
export PATH=/usr/local/cuda/bin:$PATH
export HYPERLUX_BACKEND=OPTIX
source ~/vio/.venv/bin/activate
mkdir -p "$F/data/optix_test"
cd ~
LD_DEBUG=libs viobench simulate -o "$F/data/optix_test" --episodes 1 --seed 3 --duration 10 2> /tmp/ld_debug.txt | tail -5
status=${PIPESTATUS[0]}
echo "=== exit code $status"
echo "--- OptiX libraries actually loaded:"
grep -E "calling init: .*(nvoptix|rtcore|gpucomp)" /tmp/ld_debug.txt | sed 's/.*calling init: //' | sort -u
grep -i -E "optix|error|fail" /tmp/ld_debug.txt | grep -v "find library\|search path\|trying file" | tail -5
ls -la "$F/data/optix_test"
echo "=== done $(date)"
