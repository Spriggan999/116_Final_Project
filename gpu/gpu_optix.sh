#!/usr/bin/env bash
# Runs inside WSL. Installs NVIDIA's CUDA toolkit (compiler only; the Windows driver provides the GPU) so HyperLux,
# HyperDrone's renderer, builds its OptiX backend and renders on the RTX 3060 instead of the CPU. Then a 10 s test flight.
F=/mnt/c/Users/ninad/Desktop/EECS116/Final_project
LOG=$F/gpu/optix_log.txt
exec > >(tee "$LOG") 2>&1
echo "=== start $(date)"
echo "=== 1. OptiX runtime from the Windows driver"
ls /usr/lib/wsl/lib/ | grep -i -E "optix|libcuda\.so" || echo "WARNING: no libnvoptix in /usr/lib/wsl/lib"
echo "=== 2. CUDA toolkit (needs your password once)"
if ! command -v nvcc >/dev/null && [ ! -x /usr/local/cuda/bin/nvcc ]; then
  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update -qq
  for v in 12-9 12-8 12-6; do
    if sudo apt-get install -y cuda-toolkit-$v; then echo "installed cuda-toolkit-$v"; break; fi
  done
  sudo apt-get install -y cmake git build-essential
fi
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version | tail -2 || { echo "FAIL: nvcc not available"; exit 1; }
echo "=== 3. 10 s test flight with the OptiX backend"
source ~/vio/.venv/bin/activate
export HYPERLUX_BACKEND=OPTIX
mkdir -p "$F/data/optix_test"
cd ~
time viobench simulate -o "$F/data/optix_test" --episodes 1 --seed 3 --duration 10
echo "=== exit code $?"
ls -la "$F/data/optix_test"
echo "=== done $(date)"
