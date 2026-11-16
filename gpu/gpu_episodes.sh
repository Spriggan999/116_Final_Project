#!/usr/bin/env bash
# Runs inside WSL (Ubuntu). Checks the GPU, finds the viobench venv, installs hyperdrone, generates fresh episodes.
# Everything is logged to Final_project/gpu/gpu_log.txt so Claude can read it.
F=/mnt/c/Users/ninad/Desktop/EECS116/Final_project
LOG=$F/gpu/gpu_log.txt
EPISODES=${1:-1}      # episodes per scene (default 1 for a first test)
SEED=${2:-7}
SCENES=${3:-}       # e.g. "ProcTHOR-Train-2 ProcTHOR-Train-3" (default: the generator's default scene)
exec > >(tee -a "$LOG") 2>&1
echo "=== start $(date)"

echo "=== 1. GPU visible in WSL?"
if ! nvidia-smi; then
  echo "FAIL: nvidia-smi not found or no GPU. Update the NVIDIA Windows driver (WSL CUDA support is in the driver), then rerun."
  exit 1
fi

echo "=== 2. find the viobench environment"
VENV=""
for c in ~/vio/.venv ~/vio/venv ~/.venv ~/venv ~/viobench/.venv; do
  if [ -f "$c/bin/activate" ] && "$c/bin/python" -c "import viobench" 2>/dev/null; then VENV=$c; break; fi
done
if [ -z "$VENV" ]; then
  for a in $(find ~ -maxdepth 4 -path "*bin/activate" 2>/dev/null); do
    d=$(dirname "$(dirname "$a")")
    if "$d/bin/python" -c "import viobench" 2>/dev/null; then VENV=$d; break; fi
  done
fi
if [ -z "$VENV" ]; then echo "FAIL: no Python venv with viobench found under ~"; exit 1; fi
echo "venv: $VENV"
source "$VENV/bin/activate"
python -c "import viobench, sys; print('viobench ok, python', sys.version.split()[0])"

echo "=== 3. hyperdrone"
python -c "import hyperdrone" 2>/dev/null || pip install hyperdrone
python -c "import hyperdrone; print('hyperdrone', getattr(hyperdrone, '__version__', 'installed'))" || { echo "FAIL: hyperdrone import"; exit 1; }

DZN=$(ls /usr/share/vulkan/icd.d/dzn_icd*.json 2>/dev/null | head -1)
if [ -n "$DZN" ]; then export VK_ICD_FILENAMES=$DZN; echo "rendering through Dozen (Vulkan on D3D12): $DZN"; else echo "Dozen not installed: rendering may fall back to the CPU (llvmpipe)"; fi
# GPU rendering: OptiX on the RTX 3060 via the Linux 590 OptiX libraries in ~/optix_wsl (gpu_optix_linuxlibs.sh);
# falls back to CPU rendering when they are missing. Undo: rm -rf ~/optix_wsl
export PATH=/usr/local/cuda/bin:$PATH
if [ -f ~/optix_wsl/libnvoptix.so.1 ] && [ ! -L ~/optix_wsl/libnvoptix.so.1 ] && command -v nvcc >/dev/null; then
  export LD_LIBRARY_PATH=$HOME/optix_wsl:/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
  export HYPERLUX_BACKEND=OPTIX
else
  export HYPERLUX_BACKEND=WEBGPU
fi
echo "renderer backend: $HYPERLUX_BACKEND"
# share the CPU: lowest priority (other programs always go first) and a capped number of render threads
export LP_NUM_THREADS=${LP_THREADS:-4}
echo "CPU rendering with $LP_NUM_THREADS threads at lowest priority"
echo "=== 4. generate episodes ($EPISODES per scene, seed $SEED) into $F/data/fresh"
mkdir -p "$F/data/fresh"
cd ~
if [ -n "$SCENES" ]; then
  time nice -n 19 viobench simulate -o "$F/data/fresh" --episodes "$EPISODES" --seed "$SEED" --scene-names $SCENES
else
  time nice -n 19 viobench simulate -o "$F/data/fresh" --episodes "$EPISODES" --seed "$SEED"
fi
status=$?
echo "=== simulate exit code $status"
ls -la "$F/data/fresh"
echo "=== done $(date)"
