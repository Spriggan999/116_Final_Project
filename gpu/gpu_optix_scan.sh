#!/usr/bin/env bash
# Runs inside WSL. Lists every OptiX / ray-tracing driver file WSL mounts from Windows and the functions each exports.
F=/mnt/c/Users/ninad/Desktop/EECS116/Final_project
LOG=$F/gpu/optix_scan_log.txt
exec > >(tee "$LOG") 2>&1
echo "=== start $(date)"
echo "=== files"
find /usr/lib/wsl -iname "*optix*" -o -iname "*rtcore*" -o -iname "*ptxjit*" 2>/dev/null | xargs ls -laL 2>/dev/null
echo "=== exported functions"
for f in $(find /usr/lib/wsl -iname "*optix*" -name "*.so*" 2>/dev/null); do
  echo "--- $f"
  nm -D --defined-only "$f" 2>/dev/null | awk '{print $NF}' | grep -v "^_" | head -40
done
echo "=== driver store folders"
ls /usr/lib/wsl/drivers/ 2>/dev/null | head
echo "=== done $(date)"
