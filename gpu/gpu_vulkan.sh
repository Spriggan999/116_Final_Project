#!/usr/bin/env bash
# Runs inside WSL. Makes WebGPU (HyperDrone's renderer) use the RTX 3060 via Mesa's "Dozen" driver (Vulkan on D3D12).
F=/mnt/c/Users/ninad/Desktop/EECS116/Final_project
LOG=$F/gpu/vulkan_log.txt
exec > >(tee "$LOG") 2>&1
echo "=== start $(date)"
grep -E "PRETTY_NAME" /etc/os-release
echo "=== 1. install Vulkan drivers + tools (needs your password once)"
sudo apt-get update -qq && sudo apt-get install -y mesa-vulkan-drivers vulkan-tools
echo "=== 2. Vulkan driver files"
ls /usr/share/vulkan/icd.d/ 2>&1
DZN=$(ls /usr/share/vulkan/icd.d/dzn_icd*.json 2>/dev/null | head -1)
echo "dozen ICD: ${DZN:-NOT FOUND}"
ls -la /dev/dxg 2>&1
echo "=== 3. devices seen by Vulkan (all drivers)"
vulkaninfo --summary 2>&1 | grep -E "deviceName|driverName|deviceType|apiVersion" | head -20
if [ -n "$DZN" ]; then
  echo "=== 4. devices seen through Dozen only"
  VK_ICD_FILENAMES=$DZN vulkaninfo --summary 2>&1 | grep -E "deviceName|driverName|deviceType|WARNING|ERROR" | head -20
fi
echo "=== done $(date)"
