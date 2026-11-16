@echo off
rem Double-click: checks and sets up GPU rendering for HyperDrone in WSL. Log: gpu\vulkan_log.txt
wsl bash /mnt/c/Users/ninad/Desktop/EECS116/Final_project/gpu/gpu_vulkan.sh
echo.
echo Finished - tell Claude to read gpu\vulkan_log.txt
pause
