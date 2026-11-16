@echo off
rem Double-click: runs gpu_episodes.sh in WSL (Ubuntu). Log: gpu\gpu_log.txt
wsl bash /mnt/c/Users/ninad/Desktop/EECS116/Final_project/gpu/gpu_episodes.sh %1 %2
echo.
echo Finished - tell Claude to read gpu\gpu_log.txt
pause
