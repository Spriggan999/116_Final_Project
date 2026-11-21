// LQG, IMU only (no image access): variant `lqg_imu_final` in ../../variants.py. See ../README.md.
// Build: viobench bundle <this folder> -o lqg_imu.wasm
#define LQG_VISION 0
#define VEL_PRIOR 2
#define VP_ATTITUDE 1
#define VP_CONST_SIG_X 0.255
#define VP_CONST_SIG_Y 0.075
#define VP_CONST_SIG_Z 0.255
#define VP_STILL_SIG 0.25
#define CALIB_SECONDS 0.97
#define CALIB_ACCEL_BIAS 1
#define LQG_TILT_ROT 1
#define LQG_WIND_HI 2.6
#include "../solution.cpp"
