cmake_minimum_required(VERSION 3.20)
project(solution C CXX)
# wasm (the submission): viobench bundle <this directory> -o solution.wasm
# host debug build:      cmake -S . -B build-host && cmake --build build-host && build-host/debug <feed dir>   (docs/DEBUGGING.md)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(VIOBENCH_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/native" CACHE PATH "directory with viobench_abi.h and viobench_reactor.cmake")
include(${VIOBENCH_NATIVE_DIR}/viobench_reactor.cmake)
viobench_solution(solution solution.cpp)
target_compile_options(solution PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>)
# stage-2 configuration (all switches are documented in params.h)
target_compile_definitions(solution PRIVATE
    AHEAD_FRAMES=4
    CALIB_SECONDS=0.0
    HOVER_INIT=0
    MIDPOINT=1
    MIN_EIG=12.0
    MIN_PAR_PX=0.5
    N_EXTRA=12
    N_TRACK=12
    P_FLOAT=1
    ROT_CLONE=1
    ROT_REFIT=0
    SIGN_FWD=0
    SIG_ACC_H=0.3
    SIG_BAZ0=0.03
    SIG_TILT0_HOVER=0.003
    SPEED_DEPTH=2.5
    SPEED_MODE=1
    SYM_P=1
    THRUST_ONLY=1
    TRK=2
    TRK_DISP=1
    TRK_HALF_RANK=0
    TRK_SCHED=3
    VPRIOR_REL=0.5
    VPRIOR_SIG=1.0
    MOVE_START=1
    MS_MIN_SAMPLES=100000
    REST_GYRO_MEAN=0.0
    SPEED_NN=1
    ADAPT_IMU=2)
