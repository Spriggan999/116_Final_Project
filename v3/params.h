// Tunable parameters (compile-time). Override with -DNAME=value or edit here, then re-bundle.
#pragma once
// ---- start-up ----
#ifndef CALIB_SECONDS
#define CALIB_SECONDS 0.5      // s of IMU used for gravity direction and gyro bias
#endif
#ifndef REST_ACCEL_STD
#define REST_ACCEL_STD 1.0     // m/s^2: below this the start counts as "at rest"
#endif
#ifndef REST_GYRO_MEAN
#define REST_GYRO_MEAN 0.05    // rad/s: |mean gyro| above this cannot be pure bias -> not at rest
#endif
#ifndef REST_ACCEL_MEAN
#define REST_ACCEL_MEAN 0.3    // m/s^2: | |mean accel| - g | above this -> not at rest
#endif
#ifndef REST_GYRO_STD
#define REST_GYRO_STD 0.2      // rad/s
#endif
// ---- image / tracking ----
#ifndef DS
#define DS 2                   // downsample: 320x240 -> 80x60
#endif
#ifndef PATCH_R
#define PATCH_R 4              // patch radius (9x9)
#endif
#ifndef GRID_X
#define GRID_X 14
#endif
#ifndef GRID_Y
#define GRID_Y 10
#endif
#ifndef N_TRACK
#define N_TRACK 16             // tracked patches per keyframe (most textured of the grid)
#endif
#ifndef LK_ITERS
#define LK_ITERS 3
#endif
#ifndef MIN_EIG
#define MIN_EIG 50.0           // min structure-tensor eigenvalue per pixel
#endif
#ifndef MAX_RESID
#define MAX_RESID 0.6          // max rms residual relative to patch std
#endif
#ifndef MAX_CORR_PX
#define MAX_CORR_PX 3.0        // max LK correction from the gyro prediction (small px)
#endif
#ifndef KF_LEN
#define KF_LEN 6               // frames per keyframe window (vision baseline)
#endif
#ifndef MIN_TRACKS
#define MIN_TRACKS 6
#endif
#ifndef MIN_PAR_PX
#define MIN_PAR_PX 0.5         // skip the update below this mean translational parallax (small px)
#endif
#ifndef MAX_Q
#define MAX_Q 0.3              // reject direction fits with lambda_min / lambda_mid above this
#endif
// ---- measurement noise ----
#ifndef SIG_DIR_PX
#define SIG_DIR_PX 0.5         // direction sigma term: SIG_DIR_PX / (parallax * sqrt(N))
#endif
#ifndef SIG_DIR_SCALE
#define SIG_DIR_SCALE 1.0      // direction sigma term from fit residual: sqrt(q / N) * scale
#endif
#ifndef SIG_DIR_FLOOR
#define SIG_DIR_FLOOR 0.02     // rad
#endif
#ifndef D_FLOOR
#define D_FLOOR 0.01           // m
#endif
#ifndef CHI2_GATE
#define CHI2_GATE 9.0
#endif
#ifndef USE_TILT_COUPLING
#define USE_TILT_COUPLING 1
#endif
// ---- process noise (continuous-time densities) ----
#ifndef SIG_ACC
#define SIG_ACC 0.15           // m/s^2/sqrt(Hz), inflated to cover model errors
#endif
#ifndef SIG_GYR
#define SIG_GYR 0.01           // rad/s/sqrt(Hz)
#endif
#ifndef SIG_BA
#define SIG_BA 0.01            // m/s^3/sqrt(Hz)
#endif
#ifndef SIG_BG
#define SIG_BG 0.001           // rad/s^2/sqrt(Hz)
#endif
// ---- initial uncertainty ----
#ifndef SIG_V0_REST
#define SIG_V0_REST 0.5
#endif
#ifndef SIG_V0_MOVING
#define SIG_V0_MOVING 1.0
#endif
#ifndef SIG_TILT0_REST
#define SIG_TILT0_REST 0.02
#endif
#ifndef SIG_TILT0_MOVING
#define SIG_TILT0_MOVING 0.15
#endif
#ifndef SIG_BA0
#define SIG_BA0 0.2
#endif
#ifndef SIG_BG0_REST
#define SIG_BG0_REST 0.005
#endif
#ifndef SIG_BG0_MOVING
#define SIG_BG0_MOVING 0.03
#endif
#ifndef FIX_SIGN
#define FIX_SIGN 1             // pull backwards-moving displacement (wrong sign vs cheirality) to zero
#endif
#ifndef VPRIOR_SIG
#define VPRIOR_SIG 0.5         // m/s, weak zero-velocity prior per keyframe window (0 = off)
#endif
#ifndef SMALL_FLOW_UPDATE
#define SMALL_FLOW_UPDATE 1    // below MIN_PAR_PX: soft "displacement ~ 0" update (hover / slow motion)
#endif
#ifndef SMALL_FLOW_PX
#define SMALL_FLOW_PX 0.5      // added pixel slack (small px)
#endif
#ifndef DEPTH_PRIOR
#define DEPTH_PRIOR 4.0        // m, typical upper depth indoors (sets the displacement sigma)
#endif
#ifndef PAR_SPEED
#define PAR_SPEED 6.0          // m: typical depth*geometry factor for the parallax speed proxy (0 = off)
#endif
#ifndef PAR_SPEED_REL
#define PAR_SPEED_REL 0.5      // relative sigma of the speed proxy
#endif
#ifndef ROT_FROM_VISION
#define ROT_FROM_VISION 1      // estimate the keyframe->current rotation error from the tracks (fixes gyro-bias yaw drift)
#endif
#ifndef ROT_ITERS
#define ROT_ITERS 1
#endif
#ifndef ROT_DAMP
#define ROT_DAMP 1e-4          // Levenberg damping for the rotation solve
#endif
#ifndef ROT_MAX
#define ROT_MAX 0.05           // rad: reject larger corrections as outliers
#endif
#ifndef SIG_ROT
#define SIG_ROT 0.01           // rad per track (scaled by 1/sqrt(N))
#endif
#ifndef SIG_ROT_FLOOR
#define SIG_ROT_FLOOR 0.001    // rad
#endif
#ifndef ROT_ATT_GAIN
#define ROT_ATT_GAIN 0.0       // fraction of the measured rotation error removed from the current attitude
#endif
#ifndef SIG_ROT_PAR
#define SIG_ROT_PAR 0.015      // rad per px of translational parallax (rotation/translation ambiguity)
#endif
#ifndef ROT_APPLY_DIR
#define ROT_APPLY_DIR 0        // also use the corrected rotation for the direction fit
#endif
