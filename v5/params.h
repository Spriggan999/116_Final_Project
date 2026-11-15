// Tunable parameters (compile-time). Override with -DNAME=value or edit here, then re-bundle.
#pragma once
// ---- start-up ----
#ifndef CALIB_SECONDS
#define CALIB_SECONDS 0.5      // s of IMU used for gravity direction and gyro bias
#endif
#ifndef REST_ACCEL_STD
#define REST_ACCEL_STD 1.0     // m/s^2: below this the start counts as "at rest"
#endif
#ifndef REST_GYRO_STD
#define REST_GYRO_STD 0.2      // rad/s
#endif
#ifndef REST_GYRO_MEAN
#define REST_GYRO_MEAN 0.05    // rad/s: |mean gyro| above this cannot be pure bias -> not at rest
#endif
#ifndef REST_ACCEL_MEAN
#define REST_ACCEL_MEAN 0.3    // m/s^2: | |mean accel| - g | above this -> not at rest
#endif
// ---- image / tracking ----
#ifndef DS
#define DS 2                   // tracking resolution: 320x240 / DS (160x120)
#endif
#ifndef PIX_MODE
#define PIX_MODE 1             // pixel fetch: 0 = mean of 4 green samples, 1 = 2 (diagonal), 2 = 1 sample
#endif
#ifndef SCORE_N
#define SCORE_N 4              // texture score: SCORE_N x SCORE_N raw samples per candidate cell
#endif
#ifndef RANK_SPLIT
#define RANK_SPLIT 1           // rank the candidate grid over this many frames (keyframe + first template frames)
#endif
#ifndef PATCH_SIZE
#define PATCH_SIZE 8           // patch side (multiple of 4 for SIMD)
#endif
#ifndef CROP_MARGIN
#define CROP_MARGIN 3          // crop margin around the gyro-predicted patch position (tracking px)
#endif
#ifndef GRID_X
#define GRID_X 16              // candidate grid
#endif
#ifndef GRID_Y
#define GRID_Y 11
#endif
#ifndef N_TRACK
#define N_TRACK 20             // tracked patches per window
#endif
#ifndef N_EXTRA
#define N_EXTRA 8              // extra candidates examined when some fail the texture test
#endif
#ifndef LK_ITERS
#define LK_ITERS 3
#endif
#ifndef MIN_EIG
#define MIN_EIG 50.0           // min structure-tensor eigenvalue per pixel (tracking resolution)
#endif
#ifndef CONTRAST_POW
#define CONTRAST_POW 1         // 1: scale the texture threshold with the frame's gradient energy (gain-invariant), 0: fixed
#endif
#ifndef CONTRAST_REF
#define CONTRAST_REF 2000.0    // typical frame gradient energy (mean half-trace of the ranking tensors)
#endif
#ifndef CONTRAST_MIN
#define CONTRAST_MIN 0.15      // never scale the threshold below this fraction
#endif
#ifndef MAX_RESID
#define MAX_RESID 0.6          // max rms residual relative to patch std
#endif
#ifndef MAX_CORR_PX
#define MAX_CORR_PX 3.0        // max LK correction from the gyro prediction (tracking px)
#endif
#ifndef PROP_ONCE
#define PROP_ONCE 0            // 1: propagate the covariance once per tracking burst instead of every tracking frame
#endif
#ifndef SPLIT
#define SPLIT 3                // spread templates and tracking over SPLIT consecutive frames (flattens max fuel)
#endif
#ifndef KF_LEN
#define KF_LEN 6               // frames from keyframe to tracking (vision baseline)
#endif
#ifndef MIN_TRACKS
#define MIN_TRACKS 6
#endif
#ifndef MIN_PAR_PX
#define MIN_PAR_PX 0.5         // below this mean translational parallax: "small displacement" update instead
#endif
#ifndef MAX_Q
#define MAX_Q 0.3              // reject direction fits with lambda_min / lambda_mid above this
#endif
#ifndef EIG_ITERS
#define EIG_ITERS 12
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
#ifndef SMALL_FLOW_UPDATE
#define SMALL_FLOW_UPDATE 1    // below MIN_PAR_PX: soft "displacement ~ 0" update (hover / slow motion)
#endif
#ifndef SMALL_FLOW_PX
#define SMALL_FLOW_PX 0.5
#endif
#ifndef DEPTH_PRIOR
#define DEPTH_PRIOR 4.0        // m, typical upper depth indoors (sets the small-displacement sigma)
#endif
#ifndef PAR_SPEED
#define PAR_SPEED 8.0          // m: depth factor of the parallax speed proxy |d| ~ parallax * PAR_SPEED / f
#endif
#ifndef SPEED_MODE
#define SPEED_MODE 0           // 0: mean parallax proxy, 1: direction-corrected median |d|/Z times SPEED_DEPTH
#endif
#ifndef SPEED_DEPTH
#define SPEED_DEPTH 3.0        // m, typical scene depth for SPEED_MODE 1
#endif
#ifndef SPEED_GMIN
#define SPEED_GMIN 0.2         // ignore tracks whose expected flow factor is below this (near the focus of expansion)
#endif
#ifndef PAR_SPEED_REL
#define PAR_SPEED_REL 0.5      // relative sigma of the speed proxy
#endif
#ifndef VPRIOR_SIG
#define VPRIOR_SIG 0.5         // m/s, weak zero-velocity prior per window (0 = off)
#endif
// ---- rotation from vision (gyro bias) ----
#ifndef ROT_FROM_VISION
#define ROT_FROM_VISION 1
#endif
#ifndef ROT_DAMP
#define ROT_DAMP 1e-4
#endif
#ifndef ROT_MAX
#define ROT_MAX 0.05           // rad: reject larger corrections as outliers
#endif
#ifndef SIG_ROT
#define SIG_ROT 0.01           // rad per track (scaled by 1/sqrt(N))
#endif
#ifndef SIG_ROT_FLOOR
#define SIG_ROT_FLOOR 0.001
#endif
#ifndef SIG_ROT_PAR
#define SIG_ROT_PAR 0.05       // rad per px of translational parallax (rotation/translation ambiguity)
#endif
// ---- process noise (continuous-time densities) ----
#ifndef SIG_ACC
#define SIG_ACC 0.15
#endif
#ifndef SIG_GYR
#define SIG_GYR 0.01
#endif
#ifndef SIG_BA
#define SIG_BA 0.01
#endif
#ifndef SIG_BG
#define SIG_BG 0.001
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
#ifndef PROF_SKIP_TRACK
#define PROF_SKIP_TRACK 0
#endif
#ifndef PROF_SKIP_TEMPLATES
#define PROF_SKIP_TEMPLATES 0
#endif
#ifndef PROF_SKIP_PROP
#define PROF_SKIP_PROP 0
#endif
#ifndef PROF_SKIP_DIR
#define PROF_SKIP_DIR 0
#endif
