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
#define DEPTH_PRIOR 6.0        // m, typical upper depth indoors (sets the small-displacement sigma)
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
#define SIG_ACC 0.3 
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

// ---- FIXED_GAIN: constant-gain observer instead of the EKF (no covariance) ----
#ifndef FIXED_GAIN
#define FIXED_GAIN 0
#endif
#ifndef FG_KPERP
#define FG_KPERP 0.7     // fraction of the off-direction displacement residual removed per window
#endif
#ifndef FG_KPAR
#define FG_KPAR 0.7      // fraction of the speed-proxy residual along the motion direction
#endif
#ifndef FG_KSMALL
#define FG_KSMALL 0.3    // fraction of the displacement removed when the parallax says "barely moved"
#endif
#ifndef FG_KP
#define FG_KP 1.0        // position takes this x the gained residual
#endif
#ifndef FG_KV
#define FG_KV 1.5        // velocity takes this x the gained residual / T
#endif
#ifndef FG_KB
#define FG_KB 0.4        // world accel bias takes this x the gained residual / T^2
#endif
#ifndef FG_KVP
#define FG_KVP 0.25      // velocity leak per window (zero-velocity prior)
#endif
#ifndef FG_KG_ROT
#define FG_KG_ROT 0.05   // gyro-bias gain of the vision rotation residual, low parallax
#endif
#ifndef FG_KG_TRANS
#define FG_KG_TRANS 0.005 // same while translating (rotation/translation ambiguity)
#endif
#ifndef FG_KTH
#define FG_KTH 0.0       // also rotate the current attitude by this x the bias correction
#endif
#ifndef FG_GATE
#define FG_GATE 1.0      // reject windows whose off-direction residual exceeds this x |d|
#endif
#ifndef FG_WARM
#define FG_WARM 0        // windows of boosted gains after start
#endif
#ifndef FG_WARM_GAIN
#define FG_WARM_GAIN 0.0
#endif
#ifndef FG_TILT
#define FG_TILT 1        // horizontal accel residual corrects the tilt (else a world-frame accel bias)
#endif
#ifndef FG_KTB
#define FG_KTB 0.1       // tilt correction -> gyro bias integral gain
#endif
#ifndef FG_SIG0
#define FG_SIG0 0.0      // gain schedule: gains x 1/(1+(sig_dir/FG_SIG0)^2), 0 = constant gains
#endif
#ifndef FG_WPAR
#define FG_WPAR 1.0      // exponent of that weight on the speed gain
#endif
#ifndef CLONE_PHASE
#define CLONE_PHASE 1    // frame of the window that propagates the covariance and clones the position (0 = keyframe)
#endif
#ifndef LAZY_INJECT
#define LAZY_INJECT 1    // 1: inject filter corrections once per frame instead of after every update batch
#endif
#ifndef LAST_SUB
#define LAST_SUB 0       // >0: patches in the last template subset (rest split over the others); 0 = even split
#endif
#ifndef PROP_MODE
#define PROP_MODE 3      // 0: every tracking frame; 1: once on the idle frame before tracking; 2: idle + last tracking frame;
                         // 3: idle frame, propagated ahead over the tracking frames (no propagation while tracking)
#endif
#ifndef PROF_REP
#define PROF_REP 0
#endif
#ifndef RANK_SIMD
#define RANK_SIMD 1      // 1: score 4 candidate cells per SIMD pass (same scores, ~3x cheaper)
#endif
#ifndef VPRIOR_IDLE
#define VPRIOR_IDLE 2    // weak velocity prior: 0 on the direction-update frame, 1 idle frame, 2 next frame (keyframe), 3 frame 1
#endif
#ifndef FETCH_SIMD
#define FETCH_SIMD 1     // 1: SIMD pixel fetch (same values)
#endif
#ifndef DIR_NEXT
#define DIR_NEXT 0       // 1: run the direction update on the next (keyframe) frame
#endif

// ---- adaptive measurement noise ----
#ifndef ADAPT_R
#define ADAPT_R 0        // 1: scale the direction / speed noise online from normalized innovations
#endif
#ifndef ADAPT_BETA
#define ADAPT_BETA 0.05  // per-window learning rate (geometric)
#endif
#ifndef ADAPT_MIN
#define ADAPT_MIN 0.5
#endif
#ifndef ADAPT_MAX
#define ADAPT_MAX 4.0
#endif
#ifndef ADAPT_NIS_MAX
#define ADAPT_NIS_MAX 9.0  // clip single-window NIS (outliers must not blow the factor up)
#endif
#ifndef HUBER_C
#define HUBER_C 0.0      // >0: Huber-weight vision rows beyond this many sigma (0 = off)
#endif
// ---- log-scale state of the speed proxy ----
#ifndef SCALE_STATE
#define SCALE_STATE 0
#endif
#ifndef SIG_S0
#define SIG_S0 0.3       // initial sigma of the log scale
#endif
#ifndef SIG_S_RW
#define SIG_S_RW 0.02    // random walk of the log scale [1/sqrt(s)]
#endif
#ifndef SIG_S_PRIOR
#define SIG_S_PRIOR 0.5  // per-window pseudo-measurement s = 0 (0 = off)
#endif
#ifndef SCALE_MAX
#define SCALE_MAX 1.0    // |s| clamp
#endif
#ifndef PAR_MEDIAN
#define PAR_MEDIAN 0     // 1: median instead of mean per-track parallax (speed proxy, small-flow test)
#endif
#ifndef PAR_MEDIAN_K
#define PAR_MEDIAN_K 1.0 // factor on the median (keeps MIN_PAR_PX comparable to the mean)
#endif

// ---- generator motion prior (nose-first flight) ----
#ifndef NHC_SIG
#define NHC_SIG 0.0      // m/s: soft "camera sideways velocity ~ 0" per window (0 = off)
#endif
#ifndef NHC_SIG_Y
#define NHC_SIG_Y 0.0    // m/s: same for camera vertical velocity (0 = off)
#endif
#ifndef NHC_D_REL
#define NHC_D_REL 0.0    // relative sigma of "camera sideways displacement ~ 0" in each direction update (0 = off)
#endif
#ifndef NHC_NO_TH
#define NHC_NO_TH 0      // 1: the motion prior does not touch the attitude (velocity / position only)
#endif
#ifndef SIGN_FWD
#define SIGN_FWD 0.7     // >0: if |u . camera z| exceeds this, take the motion sign from "forward" (0 = cheirality only)
#endif
