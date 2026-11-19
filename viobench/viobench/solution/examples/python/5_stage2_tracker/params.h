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
#define SCALE_MAX 3.0    // |s| clamp
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

// ---- covariance arithmetic ----
#ifndef SYM_P
#define SYM_P 0          // 1: updates touch only the upper triangle; mirrored once per frame
#endif
#ifndef P_FLOAT
#define P_FLOAT 0        // 1: covariance stored and updated in float32
#endif
#ifndef DIR_OWN
#define DIR_OWN 0        // 1: one extra frame per window that runs only the direction update and velocity prior
#endif
#ifndef PROF_DIR_STOP
#define PROF_DIR_STOP 0
#endif

// ---- long-baseline tracks: previous window's patches tracked again (~14 frames) ----
#ifndef LONG_N
#define LONG_N 0         // patches per subset re-tracked from the previous window (0 = off)
#endif
#ifndef LONG_MIN
#define LONG_MIN 4       // minimum long tracks for an update
#endif
#ifndef LONG_MIN_PAR
#define LONG_MIN_PAR 1.0 // px: minimum mean long-baseline parallax
#endif
#ifndef LONG_EXT
#define LONG_EXT 1.8     // extrapolate the last translational flow (5 -> 14 frames) for the start point
#endif
#ifndef LONG_SIG
#define LONG_SIG 1.0     // scale on the long direction noise
#endif
#ifndef LONG_SIGN_SHORT
#define LONG_SIGN_SHORT 1  // sign of the long direction from the short one (skip the long update without one)
#endif
#ifndef LONG_AGREE
#define LONG_AGREE 0.8     // skip when |cos| between long and short directions is below this
#endif
#ifndef LONG_MODE
#define LONG_MODE 1      // 1: extra filter rows on p - p_kf2; 2: blend the long direction into the short one
#endif
#ifndef LONG_W
#define LONG_W 1.0       // weight of the long direction in the blend (relative to parallax x sqrt(n))
#endif

#ifndef RANK_FAST
#define RANK_FAST 1
#endif
#ifndef SUB_CAP
#define SUB_CAP 7              // max templates cut per subset (fuel-spike cap). lite/ultra: 3
#endif
#ifndef ULTRA
#define ULTRA 0                // 1: ultra-cheap schedule, one heavy step per frame
#endif
#ifndef AHEAD_FRAMES
#define AHEAD_FRAMES (ULTRA ? SPLIT + 1 : SPLIT + DIR_OWN)
#endif
#ifndef ULTRA_SPLIT
#define ULTRA_SPLIT 0          // ULTRA: split propagation and the direction rows over two frames each
#endif

// ---- stage 2: thrust-only model, vision relative rotation with an attitude clone, optional speed row ----
#ifndef THRUST_ONLY
#define THRUST_ONLY 0      // 1: quadrotor model, specific force = thrust along body z only (a_x, a_y ignored; no drag in the simulator)
#endif
#ifndef SIG_ACC_H
#define SIG_ACC_H 0.05     // THRUST_ONLY: isotropic acceleration noise density (model error); SIG_ACC acts along the thrust axis
#endif
#ifndef SIG_TILT0_HOVER
#define SIG_TILT0_HOVER 0.005  // THRUST_ONLY: initial tilt sigma after a hover start (body z is vertical)
#endif
#ifndef SIG_BAZ0
#define SIG_BAZ0 0.05      // THRUST_ONLY: thrust-axis accel bias sigma after the hover calibration
#endif
#ifndef ROT_CLONE
#define ROT_CLONE 0        // 1: attitude clone at the keyframe; vision measures the relative rotation (corrects attitude AND gyro bias)
#endif
#ifndef ROT_ITERS
#define ROT_ITERS 2        // alternations of rotation / direction fits
#endif
#ifndef ROT_DIR_CLONE
#define ROT_DIR_CLONE 1    // direction rows: attitude Jacobian on the clone (1) or the current attitude (0)
#endif
#ifndef SIG_ROTC_K
#define SIG_ROTC_K 2.0     // inflation of the rotation covariance from the fit
#endif
#ifndef SIG_ROTC_FLOOR
#define SIG_ROTC_FLOOR 0.003  // bearing sigma floor (rad)
#endif
#ifndef SIG_ROTC_MIN
#define SIG_ROTC_MIN 0.001 // rad: floor of the rotation measurement sigma per axis
#endif
#ifndef SPEED_ROW
#define SPEED_ROW 1        // 0: no assumed-depth speed row (scale from the motion model)
#endif
#ifndef ROT_SIGN
#define ROT_SIGN 1.0
#endif
#ifndef ROT_REFIT
#define ROT_REFIT 1       // 1: re-fit the direction on rotation-corrected bearings
#endif

// ---- stage 2: persistent frame-to-frame tracker ----
#ifndef TRK
#define TRK 0              // 1: features persist across windows and are tracked every TRK_HOP frames
#endif
#ifndef TRK_WIN
#define TRK_WIN 6          // frames per window (vision update + template refresh)
#endif
#ifndef TRK_HOP
#define TRK_HOP 1          // frames between tracking hops (TRK_WIN must be a multiple)
#endif
#ifndef TRK_FLOW
#define TRK_FLOW 1.0       // fraction of the last translational flow added to the predicted start
#endif
#ifndef TRK_MIN_DIST
#define TRK_MIN_DIST 10.0  // px: new features are not started closer than this to a live one
#endif
#ifndef TRK_MERGE
#define TRK_MERGE 3.0      // px: features closer than this at a keyframe are merged
#endif

// ---- stage 2: floor plane (metric scale) ----
#ifndef FLOOR
#define FLOOR 0            // 1: one state = world height of the floor; downward tracks give |d| in units of the height
#endif
#ifndef FLOOR_Z0
#define FLOOR_Z0 -1.5      // m: initial floor height relative to the start
#endif
#ifndef FLOOR_SIG0
#define FLOOR_SIG0 1.0     // m
#endif
#ifndef FLOOR_HMIN
#define FLOOR_HMIN 0.2     // m: minimum camera height above the floor
#endif
#ifndef FLOOR_DOWN
#define FLOOR_DOWN 0.26    // rays must point at least asin(this) below the horizon
#endif
#ifndef FLOOR_MIN_PAR
#define FLOOR_MIN_PAR 0.7  // px
#endif
#ifndef FLOOR_MIN_N
#define FLOOR_MIN_N 2      // downward tracks needed
#endif
#ifndef FLOOR_RANK
#define FLOOR_RANK 1       // use the (RANK+1)-th lowest point (0 = lowest)
#endif
#ifndef FLOOR_REL
#define FLOOR_REL 0.3      // relative sigma of one floor measurement
#endif
#ifndef FLOOR_CONV
#define FLOOR_CONV 0.3     // m: below this floor-height sigma the assumed-depth speed row is dropped when a floor row was used
#endif

// ---- stage 2: long-baseline rotation (attitude clone held over many windows) ----
#ifndef LROT
#define LROT 0             // 1 (needs TRK and ROT_CLONE): relative rotation to a long keyframe, every window
#endif
#ifndef LROT_MIN
#define LROT_MIN 5         // features needed for a rotation fit
#endif
#ifndef LROT_KEEP
#define LROT_KEEP 7        // refresh the long keyframe when fewer of its features survive
#endif
#ifndef LROT_MAX_AGE
#define LROT_MAX_AGE 200   // windows
#endif
#ifndef LROT_ITERS
#define LROT_ITERS 4
#endif
#ifndef LROT_PURE_PX
#define LROT_PURE_PX 0.35  // px rms: below this the pure-rotation fit explains the tracks
#endif
#ifndef LROT_MAX_Q
#define LROT_MAX_Q 0.3
#endif
#ifndef LROT_MAX
#define LROT_MAX 0.3       // rad
#endif
#ifndef LROT_EPI_MIN
#define LROT_EPI_MIN 9     // features needed for the rotation + direction fit (camera moved)
#endif
#ifndef LROT_GATE
#define LROT_GATE 3.0      // reject rotation corrections above this many sigma of the plausible gyro drift
#endif
#ifndef LROT_BG
#define LROT_BG 0.008      // rad/s: plausible gyro bias error for that gate
#endif
#ifndef DIR_NOISE_MODE
#define DIR_NOISE_MODE 0   // 1: direction-row noise scaled by the vision displacement (parallax x DIR_DEPTH / f) instead of the filter's |d|
#endif
#ifndef DIR_DEPTH
#define DIR_DEPTH 2.5
#endif
#ifndef TRK_PAR_KEY
#define TRK_PAR_KEY 0.0    // px: >0 = adaptive baseline, the keyframe is kept until the mean translational parallax reaches this
#endif
#ifndef TRK_MAX_WINS
#define TRK_MAX_WINS 5     // ... or this many windows have passed
#endif

// ---- stage 2: Manhattan heading from edge cells ----
#ifndef MANH
#define MANH 0
#endif
#ifndef MANH_MAX
#define MANH_MAX 48        // edge cells used per keyframe
#endif
#ifndef MANH_PRE_MIN
#define MANH_PRE_MIN 800.0 // sparse (ranking) tensor: minimum larger eigenvalue
#endif
#ifndef MANH_PRE_ANISO
#define MANH_PRE_ANISO 0.3 // sparse tensor: smaller / larger eigenvalue below this
#endif
#ifndef MANH_L1_MIN
#define MANH_L1_MIN 30.0   // dense tensor: larger eigenvalue per pixel
#endif
#ifndef MANH_ANISO
#define MANH_ANISO 8.0     // dense tensor: larger / smaller eigenvalue at least this
#endif
#ifndef MANH_W_MIN
#define MANH_W_MIN 0.35
#endif
#ifndef MANH_VERT
#define MANH_VERT 0.12
#endif
#ifndef MANH_INIT
#define MANH_INIT 8        // keyframes used to learn the grid azimuth
#endif
#ifndef MANH_INIT_MIN
#define MANH_INIT_MIN 6.0  // minimum total weight of the azimuth votes
#endif
#ifndef MANH_INIT_R
#define MANH_INIT_R 0.35   // minimum resultant / total weight (a random set gives ~0.1)
#endif
#ifndef MANH_STRIDE
#define MANH_STRIDE 5      // cell order stride (coprime with the cell count)
#endif
#ifndef MANH_TILT
#define MANH_TILT 1        // 1: the yaw row also carries the tilt-error coupling
#endif
#ifndef MANH_WIN
#define MANH_WIN 0.17      // rad: residual window (first pass)
#endif
#ifndef MANH_MIN
#define MANH_MIN 4
#endif
#ifndef MANH_SIG
#define MANH_SIG 0.05      // rad at 6 edges
#endif
#ifndef MANH_SIG_MIN
#define MANH_SIG_MIN 0.02
#endif

// ---- stage 2: structure (per-feature inverse depth) ----
#ifndef STRUCT
#define STRUCT 0           // 1 (needs TRK): features with a triangulated depth give 2 rows each on the full displacement
#endif
#ifndef STRUCT_MIN
#define STRUCT_MIN 3       // with at least this many depth features the direction / assumed-depth speed rows are skipped
#endif
#ifndef STRUCT_SIG_PX
#define STRUCT_SIG_PX 0.5  // bearing noise (tracking px)
#endif
#ifndef STRUCT_SIG_M
#define STRUCT_SIG_M 0.005 // m: floor of the point-row noise
#endif
#ifndef STRUCT_SIG_K
#define STRUCT_SIG_K 2.0   // inflation of the triangulated inverse-depth sigma
#endif
#ifndef STRUCT_MIN_PAR
#define STRUCT_MIN_PAR 3.0 // triangulate only with a parallax of this many bearing sigmas
#endif
#ifndef STRUCT_MIN_D
#define STRUCT_MIN_D 0.02  // m: minimum displacement for a triangulation
#endif
#ifndef STRUCT_ZMIN
#define STRUCT_ZMIN 0.15
#endif
#ifndef STRUCT_ZMAX
#define STRUCT_ZMAX 30.0
#endif
#ifndef STRUCT_GROW
#define STRUCT_GROW 0.05   // relative depth uncertainty added per window
#endif
#ifndef STRUCT_MISS
#define STRUCT_MISS 2      // windows of rejected rows before a depth is dropped
#endif
#ifndef STRUCT_S_MIN
#define STRUCT_S_MIN 0.1   // minimum relative sigma of a newly defined map scale
#endif
#ifndef STRUCT_PAR_PX
#define STRUCT_PAR_PX 1.5  // px: minimum translational parallax of a feature for a triangulation
#endif
#ifndef STRUCT_REL
#define STRUCT_REL 0.15    // relative sigma added to every triangulated inverse depth (displacement scale / direction error)
#endif
#ifndef STRUCT_BOOT_PX
#define STRUCT_BOOT_PX 3.0 // px: mean parallax needed to start a map from the direction + speed rows
#endif
#ifndef MANH_TRY
#define MANH_TRY 176       // dense edge tensors computed per keyframe at most (fuel cap)
#endif
#ifndef STRUCT_CHK_PX
#define STRUCT_CHK_PX 2.0  // px: mean parallax above which the map is checked against the direction fit
#endif
#ifndef STRUCT_CHK_COS
#define STRUCT_CHK_COS 0.85
#endif
#ifndef STRUCT_CHK_RATIO
#define STRUCT_CHK_RATIO 4.0
#endif
#ifndef STRUCT_KEEP_SPEED
#define STRUCT_KEEP_SPEED 0  // 1: keep the assumed-depth speed row as a scale anchor while the map is used
#endif
#ifndef STRUCT_SPEED_K
#define STRUCT_SPEED_K 1.0  // noise factor of that row while the map is used
#endif
#ifndef TRK_TRIES
#define TRK_TRIES 5        // TRK == 2: template cuts tried per top-up frame (fuel cap)
#endif
#ifndef TRK_HALF_RANK
#define TRK_HALF_RANK 1    // TRK == 2: rank half of the candidate grid per top-up frame
#endif
#ifndef TRK_SCHED
#define TRK_SCHED 1        // TRK == 2 schedule: 3 = flattest (hops every 3 frames, every heavy step on its own frame), 1 = hops every 3 frames + own update frame (flat fuel), 2 = hops every 2 frames, update on the G1 frame (h02), 0 = deferred rows
#endif
#ifndef TRK_TRIES_INIT
#define TRK_TRIES_INIT 8   // template cuts tried on the first frame after calibration
#endif

// ---- stage 2: model-based refinements ----
#ifndef MIDPOINT
#define MIDPOINT 0         // 1: rotate the specific force with the mid-interval attitude
#endif
#ifndef HOVER_INIT
#define HOVER_INIT 0       // 1: hover start uses SIG_V0_HOVER and tolerates a large gyro bias in the at-rest test
#endif
#ifndef SIG_V0_HOVER
#define SIG_V0_HOVER 0.05  // m/s
#endif
#ifndef VPRIOR_REL
#define VPRIOR_REL 0.0     // velocity-prior sigma grows by this x |v| (0 = constant sigma)
#endif
#ifndef TRK_DISP
#define TRK_DISP 0         // TRK == 2: residuals use the nominal displacement over the groups' own windows (no stagger bias)
#endif
