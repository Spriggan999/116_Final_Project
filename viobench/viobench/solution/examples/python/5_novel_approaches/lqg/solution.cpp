// LQG estimator for viobench (EECS 116 Project 3, 5_novel_approaches/lqg). Built on the v8 estimator (copied below);
// LQG 0 reproduces v8 exactly.
//
// The estimate is treated as a virtual drone that a controller steers onto the real one:
//   plant        the estimated trajectory, integrated from the IMU (the corrected IMU is the control input).
//   estimator    (LQE) a steady-state Kalman filter: the covariance is NOT propagated at run time. At init the
//                Riccati recursion of v8's own 18-state error model is run over LQG_DESIGN_S seconds of a nominal
//                level, nose-first flight with nominal measurements; the converged covariance before each kind of
//                update (rotation rows per subset, direction rows, velocity prior) is stored in the heading frame.
//                At run time each update batch uses that fixed covariance, rotated to the current heading:
//                K = P H^T (H P H^T + R)^-1. No covariance maths per frame (v8 spends ~22k fuel on it).
//   controller   (LQR) the estimated error is not jumped into the state. Position / velocity / attitude errors are
//                driven to zero by an accelerometer correction u_a and a gyro correction u_w, applied over the
//                following frames; gains from a discrete Riccati equation (time constant LQR_TAU). Bias errors are
//                the controller's integral states: they stay in the IMU correction (a - b_a, w - b_g) from then on.
//   error signal LQG_VISION 1: v8's corner tracker (gyro-bias rows from the rotation residual, motion direction
//                + parallax speed from the tracks) and the velocity prior.
//                LQG_VISION 0: IMU only, no image access: hover calibration and the autopilot velocity prior.
//   THRUST_ONLY  1: the simulator has no drag, so the horizontal accelerometer axes carry only noise and bias;
//                integrate the thrust axis alone.
//
// v8 (unchanged except where marked LQG):
//
// How it works (one window = KF_LEN + SPLIT frames, + 1 in the lite build):
//   every frame   IMU: rotate with the gyro, integrate the accelerometer (gravity removed) -> attitude, velocity, position.
//   frame 0       score a grid of candidate cells for texture (SIMD) and keep the best N_TRACK + N_EXTRA.
//   frames 1..    cut 8x8 templates at 160x120 straight from the raw RGB frame, SPLIT subsets on SPLIT frames;
//                 frame 1 also clones the position in the filter.
//   frame KF_LEN-1  propagate the covariance once, ahead over the tracking frames.
//   tracking      predict where each patch moved with the gyro rotation, refine with Lucas-Kanade (SIMD); each track
//                 gives n = f_pred x f_meas, perpendicular to the camera displacement d since the clone; each subset also
//                 checks the gyro rotation (-> gyro bias).
//   direction     all tracks -> one motion direction u (smallest eigenvector of sum n n^T, one robust refit); sign by
//                 cheirality, or forward when u lies along the optical axis (the generator's autopilot flies
//                 nose-first); speed along u from the mean parallax (Sim(3) only needs a consistent scale).
//   filter        18-state error-state EKF [dp dv dtheta dba dbg dp_clone]: two constraints e . d = 0 (e perpendicular to u),
//                 one speed row, a small-motion row at low parallax, and a weak zero-velocity prior.
// Output: camera pose relative to the first camera frame and twist in the current camera frame.
#include "viobench_abi.h"
// ===== parameters =====
// VIO_LITE 1 builds the cheap second Pareto point (same method, ~45% of the fuel): fewer patches, a coarser candidate
// grid, an 11-frame window with the direction update on its own frame, float32 covariance.
#ifndef VIO_LITE
#define VIO_LITE 0
#endif
#if VIO_LITE
#define N_TRACK 11
#define GRID_X 12
#define GRID_Y 8
#define N_EXTRA 4
#define MIN_TRACKS 5
#define SPLIT 4
#define KF_LEN 6
#define DIR_OWN 1
#define VPRIOR_IDLE 1
#define P_FLOAT 1
#endif
#ifndef CALIB_SECONDS
#define CALIB_SECONDS 0.5      // s of IMU used for gravity direction and gyro bias
#endif
#ifndef CALIB_ACCEL_BIAS
#define CALIB_ACCEL_BIAS 0     // 1: at rest, take the accelerometer bias along gravity from | |mean accel| - g |
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
#ifndef DS
#define DS 2                   // tracking resolution: 320x240 / DS (160x120)
#endif
#ifndef SCORE_N
#define SCORE_N 4              // texture score: SCORE_N x SCORE_N raw samples per candidate cell
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
#ifndef RANK_FAST
#define RANK_FAST 1            // bounded-cost top-K selection of the candidate cells (flattens the ranking frame)
#endif
#ifndef SUB_CAP
#if VIO_LITE
#define SUB_CAP 3
#else
#define SUB_CAP 7
#endif              // max templates per subset: caps the catch-up after texture failures (fuel spike)
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
#ifndef SMALL_FLOW_PX
#define SMALL_FLOW_PX 0.5
#endif
#ifndef DEPTH_PRIOR
#define DEPTH_PRIOR 6.0        // m, typical upper depth indoors (sets the small-displacement sigma)
#endif
#ifndef PAR_SPEED
#define PAR_SPEED 8.0          // m: depth factor of the parallax speed proxy |d| ~ parallax * PAR_SPEED / f
#endif
#ifndef VEL_PRIOR
#define VEL_PRIOR 0            // velocity prior in the body frame, replacing the zero-velocity prior:
#endif                         //   1 learned from 2 s of IMU history, 2 constant nose-first cruise
#ifndef VP_CONST_FWD           // constant prior (VEL_PRIOR 2): mean body velocity and per-axis spread,
#define VP_CONST_FWD 0.241     // measured on training flights (scenes 20-49, velocity_prior/README)
#endif
#ifndef VP_CONST_SIG_X
#define VP_CONST_SIG_X 0.17
#endif
#ifndef VP_CONST_SIG_Y
#define VP_CONST_SIG_Y 0.05
#endif
#ifndef VP_CONST_SIG_Z
#define VP_CONST_SIG_Z 0.17
#endif
#ifndef VP_ATTITUDE
#define VP_ATTITUDE 0          // 1: prior rows also carry the attitude Jacobian (axis x v)
#endif
#ifndef VP_GATE_PX
#define VP_GATE_PX 0           // >0: cruise prior only when the last window's translational parallax exceeds this
#endif                         //     (px); still -> near-zero prior, no usable vision -> v8's weak zero prior
#ifndef VP_STILL_SIG
#define VP_STILL_SIG 0.1       // m/s per axis: velocity prior while vision sees no translation (stalled / hovering)
#endif
#ifndef VP_STALL_PX
#define VP_STALL_PX 0          // >0: stall detector: running mean of window parallax (px) below this -> still prior
#endif                         //     (a wind-stalled drone hovers for good; brief slow windows do not trigger it)
#ifndef VP_STALL_TAU
#define VP_STALL_TAU 8.0       // windows: averaging length of the stall detector
#endif
#ifndef VP_STALL_START
#define VP_STALL_START 2.0     // initial detector value in units of VP_STALL_PX (2 = start "moving", 0 = start "still":
#endif                         //     cruise only after vision has shown sustained translation)
#ifndef VP_STALL_NOTRACK
#define VP_STALL_NOTRACK 0     // 1: windows without enough tracks count as no translation (no evidence of motion)
#endif
#ifndef VMAX_RESET
#define VMAX_RESET 50.0        // m/s: a faster velocity estimate is treated as divergence (filter restart)
#endif
#ifndef VP_ADAPT_R
#define VP_ADAPT_R 0           // >0: inflate each prior axis by its running normalized innovation (NIS), with this
#endif                         //     averaging weight per application; the prior loosens on flights it does not fit
#ifndef VP_ADAPT_CAP
#define VP_ADAPT_CAP 100.0     // max variance inflation
#endif
#ifndef VP_ADAPT_MEAN
#define VP_ADAPT_MEAN 0        // >0: the forward cruise mean follows the estimated forward speed (weight per window)
#endif
#ifndef VP_SIG_SCALE
#define VP_SIG_SCALE 1.0       // multiplies the learned per-axis sigma (held-out RMSE)
#endif
#ifndef USE_PAR_SPEED
#define USE_PAR_SPEED 1        // 0: no parallax speed row (scale then comes from the accelerometer alone)
#endif
#ifndef PAR_SPEED_REL
#define PAR_SPEED_REL 0.5      // relative sigma of the speed proxy
#endif
#ifndef VPRIOR_SIG
#define VPRIOR_SIG 0.5         // m/s, weak zero-velocity prior per window (0 = off)
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
#ifndef VPRIOR_IDLE
#define VPRIOR_IDLE 2    // weak velocity prior: 2 = on the frame after the direction update (v8), 1 = on a template frame (lite)
#endif
#ifndef SIGN_FWD
#define SIGN_FWD 0.7     // >0: if |u . camera z| exceeds this, take the motion sign from "forward" (0 = cheirality only)
#endif
#ifndef P_FLOAT
#define P_FLOAT 0        // 1: covariance stored and updated in float32
#endif
#ifndef DIR_OWN
#define DIR_OWN 0        // 1: one extra frame per window that runs only the direction update and velocity prior
#endif
#ifndef LQG
#define LQG 1                 // 1: fixed-gain LQG observer + LQR IMU-correction controller; 0: v8's EKF
#endif
#ifndef LQG_VISION
#define LQG_VISION 1          // 0: IMU only (no image access), error signal from the velocity prior
#endif
#ifndef LQG_DESIGN_S
#define LQG_DESIGN_S 30.0     // s of nominal flight over which the Riccati recursion runs at init
#endif
#ifndef LQG_NOM_SPEED
#define LQG_NOM_SPEED 0.25    // m/s: nominal forward speed of the design flight
#endif
#ifndef LQG_NOM_PAR
#define LQG_NOM_PAR 1.0       // px: nominal translational parallax of a window
#endif
#ifndef LQG_NOM_TRACKS
#define LQG_NOM_TRACKS 15     // nominal tracks kept by the direction fit
#endif
#ifndef LQG_NOM_Q
#define LQG_NOM_Q 0.05        // nominal eigenvalue ratio of the direction fit
#endif
#ifndef LQR_TAU
#define LQR_TAU 0.1           // s: closed-loop time constant of the IMU-correction controller (0: apply at once)
#endif
#ifndef LQG_DIR_STATES
#define LQG_DIR_STATES 0      // 1: direction / speed / small-motion rows correct only position, velocity and the clone
#endif
#ifndef LQG_MIX_DIRS
#define LQG_MIX_DIRS 0        // 1: design flight cycles its motion through forward, up and sideways (not nose-first only)
#endif
#ifndef LQG_PRIOR_SPLIT
#define LQG_PRIOR_SPLIT (!LQG_VISION)   // 1 (IMU only): one velocity-prior axis per frame on 3 frames of the window
#endif
#ifndef LQG_WIND_HI
#define LQG_WIND_HI 0         // >0: wind-scaled cruise prior. The simulator has no drag, so a horizontal specific force in
#endif                        // the body frame during the start hover is a steady external force (wind), which stalls the
#ifndef LQG_WIND_LO           // autopilot. Prior forward speed x clamp((HI - w) / (HI - LO), 0, 1), w = |hover a_xy| (m/s^2)
#define LQG_WIND_LO 0.8
#endif
#ifndef LQG_TILT_ROT
#define LQG_TILT_ROT 0        // 1: body blocks (b_a, b_g) of the design covariance follow the body tilt, not only the heading
#endif
#ifndef LQG_SCHED
#define LQG_SCHED 0           // 1: gain scheduling: a second design for a hovering drone (zero speed, small-motion rows,
#endif                        // still prior), used while the stall detector says "still" and for small-motion updates
#ifndef LQG_SCHED_WIN
#define LQG_SCHED_WIN 0       // >0: start-up gain schedule. The design recursion starts from the (wide) initial priors and
#endif                        // the covariance of each of the first LQG_SCHED_WIN windows is stored (finite-horizon LQG)
#ifndef LQG_ADAPT
#define LQG_ADAPT 0           // >0: covariance scale per update type, prior 1, updated in flight from normalized
#endif                        // innovations: log(lam) += LQG_ADAPT * (NIS / rows - 1)
#ifndef LQG_ADAPT_MIN
#define LQG_ADAPT_MIN 0.25
#endif
#ifndef LQG_ADAPT_MAX
#define LQG_ADAPT_MAX 16.0
#endif
#ifndef LQG_STILL_CAL
#define LQG_STILL_CAL 0       // 1: motion-gated calibration: while the IMU (and vision) say the drone is still, hold the pose
#endif                        // and average gyro bias and gravity direction; ends for good at the first sign of motion
#ifndef STILL_GYRO
#define STILL_GYRO 0.08       // rad/s: frame-mean gyro deviation from the running still mean that counts as motion
#endif
#ifndef STILL_ACC
#define STILL_ACC 2.5         // m/s^2: frame-mean |accel| deviation from g that counts as motion (the hover sags ~2)
#endif
#ifndef VPRIOR_REL
#define VPRIOR_REL 0.0        // zero-velocity prior sigma grows by this x |v| (no cruise-speed assumption)
#endif
#ifndef THRUST_ONLY
#define THRUST_ONLY 0         // 1: integrate only the thrust-axis (body z) accelerometer
#endif
#if LQG
#undef P_FLOAT
#define P_FLOAT 0             // the design recursion runs in double
#endif
// ===== end of parameters =====

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

namespace {

constexpr double GRAVITY = 9.81;

// ================= small linear algebra =================
struct Mat3 { double m[9]; };
struct Vec3 { double x, y, z; };

inline Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 scale(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double norm(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3 mul(const Mat3& R, Vec3 v) {
    return {R.m[0] * v.x + R.m[1] * v.y + R.m[2] * v.z, R.m[3] * v.x + R.m[4] * v.y + R.m[5] * v.z,
            R.m[6] * v.x + R.m[7] * v.y + R.m[8] * v.z};
}
inline Vec3 mulT(const Mat3& R, Vec3 v) {
    return {R.m[0] * v.x + R.m[3] * v.y + R.m[6] * v.z, R.m[1] * v.x + R.m[4] * v.y + R.m[7] * v.z,
            R.m[2] * v.x + R.m[5] * v.y + R.m[8] * v.z};
}
inline Mat3 mul(const Mat3& A, const Mat3& B) {
    Mat3 C;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            C.m[3 * i + j] = A.m[3 * i] * B.m[j] + A.m[3 * i + 1] * B.m[3 + j] + A.m[3 * i + 2] * B.m[6 + j];
    return C;
}
inline Mat3 transpose(const Mat3& A) { return {{A.m[0], A.m[3], A.m[6], A.m[1], A.m[4], A.m[7], A.m[2], A.m[5], A.m[8]}}; }
inline Mat3 eye() { return {{1, 0, 0, 0, 1, 0, 0, 0, 1}}; }
inline Mat3 skew(Vec3 v) { return {{0, -v.z, v.y, v.z, 0, -v.x, -v.y, v.x, 0}}; }
inline Mat3 mscale(const Mat3& A, double s) { Mat3 B; for (int i = 0; i < 9; i++) B.m[i] = A.m[i] * s; return B; }

Mat3 exp_so3(Vec3 w) {  // exact (used for filter corrections, not in the IMU loop)
    double th2 = dot(w, w), a, b;
    if (th2 < 1e-12) { a = 1.0 - th2 / 6.0; b = 0.5 - th2 / 24.0; }
    else { double th = std::sqrt(th2); a = std::sin(th) / th; b = (1.0 - std::cos(th)) / th2; }
    double x = w.x, y = w.y, z = w.z;
    return {{1 - b * (y * y + z * z), -a * z + b * x * y, a * y + b * x * z,
             a * z + b * x * y, 1 - b * (x * x + z * z), -a * x + b * y * z,
             -a * y + b * x * z, a * x + b * y * z, 1 - b * (x * x + y * y)}};
}

// R <- R * exp([w]x) to second order (|w| ~ 0.01 rad per IMU sample: error ~1e-7 rad)
inline void rotate_small(Mat3& R, Vec3 w) {
    double x = w.x, y = w.y, z = w.z;
    double a = 1.0 - (x * x + y * y + z * z) / 6.0, b = 0.5;
    Mat3 E = {{1 - b * (y * y + z * z), -a * z + b * x * y, a * y + b * x * z,
               a * z + b * x * y, 1 - b * (x * x + z * z), -a * x + b * y * z,
               -a * y + b * x * z, a * x + b * y * z, 1 - b * (x * x + y * y)}};
    R = mul(R, E);
}

void orthonormalize(Mat3& R) {
    Vec3 r0{R.m[0], R.m[1], R.m[2]}, r1{R.m[3], R.m[4], R.m[5]};
    r0 = scale(r0, 1.0 / norm(r0));
    r1 = sub(r1, scale(r0, dot(r0, r1)));
    r1 = scale(r1, 1.0 / norm(r1));
    Vec3 r2 = cross(r0, r1);
    R = {{r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z}};
}

void to_quat(const Mat3& R, double q[4]) {  // x y z w
    double tr = R.m[0] + R.m[4] + R.m[8];
    if (tr > 0) {
        double s = std::sqrt(tr + 1.0) * 2;
        q[3] = 0.25 * s; q[0] = (R.m[7] - R.m[5]) / s; q[1] = (R.m[2] - R.m[6]) / s; q[2] = (R.m[3] - R.m[1]) / s;
    } else if (R.m[0] > R.m[4] && R.m[0] > R.m[8]) {
        double s = std::sqrt(1.0 + R.m[0] - R.m[4] - R.m[8]) * 2;
        q[3] = (R.m[7] - R.m[5]) / s; q[0] = 0.25 * s; q[1] = (R.m[1] + R.m[3]) / s; q[2] = (R.m[2] + R.m[6]) / s;
    } else if (R.m[4] > R.m[8]) {
        double s = std::sqrt(1.0 + R.m[4] - R.m[0] - R.m[8]) * 2;
        q[3] = (R.m[2] - R.m[6]) / s; q[0] = (R.m[1] + R.m[3]) / s; q[1] = 0.25 * s; q[2] = (R.m[5] + R.m[7]) / s;
    } else {
        double s = std::sqrt(1.0 + R.m[8] - R.m[0] - R.m[4]) * 2;
        q[3] = (R.m[3] - R.m[1]) / s; q[0] = (R.m[2] + R.m[6]) / s; q[1] = (R.m[5] + R.m[7]) / s; q[2] = 0.25 * s;
    }
    double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (!(n > 1e-12)) { q[0] = q[1] = q[2] = 0; q[3] = 1; return; }
    for (int i = 0; i < 4; i++) q[i] /= n;
}

// ================= configuration =================
int W = 320, H = 240;          // delivered frame
int SW = 160, SH = 120;        // tracking resolution (DS)
double fx_s, fy_s, cx_s, cy_s; // intrinsics at tracking resolution
Mat3 R_ci, R_ic;               // IMU <-> camera rotation
Vec3 r_cam_in_imu;             // lever arm: camera centre in IMU coordinates

// ================= pixel access straight from the raw RGB frame =================
const uint8_t* frame_px = nullptr;  // valid during step()

// value of tracking-resolution pixel (x, y): mean of the 2x2 green samples at the centre of its DS x DS cell
inline float px_ds(int x, int y) {
    const int off = DS / 2 - 1;
    const uint8_t* r0 = frame_px + ((y * DS + off) * W + x * DS + off) * 3 + 1;
    return 0.5f * (float)(r0[0] + r0[W * 3 + 3]);   // diagonal pair
}

// fill out[0..n) with tracking-resolution pixels (x0..x0+n, y): one address computation per row
// (SIMD path: may write up to 3 floats past out[n-1]; callers keep slack after the last row)
inline void fetch_row(int x0, int y, int n, float* out) {
    const int off = DS / 2 - 1;
#if defined(__wasm_simd128__)
    {   // 4 pixels per pass: green bytes 6 apart -> one shuffle of two loads per sample row
        const uint8_t* r0 = frame_px + ((y * DS + off) * W + x0 * DS + off) * 3 + 1;
        const uint8_t* r1 = r0 + W * 3 + 3;
        const v128_t half = wasm_f32x4_splat(0.5f);
        for (int i = 0; i < n; i += 4, r0 += 24, r1 += 24) {
            v128_t a = wasm_i8x16_shuffle(wasm_v128_load(r0), wasm_v128_load(r0 + 8), 0, 6, 12, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            v128_t b = wasm_i8x16_shuffle(wasm_v128_load(r1), wasm_v128_load(r1 + 8), 0, 6, 12, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            v128_t s = wasm_i32x4_add(wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(a)),
                                      wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(b)));
            wasm_v128_store(out + i, wasm_f32x4_mul(half, wasm_f32x4_convert_i32x4(s)));
        }
        return;
    }
#endif
    const uint8_t* r0 = frame_px + ((y * DS + off) * W + x0 * DS + off) * 3 + 1;
    const uint8_t* r1 = r0 + W * 3 + 3;
    for (int i = 0; i < n; i++, r0 += DS * 3, r1 += DS * 3) out[i] = 0.5f * (float)(r0[0] + r1[0]);
}

// ================= filter state =================
constexpr int IP = 0, IV = 3, ITH = 6, IBA = 9, IBG = 12, IPC = 15, NS = 18;
#if P_FLOAT
using PT = float;          // covariance storage (lite): f32x4 SIMD, half the instructions per row operation
#else
using PT = double;
#endif
PT P[NS * NS];
double dx[NS];             // accumulated error-state correction (injected after an update batch)

Mat3 R_wb;                 // body -> world (nominal)
Vec3 p_w, v_w, ba, bg;     // nominal IMU position, velocity, biases
Vec3 p_kf;                 // cloned position at the keyframe
Mat3 R_wb_kf[SPLIT];       // body orientation when each template subset was cut
Mat3 R_wb_clone;           // body orientation at the position clone
Vec3 omega_b;              // latest bias-corrected angular rate
#if VEL_PRIOR == 1
#include "velocity_prior_weights.h"   // VP_BIN, VP_HIST, VP_IN, VP_H1, VP_H2, weights, input mean/std, sigma
float vp_bins[VP_HIST][6];             // ring of 50 ms IMU means (accel / g, gyro), raw as delivered
int vp_head = 0, vp_count = 0, vp_n = 0;
double vp_sum[6];
#endif
const Vec3 g_w{0, 0, -GRAVITY};
Mat3 R_wc0;                // first camera orientation
Vec3 p_wc0;                // first camera position

#if LQG
PT P_rot[SPLIT][NS * NS], P_dir[NS * NS], P_pri[NS * NS], P_pri_ax[3][NS * NS];
int vp_axis = -1;
constexpr int NSW = LQG_SCHED_WIN > 0 ? LQG_SCHED_WIN : 1;
PT P_rot_s[NSW][SPLIT][NS * NS], P_dir_s[NSW][NS * NS], P_pri_s[NSW][NS * NS], P_ax_s[NSW][3][NS * NS];
int run_win = 0;             // windows since start (selects the start-up gains)
int design_win = 0;          // the same count in the design flight
inline bool sched_on(int w) { return LQG_SCHED_WIN > 0 && w < LQG_SCHED_WIN; }
double lam[3] = {1, 1, 1};   // LQG_ADAPT covariance scale: 0 rotation rows, 1 direction / small-motion rows, 2 prior
int cur_group = 0;
bool still_mode = false;     // LQG_STILL_CAL: still before the first motion
double still_g[3], still_a[3];
int still_n = 0;
PT P_rot_h[SPLIT][NS * NS], P_dir_h[NS * NS], P_pri_h[NS * NS];   // hover design (LQG_SCHED)
bool lqg_hover = false;      // stall detector says still: use the hover design
double wind_scale = 1.0;     // LQG_WIND_HI: fraction of the cruise speed the autopilot can still make            // >= 0: the velocity prior applies only this body axis (LQG_PRIOR_SPLIT)  // design covariance before each update batch (heading frame)
const PT* P_use = nullptr;   // design covariance of the batch being collected
bool trans_only = false;     // this batch may correct only p, v, p_clone (LQG_DIR_STATES)
bool designing = false;      // true while the Riccati recursion runs at init (rows go to the EKF)
constexpr int MAXB = 4;      // rows per batch
double bH[MAXB][NS], by[MAXB], bR[MAXB];
int bn = 0;
Vec3 c_p{0, 0, 0}, c_v{0, 0, 0}, c_th{0, 0, 0};   // error still to be removed by the controller (world frame)
double K_p = 0, K_v = 0, K_th = 0;                 // LQR gains (per frame)
double frame_T = 0;                                // IMU time integrated this frame
double fr_g[3] = {0, 0, 0}, fr_a[3] = {0, 0, 0};   // raw IMU sums this frame (still detector)
int fr_n = 0;
Vec3 frame_f{0, 0, 0};                             // summed world specific force this frame
int frame_n = 0;
#endif
// estimate including corrections the controller has not applied yet (what measurements are compared with)
inline Vec3 est_p() {
#if LQG
    return add(p_w, c_p);
#else
    return p_w;
#endif
}
inline Vec3 est_v() {
#if LQG
    return add(v_w, c_v);
#else
    return v_w;
#endif
}

// covariance-propagation accumulators (propagated lazily, once per window)
double prop_debt = 0;
bool prop_ahead_next = false;
double acc_T = 0;
int acc_n = 0;
Mat3 acc_R{{0}};
Vec3 acc_f{0, 0, 0};
double win_T[SPLIT];       // IMU time since each subset's keyframe

bool initialized = false;
double calib_t0 = -1, last_t = 0;
int calib_n = 0;
double sum_a[3], sum_a2[3], sum_g[3], sum_g2[3];

int phase = 0;             // position in the window cycle (KF_LEN + SPLIT frames)
bool have_kf = false;
int last_tracks = 0, last_used = 0;  // diagnostics
double last_quality = 0, last_parallax = 0;

// ================= config parsing =================
const char* find_key(const char* json, uint32_t size, const char* key) {
    size_t k = std::strlen(key);
    for (const char* p = json; p + k <= json + size; p++)
        if (std::memcmp(p, key, k) == 0) return p + k;
    return nullptr;
}
const char* read_number(const char* p, const char* end, double* out) {
    while (p < end && !((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' || *p == '.')) p++;
    if (p >= end) return nullptr;
    char buf[64];
    int n = 0;
    while (p < end && n < 63 && ((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' || *p == '.' || *p == 'e' || *p == 'E'))
        buf[n++] = *p++;
    buf[n] = 0;
    *out = std::strtod(buf, nullptr);
    return p;
}
bool parse_scalar(const char* json, uint32_t size, const char* key, double* out) {
    const char* p = find_key(json, size, key);
    return p && read_number(p, json + size, out);
}

// ================= EKF =================

// dst[0..NS) += a * src[0..NS)  (vectorizes to f64x2)
inline void axpy(PT* __restrict dst, PT a, const PT* __restrict src) {
    for (int j = 0; j < NS; j++) dst[j] += a * src[j];
}

// rows of F applied to the rows of M: out = F M. F = I + the nonzero 3x3 blocks of the window transition.
struct FBlocks { double T; Mat3 A1, A2, A3, B1, B2, B3, C3; };
void apply_F(const FBlocks& f, const PT* M, PT* out) {
    std::memcpy(out, M, sizeof(PT) * NS * NS);
    for (int i = 0; i < 3; i++) {
        PT* op = out + (IP + i) * NS;   // dp  += T dv + A1 dth + A2 dba + A3 dbg
        PT* ov = out + (IV + i) * NS;   // dv  += B1 dth + B2 dba + B3 dbg
        PT* ot = out + (ITH + i) * NS;  // dth += C3 dbg
        axpy(op, (PT)f.T, M + (IV + i) * NS);
        for (int c = 0; c < 3; c++) {
            axpy(op, (PT)f.A1.m[3 * i + c], M + (ITH + c) * NS);
            axpy(op, (PT)f.A2.m[3 * i + c], M + (IBA + c) * NS);
            axpy(op, (PT)f.A3.m[3 * i + c], M + (IBG + c) * NS);
            axpy(ov, (PT)f.B1.m[3 * i + c], M + (ITH + c) * NS);
            axpy(ov, (PT)f.B2.m[3 * i + c], M + (IBA + c) * NS);
            axpy(ov, (PT)f.B3.m[3 * i + c], M + (IBG + c) * NS);
            axpy(ot, (PT)f.C3.m[3 * i + c], M + (IBG + c) * NS);
        }
    }
}

// propagate over the accumulated IMU interval (T, mean rotation, mean world specific force): P = F P F^T + Q
void kf_propagate() {
#if LQG
    if (!designing) { acc_T = 0; acc_n = 0; acc_R = Mat3{{0}}; acc_f = {0, 0, 0}; prop_ahead_next = false; return; }
#endif
    if (acc_n == 0 || !(acc_T > 0)) return;
    double T = acc_T - prop_debt;     // part of this interval may already have been propagated ahead
    prop_debt = 0;
    if (prop_ahead_next) {            // propagate ahead over the frames until the direction update
        double ahead = (SPLIT + DIR_OWN) * (acc_T / acc_n) * 10.0;
        T += ahead; prop_debt = ahead; prop_ahead_next = false;
    }
    if (!(T > 0)) { acc_T = 0; acc_n = 0; acc_R = Mat3{{0}}; acc_f = {0, 0, 0}; return; }
    Mat3 Rbar = mscale(acc_R, 1.0 / acc_n);
    Vec3 f_w = scale(acc_f, 1.0 / acc_n);
    acc_T = 0; acc_n = 0; acc_R = Mat3{{0}}; acc_f = {0, 0, 0};
    Mat3 Fx = skew(f_w), FxR = mul(Fx, Rbar);
    FBlocks f{T, mscale(Fx, -0.5 * T * T), mscale(Rbar, -0.5 * T * T), mscale(FxR, T * T * T / 6.0),
              mscale(Fx, -T), mscale(Rbar, -T), mscale(FxR, 0.5 * T * T), mscale(Rbar, -T)};
    static PT Q[NS * NS], Qt[NS * NS];
    apply_F(f, P, Q);                                  // Q = F P
    for (int i = 0; i < NS; i++)                       // Qt = Q^T = P F^T
        for (int j = 0; j < NS; j++) Qt[j * NS + i] = Q[i * NS + j];
    apply_F(f, Qt, P);                                 // P = F P F^T
    double qa = SIG_ACC * SIG_ACC, qg = SIG_GYR * SIG_GYR, qba = SIG_BA * SIG_BA, qbg = SIG_BG * SIG_BG;
    for (int i = 0; i < 3; i++) {
        P[(IP + i) * NS + IP + i] += qa * T * T * T / 3.0;
        P[(IP + i) * NS + IV + i] += qa * T * T / 2.0;
        P[(IV + i) * NS + IP + i] += qa * T * T / 2.0;
        P[(IV + i) * NS + IV + i] += qa * T;
        P[(ITH + i) * NS + ITH + i] += qg * T;
        P[(IBA + i) * NS + IBA + i] += qba * T;
        P[(IBG + i) * NS + IBG + i] += qbg * T;
    }
#if THRUST_ONLY
    for (int i = 0; i < 2; i++)   // horizontal accelerometer bias does not enter the model
        for (int j = 0; j < NS; j++) P[(IBA + i) * NS + j] = P[j * NS + IBA + i] = 0;
#endif
}

// sequential scalar update: innovation y (measured - predicted), measurement row H, noise Rn
bool ekf_update_row(const double Hrow[NS], double y, double Rn) {
    double PH[NS];
    int nz[NS], nnz = 0;
    for (int j = 0; j < NS; j++) if (Hrow[j] != 0.0) nz[nnz++] = j;
    for (int i = 0; i < NS; i++) {
        double s = 0;
        for (int k = 0; k < nnz; k++) s += P[i * NS + nz[k]] * Hrow[nz[k]];
        PH[i] = s;
    }
    double S = Rn;
    for (int k = 0; k < nnz; k++) S += Hrow[nz[k]] * PH[nz[k]];
    if (!(S > 0)) return false;
    double yc = y;
    for (int k = 0; k < nnz; k++) yc -= Hrow[nz[k]] * dx[nz[k]];
    if (yc * yc / S > CHI2_GATE) return false;   // chi-square gate
    double g = yc / S, is = 1.0 / S;
    for (int i = 0; i < NS; i++) dx[i] += PH[i] * g;
    PT pht[NS];
    for (int i = 0; i < NS; i++) pht[i] = (PT)PH[i];
    for (int i = 0; i < NS; i++) axpy(P + i * NS, (PT)(-PH[i] * is), pht);
    return true;
}

#if LQG
// LQE: one batch of rows with the fixed design covariance, rotated from the heading frame to the world:
// world blocks (p, v, theta, p_clone) turn with the heading, body blocks (b_a, b_g) do not.
inline void heading_rot(double* h, double c, double s) {
    static const int blocks[4] = {IP, IV, ITH, IPC};
    for (int bi = 0; bi < 4; bi++) {
        const int b = blocks[bi];
        double x = h[b], y = h[b + 1];
        h[b] = c * x - s * y; h[b + 1] = s * x + c * y;
    }
}
void lqg_batch() {
    const int m = bn;
    bn = 0;
    if (m == 0 || P_use == nullptr) return;
    double hx = R_wb.m[0], hy = R_wb.m[3], hn = std::sqrt(hx * hx + hy * hy);
    double c = hn > 1e-9 ? hx / hn : 1.0, s = hn > 1e-9 ? hy / hn : 0.0;
    double G[MAXB][NS], y[MAXB];
#if LQG_TILT_ROT
    // body frame vs heading frame: R_t = Rz(psi)^T R_wb (the tilt). A body vector b is R_t b in the design's body frame.
    const Mat3 Rz{{c, -s, 0, s, c, 0, 0, 0, 1}};
    const Mat3 Rt = mul(transpose(Rz), R_wb);
#endif
    for (int r = 0; r < m; r++) {
        double h0[NS];
        for (int j = 0; j < NS; j++) h0[j] = bH[r][j];
        heading_rot(h0, c, -s);                       // world -> heading frame
#if LQG_TILT_ROT
        for (int b = IBA; b <= IBG; b += 3) {         // h_body . b_body = (R_t h_body) . (R_t b_body)
            Vec3 hb = mul(Rt, Vec3{h0[b], h0[b + 1], h0[b + 2]});
            h0[b] = hb.x; h0[b + 1] = hb.y; h0[b + 2] = hb.z;
        }
#endif
        for (int i = 0; i < NS; i++) G[r][i] = 0;
        const double ls = LQG_ADAPT > 0 ? lam[cur_group] : 1.0;
        for (int j = 0; j < NS; j++) {
            if (h0[j] == 0.0) continue;
            for (int i = 0; i < NS; i++) G[r][i] += ls * (double)P_use[i * NS + j] * h0[j];
        }
        heading_rot(G[r], c, s);                      // P H^T in the world frame
#if LQG_TILT_ROT
        for (int b = IBA; b <= IBG; b += 3) {         // design body frame -> actual body frame
            Vec3 gb = mulT(Rt, Vec3{G[r][b], G[r][b + 1], G[r][b + 2]});
            G[r][b] = gb.x; G[r][b + 1] = gb.y; G[r][b + 2] = gb.z;
        }
#endif
        y[r] = by[r];
        for (int j = 0; j < NS; j++) y[r] -= bH[r][j] * dx[j];   // earlier batches of this frame
    }
    double S[MAXB][MAXB];
    int keep[MAXB], k = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            double v = 0;
            for (int q = 0; q < NS; q++) v += bH[i][q] * G[j][q];
            S[i][j] = v + (i == j ? bR[i] : 0.0);
        }
        if (S[i][i] > 0 && y[i] * y[i] / S[i][i] <= CHI2_GATE) keep[k++] = i;   // chi-square gate per row
    }
    if (k == 0) return;
    if (trans_only)    // restrict the gain (S above keeps the full H P H^T)
        for (int r = 0; r < m; r++)
            for (int q = ITH; q < IPC; q++) G[r][q] = 0;
    double A[MAXB][MAXB + 1];                          // solve S_kk w = y_k (Gaussian elimination, partial pivoting)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) A[i][j] = S[keep[i]][keep[j]];
        A[i][k] = y[keep[i]];
    }
    for (int col = 0; col < k; col++) {
        int piv = col;
        for (int i = col + 1; i < k; i++) if (std::fabs(A[i][col]) > std::fabs(A[piv][col])) piv = i;
        if (!(std::fabs(A[piv][col]) > 1e-300)) return;
        if (piv != col) for (int j = 0; j <= k; j++) { double t = A[col][j]; A[col][j] = A[piv][j]; A[piv][j] = t; }
        for (int i = col + 1; i < k; i++) {
            double f = A[i][col] / A[col][col];
            for (int j = col; j <= k; j++) A[i][j] -= f * A[col][j];
        }
    }
    double w[MAXB];
    for (int i = k - 1; i >= 0; i--) {
        double v = A[i][k];
        for (int j = i + 1; j < k; j++) v -= A[i][j] * w[j];
        w[i] = v / A[i][i];
    }
    for (int i = 0; i < k; i++)
        for (int q = 0; q < NS; q++) dx[q] += G[keep[i]][q] * w[i];
    if (LQG_ADAPT > 0) {   // covariance matching: NIS = y^T S^-1 y should average the number of rows
        double nis = 0;
        for (int i = 0; i < k; i++) nis += y[keep[i]] * w[i];
        double r = nis / k;
        if (r > 9.0) r = 9.0;
        double l = lam[cur_group] * std::exp(LQG_ADAPT * (r - 1.0));
        if (l < LQG_ADAPT_MIN) l = LQG_ADAPT_MIN;
        if (l > LQG_ADAPT_MAX) l = LQG_ADAPT_MAX;
        if (std::isfinite(l)) lam[cur_group] = l;
    }
}
#endif

bool kf_update_row(const double Hrow[NS], double y, double Rn) {
#if LQG
    if (!designing) {
        if (bn < MAXB) {
            for (int j = 0; j < NS; j++) bH[bn][j] = Hrow[j];
            by[bn] = y; bR[bn] = Rn; bn++;
        }
        return true;
    }
#endif
    return ekf_update_row(Hrow, y, Rn);
}

bool inject_pending = false;
void kf_inject_now();
// corrections stay in dx (later rows see them through y - H dx) and are injected once, at the end of the frame
void kf_inject() {
#if LQG
    if (!designing) lqg_batch();
#endif
    inject_pending = true;
}
void kf_inject_now() {
    inject_pending = false;
#if LQG
    if (!designing) {
        // biases and the clone: integral states, corrected at once (the IMU correction from now on);
        // position, velocity, attitude: handed to the controller (LQR_TAU 0: at once, like the EKF)
        Vec3 dp{dx[0], dx[1], dx[2]}, dv{dx[3], dx[4], dx[5]}, dth{dx[6], dx[7], dx[8]};
        ba = add(ba, Vec3{dx[9], dx[10], dx[11]});
        bg = add(bg, Vec3{dx[12], dx[13], dx[14]});
        p_kf = add(p_kf, Vec3{dx[15], dx[16], dx[17]});
#if THRUST_ONLY
        ba.x = ba.y = 0;
#endif
        for (int i = 0; i < NS; i++) dx[i] = 0;
        if (LQR_TAU > 0) {
            c_p = add(c_p, dp); c_v = add(c_v, dv); c_th = add(c_th, dth);
        } else {
            p_w = add(p_w, dp); v_w = add(v_w, dv);
            R_wb = mul(exp_so3(dth), R_wb);
            orthonormalize(R_wb);
        }
        return;
    }
#endif
    p_w = add(p_w, Vec3{dx[0], dx[1], dx[2]});
    v_w = add(v_w, Vec3{dx[3], dx[4], dx[5]});
    R_wb = mul(exp_so3(Vec3{dx[6], dx[7], dx[8]}), R_wb);
    orthonormalize(R_wb);
    ba = add(ba, Vec3{dx[9], dx[10], dx[11]});
    bg = add(bg, Vec3{dx[12], dx[13], dx[14]});
    p_kf = add(p_kf, Vec3{dx[15], dx[16], dx[17]});
    for (int i = 0; i < NS; i++) dx[i] = 0;
    for (int i = 0; i < NS; i++)
        for (int j = i + 1; j < NS; j++) {
            PT s = (PT)0.5 * (P[i * NS + j] + P[j * NS + i]);
            P[i * NS + j] = P[j * NS + i] = s;
        }
}

void kf_clone() {
    p_kf = est_p();
#if LQG
    if (!designing) return;
#endif
    for (int i = 0; i < NS; i++)
        for (int j = 0; j < 3; j++) P[i * NS + IPC + j] = P[i * NS + IP + j];
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < NS; i++) P[(IPC + j) * NS + i] = P[(IP + j) * NS + i];
}

// row helper: H on position minus clone (camera displacement d = p - p_kf + lever change)
inline void row_d(double Hrow[NS], Vec3 e) {
    for (int i = 0; i < NS; i++) Hrow[i] = 0;
    Hrow[IP] = e.x; Hrow[IP + 1] = e.y; Hrow[IP + 2] = e.z;
    Hrow[IPC] = -e.x; Hrow[IPC + 1] = -e.y; Hrow[IPC + 2] = -e.z;
}

// ================= patches =================
// A patch is a PS x PS template (PS = 8) cut from the keyframe at tracking resolution. Positions are top-left corners;
// the patch centre is at origin + (PS - 1) / 2.
constexpr int PS = PATCH_SIZE;
constexpr int PN = PS * PS;
constexpr int NCAND = GRID_X * GRID_Y;
constexpr int MARGIN = CROP_MARGIN;           // crop margin around the predicted origin
constexpr int CSZ = PS + 2 * MARGIN + 1;      // crop side
constexpr double PC = (PS - 1) / 2.0;         // centre offset
static_assert(PS % 4 == 0, "PATCH_SIZE must be a multiple of 4 (SIMD rows)");
static_assert(KF_LEN > SPLIT, "templates (phases 1..SPLIT) must be cut before tracking starts at KF_LEN");

struct alignas(16) Patch {
    float T[PN], GX[PN], GY[PN];    // template and its gradients
    int ox, oy;                     // top-left corner (keyframe, tracking resolution)
    double i00, i01, i11;           // inverse structure tensor
    float sgx, sgy;                 // sums of the gradients (for the brightness-offset term)
    float tstd;
    int sub;                        // template subset (cut / tracked on frame offset sub)
};
Patch patches[N_TRACK];
int n_patches = 0;

constexpr int KTOP = N_TRACK + N_EXTRA < GRID_X * GRID_Y ? N_TRACK + N_EXTRA : GRID_X * GRID_Y;
int top[KTOP];
float topv[KTOP];
int nk = 0, next_cand = 0;
double samp_s = 0, samp_ss = 0;  // sums of the ranking samples (frame contrast)
int samp_n = 0;
double eig_thresh = MIN_EIG;     // texture threshold for this window (scaled with frame contrast)
float cscore[NCAND];
int cx_[NCAND], cy_[NCAND];

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
inline float hsum(v128_t v) {
    return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) + wasm_f32x4_extract_lane(v, 2) +
           wasm_f32x4_extract_lane(v, 3);
}
#endif

// Texture score of one candidate cell straight from the raw frame: an SxS grid of green samples spread over the patch
// (S = SCORE_N), structure tensor from neighbour differences, score = its smaller eigenvalue.
inline float score_cell(int x, int y) {
    constexpr int S = SCORE_N;
    constexpr int stepc = PS / S;                  // spacing in tracking px
    float v[S][S];
    const int off = DS / 2;
    for (int i = 0; i < S; i++) {
        const uint8_t* r = frame_px + (((y + stepc / 2 + i * stepc) * DS + off) * W + (x + stepc / 2) * DS + off) * 3 + 1;
        for (int j = 0; j < S; j++, r += stepc * DS * 3) v[i][j] = (float)r[0];
    }

    float a = 0, b = 0, d = 0;
    for (int i = 0; i < S - 1; i++)
        for (int j = 0; j < S - 1; j++) {
            float gx = v[i][j + 1] - v[i][j], gy = v[i + 1][j] - v[i][j];
            a += gx * gx; b += gx * gy; d += gy * gy;
        }
    float tr = 0.5f * (a + d), det = a * d - b * b;
    samp_s += tr; samp_n++;            // frame gradient energy (contrast^2), free from the score
    float disc = tr * tr - det;
    return tr - std::sqrt(disc > 0 ? disc : 0.0f);
}

#if defined(__wasm_simd128__)
// score_cell for 4 cells of one grid row at once (lanes = cells); same float operations, same order -> same scores
inline void score_cells4(const int* xs, int y, float* out) {
    static_assert(SCORE_N == 4, "SIMD ranking is written for 4x4 samples");
    constexpr int stepc = PS / 4;
    constexpr int ss = stepc * DS * 3;           // bytes between samples of one cell row
    const int off = DS / 2;
    const uint8_t* p0 = frame_px + (((y + stepc / 2) * DS + off) * W + (xs[0] + stepc / 2) * DS + off) * 3 + 1;
    const int d1 = (xs[1] - xs[0]) * DS * 3, d2 = (xs[2] - xs[0]) * DS * 3, d3 = (xs[3] - xs[0]) * DS * 3;
    const int rowb = stepc * DS * W * 3;
#define LD(r, j) wasm_f32x4_convert_i32x4(wasm_i32x4_make((r)[(j) * ss], (r)[d1 + (j) * ss], (r)[d2 + (j) * ss], (r)[d3 + (j) * ss]))
    v128_t a = wasm_f32x4_splat(0), b = a, d = a;
    const uint8_t* r = p0;
    v128_t q0 = LD(r, 0), q1 = LD(r, 1), q2 = LD(r, 2), q3 = LD(r, 3);
    for (int i = 0; i < 3; i++) {
        r += rowb;
        v128_t n0 = LD(r, 0), n1 = LD(r, 1), n2 = LD(r, 2), n3 = LD(r, 3);
#define ACC(c, cn, cd) { v128_t gx = wasm_f32x4_sub(cn, c), gy = wasm_f32x4_sub(cd, c); \
        a = wasm_f32x4_add(a, wasm_f32x4_mul(gx, gx)); b = wasm_f32x4_add(b, wasm_f32x4_mul(gx, gy)); \
        d = wasm_f32x4_add(d, wasm_f32x4_mul(gy, gy)); }
        ACC(q0, q1, n0) ACC(q1, q2, n1) ACC(q2, q3, n2)
#undef ACC
        q0 = n0; q1 = n1; q2 = n2; q3 = n3;
    }
#undef LD
    v128_t tr = wasm_f32x4_mul(wasm_f32x4_splat(0.5f), wasm_f32x4_add(a, d));
    v128_t det = wasm_f32x4_sub(wasm_f32x4_mul(a, d), wasm_f32x4_mul(b, b));
    v128_t disc = wasm_f32x4_max(wasm_f32x4_sub(wasm_f32x4_mul(tr, tr), det), wasm_f32x4_splat(0));
    wasm_v128_store(out, wasm_f32x4_sub(tr, wasm_f32x4_sqrt(disc)));
    samp_s += wasm_f32x4_extract_lane(tr, 0); samp_s += wasm_f32x4_extract_lane(tr, 1);
    samp_s += wasm_f32x4_extract_lane(tr, 2); samp_s += wasm_f32x4_extract_lane(tr, 3);
    samp_n += 4;
}
#endif

// rank a slice of the candidate grid (cells [c0, c1)) into the running top-K list
void rank_slice(int c0, int c1) {
#if defined(__wasm_simd128__)
    {   // scores only (cells are fixed); the top-K selection happens once in select_top()
        int k = c0;
        for (; k % 4 && k < c1; k++) cscore[k] = score_cell(cx_[k], cy_[k]);
        for (; k + 4 <= c1; k += 4) score_cells4(cx_ + k, cy_[k], cscore + k);  // 4 cells of one grid row
        for (; k < c1; k++) cscore[k] = score_cell(cx_[k], cy_[k]);
        return;
    }
#endif
    for (int k = c0; k < c1; k++) {
        float v = score_cell(cx_[k], cy_[k]);
        const int K = KTOP;
        if (nk == K && v <= topv[K - 1]) continue;
        int pos = nk < K ? nk++ : K - 1;
        while (pos > 0 && topv[pos - 1] < v) { topv[pos] = topv[pos - 1]; top[pos] = top[pos - 1]; pos--; }
        topv[pos] = v; top[pos] = k;
    }
}

// top-K cells by (score desc, index asc) -- the same order the running insertion gives -- via quickselect + sort
void select_top() {
#if !defined(__wasm_simd128__)
    return;   // host build: rank_slice kept the running top-K itself
#endif
    // 1) histogram on the float bit pattern (monotonic for scores >= 0): 4 bins per octave
    constexpr int NB = 128;
    int hist[NB];
    unsigned char bin[NCAND];
    for (int b = 0; b < NB; b++) hist[b] = 0;
    for (int k = 0; k < NCAND; k++) {
        float v = cscore[k] > 0 ? cscore[k] : 0.0f;
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        int b = (int)(bits >> 21) - (127 << 2);
        b = b < 0 ? 0 : (b >= NB ? NB - 1 : b);
        bin[k] = (unsigned char)b;
        hist[b]++;
    }
    // 2) threshold bin holding the K-th best
    const int K = KTOP;
    int tb = NB - 1, cum = 0;
    for (; tb > 0; tb--) { cum += hist[tb]; if (cum >= K) break; }
    // 3) the usual running top-K insertion, over the survivors only (cells below the threshold bin have K strictly
    //    better cells, so the result is identical to inserting every cell)
    nk = 0;
    if (RANK_FAST) {   // sort only the cells strictly above the threshold bin (< K of them), then fill from that bin in
                       // grid order: bounded cost even when many cells share the boundary bin (low-texture frames)
        for (int k = 0; k < NCAND; k++) {
            if (bin[k] <= tb) continue;
            float v = cscore[k];
            int pos = nk++;
            while (pos > 0 && topv[pos - 1] < v) { topv[pos] = topv[pos - 1]; top[pos] = top[pos - 1]; pos--; }
            topv[pos] = v; top[pos] = k;
        }
        for (int k = 0; k < NCAND && nk < K; k++)
            if (bin[k] == tb) { topv[nk] = cscore[k]; top[nk++] = k; }
        return;
    }
    for (int k = 0; k < NCAND; k++) {
        if (bin[k] < tb) continue;
        float v = cscore[k];
        if (nk == K && v <= topv[K - 1]) continue;
        int pos = nk < K ? nk++ : K - 1;
        while (pos > 0 && topv[pos - 1] < v) { topv[pos] = topv[pos - 1]; top[pos] = top[pos - 1]; pos--; }
        topv[pos] = v; top[pos] = k;
    }
}

// cut templates for subset `si` from the current frame at the next ranked cells, until `want` more are accepted
void cut_templates(int si, int want) {
    constexpr int BW = PS + 2;
    int target = n_patches + want, tries = 0;
    for (; next_cand < nk && n_patches < target && n_patches < N_TRACK && tries < want + N_EXTRA / SPLIT; next_cand++, tries++) {
        int i = next_cand;
        Patch& p = patches[n_patches];
        p.sub = si;
        p.ox = cx_[top[i]]; p.oy = cy_[top[i]];
        alignas(16) float buf[BW * BW + 4];
        for (int v = 0; v < BW; v++) fetch_row(p.ox - 1, p.oy - 1 + v, BW, buf + v * BW);
        float fa = 0, fb = 0, fd = 0, fs = 0, fss = 0, sgx = 0, sgy = 0;
#if defined(__wasm_simd128__)
        {
            v128_t H = wasm_f32x4_splat(0.5f), A = wasm_f32x4_splat(0), B = A, D = A, S = A, SS = A, GXs = A, GYs = A;
            for (int v = 0; v < PS; v++) {
                const float* c = buf + (v + 1) * BW + 1;
                for (int u = 0; u < PS; u += 4) {
                    v128_t C = wasm_v128_load(c + u);
                    v128_t gx = wasm_f32x4_mul(H, wasm_f32x4_sub(wasm_v128_load(c + u + 1), wasm_v128_load(c + u - 1)));
                    v128_t gy = wasm_f32x4_mul(H, wasm_f32x4_sub(wasm_v128_load(c + u + BW), wasm_v128_load(c + u - BW)));
                    wasm_v128_store(p.T + v * PS + u, C);
                    wasm_v128_store(p.GX + v * PS + u, gx);
                    wasm_v128_store(p.GY + v * PS + u, gy);
                    A = wasm_f32x4_add(A, wasm_f32x4_mul(gx, gx)); B = wasm_f32x4_add(B, wasm_f32x4_mul(gx, gy));
                    D = wasm_f32x4_add(D, wasm_f32x4_mul(gy, gy)); S = wasm_f32x4_add(S, C);
                    SS = wasm_f32x4_add(SS, wasm_f32x4_mul(C, C)); GXs = wasm_f32x4_add(GXs, gx); GYs = wasm_f32x4_add(GYs, gy);
                }
            }
            fa = hsum(A); fb = hsum(B); fd = hsum(D); fs = hsum(S); fss = hsum(SS); sgx = hsum(GXs); sgy = hsum(GYs);
        }
#else
        for (int v = 0; v < PS; v++) {
            const float* c = buf + (v + 1) * BW + 1;
            float* T = p.T + v * PS; float* GX = p.GX + v * PS; float* GY = p.GY + v * PS;
            for (int u = 0; u < PS; u++) {
                float gx = 0.5f * (c[u + 1] - c[u - 1]), gy = 0.5f * (c[u + BW] - c[u - BW]);
                T[u] = c[u]; GX[u] = gx; GY[u] = gy;
                fa += gx * gx; fb += gx * gy; fd += gy * gy; fs += c[u]; fss += c[u] * c[u];
                sgx += gx; sgy += gy;
            }
        }
#endif
        // structure tensor of the mean-removed problem (the brightness offset absorbs the gradient mean)
        double a = fa - (double)sgx * sgx / PN, b = fb - (double)sgx * sgy / PN, d = fd - (double)sgy * sgy / PN;
        double tr = 0.5 * (a + d), det = a * d - b * b;
        double mineig = (tr - std::sqrt(std::fmax(0.0, tr * tr - det))) / PN;
        if (mineig < eig_thresh || !(det > 1e-9)) continue;
        p.i00 = d / det; p.i01 = -b / det; p.i11 = a / det;
        p.sgx = sgx; p.sgy = sgy;
        double mean = fs / PN;
        p.tstd = (float)std::sqrt(std::fmax(1.0, fss / PN - mean * mean));
        n_patches++;
    }
}

// Track patch p: (px, py) is the predicted top-left corner (tracking resolution). Lucas-Kanade, inverse compositional,
// translation + brightness offset, on a small crop built straight from the raw frame. SIMD over 4-pixel row chunks.
bool track(const Patch& p, double& px, double& py) {
    int ox = (int)std::floor(px) - MARGIN, oy = (int)std::floor(py) - MARGIN;
    if (ox < 0 || oy < 0 || ox + CSZ + 1 > SW || oy + CSZ + 1 > SH) return false;
    alignas(16) float crop[CSZ * CSZ + 8];
    for (int v = 0; v < CSZ; v++) fetch_row(ox, oy + v, CSZ, crop + v * CSZ);
    double x = px - ox, y = py - oy;  // position in the crop
    const double x0 = x, y0 = y;
    float see = 0, se = 0;
    for (int it = 0; it < LK_ITERS; it++) {
        int ix = (int)std::floor(x), iy = (int)std::floor(y);
        if (ix < 0 || iy < 0 || ix + PS + 1 > CSZ || iy + PS + 1 > CSZ) return false;
        float ax = (float)(x - ix), ay = (float)(y - iy);
        float w00 = (1 - ax) * (1 - ay), w01 = ax * (1 - ay), w10 = (1 - ax) * ay, w11 = ax * ay;
        float sx, sy;
#if defined(__wasm_simd128__)
        v128_t W00 = wasm_f32x4_splat(w00), W01 = wasm_f32x4_splat(w01), W10 = wasm_f32x4_splat(w10), W11 = wasm_f32x4_splat(w11);
        v128_t Se = wasm_f32x4_splat(0), Sx = Se, Sy = Se, See = Se;
        for (int v = 0; v < PS; v++) {
            const float* r0 = crop + (iy + v) * CSZ + ix;
            const float* r1 = r0 + CSZ;
            for (int u = 0; u < PS; u += 4) {
                v128_t I = wasm_f32x4_add(
                    wasm_f32x4_add(wasm_f32x4_mul(W00, wasm_v128_load(r0 + u)), wasm_f32x4_mul(W01, wasm_v128_load(r0 + u + 1))),
                    wasm_f32x4_add(wasm_f32x4_mul(W10, wasm_v128_load(r1 + u)), wasm_f32x4_mul(W11, wasm_v128_load(r1 + u + 1))));
                v128_t E = wasm_f32x4_sub(I, wasm_v128_load(p.T + v * PS + u));
                Se = wasm_f32x4_add(Se, E);
                See = wasm_f32x4_add(See, wasm_f32x4_mul(E, E));
                Sx = wasm_f32x4_add(Sx, wasm_f32x4_mul(wasm_v128_load(p.GX + v * PS + u), E));
                Sy = wasm_f32x4_add(Sy, wasm_f32x4_mul(wasm_v128_load(p.GY + v * PS + u), E));
            }
        }
        se = hsum(Se); see = hsum(See); sx = hsum(Sx); sy = hsum(Sy);
#else
        se = 0; see = 0; sx = 0; sy = 0;
        for (int v = 0; v < PS; v++) {
            const float* r0 = crop + (iy + v) * CSZ + ix;
            const float* r1 = r0 + CSZ;
            for (int u = 0; u < PS; u++) {
                float e = w00 * r0[u] + w01 * r0[u + 1] + w10 * r1[u] + w11 * r1[u + 1] - p.T[v * PS + u];
                se += e; see += e * e; sx += p.GX[v * PS + u] * e; sy += p.GY[v * PS + u] * e;
            }
        }
#endif
        float mean = se / PN;  // brightness offset: sum g (e - mean) = sum g e - mean sum g
        sx -= mean * p.sgx;
        sy -= mean * p.sgy;
        double ddx = p.i00 * sx + p.i01 * sy, ddy = p.i01 * sx + p.i11 * sy;
        x -= ddx; y -= ddy;
        if (ddx * ddx + ddy * ddy < 1e-4) break;
    }
    float resid = std::sqrt(std::fmax(0.0f, see / PN - (se / PN) * (se / PN)));  // of the last evaluated position
    if (resid > MAX_RESID * p.tstd) return false;
    if ((x - x0) * (x - x0) + (y - y0) * (y - y0) > MAX_CORR_PX * MAX_CORR_PX) return false;
    if (x < 0 || y < 0 || x + PS + 1 > CSZ || y + PS + 1 > CSZ) return false;
    px = x + ox; py = y + oy;
    return true;
}

// ================= small solvers =================
double smallest_eig(const double M[9], Vec3& u) {  // inverse power iteration from u
    double eps = 1e-12 * (M[0] + M[4] + M[8]) + 1e-30;
    double a = M[0] + eps, b = M[1], c = M[2], d = M[4] + eps, e = M[5], f = M[8] + eps;
    double A[9] = {d * f - e * e, c * e - b * f, b * e - c * d,
                   c * e - b * f, a * f - c * c, b * c - a * e,
                   b * e - c * d, b * c - a * e, a * d - b * b};
    for (int it = 0; it < EIG_ITERS; it++) {
        Vec3 w{A[0] * u.x + A[1] * u.y + A[2] * u.z, A[3] * u.x + A[4] * u.y + A[5] * u.z, A[6] * u.x + A[7] * u.y + A[8] * u.z};
        double n = norm(w);
        if (!(n > 1e-300)) break;
        u = scale(w, 1.0 / n);
    }
    Vec3 Mu{M[0] * u.x + M[1] * u.y + M[2] * u.z, M[3] * u.x + M[4] * u.y + M[5] * u.z, M[6] * u.x + M[7] * u.y + M[8] * u.z};
    return dot(u, Mu);
}

bool solve3(const double A[9], double lam, Vec3 b, Vec3& x) {  // (A + lam I) x = b, A symmetric
    double a = A[0] + lam, bb = A[1], c = A[2], d = A[4] + lam, e = A[5], f = A[8] + lam;
    double C00 = d * f - e * e, C01 = c * e - bb * f, C02 = bb * e - c * d;
    double det = a * C00 + bb * C01 + c * C02;
    if (!(std::fabs(det) > 1e-300)) return false;
    double C11 = a * f - c * c, C12 = bb * c - a * e, C22 = a * d - bb * bb;
    x = {(C00 * b.x + C01 * b.y + C02 * b.z) / det, (C01 * b.x + C11 * b.y + C12 * b.z) / det,
         (C02 * b.x + C12 * b.y + C22 * b.z) / det};
    return true;
}

inline void accum(double M[9], Vec3 n) {
    M[0] += n.x * n.x; M[1] += n.x * n.y; M[2] += n.x * n.z;
    M[4] += n.y * n.y; M[5] += n.y * n.z; M[8] += n.z * n.z;
    M[3] = M[1]; M[6] = M[2]; M[7] = M[5];
}

// change of the camera-IMU lever arm since the keyframe clone (subset 0 attitude)
inline Vec3 sub_lever() { return sub(mul(R_wb, r_cam_in_imu), mul(R_wb_clone, r_cam_in_imu)); }

// ================= vision update (end of a keyframe window) =================
struct Track { Vec3 fr, f1; Mat3 R_wc; double par; };  // predicted / measured unit bearings (camera at tracking time) + camera attitude
Track tracks[N_TRACK];
int n_tracks = 0;
double par_sum = 0;

// Track the patches of subset `si` into the current frame; the rotation residual of this subset updates the gyro bias.
void track_subset(int si) {
    Mat3 R_wc0f = mul(R_wb_kf[si], R_ic), R_wc1f = mul(R_wb, R_ic);
    Mat3 R10 = mul(transpose(R_wc1f), R_wc0f);  // subset keyframe camera -> current camera (gyro)
    const int first = n_tracks;
    double ps = 0;
    for (int s = 0; s < n_patches; s++) {
        const Patch& p = patches[s];
        if (p.sub != si) continue;
        Vec3 f0{(p.ox + PC - cx_s) / fx_s, (p.oy + PC - cy_s) / fy_s, 1.0};
        Vec3 fr = mul(R10, f0);
        if (fr.z < 0.2) continue;
        double px0 = cx_s + fx_s * fr.x / fr.z, py0 = cy_s + fy_s * fr.y / fr.z;  // predicted centre
        double px = px0 - PC, py = py0 - PC;                                       // -> top-left
        if (!track(p, px, py)) continue;
        px += PC; py += PC;
        Vec3 f1{(px - cx_s) / fx_s, (py - cy_s) / fy_s, 1.0};
        Track& t = tracks[n_tracks++];
        t.fr = scale(fr, 1.0 / norm(fr));
        t.f1 = scale(f1, 1.0 / norm(f1));
        t.R_wc = R_wc1f;
        t.par = std::sqrt((px - px0) * (px - px0) + (py - py0) * (py - py0));
        ps += t.par;
    }
    par_sum += ps;
#if LQG
    P_use = (LQG_SCHED && lqg_hover) ? P_rot_h[si] : sched_on(run_win) ? P_rot_s[run_win][si] : P_rot[si];
    trans_only = false;
    cur_group = 0;
#endif
    const int nt = n_tracks - first;
    if (nt < MIN_TRACKS / SPLIT + 1 || !(win_T[si] > 0)) return;
    // rotation error of the gyro prediction: true f1 ~ exp(delta) fr  (delta in the current camera frame)
    double par = ps / nt;
    Vec3 dlever = sub_lever();
    Vec3 d_est = add(sub(est_p(), p_kf), dlever);
    Vec3 uc = norm(d_est) > 1e-6 ? scale(mulT(R_wc1f, d_est), 1.0 / norm(d_est)) : Vec3{0, 0, 1};
    double A[9] = {0};
    Vec3 b{0, 0, 0};
    for (int i = first; i < n_tracks; i++) {
        Vec3 fr = tracks[i].fr, f1 = tracks[i].f1;
        if (par < MIN_PAR_PX) {  // nearly pure rotation: f1 - fr = delta x fr = -[fr]x delta
            Vec3 r = sub(f1, fr);
            Vec3 J0{0, fr.z, -fr.y}, J1{-fr.z, 0, fr.x}, J2{fr.y, -fr.x, 0};
            accum(A, J0); accum(A, J1); accum(A, J2);
            b = add(b, add(scale(J0, r.x), add(scale(J1, r.y), scale(J2, r.z))));
        } else {  // epipolar residual r = f1.(u x fr), dr/ddelta = (u.fr) f1 - (f1.fr) u
            double r = dot(f1, cross(uc, fr));
            Vec3 J = sub(scale(f1, dot(uc, fr)), scale(uc, dot(f1, fr)));
            accum(A, J);
            b = add(b, scale(J, -r));
        }
    }
    Vec3 delta;
    if (solve3(A, ROT_DAMP, b, delta) && norm(delta) < ROT_MAX) {
        Vec3 db = mul(R_ic, delta);  // R10_true = exp(T dbg) R10_gyro  ->  delta_body = +T dbg
        double dbv[3] = {db.x, db.y, db.z};
        double sr = SIG_ROT / std::sqrt((double)nt) + SIG_ROT_FLOOR + SIG_ROT_PAR * (par < MIN_PAR_PX ? 0.0 : par);
        for (int k = 0; k < 3; k++) {
            double Hrow[NS] = {0};
            Hrow[IBG + k] = win_T[si];
            kf_update_row(Hrow, dbv[k], sr * sr);
        }
        kf_inject();
    }
}

// All tracks of the window -> one motion direction (+ speed proxy) for the displacement since the keyframe clone.
void direction_update() {
#if LQG
    P_use = (LQG_SCHED && lqg_hover) ? P_dir_h : sched_on(run_win) ? P_dir_s[run_win] : P_dir;
    trans_only = LQG_DIR_STATES;
    cur_group = 1;
#endif
    const int nt = n_tracks;
    last_tracks = nt;
    last_used = 0;
    last_parallax = nt ? par_sum / nt : 0;
    if (nt < MIN_TRACKS) return;
    Vec3 d_est = add(sub(est_p(), p_kf), sub_lever());
    double dn = norm(d_est);
    if (last_parallax < MIN_PAR_PX) {
#if LQG
        if (LQG_SCHED) P_use = P_dir_h;
#endif
        // little translational flow on well-tracked patches -> small displacement (indoor depths are bounded)
        {
            double sd = (last_parallax + SMALL_FLOW_PX) / fx_s * DEPTH_PRIOR;
            double dv[3] = {d_est.x, d_est.y, d_est.z};
            for (int k = 0; k < 3; k++) {
                double Hrow[NS] = {0};
                Hrow[IP + k] = 1.0; Hrow[IPC + k] = -1.0;
                if (kf_update_row(Hrow, -dv[k], sd * sd)) last_used++;
            }
        }
        kf_inject();
        return;
    }
    // the direction most orthogonal to all n = R_wc (fr x f1) (world frame)
    Vec3 ns[N_TRACK];
    for (int i = 0; i < nt; i++) ns[i] = mul(tracks[i].R_wc, cross(tracks[i].fr, tracks[i].f1));
    Vec3 u = dn > 1e-6 ? scale(d_est, 1.0 / dn) : Vec3{0.577, 0.577, 0.577};
    double M[9] = {0};
    for (int i = 0; i < nt; i++) accum(M, ns[i]);
    smallest_eig(M, u);
    double r[N_TRACK], rs[N_TRACK];  // one robust refit
    for (int i = 0; i < nt; i++) { r[i] = std::fabs(dot(ns[i], u)); rs[i] = r[i]; }
    for (int i = 1; i < nt; i++) { double v = rs[i]; int j = i - 1; while (j >= 0 && rs[j] > v) { rs[j + 1] = rs[j]; j--; } rs[j + 1] = v; }
    double med = rs[nt / 2];
    double M2[9] = {0};
    int kept = 0;
    for (int i = 0; i < nt; i++)
        if (r[i] <= 3.0 * med + 1e-12) { accum(M2, ns[i]); kept++; }
    if (kept < MIN_TRACKS) { kf_inject(); return; }
    double l1 = smallest_eig(M2, u);
    double lmid = 0.5 * (M2[0] + M2[4] + M2[8] - l1);
    if (!(lmid > 0)) { kf_inject(); return; }
    double q = l1 / lmid;
    last_quality = q;
    if (q > MAX_Q) { kf_inject(); return; }
    // sign from cheirality: a point in front of the camera flows like (x d_z - d_x, y d_z - d_y) / Z
    double score = 0;
    for (int i = 0; i < nt; i++) {
        Vec3 uc = mulT(tracks[i].R_wc, u);
        const Vec3& fr = tracks[i].fr; const Vec3& f1 = tracks[i].f1;
        double xr = fr.x / fr.z, yr = fr.y / fr.z;
        double fx_ = f1.x / f1.z - xr, fy_ = f1.y / f1.z - yr;
        score += fx_ * (xr * uc.z - uc.x) + fy_ * (yr * uc.z - uc.y);
    }
    if (score < 0) u = scale(u, -1.0);
    if (SIGN_FWD > 0) {   // generator prior: nose-first flight -> a direction clearly along the optical axis points forward
        Vec3 cz = mul(mul(R_wb, R_ic), Vec3{0, 0, 1});
        double a = dot(u, cz);
        if (std::fabs(a) > SIGN_FWD && a < 0) u = scale(u, -1.0);
    }
    Vec3 a = std::fabs(u.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 e1 = cross(u, a); e1 = scale(e1, 1.0 / norm(e1));
    Vec3 e2 = cross(u, e1);
    double sig_dir = std::sqrt(q / kept) * SIG_DIR_SCALE + SIG_DIR_PX / (last_parallax * std::sqrt((double)kept)) + SIG_DIR_FLOOR;
    double Rn = sig_dir * sig_dir * (dn * dn + D_FLOOR * D_FLOOR);
    Vec3 es[2] = {e1, e2};
    double Hrow[NS];
    for (int k = 0; k < 2; k++) {  // e . d = 0; the world e is exp(dtheta) e_est, so dh/dtheta = e x d
        Vec3 e = es[k], ed = cross(e, d_est);
        row_d(Hrow, e);
        Hrow[ITH] = ed.x; Hrow[ITH + 1] = ed.y; Hrow[ITH + 2] = ed.z;
        if (kf_update_row(Hrow, -dot(e, d_est), Rn)) last_used++;
    }
    // speed proxy along u: |d| ~ mean parallax * PAR_SPEED / f (Sim(3) only needs a consistent scale)
    if (USE_PAR_SPEED) {
        double dmeas = last_parallax / fx_s * PAR_SPEED;
        double sd = PAR_SPEED_REL * dmeas + 0.01;
        row_d(Hrow, u);
        if (kf_update_row(Hrow, dmeas - dot(u, d_est), sd * sd)) last_used++;
    }
    kf_inject();
}

// ================= IMU =================
void finish_calibration(double t) {
    double n = calib_n > 0 ? calib_n : 1;
    Vec3 mean_a{sum_a[0] / n, sum_a[1] / n, sum_a[2] / n};
    Vec3 mean_g{sum_g[0] / n, sum_g[1] / n, sum_g[2] / n};
    double var_a = 0, var_g = 0;
    for (int i = 0; i < 3; i++) {
        double ma = sum_a[i] / n, mg = sum_g[i] / n;
        var_a += std::fmax(0.0, sum_a2[i] / n - ma * ma);
        var_g += std::fmax(0.0, sum_g2[i] / n - mg * mg);
    }
    double na = norm(mean_a);
    bool at_rest = std::sqrt(var_a) < REST_ACCEL_STD && std::sqrt(var_g) < REST_GYRO_STD &&
                   norm(mean_g) < REST_GYRO_MEAN && std::fabs(na - GRAVITY) < REST_ACCEL_MEAN;
    bg = at_rest ? mean_g : Vec3{0, 0, 0};
    ba = {0, 0, 0};
#if LQG
    if (LQG_WIND_HI > 0) {
        double w = std::sqrt(mean_a.x * mean_a.x + mean_a.y * mean_a.y);
        wind_scale = (LQG_WIND_HI - w) / (LQG_WIND_HI - LQG_WIND_LO);
        if (wind_scale > 1) wind_scale = 1;
        if (!(wind_scale > 0)) wind_scale = 0;
    }
#endif
    // The start hover sags and wobbles, but returns to rest: over the whole hover the motion
    // averages out, leaving the bias along gravity (its horizontal part is inseparable from tilt).
    if (CALIB_ACCEL_BIAS && at_rest && na > 1e-6) ba = scale(mean_a, (na - GRAVITY) / na);
#if THRUST_ONLY
    ba.x = ba.y = 0;
#endif
    Vec3 z = na > 1e-6 ? scale(mean_a, 1.0 / na) : Vec3{0, 0, 1};
    Vec3 xc = std::fabs(z.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 x = sub(xc, scale(z, dot(xc, z)));
    x = scale(x, 1.0 / norm(x));
    Vec3 y = cross(z, x);
    R_wb = {{x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z}};
    p_w = v_w = {0, 0, 0};
    std::memset(P, 0, sizeof P);
    double sv = at_rest ? SIG_V0_REST : SIG_V0_MOVING;
    double st = at_rest ? SIG_TILT0_REST : SIG_TILT0_MOVING;
    double sbg = at_rest ? SIG_BG0_REST : SIG_BG0_MOVING;
    for (int i = 0; i < 3; i++) {
        P[(IP + i) * NS + IP + i] = 1e-8;
        P[(IV + i) * NS + IV + i] = sv * sv;
        P[(IBA + i) * NS + IBA + i] = SIG_BA0 * SIG_BA0;
        P[(IBG + i) * NS + IBG + i] = sbg * sbg;
    }
    P[(ITH + 0) * NS + ITH + 0] = st * st;
    P[(ITH + 1) * NS + ITH + 1] = st * st;
    P[(ITH + 2) * NS + ITH + 2] = 1e-6;  // yaw is unobservable and defines the frame
    for (int i = 0; i < NS; i++) dx[i] = 0;
    R_wc0 = mul(R_wb, R_ic);
    p_wc0 = mul(R_wb, r_cam_in_imu);
    last_t = t;
    initialized = true;
}

void integrate(const double* imu, int count) {
    for (int k = 0; k < count; k++) {
        const double* r = imu + 7 * k;
        double t = r[0];
#if VEL_PRIOR == 1
        for (int i = 0; i < 6; i++) vp_sum[i] += r[1 + i];
        if (++vp_n == VP_BIN) {
            for (int i = 0; i < 6; i++) { vp_bins[vp_head][i] = (float)(vp_sum[i] / VP_BIN / (i < 3 ? 9.81 : 1.0)); vp_sum[i] = 0; }
            vp_head = (vp_head + 1) % VP_HIST; if (vp_count < VP_HIST) vp_count++; vp_n = 0;
        }
#endif
#if THRUST_ONLY
        Vec3 a{0, 0, r[3] - ba.z};
#else
        Vec3 a = sub(Vec3{r[1], r[2], r[3]}, ba);
#endif
        Vec3 w = sub(Vec3{r[4], r[5], r[6]}, bg);
        omega_b = w;
        double dt = t - last_t;
        last_t = t;
        if (!(dt > 0) || dt > 0.1) continue;
        Vec3 f = mul(R_wb, a);
        Vec3 acc = add(f, g_w);
        p_w = add(p_w, add(scale(v_w, dt), scale(acc, 0.5 * dt * dt)));
        v_w = add(v_w, scale(acc, dt));
        rotate_small(R_wb, scale(w, dt));
        for (int i = 0; i < 9; i++) acc_R.m[i] += R_wb.m[i];
        acc_f = add(acc_f, f);
        acc_n++;
#if LQG
        frame_T += dt; frame_f = add(frame_f, f); frame_n++;
        for (int i = 0; i < 3; i++) { fr_g[i] += r[4 + i]; fr_a[i] += r[1 + i]; }
        fr_n++;
#endif
        acc_T += dt;
        for (int k = 0; k < SPLIT; k++) win_T[k] += dt;
    }
    orthonormalize(R_wb);
}

void write_identity(double* out) {
    for (int i = 0; i < VIOBENCH_STATE_VALUES; i++) out[i] = 0.0;
    out[6] = 1.0;
}

void write_state(double* out) {
    Mat3 R_wc = mul(R_wb, R_ic);
    Vec3 p_wc = add(p_w, mul(R_wb, r_cam_in_imu));
    Mat3 R = mul(transpose(R_wc0), R_wc);
    Vec3 t = mulT(R_wc0, sub(p_wc, p_wc0));
    double q[4];
    to_quat(R, q);
    Vec3 v_b = add(mulT(R_wb, v_w), cross(omega_b, r_cam_in_imu));
    Vec3 v_c = mul(R_ci, v_b);
    Vec3 w_c = mul(R_ci, omega_b);
    double vals[13] = {t.x, t.y, t.z, q[0], q[1], q[2], q[3], v_c.x, v_c.y, v_c.z, w_c.x, w_c.y, w_c.z};
    for (int i = 0; i < 13; i++) {
        if (!std::isfinite(vals[i])) { write_identity(out); return; }
        out[i] = vals[i];
    }
}

void finish_rank();
void select_top();
// first frame of a window: rank candidate cells and clone the position (templates follow on the next frames)
void make_keyframe() {
    nk = 0;
    samp_s = samp_ss = 0; samp_n = 0;
    rank_slice(0, NCAND);
    finish_rank();
    n_patches = 0;
    next_cand = 0;
    n_tracks = 0;
    par_sum = 0;
    have_kf = true;
}

void select_top();
void finish_rank() {
    select_top();
    {   // texture threshold scaled with the frame's contrast (gain-invariant patch selection)
        double energy = samp_n ? samp_s / samp_n : CONTRAST_REF;  // mean half-trace of the cell tensors ~ contrast^2
        double f = CONTRAST_POW ? energy / CONTRAST_REF : 1.0;
        if (f > 1.0) f = 1.0;
        if (f < CONTRAST_MIN) f = CONTRAST_MIN;
        eig_thresh = MIN_EIG * f;
    }
}

#if VEL_PRIOR == 1
// MLP: [VP_HIST bins oldest..newest x 6, gravity direction in body] -> body-frame velocity
bool predict_body_velocity(double out[3]) {
    if (vp_count < VP_HIST) return false;
    float x[VP_IN];
    int j = 0;
    for (int b = 0; b < VP_HIST; b++) {
        const float* bin = vp_bins[(vp_head + b) % VP_HIST];
        for (int i = 0; i < 6; i++) x[j++] = bin[i];
    }
    Vec3 gb = mulT(R_wb, Vec3{0, 0, -1});
    x[j++] = (float)gb.x; x[j++] = (float)gb.y; x[j++] = (float)gb.z;
    for (int i = 0; i < VP_IN; i++) x[i] = (x[i] - vp_mean[i]) * vp_inv_std[i];
    float h1[VP_H1], h2[VP_H2];
    for (int o = 0; o < VP_H1; o++) {
        float s = vp_b1[o]; const float* w = vp_w1 + o * VP_IN;
        for (int i = 0; i < VP_IN; i++) s += w[i] * x[i];
        h1[o] = std::tanh(s);
    }
    for (int o = 0; o < VP_H2; o++) {
        float s = vp_b2[o]; const float* w = vp_w2 + o * VP_H1;
        for (int i = 0; i < VP_H1; i++) s += w[i] * h1[i];
        h2[o] = std::tanh(s);
    }
    for (int o = 0; o < 3; o++) {
        float s = vp_b3[o]; const float* w = vp_w3 + o * VP_H2;
        for (int i = 0; i < VP_H2; i++) s += w[i] * h2[i];
        out[o] = s;
    }
    return std::isfinite(out[0] + out[1] + out[2]);
}
#endif

#if VEL_PRIOR
double vp_nis[3] = {1, 1, 1};     // running normalized innovation squared, per body axis
double vp_fwd = VP_CONST_FWD;     // (adaptive) forward cruise mean
double vp_par = VP_STALL_START * VP_STALL_PX; // running mean of the window parallax (stall detector)

// normalized innovation of a row without applying it (same S and residual as kf_update_row)
double row_nis(const double Hrow[NS], double y, double Rn) {
    double S = Rn, yc = y;
    for (int j = 0; j < NS; j++) {
        if (Hrow[j] == 0.0) continue;
        yc -= Hrow[j] * dx[j];
        for (int k = 0; k < NS; k++) if (Hrow[k] != 0.0) S += Hrow[j] * P[j * NS + k] * Hrow[k];
    }
    return S > 0 ? yc * yc / S : 0.0;
}
#endif

void apply_velocity_prior() {
#if LQG
    if (vp_axis >= 0) P_use = sched_on(run_win) ? P_ax_s[run_win][vp_axis] : P_pri_ax[vp_axis];
    else P_use = sched_on(run_win) ? P_pri_s[run_win] : P_pri;
    trans_only = false;
    cur_group = 2;
#endif
    const Vec3 v_e = est_v();
#if VEL_PRIOR
    double vb[3], sg[3];
#if VEL_PRIOR == 1
    bool ok = predict_body_velocity(vb);
    for (int k = 0; k < 3; k++) sg[k] = VP_SIG_SCALE * vp_sigma[k];
#else
    bool ok = true;
    vb[0] = vp_fwd; vb[1] = vb[2] = 0;
    sg[0] = VP_CONST_SIG_X; sg[1] = VP_CONST_SIG_Y; sg[2] = VP_CONST_SIG_Z;
#if LQG
    if (LQG_WIND_HI > 0 && !designing) {   // stalled by wind: expect wind_scale of the cruise speed
        vb[0] *= wind_scale;
        sg[0] = wind_scale * sg[0] + (1 - wind_scale) * VP_STILL_SIG;
    }
#endif
    if (VP_GATE_PX > 0) {
        // Wind can stall the autopilot: assume cruising only when vision just saw translation.
        if (last_tracks < MIN_TRACKS) ok = false;            // no evidence -> v8 weak zero prior below
        else if (last_parallax < VP_GATE_PX) {                // well tracked, little translation -> still
            vb[0] = 0; sg[0] = sg[1] = sg[2] = VP_STILL_SIG;
        }
    }
    if (VP_STALL_PX > 0) {
        if (last_tracks >= MIN_TRACKS) vp_par += (last_parallax - vp_par) / VP_STALL_TAU;
        else if (VP_STALL_NOTRACK) vp_par -= vp_par / VP_STALL_TAU;
        if (vp_par < VP_STALL_PX) { vb[0] = 0; sg[0] = sg[1] = sg[2] = VP_STILL_SIG; }
#if LQG
        if (!designing) {
            lqg_hover = vp_par < VP_STALL_PX;
            if (LQG_SCHED && lqg_hover && vp_axis < 0) P_use = P_pri_h;
        }
#endif
    }
#endif
    if (ok) {   // rows along the body axes: (R e_k) . v_w = vb[k]
        for (int k = 0; k < 3; k++) {
#if LQG
            if (vp_axis >= 0 && k != vp_axis) continue;
#endif
            Vec3 axis{R_wb.m[k], R_wb.m[3 + k], R_wb.m[6 + k]};
            double Hrow[NS] = {0};
            Hrow[IV] = axis.x; Hrow[IV + 1] = axis.y; Hrow[IV + 2] = axis.z;
#if VP_ATTITUDE
            // the body axis rotates with the attitude error (same convention as the direction rows):
            // d(axis . v)/dtheta = axis x v, so a velocity/heading disagreement can also correct heading
            Vec3 av = cross(axis, v_e);
            Hrow[ITH] = av.x; Hrow[ITH + 1] = av.y; Hrow[ITH + 2] = av.z;
#endif
            double R = sg[k] * sg[k];
            if (VP_ADAPT_R > 0) {
                // mismatch measured against the nominal spread; its running mean is the inflation
                double nis = row_nis(Hrow, vb[k] - dot(axis, v_e), R);
                vp_nis[k] += VP_ADAPT_R * (nis - vp_nis[k]);
                if (vp_nis[k] < 1.0) vp_nis[k] = 1.0;
                if (vp_nis[k] > VP_ADAPT_CAP) vp_nis[k] = VP_ADAPT_CAP;
                R *= vp_nis[k];
            }
            kf_update_row(Hrow, vb[k] - dot(axis, v_e), R);
        }
        if (VP_ADAPT_MEAN > 0) {   // follow the flight's own cruise speed, bounded
            Vec3 fwd{R_wb.m[0], R_wb.m[3], R_wb.m[6]};
            double vf = dot(fwd, v_w);
            vp_fwd += VP_ADAPT_MEAN * (vf - vp_fwd);
            if (vp_fwd < 0.1) vp_fwd = 0.1;
            if (vp_fwd > 3.0) vp_fwd = 3.0;
        }
        kf_inject();
        return;
    }
#endif
    if (!(VPRIOR_SIG > 0)) return;
    double vv[3] = {v_e.x, v_e.y, v_e.z};
    const double vsig = VPRIOR_SIG + VPRIOR_REL * norm(v_e);
    for (int k = 0; k < 3; k++) {
#if LQG
        if (vp_axis >= 0 && k != vp_axis) continue;
#endif
        double Hrow[NS] = {0};
        Hrow[IV + k] = 1.0;
        kf_update_row(Hrow, -vv[k], vsig * vsig);
    }
    kf_inject();
}

#if LQG
// Motion-gated calibration: before the first motion, hold the pose and average gyro bias / gravity direction.
void still_step() {
    const int n = fr_n;
    double g[3], a[3];
    for (int i = 0; i < 3; i++) { g[i] = n ? fr_g[i] / n : 0; a[i] = n ? fr_a[i] / n : 0; fr_g[i] = fr_a[i] = 0; }
    fr_n = 0;
    if (!still_mode || n == 0) return;
    bool moving = std::fabs(std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) - GRAVITY) > STILL_ACC;
    if (still_n > 0)
        for (int i = 0; i < 3; i++) moving = moving || std::fabs(g[i] - still_g[i] / still_n) > STILL_GYRO;
    if (LQG_VISION && last_tracks >= MIN_TRACKS && last_parallax >= MIN_PAR_PX) moving = true;
    if (moving) { still_mode = false; return; }
    for (int i = 0; i < 3; i++) { still_g[i] += g[i]; still_a[i] += a[i]; }
    still_n++;
    bg = {still_g[0] / still_n, still_g[1] / still_n, still_g[2] / still_n};
    Vec3 am{still_a[0] / still_n, still_a[1] / still_n, still_a[2] / still_n};
    double na = norm(am);
    if (na > 1e-6) {   // re-level: rotate the attitude so the mean specific force points up (heading kept)
        Vec3 up_w = mul(R_wb, scale(am, 1.0 / na));
        Vec3 ax = cross(up_w, Vec3{0, 0, 1});
        double sa = norm(ax), ca = up_w.z;
        if (sa > 1e-9) R_wb = mul(exp_so3(scale(ax, std::atan2(sa, ca) / sa)), R_wb);
        orthonormalize(R_wb);
    }
    p_w = v_w = p_kf = {0, 0, 0};
    c_p = c_v = c_th = {0, 0, 0};
}

// LQR: the remaining error x = [c_p, c_v] per axis is a double integrator driven by the accelerometer correction u_a
// (applied as a zero-order hold over a frame of length T): x+ = A x + B u, A = [1 T; 0 1], B = -[T^2/2; T].
// The attitude error c_th is driven by the gyro correction u_w: c+ = c - T u_w. Costs: c_p^2 + tau^2 c_v^2 + tau^4 u_a^2
// and c_th^2 + tau^2 u_w^2 (tau = LQR_TAU sets the closed-loop speed). Gains from iterating the discrete Riccati equation.
void lqr_design(double T) {
    const double tau = LQR_TAU > 0 ? LQR_TAU : 1.0;
    double q1 = 1.0, q2 = tau * tau, r = tau * tau * tau * tau;
    double b1 = -0.5 * T * T, b2 = -T;
    double X[4] = {q1, 0, 0, q2};   // [x11 x12; x21 x22]
    double k1 = 0, k2 = 0;
    for (int it = 0; it < 2000; it++) {
        // A^T X A, A^T X B, B^T X B
        double a11 = X[0], a12 = X[0] * T + X[1], a22 = X[0] * T * T + (X[1] + X[2]) * T + X[3];
        double xb1 = X[0] * b1 + X[1] * b2, xb2 = X[2] * b1 + X[3] * b2;   // X B
        double atxb1 = xb1, atxb2 = T * xb1 + xb2;                         // A^T X B
        double btxb = b1 * xb1 + b2 * xb2;
        double den = r + btxb;
        k1 = atxb1 / den; k2 = atxb2 / den;                                 // K = (R + B'XB)^-1 B'XA
        double N[4] = {q1 + a11 - atxb1 * k1, a12 - atxb1 * k2, a12 - atxb2 * k1, q2 + a22 - atxb2 * k2};
        for (int i = 0; i < 4; i++) X[i] = N[i];
    }
    K_p = -k1; K_v = -k2;   // u_a = K_p c_p + K_v c_v
    double x = 1.0, qt = 1.0, rt = tau * tau, b = -T;
    double kt = 0;
    for (int it = 0; it < 2000; it++) { kt = x * b / (rt + b * b * x); x = qt + x - x * b * kt; }
    K_th = -kt;              // u_w = K_th c_th
}

// one frame of control: the accelerometer and gyro corrections act on the estimate over the frame just integrated
void lqr_step() {
    const double T = frame_T;
    Vec3 fw = frame_n ? scale(frame_f, 1.0 / frame_n) : Vec3{0, 0, GRAVITY};
    frame_T = 0; frame_f = {0, 0, 0}; frame_n = 0;
    if (!(T > 0) || !(LQR_TAU > 0)) return;
    // the target (estimate + remaining error) has the corrected attitude, so its specific force differs by c_th x f
    c_v = add(c_v, scale(cross(c_th, fw), T));
    Vec3 ua = add(scale(c_p, K_p), scale(c_v, K_v));
    Vec3 uw = scale(c_th, K_th);
    p_w = add(p_w, scale(ua, 0.5 * T * T));
    v_w = add(v_w, scale(ua, T));
    R_wb = mul(exp_so3(scale(uw, T)), R_wb);
    c_p = sub(add(c_p, scale(c_v, T)), scale(ua, 0.5 * T * T));
    c_v = sub(c_v, scale(ua, T));
    c_th = sub(c_th, scale(uw, T));
}

// Riccati recursion of the EKF over a nominal flight (level, heading +x, LQG_NOM_SPEED forward, hover thrust),
// following the same frame schedule as step(). The covariance before each batch is stored for run time.
bool design_hover = false;   // design flight hovers (LQG_SCHED)
void design_rotation(int si) {
    for (int i = 0; i < NS * NS; i++) P_rot[si][i] = P[i];
    if (sched_on(design_win)) for (int i = 0; i < NS * NS; i++) P_rot_s[design_win][si][i] = P[i];
    double par = design_hover ? 0.0 : LQG_NOM_PAR, nt = (double)N_TRACK / SPLIT;
    double sr = SIG_ROT / std::sqrt(nt) + SIG_ROT_FLOOR + SIG_ROT_PAR * (par < MIN_PAR_PX ? 0.0 : par);
    for (int k = 0; k < 3; k++) {
        double Hrow[NS] = {0};
        Hrow[IBG + k] = win_T[si];
        ekf_update_row(Hrow, 0.0, sr * sr);
    }
    kf_inject();
}
void design_direction() {
    for (int i = 0; i < NS * NS; i++) P_dir[i] = P[i];
    if (sched_on(design_win)) for (int i = 0; i < NS * NS; i++) P_dir_s[design_win][i] = P[i];
    if (design_hover) {   // small-motion rows, as direction_update() at low parallax
        double sd = SMALL_FLOW_PX / fx_s * DEPTH_PRIOR;
        double Hrow[NS];
        for (int k = 0; k < 3; k++) {
            for (int i = 0; i < NS; i++) Hrow[i] = 0;
            Hrow[IP + k] = 1.0; Hrow[IPC + k] = -1.0;
            ekf_update_row(Hrow, 0.0, sd * sd);
        }
        kf_inject();
        return;
    }
    Vec3 d = sub(p_w, p_kf);
    double dn = norm(d);
    Vec3 u = dn > 1e-9 ? scale(d, 1.0 / dn) : Vec3{1, 0, 0};
    Vec3 e1 = cross(u, Vec3{0, 0, 1}); e1 = scale(e1, 1.0 / norm(e1));
    Vec3 e2 = cross(u, e1);
    double kept = LQG_NOM_TRACKS;
    double sig_dir = std::sqrt(LQG_NOM_Q / kept) * SIG_DIR_SCALE + SIG_DIR_PX / (LQG_NOM_PAR * std::sqrt(kept)) + SIG_DIR_FLOOR;
    double Rn = sig_dir * sig_dir * (dn * dn + D_FLOOR * D_FLOOR);
    Vec3 es[2] = {e1, e2};
    double Hrow[NS];
    for (int k = 0; k < 2; k++) {
        Vec3 ed = cross(es[k], d);
        row_d(Hrow, es[k]);
        Hrow[ITH] = ed.x; Hrow[ITH + 1] = ed.y; Hrow[ITH + 2] = ed.z;
        ekf_update_row(Hrow, 0.0, Rn);
    }
    if (USE_PAR_SPEED) {
        double sd = PAR_SPEED_REL * dn + 0.01;
        row_d(Hrow, u);
        ekf_update_row(Hrow, 0.0, sd * sd);
    }
    kf_inject();
}
void design_prior() {
    for (int i = 0; i < NS * NS; i++) P_pri[i] = P[i];
    if (sched_on(design_win)) for (int i = 0; i < NS * NS; i++) P_pri_s[design_win][i] = P[i];
    apply_velocity_prior();
}
void design_prior_axis(int a) {
    for (int i = 0; i < NS * NS; i++) P_pri_ax[a][i] = P[i];
    if (sched_on(design_win)) for (int i = 0; i < NS * NS; i++) P_ax_s[design_win][a][i] = P[i];
    vp_axis = a;
    apply_velocity_prior();
    vp_axis = -1;
}
void lqg_design(bool hover) {
    designing = true;
    design_hover = hover;
#if VEL_PRIOR
    vp_par = hover ? 0.0 : 1e9;   // prior mode of the design flight (still / cruise)
#endif
    const double dt = 0.05, ds = dt / 10;
    Vec3 v_nom{hover ? 0.0 : LQG_NOM_SPEED, 0, 0};
    int cyc = 0;
    R_wb = eye(); R_wb_clone = eye();
    for (int k = 0; k < SPLIT; k++) { R_wb_kf[k] = eye(); win_T[k] = 0; }
    p_w = p_kf = {0, 0, 0}; v_w = v_nom; ba = bg = {0, 0, 0};
    acc_T = 0; acc_n = 0; acc_R = Mat3{{0}}; acc_f = {0, 0, 0}; prop_debt = 0; prop_ahead_next = false;
    std::memset(P, 0, sizeof P);
    const bool rest = REST_GYRO_MEAN > 0;   // REST_GYRO_MEAN 0: the start is never treated as at rest (moving start)
    const double sv0 = rest ? SIG_V0_REST : SIG_V0_MOVING, st0 = rest ? SIG_TILT0_REST : SIG_TILT0_MOVING;
    const double sb0 = rest ? SIG_BG0_REST : SIG_BG0_MOVING;
    for (int i = 0; i < 3; i++) {
        P[(IP + i) * NS + IP + i] = 1e-8;
        P[(IV + i) * NS + IV + i] = sv0 * sv0;
        P[(IBA + i) * NS + IBA + i] = (THRUST_ONLY && i < 2) ? 0.0 : SIG_BA0 * SIG_BA0;
        P[(IBG + i) * NS + IBG + i] = sb0 * sb0;
    }
    P[ITH * NS + ITH] = P[(ITH + 1) * NS + ITH + 1] = st0 * st0;
    P[(ITH + 2) * NS + ITH + 2] = 1e-6;
    for (int i = 0; i < NS; i++) dx[i] = 0;
    for (int i = 0; i < NS * NS; i++) P_pri[i] = P_dir[i] = P[i];
    for (int a = 0; a < 3; a++) for (int i = 0; i < NS * NS; i++) P_pri_ax[a][i] = P[i];
    for (int k = 0; k < SPLIT; k++) for (int i = 0; i < NS * NS; i++) P_rot[k][i] = P[i];
    const int frames = (int)(LQG_DESIGN_S / dt);
    int ph = 0;
    design_win = 0;
    for (int f = 0; f < frames; f++) {
        for (int i = 0; i < 9; i++) acc_R.m[i] += 10.0 * (i % 4 == 0);
        acc_f = add(acc_f, Vec3{0, 0, 10.0 * GRAVITY});
        acc_n += 10; acc_T += dt;
        for (int k = 0; k < SPLIT; k++) win_T[k] += dt;
        p_w = add(p_w, scale(v_nom, dt));
        (void)ds;
        ph++;
#if LQG_VISION
        if (ph >= KF_LEN + SPLIT + DIR_OWN) {
            ph = 0;
            design_win++;
            if (LQG_MIX_DIRS && !hover) {
                const double V = LQG_NOM_SPEED, c = 0.7071 * LQG_NOM_SPEED;
                const Vec3 dirs[4] = {{V, 0, 0}, {0, 0, V}, {0, V, 0}, {c, 0, c}};
                v_nom = dirs[++cyc % 4];
            }
        }
        if (ph == 0) {
            if (VPRIOR_IDLE == 2 && !DIR_OWN && f > 0) design_prior();
        } else if (ph <= SPLIT) {
            const int si = ph - 1;
            if (VPRIOR_IDLE == 1 && ph == KF_LEN - 2) design_prior();
            if (ph == 1) { kf_propagate(); kf_clone(); }
            win_T[si] = 0;
        } else if (ph == KF_LEN - 1) {
            prop_ahead_next = true;
            kf_propagate();
        } else if (DIR_OWN && ph == KF_LEN + SPLIT) {
            design_direction();
            if (VPRIOR_IDLE != 1) design_prior();
        } else if (ph >= KF_LEN) {
            design_rotation(ph - KF_LEN);
            if (ph == KF_LEN + SPLIT - 1 && !DIR_OWN) design_direction();
        }
#else
        if (ph >= KF_LEN + SPLIT + DIR_OWN) { ph = 0; design_win++; }
        if (LQG_PRIOR_SPLIT) {
            if (ph == 0) kf_propagate();
            if (ph % 3 == 0 && ph / 3 < 3) { if (ph) kf_propagate(); design_prior_axis(ph / 3); }
        } else if (ph == 0) { kf_propagate(); design_prior(); }
#endif
        if (inject_pending) kf_inject_now();
        v_w = v_nom; R_wb = eye(); ba = bg = {0, 0, 0};   // stay on the nominal trajectory
    }
    designing = false;
    inject_pending = false;
    for (int i = 0; i < NS; i++) dx[i] = 0;
    acc_T = 0; acc_n = 0; acc_R = Mat3{{0}}; acc_f = {0, 0, 0}; prop_debt = 0; prop_ahead_next = false;
}
#endif

bool state_ok() {
    double s = p_w.x + p_w.y + p_w.z + v_w.x + v_w.y + v_w.z + ba.x + ba.y + ba.z + bg.x + bg.y + bg.z;
    for (int i = 0; i < NS; i++) s += P[i * NS + i];
    return std::isfinite(s) && norm(v_w) < VMAX_RESET;
}

}  // namespace

int32_t abi_version() { return VIOBENCH_ABI_VERSION; }
void* alloc(uint32_t size) { return std::malloc(size ? size : 1); }
int32_t frame_width() { return 320; }

int32_t init(const char* json, uint32_t size) {
    double T[16];
    const char* p = find_key(json, size, "\"T_cam_imu\"");
    bool ok = p != nullptr;
    for (int k = 0; ok && k < 16; k++) {
        p = read_number(p, json + size, &T[k]);
        ok = p != nullptr;
    }
    if (!ok)
        for (int i = 0; i < 16; i++) T[i] = (i % 5 == 0) ? 1.0 : 0.0;
    R_ci = {{T[0], T[1], T[2], T[4], T[5], T[6], T[8], T[9], T[10]}};
    Vec3 t_ci{T[3], T[7], T[11]};
    R_ic = transpose(R_ci);
    r_cam_in_imu = scale(mul(R_ic, t_ci), -1.0);

    double w = 320, h = 240, fx = 190, fy = 190, cx = 159.5, cy = 119.5;
    parse_scalar(json, size, "\"width\"", &w);
    parse_scalar(json, size, "\"height\"", &h);
    parse_scalar(json, size, "\"fx\"", &fx);
    parse_scalar(json, size, "\"fy\"", &fy);
    parse_scalar(json, size, "\"cx\"", &cx);
    parse_scalar(json, size, "\"cy\"", &cy);
    W = (int)w; H = (int)h;
    SW = W / DS; SH = H / DS;
    if (SW < 2 * CSZ + 8 || SH < 2 * CSZ + 8) return -1;
    // tracking pixel x averages full pixels around x*DS + DS/2 - 0.5
    fx_s = fx / DS; fy_s = fy / DS;
    cx_s = (cx - (DS / 2 - 0.5)) / DS; cy_s = (cy - (DS / 2 - 0.5)) / DS;

    {   // candidate cell corners (fixed grid)
        const int m = MARGIN + 2;
        for (int k = 0; k < NCAND; k++) {
            int gx = k % GRID_X, gy = k / GRID_X;
            cx_[k] = m + (int)((gx + 0.5) * (SW - PS - 2 * m) / GRID_X);
            cy_[k] = m + (int)((gy + 0.5) * (SH - PS - 2 * m) / GRID_Y);
        }
    }
#if LQG
    if (LQG_SCHED) {
        lqg_design(true);
        for (int i = 0; i < NS * NS; i++) { P_dir_h[i] = P_dir[i]; P_pri_h[i] = P_pri[i]; }
        for (int k = 0; k < SPLIT; k++) for (int i = 0; i < NS * NS; i++) P_rot_h[k][i] = P_rot[k][i];
    }
    lqg_design(false);
    lqg_hover = false;
    lqr_design(0.05);
    c_p = c_v = c_th = {0, 0, 0};
    bn = 0; frame_T = 0; frame_f = {0, 0, 0}; frame_n = 0;
#endif
    R_wb = eye();
    bg = ba = omega_b = {0, 0, 0};
    std::memset(sum_a, 0, sizeof sum_a); std::memset(sum_a2, 0, sizeof sum_a2);
    std::memset(sum_g, 0, sizeof sum_g); std::memset(sum_g2, 0, sizeof sum_g2);
#if VEL_PRIOR == 1
    vp_head = vp_count = vp_n = 0;
    std::memset(vp_sum, 0, sizeof vp_sum);
#endif
#if VEL_PRIOR
    vp_nis[0] = vp_nis[1] = vp_nis[2] = 1.0;
    vp_fwd = VP_CONST_FWD;
    vp_par = VP_STALL_START * VP_STALL_PX;
#endif
    calib_n = 0; calib_t0 = -1;
    initialized = false; have_kf = false;
    return VIOBENCH_STATUS_OK;
}

int32_t step(double t, const uint8_t* frame, const double* imu, int32_t imu_count, double* out) {
    frame_px = frame;
    if (!initialized) {
        for (int k = 0; k < imu_count; k++) {
            const double* r = imu + 7 * k;
            if (calib_t0 < 0) calib_t0 = r[0];
            for (int i = 0; i < 3; i++) {
                sum_a[i] += r[1 + i]; sum_a2[i] += r[1 + i] * r[1 + i];
                sum_g[i] += r[4 + i]; sum_g2[i] += r[4 + i] * r[4 + i];
            }
            calib_n++;
        }
        double last = imu_count > 0 ? imu[7 * (imu_count - 1)] : t;
        if (calib_n < 2 || last - calib_t0 < CALIB_SECONDS) {
            write_identity(out);
            return VIOBENCH_STATUS_OK;
        }
        finish_calibration(last);
        phase = 0;
#if LQG
        frame_T = 0; frame_f = {0, 0, 0}; frame_n = 0;
        fr_n = 0; for (int i = 0; i < 3; i++) fr_g[i] = fr_a[i] = 0;
        still_mode = LQG_STILL_CAL; still_n = 0;
        for (int i = 0; i < 3; i++) still_g[i] = still_a[i] = 0;
        run_win = 0; lam[0] = lam[1] = lam[2] = 1.0;
#endif
#if !(LQG && !LQG_VISION)
        make_keyframe();
#endif
        write_state(out);
        return VIOBENCH_STATUS_OK;
    }

    integrate(imu, imu_count);
#if LQG
    lqr_step();
    if (LQG_STILL_CAL) still_step();
#endif
#if LQG && !LQG_VISION
    // IMU only: one velocity-prior batch per window period, no image access
    phase++;
    if (phase >= KF_LEN + SPLIT + DIR_OWN) { phase = 0; run_win++; }
    if (LQG_PRIOR_SPLIT) {
        if (phase % 3 == 0 && phase / 3 < 3) { vp_axis = phase / 3; apply_velocity_prior(); vp_axis = -1; }
    } else if (phase == 0) apply_velocity_prior();
    if (false)
#endif
    {
    // window cycle of KF_LEN + SPLIT (+ DIR_OWN) frames:
    //   phase 0                         rank candidate cells (v8: + the velocity prior of the last window)
    //   phase 1 .. SPLIT                cut one template subset per frame; phase 1 also clones the position
    //   phase KF_LEN - 1                propagate the covariance, ahead over the tracking frames
    //   phase KF_LEN .. KF_LEN+SPLIT-1  track one subset per frame; v8: the last one also runs the direction update
    //   phase KF_LEN + SPLIT            lite (DIR_OWN): the direction update on its own frame
    phase++;
    if (phase >= KF_LEN + SPLIT + DIR_OWN) {
        phase = 0;
#if LQG
        run_win++;
#endif
    }
    if (phase == 0) {
        if (VPRIOR_IDLE == 2 && have_kf && !DIR_OWN) apply_velocity_prior();  // right after the direction update, on a lighter frame
        make_keyframe();
    } else if (phase <= SPLIT) {
        const int si = phase - 1;
        if (VPRIOR_IDLE == 1 && phase == KF_LEN - 2 && have_kf) apply_velocity_prior();  // lite: on a template frame
        if (phase == 1) { kf_propagate(); kf_clone(); R_wb_clone = R_wb; }  // off the ranking frame
        R_wb_kf[si] = R_wb;
        win_T[si] = 0;
        int want = (N_TRACK - n_patches + (SPLIT - si) - 1) / (SPLIT - si);
        if (SUB_CAP > 0 && want > SUB_CAP) want = SUB_CAP;  // no catch-up on the last subset (its frame also runs the direction update)
        cut_templates(si, want);
    } else if (phase == KF_LEN - 1) {
        prop_ahead_next = true;
        kf_propagate();                      // idle frame: covariance up to the direction update
    } else if (DIR_OWN && phase == KF_LEN + SPLIT) {
        if (have_kf) { direction_update(); if (VPRIOR_IDLE != 1) apply_velocity_prior(); }
    } else if (phase >= KF_LEN && have_kf) {
        track_subset(phase - KF_LEN);
        if (phase == KF_LEN + SPLIT - 1 && !DIR_OWN) direction_update();
    }
    }
    if (inject_pending) kf_inject_now();
    if (!state_ok()) {  // never output garbage: restart the filter around the current pose
        v_w = {0, 0, 0};
        ba = {0, 0, 0};
        if (!std::isfinite(p_w.x + p_w.y + p_w.z)) p_w = {0, 0, 0};
        std::memset(P, 0, sizeof P);
        for (int i = 0; i < NS; i++) P[i * NS + i] = 1e-2;
        for (int i = 0; i < NS; i++) dx[i] = 0;
        acc_T = 0; acc_n = 0; acc_R = Mat3{{0}}; acc_f = {0, 0, 0};
#if LQG
        c_p = c_v = c_th = {0, 0, 0};
        bn = 0;
#endif
        p_kf = p_w;
    }
    write_state(out);
    return VIOBENCH_STATUS_OK;
}
