// msckf: an OpenVINS-style Multi-State Constraint Kalman Filter for viobench (see README.md).
//
// State (error-state EKF, world frame = the first IMU frame, so the output needs no alignment):
//   IMU: rotation R_wi (global perturbation), position, velocity, gyro bias, accel bias, gravity vector g_w (3)
//   clones: the IMU pose (rotation, position) at each of the last N_CLONES frames.
// Features: KLT tracks on the green channel (coarse + fine level, gyro-predicted start, brightness offset).
// A feature is used once: when its track ends or spans the whole window. It is triangulated (anchored inverse depth,
// Gauss-Newton), its reprojection Jacobians are projected onto the left null space of the feature Jacobian (no
// landmark in the state), chi^2-gated, and all features of the frame form one EKF update.
#include "viobench_abi.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#ifndef __wasm__
#include <cstdio>
#endif
static int dbg_rows = 0, nrows_last = 0;

// ---------------------------------------------------------------- parameters (all overridable with -D)
#ifndef N_CLONES
#define N_CLONES 11
#endif
#ifndef MAX_FEATS
#define MAX_FEATS 32
#endif
#ifndef GRID_X
#define GRID_X 8
#endif
#ifndef GRID_Y
#define GRID_Y 6
#endif
#ifndef DETECT_CELLS
#define DETECT_CELLS 48          // max empty cells scored per frame
#endif
#ifndef DET_STEP
#define DET_STEP 4               // candidate lattice step (px) inside a cell
#endif
#ifndef MIN_SCORE
#define MIN_SCORE 10.0f          // Shi-Tomasi min eigenvalue threshold (mean per pixel, gray^2)
#endif
#ifndef TPL_BILINEAR
#define TPL_BILINEAR 1           // templates sampled at the exact sub-pixel position (else integer grid)
#endif
#ifndef DET_HALF
#define DET_HALF 0
#endif
#ifndef PRETEST
#define PRETEST 6                // skip a cell whose sparse max |gradient| (2-px differences) is below this
#endif
#ifndef COOL
#define COOL 2                   // frames a cell with no corner is skipped
#endif
#ifndef PR
#define PR 4                     // patch radius: (2PR+1)^2 samples
#endif
#ifndef LK_COARSE
#define LK_COARSE 0
#endif
#ifndef LK_FINE
#define LK_FINE 6
#endif
#ifndef MAX_RES
#define MAX_RES 14.0f            // max mean |photometric residual| after LK
#endif
#ifndef SIG_PX
#define SIG_PX 1.0               // reprojection noise (px)
#endif
#ifndef SIG_ACC
#define SIG_ACC 0.28             // accelerometer noise per 200 Hz sample (m/s^2)
#endif
#ifndef SIG_GYR
#define SIG_GYR 0.07             // gyro noise per sample (rad/s)
#endif
#ifndef SIG_BGW
#define SIG_BGW 2e-4             // gyro bias random walk (rad/s / sqrt(s))
#endif
#ifndef SIG_BAW
#define SIG_BAW 2e-3             // accel bias random walk (m/s^2 / sqrt(s))
#endif
#ifndef SIG_V0
#define SIG_V0 1.0
#endif
#ifndef SIG_TILT0
#define SIG_TILT0 0.15
#endif
#ifndef SIG_BG0
#define SIG_BG0 0.03
#endif
#ifndef SIG_BA0
#define SIG_BA0 0.1
#endif
#ifndef OBS_MAX
#define OBS_MAX 5                // observations per feature used in an update (evenly spaced)
#endif
#ifndef CHI2_ON
#define CHI2_ON 0
#endif
#ifndef CHI2_MULT
#define CHI2_MULT 1.0
#endif
#ifndef MAX_ROWS
#define MAX_ROWS 24
#endif
#ifndef MIN_PARALLAX
#define MIN_PARALLAX 0.02        // rad between first and last bearing
#endif
#ifndef VPRIOR_SIG
#define VPRIOR_SIG 0.8           // >0: weak zero-velocity pseudo-measurement (m/s) once per VPRIOR_EVERY frames
#endif
#ifndef VPRIOR_EVERY
#define VPRIOR_EVERY 10
#endif
#ifndef MAX_DEPTH
#define MAX_DEPTH 40.0
#endif
#ifndef MIN_DEPTH
#define MIN_DEPTH 0.15
#endif

#ifndef ZV_PX
#define ZV_PX 0.25               // median derotated flow (px/frame) below which the drone is taken as still
#endif
#ifndef ZV_SIG
#define ZV_SIG 0.15              // zero-velocity update sigma (m/s)
#endif
#ifndef ZV_GYR
#define ZV_GYR 0.2               // and the body rotation rate below this (rad/s)
#endif
#ifndef ZV_DEROT
#define ZV_DEROT 1               // stillness test on flow after removing a fitted residual rotation
#endif
#ifndef UPD_F32
#define UPD_F32 0                // 1: EKF update intermediates (H P, G) in single precision (4-wide SIMD); P stays double
#endif
#if UPD_F32
typedef float upd_t;
#else
typedef double upd_t;
#endif
#ifndef LK_EPS2
#define LK_EPS2 0.0009f          // LK stops when the squared step is below this (0.03 px)
#endif
#ifndef MS_LOW_N
#define MS_LOW_N 0               // with fewer than this many active tracks, detection accepts corners down to MIN_SCORE_LOW
#endif
#ifndef MIN_SCORE_LOW
#define MIN_SCORE_LOW 5.0f
#endif
#ifndef DET_REFILL_N
#define DET_REFILL_N 0           // >0: when more than this many tracks are missing, detection scores a DET_STEP_REFILL lattice
#endif
#ifndef DET_STEP_REFILL
#define DET_STEP_REFILL 8
#endif
#ifndef LOST_DEFER
#define LOST_DEFER 0             // 1: rows of tracks lost this frame are computed next frame (separates them from the refill)
#endif
#ifndef STAGGER
#define STAGGER 0                // >0: a live track finishes after N_CLONES - (slot % (STAGGER+1)) observations, so tracks
#endif                           //     detected together do not all finish (and need update rows) on the same frame
#ifndef SCHED
#define SCHED 0                  // 1: per-frame fuel budget SCHED_B: update rows, splat search and detection cells are
#endif                           //    capped from a linear cost model; what does not fit waits for the next frames
#ifndef SCHED_B
#define SCHED_B 2400000.0
#endif
#ifndef SC_BASE
#define SC_BASE 330000.0
#endif
#ifndef SC_TRACK
#define SC_TRACK 18000.0
#endif
#ifndef SC_ROW
#define SC_ROW 67000.0
#endif
#ifndef SC_CELL
#define SC_CELL 44000.0
#endif
#ifndef SC_SPLAT
#define SC_SPLAT 166000.0
#endif
#ifndef SC_MINCELLS
#define SC_MINCELLS 2
#endif
#ifndef SC_NEED_K
#define SC_NEED_K 1.3            // SCHED=2: cells reserved per missing track (some cells find no corner)
#endif
#ifndef SC_MINROWS
#define SC_MINROWS 6
#endif
#ifndef RESET_SOFT
#define RESET_SOFT 0             // 1: a triangulation-failure reset only inflates the velocity covariance (SOFT_SIG)
#endif
#ifndef SOFT_SIG
#define SOFT_SIG 0.5
#endif
#ifndef LK_TPRED
#define LK_TPRED 0               // 1: LK starts from the rotation AND the filter's predicted translation (at the mean depth)
#endif
#ifndef MANHATTAN
#define MANHATTAN 0              // 1: Manhattan-world heading measurement (yaw about gravity), see manhattan_step()
#endif
#ifndef MH_EVERY
#define MH_EVERY 5
#endif
#ifndef MH_STEP
#define MH_STEP 10
#endif
#ifndef MH_G2
#define MH_G2 900.0
#endif
#ifndef MH_VERT
#define MH_VERT 0.15
#endif
#ifndef MH_COH
#define MH_COH 0.3
#endif
#ifndef MH_T0
#define MH_T0 20
#endif
#ifndef MH_GATE
#define MH_GATE 0.17
#endif
#ifndef MH_SIG
#define MH_SIG 0.05
#endif
#ifndef MH_SIGN
#define MH_SIGN 1.0
#endif
#ifndef SPL_SKIP_ROWS
#define SPL_SKIP_ROWS 1000       // skip the splat search on frames whose feature update used more rows than this
#endif
#ifndef DET_SPARSE
#define DET_SPARSE 0             // 1: corner score from 9 gradient samples 2 px apart instead of a dense 5x5 window
#endif
#ifndef LB_ROWS
#define LB_ROWS 1000             // load balancing: if this frame's feature update has more rows than this ...
#endif
#ifndef LB_CELLS
#define LB_CELLS 8               // ... score at most this many cells (the rest wait for the next frames)
#endif
#ifndef IMU_FILT
#define IMU_FILT 0               // IMU input conditioning: 0 raw, 1 trapezoidal (mean of consecutive samples), 2 low-pass
#endif
#ifndef IMU_TAU
#define IMU_TAU 0.01             // s: low-pass time constant (IMU_FILT=2)
#endif
#ifndef VP_ADAPT
#define VP_ADAPT 0               // 1: velocity-prior sigma = VP_K x the flight's recent RMS speed, in [VP_MIN, VPRIOR_SIG]
#endif
#ifndef VP_K
#define VP_K 1.5
#endif
#ifndef VP_MIN
#define VP_MIN 0.3
#endif
#ifndef VP_TAU
#define VP_TAU 10.0
#endif
#ifndef LOWPAR
#define LOWPAR 0                 // 1: low-parallax tracks give rotation-only rows (point placed at LP_DEPTH)
#endif
#ifndef LP_BASE
#define LP_BASE 1e9              // m: use rotation-only rows only if the clone baseline over the track is below this
#endif
#ifndef LP_DEPTH
#define LP_DEPTH 10.0
#endif
#ifndef LP_SIG_PX
#define LP_SIG_PX 2.0
#endif
#ifndef LP_MAX_PX
#define LP_MAX_PX 3.0
#endif
#ifndef FEJ
#define FEJ 0                    // first-estimate Jacobians (OpenVINS): bit 1 feature updates, bit 2 propagation
#endif
#ifndef ANCHOR
#define ANCHOR 0                 // 1: refine each LK result against the template cut at the start of the observation run
#endif
#ifndef ANCHOR_IT
#define ANCHOR_IT 3
#endif
#ifndef ANCHOR_RES
#define ANCHOR_RES 10.0f
#endif
#ifndef TWO_RATE
#define TWO_RATE 0               // 1: newest DENSE clones one per frame, older clones thinned to >= SPACING frames apart
#endif
#ifndef DENSE
#define DENSE 5
#endif
#ifndef SPACING
#define SPACING 3
#endif
#ifndef KF_PX
#define KF_PX 0.0                // >0: add a clone only when the median rotation-compensated feature motion since the
#endif                           //     last clone reaches KF_PX pixels (or after KF_MAXGAP frames); 0 = every frame
#ifndef KF_MAXGAP
#define KF_MAXGAP 4
#endif
#ifndef ZV_WIN
#define ZV_WIN 1.5               // >0: stillness = median window disparity (px) below this (falls back to ZV_PX)
#endif
#ifndef ZV_SOFT
#define ZV_SOFT 0                // 1: zero-velocity sigma = ZV_SIG (1 + (disparity / ZV_WIN)^2), applied up to ZV_SOFT_MAX px
#endif
#ifndef ZV_SOFT_MAX
#define ZV_SOFT_MAX 3.0
#endif
#ifndef ZV_MOVE_WIN
#define ZV_MOVE_WIN 4.0          // ZV_START_ONLY=2: window disparity above this for 5 frames = the flight has started
#endif
#ifndef ZW_PCT
#define ZW_PCT 50                // percentile of the window disparity used for stillness (forward flight: centre is still)
#endif
#ifndef ZW_MIN_SPAN
#define ZW_MIN_SPAN 8
#endif
#ifndef ZV_START_ONLY
#define ZV_START_ONLY 0          // zero-velocity updates only until the drone first moves
#endif
#ifndef ZV_MOVE_PX
#define ZV_MOVE_PX 0.8           // median flow above this for 3 frames = the drone has started moving
#endif
#ifndef RESET_GEOM
#define RESET_GEOM 1             // divergence statistic ignores low-parallax failures
#endif
#ifndef TRI_MAX_PX
#define TRI_MAX_PX 2.0           // max reprojection error (px) after triangulation
#endif
#ifndef RECOVER_SIG
#define RECOVER_SIG 0.5          // velocity sigma added when every feature is rejected for RECOVER_N update frames
#endif
#ifndef RECOVER_N
#define RECOVER_N 3
#endif

#ifndef RESET_WIN
#define RESET_WIN 10             // frames over which triangulation success is counted
#endif
#ifndef RESET_RATE
#define RESET_RATE 0.12          // reset when fewer than this fraction of candidate features triangulate ...
#endif
#ifndef RESET_MIN
#define RESET_MIN 15             // ... out of at least this many candidates
#endif
#ifndef VMAX
#define VMAX 3.0                 // or when the speed exceeds this (m/s)
#endif
#ifndef RESET_VIS
#define RESET_VIS 1              // after a reset, start from the visual velocity guess instead of zero
#endif
#ifndef SIG_VVIS
#define SIG_VVIS 0.4
#endif
#ifndef SIG_VRESET
#define SIG_VRESET 1.0
#endif

static const int NI = 18;                          // IMU error states
static const int NMAX = NI + 6 * N_CLONES;
static const int TH = 0, PS = 3, VE = 6, BG = 9, BA = 12, GR = 15;

// ---------------------------------------------------------------- small math
struct M3 { double m[9]; };
static inline M3 mmul(const M3& a, const M3& b) {
    M3 r;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
        r.m[3 * i + j] = a.m[3 * i] * b.m[j] + a.m[3 * i + 1] * b.m[3 + j] + a.m[3 * i + 2] * b.m[6 + j];
    return r;
}
static inline M3 mtr(const M3& a) { M3 r; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) r.m[3 * i + j] = a.m[3 * j + i]; return r; }
static inline void mv(const M3& a, const double* v, double* o) {
    double x = v[0], y = v[1], z = v[2];
    o[0] = a.m[0] * x + a.m[1] * y + a.m[2] * z; o[1] = a.m[3] * x + a.m[4] * y + a.m[5] * z; o[2] = a.m[6] * x + a.m[7] * y + a.m[8] * z;
}
static inline void mtv(const M3& a, const double* v, double* o) {
    double x = v[0], y = v[1], z = v[2];
    o[0] = a.m[0] * x + a.m[3] * y + a.m[6] * z; o[1] = a.m[1] * x + a.m[4] * y + a.m[7] * z; o[2] = a.m[2] * x + a.m[5] * y + a.m[8] * z;
}
static inline M3 skew(const double* v) { M3 r = {{0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0}}; return r; }
static inline M3 expso3(const double* w) {
    double th2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2], th = std::sqrt(th2), a, b;
    if (th < 1e-7) { a = 1 - th2 / 6; b = 0.5 - th2 / 24; }
    else { a = std::sin(th) / th; b = (1 - std::cos(th)) / th2; }
    M3 K = skew(w), K2 = mmul(K, K), r;
    for (int i = 0; i < 9; i++) r.m[i] = (i % 4 == 0 ? 1.0 : 0.0) + a * K.m[i] + b * K2.m[i];
    return r;
}
static inline void orthonormalize(M3& R) {   // Gram-Schmidt on rows
    double* a = R.m; double* b = R.m + 3; double* c = R.m + 6;
    double n = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); for (int i = 0; i < 3; i++) a[i] /= n;
    double d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; for (int i = 0; i < 3; i++) b[i] -= d * a[i];
    n = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]); for (int i = 0; i < 3; i++) b[i] /= n;
    c[0] = a[1] * b[2] - a[2] * b[1]; c[1] = a[2] * b[0] - a[0] * b[2]; c[2] = a[0] * b[1] - a[1] * b[0];
}
static void rot2quat(const M3& R, double* q) {   // q = [x y z w]
    const double* m = R.m; double tr = m[0] + m[4] + m[8];
    if (tr > 0) { double s = std::sqrt(tr + 1.0) * 2; q[3] = 0.25 * s; q[0] = (m[7] - m[5]) / s; q[1] = (m[2] - m[6]) / s; q[2] = (m[3] - m[1]) / s; }
    else if (m[0] > m[4] && m[0] > m[8]) { double s = std::sqrt(1.0 + m[0] - m[4] - m[8]) * 2; q[3] = (m[7] - m[5]) / s; q[0] = 0.25 * s; q[1] = (m[1] + m[3]) / s; q[2] = (m[2] + m[6]) / s; }
    else if (m[4] > m[8]) { double s = std::sqrt(1.0 + m[4] - m[0] - m[8]) * 2; q[3] = (m[2] - m[6]) / s; q[0] = (m[1] + m[3]) / s; q[1] = 0.25 * s; q[2] = (m[5] + m[7]) / s; }
    else { double s = std::sqrt(1.0 + m[8] - m[0] - m[4]) * 2; q[3] = (m[3] - m[1]) / s; q[0] = (m[2] + m[6]) / s; q[1] = (m[5] + m[7]) / s; q[2] = 0.25 * s; }
    double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]); for (int i = 0; i < 4; i++) q[i] /= n;
}

// ---------------------------------------------------------------- globals
static int W = 320, H = 240;
static double fx = 120, fy = 120, cx = 159.5, cy = 119.5;
static M3 Rci;                      // camera <- imu
static double tci[3];
static const uint8_t* img = nullptr; // current frame (RGB)

static M3 Rwi; static double pw[3], vw[3], bg[3], ba[3], gw[3];
static double P[NMAX * NMAX];
// Schmidt map (SCHMIDT): cross-covariance between the state and the active splats (slots 0..nsch-1, 3 columns each)
// and the active splats' own covariance. The map points are never updated; their correlations with the state are.
#ifndef SCHMIDT
#define SCHMIDT 0
#endif
#ifndef SCH_N
#define SCH_N 12                 // active Schmidt splats
#endif
#ifndef SCH_INF
#define SCH_INF 1.0              // inflation of the map covariance in the innovation covariance
#endif
#ifndef SCH_EVICT
#define SCH_EVICT 200            // frames without a match before an active splat may be evicted for a new one
#endif
static const int SCH_MC = 3 * (SCHMIDT ? SCH_N : 1);
#ifndef SCH_F32
#define SCH_F32 0               // 1: cross-covariance in single precision (half the SIMD work)
#endif
#ifndef SCH_PROF
#define SCH_PROF 0              // fuel profiling only: 1 skips the update term, 2 the propagation (wrong results)
#endif
#if SCH_F32
typedef float sch_t;
#else
typedef double sch_t;
#endif
static sch_t Xc[NMAX * 3 * (SCHMIDT ? SCH_N : 1)]; static double Pmm[9 * (SCHMIDT ? SCH_N : 1) * (SCHMIDT ? SCH_N : 1)];
static int nsch = 0, sch_splat[SCHMIDT ? SCH_N : 1];
static long st_sch_upd = 0;
#define XC(i, c) Xc[(i) * SCH_MC + (c)]
#define PM(a, b) Pmm[(a) * SCH_MC + (b)]
static int n_state = NI;
struct Clone { M3 R; double p[3]; int serial; M3 Rf; double pf[3]; };   // Rf, pf: first estimates (FEJ)
static Clone clones[N_CLONES];
static int n_clones = 0, frame_no = 0;
static double t_last = 0;
static bool started = false;
static double w_last[3];

struct Obs { int serial; float u, v; };     // normalized image coordinates
struct Feat {
    bool active;
    float x, y;               // pixel position in the last frame
    float fdx, fdy;           // last derotated pixel motion (constant-velocity prediction)
    float tf[(2 * PR + 1) * (2 * PR + 1)], gxf[(2 * PR + 1) * (2 * PR + 1)], gyf[(2 * PR + 1) * (2 * PR + 1)];
    float tc[(2 * PR + 1) * (2 * PR + 1)], gxc[(2 * PR + 1) * (2 * PR + 1)], gyc[(2 * PR + 1) * (2 * PR + 1)];
    float Hf[6], Hc[6];       // inverse 3x3 normal matrices (symmetric, upper)
    float ox, oy;             // feature position minus template centre
    bool anchored;            // anchor template valid (ANCHOR)
    int nobs; Obs obs[N_CLONES + 1];
};
static Feat feats[MAX_FEATS], sub_feat, sub_feat2;
static double tri_al, tri_be, tri_rho; static float tri_hi[6]; static M3 tri_Ra;   // last triangulation (splat map)
static float fl_u[MAX_FEATS], fl_v[MAX_FEATS], fl_du[MAX_FEATS], fl_dv[MAX_FEATS];
static int nflow = 0;
struct Pending { int nobs; int fidx; Obs obs[N_CLONES + 1]; };
#ifndef MAX_PENDING
#define MAX_PENDING 24
#endif
static Pending pend[MAX_PENDING]; static int npend = 0;
static int det_rr = 0;
static bool is_kf = true;
static int cool[GRID_X * GRID_Y];
static int n_tried = 0, n_ok = 0, bad_streak = 0;
static double depth_avg = 2.5, frame_dt = 0.05;
static long st_far = 0;
static bool det_refill = false;
static int row_cap = MAX_ROWS, frame_tracks = 0;   // frame budget scheduler (SCHED)
static double sched_used = 0;
static int sched_need = 0;
static int last_reset_frame = -100000;
static double tpred_c[3] = {0, 0, 0};
static double mh_C = 0, mh_S = 0, mh_W = 0, mh_ref = 0, mh_e1[3], mh_e2[3];
static bool mh_ref_ok = false, mh_basis = false;
static long st_mh_upd = 0;

static int frame_rows = 0;
static double v2_avg = 0.25;
static int frame_tri_fail = 0, zupt_count = 0; static bool tri_par = false, moved = false; static int move_frames = 0;
static int ring_c[RESET_WIN], ring_s[RESET_WIN], n_resets = 0;
static long st_cells = 0, st_tracks = 0;
static long st_cand = 0, st_tri = 0, st_chi = 0, st_ok = 0, st_par = 0, st_rep = 0;

// ---------------------------------------------------------------- image access
static inline float pix(int x, int y) { return (float)img[(y * W + x) * 3 + 1]; }
static inline float bil(float x, float y) {
    int x0 = (int)x, y0 = (int)y; float ax = x - x0, ay = y - y0;
    const uint8_t* p = img + (y0 * W + x0) * 3 + 1;
    float a = p[0], b = p[3], c = p[3 * W], d = p[3 * W + 3];
    return (a + (b - a) * ax) + ((c + (d - c) * ax) - (a + (b - a) * ax)) * ay;
}
static inline bool inside(float x, float y, float m) { return x >= m && y >= m && x < W - 1 - m && y < H - 1 - m; }

// symmetric 3x3 inverse of [a b c; b d e; c e f] -> 6 values
static bool inv3s(double a, double b, double c, double d, double e, double f, float* o) {
    double A = d * f - e * e, B = c * e - b * f, C = b * e - c * d, D = a * f - c * c, E = b * c - a * e, F = a * d - b * b;
    double det = a * A + b * B + c * C;
    if (!(det > 1e-9)) return false;
    double id = 1.0 / det;
    o[0] = (float)(A * id); o[1] = (float)(B * id); o[2] = (float)(C * id); o[3] = (float)(D * id); o[4] = (float)(E * id); o[5] = (float)(F * id);
    return true;
}

#if TPL_BILINEAR
// templates sampled at the exact sub-pixel feature position: fine (stride 1) and coarse (stride 2). Every sample of a
// level shares the same bilinear weights, so a level is one (S+2)^2 grid of samples; gradients are grid differences.
#ifndef IMG_BLUR
#define IMG_BLUR 0               // 1: track on a 3x3 binomial-smoothed image (test only: ~9x sampling cost)
#endif
static inline float bpx(const uint8_t* p) {   // 3x3 [1 2 1] smoothed green pixel at p
    const int r = 3 * W;
    return (float)(p[-r - 3] + 2 * p[-r] + p[-r + 3] + 2 * p[-3] + 4 * p[0] + 2 * p[3] + p[r - 3] + 2 * p[r] + p[r + 3]) * (1.0f / 16);
}
#if IMG_BLUR
#define PX0(p) bpx(p)
#define PX1(p) bpx((p) + 3)
#define PX2(p) bpx((p) + 3 * W)
#define PX3(p) bpx((p) + 3 * W + 3)
#else
#define PX0(p) (p)[0]
#define PX1(p) (p)[3]
#define PX2(p) (p)[3 * W]
#define PX3(p) (p)[3 * W + 3]
#endif
static inline void grid_sample(float x0, float y0, int st, int n, float* g) {
    const int ix = (int)x0, iy = (int)y0; const float ax = x0 - ix, ay = y0 - iy;
    const float w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
    const int sx = 3 * st, sy = 3 * W * st;
    const uint8_t* base = img + (iy * W + ix) * 3 + 1;
    for (int j = 0; j < n; j++) {
        const uint8_t* p = base + j * sy;
        for (int i = 0; i < n; i++, p += sx) g[j * n + i] = w00 * PX0(p) + w10 * PX1(p) + w01 * PX2(p) + w11 * PX3(p);
    }
}
static bool cut_template(Feat& f) {
    f.ox = f.oy = 0;
    const int S = 2 * PR + 1, E = S + 2;
    if (!inside(f.x, f.y, 2 * PR + 3)) return false;
    float g[(2 * PR + 3) * (2 * PR + 3)];
    for (int lvl = 0; lvl < (LK_COARSE > 0 ? 2 : 1); lvl++) {
        const int st = lvl ? 2 : 1;
        float* t = lvl ? f.tc : f.tf; float* gx = lvl ? f.gxc : f.gxf; float* gy = lvl ? f.gyc : f.gyf;
        grid_sample(f.x - (PR + 1) * st, f.y - (PR + 1) * st, st, E, g);
        double a = 0, b = 0, c = 0, d = 0, e = 0;
        for (int j = 0; j < S; j++) for (int i = 0; i < S; i++) {
            const float* q = g + (j + 1) * E + (i + 1);
            float dx = 0.5f * (q[1] - q[-1]), dy = 0.5f * (q[E] - q[-E]);
            t[j * S + i] = q[0]; gx[j * S + i] = dx; gy[j * S + i] = dy;
            a += dx * dx; b += dx * dy; c -= dx; d += dy * dy; e -= dy;
        }
        if (!inv3s(a, b, c, d, e, (double)(S * S), lvl ? f.Hc : f.Hf)) return false;
    }
    return true;
}

// anchor template (stride 1, in the coarse slot): cut when a track's observation sequence starts
static bool cut_anchor(Feat& f) {
    const int S = 2 * PR + 1, E = S + 2;
    if (!inside(f.x, f.y, 2 * PR + 3)) return false;
    float g[(2 * PR + 3) * (2 * PR + 3)];
    grid_sample(f.x - (PR + 1), f.y - (PR + 1), 1, E, g);
    double a = 0, b = 0, c = 0, d = 0, e = 0;
    for (int j = 0; j < S; j++) for (int i = 0; i < S; i++) {
        const float* q = g + (j + 1) * E + (i + 1);
        float dx = 0.5f * (q[1] - q[-1]), dy = 0.5f * (q[E] - q[-E]);
        f.tc[j * S + i] = q[0]; f.gxc[j * S + i] = dx; f.gyc[j * S + i] = dy;
        a += dx * dx; b += dx * dy; c -= dx; d += dy * dy; e -= dy;
    }
    f.anchored = inv3s(a, b, c, d, e, (double)(S * S), f.Hc);
    return f.anchored;
}

// LK with brightness offset; returns mean |residual| or -1 on failure
static float lk_level(const Feat& f, int lvl, float& x, float& y, float& off, int iters) {
    const int S = 2 * PR + 1;
    const int st = lvl == 1 ? 2 : 1;
    const float* t = lvl ? f.tc : f.tf; const float* gx = lvl ? f.gxc : f.gxf; const float* gy = lvl ? f.gyc : f.gyf;
    const float* Hi = lvl ? f.Hc : f.Hf;
    float res = 0;
    for (int it = 0; it < iters; it++) {
        const float x0 = x - PR * st, y0 = y - PR * st;
        if (!inside(x0, y0, 1 + IMG_BLUR) || !inside(x0 + 2 * PR * st, y0 + 2 * PR * st, 1 + IMG_BLUR)) return -1;
        const int ix = (int)x0, iy = (int)y0; const float ax = x0 - ix, ay = y0 - iy;
        const float w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
        const int sx = 3 * st, sy = 3 * W * st;
        const uint8_t* base = img + (iy * W + ix) * 3 + 1;
        float bx = 0, by = 0, bo = 0; res = 0;
        for (int j = 0; j < S; j++) {
            const uint8_t* p = base + j * sy; const int r = j * S;
            for (int i = 0; i < S; i++, p += sx) {
                float e = w00 * PX0(p) + w10 * PX1(p) + w01 * PX2(p) + w11 * PX3(p) - t[r + i] - off;
                bx += gx[r + i] * e; by += gy[r + i] * e; bo -= e; res += std::fabs(e);
            }
        }
        float dx = -(Hi[0] * bx + Hi[1] * by + Hi[2] * bo);
        float dy = -(Hi[1] * bx + Hi[3] * by + Hi[4] * bo);
        float doff = -(Hi[2] * bx + Hi[4] * by + Hi[5] * bo);
        x += dx; y += dy; off += doff;
        if (dx * dx + dy * dy < LK_EPS2) break;
    }
    return res / (S * S);
}

#else
// cut templates at integer pixels around round(x, y): fine (stride 1) and coarse (stride 2, 2x2 box means, sample
// centres at +0.5). The feature sits at template centre + (ox, oy).
static bool cut_template(Feat& f) {
    const int S = 2 * PR + 1, E = S + 2;
    const int xi = (int)(f.x + 0.5f), yi = (int)(f.y + 0.5f);
    if (xi < 4 * PR + 6 || yi < 4 * PR + 6 || xi >= W - 4 * PR - 7 || yi >= H - 4 * PR - 7) return false;
    f.ox = f.x - xi; f.oy = f.y - yi;
    float g[(2 * PR + 3) * (2 * PR + 3)];
    for (int lvl = 0; lvl < 2; lvl++) {
        float* t = lvl ? f.tc : f.tf; float* gx = lvl ? f.gxc : f.gxf; float* gy = lvl ? f.gyc : f.gyf;
        if (!lvl) {
            for (int j = 0; j < E; j++) { const uint8_t* row = img + ((yi + j - PR - 1) * W + (xi - PR - 1)) * 3 + 1; for (int i = 0; i < E; i++) g[j * E + i] = row[3 * i]; }
        } else {
            for (int j = 0; j < E; j++) {
                const uint8_t* r0 = img + ((yi + 2 * (j - PR - 1)) * W + (xi - 2 * PR - 2)) * 3 + 1; const uint8_t* r1 = r0 + 3 * W;
                for (int i = 0; i < E; i++) g[j * E + i] = 0.25f * ((float)r0[6 * i] + r0[6 * i + 3] + r1[6 * i] + r1[6 * i + 3]);
            }
        }
        double a = 0, b = 0, c = 0, d = 0, e = 0;
        for (int j = 0; j < S; j++) for (int i = 0; i < S; i++) {
            const float* q = g + (j + 1) * E + (i + 1);
            float dx = 0.5f * (q[1] - q[-1]), dy = 0.5f * (q[E] - q[-E]);
            t[j * S + i] = q[0]; gx[j * S + i] = dx; gy[j * S + i] = dy;
            a += dx * dx; b += dx * dy; c -= dx; d += dy * dy; e -= dy;
        }
        if (!inv3s(a, b, c, d, e, (double)(S * S), lvl ? f.Hc : f.Hf)) return false;
    }
    return true;
}

// LK (translation + brightness offset) of the template centre (x, y); returns mean |residual| or -1
static float lk_level(const Feat& f, int lvl, float& x, float& y, float& off, int iters) {
    const int S = 2 * PR + 1;
    const float st = lvl ? 2.0f : 1.0f, o0 = lvl ? 0.5f : 0.0f;
    const float* t = lvl ? f.tc : f.tf; const float* gx = lvl ? f.gxc : f.gxf; const float* gy = lvl ? f.gyc : f.gyf;
    const float* Hi = lvl ? f.Hc : f.Hf;
    float res = 0;
    for (int it = 0; it < iters; it++) {
        const float x0 = x - PR * st + o0, y0 = y - PR * st + o0;
        if (!inside(x0, y0, 1 + IMG_BLUR) || !inside(x0 + 2 * PR * st, y0 + 2 * PR * st, 1 + IMG_BLUR)) return -1;
        // bilinear weights are the same for every sample (stride is an integer)
        const int ix = (int)x0, iy = (int)y0; const float ax = x0 - ix, ay = y0 - iy;
        const float w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
        const int sx = lvl ? 6 : 3, sy = lvl ? 6 * W : 3 * W;
        float bx = 0, by = 0, bo = 0; res = 0;
        const uint8_t* base = img + (iy * W + ix) * 3 + 1;
        for (int j = 0; j < S; j++) {
            const uint8_t* p = base + j * sy;
            for (int i = 0; i < S; i++, p += sx) {
                float v = w00 * p[0] + w10 * p[3] + w01 * p[3 * W] + w11 * p[3 * W + 3];
                float e = v - t[j * S + i] - off;
                bx += gx[j * S + i] * e; by += gy[j * S + i] * e; bo -= e; res += std::fabs(e);
            }
        }
        float dx = -(Hi[0] * bx + Hi[1] * by + Hi[2] * bo);
        float dy = -(Hi[1] * bx + Hi[3] * by + Hi[4] * bo);
        float doff = -(Hi[2] * bx + Hi[4] * by + Hi[5] * bo);
        x += dx; y += dy; off += doff;
        if (dx * dx + dy * dy < 0.0025f) break;
    }
    return res / (S * S);
}

#endif
// camera-frame velocity guess from this frame's derotated flow: direction from the epipolar normals, magnitude from
// the flow size at the average triangulated depth. Returns false if the flow is too weak or inconsistent.
static bool visual_velocity(double dt, double* vc) {
    if (nflow < 8 || dt <= 0) return false;
    double M[9] = {0};
    for (int k = 0; k < nflow; k++) {
        const double f[3] = {fl_u[k], fl_v[k], 1}, g[3] = {fl_u[k] + fl_du[k], fl_v[k] + fl_dv[k], 1};
        const double n[3] = {f[1] * g[2] - f[2] * g[1], f[2] * g[0] - f[0] * g[2], f[0] * g[1] - f[1] * g[0]};
        for (int a = 0; a < 3; a++) for (int c = 0; c < 3; c++) M[3 * a + c] += n[a] * n[c];
    }
    // smallest eigenvector by inverse iteration on M + eps I
    double t[3] = {0, 0, 1};
    for (int a = 0; a < 3; a++) M[4 * a] += 1e-12;
    float Mi[6]; if (!inv3s(M[0], M[1], M[2], M[4], M[5], M[8], Mi)) return false;
    for (int it = 0; it < 30; it++) {
        double y[3] = {Mi[0] * t[0] + Mi[1] * t[1] + Mi[2] * t[2], Mi[1] * t[0] + Mi[3] * t[1] + Mi[4] * t[2], Mi[2] * t[0] + Mi[4] * t[1] + Mi[5] * t[2]};
        double ny = std::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]); if (!(ny > 0)) return false;
        for (int a = 0; a < 3; a++) t[a] = y[a] / ny;
    }
    // scale: flow_i ~ s * ((u tz - tx), (v tz - ty)), s = |t| / depth
    double num = 0, den = 0;
    for (int k = 0; k < nflow; k++) {
        const double gu = fl_u[k] * t[2] - t[0], gv = fl_v[k] * t[2] - t[1];
        num += fl_du[k] * gu + fl_dv[k] * gv; den += gu * gu + gv * gv;
    }
    if (!(den > 0)) return false;
    double sc = num / den;
    if (sc < 0) { sc = -sc; for (int a = 0; a < 3; a++) t[a] = -t[a]; }
    const double speed = sc * depth_avg / dt;
    if (!(speed < VMAX)) return false;
    for (int a = 0; a < 3; a++) vc[a] = t[a] * speed;
    return true;
}

// median disparity (px) between each feature's first observation in the window (rotated into the current camera with
// the filter's rotations) and its current position, over features seen for at least ZW_MIN_SPAN frames; -1 if too few
static float window_disparity() {
    static float d[MAX_FEATS]; int nd = 0;
    for (int i = 0; i < MAX_FEATS; i++) {
        const Feat& f = feats[i]; if (!f.active || f.nobs < 1) continue;
        if (frame_no - f.obs[0].serial < ZW_MIN_SPAN) continue;
        int k = -1; for (int c = 0; c < n_clones; c++) if (clones[c].serial == f.obs[0].serial) { k = c; break; }
        if (k < 0) continue;
        M3 Rcc = mmul(mmul(Rci, mmul(mtr(Rwi), clones[k].R)), mtr(Rci));     // current cam <- first cam
        double b[3] = {f.obs[0].u, f.obs[0].v, 1}, q[3]; mv(Rcc, b, q);
        if (q[2] < 0.1) continue;
        const double du = fx * (q[0] / q[2]) + cx - f.x, dv = fy * (q[1] / q[2]) + cy - f.y;
        d[nd++] = (float)std::sqrt(du * du + dv * dv);
    }
    if (nd < 6) return -1;
    for (int i = 1; i < nd; i++) { float v = d[i]; int j = i - 1; while (j >= 0 && d[j] > v) { d[j + 1] = d[j]; j--; } d[j + 1] = v; }
    return d[(nd * ZW_PCT) / 100];
}

// median disparity (px) of active tracks since their observation in the newest clone (rotation-compensated)
static float disparity_since_last_clone() {
    if (n_clones == 0) return 1e9f;
    static float d[MAX_FEATS]; int nd = 0;
    const Clone& C = clones[n_clones - 1];
    M3 Rcc = mmul(mmul(Rci, mmul(mtr(Rwi), C.R)), mtr(Rci));
    for (int i = 0; i < MAX_FEATS; i++) {
        const Feat& f = feats[i]; if (!f.active || f.nobs < 1 || f.obs[f.nobs - 1].serial != C.serial) continue;
        double b[3] = {f.obs[f.nobs - 1].u, f.obs[f.nobs - 1].v, 1}, q[3]; mv(Rcc, b, q);
        if (q[2] < 0.1) continue;
        const double du = fx * (q[0] / q[2]) + cx - f.x, dv = fy * (q[1] / q[2]) + cy - f.y;
        d[nd++] = (float)std::sqrt(du * du + dv * dv);
    }
    if (nd < 5) return 1e9f;
    for (int i = 1; i < nd; i++) { float v = d[i]; int j = i - 1; while (j >= 0 && d[j] > v) { d[j + 1] = d[j]; j--; } d[j + 1] = v; }
    return d[nd / 2];
}

// ---------------------------------------------------------------- covariance helpers
#define PP(i, j) P[(i) * NMAX + (j)]
static void symmetrize() {
    for (int i = 0; i < n_state; i++) for (int j = i + 1; j < n_state; j++) { double a = 0.5 * (PP(i, j) + PP(j, i)); PP(i, j) = a; PP(j, i) = a; }
}

// ---------------------------------------------------------------- propagation over one frame of IMU samples
static double v_fej[3], p_fej[3]; static bool fej_ok = false;
static double imu_prev[6]; static bool imu_have = false;
static void propagate(const double* imu, int n) {
    double v_start[3] = {vw[0], vw[1], vw[2]}, p_start[3] = {pw[0], pw[1], pw[2]};
    double Phi[NI * NI]; std::memset(Phi, 0, sizeof Phi);
    for (int i = 0; i < NI; i++) Phi[i * NI + i] = 1;
    M3 A = {{0}}, B = {{0}}, C = {{0}}, D = {{0}}, E = {{0}}, F = {{0}}, G = {{0}};
    double T = 0, Tpg = 0;
    for (int k = 0; k < n; k++) {
        const double* r = imu + k * 7;
        double dt = r[0] - t_last;
        if (dt <= 0 || dt > 0.1) { t_last = r[0] > t_last ? r[0] : t_last; continue; }
        t_last = r[0];
        double x[6] = {r[1], r[2], r[3], r[4], r[5], r[6]};   // input conditioning of the raw sample (IMU_FILT)
        if (IMU_FILT == 1) { for (int i = 0; i < 6; i++) { const double c = x[i]; x[i] = imu_have ? 0.5 * (c + imu_prev[i]) : c; imu_prev[i] = c; } }
        else if (IMU_FILT == 2) { const double al = dt / (IMU_TAU + dt); for (int i = 0; i < 6; i++) { imu_prev[i] = imu_have ? imu_prev[i] + al * (x[i] - imu_prev[i]) : x[i]; x[i] = imu_prev[i]; } }
        imu_have = true;
        double am[3] = {x[0] - ba[0], x[1] - ba[1], x[2] - ba[2]};
        double wm[3] = {(x[3] - bg[0]) * dt, (x[4] - bg[1]) * dt, (x[5] - bg[2]) * dt};
        w_last[0] = x[3] - bg[0]; w_last[1] = x[4] - bg[1]; w_last[2] = x[5] - bg[2];
        double aw[3]; mv(Rwi, am, aw);              // R (a - ba), without gravity
        // Phi blocks: p first (uses old v blocks), then v (uses old theta blocks), then theta
        for (int i = 0; i < 9; i++) { E.m[i] += dt * B.m[i]; F.m[i] += dt * C.m[i]; G.m[i] += dt * D.m[i]; }
        Tpg += dt * T;
        M3 Sa = skew(aw), SaA = mmul(Sa, A);
        for (int i = 0; i < 9; i++) { B.m[i] -= dt * Sa.m[i]; C.m[i] -= dt * SaA.m[i]; D.m[i] -= dt * Rwi.m[i]; A.m[i] -= dt * Rwi.m[i]; }
        T += dt;
        // nominal state
        for (int i = 0; i < 3; i++) { double a = aw[i] + gw[i]; pw[i] += vw[i] * dt + 0.5 * a * dt * dt; vw[i] += a * dt; }
        Rwi = mmul(Rwi, expso3(wm));
    }
    orthonormalize(Rwi);
    if (T <= 0) return;
    if (FEJ & 2) {   // Phi_v,theta and Phi_p,theta at the first (pre-update) estimates of v and p at the frame start
        double dv[3], dp[3];
        for (int i = 0; i < 3; i++) { dv[i] = fej_ok ? v_start[i] - v_fej[i] : 0; dp[i] = fej_ok ? p_start[i] - p_fej[i] + dv[i] * T : 0; }
        M3 Sv = skew(dv), Sp = skew(dp);
        for (int i = 0; i < 9; i++) { B.m[i] -= Sv.m[i]; E.m[i] -= Sp.m[i]; }
        for (int i = 0; i < 3; i++) { v_fej[i] = vw[i]; p_fej[i] = pw[i]; }
        fej_ok = true;
    }
    auto put = [&](int r0, int c0, const M3& m) { for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Phi[(r0 + i) * NI + c0 + j] = m.m[3 * i + j]; };
    auto diag = [&](int r0, int c0, double v) { for (int i = 0; i < 3; i++) Phi[(r0 + i) * NI + c0 + i] = v; };
    put(TH, BG, A);
    put(VE, TH, B); put(VE, BG, C); put(VE, BA, D); diag(VE, GR, T);
    put(PS, TH, E); put(PS, BG, F); put(PS, BA, G); diag(PS, GR, Tpg); diag(PS, VE, T);
    // P <- Phi P Phi^T using the block structure of Phi (identity except the blocks below); rows first, then columns.
    // p rows: += E th + T v + F bg + G ba + Tpg g ; v rows: += B th + C bg + D ba + T g ; th rows: += A bg
    (void)Phi;
    const int ns = n_state;
    auto xf = [&](auto* q, const int sd) {
            double th[3], vv[3], bgv[3], bav[3], gv[3];
            for (int i = 0; i < 3; i++) { th[i] = q[(TH + i) * sd]; vv[i] = q[(VE + i) * sd]; bgv[i] = q[(BG + i) * sd]; bav[i] = q[(BA + i) * sd]; gv[i] = q[(GR + i) * sd]; }
            for (int i = 0; i < 3; i++) {
                const double* e = E.m + 3 * i; const double* f = F.m + 3 * i; const double* g = G.m + 3 * i;
                q[(PS + i) * sd] += e[0] * th[0] + e[1] * th[1] + e[2] * th[2] + T * vv[i] + f[0] * bgv[0] + f[1] * bgv[1] + f[2] * bgv[2]
                                  + g[0] * bav[0] + g[1] * bav[1] + g[2] * bav[2] + Tpg * gv[i];
                const double* bb = B.m + 3 * i; const double* c = C.m + 3 * i; const double* d = D.m + 3 * i;
                q[(VE + i) * sd] += bb[0] * th[0] + bb[1] * th[1] + bb[2] * th[2] + c[0] * bgv[0] + c[1] * bgv[1] + c[2] * bgv[2]
                                  + d[0] * bav[0] + d[1] * bav[1] + d[2] * bav[2] + T * gv[i];
                const double* aa = A.m + 3 * i;
                q[(TH + i) * sd] += aa[0] * bgv[0] + aa[1] * bgv[1] + aa[2] * bgv[2];
            }
    };
    for (int pass = 0; pass < 2; pass++)   // pass 0: column j of rows I (stride NMAX) ; pass 1: row j of columns I (stride 1)
        for (int j = 0; j < ns; j++) xf(pass == 0 ? P + j : P + j * NMAX, pass == 0 ? NMAX : 1);
    if (SCHMIDT && !(SCH_PROF & 2)) for (int c = 0; c < 3 * nsch; c++) xf(Xc + c, SCH_MC);   // cross-covariance rows of the IMU states
    const double dts = 0.005;
    const double qa = SIG_ACC * SIG_ACC * dts, qg = SIG_GYR * SIG_GYR * dts;   // continuous densities
    for (int i = 0; i < 3; i++) {
        PP(TH + i, TH + i) += qg * T;
        PP(VE + i, VE + i) += qa * T;
        PP(PS + i, PS + i) += qa * T * T * T / 3;
        PP(PS + i, VE + i) += qa * T * T / 2; PP(VE + i, PS + i) += qa * T * T / 2;
        PP(BG + i, BG + i) += SIG_BGW * SIG_BGW * T;
        PP(BA + i, BA + i) += SIG_BAW * SIG_BAW * T;
    }
}

// ---------------------------------------------------------------- clones
static void add_clone() {
    if (n_clones == N_CLONES) return;
    const int c = NI + 6 * n_clones;
    clones[n_clones].R = Rwi; for (int i = 0; i < 3; i++) clones[n_clones].p[i] = pw[i];
    clones[n_clones].Rf = Rwi; for (int i = 0; i < 3; i++) clones[n_clones].pf[i] = pw[i];
    clones[n_clones].serial = frame_no;
    // P rows/cols: clone theta = IMU theta, clone p = IMU p
    const int src[6] = {TH, TH + 1, TH + 2, PS, PS + 1, PS + 2};
    for (int j = 0; j < n_state; j++) for (int i = 0; i < 6; i++) { double v = PP(src[i], j); PP(c + i, j) = v; PP(j, c + i) = v; }
    for (int a = 0; a < 6; a++) for (int b = 0; b < 6; b++) PP(c + a, c + b) = PP(src[a], src[b]);
    if (SCHMIDT) for (int i = 0; i < 6; i++) for (int m = 0; m < 3 * nsch; m++) XC(c + i, m) = XC(src[i], m);
    n_state += 6; n_clones++;
}
static void drop_oldest_clone() {
    // remove rows/cols NI..NI+5 (memmove)
    const int c = NI, n = n_state, tail = n - c - 6;
    for (int i = 0; i < n; i++) {
        if (i >= c && i < c + 6) continue;
        double* src = P + i * NMAX; double* dst = P + (i < c ? i : i - 6) * NMAX;
        if (dst != src) std::memmove(dst, src, c * sizeof(double));
        std::memmove(dst + c, src + c + 6, tail * sizeof(double));
    }
    if (SCHMIDT) std::memmove(&XC(c, 0), &XC(c + 6, 0), (size_t)tail * SCH_MC * sizeof(sch_t));
    for (int k = 0; k + 1 < n_clones; k++) clones[k] = clones[k + 1];
    n_clones--; n_state -= 6;
}
static void drop_clone(int k) {   // remove clone k (rows/cols NI+6k..NI+6k+5)
    const int c = NI + 6 * k, n = n_state, tail = n - c - 6;
    for (int i = 0; i < n; i++) {
        if (i >= c && i < c + 6) continue;
        double* src = P + i * NMAX; double* dst = P + (i < c ? i : i - 6) * NMAX;
        if (dst != src) std::memmove(dst, src, c * sizeof(double));
        std::memmove(dst + c, src + c + 6, tail * sizeof(double));
    }
    if (SCHMIDT) std::memmove(&XC(c, 0), &XC(c + 6, 0), (size_t)tail * SCH_MC * sizeof(sch_t));
    for (int j = k; j + 1 < n_clones; j++) clones[j] = clones[j + 1];
    n_clones--; n_state -= 6;
}
static int clone_index(int serial) { for (int k = 0; k < n_clones; k++) if (clones[k].serial == serial) return k; return -1; }

// ---------------------------------------------------------------- EKF update with stacked sparse rows
#ifndef ROW_NNZ
#define ROW_NNZ (6 * OBS_MAX)
#endif
static int rnnz[MAX_ROWS], ridx[MAX_ROWS][ROW_NNZ];
static double rval[MAX_ROWS][ROW_NNZ], rs[MAX_ROWS], Rd[MAX_ROWS];
static int nrows = 0;
static double S[MAX_ROWS * MAX_ROWS];
static void add_row1(int col, double res, double var) {
    if (nrows >= MAX_ROWS) return;
    rnnz[nrows] = 1; ridx[nrows][0] = col; rval[nrows][0] = 1; rs[nrows] = res; Rd[nrows] = var; nrows++;
}

static void inject(const double* dx) {
    M3 dR = expso3(dx + TH); Rwi = mmul(dR, Rwi); orthonormalize(Rwi);
    for (int i = 0; i < 3; i++) { pw[i] += dx[PS + i]; vw[i] += dx[VE + i]; bg[i] += dx[BG + i]; ba[i] += dx[BA + i]; gw[i] += dx[GR + i]; }
    { double ng = std::sqrt(gw[0] * gw[0] + gw[1] * gw[1] + gw[2] * gw[2]); for (int i = 0; i < 3; i++) gw[i] *= 9.81 / ng; }
    for (int k = 0; k < n_clones; k++) {
        const double* d = dx + NI + 6 * k;
        M3 dRc = expso3(d); clones[k].R = mmul(dRc, clones[k].R); orthonormalize(clones[k].R);
        for (int i = 0; i < 3; i++) clones[k].p[i] += d[3 + i];
    }
}

static bool ekf_update() {
    const int n = n_state, r = nrows;
    if (r > 3) dbg_rows = r;
    if (r > 3) nrows_last = r;
    if (r == 0) return false;
    // row form (contiguous axpy loops, vectorized by the compiler): HP = H P (r x n), using P's symmetry
    static upd_t HP[MAX_ROWS * NMAX];
    for (int k = 0; k < r; k++) {
        upd_t* hp = HP + k * NMAX; for (int j = 0; j < n; j++) hp[j] = 0;
        const int nz = rnnz[k]; const int* id = ridx[k]; const double* hv = rval[k];
        int t = 0;
        for (; t + 4 <= nz; t += 4) {   // 4 rows of P per pass: hp loaded/stored once
            const double h0 = hv[t], h1 = hv[t + 1], h2 = hv[t + 2], h3 = hv[t + 3];
            const double* p0 = P + id[t] * NMAX, * p1 = P + id[t + 1] * NMAX, * p2 = P + id[t + 2] * NMAX, * p3 = P + id[t + 3] * NMAX;
            for (int j = 0; j < n; j++) hp[j] += (h0 * p0[j] + h1 * p1[j]) + (h2 * p2[j] + h3 * p3[j]);
        }
        for (; t < nz; t++) { const double h = hv[t]; const double* p = P + id[t] * NMAX; for (int j = 0; j < n; j++) hp[j] += h * p[j]; }
    }
    // S = H P H^T + R (lower triangle)
    for (int a = 0; a < r; a++) {
        const int nz = rnnz[a]; const int* id = ridx[a]; const double* hv = rval[a];
        for (int b = 0; b <= a; b++) {
            const upd_t* hp = HP + b * NMAX; double s = 0;
            for (int t = 0; t < nz; t++) s += hv[t] * hp[id[t]];
            if (a == b) s += Rd[a];
            S[a * MAX_ROWS + b] = s;
        }
    }
    // Cholesky S = L L^T (lower, in place)
    for (int j = 0; j < r; j++) {
        double* Sj = S + j * MAX_ROWS;
        double d = Sj[j];
        for (int k = 0; k < j; k++) d -= Sj[k] * Sj[k];
        if (!(d > 1e-15)) return false;
        d = std::sqrt(d); Sj[j] = d; const double id = 1.0 / d;
        for (int i = j + 1; i < r; i++) {
            double* Si = S + i * MAX_ROWS; double s = Si[j];
            for (int k = 0; k < j; k++) s -= Si[k] * Sj[k];
            Si[j] = s * id;
        }
    }
    // G = L^-1 HP (rows), y = L^-1 res, dx = G^T y, P -= G^T G
    double y[MAX_ROWS];
    for (int a = 0; a < r; a++) {
        const double* Sa = S + a * MAX_ROWS; upd_t* ga = HP + a * NMAX; double s = rs[a];
        int k = 0;
        for (; k + 4 <= a; k += 4) {
            const upd_t l0 = (upd_t)Sa[k], l1 = (upd_t)Sa[k + 1], l2 = (upd_t)Sa[k + 2], l3 = (upd_t)Sa[k + 3];
            const upd_t* q0 = HP + k * NMAX, * q1 = q0 + NMAX, * q2 = q1 + NMAX, * q3 = q2 + NMAX;
            for (int j = 0; j < n; j++) ga[j] -= (l0 * q0[j] + l1 * q1[j]) + (l2 * q2[j] + l3 * q3[j]);
            s -= (Sa[k] * y[k] + Sa[k + 1] * y[k + 1]) + (Sa[k + 2] * y[k + 2] + Sa[k + 3] * y[k + 3]);
        }
        for (; k < a; k++) {
            const double l = Sa[k]; const upd_t lf = (upd_t)l; const upd_t* gk = HP + k * NMAX;
            for (int j = 0; j < n; j++) ga[j] -= lf * gk[j];
            s -= l * y[k];
        }
        const double il = 1.0 / Sa[a]; const upd_t ilf = (upd_t)il;
        for (int j = 0; j < n; j++) ga[j] *= ilf;
        y[a] = s * il;
    }
    double dx[NMAX];
    for (int j = 0; j < n; j++) dx[j] = 0;
    for (int a = 0; a < r; a++) { const double ya = y[a]; const upd_t* ga = HP + a * NMAX; for (int j = 0; j < n; j++) dx[j] += ya * ga[j]; }
    {   // P -= G^T G, 4 rows of G per pass (each P element loaded/stored once per pass)
        int a = 0;
        for (; a + 4 <= r; a += 4) {
            const upd_t* g0 = HP + a * NMAX, * g1 = g0 + NMAX, * g2 = g1 + NMAX, * g3 = g2 + NMAX;
            for (int i = 0; i < n; i++) {
                const upd_t a0 = g0[i], a1 = g1[i], a2 = g2[i], a3 = g3[i]; double* pi = P + i * NMAX;
                for (int j = i; j < n; j++) pi[j] -= (a0 * g0[j] + a1 * g1[j]) + (a2 * g2[j] + a3 * g3[j]);
            }
        }
        for (; a < r; a++) {
            const upd_t* ga = HP + a * NMAX;
            for (int i = 0; i < n; i++) { const upd_t gi = ga[i]; double* pi = P + i * NMAX; for (int j = i; j < n; j++) pi[j] -= gi * ga[j]; }
        }
    }
    if (SCHMIDT && nsch > 0 && !(SCH_PROF & 1)) {   // Schmidt: the cross-covariance with the map follows the state update, the map does not move
        const int mc = 3 * nsch;
        static sch_t HX[MAX_ROWS * 3 * (SCHMIDT ? SCH_N : 1)];
        for (int k = 0; k < r; k++) {   // HX = H Xc, 4 nonzeros at a time
            sch_t* hx = HX + k * SCH_MC; for (int c = 0; c < mc; c++) hx[c] = 0;
            const int nz = rnnz[k]; int t = 0;
            for (; t + 4 <= nz; t += 4) {
                const sch_t h0 = rval[k][t], h1 = rval[k][t + 1], h2 = rval[k][t + 2], h3 = rval[k][t + 3];
                const sch_t *x0 = &XC(ridx[k][t], 0), *x1 = &XC(ridx[k][t + 1], 0), *x2 = &XC(ridx[k][t + 2], 0), *x3 = &XC(ridx[k][t + 3], 0);
                for (int c = 0; c < mc; c++) hx[c] += (h0 * x0[c] + h1 * x1[c]) + (h2 * x2[c] + h3 * x3[c]);
            }
            for (; t < nz; t++) { const sch_t h = rval[k][t]; const sch_t* xr = &XC(ridx[k][t], 0); for (int c = 0; c < mc; c++) hx[c] += h * xr[c]; }
        }
        for (int a = 0; a < r; a++) {   // GX = L^-1 HX
            sch_t* ga = HX + a * SCH_MC; const double* Sa = S + a * MAX_ROWS; int k = 0;
            for (; k + 4 <= a; k += 4) {
                const sch_t l0 = Sa[k], l1 = Sa[k + 1], l2 = Sa[k + 2], l3 = Sa[k + 3];
                const sch_t *g0 = HX + k * SCH_MC, *g1 = g0 + SCH_MC, *g2 = g1 + SCH_MC, *g3 = g2 + SCH_MC;
                for (int c = 0; c < mc; c++) ga[c] -= (l0 * g0[c] + l1 * g1[c]) + (l2 * g2[c] + l3 * g3[c]);
            }
            for (; k < a; k++) { const sch_t l = Sa[k]; const sch_t* gk = HX + k * SCH_MC; for (int c = 0; c < mc; c++) ga[c] -= l * gk[c]; }
            const sch_t il = 1.0 / Sa[a]; for (int c = 0; c < mc; c++) ga[c] *= il;
        }
        int a = 0;
        for (; a + 4 <= r; a += 4) {   // Xc -= G^T GX, 4 rows at a time
            const upd_t *g0 = HP + a * NMAX, *g1 = g0 + NMAX, *g2 = g1 + NMAX, *g3 = g2 + NMAX;
            const sch_t *x0 = HX + a * SCH_MC, *x1 = x0 + SCH_MC, *x2 = x1 + SCH_MC, *x3 = x2 + SCH_MC;
            for (int i = 0; i < n; i++) {
                const sch_t a0 = g0[i], a1 = g1[i], a2 = g2[i], a3 = g3[i]; sch_t* xr = &XC(i, 0);
                for (int c = 0; c < mc; c++) xr[c] -= (a0 * x0[c] + a1 * x1[c]) + (a2 * x2[c] + a3 * x3[c]);
            }
        }
        for (; a < r; a++) {
            const upd_t* g = HP + a * NMAX; const sch_t* gx = HX + a * SCH_MC;
            for (int i = 0; i < n; i++) { const sch_t gi = g[i]; sch_t* xr = &XC(i, 0); for (int c = 0; c < mc; c++) xr[c] -= gi * gx[c]; }
        }
    }
    for (int i = 0; i < n; i++) for (int j = 0; j < i; j++) PP(i, j) = PP(j, i);
    for (int i = 0; i < n; i++) if (PP(i, i) < 1e-12) PP(i, i) = 1e-12;
    inject(dx);
    return true;
}

// ---------------------------------------------------------------- feature -> MSCKF rows
static double chi2_95(int dof) {   // Wilson-Hilferty approximation of the 95% quantile
    double k = dof, z = 1.6449;
    double a = 1 - 2 / (9 * k) + z * std::sqrt(2 / (9 * k));
    return k * a * a * a;
}

// pose of camera for clone k: R_wc = R_wi R_ic, p_wc = p_wi + R_wi p_ic   (R_ic = Rci^T, p_ic = -Rci^T tci)
static void cam_pose(int k, M3& Rwc, double* pwc) {
    M3 Ric = mtr(Rci); double pic[3]; mtv(Rci, tci, pic); for (int i = 0; i < 3; i++) pic[i] = -pic[i];
    Rwc = mmul(clones[k].R, Ric); double t[3]; mv(clones[k].R, pic, t);
    for (int i = 0; i < 3; i++) pwc[i] = clones[k].p[i] + t[i];
}

static bool triangulate(const Feat& f, const int* ck, double* pf) {
    // anchored inverse depth in the first observation's camera
    const int m = f.nobs;
    M3 Ra; double pa[3]; cam_pose(ck[0], Ra, pa);
    M3 Rr[N_CLONES + 1]; double tr[N_CLONES + 1][3];   // anchor-cam -> obs-cam: p_j = R_j p_a + t_j
    for (int o = 0; o < m; o++) {
        M3 Rw; double pwc[3]; cam_pose(ck[o], Rw, pwc);
        Rr[o] = mmul(mtr(Rw), Ra);
        double d[3] = {pa[0] - pwc[0], pa[1] - pwc[1], pa[2] - pwc[2]}; mtv(Rw, d, tr[o]);
    }
    // parallax check + linear init from anchor vs last
    double b0[3] = {f.obs[0].u, f.obs[0].v, 1}, bl[3] = {f.obs[m - 1].u, f.obs[m - 1].v, 1}, rb[3];
    mv(Rr[m - 1], b0, rb);
    double n0 = std::sqrt(rb[0] * rb[0] + rb[1] * rb[1] + rb[2] * rb[2]), nl = std::sqrt(bl[0] * bl[0] + bl[1] * bl[1] + bl[2] * bl[2]);
    double cosang = (rb[0] * bl[0] + rb[1] * bl[1] + rb[2] * bl[2]) / (n0 * nl);
    if (cosang > std::cos(MIN_PARALLAX)) { st_par++; tri_par = true; return false; }
    // depth along b0: lambda * rb + t = mu * bl  -> least squares in (lambda, mu)
    const double* t = tr[m - 1];
    double a11 = rb[0] * rb[0] + rb[1] * rb[1] + rb[2] * rb[2], a12 = -(rb[0] * bl[0] + rb[1] * bl[1] + rb[2] * bl[2]), a22 = nl * nl;
    double c1 = -(rb[0] * t[0] + rb[1] * t[1] + rb[2] * t[2]), c2 = bl[0] * t[0] + bl[1] * t[1] + bl[2] * t[2];
    double det = a11 * a22 - a12 * a12; if (std::fabs(det) < 1e-12) return false;
    double lam = (c1 * a22 - a12 * c2) / det;
#ifndef __wasm__
    if (std::getenv("MSCKF_TRI") && frame_no > 120 && frame_no < 160)
        std::fprintf(stderr, "tri f %d m %d par %.3f lam %.2f |t| %.3f t %.3f %.3f %.3f\n", frame_no, m, std::acos(cosang), lam, std::sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2]), t[0], t[1], t[2]);
#endif
    if (!(lam > MIN_DEPTH && lam < MAX_DEPTH)) return false;
    double al = b0[0], be = b0[1], rho = 1.0 / lam;
    // Gauss-Newton on (alpha, beta, rho)
    float hi_last[6] = {0, 0, 0, 0, 0, 0};
    for (int it = 0; it < 6; it++) {
        double Hm[9] = {0}, g[3] = {0};
        for (int o = 0; o < m; o++) {
            const double* R = Rr[o].m;
            double h[3] = {R[0] * al + R[1] * be + R[2] + rho * tr[o][0], R[3] * al + R[4] * be + R[5] + rho * tr[o][1], R[6] * al + R[7] * be + R[8] + rho * tr[o][2]};
            if (h[2] <= 1e-6) return false;
            double iz = 1.0 / h[2];
            double e[2] = {h[0] * iz - f.obs[o].u, h[1] * iz - f.obs[o].v};
            double dh[3][3] = {{R[0], R[1], tr[o][0]}, {R[3], R[4], tr[o][1]}, {R[6], R[7], tr[o][2]}};   // dh/d(al,be,rho)
            double J[2][3];
            for (int c = 0; c < 3; c++) { J[0][c] = iz * (dh[0][c] - h[0] * iz * dh[2][c]); J[1][c] = iz * (dh[1][c] - h[1] * iz * dh[2][c]); }
            for (int a = 0; a < 3; a++) { g[a] += J[0][a] * e[0] + J[1][a] * e[1]; for (int b = 0; b < 3; b++) Hm[a * 3 + b] += J[0][a] * J[0][b] + J[1][a] * J[1][b]; }
        }
        float hi[6];
        if (!inv3s(Hm[0], Hm[1], Hm[2], Hm[4], Hm[5], Hm[8], hi)) return false;
        for (int q = 0; q < 6; q++) hi_last[q] = hi[q];
        double d0 = -(hi[0] * g[0] + hi[1] * g[1] + hi[2] * g[2]), d1 = -(hi[1] * g[0] + hi[3] * g[1] + hi[4] * g[2]), d2 = -(hi[2] * g[0] + hi[4] * g[1] + hi[5] * g[2]);
        al += d0; be += d1; rho += d2;
        if (std::fabs(d2) < 1e-4 * std::fabs(rho) && std::fabs(d0) + std::fabs(d1) < 1e-6) break;
    }
    if (!(rho > 1.0 / MAX_DEPTH && rho < 1.0 / MIN_DEPTH)) return false;
    for (int o = 0; o < m; o++) {
        const double* R = Rr[o].m;
        double h[3] = {R[0] * al + R[1] * be + R[2] + rho * tr[o][0], R[3] * al + R[4] * be + R[5] + rho * tr[o][1], R[6] * al + R[7] * be + R[8] + rho * tr[o][2]};
        if (h[2] <= 1e-6) return false;
        double eu = (h[0] / h[2] - f.obs[o].u) * fx, ev = (h[1] / h[2] - f.obs[o].v) * fy;
        if (eu * eu + ev * ev > TRI_MAX_PX * TRI_MAX_PX) { st_rep++; return false; }
    }
    depth_avg = 0.98 * depth_avg + 0.02 * (1.0 / rho);
    tri_al = al; tri_be = be; tri_rho = rho; tri_Ra = Ra; for (int q = 0; q < 6; q++) tri_hi[q] = hi_last[q];
    double pc[3] = {al / rho, be / rho, 1 / rho}, w[3]; mv(Ra, pc, w);
    for (int i = 0; i < 3; i++) pf[i] = pa[i] + w[i];
    return true;
}

#ifndef SPLAT
#define SPLAT 0                  // 1: Gaussian-splat map relocalization (splat_map.inc)
#endif
#include "splat_map.inc"
// appends the null-space projected rows of feature f to the stacked system; returns number of rows added
static int feature_rows(const Feat& f0);
// queue a finished track (its observations) for an update on this or a later frame
static Pending cand[MAX_FEATS + MAX_PENDING]; static int ncand = 0;
static Pending dlist[MAX_FEATS]; static int ndef = 0;   // LOST_DEFER: lost tracks wait one frame
static void defer_obs(const Feat& f) {
    if (f.nobs < 3 || ndef >= MAX_FEATS) return;
    dlist[ndef].nobs = f.nobs; dlist[ndef].fidx = -1; for (int o = 0; o < f.nobs; o++) dlist[ndef].obs[o] = f.obs[o]; ndef++;
}
static void queue_obs(const Feat& f) {
    if (f.nobs < 3 || ncand >= MAX_FEATS + MAX_PENDING) return;
    cand[ncand].nobs = f.nobs; cand[ncand].fidx = f.active ? (int)(&f - feats) : -1;
    for (int o = 0; o < f.nobs; o++) cand[ncand].obs[o] = f.obs[o]; ncand++;
}
// process queued tracks (pending from earlier frames first) until the row budget is used; the rest waits
static void process_candidates(int drop_serial) {
    static Pending all[MAX_FEATS + 2 * MAX_PENDING]; int na = 0;
    for (int i = 0; i < npend; i++) { all[na] = pend[i]; all[na].fidx = -1; na++; }
    for (int i = 0; i < ncand; i++) all[na++] = cand[i];
    npend = 0; ncand = 0;
    for (int i = 0; i < na; i++) {
        Pending& c = all[i];
        if (nrows + 2 * (c.nobs < OBS_MAX ? c.nobs : OBS_MAX) - 3 > row_cap) {
            // no room this frame: keep it for the next one, without the observation of the clone that leaves
            int k = 0; for (int o = 0; o < c.nobs; o++) if (c.obs[o].serial != drop_serial) c.obs[k++] = c.obs[o];
            c.nobs = k;
            if (k >= 3 && npend < MAX_PENDING) pend[npend++] = c;
            continue;
        }
        Feat& f = sub_feat2; f.nobs = c.nobs; for (int o = 0; o < c.nobs; o++) f.obs[o] = c.obs[o];
        cur_fidx = c.fidx;
        feature_rows(f);
        cur_fidx = -1;
    }
}
static int feature_rows(const Feat& f0) {
    if (f0.nobs < 3) return 0;
    Feat& f = sub_feat;          // up to OBS_MAX observations, evenly spaced, first and last kept
    if (f0.nobs <= OBS_MAX) { f.nobs = f0.nobs; for (int o = 0; o < f0.nobs; o++) f.obs[o] = f0.obs[o]; }
    else { f.nobs = OBS_MAX; for (int o = 0; o < OBS_MAX; o++) f.obs[o] = f0.obs[(o * (f0.nobs - 1) + (OBS_MAX - 1) / 2) / (OBS_MAX - 1)]; }
    const int m = f.nobs;
    int ck[N_CLONES + 1];
    for (int o = 0; o < m; o++) { ck[o] = clone_index(f.obs[o].serial); if (ck[o] < 0) return 0; }
    double pf[3];
    st_cand++;
    tri_par = false;
    bool far = false;
    if (!triangulate(f, ck, pf)) {
        st_tri++; if (!tri_par || !RESET_GEOM) frame_tri_fail++;
        if (!(LOWPAR && tri_par)) return 0;
        {   // only when the filter's own baseline over the track is tiny (nearly still): then all parallax is rotation
            const Clone& C0 = clones[ck[0]]; const Clone& C1 = clones[ck[m - 1]];
            const double bx = C1.p[0] - C0.p[0], by = C1.p[1] - C0.p[1], bz = C1.p[2] - C0.p[2];
            if (bx * bx + by * by + bz * bz > LP_BASE * LP_BASE) return 0;
        }
        // too little parallax to triangulate: place the point far away along its first bearing; after the null-space
        // projection the rows mainly constrain the relative rotation of the clones (gyro drift in slow flight)
        M3 Ra; double pa[3]; cam_pose(ck[0], Ra, pa);
        double b0[3] = {f.obs[0].u * LP_DEPTH, f.obs[0].v * LP_DEPTH, LP_DEPTH}, w[3]; mv(Ra, b0, w);
        for (int i = 0; i < 3; i++) pf[i] = pa[i] + w[i];
        far = true;
    }
    const int rr = 2 * m;
    double Hx[2 * (N_CLONES + 1)][6 * (N_CLONES + 1)]; double Hf[2 * (N_CLONES + 1)][3]; double res[2 * (N_CLONES + 1)];
    std::memset(Hx, 0, sizeof Hx);
    for (int o = 0; o < m; o++) {
        const Clone& C = clones[ck[o]];
        double d[3] = {pf[0] - C.p[0], pf[1] - C.p[1], pf[2] - C.p[2]}, pi[3], pc[3];
        mtv(C.R, d, pi); mv(Rci, pi, pc); for (int i = 0; i < 3; i++) pc[i] += tci[i];
        if (pc[2] < MIN_DEPTH) return 0;
        double iz = 1.0 / pc[2], u = pc[0] * iz, v = pc[1] * iz;
        res[2 * o] = f.obs[o].u - u; res[2 * o + 1] = f.obs[o].v - v;
        if (far && (std::fabs(res[2 * o]) * fx > LP_MAX_PX || std::fabs(res[2 * o + 1]) * fy > LP_MAX_PX)) return 0;
        double Jp[2][3] = {{iz, 0, -u * iz}, {0, iz, -v * iz}};
        const M3& RJ = (FEJ & 1) ? C.Rf : C.R;        // Jacobians at the clone's first estimate (FEJ) or the current one
        if (FEJ & 1) {
            double dj[3] = {pf[0] - C.pf[0], pf[1] - C.pf[1], pf[2] - C.pf[2]}, pij[3], pcj[3];
            mtv(C.Rf, dj, pij); mv(Rci, pij, pcj); for (int i = 0; i < 3; i++) pcj[i] += tci[i];
            if (pcj[2] < MIN_DEPTH) return 0;
            const double izj = 1.0 / pcj[2], uj = pcj[0] * izj, vj = pcj[1] * izj;
            Jp[0][0] = izj; Jp[0][2] = -uj * izj; Jp[1][1] = izj; Jp[1][2] = -vj * izj;
            for (int i = 0; i < 3; i++) d[i] = dj[i];
        }
        M3 RcRt = mmul(Rci, mtr(RJ));           // d pc / d pf
        M3 dth = mmul(RcRt, skew(d));           // d pc / d theta = Rci R^T [d]x
        for (int a = 0; a < 2; a++) for (int c = 0; c < 3; c++) {
            double sf = 0, st = 0;
            for (int k = 0; k < 3; k++) { sf += Jp[a][k] * RcRt.m[3 * k + c]; st += Jp[a][k] * dth.m[3 * k + c]; }
            Hf[2 * o + a][c] = sf;
            Hx[2 * o + a][6 * o + c] = st;
            Hx[2 * o + a][6 * o + 3 + c] = -sf;
        }
    }
    // Householder on Hf columns, apply to Hx and res; keep rows 3..rr-1
    const int nc = 6 * m;
    for (int c = 0; c < 3; c++) {
        double nrm = 0; for (int i = c; i < rr; i++) nrm += Hf[i][c] * Hf[i][c];
        nrm = std::sqrt(nrm); if (nrm < 1e-12) return 0;
        double alpha = Hf[c][c] > 0 ? -nrm : nrm;
        double vh[2 * (N_CLONES + 1)]; for (int i = 0; i < rr; i++) vh[i] = 0;
        vh[c] = Hf[c][c] - alpha; for (int i = c + 1; i < rr; i++) vh[i] = Hf[i][c];
        double vv = 0; for (int i = c; i < rr; i++) vv += vh[i] * vh[i];
        if (vv < 1e-20) continue;
        double beta = 2 / vv;
        for (int cc = c; cc < 3; cc++) { double s = 0; for (int i = c; i < rr; i++) s += vh[i] * Hf[i][cc]; s *= beta; for (int i = c; i < rr; i++) Hf[i][cc] -= s * vh[i]; }
        for (int cc = 0; cc < nc; cc++) { double s = 0; for (int i = c; i < rr; i++) s += vh[i] * Hx[i][cc]; s *= beta; if (s != 0) for (int i = c; i < rr; i++) Hx[i][cc] -= s * vh[i]; }
        { double s = 0; for (int i = c; i < rr; i++) s += vh[i] * res[i]; s *= beta; for (int i = c; i < rr; i++) res[i] -= s * vh[i]; }
    }
    const int nr = rr - 3;
    if (nrows + nr > row_cap) return 0;
    const double s2 = far ? (LP_SIG_PX / fx) * (LP_SIG_PX / fx) : (SIG_PX / fx) * (SIG_PX / fx);
#if CHI2_ON
    // chi2 test: r^T (H P H^T + s2 I)^-1 r
    static double HP[2 * (N_CLONES + 1)][NMAX];
    for (int a = 0; a < nr; a++) for (int j = 0; j < n_state; j++) {
        double s = 0;
        for (int o = 0; o < m; o++) { const int base = NI + 6 * ck[o]; for (int c = 0; c < 6; c++) { double h = Hx[3 + a][6 * o + c]; if (h != 0) s += h * PP(base + c, j); } }
        HP[a][j] = s;
    }
    static double Sm[2 * (N_CLONES + 1)][2 * (N_CLONES + 1)];
    for (int a = 0; a < nr; a++) for (int b = 0; b <= a; b++) {
        double s = 0;
        for (int o = 0; o < m; o++) { const int base = NI + 6 * ck[o]; for (int c = 0; c < 6; c++) s += HP[a][base + c] * Hx[3 + b][6 * o + c]; }
        if (a == b) s += s2;
        Sm[a][b] = s; Sm[b][a] = s;
    }
    // cholesky + solve
    for (int j = 0; j < nr; j++) {
        double d = Sm[j][j]; for (int k = 0; k < j; k++) d -= Sm[j][k] * Sm[j][k];
        if (!(d > 0)) return 0;
        d = std::sqrt(d); Sm[j][j] = d;
        for (int i = j + 1; i < nr; i++) { double s = Sm[i][j]; for (int k = 0; k < j; k++) s -= Sm[i][k] * Sm[j][k]; Sm[i][j] = s / d; }
    }
    double y[2 * (N_CLONES + 1)], chi = 0;
    for (int a = 0; a < nr; a++) { double s = res[3 + a]; for (int k = 0; k < a; k++) s -= Sm[a][k] * y[k]; y[a] = s / Sm[a][a]; chi += y[a] * y[a]; }
    n_tried++;
    if (chi > CHI2_MULT * chi2_95(nr)) { st_chi++; return 0; }
    n_ok++; st_ok++;
#else
    n_tried++; n_ok++; st_ok++;
#endif
    for (int a = 0; a < nr; a++) {
        int t = 0;
        for (int o = 0; o < m; o++) { const int base = NI + 6 * ck[o]; for (int c = 0; c < 6; c++) { ridx[nrows][t] = base + c; rval[nrows][t] = Hx[3 + a][6 * o + c]; t++; } }
        rnnz[nrows] = t; rs[nrows] = res[3 + a]; Rd[nrows] = s2; nrows++;
    }
    create_obs = &f;
    if (SPLAT && !far) splat_create(pf, ck[0]);
    if (far) st_far++;
    return nr;
}

// ---------------------------------------------------------------- tracking + detection
static float flow[MAX_FEATS];
// median flow (px) after removing the best-fitting small rotation (gyro-bias error looks like a rotation)
static float derotated_median() {
    double A[9] = {0}, b[3] = {0};
    for (int k = 0; k < nflow; k++) {
        const double u = fl_u[k], v = fl_v[k];
        const double j1[3] = {u * v, -(1 + u * u), v}, j2[3] = {1 + v * v, -u * v, -u};
        for (int a = 0; a < 3; a++) { b[a] += j1[a] * fl_du[k] + j2[a] * fl_dv[k]; for (int c = 0; c < 3; c++) A[3 * a + c] += j1[a] * j1[c] + j2[a] * j2[c]; }
    }
    for (int a = 0; a < 3; a++) A[4 * a] += 1e-6;
    float Ai[6]; double d[3] = {0, 0, 0};
    if (inv3s(A[0], A[1], A[2], A[4], A[5], A[8], Ai)) {
        d[0] = Ai[0] * b[0] + Ai[1] * b[1] + Ai[2] * b[2]; d[1] = Ai[1] * b[0] + Ai[3] * b[1] + Ai[4] * b[2]; d[2] = Ai[2] * b[0] + Ai[4] * b[1] + Ai[5] * b[2];
    }
    static float r[MAX_FEATS];
    for (int k = 0; k < nflow; k++) {
        const double u = fl_u[k], v = fl_v[k];
        const double eu = fl_du[k] - (u * v * d[0] - (1 + u * u) * d[1] + v * d[2]), ev = fl_dv[k] - ((1 + v * v) * d[0] - u * v * d[1] - u * d[2]);
        r[k] = (float)(fx * std::sqrt(eu * eu + ev * ev));
    }
    for (int i = 1; i < nflow; i++) { float v = r[i]; int j = i - 1; while (j >= 0 && r[j] > v) { r[j + 1] = r[j]; j--; } r[j + 1] = v; }
    return r[nflow / 2];
}

static void track(const M3& Rcc) {   // Rcc: new camera <- old camera rotation (gyro)
    nflow = 0; frame_tracks = 0;
    for (int i = 0; i < MAX_FEATS; i++) {
        Feat& f = feats[i]; if (!f.active) continue;
        st_tracks++; frame_tracks++;
        double b[3] = {(f.x - cx) / fx, (f.y - cy) / fy, 1}, q[3]; mv(Rcc, b, q);
        if (q[2] < 0.1) { f.active = false; continue; }
        float px = (float)(fx * q[0] / q[2] + cx), py = (float)(fy * q[1] / q[2] + cy);
        float sx = px, sy = py;   // LK start: rotation-only, or with the filter's predicted translation (LK_TPRED)
        if (LK_TPRED) {
            const double Z = depth_avg, qt[3] = {q[0] * Z - tpred_c[0], q[1] * Z - tpred_c[1], q[2] * Z - tpred_c[2]};
            if (qt[2] > 0.1) { sx = (float)(fx * qt[0] / qt[2] + cx); sy = (float)(fy * qt[1] / qt[2] + cy); }
        }
        float x = sx + f.fdx - f.ox, y = sy + f.fdy - f.oy, off = 0;
        if (!inside(x, y, 4 * PR + 4)) { x = sx - f.ox; y = sy - f.oy; }
        float r1 = LK_COARSE > 0 ? lk_level(f, 1, x, y, off, LK_COARSE) : 0;
        if (r1 < 0) { f.active = false; continue; }
        float r2 = lk_level(f, 0, x, y, off, LK_FINE);
        if (r2 < 0 || r2 > MAX_RES) { f.active = false; continue; }
        if (ANCHOR && f.anchored) {   // pull the frame-to-frame result back onto the track's first appearance (no drift)
            float xa = x, ya = y, oa = off;
            const float ra = lk_level(f, 2, xa, ya, oa, ANCHOR_IT);
            if (ra >= 0 && ra < ANCHOR_RES && std::fabs(xa - x) + std::fabs(ya - y) < 1.5f) { x = xa; y = ya; }
            else f.anchored = false;
        }
        x += f.ox; y += f.oy;
        f.fdx = 0.7f * (x - sx); f.fdy = 0.7f * (y - sy);
        fl_u[nflow] = (float)(q[0] / q[2]); fl_v[nflow] = (float)(q[1] / q[2]);
        fl_du[nflow] = (float)((x - px) / fx); fl_dv[nflow] = (float)((y - py) / fy);
        flow[nflow++] = std::sqrt((x - px) * (x - px) + (y - py) * (y - py));
        f.x = x; f.y = y;
    }
    // remove near-duplicates (features that converged onto each other)
    for (int i = 0; i < MAX_FEATS; i++) if (feats[i].active) for (int j = i + 1; j < MAX_FEATS; j++) if (feats[j].active) {
        float dx = feats[i].x - feats[j].x, dy = feats[i].y - feats[j].y;
        if (dx * dx + dy * dy < 4.0f) feats[j].active = false;
    }
}

static void detect() {
    const int cw = W / GRID_X, chh = H / GRID_Y;
    bool occ[GRID_X * GRID_Y] = {false};
    int nact = 0;
    for (int i = 0; i < MAX_FEATS; i++) if (feats[i].active) {
        nact++;
        int gx = (int)(feats[i].x / cw), gy = (int)(feats[i].y / chh);
        if (gx >= 0 && gx < GRID_X && gy >= 0 && gy < GRID_Y) occ[gy * GRID_X + gx] = true;
    }
    int scored = 0;
    det_refill = DET_REFILL_N > 0 && MAX_FEATS - nact > DET_REFILL_N;
    int det_cap = frame_rows > LB_ROWS ? LB_CELLS : DETECT_CELLS;   // load balancing: heavy update frame -> detect less
    if (SCHED == 1) { const int dc = (int)((SCHED_B - sched_used) / SC_CELL); det_cap = dc < SC_MINCELLS ? SC_MINCELLS : (dc > DETECT_CELLS ? DETECT_CELLS : dc); }
    if (SCHED == 2) { int dc = (int)((SCHED_B - sched_used) / SC_CELL); if (dc < sched_need) dc = sched_need; det_cap = dc > DETECT_CELLS ? DETECT_CELLS : dc; }
    for (int kk = 0; kk < GRID_X * GRID_Y && scored < det_cap && nact < MAX_FEATS; kk++) {
        int cell = (det_rr + kk) % (GRID_X * GRID_Y);
        if (occ[cell]) continue;
        scored++;
        int gx = cell % GRID_X, gy = cell / GRID_X;
        int x0 = gx * cw, y0 = gy * chh;
#if DET_HALF
        if (cool[cell] > frame_no) continue;
        // half-resolution (2x2 sums) patch of the cell with a 1-sample border, gradients, 3x3 structure tensor
        const int hw = cw / 2, hh = chh / 2, ew = hw + 4, eh = hh + 4;
        static float hv[64 * 64], ga[64 * 64], gb[64 * 64], gc[64 * 64];
        for (int j = 0; j < eh; j++) {
            int yy = y0 + 2 * (j - 2); if (yy < 0) yy = 0; if (yy > H - 2) yy = H - 2;
            const uint8_t* r0 = img + (yy * W) * 3 + 1; const uint8_t* r1 = r0 + 3 * W;
            for (int i = 0; i < ew; i++) {
                int xx = x0 + 2 * (i - 2); if (xx < 0) xx = 0; if (xx > W - 2) xx = W - 2;
                hv[j * ew + i] = (float)r0[3 * xx] + r0[3 * xx + 3] + r1[3 * xx] + r1[3 * xx + 3];
            }
        }
        for (int j = 1; j < eh - 1; j++) for (int i = 1; i < ew - 1; i++) {
            const float* q = hv + j * ew + i; float dx = q[1] - q[-1], dy = q[ew] - q[-ew];
            ga[j * ew + i] = dx * dx; gb[j * ew + i] = dx * dy; gc[j * ew + i] = dy * dy;
        }
        float best = nact < MS_LOW_N ? MIN_SCORE_LOW : MIN_SCORE; int bx = -1, by = -1;
        for (int j = 2; j < hh + 2; j++) for (int i = 2; i < hw + 2; i++) {
            const int x = x0 + 2 * (i - 2) + 1, y = y0 + 2 * (j - 2) + 1;   // full-res centre of the 2x2 block
            if (x < 4 * PR + 8 || y < 4 * PR + 8 || x >= W - 4 * PR - 9 || y >= H - 4 * PR - 9) continue;
            float a = 0, b = 0, c = 0;
            for (int v = -1; v <= 1; v++) { const int o = (j + v) * ew + i; a += ga[o - 1] + ga[o] + ga[o + 1]; b += gb[o - 1] + gb[o] + gb[o + 1]; c += gc[o - 1] + gc[o] + gc[o + 1]; }
            float tr = 0.5f * (a + c), dd = std::sqrt(0.25f * (a - c) * (a - c) + b * b);
            float lmin = (tr - dd) * (1.0f / 2304.0f);
            if (lmin > best) { best = lmin; bx = x; by = y; }
        }
#else
        if (cool[cell] > frame_no) continue;
        {   // cheap pre-test: any gradient on a sparse 4-px lattice of the cell?
            int mx = 0;
            for (int y = y0 + 2; y < y0 + chh - 1; y += 4) {
                if (y < 2 || y >= H - 2) continue;
                for (int x = x0 + 2; x < x0 + cw - 1; x += 4) {
                    if (x < 2 || x >= W - 2) continue;
                    const uint8_t* p = img + (y * W + x) * 3 + 1;
                    int g = std::abs((int)p[6] - (int)p[-6]) + std::abs((int)p[6 * W] - (int)p[-6 * W]);
                    if (g > mx) mx = g;
                }
            }
            if (mx < PRETEST) { cool[cell] = frame_no + COOL; continue; }
        }
        st_cells++;
#if DET_SPARSE == 2
        // sparse score as below, but each 2-px grid sample's gradient products are computed once per cell and shared by
        // the (up to 4) candidates that use it; the sums keep the same order, so the scores are identical
        static float sa[32 * 32], sb[32 * 32], sc2[32 * 32];
        const int gx0 = x0 + DET_STEP / 2 - 2, gy0 = y0 + DET_STEP / 2 - 2;   // first sample of the first candidate
        const int ncx = (cw - DET_STEP / 2 + DET_STEP - 1) / DET_STEP, ncy = (chh - DET_STEP / 2 + DET_STEP - 1) / DET_STEP;
        const int gnx = 2 * ncx + 1, gny = 2 * ncy + 1;
        for (int j = 0; j < gny; j++) {
            const int yy = gy0 + 2 * j;
            for (int i = 0; i < gnx; i++) {
                const int xx = gx0 + 2 * i; const int o = j * gnx + i;
                if (xx < 1 || yy < 1 || xx >= W - 1 || yy >= H - 1) { sa[o] = sb[o] = sc2[o] = 0; continue; }
                const uint8_t* q = img + (yy * W + xx) * 3 + 1;
                const float dx = (float)q[3] - (float)q[-3], dy = (float)q[3 * W] - (float)q[-3 * W];
                sa[o] = dx * dx; sb[o] = dx * dy; sc2[o] = dy * dy;
            }
        }
        float best = nact < MS_LOW_N ? MIN_SCORE_LOW : MIN_SCORE; int bx = -1, by = -1;
        for (int cj = 0; cj < ncy; cj++) for (int ci = 0; ci < ncx; ci++) {
            const int x = x0 + DET_STEP / 2 + DET_STEP * ci, y = y0 + DET_STEP / 2 + DET_STEP * cj;
            if (x < 2 * PR + 6 || y < 2 * PR + 6 || x >= W - 2 * PR - 7 || y >= H - 2 * PR - 7) continue;
            float a = 0, b = 0, c = 0;
            for (int j = 0; j < 3; j++) { const int o = (2 * cj + j) * gnx + 2 * ci; for (int i = 0; i < 3; i++) { a += sa[o + i]; b += sb[o + i]; c += sc2[o + i]; } }
            float tr = 0.5f * (a + c), dd = std::sqrt(0.25f * (a - c) * (a - c) + b * b);
            float lmin = (tr - dd) * (1.0f / 36.0f);
            if (lmin > best) { best = lmin; bx = x; by = y; }
        }
#elif DET_SPARSE
        // sparse score: structure tensor from a 3x3 grid of gradient samples 2 px apart (no per-pixel product pass)
        float best = nact < MS_LOW_N ? MIN_SCORE_LOW : MIN_SCORE; int bx = -1, by = -1;
        const int ds = det_refill ? DET_STEP_REFILL : DET_STEP;   // a big refill frame scores a coarser lattice
        for (int y = y0 + ds / 2; y < y0 + chh; y += ds) for (int x = x0 + ds / 2; x < x0 + cw; x += ds) {
            if (x < 2 * PR + 6 || y < 2 * PR + 6 || x >= W - 2 * PR - 7 || y >= H - 2 * PR - 7) continue;
            float a = 0, b = 0, c = 0;
            for (int j = -2; j <= 2; j += 2) {
                const uint8_t* row = img + ((y + j) * W + x) * 3 + 1;
                for (int i = -2; i <= 2; i += 2) {
                    const uint8_t* q = row + 3 * i;
                    const float dx = (float)q[3] - (float)q[-3], dy = (float)q[3 * W] - (float)q[-3 * W];
                    a += dx * dx; b += dx * dy; c += dy * dy;
                }
            }
            float tr = 0.5f * (a + c), dd = std::sqrt(0.25f * (a - c) * (a - c) + b * b);
            float lmin = (tr - dd) * (1.0f / 36.0f);   // 9 samples (scale matches 25 samples / 100)
            if (lmin > best) { best = lmin; bx = x; by = y; }
        }
#else
        // gradient products of every pixel the candidate windows touch, computed once, then 5x5 window sums
        static float pa[64 * 64], pb[64 * 64], pc[64 * 64];
        int ox = x0 + DET_STEP / 2 - 2, oy = y0 + DET_STEP / 2 - 2;     // window origin of the first candidate
        int ex = ox + ((cw - DET_STEP / 2 + DET_STEP - 1) / DET_STEP - 1) * DET_STEP + 5, ey = oy + ((chh - DET_STEP / 2 + DET_STEP - 1) / DET_STEP - 1) * DET_STEP + 5;
        if (ox < 1) ox = 1; if (oy < 1) oy = 1; if (ex > W - 1) ex = W - 1; if (ey > H - 1) ey = H - 1;   // candidates near borders are skipped
        const int nx = ex - ox, ny = ey - oy;
        {   // green rows oy-1 .. oy+ny, columns ox-1 .. ox+nx, extracted once as floats; products on contiguous rows
            static float gr[3][72];
            auto load = [&](float* dst, int y) { const uint8_t* p = img + (y * W + ox - 1) * 3 + 1; for (int i = 0; i < nx + 2; i++) dst[i] = (float)p[3 * i]; };
            load(gr[0], oy - 1); load(gr[1], oy);
            for (int j = 0; j < ny; j++) {
                float* up = gr[j % 3]; float* mid = gr[(j + 1) % 3]; float* dn = gr[(j + 2) % 3];
                load(dn, oy + j + 1);
                float* qa = pa + j * nx; float* qb = pb + j * nx; float* qc = pc + j * nx;
                for (int i = 0; i < nx; i++) {
                    const float dx = mid[i + 2] - mid[i], dy = dn[i + 1] - up[i + 1];
                    qa[i] = dx * dx; qb[i] = dx * dy; qc[i] = dy * dy;
                }
            }
        }
        float best = nact < MS_LOW_N ? MIN_SCORE_LOW : MIN_SCORE; int bx = -1, by = -1;
        for (int y = y0 + DET_STEP / 2; y < y0 + chh; y += DET_STEP) for (int x = x0 + DET_STEP / 2; x < x0 + cw; x += DET_STEP) {
            if (x < 2 * PR + 6 || y < 2 * PR + 6 || x >= W - 2 * PR - 7 || y >= H - 2 * PR - 7) continue;
            float a = 0, b = 0, c = 0;
            {
                const int o0 = (y - 2 - oy) * nx + (x - 2 - ox);
                for (int j = 0; j < 5; j++) { const int o = o0 + j * nx; a += pa[o] + pa[o + 1] + pa[o + 2] + pa[o + 3] + pa[o + 4]; b += pb[o] + pb[o + 1] + pb[o + 2] + pb[o + 3] + pb[o + 4]; c += pc[o] + pc[o + 1] + pc[o + 2] + pc[o + 3] + pc[o + 4]; }
            }
            float tr = 0.5f * (a + c), dd = std::sqrt(0.25f * (a - c) * (a - c) + b * b);
            float lmin = (tr - dd) * (1.0f / 100.0f);
            if (lmin > best) { best = lmin; bx = x; by = y; }
        }
#endif
#endif
        if (bx < 0) { cool[cell] = frame_no + COOL; continue; }
        int slot = -1; for (int i = 0; i < MAX_FEATS; i++) if (!feats[i].active) { slot = i; break; }
        if (slot < 0) break;
        Feat& f = feats[slot];
        f.x = (float)bx; f.y = (float)by; f.fdx = f.fdy = 0; f.nobs = 0;
        if (!cut_template(f)) continue;
        f.active = true; f.anchored = false; nact++;
        if (is_kf) { f.obs[0].serial = frame_no; f.obs[0].u = (float)((f.x - cx) / fx); f.obs[0].v = (float)((f.y - cy) / fy); f.nobs = 1; } else f.nobs = 0;
    }
    det_rr = (det_rr + 7) % (GRID_X * GRID_Y);
}

// ---------------------------------------------------------------- ABI
static double num_after(const char* s, uint32_t n, const char* key, double def) {
    const size_t kl = std::strlen(key);
    for (uint32_t i = 0; i + kl < n; i++) if (std::memcmp(s + i, key, kl) == 0) {
        uint32_t j = i + kl; while (j < n && (s[j] == '"' || s[j] == ':' || s[j] == ' ')) j++;
        return std::strtod(s + j, nullptr);
    }
    return def;
}

extern "C" {
VIOBENCH_EXPORT(abi_version) int32_t abi_version(void) { return VIOBENCH_ABI_VERSION; }
VIOBENCH_EXPORT(alloc) void* alloc(uint32_t size) { return std::malloc(size); }

VIOBENCH_EXPORT(init) int32_t init(const char* cfg, uint32_t n) {
    W = (int)num_after(cfg, n, "\"width\"", 320); H = (int)num_after(cfg, n, "\"height\"", 240);
    fx = num_after(cfg, n, "\"fx\"", 120); fy = num_after(cfg, n, "\"fy\"", 120);
    cx = num_after(cfg, n, "\"cx\"", 159.5); cy = num_after(cfg, n, "\"cy\"", 119.5);
    // T_cam_imu: 16 numbers after the key
    const char* k = "\"T_cam_imu\""; const size_t kl = std::strlen(k);
    double T[16] = {0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1};
    for (uint32_t i = 0; i + kl < n; i++) if (std::memcmp(cfg + i, k, kl) == 0) {
        const char* p = cfg + i + kl; const char* end = cfg + n; int got = 0;
        while (p < end && got < 16) {
            if ((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' || *p == '.') { char* e; double v = std::strtod(p, &e); if (e == p) { p++; continue; } T[got++] = v; p = e; }
            else p++;
        }
        break;
    }
    for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) Rci.m[3 * i + j] = T[4 * i + j]; tci[i] = T[4 * i + 3]; }
    std::memset(P, 0, sizeof P); std::memset(feats, 0, sizeof feats); std::memset(cool, 0, sizeof cool);
    std::memset(Xc, 0, sizeof Xc); std::memset(Pmm, 0, sizeof Pmm); nsch = 0;
    n_state = NI; n_clones = 0; frame_no = 0; started = false; npend = 0; n_splats = 0; last_reset_frame = -100000; mh_C = mh_S = mh_W = 0; mh_ref_ok = mh_basis = false; imu_have = false; v2_avg = 0.25; n_hitf = 0; fej_ok = false; depth_avg = 2.5; moved = false; move_frames = 0;
    return VIOBENCH_STATUS_OK;
}

// ---------------------------------------------------------------- Manhattan heading measurement (MANHATTAN)
// Indoor edges mostly follow a few horizontal axes 90 deg apart. Every frame a slice of a sparse gradient grid is
// turned into 3D edge directions with the filter's attitude; horizontal ones vote for the dominant axis modulo 90 deg
// (4th power of the unit direction, no trig). Each MH_EVERY frames the vote is compared with a reference taken early
// in the flight, and the difference is a yaw measurement (rotation about gravity), gated to +-MH_GATE.
static void manhattan_step() {
    double gn = std::sqrt(gw[0] * gw[0] + gw[1] * gw[1] + gw[2] * gw[2]); if (!(gn > 0)) return;
    const double g[3] = {gw[0] / gn, gw[1] / gn, gw[2] / gn};
    if (!mh_basis) {
        double e[3] = {1, 0, 0}; const double dp = e[0] * g[0] + e[1] * g[1] + e[2] * g[2];
        for (int i = 0; i < 3; i++) e[i] -= dp * g[i];
        const double en = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); for (int i = 0; i < 3; i++) mh_e1[i] = e[i] / en;
        mh_e2[0] = g[1] * mh_e1[2] - g[2] * mh_e1[1]; mh_e2[1] = g[2] * mh_e1[0] - g[0] * mh_e1[2]; mh_e2[2] = g[0] * mh_e1[1] - g[1] * mh_e1[0];
        mh_basis = true;
    }
    const M3 M = mmul(Rwi, mtr(Rci));     // world <- camera
    const int phase = frame_no % MH_EVERY;
    for (int y = MH_STEP + phase * (MH_STEP / MH_EVERY > 0 ? MH_STEP / MH_EVERY : 1); y < H - MH_STEP; y += MH_STEP) {
        const uint8_t* row = img + (y * W) * 3 + 1;
        for (int x = MH_STEP; x < W - MH_STEP; x += MH_STEP) {
            const uint8_t* q = row + 3 * x;
            const double gx = (double)q[3] - (double)q[-3], gy = (double)q[3 * W] - (double)q[-3 * W];
            const double g2 = gx * gx + gy * gy;
            if (g2 < MH_G2) continue;
            const double tnx = -gy / fx, tny = gx / fy, ux = (x - cx) / fx, uy = (y - cy) / fy;
            const double nc[3] = {-tny, tnx, ux * tny - uy * tnx};
            double n[3]; mv(M, nc, n);
            const double nn = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]); if (!(nn > 0)) continue;
            const double ng = (n[0] * g[0] + n[1] * g[1] + n[2] * g[2]) / nn;
            if (std::fabs(ng) < MH_VERT) continue;                 // a vertical edge: no heading information
            const double d[3] = {n[1] * g[2] - n[2] * g[1], n[2] * g[0] - n[0] * g[2], n[0] * g[1] - n[1] * g[0]};
            const double a = d[0] * mh_e1[0] + d[1] * mh_e1[1] + d[2] * mh_e1[2], b = d[0] * mh_e2[0] + d[1] * mh_e2[1] + d[2] * mh_e2[2];
            const double r2 = a * a + b * b; if (!(r2 > 1e-12)) continue;
            const double c2 = a * a - b * b, s2 = 2 * a * b, inv = 1.0 / (r2 * r2);
            const double w = std::sqrt(g2);
            mh_C += w * (c2 * c2 - s2 * s2) * inv; mh_S += w * (2 * c2 * s2) * inv; mh_W += w;
        }
    }
    if (phase != MH_EVERY - 1) return;
    const double C = mh_C, S = mh_S, Wt = mh_W; mh_C = mh_S = mh_W = 0;
#ifndef __wasm__
    if (std::getenv("MH_DUMP") && Wt > 0) std::fprintf(stderr, "MH %d coh %.2f psi %.1f ref %.1f\n", frame_no, std::sqrt(C * C + S * S) / Wt, 0.25 * std::atan2(S, C) * 57.2958, mh_ref * 57.2958);
#endif
    if (!(Wt > 0) || std::sqrt(C * C + S * S) < MH_COH * Wt) return;
    const double psi = 0.25 * std::atan2(S, C);
    if (!mh_ref_ok) { if (frame_no >= MH_T0) { mh_ref = psi; mh_ref_ok = true; } return; }
    double dlt = psi - mh_ref; const double q = 0.5 * 3.14159265358979;
    while (dlt > 0.5 * q) dlt -= q;
    while (dlt < -0.5 * q) dlt += q;
    if (std::fabs(dlt) > MH_GATE) return;
    nrows = 0;
    rnnz[0] = 3; for (int i = 0; i < 3; i++) { ridx[0][i] = TH + i; rval[0][i] = MH_SIGN * g[i]; }
    rs[0] = -dlt; Rd[0] = MH_SIG * MH_SIG; nrows = 1;
    ekf_update(); st_mh_upd++;
}

static void output(double* out) {
    // T_c0_ck = T_c_i * T_w_i(k) * T_i_c   (world = first IMU frame)
    M3 Ric = mtr(Rci); double pic[3]; mtv(Rci, tci, pic); for (int i = 0; i < 3; i++) pic[i] = -pic[i];
    M3 Rwc = mmul(Rwi, Ric); double t[3]; mv(Rwi, pic, t);
    double pwc[3] = {pw[0] + t[0], pw[1] + t[1], pw[2] + t[2]};
    M3 R = mmul(Rci, Rwc); double pos[3]; mv(Rci, pwc, pos); for (int i = 0; i < 3; i++) pos[i] += tci[i];
    double q[4]; rot2quat(R, q);
    // twist in camera axes
    double vi[3]; mtv(Rwi, vw, vi);
    double wx[3] = {w_last[1] * pic[2] - w_last[2] * pic[1], w_last[2] * pic[0] - w_last[0] * pic[2], w_last[0] * pic[1] - w_last[1] * pic[0]};
    for (int i = 0; i < 3; i++) vi[i] += wx[i];
    double vc[3], wc[3]; mv(Rci, vi, vc); mv(Rci, w_last, wc);
    double o[13] = {pos[0], pos[1], pos[2], q[0], q[1], q[2], q[3], vc[0], vc[1], vc[2], wc[0], wc[1], wc[2]};
    bool ok = true; for (int i = 0; i < 13; i++) ok = ok && std::isfinite(o[i]);
    if (!ok) { for (int i = 0; i < 13; i++) o[i] = 0; o[6] = 1; }
    for (int i = 0; i < 13; i++) out[i] = o[i];
}

VIOBENCH_EXPORT(step) int32_t step(double t, const uint8_t* frame, const double* imu, int32_t nimu, double* out) {
    img = frame;
    if (!started) {
        double a[3] = {0, 0, 0}; int k = 0;
        for (int i = 0; i < nimu; i++) { a[0] += imu[7 * i + 1]; a[1] += imu[7 * i + 2]; a[2] += imu[7 * i + 3]; k++; }
        double na = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        if (k == 0 || na < 1e-6) { a[0] = 0; a[1] = 0; a[2] = 1; na = 1; }
        for (int i = 0; i < 3; i++) { gw[i] = -9.81 * a[i] / na; pw[i] = vw[i] = bg[i] = ba[i] = 0; w_last[i] = 0; }
        Rwi = expso3(pw);   // identity
        std::memset(P, 0, sizeof P);
        for (int i = 0; i < 3; i++) {
            PP(VE + i, VE + i) = SIG_V0 * SIG_V0; PP(BG + i, BG + i) = SIG_BG0 * SIG_BG0; PP(BA + i, BA + i) = SIG_BA0 * SIG_BA0;
            PP(TH + i, TH + i) = 1e-10; PP(PS + i, PS + i) = 1e-10;
        }
        // gravity: tangential sigma = 9.81 * tilt sigma, radial small
        double gh[3] = {gw[0] / 9.81, gw[1] / 9.81, gw[2] / 9.81};
        double st = 9.81 * SIG_TILT0, sr = 0.02;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
            PP(GR + i, GR + j) = st * st * ((i == j) - gh[i] * gh[j]) + sr * sr * gh[i] * gh[j];
        t_last = nimu > 0 ? imu[7 * (nimu - 1)] : t;
        if (t_last > t) t_last = t;
        started = true;
        add_clone();
        detect();
        output(out);
        frame_no++;
        return VIOBENCH_STATUS_OK;
    }
    M3 Rold = Rwi;
    const double t_before = t_last, p_old[3] = {pw[0], pw[1], pw[2]};
    propagate(imu, nimu);
    if (LK_TPRED) {   // camera displacement over this frame, in the new camera's axes
        const double dw[3] = {pw[0] - p_old[0], pw[1] - p_old[1], pw[2] - p_old[2]}; double di[3]; mtv(Rwi, dw, di); mv(Rci, di, tpred_c);
    }
    if (t_last > t_before) frame_dt = t_last - t_before;
    // gyro rotation between frames, in camera axes: R_cnew_cold = Rci * (Rold^T Rnew)^T * Rci^T
    M3 dR = mmul(mtr(Rold), Rwi);
    M3 Rcc = mmul(mmul(Rci, mtr(dR)), mtr(Rci));
#if PROF_TRACK
    { static Feat save[MAX_FEATS]; std::memcpy(save, feats, sizeof feats); track(Rcc); std::memcpy(feats, save, sizeof feats); }
#endif
    track(Rcc);
    if (SCHED) {   // committed work is known: cap this frame's update rows so that a minimum of detection still fits
        const double committed = SC_BASE + SC_TRACK * frame_tracks;
        int need = SC_MINCELLS;
        if (SCHED >= 2) {   // protect detection: reserve the cells this frame will need (tracks lost), cap rows instead
            int act = 0; for (int i = 0; i < MAX_FEATS; i++) act += feats[i].active;
            need = (int)(SC_NEED_K * (MAX_FEATS - act)); if (need < SC_MINCELLS) need = SC_MINCELLS; if (need > DETECT_CELLS) need = DETECT_CELLS;
        }
        sched_need = need;
        int rc = (int)((SCHED_B - committed - SC_CELL * need) / SC_ROW);
        row_cap = rc < SC_MINROWS ? SC_MINROWS : (rc > MAX_ROWS ? MAX_ROWS : rc);
        sched_used = committed;
    }
    n_tried = n_ok = 0; frame_tri_fail = 0;
    // finished tracks: lost ones, and those spanning the whole window (the oldest clone leaves after this update)
    nrows = 0; ncand = 0;
    is_kf = KF_PX <= 0 || n_clones == 0 || frame_no - clones[n_clones - 1].serial >= KF_MAXGAP || disparity_since_last_clone() >= KF_PX;
    int thin = -1;   // two-rate window: the clone leaving the dense block is dropped if too close to its older neighbour
    if (TWO_RATE && is_kf && n_clones == N_CLONES) {
        const int j = n_clones - DENSE;
        if (j >= 1 && clones[j].serial - clones[j - 1].serial < SPACING) thin = j;
    }
    const bool full = is_kf && n_clones == N_CLONES && thin < 0;
    const int old = full ? clones[0].serial : -1;
    for (int i = 0; i < MAX_FEATS; i++) {
        Feat& f = feats[i];
        if (f.nobs == 0) continue;
        if (!f.active) { if (LOST_DEFER) defer_obs(f); else queue_obs(f); f.nobs = 0; continue; }
        if (STAGGER > 0 && is_kf && f.nobs >= N_CLONES - (i % (STAGGER + 1))) { queue_obs(f); f.nobs = 0; continue; }   // spread completions
        if (full && f.obs[0].serial == old) {
            if (f.nobs >= (TWO_RATE ? 3 : N_CLONES - 1)) { queue_obs(f); f.nobs = 0; }
            else { for (int o = 0; o + 1 < f.nobs; o++) f.obs[o] = f.obs[o + 1]; f.nobs--; }
        }
    }
    process_candidates(old);
    for (int d = 0; d < ndef; d++) {   // lost tracks of this frame join the queue for the next one
        Pending& c = dlist[d]; int k = 0; for (int o = 0; o < c.nobs; o++) if (c.obs[o].serial != old) c.obs[k++] = c.obs[o];
        c.nobs = k; if (k >= 3 && npend < MAX_PENDING) pend[npend++] = c;
    }
    ndef = 0;
    frame_rows = nrows;
    sched_used += SC_ROW * nrows;
#if PROF_UPDATE
    { static double Ps[NMAX * NMAX]; std::memcpy(Ps, P, sizeof P); M3 R0 = Rwi; double s0[15]; for (int i = 0; i < 3; i++) { s0[i] = pw[i]; s0[3 + i] = vw[i]; s0[6 + i] = bg[i]; s0[9 + i] = ba[i]; s0[12 + i] = gw[i]; }
      static Clone cs[N_CLONES]; std::memcpy(cs, clones, sizeof clones); ekf_update();
      std::memcpy(P, Ps, sizeof P); Rwi = R0; for (int i = 0; i < 3; i++) { pw[i] = s0[i]; vw[i] = s0[3 + i]; bg[i] = s0[6 + i]; ba[i] = s0[9 + i]; gw[i] = s0[12 + i]; } std::memcpy(clones, cs, sizeof clones); }
#endif
    ekf_update();
    if (full) drop_oldest_clone();
    if (thin >= 0) {
        const int ts = clones[thin].serial;
        for (int i = 0; i < MAX_FEATS; i++) { Feat& f = feats[i]; int k = 0; for (int o = 0; o < f.nobs; o++) if (f.obs[o].serial != ts) f.obs[k++] = f.obs[o]; f.nobs = k; }
        for (int i = 0; i < npend; i++) { Pending& c = pend[i]; int k = 0; for (int o = 0; o < c.nobs; o++) if (c.obs[o].serial != ts) c.obs[k++] = c.obs[o]; c.nobs = k; }
        drop_clone(thin);
    }
    if (n_tried >= 3 && n_ok == 0) bad_streak++; else if (n_ok > 0) bad_streak = 0;
    {   // divergence check: triangulation success over the last RESET_WIN frames
        const int slot = frame_no % RESET_WIN;
        ring_c[slot] = n_tried + frame_tri_fail; ring_s[slot] = n_tried;
        int c = 0, ok = 0; for (int i = 0; i < RESET_WIN; i++) { c += ring_c[i]; ok += ring_s[i]; }
        const double sp = std::sqrt(vw[0] * vw[0] + vw[1] * vw[1] + vw[2] * vw[2]);
        if ((c >= RESET_MIN && ok < RESET_RATE * c) || sp > VMAX) {
            if (RESET_SOFT && sp <= VMAX) {   // soft: keep the window and the velocity, re-open the velocity uncertainty
                for (int i = 0; i < 3; i++) PP(VE + i, VE + i) += SOFT_SIG * SOFT_SIG;
            } else {
            double vc[3];
            const bool vis = RESET_VIS && visual_velocity(frame_dt, vc);
            if (vis) { double vi[3]; mtv(Rci, vc, vi); mv(Rwi, vi, vw); }
            else for (int i = 0; i < 3; i++) vw[i] = 0;
            for (int j = 0; j < n_state; j++) for (int i = 0; i < 3; i++) { PP(VE + i, j) = 0; PP(j, VE + i) = 0; }
            if (SCHMIDT) for (int i = 0; i < 3; i++) for (int m = 0; m < 3 * nsch; m++) XC(VE + i, m) = 0;
            for (int i = 0; i < 3; i++) PP(VE + i, VE + i) = vis ? SIG_VVIS * SIG_VVIS : SIG_VRESET * SIG_VRESET;
            while (n_clones > 0) drop_oldest_clone();
            for (int i = 0; i < MAX_FEATS; i++) feats[i].nobs = 0;
            npend = 0;
            }
            for (int i = 0; i < RESET_WIN; i++) ring_c[i] = ring_s[i] = 0;
            n_resets++; last_reset_frame = frame_no;
#ifndef __wasm__
            if (std::getenv("MSCKF_RESETS")) std::fprintf(stderr, "reset f %d cand %d ok %d speed %.2f\n", frame_no, c, ok, sp);
#endif
        }
    }
    if (bad_streak >= RECOVER_N) { for (int i = 0; i < 3; i++) PP(VE + i, VE + i) += RECOVER_SIG * RECOVER_SIG; bad_streak = 0; }
    // zero-velocity update when the image says the drone is still
    if (nflow >= 8) {
        const float med = ZV_DEROT ? derotated_median() : 0;
        for (int i = 1; i < nflow; i++) { float v = flow[i]; int j = i - 1; while (j >= 0 && flow[j] > v) { flow[j + 1] = flow[j]; j--; } flow[j + 1] = v; }
        if (ZV_DEROT) flow[nflow / 2] = med;
#ifndef __wasm__
        if (std::getenv("MSCKF_FLOW") && frame_no < 120) std::fprintf(stderr, "flow f %d med %.3f n %d\n", frame_no, ZV_DEROT ? derotated_median() : flow[nflow / 2], nflow);
#endif
        const double wn = std::sqrt(w_last[0] * w_last[0] + w_last[1] * w_last[1] + w_last[2] * w_last[2]);
        if (ZV_START_ONLY == 1) { if (flow[nflow / 2] > ZV_MOVE_PX) move_frames++; else move_frames = 0; }
        if (move_frames >= 3 && ZV_START_ONLY == 1) moved = true;
        bool still = flow[nflow / 2] < ZV_PX;
        double zsig = ZV_SIG;
        if (ZV_WIN > 0) {
            const float wd = window_disparity();
            if (wd >= 0) {
                still = wd < (ZV_SOFT ? ZV_SOFT_MAX : ZV_WIN);
                if (ZV_SOFT) zsig = ZV_SIG * (1.0 + (wd / ZV_WIN) * (wd / ZV_WIN));   // weaker as the window disparity grows
                if (ZV_START_ONLY == 2) { if (wd > ZV_MOVE_WIN) move_frames++; else move_frames = 0; if (move_frames >= 5) moved = true; }
            }
        }
        if (!moved && still && wn < ZV_GYR) {
            nrows = 0;
            for (int i = 0; i < 3; i++) add_row1(VE + i, -vw[i], zsig * zsig);
            ekf_update();
            zupt_count++;
        }
    }
    if (SCHED ? sched_used + SC_CELL * (SCHED >= 2 ? sched_need : 0) + SC_SPLAT <= SCHED_B : frame_rows <= SPL_SKIP_ROWS) { splat_step(); if (SCHED) sched_used += SC_SPLAT; }
    if (MANHATTAN) manhattan_step();
    if (is_kf) add_clone();
    // record observations of the surviving tracks in the new clone, re-cut templates
    for (int i = 0; i < MAX_FEATS; i++) {
        Feat& f = feats[i]; if (!f.active) continue;
#if PROF_TPL
        cut_template(f);
#endif
        if (!cut_template(f)) { f.active = false; continue; }
        if (ANCHOR && (!f.anchored || f.nobs == 0)) cut_anchor(f);
        if (is_kf && f.nobs < N_CLONES + 1) { f.obs[f.nobs].serial = frame_no; f.obs[f.nobs].u = (float)((f.x - cx) / fx); f.obs[f.nobs].v = (float)((f.y - cy) / fy); f.nobs++; }
    }
#if PROF_DETECT
    { static Feat save[MAX_FEATS]; static int cs[GRID_X * GRID_Y]; std::memcpy(save, feats, sizeof feats); std::memcpy(cs, cool, sizeof cool); const int rr = det_rr;
      detect(); std::memcpy(feats, save, sizeof feats); std::memcpy(cool, cs, sizeof cool); det_rr = rr; }
#endif
    detect();
    // flight's typical speed: exponential mean of |v|^2 over ~VP_TAU seconds (starts at VP_V0^2)
    v2_avg += (vw[0] * vw[0] + vw[1] * vw[1] + vw[2] * vw[2] - v2_avg) * (frame_dt / VP_TAU);
    if (VPRIOR_SIG > 0 && frame_no % VPRIOR_EVERY == 0) {
        nrows = 0;
        double sig = VPRIOR_SIG;
        if (VP_ADAPT) { sig = VP_K * std::sqrt(v2_avg); if (sig < VP_MIN) sig = VP_MIN; if (sig > VPRIOR_SIG) sig = VPRIOR_SIG; }
        for (int i = 0; i < 3; i++) add_row1(VE + i, -vw[i], sig * sig);
        ekf_update();
    }
    output(out);
#ifndef __wasm__
    if (std::getenv("MSCKF_FRAMESTATS")) {
        static long pc = 0, pcell = 0, ptr = 0, pst = 0; static int pres = 0;
        std::fprintf(stderr, "FS %d tri %ld cells %ld tracks %ld rows %d resets %d pend %d nst %d splat_try %ld\n", frame_no, st_cand - pc, st_cells - pcell, st_tracks - ptr, frame_rows, n_resets - pres, npend, n_state, spl_try - pst);
        pc = st_cand; pcell = st_cells; ptr = st_tracks; pres = n_resets; pst = spl_try;
    }
    if (SPLAT && std::getenv("MSCKF_STATS") && frame_no == 1199) std::fprintf(stderr, "manhattan updates %ld | splats %d tries %ld hits %ld | late: made %ld hits %ld rejected smax %ld view %ld | schmidt upd %ld active %d\n", st_mh_upd, n_splats, spl_try, spl_hit, spl_late_made, spl_late_hit, spl_late_smax, spl_late_view, st_sch_upd, nsch);
    if (std::getenv("MSCKF_STATS") && frame_no == 1199) std::fprintf(stderr, "cand %ld tri_fail %ld (parallax %ld reproj %ld) chi_rej %ld ok %ld far %ld resets %d cells/frame %.1f tracks/frame %.1f\n", st_cand, st_tri, st_par, st_rep, st_chi, st_ok, st_far, n_resets, st_cells / 1200.0, st_tracks / 1200.0);
    if (std::getenv("MSCKF_DEBUG") && frame_no % (std::getenv("MSCKF_EVERY") ? atoi(std::getenv("MSCKF_EVERY")) : 10) == 0) {
        int na = 0; for (int i = 0; i < MAX_FEATS; i++) na += feats[i].active;
        std::fprintf(stderr, "f %d act %d rows %d v %.2f %.2f %.2f |g| %.2f bg %.3f %.3f %.3f ba %.2f %.2f %.2f sv %.3f zupts %d\n", frame_no, na, dbg_rows, vw[0], vw[1], vw[2],
            std::sqrt(gw[0]*gw[0]+gw[1]*gw[1]+gw[2]*gw[2]), bg[0], bg[1], bg[2], ba[0], ba[1], ba[2], std::sqrt(PP(VE,VE)), zupt_count);
    }
#endif
    frame_no++;
    return VIOBENCH_STATUS_OK;
}
}

extern "C" int32_t msckf_frames_since_reset(void) { return frame_no - last_reset_frame; }   // router health signal
