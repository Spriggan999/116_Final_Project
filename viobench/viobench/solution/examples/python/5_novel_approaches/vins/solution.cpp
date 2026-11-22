// vins: VINS-Mono-style tightly coupled sliding-window visual-inertial odometry (Qin, Li, Shen, T-RO 2018),
// on the KLT front end of ../msckf. See README.md.
//
// Back end (VINS-Mono):
//   - window of up to WIN frames (keyframes + the newest frame), each with R, p, v, b_g, b_a (15 error states),
//     plus gravity direction (2 states; world = first IMU frame, as in ../msckf)
//   - IMU preintegration between window frames (bias Jacobians, covariance), composed when a frame is dropped
//   - features as inverse depth anchored at their first observation in the window, Huber reprojection cost,
//     eliminated per feature (Schur complement) before the dense solve
//   - Gauss-Newton (NIT iterations per frame, small Levenberg damping)
//   - two-way marginalization: the second-newest frame is dropped when its parallax to the previous keyframe is
//     small (its IMU factors are merged), otherwise the oldest frame is marginalized into a dense prior
// Changes from the paper (from ../msckf and experiments): priors instead of the SfM initialization, a weak velocity
// prior, zero-velocity factors, divergence reset with a visual velocity guess, features marginalized once (no
// double counting) unless MARG_KEEP=1.
#include "viobench_abi.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#ifndef __wasm__
#include <cstdio>
#endif

// ---------------------------------------------------------------- settings
#ifndef PRESET
#define PRESET 0                 // 0: full (vins.wasm), 1: lite (vins_lite.wasm), 2: ultra (vins_ultra.wasm); single flags still override
#endif
#if PRESET == 1
#ifndef WIN
#define WIN 6
#endif
#ifndef DET_STEP
#define DET_STEP 4
#endif
#ifndef MARG_FMAX
#define MARG_FMAX 12
#endif
#ifndef MAX_FEATS
#define MAX_FEATS 24
#endif
#ifndef DETECT_CELLS
#define DETECT_CELLS 24
#endif
#ifndef KF_PAR
#define KF_PAR 0.07
#endif
#elif PRESET == 2
#ifndef WIN
#define WIN 5
#endif
#ifndef MARG_FMAX
#define MARG_FMAX 12
#endif
#ifndef MAX_FEATS
#define MAX_FEATS 20
#endif
#ifndef DETECT_CELLS
#define DETECT_CELLS 16
#endif
#ifndef KF_PAR
#define KF_PAR 0.08
#endif
#ifndef MAP_PF
#define MAP_PF 1
#endif
#endif
#ifndef WIN
#define WIN 11                   // frames in the window during optimization (VINS-Mono: 10 keyframes + newest)
#endif
#ifndef MAX_FEATS
#define MAX_FEATS 32             // live KLT tracks
#endif
#ifndef EDGE
#define EDGE 0                   // 1: edgelets (PL-VINS / MLINE-VINS line features, SVO-style): points on edges tracked along the
#endif                           //    edge normal, residual = distance across the edge; detected in cells without a corner
#ifndef EDGE_MAX
#define EDGE_MAX 10              // most live edgelets (extra slots beside the MAX_FEATS corner tracks)
#endif
#ifndef EDGE_MIN
#define EDGE_MIN 0.25f           // edge strength lambda_max / 36 (rendered walls: 4-level steps give about 1.3)
#endif
#ifndef EDGE_RATIO
#define EDGE_RATIO 0.15f         // lambda_min / lambda_max below this: an edge
#endif
#ifndef EDGE_PRETEST
#define EDGE_PRETEST 2           // cells below PRETEST but with a gradient of at least this are searched for edgelets
#endif
#ifndef EDGE_WHEN
#define EDGE_WHEN 1000           // edgelets detected only while fewer corner tracks than this are live
#endif
#ifndef OBS_GATE
#define OBS_GATE 0.0             // >0 (DynaVINS-style): a new observation farther than this (px) from the IMU-predicted projection of
#endif                           //    its triangulated point ends the track (the observation is not added)
#ifndef MAH_PRED
#define MAH_PRED 0               // 1: LK starts at the projection of the feature's triangulated point into the IMU-predicted pose
#endif                           //    (instead of the gyro rotation + last flow); falls back when the point is unknown
#ifndef MAH_PRED_PX
#define MAH_PRED_PX 2.0          // MAH_PRED 2: the predicted start is tried when the usual LK result is farther than this (L1 px) ...
#endif
#ifndef MAH_PRED_K
#define MAH_PRED_K 0.8           // ... and kept when its LK residual is below this fraction of the usual one
#endif
#ifndef VW_DYN
#define VW_DYN 0.0               // >0 (DM-VIO dynamic photometric weight): when the RMS reprojection error of the last solve (in SIG_PX)
#endif                           //   exceeds this, all visual information is scaled by (VW_DYN / rms)^2
#ifndef SCALE_ST
#define SCALE_ST 0               // 1 (VI-DSO: scale as an explicit state): the depth prior's centre is a window state s (log scene depth,
#endif                           //   prior N(ln DPF_Z0, S_SIG0^2), information floor S_FLOOR), estimated jointly, kept through marginalization
#ifndef S_SIG0
#define S_SIG0 0.5
#endif
#ifndef S_FLOOR
#define S_FLOOR 0.1              // the prior's sigma of s is kept at least this (scene depth changes from room to room)
#endif
#ifndef S_V0
#define S_V0 0.0                 // >0 (with SCALE_ST; router_dp's gate): below this running RMS speed (m/s) s is pinned to ln DPF_Z0 by a
#endif                           //   unary factor of sigma S_PIN (slow flights: the IMU cannot fix the scale); free above S_V1, linear ramp between
#ifndef S_V1
#define S_V1 0.0
#endif
#ifndef S_PIN
#define S_PIN 0.02
#endif
#ifndef ST_W
#define ST_W 0                   // 1 (ROVIO: the photometric innovation weights a patch by its gradients): corner observations are
#endif                           //   whitened by the template's structure tensor G / ST_G0, eigenvalues clamped to [ST_MIN, 1]
#ifndef ST_G0
#define ST_G0 8000.0
#endif
#ifndef ST_MIN
#define ST_MIN 0.1
#endif
#ifndef NB_FLOW
#define NB_FLOW 0                // DSO coarse-tracking / ROVIO-style motion-aware LK start: 1: tracks one frame old (no flow history yet)
#endif                           //   start at the derotated flow of the established tracks near them; 2: also retry failed tracks from there
#ifndef NB_R
#define NB_R 20.0f               // px: neighbour weight 1 / (d^2 + NB_R^2)
#endif
#ifndef MAH
#define MAH 0                    // 1 (VINS-Mah): before each solve, a new observation whose Mahalanobis distance to the predicted
#endif                           //    projection (newest-frame covariance from the last factorization + IMU, point, pixel noise)
#ifndef MAH_CHI
#define MAH_CHI 5.99             //    exceeds this chi-square level (2 dof; edgelets 1 dof: MAH_CHI1) ends its track
#endif
#ifndef MAH_CHI1
#define MAH_CHI1 3.84
#endif
#ifndef MAH_SIG
#define MAH_SIG 1.0              // pixel noise in the gate (px)
#endif
#ifndef MAX_WF
#define MAX_WF 96                // window features (live tracks and lost tracks whose observations are still in the window)
#endif
#ifndef GRID_X
#define GRID_X 8
#endif
#ifndef GRID_Y
#define GRID_Y 6
#endif
#ifndef DETECT_CELLS
#define DETECT_CELLS 48
#endif
#ifndef DET_STEP
#define DET_STEP 6
#endif
#ifndef MIN_SCORE
#define MIN_SCORE 10.0f
#endif
#ifndef PRETEST
#define PRETEST 6
#endif
#ifndef COOL
#define COOL 2
#endif
#ifndef PR
#define PR 3
#endif
#ifndef LK_FINE
#define LK_FINE 6
#endif
#ifndef MAX_RES
#define MAX_RES 14.0f
#endif
#ifndef LK_EPS2
#define LK_EPS2 0.0009f
#endif
#ifndef SIG_PX
#define SIG_PX 0.7               // reprojection noise (px)
#endif
#ifndef HUBER
#define HUBER 1.0                // Huber threshold (in SIG_PX)
#endif
#ifndef SIG_ACC
#define SIG_ACC 0.28
#endif
#ifndef SIG_GYR
#define SIG_GYR 0.07
#endif
#ifndef SIG_BGW
#define SIG_BGW 2e-4
#endif
#ifndef SIG_BAW
#define SIG_BAW 2e-3
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
#ifndef KF_PAR
#define KF_PAR 0.03              // rad: second-newest frame is a keyframe if its median rotation-compensated parallax to the previous keyframe is above this
#endif
#ifndef KF_MAXGAP
#define KF_MAXGAP 10             // ... or it is this many frames after the previous keyframe
#endif
#ifndef KF_MINCOM
#define KF_MINCOM 8              // ... or fewer features are seen in both
#endif
#ifndef NIT
#define NIT 1                    // Gauss-Newton iterations per frame
#endif
#ifndef LM_REL
#define LM_REL 0.0               // Levenberg damping (relative to the diagonal); 1e-4 already freezes the weak scale direction
#endif
#ifndef MIN_PARALLAX
#define MIN_PARALLAX 0.02
#endif
#ifndef MIN_DEPTH
#define MIN_DEPTH 0.15
#endif
#ifndef MAX_DEPTH
#define MAX_DEPTH 40.0
#endif
#ifndef TRI_MAX_PX
#define TRI_MAX_PX 2.0
#endif
#ifndef OUT_PX
#define OUT_PX 1.5               // feature dropped (and its track killed) when its mean reprojection error exceeds this
#endif
#ifndef MARG_KEEP
#define MARG_KEEP 0              // 1: VINS-Mono: features marginalized with the oldest frame keep their later observations
#endif
#ifndef VPRIOR_SIG
#define VPRIOR_SIG 3.0
#endif
#ifndef VP_ADAPT
#define VP_ADAPT 1
#endif
#ifndef VP_K
#define VP_K 1.2
#endif
#ifndef VP_MIN
#define VP_MIN 0.8
#endif
#ifndef VP_TAU
#define VP_TAU 10.0
#endif
#ifndef VP_DT
#define VP_DT 0.5                // the velocity prior has information 1/sig^2 per VP_DT seconds of window
#endif
#ifndef ZV_SIG
#define ZV_SIG 0.15
#endif
#ifndef ZV_GYR
#define ZV_GYR 0.2
#endif
#ifndef ZV_WIN
#define ZV_WIN 1.5
#endif
#ifndef ZW_PCT
#define ZW_PCT 90
#endif
#ifndef ZW_MIN_SPAN
#define ZW_MIN_SPAN 8
#endif
#ifndef ZV_PX
#define ZV_PX 0.25
#endif
#ifndef ZW_LAG
#define ZW_LAG 10                // stillness: disparity of each track over the last ZW_LAG frames (at least ZW_MIN_SPAN)
#endif
#ifndef VARPRO
#define VARPRO 3                 // Gauss-Newton iterations on each inverse depth (poses fixed) before every joint step
#endif
#ifndef DPF_SIG
#define DPF_SIG 1.0              // >0: log-depth prior on every feature, ln(depth / DPF_Z0) ~ N(0, DPF_SIG^2) (indoor rooms)
#endif
#ifndef DPF_Z0
#define DPF_Z0 2.2
#endif
#ifndef OPT_FMAX
#define OPT_FMAX 1000            // most features in a solve (most observations first)
#endif
#ifndef KF_PRE
#define KF_PRE 1                 // 1: keyframe decision taken before the solve
#endif
#ifndef MAP_SKIP_MARG
#define MAP_SKIP_MARG 1          // 1 (with KF_PRE): no map search on frames that marginalize
#endif
#ifndef NIT2_LIGHT
#define NIT2_LIGHT 0             // 1 (with KF_PRE): two Gauss-Newton iterations on frames that do not marginalize
#endif
#ifndef LK_R0
#define LK_R0 0.0                // >0: observation sigma = SIG_PX sqrt(1 + (LK residual / LK_R0)^2)
#endif
#ifndef OUT_KEEP_TRACK
#define OUT_KEEP_TRACK 0         // 1: an outlier feature restarts its window observations instead of ending its track
#endif
#ifndef DEFER_MARG
#define DEFER_MARG 0             // 1: no detection or map-point creation on frames that marginalize (done on the next frame)
#endif
#ifndef MARG_CACHE
#define MARG_CACHE 0             // 1 (with KF_PRE): marginalization reuses the solve's reduced feature blocks (no Jacobian rebuild)
#endif
#ifndef MARG_LIVE
#define MARG_LIVE 1              // 0: only features whose tracks ended are marginalized; live ones drop their oldest observation
#endif
#ifndef COV_FAST
#define COV_FAST 1               // 1: block-structured preintegration covariance step (same result up to rounding)
#endif
#ifndef LK_FAST
#define LK_FAST 1                // 1: SIMD LK for the tracks (PR = 3)
#endif
#ifndef IMU_MID
#define IMU_MID 0                // 1: midpoint IMU integration (mean of consecutive samples)
#endif
#ifndef MANHATTAN
#define MANHATTAN 0              // 1: Manhattan-world heading factor (from ../msckf) on slow flights, on the last keyframe
#endif
#ifndef MH_VRUN
#define MH_VRUN 0.45             // m/s: only while the running RMS speed is below this
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
#ifndef DET_MARG_CELLS
#define DET_MARG_CELLS 0         // >0: at most this many detection cells on frames that marginalize
#endif
#ifndef MS_LOW_N
#define MS_LOW_N 0               // >0: with fewer live tracks than this, detection accepts corners down to MIN_SCORE_LOW
#endif
#ifndef MIN_SCORE_LOW
#define MIN_SCORE_LOW 5.0f
#endif
#ifndef PRETEST_LOW
#define PRETEST_LOW PRETEST      // with MS_LOW_N: pretest threshold when few live tracks (rendered walls: 4-level edges, no noise)
#endif
#ifndef OUT_LOW_N
#define OUT_LOW_N 0              // >0: with fewer features in the solve than this, the outlier threshold is OUT_PX_LOW
#endif
#ifndef OUT_PX_LOW
#define OUT_PX_LOW 3.0
#endif
#ifndef MARG_FMAX
#define MARG_FMAX 8           // most features marginalized with the oldest frame (bounds the cost of that frame)
#endif
#ifndef DPF_ADAPT
#define DPF_ADAPT 0              // >0: the depth prior's centre follows the log depth of well-triangulated marginalized features (1/DPF_ADAPT per feature)
#endif
#ifndef DPF_ADAPT_PAR
#define DPF_ADAPT_PAR 0.1        // rad: triangulation angle needed for a feature to count
#endif
#ifndef DPF_INIT
#define DPF_INIT 1               // with DPF_SIG > 0: a feature that fails triangulation starts at depth DPF_Z0 (VINS-Mono: INIT_DEPTH)
#endif
#ifndef DPF_MINOBS
#define DPF_MINOBS 3
#endif
#ifndef OBS_MAX
#define OBS_MAX 7                // >0: a feature uses at most this many observations (anchor, newest, evenly spaced between)
#endif
#ifndef PROF_CHOL2
#define PROF_CHOL2 0
#endif
#ifndef PROF_BUILD2
#define PROF_BUILD2 0
#endif
#ifndef CHOL4
#define CHOL4 1                  // register-tiled Cholesky (4 rows per pass)
#endif
#ifndef CHOL_F32
#define CHOL_F32 0               // 1: the GN step solved in single precision (Jacobi-scaled) + double iterative refinement
#endif
#ifndef CHORD
#define CHORD 0                  // 1: iterations after the first reuse the first factorization (gradient-only rebuild)
#endif
#ifndef IMU_EST
#define IMU_EST 0                // 1: IMU white-noise levels estimated online (successive-sample differences, var(d) = 2 sigma^2)
#endif
#ifndef IMU_EST_N0
#define IMU_EST_N0 200
#endif
#ifndef IMU_EST_TH
#define IMU_EST_TH 1.5           // keep the nominal noise unless the estimate exceeds this x nominal
#endif
#ifndef IMU_EST_MAX
#define IMU_EST_MAX 5.0
#endif
#ifndef IMU_EST_BIAS
#define IMU_EST_BIAS 1           // once a noisy IMU is detected (after IMU_EST_BN samples), widen the prior's bias sigmas by the same factor
#endif
#ifndef IMU_EST_BN
#define IMU_EST_BN 100
#endif
#ifndef V2_MINF
#define V2_MINF 10                // the running speed (velocity prior, map gate) is updated only on frames whose solve used this many features
#endif
#ifndef ZV_KEEP
#define ZV_KEEP 1                // frames with a zero-velocity factor are keyframes
#endif
#ifndef VMAX
#define VMAX 6.0
#endif
#ifndef SIG_VVIS
#define SIG_VVIS 0.4
#endif
#ifndef SIG_VRESET
#define SIG_VRESET 1.0
#endif
#ifndef RESET_BG
#define RESET_BG 0.01
#endif
#ifndef RESET_BA
#define RESET_BA 0.05
#endif
#ifndef RESET_G
#define RESET_G 0.03
#endif
#ifndef RESET_VCAP_K
#define RESET_VCAP_K 2.0
#endif
#ifndef RESET_VCAP_MIN
#define RESET_VCAP_MIN 1.0
#endif

#ifndef PIPE
#define PIPE 0                   // 1: the optimizer runs on every second frame, its work split over two frames (A: new window
#endif                           //    frame, build, first part of the Cholesky; B: rest of the Cholesky, step, marginalization)
#ifndef PIPE_SPLIT
#define PIPE_SPLIT 0.5           // PIPE 2: fraction of the Cholesky columns factored on frame A
#endif
#ifndef PIPE_S1
#define PIPE_S1 0.3              // PIPE 3: Cholesky columns [0, S1 n) on frame A, [S1 n, S2 n) on the middle frame, the rest on B
#endif
#ifndef PIPE_S2
#define PIPE_S2 0.75
#endif
#ifndef PIPE_DET
#define PIPE_DET 0               // PIPE: 0 detection on frame B, 1 on frame A, 2 on the middle frame (PIPE 3), 3 (PIPE_SCHED) every frame, DETECT_CELLS / PIPE cells each
#endif
#ifndef PIPE_SCHED
#define PIPE_SCHED 0             // PIPE: 1 = the solve chain (build, Cholesky, step, marginalization) runs in small units spread over the
#endif                           //    PIPE frames by an estimated cost (cumulative targets PIPE_F0, PIPE_F1 of the cycle's total)
#ifndef PIPE_F0
#define PIPE_F0 (PIPE >= 3 ? 0.28 : 0.45)
#endif
#ifndef PIPE_F1
#define PIPE_F1 0.64
#endif
#ifndef PIPE_F2
#define PIPE_F2 0.82
#endif
#ifndef CU_N2
#define CU_N2 19.7               // cost model (about wasm instructions): base build per n^2
#endif
#ifndef CU_IMU
#define CU_IMU 52000.0           // per IMU factor
#endif
#ifndef CU_F0
#define CU_F0 3200.0             // per feature
#endif
#ifndef CU_FM
#define CU_FM 900.0              // per feature and squared observation count
#endif
#ifndef CU_CH
#define CU_CH 5.5                // Cholesky column j: CU_CH (n - j)(j + 8)
#endif
#ifndef CU_SOLVE
#define CU_SOLVE 24.0            // step: per n^2
#endif
#ifndef CU_SR
#define CU_SR 12.0               // marginalization Schur rows: per len r
#endif
#ifndef PIPE_MAX
#define PIPE_MAX PIPE            // PIPE_SCHED 2: a cycle may last up to PIPE_MAX frames ...
#endif
#ifndef PIPE_CAP
#define PIPE_CAP 1.9e6           // ... when its estimated work per frame would exceed this (cost units, about wasm fuel)
#endif
#ifndef PIPE_RES
#define PIPE_RES 0.0             // PIPE_SCHED 2: headroom kept on the last frame of a cycle (cost units)
#endif
#ifndef CU_TRACK
#define CU_TRACK 11600.0         // PIPE_SCHED 2: fixed work of a frame: per live track (LK + template)
#endif
#ifndef CU_CELL
#define CU_CELL 9300.0           // ... per detection cell scored
#endif
#ifndef CU_SEARCH
#define CU_SEARCH 213000.0       // ... per map search
#endif
#ifndef CU_DET
#define CU_DET 0.0           // ... per new track
#endif
#ifndef CU_PH0
#define CU_PH0 133000.0           // ... frame A (new window frame, stillness, triangulation)
#endif
#ifndef CU_PHL
#define CU_PHL 77000.0           // ... last frame of the cycle (map points)
#endif
#ifndef CU_MID
#define CU_MID 93000.0           // ... any other frame
#endif
#ifndef CU_CFULL
#define CU_CFULL 15400.0         // ... extra per detection cell that passes the pretest (fully scored)
#endif
#ifndef PIPE_SEARCH_ALL
#define PIPE_SEARCH_ALL 1        // PIPE: map search on every frame A (frame A never marginalizes)
#endif
#ifndef SHARED_BIAS
#define SHARED_BIAS 1            // one gyro and one accel bias for the whole window (else per frame, as VINS-Mono)
#endif
#ifndef BIAS_FLOOR_G
#define BIAS_FLOOR_G 2e-3        // SHARED_BIAS: the prior's gyro-bias sigma is kept at least this (rad/s): slow drift stays trackable
#endif
#ifndef BIAS_FLOOR_A
#define BIAS_FLOOR_A 0.01        // ... and accel-bias sigma (m/s^2)
#endif
#define FEAT_N (MAX_FEATS + (EDGE ? EDGE_MAX : 0))   // track slots (corners + edgelets)
static const int NF_ = SHARED_BIAS ? 9 : 15;     // states per frame: theta, p, v (+ bg, ba)
static const int G0 = (SHARED_BIAS ? 8 : 2) + (SCALE_ST ? 1 : 0);   // gravity (2) [+ bg, ba (6)] [+ log scene depth s] come first
static const int SI = G0 - 1;                    // SCALE_ST: index of s
static double sst = 0, sstl = 0;                 // SCALE_ST: s - ln DPF_Z0 and its prior linearization point
#define DPF_C (SCALE_ST ? dpf_lz0 + sst : dpf_lz0)   // ln of the depth prior's centre
static const int NX = G0 + NF_ * WIN;

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
static inline M3 eye3() { M3 r = {{1, 0, 0, 0, 1, 0, 0, 0, 1}}; return r; }
static inline M3 expso3(const double* w) {
    double th2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2], th = std::sqrt(th2), a, b;
    if (th < 1e-7) { a = 1 - th2 / 6; b = 0.5 - th2 / 24; }
    else { a = std::sin(th) / th; b = (1 - std::cos(th)) / th2; }
    M3 K = skew(w), K2 = mmul(K, K), r;
    for (int i = 0; i < 9; i++) r.m[i] = (i % 4 == 0 ? 1.0 : 0.0) + a * K.m[i] + b * K2.m[i];
    return r;
}
static inline void logso3(const M3& R, double* w) {
    const double* m = R.m;
    const double vx = m[7] - m[5], vy = m[2] - m[6], vz = m[3] - m[1];
    const double c = 0.5 * (m[0] + m[4] + m[8] - 1), s = 0.5 * std::sqrt(vx * vx + vy * vy + vz * vz);
    const double th = std::atan2(s, c);
    const double k = s < 1e-6 ? 0.5 + th * th / 12 : 0.5 * th / s;
    w[0] = k * vx; w[1] = k * vy; w[2] = k * vz;
}
static inline void orthonormalize(M3& R) {
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
static bool inv3s(double a, double b, double c, double d, double e, double f, float* o) {
    double A = d * f - e * e, B = c * e - b * f, C = b * e - c * d, D = a * f - c * c, E = b * c - a * e, F = a * d - b * b;
    double det = a * A + b * B + c * C;
    if (!(det > 1e-9)) return false;
    double id = 1.0 / det;
    o[0] = (float)(A * id); o[1] = (float)(B * id); o[2] = (float)(C * id); o[3] = (float)(D * id); o[4] = (float)(E * id); o[5] = (float)(F * id);
    return true;
}
// dot product with 4 partial sums (lanes 0,1 and 2,3 of two f64x2 accumulators); the host build sums in the same order,
// so host replay and wasm stay bit-identical
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif
static inline double dotv(const double* a, const double* b, int n) {
    int k = 0;
#ifdef __wasm_simd128__
    v128_t s0 = wasm_f64x2_splat(0.0), s1 = s0;
    for (; k + 4 <= n; k += 4) {
        s0 = wasm_f64x2_add(s0, wasm_f64x2_mul(wasm_v128_load(a + k), wasm_v128_load(b + k)));
        s1 = wasm_f64x2_add(s1, wasm_f64x2_mul(wasm_v128_load(a + k + 2), wasm_v128_load(b + k + 2)));
    }
    if (k + 2 <= n) { s0 = wasm_f64x2_add(s0, wasm_f64x2_mul(wasm_v128_load(a + k), wasm_v128_load(b + k))); k += 2; }
    double r = (wasm_f64x2_extract_lane(s0, 0) + wasm_f64x2_extract_lane(s1, 0)) + (wasm_f64x2_extract_lane(s0, 1) + wasm_f64x2_extract_lane(s1, 1));
#else
    double s00 = 0, s01 = 0, s10 = 0, s11 = 0;
    for (; k + 4 <= n; k += 4) { s00 = s00 + a[k] * b[k]; s01 = s01 + a[k + 1] * b[k + 1]; s10 = s10 + a[k + 2] * b[k + 2]; s11 = s11 + a[k + 3] * b[k + 3]; }
    if (k + 2 <= n) { s00 = s00 + a[k] * b[k]; s01 = s01 + a[k + 1] * b[k + 1]; k += 2; }
    double r = (s00 + s10) + (s01 + s11);
#endif
    for (; k < n; k++) r += a[k] * b[k];
    return r;
}
// four dot products of rows a, a+LD, a+2LD, a+3LD with b (each row: 2 partial sums, even/odd elements)
template <int LD> static inline void dot4s(const double* a, const double* b, int n, double* o) {
    int k = 0;
#ifdef __wasm_simd128__
    v128_t s0 = wasm_f64x2_splat(0.0), s1 = s0, s2 = s0, s3 = s0;
    for (; k + 4 <= n; k += 4) {
        const double* p = a + k; const double* q = b + k;
        const v128_t b0 = wasm_v128_load(q), b1 = wasm_v128_load(q + 2);
        s0 = wasm_f64x2_add(s0, wasm_f64x2_mul(wasm_v128_load(p), b0));
        s1 = wasm_f64x2_add(s1, wasm_f64x2_mul(wasm_v128_load(p + LD), b0));
        s2 = wasm_f64x2_add(s2, wasm_f64x2_mul(wasm_v128_load(p + 2 * LD), b0));
        s3 = wasm_f64x2_add(s3, wasm_f64x2_mul(wasm_v128_load(p + 3 * LD), b0));
        s0 = wasm_f64x2_add(s0, wasm_f64x2_mul(wasm_v128_load(p + 2), b1));
        s1 = wasm_f64x2_add(s1, wasm_f64x2_mul(wasm_v128_load(p + LD + 2), b1));
        s2 = wasm_f64x2_add(s2, wasm_f64x2_mul(wasm_v128_load(p + 2 * LD + 2), b1));
        s3 = wasm_f64x2_add(s3, wasm_f64x2_mul(wasm_v128_load(p + 3 * LD + 2), b1));
    }
    if (k + 2 <= n) {
        const double* p = a + k; const v128_t b0 = wasm_v128_load(b + k);
        s0 = wasm_f64x2_add(s0, wasm_f64x2_mul(wasm_v128_load(p), b0));
        s1 = wasm_f64x2_add(s1, wasm_f64x2_mul(wasm_v128_load(p + LD), b0));
        s2 = wasm_f64x2_add(s2, wasm_f64x2_mul(wasm_v128_load(p + 2 * LD), b0));
        s3 = wasm_f64x2_add(s3, wasm_f64x2_mul(wasm_v128_load(p + 3 * LD), b0));
        k += 2;
    }
    double r[4] = {wasm_f64x2_extract_lane(s0, 0) + wasm_f64x2_extract_lane(s0, 1), wasm_f64x2_extract_lane(s1, 0) + wasm_f64x2_extract_lane(s1, 1),
                   wasm_f64x2_extract_lane(s2, 0) + wasm_f64x2_extract_lane(s2, 1), wasm_f64x2_extract_lane(s3, 0) + wasm_f64x2_extract_lane(s3, 1)};
#else
    double e[4] = {0, 0, 0, 0}, od[4] = {0, 0, 0, 0};
    for (; k + 2 <= n; k += 2) for (int t = 0; t < 4; t++) { e[t] = e[t] + a[t * LD + k] * b[k]; od[t] = od[t] + a[t * LD + k + 1] * b[k + 1]; }
    double r[4] = {e[0] + od[0], e[1] + od[1], e[2] + od[2], e[3] + od[3]};
#endif
    if (k < n) for (int t = 0; t < 4; t++) r[t] += a[t * LD + k] * b[k];
    for (int t = 0; t < 4; t++) o[t] = r[t];
}
// dense Cholesky with row stride LD, 4 rows per column pass
template <int LD> static bool chol4r(double* A, int n, int j0, int j1) {   // columns j0..j1-1 (left-looking: needs columns < j0 done)
    for (int j = j0; j < j1; j++) {
        double* Aj = A + j * LD;
        const double d0 = Aj[j] - dotv(Aj, Aj, j);
        if (!(d0 > 1e-300)) return false;
        const double d = std::sqrt(d0); Aj[j] = d; const double id = 1.0 / d;
        int i = j + 1;
        for (; i + 4 <= n; i += 4) { double o[4]; double* Ai = A + i * LD; dot4s<LD>(Ai, Aj, j, o); for (int t = 0; t < 4; t++) Ai[t * LD + j] = (Ai[t * LD + j] - o[t]) * id; }
        for (; i < n; i++) { double* Ai = A + i * LD; Ai[j] = (Ai[j] - dotv(Ai, Aj, j)) * id; }
    }
    return true;
}
template <int LD> static bool chol4(double* A, int n) { return chol4r<LD>(A, n, 0, n); }
// dense Cholesky (lower, in place, row stride ld); false if not positive definite
static bool chol(double* A, int n, int ld) {
    for (int j = 0; j < n; j++) {
        double* Aj = A + j * ld;
        const double d0 = Aj[j] - dotv(Aj, Aj, j);
        if (!(d0 > 1e-300)) return false;
        const double d = std::sqrt(d0); Aj[j] = d; const double id = 1.0 / d;
        for (int i = j + 1; i < n; i++) { double* Ai = A + i * ld; Ai[j] = (Ai[j] - dotv(Ai, Aj, j)) * id; }
    }
    return true;
}
// single-precision dot product, 8 partial sums (two f32x4 accumulators); same order on the host
static inline float dotf(const float* a, const float* b, int n) {
    int k = 0;
#ifdef __wasm_simd128__
    v128_t s0 = wasm_f32x4_splat(0.0f), s1 = s0;
    for (; k + 8 <= n; k += 8) {
        s0 = wasm_f32x4_add(s0, wasm_f32x4_mul(wasm_v128_load(a + k), wasm_v128_load(b + k)));
        s1 = wasm_f32x4_add(s1, wasm_f32x4_mul(wasm_v128_load(a + k + 4), wasm_v128_load(b + k + 4)));
    }
    float r = ((wasm_f32x4_extract_lane(s0, 0) + wasm_f32x4_extract_lane(s1, 0)) + (wasm_f32x4_extract_lane(s0, 1) + wasm_f32x4_extract_lane(s1, 1)))
            + ((wasm_f32x4_extract_lane(s0, 2) + wasm_f32x4_extract_lane(s1, 2)) + (wasm_f32x4_extract_lane(s0, 3) + wasm_f32x4_extract_lane(s1, 3)));
#else
    float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (; k + 8 <= n; k += 8) for (int l = 0; l < 8; l++) s[l] = s[l] + a[k + l] * b[k + l];
    float r = ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]));
#endif
    for (; k < n; k++) r += a[k] * b[k];
    return r;
}
static void chol_solve(const double* L, int n, int ld, double* x) {   // x <- (L L^T)^-1 x
    for (int i = 0; i < n; i++) { const double* Li = L + i * ld; x[i] = (x[i] - dotv(Li, x, i)) / Li[i]; }
    for (int i = n - 1; i >= 0; i--) { double s = x[i]; for (int k = i + 1; k < n; k++) s -= L[k * ld + i] * x[k]; x[i] = s / L[i * ld + i]; }
}

// H x = b with H (lower triangle, stride ld) Jacobi-scaled, factored in single precision, then REFINE steps of
// iterative refinement in double. H is left unchanged. false if the float factorization fails.
#ifndef REFINE
#define REFINE 2
#endif
static bool solve_f32(const double* H, int n, int ld, double* x) {
    static float Lf[(2 + 15 * 64) * 64];   // generous; n <= NX
    static double D[256], r[256], y[256];
    for (int i = 0; i < n; i++) { const double d = H[i * ld + i]; if (!(d > 0)) return false; D[i] = 1.0 / std::sqrt(d); }
    const int lf = (n + 3) & ~3;
    for (int i = 0; i < n; i++) { float* li = Lf + i * lf; const double* hi = H + i * ld; for (int j = 0; j <= i; j++) li[j] = (float)(hi[j] * D[i] * D[j]); }
    for (int j = 0; j < n; j++) {
        float* Lj = Lf + j * lf;
        const float d0 = Lj[j] - dotf(Lj, Lj, j);
        if (!(d0 > 1e-30f)) return false;
        const float d = std::sqrt(d0); Lj[j] = d; const float id = 1.0f / d;
        for (int i = j + 1; i < n; i++) { float* Li = Lf + i * lf; Li[j] = (Li[j] - dotf(Li, Lj, j)) * id; }
    }
    auto lsolve = [&](double* v) {   // v <- (L L^T)^-1 v  (scaled system)
        for (int i = 0; i < n; i++) { const float* Li = Lf + i * lf; double s = v[i]; for (int k = 0; k < i; k++) s -= (double)Li[k] * v[k]; v[i] = s / Li[i]; }
        for (int i = n - 1; i >= 0; i--) { double s = v[i]; for (int k = i + 1; k < n; k++) s -= (double)Lf[k * lf + i] * v[k]; v[i] = s / Lf[i * lf + i]; }
    };
    for (int i = 0; i < n; i++) { r[i] = x[i]; y[i] = x[i] * D[i]; }   // r = b (unscaled), y = scaled rhs
    lsolve(y);
    for (int i = 0; i < n; i++) x[i] = y[i] * D[i];
    for (int it = 0; it < REFINE; it++) {
        // residual b - H x with the symmetric lower-triangle H
        for (int i = 0; i < n; i++) y[i] = r[i];
        for (int i = 0; i < n; i++) {
            const double* hi = H + i * ld; double s = 0;
            for (int j = 0; j < i; j++) { s += hi[j] * x[j]; y[j] -= hi[j] * x[i]; }
            y[i] -= s + hi[i] * x[i];
        }
        for (int i = 0; i < n; i++) y[i] *= D[i];
        lsolve(y);
        for (int i = 0; i < n; i++) x[i] += y[i] * D[i];
    }
    return true;
}
// ---------------------------------------------------------------- globals
static int W = 320, H = 240;
static double fx = 120, fy = 120, cx = 159.5, cy = 119.5;
static M3 Rci; static double tci[3];
static const uint8_t* img = nullptr;

#ifndef MAP_OBS_MAX
#define MAP_OBS_MAX 4
#endif
struct Frame {
    int mo_n; int mo_id[MAP_OBS_MAX]; float mo_u[MAP_OBS_MAX], mo_v[MAP_OBS_MAX];   // map observations (vmap.inc)
#if MANHATTAN
    bool mh_on; double mh_dlt, mh_g[3]; M3 mh_R;                                    // Manhattan heading factor
#endif
    M3 R; double p[3], v[3], bg[3], ba[3];
    int serial; double t, span, vsig; bool zv;
    M3 Rl; double pl[3], vl[3], bgl[3], bal[3];   // linearization point of the prior
};
static Frame fr[WIN + 1]; static int nfr = 0;
static double gw[3], gl[3];
struct Pre {
    double T; M3 dR; double dp[3], dv[3]; M3 Jrg, Jpg, Jpa, Jvg, Jva; double bg0[3], ba0[3]; double C[81];
    double I9[81]; bool iok;
};
static Pre pre[WIN + 1];
// the prior (Hp) and the solver system (Hs) are stored as lower triangles; marginalization swaps the two buffers
static double Hbuf0[NX * NX], Hbuf1[NX * NX]; static double* Hp = Hbuf0; static double* Hs = Hbuf1; static double gp[NX];
static double gs[NX], dxs[NX];
static bool marg_now = false;   // this frame marginalized the oldest frame
static int frame_no = 0; static double t_last = 0, frame_dt = 0.05; static bool started = false;
static double w_last[3];
static double sa2 = SIG_ACC * SIG_ACC, sg2 = SIG_GYR * SIG_GYR;
static double imu_prev[6]; static bool imu_prev_ok = false;
static double dpf_lz0 = 0.0;   // ln of the depth prior's centre (DPF_Z0, or adapted: DPF_ADAPT)
static double ie_a2 = 0, ie_g2 = 0, ie_prev[6]; static long ie_n = 0; static bool ie_have = false, ie_bias_done = false;
static double depth_avg = 2.5, v2_avg = 0.25;
static int n_resets = 0, last_reset_frame = -100000;
static long st_cells = 0, st_tri = 0, st_det = 0, st_cfull = 0;
static long st_opt = 0, st_out = 0, st_marg = 0, st_drop = 0, st_fail = 0;
static long st_nb_try = 0, st_nb_ok = 0;
static long st_k_oob = 0, st_k_res = 0, st_k_dup = 0, st_k_beh = 0, st_life = 0, st_nd = 0;   // track deaths (VINS_TRK)
#ifndef __wasm__
static struct TrkStats { ~TrkStats() { if (std::getenv("VINS_TRK")) std::fprintf(stderr, "TRK det %ld oob %ld res %ld dup %ld behind %ld out %ld life %.1f nbtry %ld nbok %ld\n", st_det, st_k_oob, st_k_res, st_k_dup, st_k_beh, st_out, st_nd ? (double)st_life / st_nd : 0.0, st_nb_try, st_nb_ok); } } trk_stats_;
#endif

struct WF {   // window feature
    bool used; int live;          // live track index or -1
    int nobs; int ser[WIN + 1]; float u[WIN + 1], v[WIN + 1], q[WIN + 1];   // q: LK residual of the observation
#if EDGE
    bool edge; float enu[WIN + 1], env[WIN + 1];   // edgelet: unit normal (pixels) of each observation
#endif
    double lam; bool lam_ok;
#if ST_W
    float sw[3];   // symmetric square root of the whitening (s00, s01, s11)
#define SWF(F) (F).sw
#else
#define SWF(F) sw_dummy
#endif
};
static float sw_dummy[3] = {1, 0, 1};
static WF wf[MAX_WF];
struct Feat {
    bool active; int wf;
    float x, y, fdx, fdy;
    float tf[(2 * PR + 1) * (2 * PR + 1)], gxf[(2 * PR + 1) * (2 * PR + 1)], gyf[(2 * PR + 1) * (2 * PR + 1)];
    float Hf[6];
    int born; float hx[16], hy[16]; float lkr;
#ifndef __wasm__
    float lkp;   // previous LK residual (VINS_TRK2)
#endif
#if ST_W
    float st[3];   // template structure tensor (sum gx^2, gx gy, gy^2)
#endif
    float t8[(2 * PR + 1) * 8], gx8[(2 * PR + 1) * 8], gy8[(2 * PR + 1) * 8];
#if EDGE
    bool edge; float enx, eny, gn[(2 * PR + 1) * (2 * PR + 1)], Hn[3];   // edgelet: normal, gradient along it, 2x2 inverse (shift, offset)
#endif   // LK_FAST: template rows padded to 8 (PR = 3)   // pixel position history (ring by frame number) for the stillness test
};
static Feat feats[FEAT_N];
static float fl_u[FEAT_N], fl_v[FEAT_N], fl_du[FEAT_N], fl_dv[FEAT_N], flow[FEAT_N];
static int nflow = 0;
static int cool[GRID_X * GRID_Y], det_rr = 0;
static bool det_obs = true;   // false (PIPE, frame B): new tracks get their first window observation on the next window frame
static M3 Rh[16];   // newest-frame attitude history (ring by frame number)

static int fidx(int serial) { for (int k = 0; k < nfr; k++) if (fr[k].serial == serial) return k; return -1; }

// ---------------------------------------------------------------- image access, templates, LK (as ../msckf)
static inline bool inside(float x, float y, float m) { return x >= m && y >= m && x < W - 1 - m && y < H - 1 - m; }
static inline void grid_sample(float x0, float y0, int n, float* g) {
    const int ix = (int)x0, iy = (int)y0; const float ax = x0 - ix, ay = y0 - iy;
    const float w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
    const uint8_t* base = img + (iy * W + ix) * 3 + 1;
    for (int j = 0; j < n; j++) {
        const uint8_t* p = base + j * 3 * W;
        for (int i = 0; i < n; i++, p += 3) g[j * n + i] = w00 * p[0] + w10 * p[3] + w01 * p[3 * W] + w11 * p[3 * W + 3];
    }
}
static bool cut_template(Feat& f) {
    const int S = 2 * PR + 1, E = S + 2;
    if (!inside(f.x, f.y, 2 * PR + 3)) return false;
    float g[(2 * PR + 3) * (2 * PR + 3)];
    grid_sample(f.x - (PR + 1), f.y - (PR + 1), E, g);
    double a = 0, b = 0, c = 0, d = 0, e = 0;
    for (int j = 0; j < S; j++) for (int i = 0; i < S; i++) {
        const float* q = g + (j + 1) * E + (i + 1);
        float dx = 0.5f * (q[1] - q[-1]), dy = 0.5f * (q[E] - q[-E]);
        f.tf[j * S + i] = q[0]; f.gxf[j * S + i] = dx; f.gyf[j * S + i] = dy;
        a += dx * dx; b += dx * dy; c -= dx; d += dy * dy; e -= dy;
    }
#if EDGE
    if (f.edge) {   // normal = gradient direction (largest eigenvector of the structure tensor); 1-D LK along it
        const double th = 0.5 * std::atan2(2 * b, a - d);
        const double nx = std::cos(th), ny = std::sin(th);
        const double lmax = 0.5 * (a + d) + std::sqrt(0.25 * (a - d) * (a - d) + b * b);
        if (!(lmax > 4.0)) return false;
        f.enx = (float)nx; f.eny = (float)ny;
        double hn = 0, ho = 0;
        for (int k = 0; k < S * S; k++) { const float g = (float)(f.gxf[k] * nx + f.gyf[k] * ny); f.gn[k] = g; hn += g * g; ho -= g; }
        const double N = S * S, det = hn * N - ho * ho; if (!(det > 1e-9)) return false;
        f.Hn[0] = (float)(N / det); f.Hn[1] = (float)(-ho / det); f.Hn[2] = (float)(hn / det);
        return true;
    }
#endif
#if ST_W
    f.st[0] = (float)a; f.st[1] = (float)b; f.st[2] = (float)d;
#endif
#ifndef __wasm__
    if (std::getenv("VINS_ST")) { const double tr = 0.5 * (a + d), dd = std::sqrt(0.25 * (a - d) * (a - d) + b * b); std::fprintf(stderr, "ST %.1f %.1f\n", tr - dd, tr + dd); }
#endif
    if (LK_FAST && PR == 3) for (int j = 0; j < S; j++) for (int i = 0; i < 8; i++) {
        const bool in = i < S; f.t8[j * 8 + i] = in ? f.tf[j * S + i] : 0; f.gx8[j * 8 + i] = in ? f.gxf[j * S + i] : 0; f.gy8[j * 8 + i] = in ? f.gyf[j * S + i] : 0;
    }
    return inv3s(a, b, c, d, e, (double)(S * S), f.Hf);
}
#if EDGE
// LK along the edge normal only (the position along the edge is not observable): unknowns shift s and brightness offset
static float lk_edge(const Feat& f, float& x, float& y, float& off, int iters) {
    const int S = 2 * PR + 1;
    const float* t = f.tf; const float* gn = f.gn; const float* Hn = f.Hn;
    float res = 0;
    for (int it = 0; it < iters; it++) {
        const float x0 = x - PR, y0 = y - PR;
        if (!inside(x0, y0, 1) || !inside(x0 + 2 * PR, y0 + 2 * PR, 1)) return -1;
        const int ix = (int)x0, iy = (int)y0; const float ax = x0 - ix, ay = y0 - iy;
        const float w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
        const uint8_t* base = img + (iy * W + ix) * 3 + 1;
        float bn = 0, bo = 0; res = 0;
        for (int j = 0; j < S; j++) {
            const uint8_t* p = base + j * 3 * W; const int r = j * S;
            for (int i = 0; i < S; i++, p += 3) {
                const float e = w00 * p[0] + w10 * p[3] + w01 * p[3 * W] + w11 * p[3 * W + 3] - t[r + i] - off;
                bn += gn[r + i] * e; bo -= e; res += std::fabs(e);
            }
        }
        const float ds = -(Hn[0] * bn + Hn[1] * bo), doff = -(Hn[1] * bn + Hn[2] * bo);
        x += ds * f.enx; y += ds * f.eny; off += doff;
        if (ds * ds < LK_EPS2) break;
    }
    return res / (S * S);
}
#endif
static float lk(const Feat& f, float& x, float& y, float& off, int iters) {
    const int S = 2 * PR + 1;
    const float* t = f.tf; const float* gx = f.gxf; const float* gy = f.gyf; const float* Hi = f.Hf;
    float res = 0;
    for (int it = 0; it < iters; it++) {
        const float x0 = x - PR, y0 = y - PR;
        if (!inside(x0, y0, 1) || !inside(x0 + 2 * PR, y0 + 2 * PR, 1)) return -1;
        const int ix = (int)x0, iy = (int)y0; const float ax = x0 - ix, ay = y0 - iy;
        const float w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
        const uint8_t* base = img + (iy * W + ix) * 3 + 1;
        float bx = 0, by = 0, bo = 0; res = 0;
        for (int j = 0; j < S; j++) {
            const uint8_t* p = base + j * 3 * W; const int r = j * S;
            for (int i = 0; i < S; i++, p += 3) {
                float e = w00 * p[0] + w10 * p[3] + w01 * p[3 * W] + w11 * p[3 * W + 3] - t[r + i] - off;
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

// LK for PR = 3, four samples at a time: the 8x9 green patch is converted to floats once per iteration, the bilinear
// residual and the sums run on f32x4 (lanes = columns 0-3 and 4-7 of a row; column 7 is padding). The host build
// computes the same lanes in the same order (bit-identical results).
static float lk_fast(const Feat& f, float& x, float& y, float& off, int iters) {
    const float* Hi = f.Hf;
    float res = 0;
    alignas(16) float g[8][12];
    for (int it = 0; it < iters; it++) {
        const float x0 = x - 3, y0 = y - 3;
        if (!inside(x0, y0, 1) || !inside(x0 + 6, y0 + 6, 1)) return -1;
        const int ix = (int)x0, iy = (int)y0; const float ax = x0 - ix, ay = y0 - iy;
        const float w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
        const uint8_t* base = img + (iy * W + ix) * 3 + 1;
        float bx, by, bo;
#ifdef __wasm_simd128__
        for (int j = 0; j < 8; j++) {
            const uint8_t* p = base + j * 3 * W;
            const v128_t lo = wasm_v128_load(p), hi = wasm_v128_load(p + 16);
            const v128_t sg = wasm_i8x16_shuffle(lo, hi, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
            const v128_t h0 = wasm_u16x8_extend_low_u8x16(sg), h1 = wasm_u16x8_extend_high_u8x16(sg);
            wasm_v128_store(g[j], wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(h0)));
            wasm_v128_store(g[j] + 4, wasm_f32x4_convert_u32x4(wasm_u32x4_extend_high_u16x8(h0)));
            wasm_v128_store(g[j] + 8, wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(h1)));
        }
        const v128_t W00 = wasm_f32x4_splat(w00), W10 = wasm_f32x4_splat(w10), W01 = wasm_f32x4_splat(w01), W11 = wasm_f32x4_splat(w11), OFF = wasm_f32x4_splat(off);
        const v128_t M = wasm_f32x4_make(1, 1, 1, 0);
        v128_t BX0 = wasm_f32x4_splat(0), BY0 = BX0, BO0 = BX0, RS0 = BX0, BX1 = BX0, BY1 = BX0, BO1 = BX0, RS1 = BX0;
        for (int j = 0; j < 7; j++) {
            const float* g0 = g[j]; const float* g1 = g[j + 1];
            v128_t e = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(W00, wasm_v128_load(g0)), wasm_f32x4_mul(W10, wasm_v128_load(g0 + 1))),
                                                     wasm_f32x4_mul(W01, wasm_v128_load(g1))), wasm_f32x4_mul(W11, wasm_v128_load(g1 + 1)));
            e = wasm_f32x4_sub(wasm_f32x4_sub(e, wasm_v128_load(f.t8 + 8 * j)), OFF);
            BX0 = wasm_f32x4_add(BX0, wasm_f32x4_mul(wasm_v128_load(f.gx8 + 8 * j), e));
            BY0 = wasm_f32x4_add(BY0, wasm_f32x4_mul(wasm_v128_load(f.gy8 + 8 * j), e));
            BO0 = wasm_f32x4_sub(BO0, e); RS0 = wasm_f32x4_add(RS0, wasm_f32x4_abs(e));
            v128_t e1 = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(W00, wasm_v128_load(g0 + 4)), wasm_f32x4_mul(W10, wasm_v128_load(g0 + 5))),
                                                      wasm_f32x4_mul(W01, wasm_v128_load(g1 + 4))), wasm_f32x4_mul(W11, wasm_v128_load(g1 + 5)));
            e1 = wasm_f32x4_mul(wasm_f32x4_sub(wasm_f32x4_sub(e1, wasm_v128_load(f.t8 + 8 * j + 4)), OFF), M);
            BX1 = wasm_f32x4_add(BX1, wasm_f32x4_mul(wasm_v128_load(f.gx8 + 8 * j + 4), e1));
            BY1 = wasm_f32x4_add(BY1, wasm_f32x4_mul(wasm_v128_load(f.gy8 + 8 * j + 4), e1));
            BO1 = wasm_f32x4_sub(BO1, e1); RS1 = wasm_f32x4_add(RS1, wasm_f32x4_abs(e1));
        }
        auto hsum = [](v128_t a, v128_t b) { const v128_t c = wasm_f32x4_add(a, b);
            return (wasm_f32x4_extract_lane(c, 0) + wasm_f32x4_extract_lane(c, 1)) + (wasm_f32x4_extract_lane(c, 2) + wasm_f32x4_extract_lane(c, 3)); };
        bx = hsum(BX0, BX1); by = hsum(BY0, BY1); bo = hsum(BO0, BO1); res = hsum(RS0, RS1);
#else
        for (int j = 0; j < 8; j++) { const uint8_t* p = base + j * 3 * W; for (int i = 0; i < 9; i++) g[j][i] = (float)p[3 * i]; }
        float BX[8] = {0}, BY[8] = {0}, BO[8] = {0}, RS[8] = {0};
        for (int j = 0; j < 7; j++) for (int l = 0; l < 8; l++) {
            const float* g0 = g[j]; const float* g1 = g[j + 1];
            float e = ((w00 * g0[l] + w10 * g0[l + 1]) + w01 * g1[l]) + w11 * g1[l + 1];
            e = (e - f.t8[8 * j + l]) - off;
            if (l == 7) e = e * 0.0f;
            BX[l] = BX[l] + f.gx8[8 * j + l] * e; BY[l] = BY[l] + f.gy8[8 * j + l] * e; BO[l] = BO[l] - e; RS[l] = RS[l] + std::fabs(e);
        }
        auto hsum = [](const float* a) { const float c0 = a[0] + a[4], c1 = a[1] + a[5], c2 = a[2] + a[6], c3 = a[3] + a[7]; return (c0 + c1) + (c2 + c3); };
        bx = hsum(BX); by = hsum(BY); bo = hsum(BO); res = hsum(RS);
#endif
        float dx = -(Hi[0] * bx + Hi[1] * by + Hi[2] * bo);
        float dy = -(Hi[1] * bx + Hi[3] * by + Hi[4] * bo);
        float doff = -(Hi[2] * bx + Hi[4] * by + Hi[5] * bo);
        x += dx; y += dy; off += doff;
        if (dx * dx + dy * dy < LK_EPS2) break;
    }
    return res / 49;
}

// camera-frame velocity guess from this frame's derotated flow (as ../msckf)
static bool visual_velocity(double dt, double* vc) {
    if (nflow < 8 || dt <= 0) return false;
    double M[9] = {0};
    for (int k = 0; k < nflow; k++) {
        const double f[3] = {fl_u[k], fl_v[k], 1}, g[3] = {fl_u[k] + fl_du[k], fl_v[k] + fl_dv[k], 1};
        const double n[3] = {f[1] * g[2] - f[2] * g[1], f[2] * g[0] - f[0] * g[2], f[0] * g[1] - f[1] * g[0]};
        for (int a = 0; a < 3; a++) for (int c = 0; c < 3; c++) M[3 * a + c] += n[a] * n[c];
    }
    double t[3] = {0, 0, 1};
    for (int a = 0; a < 3; a++) M[4 * a] += 1e-12;
    float Mi[6]; if (!inv3s(M[0], M[1], M[2], M[4], M[5], M[8], Mi)) return false;
    for (int it = 0; it < 30; it++) {
        double y[3] = {Mi[0] * t[0] + Mi[1] * t[1] + Mi[2] * t[2], Mi[1] * t[0] + Mi[3] * t[1] + Mi[4] * t[2], Mi[2] * t[0] + Mi[4] * t[1] + Mi[5] * t[2]};
        double ny = std::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]); if (!(ny > 0)) return false;
        for (int a = 0; a < 3; a++) t[a] = y[a] / ny;
    }
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

static const Frame* trk_pred = nullptr;   // MAH_PRED: predicted pose of the frame being tracked
static bool wf_point(const struct WF& F, double* pw);
static bool pred_pixel(const Feat& f, float& x, float& y) {
    if (!MAH_PRED || !trk_pred || f.wf < 0) return false;
    const WF& F = wf[f.wf]; if (!F.lam_ok || F.nobs < 2) return false;
    double pw[3]; if (!wf_point(F, pw)) return false;
    const Frame& P = *trk_pred;
    M3 Ric = mtr(Rci); double pic[3]; mtv(Rci, tci, pic); for (int i = 0; i < 3; i++) pic[i] = -pic[i];
    const M3 Rwc = mmul(P.R, Ric); double t[3]; mv(P.R, pic, t);
    double d[3] = {pw[0] - P.p[0] - t[0], pw[1] - P.p[1] - t[1], pw[2] - P.p[2] - t[2]}, q[3]; mtv(Rwc, d, q);
    if (q[2] < MIN_DEPTH) return false;
    x = (float)(fx * q[0] / q[2] + cx); y = (float)(fy * q[1] / q[2] + cy);
    return true;
}
static float nb_x[FEAT_N], nb_y[FEAT_N], nb_dx[FEAT_N], nb_dy[FEAT_N]; static int nnb = 0; static bool nb_retry[FEAT_N];
static bool nb_flow(float x, float y, float& dx, float& dy) {   // weighted derotated flow of this frame's established corner tracks
    if (nnb < 3) return false;
    float sw = 0, sx = 0, sy = 0;
    for (int k = 0; k < nnb; k++) { const float ex = nb_x[k] - x, ey = nb_y[k] - y, w = 1.0f / (ex * ex + ey * ey + NB_R * NB_R); sw += w; sx += w * nb_dx[k]; sy += w * nb_dy[k]; }
    dx = sx / sw; dy = sy / sw; return true;
}
static void track(const M3& Rcc) {
    nflow = 0; if (NB_FLOW) nnb = 0;
    for (int pass = 0; pass < (NB_FLOW ? 2 : 1); pass++)
    for (int i = 0; i < FEAT_N; i++) {
        Feat& f = feats[i]; if (!f.active) continue;
        const bool young = NB_FLOW && frame_no - f.born <= 1;
        if (NB_FLOW && pass == 0) { nb_retry[i] = false; if (young) continue; }
        if (NB_FLOW && pass == 1 && !young && !nb_retry[i]) continue;
        double b[3] = {(f.x - cx) / fx, (f.y - cy) / fy, 1}, q[3]; mv(Rcc, b, q);
        bool ok = q[2] >= 0.1;
        float px = 0, py = 0, x = 0, y = 0;
        if (ok) {
            px = (float)(fx * q[0] / q[2] + cx); py = (float)(fy * q[1] / q[2] + cy);
            x = px + f.fdx; y = py + f.fdy; float off = 0;
            if (NB_FLOW && pass == 1) { float ddx, ddy; if (nb_flow(px, py, ddx, ddy)) { x = px + ddx; y = py + ddy; if (nb_retry[i]) st_nb_try++; } else if (nb_retry[i]) { x = px; y = py; } }
            float qx = 0, qy = 0; const bool hp = MAH_PRED && pred_pixel(f, qx, qy) && inside(qx, qy, 4 * PR + 4);
            if (MAH_PRED == 1 && hp) { x = qx; y = qy; }
            if (!inside(x, y, 4 * PR + 4)) { x = px; y = py; }
#if EDGE
            float r = f.edge ? lk_edge(f, x, y, off, LK_FINE) : ((LK_FAST && PR == 3) ? lk_fast(f, x, y, off, LK_FINE) : lk(f, x, y, off, LK_FINE));
#else
            float r = (LK_FAST && PR == 3) ? lk_fast(f, x, y, off, LK_FINE) : lk(f, x, y, off, LK_FINE);
#endif
            if (MAH_PRED == 2 && hp && (r < 0 || std::fabs(x - qx) + std::fabs(y - qy) > MAH_PRED_PX)) {   // second start at the prediction; keep the better fit
                float x2 = qx, y2 = qy, off2 = 0;
#if EDGE
                const float r2 = f.edge ? lk_edge(f, x2, y2, off2, LK_FINE) : ((LK_FAST && PR == 3) ? lk_fast(f, x2, y2, off2, LK_FINE) : lk(f, x2, y2, off2, LK_FINE));
#else
                const float r2 = (LK_FAST && PR == 3) ? lk_fast(f, x2, y2, off2, LK_FINE) : lk(f, x2, y2, off2, LK_FINE);
#endif
                if (r2 >= 0 && (r < 0 || r2 < MAH_PRED_K * r)) { x = x2; y = y2; off = off2; r = r2; }
            }
            if (NB_FLOW >= 2 && pass == 0 && r > MAX_RES) { nb_retry[i] = true; continue; }   // retried from the neighbours' flow in pass 1
            if (NB_FLOW && pass == 1 && nb_retry[i] && r >= 0 && r <= MAX_RES) st_nb_ok++;
#ifndef __wasm__
            f.lkp = f.lkr;
#endif
            ok = r >= 0 && r <= MAX_RES; f.lkr = r;
        }
#ifndef __wasm__
        if (!ok && q[2] >= 0.1 && f.lkr >= 0 && std::getenv("VINS_TRK2")) std::fprintf(stderr, "RD %d %.1f %.1f %d\n", frame_no, f.lkr, f.lkp, frame_no - f.born);
        if (ok && std::getenv("VINS_TRK2")) std::fprintf(stderr, "OK %d %.1f %d\n", frame_no, f.lkr, frame_no - f.born);
#endif
#ifndef __wasm__
        if (!ok) { if (q[2] < 0.1) st_k_beh++; else if (f.lkr < 0) st_k_oob++; else st_k_res++; st_life += frame_no - f.born; st_nd++; }
#endif
        if (!ok) { f.active = false; if (f.wf >= 0) wf[f.wf].live = -1; f.wf = -1; continue; }
        f.fdx = 0.7f * (x - px); f.fdy = 0.7f * (y - py);
        f.x = x; f.y = y;
#if EDGE
        if (NB_FLOW && pass == 0 && !f.edge) { nb_x[nnb] = px; nb_y[nnb] = py; nb_dx[nnb] = x - px; nb_dy[nnb] = y - py; nnb++; }
#else
        if (NB_FLOW && pass == 0) { nb_x[nnb] = px; nb_y[nnb] = py; nb_dx[nnb] = x - px; nb_dy[nnb] = y - py; nnb++; }
#endif
#if EDGE
        if (f.edge) continue;   // edgelets: no 2-D flow (stillness, visual velocity)
#endif
        fl_u[nflow] = (float)(q[0] / q[2]); fl_v[nflow] = (float)(q[1] / q[2]);
        fl_du[nflow] = (float)((x - px) / fx); fl_dv[nflow] = (float)((y - py) / fy);
        flow[nflow++] = std::sqrt((x - px) * (x - px) + (y - py) * (y - py));
    }
    for (int i = 0; i < FEAT_N; i++) if (feats[i].active) for (int j = i + 1; j < FEAT_N; j++) if (feats[j].active) {
        float dx = feats[i].x - feats[j].x, dy = feats[i].y - feats[j].y;
        if (dx * dx + dy * dy < 4.0f) {
#ifndef __wasm__
            st_k_dup++;
#endif
            feats[j].active = false; if (feats[j].wf >= 0) wf[feats[j].wf].live = -1; feats[j].wf = -1; }
    }
}

static bool obs_gate_ok(const WF& F, const Feat& f);
#if EDGE
#define EDGE_SHIFT(F, i) do { (F).enu[i] = (F).enu[(i) + 1]; (F).env[i] = (F).env[(i) + 1]; } while (0)
#else
#define EDGE_SHIFT(F, i) do {} while (0)
#endif
static int wf_alloc(int live) {
    for (int w = 0; w < MAX_WF; w++) if (!wf[w].used) { WF& F = wf[w]; F.used = true; F.live = live; F.nobs = 0; F.lam_ok = false; F.lam = 0;
#if EDGE
        F.edge = live >= 0 && feats[live].edge;
#endif
        if (ST_W && live >= 0) {   // eigen-decomposition of the template structure tensor
#if ST_W
            const float* g = feats[live].st;
#else
            const float g[3] = {0, 0, 0};
#endif
            const double a = g[0], b = g[1], d = g[2];
            const double tr = 0.5 * (a + d), dd = std::sqrt(0.25 * (a - d) * (a - d) + b * b);
            double l1 = (tr + dd) / ST_G0, l2 = (tr - dd) / ST_G0;
            l1 = l1 < ST_MIN ? ST_MIN : (l1 > 1 ? 1 : l1); l2 = l2 < ST_MIN ? ST_MIN : (l2 > 1 ? 1 : l2);
            const double th = 0.5 * std::atan2(2 * b, a - d), c = std::cos(th), sn = std::sin(th), r1 = std::sqrt(l1), r2 = std::sqrt(l2);
            SWF(F)[0] = (float)(r1 * c * c + r2 * sn * sn); SWF(F)[1] = (float)((r1 - r2) * c * sn); SWF(F)[2] = (float)(r1 * sn * sn + r2 * c * c);
        } else if (ST_W) { SWF(F)[0] = 1; SWF(F)[1] = 0; SWF(F)[2] = 1; }
        return w; }
    return -1;
}
static void wf_add_obs(int w, const Feat& f) {
    WF& F = wf[w]; if (F.nobs >= WIN + 1) return;
    F.ser[F.nobs] = frame_no; F.u[F.nobs] = (float)((f.x - cx) / fx); F.v[F.nobs] = (float)((f.y - cy) / fy); F.q[F.nobs] = f.lkr;
#if EDGE
    F.enu[F.nobs] = f.enx; F.env[F.nobs] = f.eny;
#endif
    F.nobs++;
}

static void detect() {
    const int cw = W / GRID_X, chh = H / GRID_Y;
    bool occ[GRID_X * GRID_Y] = {false};
    int nact = 0, nedge = 0;   // corner tracks, edgelets
    for (int i = 0; i < FEAT_N; i++) if (feats[i].active) {
#if EDGE
        if (feats[i].edge) nedge++; else
#endif
        nact++;
        int gx = (int)(feats[i].x / cw), gy = (int)(feats[i].y / chh);
        if (gx >= 0 && gx < GRID_X && gy >= 0 && gy < GRID_Y) occ[gy * GRID_X + gx] = true;
    }
    int scored = 0;
    int cell_cap = marg_now && DET_MARG_CELLS > 0 && DET_MARG_CELLS < DETECT_CELLS ? DET_MARG_CELLS : DETECT_CELLS;   // marginalizing frames are the heaviest
#if PIPE
    if (PIPE_DET == 3) cell_cap = (DETECT_CELLS + PIPE - 1) / PIPE;
#endif
    const bool edge_ok0 = EDGE && nact < EDGE_WHEN;
    for (int kk = 0; kk < GRID_X * GRID_Y && scored < cell_cap && (nact < MAX_FEATS || (edge_ok0 && nedge < EDGE_MAX)); kk++) {
        int cell = (det_rr + kk) % (GRID_X * GRID_Y);
        if (occ[cell]) continue;
        scored++; st_cells++;
        int gx = cell % GRID_X, gy = cell / GRID_X;
        int x0 = gx * cw, y0 = gy * chh;
        if (cool[cell] > frame_no) continue;
        bool edge_only = nact >= MAX_FEATS;   // corners full (only with EDGE) or the cell is too faint for corners
        {
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
            if (mx < ((MS_LOW_N > 0 && nact < MS_LOW_N) ? PRETEST_LOW : PRETEST)) {
                if (!(edge_ok0 && nedge < EDGE_MAX && mx >= EDGE_PRETEST)) { cool[cell] = frame_no + COOL; continue; }
                edge_only = true;
            }
        }
        if (PIPE_SCHED) st_cfull++;
        float best = edge_only ? 1e30f : ((MS_LOW_N > 0 && nact < MS_LOW_N) ? MIN_SCORE_LOW : MIN_SCORE); int bx = -1, by = -1;
        float ebest = EDGE_MIN; int ex = -1, ey = -1;
        for (int y = y0 + DET_STEP / 2; y < y0 + chh; y += DET_STEP) for (int x = x0 + DET_STEP / 2; x < x0 + cw; x += DET_STEP) {
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
            float lmin = (tr - dd) * (1.0f / 36.0f);
            if (lmin > best) { best = lmin; bx = x; by = y; }
#if EDGE
            const float lmax = (tr + dd) * (1.0f / 36.0f);
            if (lmax > ebest && lmin < EDGE_RATIO * lmax) { ebest = lmax; ex = x; ey = y; }
#endif
        }
        bool as_edge = false;
        if (bx < 0) {
            if (!(edge_ok0 && nedge < EDGE_MAX && ex >= 0)) { cool[cell] = frame_no + COOL; continue; }
            as_edge = true; bx = ex; by = ey;
        }
        int slot = -1; for (int i = 0; i < FEAT_N; i++) if (!feats[i].active) { slot = i; break; }
        if (slot < 0) break;
        Feat& f = feats[slot];
        f.x = (float)bx; f.y = (float)by; f.fdx = f.fdy = 0; f.lkr = 0;
#if EDGE
        f.edge = as_edge;
#else
        (void)as_edge;
#endif
        if (!cut_template(f)) continue;
        f.active = true; if (as_edge) nedge++; else nact++; f.born = frame_no; st_det++; f.hx[frame_no & 15] = f.x; f.hy[frame_no & 15] = f.y;
        f.wf = det_obs ? wf_alloc(slot) : -1;
        if (f.wf >= 0) wf_add_obs(f.wf, f);
    }
    det_rr = (det_rr + ((PIPE && PIPE_DET == 3) ? (7 + PIPE - 1) / PIPE : 7)) % (GRID_X * GRID_Y);
}

// median disparity (px) between each live feature's first window observation (rotated into the newest frame) and its
// current position, over features seen for at least ZW_MIN_SPAN frames; -1 if too few
static float window_disparity() {
    static float d[FEAT_N]; int nd = 0;
    const M3& Rn = fr[nfr - 1].R;
    for (int i = 0; i < FEAT_N; i++) {
        const Feat& f = feats[i]; if (!f.active) continue;
#if EDGE
        if (f.edge) continue;
#endif
        const int age = frame_no - f.born; if (age < ZW_MIN_SPAN) continue;
        const int old = frame_no - (age < ZW_LAG ? age : ZW_LAG), r = old & 15;
        M3 Rcc = mmul(mmul(Rci, mmul(mtr(Rn), Rh[r])), mtr(Rci));
        double b[3] = {(f.hx[r] - cx) / fx, (f.hy[r] - cy) / fy, 1}, q[3]; mv(Rcc, b, q);
        if (q[2] < 0.1) continue;
        const double du = fx * (q[0] / q[2]) + cx - f.x, dv = fy * (q[1] / q[2]) + cy - f.y;
        d[nd++] = (float)std::sqrt(du * du + dv * dv);
    }
    if (nd < 6) return -1;
    for (int i = 1; i < nd; i++) { float v = d[i]; int j = i - 1; while (j >= 0 && d[j] > v) { d[j + 1] = d[j]; j--; } d[j + 1] = v; }
    return d[(nd * ZW_PCT) / 100];
}
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
    static float r[FEAT_N];
    for (int k = 0; k < nflow; k++) {
        const double u = fl_u[k], v = fl_v[k];
        const double eu = fl_du[k] - (u * v * d[0] - (1 + u * u) * d[1] + v * d[2]), ev = fl_dv[k] - ((1 + v * v) * d[0] - u * v * d[1] - u * d[2]);
        r[k] = (float)(fx * std::sqrt(eu * eu + ev * ev));
    }
    for (int i = 1; i < nflow; i++) { float v = r[i]; int j = i - 1; while (j >= 0 && r[j] > v) { r[j + 1] = r[j]; j--; } r[j + 1] = v; }
    return r[nflow / 2];
}

// ---------------------------------------------------------------- IMU preintegration
static void pre_reset(Pre& P, const double* bg0, const double* ba0) {
    P.T = 0; P.dR = eye3(); for (int i = 0; i < 3; i++) { P.dp[i] = P.dv[i] = 0; P.bg0[i] = bg0[i]; P.ba0[i] = ba0[i]; }
    M3 Z = {{0}}; P.Jrg = P.Jpg = P.Jpa = P.Jvg = P.Jva = Z;
    for (int i = 0; i < 81; i++) P.C[i] = 0;
    P.iok = false;
}
// C <- A C A^T + Q for the 9x9 [theta, p, v] covariance
static void cov9(double* C, const double* A, const double* Q) {
    double B[81];
    for (int i = 0; i < 9; i++) for (int j = 0; j < 9; j++) { double s = 0; for (int k = 0; k < 9; k++) s += A[9 * i + k] * C[9 * k + j]; B[9 * i + j] = s; }
    for (int i = 0; i < 9; i++) for (int j = 0; j < 9; j++) { double s = Q ? Q[9 * i + j] : 0; for (int k = 0; k < 9; k++) s += B[9 * i + k] * A[9 * j + k]; C[9 * i + j] = s; }
}
static void put3(double* A, int r0, int c0, const M3& m) { for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) A[9 * (r0 + i) + c0 + j] = m.m[3 * i + j]; }
// C <- A C A^T + Q for A = [[Et 0 0]; [Ap I dt I]; [Av 0 I]] (blocks theta, p, v), Q diagonal blocks qth, qpp, qpv, qvv
static inline void get3(const double* C, int r0, int c0, M3& m) { for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) m.m[3 * i + j] = C[9 * (r0 + i) + c0 + j]; }
static inline void set3(double* C, int r0, int c0, const M3& m) { for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { C[9 * (r0 + i) + c0 + j] = m.m[3 * i + j]; C[9 * (c0 + j) + r0 + i] = m.m[3 * i + j]; } }
static void cov_prop(double* C, const M3& Et, const M3& Ap, const M3& Av, double dt, double qth, double qpp, double qpv, double qvv) {
    M3 X, Cp, Cv, Pp, Pv, Vv; get3(C, 0, 0, X); get3(C, 0, 3, Cp); get3(C, 0, 6, Cv); get3(C, 3, 3, Pp); get3(C, 3, 6, Pv); get3(C, 6, 6, Vv);
    M3 Y; for (int i = 0; i < 9; i++) Y.m[i] = Cp.m[i] + dt * Cv.m[i];                 // C_theta,p + dt C_theta,v
    const M3 ApT = mtr(Ap), AvT = mtr(Av), EtT = mtr(Et);
    const M3 XAp = mmul(X, ApT), XAv = mmul(X, AvT);
    M3 tp = XAp, tv = XAv; for (int i = 0; i < 9; i++) { tp.m[i] += Y.m[i]; tv.m[i] += Cv.m[i]; }
    M3 thth = mmul(mmul(Et, X), EtT), thp = mmul(Et, tp), thv = mmul(Et, tv);
    const M3 ApXAp = mmul(Ap, XAp), ApY = mmul(Ap, Y), ApXAv = mmul(Ap, XAv), ApCv = mmul(Ap, Cv), YtAvT = mmul(mtr(Y), AvT);
    const M3 AvXAv = mmul(Av, XAv), AvCv = mmul(Av, Cv);
    M3 pp, pv, vv;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        const int k = 3 * i + j, kt = 3 * j + i;
        pp.m[k] = ApXAp.m[k] + ApY.m[k] + ApY.m[kt] + Pp.m[k] + dt * (Pv.m[k] + Pv.m[kt]) + dt * dt * Vv.m[k];
        pv.m[k] = ApXAv.m[k] + ApCv.m[k] + YtAvT.m[k] + Pv.m[k] + dt * Vv.m[k];
        vv.m[k] = AvXAv.m[k] + AvCv.m[k] + AvCv.m[kt] + Vv.m[k];
    }
    for (int i = 0; i < 3; i++) { thth.m[4 * i] += qth; pp.m[4 * i] += qpp; pv.m[4 * i] += qpv; vv.m[4 * i] += qvv; }
    set3(C, 0, 0, thth); set3(C, 0, 3, thp); set3(C, 0, 6, thv); set3(C, 3, 3, pp); set3(C, 3, 6, pv); set3(C, 6, 6, vv);
}
static void pre_add(Pre& P, double dt, const double* x) {
    double a[3] = {x[0] - P.ba0[0], x[1] - P.ba0[1], x[2] - P.ba0[2]};
    double w[3] = {(x[3] - P.bg0[0]) * dt, (x[4] - P.bg0[1]) * dt, (x[5] - P.bg0[2]) * dt};
    const double dt2 = dt * dt;
    M3 Ra = mmul(P.dR, skew(a)), RaJ = mmul(Ra, P.Jrg);
    for (int i = 0; i < 9; i++) {
        P.Jpa.m[i] += P.Jva.m[i] * dt - 0.5 * P.dR.m[i] * dt2;
        P.Jpg.m[i] += P.Jvg.m[i] * dt - 0.5 * RaJ.m[i] * dt2;
        P.Jva.m[i] -= P.dR.m[i] * dt;
        P.Jvg.m[i] -= RaJ.m[i] * dt;
    }
    double A[81] = {0};
    M3 E = expso3(w), Et = mtr(E);
    put3(A, 0, 0, Et);
    for (int i = 0; i < 9; i++) { Ra.m[i] *= -1; }
    M3 Ap = Ra, Av = Ra; for (int i = 0; i < 9; i++) { Ap.m[i] *= 0.5 * dt2; Av.m[i] *= dt; }
    put3(A, 3, 0, Ap); put3(A, 6, 0, Av);
    for (int i = 0; i < 3; i++) { A[9 * (3 + i) + 3 + i] = 1; A[9 * (3 + i) + 6 + i] = dt; A[9 * (6 + i) + 6 + i] = 1; }
    const double DTS = 0.005, qa = sa2 * DTS, qg = sg2 * DTS;
    double Q[81] = {0};
    for (int i = 0; i < 3; i++) {
        Q[9 * i + i] = qg * dt;
        Q[9 * (3 + i) + 3 + i] = qa * dt2 * dt / 3; Q[9 * (3 + i) + 6 + i] = Q[9 * (6 + i) + 3 + i] = qa * dt2 / 2;
        Q[9 * (6 + i) + 6 + i] = qa * dt;
    }
    if (COV_FAST) {
        M3 Ap2, Av2; get3(A, 3, 0, Ap2); get3(A, 6, 0, Av2);
        cov_prop(P.C, Et, Ap2, Av2, dt, qg * dt, qa * dt2 * dt / 3, qa * dt2 / 2, qa * dt);
    } else cov9(P.C, A, Q);
    double Rav[3]; mv(P.dR, a, Rav);
    for (int i = 0; i < 3; i++) { P.dp[i] += P.dv[i] * dt + 0.5 * Rav[i] * dt2; P.dv[i] += Rav[i] * dt; }
    M3 Jr = eye3(); { M3 S = skew(w); for (int i = 0; i < 9; i++) Jr.m[i] -= 0.5 * S.m[i]; }
    M3 EJ = mmul(Et, P.Jrg); for (int i = 0; i < 9; i++) P.Jrg.m[i] = EJ.m[i] - Jr.m[i] * dt;
    P.dR = mmul(P.dR, E); orthonormalize(P.dR);
    P.T += dt; P.iok = false;
}
// b's measurements re-expressed at a's bias linearization point (first order), then c = a followed by b
static void pre_compose(Pre& a, Pre b) {
    double dg[3], da[3];
    for (int i = 0; i < 3; i++) { dg[i] = a.bg0[i] - b.bg0[i]; da[i] = a.ba0[i] - b.ba0[i]; }
    { double t[3]; mv(b.Jrg, dg, t); b.dR = mmul(b.dR, expso3(t)); orthonormalize(b.dR); }
    { double t1[3], t2[3]; mv(b.Jvg, dg, t1); mv(b.Jva, da, t2); for (int i = 0; i < 3; i++) b.dv[i] += t1[i] + t2[i];
      mv(b.Jpg, dg, t1); mv(b.Jpa, da, t2); for (int i = 0; i < 3; i++) b.dp[i] += t1[i] + t2[i]; }
    const M3 Ra = a.dR, RbT = mtr(b.dR);
    const M3 Sdv = skew(b.dv), Sdp = skew(b.dp);
    const M3 RaSdv = mmul(Ra, Sdv), RaSdp = mmul(Ra, Sdp);
    const double Tb = b.T;
    // covariance
    double A[81] = {0}, B[81] = {0}, Cb[81];
    put3(A, 0, 0, RbT);
    { M3 m = RaSdp; for (int i = 0; i < 9; i++) m.m[i] = -m.m[i]; put3(A, 3, 0, m); }
    { M3 m = RaSdv; for (int i = 0; i < 9; i++) m.m[i] = -m.m[i]; put3(A, 6, 0, m); }
    for (int i = 0; i < 3; i++) { A[9 * (3 + i) + 3 + i] = 1; A[9 * (3 + i) + 6 + i] = Tb; A[9 * (6 + i) + 6 + i] = 1; B[9 * i + i] = 1; }
    put3(B, 3, 3, Ra); put3(B, 6, 6, Ra);
    for (int i = 0; i < 81; i++) Cb[i] = b.C[i];
    cov9(Cb, B, nullptr);
    cov9(a.C, A, Cb);
    // Jacobians (use a's old values)
    const M3 RaJrg = mmul(RaSdv, a.Jrg), RaJrgp = mmul(RaSdp, a.Jrg);
    const M3 RaJvgb = mmul(Ra, b.Jvg), RaJvab = mmul(Ra, b.Jva), RaJpgb = mmul(Ra, b.Jpg), RaJpab = mmul(Ra, b.Jpa);
    const M3 Jrg = mmul(RbT, a.Jrg);
    for (int i = 0; i < 9; i++) {
        const double jpg = a.Jpg.m[i] + a.Jvg.m[i] * Tb + RaJpgb.m[i] - RaJrgp.m[i];
        const double jpa = a.Jpa.m[i] + a.Jva.m[i] * Tb + RaJpab.m[i];
        const double jvg = a.Jvg.m[i] + RaJvgb.m[i] - RaJrg.m[i];
        const double jva = a.Jva.m[i] + RaJvab.m[i];
        a.Jpg.m[i] = jpg; a.Jpa.m[i] = jpa; a.Jvg.m[i] = jvg; a.Jva.m[i] = jva;
        a.Jrg.m[i] = Jrg.m[i] + b.Jrg.m[i];
    }
    double t[3]; mv(Ra, b.dp, t);
    for (int i = 0; i < 3; i++) a.dp[i] += a.dv[i] * Tb + t[i];
    mv(Ra, b.dv, t); for (int i = 0; i < 3; i++) a.dv[i] += t[i];
    a.dR = mmul(Ra, b.dR); orthonormalize(a.dR);
    a.T += Tb; a.iok = false;
}
static void pre_info(Pre& P) {
    if (P.iok) return;
    double L[81]; for (int i = 0; i < 81; i++) L[i] = P.C[i];
    for (int i = 0; i < 9; i++) L[10 * i] += 1e-14;
    if (!chol(L, 9, 9)) { for (int i = 0; i < 81; i++) P.I9[i] = 0; for (int i = 0; i < 9; i++) P.I9[10 * i] = 1e6; P.iok = true; return; }
    for (int j = 0; j < 9; j++) { double e[9] = {0}; e[j] = 1; chol_solve(L, 9, 9, e); for (int i = 0; i < 9; i++) P.I9[9 * i + j] = e[i]; }
    P.iok = true;
}

// ---------------------------------------------------------------- cost terms (normal equations H dx = -g)
static void gbasis(double* b1, double* b2) {
    const double n = std::sqrt(gw[0] * gw[0] + gw[1] * gw[1] + gw[2] * gw[2]);
    const double e[3] = {gw[0] / n, gw[1] / n, gw[2] / n};
    double a[3] = {1, 0, 0}; if (std::fabs(e[0]) > 0.9) { a[0] = 0; a[1] = 1; }
    const double d = a[0] * e[0] + a[1] * e[1] + a[2] * e[2];
    for (int i = 0; i < 3; i++) b1[i] = a[i] - d * e[i];
    const double m = std::sqrt(b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2]); for (int i = 0; i < 3; i++) b1[i] /= m;
    b2[0] = e[1] * b1[2] - e[2] * b1[1]; b2[1] = e[2] * b1[0] - e[0] * b1[2]; b2[2] = e[0] * b1[1] - e[1] * b1[0];
}
#define HH(i, j) Hs[(i) * NX + (j)]
static bool want_H = true;   // false: builders accumulate the gradient only (chord iterations reuse the factorization)
static inline int FB(int k) { return G0 + NF_ * k; }
static inline int BGI(int k) { return SHARED_BIAS ? 2 : FB(k) + 9; }
static inline int BAI(int k) { return SHARED_BIAS ? 5 : FB(k) + 12; }
static inline int NSYS() { return G0 + NF_ * nfr; }
static double bgl_s[3], bal_s[3];   // SHARED_BIAS: prior linearization point of the biases

static void prior_dx(double* d) {   // x - x_lin in the prior's coordinates
    double b1[3], b2[3]; gbasis(b1, b2);
    const double n = std::sqrt(gw[0] * gw[0] + gw[1] * gw[1] + gw[2] * gw[2]), nl = std::sqrt(gl[0] * gl[0] + gl[1] * gl[1] + gl[2] * gl[2]);
    double dg[3]; for (int i = 0; i < 3; i++) dg[i] = gw[i] / n - gl[i] / nl;
    d[0] = b1[0] * dg[0] + b1[1] * dg[1] + b1[2] * dg[2]; d[1] = b2[0] * dg[0] + b2[1] * dg[1] + b2[2] * dg[2];
    if (SHARED_BIAS && nfr > 0) for (int i = 0; i < 3; i++) { d[2 + i] = fr[0].bg[i] - bgl_s[i]; d[5 + i] = fr[0].ba[i] - bal_s[i]; }
    if (SCALE_ST) d[SI] = sst - sstl;
    for (int k = 0; k < nfr; k++) {
        const Frame& F = fr[k]; double* o = d + FB(k);
        logso3(mmul(mtr(F.Rl), F.R), o);
        for (int i = 0; i < 3; i++) { o[3 + i] = F.p[i] - F.pl[i]; o[6 + i] = F.v[i] - F.vl[i]; if (!SHARED_BIAS) { o[9 + i] = F.bg[i] - F.bgl[i]; o[12 + i] = F.ba[i] - F.bal[i]; } }
    }
}
static void add_prior() {
    const int n = NSYS();
    static double d[NX]; prior_dx(d);
    for (int i = 0; i < n; i++) {
        const double* hp = Hp + i * NX; double* h = Hs + i * NX;
        if (want_H) for (int j = 0; j <= i; j++) h[j] += hp[j];
        gs[i] += gp[i] + dotv(hp, d, i + 1);   // symmetric product from the lower triangle: row part ...
        const double di = d[i]; for (int j = 0; j < i; j++) gs[j] += hp[j] * di;   // ... and column part
    }
}

static inline int col_index(int k, int c) { return c < 9 ? FB(k) + c : (c < 12 ? BGI(k) + c - 9 : BAI(k) + c - 12); }
static bool dbg_cap = false; static double dbg_r[15], dbg_J[15][32];
// IMU factor between frames k-1 and k
static double imu_cost = 0;
static void add_imu_full(int k);
#if SHARED_BIAS
// IMU factor between frames k-1 and k, shared biases: 9 residuals (theta, p, v) on 26 columns
// [theta_i p_i v_i | theta_j p_j v_j | b_g b_a | gravity]; the lower triangle of H only
static void add_imu(int k) {
    if (dbg_cap) { add_imu_full(k); return; }
    Pre& P = pre[k]; pre_info(P);
    const Frame& A = fr[k - 1]; const Frame& B = fr[k];
    double dbg[3], dba[3]; for (int i = 0; i < 3; i++) { dbg[i] = A.bg[i] - P.bg0[i]; dba[i] = A.ba[i] - P.ba0[i]; }
    double t[3]; mv(P.Jrg, dbg, t);
    const M3 dRc = mmul(P.dR, expso3(t));
    double dvc[3], dpc[3], t1[3], t2[3];
    mv(P.Jvg, dbg, t1); mv(P.Jva, dba, t2); for (int i = 0; i < 3; i++) dvc[i] = P.dv[i] + t1[i] + t2[i];
    mv(P.Jpg, dbg, t1); mv(P.Jpa, dba, t2); for (int i = 0; i < 3; i++) dpc[i] = P.dp[i] + t1[i] + t2[i];
    const double T = P.T;
    const M3 RiT = mtr(A.R);
    const M3 E = mmul(mmul(mtr(dRc), RiT), B.R);
    double r[9]; logso3(E, r);
    double xp[3], xv[3], ip[3], iv[3];
    for (int i = 0; i < 3; i++) { xp[i] = B.p[i] - A.p[i] - A.v[i] * T - 0.5 * gw[i] * T * T; xv[i] = B.v[i] - A.v[i] - gw[i] * T; }
    mv(RiT, xp, ip); mv(RiT, xv, iv);
    for (int i = 0; i < 3; i++) { r[3 + i] = ip[i] - dpc[i]; r[6 + i] = iv[i] - dvc[i]; }
    // J^T stored by column: JT[c][a], a = residual row (padded to 10)
    static double JT[26][10], MT[26][10];
    std::memset(JT, 0, sizeof JT);
    auto put = [&](int r0, int c0, const M3& m, double sc) { for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) JT[c0 + b][r0 + a] = sc * m.m[3 * a + b]; };
    M3 Jri = eye3(); { M3 S = skew(r); for (int i = 0; i < 9; i++) Jri.m[i] += 0.5 * S.m[i]; }
    put(0, 0, mmul(Jri, mmul(mtr(B.R), A.R)), -1);
    put(0, 9, Jri, 1);
    put(0, 18, mmul(Jri, mmul(mtr(E), P.Jrg)), -1);
    put(3, 0, skew(ip), 1); put(3, 3, RiT, -1); put(3, 6, RiT, -T); put(3, 12, RiT, 1); put(3, 18, P.Jpg, -1); put(3, 21, P.Jpa, -1);
    put(6, 0, skew(iv), 1); put(6, 6, RiT, -1); put(6, 15, RiT, 1); put(6, 18, P.Jvg, -1); put(6, 21, P.Jva, -1);
    {
        double b1[3], b2[3]; gbasis(b1, b2);
        double c1[3], c2[3]; mv(RiT, b1, c1); mv(RiT, b2, c2);
        for (int i = 0; i < 3; i++) {
            JT[24][3 + i] = -0.5 * T * T * 9.81 * c1[i]; JT[25][3 + i] = -0.5 * T * T * 9.81 * c2[i];
            JT[24][6 + i] = -T * 9.81 * c1[i]; JT[25][6 + i] = -T * 9.81 * c2[i];
        }
    }
    double I9p[9][10]; for (int a = 0; a < 9; a++) { for (int b = 0; b < 9; b++) I9p[a][b] = P.I9[9 * a + b]; I9p[a][9] = 0; }
    for (int c = 0; c < 26; c++) { for (int a = 0; a < 9; a++) MT[c][a] = dotv(I9p[a], JT[c], 10); MT[c][9] = 0; }
    double rr[10]; for (int a = 0; a < 9; a++) rr[a] = r[a]; rr[9] = 0;
    { double cost = 0; for (int a = 0; a < 9; a++) cost += r[a] * dotv(I9p[a], rr, 10); imu_cost += cost; }
    int gi[26];
    for (int c = 0; c < 9; c++) { gi[c] = FB(k - 1) + c; gi[9 + c] = FB(k) + c; }
    for (int c = 0; c < 6; c++) gi[18 + c] = 2 + c;
    gi[24] = 0; gi[25] = 1;
    for (int c = 0; c < 26; c++) {
        gs[gi[c]] += dotv(MT[c], rr, 10);
        if (want_H) for (int d = 0; d <= c; d++) {
            const double v = dotv(JT[c], MT[d], 10);
            const int rI = gi[c], cI = gi[d];
            if (rI >= cI) Hs[rI * NX + cI] += v; else Hs[cI * NX + rI] += v;
        }
    }
}
#else
static void add_imu(int k) { add_imu_full(k); }
#endif
static void add_imu_full(int k) {
    Pre& P = pre[k]; pre_info(P);
    const Frame& A = fr[k - 1]; const Frame& B = fr[k];
    double dbg[3], dba[3]; for (int i = 0; i < 3; i++) { dbg[i] = A.bg[i] - P.bg0[i]; dba[i] = A.ba[i] - P.ba0[i]; }
    double t[3]; mv(P.Jrg, dbg, t);
    const M3 dRc = mmul(P.dR, expso3(t));
    double dvc[3], dpc[3], t1[3], t2[3];
    mv(P.Jvg, dbg, t1); mv(P.Jva, dba, t2); for (int i = 0; i < 3; i++) dvc[i] = P.dv[i] + t1[i] + t2[i];
    mv(P.Jpg, dbg, t1); mv(P.Jpa, dba, t2); for (int i = 0; i < 3; i++) dpc[i] = P.dp[i] + t1[i] + t2[i];
    const double T = P.T;
    const M3 RiT = mtr(A.R);
    const M3 E = mmul(mmul(mtr(dRc), RiT), B.R);
    double r[15]; logso3(E, r);
    double xp[3], xv[3], ip[3], iv[3];
    for (int i = 0; i < 3; i++) { xp[i] = B.p[i] - A.p[i] - A.v[i] * T - 0.5 * gw[i] * T * T; xv[i] = B.v[i] - A.v[i] - gw[i] * T; }
    mv(RiT, xp, ip); mv(RiT, xv, iv);
    for (int i = 0; i < 3; i++) { r[3 + i] = ip[i] - dpc[i]; r[6 + i] = iv[i] - dvc[i]; r[9 + i] = B.bg[i] - A.bg[i]; r[12 + i] = B.ba[i] - A.ba[i]; }
    // Jacobian 15 x 32: [frame i (15) | frame j (15) | gravity (2)]
    static double J[15][32];
    std::memset(J, 0, sizeof J);
    auto put = [&](int r0, int c0, const M3& m, double s) { for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) J[r0 + a][c0 + b] = s * m.m[3 * a + b]; };
    M3 Jri = eye3(); { M3 S = skew(r); for (int i = 0; i < 9; i++) Jri.m[i] += 0.5 * S.m[i]; }
    put(0, 0, mmul(Jri, mmul(mtr(B.R), A.R)), -1);
    put(0, 15, Jri, 1);
    put(0, 9, mmul(Jri, mmul(mtr(E), P.Jrg)), -1);
    put(3, 0, skew(ip), 1); put(3, 3, RiT, -1); put(3, 6, RiT, -T); put(3, 9, P.Jpg, -1); put(3, 12, P.Jpa, -1); put(3, 18, RiT, 1);
    put(6, 0, skew(iv), 1); put(6, 6, RiT, -1); put(6, 9, P.Jvg, -1); put(6, 12, P.Jva, -1); put(6, 21, RiT, 1);
    for (int i = 0; i < 3; i++) { J[9 + i][9 + i] = -1; J[9 + i][24 + i] = 1; J[12 + i][12 + i] = -1; J[12 + i][27 + i] = 1; }
    {
        double b1[3], b2[3]; gbasis(b1, b2);
        double c1[3], c2[3]; mv(RiT, b1, c1); mv(RiT, b2, c2);
        for (int i = 0; i < 3; i++) {
            J[3 + i][30] = -0.5 * T * T * 9.81 * c1[i]; J[3 + i][31] = -0.5 * T * T * 9.81 * c2[i];
            J[6 + i][30] = -T * 9.81 * c1[i]; J[6 + i][31] = -T * 9.81 * c2[i];
        }
    }
#ifndef __wasm__
    if (dbg_cap) { std::memcpy(dbg_r, r, sizeof dbg_r); std::memcpy(dbg_J, J, sizeof dbg_J); }
#endif
    // M = Info J ; H += J^T M ; g += M^T r
    static double M[15][32];
    const double ibg = SHARED_BIAS ? 0.0 : 1.0 / (SIG_BGW * SIG_BGW * T), iba = SHARED_BIAS ? 0.0 : 1.0 / (SIG_BAW * SIG_BAW * T);
    for (int a = 0; a < 9; a++) for (int c = 0; c < 32; c++) { double s = 0; for (int b = 0; b < 9; b++) s += P.I9[9 * a + b] * J[b][c]; M[a][c] = s; }
    for (int a = 9; a < 12; a++) for (int c = 0; c < 32; c++) M[a][c] = ibg * J[a][c];
    for (int a = 12; a < 15; a++) for (int c = 0; c < 32; c++) M[a][c] = iba * J[a][c];
    int gi[32]; for (int c = 0; c < 15; c++) { gi[c] = col_index(k - 1, c); gi[15 + c] = col_index(k, c); } gi[30] = 0; gi[31] = 1;
    double cost = 0;
    for (int a = 0; a < 15; a++) { double s = 0; for (int b = 0; b < 15; b++) s += (a < 9 && b < 9 ? P.I9[9 * a + b] : (a == b ? (a < 12 ? ibg : iba) : 0)) * r[b]; cost += r[a] * s; }
    imu_cost += cost;
    static double JT[32][16], MT[32][16];   // transposed, padded rows for dot products
    for (int c = 0; c < 32; c++) { for (int a = 0; a < 15; a++) { JT[c][a] = J[a][c]; MT[c][a] = M[a][c]; } JT[c][15] = MT[c][15] = 0; }
    double rr[16]; for (int a = 0; a < 15; a++) rr[a] = r[a]; rr[15] = 0;
    for (int c = 0; c < 32; c++) {
        gs[gi[c]] += dotv(MT[c], rr, 16);
        for (int d = c; d < 32; d++) {
            const double v = dotv(JT[c], MT[d], 16);
            Hs[gi[c] * NX + gi[d]] += v; if (d != c) Hs[gi[d] * NX + gi[c]] += v;
        }
    }
}

static void add_s_pin() {   // SCALE_ST + S_V0: speed-gated pin of s (router_dp); solves only, never marginalized
    if (SCALE_ST && S_V0 > 0) {
        const double vr = std::sqrt(v2_avg), w = vr <= S_V0 ? 1.0 : (vr >= S_V1 ? 0.0 : (S_V1 - vr) / (S_V1 - S_V0 + 1e-9));
        if (w > 0) { const double info = w / (S_PIN * S_PIN); if (want_H) HH(SI, SI) += info; gs[SI] += info * sst; }
    }
}
static void add_unary() {
    for (int k = 0; k < nfr; k++) {
        const Frame& F = fr[k];
        double info = 0;
        if (VPRIOR_SIG > 0) info += (F.span / VP_DT) / (F.vsig * F.vsig);
        if (F.zv) info += 1.0 / (ZV_SIG * ZV_SIG);
        if (info <= 0) continue;
        const int o = FB(k) + 6;
        for (int i = 0; i < 3; i++) { if (want_H) HH(o + i, o + i) += info; gs[o + i] += info * F.v[i]; }
    }
}

// feature: anchored inverse depth; contributions with the inverse depth eliminated (Schur complement)
// marginalization cache (MARG_CACHE): reduced blocks of the oldest frame's features from the solve of a marginalizing frame
#ifndef MC_N
#define MC_N 24
#endif
static bool mc_on = false, mc_valid = false; static int mc_n = 0, mc_w[MC_N], mc_m[MC_N], mc_fl[MC_N][WIN + 1];
static double mc_H[MC_N][6 * (WIN + 1)][6 * (WIN + 1)], mc_g[MC_N][6 * (WIN + 1)];
static bool dbg_noschur = false, dbg_nofeat = false, dbg_noimu = false;
#ifndef PROF_NOVIS
#define PROF_NOVIS 0
#endif
static bool novis = PROF_NOVIS || std::getenv("VINS_NOVIS") != nullptr; static double fb_cost = 0;
static int fb_nf[MAX_WF], fb_fl[MAX_WF][WIN + 1];
static double vw_k = 1.0, vw_e2 = 0; static long vw_n = 0;   // VW_DYN: current visual weight, squared-error sum and count of this solve
static double fb_hxf[MAX_WF][6 * (WIN + 1)], fb_hff[MAX_WF], fb_gf[MAX_WF], fb_err[MAX_WF], fb_hls[MAX_WF];
static bool fb_on[MAX_WF];
static void cam_pose(int k, M3& Rwc, double* pwc) {
    M3 Ric = mtr(Rci); double pic[3]; mtv(Rci, tci, pic); for (int i = 0; i < 3; i++) pic[i] = -pic[i];
    Rwc = mmul(fr[k].R, Ric); double t[3]; mv(fr[k].R, pic, t);
    for (int i = 0; i < 3; i++) pwc[i] = fr[k].p[i] + t[i];
}
static bool wf_point(const WF& F, double* pw) {
    const int a = fidx(F.ser[0]); if (a < 0) return false;
    M3 Ra; double pa[3]; cam_pose(a, Ra, pa);
    double b[3] = {F.u[0] / F.lam, F.v[0] / F.lam, 1.0 / F.lam}, t[3]; mv(Ra, b, t);
    for (int i = 0; i < 3; i++) pw[i] = pa[i] + t[i];
    return true;
}
static int obs_sel(const WF& F, int* sel) {
    const int m = F.nobs;
    if (OBS_MAX <= 1 || m <= OBS_MAX) { for (int o = 0; o < m; o++) sel[o] = o; return m; }
    for (int o = 0; o < OBS_MAX; o++) sel[o] = (o * (m - 1) + (OBS_MAX - 1) / 2) / (OBS_MAX - 1);
    sel[OBS_MAX - 1] = m - 1; sel[0] = 0;
    return OBS_MAX;
}
// inverse depth refined with the poses held fixed (variable projection: depths at their conditional optimum)
static bool refine_lam(WF& F, int iters) {
    const int a = fidx(F.ser[0]); if (a < 0) return false;
    const Frame& FA = fr[a];
    const double b[3] = {F.u[0], F.v[0], 1};
    double cb[3]; mtv(Rci, b, cb); double rb[3]; mv(FA.R, cb, rb);   // world direction of the anchor bearing (IMU-frame scaled)
    double tc[3]; mtv(Rci, tci, tc); double rt[3]; mv(FA.R, tc, rt);
    int sel[WIN + 1]; const int ms = obs_sel(F, sel);
    for (int it = 0; it < iters; it++) {
        const double il = 1.0 / F.lam;
        double h2 = 0, g = 0;
        for (int oo = 1; oo < ms; oo++) {
            const int o = sel[oo];
            const int j = fidx(F.ser[o]); if (j < 0) return false;
            const Frame& FJ = fr[j];
            double d[3], y[3], pc[3], dd[3], yd[3], pcd[3];
            for (int i = 0; i < 3; i++) { d[i] = rb[i] * il - rt[i] + FA.p[i] - FJ.p[i]; dd[i] = -rb[i] * il * il; }
            mtv(FJ.R, d, y); mv(Rci, y, pc); for (int i = 0; i < 3; i++) pc[i] += tci[i];
            mtv(FJ.R, dd, yd); mv(Rci, yd, pcd);
            if (pc[2] < MIN_DEPTH) return false;
            const double iz = 1.0 / pc[2], u = pc[0] * iz, v = pc[1] * iz;
            double e0 = fx * (u - F.u[o]), e1 = fy * (v - F.v[o]);
            double j0 = fx * iz * (pcd[0] - u * pcd[2]), j1 = fy * iz * (pcd[1] - v * pcd[2]);
#if EDGE
            if (F.edge) { const double nu = F.enu[o], nv = F.env[o]; e0 = nu * e0 + nv * e1; e1 = 0; j0 = nu * j0 + nv * j1; j1 = 0; } else
#endif
            if (ST_W) { const double t0 = SWF(F)[0] * e0 + SWF(F)[1] * e1, t1 = SWF(F)[1] * e0 + SWF(F)[2] * e1, k0 = SWF(F)[0] * j0 + SWF(F)[1] * j1, k1 = SWF(F)[1] * j0 + SWF(F)[2] * j1; e0 = t0; e1 = t1; j0 = k0; j1 = k1; }
            const double en = std::sqrt(e0 * e0 + e1 * e1), wh = en / SIG_PX <= HUBER ? 1.0 : HUBER * SIG_PX / en;
            h2 += wh * (j0 * j0 + j1 * j1); g += wh * (j0 * e0 + j1 * e1);
        }
        if (DPF_SIG > 0) { const double r = (std::log(F.lam) + DPF_C) / DPF_SIG, J = 1.0 / (F.lam * DPF_SIG); h2 += J * J; g += J * r; }
        if (!(h2 > 1e-12)) return true;
        double nl = F.lam - g / h2;
        if (nl < 0.5 * F.lam) nl = 0.5 * F.lam; if (nl > 2 * F.lam) nl = 2 * F.lam;
        F.lam = nl;
    }
    return F.lam > 1.0 / MAX_DEPTH && F.lam < 1.0 / MIN_DEPTH;
}
static bool add_feature(int w) {
    WF& F = wf[w]; fb_on[w] = false;
    const int a = fidx(F.ser[0]); if (a < 0) return false;
    if (VARPRO > 0 && !dbg_noschur && !refine_lam(F, VARPRO)) return false;
    int sel[WIN + 1]; const int m = obs_sel(F, sel);
    // blocks of the feature's 6m x 6m pose Hessian: anchor-anchor, anchor-k, k-k (no other blocks are nonzero)
    static double Haa[6][6], Hak[WIN + 1][6][6], Hkk[WIN + 1][6][6];
    for (int p = 0; p < 6; p++) for (int q = 0; q < 6; q++) Haa[p][q] = 0;
    double gl_[6 * (WIN + 1)], hxf[6 * (WIN + 1)], hff = 0, gf = 0, err = 0, cost_f = 0, hls = 0;
    for (int i = 0; i < 6 * m; i++) gl_[i] = hxf[i] = 0;
    const Frame& FA = fr[a];
    const double lam = F.lam, il = 1.0 / lam;
    const double b[3] = {F.u[0], F.v[0], 1};
    double xa[3] = {b[0] * il - tci[0], b[1] * il - tci[1], b[2] * il - tci[2]}, xi[3]; mtv(Rci, xa, xi);   // point in IMU-a
    double pwv[3]; mv(FA.R, xi, pwv); for (int i = 0; i < 3; i++) pwv[i] += FA.p[i];
    const M3 RaSx = mmul(FA.R, skew(xi));
    double cb[3]; mtv(Rci, b, cb); double dl[3]; mv(FA.R, cb, dl); for (int i = 0; i < 3; i++) dl[i] *= -il * il;   // d pw / d lam
    fb_fl[w][0] = a;
    for (int o = 1; o < m; o++) {
        const int q0 = sel[o];
        const int j = fidx(F.ser[q0]); if (j < 0) return false;
        fb_fl[w][o] = j;
        const Frame& FJ = fr[j];
        double d[3] = {pwv[0] - FJ.p[0], pwv[1] - FJ.p[1], pwv[2] - FJ.p[2]}, y[3], pc[3]; mtv(FJ.R, d, y); mv(Rci, y, pc);
        for (int i = 0; i < 3; i++) pc[i] += tci[i];
        if (pc[2] < MIN_DEPTH) return false;
        const double iz = 1.0 / pc[2], u = pc[0] * iz, v = pc[1] * iz;
        double e[2] = {fx * (u - F.u[q0]), fy * (v - F.v[q0])};
#if EDGE
        double Jp[2][3] = {{fx * iz, 0, -fx * u * iz}, {0, fy * iz, -fy * v * iz}};
        if (F.edge) {   // distance across the edge: project the residual and its Jacobian on the observed normal
            const double nu = F.enu[q0], nv = F.env[q0];
            e[0] = nu * e[0] + nv * e[1]; e[1] = 0;
            for (int c = 0; c < 3; c++) { Jp[0][c] = nu * Jp[0][c] + nv * Jp[1][c]; Jp[1][c] = 0; }
        }
#endif
        err += std::sqrt(e[0] * e[0] + e[1] * e[1]);
#if !EDGE
        double Jp[2][3] = {{fx * iz, 0, -fx * u * iz}, {0, fy * iz, -fy * v * iz}};
#endif
#if EDGE
        if (ST_W && !F.edge) {
#else
        if (ST_W) {
#endif
            const double t0 = SWF(F)[0] * e[0] + SWF(F)[1] * e[1], t1 = SWF(F)[1] * e[0] + SWF(F)[2] * e[1]; e[0] = t0; e[1] = t1;
            for (int c = 0; c < 3; c++) { const double k0 = SWF(F)[0] * Jp[0][c] + SWF(F)[1] * Jp[1][c], k1 = SWF(F)[1] * Jp[0][c] + SWF(F)[2] * Jp[1][c]; Jp[0][c] = k0; Jp[1][c] = k1; }
        }
        const double en = std::sqrt(e[0] * e[0] + e[1] * e[1]);
        cost_f += 0.5 * (e[0] * e[0] + e[1] * e[1]) / (SIG_PX * SIG_PX);
        const double sq = LK_R0 > 0 ? SIG_PX * std::sqrt(1.0 + (F.q[q0] / LK_R0) * (F.q[q0] / LK_R0)) : SIG_PX;   // weaker for a poor LK fit
        const double wh = en / sq <= HUBER ? 1.0 : HUBER * sq / en;
        const double info = (VW_DYN > 0 ? vw_k : 1.0) * wh / (sq * sq);
        if (VW_DYN > 0) { vw_e2 += (e[0] * e[0] + e[1] * e[1]) / (SIG_PX * SIG_PX); vw_n += 1; }
        double G[2][3], GR[2][3];
        for (int r = 0; r < 2; r++) for (int c = 0; c < 3; c++) G[r][c] = Jp[r][0] * Rci.m[c] + Jp[r][1] * Rci.m[3 + c] + Jp[r][2] * Rci.m[6 + c];
        for (int r = 0; r < 2; r++) for (int c = 0; c < 3; c++) GR[r][c] = G[r][0] * FJ.R.m[3 * c] + G[r][1] * FJ.R.m[3 * c + 1] + G[r][2] * FJ.R.m[3 * c + 2];
        const M3 Sy = skew(y);
        double Ja[2][6], Jj[2][6], Jl[2];
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 3; c++) {
                Jj[r][c] = G[r][0] * Sy.m[c] + G[r][1] * Sy.m[3 + c] + G[r][2] * Sy.m[6 + c];
                Jj[r][3 + c] = -GR[r][c];
                Ja[r][c] = -(GR[r][0] * RaSx.m[c] + GR[r][1] * RaSx.m[3 + c] + GR[r][2] * RaSx.m[6 + c]);
                Ja[r][3 + c] = GR[r][c];
            }
            Jl[r] = GR[r][0] * dl[0] + GR[r][1] * dl[1] + GR[r][2] * dl[2];
        }
        const int oj = 6 * o;
        double (*hak)[6] = Hak[o]; double (*hkk)[6] = Hkk[o];
        for (int p = 0; p < 6; p++) {
            const double a0 = info * Ja[0][p], a1 = info * Ja[1][p], j0 = info * Jj[0][p], j1 = info * Jj[1][p];
            for (int q = 0; q < 6; q++) {
                Haa[p][q] += a0 * Ja[0][q] + a1 * Ja[1][q];
                hak[p][q] = a0 * Jj[0][q] + a1 * Jj[1][q];
                hkk[p][q] = j0 * Jj[0][q] + j1 * Jj[1][q];
            }
            gl_[p] += a0 * e[0] + a1 * e[1]; gl_[oj + p] += j0 * e[0] + j1 * e[1];
            hxf[p] += a0 * Jl[0] + a1 * Jl[1]; hxf[oj + p] += j0 * Jl[0] + j1 * Jl[1];
        }
        hff += info * (Jl[0] * Jl[0] + Jl[1] * Jl[1]); gf += info * (Jl[0] * e[0] + Jl[1] * e[1]);
    }
    fb_err[w] = err / (m - 1);
    if (DPF_SIG > 0) {   // log-depth prior
        const double r = (std::log(lam) + DPF_C) / DPF_SIG, J = 1.0 / (lam * DPF_SIG);
        hff += J * J; gf += J * r; cost_f += 0.5 * r * r;
        if (SCALE_ST) { hls = J / DPF_SIG; gs[SI] += r / DPF_SIG; if (want_H) HH(SI, SI) += 1.0 / (DPF_SIG * DPF_SIG); }
    }
    fb_cost += cost_f;
    hff += 1e-6;
    const double ih = dbg_noschur ? 0.0 : 1.0 / hff;
    int mc = -1;
    if (MARG_CACHE && mc_on && want_H && a == 0 && mc_n < MC_N) { mc = mc_n++; mc_w[mc] = w; mc_m[mc] = m; for (int q = 0; q < m; q++) mc_fl[mc][q] = fb_fl[w][q]; }
    // scatter the lower triangle of (blocks - hxf hxf^T / hff), and gl - hxf gf / hff
    for (int s1 = 0; s1 < m; s1++) {
        const int g1 = FB(fb_fl[w][s1]);
        for (int p = 0; p < 6; p++) { const double gv = gl_[6 * s1 + p] - hxf[6 * s1 + p] * gf * ih; gs[g1 + p] += gv; if (mc >= 0) mc_g[mc][6 * s1 + p] = gv; }
        if (want_H) for (int s2 = 0; s2 < m; s2++) {
            const int g2 = FB(fb_fl[w][s2]);
            if (g2 > g1) continue;                     // lower triangle only (frames are in index order)
            double (*blk)[6] = nullptr; bool tr = false;
            if (s1 == s2) blk = s1 == 0 ? Haa : Hkk[s1];
            else if (s1 == 0) blk = Hak[s2];           // rows anchor, cols s2
            else if (s2 == 0) { blk = Hak[s1]; tr = true; }
            for (int p = 0; p < 6; p++) {
                double* h = Hs + (g1 + p) * NX + g2; const double hp = hxf[6 * s1 + p] * ih;
                const int qmax = g1 == g2 ? p + 1 : 6;
                for (int q = 0; q < qmax; q++) {
                    double v = -hp * hxf[6 * s2 + q];
                    if (blk) v += tr ? blk[q][p] : blk[p][q];
                    h[q] += v;
                    if (mc >= 0) mc_H[mc][6 * s1 + p][6 * s2 + q] = v;
                }
            }
        }
    }
    if (SCALE_ST) {   // Schur terms of s (lambda couples to s only through the depth prior)
        gs[SI] -= hls * gf * ih;
        if (want_H) { HH(SI, SI) -= hls * hls * ih;
            for (int s1 = 0; s1 < m; s1++) { const int g1 = FB(fb_fl[w][s1]); for (int p = 0; p < 6; p++) HH(g1 + p, SI) -= hxf[6 * s1 + p] * hls * ih; } }
    }
    fb_nf[w] = m; fb_hff[w] = hff; fb_gf[w] = gf; if (SCALE_ST) fb_hls[w] = hls;
    for (int i = 0; i < 6 * m; i++) fb_hxf[w][i] = hxf[i];
    fb_on[w] = true;
    return true;
}
static bool wf_usable(const WF& F) { return F.used && F.lam_ok && F.nobs >= 2; }

// ---------------------------------------------------------------- triangulation (inverse depth along the anchor bearing)
static bool triangulate(WF& F) {
#if EDGE
    if (F.edge) return false;   // the along-edge position is not measured: start from the depth prior (DPF_INIT)
#endif
    const int m = F.nobs; const int a = fidx(F.ser[0]); if (a < 0) return false;
    M3 Ra; double pa[3]; cam_pose(a, Ra, pa);
    M3 Rr[WIN + 1]; double tr[WIN + 1][3];
    for (int o = 0; o < m; o++) {
        const int k = fidx(F.ser[o]); if (k < 0) return false;
        M3 Rw; double pwc[3]; cam_pose(k, Rw, pwc);
        Rr[o] = mmul(mtr(Rw), Ra);
        double d[3] = {pa[0] - pwc[0], pa[1] - pwc[1], pa[2] - pwc[2]}; mtv(Rw, d, tr[o]);
    }
    const double b0[3] = {F.u[0], F.v[0], 1}, bl[3] = {F.u[m - 1], F.v[m - 1], 1}; double rb[3];
    mv(Rr[m - 1], b0, rb);
    const double n0 = std::sqrt(rb[0] * rb[0] + rb[1] * rb[1] + rb[2] * rb[2]), nl = std::sqrt(bl[0] * bl[0] + bl[1] * bl[1] + bl[2] * bl[2]);
    const double cosang = (rb[0] * bl[0] + rb[1] * bl[1] + rb[2] * bl[2]) / (n0 * nl);
    if (cosang > std::cos(MIN_PARALLAX)) return false;
    const double* t = tr[m - 1];
    const double a11 = n0 * n0, a12 = -(rb[0] * bl[0] + rb[1] * bl[1] + rb[2] * bl[2]), a22 = nl * nl;
    const double c1 = -(rb[0] * t[0] + rb[1] * t[1] + rb[2] * t[2]), c2 = bl[0] * t[0] + bl[1] * t[1] + bl[2] * t[2];
    const double det = a11 * a22 - a12 * a12; if (std::fabs(det) < 1e-12) return false;
    const double lam0 = (c1 * a22 - a12 * c2) / det;
    if (!(lam0 > MIN_DEPTH && lam0 < MAX_DEPTH)) return false;
    double rho = 1.0 / lam0;
    for (int it = 0; it < 5; it++) {
        double h2 = 0, g = 0;
        for (int o = 1; o < m; o++) {
            const double* R = Rr[o].m;
            double hb[3] = {R[0] * b0[0] + R[1] * b0[1] + R[2], R[3] * b0[0] + R[4] * b0[1] + R[5], R[6] * b0[0] + R[7] * b0[1] + R[8]};
            double h[3] = {hb[0] + rho * tr[o][0], hb[1] + rho * tr[o][1], hb[2] + rho * tr[o][2]};
            if (h[2] <= 1e-6) return false;
            const double iz = 1.0 / h[2];
            const double e0 = h[0] * iz - F.u[o], e1 = h[1] * iz - F.v[o];
            const double j0 = iz * (tr[o][0] - h[0] * iz * tr[o][2]), j1 = iz * (tr[o][1] - h[1] * iz * tr[o][2]);
            h2 += j0 * j0 + j1 * j1; g += j0 * e0 + j1 * e1;
        }
        if (!(h2 > 1e-12)) return false;
        const double d = -g / h2; rho += d;
        if (std::fabs(d) < 1e-4 * std::fabs(rho)) break;
    }
    if (!(rho > 1.0 / MAX_DEPTH && rho < 1.0 / MIN_DEPTH)) return false;
    for (int o = 1; o < m; o++) {
        const double* R = Rr[o].m;
        double h[3] = {R[0] * b0[0] + R[1] * b0[1] + R[2] + rho * tr[o][0], R[3] * b0[0] + R[4] * b0[1] + R[5] + rho * tr[o][1], R[6] * b0[0] + R[7] * b0[1] + R[8] + rho * tr[o][2]};
        if (h[2] <= 1e-6) return false;
        const double eu = (h[0] / h[2] - F.u[o]) * fx, ev = (h[1] / h[2] - F.v[o]) * fy;
        if (eu * eu + ev * ev > TRI_MAX_PX * TRI_MAX_PX) return false;
    }
    F.lam = rho; F.lam_ok = true;
    depth_avg = 0.98 * depth_avg + 0.02 * (1.0 / rho);
    return true;
}

static bool obs_gate_ok(const WF& F, const Feat& f) {   // newest window frame = predicted pose of this frame
    if (!(OBS_GATE > 0) || !F.lam_ok || F.nobs < 2) return true;
    double pw[3]; if (!wf_point(F, pw)) return true;
    M3 Rw; double pc[3]; cam_pose(nfr - 1, Rw, pc);
    double d[3] = {pw[0] - pc[0], pw[1] - pc[1], pw[2] - pc[2]}, q[3]; mtv(Rw, d, q);
    if (q[2] < MIN_DEPTH) return false;
    double eu = fx * q[0] / q[2] + cx - f.x, ev = fy * q[1] / q[2] + cy - f.y;
#if EDGE
    if (F.edge) { const double e = eu * f.enx + ev * f.eny; eu = e; ev = 0; }
#endif
    return eu * eu + ev * ev <= OBS_GATE * OBS_GATE;
}
// VINS-Mah gate. The newest frame is the last block of the solve, so its marginal information is L_NN L_NN^T (last
// 9x9 block of the Cholesky factor): no extra solves.
static double mah_S[81]; static bool mah_ok = false; static int mah_ser = -1;
static long st_mah_test = 0, st_mah_rej = 0;
static void mah_store(int n) {
    if (!MAH) return;
    const int o = FB(nfr - 1); if (o + 9 > n) { mah_ok = false; return; }
    double Lm[81], A[81];
    for (int i = 0; i < 9; i++) for (int j = 0; j < 9; j++) Lm[9 * i + j] = j <= i ? Hs[(o + i) * NX + o + j] : 0;
    for (int i = 0; i < 9; i++) for (int j = 0; j < 9; j++) { double t = 0; for (int k = 0; k < 9; k++) t += Lm[9 * i + k] * Lm[9 * j + k]; A[9 * i + j] = t; }
    if (!chol(A, 9, 9)) { mah_ok = false; return; }
    for (int c = 0; c < 9; c++) { double col[9] = {0}; col[c] = 1; chol_solve(A, 9, 9, col); for (int r = 0; r < 9; r++) mah_S[9 * r + c] = col[r]; }
    mah_ok = true; mah_ser = fr[nfr - 1].serial;
}
static bool mah_gate_ok(int w, const Feat& f) {   // newest window frame = this frame (predicted), previous one = the solved frame
    const WF& F = wf[w];
    if (!mah_ok || nfr < 2 || fr[nfr - 2].serial != mah_ser || !F.lam_ok || F.nobs < 2) return true;
    const int a = fidx(F.ser[0]); if (a < 0) return true;
    const Frame& N = fr[nfr - 2]; const Frame& FJ = fr[nfr - 1]; const Pre& P = pre[nfr - 1];
    // predicted pose covariance (theta, p) of the new frame: A Sigma_N A^T + preintegration noise
    double Am[6][9] = {{0}};
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Am[i][j] = P.dR.m[3 * j + i];   // dR^T
    {   double rdp[3]; mv(N.R, P.dp, rdp); const M3 S = skew(rdp);
        for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) Am[3 + i][j] = -S.m[3 * i + j]; Am[3 + i][3 + i] = 1; Am[3 + i][6 + i] = P.T; } }
    double AS[6][9], C6[6][6];
    for (int i = 0; i < 6; i++) for (int j = 0; j < 9; j++) { double t = 0; for (int k = 0; k < 9; k++) t += Am[i][k] * mah_S[9 * k + j]; AS[i][j] = t; }
    for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) { double t = 0; for (int k = 0; k < 9; k++) t += AS[i][k] * Am[j][k]; C6[i][j] = t; }
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {   // preintegration noise: rotation in the body frame, position rotated to world
        C6[i][j] += P.C[9 * i + j];
        double t = 0; for (int k = 0; k < 3; k++) for (int l = 0; l < 3; l++) t += N.R.m[3 * i + k] * P.C[9 * (3 + k) + 3 + l] * N.R.m[3 * j + l];
        C6[3 + i][3 + j] += t;
    }
    // projection, Jacobians (as add_feature)
    const Frame& FA = fr[a];
    const double lam = F.lam, il = 1.0 / lam, b[3] = {F.u[0], F.v[0], 1};
    double xa[3] = {b[0] * il - tci[0], b[1] * il - tci[1], b[2] * il - tci[2]}, xi[3]; mtv(Rci, xa, xi);
    double pwv[3]; mv(FA.R, xi, pwv); for (int i = 0; i < 3; i++) pwv[i] += FA.p[i];
    double cb[3]; mtv(Rci, b, cb); double dl[3]; mv(FA.R, cb, dl); for (int i = 0; i < 3; i++) dl[i] *= -il * il;
    double d[3] = {pwv[0] - FJ.p[0], pwv[1] - FJ.p[1], pwv[2] - FJ.p[2]}, y[3], pc[3]; mtv(FJ.R, d, y); mv(Rci, y, pc);
    for (int i = 0; i < 3; i++) pc[i] += tci[i];
    if (pc[2] < MIN_DEPTH) return false;
    const double iz = 1.0 / pc[2], u = pc[0] * iz, v = pc[1] * iz;
    double e[2] = {fx * u + cx - f.x, fy * v + cy - f.y};
    double Jp[2][3] = {{fx * iz, 0, -fx * u * iz}, {0, fy * iz, -fy * v * iz}};
    int rows = 2;
#if EDGE
    if (F.edge) { const double nu = f.enx, nv = f.eny; e[0] = nu * e[0] + nv * e[1]; for (int c = 0; c < 3; c++) Jp[0][c] = nu * Jp[0][c] + nv * Jp[1][c]; rows = 1; }
#endif
    double G[2][3], GR[2][3], J[2][6], Jl[2];
    for (int r = 0; r < rows; r++) for (int c = 0; c < 3; c++) G[r][c] = Jp[r][0] * Rci.m[c] + Jp[r][1] * Rci.m[3 + c] + Jp[r][2] * Rci.m[6 + c];
    for (int r = 0; r < rows; r++) for (int c = 0; c < 3; c++) GR[r][c] = G[r][0] * FJ.R.m[3 * c] + G[r][1] * FJ.R.m[3 * c + 1] + G[r][2] * FJ.R.m[3 * c + 2];
    const M3 Sy = skew(y);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < 3; c++) { J[r][c] = G[r][0] * Sy.m[c] + G[r][1] * Sy.m[3 + c] + G[r][2] * Sy.m[6 + c]; J[r][3 + c] = -GR[r][c]; }
        Jl[r] = GR[r][0] * dl[0] + GR[r][1] * dl[1] + GR[r][2] * dl[2];
    }
    // inverse-depth variance: from the last solve if the feature was in it, else the depth prior
    double sl2 = (lam * DPF_SIG) * (lam * DPF_SIG);
    if (fb_on[w] && fb_hff[w] > 0) { const double s2 = 1.0 / fb_hff[w]; if (s2 < sl2) sl2 = s2; }
    double Sm[2][2];
    for (int r = 0; r < rows; r++) for (int q = 0; q < rows; q++) {
        double t = 0; for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) t += J[r][i] * C6[i][j] * J[q][j];
        Sm[r][q] = t + sl2 * Jl[r] * Jl[q] + (r == q ? MAH_SIG * MAH_SIG : 0);
    }
    st_mah_test++;
    if (rows == 1) { const bool ok1 = e[0] * e[0] / Sm[0][0] <= MAH_CHI1; st_mah_rej += !ok1; return ok1; }
    const double det = Sm[0][0] * Sm[1][1] - Sm[0][1] * Sm[1][0]; if (!(det > 0)) return true;
    const double m2 = (Sm[1][1] * e[0] * e[0] - 2 * Sm[0][1] * e[0] * e[1] + Sm[0][0] * e[1] * e[1]) / det;
    st_mah_rej += m2 > MAH_CHI;
    return m2 <= MAH_CHI;
}
// ---------------------------------------------------------------- solve
static void apply_dx(const double* dx) {
    double b1[3], b2[3]; gbasis(b1, b2);
    for (int i = 0; i < 3; i++) gw[i] += 9.81 * (b1[i] * dx[0] + b2[i] * dx[1]);
    { const double n = std::sqrt(gw[0] * gw[0] + gw[1] * gw[1] + gw[2] * gw[2]); for (int i = 0; i < 3; i++) gw[i] *= 9.81 / n; }
#ifndef __wasm__
    if (SCALE_ST && std::getenv("VINS_SST")) std::fprintf(stderr, "SST %d %.3f %.3f\n", frame_no, sst, dx[SI]);
#endif
    if (SCALE_ST) { double t = dx[SI]; if (t > 0.5) t = 0.5; if (t < -0.5) t = -0.5; sst += t; if (sst > 1.6) sst = 1.6; if (sst < -1.0) sst = -1.0; }
    for (int k = 0; k < nfr; k++) {
        Frame& F = fr[k]; const double* d = dx + FB(k);
        F.R = mmul(F.R, expso3(d)); orthonormalize(F.R);
        for (int i = 0; i < 3; i++) { F.p[i] += d[3 + i]; F.v[i] += d[6 + i]; F.bg[i] += dx[BGI(k) + i]; F.ba[i] += dx[BAI(k) + i]; }
    }
}
#include "vmap.inc"
#if MANHATTAN
// ---------------------------------------------------------------- Manhattan-world heading (MANHATTAN), ported from ../msckf
// Horizontal edges vote for the dominant axis modulo 90 deg (4th power of the unit direction). Every MH_EVERY frames the
// vote is compared with a reference taken early in the flight; the difference (a yaw error about gravity) becomes a
// factor on the last keyframe (never dropped, only marginalized).
static double mh_C = 0, mh_S = 0, mh_W = 0, mh_ref = 0, mh_e1[3], mh_e2[3]; static bool mh_ref_ok = false, mh_basis = false;
static long st_mh = 0;
static void manhattan_step() {
    if (!MANHATTAN || nfr < 2) return;
    if (v2_avg > MH_VRUN * MH_VRUN) { mh_C = mh_S = mh_W = 0; return; }
    double gn = std::sqrt(gw[0] * gw[0] + gw[1] * gw[1] + gw[2] * gw[2]); if (!(gn > 0)) return;
    const double g[3] = {gw[0] / gn, gw[1] / gn, gw[2] / gn};
    if (!mh_basis) {
        double e[3] = {1, 0, 0}; const double dp = e[0] * g[0] + e[1] * g[1] + e[2] * g[2];
        for (int i = 0; i < 3; i++) e[i] -= dp * g[i];
        const double en = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); for (int i = 0; i < 3; i++) mh_e1[i] = e[i] / en;
        mh_e2[0] = g[1] * mh_e1[2] - g[2] * mh_e1[1]; mh_e2[1] = g[2] * mh_e1[0] - g[0] * mh_e1[2]; mh_e2[2] = g[0] * mh_e1[1] - g[1] * mh_e1[0];
        mh_basis = true;
    }
    const M3 M = mmul(fr[nfr - 1].R, mtr(Rci));   // world <- camera (newest frame)
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
            double nv[3]; mv(M, nc, nv);
            const double nn = std::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]); if (!(nn > 0)) continue;
            const double ng = (nv[0] * g[0] + nv[1] * g[1] + nv[2] * g[2]) / nn;
            if (std::fabs(ng) < MH_VERT) continue;
            const double d[3] = {nv[1] * g[2] - nv[2] * g[1], nv[2] * g[0] - nv[0] * g[2], nv[0] * g[1] - nv[1] * g[0]};
            const double a = d[0] * mh_e1[0] + d[1] * mh_e1[1] + d[2] * mh_e1[2], b = d[0] * mh_e2[0] + d[1] * mh_e2[1] + d[2] * mh_e2[2];
            const double r2 = a * a + b * b; if (!(r2 > 1e-12)) continue;
            const double c2 = a * a - b * b, s2 = 2 * a * b, inv = 1.0 / (r2 * r2), w = std::sqrt(g2);
            mh_C += w * (c2 * c2 - s2 * s2) * inv; mh_S += w * (2 * c2 * s2) * inv; mh_W += w;
        }
    }
    if (phase != MH_EVERY - 1) return;
    const double C = mh_C, S = mh_S, Wt = mh_W; mh_C = mh_S = mh_W = 0;
    if (!(Wt > 0) || std::sqrt(C * C + S * S) < MH_COH * Wt) return;
    const double psi = 0.25 * std::atan2(S, C);
    if (!mh_ref_ok) { if (frame_no >= MH_T0) { mh_ref = psi; mh_ref_ok = true; } return; }
    double dlt = psi - mh_ref; const double qq = 0.5 * 3.14159265358979;
    while (dlt > 0.5 * qq) dlt -= qq;
    while (dlt < -0.5 * qq) dlt += qq;
    if (std::fabs(dlt) > MH_GATE) return;
    Frame& K = fr[nfr - 2];   // last keyframe
    K.mh_on = true; K.mh_dlt = dlt; for (int i = 0; i < 3; i++) K.mh_g[i] = g[i]; K.mh_R = K.R; st_mh++;
}
// heading factors of frames k0..k1-1: e = dlt + sign g . Log(R R_meas^T), on the frame's attitude
static void add_mh_factors(int k0, int k1) {
    if (!MANHATTAN) return;
    const double info = 1.0 / (MH_SIG * MH_SIG);
    for (int k = k0; k < k1; k++) {
        const Frame& F = fr[k]; if (!F.mh_on) continue;
        double w[3]; logso3(mmul(F.R, mtr(F.mh_R)), w);
        const double e = F.mh_dlt + MH_SIGN * (F.mh_g[0] * w[0] + F.mh_g[1] * w[1] + F.mh_g[2] * w[2]);
        double J[3]; mtv(F.R, F.mh_g, J); for (int i = 0; i < 3; i++) J[i] *= MH_SIGN;   // d e / d theta_body = sign g^T R
        const int o = FB(k);
        for (int p = 0; p < 3; p++) {
            gs[o + p] += info * J[p] * e;
            if (want_H) for (int q = 0; q <= p; q++) Hs[(o + p) * NX + o + q] += info * J[p] * J[q];
        }
    }
}
#else
static void manhattan_step() {}
static void add_mh_factors(int, int) {}
#endif
static void vw_update() {   // VW_DYN: weight from the errors of the previous solve (one solve of lag)
    if (VW_DYN > 0 && vw_n >= 10) { const double rms = std::sqrt(vw_e2 / vw_n); vw_k = rms > VW_DYN ? (VW_DYN / rms) * (VW_DYN / rms) : 1.0; }
#ifndef __wasm__
    if (std::getenv("VINS_VW") && vw_n >= 10) std::fprintf(stderr, "VW %.3f\n", std::sqrt(vw_e2 / vw_n));
#endif
    vw_e2 = 0; vw_n = 0;
}
static void build() {
    const int n = NSYS();
    if (VW_DYN > 0) vw_update();
    for (int i = 0; i < n; i++) { if (want_H) { double* h = Hs + i * NX; for (int j = 0; j < n; j++) h[j] = 0; } gs[i] = 0; }
    imu_cost = 0;
    add_prior();
    for (int k = 1; k < nfr; k++) add_imu(k);
    add_unary(); add_s_pin();
    add_map_factors(0, nfr);
    add_mh_factors(0, nfr);
    for (int w = 0; w < MAX_WF; w++) fb_on[w] = false;
    if (!(dbg_nofeat || novis)) {
        if (OPT_FMAX >= MAX_WF) { for (int w = 0; w < MAX_WF; w++) if (wf_usable(wf[w])) { if (!add_feature(w)) wf[w].lam_ok = false; } }
        else {   // at most OPT_FMAX features, those with the most observations first
            int cnt = 0;
            for (int nob = WIN + 1; nob >= 2 && cnt < OPT_FMAX; nob--)
                for (int w = 0; w < MAX_WF && cnt < OPT_FMAX; w++) if (wf_usable(wf[w]) && wf[w].nobs == nob) { if (add_feature(w)) cnt++; else wf[w].lam_ok = false; }
        }
    }
    if (dbg_noimu) { for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) Hs[i * NX + j] = 0; gs[i] = 0; } for (int w = 0; w < MAX_WF; w++) if (wf_usable(wf[w])) add_feature(w); }
}
#ifndef __wasm__
// numerical gradient check of build() (debug): total cost with features at fixed inverse depth, no Schur, no Huber
static double total_cost() {
    const int n = NSYS(); static double d[NX]; prior_dx(d);
    double c = 0;
    for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += (j <= i ? Hp[i * NX + j] : Hp[j * NX + i]) * d[j]; c += 0.5 * d[i] * s + gp[i] * d[i]; }
    for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) Hs[i * NX + j] = 0; gs[i] = 0; }
    imu_cost = 0; for (int k = 1; k < nfr; k++) add_imu(k); c += 0.5 * imu_cost;
    for (int k = 0; k < nfr; k++) { const Frame& F = fr[k]; double info = (F.span / VP_DT) / (F.vsig * F.vsig) + (F.zv ? 1.0 / (ZV_SIG * ZV_SIG) : 0);
        c += 0.5 * info * (F.v[0] * F.v[0] + F.v[1] * F.v[1] + F.v[2] * F.v[2]); }
    fb_cost = 0; for (int w = 0; w < MAX_WF; w++) if (wf_usable(wf[w]) && !dbg_nofeat) add_feature(w); c += fb_cost;
    if (dbg_noimu) c = fb_cost;
    return c;
}
static void jac_check(int k) {
    const int n = NSYS();
    dbg_cap = true; for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) Hs[i * NX + j] = 0; gs[i] = 0; }
    add_imu(k); double r0[15], J0[15][32]; std::memcpy(r0, dbg_r, sizeof r0); std::memcpy(J0, dbg_J, sizeof J0);
    static Frame save[WIN + 1]; double gsave[3]; std::memcpy(save, fr, sizeof fr); for (int i = 0; i < 3; i++) gsave[i] = gw[i];
    for (int c = 0; c < 32; c++) {
        const int gi = c < 15 ? col_index(k - 1, c) : (c < 30 ? col_index(k, c - 15) : c - 30);
        double rp[15], rm[15];
        for (int sg = 0; sg < 2; sg++) {
            static double dx[NX]; for (int j = 0; j < n; j++) dx[j] = 0; dx[gi] = sg ? -1e-6 : 1e-6;
            apply_dx(dx); add_imu(k); std::memcpy(sg ? rm : rp, dbg_r, sizeof rp);
            std::memcpy(fr, save, sizeof fr); for (int q = 0; q < 3; q++) gw[q] = gsave[q];
        }
        double worst = 0; int wr = -1;
        for (int r = 0; r < 15; r++) { const double num = (rp[r] - rm[r]) / 2e-6, e = std::fabs(num - J0[r][c]); if (e > worst) { worst = e; wr = r; } }
        if (worst > 1e-4) std::fprintf(stderr, "JAC k %d col %d: worst abs err %.4g at row %d (analytic %.5f numeric %.5f)\n", k, c, worst, wr, J0[wr][c], (rp[wr] - rm[wr]) / 2e-6);
    }
    dbg_cap = false;
}
static void grad_check() {
    if (std::getenv("VINS_JAC")) { for (int k = 1; k < nfr; k++) jac_check(k); return; }
    const int n = NSYS();
    dbg_noschur = true; dbg_nofeat = std::getenv("VINS_NOFEAT") != nullptr; dbg_noimu = std::getenv("VINS_NOIMU") != nullptr;
    build(); static double g0[NX]; for (int i = 0; i < n; i++) g0[i] = gs[i];
    static Frame save[WIN + 1]; double gsave[3]; std::memcpy(save, fr, sizeof fr); for (int i = 0; i < 3; i++) gsave[i] = gw[i];
    double worst = 0; int wi = -1;
    for (int i = 0; i < n; i++) {
        double c2[2];
        for (int sg = 0; sg < 2; sg++) {
            static double dx[NX]; for (int j = 0; j < n; j++) dx[j] = 0; dx[i] = sg ? -1e-6 : 1e-6;
            apply_dx(dx); c2[sg] = total_cost();
            std::memcpy(fr, save, sizeof fr); for (int q = 0; q < 3; q++) gw[q] = gsave[q];
        }
        const double num = (c2[0] - c2[1]) / 2e-6;
        const double rel = std::fabs(num - g0[i]) / (std::fabs(num) + std::fabs(g0[i]) + 1e-3);
        std::fprintf(stderr, "  g[%3d] analytic %12.5f numeric %12.5f\n", i, g0[i], num);
        if (rel > worst) { worst = rel; wi = i; }
    }
    std::fprintf(stderr, "GRADCHECK f %d n %d worst rel %.3g at %d\n", frame_no, n, worst, wi);
    dbg_noschur = false; dbg_nofeat = dbg_noimu = false;
}
#endif
static void kill_wf(int w) {
    WF& F = wf[w];
    if (OUT_KEEP_TRACK && F.live >= 0) { F.nobs = 0; F.lam_ok = false; return; }   // restart the window feature, keep the track
    if (F.live >= 0) { feats[F.live].active = false; feats[F.live].wf = -1; }
    F.used = false; F.live = -1; F.nobs = 0; F.lam_ok = false;
}
static bool gn_iteration(bool last, bool chord) {
    const int n = NSYS();
    want_H = !chord; build(); want_H = true;
    int nfeat = 0; for (int w = 0; w < MAX_WF; w++) nfeat += fb_on[w];
    const double out_px = (OUT_LOW_N > 0 && nfeat < OUT_LOW_N) ? OUT_PX_LOW : OUT_PX;   // few features: keep more of them
    for (int i = 0; i < n; i++) { if (!chord) HH(i, i) += LM_REL * HH(i, i) + 1e-9; dxs[i] = -gs[i]; }
#if PROF_CHOL2
    { static double Hc[NX * NX]; std::memcpy(Hc, Hs, sizeof(double) * NX * n); chol(Hc, n, NX); }
#endif
#if PROF_BUILD2
    { static double Hc[NX * NX], gc[NX]; std::memcpy(Hc, Hs, sizeof(double) * NX * n); std::memcpy(gc, gs, sizeof(double) * n); build(); std::memcpy(Hs, Hc, sizeof(double) * NX * n); std::memcpy(gs, gc, sizeof(double) * n); }
#endif
    if (CHOL_F32) { if (!solve_f32(Hs, n, NX, dxs)) { st_fail++; return false; } }
    else {
        if (!chord && !(CHOL4 ? chol4<NX>(Hs, n) : chol(Hs, n, NX))) { st_fail++; return false; }
        if (MAH && last) mah_store(n);
        chol_solve(Hs, n, NX, dxs);
    }
    for (int i = 0; i < n; i++) if (!std::isfinite(dxs[i])) { st_fail++; return false; }
    apply_dx(dxs);
    for (int w = 0; w < MAX_WF; w++) if (fb_on[w]) {
        double s = fb_gf[w]; if (SCALE_ST) s += fb_hls[w] * dxs[SI];
        for (int s1 = 0; s1 < fb_nf[w]; s1++) { const double* d = dxs + FB(fb_fl[w][s1]); const double* h = fb_hxf[w] + 6 * s1; for (int p = 0; p < 6; p++) s += h[p] * d[p]; }
        WF& F = wf[w];
        F.lam -= s / fb_hff[w];
        if (!(F.lam > 1.0 / MAX_DEPTH && F.lam < 1.0 / MIN_DEPTH) || (last && fb_err[w] > out_px)) { st_out++; kill_wf(w); }
    }
    return true;
}

// ---------------------------------------------------------------- window management
// remove block [o, o+len) of an n x n system (row stride NX) by Schur complement, compacting in place
static struct { int n, o, len; bool ok; double A[15 * 15], gx[15]; } so_;   // schur_out in parts (PIPE scheduler)
static double so_B[15][NX], so_X[15][NX];   // B = H_block,c ; X = A^-1 B
static void schur_prep(double* Hm, double* gm, int n, int o, int len) {
    const int e = o + len; double* A = so_.A;
    so_.n = n; so_.o = o; so_.len = len; so_.ok = false;
    bool any = false;
    for (int i = 0; i < len; i++) for (int j = 0; j <= i; j++) { const double v = Hm[(o + i) * NX + o + j]; A[i * len + j] = A[j * len + i] = v; any = any || v != 0; }
    if (!any) return;
    double mx = 0; for (int i = 0; i < len; i++) if (A[i * len + i] > mx) mx = A[i * len + i];
    for (int i = 0; i < len; i++) A[i * len + i] += 1e-9 * mx + 1e-12;
    if (!chol(A, len, len)) return;
    for (int i = 0; i < len; i++) {
        for (int c = 0; c < o; c++) so_B[i][c] = Hm[(o + i) * NX + c];
        for (int c = o; c < e; c++) so_B[i][c] = 0;
        for (int c = e; c < n; c++) so_B[i][c] = Hm[c * NX + o + i];
    }
    for (int c = 0; c < n; c++) {
        if (c >= o && c < e) { for (int i = 0; i < len; i++) so_X[i][c] = 0; continue; }
        double col[15]; for (int i = 0; i < len; i++) col[i] = so_B[i][c];
        chol_solve(A, len, len, col);
        for (int i = 0; i < len; i++) so_X[i][c] = col[i];
    }
    for (int i = 0; i < len; i++) so_.gx[i] = gm[o + i];
    chol_solve(A, len, len, so_.gx);
    so_.ok = true;
}
static void schur_rows(double* Hm, double* gm, int r0, int r1) {   // lower triangle (c <= r), axpy over contiguous c
    if (!so_.ok) return;
    const int o = so_.o, e = o + so_.len, len = so_.len;
    for (int r = r0; r < r1; r++) {
        if (r >= o && r < e) continue;
        double* hr = Hm + r * NX; double s = 0;
        for (int i = 0; i < len; i++) { const double a = so_B[i][r]; const double* xi = so_X[i]; for (int c = 0; c <= r; c++) hr[c] -= a * xi[c]; s += a * so_.gx[i]; }
        gm[r] -= s;
    }
}
static void schur_fin(double* Hm, double* gm) {   // compact (lower triangle), clear the freed rows
    const int n = so_.n, o = so_.o, len = so_.len, e = o + len;
    for (int r = 0, rr = 0; r < n; r++) {
        if (r >= o && r < e) continue;
        double* src = Hm + r * NX; double* dst = Hm + rr * NX;
        int cc = 0; for (int c = 0; c < o && c <= r; c++) dst[cc++] = src[c];
        for (int c = e; c <= r; c++) dst[cc++] = src[c];
        gm[rr] = gm[r]; rr++;
    }
    for (int r = n - len; r < NX; r++) { double* h = Hm + r * NX; for (int c = 0; c < NX; c++) h[c] = 0; }   // freed rows
    for (int r = n - len; r < n; r++) gm[r] = 0;
}
// remove block [o, o+len) of an n x n system (row stride NX) by Schur complement, compacting in place
static void schur_out(double* Hm, double* gm, int n, int o, int len) {   // lower-triangle storage in and out
    schur_prep(Hm, gm, n, o, len); schur_rows(Hm, gm, 0, n); schur_fin(Hm, gm);
}
static void set_lin() {
    for (int i = 0; i < 3; i++) { gl[i] = gw[i]; bgl_s[i] = fr[0].bg[i]; bal_s[i] = fr[0].ba[i]; }
    sstl = sst;
    for (int k = 0; k < nfr; k++) { Frame& F = fr[k]; F.Rl = F.R; for (int i = 0; i < 3; i++) { F.pl[i] = F.p[i]; F.vl[i] = F.v[i]; F.bgl[i] = F.bg[i]; F.bal[i] = F.ba[i]; } }
}
// re-anchor a feature on its observation o1 (the anchor observation is being removed)
static void reanchor(WF& F, int o1) {
    if (!F.lam_ok) return;
    double pwv[3];
    if (!wf_point(F, pwv)) { F.lam_ok = false; return; }
    const int k = fidx(F.ser[o1]); if (k < 0) { F.lam_ok = false; return; }
    M3 Rw; double pc[3]; cam_pose(k, Rw, pc);
    double d[3] = {pwv[0] - pc[0], pwv[1] - pc[1], pwv[2] - pc[2]}, q[3]; mtv(Rw, d, q);
    if (q[2] > MIN_DEPTH && q[2] < MAX_DEPTH) F.lam = 1.0 / q[2]; else F.lam_ok = false;
}
static void remove_obs(int serial) {
    for (int w = 0; w < MAX_WF; w++) {
        WF& F = wf[w]; if (!F.used) continue;
        int o = -1; for (int i = 0; i < F.nobs; i++) if (F.ser[i] == serial) { o = i; break; }
        if (o < 0) continue;
        if (o == 0 && F.nobs >= 2) reanchor(F, 1);
        for (int i = o; i + 1 < F.nobs; i++) { F.ser[i] = F.ser[i + 1]; F.u[i] = F.u[i + 1]; F.v[i] = F.v[i + 1]; F.q[i] = F.q[i + 1]; EDGE_SHIFT(F, i); }
        F.nobs--;
        if (F.nobs < 2) F.lam_ok = false;
        if (F.nobs == 0 && F.live < 0) F.used = false;
    }
}
static void drop_second_newest() {
    const int s = nfr - 2, n = NSYS();
    remove_obs(fr[s].serial);
    schur_out(Hp, gp, n, FB(s), NF_);
    pre_compose(pre[s], pre[s + 1]);
    const double span = fr[s].span;
    fr[s] = fr[s + 1]; fr[s].span += span;
    nfr--; st_drop++;
}
static void marg_build() {   // system of everything that touches frame 0, in Hs/gs (marginalize_oldest, first part)
    const int n = NSYS();
    // system of everything that touches frame 0: prior, IMU factor 0-1, frame 0's priors, features anchored at frame 0
    for (int i = 0; i < n; i++) { double* h = Hs + i * NX; for (int j = 0; j < n; j++) h[j] = 0; gs[i] = 0; }
    add_prior();
    add_imu(1);
    { const int keep = nfr; nfr = 1; add_unary(); nfr = keep; }
    add_map_factors(0, 1);
    add_mh_factors(0, 1);
    const int s0 = fr[0].serial;
    {   // features anchored at frame 0 go into the prior; at most MARG_FMAX (most observations first), the others are re-anchored
        int cnt = 0;
        for (int w = 0; w < MAX_WF; w++) if (wf[w].used && wf[w].nobs > 0 && wf[w].ser[0] == s0) fb_on[w] = false;
        if (MARG_CACHE && mc_valid) {   // blocks from this frame's solve, gradient moved by the step: g + H dx
            for (int nob = WIN + 1; nob >= 2 && cnt < MARG_FMAX; nob--)
                for (int c = 0; c < mc_n && cnt < MARG_FMAX; c++) {
                    const int w = mc_w[c]; WF& F = wf[w];
                    if (!wf_usable(F) || F.ser[0] != s0 || F.nobs != nob) continue;
                    const int m = mc_m[c];
                    static double dl[6 * (WIN + 1)]; for (int s1 = 0; s1 < m; s1++) for (int p = 0; p < 6; p++) dl[6 * s1 + p] = dxs[FB(mc_fl[c][s1]) + p];
                    for (int s1 = 0; s1 < m; s1++) {
                        const int g1 = FB(mc_fl[c][s1]);
                        for (int p = 0; p < 6; p++) {
                            const int r = 6 * s1 + p; double gv = mc_g[c][r];
                            for (int t = 0; t < 6 * m; t++) gv += (t <= r ? mc_H[c][r][t] : mc_H[c][t][r]) * dl[t];
                            gs[g1 + p] += gv;
                            for (int s2 = 0; s2 <= s1; s2++) {
                                const int g2 = FB(mc_fl[c][s2]); double* h = Hs + (g1 + p) * NX + g2;
                                const int qmax = s1 == s2 ? p + 1 : 6;
                                for (int q = 0; q < qmax; q++) h[q] += mc_H[c][r][6 * s2 + q];
                            }
                        }
                    }
                    fb_on[w] = true; cnt++;
                }
        } else
        for (int nob = WIN + 1; nob >= 2 && cnt < MARG_FMAX; nob--)
            for (int w = 0; w < MAX_WF && cnt < MARG_FMAX; w++) { WF& F = wf[w]; if (wf_usable(F) && F.ser[0] == s0 && F.nobs == nob && (MARG_LIVE || F.live < 0)) { if (add_feature(w)) cnt++; } }
    }
    for (int r = n; r < NX; r++) { double* h = Hs + r * NX; for (int c = 0; c < NX; c++) h[c] = 0; }   // rows the build did not touch
}
static void marg_fin() {   // after frame 0 is eliminated from Hs: it becomes the prior; features and frames shift
    const int n = NSYS();
    const int s0 = fr[0].serial;
    { double* t = Hp; Hp = Hs; Hs = t; } for (int i = 0; i < n; i++) gp[i] = gs[i];
    if (SHARED_BIAS) for (int i = 2; i < 8; i++) {   // bias information floor (random walk): scale its row and column
        const double cap = 1.0 / (i < 5 ? BIAS_FLOOR_G * BIAS_FLOOR_G : BIAS_FLOOR_A * BIAS_FLOOR_A), h = Hp[i * NX + i];
        if (h > cap) { const double f = std::sqrt(cap / h); for (int j = 0; j <= i; j++) Hp[i * NX + j] *= f; for (int j = i; j < n; j++) Hp[j * NX + i] *= f; gp[i] *= f; }
    }
    if (SCALE_ST) { const int i = SI; const double cap = 1.0 / (S_FLOOR * S_FLOOR), h = Hp[i * NX + i];
        if (h > cap) { const double f = std::sqrt(cap / h); for (int j = 0; j <= i; j++) Hp[i * NX + j] *= f; for (int j = i; j < n; j++) Hp[j * NX + i] *= f; gp[i] *= f; } }
    // features: marginalized ones leave the window (their information is in the prior), others lose their frame-0 observation
    for (int w = 0; w < MAX_WF; w++) {
        WF& F = wf[w]; if (!F.used || F.nobs == 0 || F.ser[0] != s0) continue;
        if (DPF_ADAPT > 0 && fb_on[w] && F.nobs >= 2) {   // room depth scale from features with a large triangulation angle
            double pwv[3]; const int kl = fidx(F.ser[F.nobs - 1]);
            if (kl >= 0 && wf_point(F, pwv)) {
                M3 Ra, Rl; double ca[3], cl[3]; cam_pose(fidx(F.ser[0]), Ra, ca); cam_pose(kl, Rl, cl);
                double a[3], b[3]; for (int i = 0; i < 3; i++) { a[i] = pwv[i] - ca[i]; b[i] = pwv[i] - cl[i]; }
                const double na = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]), nb = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
                const double cs = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
                if (cs < std::cos(DPF_ADAPT_PAR)) {
                    dpf_lz0 += (-std::log(F.lam) - dpf_lz0) / (DPF_ADAPT > 0 ? DPF_ADAPT : 1);
                    const double lo = std::log(1.2), hi = std::log(4.0); if (dpf_lz0 < lo) dpf_lz0 = lo; if (dpf_lz0 > hi) dpf_lz0 = hi;
                }
            }
        }
        if (fb_on[w] && !MARG_KEEP) { map_note_candidate(w); F.nobs = 0; F.lam_ok = false; if (F.live < 0) F.used = false; continue; }
        if (F.nobs >= 2) reanchor(F, 1);
        for (int i = 0; i + 1 < F.nobs; i++) { F.ser[i] = F.ser[i + 1]; F.u[i] = F.u[i + 1]; F.v[i] = F.v[i + 1]; F.q[i] = F.q[i + 1]; EDGE_SHIFT(F, i); }
        F.nobs--; if (F.nobs < 2) F.lam_ok = false;
        if (F.nobs == 0 && F.live < 0) F.used = false;
    }
    for (int k = 0; k + 1 < nfr; k++) fr[k] = fr[k + 1];
    for (int k = 1; k + 1 < nfr; k++) pre[k] = pre[k + 1];
    nfr--;
    set_lin();
    st_marg++;
}
static void marginalize_oldest() {
    marg_build();
    schur_out(Hs, gs, NSYS(), FB(0), NF_);
    marg_fin();
}
static bool second_newest_is_kf() {
    if (nfr < 3) return true;
    const int s = nfr - 2, q = nfr - 3;
    if (fr[s].serial - fr[q].serial >= KF_MAXGAP) return true;
    if (ZV_KEEP && fr[s].zv) return true;   // keep the zero-velocity information (it would be lost with a dropped frame)
    const M3 Rcc = mmul(mmul(Rci, mmul(mtr(fr[s].R), fr[q].R)), mtr(Rci));
    static float d[MAX_WF]; int nd = 0;
    for (int w = 0; w < MAX_WF; w++) {
        const WF& F = wf[w]; if (!F.used) continue;
#if EDGE
        if (F.edge) continue;
#endif
        int a = -1, b = -1;
        for (int i = 0; i < F.nobs; i++) { if (F.ser[i] == fr[q].serial) a = i; if (F.ser[i] == fr[s].serial) b = i; }
        if (a < 0 || b < 0) continue;
        double bb[3] = {F.u[a], F.v[a], 1}, r[3]; mv(Rcc, bb, r);
        if (r[2] < 0.1) continue;
        const double du = r[0] / r[2] - F.u[b], dv = r[1] / r[2] - F.v[b];
        d[nd++] = (float)std::sqrt(du * du + dv * dv);
    }
    if (nd < KF_MINCOM) return true;
    for (int i = 1; i < nd; i++) { float v = d[i]; int j = i - 1; while (j >= 0 && d[j] > v) { d[j + 1] = d[j]; j--; } d[j + 1] = v; }
    return d[nd / 2] >= KF_PAR;
}

static void reset_prior(double sv, double sbg, double sba, double sg) {
    for (int i = 0; i < NX; i++) { for (int j = 0; j < NX; j++) Hp[i * NX + j] = 0; gp[i] = 0; }
    Hp[0] = Hp[NX + 1] = 1.0 / (sg * sg);
    if (SCALE_ST) { Hp[SI * NX + SI] = 1.0 / (S_SIG0 * S_SIG0); gp[SI] = sst / (S_SIG0 * S_SIG0); }   // centred on ln DPF_Z0
    const int o = FB(0);
    for (int i = 0; i < 6; i++) Hp[(o + i) * NX + o + i] = 1e8;
    for (int i = 0; i < 3; i++) {
        Hp[(o + 6 + i) * NX + o + 6 + i] = 1.0 / (sv * sv);
        Hp[(BGI(0) + i) * NX + BGI(0) + i] = 1.0 / (sbg * sbg);
        Hp[(BAI(0) + i) * NX + BAI(0) + i] = 1.0 / (sba * sba);
    }
    set_lin();
}
static void reset_window() {
    double vc[3];
    const bool vis = visual_velocity(frame_dt, vc);
    fr[0] = fr[nfr - 1]; nfr = 1;
    Frame& F = fr[0];
    if (vis) {
        double vi[3]; mtv(Rci, vc, vi); mv(F.R, vi, F.v);
        if (RESET_VCAP_K > 0) {
            double cap = RESET_VCAP_K * std::sqrt(v2_avg); if (cap < RESET_VCAP_MIN) cap = RESET_VCAP_MIN;
            const double sv = std::sqrt(F.v[0] * F.v[0] + F.v[1] * F.v[1] + F.v[2] * F.v[2]);
            if (sv > cap) for (int i = 0; i < 3; i++) F.v[i] *= cap / sv;
        }
    } else for (int i = 0; i < 3; i++) F.v[i] = 0;
    reset_prior(vis ? SIG_VVIS : SIG_VRESET, RESET_BG, RESET_BA, RESET_G);
    for (int w = 0; w < MAX_WF; w++) { WF& F2 = wf[w]; F2.nobs = 0; F2.lam_ok = false; if (F2.live < 0) F2.used = false; }
    n_resets++; last_reset_frame = frame_no;
#if MAH
    mah_ok = false;
#endif
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
static void output_frame(const Frame& F, double* out) {
    M3 Ric = mtr(Rci); double pic[3]; mtv(Rci, tci, pic); for (int i = 0; i < 3; i++) pic[i] = -pic[i];
    M3 Rwc = mmul(F.R, Ric); double t[3]; mv(F.R, pic, t);
    double pwc[3] = {F.p[0] + t[0], F.p[1] + t[1], F.p[2] + t[2]};
    M3 R = mmul(Rci, Rwc); double pos[3]; mv(Rci, pwc, pos); for (int i = 0; i < 3; i++) pos[i] += tci[i];
    double q[4]; rot2quat(R, q);
    double vi[3]; mtv(F.R, F.v, vi);
    double wx[3] = {w_last[1] * pic[2] - w_last[2] * pic[1], w_last[2] * pic[0] - w_last[0] * pic[2], w_last[0] * pic[1] - w_last[1] * pic[0]};
    for (int i = 0; i < 3; i++) vi[i] += wx[i];
    double vc[3], wc[3]; mv(Rci, vi, vc); mv(Rci, w_last, wc);
    double o[13] = {pos[0], pos[1], pos[2], q[0], q[1], q[2], q[3], vc[0], vc[1], vc[2], wc[0], wc[1], wc[2]};
    bool ok = true; for (int i = 0; i < 13; i++) ok = ok && std::isfinite(o[i]);
    if (!ok) { for (int i = 0; i < 13; i++) o[i] = 0; o[6] = 1; }
    for (int i = 0; i < 13; i++) out[i] = o[i];
}
static void output(double* out) { output_frame(fr[nfr - 1], out); }

// ---------------------------------------------------------------- PIPE: optimizer on every second frame, split over two frames
#if PIPE
#if NIT != 1 || IMU_EST || IMU_MID || MARG_CACHE || CHOL_F32
#error "PIPE supports NIT 1 without IMU_EST, IMU_MID, MARG_CACHE, CHOL_F32"
#endif
static Pre pp; static bool pp_on = false; static M3 pp_dR_last;   // pending preintegration since the newest window frame
static int pipe_ph = 0;                                           // 0: this frame adds a window frame (A), 1: it finishes the solve (B)
static Frame pipe_F; static bool pipe_kf = false, pipe_ok = false; static int pipe_n = 0, pipe_js = 0;
static int pipe_cut(double f) { int c = (int)(f * pipe_n); return c < 0 ? 0 : (c > pipe_n ? pipe_n : c); }
static void predict_frame(const Frame& N, const Pre& P, Frame& F) {
    F = N;
    F.R = mmul(N.R, P.dR); orthonormalize(F.R);
    double a[3], b[3]; mv(N.R, P.dv, a); mv(N.R, P.dp, b);
    for (int i = 0; i < 3; i++) { F.p[i] = N.p[i] + N.v[i] * P.T + 0.5 * gw[i] * P.T * P.T + b[i]; F.v[i] = N.v[i] + gw[i] * P.T + a[i]; }
}
static bool gn_begin() {   // build, damping, Cholesky columns [0, js)
    const int n = NSYS(); pipe_n = n;
    build();
    for (int i = 0; i < n; i++) { HH(i, i) += LM_REL * HH(i, i) + 1e-9; dxs[i] = -gs[i]; }
    pipe_js = pipe_cut(PIPE >= 3 ? PIPE_S1 : PIPE_SPLIT);
    if (!chol4r<NX>(Hs, n, 0, pipe_js)) { st_fail++; return false; }
    return true;
}
static bool gn_middle() {   // PIPE 3: middle part of the Cholesky
    const int j1 = pipe_cut(PIPE_S2) > pipe_js ? pipe_cut(PIPE_S2) : pipe_js;
    if (!chol4r<NX>(Hs, pipe_n, pipe_js, j1)) { st_fail++; return false; }
    pipe_js = j1; return true;
}
static bool gn_finish() {   // rest of the Cholesky, step, inverse depths, outliers
    const int n = pipe_n;
    int nfeat = 0; for (int w = 0; w < MAX_WF; w++) nfeat += fb_on[w];
    const double out_px = (OUT_LOW_N > 0 && nfeat < OUT_LOW_N) ? OUT_PX_LOW : OUT_PX;
    if (!chol4r<NX>(Hs, n, pipe_js, n)) { st_fail++; return false; }
    if (MAH) mah_store(n);
    chol_solve(Hs, n, NX, dxs);
    for (int i = 0; i < n; i++) if (!std::isfinite(dxs[i])) { st_fail++; return false; }
    apply_dx(dxs);
    for (int w = 0; w < MAX_WF; w++) if (fb_on[w]) {
        double s = fb_gf[w]; if (SCALE_ST) s += fb_hls[w] * dxs[SI];
        for (int s1 = 0; s1 < fb_nf[w]; s1++) { const double* d = dxs + FB(fb_fl[w][s1]); const double* h = fb_hxf[w] + 6 * s1; for (int p = 0; p < 6; p++) s += h[p] * d[p]; }
        WF& F = wf[w];
        F.lam -= s / fb_hff[w];
        if (!(F.lam > 1.0 / MAX_DEPTH && F.lam < 1.0 / MIN_DEPTH) || fb_err[w] > out_px) { st_out++; kill_wf(w); }
    }
    return true;
}
static void pipe_templates(const M3& Rnow) {
    for (int i = 0; i < FEAT_N; i++) {
        Feat& f = feats[i]; if (!f.active) continue;
        if (!cut_template(f)) { f.active = false; if (f.wf >= 0) wf[f.wf].live = -1; f.wf = -1; continue; }
        f.hx[frame_no & 15] = f.x; f.hy[frame_no & 15] = f.y;
    }
    Rh[frame_no & 15] = Rnow;
}

#if PIPE_SCHED
// ---- scheduler: the chain of one cycle in units with estimated costs
enum { SC_BASE, SC_FEAT, SC_DAMP, SC_CHOL, SC_SOLVE, SC_MBUILD, SC_MPREP, SC_MROWS, SC_MFIN, SC_BIMU, SC_MBF, SC_DROP0, SC_DROWS, SC_DFIN, SC_DONE };
static int sc_ml[MAX_WF], sc_mln = 0, sc_mcnt = 0;
static int sc_len = PIPE;   // frames in the current cycle   // marginalization: candidate features in order, features added
static int sc_st = SC_DONE, sc_cur = 0, sc_n = 0; static double sc_done = 0, sc_total = 1; static bool sc_marg = false;
static double sc_log[SC_DONE + 1];   // host debug: estimated units per type this frame
static inline int feat_m(const WF& F) { return (OBS_MAX > 0 && F.nobs > OBS_MAX) ? OBS_MAX : F.nobs; }
static double cu_base(int n) { return CU_N2 * n * n + CU_IMU * (nfr - 1); }
static double cu_feat(const WF& F) { const double m = feat_m(F); return CU_F0 + CU_FM * m * m; }
static double cu_col(int n, int j) { return CU_CH * (n - j) * (j + 8.0); }
static double cu_marg_build(int n) {
    double c = CU_N2 * n * n + CU_IMU; int cnt = 0; const int s0 = fr[0].serial;
    for (int w = 0; w < MAX_WF && cnt < MARG_FMAX; w++) if (wf_usable(wf[w]) && wf[w].ser[0] == s0) { c += cu_feat(wf[w]); cnt++; }
    return c;
}
static double cu_mprep(int n) { return 17.0 * NF_ * NF_ * n + 12000; }
static double cu_mfin(int n) { return 0.6 * n * n + 5000; }
static double sc_estimate() {
    const int n = NSYS(); double c = cu_base(n) + 2000;
    for (int w = 0; w < MAX_WF; w++) if (wf_usable(wf[w])) c += cu_feat(wf[w]);
    for (int j = 0; j < n; j++) c += cu_col(n, j);
    c += CU_SOLVE * n * n + 20000;
    if (sc_marg) { const int nm = n; c += cu_marg_build(nm) + cu_mprep(nm) + CU_SR * NF_ * nm * nm * 0.5 + cu_mfin(nm); }
    else if (nfr >= 3 && !pipe_kf && Hp[FB(nfr - 2) * NX + FB(nfr - 2)] != 0) c += cu_mprep(n) + CU_SR * NF_ * n * n * 0.5 + 40000;   // drop with a prior Schur step
    return c;
}
static void sc_start() { sc_st = SC_BASE; sc_cur = 0; sc_done = 0; sc_marg = pipe_kf && nfr == WIN; sc_total = sc_estimate(); }
static void sc_solve_done(bool ok) {   // step applied (or failed): resets, drop, or marginalization next
    st_opt++;
    if (ok) map_check();
    const Frame& Nw = fr[nfr - 1];
    const double sp = std::sqrt(Nw.v[0] * Nw.v[0] + Nw.v[1] * Nw.v[1] + Nw.v[2] * Nw.v[2]);
    bool finite = true; for (int i = 0; i < 3; i++) finite = finite && std::isfinite(Nw.p[i]) && std::isfinite(Nw.v[i]);
    {   int nu = 0; for (int w = 0; w < MAX_WF; w++) nu += fb_on[w];
        if (ok && finite && nu >= V2_MINF) v2_avg += (Nw.v[0] * Nw.v[0] + Nw.v[1] * Nw.v[1] + Nw.v[2] * Nw.v[2] - v2_avg) * (sc_len * frame_dt / VP_TAU); }
    if (!ok || !finite || sp > VMAX) {
        if (!finite) { fr[nfr - 1] = pipe_F; }
        reset_window(); sc_st = SC_DONE; return;
    }
    if (nfr >= 3 && !pipe_kf) { sc_st = SC_DROP0; return; }
    if (nfr >= 3 && nfr == WIN) { sc_st = SC_MBUILD; return; }
    sc_st = SC_DONE;
}
static void sc_unit();
static void sc_work(double target) {   // run units until the estimated work reaches target (or the chain ends)
    while (sc_st != SC_DONE && sc_done < target) {
        const int st0 = sc_st; const double d0 = sc_done;
        sc_unit();
        sc_log[st0] += sc_done - d0;
    }
}
static void sc_unit() {
    {
        switch (sc_st) {
        case SC_BASE: {   // zero, prior (then one unit per IMU factor)
            const int n = NSYS(); sc_n = n;
            for (int i = 0; i < n; i++) { double* h = Hs + i * NX; for (int j = 0; j < n; j++) h[j] = 0; gs[i] = 0; }
            imu_cost = 0; if (VW_DYN > 0) vw_update();
            add_prior();
            sc_done += CU_N2 * n * n; sc_st = SC_BIMU; sc_cur = 1; break; }
        case SC_BIMU:
            if (sc_cur < nfr) { add_imu(sc_cur); sc_cur++; sc_done += CU_IMU; break; }
            add_unary(); add_s_pin();
            add_map_factors(0, nfr);
            add_mh_factors(0, nfr);
            for (int w = 0; w < MAX_WF; w++) fb_on[w] = false;
            sc_st = SC_FEAT; sc_cur = 0; break;
        case SC_FEAT:
            if (dbg_nofeat || novis) { sc_st = SC_DAMP; break; }
            for (; sc_cur < MAX_WF; sc_cur++) if (wf_usable(wf[sc_cur])) break;
            if (sc_cur >= MAX_WF) { sc_st = SC_DAMP; break; }
            sc_done += cu_feat(wf[sc_cur]);
            if (!add_feature(sc_cur)) wf[sc_cur].lam_ok = false;
            sc_cur++; break;
        case SC_DAMP:
            for (int i = 0; i < sc_n; i++) { HH(i, i) += LM_REL * HH(i, i) + 1e-9; dxs[i] = -gs[i]; }
            sc_done += 2000; sc_st = SC_CHOL; sc_cur = 0; break;
        case SC_CHOL:
            if (!pipe_ok) { sc_st = SC_SOLVE; break; }
            if (sc_cur >= sc_n) { sc_st = SC_SOLVE; break; }
            sc_done += cu_col(sc_n, sc_cur);
            if (!chol4r<NX>(Hs, sc_n, sc_cur, sc_cur + 1)) { st_fail++; pipe_ok = false; }
            sc_cur++; break;
        case SC_SOLVE: {
            bool ok = pipe_ok;
            if (ok) {
                const int n = sc_n;
                int nfeat = 0; for (int w = 0; w < MAX_WF; w++) nfeat += fb_on[w];
                const double out_px = (OUT_LOW_N > 0 && nfeat < OUT_LOW_N) ? OUT_PX_LOW : OUT_PX;
                if (MAH) mah_store(n);
                chol_solve(Hs, n, NX, dxs);
                for (int i = 0; i < n && ok; i++) if (!std::isfinite(dxs[i])) { st_fail++; ok = false; }
                if (ok) {
                    apply_dx(dxs);
                    for (int w = 0; w < MAX_WF; w++) if (fb_on[w]) {
                        double sv = fb_gf[w];
                        if (SCALE_ST) sv += fb_hls[w] * dxs[SI]; for (int s1 = 0; s1 < fb_nf[w]; s1++) { const double* d = dxs + FB(fb_fl[w][s1]); const double* h = fb_hxf[w] + 6 * s1; for (int q = 0; q < 6; q++) sv += h[q] * d[q]; }
                        WF& F = wf[w];
                        F.lam -= sv / fb_hff[w];
                        if (!(F.lam > 1.0 / MAX_DEPTH && F.lam < 1.0 / MIN_DEPTH) || fb_err[w] > out_px) { st_out++; kill_wf(w); }
                    }
                }
            }
            sc_done += CU_SOLVE * sc_n * sc_n + 20000;
            sc_solve_done(ok); break; }
        case SC_MBUILD: {   // marg_build in units: prior and IMU factor 0-1 first, then one feature per unit
            const int n = NSYS(); sc_done += CU_N2 * n * n + CU_IMU;
            for (int i = 0; i < n; i++) { double* h = Hs + i * NX; for (int j = 0; j < n; j++) h[j] = 0; gs[i] = 0; }
            add_prior();
            add_imu(1);
            { const int keep = nfr; nfr = 1; add_unary(); nfr = keep; }
            add_map_factors(0, 1);
            add_mh_factors(0, 1);
            const int s0 = fr[0].serial;
            for (int w = 0; w < MAX_WF; w++) if (wf[w].used && wf[w].nobs > 0 && wf[w].ser[0] == s0) fb_on[w] = false;
            sc_mln = 0; sc_mcnt = 0;
            for (int nob = WIN + 1; nob >= 2; nob--)
                for (int w = 0; w < MAX_WF; w++) { const WF& F = wf[w]; if (wf_usable(F) && F.ser[0] == s0 && F.nobs == nob && (MARG_LIVE || F.live < 0)) sc_ml[sc_mln++] = w; }
            sc_st = SC_MBF; sc_cur = 0; break; }
        case SC_MBF:
            if (sc_cur < sc_mln && sc_mcnt < MARG_FMAX) { const int w = sc_ml[sc_cur++]; sc_done += cu_feat(wf[w]); if (add_feature(w)) sc_mcnt++; break; }
            {   const int n = NSYS(); for (int r = n; r < NX; r++) { double* h = Hs + r * NX; for (int c = 0; c < NX; c++) h[c] = 0; } }
            sc_st = SC_MPREP; break;
        case SC_MPREP: {
            const int n = NSYS(); sc_done += cu_mprep(n);
            schur_prep(Hs, gs, n, FB(0), NF_); sc_st = SC_MROWS; sc_cur = 0; break; }
        case SC_MROWS: {
            const int n = NSYS();
            if (sc_cur >= n) { sc_st = SC_MFIN; break; }
            const int r1 = sc_cur + 4 < n ? sc_cur + 4 : n;
            for (int r = sc_cur; r < r1; r++) sc_done += CU_SR * NF_ * (r + 1.0);
            schur_rows(Hs, gs, sc_cur, r1); sc_cur = r1; break; }
        case SC_DROP0: {   // drop_second_newest in units
            const int sd = nfr - 2, n = NSYS();
            remove_obs(fr[sd].serial);
            schur_prep(Hp, gp, n, FB(sd), NF_);
            sc_done += cu_mprep(n) + 20000; sc_st = SC_DROWS; sc_cur = 0; break; }
        case SC_DROWS: {
            const int n = NSYS();
            if (sc_cur >= n || !so_.ok) { sc_st = SC_DFIN; break; }
            const int r1 = sc_cur + 4 < n ? sc_cur + 4 : n;
            for (int r = sc_cur; r < r1; r++) sc_done += CU_SR * NF_ * (r + 1.0);
            schur_rows(Hp, gp, sc_cur, r1); sc_cur = r1; break; }
        case SC_DFIN: {
            const int sd = nfr - 2;
            schur_fin(Hp, gp);
            pre_compose(pre[sd], pre[sd + 1]);
            const double span = fr[sd].span;
            fr[sd] = fr[sd + 1]; fr[sd].span += span;
            nfr--; st_drop++;
            sc_done += 20000; sc_st = SC_DONE; break; }
        case SC_MFIN: {
            const int n = NSYS(); sc_done += cu_mfin(n);
            schur_fin(Hs, gs); marg_fin(); marg_now = true; sc_st = SC_DONE; break; }
        }
    }
}
static double sc_fixed_prev[4] = {6e5, 6e5, 6e5, 6e5}, sc_level = 0, sc_target = 0;
static double sc_frac(int ph) {
    if (ph >= PIPE - 1) return 1e300;
    const double f[3] = {PIPE_F0, PIPE >= 4 ? PIPE_F1 : PIPE_F1, PIPE_F2};
    return f[ph < 3 ? ph : 2] * sc_total;
}
static void step_sched(const double* imu, int32_t nimu, double* out) {
    for (int i = 0; i <= SC_DONE; i++) sc_log[i] = 0;
    const long c0 = st_cells, m0 = st_map_try, d0 = st_det, cf0 = st_cfull;
    if (pipe_ph == 0 && sc_st != SC_DONE) sc_work(1e300);   // never happens with the last frame finishing the chain
    {   const Frame& N = fr[nfr - 1];
        if (!pp_on) { pre_reset(pp, N.bg, N.ba); pp_on = true; pp_dR_last = eye3(); }
        const double t_before = t_last;
        for (int k = 0; k < nimu; k++) {
            const double* r = imu + 7 * k;
            const double dt = r[0] - t_last;
            if (dt <= 0 || dt > 0.1) { t_last = r[0] > t_last ? r[0] : t_last; continue; }
            t_last = r[0];
            pre_add(pp, dt, r + 1);
            for (int i = 0; i < 3; i++) w_last[i] = r[4 + i] - N.bg[i];
        }
        if (t_last > t_before) frame_dt = t_last - t_before;
    }
    const M3 dRi = mmul(mtr(pp_dR_last), pp.dR); pp_dR_last = pp.dR;
    const M3 Rcc = mmul(mmul(Rci, mtr(dRi)), mtr(Rci));
    Frame Fpred; if (MAH_PRED) { predict_frame(fr[nfr - 1], pp, Fpred); trk_pred = &Fpred; }
    track(Rcc);
    if (pipe_ph == 0) {
        Frame F; predict_frame(fr[nfr - 1], pp, F);
        F.serial = frame_no; F.t = t_last; F.span = pp.T > 0 ? pp.T : PIPE * frame_dt; F.zv = false; F.mo_n = 0;
#if MANHATTAN
        F.mh_on = false;
#endif
        {   double sig = VPRIOR_SIG;
            if (VP_ADAPT) { sig = VP_K * std::sqrt(v2_avg); if (sig < VP_MIN) sig = VP_MIN; if (sig > VPRIOR_SIG) sig = VPRIOR_SIG; }
            F.vsig = sig; }
        fr[nfr] = F; pre[nfr] = pp; nfr++; pp_on = false;
        for (int i = 0; i < FEAT_N; i++) {
            Feat& f = feats[i]; if (!f.active) continue;
            if (f.wf < 0) { f.wf = wf_alloc(i); }
            if (OBS_GATE > 0 && f.wf >= 0 && !obs_gate_ok(wf[f.wf], f)) { f.active = false; wf[f.wf].live = -1; f.wf = -1; continue; }
            if (MAH && f.wf >= 0 && !mah_gate_ok(f.wf, f)) { f.active = false; wf[f.wf].live = -1; f.wf = -1; continue; }
            if (f.wf >= 0) wf_add_obs(f.wf, f);
        }
        if (nflow >= 8) {
            const float med = derotated_median();
            bool still = med < ZV_PX;
            if (ZV_WIN > 0) { const float wd = window_disparity(); if (wd >= 0) still = wd < ZV_WIN; }
            const double wn = std::sqrt(w_last[0] * w_last[0] + w_last[1] * w_last[1] + w_last[2] * w_last[2]);
            fr[nfr - 1].zv = still && wn < ZV_GYR;
        }
        for (int w = 0; w < MAX_WF; w++) {
            WF& Fw = wf[w]; if (!Fw.used || Fw.lam_ok || Fw.nobs < 2) continue;
            st_tri++;
            if (!triangulate(Fw) && DPF_SIG > 0 && DPF_INIT && Fw.nobs >= DPF_MINOBS) { Fw.lam = std::exp(-DPF_C); Fw.lam_ok = true; }
        }
        pipe_kf = nfr < 3 || second_newest_is_kf();
        map_search();
        pipe_F = F; pipe_ok = true;
        sc_start();
    }
    const int last_ph = PIPE_SCHED == 2 ? sc_len - 1 : PIPE - 1;
    const bool det_here = (PIPE_DET == 0 && pipe_ph == last_ph) || (PIPE_DET == 1 && pipe_ph == 0) || (PIPE_DET == 2 && pipe_ph == 1) || PIPE_DET == 3;
    if (PIPE_SCHED == 1) sc_work(sc_frac(pipe_ph));
    Frame Fc; predict_frame(fr[nfr - 1], pp, Fc);
    if (pipe_ph == 0) Fc = fr[nfr - 1];
    pipe_templates(Fc.R);
    if (det_here) { det_obs = pipe_ph == 0; detect(); det_obs = true; }
    if (PIPE_SCHED == 2) {   // fixed work of this frame first, then the chain up to a common level
        int na = 0; for (int i = 0; i < FEAT_N; i++) na += feats[i].active;
        const double fx_k = (pipe_ph == 0 ? CU_PH0 : (pipe_ph == last_ph ? CU_PHL : CU_MID)) + CU_TRACK * na
                          + CU_CELL * (st_cells - c0) + CU_CFULL * (st_cfull - cf0) + CU_SEARCH * (st_map_try - m0) + CU_DET * (st_det - d0);
        if (pipe_ph == 0) {   // cycle length and per-frame level (fixed work of later frames as in the previous cycle)
            int len = PIPE;
            for (;;) {
                sc_level = (sc_total + fx_k + (len - 2) * sc_fixed_prev[1] + sc_fixed_prev[2] + PIPE_RES) / len;
                if (sc_level <= PIPE_CAP || len >= PIPE_MAX) break;
                len++;
            }
            sc_len = len; sc_target = 0;
        }
        const int role = pipe_ph == 0 ? 0 : (pipe_ph == sc_len - 1 ? 2 : 1);
        sc_fixed_prev[role] = fx_k;
        double b = sc_level - fx_k; if (b < 0) b = 0;
        sc_target += b;
        sc_work(pipe_ph == sc_len - 1 ? 1e300 : sc_target);
        predict_frame(fr[nfr - 1], pp, Fc); if (pipe_ph == 0) Fc = fr[nfr - 1];
    }
    if (pipe_ph == last_ph) { map_create_at(Fc); marg_now = false; }
    output_frame(Fc, out);
#ifndef __wasm__
    if (std::getenv("VINS_SCLOG")) { int na = 0; for (int i = 0; i < FEAT_N; i++) na += feats[i].active;
        std::fprintf(stderr, "SC %d %d %d %ld %ld %ld %ld", frame_no, pipe_ph, na, st_cells - c0, st_map_try - m0, st_det - d0, st_cfull - cf0);
        for (int i = 0; i < SC_DONE; i++) std::fprintf(stderr, " %.0f", sc_log[i]); std::fprintf(stderr, "\n"); }
#endif
    pipe_ph = (pipe_ph + 1) % (PIPE_SCHED == 2 ? sc_len : PIPE);
}
#endif
static void step_pipe(const double* imu, int32_t nimu, double* out) {
    {   const Frame& N = fr[nfr - 1];
        if (!pp_on) { pre_reset(pp, N.bg, N.ba); pp_on = true; pp_dR_last = eye3(); }
        const double t_before = t_last;
        for (int k = 0; k < nimu; k++) {
            const double* r = imu + 7 * k;
            const double dt = r[0] - t_last;
            if (dt <= 0 || dt > 0.1) { t_last = r[0] > t_last ? r[0] : t_last; continue; }
            t_last = r[0];
            pre_add(pp, dt, r + 1);
            for (int i = 0; i < 3; i++) w_last[i] = r[4 + i] - N.bg[i];
        }
        if (t_last > t_before) frame_dt = t_last - t_before;
    }
    const M3 dRi = mmul(mtr(pp_dR_last), pp.dR); pp_dR_last = pp.dR;   // rotation since the previous camera frame
    const M3 Rcc = mmul(mmul(Rci, mtr(dRi)), mtr(Rci));
    if (pipe_ph == 0) {   // ---- frame A: new window frame, build, first part of the factorization
        Frame F; predict_frame(fr[nfr - 1], pp, F);
        F.serial = frame_no; F.t = t_last; F.span = pp.T > 0 ? pp.T : PIPE * frame_dt; F.zv = false; F.mo_n = 0;
#if MANHATTAN
        F.mh_on = false;
#endif
        {   double sig = VPRIOR_SIG;
            if (VP_ADAPT) { sig = VP_K * std::sqrt(v2_avg); if (sig < VP_MIN) sig = VP_MIN; if (sig > VPRIOR_SIG) sig = VPRIOR_SIG; }
            F.vsig = sig; }
        track(Rcc);
        fr[nfr] = F; pre[nfr] = pp; nfr++; pp_on = false;
        for (int i = 0; i < FEAT_N; i++) {
            Feat& f = feats[i]; if (!f.active) continue;
            if (f.wf < 0) { f.wf = wf_alloc(i); }
            if (OBS_GATE > 0 && f.wf >= 0 && !obs_gate_ok(wf[f.wf], f)) { f.active = false; wf[f.wf].live = -1; f.wf = -1; continue; }
            if (MAH && f.wf >= 0 && !mah_gate_ok(f.wf, f)) { f.active = false; wf[f.wf].live = -1; f.wf = -1; continue; }
            if (f.wf >= 0) wf_add_obs(f.wf, f);
        }
        if (nflow >= 8) {
            const float med = derotated_median();
            bool still = med < ZV_PX;
            if (ZV_WIN > 0) { const float wd = window_disparity(); if (wd >= 0) still = wd < ZV_WIN; }
            const double wn = std::sqrt(w_last[0] * w_last[0] + w_last[1] * w_last[1] + w_last[2] * w_last[2]);
            fr[nfr - 1].zv = still && wn < ZV_GYR;
        }
        for (int w = 0; w < MAX_WF; w++) {
            WF& Fw = wf[w]; if (!Fw.used || Fw.lam_ok || Fw.nobs < 2) continue;
            st_tri++;
            if (!triangulate(Fw) && DPF_SIG > 0 && DPF_INIT && Fw.nobs >= DPF_MINOBS) { Fw.lam = std::exp(-DPF_C); Fw.lam_ok = true; }
        }
        pipe_kf = nfr < 3 || second_newest_is_kf();
        if (PIPE_SEARCH_ALL || !(MAP_SKIP_MARG && pipe_kf && nfr == WIN)) map_search();
        pipe_F = F;
        pipe_ok = gn_begin();
        pipe_templates(fr[nfr - 1].R);
        if (PIPE_DET) { det_obs = true; detect(); }
        output(out);
        pipe_ph = 1;
    } else if (pipe_ph < PIPE - 1) {   // ---- middle frame (PIPE 3): track, more of the factorization
        track(Rcc);
        if (pipe_ok) pipe_ok = gn_middle();
        Frame Fm; predict_frame(fr[nfr - 1], pp, Fm);
        pipe_templates(Fm.R);
        if (PIPE_DET == 2) { det_obs = false; detect(); det_obs = true; }
        output_frame(Fm, out);
        pipe_ph++;
    } else {               // ---- frame B: finish the solve, marginalize, refill
        track(Rcc);
        const bool ok = pipe_ok && gn_finish();
        st_opt++;
        if (ok) map_check();
        const Frame& Nw = fr[nfr - 1];
        const double sp = std::sqrt(Nw.v[0] * Nw.v[0] + Nw.v[1] * Nw.v[1] + Nw.v[2] * Nw.v[2]);
        bool finite = true; for (int i = 0; i < 3; i++) finite = finite && std::isfinite(Nw.p[i]) && std::isfinite(Nw.v[i]);
        if (!ok || !finite || sp > VMAX) {
            if (!finite) { fr[nfr - 1] = pipe_F; }
#ifndef __wasm__
            if (std::getenv("VINS_RESETS")) std::fprintf(stderr, "reset f %d ok %d speed %.2f\n", frame_no, (int)ok, sp);
#endif
            reset_window();
        } else if (nfr >= 3) {
            if (!pipe_kf) drop_second_newest();
            else if (nfr == WIN) { marginalize_oldest(); marg_now = true; }
        }
        Frame Fb; predict_frame(fr[nfr - 1], pp, Fb);
        pipe_templates(Fb.R);
        if (!PIPE_DET) { det_obs = false; detect(); det_obs = true; }
        map_create_at(Fb);
        marg_now = false;
        {   int nu = 0; for (int w = 0; w < MAX_WF; w++) nu += fb_on[w];
            if (nu >= V2_MINF) v2_avg += (Fb.v[0] * Fb.v[0] + Fb.v[1] * Fb.v[1] + Fb.v[2] * Fb.v[2] - v2_avg) * (PIPE * frame_dt / VP_TAU); }
        output_frame(Fb, out);
        pipe_ph = 0;
    }
}
#endif

extern "C" {
VIOBENCH_EXPORT(abi_version) int32_t abi_version(void) { return VIOBENCH_ABI_VERSION; }
VIOBENCH_EXPORT(alloc) void* alloc(uint32_t size) { return std::malloc(size); }
VIOBENCH_EXPORT(init) int32_t init(const char* cfg, uint32_t n) {
    W = (int)num_after(cfg, n, "\"width\"", 320); H = (int)num_after(cfg, n, "\"height\"", 240);
    fx = num_after(cfg, n, "\"fx\"", 120); fy = num_after(cfg, n, "\"fy\"", 120);
    cx = num_after(cfg, n, "\"cx\"", 159.5); cy = num_after(cfg, n, "\"cy\"", 119.5);
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
    std::memset(feats, 0, sizeof feats); std::memset(cool, 0, sizeof cool); std::memset(wf, 0, sizeof wf);
    for (int i = 0; i < FEAT_N; i++) feats[i].wf = -1;
    ie_a2 = ie_g2 = 0; ie_n = 0; ie_have = false; ie_bias_done = false; sa2 = SIG_ACC * SIG_ACC; sg2 = SIG_GYR * SIG_GYR;
    dpf_lz0 = std::log(DPF_Z0); imu_prev_ok = false;
#if MANHATTAN
    mh_C = mh_S = mh_W = 0; mh_ref_ok = mh_basis = false;
#endif
#if PIPE
    pp_on = false; pipe_ph = 0; pipe_ok = false; det_obs = true;
#if PIPE_SCHED
    sc_st = SC_DONE; sc_len = PIPE;
#endif
#endif
    nfr = 0; frame_no = 0; started = false; map_rr = 0; n_map_cand = 0; for (int i = 0; i < (MAP ? MAP_N : 1); i++) mp[i].used = false; depth_avg = 2.5; v2_avg = 0.25; n_resets = 0; last_reset_frame = -100000; det_rr = 0;
    return VIOBENCH_STATUS_OK;
}

VIOBENCH_EXPORT(step) int32_t step(double t, const uint8_t* frame, const double* imu, int32_t nimu, double* out) {
    img = frame;
    if (!started) {
        double a[3] = {0, 0, 0}; int k = 0;
        for (int i = 0; i < nimu; i++) { a[0] += imu[7 * i + 1]; a[1] += imu[7 * i + 2]; a[2] += imu[7 * i + 3]; k++; }
        double na = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        if (k == 0 || na < 1e-6) { a[0] = 0; a[1] = 0; a[2] = 1; na = 1; }
        Frame& F = fr[0];
        F.R = eye3();
        for (int i = 0; i < 3; i++) { gw[i] = -9.81 * a[i] / na; F.p[i] = F.v[i] = F.bg[i] = F.ba[i] = 0; w_last[i] = 0; }
        t_last = nimu > 0 ? imu[7 * (nimu - 1)] : t;
        if (t_last > t) t_last = t;
        F.serial = 0; F.t = t_last; F.span = 0.05; F.vsig = VPRIOR_SIG; F.zv = false; F.mo_n = 0;
#if MANHATTAN
    F.mh_on = false;
#endif

        nfr = 1;
        reset_prior(SIG_V0, SIG_BG0, SIG_BA0, SIG_TILT0);
        started = true;
        Rh[0] = F.R;
        detect();
        output(out);
        frame_no++;
        return VIOBENCH_STATUS_OK;
    }
#if PIPE
#if PIPE_SCHED
    step_sched(imu, nimu, out);
#else
    step_pipe(imu, nimu, out);
#endif
    frame_no++; return VIOBENCH_STATUS_OK;
#endif
    // IMU: preintegrate from the newest frame
    const Frame& N = fr[nfr - 1];
    Pre P; pre_reset(P, N.bg, N.ba);
    const double t_before = t_last;
    if (IMU_EST) {   // noise levels for this frame's preintegration from the statistics so far
        double ea = (IMU_EST_N0 * SIG_ACC * SIG_ACC + ie_a2 / 6) / (IMU_EST_N0 + ie_n), eg = (IMU_EST_N0 * SIG_GYR * SIG_GYR + ie_g2 / 6) / (IMU_EST_N0 + ie_n);
        const double hi = IMU_EST_MAX * IMU_EST_MAX;
        if (ea > hi * SIG_ACC * SIG_ACC) ea = hi * SIG_ACC * SIG_ACC;
        if (eg > hi * SIG_GYR * SIG_GYR) eg = hi * SIG_GYR * SIG_GYR;
        sa2 = ea < IMU_EST_TH * IMU_EST_TH * SIG_ACC * SIG_ACC ? SIG_ACC * SIG_ACC : ea;
        sg2 = eg < IMU_EST_TH * IMU_EST_TH * SIG_GYR * SIG_GYR ? SIG_GYR * SIG_GYR : eg;
        if (IMU_EST_BIAS && !ie_bias_done && ie_n >= IMU_EST_BN) {
            ie_bias_done = true;
            const double kk[2] = {std::sqrt(sg2) / SIG_GYR, std::sqrt(sa2) / SIG_ACC};
            for (int q = 0; q < 2; q++) if (kk[q] > 1.0001) for (int i = 0; i < 3; i++) {
                const int c = q == 0 ? BGI(0) + i : BAI(0) + i, n = NSYS();
                for (int j = 0; j <= c; j++) Hp[c * NX + j] /= kk[q]; for (int j = c; j < n; j++) Hp[j * NX + c] /= kk[q];
                gp[c] /= kk[q];
            }
        }
    }
    for (int k = 0; k < nimu; k++) {
        const double* r = imu + 7 * k;
        const double dt = r[0] - t_last;
        if (IMU_EST) {
            if (ie_have) { for (int i = 0; i < 3; i++) { const double da = r[1 + i] - ie_prev[i], dg = r[4 + i] - ie_prev[3 + i]; ie_a2 += da * da; ie_g2 += dg * dg; } ie_n++; }
            for (int i = 0; i < 6; i++) ie_prev[i] = r[1 + i]; ie_have = true;
        }
        if (dt <= 0 || dt > 0.1) { t_last = r[0] > t_last ? r[0] : t_last; continue; }
        t_last = r[0];
        if (IMU_MID) {   // midpoint integration (VINS-Mono): average of this and the previous sample
            double xm[6]; for (int i = 0; i < 6; i++) xm[i] = imu_prev_ok ? 0.5 * (r[1 + i] + imu_prev[i]) : r[1 + i];
            pre_add(P, dt, xm);
        } else pre_add(P, dt, r + 1);
        if (IMU_MID) { for (int i = 0; i < 6; i++) imu_prev[i] = r[1 + i]; imu_prev_ok = true; }
        for (int i = 0; i < 3; i++) w_last[i] = r[4 + i] - N.bg[i];
    }
    if (t_last > t_before) frame_dt = t_last - t_before;
    // new frame predicted from the newest
    Frame F = N;
    F.R = mmul(N.R, P.dR); orthonormalize(F.R);
    { double a[3], b[3]; mv(N.R, P.dv, a); mv(N.R, P.dp, b);
      for (int i = 0; i < 3; i++) { F.p[i] = N.p[i] + N.v[i] * P.T + 0.5 * gw[i] * P.T * P.T + b[i]; F.v[i] = N.v[i] + gw[i] * P.T + a[i]; } }
    F.serial = frame_no; F.t = t_last; F.span = P.T > 0 ? P.T : frame_dt; F.zv = false; F.mo_n = 0;
#if MANHATTAN
    F.mh_on = false;
#endif

    {   double sig = VPRIOR_SIG;
        if (VP_ADAPT) { sig = VP_K * std::sqrt(v2_avg); if (sig < VP_MIN) sig = VP_MIN; if (sig > VPRIOR_SIG) sig = VPRIOR_SIG; }
        F.vsig = sig; }
    // track with the gyro rotation
    const M3 Rcc = mmul(mmul(Rci, mtr(P.dR)), mtr(Rci));
    trk_pred = &F;
    track(Rcc);
    fr[nfr] = F; pre[nfr] = P; nfr++;
    for (int i = 0; i < FEAT_N; i++) {
        Feat& f = feats[i]; if (!f.active) continue;
        if (f.wf < 0) { f.wf = wf_alloc(i); }
        if (OBS_GATE > 0 && f.wf >= 0 && !obs_gate_ok(wf[f.wf], f)) { f.active = false; wf[f.wf].live = -1; f.wf = -1; continue; }
        if (MAH && f.wf >= 0 && !mah_gate_ok(f.wf, f)) { f.active = false; wf[f.wf].live = -1; f.wf = -1; continue; }
        if (f.wf >= 0) wf_add_obs(f.wf, f);
    }
    // stillness -> zero-velocity factor on the newest frame
    if (nflow >= 8) {
        const float med = derotated_median();
        bool still = med < ZV_PX;
        if (ZV_WIN > 0) { const float wd = window_disparity(); if (wd >= 0) still = wd < ZV_WIN; }
        const double wn = std::sqrt(w_last[0] * w_last[0] + w_last[1] * w_last[1] + w_last[2] * w_last[2]);
        fr[nfr - 1].zv = still && wn < ZV_GYR;
#ifndef __wasm__
        if (std::getenv("VINS_ZV") && frame_no < std::atoi(std::getenv("VINS_ZV"))) std::fprintf(stderr, "zv f %d med %.2f wd %.2f wn %.2f -> %d\n", frame_no, med, ZV_WIN > 0 ? window_disparity() : -1.0f, wn, (int)fr[nfr - 1].zv);
#endif
    }
    for (int w = 0; w < MAX_WF; w++) {
        WF& Fw = wf[w]; if (!Fw.used || Fw.lam_ok || Fw.nobs < 2) continue;
        st_tri++;
        if (!triangulate(Fw) && DPF_SIG > 0 && DPF_INIT && Fw.nobs >= DPF_MINOBS) { Fw.lam = std::exp(-DPF_C); Fw.lam_ok = true; }
    }
    // keyframe decision before the solve (it compares the two previous frames): a frame that will marginalize skips the map search
    const bool kf_pre = KF_PRE ? (nfr < 3 || second_newest_is_kf()) : false;
    if (!(KF_PRE && MAP_SKIP_MARG && kf_pre && nfr == WIN)) map_search();
    // optimize
#ifndef __wasm__
    if (std::getenv("VINS_GCHECK") && frame_no == std::atoi(std::getenv("VINS_GCHECK"))) grad_check();
#endif
    bool ok = true;
#ifndef __wasm__
    const bool dbgc = std::getenv("VINS_COST") && frame_no < std::atoi(std::getenv("VINS_COST"));
    if (dbgc) std::fprintf(stderr, "f %d nfr %d cost %.2f", frame_no, nfr, total_cost());
#endif
    mc_n = 0; mc_valid = false; mc_on = MARG_CACHE && KF_PRE && kf_pre && nfr == WIN;
    const int nit = (NIT2_LIGHT && KF_PRE && !(kf_pre && nfr == WIN)) ? 2 : NIT;   // frames that do not marginalize can afford a second iteration
    for (int it = 0; it < nit && ok; it++) {
        ok = gn_iteration(it == nit - 1, CHORD && it > 0);
#ifndef __wasm__
        if (dbgc) std::fprintf(stderr, " -> %.2f", total_cost());
#endif
    }
#ifndef __wasm__
    if (dbgc) std::fprintf(stderr, "\n");
#endif
    st_opt++;
    mc_valid = mc_on && ok; mc_on = false;
    if (ok) map_check();
    const Frame& Nw = fr[nfr - 1];
    const double sp = std::sqrt(Nw.v[0] * Nw.v[0] + Nw.v[1] * Nw.v[1] + Nw.v[2] * Nw.v[2]);
    bool finite = true; for (int i = 0; i < 3; i++) finite = finite && std::isfinite(Nw.p[i]) && std::isfinite(Nw.v[i]);
    if (!ok || !finite || sp > VMAX) {
        if (!finite) { fr[nfr - 1] = F; }
#ifndef __wasm__
        if (std::getenv("VINS_RESETS")) std::fprintf(stderr, "reset f %d ok %d speed %.2f\n", frame_no, (int)ok, sp);
#endif
        reset_window();
    } else if (nfr >= 3) {
        if (!(KF_PRE ? kf_pre : second_newest_is_kf())) drop_second_newest();
        else if (nfr == WIN) { marginalize_oldest(); marg_now = true; }
    }
    // templates of the live tracks, detection
    for (int i = 0; i < FEAT_N; i++) {
        Feat& f = feats[i]; if (!f.active) continue;
        if (!cut_template(f)) { f.active = false; if (f.wf >= 0) wf[f.wf].live = -1; f.wf = -1; continue; }
        f.hx[frame_no & 15] = f.x; f.hy[frame_no & 15] = f.y;
    }
    Rh[frame_no & 15] = fr[nfr - 1].R;
    if (!(DEFER_MARG && marg_now)) { map_create(); detect(); }   // a marginalizing frame leaves map points and detection to the next one
    marg_now = false;
    manhattan_step();
    {   const Frame& L = fr[nfr - 1];
        int nu = 0; for (int w = 0; w < MAX_WF; w++) nu += fb_on[w];
        if (nu >= V2_MINF) v2_avg += (L.v[0] * L.v[0] + L.v[1] * L.v[1] + L.v[2] * L.v[2] - v2_avg) * (frame_dt / VP_TAU); }
    output(out);
#ifndef __wasm__
    if (MAH && std::getenv("VINS_MAHSTATS") && frame_no == 1199) std::fprintf(stderr, "mah tests %ld rejected %ld\n", st_mah_test, st_mah_rej);
    if (MAP && std::getenv("VINS_MAPSTATS") && frame_no == 1199) std::fprintf(stderr, "map made %ld tries %ld hits %ld out %ld\n", st_map_made, st_map_try, st_map_hit, st_map_out);
    if (std::getenv("VINS_FS")) { int nu = 0; for (int w = 0; w < MAX_WF; w++) nu += fb_on[w]; static long pm = 0, pd = 0; static long pt = 0, pmk = 0, pr = 0; int nact = 0; for (int i = 0; i < FEAT_N; i++) nact += feats[i].active; int nwf = 0; for (int w = 0; w < MAX_WF; w++) nwf += wf[w].used;
        static long pc = 0, ptr = 0, pde = 0;
        std::fprintf(stderr, "FS %d %d %d %ld %ld %ld %ld %d %d %d %ld %ld %ld\n", frame_no, nfr, nu, st_marg - pm, st_drop - pd, st_map_try - pt, st_map_made - pmk, n_resets - (int)pr, nact, nwf, st_cells - pc, st_tri - ptr, st_det - pde); pm = st_marg; pd = st_drop; pt = st_map_try; pmk = st_map_made; pr = n_resets; pc = st_cells; ptr = st_tri; pde = st_det; }
    if (std::getenv("VINS_DEBUG") && frame_no % std::atoi(std::getenv("VINS_DEBUG")) == 0) {
        int nu = 0, nl = 0; for (int w = 0; w < MAX_WF; w++) { nu += wf_usable(wf[w]); nl += wf[w].used; }
        const Frame& L = fr[nfr - 1];
        std::fprintf(stderr, "f %d zv %d g %.2f %.2f %.2f nfr %d span %.2f wf %d usable %d v %.2f %.2f %.2f bg %.3f %.3f %.3f ba %.2f %.2f %.2f imu_cost %.1f resets %d drop %ld marg %ld out %ld\n",
            frame_no, (int)L.zv, gw[0], gw[1], gw[2], nfr, L.t - fr[0].t, nl, nu, L.v[0], L.v[1], L.v[2], L.bg[0], L.bg[1], L.bg[2], L.ba[0], L.ba[1], L.ba[2], imu_cost, n_resets, st_drop, st_marg, st_out);
    }
#endif
    frame_no++;
    return VIOBENCH_STATUS_OK;
}
}

extern "C" int32_t vins_frames_since_reset(void) { return frame_no - last_reset_frame; }
extern "C" int32_t vins_active_tracks(void) { int n = 0; for (int i = 0; i < FEAT_N; i++) n += feats[i].active; return n; }
