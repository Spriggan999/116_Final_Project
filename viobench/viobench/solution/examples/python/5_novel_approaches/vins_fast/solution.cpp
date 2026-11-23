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
#ifndef LEARN_P
#define LEARN_P 0                // 1: learned noise weights (lp_weights.h, CMA-ES): 9 log multipliers, each constant + slope x (running
#endif                           //   speed - 0.5 m/s): pixel noise, depth-prior sigma, depth-prior centre (additive ln), accel noise,
                                 //   gyro noise, velocity prior, zero-velocity sigma, outlier threshold, bias random-walk floor
#ifndef LP_GATE
#define LP_GATE 0                // 1 (with LEARN_P): the learned weights apply only as far as the IMU is noisier than nominal:
#endif                           //   gate = clamp((k - LP_K0) / (LP_K1 - LP_K0), 0, 1), k = measured accel noise / SIG_ACC (router_dpn's test)
#ifndef LP_K0
#define LP_K0 1.3
#endif
#ifndef LP_K1
#define LP_K1 1.8
#endif
#ifndef NN_DEPTH
#define NN_DEPTH 0               // learned depth prior (nn_depth_weights.h): each corner's log-depth prior centre is predicted from its
#endif                           //   elevation below the horizon, image position, patch texture, height, speeds, FOV. 1: linear model, 2: MLP
#ifndef NN_OFF
#define NN_OFF 0.2               // added to the predicted ln depth (targets are well-triangulated features, nearer than average)
#endif
#ifndef NN_SIG
#define NN_SIG 1.0               // prior sigma (ln depth) of features with a learned centre
#endif
#ifndef NN_HET
#define NN_HET 0                 // 1 (NN_DEPTH 2): sigma = NN_SIG x predicted sigma / 0.39 (the median prediction)
#endif
#ifndef NN_EDGE
#define NN_EDGE 0                // 1: edgelets use the learned centre too (trained on corners only)
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
#ifndef NO_UPPER0
#define NO_UPPER0 1              // 1: base_prior leaves the upper triangle of H as it is (never read)
#endif
#ifndef VARPRO_FAST
#define VARPRO_FAST 1            // 1: inverse-depth refinement with the camera point affine in 1/lam (same maths, fewer operations)
#endif
#ifndef DET_FAST
#define DET_FAST 1               // 1: detection scores from exact integer SIMD sums (bit-identical)
#endif
#ifndef MEM_OWN
#define MEM_OWN 1                // 1: own SIMD memset/memcpy/memmove (needs -mno-bulk-memory; identical results)
#endif
#ifndef FEAT_FAST
#define FEAT_FAST 1              // 1: feature 6x6 blocks two columns at a time (bit-identical)
#endif
#ifndef CHOL8
#define CHOL8 1                  // 1: Cholesky columns 8 rows per pass, 8 elements per iteration (bit-identical)
#endif
#ifndef BASE_FUSE
#define BASE_FUSE 1                // 1: base_prior in one fused SIMD pass per row (bit-identical)
#endif
#ifndef TPL_FAST
#define TPL_FAST 2               // 1: SIMD template cut (bit-identical)
#endif
#ifndef SCATTER_FAST
#define SCATTER_FAST 1           // 1: feature Hessian blocks scattered two columns at a time (bit-identical)
#endif
#ifndef SCHUR_FAST
#define SCHUR_FAST 1             // 1: Schur rows of a 9-state elimination with all 9 updates in registers (bit-identical)
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
#ifndef CU_MPK
#define CU_MPK 17.0              // marginalization prep: per NF^2 n
#endif
#ifndef CU_MPC
#define CU_MPC 12000.0
#endif
#ifndef CU_MFK
#define CU_MFK 0.6               // marginalization finish: per n^2
#endif
#ifndef CU_MFC
#define CU_MFC 5000.0
#endif
#ifndef CU_DRC
#define CU_DRC 40000.0           // drop with a prior Schur step: fixed part
#endif
#ifndef CU_SOLC
#define CU_SOLC 20000.0          // step: fixed part
#endif
#ifndef CU_DAMPC
#define CU_DAMPC 2000.0          // damping
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
static const int G0 = SHARED_BIAS ? 8 : 2;       // gravity (2) [+ bg, ba (6)] come first
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
#if MEM_OWN && defined(__wasm_simd128__) && !defined(__wasm_bulk_memory__)
// memset / memcpy / memmove as v128 loops: the sandbox charges bulk memory ops about one fuel per byte, these cost about
// a quarter of that (build with -mno-bulk-memory so that the compiler calls these instead of emitting memory.fill/copy)
extern "C" __attribute__((no_builtin)) void* memset(void* d, int c, size_t n) {
    unsigned char* p = (unsigned char*)d; const v128_t v = wasm_i8x16_splat((int8_t)c);
    for (; n >= 64; n -= 64, p += 64) { wasm_v128_store(p, v); wasm_v128_store(p + 16, v); wasm_v128_store(p + 32, v); wasm_v128_store(p + 48, v); }
    for (; n >= 16; n -= 16, p += 16) wasm_v128_store(p, v);
    for (; n >= 8; n -= 8, p += 8) wasm_v128_store64_lane(p, v, 0);
    for (; n > 0; n--) *p++ = (unsigned char)c;
    return d;
}
extern "C" __attribute__((no_builtin)) void* memcpy(void* __restrict d, const void* __restrict s, size_t n) {
    unsigned char* p = (unsigned char*)d; const unsigned char* q = (const unsigned char*)s;
    for (; n >= 64; n -= 64, p += 64, q += 64) {
        const v128_t a = wasm_v128_load(q), b = wasm_v128_load(q + 16), c = wasm_v128_load(q + 32), e = wasm_v128_load(q + 48);
        wasm_v128_store(p, a); wasm_v128_store(p + 16, b); wasm_v128_store(p + 32, c); wasm_v128_store(p + 48, e);
    }
    for (; n >= 16; n -= 16, p += 16, q += 16) wasm_v128_store(p, wasm_v128_load(q));
    for (; n >= 8; n -= 8, p += 8, q += 8) { uint64_t t; __builtin_memcpy_inline(&t, q, 8); __builtin_memcpy_inline(p, &t, 8); }
    for (; n > 0; n--) *p++ = *q++;
    return d;
}
extern "C" __attribute__((no_builtin)) void* memmove(void* d, const void* s, size_t n) {
    unsigned char* p = (unsigned char*)d; const unsigned char* q = (const unsigned char*)s;
    if (p == q || n == 0) return d;
    if (p < q || p >= q + n) {   // forward (16 bytes are loaded before they are stored: safe for p < q)
        for (; n >= 16; n -= 16, p += 16, q += 16) wasm_v128_store(p, wasm_v128_load(q));
        for (; n > 0; n--) *p++ = *q++;
    } else {                      // backward
        p += n; q += n;
        for (; n >= 16; n -= 16) { p -= 16; q -= 16; wasm_v128_store(p, wasm_v128_load(q)); }
        for (; n > 0; n--) *--p = *--q;
    }
    return d;
}
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
#if defined(__wasm_simd128__) && CHOL8
// R rows (a, a+LD, ...) dot b: per row one f64x2 accumulator fed chunk by chunk in k order (the same sums as dot4s,
// bit-identical), eight elements per iteration with the chunk loads folded into constant offsets (fewer wasm instructions)
template <int LD, int R> static inline void dotRs(const double* a, const double* b, int n, double* o) {
    v128_t s[R]; for (int t = 0; t < R; t++) s[t] = wasm_f64x2_splat(0.0);
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        const double* p = a + k; const double* q = b + k;
        const v128_t b0 = wasm_v128_load(q), b1 = wasm_v128_load(q + 2), b2 = wasm_v128_load(q + 4), b3 = wasm_v128_load(q + 6);
        for (int t = 0; t < R; t++) {
            v128_t x = s[t];
            x = wasm_f64x2_add(x, wasm_f64x2_mul(wasm_v128_load(p + t * LD), b0));
            x = wasm_f64x2_add(x, wasm_f64x2_mul(wasm_v128_load(p + t * LD + 2), b1));
            x = wasm_f64x2_add(x, wasm_f64x2_mul(wasm_v128_load(p + t * LD + 4), b2));
            x = wasm_f64x2_add(x, wasm_f64x2_mul(wasm_v128_load(p + t * LD + 6), b3));
            s[t] = x;
        }
    }
    for (; k + 2 <= n; k += 2) {
        const v128_t b0 = wasm_v128_load(b + k);
        for (int t = 0; t < R; t++) s[t] = wasm_f64x2_add(s[t], wasm_f64x2_mul(wasm_v128_load(a + t * LD + k), b0));
    }
    for (int t = 0; t < R; t++) { double r = wasm_f64x2_extract_lane(s[t], 0) + wasm_f64x2_extract_lane(s[t], 1); if (k < n) r += a[t * LD + k] * b[k]; o[t] = r; }
}
#endif
// dense Cholesky with row stride LD, 4 rows per column pass
template <int LD> static bool chol4r(double* A, int n, int j0, int j1) {   // columns j0..j1-1 (left-looking: needs columns < j0 done)
    for (int j = j0; j < j1; j++) {
        double* Aj = A + j * LD;
        const double d0 = Aj[j] - dotv(Aj, Aj, j);
        if (!(d0 > 1e-300)) return false;
        const double d = std::sqrt(d0); Aj[j] = d; const double id = 1.0 / d;
        int i = j + 1;
#if defined(__wasm_simd128__) && CHOL8
        for (; i + 8 <= n; i += 8) { double o[8]; double* Ai = A + i * LD; dotRs<LD, 8>(Ai, Aj, j, o); for (int t = 0; t < 8; t++) Ai[t * LD + j] = (Ai[t * LD + j] - o[t]) * id; }
        for (; i + 4 <= n; i += 4) { double o[4]; double* Ai = A + i * LD; dotRs<LD, 4>(Ai, Aj, j, o); for (int t = 0; t < 4; t++) Ai[t * LD + j] = (Ai[t * LD + j] - o[t]) * id; }
#endif
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
#if LEARN_P
#include "lp_weights.h"   // LP_W[18]: (constant, slope) per multiplier
static double lp_w[18], lp_m[9] = {1, 1, 0, 1, 1, 1, 1, 1, 1}; static bool lp_init = false;
#define SIGPX_ (SIG_PX * lp_m[0])
#define DPFSIG_ (DPF_SIG * lp_m[1])
#define ZVSIG_ (ZV_SIG * lp_m[6])
#define OUTPX_ (OUT_PX * lp_m[7])
#define VPK_ (VP_K * lp_m[5])
#define VPMIN_ (VP_MIN * lp_m[5])
#define BFG_ (BIAS_FLOOR_G * lp_m[8])
#define BFA_ (BIAS_FLOOR_A * lp_m[8])
#else
#define SIGPX_ SIG_PX
#define DPFSIG_ DPF_SIG
#define ZVSIG_ ZV_SIG
#define OUTPX_ OUT_PX
#define VPK_ VP_K
#define VPMIN_ VP_MIN
#define BFG_ BIAS_FLOOR_G
#define BFA_ BIAS_FLOOR_A
#endif
static double imu_prev[6]; static bool imu_prev_ok = false;
static double dpf_lz0 = 0.0;   // ln of the depth prior's centre (DPF_Z0, or adapted: DPF_ADAPT)
static double ie_a2 = 0, ie_g2 = 0, ie_prev[6]; static long ie_n = 0; static bool ie_have = false, ie_bias_done = false;
static double depth_avg = 2.5, v2_avg = 0.25;
static int n_resets = 0, last_reset_frame = -100000;
static long st_cells = 0, st_tri = 0, st_det = 0, st_cfull = 0;
static long st_opt = 0, st_out = 0, st_marg = 0, st_drop = 0, st_fail = 0;

struct WF {   // window feature
    bool used; int live;          // live track index or -1
    int nobs; int ser[WIN + 1]; float u[WIN + 1], v[WIN + 1], q[WIN + 1];   // q: LK residual of the observation
#if EDGE
    bool edge; float enu[WIN + 1], env[WIN + 1];   // edgelet: unit normal (pixels) of each observation
#endif
    double lam; bool lam_ok;
#if NN_DEPTH
    double dmu, dsig;   // learned depth prior: ln-depth centre (anchor camera) and sigma
#define DPF_LZ(F) ((F).dmu)
#define DPF_S(F) ((F).dsig)
#else
#define DPF_LZ(F) dpf_lz0
#define DPF_S(F) DPFSIG_
#endif
#ifndef __wasm__
    long uid;   // VINS_DLOG: feature id in the depth-prior training log
#endif
};
static WF wf[MAX_WF];
struct Feat {
    bool active; int wf;
    float x, y, fdx, fdy;
    float tf[(2 * PR + 1) * (2 * PR + 1)], gxf[(2 * PR + 1) * (2 * PR + 1)], gyf[(2 * PR + 1) * (2 * PR + 1)];
    float Hf[6];
    int born; float hx[16], hy[16]; float lkr;
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
#ifdef __wasm_simd128__
#define HAS_SIMD_ 1
#else
#define HAS_SIMD_ 0
#endif
static bool cut_template(Feat& f) {
    const int S = 2 * PR + 1, E = S + 2;
    if (!inside(f.x, f.y, 2 * PR + 3)) return false;
    double a = 0, b = 0, c = 0, d = 0, e = 0;
#if defined(__wasm_simd128__) && TPL_FAST && PR == 3
    {   // same float operations as grid_sample + the gradient loop, four columns at a time; the double sums stay sequential
        const float x0 = f.x - (PR + 1), y0 = f.y - (PR + 1);
        const int ix = (int)x0, iy = (int)y0; const float ax = x0 - ix, ay = y0 - iy;
        const float w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
        const uint8_t* base = img + (iy * W + ix) * 3 + 1;
        alignas(16) float G[10][16], g[9][12], tl[7][8], gxl[7][8], gyl[7][8], pxx[7][8], pxy[7][8], pyy[7][8];
        for (int j = 0; j < 10; j++) {   // green samples 0..10 of row j (as lk_fast)
            const uint8_t* p = base + j * 3 * W;
            const v128_t lo = wasm_v128_load(p), hi = wasm_v128_load(p + 16);
            const v128_t sg = wasm_i8x16_shuffle(lo, hi, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
            const v128_t h0 = wasm_u16x8_extend_low_u8x16(sg), h1 = wasm_u16x8_extend_high_u8x16(sg);
            wasm_v128_store(G[j], wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(h0)));
            wasm_v128_store(G[j] + 4, wasm_f32x4_convert_u32x4(wasm_u32x4_extend_high_u16x8(h0)));
            wasm_v128_store(G[j] + 8, wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(h1)));
            wasm_v128_store(G[j] + 12, wasm_f32x4_splat(0.0f));
        }
        const v128_t W00 = wasm_f32x4_splat(w00), W10 = wasm_f32x4_splat(w10), W01 = wasm_f32x4_splat(w01), W11 = wasm_f32x4_splat(w11);
        for (int j = 0; j < 9; j++)
            for (int i = 0; i < 12; i += 4) {
                const v128_t s0 = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(W00, wasm_v128_load(G[j] + i)), wasm_f32x4_mul(W10, wasm_v128_load(G[j] + i + 1))),
                                                                wasm_f32x4_mul(W01, wasm_v128_load(G[j + 1] + i))), wasm_f32x4_mul(W11, wasm_v128_load(G[j + 1] + i + 1)));
                wasm_v128_store(g[j] + i, s0);
            }
        const v128_t half = wasm_f32x4_splat(0.5f);
        for (int j = 0; j < 7; j++)
            for (int i = 0; i < 8; i += 4) {
                const v128_t q0 = wasm_v128_load(g[j + 1] + i + 1);
                const v128_t dx = wasm_f32x4_mul(half, wasm_f32x4_sub(wasm_v128_load(g[j + 1] + i + 2), wasm_v128_load(g[j + 1] + i)));
                const v128_t dy = wasm_f32x4_mul(half, wasm_f32x4_sub(wasm_v128_load(g[j + 2] + i + 1), wasm_v128_load(g[j] + i + 1)));
                wasm_v128_store(tl[j] + i, q0); wasm_v128_store(gxl[j] + i, dx); wasm_v128_store(gyl[j] + i, dy);
                wasm_v128_store(pxx[j] + i, wasm_f32x4_mul(dx, dx)); wasm_v128_store(pxy[j] + i, wasm_f32x4_mul(dx, dy)); wasm_v128_store(pyy[j] + i, wasm_f32x4_mul(dy, dy));
            }
#if TPL_FAST >= 2
        {   // stores: 7-float rows as two overlapping 4-lane stores; padded rows for LK_FAST with lane 7 zeroed
            const v128_t M3 = wasm_i32x4_make(-1, -1, -1, 0);
            for (int j = 0; j < 7; j++) {
                const v128_t t0 = wasm_v128_load(tl[j]), t1 = wasm_v128_load(tl[j] + 4), x0 = wasm_v128_load(gxl[j]), x1 = wasm_v128_load(gxl[j] + 4);
                const v128_t y0 = wasm_v128_load(gyl[j]), y1 = wasm_v128_load(gyl[j] + 4);
                wasm_v128_store(f.tf + 7 * j, t0); wasm_v128_store(f.tf + 7 * j + 3, wasm_v128_load(tl[j] + 3));
                wasm_v128_store(f.gxf + 7 * j, x0); wasm_v128_store(f.gxf + 7 * j + 3, wasm_v128_load(gxl[j] + 3));
                wasm_v128_store(f.gyf + 7 * j, y0); wasm_v128_store(f.gyf + 7 * j + 3, wasm_v128_load(gyl[j] + 3));
                if (LK_FAST) {
                    wasm_v128_store(f.t8 + 8 * j, t0); wasm_v128_store(f.t8 + 8 * j + 4, wasm_v128_and(t1, M3));
                    wasm_v128_store(f.gx8 + 8 * j, x0); wasm_v128_store(f.gx8 + 8 * j + 4, wasm_v128_and(x1, M3));
                    wasm_v128_store(f.gy8 + 8 * j, y0); wasm_v128_store(f.gy8 + 8 * j + 4, wasm_v128_and(y1, M3));
                }
            }
#if TPL_FAST >= 3
            // the five sums in f32x4 lanes (column-wise partial sums, about 1e-6 relative to the double sums), then in double
            {   v128_t SA = wasm_f32x4_splat(0.0f), SB = SA, SC = SA, SD = SA, SE = SA;
                for (int j = 0; j < 7; j++) {
                    SA = wasm_f32x4_add(SA, wasm_f32x4_add(wasm_v128_load(pxx[j]), wasm_v128_and(wasm_v128_load(pxx[j] + 4), M3)));
                    SB = wasm_f32x4_add(SB, wasm_f32x4_add(wasm_v128_load(pxy[j]), wasm_v128_and(wasm_v128_load(pxy[j] + 4), M3)));
                    SD = wasm_f32x4_add(SD, wasm_f32x4_add(wasm_v128_load(pyy[j]), wasm_v128_and(wasm_v128_load(pyy[j] + 4), M3)));
                    SC = wasm_f32x4_add(SC, wasm_f32x4_add(wasm_v128_load(gxl[j]), wasm_v128_and(wasm_v128_load(gxl[j] + 4), M3)));
                    SE = wasm_f32x4_add(SE, wasm_f32x4_add(wasm_v128_load(gyl[j]), wasm_v128_and(wasm_v128_load(gyl[j] + 4), M3)));
                }
                auto hs = [](v128_t v) { return ((double)wasm_f32x4_extract_lane(v, 0) + (double)wasm_f32x4_extract_lane(v, 1)) + ((double)wasm_f32x4_extract_lane(v, 2) + (double)wasm_f32x4_extract_lane(v, 3)); };
                a = hs(SA); b = hs(SB); c = -hs(SC); d = hs(SD); e = -hs(SE);
            }
            if (0)
#endif
            {
            // the five double sums, two per f64x2 (each lane keeps its own order): (a, d), (b, c), (e, -); c, e add the negation
            v128_t AD = wasm_f64x2_splat(0.0), BC = AD, EE = AD;
            for (int j = 0; j < 7; j++) {
                const v128_t P = wasm_v128_load(pxx[j]), Q = wasm_v128_load(pyy[j]), R = wasm_v128_load(pxy[j]), G = wasm_f32x4_neg(wasm_v128_load(gxl[j]));
                const v128_t Hy = wasm_f32x4_neg(wasm_v128_load(gyl[j]));
                const v128_t P1 = wasm_v128_load(pxx[j] + 4), Q1 = wasm_v128_load(pyy[j] + 4), R1 = wasm_v128_load(pxy[j] + 4), G1 = wasm_f32x4_neg(wasm_v128_load(gxl[j] + 4));
                const v128_t H1 = wasm_f32x4_neg(wasm_v128_load(gyl[j] + 4));
#define TS(PP, QQ, RR, GG, HH, l) { \
                    AD = wasm_f64x2_add(AD, wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(PP, QQ, l, 4 + l, 0, 0))); \
                    BC = wasm_f64x2_add(BC, wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(RR, GG, l, 4 + l, 0, 0))); \
                    EE = wasm_f64x2_add(EE, wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(HH, HH, l, l, 0, 0))); }
                TS(P, Q, R, G, Hy, 0) TS(P, Q, R, G, Hy, 1) TS(P, Q, R, G, Hy, 2) TS(P, Q, R, G, Hy, 3)
                TS(P1, Q1, R1, G1, H1, 0) TS(P1, Q1, R1, G1, H1, 1) TS(P1, Q1, R1, G1, H1, 2)
#undef TS
            }
            a = wasm_f64x2_extract_lane(AD, 0); d = wasm_f64x2_extract_lane(AD, 1);
            b = wasm_f64x2_extract_lane(BC, 0); c = wasm_f64x2_extract_lane(BC, 1); e = wasm_f64x2_extract_lane(EE, 0);
            }
        }
#else
        for (int j = 0; j < 7; j++) for (int i = 0; i < 7; i++) {
            f.tf[j * 7 + i] = tl[j][i]; f.gxf[j * 7 + i] = gxl[j][i]; f.gyf[j * 7 + i] = gyl[j][i];
            a += pxx[j][i]; b += pxy[j][i]; c -= gxl[j][i]; d += pyy[j][i]; e -= gyl[j][i];
        }
#endif
    }
#else
    float g[(2 * PR + 3) * (2 * PR + 3)];
    grid_sample(f.x - (PR + 1), f.y - (PR + 1), E, g);
    for (int j = 0; j < S; j++) for (int i = 0; i < S; i++) {
        const float* q = g + (j + 1) * E + (i + 1);
        float dx = 0.5f * (q[1] - q[-1]), dy = 0.5f * (q[E] - q[-E]);
        f.tf[j * S + i] = q[0]; f.gxf[j * S + i] = dx; f.gyf[j * S + i] = dy;
        a += dx * dx; b += dx * dy; c -= dx; d += dy * dy; e -= dy;
    }
#endif
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
    if (LK_FAST && PR == 3 && !(TPL_FAST >= 2 && HAS_SIMD_)) for (int j = 0; j < S; j++) for (int i = 0; i < 8; i++) {
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
    alignas(16) float g[8][12]; int gix = -1 << 30, giy = 0;
    for (int it = 0; it < iters; it++) {
        const float x0 = x - 3, y0 = y - 3;
        if (!inside(x0, y0, 1) || !inside(x0 + 6, y0 + 6, 1)) return -1;
        const int ix = (int)x0, iy = (int)y0; const float ax = x0 - ix, ay = y0 - iy;
        const float w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
        const uint8_t* base = img + (iy * W + ix) * 3 + 1;
        float bx, by, bo;
#ifdef __wasm_simd128__
        if (ix != gix || iy != giy) { gix = ix; giy = iy;   // the converted patch is reused while the integer position stays
        for (int j = 0; j < 8; j++) {
            const uint8_t* p = base + j * 3 * W;
            const v128_t lo = wasm_v128_load(p), hi = wasm_v128_load(p + 16);
            const v128_t sg = wasm_i8x16_shuffle(lo, hi, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
            const v128_t h0 = wasm_u16x8_extend_low_u8x16(sg), h1 = wasm_u16x8_extend_high_u8x16(sg);
            wasm_v128_store(g[j], wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(h0)));
            wasm_v128_store(g[j] + 4, wasm_f32x4_convert_u32x4(wasm_u32x4_extend_high_u16x8(h0)));
            wasm_v128_store(g[j] + 8, wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(h1)));
        }
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
static void track(const M3& Rcc) {
    nflow = 0;
    for (int i = 0; i < FEAT_N; i++) {
        Feat& f = feats[i]; if (!f.active) continue;
        double b[3] = {(f.x - cx) / fx, (f.y - cy) / fy, 1}, q[3]; mv(Rcc, b, q);
        bool ok = q[2] >= 0.1;
        float px = 0, py = 0, x = 0, y = 0;
        if (ok) {
            px = (float)(fx * q[0] / q[2] + cx); py = (float)(fy * q[1] / q[2] + cy);
            x = px + f.fdx; y = py + f.fdy; float off = 0;
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
            ok = r >= 0 && r <= MAX_RES; f.lkr = r;
        }
        if (!ok) { f.active = false; if (f.wf >= 0) wf[f.wf].live = -1; f.wf = -1; continue; }
        f.fdx = 0.7f * (x - px); f.fdy = 0.7f * (y - py);
        f.x = x; f.y = y;
#if EDGE
        if (f.edge) continue;   // edgelets: no 2-D flow (stillness, visual velocity)
#endif
        fl_u[nflow] = (float)(q[0] / q[2]); fl_v[nflow] = (float)(q[1] / q[2]);
        fl_du[nflow] = (float)((x - px) / fx); fl_dv[nflow] = (float)((y - py) / fy);
        flow[nflow++] = std::sqrt((x - px) * (x - px) + (y - py) * (y - py));
    }
    for (int i = 0; i < FEAT_N; i++) if (feats[i].active) for (int j = i + 1; j < FEAT_N; j++) if (feats[j].active) {
        float dx = feats[i].x - feats[j].x, dy = feats[i].y - feats[j].y;
        if (dx * dx + dy * dy < 4.0f) { feats[j].active = false; if (feats[j].wf >= 0) wf[feats[j].wf].live = -1; feats[j].wf = -1; }
    }
}

static bool obs_gate_ok(const WF& F, const Feat& f);
#if EDGE
#define EDGE_SHIFT(F, i) do { (F).enu[i] = (F).enu[(i) + 1]; (F).env[i] = (F).env[(i) + 1]; } while (0)
#else
#define EDGE_SHIFT(F, i) do {} while (0)
#endif
#ifndef __wasm__
static FILE* dlog = nullptr; static bool dlog_init = false; static long dlog_uid = 0;
static void dlog_feature(WF& F, const Feat& f) {   // VINS_DLOG=path: inputs of a new window feature (depth-prior training data)
    if (!dlog_init) { dlog_init = true; const char* p = std::getenv("VINS_DLOG"); if (p) dlog = std::fopen(p, "w"); }
    F.uid = dlog_uid++;
    if (!dlog || nfr < 1) return;
    const Frame& N = fr[nfr - 1];
    const double bc[3] = {(f.x - cx) / fx, (f.y - cy) / fy, 1}; double bi[3], bw[3]; mtv(Rci, bc, bi); mv(N.R, bi, bw);
    const double ng = std::sqrt(gw[0] * gw[0] + gw[1] * gw[1] + gw[2] * gw[2]), nb = std::sqrt(bw[0] * bw[0] + bw[1] * bw[1] + bw[2] * bw[2]);
    const double se = (bw[0] * gw[0] + bw[1] * gw[1] + bw[2] * gw[2]) / (ng * nb);
    const double zc[3] = {0, 0, 1}; double zi[3], zw[3]; mtv(Rci, zc, zi); mv(N.R, zi, zw);
    const double sp = (zw[0] * gw[0] + zw[1] * gw[1] + zw[2] * gw[2]) / ng;
    const int S = 2 * PR + 1; double a = 0, b = 0, d = 0, m = 0, m2 = 0;
    for (int k = 0; k < S * S; k++) { a += f.gxf[k] * f.gxf[k]; b += f.gxf[k] * f.gyf[k]; d += f.gyf[k] * f.gyf[k]; m += f.tf[k]; m2 += f.tf[k] * f.tf[k]; }
    m /= S * S; m2 = m2 / (S * S) - m * m; const double tr = 0.5 * (a + d), dd = std::sqrt(0.25 * (a - d) * (a - d) + b * b);
    int nact = 0; for (int i = 0; i < FEAT_N; i++) nact += feats[i].active;
    const double vn = std::sqrt(N.v[0] * N.v[0] + N.v[1] * N.v[1] + N.v[2] * N.v[2]);
    int edge = 0;
#if EDGE
    edge = f.edge;
#endif
    std::fprintf(dlog, "F %ld %d %.5f %.5f %.5f %.5f %.4f %.4f %.4f %.3f %.3f %.3f %.3f %d %d %.4f %.4f %.4f\n", F.uid, frame_no, se, sp, bc[0], bc[1],
                 N.p[2], std::sqrt(v2_avg), vn, std::log1p(std::max(0.0, tr - dd)), std::log1p(tr + dd), m, std::sqrt(std::max(0.0, m2)), edge, nact, fx / W, f.x, f.y);
}
static void dlog_obs(const WF& F, const Feat& f) {
    if (dlog) std::fprintf(dlog, "O %ld %d %.6f %.6f\n", F.uid, frame_no, (f.x - cx) / fx, (f.y - cy) / fy);
}
#endif
#if NN_DEPTH
#ifndef NN_WSET
#define NN_WSET 0                // weights: 0 nn_depth_weights.h (chosen), 1/2: the same training with other seeds (robustness check)
#endif
#if NN_WSET == 1
#include "nn_depth_weights_s1.h"
#elif NN_WSET == 2
#include "nn_depth_weights_s2.h"
#elif NN_WSET == 3
#include "nn_depth_weights_g.h"   // geometry + appearance only (no speed, no height)
#else
#include "nn_depth_weights.h"
#endif
static void nn_depth_set(WF& F, const Feat& f) {
    F.dmu = dpf_lz0; F.dsig = DPF_SIG;
#if EDGE
    if (f.edge && !NN_EDGE) return;
#endif
    if (nfr < 1) return;
    const Frame& N = fr[nfr - 1];
    const double bc[3] = {(f.x - cx) / fx, (f.y - cy) / fy, 1}; double bi[3], bw[3]; mtv(Rci, bc, bi); mv(N.R, bi, bw);
    const double ng = std::sqrt(gw[0] * gw[0] + gw[1] * gw[1] + gw[2] * gw[2]), nb = std::sqrt(bw[0] * bw[0] + bw[1] * bw[1] + bw[2] * bw[2]);
    const double se = (bw[0] * gw[0] + bw[1] * gw[1] + bw[2] * gw[2]) / (ng * nb);
    const double zc[3] = {0, 0, 1}; double zi[3], zw[3]; mtv(Rci, zc, zi); mv(N.R, zi, zw);
    const double sp = (zw[0] * gw[0] + zw[1] * gw[1] + zw[2] * gw[2]) / ng;
    const int S = 2 * PR + 1; double a = 0, b = 0, d = 0, m = 0, m2 = 0;
    for (int k = 0; k < S * S; k++) { a += f.gxf[k] * f.gxf[k]; b += f.gxf[k] * f.gyf[k]; d += f.gyf[k] * f.gyf[k]; m += f.tf[k]; m2 += f.tf[k] * f.tf[k]; }
    m /= S * S; m2 = m2 / (S * S) - m * m; const double tr = 0.5 * (a + d), dd = std::sqrt(0.25 * (a - d) * (a - d) + b * b);
    const double vn = std::sqrt(N.v[0] * N.v[0] + N.v[1] * N.v[1] + N.v[2] * N.v[2]);
    const double pz = N.p[2] < -2 ? -2 : (N.p[2] > 2 ? 2 : N.p[2]);
#if NN_NIN == 11
    (void)vn; (void)pz;
    const double x[NN_NIN] = {se, se * se, sp, bc[0], bc[1], std::fabs(bc[0]),
                              std::log1p(tr - dd > 0 ? tr - dd : 0) / 5, std::log1p(tr + dd) / 7, m / 255, std::sqrt(m2 > 0 ? m2 : 0) / 40, 3.0 * fx / W};
#else
    const double x[NN_NIN] = {se, se * se, sp, bc[0], bc[1], std::fabs(bc[0]), pz, std::sqrt(v2_avg), vn > 3 ? 3 : vn,
                              std::log1p(tr - dd > 0 ? tr - dd : 0) / 5, std::log1p(tr + dd) / 7, m / 255, std::sqrt(m2 > 0 ? m2 : 0) / 40, 3.0 * fx / W};
#endif
    float xn[NN_NIN]; for (int i = 0; i < NN_NIN; i++) xn[i] = (float)((x[i] - NN_MU[i]) / NN_SD[i]);
    double mu, ls = std::log(0.39);
    if (NN_DEPTH == 1) { mu = NN_WL[NN_NIN]; for (int i = 0; i < NN_NIN; i++) mu += NN_WL[i] * xn[i]; }
    else {
        mu = NN_B2[0]; ls = NN_B2[1];
        for (int h = 0; h < NN_H; h++) {
            float t = NN_B1[h]; for (int i = 0; i < NN_NIN; i++) t += xn[i] * NN_W1[i * NN_H + h];
            const double th = std::tanh((double)t); mu += th * NN_W2[2 * h]; ls += th * NN_W2[2 * h + 1];
        }
    }
    mu += NN_OFF; if (mu < -0.7) mu = -0.7; if (mu > 2.1) mu = 2.1;
    F.dmu = mu;
    if (NN_HET) { if (ls < -3) ls = -3; if (ls > 1) ls = 1; F.dsig = NN_SIG * std::exp(ls) / 0.39; } else F.dsig = NN_SIG;
}
#endif
static int wf_alloc(int live) {
    for (int w = 0; w < MAX_WF; w++) if (!wf[w].used) { WF& F = wf[w]; F.used = true; F.live = live; F.nobs = 0; F.lam_ok = false; F.lam = 0;
#if EDGE
        F.edge = live >= 0 && feats[live].edge;
#endif
#ifndef __wasm__
        if (live >= 0) dlog_feature(F, feats[live]);
#endif
#if NN_DEPTH
        if (live >= 0) nn_depth_set(F, feats[live]); else { F.dmu = dpf_lz0; F.dsig = DPF_SIG; }
#endif
        return w; }
    return -1;
}
static void wf_add_obs(int w, const Feat& f) {
    WF& F = wf[w]; if (F.nobs >= WIN + 1) return;
    F.ser[F.nobs] = frame_no; F.u[F.nobs] = (float)((f.x - cx) / fx); F.v[F.nobs] = (float)((f.y - cy) / fy); F.q[F.nobs] = f.lkr;
#ifndef __wasm__
    dlog_obs(F, f);
#endif
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
#if defined(__wasm_simd128__) && DET_FAST
            int nxs = 0; for (int x = x0 + 2; x < x0 + cw - 1 && x < W - 2; x += 4) nxs++;   // samples per row (a prefix is valid)
            if (nxs <= 10) {   // integer max of the same sums, ten samples of a row at a time (exact)
                const v128_t iota = wasm_i16x8_make(0, 1, 2, 3, 4, 5, 6, 7);
                const v128_t mlo = wasm_i16x8_lt(iota, wasm_i16x8_splat(nxs)), mhi = wasm_i16x8_lt(iota, wasm_i16x8_splat(nxs - 8));
                v128_t M = wasm_i16x8_splat(0);
                for (int y = y0 + 2; y < y0 + chh - 1; y += 4) {
                    if (y < 2 || y >= H - 2) continue;
                    const uint8_t* pc = img + (y * W + x0) * 3 + 1;           // green of (x0 + 4t, y): t = s is p[-6], t = s + 1 is p[6]
                    const uint8_t* pu = pc - 6 * W + 6; const uint8_t* pd = pc + 6 * W + 6;   // (x0 + 2 + 4s, y -+ 2)
                    v128_t A = wasm_i8x16_splat(0), U = A, D = A;
#define DLA(t) A = wasm_v128_load8_lane(pc + 12 * (t), A, t)
#define DLU(t) U = wasm_v128_load8_lane(pu + 12 * (t), U, t); D = wasm_v128_load8_lane(pd + 12 * (t), D, t)
                    DLA(0); DLA(1); DLA(2); DLA(3); DLA(4); DLA(5); DLA(6); DLA(7); DLA(8); DLA(9); DLA(10);
                    DLU(0); DLU(1); DLU(2); DLU(3); DLU(4); DLU(5); DLU(6); DLU(7); DLU(8); DLU(9);
#undef DLA
#undef DLU
                    const v128_t As = wasm_i8x16_shuffle(A, A, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0);
                    const v128_t dx = wasm_i8x16_sub(wasm_u8x16_max(As, A), wasm_u8x16_min(As, A));
                    const v128_t dy = wasm_i8x16_sub(wasm_u8x16_max(D, U), wasm_u8x16_min(D, U));
                    const v128_t lo = wasm_v128_and(wasm_i16x8_add(wasm_u16x8_extend_low_u8x16(dx), wasm_u16x8_extend_low_u8x16(dy)), mlo);
                    const v128_t hi = wasm_v128_and(wasm_i16x8_add(wasm_u16x8_extend_high_u8x16(dx), wasm_u16x8_extend_high_u8x16(dy)), mhi);
                    M = wasm_u16x8_max(M, wasm_u16x8_max(lo, hi));
                }
                M = wasm_u16x8_max(M, wasm_i8x16_shuffle(M, M, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7));
                M = wasm_u16x8_max(M, wasm_i8x16_shuffle(M, M, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3));
                M = wasm_u16x8_max(M, wasm_i8x16_shuffle(M, M, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1));
                mx = wasm_u16x8_extract_lane(M, 0);
            } else
#endif
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
#if defined(__wasm_simd128__) && DET_FAST
            {   // integer sums of the 9 samples (exact, as the float sums of integers below 2^24): one row of 3 samples per step
                const int W3 = 3 * W; v128_t SA = wasm_i32x4_splat(0), SB = SA, SC = SA;
                for (int j = -2; j <= 2; j += 2) {
                    const uint8_t* q = img + ((y + j) * W + x) * 3 + 1;
                    const v128_t r0 = wasm_v128_load(q - 9), r1 = wasm_v128_load(q + 7);   // row: q-9 .. q+22
                    const v128_t ru = wasm_v128_load(q + W3 - 6), rd = wasm_v128_load(q - W3 - 6);   // rows below / above: q-6 .. q+9
                    // X: dx+ (q-3, q+3, q+9) in 0-2, dx- (q-9, q-3, q+3) in 8-10;  D: dy+ in 4-6, dy- in 12-14
                    const v128_t X = wasm_i8x16_shuffle(r0, r1, 6, 12, 18, 0, 0, 0, 0, 0, 0, 6, 12, 0, 0, 0, 0, 0);
                    const v128_t D = wasm_i8x16_shuffle(ru, rd, 0, 0, 0, 0, 0, 6, 12, 0, 0, 0, 0, 0, 16, 22, 28, 0);
                    const v128_t Z = wasm_i8x16_shuffle(X, D, 0, 1, 2, 3, 20, 21, 22, 7, 8, 9, 10, 11, 28, 29, 30, 15);
                    const v128_t G = wasm_i16x8_sub(wasm_u16x8_extend_low_u8x16(Z), wasm_u16x8_extend_high_u8x16(Z));   // dx 0-2, dy 4-6
                    const v128_t Gs = wasm_i8x16_shuffle(G, G, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
                    SA = wasm_i32x4_add(SA, wasm_i32x4_extmul_low_i16x8(G, G));
                    SC = wasm_i32x4_add(SC, wasm_i32x4_extmul_high_i16x8(G, G));
                    SB = wasm_i32x4_add(SB, wasm_i32x4_extmul_low_i16x8(G, Gs));
                }
                a = (float)(wasm_i32x4_extract_lane(SA, 0) + wasm_i32x4_extract_lane(SA, 1) + wasm_i32x4_extract_lane(SA, 2));
                b = (float)(wasm_i32x4_extract_lane(SB, 0) + wasm_i32x4_extract_lane(SB, 1) + wasm_i32x4_extract_lane(SB, 2));
                c = (float)(wasm_i32x4_extract_lane(SC, 0) + wasm_i32x4_extract_lane(SC, 1) + wasm_i32x4_extract_lane(SC, 2));
            }
#else
            for (int j = -2; j <= 2; j += 2) {
                const uint8_t* row = img + ((y + j) * W + x) * 3 + 1;
                for (int i = -2; i <= 2; i += 2) {
                    const uint8_t* q = row + 3 * i;
                    const float dx = (float)q[3] - (float)q[-3], dy = (float)q[3 * W] - (float)q[-3 * W];
                    a += dx * dx; b += dx * dy; c += dy * dy;
                }
            }
#endif
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
    for (int k = 0; k < nfr; k++) {
        const Frame& F = fr[k]; double* o = d + FB(k);
        logso3(mmul(mtr(F.Rl), F.R), o);
        for (int i = 0; i < 3; i++) { o[3 + i] = F.p[i] - F.pl[i]; o[6 + i] = F.v[i] - F.vl[i]; if (!SHARED_BIAS) { o[9 + i] = F.bg[i] - F.bgl[i]; o[12 + i] = F.ba[i] - F.bal[i]; } }
    }
}
static void add_prior();
// zero the n x n system (whole rows: the columns past n are unused) and add the prior: same values as zero + add_prior()
// (0 + x = x), with bulk memory ops for the zeroing and the copy of the prior's lower triangle
// bulk memory ops (memset / memcpy) cost about one fuel per byte in the sandbox; v128 loops cost about a quarter of that
static inline void vzero(double* p, int n) {
    int k = 0; const v128_t z = wasm_f64x2_splat(0.0);
    for (; k + 2 <= n; k += 2) wasm_v128_store(p + k, z);
    if (k < n) p[k] = 0.0;
}
static inline void vcopy(double* dst, const double* src, int n) {
    int k = 0;
    for (; k + 2 <= n; k += 2) wasm_v128_store(dst + k, wasm_v128_load(src + k));
    if (k < n) dst[k] = src[k];
}
static void base_prior(int n) {
    static double d[NX]; prior_dx(d);
    vzero(gs, n);
#if BASE_FUSE
    // one pass over each prior row: copy to H, the dot product with d (dotv's exact summation order) and the column part
    // gs[j] += hp[j] d[i] two outputs at a time (each output keeps its order of additions): bit-identical
    for (int i = 0; i < n; i++) {
        const double* hp = Hp + i * NX; double* h = Hs + i * NX; const int m = i + 1;
        const v128_t DI = wasm_f64x2_splat(d[i]);
        v128_t s0 = wasm_f64x2_splat(0.0), s1 = s0; int k = 0;
        if (want_H) {
            for (; k + 4 <= m; k += 4) {
                const v128_t a0 = wasm_v128_load(hp + k), a1 = wasm_v128_load(hp + k + 2);
                wasm_v128_store(h + k, a0); wasm_v128_store(h + k + 2, a1);
                s0 = wasm_f64x2_add(s0, wasm_f64x2_mul(a0, wasm_v128_load(d + k)));
                s1 = wasm_f64x2_add(s1, wasm_f64x2_mul(a1, wasm_v128_load(d + k + 2)));
                if (k + 4 <= i) { wasm_v128_store(gs + k, wasm_f64x2_add(wasm_v128_load(gs + k), wasm_f64x2_mul(a0, DI)));
                                  wasm_v128_store(gs + k + 2, wasm_f64x2_add(wasm_v128_load(gs + k + 2), wasm_f64x2_mul(a1, DI))); }
                else { for (int j = k; j < i; j++) gs[j] += hp[j] * d[i]; }
            }
        } else {
            for (; k + 4 <= m; k += 4) {
                const v128_t a0 = wasm_v128_load(hp + k), a1 = wasm_v128_load(hp + k + 2);
                s0 = wasm_f64x2_add(s0, wasm_f64x2_mul(a0, wasm_v128_load(d + k)));
                s1 = wasm_f64x2_add(s1, wasm_f64x2_mul(a1, wasm_v128_load(d + k + 2)));
                if (k + 4 <= i) { wasm_v128_store(gs + k, wasm_f64x2_add(wasm_v128_load(gs + k), wasm_f64x2_mul(a0, DI)));
                                  wasm_v128_store(gs + k + 2, wasm_f64x2_add(wasm_v128_load(gs + k + 2), wasm_f64x2_mul(a1, DI))); }
                else { for (int j = k; j < i; j++) gs[j] += hp[j] * d[i]; }
            }
        }
        const int k4 = k;
        if (k + 2 <= m) { s0 = wasm_f64x2_add(s0, wasm_f64x2_mul(wasm_v128_load(hp + k), wasm_v128_load(d + k))); k += 2; }
        double r = (wasm_f64x2_extract_lane(s0, 0) + wasm_f64x2_extract_lane(s1, 0)) + (wasm_f64x2_extract_lane(s0, 1) + wasm_f64x2_extract_lane(s1, 1));
        for (; k < m; k++) r += hp[k] * d[k];
        for (int j = k4; j < i; j++) gs[j] += hp[j] * d[i];
        if (want_H) { for (int j = k4; j < m; j++) h[j] = hp[j]; if (!NO_UPPER0) vzero(h + m, n - m); } else vzero(h, n);
        gs[i] += gp[i] + r;
    }
#else
    for (int i = 0; i < n; i++) {
        const double* hp = Hp + i * NX; double* h = Hs + i * NX;
        if (want_H) { vcopy(h, hp, i + 1); vzero(h + i + 1, n - i - 1); } else vzero(h, n);
        gs[i] += gp[i] + dotv(hp, d, i + 1);
        const double di = d[i]; for (int j = 0; j < i; j++) gs[j] += hp[j] * di;
    }
#endif
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

static void add_unary() {
    for (int k = 0; k < nfr; k++) {
        const Frame& F = fr[k];
        double info = 0;
        if (VPRIOR_SIG > 0) info += (F.span / VP_DT) / (F.vsig * F.vsig);
        if (F.zv) info += 1.0 / (ZVSIG_ * ZVSIG_);
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
static double fb_hxf[MAX_WF][6 * (WIN + 1)], fb_hff[MAX_WF], fb_gf[MAX_WF], fb_err[MAX_WF];
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
#if VARPRO_FAST
    // camera point as an affine function of the inverse depth: pc = A / lam + B (per observation, once), d pc / d(1/lam) = A
    double PA[WIN + 1][3], PB[WIN + 1][3];
    for (int oo = 1; oo < ms; oo++) {
        const int j = fidx(F.ser[sel[oo]]); if (j < 0) return false;
        const Frame& FJ = fr[j];
        double d[3], y[3]; for (int i = 0; i < 3; i++) d[i] = FA.p[i] - rt[i] - FJ.p[i];
        mtv(FJ.R, rb, y); mv(Rci, y, PA[oo]);
        mtv(FJ.R, d, y); mv(Rci, y, PB[oo]); for (int i = 0; i < 3; i++) PB[oo][i] += tci[i];
    }
#endif
    for (int it = 0; it < iters; it++) {
        const double il = 1.0 / F.lam;
        double h2 = 0, g = 0;
#if VARPRO_FAST
        for (int oo = 1; oo < ms; oo++) {
            const int o = sel[oo]; const double* A = PA[oo]; const double* B = PB[oo]; const double il2 = -il * il;
            const double pc[3] = {A[0] * il + B[0], A[1] * il + B[1], A[2] * il + B[2]}, pcd[3] = {A[0] * il2, A[1] * il2, A[2] * il2};
            if (pc[2] < MIN_DEPTH) return false;
            const double iz = 1.0 / pc[2], u = pc[0] * iz, v = pc[1] * iz;
            double e0 = fx * (u - F.u[o]), e1 = fy * (v - F.v[o]);
            double j0 = fx * iz * (pcd[0] - u * pcd[2]), j1 = fy * iz * (pcd[1] - v * pcd[2]);
#if EDGE
            if (F.edge) { const double nu = F.enu[o], nv = F.env[o]; e0 = nu * e0 + nv * e1; e1 = 0; j0 = nu * j0 + nv * j1; j1 = 0; }
#endif
            const double en = std::sqrt(e0 * e0 + e1 * e1), wh = en / SIGPX_ <= HUBER ? 1.0 : HUBER * SIGPX_ / en;
            h2 += wh * (j0 * j0 + j1 * j1); g += wh * (j0 * e0 + j1 * e1);
        }
#else
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
            if (F.edge) { const double nu = F.enu[o], nv = F.env[o]; e0 = nu * e0 + nv * e1; e1 = 0; j0 = nu * j0 + nv * j1; j1 = 0; }
#endif
            const double en = std::sqrt(e0 * e0 + e1 * e1), wh = en / SIGPX_ <= HUBER ? 1.0 : HUBER * SIGPX_ / en;
            h2 += wh * (j0 * j0 + j1 * j1); g += wh * (j0 * e0 + j1 * e1);
        }
#endif
        if (DPF_SIG > 0) { const double r = (std::log(F.lam) + DPF_LZ(F)) / DPF_S(F), J = 1.0 / (F.lam * DPF_S(F)); h2 += J * J; g += J * r; }
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
    alignas(16) static double Haa[6][6], Hak[WIN + 1][6][6], Hkk[WIN + 1][6][6];
    for (int p = 0; p < 6; p++) for (int q = 0; q < 6; q++) Haa[p][q] = 0;
    double gl_[6 * (WIN + 1)], hxf[6 * (WIN + 1)], hff = 0, gf = 0, err = 0, cost_f = 0;
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
        const double en = std::sqrt(e[0] * e[0] + e[1] * e[1]);
        err += en;
        cost_f += 0.5 * (e[0] * e[0] + e[1] * e[1]) / (SIGPX_ * SIGPX_);
        const double sq = LK_R0 > 0 ? SIGPX_ * std::sqrt(1.0 + (F.q[q0] / LK_R0) * (F.q[q0] / LK_R0)) : SIGPX_;   // weaker for a poor LK fit
        const double wh = en / sq <= HUBER ? 1.0 : HUBER * sq / en;
        const double info = wh / (sq * sq);
#if !EDGE
        const double Jp[2][3] = {{fx * iz, 0, -fx * u * iz}, {0, fy * iz, -fy * v * iz}};
#endif
        double G[2][3], GR[2][3];
        for (int r = 0; r < 2; r++) for (int c = 0; c < 3; c++) G[r][c] = Jp[r][0] * Rci.m[c] + Jp[r][1] * Rci.m[3 + c] + Jp[r][2] * Rci.m[6 + c];
        for (int r = 0; r < 2; r++) for (int c = 0; c < 3; c++) GR[r][c] = G[r][0] * FJ.R.m[3 * c] + G[r][1] * FJ.R.m[3 * c + 1] + G[r][2] * FJ.R.m[3 * c + 2];
        const M3 Sy = skew(y);
        alignas(16) double Ja[2][6], Jj[2][6]; double Jl[2];
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
#if defined(__wasm_simd128__) && FEAT_FAST
        {   // the three 6x6 blocks two columns at a time (each element: the same two products and sums, bit-identical)
            v128_t A0[3], A1[3], B0[3], B1[3];
            for (int c = 0; c < 3; c++) { A0[c] = wasm_v128_load(&Ja[0][2 * c]); A1[c] = wasm_v128_load(&Ja[1][2 * c]); B0[c] = wasm_v128_load(&Jj[0][2 * c]); B1[c] = wasm_v128_load(&Jj[1][2 * c]); }
            for (int p = 0; p < 6; p++) {
                const double a0 = info * Ja[0][p], a1 = info * Ja[1][p], j0 = info * Jj[0][p], j1 = info * Jj[1][p];
                const v128_t sa0 = wasm_f64x2_splat(a0), sa1 = wasm_f64x2_splat(a1), sj0 = wasm_f64x2_splat(j0), sj1 = wasm_f64x2_splat(j1);
                for (int c = 0; c < 3; c++) {
                    double* ha = &Haa[p][2 * c];
                    wasm_v128_store(ha, wasm_f64x2_add(wasm_v128_load(ha), wasm_f64x2_add(wasm_f64x2_mul(sa0, A0[c]), wasm_f64x2_mul(sa1, A1[c]))));
                    wasm_v128_store(&hak[p][2 * c], wasm_f64x2_add(wasm_f64x2_mul(sa0, B0[c]), wasm_f64x2_mul(sa1, B1[c])));
                    wasm_v128_store(&hkk[p][2 * c], wasm_f64x2_add(wasm_f64x2_mul(sj0, B0[c]), wasm_f64x2_mul(sj1, B1[c])));
                }
                gl_[p] += a0 * e[0] + a1 * e[1]; gl_[oj + p] += j0 * e[0] + j1 * e[1];
                hxf[p] += a0 * Jl[0] + a1 * Jl[1]; hxf[oj + p] += j0 * Jl[0] + j1 * Jl[1];
            }
        }
#else
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
#endif
        hff += info * (Jl[0] * Jl[0] + Jl[1] * Jl[1]); gf += info * (Jl[0] * e[0] + Jl[1] * e[1]);
    }
    fb_err[w] = err / (m - 1);
    if (DPF_SIG > 0) {   // log-depth prior
        const double r = (std::log(lam) + DPF_LZ(F)) / DPF_S(F), J = 1.0 / (lam * DPF_S(F));
        hff += J * J; gf += J * r; cost_f += 0.5 * r * r;
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
#if defined(__wasm_simd128__) && SCATTER_FAST
            if (g1 != g2) {   // full 6x6 off-diagonal block, branch-free and two columns at a time (same operations)
                const double* x2 = hxf + 6 * s2;
                const v128_t X0 = wasm_v128_load(x2), X1 = wasm_v128_load(x2 + 2), X2 = wasm_v128_load(x2 + 4);
                for (int p = 0; p < 6; p++) {
                    double* h = Hs + (g1 + p) * NX + g2;
                    const v128_t NH = wasm_f64x2_splat(-(hxf[6 * s1 + p] * ih));
                    v128_t v0 = wasm_f64x2_mul(NH, X0), v1 = wasm_f64x2_mul(NH, X1), v2 = wasm_f64x2_mul(NH, X2);
                    if (blk) {
                        if (tr) {
                            v0 = wasm_f64x2_add(v0, wasm_f64x2_make(blk[0][p], blk[1][p]));
                            v1 = wasm_f64x2_add(v1, wasm_f64x2_make(blk[2][p], blk[3][p]));
                            v2 = wasm_f64x2_add(v2, wasm_f64x2_make(blk[4][p], blk[5][p]));
                        } else {
                            v0 = wasm_f64x2_add(v0, wasm_v128_load(&blk[p][0]));
                            v1 = wasm_f64x2_add(v1, wasm_v128_load(&blk[p][2]));
                            v2 = wasm_f64x2_add(v2, wasm_v128_load(&blk[p][4]));
                        }
                    }
                    wasm_v128_store(h, wasm_f64x2_add(wasm_v128_load(h), v0));
                    wasm_v128_store(h + 2, wasm_f64x2_add(wasm_v128_load(h + 2), v1));
                    wasm_v128_store(h + 4, wasm_f64x2_add(wasm_v128_load(h + 4), v2));
                    if (mc >= 0) { double* mh = &mc_H[mc][6 * s1 + p][6 * s2]; wasm_v128_store(mh, v0); wasm_v128_store(mh + 2, v1); wasm_v128_store(mh + 4, v2); }
                }
                continue;
            }
#endif
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
    fb_nf[w] = m; fb_hff[w] = hff; fb_gf[w] = gf;
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
static void build() {
    const int n = NSYS();
    for (int i = 0; i < n; i++) { if (want_H) { double* h = Hs + i * NX; for (int j = 0; j < n; j++) h[j] = 0; } gs[i] = 0; }
    imu_cost = 0;
    add_prior();
    for (int k = 1; k < nfr; k++) add_imu(k);
    add_unary();
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
    for (int k = 0; k < nfr; k++) { const Frame& F = fr[k]; double info = (F.span / VP_DT) / (F.vsig * F.vsig) + (F.zv ? 1.0 / (ZVSIG_ * ZVSIG_) : 0);
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
    const double out_px = (OUT_LOW_N > 0 && nfeat < OUT_LOW_N) ? OUT_PX_LOW : OUTPX_;   // few features: keep more of them
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
        double s = fb_gf[w];
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
#if defined(__wasm_simd128__) && SCHUR_FAST
    if (len == 9) {   // all 9 updates of an element pair in registers, then one store: same operations in the same order
        for (int r = r0; r < r1; r++) {
            if (r >= o && r < e) continue;
            double* hr = Hm + r * NX;
            double a[9]; for (int i = 0; i < 9; i++) a[i] = so_B[i][r];
            const v128_t A0 = wasm_f64x2_splat(a[0]), A1 = wasm_f64x2_splat(a[1]), A2 = wasm_f64x2_splat(a[2]), A3 = wasm_f64x2_splat(a[3]),
                         A4 = wasm_f64x2_splat(a[4]), A5 = wasm_f64x2_splat(a[5]), A6 = wasm_f64x2_splat(a[6]), A7 = wasm_f64x2_splat(a[7]), A8 = wasm_f64x2_splat(a[8]);
            const int cend = r + 1; int c = 0;
            for (; c + 2 <= cend; c += 2) {
                const double* x = &so_X[0][c];
                v128_t h = wasm_v128_load(hr + c);
                h = wasm_f64x2_sub(h, wasm_f64x2_mul(A0, wasm_v128_load(x)));
                h = wasm_f64x2_sub(h, wasm_f64x2_mul(A1, wasm_v128_load(x + NX)));
                h = wasm_f64x2_sub(h, wasm_f64x2_mul(A2, wasm_v128_load(x + 2 * NX)));
                h = wasm_f64x2_sub(h, wasm_f64x2_mul(A3, wasm_v128_load(x + 3 * NX)));
                h = wasm_f64x2_sub(h, wasm_f64x2_mul(A4, wasm_v128_load(x + 4 * NX)));
                h = wasm_f64x2_sub(h, wasm_f64x2_mul(A5, wasm_v128_load(x + 5 * NX)));
                h = wasm_f64x2_sub(h, wasm_f64x2_mul(A6, wasm_v128_load(x + 6 * NX)));
                h = wasm_f64x2_sub(h, wasm_f64x2_mul(A7, wasm_v128_load(x + 7 * NX)));
                h = wasm_f64x2_sub(h, wasm_f64x2_mul(A8, wasm_v128_load(x + 8 * NX)));
                wasm_v128_store(hr + c, h);
            }
            for (; c < cend; c++) { double v = hr[c]; for (int i = 0; i < 9; i++) v -= a[i] * so_X[i][c]; hr[c] = v; }
            double s = 0; for (int i = 0; i < 9; i++) s += a[i] * so_.gx[i];
            gm[r] -= s;
        }
        return;
    }
#endif
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
#if defined(__wasm_simd128__) && SCHUR_FAST
        {   const int m1 = o < r + 1 ? o : r + 1;   // two contiguous pieces (rows rr < r never overlap; rr == r copies in place)
            if (dst != src) vcopy(dst, src, m1);
            if (r >= e) vcopy(dst + m1, src + e, r + 1 - e); }
#else
        int cc = 0; for (int c = 0; c < o && c <= r; c++) dst[cc++] = src[c];
        for (int c = e; c <= r; c++) dst[cc++] = src[c];
#endif
        gm[rr] = gm[r]; rr++;
    }
    for (int r = n - len; r < NX; r++) vzero(Hm + r * NX, NX);   // freed rows
    for (int r = n - len; r < n; r++) gm[r] = 0;
}
// remove block [o, o+len) of an n x n system (row stride NX) by Schur complement, compacting in place
static void schur_out(double* Hm, double* gm, int n, int o, int len) {   // lower-triangle storage in and out
    schur_prep(Hm, gm, n, o, len); schur_rows(Hm, gm, 0, n); schur_fin(Hm, gm);
}
static void set_lin() {
    for (int i = 0; i < 3; i++) { gl[i] = gw[i]; bgl_s[i] = fr[0].bg[i]; bal_s[i] = fr[0].ba[i]; }
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
#if NN_DEPTH
    if (q[2] > MIN_DEPTH && q[2] < MAX_DEPTH) F.dmu += std::log(q[2] * F.lam);   // the prior follows the point to the new anchor
#endif
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
        const double cap = 1.0 / (i < 5 ? BFG_ * BFG_ : BFA_ * BFA_), h = Hp[i * NX + i];
        if (h > cap) { const double f = std::sqrt(cap / h); for (int j = 0; j <= i; j++) Hp[i * NX + j] *= f; for (int j = i; j < n; j++) Hp[j * NX + i] *= f; gp[i] *= f; }
    }
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
#if NIT != 1 || IMU_EST || IMU_MID || CHOL_F32
#error "PIPE supports NIT 1 without IMU_EST, IMU_MID, CHOL_F32"
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
    const double out_px = (OUT_LOW_N > 0 && nfeat < OUT_LOW_N) ? OUT_PX_LOW : OUTPX_;
    if (!chol4r<NX>(Hs, n, pipe_js, n)) { st_fail++; return false; }
    if (MAH) mah_store(n);
    chol_solve(Hs, n, NX, dxs);
    for (int i = 0; i < n; i++) if (!std::isfinite(dxs[i])) { st_fail++; return false; }
    apply_dx(dxs);
    for (int w = 0; w < MAX_WF; w++) if (fb_on[w]) {
        double s = fb_gf[w];
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
#ifdef PROF_CUT
        { static Feat tmp; tmp = f; for (int r = 0; r < PROF_CUT; r++) { cut_template(tmp); __asm__ volatile("" :: "r"(&tmp) : "memory"); } }
#endif
#ifdef PROF_CPY1
        { static Feat tmp; tmp = f; __asm__ volatile("" :: "r"(&tmp) : "memory"); }
#endif
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
static double sc_log[SC_DONE + 1];
static double vz_zero = 0, vz_env = 0, vz_n = 0;   // host debug: estimated units per type this frame
static inline int feat_m(const WF& F) { return (OBS_MAX > 0 && F.nobs > OBS_MAX) ? OBS_MAX : F.nobs; }
static double cu_base(int n) { return CU_N2 * n * n + CU_IMU * (nfr - 1); }
static double cu_feat(const WF& F) { const double m = feat_m(F); return CU_F0 + CU_FM * m * m; }
static double cu_col(int n, int j) { return CU_CH * (n - j) * (j + 8.0); }
#ifndef CU_SKYB
#define CU_SKYB 3.0              // SKYLINE: building the permuted copy, per n^2
#endif
#ifndef CU_MC0
#define CU_MC0 4000.0            // PIPE + MARG_CACHE: a cached marginalization feature, base
#endif
#ifndef CU_MCM
#define CU_MCM 250.0             // ... and per squared observation count
#endif
static int mc_find(int w) { for (int c = 0; c < mc_n; c++) if (mc_w[c] == w) return c; return -1; }
// cached reduced blocks of feature c into Hs/gs, gradient moved by the solve's step (g + H dx)
static void mc_add(int c) {
    const int m = mc_m[c], M = 6 * m;
    alignas(16) static double S[6 * (WIN + 1)][6 * (WIN + 1) + 2], dl[6 * (WIN + 1) + 2];
    for (int r = 0; r < M; r++) for (int t = 0; t <= r; t++) { const double v = mc_H[c][r][t]; S[r][t] = v; S[t][r] = v; }
    for (int s1 = 0; s1 < m; s1++) for (int p = 0; p < 6; p++) dl[6 * s1 + p] = dxs[FB(mc_fl[c][s1]) + p];
    for (int s1 = 0; s1 < m; s1++) {
        const int g1 = FB(mc_fl[c][s1]);
        for (int p = 0; p < 6; p++) {
            const int r = 6 * s1 + p;
            gs[g1 + p] += mc_g[c][r] + dotv(S[r], dl, M);
            for (int s2 = 0; s2 <= s1; s2++) {
                const int g2 = FB(mc_fl[c][s2]); double* h = Hs + (g1 + p) * NX + g2; const double* src = &mc_H[c][r][6 * s2];
                const int qmax = s1 == s2 ? p + 1 : 6;
                for (int q = 0; q < qmax; q++) h[q] += src[q];
            }
        }
    }
}
static double cu_marg_build(int n) {
    double c = CU_N2 * n * n + CU_IMU; int cnt = 0; const int s0 = fr[0].serial;
    for (int w = 0; w < MAX_WF && cnt < MARG_FMAX; w++) if (wf_usable(wf[w]) && wf[w].ser[0] == s0) {
        if (MARG_CACHE) { const double m = feat_m(wf[w]); c += CU_MC0 + CU_MCM * m * m; } else c += cu_feat(wf[w]);
        cnt++; }
    return c;
}
static double cu_mprep(int n) { return CU_MPK * NF_ * NF_ * n + CU_MPC; }
static double cu_mfin(int n) { return CU_MFK * n * n + CU_MFC; }
static double sc_estimate() {
    const int n = NSYS(); double c = cu_base(n) + CU_DAMPC;
    for (int w = 0; w < MAX_WF; w++) if (wf_usable(wf[w])) c += cu_feat(wf[w]);
    for (int j = 0; j < n; j++) c += cu_col(n, j);
    c += CU_SOLVE * n * n + CU_SOLC;
    if (sc_marg) { const int nm = n; c += cu_marg_build(nm) + cu_mprep(nm) + CU_SR * NF_ * nm * nm * 0.5 + cu_mfin(nm); }
    else if (nfr >= 3 && !pipe_kf && Hp[FB(nfr - 2) * NX + FB(nfr - 2)] != 0) c += cu_mprep(n) + CU_SR * NF_ * n * n * 0.5 + CU_DRC;   // drop with a prior Schur step
    return c;
}
static void sc_start() { sc_st = SC_BASE; sc_cur = 0; sc_done = 0; sc_marg = pipe_kf && nfr == WIN; sc_total = sc_estimate();
    mc_n = 0; mc_valid = false; mc_on = MARG_CACHE && sc_marg; }
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
#ifndef SKYLINE
#define SKYLINE 0                // 1 (PIPE): Cholesky on a copy with the shared states last, skipping each row's leading zeros
#endif
#if SKYLINE
// The frames' block of H is a band-like profile (features and the prior couple nearby frames); the 8 shared states
// couple everything, so they go last. Row i of the permuted matrix starts at its first nonzero fz[i]; L keeps that
// profile (no fill to the left of a row's first nonzero), so every dot product starts there.
alignas(16) static double Hq[NX * NX];
static int fz[NX], sk_n = 0;
#ifndef SKY_ORDER
#define SKY_ORDER 1              // 0: frames (theta p v) then shared; 1: all theta p, then all v, then shared (rows of a kind contiguous)
#endif
static int sk_po[NX];   // permuted -> original index
static inline int sk_old(int i) { return sk_po[i]; }
static void sky_build() {
    const int n = sc_n, nf = n - G0, nk = nf / NF_; sk_n = n;
#if SKY_ORDER == 1 && SHARED_BIAS && !defined(SKY_GENERIC)
    // permuted order: theta p of frames 0..nk-1 (6 each), v of frames (3 each), shared (G0); copies by blocks
    const int NV = NF_ - 6, VB = 6 * nk, SB = VB + NV * nk;
    for (int k = 0; k < nk; k++) { for (int c = 0; c < 6; c++) sk_po[6 * k + c] = FB(k) + c; for (int c = 0; c < NV; c++) sk_po[VB + NV * k + c] = FB(k) + 6 + c; }
    for (int g = 0; g < G0; g++) sk_po[SB + g] = g;
    for (int k = 0; k < nk; k++) for (int c = 0; c < 6; c++) {   // theta p rows: columns theta p of frames <= k (row r, contiguous)
        const int i = 6 * k + c; const double* hr = Hs + (FB(k) + c) * NX; double* d = Hq + i * NX;
        for (int k2 = 0; k2 < k; k2++) { const double* q = hr + FB(k2); double* o = d + 6 * k2; vcopy(o, q, 6); }
        for (int c2 = 0; c2 <= c; c2++) d[6 * k + c2] = hr[FB(k) + c2];
    }
    for (int k = 0; k < nk; k++) for (int c = 0; c < NV; c++) {   // v rows
        const int r = FB(k) + 6 + c, i = VB + NV * k + c; const double* hr = Hs + r * NX; double* d = Hq + i * NX;
        for (int k2 = 0; k2 <= k; k2++) vcopy(d + 6 * k2, hr + FB(k2), 6);                       // theta p of frames <= k: row r
        for (int k2 = k + 1; k2 < nk; k2++) { const double* q = Hs + FB(k2) * NX + r; double* o = d + 6 * k2;   // later frames: column r
            for (int c2 = 0; c2 < 6; c2++) o[c2] = q[c2 * NX]; }
        for (int k2 = 0; k2 < k; k2++) { const double* q = hr + FB(k2) + 6; double* o = d + VB + NV * k2; for (int c2 = 0; c2 < NV; c2++) o[c2] = q[c2]; }
        for (int c2 = 0; c2 <= c; c2++) d[VB + NV * k + c2] = hr[FB(k) + 6 + c2];
    }
    for (int g = 0; g < G0; g++) {   // shared rows: every frame column is column g of a later row
        const int i = SB + g; double* d = Hq + i * NX;
        for (int k = 0; k < nk; k++) { const double* q = Hs + FB(k) * NX + g;
            for (int c2 = 0; c2 < 6; c2++) d[6 * k + c2] = q[c2 * NX];
            for (int c2 = 0; c2 < NV; c2++) d[VB + NV * k + c2] = q[(6 + c2) * NX]; }
        for (int c2 = 0; c2 <= g; c2++) d[SB + c2] = Hs[g * NX + c2];
    }
    for (int i = 0; i < n; i++) { const double* d = Hq + i * NX; int f = 0; while (f < i && d[f] == 0) f++; fz[i] = f; }
#else
    // segments of contiguous original indices in permuted order
    int sn[3 * WIN + 2], so[3 * WIN + 2], sl[3 * WIN + 2], ns = 0, p = 0;
    auto seg = [&](int o, int l) { sn[ns] = p; so[ns] = o; sl[ns] = l; ns++; for (int t = 0; t < l; t++) sk_po[p++] = o + t; };
    if (SKY_ORDER == 1) { for (int k = 0; k < nk; k++) seg(FB(k), 6); for (int k = 0; k < nk; k++) seg(FB(k) + 6, NF_ - 6); }
    else for (int k = 0; k < nk; k++) seg(FB(k), NF_);
    seg(0, G0);
    for (int i = 0; i < n; i++) {
        const int r = sk_po[i]; double* dst = Hq + i * NX; const double* hr = Hs + r * NX;
        for (int s2 = 0; s2 < ns && sn[s2] <= i; s2++) {
            const int c0 = so[s2]; int cnt = sl[s2]; if (sn[s2] + cnt > i + 1) cnt = i + 1 - sn[s2];
            double* d = dst + sn[s2];
            if (c0 + cnt - 1 <= r) vcopy(d, hr + c0, cnt);
            else for (int t = 0; t < cnt; t++) { const int c = c0 + t; d[t] = c <= r ? hr[c] : Hs[c * NX + r]; }
        }
        int f = 0; while (f < i && dst[f] == 0) f++;
        fz[i] = f;
    }
#endif
}
static bool sky_col(int j) {   // column j of L in place (left-looking), rows below j
    const int n = sk_n; double* Aj = Hq + j * NX; const int fj = fz[j];
    const double d0 = Aj[j] - dotv(Aj + fj, Aj + fj, j - fj);
    if (!(d0 > 1e-300)) return false;
    const double d = std::sqrt(d0); Aj[j] = d; const double id = 1.0 / d;
    int i = j + 1;
    while (i < n) {
        if (fz[i] > j) { i++; continue; }                 // L[i][j] = 0: left of row i's profile
#if CHOL8
        if (i + 8 <= n) {   // eight rows from their common start (the zeros left of a row's profile are stored)
            int m = fz[i]; bool all = true;
            for (int t = 1; t < 8; t++) { const int f = fz[i + t]; all = all && f <= j; if (f < m) m = f; }
            if (all) {
                const int k0 = (m > fj ? m : fj) & ~1;
                double o[8]; double* Ai = Hq + i * NX; dotRs<NX, 8>(Ai + k0, Aj + k0, j - k0, o);
                for (int t = 0; t < 8; t++) Ai[t * NX + j] = (Ai[t * NX + j] - o[t]) * id;
                i += 8; continue;
            }
        }
#endif
        if (i + 4 <= n && fz[i + 1] <= j && fz[i + 2] <= j && fz[i + 3] <= j) {   // four rows from their common start
            int m = fz[i]; for (int t = 1; t < 4; t++) if (fz[i + t] < m) m = fz[i + t];
            const int k0 = (m > fj ? m : fj) & ~1;
            double o[4]; double* Ai = Hq + i * NX; dot4s<NX>(Ai + k0, Aj + k0, j - k0, o);
            for (int t = 0; t < 4; t++) Ai[t * NX + j] = (Ai[t * NX + j] - o[t]) * id;
            i += 4; continue;
        }
        double* Ai = Hq + i * NX; const int k0 = fz[i] > fj ? fz[i] : fj;
        Ai[j] = (Ai[j] - dotv(Ai + k0, Aj + k0, j - k0)) * id;
        i++;
    }
    return true;
}
static void sky_solve(double* x) {   // x <- H^-1 x through the permuted profile factor
    const int n = sk_n; alignas(16) static double y[NX];
    for (int i = 0; i < n; i++) y[i] = x[sk_old(i)];
    for (int i = 0; i < n; i++) { const double* Li = Hq + i * NX; const int f = fz[i]; y[i] = (y[i] - dotv(Li + f, y + f, i - f)) / Li[i]; }
    // backward, row-oriented: y[k] final, then y[f..k) -= L[k][f..k) y[k] (contiguous row, two at a time)
    for (int k = n - 1; k >= 0; k--) {
        const double* Lk = Hq + k * NX; const double yk = (y[k] /= Lk[k]);
        int i = fz[k];
#ifdef __wasm_simd128__
        const v128_t Y = wasm_f64x2_splat(yk);
        if ((i & 1) && i < k) { y[i] -= Lk[i] * yk; i++; }
        for (; i + 2 <= k; i += 2) wasm_v128_store(y + i, wasm_f64x2_sub(wasm_v128_load(y + i), wasm_f64x2_mul(wasm_v128_load(Lk + i), Y)));
#endif
        for (; i < k; i++) y[i] -= Lk[i] * yk;
    }
    for (int i = 0; i < n; i++) x[sk_old(i)] = y[i];
}
#endif
static void sc_unit();
#ifndef PIPE_FXP
#define PIPE_FXP 0               // 1: the planner's fixed work of later frames is at least CU_MID / CU_PHL + CU_TRACK x live tracks
#endif
#ifndef PIPE_LX
#define PIPE_LX 0                // 1: a 2-frame cycle whose last frame is predicted (learned linear model) above PIPE_CAPX gets 3 frames
#endif
#ifndef PIPE_CAPX
#define PIPE_CAPX 1.50           // M fuel
#endif
#ifndef PIPE_LA_MID
#define PIPE_LA_MID 0            // 1: the look-ahead only on middle frames (not on a cycle's first frame)
#endif
#ifndef PIPE_LA
#define PIPE_LA -1               // >= 0: on frames before the cycle's last, a large indivisible unit (build, solve, marginalization build /
#endif                           //   prep, drop) is not started when it would end more than PIPE_LA above the frame's target
static double sc_next_cost() {   // estimated cost of the next unit when it is a large indivisible one (else 0)
    const int n = NSYS();
    switch (sc_st) {
    case SC_BASE: return CU_N2 * n * n;
    case SC_SOLVE: return CU_SOLVE * sc_n * sc_n + CU_SOLC;
    case SC_MBUILD: return CU_N2 * n * n + CU_IMU;
    case SC_MPREP: return cu_mprep(n);
    case SC_DROP0: return cu_mprep(n) + CU_DRC / 2;
    default: return 0;
    }
}
static void sc_work(double target) {   // run units until the estimated work reaches target (or the chain ends)
    while (sc_st != SC_DONE && sc_done < target) {
        if (PIPE_LA >= 0 && target < 1e299 && (!PIPE_LA_MID || pipe_ph > 0) && sc_done + sc_next_cost() > target + PIPE_LA) break;
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
#ifdef PROF_BASE
            for (int i = 0; i < n; i++) { double* h = Hs + i * NX; for (int j = 0; j < n; j++) h[j] = 0; gs[i] = 0; }
#if PROF_BASE > 1
            add_prior();
#endif
#endif
            imu_cost = 0;
            base_prior(n);
            sc_done += CU_N2 * n * n; sc_st = SC_BIMU; sc_cur = 1; break; }
        case SC_BIMU:
            if (sc_cur < nfr) { add_imu(sc_cur); sc_cur++; sc_done += CU_IMU; break; }
            add_unary();
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
#ifdef VZ
            {   long z = 0, tot = 0, env = 0; const int n = sc_n;
                for (int i = 0; i < n; i++) { int fnz = i; for (int j = 0; j <= i; j++) { tot++; if (Hs[i * NX + j] == 0) z++; } for (int j = 0; j <= i; j++) if (Hs[i * NX + j] != 0) { fnz = j; break; } env += i - fnz + 1; }
                vz_zero = (double)z / tot; vz_env = (double)env / tot; vz_n = n;
                // frames first, the 8 shared states last: skyline cost sum over rows of (row envelope)^2 relative to dense
                double dense = 0, sky = 0; const int G = 8;
                for (int i = G; i < n; i++) { int fnz = i; for (int j = G; j <= i; j++) if (Hs[i * NX + j] != 0) { fnz = j; break; } const double w = i - fnz + 1; sky += w * w; dense += (double)(i - G + 1) * (i - G + 1); }
                for (int i = 0; i < G; i++) { const double w = n - G + i + 1; sky += w * w; dense += w * w; }
                vz_n = sky / dense; }
#endif
            for (int i = 0; i < sc_n; i++) { HH(i, i) += LM_REL * HH(i, i) + 1e-9; dxs[i] = -gs[i]; }
#if SKYLINE
            sky_build(); sc_done += CU_SKYB * sc_n * sc_n;
#endif
#ifdef PROF_CHOL
            {   static double T[NX * NX]; for (int i = 0; i < sc_n; i++) vcopy(T + i * NX, Hs + i * NX, i + 1);
#if PROF_CHOL == 2
                for (int i = 0; i < sc_n; i++) vcopy(T + i * NX, Hs + i * NX, i + 1);
#else
                chol4r<NX>(T, sc_n, 0, sc_n);
#endif
                __asm__ volatile("" :: "r"(T) : "memory"); }
#endif
            sc_done += CU_DAMPC; sc_st = SC_CHOL; sc_cur = 0; break;
        case SC_CHOL:
            if (!pipe_ok) { sc_st = SC_SOLVE; break; }
            if (sc_cur >= sc_n) { sc_st = SC_SOLVE; break; }
            sc_done += cu_col(sc_n, sc_cur);
#if SKYLINE
            if (!sky_col(sc_cur)) { st_fail++; pipe_ok = false; }
#else
            if (!chol4r<NX>(Hs, sc_n, sc_cur, sc_cur + 1)) { st_fail++; pipe_ok = false; }
#endif
            sc_cur++; break;
        case SC_SOLVE: {
            bool ok = pipe_ok;
            if (ok) {
                const int n = sc_n;
                int nfeat = 0; for (int w = 0; w < MAX_WF; w++) nfeat += fb_on[w];
                const double out_px = (OUT_LOW_N > 0 && nfeat < OUT_LOW_N) ? OUT_PX_LOW : OUTPX_;
                if (MAH) mah_store(n);
#if SKYLINE
                sky_solve(dxs);
#else
                chol_solve(Hs, n, NX, dxs);
#endif
                for (int i = 0; i < n && ok; i++) if (!std::isfinite(dxs[i])) { st_fail++; ok = false; }
                if (ok) {
                    apply_dx(dxs);
                    for (int w = 0; w < MAX_WF; w++) if (fb_on[w]) {
                        double sv = fb_gf[w];
                        for (int s1 = 0; s1 < fb_nf[w]; s1++) { const double* d = dxs + FB(fb_fl[w][s1]); const double* h = fb_hxf[w] + 6 * s1; for (int q = 0; q < 6; q++) sv += h[q] * d[q]; }
                        WF& F = wf[w];
                        F.lam -= sv / fb_hff[w];
                        if (!(F.lam > 1.0 / MAX_DEPTH && F.lam < 1.0 / MIN_DEPTH) || fb_err[w] > out_px) { st_out++; kill_wf(w); }
                    }
                }
            }
            sc_done += CU_SOLVE * sc_n * sc_n + CU_SOLC;
            mc_valid = mc_on && ok; mc_on = false;
            sc_solve_done(ok); break; }
        case SC_MBUILD: {   // marg_build in units: prior and IMU factor 0-1 first, then one feature per unit
            const int n = NSYS(); sc_done += CU_N2 * n * n + CU_IMU;
            base_prior(n);
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
            if (sc_cur < sc_mln && sc_mcnt < MARG_FMAX) {
                const int w = sc_ml[sc_cur++];
                const int c = (MARG_CACHE && mc_valid) ? mc_find(w) : -1;
                if (c >= 0) { const double m = mc_m[c]; sc_done += CU_MC0 + CU_MCM * m * m; mc_add(c); fb_on[w] = true; sc_mcnt++; }
                else { sc_done += cu_feat(wf[w]); if (add_feature(w)) sc_mcnt++; }
                break; }
            {   const int n = NSYS(); for (int r = n; r < NX; r++) vzero(Hs + r * NX, NX); }
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
            sc_done += cu_mprep(n) + CU_DRC / 2; sc_st = SC_DROWS; sc_cur = 0; break; }
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
            sc_done += CU_DRC / 2; sc_st = SC_DONE; break; }
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
#if defined(PROF_TRACK) || defined(PROF_TPL)
    {   static Feat sv[FEAT_N]; std::memcpy(sv, feats, sizeof feats); const int nf0 = nflow;
#ifdef PROF_TRACK
        track(Rcc);
#endif
#ifdef PROF_TPL
        pipe_templates(fr[nfr - 1].R);
#endif
        std::memcpy(feats, sv, sizeof feats); nflow = nf0; }
#endif
#ifdef PROF_MS
    {   static char buf[100000]; std::memset(buf, (int)(frame_no & 0x7f), sizeof buf); __asm__ volatile("" :: "r"(buf) : "memory"); }
#endif
#ifdef PROF_COPY
    {   static Feat sv[FEAT_N]; std::memcpy(sv, feats, sizeof feats); std::memcpy(feats, sv, sizeof feats); }
#endif
    track(Rcc);
    if (pipe_ph == 0) {
        Frame F; predict_frame(fr[nfr - 1], pp, F);
        F.serial = frame_no; F.t = t_last; F.span = pp.T > 0 ? pp.T : PIPE * frame_dt; F.zv = false; F.mo_n = 0;
#if MANHATTAN
        F.mh_on = false;
#endif
        {   double sig = VPRIOR_SIG;
            if (VP_ADAPT) { sig = VPK_ * std::sqrt(v2_avg); if (sig < VPMIN_) sig = VPMIN_; if (sig > VPRIOR_SIG) sig = VPRIOR_SIG; }
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
            if (!triangulate(Fw) && DPF_SIG > 0 && DPF_INIT && Fw.nobs >= DPF_MINOBS) { Fw.lam = std::exp(-DPF_LZ(Fw)); Fw.lam_ok = true; }
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
            if (PIPE_FXP) {   // later frames' fixed work: at least what the current track count implies (start-up: tracks grow)
                const double m1 = CU_MID + CU_TRACK * na, m2 = CU_PHL + CU_TRACK * na;
                if (sc_fixed_prev[1] < m1) sc_fixed_prev[1] = m1; if (sc_fixed_prev[2] < m2) sc_fixed_prev[2] = m2;
            }
            for (;;) {
                sc_level = (sc_total + fx_k + (len - 2) * sc_fixed_prev[1] + sc_fixed_prev[2] + PIPE_RES) / len;
                if (sc_level <= PIPE_CAP || len >= PIPE_MAX) break;
                len++;
            }
#if PIPE_LX
            if (len == 2 && PIPE_MAX >= 3) {   // learned overshoot of the 2-frame cycle's last frame (linear model fitted on sandbox fuel)
                static const double LXW[13] = {0.0158163, -0.137342, 0.0536243, -0.884504, 0.229686, 0.0122956, -0.00430943, -0.00185604, -0.00931902, 0.0402415, 0.232175, -0.0270902, 0.174593};
                int nwf = 0, nob = 0; for (int w = 0; w < MAX_WF; w++) if (wf[w].used) { nwf++; nob += wf[w].nobs; }
                const double f[13] = {1, sc_total / 1e6, fx_k / 1e6, sc_fixed_prev[2] / 1e6, na / 32.0, (st_cells - c0) / 30.0, (st_cfull - cf0) / 10.0,
                                      (double)(st_map_try - m0), (st_det - d0) / 10.0, (double)sc_marg, nfr / 11.0, nwf / 50.0, nob / 300.0};
                double ex = 0; for (int i = 0; i < 13; i++) ex += LXW[i] * f[i];
                if (sc_level / 1e6 + ex > PIPE_CAPX) { len = 3; sc_level = (sc_total + fx_k + sc_fixed_prev[1] + sc_fixed_prev[2] + PIPE_RES) / 3; }
            }
#endif
            sc_len = len; sc_target = 0;
#ifndef __wasm__
            if (std::getenv("VINS_PLOG")) { int nwf = 0, nob = 0; for (int w = 0; w < MAX_WF; w++) if (wf[w].used) { nwf++; nob += wf[w].nobs; }
                std::fprintf(stderr, "PL %d %d %.0f %.0f %.0f %.0f %.0f %d %ld %ld %ld %ld %d %d %d %d\n", frame_no, len, sc_total, fx_k, sc_fixed_prev[1], sc_fixed_prev[2], sc_level,
                             na, st_cells - c0, st_cfull - cf0, st_map_try - m0, st_det - d0, (int)sc_marg, nfr, nwf, nob); }
#endif
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
#ifdef VCNT
    { int na = 0, ne = 0; for (int i = 0; i < FEAT_N; i++) if (feats[i].active) { na++; if (feats[i].edge) ne++; } out[7] = na; out[8] = ne; out[9] = FEAT_N; }
#endif
#ifdef VZ
    out[7] = vz_zero; out[8] = vz_env; out[9] = vz_n;
#endif
#ifdef VDIAG2
    {   // per-frame unit estimates by group (regression of the cost model); quaternion stays valid
        auto L = [&](int k) { return sc_log[k]; };
        int na = 0; for (int i = 0; i < FEAT_N; i++) na += feats[i].active;
        out[0] = pipe_ph + 10.0 * sc_len + 100.0 * na + 10000.0 * (double)(st_cells - c0);
        out[1] = (double)(st_cfull - cf0) + 100.0 * (double)(st_map_try - m0) + 10000.0 * (double)(st_det - d0);
        out[2] = L(SC_BASE) + L(SC_MBUILD);
        out[3] = 0; out[4] = 0; out[5] = 0; out[6] = 1;
        out[7] = L(SC_BIMU); out[8] = L(SC_FEAT); out[9] = L(SC_MBF); out[10] = L(SC_CHOL);
        out[11] = L(SC_SOLVE) + L(SC_DAMP);
        out[12] = L(SC_MPREP) + L(SC_MROWS) + L(SC_MFIN) + L(SC_DROP0) + L(SC_DROWS) + L(SC_DFIN);
    }
#endif
#ifdef VDIAG
    {   double ch = 0; for (int i = 0; i < SC_DONE; i++) ch += sc_log[i];
        int na = 0; for (int i = 0; i < FEAT_N; i++) na += feats[i].active;
        const double fxd = (pipe_ph == 0 ? CU_PH0 : (pipe_ph == last_ph ? CU_PHL : CU_MID)) + CU_TRACK * na
                         + CU_CELL * (st_cells - c0) + CU_CFULL * (st_cfull - cf0) + CU_SEARCH * (st_map_try - m0) + CU_DET * (st_det - d0);
        out[7] = pipe_ph + 10 * sc_len; out[8] = fxd; out[9] = ch; out[10] = sc_level; out[11] = na + 100 * (double)(st_cells - c0) + 10000 * (double)(st_cfull - cf0); out[12] = (double)(st_map_try - m0) + 10 * (double)sc_marg; }
#endif
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
            if (VP_ADAPT) { sig = VPK_ * std::sqrt(v2_avg); if (sig < VPMIN_) sig = VPMIN_; if (sig > VPRIOR_SIG) sig = VPRIOR_SIG; }
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
            if (!triangulate(Fw) && DPF_SIG > 0 && DPF_INIT && Fw.nobs >= DPF_MINOBS) { Fw.lam = std::exp(-DPF_LZ(Fw)); Fw.lam_ok = true; }
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

#if LEARN_P
static double lpg_d2 = 0, lpg_prev[3]; static long lpg_n = 0;   // LP_GATE: sum of squared successive accel differences
static void lp_gate_imu(const double* imu, int nimu) {
    for (int i = 0; i < nimu; i++) {
        const double* a = imu + 7 * i + 1;
        if (lpg_n > 0 || i > 0) { for (int k = 0; k < 3; k++) { const double d = a[k] - lpg_prev[k]; lpg_d2 += d * d; } lpg_n++; }
        for (int k = 0; k < 3; k++) lpg_prev[k] = a[k];
        if (lpg_n == 0 && i == 0) lpg_n = 0;
    }
}
static void lp_update() {   // multipliers for this frame from the running speed
    if (!lp_init) {
        lp_init = true; for (int i = 0; i < 18; i++) lp_w[i] = LP_W[i];
#ifndef __wasm__
        if (const char* e = std::getenv("VINS_LP")) { for (int i = 0; i < 18 && *e; i++) { char* q; lp_w[i] = std::strtod(e, &q); e = *q ? q + 1 : q; } }
#endif
    }
    double vr = std::sqrt(v2_avg); if (vr > 2) vr = 2; const double dv = vr - 0.5;
    double g = 1;
    if (LP_GATE) {   // var(successive difference) = 2 sigma^2 per axis; 200-sample prior at the nominal level
        const double v = (200 * 2 * 3 * SIG_ACC * SIG_ACC + lpg_d2) / (200 + lpg_n), k = std::sqrt(v / (6 * SIG_ACC * SIG_ACC));
        g = (k - LP_K0) / (LP_K1 - LP_K0); if (g < 0) g = 0; if (g > 1) g = 1;
    }
    for (int i = 0; i < 9; i++) { double l = g * (lp_w[2 * i] + lp_w[2 * i + 1] * dv); if (l < -2) l = -2; if (l > 2) l = 2; lp_m[i] = i == 2 ? l : std::exp(l); }
    dpf_lz0 = std::log(DPF_Z0) + lp_m[2];
    sa2 = SIG_ACC * SIG_ACC * lp_m[3] * lp_m[3]; sg2 = SIG_GYR * SIG_GYR * lp_m[4] * lp_m[4];
}
#endif
#ifndef __wasm__
static double aug_imu[7 * 64]; static double aug_ba[3], aug_bg[3]; static double aug_k[2] = {0, 0}; static uint64_t aug_s = 0; static bool aug_init = false;
static double aug_gauss() {   // xorshift64* + Box-Muller (host augmentation only)
    auto u = [&]() { aug_s ^= aug_s >> 12; aug_s ^= aug_s << 25; aug_s ^= aug_s >> 27; return ((aug_s * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0) + 1e-12; };
    const double a = u(), b = u(); return std::sqrt(-2 * std::log(a)) * std::cos(6.283185307179586 * b);
}
#endif
VIOBENCH_EXPORT(step) int32_t step(double t, const uint8_t* frame, const double* imu, int32_t nimu, double* out) {
    img = frame;
#ifndef __wasm__
    if (!aug_init) {   // VINS_AUG="k_noise,k_bias,seed": IMU white noise and bias turn-on scaled to k x the nominal (training only)
        aug_init = true;
        if (const char* e = std::getenv("VINS_AUG")) { double a = 1, b = 1, sd = 1; std::sscanf(e, "%lf,%lf,%lf", &a, &b, &sd);
            aug_k[0] = std::sqrt(std::max(0.0, a * a - 1)); aug_k[1] = std::sqrt(std::max(0.0, b * b - 1)); aug_s = 88172645463325252ULL ^ (uint64_t)(sd * 7919);
            for (int i = 0; i < 3; i++) { aug_ba[i] = 0.2 * aug_k[1] * aug_gauss(); aug_bg[i] = 0.02 * aug_k[1] * aug_gauss(); } }
    }
    if ((aug_k[0] > 0 || aug_k[1] > 0) && nimu <= 64) {
        for (int i = 0; i < 7 * nimu; i++) aug_imu[i] = imu[i];
        for (int i = 0; i < nimu; i++) for (int a = 0; a < 3; a++) {
            aug_imu[7 * i + 1 + a] += aug_ba[a] + 0.28 * aug_k[0] * aug_gauss();
            aug_imu[7 * i + 4 + a] += aug_bg[a] + 0.07 * aug_k[0] * aug_gauss();
        }
        imu = aug_imu;
    }
#endif
#if LEARN_P
    if (LP_GATE) lp_gate_imu(imu, nimu);
    lp_update();
#endif
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
        if (VP_ADAPT) { sig = VPK_ * std::sqrt(v2_avg); if (sig < VPMIN_) sig = VPMIN_; if (sig > VPRIOR_SIG) sig = VPRIOR_SIG; }
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
        if (!triangulate(Fw) && DPF_SIG > 0 && DPF_INIT && Fw.nobs >= DPF_MINOBS) { Fw.lam = std::exp(-DPF_LZ(Fw)); Fw.lam_ok = true; }
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
