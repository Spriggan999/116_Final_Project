// v8 = v7 + generator prior on the motion sign: the episode autopilot flies nose-first (FPV), so when the vision
// direction is clearly along the optical axis (|u . z_cam| > SIGN_FWD) its sign is taken as forward.
// VIO v7 for viobench: compute-light visual-inertial odometry. Native C++, no OpenCV, no ML, deterministic.
// v7 = v6 estimator with the per-frame work rebalanced (max fuel 158k -> 99k): SIMD candidate scoring + histogram top-K,
// SIMD pixel fetch, covariance propagated once per window on an idle frame (ahead over the tracking frames), position
// clone moved off the ranking frame, filter corrections injected once per frame. FIXED_GAIN=1 builds a covariance-free
// constant-gain observer instead (tested: clearly worse, kept for reference).
//
// How it works (one keyframe window = KF_LEN frames):
//   every frame   IMU: rotate with the gyro, integrate the accelerometer (gravity removed) -> attitude, velocity, position.
//   keyframe      rank candidate patches on a cheap 40x30 overview, cut 8x8 templates at 160x120 straight from the raw
//                 frame (split over SPLIT consecutive frames to flatten fuel), clone the position in the filter.
//   window end    predict where every patch moved using the gyro rotation, refine with Lucas-Kanade (SIMD) on small
//                 crops built straight from the raw frame (again split over SPLIT frames). Each track gives n = f_pred x f_meas, perpendicular to the camera
//                 displacement d. All tracks -> one motion direction u (smallest eigenvector of sum n n^T) and a small
//                 rotation correction (-> gyro bias). Speed along u from the parallax (consistent-scale proxy).
//   filter        18-state error-state EKF [dp dv dtheta dba dbg dp_kf]; covariance propagated once per window.
// Output: camera pose relative to the first camera frame and twist in the current camera frame.
#include "viobench_abi.h"
#include "params.h"

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
#if PIX_MODE == 0
    const uint8_t* r1 = r0 + W * 3;
    return 0.25f * (float)(r0[0] + r0[3] + r1[0] + r1[3]);
#elif PIX_MODE == 1
    return 0.5f * (float)(r0[0] + r0[W * 3 + 3]);   // diagonal pair
#else
    return (float)r0[W * 3 + 3];                  // single sample
#endif
}

// fill out[0..n) with tracking-resolution pixels (x0..x0+n, y): one address computation per row
// (FETCH_SIMD: may write up to 3 floats past out[n-1]; callers keep slack after the last row)
inline void fetch_row(int x0, int y, int n, float* out) {
    const int off = DS / 2 - 1;
#if defined(__wasm_simd128__) && PIX_MODE == 1 && DS == 2
    if (FETCH_SIMD) {   // 4 pixels per pass: green bytes 6 apart -> one shuffle of two loads per sample row
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
#if PIX_MODE == 0
    const uint8_t* r1 = r0 + W * 3;
    for (int i = 0; i < n; i++, r0 += DS * 3, r1 += DS * 3) out[i] = 0.25f * (float)(r0[0] + r0[3] + r1[0] + r1[3]);
#elif PIX_MODE == 1
    const uint8_t* r1 = r0 + W * 3 + 3;
    for (int i = 0; i < n; i++, r0 += DS * 3, r1 += DS * 3) out[i] = 0.5f * (float)(r0[0] + r1[0]);
#else
    const uint8_t* r1 = r0 + W * 3 + 3;
    for (int i = 0; i < n; i++, r1 += DS * 3) out[i] = (float)r1[0];
#endif
}

// ================= filter state =================
constexpr int IP = 0, IV = 3, ITH = 6, IBA = 9, IBG = 12, IPC = 15;
#if SCALE_STATE
constexpr int IS = 18, NS1 = 19;  // + log scale of the speed proxy
#else
constexpr int NS1 = 18;
#endif
#if ROT_CLONE
constexpr int ITC = NS1, NS2 = NS1 + 3;  // + attitude clone at the keyframe (vision relative-rotation update)
#else
constexpr int NS2 = NS1;
#endif
#if FLOOR
constexpr int IZF = NS2, NS3 = NS2 + 1;  // + world height of the floor plane (constant): metric scale from floor tracks
#else
constexpr int NS3 = NS2;
#endif
#if LONG_N > 0
constexpr int IPC2 = NS3, NS = NS3 + 3;  // + the previous window's position clone (long-baseline tracks)
#else
constexpr int NS = NS3;
#endif
double z_floor = FLOOR_Z0;  // FLOOR: floor height in the world frame (start position = 0, z up)
Vec3 p_kf2{0, 0, 0};       // LONG_N: previous window's position clone
Vec3 last_u_short{0, 0, 0};
bool short_ok = false;
double s_log = 0;          // SCALE_STATE: log scale of the parallax speed proxy
// adaptive measurement noise (ADAPT_R): running factors on the direction / speed noise, from normalized innovations
double lam_dir = 1.0, lam_spd = 1.0;
double last_nis = 0;       // normalized innovation squared of the last kf_update_row (before any Huber inflation)
bool huber_on = false;     // Huber weighting for the current measurement rows
#if P_FLOAT
using PT = float;          // covariance storage: f32x4 SIMD, half the instructions per row operation
#else
using PT = double;
#endif
PT P[NS * NS];
double dx[NS];             // accumulated error-state correction (injected after an update batch)

Mat3 R_wb;                 // body -> world (nominal)
Vec3 p_w, v_w, ba, bg;     // nominal IMU position, velocity, biases
Vec3 p_kf;                 // cloned position at the keyframe
Vec3 bw{0, 0, 0};          // FIXED_GAIN: world-frame accel bias (also absorbs tilt error)
double t_clone = 0;        // time of the keyframe clone
int n_windows = 0;         // vision windows since start (FIXED_GAIN warm-up)
Mat3 R_wb_kf[SPLIT];       // body orientation when each template subset was cut
Mat3 R_wb_clone;
Mat3 R_wb_hop, R_wb_key;   // TRK: body orientation at the last tracking hop / at the tracker keyframe
static_assert(TRK_WIN > 0 && TRK_HOP > 0 && TRK_WIN % TRK_HOP == 0, "TRK_WIN must be a positive multiple of TRK_HOP");
static_assert(!LROT || (TRK && ROT_CLONE), "LROT needs TRK and ROT_CLONE");
static_assert(TRK != 2 || N_TRACK % 2 == 0, "TRK == 2 needs an even N_TRACK");
constexpr int TG = 2;      // TRK == 2: feature groups (slot % TG), one group tracked per frame
Mat3 R_hopg[TG], R_keyg[TG];
Vec3 p_keyg[TG], dispg[TG]; // TRK_DISP: IMU position at each group's keyframe / nominal displacement over its window
Mat3 R_wb_long;            // LROT: body orientation at the long rotation keyframe (attitude clone)
           // body orientation at the position clone
Vec3 omega_b;              // latest bias-corrected angular rate
const Vec3 g_w{0, 0, -GRAVITY};
Mat3 R_wc0;                // first camera orientation
Vec3 p_wc0;                // first camera position

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
#if MOVE_START
Mat3 R_cal;            // MOVE_START: raw-gyro rotation since the first frame (body at c0 -> body now)
double t_first = -1, cal_prev_t = -1;
#endif
double sum_a[3], sum_a2[3], sum_g[3], sum_g2[3];

int phase = 0;             // position in the window cycle (KF_LEN + SPLIT frames)
bool have_kf = false;
int last_tracks = 0, last_used = 0;  // diagnostics
double last_quality = 0, last_parallax = 0;
int diag_reason = 0, diag_start = 0, diag_now = 0;
double diag_eta[4] = {0, 0, 0, 0};
double diag_sp[4] = {0, 0, 0, 0};
#if ADAPT_IMU
// ADAPT_IMU: white-noise level of the IMU measured online from second differences of consecutive samples
// (E|x[k] - 2x[k-1] + x[k-2]|^2 = 6 sigma^2 per axis; the true signal adds ~nothing at 200 Hz)
double ad_pg[2][3], ad_pa[2][3], ad_sg = 0, ad_sa = 0; int ad_n = 0, ad_cnt = 0;
double imu_rg = 1.0, imu_ra = 1.0;   // measured / nominal (0.07 rad/s, 0.28 m/s^2 per sample)
bool ad_bg_done = false;
#endif
#define NFEAT 9
double diag9[13] = {0};
int d10_rows = 0, d10_acc = 0, d10_kept = 0, d10_win = 0;
double d11_rn[3] = {0, 0, 0};
int diag9_n = 0;
#if SPEED_NN
#include "speednn.h"
#endif
double diag_manh[3] = {0, 0, 0};
int trk_idx[N_TRACK];      // TRK: feature index of each entry of tracks[]
bool struct_wipe = false;   // STRUCT: the map contradicted the direction fit, drop all depths
bool struct_skip = false;   // STRUCT: enough depth rows were applied, skip the direction / assumed-depth speed rows
int struct_n = 0, struct_known = 0, struct_resets = 0;
double manh_sw = 0;
double manh_c4 = 0, manh_s4 = 0, manh_psi = 0;   // MANH: azimuth votes and the learned grid azimuth
int manh_init_n = 0;
bool manh_ready = false;
int diag_winlen = 0;
int lrot_n = 0, lrot_age = 0, lrot_code = 0;
double lrot_eps = 0;

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
inline void axpy_n(PT* __restrict dst, PT a, const PT* __restrict src, int n) {
    for (int j = 0; j < n; j++) dst[j] += a * src[j];
}
// SYM_P: the updates write only the upper triangle; this restores the lower one (once per frame, before anything
// that reads whole rows)
inline void mirror_P() {
    for (int i = 0; i < NS; i++)
        for (int j = i + 1; j < NS; j++) P[j * NS + i] = P[i * NS + j];
}
bool p_lower_stale = false;

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
bool prop_split = false, prop_half = false;   // ULTRA: covariance propagation split over two frames
double prop_T = 0;
FBlocks prop_f;
Vec3 prop_z{0, 0, 1};       // THRUST_ONLY: mean thrust direction (world) of the propagated interval
PT prop_Q[NS * NS], prop_Qt[NS * NS];
void prop_finish_(const FBlocks& f, double T, PT* Q, PT* Qt);
inline void prop_finish() { if (prop_half) { prop_half = false; prop_finish_(prop_f, prop_T, prop_Q, prop_Qt); } }
void kf_propagate() {
#if PROF_SKIP_PROP || FIXED_GAIN
    acc_T = 0; acc_n = 0; acc_R = Mat3{{0}}; acc_f = {0, 0, 0}; return;
#endif
    if (acc_n == 0 || !(acc_T > 0)) return;
    double T = acc_T - prop_debt;     // part of this interval may already have been propagated ahead
    prop_debt = 0;
    if (prop_ahead_next) {            // propagate ahead over the frames until the direction update
        double ahead = AHEAD_FRAMES * (acc_T / acc_n) * 10.0;
        T += ahead; prop_debt = ahead; prop_ahead_next = false;
    }
    if (!(T > 0)) { acc_T = 0; acc_n = 0; acc_R = Mat3{{0}}; acc_f = {0, 0, 0}; return; }
    Mat3 Rbar = mscale(acc_R, 1.0 / acc_n);
    Vec3 f_w = scale(acc_f, 1.0 / acc_n);
    acc_T = 0; acc_n = 0; acc_R = Mat3{{0}}; acc_f = {0, 0, 0};
    { double nf = norm(f_w); if (nf > 1e-6) prop_z = scale(f_w, 1.0 / nf); }
    Mat3 Fx = skew(f_w), FxR = mul(Fx, Rbar);
    FBlocks f{T, mscale(Fx, -0.5 * T * T), mscale(Rbar, -0.5 * T * T), mscale(FxR, T * T * T / 6.0),
              mscale(Fx, -T), mscale(Rbar, -T), mscale(FxR, 0.5 * T * T), mscale(Rbar, -T)};
#if WIND_STATE && THRUST_ONLY
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 2; c++) {   // world-frame x, y columns: d acc / d b = -I
            f.A2.m[3 * r + c] = r == c ? -0.5 * T * T : 0.0;
            f.B2.m[3 * r + c] = r == c ? -T : 0.0;
        }
#endif
    if (SYM_P && p_lower_stale) { mirror_P(); p_lower_stale = false; }
    PT* Q = prop_Q; PT* Qt = prop_Qt;
    apply_F(f, P, Q);                                  // Q = F P
    if (prop_split) { prop_f = f; prop_T = T; prop_half = true; return; }   // ULTRA: finish on the next frame
    prop_finish_(f, T, Q, Qt);
}
void prop_finish_(const FBlocks& f, double T, PT* Q, PT* Qt) {
    for (int i = 0; i < NS; i++)                       // Qt = Q^T = P F^T
        for (int j = 0; j < NS; j++) Qt[j * NS + i] = Q[i * NS + j];
    apply_F(f, Qt, P);                                 // P = F P F^T
    double qa = SIG_ACC * SIG_ACC, qg = SIG_GYR * SIG_GYR, qba = SIG_BA * SIG_BA, qbg = SIG_BG * SIG_BG;
#if ADAPT_IMU == 2
    // accelerometer model error grows with a lower-grade IMU: the isotropic term follows the measured accel noise ratio
    const double ra2 = 1.0;
    { const double r = imu_ra < AD_ACC_CAP ? imu_ra : AD_ACC_CAP; qa = SIG_ACC_H * SIG_ACC_H * r * r; }
#elif ADAPT_IMU
    qg *= imu_rg * imu_rg; qbg *= imu_rg * imu_rg;
    const double ra2 = imu_ra * imu_ra;
#else
    const double ra2 = 1.0;
#endif
    if (THRUST_ONLY) {   // accelerometer noise acts along the thrust axis only; a small isotropic term covers model error
#if ADAPT_IMU == 2
        const double qz = (SIG_ACC * SIG_ACC - SIG_ACC_H * SIG_ACC_H) * ra2;   // qa already holds the adapted isotropic term
#else
        const double qz = (qa - SIG_ACC_H * SIG_ACC_H) * ra2;
        qa = SIG_ACC_H * SIG_ACC_H;
#endif
        const double z[3] = {prop_z.x, prop_z.y, prop_z.z};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                const double zz = qz * z[i] * z[j];
                P[(IP + i) * NS + IP + j] += zz * T * T * T / 3.0;
                P[(IP + i) * NS + IV + j] += zz * T * T / 2.0;
                P[(IV + i) * NS + IP + j] += zz * T * T / 2.0;
                P[(IV + i) * NS + IV + j] += zz * T;
            }
    }
    for (int i = 0; i < 3; i++) {
        P[(IP + i) * NS + IP + i] += qa * T * T * T / 3.0;
        P[(IP + i) * NS + IV + i] += qa * T * T / 2.0;
        P[(IV + i) * NS + IP + i] += qa * T * T / 2.0;
        P[(IV + i) * NS + IV + i] += qa * T;
        P[(ITH + i) * NS + ITH + i] += qg * T;
        if (!THRUST_ONLY || i == 2) P[(IBA + i) * NS + IBA + i] += qba * T;
#if WIND_STATE && THRUST_ONLY
        else P[(IBA + i) * NS + IBA + i] += SIG_WIND_RW * SIG_WIND_RW * T;
#endif
        P[(IBG + i) * NS + IBG + i] += qbg * T;
#if SCALE_STATE
    if (i == 0) P[IS * NS + IS] += SIG_S_RW * SIG_S_RW * T;
#endif
    }
}

// sequential scalar update: innovation y (measured - predicted), measurement row H, noise Rn
struct QRow { double H[NS]; double y, R; bool hub; };
QRow qrows[8]; int nq = 0; bool defer_rows = false;   // ULTRA: rows computed on one frame, applied on the next
inline bool huber_on_fwd() { return huber_on; }
bool kf_update_row(const double Hrow[NS], double y, double Rn) {
    if (defer_rows) {
        if (nq >= 8) return false;
        for (int j = 0; j < NS; j++) qrows[nq].H[j] = Hrow[j];
        qrows[nq].y = y; qrows[nq].R = Rn; qrows[nq].hub = huber_on_fwd(); nq++;
        return true;
    }
    double PH[NS];
    int nz[NS], nnz = 0;
    for (int j = 0; j < NS; j++) if (Hrow[j] != 0.0) nz[nnz++] = j;
    if (SYM_P) {   // only the upper triangle is current: P(i, c) = P[min * NS + max]
        for (int i = 0; i < NS; i++) PH[i] = 0;
        for (int k = 0; k < nnz; k++) {
            const int c = nz[k];
            const double h = Hrow[c];
            for (int i = 0; i < c; i++) PH[i] += P[i * NS + c] * h;
            for (int i = c; i < NS; i++) PH[i] += P[c * NS + i] * h;
        }
    } else
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
    last_nis = yc * yc / S;
    if (!(last_nis <= CHI2_GATE)) return false;   // also rejects NaN
    if (HUBER_C > 0 && huber_on && last_nis > HUBER_C * HUBER_C) {   // Huber: weight c/|z| == noise inflated by |z|/c
        double infl = std::sqrt(last_nis) / HUBER_C;
        S += (infl - 1.0) * Rn;
    }
    double g = yc / S, is = 1.0 / S;
    for (int i = 0; i < NS; i++) dx[i] += PH[i] * g;
    PT pht[NS];
    for (int i = 0; i < NS; i++) pht[i] = (PT)PH[i];
    if (SYM_P) {
        for (int i = 0; i < NS; i++) axpy_n(P + i * NS + i, (PT)(-PH[i] * is), pht + i, NS - i);
        p_lower_stale = true;
    } else
        for (int i = 0; i < NS; i++) axpy(P + i * NS, (PT)(-PH[i] * is), pht);
    return true;
}

bool inject_pending = false;
void kf_inject_now();
// LAZY_INJECT: corrections stay in dx (later rows see them through y - H dx) and are injected once per frame
void kf_inject() {
#if FIXED_GAIN
    return;
#endif
    if (LAZY_INJECT) { inject_pending = true; return; }
    kf_inject_now();
}
void kf_inject_now() {
    inject_pending = false;
    p_w = add(p_w, Vec3{dx[0], dx[1], dx[2]});
    v_w = add(v_w, Vec3{dx[3], dx[4], dx[5]});
    {
        const Mat3 Eth = exp_so3(Vec3{dx[6], dx[7], dx[8]});
        R_wb = mul(Eth, R_wb);
#if TRK
        R_wb_hop = mul(Eth, R_wb_hop);   // the hop reference moves with the current attitude (prediction uses their difference)
        for (int g = 0; g < TG; g++) { R_hopg[g] = mul(Eth, R_hopg[g]); p_keyg[g] = add(p_keyg[g], Vec3{dx[0], dx[1], dx[2]}); }   // recent keyframe positions share the position correction
#endif
    }
    orthonormalize(R_wb);
    ba = add(ba, Vec3{dx[9], dx[10], dx[11]});
    bg = add(bg, Vec3{dx[12], dx[13], dx[14]});
    p_kf = add(p_kf, Vec3{dx[15], dx[16], dx[17]});
#if ROT_CLONE
    {   // the attitude clone (and the subset keyframe attitudes cut right after it) take the clone's correction
        Mat3 E = exp_so3(Vec3{dx[ITC], dx[ITC + 1], dx[ITC + 2]});
#if TRK
        R_wb_key = mul(E, R_wb_key);
        for (int g = 0; g < TG; g++) R_keyg[g] = mul(E, R_keyg[g]);
#endif
#if LROT
        R_wb_long = mul(E, R_wb_long);
#else
        R_wb_clone = mul(E, R_wb_clone);
        for (int s = 0; s < SPLIT; s++) R_wb_kf[s] = mul(E, R_wb_kf[s]);
#endif
    }
#endif
#if LONG_N > 0
    p_kf2 = add(p_kf2, Vec3{dx[IPC2], dx[IPC2 + 1], dx[IPC2 + 2]});
#endif
#if FLOOR
    z_floor += dx[IZF];
    if (z_floor > -FLOOR_HMIN) z_floor = -FLOOR_HMIN;
#endif
#if SCALE_STATE
    s_log += dx[IS];
    if (s_log > SCALE_MAX) s_log = SCALE_MAX;
    if (s_log < -SCALE_MAX) s_log = -SCALE_MAX;
#endif
    for (int i = 0; i < NS; i++) dx[i] = 0;
    if (SYM_P) { if (p_lower_stale) { mirror_P(); p_lower_stale = false; } return; }
    for (int i = 0; i < NS; i++)
        for (int j = i + 1; j < NS; j++) {
            PT s = (PT)0.5 * (P[i * NS + j] + P[j * NS + i]);
            P[i * NS + j] = P[j * NS + i] = s;
        }
}

Mat3 R_wb_clone2;
void kf_clone() {
    if (inject_pending) kf_inject_now();   // the clone must copy the corrected state
#if LONG_N > 0
    p_kf2 = p_kf;               // the current clone becomes the previous one (state and covariance)
    R_wb_clone2 = R_wb_clone;
    if (!FIXED_GAIN) {
        if (SYM_P && p_lower_stale) { mirror_P(); p_lower_stale = false; }
        for (int i = 0; i < NS; i++)
            for (int j = 0; j < 3; j++) P[i * NS + IPC2 + j] = P[i * NS + IPC + j];
        for (int j = 0; j < 3; j++)
            for (int i = 0; i < NS; i++) P[(IPC2 + j) * NS + i] = P[(IPC + j) * NS + i];
    }
#endif
    p_kf = p_w;
    t_clone = last_t;
#if FIXED_GAIN
    return;
#endif
    if (SYM_P && p_lower_stale) { mirror_P(); p_lower_stale = false; }
    for (int i = 0; i < NS; i++)
        for (int j = 0; j < 3; j++) P[i * NS + IPC + j] = P[i * NS + IP + j];
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < NS; i++) P[(IPC + j) * NS + i] = P[(IP + j) * NS + i];
#if ROT_CLONE && !LROT
    for (int i = 0; i < NS; i++)
        for (int j = 0; j < 3; j++) P[i * NS + ITC + j] = P[i * NS + ITH + j];
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < NS; i++) P[(ITC + j) * NS + i] = P[(ITH + j) * NS + i];
#endif
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
    double lx, ly, lfx, lfy;        // LONG_N: last tracked top-left and its translational flow (px)
    int ok;                         // LONG_N: tracked in its own window
};
Patch patches_buf[2][N_TRACK];
Patch* patches = patches_buf[0];    // this window's templates
Patch* prevp = patches_buf[1];      // LONG_N: the previous window's templates
int n_prev = 0;
Mat3 R_wb_kf_prev[SPLIT], R_wb_trk[SPLIT], R_wb_trk_prev[SPLIT];
int n_patches = 0;

constexpr int KTOP = N_TRACK + N_EXTRA < GRID_X * GRID_Y ? N_TRACK + N_EXTRA : GRID_X * GRID_Y;
int top[KTOP];
float topv[KTOP];
int nk = 0, next_cand = 0;
double samp_s = 0, samp_ss = 0;  // sums of the ranking samples (frame contrast)
int samp_n = 0;
double eig_thresh = MIN_EIG;     // texture threshold for this window (scaled with frame contrast)
float cscore[NCAND];
float ctr[NCAND];           // half-trace of each cell's sparse tensor (edge strength = 2 ctr - cscore)
float* ctr_p = ctr;
float g_tr = 0;
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
    g_tr = tr;
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
    wasm_v128_store(ctr_p, tr);
    samp_s += wasm_f32x4_extract_lane(tr, 0); samp_s += wasm_f32x4_extract_lane(tr, 1);
    samp_s += wasm_f32x4_extract_lane(tr, 2); samp_s += wasm_f32x4_extract_lane(tr, 3);
    samp_n += 4;
}
#endif

// rank a slice of the candidate grid (cells [c0, c1)) into the running top-K list
void rank_slice(int c0, int c1) {
#if defined(__wasm_simd128__)
    if (RANK_SIMD) {   // scores only (cells are fixed); the top-K selection happens once in select_top()
        int k = c0;
        for (; k % 4 && k < c1; k++) { cscore[k] = score_cell(cx_[k], cy_[k]); ctr[k] = g_tr; }
        for (; k + 4 <= c1; k += 4) { ctr_p = ctr + k; score_cells4(cx_ + k, cy_[k], cscore + k); }  // 4 cells of one grid row
        for (; k < c1; k++) { cscore[k] = score_cell(cx_[k], cy_[k]); ctr[k] = g_tr; }
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
    if (!RANK_SIMD) return;
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

// cut the template of patch p at (p.ox, p.oy) from the current frame; false if the texture test fails
bool cut_at(Patch& p) {
    constexpr int BW = PS + 2;
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
    if (mineig < eig_thresh || !(det > 1e-9)) return false;
    p.i00 = d / det; p.i01 = -b / det; p.i11 = a / det;
    p.sgx = sgx; p.sgy = sgy;
    double mean = fs / PN;
    p.tstd = (float)std::sqrt(std::fmax(1.0, fss / PN - mean * mean));
    return true;
}

// cut templates for subset `si` from the current frame at the next ranked cells, until `want` more are accepted
void cut_templates(int si, int want) {
#if PROF_SKIP_TEMPLATES
    return;
#endif
    int target = n_patches + want, tries = 0;
    for (; next_cand < nk && n_patches < target && n_patches < N_TRACK && tries < want + N_EXTRA / SPLIT; next_cand++, tries++) {
        int i = next_cand;
        Patch& p = patches[n_patches];
        p.sub = si;
        p.ok = 0;
        p.ox = cx_[top[i]]; p.oy = cy_[top[i]];
        if (!cut_at(p)) continue;
        n_patches++;
    }
}

// Track patch p: (px, py) is the predicted top-left corner (tracking resolution). Lucas-Kanade, inverse compositional,
// translation + brightness offset, on a small crop built straight from the raw frame. SIMD over 4-pixel row chunks.
bool track(const Patch& p, double& px, double& py) {
#if PROF_SKIP_TRACK
    return false;
#endif
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
Track ltracks[N_TRACK];   // LONG_N: previous window's patches tracked again (about 14-frame baseline)
int n_long = 0;
double lpar_sum = 0;

// Track the patches of subset `si` into the current frame; the rotation residual of this subset updates the gyro bias.
void track_subset(int si) {
    if (PROP_MODE == 0 && (!PROP_ONCE || si == 0)) kf_propagate();
    if (PROP_MODE == 2 && si == SPLIT - 1) kf_propagate();  // PROP_ONCE: one propagation per tracking burst (IMU of the next
                                                // frames is folded into the following propagation)
    Mat3 R_wc0f = mul(R_wb_kf[si], R_ic), R_wc1f = mul(R_wb, R_ic);
    Mat3 R10 = mul(transpose(R_wc1f), R_wc0f);  // subset keyframe camera -> current camera (gyro)
    const int first = n_tracks;
    double ps = 0;
    R_wb_trk[si] = R_wb;
    for (int s = 0; s < n_patches; s++) {
        Patch& p = patches[s];
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
        p.ok = 1; p.lx = px - PC; p.ly = py - PC; p.lfx = px - px0; p.lfy = py - py0;
    }
    par_sum += ps;
#if LONG_N > 0
    if (n_prev > 0) {   // the previous window's patches of this subset, tracked again: ~3x the baseline
        Mat3 R_tr = mul(transpose(R_wc1f), mul(R_wb_trk_prev[si], R_ic));   // camera at their last tracking -> now
        Mat3 R10L = mul(transpose(R_wc1f), mul(R_wb_kf_prev[si], R_ic));    // camera at their template -> now
        int used = 0;
        for (int s = 0; s < n_prev && used < LONG_N && n_long < N_TRACK; s++) {
            const Patch& p = prevp[s];
            if (p.sub != si || !p.ok) continue;
            used++;
            Vec3 b{(p.lx + PC - cx_s) / fx_s, (p.ly + PC - cy_s) / fy_s, 1.0};
            Vec3 br = mul(R_tr, b);
            if (br.z < 0.2) continue;
            double px = cx_s + fx_s * br.x / br.z + LONG_EXT * p.lfx - PC;
            double py = cy_s + fy_s * br.y / br.z + LONG_EXT * p.lfy - PC;
            if (!track(p, px, py)) continue;
            px += PC; py += PC;
            Vec3 f0{(p.ox + PC - cx_s) / fx_s, (p.oy + PC - cy_s) / fy_s, 1.0};
            Vec3 fr = mul(R10L, f0);
            if (fr.z < 0.2) continue;
            double qx = cx_s + fx_s * fr.x / fr.z, qy = cy_s + fy_s * fr.y / fr.z;
            Vec3 f1{(px - cx_s) / fx_s, (py - cy_s) / fy_s, 1.0};
            Track& t = ltracks[n_long++];
            t.fr = scale(fr, 1.0 / norm(fr));
            t.f1 = scale(f1, 1.0 / norm(f1));
            t.R_wc = R_wc1f;
            t.par = std::sqrt((px - qx) * (px - qx) + (py - qy) * (py - qy));
            lpar_sum += t.par;
        }
    }
#endif
    const int nt = n_tracks - first;
    if (!ROT_FROM_VISION || ROT_CLONE || PROF_SKIP_DIR > 1 || nt < MIN_TRACKS / SPLIT + 1 || !(win_T[si] > 0)) return;
    // rotation error of the gyro prediction: true f1 ~ exp(delta) fr  (delta in the current camera frame)
    double par = ps / nt;
    Vec3 dlever = sub_lever();
    Vec3 d_est = add(sub(p_w, p_kf), dlever);
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
#if FIXED_GAIN
        {   // fixed gains: gyro bias and current attitude take a constant fraction of the rotation residual
            double kg = par < MIN_PAR_PX ? FG_KG_ROT : FG_KG_TRANS;
            bg = add(bg, scale(db, kg / win_T[si]));
            if (FG_KTH > 0) { R_wb = mul(R_wb, exp_so3(scale(db, -FG_KTH * kg))); orthonormalize(R_wb); }
            return;
        }
#endif
        double sr = SIG_ROT / std::sqrt((double)nt) + SIG_ROT_FLOOR + SIG_ROT_PAR * (par < MIN_PAR_PX ? 0.0 : par);
        for (int k = 0; k < 3; k++) {
            double Hrow[NS] = {0};
            Hrow[IBG + k] = win_T[si];
            kf_update_row(Hrow, dbv[k], sr * sr);
        }
        kf_inject();
    }
}

// FIXED_GAIN: apply a gained displacement residual r (measured - predicted camera displacement since the clone)
void fg_apply(Vec3 r) {
    double T = last_t - t_clone;
    if (!(T > 0.05)) return;
    double warm = FG_WARM > 0 && n_windows < FG_WARM ? 1.0 + (FG_WARM - n_windows) * FG_WARM_GAIN : 1.0;
    p_w = add(p_w, scale(r, FG_KP * warm));
    v_w = add(v_w, scale(r, FG_KV * warm / T));
    Vec3 c = scale(r, FG_KB / (T * T));        // wanted increase of the world acceleration
#if FG_TILT
    // horizontal part -> tilt (a tilt error phi shifts the world acceleration by phi x g_up); vertical part -> bias
    bw.z -= c.z;
    Vec3 phi{-c.y / GRAVITY, c.x / GRAVITY, 0.0};
    R_wb = mul(exp_so3(phi), R_wb);
    orthonormalize(R_wb);
    // integral term (Mahony): a persistent tilt correction means a roll/pitch gyro bias
    bg = sub(bg, scale(mulT(R_wb, phi), FG_KTB / T));
#else
    bw = sub(bw, c);
#endif
}

// ADAPT_R: geometric running estimate of how much the nominal noise must be scaled for consistent innovations
inline double adapt(double lam, double nis) {
    if (!(nis >= 0)) return lam;
    double z = nis < ADAPT_NIS_MAX ? nis : ADAPT_NIS_MAX;
    double r = lam * (z > 1e-3 ? z : 1e-3);            // the factor that would have made this window's NIS ~ 1
    double l = std::exp((1.0 - ADAPT_BETA) * std::log(lam) + ADAPT_BETA * std::log(r));
    return l < ADAPT_MIN ? ADAPT_MIN : (l > ADAPT_MAX ? ADAPT_MAX : l);
}

#if LONG_N > 0
Vec3 u_long{0, 0, 0};
bool long_ok = false;
double long_w = 0;
// LONG_MODE 2: direction of the long tracks only (no filter rows); blended into the short direction
void long_direction() {
    long_ok = false;
    const int nt = n_long;
    if (nt < LONG_MIN) return;
    const double lpar = lpar_sum / nt;
    if (lpar < LONG_MIN_PAR) return;
    Vec3 ns[N_TRACK];
    for (int i = 0; i < nt; i++) ns[i] = mul(ltracks[i].R_wc, cross(ltracks[i].fr, ltracks[i].f1));
    Vec3 u{0.577, 0.577, 0.577};
    double M[9] = {0};
    for (int i = 0; i < nt; i++) accum(M, ns[i]);
    double l1 = smallest_eig(M, u);
    double lmid = 0.5 * (M[0] + M[4] + M[8] - l1);
    if (!(lmid > 0) || l1 / lmid > MAX_Q) return;
    u_long = u;
    long_w = LONG_W * lpar * std::sqrt((double)nt);   // weight ~ parallax x sqrt(tracks), like the inverse noise
    long_ok = true;
}
// Long-baseline direction: the previous window's patches, tracked from their templates (previous window) to now,
// constrain the displacement since the previous clone, d_L = p - p_kf2: e . d_L = 0 for e perpendicular to u_L.
void long_update() {
    const int nt = n_long;
    if (nt < LONG_MIN) return;
    const double lpar = lpar_sum / nt;
    if (lpar < LONG_MIN_PAR) return;
    Vec3 d = add(sub(p_w, p_kf2), sub(mul(R_wb, r_cam_in_imu), mul(R_wb_clone2, r_cam_in_imu)));
    double dn = norm(d);
    Vec3 ns[N_TRACK];
    for (int i = 0; i < nt; i++) ns[i] = mul(ltracks[i].R_wc, cross(ltracks[i].fr, ltracks[i].f1));
    Vec3 u = dn > 1e-6 ? scale(d, 1.0 / dn) : Vec3{0.577, 0.577, 0.577};
    double M[9] = {0};
    for (int i = 0; i < nt; i++) accum(M, ns[i]);
    double l1 = smallest_eig(M, u);
    double lmid = 0.5 * (M[0] + M[4] + M[8] - l1);
    if (!(lmid > 0)) return;
    double q = l1 / lmid;
    if (q > MAX_Q) return;
    double score = 0;   // sign: cheirality, then the generator's forward prior (as for the short tracks)
    for (int i = 0; i < nt; i++) {
        Vec3 uc = mulT(ltracks[i].R_wc, u);
        const Vec3& fr = ltracks[i].fr; const Vec3& f1 = ltracks[i].f1;
        double xr = fr.x / fr.z, yr = fr.y / fr.z;
        score += (f1.x / f1.z - xr) * (xr * uc.z - uc.x) + (f1.y / f1.z - yr) * (yr * uc.z - uc.y);
    }
    if (score < 0) u = scale(u, -1.0);
    if (SIGN_FWD > 0) {
        Vec3 cz = mul(mul(R_wb, R_ic), Vec3{0, 0, 1});
        double a = dot(u, cz);
        if (std::fabs(a) > SIGN_FWD && a < 0) u = scale(u, -1.0);
    }
    if (LONG_SIGN_SHORT && short_ok) {   // few tracks: take the sign from this window's short direction
        double c = dot(u, last_u_short);
        if (c < 0) u = scale(u, -1.0);
        if (std::fabs(c) < LONG_AGREE) return;   // the two directions disagree: skip
    }
#ifdef VIO_DEBUG
    std::fprintf(stderr, "L n %d lpar %.2f q %.3f dot_short %.3f dot_dest %.3f dn %.3f\n", nt, lpar, q, dot(u, last_u_short), dn > 1e-6 ? dot(u, d) / dn : 0.0, dn);
#endif
    Vec3 a = std::fabs(u.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 e1 = cross(u, a); e1 = scale(e1, 1.0 / norm(e1));
    Vec3 e2 = cross(u, e1);
    double sig = (std::sqrt(q / nt) * SIG_DIR_SCALE + SIG_DIR_PX / (lpar * std::sqrt((double)nt)) + SIG_DIR_FLOOR) * LONG_SIG;
    double Rn = sig * sig * (dn * dn + D_FLOOR * D_FLOOR);
    Vec3 es[2] = {e1, e2};
    for (int k = 0; k < 2; k++) {
        double Hrow[NS] = {0};
        Vec3 e = es[k], ed = cross(e, d);
        Hrow[IP] = e.x; Hrow[IP + 1] = e.y; Hrow[IP + 2] = e.z;
        Hrow[IPC2] = -e.x; Hrow[IPC2 + 1] = -e.y; Hrow[IPC2 + 2] = -e.z;
        Hrow[ITH] = ed.x; Hrow[ITH + 1] = ed.y; Hrow[ITH + 2] = ed.z;
        kf_update_row(Hrow, -dot(e, d), Rn);
    }
    kf_inject();
}
#endif

// All tracks of the window -> one motion direction (+ speed proxy) for the displacement since the keyframe clone.
void direction_update_short();
void direction_update() {
#if PROF_SKIP_DIR
    return;
#endif
    short_ok = false;
#if LONG_N > 0
    if (LONG_MODE == 2) long_direction();
#endif
    direction_update_short();
#if LONG_N > 0
    if (LONG_MODE == 1) long_update();
#endif
}
#if ROT_CLONE
constexpr int DIR_TH = (ROT_DIR_CLONE && !LROT) ? ITC : ITH;
// relative-rotation measurement eps = dtheta(now) - dtheta(clone); noise from the normal matrix A and bearing sigma sb
void rot_rows(const double A[9], Vec3 eps, double sb) {
    const double ev[3] = {eps.x, eps.y, eps.z};
    for (int k = 0; k < 3; k++) {
        Vec3 unit{k == 0 ? 1.0 : 0.0, k == 1 ? 1.0 : 0.0, k == 2 ? 1.0 : 0.0}, col;
        double var = 1.0;
        if (solve3(A, ROT_DAMP, unit, col)) var = k == 0 ? col.x : (k == 1 ? col.y : col.z);
        if (!(var > 0)) var = 1.0;
        double Rn = SIG_ROTC_K * SIG_ROTC_K * sb * sb * var + SIG_ROTC_MIN * SIG_ROTC_MIN;
        double Hrow[NS] = {0};
        Hrow[ITH + k] = 1.0; Hrow[ITC + k] = -1.0;
        d10_rows++; d11_rn[k] = std::sqrt(Rn);
        if (kf_update_row(Hrow, ROT_SIGN * ev[k], Rn)) d10_acc++;
    }
}
#else
constexpr int DIR_TH = ITH;
#endif
void direction_update_short() {
    const int nt = n_tracks;
    last_tracks = nt;
    last_used = 0;
    last_parallax = nt ? par_sum / nt : 0;
    if (PAR_MEDIAN && nt >= 3) {   // median per-track parallax: one bad track cannot inflate the speed proxy
        double pv[N_TRACK];
        for (int i = 0; i < nt; i++) pv[i] = tracks[i].par;
        for (int i = 1; i < nt; i++) { double v = pv[i]; int j = i - 1; while (j >= 0 && pv[j] > v) { pv[j + 1] = pv[j]; j--; } pv[j + 1] = v; }
        last_parallax = PAR_MEDIAN_K * pv[nt / 2];
    }
    diag_reason = 1;
    if (nt < MIN_TRACKS) return;
    Vec3 d_est = add(sub(p_w, p_kf), sub_lever());
    double w1g = 0;
#if TRK == 2
    if (TRK_DISP) {   // the tracks span each group's own window (1-2 frames earlier than clone -> now): use that displacement
        int c1 = 0;
        for (int i = 0; i < nt; i++) c1 += trk_idx[i] & 1;
        const double w1 = (double)c1 / nt;
        w1g = w1;
        d_est = add(scale(dispg[0], 1.0 - w1), scale(dispg[1], w1));
    }
#endif
    double dn = norm(d_est);
    if (last_parallax < MIN_PAR_PX) {
        diag_reason = 2;
        // little translational flow on well-tracked patches -> small displacement (indoor depths are bounded)
#if FIXED_GAIN
        if (SMALL_FLOW_UPDATE) { fg_apply(scale(d_est, -FG_KSMALL)); last_used = 1; }
        n_windows++;
        return;
#endif
        if (SMALL_FLOW_UPDATE) {
            double sd = (last_parallax + SMALL_FLOW_PX) / fx_s * DEPTH_PRIOR;
#if SCALE_STATE && !STRUCT
            sd *= std::exp(s_log);
#endif
            double dv[3] = {d_est.x, d_est.y, d_est.z};
            for (int k = 0; k < 3; k++) {
                double Hrow[NS] = {0};
                Hrow[IP + k] = 1.0; Hrow[IPC + k] = -1.0;
                if (kf_update_row(Hrow, -dv[k], sd * sd)) last_used++;
            }
        }
#if ROT_CLONE && !LROT
        {   // nearly pure rotation: wr = w1 + eps x w1  (eps = attitude error now - at the clone, world frame)
            double A[9] = {0};
            Vec3 b{0, 0, 0};
            double ss = 0;
            for (int i = 0; i < nt; i++) {
                Vec3 wr = mul(tracks[i].R_wc, tracks[i].fr), w1 = mul(tracks[i].R_wc, tracks[i].f1);
                A[0] += 1 - w1.x * w1.x; A[1] -= w1.x * w1.y; A[2] -= w1.x * w1.z;
                A[4] += 1 - w1.y * w1.y; A[5] -= w1.y * w1.z; A[8] += 1 - w1.z * w1.z;
                b = add(b, cross(w1, wr));
            }
            A[3] = A[1]; A[6] = A[2]; A[7] = A[5];
            Vec3 eps;
            if (solve3(A, ROT_DAMP, b, eps) && norm(eps) < ROT_MAX) {
                for (int i = 0; i < nt; i++) {
                    Vec3 wr = mul(tracks[i].R_wc, tracks[i].fr), w1 = mul(tracks[i].R_wc, tracks[i].f1);
                    Vec3 rr = sub(sub(wr, w1), cross(eps, w1));
                    ss += dot(rr, rr);
                }
                double sb = std::sqrt(ss / (2.0 * nt)) + SIG_ROTC_FLOOR + last_parallax / fx_s;
                rot_rows(A, eps, sb);
            }
        }
#endif
        kf_inject();
        return;
    }
    // the direction most orthogonal to all n = R_wc (fr x f1) (world frame)
    Vec3 ns[N_TRACK];
    for (int i = 0; i < nt; i++) ns[i] = mul(tracks[i].R_wc, cross(tracks[i].fr, tracks[i].f1));
#if ROT_CLONE && !LROT
    Vec3 wr_[N_TRACK], w1_[N_TRACK];
    for (int i = 0; i < nt; i++) { wr_[i] = mul(tracks[i].R_wc, tracks[i].fr); w1_[i] = mul(tracks[i].R_wc, tracks[i].f1); }
#endif
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
    diag_reason = 3;
    if (kept < MIN_TRACKS) { kf_inject(); return; }
    double l1 = smallest_eig(M2, u);
#if ROT_CLONE && !LROT
    // joint rotation + direction: alternate  eps <- least squares of the epipolar residuals u.(wr x w1) for w1 -> w1 + eps x w1,
    // then re-fit u on the corrected bearings. eps = attitude error now minus at the clone (world frame).
    Vec3 eps_tot{0, 0, 0};
    double Arot[9] = {0};
    bool rot_ok = true;
    for (int it = 0; it < ROT_ITERS && rot_ok; it++) {
        double A[9] = {0};
        Vec3 b{0, 0, 0};
        for (int i = 0; i < nt; i++) {
            if (!(r[i] <= 3.0 * med + 1e-12)) continue;
            Vec3 J = sub(scale(u, dot(wr_[i], w1_[i])), scale(wr_[i], dot(u, w1_[i])));
            accum(A, J);
            b = add(b, scale(J, -dot(u, ns[i])));
        }
        Vec3 e;
        if (!solve3(A, ROT_DAMP, b, e)) { rot_ok = false; break; }
        eps_tot = add(eps_tot, e);
        for (int k = 0; k < 9; k++) Arot[k] = A[k];
        if (!ROT_REFIT) break;   // keep the gyro-derotated direction; eps is only a measurement
        for (int k = 0; k < 9; k++) M2[k] = 0;
        for (int i = 0; i < nt; i++) {
            w1_[i] = add(w1_[i], cross(e, w1_[i]));
            ns[i] = cross(wr_[i], w1_[i]);
            if (r[i] <= 3.0 * med + 1e-12) accum(M2, ns[i]);
        }
        l1 = smallest_eig(M2, u);
    }
    diag_reason = 5;
    if (!rot_ok || norm(eps_tot) > ROT_MAX) { kf_inject(); return; }
#endif
    double lmid = 0.5 * (M2[0] + M2[4] + M2[8] - l1);
    if (!(lmid > 0)) { kf_inject(); return; }
    double q = l1 / lmid;
    if (q < 0) q = 0;
    last_quality = q;
    diag_reason = 4;
    if (q > MAX_Q) { kf_inject(); return; }
    diag_reason = 6;
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
#if LONG_N > 0
    if (LONG_MODE == 2 && long_ok) {   // blend in the long-baseline direction (same sign as the short one)
        Vec3 ul = dot(u_long, u) < 0 ? scale(u_long, -1.0) : u_long;
        if (dot(ul, u) > LONG_AGREE) {
            double ws = last_parallax * std::sqrt((double)kept);
            Vec3 b = add(scale(u, ws), scale(ul, long_w));
            u = scale(b, 1.0 / norm(b));
            q *= ws / (ws + long_w);   // a better-determined direction: smaller noise below
        }
    }
#endif
    last_u_short = u;
    short_ok = true;
#if STRUCT
    if (struct_skip && last_parallax >= STRUCT_CHK_PX) {
        // the map-based displacement must agree with the depth-free direction and with a plausible scene depth
        const double cs = dn > 1e-6 ? dot(u, d_est) / dn : 1.0;
        const double dm0 = last_parallax / fx_s * SPEED_DEPTH;
        if (cs < STRUCT_CHK_COS || dn > STRUCT_CHK_RATIO * dm0 || dn * STRUCT_CHK_RATIO < dm0) { struct_wipe = true; struct_skip = false; }
    }
#endif
#if FLOOR
    double eta_f = 0;      // floor: height of the keyframe camera above the floor in units of |d| (0 = no floor tracks)
    {
        double et[N_TRACK]; int ne = 0;
        for (int i = 0; i < nt; i++) {
            const Vec3 fr = tracks[i].fr, f1 = tracks[i].f1;
            Vec3 wr = mul(tracks[i].R_wc, fr);
            if (wr.z > -FLOOR_DOWN || tracks[i].par < FLOOR_MIN_PAR) continue;
            if (!(r[i] <= 3.0 * med + 1e-12)) continue;
            Vec3 uc = mulT(tracks[i].R_wc, u);
            Vec3 c = cross(fr, f1);
            double c2 = dot(c, c);
            if (c2 < 1e-12) continue;
            double lam = dot(cross(uc, f1), c) / c2;
            if (lam > 0) et[ne++] = -wr.z * lam;
        }
        for (int i = 1; i < ne; i++) { double v = et[i]; int j = i - 1; while (j >= 0 && et[j] < v) { et[j + 1] = et[j]; j--; } et[j + 1] = v; }
        if (ne >= FLOOR_MIN_N) eta_f = et[FLOOR_RANK < ne ? FLOOR_RANK : ne - 1];
    }
#endif
#ifdef DIAG2
    {   // floor-plane check: height below the keyframe camera of each downward track, in units of |d|
        double et[N_TRACK]; int ne = 0;
        for (int i = 0; i < nt; i++) {
            const Vec3 fr = tracks[i].fr, f1 = tracks[i].f1;
            Vec3 wr = mul(tracks[i].R_wc, fr);
            if (wr.z > -0.26 || tracks[i].par < 0.7) continue;
            Vec3 uc = mulT(tracks[i].R_wc, u);
            Vec3 c = cross(fr, f1);
            double c2 = dot(c, c);
            if (c2 < 1e-12) continue;
            double lam = dot(cross(uc, f1), c) / c2;
            if (lam > 0) et[ne++] = -wr.z * lam;
        }
        for (int i = 1; i < ne; i++) { double v = et[i]; int j = i - 1; while (j >= 0 && et[j] < v) { et[j + 1] = et[j]; j--; } et[j + 1] = v; }
        diag_eta[0] = ne; diag_eta[1] = ne > 0 ? et[0] : 0; diag_eta[2] = ne > 1 ? et[1] : 0; diag_eta[3] = ne > 2 ? et[ne / 2] : 0;
    }
#endif

#if FIXED_GAIN
    {
        double dmeas = last_parallax / fx_s * PAR_SPEED;
        double along = dot(u, d_est);
        Vec3 perp = sub(d_est, scale(u, along));
        // gate: perpendicular residual far larger than the direction can explain
        if (norm(perp) > FG_GATE * (dn + D_FLOOR)) { n_windows++; return; }
        double w = 1.0;
        if (FG_SIG0 > 0) {   // gain schedule: same constants, scaled by this window's direction quality
            double sig_dir = std::sqrt(q / kept) * SIG_DIR_SCALE + SIG_DIR_PX / (last_parallax * std::sqrt((double)kept)) + SIG_DIR_FLOOR;
            w = 1.0 / (1.0 + (sig_dir / FG_SIG0) * (sig_dir / FG_SIG0));
        }
        fg_apply(add(scale(u, FG_KPAR * std::pow(w, FG_WPAR) * (dmeas - along)), scale(perp, -FG_KPERP * w)));
        last_used = 3;
        n_windows++;
        return;
    }
#endif
    if (PROF_DIR_STOP) { kf_inject(); return; }
    Vec3 a = std::fabs(u.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 e1 = cross(u, a); e1 = scale(e1, 1.0 / norm(e1));
    Vec3 e2 = cross(u, e1);
    double sig_dir = std::sqrt(q / kept) * SIG_DIR_SCALE + SIG_DIR_PX / (last_parallax * std::sqrt((double)kept)) + SIG_DIR_FLOOR;
    double Rn = sig_dir * sig_dir * (dn * dn + D_FLOOR * D_FLOOR);
    if (DIR_NOISE_MODE == 1) {   // displacement scale from the image (parallax x typical depth), not from the filter's own |d|
        const double dv = last_parallax / fx_s * DIR_DEPTH;
        Rn = sig_dir * sig_dir * (dv * dv + D_FLOOR * D_FLOOR);
    }
    if (ADAPT_R) Rn *= lam_dir;
    huber_on = true;
    Vec3 es[2] = {e1, e2};
    double Hrow[NS];
#if ROT_CLONE && !LROT
    if (ROT_ITERS > 0) {
        double ss = 0;
        for (int i = 0; i < nt; i++)
            if (r[i] <= 3.0 * med + 1e-12) { double e_ = dot(u, ns[i]); ss += e_ * e_; }
        d10_win++; d10_kept = kept;
        if (kept >= ROTC_MIN_KEPT) {
        double sb = std::sqrt(ss / (kept > 6 ? kept - 5 : 1)) + SIG_ROTC_FLOOR;
        rot_rows(Arot, eps_tot, sb);
        }
    }
#endif
    double nis_sum = 0;
    int nis_n = 0;
    for (int k = 0; k < 2; k++) {  // e . d = 0; the world e is exp(dtheta) e_est, so dh/dtheta = e x d
        Vec3 e = es[k], ed = cross(e, d_est);
        row_d(Hrow, e);
        Hrow[DIR_TH] = ed.x; Hrow[DIR_TH + 1] = ed.y; Hrow[DIR_TH + 2] = ed.z;
        if (!struct_skip && kf_update_row(Hrow, -dot(e, d_est), Rn)) last_used++;
        nis_sum += last_nis; nis_n++;
    }
    if (ADAPT_R) lam_dir = adapt(lam_dir, nis_sum / nis_n);
    // speed proxy along u (Sim(3) only needs a consistent scale):
    //   SPEED_MODE 0: |d| ~ mean parallax * PAR_SPEED / f
    //   SPEED_MODE 1: per track, flow / (flow that a unit displacement along u gives at unit depth) = |d| / Z;
    //                 median over tracks, times a typical depth (removes the bias of forward vs sideways motion)
    double dmeas = last_parallax / fx_s * PAR_SPEED;
    diag_sp[0] = last_parallax / fx_s;
    double sp_med = 0, sp_q3 = 0; int sp_nr = 0;
#if FLOOR
    diag_sp[2] = eta_f;
#endif
    if (SPEED_MODE == 1) {
        double rr[N_TRACK];
        int nr = 0;
        for (int i = 0; i < nt; i++) {
            Vec3 uc = mulT(tracks[i].R_wc, u);
            const Vec3& fr = tracks[i].fr; const Vec3& f1 = tracks[i].f1;
            double xr = fr.x / fr.z, yr = fr.y / fr.z;
            double fx_ = f1.x / f1.z - xr, fy_ = f1.y / f1.z - yr;
            double gx = xr * uc.z - uc.x, gy = yr * uc.z - uc.y;
            double g2 = gx * gx + gy * gy;
            if (g2 < SPEED_GMIN * SPEED_GMIN) continue;      // near the focus of expansion: no speed information
            double r = (fx_ * gx + fy_ * gy) / g2;             // projection of the flow on the expected flow direction
            if (r > 0) rr[nr++] = r;
        }
        diag_sp[1] = nr >= 3 ? 1 : 0;
        sp_nr = nr;
        if (nr >= 3) {
            for (int i = 1; i < nr; i++) { double v = rr[i]; int j = i - 1; while (j >= 0 && rr[j] > v) { rr[j + 1] = rr[j]; j--; } rr[j + 1] = v; }
            dmeas = rr[nr / 2] * SPEED_DEPTH;
            diag_sp[1] = rr[nr / 2]; diag_sp[3] = rr[(3 * nr) / 4];
            sp_med = rr[nr / 2]; sp_q3 = rr[(3 * nr) / 4];
        }
    }
    if (NHC_D_REL > 0) {   // generator prior on the window displacement: little sideways camera motion
        const Mat3 R_wc = mul(R_wb, R_ic);
        Vec3 c = mul(R_wc, Vec3{1, 0, 0});
        row_d(Hrow, c);
        if (!NHC_NO_TH) { Vec3 cd = cross(c, d_est); Hrow[ITH] = cd.x; Hrow[ITH + 1] = cd.y; Hrow[ITH + 2] = cd.z; }
        double sn = NHC_D_REL * (dn + D_FLOOR);
        kf_update_row(Hrow, -dot(c, d_est), sn * sn);
        for (int j = ITH; j < ITH + 3; j++) Hrow[j] = 0;
    }
    bool floor_used = false;
#if FLOOR
    if (eta_f > 0) {   // eta (u . d) = height of the keyframe above the floor = p_kf.z - z_floor
        const double hk = p_kf.z - z_floor;
        row_d(Hrow, scale(u, eta_f));
        Hrow[IPC + 2] -= 1.0;
        Hrow[IZF] = 1.0;
        const double sf = FLOOR_REL * (hk > FLOOR_HMIN ? hk : FLOOR_HMIN) + 0.03;
        if (kf_update_row(Hrow, hk - eta_f * dot(u, d_est), sf * sf)) { last_used++; floor_used = true; }
    }
    const bool floor_conv = P[IZF * NS + IZF] < FLOOR_CONV * FLOOR_CONV;
#else
    const bool floor_conv = false;
#endif
#if SPEED_NN || defined(DIAG9)
    double feat[NFEAT];
    {   // speed-model features (dimensionless): flow/depth ratio, its spread, support, direction quality,
        // filter-vs-measurement ratio, turn rate, camera pitch, track elevation, field of view
        const double base = sp_nr >= 3 ? sp_med : last_parallax / fx_s;
        const Mat3 R_wc = mul(R_wb, R_ic);
        double ez = 0;
        for (int i = 0; i < nt; i++) { Vec3 wr = mul(tracks[i].R_wc, tracks[i].f1); ez += wr.z / std::sqrt(dot(wr, wr)); }
        feat[0] = std::log(base + 1e-4);
        feat[1] = sp_nr >= 3 ? std::log((sp_q3 + 1e-4) / (sp_med + 1e-4)) : 0.0;
        feat[2] = (double)sp_nr / N_TRACK;
        feat[3] = std::sqrt(q);
        feat[4] = std::log((std::fabs(dot(u, d_est)) + 0.01) / (dmeas + 0.01));
        feat[5] = norm(omega_b);
        feat[6] = R_wc.m[8];
        feat[7] = nt ? ez / nt : 0.0;
        feat[8] = fx_s / (0.5 * SW);
        for (int i = 0; i < NFEAT; i++) if (!std::isfinite(feat[i])) feat[i] = 0;
    }
#if SPEED_NN
    {
        double y = speednn_eval(feat);
        if (y > SPEED_NN_CLAMP) y = SPEED_NN_CLAMP;
        if (y < -SPEED_NN_CLAMP) y = -SPEED_NN_CLAMP;
        dmeas *= std::exp(y);
    }
#endif
#ifdef DIAG9
    diag9[0] = dmeas;
    for (int i = 0; i < NFEAT; i++) diag9[1 + i] = feat[i];
    diag9[10] = w1g; diag9[11] = ++diag9_n; diag9[12] = 0;
#endif
#endif
    if (SPEED_ROW && (!struct_skip || STRUCT_KEEP_SPEED) && !(floor_conv && floor_used)) {
#ifdef DIAG9
    diag9[12] = 1;
#endif
    double sd = PAR_SPEED_REL * dmeas + 0.01;
    if (struct_skip) sd *= STRUCT_SPEED_K;   // with a map the assumed-depth speed is only a weak scale anchor
    if (ADAPT_R) sd *= std::sqrt(lam_spd);
    row_d(Hrow, u);
#if SCALE_STATE && !STRUCT
    // u . d = dmeas * exp(s): y = dmeas e^s - u.d, dh/ds = -dmeas e^s
    double es_ = std::exp(s_log);
    Hrow[IS] = -dmeas * es_;
    if (kf_update_row(Hrow, dmeas * es_ - dot(u, d_est), sd * sd * es_ * es_)) last_used++;
#else
    if (kf_update_row(Hrow, dmeas - dot(u, d_est), sd * sd)) last_used++;
#endif
    if (ADAPT_R) lam_spd = adapt(lam_spd, last_nis);
    }
    huber_on = false;
#if SCALE_STATE && !STRUCT
    if (SIG_S_PRIOR > 0) {   // weak pull of the log scale toward 0 (keeps it from collapsing)
        double H2[NS] = {0};
        H2[IS] = 1.0;
        kf_update_row(H2, -s_log, SIG_S_PRIOR * SIG_S_PRIOR);
    }
#endif
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
    const bool quiet = std::sqrt(var_a) < REST_ACCEL_STD && std::sqrt(var_g) < REST_GYRO_STD;
    bool at_rest = quiet && norm(mean_g) < REST_GYRO_MEAN && std::fabs(na - GRAVITY) < REST_ACCEL_MEAN;
    // HOVER_INIT: a quiet IMU with a plausible mean rate is a hover with a large gyro bias, not motion
    if (HOVER_INIT && quiet && norm(mean_g) < 2.0 * REST_GYRO_MEAN && std::fabs(na - GRAVITY) < 1.5 * REST_ACCEL_MEAN) at_rest = true;
    bg = at_rest ? mean_g : Vec3{0, 0, 0};
    ba = {0, 0, 0};
    Vec3 z = na > 1e-6 ? scale(mean_a, 1.0 / na) : Vec3{0, 0, 1};
    Vec3 xc = std::fabs(z.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 x = sub(xc, scale(z, dot(xc, z)));
    x = scale(x, 1.0 / norm(x));
    Vec3 y = cross(z, x);
    R_wb = {{x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z}};
    if (THRUST_ONLY) {   // thrust is along body z: at hover body z is vertical, and the mean a_z gives the thrust bias
        R_wb = eye();
        if (at_rest) ba.z = mean_a.z - GRAVITY;
    }
#if MOVE_START
    Mat3 R_b0 = R_wb;                  // body orientation at the first frame (world = level frame of that body)
    if (!at_rest) { orthonormalize(R_cal); R_wb = mul(R_b0, R_cal); }   // the body has turned since the first frame: keep c0 = first frame
#endif
    p_w = v_w = {0, 0, 0};
    std::memset(P, 0, sizeof P);
    double sv = at_rest ? (HOVER_INIT ? SIG_V0_HOVER : SIG_V0_REST) : SIG_V0_MOVING;
    double st = at_rest ? (THRUST_ONLY ? SIG_TILT0_HOVER : SIG_TILT0_REST) : SIG_TILT0_MOVING;
    double sbg = at_rest ? SIG_BG0_REST : SIG_BG0_MOVING;
    for (int i = 0; i < 3; i++) {
        P[(IP + i) * NS + IP + i] = 1e-8;
        P[(IV + i) * NS + IV + i] = sv * sv;
        P[(IBA + i) * NS + IBA + i] = THRUST_ONLY ? (i == 2 ? (at_rest ? SIG_BAZ0 * SIG_BAZ0 : SIG_BA0 * SIG_BA0) : 0.0) : SIG_BA0 * SIG_BA0;
#if WIND_STATE && THRUST_ONLY
        if (i < 2) P[(IBA + i) * NS + IBA + i] = SIG_WIND0 * SIG_WIND0;
#endif
        P[(IBG + i) * NS + IBG + i] = sbg * sbg;
#if SCALE_STATE
    P[IS * NS + IS] = SIG_S0 * SIG_S0;
#endif
    }
    P[(ITH + 0) * NS + ITH + 0] = st * st;
    P[(ITH + 1) * NS + ITH + 1] = st * st;
    P[(ITH + 2) * NS + ITH + 2] = 1e-6;  // yaw is unobservable and defines the frame
#if MOVE_START && MS_PV
    if (!at_rest && t_first >= 0) {   // unknown motion since the first frame: position = T * initial velocity (to first order)
        double T = std::fmax(0.0, t - t_first);
        for (int i = 0; i < 3; i++) {
            P[(IP + i) * NS + IP + i] = T * T * sv * sv + 1e-8;
            P[(IP + i) * NS + IV + i] = P[(IV + i) * NS + IP + i] = T * sv * sv;
        }
    }
#endif
#if FLOOR
    P[IZF * NS + IZF] = FLOOR_SIG0 * FLOOR_SIG0;
    z_floor = FLOOR_Z0;
#endif
    for (int i = 0; i < NS; i++) dx[i] = 0;
#if MOVE_START
    R_wc0 = mul(R_b0, R_ic);
    p_wc0 = mul(R_b0, r_cam_in_imu);
#else
    R_wc0 = mul(R_wb, R_ic);
    p_wc0 = mul(R_wb, r_cam_in_imu);
#endif
    last_t = t;
    initialized = true;
}

void integrate(const double* imu, int count) {
    for (int k = 0; k < count; k++) {
        const double* r = imu + 7 * k;
        double t = r[0];
#if ADAPT_IMU
        if (ad_n >= 2) {
            for (int i = 0; i < 3; i++) {
                const double dg = r[4 + i] - 2.0 * ad_pg[0][i] + ad_pg[1][i], da = r[1 + i] - 2.0 * ad_pa[0][i] + ad_pa[1][i];
                ad_sg += dg * dg; ad_sa += da * da;
            }
            ad_cnt++;
        }
        for (int i = 0; i < 3; i++) { ad_pg[1][i] = ad_pg[0][i]; ad_pg[0][i] = r[4 + i]; ad_pa[1][i] = ad_pa[0][i]; ad_pa[0][i] = r[1 + i]; }
        ad_n++;
#endif
        Vec3 a = THRUST_ONLY ? Vec3{0, 0, r[3] - ba.z} : sub(Vec3{r[1], r[2], r[3]}, ba);
        Vec3 w = sub(Vec3{r[4], r[5], r[6]}, bg);
        omega_b = w;
        double dt = t - last_t;
        last_t = t;
        if (!(dt > 0) || dt > 0.1) continue;
        // MIDPOINT: specific force rotated with the attitude at the middle of the sample interval (first order)
        Vec3 f = mul(R_wb, MIDPOINT ? add(a, scale(cross(w, a), 0.5 * dt)) : a);
        Vec3 acc = sub(add(f, g_w), bw);
#if WIND_STATE && THRUST_ONLY
        acc.x -= ba.x; acc.y -= ba.y;   // ba.x, ba.y = world-frame horizontal acceleration bias (wind force / mass)
#endif
        p_w = add(p_w, add(scale(v_w, dt), scale(acc, 0.5 * dt * dt)));
        v_w = add(v_w, scale(acc, dt));
        rotate_small(R_wb, scale(w, dt));
        for (int i = 0; i < 9; i++) acc_R.m[i] += R_wb.m[i];
        acc_f = add(acc_f, f);
        acc_n++;
        acc_T += dt;
        for (int k = 0; k < SPLIT; k++) win_T[k] += dt;
    }
    orthonormalize(R_wb);
#if ADAPT_IMU
    if (ad_cnt >= 8) {
        const double c = 1.0 / (18.0 * ad_cnt);
        imu_rg = std::sqrt(ad_sg * c) / 0.07; imu_ra = std::sqrt(ad_sa * c) / 0.28;
        imu_rg = imu_rg < 1.0 ? 1.0 : (imu_rg > AD_MAX ? AD_MAX : imu_rg);
        imu_ra = imu_ra < 1.0 ? 1.0 : (imu_ra > AD_MAX ? AD_MAX : imu_ra);
    }
    // once the level is known (AD_BG_T s of data): open the gyro-bias prior in proportion (bias and noise scale together
    // in an IMU of lower grade); never tightens what the filter already has
    if (ADAPT_IMU == 1 && initialized && !ad_bg_done && ad_cnt >= (int)(AD_BG_T * 200)) {
        ad_bg_done = true;
        const double sb = SIG_BG0_MOVING * (AD_BG_K * (imu_rg - 1.0) + 1.0);
        for (int i = 0; i < 3; i++) {
            const int d = (IBG + i) * NS + IBG + i;
            if (P[d] < sb * sb) {   // inflate the variance only; keep correlations consistent by scaling the row/column
                const double f = std::sqrt(sb * sb / (P[d] > 1e-30 ? P[d] : 1e-30));
                for (int j = 0; j < NS; j++) { P[(IBG + i) * NS + j] *= f; P[j * NS + IBG + i] *= f; }
            }
        }
    }
#endif
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
#ifdef DIAG
    vals[7] = last_tracks; vals[8] = last_used; vals[9] = last_parallax; vals[10] = diag_reason + 10 * diag_start; vals[11] = norm(v_w); vals[12] = phase;
#endif
#if defined(DIAG3) && FLOOR
    vals[7] = z_floor; vals[8] = std::sqrt((double)P[IZF * NS + IZF]); vals[9] = p_w.z; vals[10] = last_used; vals[11] = norm(v_w); vals[12] = phase;
#endif
#if defined(DIAG4) && LROT
    vals[7] = lrot_n; vals[8] = lrot_eps; vals[9] = lrot_age; vals[10] = lrot_code; vals[11] = norm(v_w); vals[12] = phase;
#endif
#ifdef DIAG5
    { Vec3 uc = mulT(mul(R_wb, R_ic), last_u_short); vals[7] = uc.x; vals[8] = uc.y; vals[9] = uc.z; vals[10] = diag_reason + 10 * last_tracks; vals[11] = last_parallax; vals[12] = phase + 100 * diag_winlen; }
#endif
#ifdef DIAG6
    vals[7] = diag_sp[0]; vals[8] = diag_sp[1]; vals[9] = diag_sp[2]; vals[10] = diag_reason; vals[11] = diag_sp[3]; vals[12] = phase;
#endif
#if defined(DIAG7) && MANH
    vals[7] = diag_manh[0]; vals[8] = diag_manh[1]; vals[9] = diag_manh[2]; vals[10] = manh_psi; vals[11] = manh_ready; vals[12] = phase + 100 * diag_winlen;
#endif
#if defined(DIAG8) && STRUCT
    vals[7] = struct_known; vals[8] = struct_n; vals[9] = last_tracks; vals[10] = diag_reason;
#if SCALE_STATE
    vals[9] = s_log; vals[10] = diag_reason + 10 * (int)(100 * std::sqrt((double)P[IS * NS + IS]));
#endif
    vals[11] = norm(v_w); vals[12] = phase + 100 * diag_winlen;
#endif
#ifdef DIAG11
    vals[7] = std::sqrt((double)P[(IBG + 0) * NS + IBG + 0]); vals[8] = std::sqrt((double)P[(IBG + 1) * NS + IBG + 1]); vals[9] = std::sqrt((double)P[(IBG + 2) * NS + IBG + 2]);
    vals[10] = d11_rn[0]; vals[11] = d11_rn[1]; vals[12] = d11_rn[2];
#endif
#ifdef DIAG10
    vals[7] = bg.x; vals[8] = bg.y; vals[9] = bg.z; vals[10] = d10_rows + 1e4 * d10_win; vals[11] = d10_acc; vals[12] = d10_kept;
#endif
#ifdef DIAG9
    {   // 9 free slots (the quaternion must stay valid): alternate two halves of the record
        static int par9 = 0; par9 ^= 1;
        const int sl[9] = {0, 1, 2, 7, 8, 9, 10, 11, 12};
        vals[3] = vals[4] = vals[5] = 0; vals[6] = 1;
        vals[0] = par9;
        if (par9 == 0) for (int i = 1; i < 9; i++) vals[sl[i]] = diag9[i - 1];          // dmeas, feat0..6
        else { for (int i = 1; i < 9; i++) vals[sl[i]] = 0; for (int i = 8; i < 13; i++) vals[sl[i - 7]] = diag9[i]; }   // feat7, feat8, w1, n, valid
    }
#endif
#ifdef DIAG2
    vals[7] = diag_eta[0]; vals[8] = diag_eta[1]; vals[9] = diag_eta[2]; vals[10] = diag_eta[3]; vals[11] = diag_reason; vals[12] = phase;
#endif
    for (int i = 0; i < 13; i++) {
        if (!std::isfinite(vals[i])) { write_identity(out); return; }
        out[i] = vals[i];
    }
}

void finish_rank();
void select_top();
// first frame of a window: rank candidate cells and clone the position (templates follow on the next frames)
void make_keyframe() {
    if (CLONE_PHASE == 0) kf_propagate();
    if (PROF_REP == 4) { acc_T = 0.05; acc_n = 10; acc_R = eye(); kf_propagate(); }
    if (PROF_REP == 5) { nk = 0; rank_slice(0, NCAND); }
    if (PROF_REP == 7) { select_top(); }
    if (PROF_REP == 8) { finish_rank(); }
#if defined(__wasm_simd128__)
    if (PROF_REP == 6) { float sc[4]; int xs[4] = {10, 20, 30, 40}; volatile float sink = 0; for (int k = 0; k < NCAND / 4; k++) { score_cells4(xs, 10 + (k % 8), sc); sink = sink + sc[0]; } }
#endif
#if LONG_N > 0
    { Patch* t = patches; patches = prevp; prevp = t; n_prev = n_patches;
      for (int k = 0; k < SPLIT; k++) { R_wb_kf_prev[k] = R_wb_kf[k]; R_wb_trk_prev[k] = R_wb_trk[k]; } }
#endif
    nk = 0;
    samp_s = samp_ss = 0; samp_n = 0;
    rank_slice(0, NCAND / RANK_SPLIT);
    if (RANK_SPLIT == 1) finish_rank();
    n_patches = 0;
    next_cand = 0;
    if (CLONE_PHASE == 0) { kf_clone(); R_wb_clone = R_wb; }
    n_tracks = 0;
    par_sum = 0;
    n_long = 0;
    lpar_sum = 0;
    have_kf = true;
}

void select_top();
void finish_rank() {
    select_top();
    {   // texture threshold scaled with the frame's contrast (gain-invariant patch selection)
        double energy = samp_n ? samp_s / samp_n : CONTRAST_REF;  // mean half-trace of the cell tensors ~ contrast^2
        double f = CONTRAST_POW ? energy / CONTRAST_REF : 1.0;
#ifdef VIO_DEBUG
        std::fprintf(stderr, "ENERGY %.1f\n", energy);
#endif
        if (f > 1.0) f = 1.0;
        if (f < CONTRAST_MIN) f = CONTRAST_MIN;
        eig_thresh = MIN_EIG * f;
    }
}

// Motion prior of the episode generator: its autopilot flies nose-first (FPV camera), so the camera's sideways
// (camera x) and, more weakly, vertical (camera y) velocity are small. Soft pseudo-measurements c . v = 0 with
// c the camera axis in the world frame; c = exp(dtheta) c_est, so dh/dtheta = c x v. (Lever arm ignored.)
void apply_motion_prior() {
    if (!(NHC_SIG > 0) && !(NHC_SIG_Y > 0)) return;
    const Mat3 R_wc = mul(R_wb, R_ic);
    for (int k = 0; k < 2; k++) {
        const double sig = k == 0 ? NHC_SIG : NHC_SIG_Y;
        if (!(sig > 0)) continue;
        Vec3 c = mul(R_wc, k == 0 ? Vec3{1, 0, 0} : Vec3{0, 1, 0});
        Vec3 cv = cross(c, v_w);
        double Hrow[NS] = {0};
        Hrow[IV] = c.x; Hrow[IV + 1] = c.y; Hrow[IV + 2] = c.z;
        if (!NHC_NO_TH) { Hrow[ITH] = cv.x; Hrow[ITH + 1] = cv.y; Hrow[ITH + 2] = cv.z; }
        kf_update_row(Hrow, -dot(c, v_w), sig * sig);
    }
}

void apply_velocity_prior() {
#if FIXED_GAIN
    v_w = scale(v_w, 1.0 - FG_KVP);
    return;
#endif
    if (!(VPRIOR_SIG > 0)) return;
    double vv[3] = {v_w.x, v_w.y, v_w.z};
    const double vsig = VPRIOR_SIG + VPRIOR_REL * norm(v_w);   // weaker pull toward zero the faster the estimate already is
    for (int k = 0; k < 3; k++) {
        double Hrow[NS] = {0};
        Hrow[IV + k] = 1.0;
        kf_update_row(Hrow, -vv[k], vsig * vsig);
    }
    apply_motion_prior();
    kf_inject();
}

#if TRK
// ================= persistent tracker (stage 2) =================
// Features live across windows. Every TRK_HOP frames each feature is tracked from its last position (start = gyro
// rotation of the last bearing + the last translational flow), against the template cut at the last keyframe.
// Every TRK_WIN frames: vision update from (keyframe bearing -> current position), then templates are re-cut at the
// tracked positions (same features, new keyframe) and free slots are filled from the ranked cells.
struct Feat { int alive, keyed; double px, py, lfx, lfy; double rho; int hasL; Vec3 bL; double rvar; int miss; };
Feat feats[N_TRACK];
int trk_frame = 0, trk_wins = 0;

#if LROT
// ---- long-baseline rotation: attitude clone held while enough features from its frame survive ----
Vec3 u_long_dir{0.6, 0.6, 0.5};
void lrot_refresh() {
    if (SYM_P && p_lower_stale) { mirror_P(); p_lower_stale = false; }
    for (int i = 0; i < NS; i++)
        for (int j = 0; j < 3; j++) P[i * NS + ITC + j] = P[i * NS + ITH + j];
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < NS; i++) P[(ITC + j) * NS + i] = P[(ITH + j) * NS + i];
    R_wb_long = R_wb;
    for (int i = 0; i < N_TRACK; i++) {
        Feat& f = feats[i];
        if (!f.alive) { f.hasL = 0; continue; }
        Vec3 b{(f.px + PC - cx_s) / fx_s, (f.py + PC - cy_s) / fy_s, 1.0};
        f.bL = scale(b, 1.0 / norm(b)); f.hasL = 1;
    }
    u_long_dir = Vec3{0.6, 0.6, 0.5};
    lrot_age = 0;
}
int lrot_count() { int n = 0; for (int i = 0; i < N_TRACK; i++) n += feats[i].alive && feats[i].hasL; return n; }
void lrot_update() {
    Vec3 wr[N_TRACK], w1[N_TRACK];
    int n = 0;
    const Mat3 Rc = mul(R_wb, R_ic), RL = mul(R_wb_long, R_ic);
    for (int i = 0; i < N_TRACK; i++) {
        const Feat& f = feats[i];
        if (!f.alive || !f.hasL) continue;
        Vec3 b{(f.px + PC - cx_s) / fx_s, (f.py + PC - cy_s) / fy_s, 1.0};
        wr[n] = mul(RL, f.bL); w1[n] = mul(Rc, scale(b, 1.0 / norm(b))); n++;
    }
    lrot_n = n; lrot_age++; lrot_code = 1;
    if (n < LROT_MIN) return;
    Vec3 eps_tot{0, 0, 0};
    double Arot[9] = {0}, sb = 0;
    bool in[N_TRACK];
    for (int i = 0; i < n; i++) in[i] = true;
    // 1) pure rotation: wr = w1 + eps x w1 (two passes, the second without the worst residuals)
    double rms = 0;
    for (int pass = 0; pass < 2; pass++) {
        double A[9] = {0};
        Vec3 b{0, 0, 0};
        int k = 0;
        for (int i = 0; i < n; i++) {
            if (!in[i]) continue;
            const Vec3 w = w1[i];
            A[0] += 1 - w.x * w.x; A[1] -= w.x * w.y; A[2] -= w.x * w.z;
            A[4] += 1 - w.y * w.y; A[5] -= w.y * w.z; A[8] += 1 - w.z * w.z;
            b = add(b, cross(w, wr[i])); k++;
        }
        A[3] = A[1]; A[6] = A[2]; A[7] = A[5];
        Vec3 e;
        if (k < LROT_MIN || !solve3(A, ROT_DAMP, b, e)) return;
        double rr[N_TRACK], rs[N_TRACK], ss = 0;
        for (int i = 0; i < n; i++) {
            w1[i] = add(w1[i], cross(e, w1[i]));
            w1[i] = scale(w1[i], 1.0 / norm(w1[i]));
            Vec3 d = sub(wr[i], w1[i]);
            rr[i] = norm(d); rs[i] = rr[i];
            if (in[i]) ss += rr[i] * rr[i];
        }
        eps_tot = add(eps_tot, e);
        for (int q = 0; q < 9; q++) Arot[q] = A[q];
        rms = std::sqrt(ss / (2.0 * k));
        if (pass == 0) {
            for (int i = 1; i < n; i++) { double v = rs[i]; int j = i - 1; while (j >= 0 && rs[j] > v) { rs[j + 1] = rs[j]; j--; } rs[j + 1] = v; }
            const double med = rs[n / 2];
            for (int i = 0; i < n; i++) in[i] = rr[i] <= 2.5 * med + 1e-9;
        }
    }
    sb = rms; lrot_code = 2;
    if (rms * fx_s > LROT_PURE_PX) {
        lrot_code = 3;
        if (n < LROT_EPI_MIN) return;
        // 2) the camera also moved: epipolar residuals u . (wr x w1), alternate direction u and rotation eps
        for (int i = 0; i < n; i++) in[i] = true;
        Vec3 u = u_long_dir;
        double q = 1, ssr = 0;
        int kept = n;
        for (int it = 0; it < LROT_ITERS; it++) {
            Vec3 ns[N_TRACK];
            double M[9] = {0};
            for (int i = 0; i < n; i++) { ns[i] = cross(wr[i], w1[i]); if (in[i]) accum(M, ns[i]); }
            double l1 = smallest_eig(M, u);
            double lmid = 0.5 * (M[0] + M[4] + M[8] - l1);
            q = lmid > 0 ? l1 / lmid : 1;
            double r[N_TRACK], rs[N_TRACK];
            for (int i = 0; i < n; i++) { r[i] = std::fabs(dot(ns[i], u)); rs[i] = r[i]; }
            for (int i = 1; i < n; i++) { double v = rs[i]; int j = i - 1; while (j >= 0 && rs[j] > v) { rs[j + 1] = rs[j]; j--; } rs[j + 1] = v; }
            const double med = rs[n / 2];
            kept = 0; ssr = 0;
            double A[9] = {0};
            Vec3 b{0, 0, 0};
            for (int i = 0; i < n; i++) {
                in[i] = r[i] <= 3.0 * med + 1e-12;
                if (!in[i]) continue;
                kept++; ssr += r[i] * r[i];
                Vec3 J = sub(scale(u, dot(wr[i], w1[i])), scale(wr[i], dot(u, w1[i])));
                accum(A, J);
                b = add(b, scale(J, -dot(u, ns[i])));
            }
            if (kept < LROT_EPI_MIN - 1) return;
            for (int k = 0; k < 9; k++) Arot[k] = A[k];
            if (it == LROT_ITERS - 1) break;
            Vec3 e;
            if (!solve3(A, ROT_DAMP, b, e)) return;
            eps_tot = add(eps_tot, e);
            for (int i = 0; i < n; i++) { w1[i] = add(w1[i], cross(e, w1[i])); w1[i] = scale(w1[i], 1.0 / norm(w1[i])); }
        }
        lrot_code = 4;
        if (q > LROT_MAX_Q) return;
        lrot_code = 5;
        u_long_dir = u;
        sb = std::sqrt(ssr / kept);
    }
    lrot_eps = norm(eps_tot);
    {   // plausibility: the gyro cannot have drifted further than this since the long keyframe
        const double T = lrot_age * TRK_WIN * 0.05;
        const double lim = LROT_GATE * std::sqrt(LROT_BG * LROT_BG * T * T + SIG_GYR * SIG_GYR * T + 1e-6);
        if (lrot_eps > lim || lrot_eps > LROT_MAX) { lrot_code = 6; return; }
    }
    rot_rows(Arot, eps_tot, sb + SIG_ROTC_FLOOR);
    kf_inject();
}
#endif

#if MANH
// ================= Manhattan heading (stage 2) =================
// Edge cells (one strong gradient direction) are images of straight 3D edges. For a horizontal edge the direction is
// up x m (m = normal of the plane through the camera and the edge). Indoors those directions cluster on a 90-degree
// grid (walls, floor tiles, door frames): its azimuth is learned in the first seconds and then every keyframe gives
// an absolute heading measurement (no drift), fused as one filter row on the yaw error.
void edge_tensor(int ox, int oy, double& a, double& b, double& d, double& ux, double& uy) {
    constexpr int BW = PS + 2;
    alignas(16) float buf[BW * BW + 4];
    for (int v = 0; v < BW; v++) fetch_row(ox - 1, oy - 1 + v, BW, buf + v * BW);
    float fa = 0, fb = 0, fd = 0, su = 0, sv = 0;
    for (int v = 0; v < PS; v++) {
        const float* c = buf + (v + 1) * BW + 1;
        for (int u = 0; u < PS; u++) {
            const float gx = 0.5f * (c[u + 1] - c[u - 1]), gy = 0.5f * (c[u + BW] - c[u - BW]);
            const float g2 = gx * gx + gy * gy;
            fa += gx * gx; fb += gx * gy; fd += gy * gy; su += g2 * u; sv += g2 * v;
        }
    }
    a = fa; b = fb; d = fd;
    const float sg = fa + fd;
    ux = sg > 0 ? su / sg : PC; uy = sg > 0 ? sv / sg : PC;   // gradient-weighted centroid: a point on the edge
}
void manh_update() {
    const Mat3 Rwc = mul(R_wb, R_ic);
    double az[MANH_MAX], wt[MANH_MAX], kx[MANH_MAX], ky[MANH_MAX];
    int n = 0;
    static int k0 = 0;
    k0 = (k0 + 7) % NCAND;
    int tried = 0;
    for (int ii = 0; ii < NCAND && n < MANH_MAX && tried < MANH_TRY; ii++) {
        const int k = (k0 + ii * MANH_STRIDE) % NCAND;     // spread over the image, different cells every keyframe
        const float lmin = cscore[k], lmax = 2 * ctr[k] - lmin;
        if (lmax < MANH_PRE_MIN || lmin > MANH_PRE_ANISO * lmax) continue;
        double a, b, d, ux, uy;
        tried++;
        edge_tensor(cx_[k], cy_[k], a, b, d, ux, uy);
        const double tr = 0.5 * (a + d), disc = std::sqrt(std::fmax(tr * tr - (a * d - b * b), 0.0));
        const double l1 = tr + disc, l2 = tr - disc;
        if (l1 < MANH_L1_MIN * PN || l1 < MANH_ANISO * l2) continue;
        double vx, vy;
        if (std::fabs(b) > 1e-9 * l1) { vx = b; vy = l1 - a; } else if (a > d) { vx = 1; vy = 0; } else { vx = 0; vy = 1; }
        const double xn = (cx_[k] + ux - cx_s) / fx_s, yn = (cy_[k] + uy - cy_s) / fy_s;
        vx *= fx_s; vy *= fy_s;                             // image-line normal in normalised coordinates
        Vec3 mc{vx, vy, -(vx * xn + vy * yn)};
        mc = scale(mc, 1.0 / norm(mc));
        const Vec3 mw = mul(Rwc, mc);
        const double w2 = mw.x * mw.x + mw.y * mw.y;        // |up x m|^2: small when the edge is near camera height
        if (w2 < MANH_W_MIN * MANH_W_MIN || std::fabs(mw.z) < MANH_VERT) continue;  // also skips vertical edges (m horizontal)
        az[n] = std::atan2(mw.x, -mw.y); wt[n] = w2;        // azimuth of up x m = (-m.y, m.x, 0); noise ~ 1/|up x m|
        kx[n] = mw.z * mw.x / w2; ky[n] = mw.z * mw.y / w2; // tilt-error coupling of this azimuth
        n++;
    }
    diag_manh[0] = n; diag_manh[1] = 0; diag_manh[2] = 0;
    if (!manh_ready) {
        for (int i = 0; i < n; i++) { manh_c4 += wt[i] * std::cos(4 * az[i]); manh_s4 += wt[i] * std::sin(4 * az[i]); manh_sw += wt[i]; }
        if (++manh_init_n >= MANH_INIT) {
            manh_psi = 0.25 * std::atan2(manh_s4, manh_c4);
            manh_ready = manh_sw > MANH_INIT_MIN && manh_c4 * manh_c4 + manh_s4 * manh_s4 > MANH_INIT_R * MANH_INIT_R * manh_sw * manh_sw;
            if (!manh_ready) { manh_init_n = 0; manh_c4 = manh_s4 = manh_sw = 0; }
        }
        return;
    }
    const double Q = 1.5707963267948966;
    double r[MANH_MAX], c = 0, bx = 0, by = 0;
    for (int i = 0; i < n; i++) { double v = az[i] - manh_psi; r[i] = v - Q * std::floor(v / Q + 0.5); }
    double win = MANH_WIN;
    int m = 0;
    for (int pass = 0; pass < 2; pass++) {
        double sw = 0, sr = 0, sx = 0, sy = 0;
        m = 0;
        for (int i = 0; i < n; i++)
            if (std::fabs(r[i] - c) < win) { sw += wt[i]; sr += wt[i] * r[i]; sx += wt[i] * kx[i]; sy += wt[i] * ky[i]; m++; }
        if (m < MANH_MIN || !(sw > 0)) return;
        c = sr / sw; bx = sx / sw; by = sy / sw; win *= 0.5;
    }
    diag_manh[1] = m; diag_manh[2] = c;
    // measured azimuth = true + (estimate rotated by -dtheta_z) ...: c = -dtheta_z + kx dtheta_x + ky dtheta_y
    double Hrow[NS] = {0};
    Hrow[ITH + 2] = -1.0;
    if (MANH_TILT) { Hrow[ITH] = bx; Hrow[ITH + 1] = by; }
    const double sg = MANH_SIG / std::sqrt(m / 6.0) + MANH_SIG_MIN;
    kf_update_row(Hrow, c, sg * sg);
    kf_inject();
}
#endif

void trk_hop() {
    const Mat3 Rh = mul(transpose(mul(R_wb, R_ic)), mul(R_wb_hop, R_ic));   // camera at the last hop -> now (gyro)
    for (int i = 0; i < N_TRACK; i++) {
        Feat& f = feats[i];
        if (!f.alive) continue;
        Vec3 b{(f.px + PC - cx_s) / fx_s, (f.py + PC - cy_s) / fy_s, 1.0};
        Vec3 br = mul(Rh, b);
        if (br.z < 0.2) { f.alive = 0; continue; }
        const double qx = cx_s + fx_s * br.x / br.z - PC, qy = cy_s + fy_s * br.y / br.z - PC;  // rotation-predicted top-left
        double sx = qx + TRK_FLOW * f.lfx, sy = qy + TRK_FLOW * f.lfy;
        if (!track(patches[i], sx, sy)) { f.alive = 0; continue; }
        f.lfx = sx - qx; f.lfy = sy - qy; f.px = sx; f.py = sy;
    }
    R_wb_hop = R_wb;
}

void trk_collect() {
    n_tracks = 0; par_sum = 0;
    const Mat3 Rc = mul(R_wb, R_ic);
    const Mat3 R10 = mul(transpose(Rc), mul(R_wb_key, R_ic));
    for (int i = 0; i < N_TRACK; i++) {
        const Feat& f = feats[i];
        if (!f.alive || !f.keyed) continue;
        const Patch& p = patches[i];
        Vec3 f0{(p.ox + PC - cx_s) / fx_s, (p.oy + PC - cy_s) / fy_s, 1.0};
        Vec3 fr = mul(R10, f0);
        if (fr.z < 0.2) continue;
        const double px0 = cx_s + fx_s * fr.x / fr.z, py0 = cy_s + fy_s * fr.y / fr.z;
        const double px = f.px + PC, py = f.py + PC;
        Vec3 f1{(px - cx_s) / fx_s, (py - cy_s) / fy_s, 1.0};
        trk_idx[n_tracks] = i;
        Track& t = tracks[n_tracks++];
        t.fr = scale(fr, 1.0 / norm(fr));
        t.f1 = scale(f1, 1.0 / norm(f1));
        t.R_wc = Rc;
        t.par = std::sqrt((px - px0) * (px - px0) + (py - py0) * (py - py0));
        par_sum += t.par;
    }
}

#if STRUCT
// ================= structure (stage 2) =================
// Every persistent feature carries an inverse depth rho along its keyframe ray. A feature with a depth is a 3D point:
// its new image position constrains the whole displacement (direction AND length), two rows per feature. Depths are
// triangulated after each update from the filter's displacement and moved to the new keyframe, so the metric scale
// is carried by the scene from window to window instead of by an assumed room depth.
void struct_rows() {
    struct_n = 0; struct_known = 0; struct_skip = false;
    const Vec3 d = add(sub(p_w, p_kf), sub_lever());
    const double sb = STRUCT_SIG_PX / fx_s;
#if SCALE_STATE
    const double MS = std::exp(s_log);
#else
    const double MS = 1.0;
#endif
    for (int i = 0; i < n_tracks; i++) {
        Feat& f = feats[trk_idx[i]];
        if (!(f.rho > 0)) continue;
        struct_known++;
        const Vec3 wr = mul(tracks[i].R_wc, tracks[i].fr), w1 = mul(tracks[i].R_wc, tracks[i].f1);
        const Vec3 X = scale(wr, MS / f.rho), Y = sub(X, d);          // metric point = map point x map scale
        const double dist = norm(Y);
        Vec3 a = std::fabs(w1.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
        Vec3 t1 = cross(w1, a); t1 = scale(t1, 1.0 / norm(t1));
        const Vec3 t2 = cross(w1, t1);
        const Vec3 ts[2] = {t1, t2};
        int ok = 0;
        for (int k = 0; k < 2; k++) {
            const Vec3 t = ts[k];
            double Hrow[NS];
            row_d(Hrow, scale(t, -1.0));                       // h = t . (X - d), measured 0
            const Vec3 dt = cross(d, t);
            Hrow[ITH] = dt.x; Hrow[ITH + 1] = dt.y; Hrow[ITH + 2] = dt.z;
            const double tw = MS * dot(t, wr) / (f.rho * f.rho);
            const double Rn = sb * sb * dist * dist + f.rvar * tw * tw + STRUCT_SIG_M * STRUCT_SIG_M;
#if SCALE_STATE
            Hrow[IS] = dot(t, X);                                // d/ds of t . (e^s X_map - d)
#endif
            if (kf_update_row(Hrow, -dot(t, Y), Rn)) ok++;
        }
        if (ok == 2) { struct_n++; f.miss = 0; }
        else if (++f.miss >= STRUCT_MISS) { f.rho = 0; f.miss = 0; }   // repeatedly inconsistent: forget the depth
    }
    struct_skip = struct_n >= STRUCT_MIN;
    last_used += 2 * struct_n;
    kf_inject();
}
// after the update: triangulate with the corrected displacement, fuse, and move every depth to the new keyframe (now)
void struct_depths() {
    Vec3 d = add(sub(p_w, p_kf), sub_lever());
    const double sb = STRUCT_SIG_PX / fx_s;
#if SCALE_STATE
    if (struct_known == 0) {   // no depth survived: the map scale is defined anew by this window's displacement
        const double dm = norm(d);
        double vd = 0;         // variance of the displacement along itself
        if (dm > 1e-6)
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) {
                    const double ua = (a == 0 ? d.x : a == 1 ? d.y : d.z) / dm, ub = (b == 0 ? d.x : b == 1 ? d.y : d.z) / dm;
                    vd += ua * ub * (P[(IP + a) * NS + IP + b] + P[(IPC + a) * NS + IPC + b] - P[(IP + a) * NS + IPC + b] - P[(IPC + a) * NS + IP + b]);
                }
        double rel = dm > 1e-6 ? vd / (dm * dm) : 1.0;
        if (!(rel > STRUCT_S_MIN * STRUCT_S_MIN)) rel = STRUCT_S_MIN * STRUCT_S_MIN;
        if (rel > 1.0) rel = 1.0;
        s_log = 0;
        for (int i = 0; i < NS; i++) { P[IS * NS + i] = 0; P[i * NS + IS] = 0; }
        P[IS * NS + IS] = rel;
    }
    d = scale(d, std::exp(-s_log));   // displacement in map units
#endif
    const double dn = norm(d);
    // new depths only from a window whose displacement is trustworthy: a good direction fit or a consistent map
    if (struct_wipe) {
        for (int i = 0; i < N_TRACK; i++) feats[i].rho = 0;
        struct_n = 0; struct_wipe = false; struct_resets++;
    }
    const bool tri_ok = struct_n >= STRUCT_MIN || (diag_reason == 6 && last_parallax >= STRUCT_BOOT_PX);
    for (int i = 0; i < n_tracks; i++) {
        Feat& f = feats[trk_idx[i]];
        const Vec3 wr = mul(tracks[i].R_wc, tracks[i].fr), w1 = mul(tracks[i].R_wc, tracks[i].f1);
        const Vec3 c = cross(wr, w1);
        const double c2 = dot(c, c);
        if (tri_ok && dn > STRUCT_MIN_D && tracks[i].par >= STRUCT_PAR_PX && c2 > STRUCT_MIN_PAR * STRUCT_MIN_PAR * sb * sb) {
            const double lam = dot(cross(d, w1), c) / c2;
            if (lam > STRUCT_ZMIN && lam < STRUCT_ZMAX) {
                const double rm = 1.0 / lam, sr = rm * (sb / std::sqrt(c2) * STRUCT_SIG_K + STRUCT_REL);
                if (f.rho > 0) { const double w = f.rvar / (f.rvar + sr * sr); f.rho += w * (rm - f.rho); f.rvar *= 1.0 - w; }
                else { f.rho = rm; f.rvar = sr * sr; }
            }
        }
        if (f.rho > 0) {
            const Vec3 Y = sub(scale(wr, 1.0 / f.rho), d);
            const double l2 = norm(Y);
            if (!(l2 > STRUCT_ZMIN) || dot(Y, w1) <= 0) { f.rho = 0; continue; }
            const double r2 = 1.0 / l2, k = (r2 / f.rho) * (r2 / f.rho);
            f.rvar = f.rvar * k * k + STRUCT_GROW * STRUCT_GROW * r2 * r2;
            f.rho = r2;
        }
    }
}
#endif

void trk_rekey() {
    R_wb_key = R_wb;
    nk = 0; samp_s = samp_ss = 0; samp_n = 0;
    rank_slice(0, NCAND);
    finish_rank();
    const int m = MARGIN + 2;
    double mfx = 0, mfy = 0;
    int na = 0;
    for (int i = 0; i < N_TRACK; i++) {
        Feat& f = feats[i];
        if (!f.alive) continue;
        Patch& p = patches[i];
        const int ox = (int)std::floor(f.px + 0.5), oy = (int)std::floor(f.py + 0.5);
        bool ok = !(ox < m || oy < m || ox + PS + m > SW || oy + PS + m > SH);
        for (int j = 0; ok && j < i; j++)   // two features that converged on the same point: keep the first
            if (feats[j].alive && std::fabs(feats[j].px - ox) < TRK_MERGE && std::fabs(feats[j].py - oy) < TRK_MERGE) ok = false;
        if (ok) { p.ox = ox; p.oy = oy; ok = cut_at(p); }
        if (!ok) { f.alive = 0; continue; }
        f.px = ox; f.py = oy; f.keyed = 1;
        mfx += f.lfx; mfy += f.lfy; na++;
    }
    if (na) { mfx /= na; mfy /= na; }
    for (int c = 0; c < nk && na < N_TRACK; c++) {
        const int ox = cx_[top[c]], oy = cy_[top[c]];
        bool near = false;
        int slot = -1;
        for (int j = 0; j < N_TRACK; j++) {
            if (!feats[j].alive) { if (slot < 0) slot = j; continue; }
            if (std::fabs(feats[j].px - ox) < TRK_MIN_DIST && std::fabs(feats[j].py - oy) < TRK_MIN_DIST) { near = true; break; }
        }
        if (near || slot < 0) continue;
        Patch& p = patches[slot];
        p.ox = ox; p.oy = oy;
        if (!cut_at(p)) continue;
        feats[slot] = Feat{1, 1, (double)ox, (double)oy, mfx, mfy, 0.0, 0, Vec3{0, 0, 1}, 0.0, 0};
        na++;
    }
    n_patches = na;
    have_kf = true;
}
#endif

#if TRK == 2
// ---- two feature groups, one tracked per frame (flat fuel) ----
// frame ph (0..5):  ph even -> group 0 hops, ph odd -> group 1 hops.
//   ph0: G0 measurement (keyframe bearing -> now), re-key G0, covariance propagation (ahead by one frame)
//   ph1: G1 measurement, vision update from both groups, clone, re-key G1
//   ph2: rank + new features into free G0 slots      ph3: the same for G1 slots
void trkg_hop(int g) {
    const Mat3 Rh = mul(transpose(mul(R_wb, R_ic)), mul(R_hopg[g], R_ic));
    for (int i = g; i < N_TRACK; i += TG) {
        Feat& f = feats[i];
        if (!f.alive) continue;
        Vec3 b{(f.px + PC - cx_s) / fx_s, (f.py + PC - cy_s) / fy_s, 1.0};
        Vec3 br = mul(Rh, b);
        if (!(br.z >= 0.2)) { f.alive = 0; continue; }
        const double qx = cx_s + fx_s * br.x / br.z - PC, qy = cy_s + fy_s * br.y / br.z - PC;
        double sx = qx + TRK_FLOW * f.lfx, sy = qy + TRK_FLOW * f.lfy;
        if (!track(patches[i], sx, sy)) { f.alive = 0; continue; }
        f.lfx = sx - qx; f.lfy = sy - qy; f.px = sx; f.py = sy;
    }
    R_hopg[g] = R_wb;
}
void trkg_collect(int g) {
    dispg[g] = sub(p_w, p_keyg[g]);
    const Mat3 Rc = mul(R_wb, R_ic);
    const Mat3 R10 = mul(transpose(Rc), mul(R_keyg[g], R_ic));
    for (int i = g; i < N_TRACK; i += TG) {
        const Feat& f = feats[i];
        if (!f.alive || !f.keyed) continue;
        const Patch& p = patches[i];
        Vec3 f0{(p.ox + PC - cx_s) / fx_s, (p.oy + PC - cy_s) / fy_s, 1.0};
        Vec3 fr = mul(R10, f0);
        if (!(fr.z >= 0.2)) continue;
        const double px0 = cx_s + fx_s * fr.x / fr.z, py0 = cy_s + fy_s * fr.y / fr.z;
        const double px = f.px + PC, py = f.py + PC;
        Vec3 f1{(px - cx_s) / fx_s, (py - cy_s) / fy_s, 1.0};
        trk_idx[n_tracks] = i;
        Track& t = tracks[n_tracks++];
        t.fr = scale(fr, 1.0 / norm(fr));
        t.f1 = scale(f1, 1.0 / norm(f1));
        t.R_wc = Rc;
        t.par = std::sqrt((px - px0) * (px - px0) + (py - py0) * (py - py0));
        par_sum += t.par;
    }
}
void trkg_rekey(int g) {
    R_keyg[g] = R_wb; p_keyg[g] = p_w;
    const int m = MARGIN + 2;
    for (int i = g; i < N_TRACK; i += TG) {
        Feat& f = feats[i];
        if (!f.alive) continue;
        Patch& p = patches[i];
        const int ox = (int)std::floor(f.px + 0.5), oy = (int)std::floor(f.py + 0.5);
        bool ok = !(ox < m || oy < m || ox + PS + m > SW || oy + PS + m > SH);
        for (int j = 0; ok && j < N_TRACK; j++)
            if (j != i && feats[j].alive && (j % TG != g || j < i) && std::fabs(feats[j].px - ox) < TRK_MERGE && std::fabs(feats[j].py - oy) < TRK_MERGE) ok = false;
        if (ok) { p.ox = ox; p.oy = oy; ok = cut_at(p); }
        if (!ok) { f.alive = 0; continue; }
        f.px = ox; f.py = oy; f.keyed = 1;
    }
}
// rank the grid on this frame and start new features in the free slots of group g (g < 0: all groups)
void trkg_topup(int g, int keyed) {
    nk = 0; samp_s = samp_ss = 0; samp_n = 0;
    if (g < 0 || !TRK_HALF_RANK) rank_slice(0, NCAND);
    else {   // rank half of the grid per top-up frame (upper / lower half, swapped every window); the other half is not used
        constexpr int HC = (GRID_Y / 2) * GRID_X;
        const int h = (g + trk_frame / 6) & 1;
        if (h == 0) { rank_slice(0, HC); for (int k = HC; k < NCAND; k++) cscore[k] = 0; }
        else { rank_slice(HC, NCAND); for (int k = 0; k < HC; k++) cscore[k] = 0; }
    }
    finish_rank();
    double mfx = 0, mfy = 0;
    int na = 0, free_g = 0;
    for (int i = 0; i < N_TRACK; i++) {
        if (feats[i].alive) { mfx += feats[i].lfx; mfy += feats[i].lfy; na++; }
        else if (g < 0 || i % TG == g) free_g++;
    }
    if (na) { mfx /= na; mfy /= na; }
    n_patches = na;
    int tries = 0;
    const int max_tries = g == -1 ? TRK_TRIES_INIT : TRK_TRIES;
    for (int c = 0; c < nk && free_g > 0 && tries < max_tries; c++) {
        const int ox = cx_[top[c]], oy = cy_[top[c]];
        bool near = false;
        int slot = -1;
        for (int j = 0; j < N_TRACK; j++) {
            if (!feats[j].alive) { if (slot < 0 && (g < 0 || j % TG == g)) slot = j; continue; }
            if (std::fabs(feats[j].px - ox) < TRK_MIN_DIST && std::fabs(feats[j].py - oy) < TRK_MIN_DIST) { near = true; break; }
        }
        if (near || slot < 0) continue;
        Patch& p = patches[slot];
        p.ox = ox; p.oy = oy;
        tries++;
        if (!cut_at(p)) continue;
        feats[slot] = Feat{1, keyed, (double)ox, (double)oy, mfx, mfy, 0.0, 0, Vec3{0, 0, 1}, 0.0, 0};
        if (g == -2) {   // the group's last hop was 1-2 frames ago: express the position in that hop's camera (rotation only)
            const Mat3 Rb = mul(transpose(mul(R_hopg[slot % TG], R_ic)), mul(R_wb, R_ic));
            Vec3 b{(ox + PC - cx_s) / fx_s, (oy + PC - cy_s) / fy_s, 1.0};
            Vec3 br = mul(Rb, b);
            if (br.z >= 0.2) { feats[slot].px = cx_s + fx_s * br.x / br.z - PC; feats[slot].py = cy_s + fy_s * br.y / br.z - PC; }
        }
        free_g--; n_patches++;
    }
    have_kf = true;
}
#endif

bool state_ok() {
    double s = p_w.x + p_w.y + p_w.z + v_w.x + v_w.y + v_w.z + ba.x + ba.y + ba.z + bg.x + bg.y + bg.z;
    for (int i = 0; i < NS; i++) s += P[i * NS + i];
    for (int i = 0; i < 9; i++) s += R_wb.m[i];
    return std::isfinite(s) && norm(v_w) < 50.0;
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
    R_wb = eye();
    bg = ba = omega_b = {0, 0, 0};
    std::memset(sum_a, 0, sizeof sum_a); std::memset(sum_a2, 0, sizeof sum_a2);
    std::memset(sum_g, 0, sizeof sum_g); std::memset(sum_g2, 0, sizeof sum_g2);
    calib_n = 0; calib_t0 = -1;
#if MOVE_START
    R_cal = eye(); t_first = -1; cal_prev_t = -1;
#endif
#if ADAPT_IMU
    ad_sg = ad_sa = 0; ad_n = ad_cnt = 0; imu_rg = imu_ra = 1.0; ad_bg_done = false;
#endif
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
#if MOVE_START
            if (cal_prev_t >= 0 && r[0] > t_first) {
                double dt = r[0] - cal_prev_t;
                rotate_small(R_cal, Vec3{r[4] * dt, r[5] * dt, r[6] * dt});
            }
            cal_prev_t = r[0];
#endif
        }
        double last = imu_count > 0 ? imu[7 * (imu_count - 1)] : t;
#if MOVE_START
        if (t_first < 0) t_first = t;
        bool moving_now = false;
        if (calib_n >= MS_MIN_SAMPLES) {   // clear motion: start the filter now instead of waiting for a hover
            double n = calib_n, gm = 0, am = 0;
            for (int i = 0; i < 3; i++) { double g = sum_g[i] / n, a = sum_a[i] / n; gm += g * g; am += a * a; }
            moving_now = std::sqrt(gm) > MS_GYRO_MEAN || std::fabs(std::sqrt(am) - GRAVITY) > MS_ACCEL_MEAN;
        }
        if (!moving_now && (calib_n < 2 || last - calib_t0 < CALIB_SECONDS)) {
#else
        if (calib_n < 2 || last - calib_t0 < CALIB_SECONDS) {
#endif
            write_identity(out);
            return VIOBENCH_STATUS_OK;
        }
        finish_calibration(last);
        phase = 0;
#if TRK
        for (int i = 0; i < N_TRACK; i++) feats[i].alive = 0;
        trk_frame = 0;
#if MANH
        manh_c4 = manh_s4 = manh_sw = 0; manh_init_n = 0; manh_ready = false;
#endif
        kf_clone(); R_wb_clone = R_wb; R_wb_hop = R_wb;
#if TRK == 2
        for (int g = 0; g < TG; g++) { R_hopg[g] = R_keyg[g] = R_wb; p_keyg[g] = p_w; dispg[g] = Vec3{0, 0, 0}; }
        trkg_topup(-1, 1);
        n_tracks = 0; par_sum = 0;
#else
        trk_rekey();
#endif
#if LROT
        lrot_refresh();
#endif
#else
        make_keyframe();
#endif
        write_state(out);
        return VIOBENCH_STATUS_OK;
    }

    integrate(imu, imu_count);

    // window cycle of KF_LEN + SPLIT frames:
    //   phase 0                        rank candidate cells, clone the position
    //   phase 1 .. SPLIT               cut one template subset per frame
    //   phase KF_LEN .. KF_LEN+SPLIT-1 track one subset per frame; the last one also runs the direction update
#if TRK == 2
    {
        trk_frame++;
        const int ph = trk_frame % 6;
        phase = ph;
#if TRK_SCHED == 3
        // flat schedule: hops every 3 frames, update / velocity prior / propagation / ranking each on their own frame
        if (ph == 0) { trkg_hop(0); n_tracks = 0; par_sum = 0; trkg_collect(0); trkg_rekey(0); }
        else if (ph == 1) { trkg_hop(1); trkg_collect(1); trkg_rekey(1); }
        else if (ph == 2) {
            diag_start = n_patches;
            direction_update();
            if (inject_pending) kf_inject_now();
            kf_clone(); R_wb_clone = R_wb;
        }
        else if (ph == 3) { trkg_hop(0); apply_velocity_prior(); }
        else if (ph == 4) { trkg_hop(1); if (inject_pending) kf_inject_now(); prop_ahead_next = true; kf_propagate(); }
        else trkg_topup(-2, 0);
#elif TRK_SCHED == 1
        // hops every 3 frames (G0 on ph 0/3, G1 on ph 1/4); the update has its own frame
        if (ph == 0) { trkg_hop(0); n_tracks = 0; par_sum = 0; trkg_collect(0); trkg_rekey(0); }
        else if (ph == 1) { trkg_hop(1); trkg_collect(1); trkg_rekey(1); }
        else if (ph == 2) {
            diag_start = n_patches;
            direction_update();
            if (inject_pending) kf_inject_now();
            kf_clone(); R_wb_clone = R_wb;
        }
        else if (ph == 3) { trkg_hop(0); trkg_topup(0, 0); }
        else if (ph == 4) { trkg_hop(1); trkg_topup(1, 0); }
        else { apply_velocity_prior(); if (inject_pending) kf_inject_now(); prop_ahead_next = true; kf_propagate(); }
#else
        const int g = ph & 1;
        trkg_hop(g);
        if (ph == 0) {
            n_tracks = 0; par_sum = 0;
            trkg_collect(0);
            trkg_rekey(0);
            if (TRK_SCHED == 2) { prop_ahead_next = true; kf_propagate(); }
        } else if (ph == 1) {
            trkg_collect(1);
            diag_start = n_patches;
            if (TRK_SCHED == 0) { defer_rows = true; nq = 0; }
            direction_update();
            defer_rows = false;
            if (TRK_SCHED == 2) { apply_velocity_prior(); if (inject_pending) kf_inject_now(); kf_clone(); R_wb_clone = R_wb; }
            trkg_rekey(1);
        } else if (ph == 2) {
            if (TRK_SCHED == 0) {
                for (int k = 0; k < nq; k++) { huber_on = qrows[k].hub; kf_update_row(qrows[k].H, qrows[k].y, qrows[k].R); }
                huber_on = false; nq = 0;
                kf_inject();
                if (inject_pending) kf_inject_now();
                kf_clone(); R_wb_clone = R_wb;
            } else trkg_topup(0, 0);
        } else if (ph == 3) trkg_topup(1, 0);
        else if (ph == 4) { if (TRK_SCHED == 0) trkg_topup(0, 0); }
        else if (TRK_SCHED == 0) { apply_velocity_prior(); if (inject_pending) kf_inject_now(); prop_ahead_next = true; kf_propagate(); }
#endif
    }
    if (false)
#elif TRK
    {
        trk_frame++;
        const int ph = trk_frame % TRK_WIN;
        phase = ph;
        if (ph % TRK_HOP == 0 && have_kf) trk_hop();
        bool do_key = ph == 0;
        diag_winlen = 0;
        if (ph == 0) { kf_propagate(); trk_wins++; }
        if (do_key && have_kf && TRK_PAR_KEY > 0) {   // adaptive baseline: keep the keyframe until the tracks have moved enough
            trk_collect();
            const double pm = n_tracks ? par_sum / n_tracks : 0;
            if (n_tracks >= MIN_TRACKS && pm < TRK_PAR_KEY && trk_wins < TRK_MAX_WINS) do_key = false;
        }
        if (do_key) {
            diag_winlen = trk_wins * TRK_WIN;
            trk_wins = 0;
            diag_start = n_patches;
#if LROT
            if (have_kf) { lrot_update(); if (inject_pending) kf_inject_now(); }
#endif
#if STRUCT
            if (have_kf) {
                trk_collect(); struct_rows();
                if (inject_pending) kf_inject_now();
                direction_update(); apply_velocity_prior();
                if (inject_pending) kf_inject_now();
                struct_depths();
            }
#else
            if (have_kf) { trk_collect(); direction_update(); apply_velocity_prior(); }
#endif
            if (inject_pending) kf_inject_now();
            kf_clone(); R_wb_clone = R_wb;
            trk_rekey();
#if MANH
            manh_update();
            if (inject_pending) kf_inject_now();
#endif
#if LROT
            if (lrot_count() < LROT_KEEP || lrot_age >= LROT_MAX_AGE) lrot_refresh();
#endif
        }
    }
    if (false)
#endif
#if ULTRA
    // ULTRA window: every frame does at most one heavy step (H = ULTRA_SPLIT: covariance propagation and the
    // direction rows each split over two frames).
    {
        constexpr int H = ULTRA_SPLIT ? 1 : 0;
        constexpr int R = RANK_SPLIT, S = SPLIT, C0 = R + 1 + H;          // first cut frame
        constexpr int T0 = C0 + KF_LEN, D = T0 + S, WIN = D + 3 + H;
        static_assert(KF_LEN > S + H, "tracking (and its propagation) must start after the last template subset");
        phase++;
        if (phase >= WIN) phase = 0;
        prop_split = H;
        if (phase == 0) make_keyframe();
        else if (phase < R) {
            rank_slice(phase * NCAND / R, (phase + 1) * NCAND / R);
            if (phase == R - 1) finish_rank();
        } else if (phase == R) { kf_propagate(); if (!H) { kf_clone(); R_wb_clone = R_wb; } }
        else if (H && phase == R + 1) { prop_finish(); kf_clone(); R_wb_clone = R_wb; }
        else if (phase >= C0 && phase < C0 + S) {
            const int si = phase - C0;
            R_wb_kf[si] = R_wb; win_T[si] = 0;
            int want = (N_TRACK - n_patches + (S - si) - 1) / (S - si);
            if (SUB_CAP > 0 && want > SUB_CAP) want = SUB_CAP;
            cut_templates(si, want);
        } else if (phase == T0 - 1 - H) { prop_ahead_next = true; kf_propagate(); }
        else if (H && phase == T0 - 1) prop_finish();
        else if (phase >= T0 && phase < D && have_kf) track_subset(phase - T0);
        else if (phase == D && have_kf) { defer_rows = true; nq = 0; direction_update(); defer_rows = false; }
        else if (phase == D + 1 || (H && phase == D + 2)) {
            int k0 = 0, k1 = nq;
            if (H) { k0 = phase == D + 1 ? 0 : (nq + 1) / 2; k1 = phase == D + 1 ? (nq + 1) / 2 : nq; }
            for (int k = k0; k < k1; k++) { huber_on = qrows[k].hub; kf_update_row(qrows[k].H, qrows[k].y, qrows[k].R); }
            huber_on = false;
            if (k1 == nq) { nq = 0; kf_inject(); }
        }
        else if (phase == D + 2 + H && have_kf) apply_velocity_prior();
        prop_split = false;
    }
    if (false)
#endif
    {
    phase++;
    if (phase >= KF_LEN + SPLIT + DIR_OWN) phase = 0;
    if (phase == 0) {
        if (DIR_NEXT && have_kf) direction_update();               // one frame late, off the busiest tracking frame
        if (VPRIOR_IDLE == 2 && have_kf && !DIR_OWN) apply_velocity_prior();  // right after the direction update, on a lighter frame
        make_keyframe();
    } else if (phase <= SPLIT) {
        const int si = phase - 1;
        if (VPRIOR_IDLE == 3 && phase == 1) apply_velocity_prior();
        if (VPRIOR_IDLE == 1 && phase == KF_LEN - 2 && have_kf) apply_velocity_prior();  // a cut frame when SPLIT >= KF_LEN - 2
        if (CLONE_PHASE == phase) { kf_propagate(); kf_clone(); R_wb_clone = R_wb; }  // off the ranking frame
        if (phase < RANK_SPLIT) {
            rank_slice(phase * NCAND / RANK_SPLIT, (phase + 1) * NCAND / RANK_SPLIT);
            if (phase == RANK_SPLIT - 1) finish_rank();
        }
        R_wb_kf[si] = R_wb;
        win_T[si] = 0;
        int want = (N_TRACK - n_patches + (SPLIT - si) - 1) / (SPLIT - si);
        if (LAST_SUB > 0 && si < SPLIT - 1)  // lighter last subset: its frame also runs the direction update
            want = (N_TRACK - LAST_SUB - n_patches + (SPLIT - 1 - si) - 1) / (SPLIT - 1 - si);
        if (SUB_CAP > 0 && want > SUB_CAP) want = SUB_CAP;
        cut_templates(si, want);
    } else if (VPRIOR_IDLE == 1 && phase == KF_LEN - 2) {
        apply_velocity_prior();              // weak zero-velocity prior on an otherwise idle frame
        if (PROP_MODE != 0 && KF_LEN - 2 == KF_LEN - 1) kf_propagate();
    } else if (PROP_MODE != 0 && phase == KF_LEN - 1) {
        if (PROP_MODE == 3) prop_ahead_next = true;
        kf_propagate();                      // idle frame: covariance up to just before tracking
    } else if (DIR_OWN && phase == KF_LEN + SPLIT) {
        if (have_kf) { direction_update(); if (VPRIOR_IDLE != 1) apply_velocity_prior(); }  // DIR_OWN: the window's last frame does only this
        if (PROF_REP == 11 && have_kf) direction_update();
        if (PROF_REP == 12 && have_kf) apply_velocity_prior();
        if (PROF_REP == 13 && have_kf) { inject_pending = false; kf_inject_now(); }
    } else if (phase >= KF_LEN && have_kf) {
        track_subset(phase - KF_LEN);
        if (phase == KF_LEN + SPLIT - 1 && !DIR_OWN) {
            if (!DIR_NEXT) direction_update();
            if (PROF_REP == 1) direction_update();
            if (!VPRIOR_IDLE) apply_velocity_prior();
            if (PROF_REP == 2) apply_velocity_prior();
            if (PROF_REP == 3) kf_inject_now();
        }
    }
    }
    if (inject_pending) kf_inject_now();
    if (!state_ok()) {  // never output garbage: restart the filter around the current pose
        v_w = {0, 0, 0};
        if (!THRUST_ONLY || !std::isfinite(ba.z)) ba = {0, 0, 0};   // THRUST_ONLY keeps the calibrated thrust-axis bias
        ba.x = ba.y = 0;
        if (!std::isfinite(bg.x + bg.y + bg.z)) bg = {0, 0, 0};
        { double sr = 0; for (int i = 0; i < 9; i++) sr += R_wb.m[i]; if (!std::isfinite(sr)) R_wb = eye(); }
        inject_pending = false; nq = 0;
        if (!std::isfinite(p_w.x + p_w.y + p_w.z)) p_w = {0, 0, 0};
        std::memset(P, 0, sizeof P);
        for (int i = 0; i < NS; i++) P[i * NS + i] = 1e-2;
        for (int i = 0; i < NS; i++) dx[i] = 0;
        acc_T = 0; acc_n = 0; acc_R = Mat3{{0}}; acc_f = {0, 0, 0};
        p_kf = p_w;
        p_kf2 = p_w;
        bw = {0, 0, 0};
        if (THRUST_ONLY) P[(IBA + 0) * NS + IBA + 0] = P[(IBA + 1) * NS + IBA + 1] = WIND_STATE ? SIG_WIND0 * SIG_WIND0 : 0;   // frozen states stay frozen
#if TRK
        R_wb_clone = R_wb_hop = R_wb_key = R_wb;
        for (int g = 0; g < TG; g++) { R_hopg[g] = R_keyg[g] = R_wb; p_keyg[g] = p_w; dispg[g] = Vec3{0, 0, 0}; }
        for (int i = 0; i < N_TRACK; i++) feats[i].alive = 0;   // start the tracker again from this frame's next top-up
        n_tracks = 0; par_sum = 0;
        for (int i = 0; i < NS; i++)
            for (int j = 0; j < 3; j++) { P[(IPC + j) * NS + i] = P[(IP + j) * NS + i]; }
        for (int i = 0; i < NS; i++)
            for (int j = 0; j < 3; j++) { P[i * NS + IPC + j] = P[i * NS + IP + j]; }
#endif
    }
#ifdef VIO_DEBUG
    if (phase == KF_LEN + SPLIT - 1)
        std::fprintf(stderr, "t %.2f tracks %d used %d q %.3f par %.2f v %.2f %.2f %.2f bg %.4f %.4f %.4f\n",
                     t, last_tracks, last_used, last_quality, last_parallax, v_w.x, v_w.y, v_w.z, bg.x, bg.y, bg.z);
#endif
    write_state(out);
    return VIOBENCH_STATUS_OK;
}
