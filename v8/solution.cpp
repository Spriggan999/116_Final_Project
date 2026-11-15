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
#if SCALE_STATE
constexpr int IP = 0, IV = 3, ITH = 6, IBA = 9, IBG = 12, IPC = 15, IS = 18, NS = 19;  // + log scale of the speed proxy
#else
constexpr int IP = 0, IV = 3, ITH = 6, IBA = 9, IBG = 12, IPC = 15, NS = 18;
#endif
double s_log = 0;          // SCALE_STATE: log scale of the parallax speed proxy
// adaptive measurement noise (ADAPT_R): running factors on the direction / speed noise, from normalized innovations
double lam_dir = 1.0, lam_spd = 1.0;
double last_nis = 0;       // normalized innovation squared of the last kf_update_row (before any Huber inflation)
bool huber_on = false;     // Huber weighting for the current measurement rows
double P[NS * NS];
double dx[NS];             // accumulated error-state correction (injected after an update batch)

Mat3 R_wb;                 // body -> world (nominal)
Vec3 p_w, v_w, ba, bg;     // nominal IMU position, velocity, biases
Vec3 p_kf;                 // cloned position at the keyframe
Vec3 bw{0, 0, 0};          // FIXED_GAIN: world-frame accel bias (also absorbs tilt error)
double t_clone = 0;        // time of the keyframe clone
int n_windows = 0;         // vision windows since start (FIXED_GAIN warm-up)
Mat3 R_wb_kf[SPLIT];       // body orientation when each template subset was cut
Mat3 R_wb_clone;           // body orientation at the position clone
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
inline void axpy(double* __restrict dst, double a, const double* __restrict src) {
    for (int j = 0; j < NS; j++) dst[j] += a * src[j];
}

// rows of F applied to the rows of M: out = F M. F = I + the nonzero 3x3 blocks of the window transition.
struct FBlocks { double T; Mat3 A1, A2, A3, B1, B2, B3, C3; };
void apply_F(const FBlocks& f, const double* M, double* out) {
    std::memcpy(out, M, sizeof(double) * NS * NS);
    for (int i = 0; i < 3; i++) {
        double* op = out + (IP + i) * NS;   // dp  += T dv + A1 dth + A2 dba + A3 dbg
        double* ov = out + (IV + i) * NS;   // dv  += B1 dth + B2 dba + B3 dbg
        double* ot = out + (ITH + i) * NS;  // dth += C3 dbg
        axpy(op, f.T, M + (IV + i) * NS);
        for (int c = 0; c < 3; c++) {
            axpy(op, f.A1.m[3 * i + c], M + (ITH + c) * NS);
            axpy(op, f.A2.m[3 * i + c], M + (IBA + c) * NS);
            axpy(op, f.A3.m[3 * i + c], M + (IBG + c) * NS);
            axpy(ov, f.B1.m[3 * i + c], M + (ITH + c) * NS);
            axpy(ov, f.B2.m[3 * i + c], M + (IBA + c) * NS);
            axpy(ov, f.B3.m[3 * i + c], M + (IBG + c) * NS);
            axpy(ot, f.C3.m[3 * i + c], M + (IBG + c) * NS);
        }
    }
}

// propagate over the accumulated IMU interval (T, mean rotation, mean world specific force): P = F P F^T + Q
void kf_propagate() {
#if PROF_SKIP_PROP || FIXED_GAIN
    acc_T = 0; acc_n = 0; acc_R = Mat3{{0}}; acc_f = {0, 0, 0}; return;
#endif
    if (acc_n == 0 || !(acc_T > 0)) return;
    double T = acc_T - prop_debt;     // part of this interval may already have been propagated ahead
    prop_debt = 0;
    if (prop_ahead_next) {            // propagate ahead over the frames until the direction update
        double ahead = SPLIT * (acc_T / acc_n) * 10.0;
        T += ahead; prop_debt = ahead; prop_ahead_next = false;
    }
    if (!(T > 0)) { acc_T = 0; acc_n = 0; acc_R = Mat3{{0}}; acc_f = {0, 0, 0}; return; }
    Mat3 Rbar = mscale(acc_R, 1.0 / acc_n);
    Vec3 f_w = scale(acc_f, 1.0 / acc_n);
    acc_T = 0; acc_n = 0; acc_R = Mat3{{0}}; acc_f = {0, 0, 0};
    Mat3 Fx = skew(f_w), FxR = mul(Fx, Rbar);
    FBlocks f{T, mscale(Fx, -0.5 * T * T), mscale(Rbar, -0.5 * T * T), mscale(FxR, T * T * T / 6.0),
              mscale(Fx, -T), mscale(Rbar, -T), mscale(FxR, 0.5 * T * T), mscale(Rbar, -T)};
    static double Q[NS * NS], Qt[NS * NS];
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
#if SCALE_STATE
    if (i == 0) P[IS * NS + IS] += SIG_S_RW * SIG_S_RW * T;
#endif
    }
}

// sequential scalar update: innovation y (measured - predicted), measurement row H, noise Rn
bool kf_update_row(const double Hrow[NS], double y, double Rn) {
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
    last_nis = yc * yc / S;
    if (last_nis > CHI2_GATE) return false;
    if (HUBER_C > 0 && huber_on && last_nis > HUBER_C * HUBER_C) {   // Huber: weight c/|z| == noise inflated by |z|/c
        double infl = std::sqrt(last_nis) / HUBER_C;
        S += (infl - 1.0) * Rn;
    }
    double g = yc / S, is = 1.0 / S;
    for (int i = 0; i < NS; i++) dx[i] += PH[i] * g;
    for (int i = 0; i < NS; i++) axpy(P + i * NS, -PH[i] * is, PH);
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
    R_wb = mul(exp_so3(Vec3{dx[6], dx[7], dx[8]}), R_wb);
    orthonormalize(R_wb);
    ba = add(ba, Vec3{dx[9], dx[10], dx[11]});
    bg = add(bg, Vec3{dx[12], dx[13], dx[14]});
    p_kf = add(p_kf, Vec3{dx[15], dx[16], dx[17]});
#if SCALE_STATE
    s_log += dx[IS];
    if (s_log > SCALE_MAX) s_log = SCALE_MAX;
    if (s_log < -SCALE_MAX) s_log = -SCALE_MAX;
#endif
    for (int i = 0; i < NS; i++) dx[i] = 0;
    for (int i = 0; i < NS; i++)
        for (int j = i + 1; j < NS; j++) {
            double s = 0.5 * (P[i * NS + j] + P[j * NS + i]);
            P[i * NS + j] = P[j * NS + i] = s;
        }
}

void kf_clone() {
    p_kf = p_w;
    t_clone = last_t;
#if FIXED_GAIN
    return;
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
    if (RANK_SIMD) {   // scores only (cells are fixed); the top-K selection happens once in select_top()
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
#if PROF_SKIP_TEMPLATES
    return;
#endif
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

// Track the patches of subset `si` into the current frame; the rotation residual of this subset updates the gyro bias.
void track_subset(int si) {
    if (PROP_MODE == 0 && (!PROP_ONCE || si == 0)) kf_propagate();
    if (PROP_MODE == 2 && si == SPLIT - 1) kf_propagate();  // PROP_ONCE: one propagation per tracking burst (IMU of the next
                                                // frames is folded into the following propagation)
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
    const int nt = n_tracks - first;
    if (!ROT_FROM_VISION || PROF_SKIP_DIR > 1 || nt < MIN_TRACKS / SPLIT + 1 || !(win_T[si] > 0)) return;
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

// All tracks of the window -> one motion direction (+ speed proxy) for the displacement since the keyframe clone.
void direction_update() {
#if PROF_SKIP_DIR
    return;
#endif
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
    if (nt < MIN_TRACKS) return;
    Vec3 d_est = add(sub(p_w, p_kf), sub_lever());
    double dn = norm(d_est);
    if (last_parallax < MIN_PAR_PX) {
        // little translational flow on well-tracked patches -> small displacement (indoor depths are bounded)
#if FIXED_GAIN
        if (SMALL_FLOW_UPDATE) { fg_apply(scale(d_est, -FG_KSMALL)); last_used = 1; }
        n_windows++;
        return;
#endif
        if (SMALL_FLOW_UPDATE) {
            double sd = (last_parallax + SMALL_FLOW_PX) / fx_s * DEPTH_PRIOR;
#if SCALE_STATE
            sd *= std::exp(s_log);
#endif
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
    Vec3 a = std::fabs(u.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 e1 = cross(u, a); e1 = scale(e1, 1.0 / norm(e1));
    Vec3 e2 = cross(u, e1);
    double sig_dir = std::sqrt(q / kept) * SIG_DIR_SCALE + SIG_DIR_PX / (last_parallax * std::sqrt((double)kept)) + SIG_DIR_FLOOR;
    double Rn = sig_dir * sig_dir * (dn * dn + D_FLOOR * D_FLOOR);
    if (ADAPT_R) Rn *= lam_dir;
    huber_on = true;
    Vec3 es[2] = {e1, e2};
    double Hrow[NS];
    double nis_sum = 0;
    int nis_n = 0;
    for (int k = 0; k < 2; k++) {  // e . d = 0; the world e is exp(dtheta) e_est, so dh/dtheta = e x d
        Vec3 e = es[k], ed = cross(e, d_est);
        row_d(Hrow, e);
        Hrow[ITH] = ed.x; Hrow[ITH + 1] = ed.y; Hrow[ITH + 2] = ed.z;
        if (kf_update_row(Hrow, -dot(e, d_est), Rn)) last_used++;
        nis_sum += last_nis; nis_n++;
    }
    if (ADAPT_R) lam_dir = adapt(lam_dir, nis_sum / nis_n);
    // speed proxy along u (Sim(3) only needs a consistent scale):
    //   SPEED_MODE 0: |d| ~ mean parallax * PAR_SPEED / f
    //   SPEED_MODE 1: per track, flow / (flow that a unit displacement along u gives at unit depth) = |d| / Z;
    //                 median over tracks, times a typical depth (removes the bias of forward vs sideways motion)
    double dmeas = last_parallax / fx_s * PAR_SPEED;
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
        if (nr >= 3) {
            for (int i = 1; i < nr; i++) { double v = rr[i]; int j = i - 1; while (j >= 0 && rr[j] > v) { rr[j + 1] = rr[j]; j--; } rr[j + 1] = v; }
            dmeas = rr[nr / 2] * SPEED_DEPTH;
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
    double sd = PAR_SPEED_REL * dmeas + 0.01;
    if (ADAPT_R) sd *= std::sqrt(lam_spd);
    row_d(Hrow, u);
#if SCALE_STATE
    // u . d = dmeas * exp(s): y = dmeas e^s - u.d, dh/ds = -dmeas e^s
    double es_ = std::exp(s_log);
    Hrow[IS] = -dmeas * es_;
    if (kf_update_row(Hrow, dmeas * es_ - dot(u, d_est), sd * sd * es_ * es_)) last_used++;
#else
    if (kf_update_row(Hrow, dmeas - dot(u, d_est), sd * sd)) last_used++;
#endif
    if (ADAPT_R) lam_spd = adapt(lam_spd, last_nis);
    huber_on = false;
#if SCALE_STATE
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
    bool at_rest = std::sqrt(var_a) < REST_ACCEL_STD && std::sqrt(var_g) < REST_GYRO_STD &&
                   norm(mean_g) < REST_GYRO_MEAN && std::fabs(na - GRAVITY) < REST_ACCEL_MEAN;
    bg = at_rest ? mean_g : Vec3{0, 0, 0};
    ba = {0, 0, 0};
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
#if SCALE_STATE
    P[IS * NS + IS] = SIG_S0 * SIG_S0;
#endif
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
        Vec3 a = sub(Vec3{r[1], r[2], r[3]}, ba);
        Vec3 w = sub(Vec3{r[4], r[5], r[6]}, bg);
        omega_b = w;
        double dt = t - last_t;
        last_t = t;
        if (!(dt > 0) || dt > 0.1) continue;
        Vec3 f = mul(R_wb, a);
        Vec3 acc = sub(add(f, g_w), bw);
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
    if (CLONE_PHASE == 0) kf_propagate();
    if (PROF_REP == 4) { acc_T = 0.05; acc_n = 10; acc_R = eye(); kf_propagate(); }
    if (PROF_REP == 5) { nk = 0; rank_slice(0, NCAND); }
    if (PROF_REP == 7) { select_top(); }
    if (PROF_REP == 8) { finish_rank(); }
#if defined(__wasm_simd128__)
    if (PROF_REP == 6) { float sc[4]; int xs[4] = {10, 20, 30, 40}; volatile float sink = 0; for (int k = 0; k < NCAND / 4; k++) { score_cells4(xs, 10 + (k % 8), sc); sink = sink + sc[0]; } }
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
    for (int k = 0; k < 3; k++) {
        double Hrow[NS] = {0};
        Hrow[IV + k] = 1.0;
        kf_update_row(Hrow, -vv[k], VPRIOR_SIG * VPRIOR_SIG);
    }
    apply_motion_prior();
    kf_inject();
}

bool state_ok() {
    double s = p_w.x + p_w.y + p_w.z + v_w.x + v_w.y + v_w.z + ba.x + ba.y + ba.z + bg.x + bg.y + bg.z;
    for (int i = 0; i < NS; i++) s += P[i * NS + i];
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
        make_keyframe();
        write_state(out);
        return VIOBENCH_STATUS_OK;
    }

    integrate(imu, imu_count);

    // window cycle of KF_LEN + SPLIT frames:
    //   phase 0                        rank candidate cells, clone the position
    //   phase 1 .. SPLIT               cut one template subset per frame
    //   phase KF_LEN .. KF_LEN+SPLIT-1 track one subset per frame; the last one also runs the direction update
    phase++;
    if (phase >= KF_LEN + SPLIT) phase = 0;
    if (phase == 0) {
        if (DIR_NEXT && have_kf) direction_update();               // one frame late, off the busiest tracking frame
        if (VPRIOR_IDLE == 2 && have_kf) apply_velocity_prior();  // right after the direction update, on a lighter frame
        make_keyframe();
    } else if (phase <= SPLIT) {
        const int si = phase - 1;
        if (VPRIOR_IDLE == 3 && phase == 1) apply_velocity_prior();
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
        cut_templates(si, want);
    } else if (VPRIOR_IDLE == 1 && phase == KF_LEN - 2) {
        apply_velocity_prior();              // weak zero-velocity prior on an otherwise idle frame
        if (PROP_MODE != 0 && KF_LEN - 2 == KF_LEN - 1) kf_propagate();
    } else if (PROP_MODE != 0 && phase == KF_LEN - 1) {
        if (PROP_MODE == 3) prop_ahead_next = true;
        kf_propagate();                      // idle frame: covariance up to just before tracking
    } else if (phase >= KF_LEN && have_kf) {
        track_subset(phase - KF_LEN);
        if (phase == KF_LEN + SPLIT - 1) {
            if (!DIR_NEXT) direction_update();
            if (PROF_REP == 1) direction_update();
            if (!VPRIOR_IDLE) apply_velocity_prior();
            if (PROF_REP == 2) apply_velocity_prior();
            if (PROF_REP == 3) kf_inject_now();
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
        p_kf = p_w;
        bw = {0, 0, 0};
    }
#ifdef VIO_DEBUG
    if (phase == KF_LEN + SPLIT - 1)
        std::fprintf(stderr, "t %.2f tracks %d used %d q %.3f par %.2f v %.2f %.2f %.2f bg %.4f %.4f %.4f\n",
                     t, last_tracks, last_used, last_quality, last_parallax, v_w.x, v_w.y, v_w.z, bg.x, bg.y, bg.z);
#endif
    write_state(out);
    return VIOBENCH_STATUS_OK;
}
