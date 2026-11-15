// v1: compute-light VIO for viobench. Native C++, no OpenCV, flat per-frame cost.
//
// Estimator: 18-state error-state EKF
//   [dp(3) dv(3) dtheta(3, world frame) dba(3) dbg(3) dp_kf(3)]
// - IMU (200 Hz) propagates the nominal state every frame; covariance propagated once per frame.
// - Every KF_LEN frames (keyframe window), N_TRACK textured patches of the keyframe (80x60 green image) are tracked into
//   the current frame with inverse-compositional Lucas-Kanade, started at the gyro-predicted location.
//   Each track gives n = (R_10 f0) x f1, orthogonal to the camera displacement d (depth-free epipolar constraint).
//   All tracks are aggregated into one motion direction u (smallest eigenvector of sum n n^T, one robust refit),
//   and the filter applies two constraints e_k . d = 0 (e_k orthogonal to u) with noise from the parallax.
//   Scale comes from the accelerometer; the attitude coupling (H_theta = e x d) makes tilt and biases observable.
// - Pose of the camera relative to the first camera frame; twist in the current camera frame.
#include "viobench_abi.h"
#include "params.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr double GRAVITY = 9.81;

// ---------------- small linear algebra ----------------
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

Mat3 exp_so3(Vec3 w) {
    double th2 = dot(w, w), a, b;
    if (th2 < 1e-12) { a = 1.0 - th2 / 6.0; b = 0.5 - th2 / 24.0; }
    else { double th = std::sqrt(th2); a = std::sin(th) / th; b = (1.0 - std::cos(th)) / th2; }
    double x = w.x, y = w.y, z = w.z;
    return {{1 - b * (y * y + z * z), -a * z + b * x * y, a * y + b * x * z,
             a * z + b * x * y, 1 - b * (x * x + z * z), -a * x + b * y * z,
             -a * y + b * x * z, a * x + b * y * z, 1 - b * (x * x + y * y)}};
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

// ---------------- configuration ----------------
int W = 320, H = 240;          // delivered frame
int SW = 80, SH = 60;          // small image
double fx_s, fy_s, cx_s, cy_s; // small-image intrinsics
Mat3 R_ci, R_ic;               // IMU <-> camera rotation
Vec3 r_cam_in_imu;             // lever arm: camera centre in IMU coordinates

// ---------------- images ----------------
constexpr int MAX_SMALL = 320 * 240;
float img_a[MAX_SMALL], img_b[MAX_SMALL];
float* img_kf = img_a;
float* img_cur = img_b;
bool have_kf = false;
bool pending_select = false;  // patch selection on the new keyframe is deferred by one frame to flatten fuel
int frames_since_kf = 0;

void downsample(const uint8_t* frame) {
    // mean of a 2x2 block of green samples at the centre of each DS x DS cell
    const int off = DS / 2 - 1;
    const int stride = W * 3;
    for (int y = 0; y < SH; y++) {
        const uint8_t* r0 = frame + (y * DS + off) * stride + 1;
        const uint8_t* r1 = r0 + stride;
        float* o = img_cur + y * SW;
        for (int x = 0; x < SW; x++) {
            int c = (x * DS + off) * 3;
            o[x] = 0.25f * (float)(r0[c] + r0[c + 3] + r1[c] + r1[c + 3]);
        }
    }
}

// ---------------- filter state ----------------
constexpr int IP = 0, IV = 3, ITH = 6, IBA = 9, IBG = 12, IPC = 15, NS = 18;
double P[NS * NS];
double dx[NS];             // accumulated error-state correction (injected after an update batch)

Mat3 R_wb;                 // body -> world (nominal)
Vec3 p_w, v_w, ba, bg;     // nominal IMU position, velocity, biases
Vec3 p_kf;                 // cloned position at the keyframe
Mat3 R_wb_kf;              // body orientation at the keyframe
Vec3 omega_b;              // latest bias-corrected angular rate
const Vec3 g_w{0, 0, -GRAVITY};
Mat3 R_wc0;                // first camera orientation
Vec3 p_wc0;                // first camera position

bool initialized = false;
double calib_t0 = -1, last_t = 0;
int calib_n = 0;
double sum_a[3], sum_a2[3], sum_g[3], sum_g2[3];

int last_tracks = 0, last_used = 0;  // diagnostics
double last_quality = 0, last_parallax = 0;

// ---------------- config parsing ----------------
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

// ---------------- EKF ----------------
inline void set_block(double* F, int r, int c, const Mat3& M) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) F[(r + i) * NS + c + j] = M.m[3 * i + j];
}

// T: window length, Rbar: mean body->world rotation, f_w: mean world specific force R (a - ba)
void kf_propagate(double T, const Mat3& Rbar, Vec3 f_w) {
    static double F[NS * NS], FP[NS * NS];
    std::memset(F, 0, sizeof F);
    for (int i = 0; i < NS; i++) F[i * NS + i] = 1.0;
    Mat3 Fx = skew(f_w), FxR = mul(Fx, Rbar);
    set_block(F, IP, IV, mscale(eye(), T));
    set_block(F, IP, ITH, mscale(Fx, -0.5 * T * T));
    set_block(F, IP, IBA, mscale(Rbar, -0.5 * T * T));
    set_block(F, IP, IBG, mscale(FxR, T * T * T / 6.0));
    set_block(F, IV, ITH, mscale(Fx, -T));
    set_block(F, IV, IBA, mscale(Rbar, -T));
    set_block(F, IV, IBG, mscale(FxR, 0.5 * T * T));
    set_block(F, ITH, IBG, mscale(Rbar, -T));
    // P = F P F^T; F is identity outside rows 0..8 and its nonzeros lie in columns 0..14
    for (int i = 0; i < NS; i++)
        for (int j = 0; j < NS; j++) {
            double s;
            if (i >= 9) s = P[i * NS + j];
            else { s = 0; for (int k = 0; k < 15; k++) s += F[i * NS + k] * P[k * NS + j]; }
            FP[i * NS + j] = s;
        }
    for (int i = 0; i < NS; i++)
        for (int j = 0; j < NS; j++) {
            double s;
            if (j >= 9) s = FP[i * NS + j];
            else { s = 0; for (int k = 0; k < 15; k++) s += FP[i * NS + k] * F[j * NS + k]; }
            P[i * NS + j] = s;
        }
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
}

// sequential scalar update: innovation y (measured - predicted), measurement row H, noise Rn
bool kf_update_row(const double Hrow[NS], double y, double Rn) {
    double PH[NS];
    for (int i = 0; i < NS; i++) {
        double s = 0;
        for (int j = 0; j < NS; j++) if (Hrow[j] != 0.0) s += P[i * NS + j] * Hrow[j];
        PH[i] = s;
    }
    double S = Rn;
    for (int j = 0; j < NS; j++) S += Hrow[j] * PH[j];
    if (!(S > 0)) return false;
    double yc = y;
    for (int j = 0; j < NS; j++) yc -= Hrow[j] * dx[j];
    if (yc * yc / S > CHI2_GATE) return false;
    for (int i = 0; i < NS; i++) dx[i] += PH[i] / S * yc;
    for (int i = 0; i < NS; i++)
        for (int j = 0; j < NS; j++) P[i * NS + j] -= PH[i] * PH[j] / S;
    return true;
}

void kf_inject() {
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
            double s = 0.5 * (P[i * NS + j] + P[j * NS + i]);
            P[i * NS + j] = P[j * NS + i] = s;
        }
}

void kf_clone() {
    p_kf = p_w;
    for (int i = 0; i < NS; i++)
        for (int j = 0; j < 3; j++) P[i * NS + IPC + j] = P[i * NS + IP + j];
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < NS; i++) P[(IPC + j) * NS + i] = P[(IP + j) * NS + i];
}

// ---------------- patches / Lucas-Kanade ----------------
constexpr int PS = 2 * PATCH_R + 1;
constexpr int PN = PS * PS;
constexpr int NCAND = GRID_X * GRID_Y;

struct Cand { int x, y; double score, gxx, gxy, gyy; };
Cand cands[NCAND];
int sel_idx[NCAND];
int n_sel = 0;

inline float bilinear(const float* img, double x, double y) {
    int x0 = (int)x, y0 = (int)y;
    float ax = (float)(x - x0), ay = (float)(y - y0);
    const float* p = img + y0 * SW + x0;
    return (1 - ay) * ((1 - ax) * p[0] + ax * p[1]) + ay * ((1 - ax) * p[SW] + ax * p[SW + 1]);
}

void select_patches() {  // structure-tensor score on the keyframe image, keep the N_TRACK best
    const int m = PATCH_R + 2;
    for (int gy = 0; gy < GRID_Y; gy++)
        for (int gx = 0; gx < GRID_X; gx++) {
            Cand& c = cands[gy * GRID_X + gx];
            c.x = m + (int)((gx + 0.5) * (SW - 2 * m) / GRID_X);
            c.y = m + (int)((gy + 0.5) * (SH - 2 * m) / GRID_Y);
            double a = 0, b = 0, d = 0;
            for (int v = -PATCH_R; v <= PATCH_R; v++) {
                const float* row = img_kf + (c.y + v) * SW + c.x;
                for (int u = -PATCH_R; u <= PATCH_R; u++) {
                    double ix = 0.5 * (row[u + 1] - row[u - 1]);
                    double iy = 0.5 * (row[u + SW] - row[u - SW]);
                    a += ix * ix; b += ix * iy; d += iy * iy;
                }
            }
            c.gxx = a; c.gxy = b; c.gyy = d;
            double tr = 0.5 * (a + d), det = a * d - b * b;
            c.score = (tr - std::sqrt(std::fmax(0.0, tr * tr - det))) / PN;
        }
    for (int i = 0; i < NCAND; i++) sel_idx[i] = i;
    n_sel = N_TRACK < NCAND ? N_TRACK : NCAND;
    for (int i = 0; i < n_sel; i++) {
        int best = i;
        for (int j = i + 1; j < NCAND; j++)
            if (cands[sel_idx[j]].score > cands[sel_idx[best]].score) best = j;
        int t = sel_idx[i]; sel_idx[i] = sel_idx[best]; sel_idx[best] = t;
    }
    while (n_sel > 0 && cands[sel_idx[n_sel - 1]].score < MIN_EIG) n_sel--;
}

// Track keyframe patch c into img_cur starting at (px, py): translation + brightness offset, inverse compositional.
bool track(const Cand& c, double& px, double& py) {
    float T[PN], GX[PN], GY[PN], E[PN];
    int k = 0;
    double tmean = 0, tsq = 0;
    for (int v = -PATCH_R; v <= PATCH_R; v++) {
        const float* row = img_kf + (c.y + v) * SW + c.x;
        for (int u = -PATCH_R; u <= PATCH_R; u++, k++) {
            T[k] = row[u];
            GX[k] = 0.5f * (row[u + 1] - row[u - 1]);
            GY[k] = 0.5f * (row[u + SW] - row[u - SW]);
            tmean += T[k];
            tsq += (double)T[k] * T[k];
        }
    }
    tmean /= PN;
    double tstd = std::sqrt(std::fmax(1.0, tsq / PN - tmean * tmean));
    double det = c.gxx * c.gyy - c.gxy * c.gxy;
    if (!(det > 1e-9)) return false;
    double i00 = c.gyy / det, i01 = -c.gxy / det, i11 = c.gxx / det;
    const double x_start = px, y_start = py;
    double resid = 1e9;
    for (int it = 0; it < LK_ITERS; it++) {
        if (px < PATCH_R + 1 || py < PATCH_R + 1 || px >= SW - PATCH_R - 2 || py >= SH - PATCH_R - 2) return false;
        double se = 0;
        k = 0;
        for (int v = -PATCH_R; v <= PATCH_R; v++)
            for (int u = -PATCH_R; u <= PATCH_R; u++, k++) {
                float e = bilinear(img_cur, px + u, py + v) - T[k];
                E[k] = e;
                se += e;
            }
        double mean = se / PN, sx = 0, sy = 0, see = 0;
        for (k = 0; k < PN; k++) {
            double e = E[k] - mean;
            sx += GX[k] * e; sy += GY[k] * e; see += e * e;
        }
        resid = std::sqrt(see / PN);
        double ddx = i00 * sx + i01 * sy, ddy = i01 * sx + i11 * sy;
        px -= ddx; py -= ddy;
        if (ddx * ddx + ddy * ddy < 1e-4) break;
    }
    if (resid > MAX_RESID * tstd) return false;
    double cx = px - x_start, cy = py - y_start;
    if (cx * cx + cy * cy > MAX_CORR_PX * MAX_CORR_PX) return false;
    return px >= 0 && py >= 0 && px <= SW - 1 && py <= SH - 1;
}

// smallest-eigenvalue eigenvector of a symmetric 3x3 (inverse power iteration from u); returns the eigenvalue
double smallest_eig(const double M[9], Vec3& u) {
    double eps = 1e-12 * (M[0] + M[4] + M[8]) + 1e-30;
    double a = M[0] + eps, b = M[1], c = M[2], d = M[4] + eps, e = M[5], f = M[8] + eps;
    double A[9] = {d * f - e * e, c * e - b * f, b * e - c * d,
                   c * e - b * f, a * f - c * c, b * c - a * e,
                   b * e - c * d, b * c - a * e, a * d - b * b};
    for (int it = 0; it < 12; it++) {
        Vec3 w{A[0] * u.x + A[1] * u.y + A[2] * u.z, A[3] * u.x + A[4] * u.y + A[5] * u.z, A[6] * u.x + A[7] * u.y + A[8] * u.z};
        double n = norm(w);
        if (!(n > 1e-300)) break;
        u = scale(w, 1.0 / n);
    }
    Vec3 Mu{M[0] * u.x + M[1] * u.y + M[2] * u.z, M[3] * u.x + M[4] * u.y + M[5] * u.z, M[6] * u.x + M[7] * u.y + M[8] * u.z};
    return dot(u, Mu);
}

inline void accum(double M[9], Vec3 n) {
    M[0] += n.x * n.x; M[1] += n.x * n.y; M[2] += n.x * n.z;
    M[4] += n.y * n.y; M[5] += n.y * n.z; M[8] += n.z * n.z;
    M[3] = M[1]; M[6] = M[2]; M[7] = M[5];
}

void vision_update() {
    Mat3 R_wc0f = mul(R_wb_kf, R_ic), R_wc1f = mul(R_wb, R_ic);
    Mat3 R10 = mul(transpose(R_wc1f), R_wc0f);  // keyframe camera -> current camera
    Vec3 dlever = sub(mul(R_wb, r_cam_in_imu), mul(R_wb_kf, r_cam_in_imu));
    Vec3 ns[NCAND], frs[NCAND], f1s[NCAND];
    int nt = 0;
    double par_sum = 0;
    for (int s = 0; s < n_sel; s++) {
        const Cand& c = cands[sel_idx[s]];
        Vec3 f0{(c.x - cx_s) / fx_s, (c.y - cy_s) / fy_s, 1.0};
        Vec3 fr = mul(R10, f0);
        if (fr.z < 0.2) continue;
        double px0 = cx_s + fx_s * fr.x / fr.z, py0 = cy_s + fy_s * fr.y / fr.z;
        double px = px0, py = py0;
        if (!track(c, px, py)) continue;
        Vec3 f1{(px - cx_s) / fx_s, (py - cy_s) / fy_s, 1.0};
        Vec3 frn = scale(fr, 1.0 / norm(fr)), f1n = scale(f1, 1.0 / norm(f1));
        frs[nt] = fr; f1s[nt] = f1;
        ns[nt++] = mul(R_wc1f, cross(frn, f1n));  // world frame, |n| ~ parallax angle
        par_sum += std::sqrt((px - px0) * (px - px0) + (py - py0) * (py - py0));
    }
    last_tracks = nt;
    last_used = 0;
    last_parallax = nt ? par_sum / nt : 0;
    if (nt < MIN_TRACKS) return;
    Vec3 d_est = add(sub(p_w, p_kf), dlever);
    double dn = norm(d_est);
    if (last_parallax < MIN_PAR_PX) {
        // little translational flow on well-tracked patches => small displacement (indoor depths are bounded)
        if (SMALL_FLOW_UPDATE) {
            double sd = (last_parallax + SMALL_FLOW_PX) / fx_s * DEPTH_PRIOR;
            for (int k = 0; k < 3; k++) {
                double Hrow[NS] = {0};
                Hrow[IP + k] = 1.0; Hrow[IPC + k] = -1.0;
                double dk = k == 0 ? d_est.x : (k == 1 ? d_est.y : d_est.z);
                if (kf_update_row(Hrow, -dk, sd * sd)) last_used++;
            }
            kf_inject();
        }
        return;
    }
    Vec3 u = dn > 1e-6 ? scale(d_est, 1.0 / dn) : Vec3{0.577, 0.577, 0.577};
    double M[9] = {0};
    for (int i = 0; i < nt; i++) accum(M, ns[i]);
    smallest_eig(M, u);
    // robust refit: drop tracks far from the first fit
    double r[NCAND], rs[NCAND];
    for (int i = 0; i < nt; i++) { r[i] = std::fabs(dot(ns[i], u)); rs[i] = r[i]; }
    for (int i = 1; i < nt; i++) { double v = rs[i]; int j = i - 1; while (j >= 0 && rs[j] > v) { rs[j + 1] = rs[j]; j--; } rs[j + 1] = v; }
    double med = rs[nt / 2];
    double M2[9] = {0};
    int kept = 0;
    for (int i = 0; i < nt; i++)
        if (r[i] <= 3.0 * med + 1e-12) { accum(M2, ns[i]); kept++; }
    if (kept < MIN_TRACKS) return;
    double l1 = smallest_eig(M2, u);
    double lmid = 0.5 * (M2[0] + M2[4] + M2[8] - l1);
    if (!(lmid > 0)) return;
    double q = l1 / lmid;
    last_quality = q;
    if (q > MAX_Q) return;
    // sign from cheirality: translational flow of a point in front of the camera is (x d_z - d_x, y d_z - d_y) / Z
    {
        Vec3 uc = mulT(R_wc1f, u);
        double score = 0;
        for (int i = 0; i < nt; i++) {
            double xr = frs[i].x / frs[i].z, yr = frs[i].y / frs[i].z;
            double fx_ = f1s[i].x / f1s[i].z - xr, fy_ = f1s[i].y / f1s[i].z - yr;
            score += fx_ * (xr * uc.z - uc.x) + fy_ * (yr * uc.z - uc.y);
        }
        if (score < 0) u = scale(u, -1.0);
    }
    bool wrong_sign = dot(u, d_est) < 0;
#ifdef VIO_DEBUG
    { Vec3 uc = mulT(R_wc1f, u), dc = mulT(R_wc1f, d_est);
      std::fprintf(stderr, "DIR %.4f %.4f %.4f %.4f %.4f %.4f %.4f n %d par %.2f q %.4f\n", last_t, uc.x, uc.y, uc.z, dc.x, dc.y, dc.z, kept, last_parallax, q); }
#endif

    Vec3 a = std::fabs(u.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 e1 = cross(u, a); e1 = scale(e1, 1.0 / norm(e1));
    Vec3 e2 = cross(u, e1);
    double sig_dir = std::sqrt(q / kept) * SIG_DIR_SCALE + SIG_DIR_PX / (last_parallax * std::sqrt((double)kept)) + SIG_DIR_FLOOR;
    double Rn = sig_dir * sig_dir * (dn * dn + D_FLOOR * D_FLOOR);
    Vec3 es[2] = {e1, e2};
    for (int k = 0; k < 2; k++) {
        // h = e . d; the true world e is exp(dtheta) e_est, so dh/dtheta = e x d
        double Hrow[NS] = {0};
        Vec3 e = es[k], ed = cross(e, d_est);
        Hrow[IP] = e.x; Hrow[IP + 1] = e.y; Hrow[IP + 2] = e.z;
        Hrow[IPC] = -e.x; Hrow[IPC + 1] = -e.y; Hrow[IPC + 2] = -e.z;
        if (USE_TILT_COUPLING) { Hrow[ITH] = ed.x; Hrow[ITH + 1] = ed.y; Hrow[ITH + 2] = ed.z; }
        if (kf_update_row(Hrow, -dot(e, d_est), Rn)) last_used++;
    }
    if (PAR_SPEED > 0) {  // speed proxy: |d| ~ parallax * depth / f, with a constant typical depth (only consistency matters)
        double dmeas = last_parallax / fx_s * PAR_SPEED;
        double Hrow[NS] = {0};
        Hrow[IP] = u.x; Hrow[IP + 1] = u.y; Hrow[IP + 2] = u.z;
        Hrow[IPC] = -u.x; Hrow[IPC + 1] = -u.y; Hrow[IPC + 2] = -u.z;
        double sd = PAR_SPEED_REL * dmeas + 0.01;
        if (kf_update_row(Hrow, dmeas - dot(u, d_est), sd * sd)) last_used++;
        wrong_sign = false;
    }
    if (wrong_sign && FIX_SIGN) {  // estimate moves backwards along the measured direction: pull that component to zero
        double Hrow[NS] = {0};
        Hrow[IP] = u.x; Hrow[IP + 1] = u.y; Hrow[IP + 2] = u.z;
        Hrow[IPC] = -u.x; Hrow[IPC + 1] = -u.y; Hrow[IPC + 2] = -u.z;
        if (kf_update_row(Hrow, -dot(u, d_est), Rn)) last_used++;
    }
    kf_inject();
}

// ---------------- IMU ----------------
void finish_calibration(double t) {
    double n = calib_n > 0 ? calib_n : 1;
    Vec3 mean_a{sum_a[0] / n, sum_a[1] / n, sum_a[2] / n};
    double var_a = 0, var_g = 0;
    for (int i = 0; i < 3; i++) {
        double ma = sum_a[i] / n, mg = sum_g[i] / n;
        var_a += std::fmax(0.0, sum_a2[i] / n - ma * ma);
        var_g += std::fmax(0.0, sum_g2[i] / n - mg * mg);
    }
    Vec3 mean_g{sum_g[0] / n, sum_g[1] / n, sum_g[2] / n};
    double na0 = norm(mean_a);
    bool at_rest = std::sqrt(var_a) < REST_ACCEL_STD && std::sqrt(var_g) < REST_GYRO_STD &&
                   norm(mean_g) < REST_GYRO_MEAN && std::fabs(na0 - GRAVITY) < REST_ACCEL_MEAN;
    bg = at_rest ? Vec3{sum_g[0] / n, sum_g[1] / n, sum_g[2] / n} : Vec3{0, 0, 0};
    ba = {0, 0, 0};
    double na = norm(mean_a);
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

// Integrate the frame's IMU samples; returns the window length, mean rotation and mean world specific force.
double integrate(const double* imu, int count, Mat3& Rbar, Vec3& f_mean) {
    double T = 0;
    Mat3 Rsum{{0}};
    Vec3 fsum{0, 0, 0};
    int n = 0;
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
        Vec3 acc = add(f, g_w);
        p_w = add(p_w, add(scale(v_w, dt), scale(acc, 0.5 * dt * dt)));
        v_w = add(v_w, scale(acc, dt));
        R_wb = mul(R_wb, exp_so3(scale(w, dt)));
        for (int i = 0; i < 9; i++) Rsum.m[i] += R_wb.m[i];
        fsum = add(fsum, f);
        n++;
        T += dt;
    }
    orthonormalize(R_wb);
    if (n > 0) {
        for (int i = 0; i < 9; i++) Rbar.m[i] = Rsum.m[i] / n;
        f_mean = scale(fsum, 1.0 / n);
    } else {
        Rbar = R_wb;
        f_mean = {0, 0, GRAVITY};
    }
    return T;
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

void make_keyframe() {
    float* t = img_kf; img_kf = img_cur; img_cur = t;
    have_kf = true;
    frames_since_kf = 0;
    n_sel = 0;
    pending_select = true;
    kf_clone();
    R_wb_kf = R_wb;
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
    if (SW * SH > MAX_SMALL || SW < 2 * PATCH_R + 6 || SH < 2 * PATCH_R + 6) return -1;
    // small pixel x averages full pixels around x*DS + DS/2 - 0.5
    fx_s = fx / DS; fy_s = fy / DS;
    cx_s = (cx - (DS / 2 - 0.5)) / DS; cy_s = (cy - (DS / 2 - 0.5)) / DS;

    R_wb = eye();
    bg = ba = omega_b = {0, 0, 0};
    std::memset(sum_a, 0, sizeof sum_a); std::memset(sum_a2, 0, sizeof sum_a2);
    std::memset(sum_g, 0, sizeof sum_g); std::memset(sum_g2, 0, sizeof sum_g2);
    calib_n = 0; calib_t0 = -1;
    initialized = false; have_kf = false;
    return VIOBENCH_STATUS_OK;
}

int32_t step(double t, const uint8_t* frame, const double* imu, int32_t imu_count, double* out) {
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
        downsample(frame);
        make_keyframe();
        write_state(out);
        return VIOBENCH_STATUS_OK;
    }

    Mat3 Rbar;
    Vec3 f_mean;
    double T = integrate(imu, imu_count, Rbar, f_mean);
    if (T > 0) kf_propagate(T, Rbar, f_mean);

    if (pending_select) {
        select_patches();
        pending_select = false;
    }
    if (++frames_since_kf >= KF_LEN) {
        downsample(frame);
        if (have_kf && n_sel > 0) vision_update();
#ifdef VIO_DEBUG
        std::fprintf(stderr, "KF t %.2f tr %d used %d q %.3f par %.2f |v| %.2f v %.2f %.2f %.2f ba %.2f %.2f %.2f Pv %.3f Pth %.4f\n", last_t, last_tracks, last_used, last_quality, last_parallax,
                     norm(v_w), v_w.x, v_w.y, v_w.z, ba.x, ba.y, ba.z, std::sqrt(P[IV * NS + IV]), std::sqrt(P[ITH * NS + ITH]));
#endif
        if (VPRIOR_SIG > 0) {  // weak prior: the platform's speed is bounded
            for (int k = 0; k < 3; k++) {
                double Hrow[NS] = {0};
                Hrow[IV + k] = 1.0;
                double vk = k == 0 ? v_w.x : (k == 1 ? v_w.y : v_w.z);
                kf_update_row(Hrow, -vk, VPRIOR_SIG * VPRIOR_SIG);
            }
            kf_inject();
        }
        make_keyframe();
    }
    if (!state_ok()) {  // never output garbage: restart the filter around the current pose
        v_w = {0, 0, 0};
        ba = {0, 0, 0};
        if (!std::isfinite(p_w.x + p_w.y + p_w.z)) p_w = {0, 0, 0};
        std::memset(P, 0, sizeof P);
        for (int i = 0; i < NS; i++) P[i * NS + i] = 1e-2;
        for (int i = 0; i < NS; i++) dx[i] = 0;
        p_kf = p_w;
    }
#ifdef VIO_DEBUG
    static int fc = 0;
    if (++fc % 20 == 0)
        std::fprintf(stderr, "t %.2f tracks %d used %d q %.3f par %.2f v %.2f %.2f %.2f ba %.3f %.3f %.3f bg %.4f %.4f %.4f\n",
                     t, last_tracks, last_used, last_quality, last_parallax, v_w.x, v_w.y, v_w.z, ba.x, ba.y, ba.z, bg.x, bg.y, bg.z);
#endif
    write_state(out);
    return VIOBENCH_STATUS_OK;
}
