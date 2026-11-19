// IMU-only estimator with a nose-first cruise prior (EECS 116 Project 3, 5_novel_approaches).
// Native C++, no image access, constant small fuel per frame.
//
//   calibrate   average the IMU over the start hover (HOVER_S): gyro bias, gravity direction and the
//               accelerometer bias along gravity. The hover sags and wobbles but returns to rest, so the
//               motion averages out over the whole hover (over half of it, it does not).
//   attitude    integrate the bias-corrected gyro.
//   velocity    integrate the accelerometer, then pull the body-frame velocity toward the autopilot's
//               cruise habit: forward CRUISE m/s, no sideslip, no climb, with per-axis time constants.
//               A slow integral of the pull learns what remains of the accelerometer bias and of the
//               gravity that leaks through small tilt errors (a PI loop instead of a full filter).
//   position    integrate the velocity.
// Output: camera pose relative to the first camera frame, twist in the current camera frame.
#include "viobench_abi.h"
#include <cmath>
#include <cstdlib>
#include <cstring>

#ifndef HOVER_S
#define HOVER_S 0.9          // s of start hover used for calibration (the generator hovers 1 s)
#endif
#ifndef CRUISE
#define CRUISE 0.241         // m/s forward (training flights, scenes 20-49)
#endif
#ifndef TAU_X
#define TAU_X 1.0            // s: forward pull toward CRUISE
#endif
#ifndef TAU_Y
#define TAU_Y 0.3            // s: lateral pull toward 0 (no sideslip)
#endif
#ifndef TAU_Z
#define TAU_Z 0.5            // s: vertical pull toward 0
#endif
#ifndef TAU_I
#define TAU_I 20.0           // s: integral time of the bias-learning loop (0 = off)
#endif

namespace {
constexpr double G = 9.81;
struct V3 { double x, y, z; };
struct M3 { double m[9]; };
inline V3 add(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 scl(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline V3 mul(const M3& R, V3 v) {
    return {R.m[0] * v.x + R.m[1] * v.y + R.m[2] * v.z, R.m[3] * v.x + R.m[4] * v.y + R.m[5] * v.z,
            R.m[6] * v.x + R.m[7] * v.y + R.m[8] * v.z};
}
inline V3 mulT(const M3& R, V3 v) {
    return {R.m[0] * v.x + R.m[3] * v.y + R.m[6] * v.z, R.m[1] * v.x + R.m[4] * v.y + R.m[7] * v.z,
            R.m[2] * v.x + R.m[5] * v.y + R.m[8] * v.z};
}
inline M3 mul(const M3& A, const M3& B) {
    M3 C;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            C.m[3 * i + j] = A.m[3 * i] * B.m[j] + A.m[3 * i + 1] * B.m[3 + j] + A.m[3 * i + 2] * B.m[6 + j];
    return C;
}
inline M3 tr(const M3& A) { return {{A.m[0], A.m[3], A.m[6], A.m[1], A.m[4], A.m[7], A.m[2], A.m[5], A.m[8]}}; }

// R <- R exp([w]x), second order (|w| ~ 0.005 rad per sample)
void rotate(M3& R, V3 w) {
    double x = w.x, y = w.y, z = w.z, a = 1.0 - (x * x + y * y + z * z) / 6.0, b = 0.5;
    M3 E = {{1 - b * (y * y + z * z), -a * z + b * x * y, a * y + b * x * z,
             a * z + b * x * y, 1 - b * (x * x + z * z), -a * x + b * y * z,
             -a * y + b * x * z, a * x + b * y * z, 1 - b * (x * x + y * y)}};
    R = mul(R, E);
}
void orthonormalize(M3& R) {   // Gram-Schmidt on the columns
    V3 c0{R.m[0], R.m[3], R.m[6]}, c1{R.m[1], R.m[4], R.m[7]};
    c0 = scl(c0, 1.0 / std::sqrt(dot(c0, c0)));
    c1 = sub(c1, scl(c0, dot(c0, c1)));
    c1 = scl(c1, 1.0 / std::sqrt(dot(c1, c1)));
    V3 c2 = cross(c0, c1);
    R = {{c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z}};
}
void to_quat(const M3& R, double q[4]) {  // x y z w
    double t = R.m[0] + R.m[4] + R.m[8];
    if (t > 0) {
        double s = std::sqrt(t + 1.0) * 2;
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
}

M3 R_wb, R_ci, R_ic, R_wc0;
V3 r_cam, p_wc0, p_w, v_w, bg, ba, bi, omega;   // bi: learned body-frame acceleration correction
bool ready = false;
double t0 = -1, last_t = 0, sa[3], sg[3];
int n_cal = 0;

const char* find_key(const char* json, uint32_t size, const char* key) {
    size_t k = std::strlen(key);
    for (const char* p = json; p + k <= json + size; p++)
        if (std::memcmp(p, key, k) == 0) return p + k;
    return nullptr;
}
const char* read_number(const char* p, const char* end, double* out) {
    while (p < end && !((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' || *p == '.')) p++;
    if (p >= end) return nullptr;
    char* stop;
    *out = std::strtod(p, &stop);
    return stop;
}

void calibrate(double t) {
    V3 a{sa[0] / n_cal, sa[1] / n_cal, sa[2] / n_cal};
    bg = {sg[0] / n_cal, sg[1] / n_cal, sg[2] / n_cal};
    double na = std::sqrt(dot(a, a));
    V3 z = scl(a, 1.0 / na);
    ba = scl(z, na - G);                       // bias along gravity; the rest is inseparable from tilt
    V3 xc = std::fabs(z.x) < 0.9 ? V3{1, 0, 0} : V3{0, 1, 0};
    V3 x = sub(xc, scl(z, dot(xc, z)));
    x = scl(x, 1.0 / std::sqrt(dot(x, x)));
    V3 y = cross(z, x);
    R_wb = {{x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z}};
    p_w = v_w = bi = {0, 0, 0};
    R_wc0 = mul(R_wb, R_ic);
    p_wc0 = mul(R_wb, r_cam);
    last_t = t;
    ready = true;
}

void sample(const double* r) {
    double dt = r[0] - last_t;
    last_t = r[0];
    if (!(dt > 0) || dt > 0.1) return;
    omega = sub(V3{r[4], r[5], r[6]}, bg);
    V3 f = sub(sub(V3{r[1], r[2], r[3]}, ba), bi);
    V3 acc = add(mul(R_wb, f), V3{0, 0, -G});
    p_w = add(p_w, add(scl(v_w, dt), scl(acc, 0.5 * dt * dt)));
    v_w = add(v_w, scl(acc, dt));
    rotate(R_wb, scl(omega, dt));
    // cruise prior in the body frame
    V3 vb = mulT(R_wb, v_w);
    V3 e{CRUISE - vb.x, -vb.y, -vb.z};
    V3 pull{e.x * dt / TAU_X, e.y * dt / TAU_Y, e.z * dt / TAU_Z};
    v_w = add(v_w, mul(R_wb, pull));
    if (TAU_I > 0) bi = sub(bi, scl(pull, 1.0 / TAU_I));   // persistent pull = unmodelled acceleration
}

void write(double* out) {
    M3 R_wc = mul(R_wb, R_ic);
    V3 p_wc = add(p_w, mul(R_wb, r_cam));
    M3 R = mul(tr(R_wc0), R_wc);
    V3 t = mulT(R_wc0, sub(p_wc, p_wc0));
    double q[4];
    to_quat(R, q);
    V3 vb = add(mulT(R_wb, v_w), cross(omega, r_cam));
    V3 vc = mul(R_ci, vb), wc = mul(R_ci, omega);
    double v[13] = {t.x, t.y, t.z, q[0], q[1], q[2], q[3], vc.x, vc.y, vc.z, wc.x, wc.y, wc.z};
    for (int i = 0; i < 13; i++) {
        if (!std::isfinite(v[i])) { for (int k = 0; k < 13; k++) out[k] = k == 6; return; }
        out[i] = v[i];
    }
}
}  // namespace

int32_t abi_version() { return VIOBENCH_ABI_VERSION; }
void* alloc(uint32_t size) { return std::malloc(size ? size : 1); }

int32_t init(const char* json, uint32_t size) {
    double T[16];
    const char* p = find_key(json, size, "\"T_cam_imu\"");
    bool ok = p != nullptr;
    for (int k = 0; ok && k < 16; k++) { p = read_number(p, json + size, &T[k]); ok = p != nullptr; }
    if (!ok) for (int i = 0; i < 16; i++) T[i] = (i % 5 == 0) ? 1.0 : 0.0;
    R_ci = {{T[0], T[1], T[2], T[4], T[5], T[6], T[8], T[9], T[10]}};
    R_ic = tr(R_ci);
    r_cam = scl(mul(R_ic, V3{T[3], T[7], T[11]}), -1.0);
    R_wb = {{1, 0, 0, 0, 1, 0, 0, 0, 1}};
    bg = ba = bi = omega = {0, 0, 0};
    for (int i = 0; i < 3; i++) sa[i] = sg[i] = 0;
    n_cal = 0; t0 = -1; ready = false;
    return VIOBENCH_STATUS_OK;
}

int32_t step(double t, const uint8_t*, const double* imu, int32_t n, double* out) {
    if (!ready) {
        for (int k = 0; k < n; k++) {
            const double* r = imu + 7 * k;
            if (t0 < 0) t0 = r[0];
            for (int i = 0; i < 3; i++) { sa[i] += r[1 + i]; sg[i] += r[4 + i]; }
            n_cal++;
        }
        double last = n > 0 ? imu[7 * (n - 1)] : t;
        if (n_cal < 2 || last - t0 < HOVER_S) {
            for (int k = 0; k < 13; k++) out[k] = k == 6;
            return VIOBENCH_STATUS_OK;
        }
        calibrate(last);
        write(out);
        return VIOBENCH_STATUS_OK;
    }
    for (int k = 0; k < n; k++) sample(imu + 7 * k);
    orthonormalize(R_wb);
    write(out);
    return VIOBENCH_STATUS_OK;
}
