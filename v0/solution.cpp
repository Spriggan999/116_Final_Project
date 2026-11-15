// v0: native IMU-only dead reckoning (viobench ABI v1).
// - First CALIB_SECONDS of IMU: gravity direction from mean accel; gyro bias from mean gyro if at rest.
// - Then: gyro integration for attitude, accel double integration for velocity/position (gravity removed).
// - Optional velocity leak (VEL_LEAK_TAU) to bound drift; set <= 0 for pure dead reckoning.
// Output: pose T_c0_ck and twist in the current camera frame. Never NaN.
#include "viobench_abi.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

constexpr double GRAVITY = 9.81;
constexpr double CALIB_SECONDS = 0.5;
constexpr double REST_ACCEL_STD = 1.0;  // m/s^2, norm of per-axis std
constexpr double REST_GYRO_STD = 0.2;   // rad/s
constexpr double VEL_LEAK_TAU = 0.0;    // s; 0 = off. Try 2-10 s to damp drift.

struct Mat3 { double m[9]; };  // row-major
struct Vec3 { double x, y, z; };

inline Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 scale(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double norm(Vec3 a) { return std::sqrt(dot(a, a)); }

inline Vec3 mul(const Mat3& R, Vec3 v) {
    return {R.m[0] * v.x + R.m[1] * v.y + R.m[2] * v.z,
            R.m[3] * v.x + R.m[4] * v.y + R.m[5] * v.z,
            R.m[6] * v.x + R.m[7] * v.y + R.m[8] * v.z};
}
inline Vec3 mulT(const Mat3& R, Vec3 v) {  // R^T v
    return {R.m[0] * v.x + R.m[3] * v.y + R.m[6] * v.z,
            R.m[1] * v.x + R.m[4] * v.y + R.m[7] * v.z,
            R.m[2] * v.x + R.m[5] * v.y + R.m[8] * v.z};
}
inline Mat3 mul(const Mat3& A, const Mat3& B) {
    Mat3 C;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            C.m[3 * i + j] = A.m[3 * i] * B.m[j] + A.m[3 * i + 1] * B.m[3 + j] + A.m[3 * i + 2] * B.m[6 + j];
    return C;
}
inline Mat3 transpose(const Mat3& A) {
    return {{A.m[0], A.m[3], A.m[6], A.m[1], A.m[4], A.m[7], A.m[2], A.m[5], A.m[8]}};
}
inline Mat3 eye() { return {{1, 0, 0, 0, 1, 0, 0, 0, 1}}; }

Mat3 exp_so3(Vec3 w) {
    double th2 = dot(w, w);
    double a, b;  // R = I + a K + b K^2, K = [w]x
    if (th2 < 1e-12) { a = 1.0 - th2 / 6.0; b = 0.5 - th2 / 24.0; }
    else { double th = std::sqrt(th2); a = std::sin(th) / th; b = (1.0 - std::cos(th)) / th2; }
    double x = w.x, y = w.y, z = w.z;
    Mat3 R;
    R.m[0] = 1 - b * (y * y + z * z); R.m[1] = -a * z + b * x * y;      R.m[2] = a * y + b * x * z;
    R.m[3] = a * z + b * x * y;       R.m[4] = 1 - b * (x * x + z * z); R.m[5] = -a * x + b * y * z;
    R.m[6] = -a * y + b * x * z;      R.m[7] = a * x + b * y * z;       R.m[8] = 1 - b * (x * x + y * y);
    return R;
}

// Re-orthonormalize (Gram-Schmidt on rows) to stop numerical drift.
void orthonormalize(Mat3& R) {
    Vec3 r0{R.m[0], R.m[1], R.m[2]}, r1{R.m[3], R.m[4], R.m[5]};
    r0 = scale(r0, 1.0 / norm(r0));
    r1 = sub(r1, scale(r0, dot(r0, r1)));
    r1 = scale(r1, 1.0 / norm(r1));
    Vec3 r2 = cross(r0, r1);
    R = {{r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z}};
}

void to_quat(const Mat3& R, double q[4]) {  // x, y, z, w
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

// ---- state ----
Mat3 R_ci;        // rotation IMU -> camera (from T_cam_imu)
Vec3 t_ci;        // translation part of T_cam_imu
Mat3 R_ic;        // camera -> IMU
Vec3 r_cam_in_imu;  // camera centre in IMU coordinates (lever arm)

bool initialized = false;
double calib_t0 = -1, last_t = 0;
int calib_n = 0;
double sum_a[3], sum_a2[3], sum_g[3], sum_g2[3];

Mat3 R_wb;        // body (IMU) -> world
Vec3 v_w, p_w;    // IMU velocity/position in world
Vec3 gyro_bias;
Vec3 omega_b;     // latest bias-corrected angular rate (body)
const Vec3 g_w{0, 0, -GRAVITY};
Mat3 R_wc0;       // first camera orientation in world
Vec3 p_wc0;       // first camera position in world

// Minimal parser: find "T_cam_imu" and read the next 16 numbers.
bool parse_T(const char* json, uint32_t size, double T[16]) {
    const char* key = "\"T_cam_imu\"";
    size_t klen = std::strlen(key);
    const char* end = json + size;
    const char* p = json;
    for (; p + klen <= end; p++)
        if (std::memcmp(p, key, klen) == 0) break;
    if (p + klen > end) return false;
    p += klen;
    for (int k = 0; k < 16; k++) {
        while (p < end && !((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' || *p == '.')) p++;
        if (p >= end) return false;
        char buf[64];
        int n = 0;
        while (p < end && n < 63 && ((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' || *p == '.' || *p == 'e' || *p == 'E'))
            buf[n++] = *p++;
        buf[n] = 0;
        T[k] = std::strtod(buf, nullptr);
    }
    return true;
}

void camera_pose_world(Mat3& R_wc, Vec3& p_wc) {
    R_wc = mul(R_wb, R_ic);
    p_wc = add(p_w, mul(R_wb, r_cam_in_imu));
}

void write_identity(double* out) {
    for (int i = 0; i < VIOBENCH_STATE_VALUES; i++) out[i] = 0.0;
    out[6] = 1.0;
}

void finish_calibration(double t) {
    double n = calib_n > 0 ? calib_n : 1;
    Vec3 mean_a{sum_a[0] / n, sum_a[1] / n, sum_a[2] / n};
    double var_a = 0, var_g = 0;
    for (int i = 0; i < 3; i++) {
        double ma = sum_a[i] / n, mg = sum_g[i] / n;
        var_a += std::fmax(0.0, sum_a2[i] / n - ma * ma);
        var_g += std::fmax(0.0, sum_g2[i] / n - mg * mg);
    }
    bool at_rest = std::sqrt(var_a) < REST_ACCEL_STD && std::sqrt(var_g) < REST_GYRO_STD;
    gyro_bias = at_rest ? Vec3{sum_g[0] / n, sum_g[1] / n, sum_g[2] / n} : Vec3{0, 0, 0};

    // world z = "up" in body coords; R_wb rows are world axes in body coords
    double na = norm(mean_a);
    Vec3 z = na > 1e-6 ? scale(mean_a, 1.0 / na) : Vec3{0, 0, 1};
    Vec3 xc = std::fabs(z.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 x = sub(xc, scale(z, dot(xc, z)));
    x = scale(x, 1.0 / norm(x));
    Vec3 y = cross(z, x);
    R_wb = {{x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z}};
    v_w = {0, 0, 0};
    p_w = {0, 0, 0};
    camera_pose_world(R_wc0, p_wc0);
    last_t = t;
    initialized = true;
}

void integrate(const double* imu, int count) {
    double leak = VEL_LEAK_TAU > 0 ? 1.0 / VEL_LEAK_TAU : 0.0;
    for (int k = 0; k < count; k++) {
        const double* r = imu + 7 * k;
        double t = r[0];
        Vec3 a{r[1], r[2], r[3]};
        Vec3 w = sub(Vec3{r[4], r[5], r[6]}, gyro_bias);
        omega_b = w;
        double dt = t - last_t;
        last_t = t;
        if (!(dt > 0) || dt > 0.1) continue;
        Vec3 acc = add(mul(R_wb, a), g_w);  // world acceleration (before rotating forward)
        p_w = add(p_w, add(scale(v_w, dt), scale(acc, 0.5 * dt * dt)));
        v_w = add(v_w, scale(acc, dt));
        if (leak > 0) v_w = scale(v_w, 1.0 / (1.0 + leak * dt));
        R_wb = mul(R_wb, exp_so3(scale(w, dt)));
    }
    orthonormalize(R_wb);
}

void write_state(double* out) {
    Mat3 R_wc;
    Vec3 p_wc;
    camera_pose_world(R_wc, p_wc);
    // T_c0_c = T_w_c0^-1 T_w_c
    Mat3 R = mul(transpose(R_wc0), R_wc);
    Vec3 t = mulT(R_wc0, sub(p_wc, p_wc0));
    double q[4];
    to_quat(R, q);
    // twist in camera axes: v_cam = R_ci (R_wb^T v_w + omega x r), w_cam = R_ci omega
    Vec3 v_b = add(mulT(R_wb, v_w), cross(omega_b, r_cam_in_imu));
    Vec3 v_c = mul(R_ci, v_b);
    Vec3 w_c = mul(R_ci, omega_b);
    double vals[13] = {t.x, t.y, t.z, q[0], q[1], q[2], q[3], v_c.x, v_c.y, v_c.z, w_c.x, w_c.y, w_c.z};
    for (int i = 0; i < 13; i++) {
        if (!std::isfinite(vals[i])) { write_identity(out); return; }
        out[i] = vals[i];
    }
}

}  // namespace

int32_t abi_version() { return VIOBENCH_ABI_VERSION; }

void* alloc(uint32_t size) { return std::malloc(size ? size : 1); }

int32_t frame_width() { return 320; }

int32_t init(const char* json, uint32_t size) {
    double T[16];
    if (!parse_T(json, size, T)) {
        // fall back to identity extrinsics
        for (int i = 0; i < 16; i++) T[i] = (i % 5 == 0) ? 1.0 : 0.0;
    }
    R_ci = {{T[0], T[1], T[2], T[4], T[5], T[6], T[8], T[9], T[10]}};
    t_ci = {T[3], T[7], T[11]};
    R_ic = transpose(R_ci);
    r_cam_in_imu = scale(mul(R_ic, t_ci), -1.0);  // T_imu_cam translation = -R_ic t_ci
    R_wb = eye();
    v_w = p_w = gyro_bias = omega_b = {0, 0, 0};
    std::memset(sum_a, 0, sizeof sum_a); std::memset(sum_a2, 0, sizeof sum_a2);
    std::memset(sum_g, 0, sizeof sum_g); std::memset(sum_g2, 0, sizeof sum_g2);
    calib_n = 0;
    calib_t0 = -1;
    initialized = false;
    return VIOBENCH_STATUS_OK;
}

int32_t step(double t, const uint8_t*, const double* imu, int32_t imu_count, double* out) {
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
        write_state(out);
        return VIOBENCH_STATUS_OK;
    }
    integrate(imu, imu_count);
    write_state(out);
    return VIOBENCH_STATUS_OK;
}
