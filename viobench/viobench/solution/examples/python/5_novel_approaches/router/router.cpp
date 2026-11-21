// router: runs the MSCKF (+ splat map) and learned_lin side by side and blends their motion by flight regime.
// Fast flights (the MSCKF's regime): follow the MSCKF. Slow flights (start-hover, ~0.3 m/s, where learned_lin is
// competitive and the two errors are fairly independent): average the two models' position increments.
// Position increments are blended, never positions, so the output never jumps. Attitude and twist: the MSCKF.
// See README.md.
#include "viobench_abi.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#ifndef RT_LO
#define RT_LO 0.30     // m/s: at or below this running speed, weight RT_WMIN on the MSCKF
#endif
#ifndef RT_HI
#define RT_HI 0.45     // m/s: at or above this, weight 1 (MSCKF only)
#endif
#ifndef RT_TAU
#define RT_TAU 10.0    // s: time constant of the running speed (the MSCKF's own speed)
#endif
#ifndef RT_V0
#define RT_V0 0.7      // m/s: start value (default to the MSCKF)
#endif
#ifndef RT_RESET_FR
#define RT_RESET_FR 0     // frames after an MSCKF divergence reset during which its weight is capped at RT_WRESET
#endif
#ifndef RT_WRESET
#define RT_WRESET 0.5
#endif
#ifndef RT_SHRINK
#define RT_SHRINK 1.0  // test: scale on the blended position step
#endif
#ifndef RT_WMAX
#define RT_WMAX 1.0    // weight on the MSCKF at or above RT_HI
#endif
#ifndef RT_WMIN
#define RT_WMIN 0.5
#endif
#ifndef RT_LOWTR
#define RT_LOWTR 0     // >0: while the MSCKF has fewer active tracks than this (and RT_LOWHOLD frames after), cap its weight
#endif
#ifndef RT_WLOW
#define RT_WLOW 0.5
#endif
#ifndef RT_LOWHOLD
#define RT_LOWHOLD 20
#endif
#ifndef RT_VCAP
#define RT_VCAP 0.0    // m/s: >0 caps the output position step at this speed (an MSCKF update can teleport the position)
#endif
#ifndef RT_VCAPK
#define RT_VCAPK 0.0   // the cap is at least RT_VCAPK x the running speed
#endif
#ifndef RT_CAPLIN
#define RT_CAPLIN 0    // 1: when the MSCKF step is over the cap, use learned_lin's step instead (then cap)
#endif
extern "C" {
int32_t msckf_init(const char*, uint32_t); int32_t msckf_step(double, const uint8_t*, const double*, int32_t, double*);
int32_t msckf_frames_since_reset(void); int32_t msckf_active_tracks(void);
int32_t lin_init(const char*, uint32_t);   int32_t lin_step(double, const uint8_t*, const double*, int32_t, double*);
}
static double pa_prev[3], pb_prev[3], pout[3], vrun = RT_V0, t_prev = 0, w_last = 1;
static bool first = true;
extern "C" {
VIOBENCH_EXPORT(abi_version) int32_t abi_version(void) { return VIOBENCH_ABI_VERSION; }
VIOBENCH_EXPORT(alloc) void* alloc(uint32_t size) { return std::malloc(size); }
VIOBENCH_EXPORT(init) int32_t init(const char* cfg, uint32_t n) {
    first = true; vrun = RT_V0;
    if (msckf_init(cfg, n) != VIOBENCH_STATUS_OK) return VIOBENCH_STATUS_ERROR;
    return lin_init(cfg, n);
}
VIOBENCH_EXPORT(step) int32_t step(double t, const uint8_t* frame, const double* imu, int32_t n, double* out) {
    double a[VIOBENCH_STATE_VALUES], b[VIOBENCH_STATE_VALUES];
    if (msckf_step(t, frame, imu, n, a) != VIOBENCH_STATUS_OK) return VIOBENCH_STATUS_ERROR;
    const bool bok = lin_step(t, frame, imu, n, b) == VIOBENCH_STATUS_OK && std::isfinite(b[0]) && std::isfinite(b[1]) && std::isfinite(b[2]);
    if (first) {
        for (int i = 0; i < 3; i++) { pa_prev[i] = a[i]; pb_prev[i] = bok ? b[i] : a[i]; pout[i] = a[i]; }
        first = false;
    } else {
        double da[3], db[3], sa = 0;
        for (int i = 0; i < 3; i++) { da[i] = a[i] - pa_prev[i]; db[i] = bok ? b[i] - pb_prev[i] : da[i]; sa += da[i] * da[i]; }
        const double dt = t - t_prev;
        if (dt > 1e-4) vrun += (std::sqrt(sa) / dt - vrun) * (dt / RT_TAU);
        double x = (vrun - RT_LO) / (RT_HI - RT_LO); x = x < 0 ? 0 : (x > 1 ? 1 : x);
        const double w = RT_WMIN + (RT_WMAX - RT_WMIN) * x;
        const double wr = msckf_frames_since_reset() < RT_RESET_FR ? RT_WRESET : 1.0;   // just reset: lean on learned_lin
        w_last = w;
        static int low_left = 0;
        if (RT_LOWTR > 0) { if (msckf_active_tracks() < RT_LOWTR) low_left = RT_LOWHOLD + 1; if (low_left > 0) low_left--; }
        double wu = w < wr ? w : wr;
        if (RT_LOWTR > 0 && low_left > 0 && wu > RT_WLOW) wu = RT_WLOW;
        double st[3], s2 = 0;
        for (int i = 0; i < 3; i++) { st[i] = RT_SHRINK * (wu * da[i] + (1 - wu) * db[i]); s2 += st[i] * st[i]; }
        if (RT_VCAP > 0 && dt > 1e-4) {
            double cap = RT_VCAP; if (RT_VCAPK * vrun > cap) cap = RT_VCAPK * vrun;
            cap *= dt;
            if (RT_CAPLIN && bok && sa > cap * cap) { s2 = 0; for (int i = 0; i < 3; i++) { st[i] = db[i]; s2 += st[i] * st[i]; } }
            if (s2 > cap * cap) { const double k = cap / std::sqrt(s2); for (int i = 0; i < 3; i++) st[i] *= k; }
        }
        for (int i = 0; i < 3; i++) { pout[i] += st[i]; pa_prev[i] = a[i]; if (bok) pb_prev[i] = b[i]; }
    }
    t_prev = t;
    for (int i = 0; i < VIOBENCH_STATE_VALUES; i++) out[i] = a[i];
    for (int i = 0; i < 3; i++) out[i] = std::isfinite(pout[i]) ? pout[i] : a[i];
    return VIOBENCH_STATUS_OK;
}
}
