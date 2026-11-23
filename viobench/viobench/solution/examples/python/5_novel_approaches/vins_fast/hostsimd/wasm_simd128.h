// Portable lane-by-lane emulation of the wasm_simd128.h intrinsics used by vins_fast, for host replay builds
// (build_host.sh compiles with -D__wasm_simd128__ -I hostsimd -ffp-contract=off). Each lane is computed with the same
// IEEE operation as the wasm instruction, so results are bit-identical to the wasm build.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

struct v128_t { alignas(16) uint8_t b[16]; };

#define VE_GET(T, v, i) ([&]() { T x_; std::memcpy(&x_, (v).b + sizeof(T) * (i), sizeof(T)); return x_; }())
template <class T> static inline void ve_set(v128_t& v, int i, T x) { std::memcpy(v.b + sizeof(T) * i, &x, sizeof(T)); }
template <class T> static inline T ve_get(const v128_t& v, int i) { T x; std::memcpy(&x, v.b + sizeof(T) * i, sizeof(T)); return x; }

static inline v128_t wasm_v128_load(const void* p) { v128_t r; std::memcpy(r.b, p, 16); return r; }
static inline void wasm_v128_store(void* p, v128_t a) { std::memcpy(p, a.b, 16); }
static inline v128_t wasm_v128_and(v128_t a, v128_t b) { v128_t r; for (int i = 0; i < 16; i++) r.b[i] = a.b[i] & b.b[i]; return r; }
static inline v128_t ve_load8_lane(const void* p, v128_t v, int l) { v.b[l] = *(const uint8_t*)p; return v; }
#define wasm_v128_load8_lane(p, v, l) ve_load8_lane((p), (v), (l))
static inline void ve_store64_lane(void* p, v128_t v, int l) { std::memcpy(p, v.b + 8 * l, 8); }
#define wasm_v128_store64_lane(p, v, l) ve_store64_lane((p), (v), (l))

#define VE_BIN(name, T, n, expr) static inline v128_t name(v128_t a, v128_t b) { v128_t r; for (int i = 0; i < n; i++) { T x = ve_get<T>(a, i), y = ve_get<T>(b, i); ve_set<T>(r, i, (T)(expr)); } return r; }
#define VE_SPLAT(name, T, n) static inline v128_t name(T x) { v128_t r; for (int i = 0; i < n; i++) ve_set<T>(r, i, x); return r; }

// f64x2
VE_BIN(wasm_f64x2_add, double, 2, x + y)
VE_BIN(wasm_f64x2_sub, double, 2, x - y)
VE_BIN(wasm_f64x2_mul, double, 2, x * y)
VE_SPLAT(wasm_f64x2_splat, double, 2)
static inline v128_t wasm_f64x2_make(double a, double b) { v128_t r; ve_set<double>(r, 0, a); ve_set<double>(r, 1, b); return r; }
#define wasm_f64x2_extract_lane(v, l) ve_get<double>((v), (l))
static inline v128_t wasm_f64x2_promote_low_f32x4(v128_t a) { v128_t r; for (int i = 0; i < 2; i++) ve_set<double>(r, i, (double)ve_get<float>(a, i)); return r; }

// f32x4
VE_BIN(wasm_f32x4_add, float, 4, x + y)
VE_BIN(wasm_f32x4_sub, float, 4, x - y)
VE_BIN(wasm_f32x4_mul, float, 4, x * y)
VE_SPLAT(wasm_f32x4_splat, float, 4)
static inline v128_t wasm_f32x4_make(float a, float b, float c, float d) { v128_t r; ve_set<float>(r, 0, a); ve_set<float>(r, 1, b); ve_set<float>(r, 2, c); ve_set<float>(r, 3, d); return r; }
#define wasm_f32x4_extract_lane(v, l) ve_get<float>((v), (l))
static inline v128_t wasm_f32x4_neg(v128_t a) { v128_t r = a; for (int i = 0; i < 4; i++) r.b[4 * i + 3] ^= 0x80; return r; }
static inline v128_t wasm_f32x4_abs(v128_t a) { v128_t r = a; for (int i = 0; i < 4; i++) r.b[4 * i + 3] &= 0x7f; return r; }
static inline v128_t wasm_f32x4_convert_u32x4(v128_t a) { v128_t r; for (int i = 0; i < 4; i++) ve_set<float>(r, i, (float)ve_get<uint32_t>(a, i)); return r; }

// i32x4 / u32x4
VE_BIN(wasm_i32x4_add, uint32_t, 4, x + y)
VE_SPLAT(wasm_i32x4_splat, int32_t, 4)
static inline v128_t wasm_i32x4_make(int32_t a, int32_t b, int32_t c, int32_t d) { v128_t r; ve_set<int32_t>(r, 0, a); ve_set<int32_t>(r, 1, b); ve_set<int32_t>(r, 2, c); ve_set<int32_t>(r, 3, d); return r; }
#define wasm_i32x4_extract_lane(v, l) ve_get<int32_t>((v), (l))
static inline v128_t wasm_i32x4_extmul_low_i16x8(v128_t a, v128_t b) { v128_t r; for (int i = 0; i < 4; i++) ve_set<int32_t>(r, i, (int32_t)ve_get<int16_t>(a, i) * (int32_t)ve_get<int16_t>(b, i)); return r; }
static inline v128_t wasm_i32x4_extmul_high_i16x8(v128_t a, v128_t b) { v128_t r; for (int i = 0; i < 4; i++) ve_set<int32_t>(r, i, (int32_t)ve_get<int16_t>(a, i + 4) * (int32_t)ve_get<int16_t>(b, i + 4)); return r; }
static inline v128_t wasm_u32x4_extend_low_u16x8(v128_t a) { v128_t r; for (int i = 0; i < 4; i++) ve_set<uint32_t>(r, i, ve_get<uint16_t>(a, i)); return r; }
static inline v128_t wasm_u32x4_extend_high_u16x8(v128_t a) { v128_t r; for (int i = 0; i < 4; i++) ve_set<uint32_t>(r, i, ve_get<uint16_t>(a, i + 4)); return r; }

// i16x8 / u16x8
VE_BIN(wasm_i16x8_add, uint16_t, 8, x + y)
VE_BIN(wasm_i16x8_sub, uint16_t, 8, x - y)
VE_BIN(wasm_u16x8_max, uint16_t, 8, x > y ? x : y)
VE_SPLAT(wasm_i16x8_splat, int16_t, 8)
static inline v128_t wasm_i16x8_lt(v128_t a, v128_t b) { v128_t r; for (int i = 0; i < 8; i++) ve_set<int16_t>(r, i, ve_get<int16_t>(a, i) < ve_get<int16_t>(b, i) ? (int16_t)-1 : (int16_t)0); return r; }
static inline v128_t wasm_i16x8_make(int16_t a0, int16_t a1, int16_t a2, int16_t a3, int16_t a4, int16_t a5, int16_t a6, int16_t a7) {
    v128_t r; const int16_t a[8] = {a0, a1, a2, a3, a4, a5, a6, a7}; std::memcpy(r.b, a, 16); return r; }
#define wasm_u16x8_extract_lane(v, l) ve_get<uint16_t>((v), (l))
static inline v128_t wasm_u16x8_extend_low_u8x16(v128_t a) { v128_t r; for (int i = 0; i < 8; i++) ve_set<uint16_t>(r, i, a.b[i]); return r; }
static inline v128_t wasm_u16x8_extend_high_u8x16(v128_t a) { v128_t r; for (int i = 0; i < 8; i++) ve_set<uint16_t>(r, i, a.b[i + 8]); return r; }

// i8x16 / u8x16
VE_BIN(wasm_i8x16_sub, uint8_t, 16, x - y)
VE_BIN(wasm_u8x16_min, uint8_t, 16, x < y ? x : y)
VE_BIN(wasm_u8x16_max, uint8_t, 16, x > y ? x : y)
VE_SPLAT(wasm_i8x16_splat, int8_t, 16)

// shuffles (indices 0-15 pick from a, 16-31 from b; i32x4: 0-3 a, 4-7 b)
static inline v128_t ve_shuf8(v128_t a, v128_t b, const int* c) { v128_t r; for (int i = 0; i < 16; i++) r.b[i] = c[i] < 16 ? a.b[c[i]] : b.b[c[i] - 16]; return r; }
#define wasm_i8x16_shuffle(a, b, ...) ([&]() { const int c_[16] = {__VA_ARGS__}; return ve_shuf8((a), (b), c_); }())
static inline v128_t ve_shuf32(v128_t a, v128_t b, const int* c) { v128_t r; for (int i = 0; i < 4; i++) ve_set<uint32_t>(r, i, c[i] < 4 ? ve_get<uint32_t>(a, c[i]) : ve_get<uint32_t>(b, c[i] - 4)); return r; }
#define wasm_i32x4_shuffle(a, b, ...) ([&]() { const int c_[4] = {__VA_ARGS__}; return ve_shuf32((a), (b), c_); }())

// used by the SIMD map search (mini/, z145/ vmap.inc)
VE_BIN(wasm_f32x4_div, float, 4, x / y)
static inline v128_t wasm_f32x4_sqrt(v128_t a) { v128_t r; for (int i = 0; i < 4; i++) ve_set<float>(r, i, std::sqrt(ve_get<float>(a, i))); return r; }
static inline v128_t wasm_f32x4_gt(v128_t a, v128_t b) { v128_t r; for (int i = 0; i < 4; i++) ve_set<int32_t>(r, i, ve_get<float>(a, i) > ve_get<float>(b, i) ? -1 : 0); return r; }
