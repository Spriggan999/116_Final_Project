// vmap.inc: relocalization map for vins (included by solution.cpp), the role of VINS-Mono's loop closure.
//
// Map points come from features the window marginalizes (their 3D position is then well estimated) while the
// track is still live: world position, the track's current 7x7 template (for NCC search and LK refinement) and the
// viewing direction / distance at creation. Each frame up to MAP_PF visible map points are searched around their
// predicted projection (NCC on a MAP_STEP grid within +-MAP_R px, then LK). A match is a reprojection factor of the
// fixed map point on the newest frame (noise MAP_SIG_PX, Huber). It lives with that frame: lost if the frame is
// dropped as a non-keyframe, folded into the prior when the frame is marginalized.
#ifndef MAP
#define MAP 1
#endif
#ifndef MAP_N
#define MAP_N 96
#endif
#ifndef MAP_T
#define MAP_T 20.0               // s: map points made freely before this time ...
#endif
#ifndef MAP_VIS_MIN
#define MAP_VIS_MIN 6            // ... and afterwards only where fewer map points than this are in view
#endif
#ifndef MAP_MINOBS
#define MAP_MINOBS 4
#endif
#ifndef MAP_ZMIN
#define MAP_ZMIN 0.4
#endif
#ifndef MAP_ZMAX
#define MAP_ZMAX 8.0
#endif
#ifndef MAP_DEDUP
#define MAP_DEDUP 12.0           // px
#endif
#ifndef MAP_PF
#define MAP_PF 3                 // searches per frame
#endif
#ifndef MAP_OBS_MAX
#define MAP_OBS_MAX 4            // map observations kept per frame
#endif
#ifndef MAP_R
#define MAP_R 12                 // px: search half-window
#endif
#ifndef MAP_STEP
#define MAP_STEP 2
#endif
#ifndef MAP_NCC
#define MAP_NCC 0.85
#endif
#ifndef MAP_SIMD
#define MAP_SIMD 1               // SIMD coarse search (bit-identical to the scalar one)
#endif
#ifndef MAP_LK_MAX
#define MAP_LK_MAX 3.0
#endif
#ifndef MAP_SIG_PX
#define MAP_SIG_PX 2.0
#endif
#ifndef MAP_HUBER
#define MAP_HUBER 1.0
#endif
#ifndef MAP_OUT
#define MAP_OUT 4.0              // px: a map observation with a larger residual after the solve is removed
#endif
#ifndef MAP_MIN_AGE
#define MAP_MIN_AGE 4.0          // s
#endif
#ifndef MAP_RETRY
#define MAP_RETRY 3              // frames
#endif
#ifndef MAP_VIEW_COS
#define MAP_VIEW_COS 0.5
#endif
#ifndef MAP_DRATIO
#define MAP_DRATIO 2.0
#endif
#ifndef MAP_VRUN
#define MAP_VRUN 0.45             // m/s: map search only while the flight's running RMS speed is above this
#endif
#ifndef MAP_CREATE_MAX
#define MAP_CREATE_MAX 6      // most map points made per frame
#endif
#ifndef MAP_MIN_PAR
#define MAP_MIN_PAR 0.0          // rad: >0 makes map points only from features with at least this triangulation angle
#endif
#ifndef MAP_BAD
#define MAP_BAD 3                // a map point removed after this many rejected matches
#endif

static const int MS_ = 2 * PR + 1;
struct MapPt {
    bool used; double p[3], vdir[3], dist, t_made; int last_try, bad;
    float tf[MS_ * MS_], gxf[MS_ * MS_], gyf[MS_ * MS_], Hf[6], tn[MS_ * MS_];
};
static MapPt mp[MAP && MAP_N > 0 ? MAP_N : 1];
static int map_rr = 0;
static long st_map_try = 0, st_map_hit = 0, st_map_made = 0, st_map_out = 0;
struct MapCand { int live; double p[3]; };
static MapCand map_cand[MAX_WF]; static int n_map_cand = 0;

static void cam_of(const Frame& F, M3& Rwc, double* pwc) {
    M3 Ric = mtr(Rci); double pic[3]; mtv(Rci, tci, pic); for (int i = 0; i < 3; i++) pic[i] = -pic[i];
    Rwc = mmul(F.R, Ric); double t[3]; mv(F.R, pic, t);
    for (int i = 0; i < 3; i++) pwc[i] = F.p[i] + t[i];
}
// projection of a world point into frame F; false if behind / outside
static bool map_project(const Frame& F, const double* pw, double& px, double& py, double& z) {
    M3 Rwc; double pc[3]; cam_of(F, Rwc, pc);
    double d[3] = {pw[0] - pc[0], pw[1] - pc[1], pw[2] - pc[2]}, q[3]; mtv(Rwc, d, q);
    if (q[2] < MAP_ZMIN * 0.5) return false;
    z = q[2]; px = fx * q[0] / q[2] + cx; py = fy * q[1] / q[2] + cy;
    return true;
}
static int map_visible_count(const Frame& F) {
    int c = 0;
    for (int i = 0; i < MAP_N; i++) {
        if (!mp[i].used) continue; double x, y, z;
        if (map_project(F, mp[i].p, x, y, z) && inside((float)x, (float)y, 2 * PR + 4)) c++;
    }
    return c;
}
// candidate map points from the features marginalized with the oldest frame (made after the templates are re-cut)
static void map_note_candidate(int w) {
    if (!MAP) return;
    const WF& F = wf[w];
    if (F.live < 0 || !feats[F.live].active || F.nobs < MAP_MINOBS || fb_err[w] > 1.0 || n_map_cand >= MAX_WF) return;
#if EDGE
    if (F.edge) return;
#endif
    double pw[3]; if (!wf_point(F, pw)) return;
    if (MAP_MIN_PAR > 0) {   // the point's depth must come from geometry (triangulation angle), not from the depth prior
        const int ka = fidx(F.ser[0]), kl = fidx(F.ser[F.nobs - 1]); if (ka < 0 || kl < 0) return;
        M3 Ra, Rl; double ca[3], cl[3]; cam_pose(ka, Ra, ca); cam_pose(kl, Rl, cl);
        double a[3], b[3]; for (int i = 0; i < 3; i++) { a[i] = pw[i] - ca[i]; b[i] = pw[i] - cl[i]; }
        const double cs = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / std::sqrt((a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) * (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]));
        if (cs > std::cos(MAP_MIN_PAR)) return;
    }
    MapCand& c = map_cand[n_map_cand++]; c.live = F.live; for (int i = 0; i < 3; i++) c.p[i] = pw[i];
}
static void map_create_at(const Frame& N) {
    if (!MAP) { n_map_cand = 0; return; }
    bool allow = frame_no * 0.05 < MAP_T;
    if (!allow && n_map_cand > 0) allow = map_visible_count(N) < MAP_VIS_MIN;
    if (!allow || n_map_cand == 0) { n_map_cand = 0; return; }
    // projections of the existing map points, once (duplicate test)
    static float qx[MAP && MAP_N > 0 ? MAP_N : 1], qy[MAP && MAP_N > 0 ? MAP_N : 1]; static bool qok[MAP && MAP_N > 0 ? MAP_N : 1];
    {   M3 Rwc; double pc[3]; cam_of(N, Rwc, pc);
        for (int i = 0; i < MAP_N; i++) {
            qok[i] = false; if (!mp[i].used) continue;
            double d[3] = {mp[i].p[0] - pc[0], mp[i].p[1] - pc[1], mp[i].p[2] - pc[2]}, q[3]; mtv(Rwc, d, q);
            if (q[2] < MAP_ZMIN * 0.5) continue;
            qok[i] = true; qx[i] = (float)(fx * q[0] / q[2] + cx); qy[i] = (float)(fy * q[1] / q[2] + cy);
        }
    }
    int made = 0;
    for (int c = 0; c < n_map_cand && made < MAP_CREATE_MAX; c++) {
        const Feat& f = feats[map_cand[c].live]; if (!f.active) continue;
        double x, y, z; if (!map_project(N, map_cand[c].p, x, y, z) || z < MAP_ZMIN || z > MAP_ZMAX) continue;
        if (std::fabs(x - f.x) + std::fabs(y - f.y) > 3.0) continue;   // the point must project onto its own track
        bool dup = false;
        for (int i = 0; i < MAP_N && !dup; i++) if (qok[i] && (qx[i] - f.x) * (qx[i] - f.x) + (qy[i] - f.y) * (qy[i] - f.y) < MAP_DEDUP * MAP_DEDUP) dup = true;
        if (dup) continue;
        int slot = -1; for (int i = 0; i < MAP_N; i++) if (!mp[i].used) { slot = i; break; }
        if (slot < 0) break;
        MapPt& m = mp[slot];
        m.used = true; m.bad = 0; m.last_try = -1000; m.t_made = frame_no * 0.05;
        for (int i = 0; i < 3; i++) m.p[i] = map_cand[c].p[i];
        M3 Rwc; double pc[3]; cam_of(N, Rwc, pc);
        double d[3] = {m.p[0] - pc[0], m.p[1] - pc[1], m.p[2] - pc[2]}; m.dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        for (int i = 0; i < 3; i++) m.vdir[i] = d[i] / m.dist;
        for (int i = 0; i < MS_ * MS_; i++) { m.tf[i] = f.tf[i]; m.gxf[i] = f.gxf[i]; m.gyf[i] = f.gyf[i]; }
        for (int i = 0; i < 6; i++) m.Hf[i] = f.Hf[i];
        double mean = 0; for (int i = 0; i < MS_ * MS_; i++) mean += m.tf[i]; mean /= MS_ * MS_;
        double ss = 0; for (int i = 0; i < MS_ * MS_; i++) { m.tn[i] = (float)(m.tf[i] - mean); ss += m.tn[i] * m.tn[i]; }
        if (ss < 1e-6) { m.used = false; continue; }
        const double is = 1.0 / std::sqrt(ss); for (int i = 0; i < MS_ * MS_; i++) m.tn[i] = (float)(m.tn[i] * is);
        st_map_made++; made++;
        qok[slot] = true; qx[slot] = (float)x; qy[slot] = (float)y;
    }
    n_map_cand = 0;
}
static void map_create() { map_create_at(fr[nfr - 1]); }
// NCC of the normalized template at integer centre (x0, y0)
static float map_ncc(const MapPt& m, int x0, int y0) {
    float st = 0, s1 = 0, s2 = 0;
    for (int j = 0; j < MS_; j++) {
        const uint8_t* p = img + ((y0 - PR + j) * W + (x0 - PR)) * 3 + 1; const float* t = m.tn + j * MS_;
        for (int i = 0; i < MS_; i++) { const float v = p[3 * i]; st += t[i] * v; s1 += v; s2 += v * v; }
    }
    const float var = s2 - s1 * s1 / (MS_ * MS_);
    return var > 1e-3f ? st / std::sqrt(var) : 0.0f;
}
static void map_search() {
    if (!MAP) return;
    Frame& N = fr[nfr - 1]; N.mo_n = 0;
    if (MAP_VRUN > 0 && v2_avg < MAP_VRUN * MAP_VRUN) return;
    const double tnow = frame_no * 0.05;
    int tries = 0;
    for (int kk = 0; kk < MAP_N && tries < MAP_PF && N.mo_n < MAP_OBS_MAX; kk++) {
        const int i = (map_rr + kk) % MAP_N; MapPt& m = mp[i];
        if (!m.used || tnow - m.t_made < MAP_MIN_AGE || frame_no - m.last_try < MAP_RETRY) continue;
        double x, y, z; if (!map_project(N, m.p, x, y, z)) continue;
        if (!inside((float)x, (float)y, (float)(2 * PR + 4 + MAP_R))) continue;
        M3 Rwc; double pc[3]; cam_of(N, Rwc, pc);
        double d[3] = {m.p[0] - pc[0], m.p[1] - pc[1], m.p[2] - pc[2]}; const double dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if ((d[0] * m.vdir[0] + d[1] * m.vdir[1] + d[2] * m.vdir[2]) / dist < MAP_VIEW_COS) continue;
        if (dist > MAP_DRATIO * m.dist || dist * MAP_DRATIO < m.dist) continue;
        // skip points that a live track already follows (no new information)
        bool tracked = false;
        for (int q = 0; q < MAX_FEATS && !tracked; q++) if (feats[q].active && std::fabs(feats[q].x - x) + std::fabs(feats[q].y - y) < 2.0) tracked = true;
        if (tracked) continue;
        tries++; m.last_try = frame_no; st_map_try++;
        const int xc = (int)(x + 0.5), yc = (int)(y + 0.5);
        float best = -2, second = -2; int bx = xc, by = yc;
#if MAP_SIMD && defined(__wasm_simd128__) && MAP_STEP == 2 && PR == 3
        {   // same scores as map_ncc, four dx positions per pass (lanes = dx, dx+2, dx+4, dx+6); every lane sums its 49
            // samples in map_ncc's order, so the results are bit-identical. The green channel of the search region is
            // converted once and split by column parity, which makes the stride-2 lanes contiguous.
            constexpr int RW = 2 * MAP_R + 2 * PR + 1;          // region side (31 for MAP_R 12)
            constexpr int NQ = (2 * MAP_R / MAP_STEP + 1 + 3) / 4;   // groups of 4 positions per row
            constexpr int HW = (2 * MAP_R + 6 * 2 + 2 * PR) / 2 + 8; // parity row length incl. padding lanes
            alignas(16) static float Ev[RW][HW], Od[RW][HW];
            const int rx0 = xc - MAP_R - PR, ry0 = yc - MAP_R - PR;
            for (int r = 0; r < RW; r++) {
                const uint8_t* p = img + ((ry0 + r) * W + rx0) * 3 + 1;
                int c = 0;
                for (; c + 1 < RW; c += 2) { Ev[r][c >> 1] = (float)p[3 * c]; Od[r][c >> 1] = (float)p[3 * c + 3]; }
                if (c < RW) { Ev[r][c >> 1] = (float)p[3 * c]; Od[r][c >> 1] = 0.0f; c += 2; }
                for (int q = c >> 1; q < HW; q++) { Ev[r][q] = 0.0f; Od[r][q] = 0.0f; }
            }
            const v128_t thr = wasm_f32x4_splat(1e-3f), n2 = wasm_f32x4_splat((float)(MS_ * MS_)), zero = wasm_f32x4_splat(0.0f);
            for (int dy = -MAP_R; dy <= MAP_R; dy += MAP_STEP) {
                const int rb = dy + MAP_R;
                for (int g = 0; g < NQ; g++) {
                    const int a = 8 * g;                         // dx + MAP_R of lane 0 (even)
                    v128_t ST = zero, S1 = zero, S2 = zero;
                    for (int j = 0; j < MS_; j++) {
                        const float* er = Ev[rb + j]; const float* orr = Od[rb + j]; const float* t = m.tn + j * MS_;
                        for (int i = 0; i < MS_; i++) {
                            const int col = a + i;
                            const v128_t V = (col & 1) ? wasm_v128_load(orr + (col >> 1)) : wasm_v128_load(er + (col >> 1));
                            ST = wasm_f32x4_add(ST, wasm_f32x4_mul(wasm_f32x4_splat(t[i]), V));
                            S1 = wasm_f32x4_add(S1, V);
                            S2 = wasm_f32x4_add(S2, wasm_f32x4_mul(V, V));
                        }
                    }
                    const v128_t var = wasm_f32x4_sub(S2, wasm_f32x4_div(wasm_f32x4_mul(S1, S1), n2));
                    const v128_t sc = wasm_v128_and(wasm_f32x4_div(ST, wasm_f32x4_sqrt(var)), wasm_f32x4_gt(var, thr));
                    alignas(16) float cs[4]; wasm_v128_store(cs, sc);
                    for (int k = 0; k < 4; k++) {
                        const int dx = a + 2 * k - MAP_R;
                        if (dx > MAP_R) break;
                        const float c = cs[k];
                        if (c > best) { if (std::abs(dx + xc - bx) + std::abs(dy + yc - by) > 2 * MAP_STEP) second = best; best = c; bx = xc + dx; by = yc + dy; }
                        else if (c > second && std::abs(dx + xc - bx) + std::abs(dy + yc - by) > 2 * MAP_STEP) second = c;
                    }
                }
            }
        }
#else
        for (int dy = -MAP_R; dy <= MAP_R; dy += MAP_STEP) for (int dx = -MAP_R; dx <= MAP_R; dx += MAP_STEP) {
            const float c = map_ncc(m, xc + dx, yc + dy);
            if (c > best) { if (std::abs(dx + xc - bx) + std::abs(dy + yc - by) > 2 * MAP_STEP) second = best; best = c; bx = xc + dx; by = yc + dy; }
            else if (c > second && std::abs(dx + xc - bx) + std::abs(dy + yc - by) > 2 * MAP_STEP) second = c;
        }
#endif
        if (MAP_STEP > 1) {
            const int cx0 = bx, cy0 = by;
            for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) { if (!dx && !dy) continue; const float c = map_ncc(m, cx0 + dx, cy0 + dy); if (c > best) { best = c; bx = cx0 + dx; by = cy0 + dy; } }
        }
        if (best < MAP_NCC || best - second < 0.05f) { if (++m.bad >= MAP_BAD) m.used = false; continue; }
        static Feat tmp;
        for (int q = 0; q < MS_ * MS_; q++) { tmp.tf[q] = m.tf[q]; tmp.gxf[q] = m.gxf[q]; tmp.gyf[q] = m.gyf[q]; }
        for (int q = 0; q < 6; q++) tmp.Hf[q] = m.Hf[q];
        float lx = (float)bx, ly = (float)by, off = 0;
        const float r = lk(tmp, lx, ly, off, LK_FINE);
        if (r < 0 || r > MAX_RES || std::fabs(lx - bx) + std::fabs(ly - by) > MAP_LK_MAX) { if (++m.bad >= MAP_BAD) m.used = false; continue; }
        N.mo_id[N.mo_n] = i; N.mo_u[N.mo_n] = (float)((lx - cx) / fx); N.mo_v[N.mo_n] = (float)((ly - cy) / fy); N.mo_n++;
        st_map_hit++;
    }
    map_rr = (map_rr + 1) % MAP_N;
}
// reprojection factors of the map observations of frames k0..k1-1 (lower triangle of H)
static void add_map_factors(int k0, int k1) {
    if (!MAP) return;
    for (int k = k0; k < k1; k++) {
        const Frame& F = fr[k];
        for (int o = 0; o < F.mo_n; o++) {
            const MapPt& m = mp[F.mo_id[o]];
            double d[3] = {m.p[0] - F.p[0], m.p[1] - F.p[1], m.p[2] - F.p[2]}, y[3], pc[3]; mtv(F.R, d, y); mv(Rci, y, pc);
            for (int i = 0; i < 3; i++) pc[i] += tci[i];
            if (pc[2] < MIN_DEPTH) continue;
            const double iz = 1.0 / pc[2], u = pc[0] * iz, v = pc[1] * iz;
            const double e[2] = {fx * (u - F.mo_u[o]), fy * (v - F.mo_v[o])}, en = std::sqrt(e[0] * e[0] + e[1] * e[1]);
            const double wh = en / MAP_SIG_PX <= MAP_HUBER ? 1.0 : MAP_HUBER * MAP_SIG_PX / en, info = wh / (MAP_SIG_PX * MAP_SIG_PX);
            const double Jp[2][3] = {{fx * iz, 0, -fx * u * iz}, {0, fy * iz, -fy * v * iz}};
            double G[2][3], J[2][6];
            for (int r = 0; r < 2; r++) for (int c = 0; c < 3; c++) G[r][c] = Jp[r][0] * Rci.m[c] + Jp[r][1] * Rci.m[3 + c] + Jp[r][2] * Rci.m[6 + c];
            const M3 Sy = skew(y);
            for (int r = 0; r < 2; r++) for (int c = 0; c < 3; c++) {
                J[r][c] = G[r][0] * Sy.m[c] + G[r][1] * Sy.m[3 + c] + G[r][2] * Sy.m[6 + c];
                J[r][3 + c] = -(G[r][0] * F.R.m[3 * c] + G[r][1] * F.R.m[3 * c + 1] + G[r][2] * F.R.m[3 * c + 2]);
            }
            const int o0 = FB(k);
            for (int p = 0; p < 6; p++) {
                gs[o0 + p] += info * (J[0][p] * e[0] + J[1][p] * e[1]);
                if (want_H) for (int q = 0; q <= p; q++) Hs[(o0 + p) * NX + o0 + q] += info * (J[0][p] * J[0][q] + J[1][p] * J[1][q]);
            }
        }
    }
}
// after the solve: drop map observations that do not fit
static void map_check() {
    if (!MAP) return;
    for (int k = 0; k < nfr; k++) {
        Frame& F = fr[k]; int n = 0;
        for (int o = 0; o < F.mo_n; o++) {
            MapPt& m = mp[F.mo_id[o]];
            double x, y, z; bool ok = m.used && map_project(F, m.p, x, y, z);
            if (ok) { const double ex = x - (fx * F.mo_u[o] + cx), ey = y - (fy * F.mo_v[o] + cy); ok = ex * ex + ey * ey < MAP_OUT * MAP_OUT; }
            if (ok) { F.mo_id[n] = F.mo_id[o]; F.mo_u[n] = F.mo_u[o]; F.mo_v[n] = F.mo_v[o]; n++; }
            else { st_map_out++; if (m.used && ++m.bad >= MAP_BAD) m.used = false; }
        }
        F.mo_n = n;
    }
}
