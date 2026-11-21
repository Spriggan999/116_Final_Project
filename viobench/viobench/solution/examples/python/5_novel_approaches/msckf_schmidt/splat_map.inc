// splat_map.inc: a persistent map of 3D Gaussian splats on top of the MSCKF (included by solution.cpp).
//
// A splat = mean (world = first IMU frame), 3x3 covariance, appearance (box-averaged 9x9 signature for NCC search,
// 9x9 LK templates at stride 2 and 1), creation viewing direction and distance.
// Creation: tracks the MSCKF accepted during the first SPL_T_MAP seconds; covariance = triangulation information
// (anchored inverse depth) + the anchor clone's pose covariance. The map is frozen afterwards.
// Use: every frame up to SPL_PF visible splats are projected; the innovation covariance (pose + splat, the EWA
// splatting projection J Sigma J^T) sets the search ellipse; coarse NCC on a 2 px grid, LK refinement, chi^2 gate,
// then two whitened EKF rows on the current attitude and position.
#ifndef SPLAT
#define SPLAT 0
#endif
#ifndef SPL_N
#define SPL_N 96
#endif
#ifndef SPL_T_MAP
#define SPL_T_MAP 20.0          // s: splats are created only before this time
#endif
#ifndef SPL_SMAX
#define SPL_SMAX 0.5            // m: largest allowed position sigma of a new splat
#endif
#ifndef SPL_GROW
#define SPL_GROW 0              // 1: keep creating splats after SPL_T_MAP where the view holds < SPL_VIS_MIN splats
#endif
#ifndef SPL_RECENT
#define SPL_RECENT 40           // frames (SPL_GROW=2)
#endif
#ifndef SPL_HIT_MIN
#define SPL_HIT_MIN 2           // SPL_GROW=2: grow only where fewer matches than this happened in the last SPL_RECENT frames
#endif
#ifndef SPL_VIS_MIN
#define SPL_VIS_MIN 6
#endif
#ifndef SPL_SMAX_LATE
#define SPL_SMAX_LATE SPL_SMAX  // m: sigma cap for splats made after SPL_T_MAP (growing map)
#endif
#ifndef SPL_POSE_K
#define SPL_POSE_K 1.0          // weight of the anchor pose covariance in a new splat's covariance
#endif
#ifndef SPL_ZMIN
#define SPL_ZMIN 0.4
#endif
#ifndef SPL_ZMAX
#define SPL_ZMAX 8.0
#endif
#ifndef SPL_R
#define SPL_R 30                // px: search half-window cap
#endif
#ifndef SPL_NCC
#define SPL_NCC 0.85
#endif
#ifndef SPL_MARGIN
#define SPL_MARGIN 0.05
#endif
#ifndef SPL_PF
#define SPL_PF 1                // searches per frame
#endif
#ifndef SPL_RINF
#define SPL_RINF 4.0            // inflation of the splat covariance in the measurement noise
#endif
#ifndef SPL_SIG_PX
#define SPL_SIG_PX 1.5
#endif
#ifndef SPL_VIEW_COS
#define SPL_VIEW_COS 0.5
#endif
#ifndef SPL_DRATIO
#define SPL_DRATIO 2.0
#endif
#ifndef SPL_GATE
#define SPL_GATE 13.8           // 2-dof chi^2 (99.9%)
#endif
#ifndef SPL_DEDUP
#define SPL_DEDUP 12.0          // px: no new splat this close to an existing one's projection
#endif
#ifndef SPL_RETRY
#define SPL_RETRY 3             // frames before the same splat is searched again
#endif
#ifndef SPL_MIN_AGE
#define SPL_MIN_AGE 4.0         // s: a splat is searched for only once it is this old
#endif
#ifndef SPL_LK_MAX
#define SPL_LK_MAX 3.0          // px: LK may move this far from the NCC peak
#endif
#ifndef SPL_MAX_RES
#define SPL_MAX_RES MAX_RES   // max mean |LK residual| of an accepted match
#endif
#ifndef SPL_FNCC
#define SPL_FNCC 0.0          // >0: minimum NCC of the fine template after LK
#endif
#ifndef SPL_FAST
#define SPL_FAST 1              // coarse-to-fine NCC search
#endif
#ifndef SPL_PEAKS
#define SPL_PEAKS 3
#endif
#ifndef SPL_CONS
#define SPL_CONS 4.0            // px: offsets (match - prediction) that agree
#endif
#ifndef SPL_CONS_MIN
#define SPL_CONS_MIN 1          // matches that must agree before any is used (1: no consensus)
#endif
#ifndef SPL_ORACLE_MAP
#define SPL_ORACLE_MAP 0
#endif
#ifndef SPL_ORACLE_MATCH
#define SPL_ORACLE_MATCH 0
#endif
#ifndef SPL_SOFT
#define SPL_SOFT 0.0            // >0: matches outside the consensus group are used with noise x this
#endif
#ifndef SPL_REFINE
#define SPL_REFINE 0            // 1: refine the splat position with each used match
#endif
#ifndef SPL_REF_POSE
#define SPL_REF_POSE 1.0        // weight of the pose covariance in the refinement noise
#endif
#ifndef SPL_REFRESH
#define SPL_REFRESH 0.0         // >0: new appearance after a used match with NCC >= this
#endif
#ifndef SPL_INFO
#define SPL_INFO 0.0            // >0: search only if pose variance > this x splat variance (in the image)
#endif
#ifndef SPL_HTH
#define SPL_HTH 1               // rows act on attitude
#endif

#ifndef __wasm__
// ---- host-only ground-truth diagnostics (env SPL_GT = path to the feed's gt.f64)
static double* gtv = nullptr; static int gt_n = 0; static bool gt_tried = false;
static long dg_try = 0, dg_inwin = 0, dg_acc = 0, dg_acc2 = 0, dg_valid = 0; static double dg_cerr[4096], dg_perr[4096]; static int dg_nc = 0;
static double dg_merr[8192]; static int dg_nm = 0; static long dg_cons = 0, dg_ncons = 0, dg_peak2 = 0, dg_far = 0, dg_mid = 0;
#ifndef SPL_GT_CONS
#define SPL_GT_CONS 1.0
#endif
static void gt_load() {
    if (gt_tried) return; gt_tried = true;
    const char* p = std::getenv("SPL_GT"); if (!p) return;
    FILE* f = std::fopen(p, "rb"); if (!f) return;
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    gtv = (double*)std::malloc(n); if (std::fread(gtv, 1, n, f) != (size_t)n) { gtv = nullptr; } std::fclose(f); gt_n = (int)(n / 56);
}
static M3 q2R(const double* q) { const double x = q[0], y = q[1], z = q[2], w = q[3];
    M3 R = {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}}; return R; }
static int dg_median_print(const char* name, double* a, int n) {
    for (int i = 1; i < n; i++) { double v = a[i]; int j = i - 1; while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; } a[j + 1] = v; }
    std::fprintf(stderr, "%s n %d median %.3f p80 %.3f  ", name, n, n ? a[n / 2] : -1.0, n ? a[(n * 4) / 5] : -1.0); return 0;
}
#endif

struct Splat {
    double p[3], C[9], vdir[3], dist, t_made;
    float sig[81], smean, snorm;     // NCC signature (box-averaged, stride 2), zero-mean, norm
    Feat tpl;                        // LK templates: tc (stride 2), tf (stride 1)
    int last_try, tries, hits;
    int slot, last_hit;              // Schmidt map: active slot (-1: none), frame of the last match
    double gt_p[3]; bool gt_ok;      // host diagnostics: true point in camera-0 frame
};
static Splat splats[SPLAT ? SPL_N : 1];
static int n_splats = 0;
#ifndef SCH_PREF
#define SCH_PREF 1              // search active (Schmidt) splats first
#endif
// ---- Schmidt map: slots hold splats whose cross-covariance with the state is kept (Xc, Pmm in solution.cpp)
static void sch_release(int k) {   // drop slot k: the last slot moves into it
    const int L = nsch - 1, mc = 3 * nsch;
    splats[sch_splat[k]].slot = -1;
    if (k != L) {
        for (int i = 0; i < n_state; i++) for (int c = 0; c < 3; c++) XC(i, 3 * k + c) = XC(i, 3 * L + c);
        for (int a = 0; a < 3; a++) for (int c = 0; c < mc; c++) PM(3 * k + a, c) = PM(3 * L + a, c);
        for (int r = 0; r < mc; r++) for (int a = 0; a < 3; a++) PM(r, 3 * k + a) = PM(r, 3 * L + a);
        sch_splat[k] = sch_splat[L]; splats[sch_splat[k]].slot = k;
    }
    nsch--;
}
// new splat si triangulated from anchor clone ac: p_f = p_a + R_a q, so d p_f = d p_a - [w]x d th_a (w = p_f - p_a)
static void sch_activate(int si, const double* Ct, int ac) {
    if (nsch >= SCH_N) {   // full: evict the active splat matched longest ago, if old enough
        int best = -1, age = -1;
        for (int j = 0; j < nsch; j++) { const int a = frame_no - splats[sch_splat[j]].last_hit; if (a > age) { age = a; best = j; } }
        if (age < SCH_EVICT) return;
        sch_release(best);
    }
    Splat& s = splats[si];
    const int k = nsch, k3 = 3 * k, b = NI + 6 * ac, n = n_state;
    const double w[3] = {s.p[0] - clones[ac].p[0], s.p[1] - clones[ac].p[1], s.p[2] - clones[ac].p[2]};
    const double Jt[3][3] = {{0, w[2], -w[1]}, {-w[2], 0, w[0]}, {w[1], -w[0], 0}};   // -[w]x
    for (int i = 0; i < n; i++) {
        const double* pr = P + i * NMAX + b;
        for (int c = 0; c < 3; c++) XC(i, k3 + c) = Jt[c][0] * pr[0] + Jt[c][1] * pr[1] + Jt[c][2] * pr[2] + pr[3 + c];
    }
    for (int j = 0; j <= k; j++) for (int a = 0; a < 3; a++) for (int c = 0; c < 3; c++) {
        double v = Jt[a][0] * XC(b, 3 * j + c) + Jt[a][1] * XC(b + 1, 3 * j + c) + Jt[a][2] * XC(b + 2, 3 * j + c) + XC(b + 3 + a, 3 * j + c);
        if (j == k) v += Ct[3 * a + c];
        PM(k3 + a, 3 * j + c) = v; PM(3 * j + c, k3 + a) = v;
    }
    for (int a = 0; a < 3; a++) for (int c = a + 1; c < 3; c++) { const double v = 0.5 * (PM(k3 + a, k3 + c) + PM(k3 + c, k3 + a)); PM(k3 + a, k3 + c) = v; PM(k3 + c, k3 + a) = v; }
    for (int a = 0; a < 3; a++) for (int c = 0; c < 3; c++) s.C[3 * a + c] = PM(k3 + a, k3 + c);
    sch_splat[k] = si; s.slot = k; s.last_hit = frame_no; nsch++;
}
static int cur_fidx = -1;           // index of the live feature behind the track being processed (-1: none)
static long spl_try = 0, spl_hit = 0, spl_gate = 0, spl_late_made = 0, spl_late_hit = 0, spl_late_smax = 0, spl_late_view = 0;

static bool splat_cut(Feat& f) {    // both LK levels at the feature's exact position
    const int S = 2 * PR + 1, E = S + 2;
    if (!inside(f.x, f.y, 4 * PR + 6)) return false;
    float g[(2 * PR + 3) * (2 * PR + 3)];
    for (int lvl = 0; lvl < 2; lvl++) {
        const int st = lvl ? 2 : 1;
        float* t = lvl ? f.tc : f.tf; float* gx = lvl ? f.gxc : f.gxf; float* gy = lvl ? f.gyc : f.gyf;
        grid_sample(f.x - (PR + 1) * st, f.y - (PR + 1) * st, st, E, g);
        double a = 0, b = 0, c = 0, d = 0, e = 0;
        for (int j = 0; j < S; j++) for (int i = 0; i < S; i++) {
            const float* q = g + (j + 1) * E + (i + 1);
            float dx = 0.5f * (q[1] - q[-1]), dy = 0.5f * (q[E] - q[-E]);
            t[j * S + i] = q[0]; gx[j * S + i] = dx; gy[j * S + i] = dy;
            a += dx * dx; b += dx * dy; c -= dx; d += dy * dy; e -= dy;
        }
        if (!inv3s(a, b, c, d, e, (double)(S * S), lvl ? f.Hc : f.Hf)) return false;
    }
    f.ox = f.oy = 0;
    return true;
}

// growing map, mode 2: matches in the last SPL_RECENT frames (the map is useful here if this is high)
static int hit_frames[16]; static int n_hitf = 0;
static int recent_hits() { int c = 0; for (int i = 0; i < 16 && i < n_hitf; i++) if (frame_no - hit_frames[i] <= SPL_RECENT) c++; return c; }
// called from feature_rows after a track was accepted: maybe turn it into a splat
static const Feat* create_obs = nullptr;
static void splat_create(const double* pf, int anchor_clone) {
    const bool late = frame_no * frame_dt > SPL_T_MAP;
    if (!SPLAT || cur_fidx < 0 || (late && !SPL_GROW) || (n_splats >= SPL_N && !SPL_GROW)) return;
    Feat& f = feats[cur_fidx];
    if (!f.active) return;
    // camera now
    M3 Ric = mtr(Rci); double pic[3]; mtv(Rci, tci, pic); for (int i = 0; i < 3; i++) pic[i] = -pic[i];
    double cam[3]; mv(Rwi, pic, cam); for (int i = 0; i < 3; i++) cam[i] += pw[i];
    double d[3] = {pf[0] - cam[0], pf[1] - cam[1], pf[2] - cam[2]};
    const double dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (!(dist > SPL_ZMIN && dist < SPL_ZMAX)) return;
    // covariance: inverse-depth information -> xyz, plus the anchor clone's position and attitude uncertainty
    const double s2 = (SIG_PX / fx) * (SIG_PX / fx), r = tri_rho, ir = 1.0 / r;
    const double Mm[9] = {ir, 0, -tri_al * ir * ir, 0, ir, -tri_be * ir * ir, 0, 0, -ir * ir};
    double J[9];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) J[3 * i + j] = tri_Ra.m[3 * i] * Mm[j] + tri_Ra.m[3 * i + 1] * Mm[3 + j] + tri_Ra.m[3 * i + 2] * Mm[6 + j];
    const double Hi[9] = {tri_hi[0], tri_hi[1], tri_hi[2], tri_hi[1], tri_hi[3], tri_hi[4], tri_hi[2], tri_hi[4], tri_hi[5]};
    double T[9], C[9];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) T[3 * i + j] = J[3 * i] * Hi[j] + J[3 * i + 1] * Hi[3 + j] + J[3 * i + 2] * Hi[6 + j];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) C[3 * i + j] = s2 * (T[3 * i] * J[3 * j] + T[3 * i + 1] * J[3 * j + 1] + T[3 * i + 2] * J[3 * j + 2]);
    double Ctri[9]; for (int i = 0; i < 9; i++) Ctri[i] = C[i];   // triangulation part only (Schmidt map)
    const int b = NI + 6 * anchor_clone;
    const double da = 1.0 / r;   // anchor depth
    double sth = (PP(b, b) + PP(b + 1, b + 1) + PP(b + 2, b + 2)) / 3.0;
    for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) C[3 * i + j] += SPL_POSE_K * PP(b + 3 + i, b + 3 + j); C[4 * i] += SPL_POSE_K * da * da * sth; }
    const double smax = std::sqrt(C[0] > C[4] ? (C[0] > C[8] ? C[0] : C[8]) : (C[4] > C[8] ? C[4] : C[8]));
    if (!(smax < (late ? SPL_SMAX_LATE : SPL_SMAX))) { if (late) spl_late_smax++; return; }
    // not next to an existing splat in the current view; count the splats in view (growing map: only new areas)
    int in_view = 0;
    for (int k = 0; k < n_splats; k++) {
        double e[3] = {splats[k].p[0] - pw[0], splats[k].p[1] - pw[1], splats[k].p[2] - pw[2]}, pi[3], pc[3];
        mtv(Rwi, e, pi); mv(Rci, pi, pc); for (int i = 0; i < 3; i++) pc[i] += tci[i];
        if (pc[2] < 0.1) continue;
        const double u = fx * pc[0] / pc[2] + cx, v = fy * pc[1] / pc[2] + cy;
        const double dx = u - f.x, dy = v - f.y;
        if (dx * dx + dy * dy < SPL_DEDUP * SPL_DEDUP) return;
        if (pc[2] > SPL_ZMIN && pc[2] < SPL_ZMAX && u > 0 && v > 0 && u < W && v < H) in_view++;
    }
    if (late && (SPL_GROW == 2 ? recent_hits() >= SPL_HIT_MIN : in_view >= SPL_VIS_MIN)) { spl_late_view++; return; }
    int slot = n_splats;
    if (slot >= SPL_N) {   // full: recycle the splat seen (tried or made) longest ago
        int best = 1 << 30;
        for (int k = 0; k < n_splats; k++) {
            const int seen = splats[k].last_try > (int)(splats[k].t_made / frame_dt) ? splats[k].last_try : (int)(splats[k].t_made / frame_dt);
            if (seen < best) { best = seen; slot = k; }
        }
        if (frame_no - best < 100) return;   // nothing stale enough (5 s)
    }
    Splat& s = splats[slot];
    s.tpl.x = f.x; s.tpl.y = f.y;
    if (!splat_cut(s.tpl)) return;
    // NCC signature: 2x2 box means on the stride-2 grid (box centre = sample position + 0.5)
    double sum = 0;
    for (int j = 0; j < 9; j++) for (int i = 0; i < 9; i++) {
        const float x = f.x + 2 * (i - 4), y = f.y + 2 * (j - 4);
        const float v = 0.25f * (bil(x, y) + bil(x + 1, y) + bil(x, y + 1) + bil(x + 1, y + 1));
        s.sig[9 * j + i] = v; sum += v;
    }
    const float mean = (float)(sum / 81); double nn = 0;
    for (int i = 0; i < 81; i++) { s.sig[i] -= mean; nn += s.sig[i] * s.sig[i]; }
    if (nn < 81 * 4.0) return;   // flat
    s.snorm = (float)std::sqrt(nn); s.smean = mean;
    for (int i = 0; i < 3; i++) { s.p[i] = pf[i]; s.vdir[i] = d[i] / dist; }
    for (int i = 0; i < 9; i++) s.C[i] = C[i];
    s.dist = dist; s.t_made = frame_no * frame_dt; s.last_try = -1000; s.tries = s.hits = 0;
    s.gt_ok = false;
#ifndef __wasm__
    gt_load();
    if (gtv && create_obs) {   // least-squares ray intersection with true camera poses (all observations + now)
        double A[9] = {0}, bb[3] = {0};
        auto addray = [&](int k, double uu, double vv) {
            const double* g = gtv + 7 * k; M3 R = q2R(g + 3); double dd[3] = {uu, vv, 1}, w[3]; mv(R, dd, w);
            double nn = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); for (int i = 0; i < 3; i++) w[i] /= nn;
            for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { double m = (i == j) - w[i] * w[j]; A[3 * i + j] += m; bb[i] += m * g[j]; } };
        for (int o = 0; o < create_obs->nobs; o++) if (create_obs->obs[o].serial < gt_n) addray(create_obs->obs[o].serial, create_obs->obs[o].u, create_obs->obs[o].v);
        if (frame_no < gt_n) addray(frame_no, (f.x - cx) / fx, (f.y - cy) / fy);
        float Ai[6];
        if (inv3s(A[0], A[1], A[2], A[4], A[5], A[8], Ai)) {
            s.gt_p[0] = Ai[0] * bb[0] + Ai[1] * bb[1] + Ai[2] * bb[2]; s.gt_p[1] = Ai[1] * bb[0] + Ai[3] * bb[1] + Ai[4] * bb[2]; s.gt_p[2] = Ai[2] * bb[0] + Ai[4] * bb[1] + Ai[5] * bb[2];
            s.gt_ok = true;
            {   // self-consistency: the true point should reproject onto the feature now (true pose)
                const double* g0 = gtv + 7 * frame_no; M3 R0 = q2R(g0 + 3);
                double e0[3] = {s.gt_p[0] - g0[0], s.gt_p[1] - g0[1], s.gt_p[2] - g0[2]}, q0[3]; mtv(R0, e0, q0);
                const double ex = fx * q0[0] / q0[2] + cx - f.x, ey = fy * q0[1] / q0[2] + cy - f.y;
                s.gt_ok = q0[2] > 0.1 && ex * ex + ey * ey < SPL_GT_CONS * SPL_GT_CONS;
                dg_ncons++; if (s.gt_ok) dg_cons++;
            }
            if (SPL_ORACLE_MAP && s.gt_ok) {   // true point, in the filter's world (first IMU) frame
                double e[3] = {s.gt_p[0] - tci[0], s.gt_p[1] - tci[1], s.gt_p[2] - tci[2]}; mtv(Rci, e, s.p);
                for (int i = 0; i < 9; i++) s.C[i] = (i % 4 == 0) ? 1e-4 : 0;
            }
            double pe[3]; mv(Rci, s.p, pe); for (int i = 0; i < 3; i++) pe[i] += tci[i];
            double camc0[3]; mv(Rci, cam, camc0); for (int i = 0; i < 3; i++) camc0[i] += tci[i];
            const double* g = gtv + 7 * frame_no;
            if (dg_nc < 4096) { dg_cerr[dg_nc] = std::sqrt((pe[0]-s.gt_p[0])*(pe[0]-s.gt_p[0])+(pe[1]-s.gt_p[1])*(pe[1]-s.gt_p[1])+(pe[2]-s.gt_p[2])*(pe[2]-s.gt_p[2]));
                dg_perr[dg_nc] = std::sqrt((camc0[0]-g[0])*(camc0[0]-g[0])+(camc0[1]-g[1])*(camc0[1]-g[1])+(camc0[2]-g[2])*(camc0[2]-g[2])); dg_nc++; }
        }
    }
#endif
    if (late) spl_late_made++;
    if (SCHMIDT) {
        if (slot < n_splats && s.slot >= 0) sch_release(s.slot);   // recycled an active splat
        s.slot = -1; s.last_hit = frame_no;
        sch_activate(slot, Ctri, anchor_clone);
    }
    if (slot == n_splats) n_splats++;
}

// search one splat in the current image; on success add 2 whitened rows (returns true)
struct SCand { double r0, r1, Rm[3], Hx[2][6]; double dx, dy; bool good; int k; float x, y, ncc; };
static SCand scand[SPL_PF > 0 ? SPL_PF : 1]; static int ncand_s = 0;
static long dg_cons_acc = 0, dg_cons_good = 0;
static bool splat_measure(Splat& s) {
    double d[3] = {s.p[0] - pw[0], s.p[1] - pw[1], s.p[2] - pw[2]}, pi[3], pc[3];
    mtv(Rwi, d, pi); mv(Rci, pi, pc); for (int i = 0; i < 3; i++) pc[i] += tci[i];
    if (pc[2] < SPL_ZMIN) return false;
    const double iz = 1.0 / pc[2], u = pc[0] * iz, v = pc[1] * iz;
    const double xp = fx * u + cx, yp = fy * v + cy;
    const double Jp[2][3] = {{iz, 0, -u * iz}, {0, iz, -v * iz}};
    M3 A = mmul(Rci, mtr(Rwi));                 // d pc / d pf
    M3 Bt = mmul(A, skew(d));                   // d pc / d theta
    double Hf[2][3], Hth[2][3];
    for (int a = 0; a < 2; a++) for (int c = 0; c < 3; c++) {
        double sf = 0, st = 0;
        for (int k = 0; k < 3; k++) { sf += Jp[a][k] * A.m[3 * k + c]; st += Jp[a][k] * Bt.m[3 * k + c]; }
        Hf[a][c] = sf; Hth[a][c] = SPL_HTH ? st : 0;
    }
    // R = Hf C Hf^T * RINF + sig^2 ; S = Hx P Hx^T + R  (Hx = [Hth, -Hf] on [TH, PS])
    double Rm[3] = {0, 0, 0};   // 00, 01, 11
    for (int a = 0; a < 2; a++) for (int b2 = a; b2 < 2; b2++) {
        double sacc = 0;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) sacc += Hf[a][i] * s.C[3 * i + j] * Hf[b2][j];
        Rm[a + b2] = sacc * SPL_RINF;
    }
    const double sp = SPL_SIG_PX / fx; Rm[0] += sp * sp; Rm[2] += sp * sp;
    double Hx[2][6]; int idx[6] = {TH, TH + 1, TH + 2, PS, PS + 1, PS + 2};
    for (int a = 0; a < 2; a++) for (int c = 0; c < 3; c++) { Hx[a][c] = Hth[a][c]; Hx[a][3 + c] = -Hf[a][c]; }
    if (SCHMIDT && s.slot >= 0) {   // joint covariance: Hf Pmm Hf^T * SCH_INF + Hx Pxm Hf^T + its transpose
        const int k3 = 3 * s.slot;
        double XH[6][2];   // Pxm[idx, slot] Hf^T
        for (int i = 0; i < 6; i++) for (int a = 0; a < 2; a++) XH[i][a] = XC(idx[i], k3) * Hf[a][0] + XC(idx[i], k3 + 1) * Hf[a][1] + XC(idx[i], k3 + 2) * Hf[a][2];
        for (int a = 0; a < 2; a++) for (int b2 = a; b2 < 2; b2++) {
            double m = 0, x = 0;
            for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) m += Hf[a][i] * PM(k3 + i, k3 + j) * Hf[b2][j];
            for (int i = 0; i < 6; i++) x += Hx[a][i] * XH[i][b2] + Hx[b2][i] * XH[i][a];
            Rm[a + b2] = m * SCH_INF + x + (a == b2 ? sp * sp : 0);
        }
    }
    double Sm[3] = {Rm[0], Rm[1], Rm[2]};
    for (int a = 0; a < 2; a++) for (int b2 = a; b2 < 2; b2++) {
        double sacc = 0;
        for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) sacc += Hx[a][i] * PP(idx[i], idx[j]) * Hx[b2][j];
        Sm[a + b2] += sacc;
    }
    if (SPL_INFO > 0) {   // only when the map is more informative than the current pose (projected traces)
        const double pose = (Sm[0] + Sm[2]) - (Rm[0] + Rm[2]), map = (Rm[0] + Rm[2]) / SPL_RINF - 2 * (SPL_SIG_PX / fx) * (SPL_SIG_PX / fx) / SPL_RINF;
        if (pose < SPL_INFO * map) return false;
    }
    // search window from the innovation covariance (3 sigma), capped
    int rx = (int)(3 * std::sqrt(Sm[0]) * fx + 2), ry = (int)(3 * std::sqrt(Sm[2]) * fy + 2);
    if (rx > SPL_R) rx = SPL_R; if (ry > SPL_R) ry = SPL_R;
    rx &= ~1; ry &= ~1;
    double xt = -1e9, yt = -1e9;
#ifndef __wasm__
    if (gtv && s.gt_ok && frame_no < gt_n) {   // true projection of the true point with the true pose
        const double* g = gtv + 7 * frame_no; M3 R = q2R(g + 3);
        double e[3] = {s.gt_p[0] - g[0], s.gt_p[1] - g[1], s.gt_p[2] - g[2]}, q[3]; mtv(R, e, q);
        if (q[2] > 0.1) { xt = fx * q[0] / q[2] + cx; yt = fy * q[1] / q[2] + cy; dg_valid++;
            if (std::fabs(xt - xp) <= rx && std::fabs(yt - yp) <= ry) dg_inwin++; }
        dg_try++;
    }
#endif
    const int xi = (int)std::floor(xp), yi = (int)std::floor(yp);
    // box-averaged half-res patch covering candidates x = xi + 2a (|2a| <= rx) and their 9x9 signatures (+-8 px)
    const int X0 = xi - rx - 8, Y0 = yi - ry - 8, nu = rx + 9, nv = ry + 9;   // H[v][u] = box at (X0+2u, Y0+2v)
    if (X0 < 1 || Y0 < 1 || X0 + 2 * nu + 2 >= W || Y0 + 2 * nv + 2 >= H) return false;
    static float Hb[(SPL_R + 10) * (SPL_R + 10)];
    for (int vv = 0; vv < nv; vv++) {
        const uint8_t* r0 = img + ((Y0 + 2 * vv) * W + X0) * 3 + 1; const uint8_t* r1 = r0 + 3 * W;
        for (int uu = 0; uu < nu; uu++) Hb[vv * nu + uu] = 0.25f * ((float)r0[6 * uu] + r0[6 * uu + 3] + r1[6 * uu] + r1[6 * uu + 3]);
    }
    float best = -2, second = -2; int bu = -1, bv = -1;
    const int nca = rx / 2, ncb = ry / 2;
    auto ncc_at = [&](int a, int b2, int step) -> float {      // NCC of the signature (every step-th sample)
        const int u0 = rx / 2 + a, v0 = ry / 2 + b2;
        float sw = 0, sw2 = 0, swt = 0, st = 0, st2 = 0; int n = 0;
        for (int j = 0; j < 9; j += step) {
            const float* hr = Hb + (v0 + j) * nu + u0; const float* tr = s.sig + 9 * j;
            for (int i = 0; i < 9; i += step) { const float w = hr[i], t = tr[i]; sw += w; sw2 += w * w; swt += w * t; st += t; st2 += t * t; n++; }
        }
        const float vw = sw2 - sw * sw / n, vt = st2 - st * st / n;
        return vw > 1.0f && vt > 1.0f ? (swt - sw * st / n) / std::sqrt(vw * vt) : -1.0f;
    };
#if SPL_FAST
    // coarse: 5x5 signature on a 4 px grid; refine the best SPL_PEAKS distinct peaks with the full signature (+-2 px)
    float pk[SPL_PEAKS]; int pa[SPL_PEAKS], pb[SPL_PEAKS]; int np = 0;
    for (int i = 0; i < SPL_PEAKS; i++) pk[i] = -2;
    static float cs[(SPL_R + 1) * (SPL_R + 1)]; static int ca[(SPL_R + 1) * (SPL_R + 1)], cb[(SPL_R + 1) * (SPL_R + 1)]; int nc = 0;
    for (int b2 = -ncb; b2 <= ncb; b2 += 2) for (int a = -nca; a <= nca; a += 2) { cs[nc] = ncc_at(a, b2, 2); ca[nc] = a; cb[nc] = b2; nc++; }
    for (int p = 0; p < SPL_PEAKS; p++) {   // greedy distinct peaks
        int bi = -1; float bs = -2;
        for (int i = 0; i < nc; i++) {
            if (cs[i] <= bs) continue;
            bool near = false; for (int q = 0; q < np; q++) if (std::abs(ca[i] - pa[q]) <= 2 && std::abs(cb[i] - pb[q]) <= 2) near = true;
            if (!near) { bs = cs[i]; bi = i; }
        }
        if (bi < 0) break;
        pa[np] = ca[bi]; pb[np] = cb[bi]; np++;
    }
    for (int p = 0; p < np; p++) {
        float pbest = -2; int ba = pa[p], bb = pb[p];
        for (int db = -1; db <= 1; db++) for (int da = -1; da <= 1; da++) {
            const int a = pa[p] + da, b2 = pb[p] + db;
            if (a < -nca || a > nca || b2 < -ncb || b2 > ncb) continue;
            const float c = ncc_at(a, b2, 1);
            if (c > pbest) { pbest = c; ba = a; bb = b2; }
        }
        if (pbest > best) { second = best; best = pbest; bu = ba; bv = bb; } else if (pbest > second) second = pbest;
    }
    if (SPL_ORACLE_MATCH) { if (!(xt > -1e8 && std::fabs(xt - xp) <= rx && std::fabs(yt - yp) <= ry)) return false; best = 1; second = 0; }
    else if (best < SPL_NCC || best - second < SPL_MARGIN) return false;
#else
    static float sc[(SPL_R + 1) * (SPL_R + 1)];
    for (int b2 = -ncb; b2 <= ncb; b2++) for (int a = -nca; a <= nca; a++) {
        const float ncc = ncc_at(a, b2, 1);
        sc[(b2 + ncb) * (2 * nca + 1) + (a + nca)] = ncc;
        if (ncc > best) { best = ncc; bu = a; bv = b2; }
    }
    if (best < SPL_NCC) return false;
    for (int b2 = -ncb; b2 <= ncb; b2++) for (int a = -nca; a <= nca; a++) {
        if (std::abs(a - bu) <= 1 && std::abs(b2 - bv) <= 1) continue;
        const float c = sc[(b2 + ncb) * (2 * nca + 1) + (a + nca)];
        if (c > second) second = c;
    }
    if (best - second < SPL_MARGIN) return false;
#endif
    // candidate centre: Hb index (rx/2 + bu + 4) -> full-res x = X0 + 2*(rx/2 + bu + 4) = xi + 2*bu
    float x = (float)(xi + 2 * bu), y = (float)(yi + 2 * bv), off = 0;
    if (SPL_ORACLE_MATCH) { x = (float)xt; y = (float)yt; }
    const float x0 = x, y0 = y;
    if (!SPL_ORACLE_MATCH && lk_level(s.tpl, 1, x, y, off, 3) < 0) return false;
    off = 0;
    const float res = SPL_ORACLE_MATCH ? 0 : lk_level(s.tpl, 0, x, y, off, 5);
    if (res < 0 || res > SPL_MAX_RES) return false;
    if ((x - x0) * (x - x0) + (y - y0) * (y - y0) > SPL_LK_MAX * SPL_LK_MAX) return false;
    if (SPL_FNCC > 0 && !SPL_ORACLE_MATCH) {   // NCC of the fine (stride 1) template at the refined position
        const int S = 2 * PR + 1; static float g[(2 * PR + 1) * (2 * PR + 1)];
        if (!inside(x - PR, y - PR, 1) || !inside(x + PR, y + PR, 1)) return false;
        grid_sample(x - PR, y - PR, 1, S, g);
        double ma = 0, mb = 0; for (int i = 0; i < S * S; i++) { ma += g[i]; mb += s.tpl.tf[i]; } ma /= S * S; mb /= S * S;
        double sab = 0, saa = 0, sbb = 0;
        for (int i = 0; i < S * S; i++) { const double a = g[i] - ma, b = s.tpl.tf[i] - mb; sab += a * b; saa += a * a; sbb += b * b; }
        if (!(saa > 0 && sbb > 0) || sab / std::sqrt(saa * sbb) < SPL_FNCC) return false;
    }
    // innovation and gate
    const double r0 = (x - cx) / fx - u, r1 = (y - cy) / fy - v;
    const double det = Sm[0] * Sm[2] - Sm[1] * Sm[1];
    if (!(det > 0)) return false;
    const double nis = (Sm[2] * r0 * r0 - 2 * Sm[1] * r0 * r1 + Sm[0] * r1 * r1) / det;
    if (nis > SPL_GATE) { spl_gate++; return false; }
#ifndef __wasm__
    if (xt > -1e8) { const double me = std::sqrt((x - xt) * (x - xt) + (y - yt) * (y - yt)); dg_acc++; if (me < 2) dg_acc2++; if (dg_nm < 8192) dg_merr[dg_nm++] = me;
        const double pe = std::sqrt((x0 - xt) * (x0 - xt) + (y0 - yt) * (y0 - yt)); if (pe < 2) dg_peak2++; if (me > 5) dg_far++; if (me >= 2 && me <= 5) dg_mid++;
        if (std::getenv("SPL_DUMP")) std::fprintf(stderr, "M f %d ncc %.3f sec %.3f peakerr %.2f lkerr %.2f dist %.2f/%.2f win %d %d\n", frame_no, best, second, pe, me, std::sqrt((s.p[0]-pw[0])*(s.p[0]-pw[0])+(s.p[1]-pw[1])*(s.p[1]-pw[1])+(s.p[2]-pw[2])*(s.p[2]-pw[2])), s.dist, rx, ry); }
#endif
    if (ncand_s >= SPL_PF) return false;
    SCand& c = scand[ncand_s++];
    c.k = (int)(&s - splats); c.x = x; c.y = y; c.ncc = best;
    c.r0 = r0; c.r1 = r1; c.dx = x - xp; c.dy = y - yp; c.good = xt > -1e8 && std::sqrt((x - xt) * (x - xt) + (y - yt) * (y - yt)) < 2;
    for (int i = 0; i < 3; i++) c.Rm[i] = Rm[i];
    for (int aa = 0; aa < 2; aa++) for (int cc = 0; cc < 6; cc++) c.Hx[aa][cc] = Hx[aa][cc];
    return true;
}
// Schmidt update with a match of an active splat: state and cross-covariances move, the map does not
static void schmidt_update(Splat& s, float x, float y, double inf) {
    double d[3] = {s.p[0] - pw[0], s.p[1] - pw[1], s.p[2] - pw[2]}, pi[3], pc[3];
    mtv(Rwi, d, pi); mv(Rci, pi, pc); for (int i = 0; i < 3; i++) pc[i] += tci[i];
    if (pc[2] < SPL_ZMIN) return;
    const double iz = 1.0 / pc[2], u = pc[0] * iz, v = pc[1] * iz;
    const double Jp[2][3] = {{iz, 0, -u * iz}, {0, iz, -v * iz}};
    M3 A = mmul(Rci, mtr(Rwi)), Bt = mmul(A, skew(d));
    double Hf[2][3], Hx[2][6]; const int idx[6] = {TH, TH + 1, TH + 2, PS, PS + 1, PS + 2};
    for (int a = 0; a < 2; a++) for (int c = 0; c < 3; c++) {
        double sf = 0, st = 0; for (int k = 0; k < 3; k++) { sf += Jp[a][k] * A.m[3 * k + c]; st += Jp[a][k] * Bt.m[3 * k + c]; }
        Hf[a][c] = sf; Hx[a][c] = SPL_HTH ? st : 0; Hx[a][3 + c] = -sf;
    }
    const int n = n_state, k3 = 3 * s.slot, mc = 3 * nsch;
#ifndef __wasm__
    if (std::getenv("SCH_CHECK")) {   // joint [pose; map point] covariance must be positive definite
        const int id9[6] = {TH, TH + 1, TH + 2, PS, PS + 1, PS + 2}; double Jm[81];
        for (int i = 0; i < 9; i++) for (int j = 0; j < 9; j++)
            Jm[9 * i + j] = i < 6 && j < 6 ? PP(id9[i], id9[j]) : i < 6 ? XC(id9[i], k3 + j - 6) : j < 6 ? XC(id9[j], k3 + i - 6) : PM(k3 + i - 6, k3 + j - 6);
        bool ok = true; double mind = 1e9;
        for (int j = 0; j < 9 && ok; j++) {
            double d = Jm[9 * j + j]; for (int k = 0; k < j; k++) d -= Jm[9 * j + k] * Jm[9 * j + k];
            if (d <= 0) { ok = false; break; } d = std::sqrt(d); Jm[9 * j + j] = d; if (j >= 6 && d * d < mind) mind = d * d;
            for (int i = j + 1; i < 9; i++) { double v = Jm[9 * i + j]; for (int k = 0; k < j; k++) v -= Jm[9 * i + k] * Jm[9 * j + k]; Jm[9 * i + j] = v / d; }
        }
        std::fprintf(stderr, "SCHK f %d slot %d ok %d condvar_map %.3g pmm %.3g %.3g %.3g\n", frame_no, s.slot, ok, mind, PM(k3, k3), PM(k3 + 1, k3 + 1), PM(k3 + 2, k3 + 2));
    }
#endif
    static double PHt[NMAX][2], M[2][3 * (SCHMIDT ? SCH_N : 1)], dx[NMAX];
    for (int i = 0; i < n; i++) {   // joint P H^T, state rows
        const double* pr = P + i * NMAX; const sch_t* xr = &XC(i, k3);
        for (int a = 0; a < 2; a++) {
            double acc = Hf[a][0] * xr[0] + Hf[a][1] * xr[1] + Hf[a][2] * xr[2];
            for (int t = 0; t < 6; t++) acc += pr[idx[t]] * Hx[a][t];
            PHt[i][a] = acc;
        }
    }
    for (int a = 0; a < 2; a++) for (int col = 0; col < mc; col++) {   // H [Pxm; Pmm] (map columns)
        double acc = Hf[a][0] * PM(k3, col) + Hf[a][1] * PM(k3 + 1, col) + Hf[a][2] * PM(k3 + 2, col);
        for (int t = 0; t < 6; t++) acc += Hx[a][t] * XC(idx[t], col);
        M[a][col] = acc;
    }
    const double sp = SPL_SIG_PX / fx;
    double Sm[3];
    for (int a = 0; a < 2; a++) for (int b2 = a; b2 < 2; b2++) {
        double hph = 0, m = 0, xm = 0;
        for (int t = 0; t < 6; t++) hph += Hx[a][t] * PHt[idx[t]][b2];   // Hx Pxx Hx^T + Hx Pxm Hf^T
        for (int c = 0; c < 3; c++) xm += Hf[a][c] * (M[b2][k3 + c] - Hf[b2][0] * PM(k3, k3 + c) - Hf[b2][1] * PM(k3 + 1, k3 + c) - Hf[b2][2] * PM(k3 + 2, k3 + c));   // Hf Pmx Hx^T
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) m += Hf[a][i] * PM(k3 + i, k3 + j) * Hf[b2][j];
        Sm[a + b2] = hph + xm + inf * (m * SCH_INF + (a == b2 ? sp * sp : 0));
    }
    const double det = Sm[0] * Sm[2] - Sm[1] * Sm[1]; if (!(det > 0)) return;
    const double Si[3] = {Sm[2] / det, -Sm[1] / det, Sm[0] / det};
    const double r0 = (x - cx) / fx - u, r1 = (y - cy) / fy - v;
    if ((Si[0] * r0 * r0 + 2 * Si[1] * r0 * r1 + Si[2] * r1 * r1) > SPL_GATE) { spl_gate++; return; }
    static double K[NMAX][2];
    for (int i = 0; i < n; i++) {
        K[i][0] = PHt[i][0] * Si[0] + PHt[i][1] * Si[1]; K[i][1] = PHt[i][0] * Si[1] + PHt[i][1] * Si[2];
        dx[i] = K[i][0] * r0 + K[i][1] * r1;
    }
    for (int i = 0; i < n; i++) {
        const double k0 = K[i][0], k1 = K[i][1]; double* pr = P + i * NMAX;
        for (int j = i; j < n; j++) pr[j] -= k0 * PHt[j][0] + k1 * PHt[j][1];
        sch_t* xr = &XC(i, 0);
        for (int c = 0; c < mc; c++) xr[c] -= k0 * M[0][c] + k1 * M[1][c];
    }
    for (int i = 0; i < n; i++) for (int j = 0; j < i; j++) PP(i, j) = PP(j, i);
    for (int i = 0; i < n; i++) if (PP(i, i) < 1e-12) PP(i, i) = 1e-12;
    inject(dx);
    s.last_hit = frame_no; st_sch_upd++;
}
static void splat_rows(const SCand& c, double inf = 1.0) {   // whiten with R = L L^T: rows L^-1 [H r]
    const int idx[6] = {TH, TH + 1, TH + 2, PS, PS + 1, PS + 2};
    const double l00 = std::sqrt(c.Rm[0] * inf), l10 = c.Rm[1] * inf / l00, l11 = std::sqrt(c.Rm[2] * inf - l10 * l10);
    if (!(l11 > 0) || nrows + 2 > MAX_ROWS) return;
    for (int a = 0; a < 2; a++) {
        double h[6], rr;
        for (int k = 0; k < 6; k++) h[k] = a == 0 ? c.Hx[0][k] / l00 : (c.Hx[1][k] - l10 * c.Hx[0][k] / l00) / l11;
        rr = a == 0 ? c.r0 / l00 : (c.r1 - l10 * c.r0 / l00) / l11;
        rnnz[nrows] = 6; for (int k = 0; k < 6; k++) { ridx[nrows][k] = idx[k]; rval[nrows][k] = h[k]; }
        rs[nrows] = rr; Rd[nrows] = 1.0; nrows++;
    }
}

// per-splat Kalman update with a match, after the pose update (the pose uncertainty is part of the noise)
static void splat_refine(Splat& s, float x, float y) {
    double d[3] = {s.p[0] - pw[0], s.p[1] - pw[1], s.p[2] - pw[2]}, pi[3], pc[3];
    mtv(Rwi, d, pi); mv(Rci, pi, pc); for (int i = 0; i < 3; i++) pc[i] += tci[i];
    if (pc[2] < SPL_ZMIN) return;
    const double iz = 1.0 / pc[2], u = pc[0] * iz, v = pc[1] * iz;
    const double Jp[2][3] = {{iz, 0, -u * iz}, {0, iz, -v * iz}};
    M3 A = mmul(Rci, mtr(Rwi)), Bt = mmul(A, skew(d));
    double Hf[2][3], Hx[2][6]; const int idx[6] = {TH, TH + 1, TH + 2, PS, PS + 1, PS + 2};
    for (int a = 0; a < 2; a++) for (int c = 0; c < 3; c++) {
        double sf = 0, st = 0; for (int k = 0; k < 3; k++) { sf += Jp[a][k] * A.m[3 * k + c]; st += Jp[a][k] * Bt.m[3 * k + c]; }
        Hf[a][c] = sf; Hx[a][c] = st; Hx[a][3 + c] = -sf;
    }
    double S[3] = {0, 0, 0};
    const double sp = SPL_SIG_PX / fx;
    for (int a = 0; a < 2; a++) for (int b = a; b < 2; b++) {
        double acc = 0;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) acc += Hf[a][i] * s.C[3 * i + j] * Hf[b][j];
        for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) acc += SPL_REF_POSE * Hx[a][i] * PP(idx[i], idx[j]) * Hx[b][j];
        S[a + b] = acc + (a == b ? sp * sp : 0);
    }
    const double det = S[0] * S[2] - S[1] * S[1]; if (!(det > 0)) return;
    const double Si[3] = {S[2] / det, -S[1] / det, S[0] / det};
    double CH[3][2];   // C Hf^T
    for (int i = 0; i < 3; i++) for (int a = 0; a < 2; a++) CH[i][a] = s.C[3 * i] * Hf[a][0] + s.C[3 * i + 1] * Hf[a][1] + s.C[3 * i + 2] * Hf[a][2];
    double K[3][2];
    for (int i = 0; i < 3; i++) { K[i][0] = CH[i][0] * Si[0] + CH[i][1] * Si[1]; K[i][1] = CH[i][0] * Si[1] + CH[i][1] * Si[2]; }
    const double r0 = (x - cx) / fx - u, r1 = (y - cy) / fy - v;
    for (int i = 0; i < 3; i++) s.p[i] += K[i][0] * r0 + K[i][1] * r1;
    double Cn[9];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Cn[3 * i + j] = s.C[3 * i + j] - (K[i][0] * CH[j][0] + K[i][1] * CH[j][1]);
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) s.C[3 * i + j] = 0.5 * (Cn[3 * i + j] + Cn[3 * j + i]);
}
// new appearance from the current view after a confident match
static void splat_refresh(Splat& s, float x, float y) {
    Feat t; t.x = x; t.y = y;
    if (!splat_cut(t)) return;
    float sig[81]; double sum = 0;
    for (int j = 0; j < 9; j++) for (int i = 0; i < 9; i++) {
        const float xx = x + 2 * (i - 4), yy = y + 2 * (j - 4);
        if (!inside(xx, yy, 1) || !inside(xx + 1, yy + 1, 1)) return;
        const float v = 0.25f * (bil(xx, yy) + bil(xx + 1, yy) + bil(xx, yy + 1) + bil(xx + 1, yy + 1));
        sig[9 * j + i] = v; sum += v;
    }
    const float mean = (float)(sum / 81); double nn = 0;
    for (int i = 0; i < 81; i++) { sig[i] -= mean; nn += sig[i] * sig[i]; }
    if (nn < 81 * 4.0) return;
    for (int i = 0; i < 81; i++) s.sig[i] = sig[i];
    s.snorm = (float)std::sqrt(nn); s.tpl = t;
    M3 Ric = mtr(Rci); double pic[3]; mtv(Rci, tci, pic); for (int i = 0; i < 3; i++) pic[i] = -pic[i];
    double cam[3]; mv(Rwi, pic, cam); for (int i = 0; i < 3; i++) cam[i] += pw[i];
    double d[3] = {s.p[0] - cam[0], s.p[1] - cam[1], s.p[2] - cam[2]};
    const double dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (dist > 0.1) { s.dist = dist; for (int i = 0; i < 3; i++) s.vdir[i] = d[i] / dist; }
}

static void splat_step() {
    if (!SPLAT || n_splats == 0) return;
    const double now = frame_no * frame_dt;
    M3 Ric = mtr(Rci); double pic[3]; mtv(Rci, tci, pic); for (int i = 0; i < 3; i++) pic[i] = -pic[i];
    double cam[3]; mv(Rwi, pic, cam); for (int i = 0; i < 3; i++) cam[i] += pw[i];
    nrows = 0;
    for (int n = 0; n < SPL_PF; n++) {
        int pick = -1, oldest = 1 << 30;
        for (int k = 0; k < n_splats; k++) {
            Splat& s = splats[k];
            if (frame_no - s.last_try < SPL_RETRY || now - s.t_made < SPL_MIN_AGE) continue;
            double d[3] = {s.p[0] - cam[0], s.p[1] - cam[1], s.p[2] - cam[2]};
            const double dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            if (dist < SPL_ZMIN || dist > s.dist * SPL_DRATIO || dist < s.dist / SPL_DRATIO) continue;
            if ((d[0] * s.vdir[0] + d[1] * s.vdir[1] + d[2] * s.vdir[2]) / dist < SPL_VIEW_COS) continue;
            double e[3] = {s.p[0] - pw[0], s.p[1] - pw[1], s.p[2] - pw[2]}, pi[3], pc[3];
            mtv(Rwi, e, pi); mv(Rci, pi, pc); for (int i = 0; i < 3; i++) pc[i] += tci[i];
            if (pc[2] < SPL_ZMIN) continue;
            const double x = fx * pc[0] / pc[2] + cx, y = fy * pc[1] / pc[2] + cy;
            if (x < 30 || y < 30 || x > W - 31 || y > H - 31) continue;
            const int key = s.last_try - ((SCHMIDT && SCH_PREF && s.slot >= 0) ? (1 << 20) : 0);
            if (key < oldest) { oldest = key; pick = k; }
        }
        if (pick < 0) break;
        Splat& s = splats[pick];
        s.last_try = frame_no; s.tries++; spl_try++;
        if (splat_measure(s)) { s.hits++; spl_hit++; if (s.t_made > SPL_T_MAP) spl_late_hit++; hit_frames[n_hitf++ % 16] = frame_no; }
    }
    // consensus: the pose error shifts all projections alike; keep the largest group of agreeing offsets
    int bi = -1, bc = 0;
    for (int i = 0; i < ncand_s; i++) {
        int c = 0;
        for (int j = 0; j < ncand_s; j++) if (std::fabs(scand[i].dx - scand[j].dx) <= SPL_CONS && std::fabs(scand[i].dy - scand[j].dy) <= SPL_CONS) c++;
        if (c > bc) { bc = c; bi = i; }
    }
    static bool used[SPL_PF > 0 ? SPL_PF : 1];
    for (int j = 0; j < ncand_s; j++) {
        const bool agree = bi >= 0 && bc >= SPL_CONS_MIN && std::fabs(scand[bi].dx - scand[j].dx) <= SPL_CONS && std::fabs(scand[bi].dy - scand[j].dy) <= SPL_CONS;
        used[j] = agree || SPL_SOFT > 0;
        if (!used[j]) continue;
        dg_cons_acc++; dg_cons_good += scand[j].good;
        if (SCHMIDT && splats[scand[j].k].slot >= 0) continue;   // Schmidt update below
        splat_rows(scand[j], agree ? 1.0 : SPL_SOFT);
    }
    if (nrows > 0) ekf_update();
    if (SCHMIDT) for (int j = 0; j < ncand_s; j++) {
        if (!used[j] || splats[scand[j].k].slot < 0) continue;
        const bool agree = bi >= 0 && bc >= SPL_CONS_MIN && std::fabs(scand[bi].dx - scand[j].dx) <= SPL_CONS && std::fabs(scand[bi].dy - scand[j].dy) <= SPL_CONS;
        schmidt_update(splats[scand[j].k], scand[j].x, scand[j].y, agree ? 1.0 : SPL_SOFT);
    }
    for (int j = 0; j < ncand_s; j++) {
        if (!used[j]) continue;
        Splat& sp = splats[scand[j].k];
        if (SPL_REFINE && !(SCHMIDT && sp.slot >= 0)) splat_refine(sp, scand[j].x, scand[j].y);
        if (SPL_REFRESH && scand[j].ncc >= SPL_REFRESH) splat_refresh(sp, scand[j].x, scand[j].y);
    }
    ncand_s = 0;
}
