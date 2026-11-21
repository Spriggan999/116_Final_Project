// ================= Gaussian-splat map (SPLAT=1) =================
// A persistent map of 3D Gaussians ("splats") in the world frame: mean X, covariance S (3x3), an 8x8 green
// appearance template, hit / miss counts. Splats are born from tracked corners (two-ray triangulation between the
// feature's first keyframe and now), projected into each frame with the EWA splatting Jacobian (2D mean + 2D
// covariance J (S + pose uncertainty) J^T), found again with Lucas-Kanade optical flow started at the projected mean
// (several starts inside the 2D ellipse when it is wide) and gated by the Mahalanobis distance. Every match is an
// EKF reprojection pair of rows on position and attitude (splat covariance folded into the noise), then the splat
// itself is refined with a per-splat Kalman update. Old splats seen again tie the pose to the map built earlier:
// that is what bounds drift; a live tracked feature feeds its own splat at no extra tracking cost.
#ifndef SPL_BIG
#define SPL_BIG 0           // 1: coarse relocalization uses a 16x16 context template (4x4 samples, 4 px apart)
#endif
#ifndef SPL_KEEP_ACC
#define SPL_KEEP_ACC 0      // 1: a full map replaces its least accurate splat (largest covariance)
#endif
#if SPLAT == 2
#define SPL_N SPL_K         // every splat is a filter state
#endif
#ifndef SPL_N
#define SPL_N 24            // map size
#endif
#ifndef SPL_SIG_PX
#define SPL_SIG_PX 1.0      // bearing noise of a measurement (tracking px)
#endif
#ifndef SPL_MIN_ANG
#define SPL_MIN_ANG 0.05    // rad: minimum ray angle for a new splat
#endif
#ifndef SPL_ZMIN
#define SPL_ZMIN 0.3
#endif
#ifndef SPL_ZMAX
#define SPL_ZMAX 8.0
#endif
#ifndef SPL_REL
#define SPL_REL 0.15        // relative depth error from the filter's own baseline (scale) error
#endif
#ifndef SPL_RINF
#define SPL_RINF 2.0        // measurement noise inflation (the splat error is not white across observations)
#endif
#ifndef SPL_REACQ
#define SPL_REACQ 4         // re-acquisition attempts (unlinked splats) per 6-frame cycle
#endif
#ifndef SPL_RMAX
#define SPL_RMAX 8.0        // px: widest re-acquisition search radius
#endif
#ifndef SPL_MAXMISS
#define SPL_MAXMISS 4
#endif
#ifndef SPL_MIN_AGE
#define SPL_MIN_AGE 1.0     // s: a splat constrains the pose only once it is this old (long baseline)
#endif
#ifndef SPL_LINKED
#define SPL_LINKED 1        // 1: live tracked features measure their splat every cycle
#endif
#ifndef SPL_GATE2
#define SPL_GATE2 9.21      // 2D chi^2 gate of a re-acquired match
#endif
#ifndef SPL_DRATIO
#define SPL_DRATIO 1.35     // max viewing-distance ratio (template scale change) for re-acquisition
#endif
#ifndef SPL_VIEW_COS
#define SPL_VIEW_COS 0.9    // min cosine between the creation and current viewing directions
#endif
#ifndef SPL_SMAX
#define SPL_SMAX 0.5        // m: splats with a larger position sigma are not used for the pose
#endif
#ifndef SPL_POSE_COV
#define SPL_POSE_COV 0      // 1: new splat covariance includes the current pose uncertainty
#endif
#ifndef SPL_OWN
#define SPL_OWN 1           // 1: linked splats are measured with their own (creation) template, started at the feature
#endif
#ifndef SPL_NIS_KILL
#define SPL_NIS_KILL 16.0   // 2D NIS of a refinement above this counts as a miss (and is not used)
#endif
#ifndef SPL_DMIN_USE
#define SPL_DMIN_USE 0.4    // m: closer splats are not used for the pose
#endif
#ifndef SPL_DEPTH_GAIN
#define SPL_DEPTH_GAIN 1.0  // new splat depth x this (undo the filter's local under-scale)
#endif
#ifndef SPL_HTH
#define SPL_HTH 1           // rows act on attitude
#endif
#ifndef SPL_HP
#define SPL_HP 1            // rows act on position
#endif
#ifndef SPL_DISP_FIX
#define SPL_DISP_FIX 1
#endif
#ifndef SPL_PH
#define SPL_PH 3            // cycle phase of the re-acquisition step
#endif

struct Splat {
    int alive, link, gen, hits, miss;
    Vec3 X; double S[9];
    Vec3 view; double dist;     // viewing direction (camera -> X) and distance at creation / last template
    double t0, ptr0;            // creation time, pose-covariance trace at creation
    int uid;
    float tm, ts, tm4, ts4;     // template statistics (relocalization)
    int act;                    // frames left of continued tracking after a relocalization
#if SPL_BIG
    float big[16 * 16];         // 16x16 context template around the splat (coarse relocalization signature)
#endif
    double ax, ay; Mat3 aR;     // last tracked top-left (tracking px) and body attitude then
#ifdef SPL_ORACLE_MATCH
    Vec3 Xg;
#endif
    Patch tp;
};
Splat spl[SPL_N];
void tmpl_stats(Splat& s);
int feat_gen[N_TRACK];
int fa_set[N_TRACK];
Vec3 fa_c[N_TRACK], fa_w[N_TRACK];
double fa_t[N_TRACK], fa_bx[N_TRACK], fa_by[N_TRACK];
int spl_uid = 0;
#ifdef SPL_GTLOG
FILE* gtlog = nullptr;
#define GTL(...) do { if (!gtlog) { gtlog = std::fopen(std::getenv("SPL_GTLOG_FILE") ? std::getenv("SPL_GTLOG_FILE") : "/dev/null", "w"); std::fprintf(gtlog, "I"); for (int q = 0; q < 9; q++) std::fprintf(gtlog, " %.10g", R_wc0.m[q]); std::fprintf(gtlog, " %.10g %.10g %.10g\n", p_wc0.x, p_wc0.y, p_wc0.z); } std::fprintf(gtlog, __VA_ARGS__); } while (0)
#else
#define GTL(...) do {} while (0)
#endif
int feat_spl[N_TRACK];          // splat linked to each feature slot (-1: none)
int spl_diag_rows = 0, spl_diag_reacq = 0, spl_diag_n = 0;
int sd[16] = {0};
int gd[8] = {0}, gr[3] = {0}, go[6] = {0};
#define SD(i) (sd[i]++)

inline Vec3 cam_center() { return add(p_w, mul(R_wb, r_cam_in_imu)); }
inline Vec3 vnorm(Vec3 a) { const double n = norm(a); return n > 0 ? scale(a, 1.0 / n) : a; }

#if SPLAT == 2
// remove splat k's block from the filter (zero rows / columns)
void splat_free_state(int k) {
    const int b = IXL + 3 * k;
    for (int i = 0; i < NS; i++)
        for (int j = 0; j < 3; j++) { P[i * NS + b + j] = 0; P[(b + j) * NS + i] = 0; }
    dx[b] = dx[b + 1] = dx[b + 2] = 0;
}
void splat_inject() {
    for (int k = 0; k < SPL_K; k++)
        if (spl[k].alive) spl[k].X = add(spl[k].X, Vec3{dx[IXL + 3 * k], dx[IXL + 3 * k + 1], dx[IXL + 3 * k + 2]});
}
#endif
#ifdef SPL_DEBUG
int spl_diag_created = 0;
void spl_report() { std::fprintf(stderr, "splat rows %d reacq %d created %d |", spl_diag_rows, spl_diag_reacq, spl_diag_created); for (int i = 0; i < 16; i++) std::fprintf(stderr, " %d", sd[i]); std::fprintf(stderr, "\n"); }
#endif
void splat_reset() {
#ifdef SPL_DEBUG
    static bool reg = false; if (!reg) { reg = true; std::atexit(spl_report); }
#endif
    for (int i = 0; i < SPL_N; i++) spl[i].alive = 0;
    for (int i = 0; i < N_TRACK; i++) { feat_gen[i] = 0; fa_set[i] = 0; feat_spl[i] = -1; }
#if SPLAT == 2
    for (int k = 0; k < SPL_K; k++) splat_free_state(k);
#endif
}
// a feature slot gets a new feature
void splat_on_birth(int i) {
    feat_gen[i]++; fa_set[i] = 0;
    if (feat_spl[i] >= 0) { spl[feat_spl[i]].link = -1; feat_spl[i] = -1; }
}
// the feature was keyed (template cut) at tracking px (cx, cy) of the current frame: first observation
void splat_on_key(int i, double cxp, double cyp) {
    if (fa_set[i]) return;
    Vec3 b{(cxp - cx_s) / fx_s, (cyp - cy_s) / fy_s, 1.0};
    fa_w[i] = vnorm(mul(mul(R_wb, R_ic), b));
    fa_t[i] = last_t; fa_bx[i] = b.x; fa_by[i] = b.y;
    fa_c[i] = cam_center();
    fa_set[i] = 1;
}

// 3x3 helpers on row-major double[9]
inline double qform(const double S[9], Vec3 t) {
    const double a[3] = {t.x, t.y, t.z};
    double s = 0;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) s += a[i] * S[3 * i + j] * a[j];
    return s;
}
inline Vec3 mulS(const double S[9], Vec3 t) {
    return Vec3{S[0] * t.x + S[1] * t.y + S[2] * t.z, S[3] * t.x + S[4] * t.y + S[5] * t.z, S[6] * t.x + S[7] * t.y + S[8] * t.z};
}
// pose part of a point's uncertainty: P_pp + [Y]x P_thth [Y]x^T  (Y = X - p)
double pose_trace() {
    double s = 0;
    for (int i = 0; i < 3; i++) s += (double)P[(IP + i) * NS + IP + i] + (double)P[(ITH + i) * NS + ITH + i];
    return s;
}
void pose_cov(Vec3 Y, double C[9]) {
    double Pp[9], Pt[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            const int a = i < j ? i : j, b = i < j ? j : i;   // upper triangle is current under SYM_P
            Pp[3 * i + j] = (double)P[(IP + a) * NS + IP + b];
            Pt[3 * i + j] = (double)P[(ITH + a) * NS + ITH + b];
        }
    const double y[3] = {Y.x, Y.y, Y.z};
    // K = [Y]x: K v = Y x v
    const double K[9] = {0, -y[2], y[1], y[2], 0, -y[0], -y[1], y[0], 0};
    double KP[9];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { double s = 0; for (int k = 0; k < 3; k++) s += K[3 * i + k] * Pt[3 * k + j]; KP[3 * i + j] = s; }
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { double s = 0; for (int k = 0; k < 3; k++) s += KP[3 * i + k] * K[3 * j + k]; C[3 * i + j] = Pp[3 * i + j] + s; }
}

#ifdef SPL_GTCHK
// debug: ground truth from env SPL_GT (feed dir): does a splat correction move p / v toward the truth?
int gt_frame = -1; std::vector<double> gt_pose, gt_ft;
double gc_n = 0, gc_p_before = 0, gc_p_after = 0, gc_v_before = 0, gc_v_after = 0;
void gtchk_report() {
    std::fprintf(stderr, "reloc: searches %d gt-visible %d gt-in-window %d | accepted %d: correct(<2px) %d near(<5) %d wrong %d | accepted with gt in window %d\n", gd[0], gd[1], gd[2], gd[3], gd[4], gd[5], gd[6], gd[7]);
    std::fprintf(stderr, "outside: image-edge %d beyond-cap %d ellipse-too-small %d | sigma-distance <4 %d 4-6 %d >6 %d\n", go[0], go[1], go[2], go[3], go[4], go[5]);
    std::fprintf(stderr, "rows: both rows accepted for correct %d near %d wrong %d\n", gr[0], gr[1], gr[2]); std::fprintf(stderr, "gtchk n %.0f  |p err| before %.4f after %.4f   |v err| before %.4f after %.4f\n", gc_n, gc_p_before / gc_n, gc_p_after / gc_n, gc_v_before / gc_n, gc_v_after / gc_n); }
Vec3 gt_world(int k) {
    const double* g = &gt_pose[7 * k];
    return add(p_wc0, mul(R_wc0, Vec3{g[0], g[1], g[2]}));
}
void gtchk(Vec3 dp, Vec3 dv) {
    if (gt_pose.empty()) {
        const char* d = std::getenv("SPL_GT"); if (!d) return;
        char path[4096]; std::snprintf(path, sizeof path, "%s/gt.f64", d);
        FILE* f = std::fopen(path, "rb"); if (!f) return;
        double x; while (std::fread(&x, 8, 1, f) == 1) gt_pose.push_back(x); std::fclose(f);
        std::atexit(gtchk_report);
    }
    const int k = gt_frame;
    if (k < 1 || 7 * (k + 2) > (int)gt_pose.size()) return;
    const Vec3 c = cam_center(), cg = gt_world(k);
    const Vec3 vg = scale(sub(gt_world(k + 1), gt_world(k - 1)), 10.0);   // 20 Hz central difference
    gc_n++;
    gc_p_before += norm(sub(c, cg)); gc_p_after += norm(sub(add(c, dp), cg));
    gc_v_before += norm(sub(v_w, vg)); gc_v_after += norm(sub(add(v_w, dv), vg));
}
#endif
// tangent basis of a unit bearing
inline void tangents(Vec3 w, Vec3& t1, Vec3& t2) {
    Vec3 a = std::fabs(w.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    t1 = vnorm(cross(w, a)); t2 = cross(w, t1);
}

// EKF rows of one splat seen along world bearing w (unit) from the current camera; returns rows accepted
int splat_rows(Splat& s, Vec3 w) {
    SD(0);
#if SPLAT == 2
    const int kb = IXL + 3 * (int)(&s - spl);
    {
        const double tr = (double)P[kb * NS + kb] + (double)P[(kb + 1) * NS + kb + 1] + (double)P[(kb + 2) * NS + kb + 2];
        if (last_t - s.t0 < SPL_MIN_AGE) { SD(1); return 0; }
        if (tr > SPL_SMAX * SPL_SMAX) { SD(2); return 0; }
    }
#else
    if (last_t - s.t0 < SPL_MIN_AGE) { SD(1); return 0; }
    if (qform(s.S, Vec3{1, 0, 0}) + qform(s.S, Vec3{0, 1, 0}) + qform(s.S, Vec3{0, 0, 1}) > SPL_SMAX * SPL_SMAX) { SD(2); return 0; }
#endif
    const Vec3 c = cam_center();
    const Vec3 Y = sub(s.X, c), Xp = sub(s.X, p_w);
    const double dist = norm(Y);
    if (dist < SPL_DMIN_USE) { SD(3); return 0; }
    const double sb = SPL_SIG_PX / fx_s;
    Vec3 t1, t2; tangents(w, t1, t2);
    const Vec3 ts[2] = {t1, t2};
    int ok = 0;
    for (int k = 0; k < 2; k++) {
        const Vec3 t = ts[k];
        double Hrow[NS];
        for (int i = 0; i < NS; i++) Hrow[i] = 0;
        Hrow[IP] = -t.x; Hrow[IP + 1] = -t.y; Hrow[IP + 2] = -t.z;
        const Vec3 ht = cross(t, Xp);
        if (SPL_HTH) { Hrow[ITH] = ht.x; Hrow[ITH + 1] = ht.y; Hrow[ITH + 2] = ht.z; }
        if (!SPL_HP) { Hrow[IP] = Hrow[IP + 1] = Hrow[IP + 2] = 0; }
#if SPLAT == 2
        Hrow[kb] = t.x; Hrow[kb + 1] = t.y; Hrow[kb + 2] = t.z;
        const double Rn = SPL_RINF * sb * sb * dist * dist;
#else
        const double Rn = SPL_RINF * (sb * sb * dist * dist + qform(s.S, t));
#endif
        const bool acc = kf_update_row(Hrow, -dot(t, Y), Rn);
        if (acc) ok++;
#ifdef SPL_DEBUG
        std::fprintf(stderr, "row t %.2f age %.2f dist %.2f innov_px %.2f sigR_px %.2f nis %.2f acc %d hits %d\n", last_t, last_t - s.t0, dist, -dot(t, Y) / dist * fx_s, std::sqrt(Rn) / dist * fx_s, last_nis, (int)acc, s.hits);
#endif
    }
    spl_diag_rows += ok;
    return ok;
}

// per-splat Kalman refinement with bearing w from the (corrected) current camera
#ifndef SPL_FREEZE
#define SPL_FREEZE 0        // 1: splats are not refined after creation (the early map stays as built)
#endif
bool splat_refine(Splat& s, Vec3 w) {
    if (SPL_FREEZE) { (void)s; (void)w; return true; }
#if defined(SPL_GTMAP) && !defined(SPL_ORACLE_MATCH)
    (void)s; (void)w; return true;
#endif
    const Vec3 c = cam_center();
    const Vec3 Y = sub(s.X, c);
    const double dist = norm(Y);
    if (!(dist > 1e-3) || dot(Y, w) <= 0) return false;
    const double sb = SPL_SIG_PX / fx_s;
    double C[9];
    pose_cov(sub(s.X, p_w), C);
    { const double tr = pose_trace(), f = tr > s.ptr0 ? 1.0 - s.ptr0 / tr : 0.0; for (int i = 0; i < 9; i++) C[i] *= f; }
    Vec3 t1, t2; tangents(w, t1, t2);
    const Vec3 ts[2] = {t1, t2};
    {   // consistency: 2D NIS of the bearing against splat + pose-drift + pixel noise (diagonal approximation)
        double nis = 0;
        for (int k = 0; k < 2; k++) {
            const double r = dot(ts[k], Y), den = qform(s.S, ts[k]) + qform(C, ts[k]) + sb * sb * dist * dist;
            nis += r * r / den;
        }
        if (!(nis < SPL_NIS_KILL)) { SD(11); return false; }
    }
    for (int k = 0; k < 2; k++) {
        const Vec3 t = ts[k];
        const double r = -dot(t, sub(s.X, c));
        const Vec3 St = mulS(s.S, t);
        const double den = dot(t, St) + sb * sb * dist * dist + qform(C, t);
        if (!(den > 0)) return false;
        const double g = r / den;
        s.X = add(s.X, scale(St, g));
        const double a[3] = {St.x, St.y, St.z};
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) s.S[3 * i + j] -= a[i] * a[j] / den;
    }
    for (int i = 0; i < 3; i++) for (int j = i + 1; j < 3; j++) { const double m = 0.5 * (s.S[3 * i + j] + s.S[3 * j + i]); s.S[3 * i + j] = s.S[3 * j + i] = m; }
    return true;
}

int splat_slot() {
    int best = -1; double bs = 1e30;
    for (int i = 0; i < SPL_N; i++) {
        if (!spl[i].alive) return i;
        if (spl[i].link >= 0) continue;
        const double sc = SPL_KEEP_ACC ? -(spl[i].S[0] + spl[i].S[4] + spl[i].S[8])   // drop the least accurate splat
                                       : spl[i].hits - 3.0 * spl[i].miss - 0.2 * (last_t - spl[i].t0);
        if (sc < bs) { bs = sc; best = i; }
    }
    return best;
}

// new splat from feature i: rays (fa_c, fa_w) and (current camera, w)
#ifndef SPL_KEEP_ACC
#define SPL_KEEP_ACC 0      // 1: a full map replaces its least accurate splat (largest covariance)
#endif
#ifndef SPL_RELOC
#define SPL_RELOC 0         // 1: wide ZNCC relocalization of map splats instead of the narrow re-acquisition
#endif
#ifndef SPL_T_MAP
#define SPL_T_MAP 1e9       // s: splats are only created before this time (early, nearly drift-free map)
#endif
#ifndef SPL_T_START
#define SPL_T_START 0.0     // s after the filter start before splats are created
#endif
void splat_create(int i, Vec3 w) {
    if (last_t - t_first < SPL_T_START) return;
    if (last_t - t_first > SPL_T_MAP) return;
    const Vec3 ca = fa_c[i], wa = fa_w[i], cb = cam_center();
    const double cosang = dot(wa, w);
    if (!(cosang < std::cos(SPL_MIN_ANG))) return;
    // midpoint: minimize |ca + la wa - cb - lb w|
    const Vec3 d = sub(cb, ca);
    const double b = dot(wa, w), den = 1.0 - b * b;
    if (!(den > 1e-6)) return;
    const double la = (dot(d, wa) - b * dot(d, w)) / den, lb = (b * dot(d, wa) - dot(d, w)) / den;
    if (!(la > SPL_ZMIN && lb > SPL_ZMIN && lb < SPL_ZMAX && la < SPL_ZMAX)) return;
    const Vec3 Xa = add(ca, scale(wa, la)), Xb = add(cb, scale(w, lb));
    if (norm(sub(Xa, Xb)) > 0.1 * lb + 0.05) return;            // rays do not meet: bad track
    const int k = splat_slot();
    if (k < 0) return;
    Splat& s = spl[k];
    if (s.alive && s.link >= 0) feat_spl[s.link] = -1;
#ifdef SPL_DEBUG
    spl_diag_created++;
#endif
    s.alive = 1; s.link = i; s.gen = feat_gen[i]; s.hits = 0; s.miss = 0; s.t0 = last_t; s.ptr0 = pose_trace(); s.act = 0;
    s.X = scale(add(Xa, Xb), 0.5);
    if (SPL_DEPTH_GAIN != 1.0) s.X = add(cb, scale(sub(s.X, cb), SPL_DEPTH_GAIN));
    const double sb = SPL_SIG_PX / fx_s, ang = std::acos(cosang < 1 ? cosang : 1.0);
    const double sl = lb * (sb * 1.4 / ang + SPL_REL), st = lb * sb;
    double Pc[9] = {0};
    if (SPL_POSE_COV && SPLAT == 1) pose_cov(sub(s.X, p_w), Pc);
    const double wv[3] = {w.x, w.y, w.z};
    for (int a = 0; a < 3; a++)
        for (int c = 0; c < 3; c++)
            s.S[3 * a + c] = (sl * sl - st * st) * wv[a] * wv[c] + (a == c ? st * st : 0.0) + Pc[3 * a + c];
#if SPLAT == 2
    {   // X = p + Y (Y = X - p, attitude-carried): dX = dp - [Y]x dth.  P_X* = J P_pose*, P_XX = J P_pose J^T + N
        if (SYM_P && p_lower_stale) { mirror_P(); p_lower_stale = false; }
        splat_free_state(k);
        const int kb = IXL + 3 * k;
        const Vec3 Y = sub(s.X, p_w);
        const double y[3] = {Y.x, Y.y, Y.z};
        double J[3][6];   // columns: p (3), th (3);  -[Y]x
        for (int a = 0; a < 3; a++) for (int c = 0; c < 3; c++) J[a][c] = a == c ? 1.0 : 0.0;
        const double K[9] = {0, -y[2], y[1], y[2], 0, -y[0], -y[1], y[0], 0};
        for (int a = 0; a < 3; a++) for (int c = 0; c < 3; c++) J[a][3 + c] = -K[3 * a + c];
        const int idx[6] = {IP, IP + 1, IP + 2, ITH, ITH + 1, ITH + 2};
        for (int j = 0; j < NS; j++) {
            if (j >= IXL && j < IXL + 3 * SPL_K) continue;
            for (int a = 0; a < 3; a++) {
                double v = 0;
                for (int c = 0; c < 6; c++) v += J[a][c] * (double)P[idx[c] * NS + j];
                P[(kb + a) * NS + j] = (PT)v; P[j * NS + kb + a] = (PT)v;
            }
        }
        for (int j = 0; j < SPL_K; j++) {   // other splats: through their pose cross-covariance
            if (j == k || !spl[j].alive) continue;
            for (int jj = 0; jj < 3; jj++) {
                const int col = IXL + 3 * j + jj;
                for (int a = 0; a < 3; a++) {
                    double v = 0;
                    for (int c = 0; c < 6; c++) v += J[a][c] * (double)P[idx[c] * NS + col];
                    P[(kb + a) * NS + col] = (PT)v; P[col * NS + kb + a] = (PT)v;
                }
            }
        }
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++) {
                double v = s.S[3 * a + b];
                for (int c = 0; c < 6; c++) v += J[a][c] * (double)P[(kb + b) * NS + idx[c]];
                P[(kb + a) * NS + kb + b] = (PT)v;
            }
    }
#endif
    s.view = w; s.dist = lb;
#ifdef SPL_GTMAP
    {   // oracle map: the same two rays triangulated with ground-truth camera poses (c0 frame -> world via R_wc0, p_wc0)
        static std::vector<double> ft;
        if (ft.empty()) { char path[4096]; std::snprintf(path, sizeof path, "%s/frame_times.f64", std::getenv("SPL_GT")); FILE* f = std::fopen(path, "rb"); double x; while (f && std::fread(&x, 8, 1, f) == 1) ft.push_back(x); if (f) std::fclose(f); }
        if (gt_pose.empty()) gtchk(Vec3{0, 0, 0}, Vec3{0, 0, 0});
        auto frame_of = [&](double tt) { int best = 0; for (int q = 0; q < (int)ft.size(); q++) if (std::fabs(ft[q] - tt) < std::fabs(ft[best] - tt)) best = q; return best; };
        auto gpose = [&](int k, Mat3& Rw, Vec3& cw) {
            const double* g = &gt_pose[7 * k]; double q[4] = {g[3], g[4], g[5], g[6]};
            const double x = q[0], y = q[1], z = q[2], ww = q[3];
            Mat3 R{{1 - 2 * (y * y + z * z), 2 * (x * y - z * ww), 2 * (x * z + y * ww), 2 * (x * y + z * ww), 1 - 2 * (x * x + z * z), 2 * (y * z - x * ww), 2 * (x * z - y * ww), 2 * (y * z + x * ww), 1 - 2 * (x * x + y * y)}};
            Rw = mul(R_wc0, R); cw = add(p_wc0, mul(R_wc0, Vec3{g[0], g[1], g[2]}));
        };
        Mat3 Ra, Rb; Vec3 ga, gb;
        gpose(frame_of(fa_t[i]), Ra, ga); gpose(gt_frame, Rb, gb);
        const Vec3 da = vnorm(mul(Ra, Vec3{fa_bx[i], fa_by[i], 1.0}));
        const Vec3 wc = mul(transpose(mul(R_wb, R_ic)), w);
        const Vec3 db = vnorm(mul(Rb, scale(wc, 1.0 / wc.z)));
        const Vec3 dd = sub(gb, ga);
        const double bb = dot(da, db), den2 = 1.0 - bb * bb;
        if (!(den2 > 1e-6)) { s.alive = 0; feat_spl[i] = -1; return; }
        const double l1 = (dot(dd, da) - bb * dot(dd, db)) / den2, l2 = (bb * dot(dd, da) - dot(dd, db)) / den2;
        if (!(l1 > 0.2 && l2 > 0.2)) { s.alive = 0; feat_spl[i] = -1; return; }
#ifdef SPL_ORACLE_MATCH
        s.Xg = scale(add(add(ga, scale(da, l1)), add(gb, scale(db, l2))), 0.5);
#else
        s.X = scale(add(add(ga, scale(da, l1)), add(gb, scale(db, l2))), 0.5);
        for (int q = 0; q < 9; q++) s.S[q] = (q % 4 == 0) ? 1e-4 : 0.0;
#endif
    }
#endif
    s.uid = ++spl_uid;
    GTL("C %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", s.uid, last_t, s.X.x, s.X.y, s.X.z, fa_t[i], fa_bx[i], fa_by[i], cb.x, cb.y, cb.z, ca.x, ca.y, ca.z);
    { Vec3 bc = mul(transpose(mul(R_wb, R_ic)), w); GTL("O %d %.6f %.6f %.6f\n", s.uid, last_t, bc.x / bc.z, bc.y / bc.z); }
    s.tp = patches[i];
#if SPL_BIG
    {
        const int bx = (int)std::floor(feats[i].px + 0.5) - 4, by = (int)std::floor(feats[i].py + 0.5) - 4;
        if (bx < 0 || by < 0 || bx + 16 + 4 > SW || by + 16 > SH) { s.alive = 0; feat_spl[i] = -1; return; }
        alignas(16) float rowb[24];
        for (int r = 0; r < 16; r++) { fetch_row(bx, by + r, 16, rowb); for (int q = 0; q < 16; q++) s.big[r * 16 + q] = rowb[q]; }
    }
#endif
#if SPL_RELOC
    tmpl_stats(s);
#endif
    feat_spl[i] = k;
}

// group g was just collected (trkg_collect): linked splats are measured by their live features, new splats born
void splat_group(int g) {
    const Mat3 Rc = mul(R_wb, R_ic);
    Vec3 wb[N_TRACK]; int has[N_TRACK], rows_ok[N_TRACK];
    int nrows = 0;
    for (int i = g; i < N_TRACK; i += TG) {
        has[i] = 0;
        const Feat& f = feats[i];
        if (!f.alive || !f.keyed) continue;
        double mx = f.px, my = f.py;
        int k = feat_spl[i];
        if (k >= 0 && !(spl[k].alive && spl[k].link == i && spl[k].gen == feat_gen[i])) { feat_spl[i] = -1; k = -1; }
        if (k >= 0 && SPL_OWN && !track(spl[k].tp, mx, my)) {   // the splat's own appearance is no longer at the feature
            spl[k].link = -1; feat_spl[i] = -1; spl[k].miss++; SD(12);
            continue;
        }
        Vec3 b{(mx + PC - cx_s) / fx_s, (my + PC - cy_s) / fy_s, 1.0};
        wb[i] = vnorm(mul(Rc, b)); has[i] = 1;
        if (k >= 0) GTL("O %d %.6f %.6f %.6f\n", spl[k].uid, last_t, b.x, b.y);
        rows_ok[i] = 1;
        if (k >= 0 && SPL_LINKED) { const int r = splat_rows(spl[k], wb[i]); nrows += r; rows_ok[i] = r > 0 || SD(13) < 0; }
    }
    if (nrows) {   // the direction update uses the displacement frozen at collect: it takes the same correction
        const Vec3 dd{dx[IP] - dx[IPC], dx[IP + 1] - dx[IPC + 1], dx[IP + 2] - dx[IPC + 2]};
#ifdef SPL_GTCHK
        gtchk(Vec3{dx[IP], dx[IP + 1], dx[IP + 2]}, Vec3{dx[IV], dx[IV + 1], dx[IV + 2]});
#endif
        kf_inject_now();
        if (SPL_DISP_FIX) for (int q = 0; q < TG; q++) dispg[q] = add(dispg[q], dd);
    }
    for (int i = g; i < N_TRACK; i += TG) {
        if (!has[i]) continue;
        const int k = feat_spl[i];
        if (k >= 0) {
#if SPLAT == 2
            if (rows_ok[i]) { spl[k].hits++; spl[k].miss = 0; }
            else if (++spl[k].miss >= SPL_MAXMISS) { spl[k].alive = 0; feat_spl[i] = -1; }
#else
            if (splat_refine(spl[k], wb[i])) { spl[k].hits++; spl[k].miss = 0; }
            else if (++spl[k].miss >= SPL_MAXMISS) { spl[k].alive = 0; feat_spl[i] = -1; }
#endif
        }
        else if (fa_set[i]) splat_create(i, wb[i]);
    }
    // features that died drop their link (the splat stays in the map)
    for (int i = g; i < N_TRACK; i += TG)
        if (!feats[i].alive && feat_spl[i] >= 0) { spl[feat_spl[i]].link = -1; feat_spl[i] = -1; }
}

// project unlinked splats into the current frame; LK from the projected mean (several starts inside the 2D
// ellipse), gate with the EWA 2D covariance; matched splats update the pose and themselves
int reacq_next = 0;
#if SPL_RELOC
// relocalization against the (early, nearly drift-free) splat map: the splat's EWA ellipse under the full pose
// uncertainty gives a wide search window; ZNCC of the splat template on a coarse grid, a unique peak, LK refinement,
// then the EKF chi^2 gate. Re-finding old splats acts like a loop closure.
#ifndef SPL_RELOC_R
#define SPL_RELOC_R 30      // px (tracking res): widest search half-window
#endif
#ifndef SPL_NCC
#define SPL_NCC 0.85
#endif
#ifndef SPL_NCC_MARGIN
#define SPL_NCC_MARGIN 0.05
#endif
#ifndef SPL_NCC_STEP
#define SPL_NCC_STEP 3
#endif
float zncc_at(const float* img, int iw, int x0, int y0, const Patch& p, float tm, float ts) {
    float s = 0, ss = 0, st = 0;
    for (int v = 0; v < PS; v++) {
        const float* r = img + (y0 + v) * iw + x0;
        const float* tt = p.T + v * PS;
        for (int u = 0; u < PS; u++) { const float a = r[u]; s += a; ss += a * a; st += a * tt[u]; }
    }
    const float m = s / PN, var = ss / PN - m * m;
    if (!(var > 4.0f)) return -1.0f;
    return (st / PN - m * tm) / (std::sqrt(var) * ts);
}
#ifndef SPL_ACT
#define SPL_ACT 0           // frames of continued LK tracking after a relocalization (0: off)
#endif
#ifndef SPL_ACT_EVERY
#define SPL_ACT_EVERY 2     // rows every this many tracked frames
#endif
#ifndef SPL_ACT_MAX
#define SPL_ACT_MAX 2       // tracked splats per frame
#endif
#ifndef SPL_COARSE_YSTEP
#define SPL_COARSE_YSTEP 1  // coarse ZNCC rows: every row of the subsampled window (1) or every second (2)
#endif
#ifndef SPL_GATE2D
#define SPL_GATE2D 0.0      // chi^2 (2 dof) gate of a relocalization match against the EWA ellipse (0: off)
#endif
#ifndef SPL_PROF
#define SPL_PROF 0
#endif
#ifndef SPL_FINE_H
#define SPL_FINE_H 2        // fine ZNCC search half-width around the coarse peak (LK refines the rest)
#endif
#ifndef SPL_RELOC_PF
#define SPL_RELOC_PF 2      // relocalization searches per frame (flat fuel)
#endif
#ifndef SPL_SCAN_PF
#define SPL_SCAN_PF 24      // map splats checked for visibility per frame
#endif
// template statistics: full 8x8 and the 4x4 subsample (every second pixel) used by the coarse search
void tmpl_stats(Splat& s) {
    float a = 0, b = 0, a4 = 0, b4 = 0;
    for (int v = 0; v < PS; v++)
        for (int u = 0; u < PS; u++) {
            const float x = s.tp.T[v * PS + u];
            a += x; b += x * x;
            if (!(u & 1) && !(v & 1)) { a4 += x; b4 += x * x; }
        }
    constexpr float N4 = (PS / 2) * (PS / 2);
#if SPL_BIG
    a4 = b4 = 0;
    for (int v = 0; v < 16; v += 4) for (int u = 0; u < 16; u += 4) { const float x = s.big[v * 16 + u]; a4 += x; b4 += x * x; }
#endif
    s.tm = a / PN; s.ts = std::sqrt(std::fmax(1e-3f, b / PN - s.tm * s.tm));
    s.tm4 = a4 / N4; s.ts4 = std::sqrt(std::fmax(1e-3f, b4 / N4 - s.tm4 * s.tm4));
}
// ZNCC of the template at window position (x0, y0); sub = 2: the 4x4 subsample
inline float zncc_at(const float* img, int iw, int x0, int y0, const Splat& s, int sub) {
    float a = 0, b = 0, c = 0;
    for (int v = 0; v < PS; v += sub) {
        const float* r = img + (y0 + v) * iw + x0;
        const float* tt = s.tp.T + v * PS;
        for (int u = 0; u < PS; u += sub) { const float x = r[u]; a += x; b += x * x; c += x * tt[u]; }
    }
    const float n = (float)((PS / sub) * (PS / sub));
    const float m = a / n, var = b / n - m * m;
    if (!(var > 4.0f)) return -1.0f;
    return sub == 1 ? (c / n - m * s.tm) / (std::sqrt(var) * s.ts) : (c / n - m * s.tm4) / (std::sqrt(var) * s.ts4);
}
// one relocalization search for splat k on the current frame; returns 1 if its rows were applied
// coarse ZNCC of the 4x4 subsampled template at 4 consecutive positions (x .. x+3) of the subsampled window
// (stride 1 there = 2 px at tracking resolution). Sums run in the same order as the scalar version.
inline void zncc4(const float* sw, int sws, int x, int y, const float* t4, float tm4, float ts4, float out[4], int sp = 1) {
#if defined(__wasm_simd128__)
    v128_t A = wasm_f32x4_splat(0), B = A, Cc = A;
    for (int v = 0; v < 4; v++) {
        const float* r = sw + (y + sp * v) * sws + x;
        for (int u = 0; u < 4; u++) {
            const v128_t X = wasm_v128_load(r + sp * u), T = wasm_f32x4_splat(t4[v * 4 + u]);
            A = wasm_f32x4_add(A, X); B = wasm_f32x4_add(B, wasm_f32x4_mul(X, X)); Cc = wasm_f32x4_add(Cc, wasm_f32x4_mul(X, T));
        }
    }
    const v128_t n = wasm_f32x4_splat(1.0f / 16.0f);
    const v128_t m = wasm_f32x4_mul(A, n), var = wasm_f32x4_sub(wasm_f32x4_mul(B, n), wasm_f32x4_mul(m, m));
    const v128_t num = wasm_f32x4_sub(wasm_f32x4_mul(Cc, n), wasm_f32x4_mul(m, wasm_f32x4_splat(tm4)));
    const v128_t den = wasm_f32x4_mul(wasm_f32x4_sqrt(wasm_f32x4_max(var, wasm_f32x4_splat(1e-6f))), wasm_f32x4_splat(ts4));
    v128_t z = wasm_f32x4_div(num, den);
    z = wasm_v128_bitselect(z, wasm_f32x4_splat(-1.0f), wasm_f32x4_gt(var, wasm_f32x4_splat(4.0f)));
    wasm_v128_store(out, z);
#else
    float A[4] = {0, 0, 0, 0}, B[4] = {0, 0, 0, 0}, Cc[4] = {0, 0, 0, 0};
    for (int v = 0; v < 4; v++) {
        const float* r = sw + (y + sp * v) * sws + x;
        for (int u = 0; u < 4; u++) {
            const float T = t4[v * 4 + u];
            for (int q = 0; q < 4; q++) { const float X = r[sp * u + q]; A[q] = A[q] + X; B[q] = B[q] + X * X; Cc[q] = Cc[q] + X * T; }
        }
    }
    for (int q = 0; q < 4; q++) {
        const float m = A[q] * (1.0f / 16.0f), var = B[q] * (1.0f / 16.0f) - m * m;
        const float num = Cc[q] * (1.0f / 16.0f) - m * tm4, den = std::sqrt(std::fmax(var, 1e-6f)) * ts4;
        out[q] = var > 4.0f ? num / den : -1.0f;
    }
#endif
}
int reloc_one(Splat& s, const Mat3& Rc, const Mat3& Rcw, Vec3 c, Vec3 Y, Vec3 Yc) {
    constexpr int SWW = SPL_RELOC_R + PS / 2 + 8;            // subsampled window side (+ SIMD slack)
    alignas(16) static float sw[(SWW + 12) * (SWW + 16) + 16];
    alignas(16) static float row[2 * SWW + 16];
    const double iz = 1.0 / Yc.z;
    const double u = cx_s + fx_s * Yc.x * iz, v = cy_s + fy_s * Yc.y * iz;
    // EWA: 2D covariance of the projected splat under splat + full pose uncertainty
    double C[9];
    pose_cov(sub(s.X, p_w), C);
    for (int i = 0; i < 9; i++) C[i] += s.S[i];
    const double* R = Rcw.m;
    const double J0[3] = {fx_s * iz, 0, -fx_s * Yc.x * iz * iz}, J1[3] = {0, fy_s * iz, -fy_s * Yc.y * iz * iz};
    const Vec3 j0{J0[0] * R[0] + J0[2] * R[6], J0[0] * R[1] + J0[2] * R[7], J0[0] * R[2] + J0[2] * R[8]};
    const Vec3 j1{J1[1] * R[3] + J1[2] * R[6], J1[1] * R[4] + J1[2] * R[7], J1[1] * R[5] + J1[2] * R[8]};
    const double a = dot(j0, mulS(C, j0)) + 1, d = dot(j1, mulS(C, j1)) + 1, bco = dot(j0, mulS(C, j1));
    int rx = (int)std::ceil(3.0 * std::sqrt(a)), ry = (int)std::ceil(3.0 * std::sqrt(d));
    if (rx > SPL_RELOC_R) rx = SPL_RELOC_R;
    if (ry > SPL_RELOC_R) ry = SPL_RELOC_R;
    if (rx < 4) rx = 4;
    if (ry < 4) ry = 4;
    const int m = MARGIN + 2;
    int x0 = (int)std::floor(u - PC) - rx, x1 = (int)std::floor(u - PC) + rx, y0 = (int)std::floor(v - PC) - ry, y1 = (int)std::floor(v - PC) + ry;
    if (x0 < m) x0 = m;
    if (y0 < m) y0 = m;
    if (x1 > SW - PS - m - 1) x1 = SW - PS - m - 1;
    if (y1 > SH - PS - m - 1) y1 = SH - PS - m - 1;
    if (x1 < x0 || y1 < y0) return 0;
    SD(8);
#ifdef SPL_ORACLE_MATCH
    double gu = -1e9, gv = -1e9; bool g_vis = false;
    if (!gt_pose.empty() && gt_frame >= 1) {
        const double* g = &gt_pose[7 * gt_frame];
        const double qx = g[3], qy = g[4], qz = g[5], qw = g[6];
        Mat3 Rq{{1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw), 2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw), 2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)}};
        const Mat3 Rg = mul(R_wc0, Rq); const Vec3 cg = add(p_wc0, mul(R_wc0, Vec3{g[0], g[1], g[2]}));
        const Vec3 Yg = mul(transpose(Rg), sub(s.Xg, cg));
        if (Yg.z > 0.3) { gu = cx_s + fx_s * Yg.x / Yg.z - PC; gv = cy_s + fy_s * Yg.y / Yg.z - PC; g_vis = true; }
    }
    const bool g_in = g_vis && gu >= x0 && gu <= x1 && gv >= y0 && gv <= y1;
    gd[0]++; if (g_vis) gd[1]++; if (g_in) gd[2]++;
    if (g_vis && !g_in) {   // why outside: capped window or an ellipse that is too small?
        const double ex = gu - (u - PC), ey = gv - (v - PC);
        const bool capped = std::fabs(ex) > SPL_RELOC_R || std::fabs(ey) > SPL_RELOC_R;
        const bool clipped = gu < x0 - 0.5 && (int)std::floor(u - PC) - rx < MARGIN + 2 || gu > x1 + 0.5 && (int)std::floor(u - PC) + rx > SW - PS - MARGIN - 3 || gv < y0 - 0.5 && (int)std::floor(v - PC) - ry < MARGIN + 2 || gv > y1 + 0.5 && (int)std::floor(v - PC) + ry > SH - PS - MARGIN - 3;
        if (clipped) go[0]++; else if (capped) go[1]++; else go[2]++;
        const double m = std::sqrt(ex * ex / a + ey * ey / d);
        if (m < 4) go[3]++; else if (m < 6) go[4]++; else go[5]++;
    }
#endif
    // subsampled window: every second row and column (top-left corners x0 .. x1 -> nx positions at stride 2)
    const int nx = (x1 - x0) / 2 + 1, ny = (y1 - y0) / 2 + 1;
    constexpr int EXT = SPL_BIG ? 8 : 3;                       // template extent beyond the position (sw px)
    constexpr int OFF = SPL_BIG ? 4 : 0;                       // big template starts 4 px before the 8x8 one
    const int sws = ((nx + EXT + 3) / 4) * 4 + 4;             // row stride (multiple of 4, room for SIMD reads)
    const int scols = nx + EXT + 4, srows = ny + EXT;
    const int wx0 = x0 - OFF, wy0 = y0 - OFF;
    for (int r = 0; r < srows; r++) {
        const int yy = wy0 + 2 * r;
        const int ncol = 2 * scols;
        if (yy >= SH || wx0 + ncol > SW) {   // past the image edge: zeros (scores there are not used)
            for (int q = 0; q < scols; q++) sw[r * sws + q] = 0;
            int nfit = (SW - wx0) / 2; if (nfit > scols) nfit = scols;
            if (yy < SH && nfit > 0) { fetch_row(wx0, yy, 2 * nfit, row); for (int q = 0; q < nfit; q++) sw[r * sws + q] = row[2 * q]; }
            continue;
        }
        fetch_row(wx0, yy, ncol, row);
        for (int q = 0; q < scols; q++) sw[r * sws + q] = row[2 * q];
    }
    if (SPL_PROF == 1) return 0;
    float t4[16];
#if SPL_BIG
    for (int vv = 0; vv < 4; vv++) for (int uu = 0; uu < 4; uu++) t4[vv * 4 + uu] = s.big[(4 * vv) * 16 + 4 * uu];
    constexpr int SP = 2;
#else
    for (int vv = 0; vv < 4; vv++) for (int uu = 0; uu < 4; uu++) t4[vv * 4 + uu] = s.tp.T[(2 * vv) * PS + 2 * uu];
    constexpr int SP = 1;
#endif
    float b1 = -2, b2 = -2; int bx = 0, by = 0;
    alignas(16) float z4[4];
    for (int yy = 0; yy < ny; yy += SPL_COARSE_YSTEP)
        for (int xx = 0; xx < nx; xx += 4) {
            zncc4(sw, sws, xx, yy, t4, s.tm4, s.ts4, z4, SP);
            {   // most positions beat neither score: skip the bookkeeping
                const float zm = std::fmax(std::fmax(z4[0], z4[1]), std::fmax(z4[2], z4[3]));
                if (!(zm > b2)) continue;
            }
            for (int q = 0; q < 4 && xx + q < nx; q++) {
                const float z = z4[q]; const int px = xx + q;
                if (z > b1) {
                    if (std::abs(px - bx) > 2 || std::abs(yy - by) > 2) b2 = b1;
                    b1 = z; bx = px; by = yy;
                } else if (z > b2 && (std::abs(px - bx) > 2 || std::abs(yy - by) > 2)) b2 = z;
            }
        }
    if (SPL_PROF == 2) return b1 > 10.0f && b2 > 10.0f ? 1 : 0;
    if (!(b1 - b2 >= SPL_NCC_MARGIN)) { s.miss++; return 0; }
    // fine: full template at +-SPL_FINE_H px around the coarse peak
    const int cxp = x0 + 2 * bx, cyp = y0 + 2 * by;
    constexpr int FW = PS + 2 * SPL_FINE_H;
    alignas(16) float fw[FW * FW + 8];
    const int fx0 = cxp - SPL_FINE_H, fy0 = cyp - SPL_FINE_H;
    if (fx0 < 0 || fy0 < 0 || fx0 + FW + 4 > SW || fy0 + FW > SH) { s.miss++; return 0; }
    for (int r = 0; r < FW; r++) fetch_row(fx0, fy0 + r, FW, fw + r * FW);
    float f1 = -2; int fxb = 0, fyb = 0;
    for (int yy = 0; yy <= 2 * SPL_FINE_H; yy++)
        for (int xx = 0; xx <= 2 * SPL_FINE_H; xx++) {
            const float z = zncc_at(fw, FW, xx, yy, s, 1);
            if (z > f1) { f1 = z; fxb = xx; fyb = yy; }
        }
    SD(14);
    if (SPL_PROF == 3) return f1 > 10.0f ? 1 : 0;
    if (!(f1 >= SPL_NCC)) { s.miss++; return 0; }
    SD(9);
    double sx = fx0 + fxb, sy = fy0 + fyb;
    if (!track(s.tp, sx, sy)) { s.miss++; return 0; }
    if (SPL_GATE2D > 0) {   // EWA ellipse gate on the match offset
        const double ex = sx - (u - PC), ey = sy - (v - PC), det = a * d - bco * bco;
        if (!(det > 0) || (d * ex * ex - 2 * bco * ex * ey + a * ey * ey) / det > SPL_GATE2D) { s.miss++; return 0; }
    }
    SD(15);
#ifdef SPL_ORACLE_MATCH
    { const double e = std::hypot(sx - gu, sy - gv); gd[3]++; if (e < 2.0) gd[4]++; else if (e < 5.0) gd[5]++; else gd[6]++; if (g_in) gd[7]++; }
#endif
    Vec3 bb{(sx + PC - cx_s) / fx_s, (sy + PC - cy_s) / fy_s, 1.0};
    const Vec3 w = vnorm(mul(Rc, bb));
    (void)c; (void)Y;
    const int r = splat_rows(s, w);
#ifdef SPL_ORACLE_MATCH
    { const double e = std::hypot(sx - gu, sy - gv); if (r == 2) { if (e < 2.0) gr[0]++; else if (e < 5.0) gr[1]++; else gr[2]++; } }
#endif
    if (r) { s.hits++; s.miss = 0; spl_diag_reacq++; if (SPL_ACT > 0) { s.act = SPL_ACT; s.ax = sx; s.ay = sy; s.aR = R_wb; } }
    return r;
}
// continued tracking of a relocalized splat: gyro-predicted start, LK with its own template, rows every SPL_ACT_EVERY frames
int act_one(Splat& s, const Mat3& Rc) {
    const Mat3 Rh = mul(transpose(Rc), mul(s.aR, R_ic));
    Vec3 b{(s.ax + PC - cx_s) / fx_s, (s.ay + PC - cy_s) / fy_s, 1.0};
    Vec3 br = mul(Rh, b);
    s.act--;
    if (!(br.z >= 0.2)) { s.act = 0; return 0; }
    double sx = cx_s + fx_s * br.x / br.z - PC, sy = cy_s + fy_s * br.y / br.z - PC;
    if (!track(s.tp, sx, sy)) { s.act = 0; return 0; }
    s.ax = sx; s.ay = sy; s.aR = R_wb;
    if ((SPL_ACT - s.act) % SPL_ACT_EVERY) return 0;
    Vec3 bb{(sx + PC - cx_s) / fx_s, (sy + PC - cy_s) / fy_s, 1.0};
    const int r = splat_rows(s, vnorm(mul(Rc, bb)));
    if (!r) s.act = 0;
    return r;
}
// every frame: scan a few map splats for visibility, search at most SPL_RELOC_PF of them
void splat_reloc_frame() {
    if (inject_pending) kf_inject_now();
    const Mat3 Rc = mul(R_wb, R_ic), Rcw = transpose(Rc);
    const Vec3 c = cam_center();
    int tries = 0, nrows = 0;
    if (SPL_ACT > 0) {
        int na = 0;
        for (int k = 0; k < SPL_N && na < SPL_ACT_MAX; k++)
            if (spl[k].alive && spl[k].act > 0) { na++; nrows += act_one(spl[k], Rc); }
    }
    for (int n = 0; n < SPL_SCAN_PF && tries < SPL_RELOC_PF; n++) {
        const int k = reacq_next;
        reacq_next = (reacq_next + 1) % SPL_N;
        Splat& s = spl[k];
        if (!s.alive || s.link >= 0 || s.act > 0) continue;
        SD(4);
        const Vec3 Y = sub(s.X, c);
        const Vec3 Yc = mul(Rcw, Y);
        if (!(Yc.z > SPL_ZMIN)) continue;
        const double dist = norm(Y);
        if (dist > s.dist * SPL_DRATIO || dist * SPL_DRATIO < s.dist) continue;
        if (dot(scale(Y, 1.0 / dist), s.view) < SPL_VIEW_COS) continue;
        const double iz = 1.0 / Yc.z;
        const double u = cx_s + fx_s * Yc.x * iz, v = cy_s + fy_s * Yc.y * iz;
        if (u < -SPL_RELOC_R || v < -SPL_RELOC_R || u > SW + SPL_RELOC_R || v > SH + SPL_RELOC_R) continue;
        SD(7);
        tries++;
        nrows += reloc_one(s, Rc, Rcw, c, Y, Yc);
    }
    if (nrows) {   // a direction update still to come uses displacements frozen at collect: they take the same correction
        const Vec3 dd{dx[IP] - dx[IPC], dx[IP + 1] - dx[IPC + 1], dx[IP + 2] - dx[IPC + 2]};
        kf_inject_now();
        for (int q = 0; q < TG; q++) dispg[q] = add(dispg[q], dd);
    }
}
void splat_reacquire() {}
#elif defined(SPL_ORACLE_MATCH)
// upper bound: perfect place recognition of the self-built map (measured bearing synthesized from the true point / pose)
void splat_reacquire() {
    if (gt_pose.empty() || gt_frame < 1) return;
    const double* g = &gt_pose[7 * gt_frame];
    const double x = g[3], y = g[4], z = g[5], ww = g[6];
    Mat3 R{{1 - 2 * (y * y + z * z), 2 * (x * y - z * ww), 2 * (x * z + y * ww), 2 * (x * y + z * ww), 1 - 2 * (x * x + z * z), 2 * (y * z - x * ww), 2 * (x * z - y * ww), 2 * (y * z + x * ww), 1 - 2 * (x * x + y * y)}};
    const Mat3 Rg = mul(R_wc0, R); const Vec3 cg = add(p_wc0, mul(R_wc0, Vec3{g[0], g[1], g[2]}));
    const Mat3 Rc = mul(R_wb, R_ic);
    int tries = 0, nrows = 0;
    for (int n = 0; n < SPL_N && tries < SPL_REACQ; n++) {
        const int k = (reacq_next + n) % SPL_N;
        Splat& s = spl[k];
        if (!s.alive || s.link >= 0) continue;
        const Vec3 Yc = mul(transpose(Rg), sub(s.Xg, cg));
        if (!(Yc.z > 0.3)) continue;
        const double u = cx_s + fx_s * Yc.x / Yc.z, v = cy_s + fy_s * Yc.y / Yc.z;
        if (u < 8 || v < 8 || u > SW - 8 || v > SH - 8) continue;
        if (dot(vnorm(sub(s.Xg, cg)), s.view) < SPL_VIEW_COS) continue;
        tries++;
        const Vec3 w = vnorm(mul(Rc, Yc));
        nrows += splat_rows(s, w);
        spl_diag_reacq++;
    }
    reacq_next = (reacq_next + 7) % SPL_N;
    if (nrows) kf_inject_now();
}
#else
void splat_reacquire() {
    const Mat3 Rc = mul(R_wb, R_ic), Rcw = transpose(Rc);
    const Vec3 c = cam_center();
    const double sb = SPL_SIG_PX;
    int tries = 0, nrows = 0;
    int found_k[SPL_REACQ]; Vec3 found_w[SPL_REACQ]; int nf = 0;
    const int m = MARGIN + 2;
    for (int n = 0; n < SPL_N && tries < SPL_REACQ; n++) {
        const int k = (reacq_next + n) % SPL_N;
        Splat& s = spl[k];
        if (!s.alive || s.link >= 0) continue;
        const Vec3 Y = sub(s.X, c);
        const Vec3 Yc = mul(Rcw, Y);
        SD(4);
        if (!(Yc.z > SPL_ZMIN)) continue;
        SD(5);
        const double dist = norm(Y);
        if (dist > s.dist * SPL_DRATIO || dist * SPL_DRATIO < s.dist) continue;
        SD(6);
        if (dot(scale(Y, 1.0 / dist), s.view) < SPL_VIEW_COS) continue;
        SD(7);
        const double iz = 1.0 / Yc.z;
        const double u = cx_s + fx_s * Yc.x * iz, v = cy_s + fy_s * Yc.y * iz;
        const double ox = u - PC, oy = v - PC;
        if (ox < m + 1 || oy < m + 1 || ox + PS + m + 1 > SW || oy + PS + m + 1 > SH) continue;
        // EWA projection of the splat (+ pose uncertainty) to a 2D covariance
        double C[9];
#if SPLAT == 2
        {   // exact: covariance of X - p_cam from the filter (splat, pose and their cross terms), then EWA projection
            if (SYM_P && p_lower_stale) { mirror_P(); p_lower_stale = false; }
            const int kb = IXL + 3 * k;
            const Vec3 Yp = sub(s.X, p_w);
            const double y[3] = {Yp.x, Yp.y, Yp.z};
            const double K[9] = {0, -y[2], y[1], y[2], 0, -y[0], -y[1], y[0], 0};
            // e = dX - dp + [Y]x dth  ->  G = [ -I | K | I ] over (p, th, X)
            double G[3][9];
            for (int a = 0; a < 3; a++) for (int c = 0; c < 3; c++) { G[a][c] = a == c ? -1.0 : 0.0; G[a][3 + c] = K[3 * a + c]; G[a][6 + c] = a == c ? 1.0 : 0.0; }
            const int idx[9] = {IP, IP + 1, IP + 2, ITH, ITH + 1, ITH + 2, kb, kb + 1, kb + 2};
            double GP[3][9];
            for (int a = 0; a < 3; a++) for (int c = 0; c < 9; c++) { double v = 0; for (int e = 0; e < 9; e++) v += G[a][e] * (double)P[idx[e] * NS + idx[c]]; GP[a][c] = v; }
            for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) { double v = 0; for (int c = 0; c < 9; c++) v += GP[a][c] * G[b][c]; C[3 * a + b] = v; }
        }
#else
        pose_cov(sub(s.X, p_w), C);
        { const double tr = pose_trace(), f = tr > s.ptr0 ? 1.0 - s.ptr0 / tr : 0.0; for (int i = 0; i < 9; i++) C[i] = C[i] * f + s.S[i]; }
#endif
        // J = d(u,v)/dX = [fx/z 0 -fx x/z^2; 0 fy/z -fy y/z^2] Rcw
        const double J0[3] = {fx_s * iz, 0, -fx_s * Yc.x * iz * iz}, J1[3] = {0, fy_s * iz, -fy_s * Yc.y * iz * iz};
        Vec3 j0{0, 0, 0}, j1{0, 0, 0};
        {   // rows of J0/J1 times Rcw
            const double* R = Rcw.m;
            j0 = Vec3{J0[0] * R[0] + J0[1] * R[3] + J0[2] * R[6], J0[0] * R[1] + J0[1] * R[4] + J0[2] * R[7], J0[0] * R[2] + J0[1] * R[5] + J0[2] * R[8]};
            j1 = Vec3{J1[0] * R[0] + J1[1] * R[3] + J1[2] * R[6], J1[0] * R[1] + J1[1] * R[4] + J1[2] * R[7], J1[0] * R[2] + J1[1] * R[5] + J1[2] * R[8]};
        }
        const Vec3 Cj0 = mulS(C, j0), Cj1 = mulS(C, j1);
        const double a = dot(j0, Cj0) + sb * sb, b = dot(j0, Cj1), d = dot(j1, Cj1) + sb * sb;
        const double det = a * d - b * b;
        SD(8);
        if (!(det > 0)) continue;
        const double tr = 0.5 * (a + d), lmax = tr + std::sqrt(std::fmax(0.0, tr * tr - det));
        double rad = 3.0 * std::sqrt(lmax);
        if (rad > SPL_RMAX) SD(9);
        if (rad > SPL_RMAX) { s.miss++; if (s.miss >= SPL_MAXMISS) s.alive = 0; continue; }   // too uncertain to search
        tries++;
        // LK starts: the mean, then a ring of starts 4 px apart inside the ellipse
        double best = 1e30, bx = 0, by = 0;
        const int ring = rad > 3.0 ? (rad > 6.0 ? 2 : 1) : 0;
        for (int dy = -ring; dy <= ring; dy++)
            for (int dxs = -ring; dxs <= ring; dxs++) {
                double sx = ox + 4.0 * dxs, sy = oy + 4.0 * dy;
                if (!track(s.tp, sx, sy)) continue;
                const double ex = sx - ox, ey = sy - oy;
                const double m2 = (d * ex * ex - 2 * b * ex * ey + a * ey * ey) / det;
                if (m2 < best) { best = m2; bx = sx; by = sy; }
            }
        if (best < 1e29) SD(10);
        if (!(best < SPL_GATE2)) { s.miss++; if (s.miss >= SPL_MAXMISS) s.alive = 0; continue; }
        Vec3 bb{(bx + PC - cx_s) / fx_s, (by + PC - cy_s) / fy_s, 1.0};
        GTL("O %d %.6f %.6f %.6f\n", s.uid, last_t, bb.x, bb.y);
        const Vec3 w = vnorm(mul(Rc, bb));
        nrows += splat_rows(s, w);
        found_k[nf] = k; found_w[nf] = w; nf++;
        spl_diag_reacq++;
    }
    reacq_next = (reacq_next + 7) % SPL_N;
    if (nrows) kf_inject_now();
    for (int n = 0; n < nf; n++) {
        Splat& s = spl[found_k[n]];
        if (SPLAT == 2 || splat_refine(s, found_w[n])) { s.hits++; s.miss = 0; }
        else if (++s.miss >= SPL_MAXMISS) s.alive = 0;
    }
}
#endif

// camera depth of the splat linked to feature slot `slot` (splat-aware speed)
bool splat_track_depth(int slot, double& z) {
    const int k = feat_spl[slot];
    if (k < 0 || !spl[k].alive || spl[k].link != slot || last_t - spl[k].t0 < SPL_SPEED_AGE) return false;
    const Vec3 Yc = mulT(mul(R_wb, R_ic), sub(spl[k].X, cam_center()));
    if (!(Yc.z > SPL_ZMIN)) return false;
    z = Yc.z;
    return true;
}
