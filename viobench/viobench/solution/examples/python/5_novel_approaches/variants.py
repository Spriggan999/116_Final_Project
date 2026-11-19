"""Estimator variants: a C++ source (relative to this directory) plus -D overrides."""
VARIANTS = {
    # Copy of 2_deterministic/solution.cpp (v8); must reproduce its results.
    'base': dict(source='estimator/solution.cpp'),
    # Scale from the accelerometer: no fixed-depth parallax speed row, no zero-velocity prior.
    'imu_scale': dict(source='estimator/solution.cpp', defines=dict(USE_PAR_SPEED=0, VPRIOR_SIG=0)),
    # Keep both rows but make them weak, so IMU-derived scale dominates.
    'imu_scale_weak': dict(source='estimator/solution.cpp', defines=dict(PAR_SPEED_REL=2.0, VPRIOR_SIG=2.0)),
}
# Body-frame velocity priors replacing the zero-velocity prior (see velocity_prior/).
VARIANTS['cruise_prior'] = dict(source='estimator/solution.cpp', defines=dict(VEL_PRIOR=2))
VARIANTS['learned_prior'] = dict(source='estimator/solution.cpp', defines=dict(VEL_PRIOR=1))
# Repeated prior rows carry correlated information (cruise-speed errors persist for
# seconds, the prior is applied every ~0.45 s): inflate the spread by ~sqrt(2 s / 0.45 s).
for k in (2, 3):
    VARIANTS[f'cruise_prior_x{k}'] = dict(source='estimator/solution.cpp', defines=dict(
        VEL_PRIOR=2, VP_CONST_SIG_X=0.17*k, VP_CONST_SIG_Y=0.05*k, VP_CONST_SIG_Z=0.17*k))
    VARIANTS[f'learned_prior_x{k}'] = dict(source='estimator/solution.cpp', defines=dict(VEL_PRIOR=1, VP_SIG_SCALE=k))
# No-sideslip only: lateral and vertical constraint, forward speed left to vision.
VARIANTS['lateral_prior'] = dict(source='estimator/solution.cpp', defines=dict(
    VEL_PRIOR=2, VP_CONST_FWD=0.0, VP_CONST_SIG_X=0.5, VP_CONST_SIG_Y=0.1, VP_CONST_SIG_Z=0.34))
# Same priors with the attitude Jacobian in the measurement rows.
for name in ('cruise_prior', 'cruise_prior_x2', 'cruise_prior_x3', 'lateral_prior', 'learned_prior_x3'):
    spec = VARIANTS[name]
    VARIANTS[name+'_att'] = dict(source=spec['source'], defines=dict(spec['defines'], VP_ATTITUDE=1))
# Calibrate over the whole 1 s start hover (the settling wobble averages out over it;
# over the first 0.5 s it leaves a gyro-bias error about as large as the bias).
VARIANTS['hover_calib'] = dict(source='estimator/solution.cpp', defines=dict(CALIB_SECONDS=0.9))
VARIANTS['hover_calib_ba'] = dict(source='estimator/solution.cpp', defines=dict(CALIB_SECONDS=0.9, CALIB_ACCEL_BIAS=1))
VARIANTS['hover_cruise_x3'] = dict(source='estimator/solution.cpp', defines=dict(
    VARIANTS['cruise_prior_x3_att']['defines'], CALIB_SECONDS=0.9, CALIB_ACCEL_BIAS=1))
# Dev-set bracket of the prior inflation, with and without full-hover calibration.
for k in (2, 3, 4, 5):
    d = dict(VEL_PRIOR=2, VP_ATTITUDE=1, VP_CONST_SIG_X=0.17*k, VP_CONST_SIG_Y=0.05*k, VP_CONST_SIG_Z=0.17*k)
    VARIANTS[f'cruise_att_x{k}'] = dict(source='estimator/solution.cpp', defines=d)
    VARIANTS[f'hover_cruise_att_x{k}'] = dict(source='estimator/solution.cpp', defines=dict(d, CALIB_SECONDS=0.9, CALIB_ACCEL_BIAS=1))
for k in (1, 1.5):
    d = dict(VEL_PRIOR=2, VP_ATTITUDE=1, VP_CONST_SIG_X=0.17*k, VP_CONST_SIG_Y=0.05*k, VP_CONST_SIG_Z=0.17*k,
             CALIB_SECONDS=0.9, CALIB_ACCEL_BIAS=1)
    VARIANTS[f'hover_cruise_att_x{k}'] = dict(source='estimator/solution.cpp', defines=d)
for k in (1, 1.5, 2, 3):
    VARIANTS[f'hover_learned_att_x{k}'] = dict(source='estimator/solution.cpp', defines=dict(
        VEL_PRIOR=1, VP_ATTITUDE=1, VP_SIG_SCALE=k, CALIB_SECONDS=0.9, CALIB_ACCEL_BIAS=1))
# Lite build (second Pareto point of 2_deterministic) with the same additions.
VARIANTS['lite_base'] = dict(source='estimator/solution.cpp', defines=dict(VIO_LITE=1))
VARIANTS['lite_hover_cruise_x1.5'] = dict(source='estimator/solution.cpp', defines=dict(
    VARIANTS['hover_cruise_att_x1.5']['defines'], VIO_LITE=1))
# IMU-only cruise estimator (no image access): very low fuel.
VARIANTS['imu_cruise'] = dict(source='imu_cruise/solution.cpp')
VARIANTS['imu_cruise_noint'] = dict(source='imu_cruise/solution.cpp', defines=dict(TAU_I=0))
VARIANTS['imu_cruise_soft'] = dict(source='imu_cruise/solution.cpp', defines=dict(TAU_X=3.0, TAU_Y=1.0, TAU_Z=1.5))
VARIANTS['imu_cruise_hard'] = dict(source='imu_cruise/solution.cpp', defines=dict(TAU_X=0.3, TAU_Y=0.1, TAU_Z=0.2))
# Ablation for the final comparison: cruise prior without the full-hover calibration.
VARIANTS['cruise_att_x1.5'] = dict(source='estimator/solution.cpp', defines=dict(
    VEL_PRIOR=2, VP_ATTITUDE=1, VP_CONST_SIG_X=0.17*1.5, VP_CONST_SIG_Y=0.05*1.5, VP_CONST_SIG_Z=0.17*1.5))
VARIANTS['lite_cruise_x1.5'] = dict(source='estimator/solution.cpp', defines=dict(
    VARIANTS['cruise_att_x1.5']['defines'], VIO_LITE=1))

# ===== Recommended (selected on dev, reported on held-out; see README) =====
VARIANTS['vio_cruise'] = VARIANTS['cruise_att_x1.5']
VARIANTS['vio_cruise_lite'] = VARIANTS['lite_cruise_x1.5']
# Existing ultra-lite point (24k max fuel), the strongest model on the server's harder set.
VARIANTS['ultra'] = dict(source='../2_deterministic/solution_ultra.cpp')
# Robust cruise prior for harder flights: innovation-adaptive spread and/or adaptive cruise speed.
for lite in (0, 1):
    pre = 'lite_' if lite else ''
    base_d = dict(VARIANTS['cruise_att_x1.5']['defines'], VIO_LITE=lite)
    VARIANTS[f'{pre}cruise_adaptR'] = dict(source='estimator/solution.cpp', defines=dict(base_d, VP_ADAPT_R=0.1))
    VARIANTS[f'{pre}cruise_adaptRM'] = dict(source='estimator/solution.cpp', defines=dict(base_d, VP_ADAPT_R=0.1, VP_ADAPT_MEAN=0.05))
# Harder-flight hypotheses (fast motion outruns the 3 px LK search; noisier IMU):
#   wide search: larger crop margin and LK correction limit; short window: less flow per window;
#   imu2: filter IMU noise / bias priors doubled for a noisier sensor.
_cr = VARIANTS['cruise_att_x1.5']['defines']
VARIANTS['cruise_wide'] = dict(source='estimator/solution.cpp', defines=dict(_cr, CROP_MARGIN=6, MAX_CORR_PX=6.0))
VARIANTS['cruise_short'] = dict(source='estimator/solution.cpp', defines=dict(_cr, KF_LEN=4))
VARIANTS['cruise_imu2'] = dict(source='estimator/solution.cpp', defines=dict(_cr, SIG_ACC=0.6, SIG_GYR=0.02, SIG_BA0=0.4, SIG_BG0_REST=0.01, SIG_BG0_MOVING=0.06, SIG_BA=0.02, SIG_BG=0.002))
VARIANTS['base_wide'] = dict(source='estimator/solution.cpp', defines=dict(CROP_MARGIN=6, MAX_CORR_PX=6.0))
# Motion-gated cruise prior (wind can stall the autopilot; the ungated prior invents motion).
for px in (0.5, 1.0, 2.0):
    for lite in (0, 1):
        pre = 'lite_' if lite else ''
        VARIANTS[f'{pre}cruise_gate{px:g}'] = dict(source='estimator/solution.cpp', defines=dict(
            VARIANTS['cruise_att_x1.5']['defines'], VIO_LITE=lite, VP_GATE_PX=px))
# Sustained-stall detector instead of a per-window gate.
for px in (0.5, 1.0):
    for tau in (8, 16):
        for lite in (0, 1):
            pre = 'lite_' if lite else ''
            VARIANTS[f'{pre}cruise_stall{px:g}_t{tau}'] = dict(source='estimator/solution.cpp', defines=dict(
                VARIANTS['cruise_att_x1.5']['defines'], VIO_LITE=lite, VP_STALL_PX=px, VP_STALL_TAU=float(tau)))
# Stall detector that starts "still" (the start hover is still) and/or treats track loss as no motion.
for lite in (0, 1):
    pre = 'lite_' if lite else ''
    d = dict(VARIANTS['cruise_att_x1.5']['defines'], VIO_LITE=lite, VP_STALL_PX=0.5, VP_STALL_TAU=8.0)
    VARIANTS[f'{pre}stall_s0'] = dict(source='estimator/solution.cpp', defines=dict(d, VP_STALL_START=0.0))
    VARIANTS[f'{pre}stall_nt'] = dict(source='estimator/solution.cpp', defines=dict(d, VP_STALL_NOTRACK=1))
    VARIANTS[f'{pre}stall_s0nt'] = dict(source='estimator/solution.cpp', defines=dict(d, VP_STALL_START=0.0, VP_STALL_NOTRACK=1))
# Robust candidates: still-start stall detector + realistic divergence guard (+ gentler still prior).
for lite in (0, 1):
    pre = 'lite_' if lite else ''
    d = dict(VARIANTS[f'{pre}stall_s0']['defines'], VMAX_RESET=5.0)
    VARIANTS[f'{pre}robust'] = dict(source='estimator/solution.cpp', defines=d)
    VARIANTS[f'{pre}robust_soft'] = dict(source='estimator/solution.cpp', defines=dict(d, VP_STILL_SIG=0.25))

# ===== Recommended after the harder server set (selected on dev levels, checked on held-out levels) =====
VARIANTS['vio_robust'] = VARIANTS['robust_soft']
VARIANTS['vio_robust_lite'] = VARIANTS['lite_robust_soft']
