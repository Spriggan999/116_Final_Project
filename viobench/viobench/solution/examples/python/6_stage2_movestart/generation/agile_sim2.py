"""Like agile_sim.py, plus the IMU error model and wind (same template edit as 5_novel_approaches/data/simulate_hard.py).
Env AGILE = 7 or 8 autopilot numbers (see agile_sim.py), IMU = white-noise scale, BIAS = bias turn-on scale, WIND = random force std (N)."""
import os, sys, hashlib
from pathlib import Path
import viobench.tools.datasets.generate.simulation as sim
import viobench.tools.datasets.storage as storage
from viobench.tools.cli import main
NAMES = ["CRUISE_SPEED_MIN", "CRUISE_SPEED_MAX", "MAX_ACCELERATION", "MAX_JERK", "MAX_YAW_RATE", "PHASE_ACCELERATION", "PHASE_JERK"]
vals = [float(x) for x in os.environ.get("AGILE", "2.0 3.5 4.0 12.0 2.5 1.2 3.0").split()]
if len(vals) == 8: NAMES.append("TRACKING_MARGIN")
assert len(vals) == len(NAMES), "AGILE needs 7 or 8 numbers"
imu, bias, wind = (float(os.environ.get(k, d)) for k, d in (("IMU", "1"), ("BIAS", "1"), ("WIND", "0")))
text = sim.TEMPLATE.read_text()
anchor = "static constexpr TI INITIALIZATION_HOLD_STEPS = @HOLD_STEPS@;"
old_spec = "    struct SPEC: hyperdrone::presets::X500FPVIMU<T, TI> {"
assert anchor in text and text.count(old_spec) == 1, "spec template changed: update agile_sim2.py"
DYN = f"""
    struct DYN: hyperdrone::presets::x500_fpv_imu::DynamicsStaticParameters<T, TI> {{
        using BASE = hyperdrone::presets::x500_fpv_imu::DynamicsStaticParameters<T, TI>;
        using PT = typename BASE::PARAMETERS_TYPE;
        static constexpr typename PT::IMU imu = {{
                {{{{ {{0, (T)(0.28 * {imu})}}, {{(T)(0.2 * {bias}), (T)0, (T)0.001}} }}}},
                {{{{ {{0, (T)(0.07 * {imu})}}, {{(T)(0.02 * {bias}), (T)0, (T)0.0006}} }}}}
        }};
        static constexpr typename PT::Disturbances disturbances = {{{{0, (T){wind}}}, {{0, 0}}}};
        static constexpr PT PARAMETER_VALUES = {{{{{{BASE::dynamics, BASE::integration, BASE::mdp}}, imu}}, disturbances}};
    }};
    struct SPEC: hyperdrone::Specification<T, TI, DYN> {{
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 8;"""
text = text.replace(old_spec, DYN)
text = text.replace(anchor, anchor + "".join(f"\n        static constexpr T {n} = {v!r};" for n, v in zip(NAMES, vals)))
cache = Path.home() / ".cache" / "viobench"; cache.mkdir(parents=True, exist_ok=True)
tpl = cache / f"agile2_world-{hashlib.sha256(text.encode()).hexdigest()[:12]}.h.in"
tpl.write_text(text)
sim.TEMPLATE = tpl
sim.PARALLEL_FLIGHTS = int(os.environ.get("VIO_PAR", "8"))
sim.PNG_COMPRESSION = 1; storage.PNG_COMPRESSION = 1
print(f"agile_sim2: {dict(zip(NAMES, vals))} imu x{imu} bias x{bias} wind {wind} N, {sim.PARALLEL_FLIGHTS} flights at once", flush=True)
sys.exit(main(["simulate"] + sys.argv[1:]))
