"""viobench simulate with a faster, less predictable autopilot (stage-2 style flights). Same arguments as `viobench simulate`.
Env AGILE = "cruise_min cruise_max max_accel max_jerk max_yaw_rate phase_accel phase_jerk"
(course defaults: 0.45 0.8 0.7 2.0 0.8 0.2 0.5). The values are compiled into the autopilot task, so each new set
triggers one silent ~5-7 min environment compile."""
import os, sys, hashlib
from pathlib import Path
import viobench.tools.datasets.generate.simulation as sim
import viobench.tools.datasets.storage as storage
from viobench.tools.cli import main
NAMES = ["CRUISE_SPEED_MIN", "CRUISE_SPEED_MAX", "MAX_ACCELERATION", "MAX_JERK", "MAX_YAW_RATE", "PHASE_ACCELERATION", "PHASE_JERK"]
vals = [float(x) for x in os.environ.get("AGILE", "1.0 2.0 2.0 6.0 1.6 0.6 1.5").split()]
if len(vals) == 8: NAMES.append("TRACKING_MARGIN")   # optional 8th number: larger = the autopilot keeps its speed further from the route
assert len(vals) == len(NAMES), "AGILE needs 7 or 8 numbers"
text = sim.TEMPLATE.read_text()
anchor = "static constexpr TI INITIALIZATION_HOLD_STEPS = @HOLD_STEPS@;"
assert anchor in text, "spec template changed: update agile_sim.py"
text = text.replace(anchor, anchor + "".join(f"\n        static constexpr T {n} = {v!r};" for n, v in zip(NAMES, vals)))
cache = Path.home() / ".cache" / "viobench"; cache.mkdir(parents=True, exist_ok=True)
tpl = cache / f"agile_world-{hashlib.sha256(text.encode()).hexdigest()[:12]}.h.in"
tpl.write_text(text)
sim.TEMPLATE = tpl
sim.PARALLEL_FLIGHTS = int(os.environ.get("VIO_PAR", "8"))
sim.PNG_COMPRESSION = 1; storage.PNG_COMPRESSION = 1
print(f"agile_sim: {dict(zip(NAMES, vals))}, {sim.PARALLEL_FLIGHTS} flights at once", flush=True)
sys.exit(main(["simulate"] + sys.argv[1:]))
