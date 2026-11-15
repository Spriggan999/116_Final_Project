"""Claimed-area metric as described on viobench: log-log, floors 1 fuel / 0.001 m, upper refs 1.2*step_fuel and
1.2*worst baseline ATE. Area = region dominated by (baselines + ours) minus region dominated by baselines, as % of rectangle."""
import numpy as np
STEP_FUEL = 2e10
BASE = [(113, 2.695), (4.0e6, 1.93), (1.5e7, 0.70), (1.4e9, 0.55), (2.7e9, 0.54)]  # read off the plot (approx)
WORST = 2.695  # worst baseline mean ATE (the plot's top 3.23 = 1.2 * 2.69)
FX = (0.0, np.log10(1.2 * STEP_FUEL)); AY = (np.log10(0.001), np.log10(1.2 * WORST))
def dominated_area(points, n=4000):
    xs = np.linspace(*FX, n); area = 0.0
    for x in xs:
        best = AY[1]
        for f, a in points:
            if np.log10(max(f, 1)) <= x: best = min(best, np.log10(max(a, 0.001)))
        area += (AY[1] - best)
    return area * (FX[1] - FX[0]) / n
total = (FX[1] - FX[0]) * (AY[1] - AY[0]); base = dominated_area(BASE)
def claimed(ours): return 100 * (dominated_area(BASE + ours) - base) / total
if __name__ == "__main__":
    print(f"rectangle {total:.1f} decade^2, baselines already cover {100*base/total:.1f}%")
    tests = {"teammate 150/1.93": [(150, 1.93)], "v3 1e6/0.597": [(1.0e6, 0.597)],
             "v3 + teammate": [(1.0e6, 0.597), (150, 1.93)],
             "1e5/0.6": [(1e5, 0.6)], "2e4/0.8": [(2e4, 0.8)], "1e4/1.0": [(1e4, 1.0)],
             "ladder 2e4/0.8 + 1e5/0.6 + 1e6/0.5": [(2e4, 0.8), (1e5, 0.6), (1e6, 0.5)],
             "dream 1e4/0.3": [(1e4, 0.3)], "fantasy 1e3/0.05": [(1e3, 0.05)]}
    for k, v in tests.items(): print(f"{k:40s} {claimed(v):5.1f}%")
