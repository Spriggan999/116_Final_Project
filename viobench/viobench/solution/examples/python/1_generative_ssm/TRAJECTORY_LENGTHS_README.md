# Leaderboard trajectory lengths

The website table supplied on **2026-11-14** contains **20 episodes**, each with
**1,200 camera frames**. At the benchmark's **20 Hz camera rate**, each episode
is nominally **60 seconds long**. All four baselines are evaluated on the same
selected episodes; there is no separate trajectory duration for each model.

The 20 hash prefixes in that table match the repository's
[`public+private-10` preset](../../../../episodes/__init__.py) exactly, in order:
the ten public episodes followed by the first ten additional episodes. This
identifies the set shown in the supplied table. Instructors can change the live
selection later, so compare episode hashes when using newer results.

## Camera frames and IMU samples are different counts

| Quantity | Rate | Count over a nominal 60-second episode |
|---|---:|---:|
| Camera frames / solution `step` calls | 20 Hz | 1,200 |
| IMU measurements | 200 Hz | About 12,000 |
| IMU measurements per camera interval | 200 / 20 | 10 |

**1,200 IMU samples at 200 Hz would span 6 seconds. The website's Frames column
counts camera frames, so that calculation does not apply to this table.**

For uniformly sampled frames starting at time zero, frame 1,200 is timestamped
59.95 seconds: `(1200 - 1) / 20`. Consequently the local report's `duration_s`
is **59.95 s**, while the nominal recording duration is `1200 / 20 = 60 s`.
This is a first-to-last timestamp convention, not a shorter flight.

The simulator's native 10-second informational limit, the training curriculum's
sequence lengths, the baselines' 0.5-second calibration periods, and the server's
24-minute processing timeout are separate quantities. They do not set the
duration of these recorded leaderboard episodes.

## What the four baseline scores tell us

These scores were supplied from the website, rather than recomputed locally:

| Baseline | Mean Sim(3) ATE (m) | Maximum fuel/frame | Episode duration in the supplied set |
|---|---:|---:|---:|
| `tiny_cnn` | 0.668412 | 14,684,708 | 60 s nominal |
| `cpp_minimal` | 2.695240 | 113 | 60 s nominal |
| `dead_reckoning` | 1.935490 | 4,221,379 | 60 s nominal |
| `klt_imu` | 1.852220 | 289,275,943 | 60 s nominal |

The baseline implementations process observations supplied by the runner:

- [`tiny_cnn`](../tiny_cnn/solution/solution.py) predicts and chains motion per frame.
- [`dead_reckoning`](../dead_reckoning/solution.py) integrates IMU observations.
- [`klt_imu`](../klt_imu/solution.py) combines visual tracking and IMU integration.
- [`cpp_minimal`](../../native/cpp_minimal/solution.cpp) always returns an identity pose and zero twist.

`cpp_minimal` is under the neighboring **native** examples directory; the other
three are Python examples. Their source does not reveal the instructor's live
episode selection or uniquely determine the physical length of the routes.
The tiny-CNN training script's simulation defaults describe generated training
data, not necessarily the server's selected evaluation episodes.

ATE measures trajectory error, not distance travelled. In particular, the
constant-output C++ baseline is translated to the ground-truth centroid by
Sim(3) alignment: its score measures the trajectory's spatial spread, not its
path length. Maximum fuel/frame measures work for one call, not the number of
calls. Neither headline value can be inverted to recover trajectory duration
or physical route length.

## Physical path lengths we can verify locally

Physical trajectory length is the ground-truth camera-centre path length:

`distance_travelled = sum(norm(p[k] - p[k-1]))`, in metres.

The official [trajectory scorer](../../../../tools/runner/metrics.py) calculates
this from ground-truth positions sampled at camera timestamps, before measuring
prediction error. It is independent of which model is evaluated and does not
use the model's fitted Sim(3) scale. Sampling can omit motion between timestamps.

The saved [public-set report](report-first-test/report.json) contains all ten
public hashes in the catalog, in the same order. Its exact metadata gives:

| Public episode suffix | Episode hash prefix | Frames | Timestamp span (s) | Ground-truth path (m) |
|---|---|---:|---:|---:|
| `0000` | `0386e5c941b5` | 1,200 | 59.95 | 19.274 |
| `0001` | `9b6a92791e46` | 1,200 | 59.95 | 19.689 |
| `0002` | `6e8e3766f462` | 1,200 | 59.95 | 19.731 |
| `0003` | `f8fb89fff0c3` | 1,200 | 59.95 | 13.677 |
| `0004` | `489ad518363d` | 1,200 | 59.95 | 14.114 |
| `0005` | `51ea6311a50f` | 1,200 | 59.95 | 19.937 |
| `0006` | `60ed4323f3aa` | 1,200 | 59.95 | 19.173 |
| `0007` | `8fff5c8dc637` | 1,200 | 59.95 | 18.520 |
| `0008` | `41e11443a9a5` | 1,200 | 59.95 | 19.092 |
| `0009` | `8321da1faa30` | 1,200 | 59.95 | 19.084 |

For these ten public routes, the mean path length is **18.229 m**, the range is
**13.677–19.937 m**, and the total is **182.290 m**. These are distances for the
public half of the displayed set, not statistics for all twenty episodes.

The other ten displayed hashes, starting at `53b9f0801513` and ending at
`1b38f5e9c6a8`, also have 1,200 frames in the supplied website table. Their
physical distances are **not established by the local public report or the
rounded website columns**. Read their `metrics.distance_travelled` values from
an existing server result's Raw JSON to determine those distances exactly.
`Drift/m` alone is insufficient without the corresponding final drift.

## Our shorter held-out evaluations

The current teacher's local reports use shorter episodes:

| Local report suffix | Episodes | Frames per episode | Nominal duration | Mean ground-truth path | Path range |
|---|---:|---:|---:|---:|---:|
| `20261114-160102` | 2 | 100 | 5 s | 0.189 m | 0.146–0.231 m |
| `20261114-155426` | 2 | 200 | 10 s | 1.949 m | 1.877–2.020 m |
| `20261114-154035` | 2 | 400 | 20 s | 4.331 m | 4.222–4.440 m |

These reports are under
`checkpoints/20261113-192121-curriculum-tujmpo0u/report-heldout-<suffix>/report.json`.
Their timestamp spans are respectively 4.95, 9.95 and 19.95 seconds. Their Mean
ATE values are not direct comparisons with the twenty one-minute leaderboard
episodes. Compare models on identical episode hashes and input settings.

## How to reproduce the measurements

From this directory, this reads an existing local report and prints both length
definitions. It does not rerun a model or download data:

```bash
python - <<'PY'
import json
from pathlib import Path
from statistics import mean

path = Path('report-first-test/report.json')
report = json.loads(path.read_text())
episodes = report.get('results', report)['episodes']
distances = []
for episode in episodes:
    distance = episode['metrics']['distance_travelled']
    distances.append(distance)
    print(f"{episode['hash'][:12]}: {episode['frames']} camera frames, "
          f"{episode['duration_s']:.2f} s timestamp span, "
          f"{distance:.6f} m ground-truth path")
print(f'Mean path: {mean(distances):.6f} m')
print(f'Range: {min(distances):.6f}–{max(distances):.6f} m')
PY
```

Change `path` to another `report.json`, or to a saved copy of your own server
submission's Raw JSON. Both formats expose the necessary per-episode fields.
For a changing live set, compare its `evaluation_hashes` against the report's
episode hashes and require all selected reports to be present and current.

On the signed-in website, these browser-console commands perform only GET
requests and do not create submissions or consume upload allowance:

```javascript
JSON.stringify((await api('/config')).evaluation_hashes)
```

For an **existing submission that belongs to your account**, replace the ID:

```javascript
const result = await api('/submissions/YOUR_EXISTING_SUBMISSION_ID');
console.table(result.results.episodes.map(e => ({
  episode: e.hash,
  frames: e.frames,
  seconds: e.duration_s,
  ground_truth_metres: e.metrics.distance_travelled
})));
```

Baseline accounts' detailed reports may not be visible to your account. Your
own completed submission on the same selected hashes contains the same
ground-truth route lengths. No new upload is needed to inspect an existing
result. This investigation made only read-only website requests; it did not
upload a model, create a submission, or schedule an evaluation. The unauthenticated
configuration/comparison endpoints returned HTTP 401, so identification of the
displayed twenty-episode set relies on the table you supplied, not an authenticated
API response.

Sources: the local [episode catalog](../../../../episodes/__init__.py),
[frame-count/duration implementation](../../../../tools/datasets/episode.py),
[simulator rates/defaults](../../../../tools/datasets/generate/simulation.py),
[server aggregation](../../../../tools/server/evaluations.py), and
[student guide](../../../../../docs/STUDENT_GUIDE.md).
