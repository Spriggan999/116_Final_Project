"""Check upload size and resource caps on a full synthetic camera/IMU episode.

This measures resource use, not trajectory accuracy. Wasmtime enforces every
store, fuel and wall-clock cap. Real evaluation episodes should be checked too.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

REPO = Path(__file__).resolve().parents[5]
LIMIT_VALUES = dict(init_fuel=200_000_000_000, step_fuel=20_000_000_000,
                    memory_bytes=2*2**30, table_elements=1_048_576,
                    init_wall_seconds=300., step_wall_seconds=60.)
MAX_UPLOAD_BYTES = 128*2**20
EPISODE_TIMEOUT_SECONDS = 24*60


def verify(module, frames=1200, report=None, log=print, step_fuel=None):
    from viobench import Sandbox
    from viobench.tools.abi.config import default_guest_config
    from viobench.tools.runner.sandbox import Limits
    if frames < 1:
        raise ValueError('frames must be positive')
    limits=dict(LIMIT_VALUES)
    if step_fuel is not None:
        if not isinstance(step_fuel,int) or step_fuel < 1:
            raise ValueError('step_fuel must be a positive integer')
        limits['step_fuel']=step_fuel
    module = Path(module)
    result = dict(module=str(module.resolve()), module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                  bytes=module.stat().st_size, frames_requested=frames, frames_completed=0,
                  simulated_duration_seconds=frames/20., limits=limits,
                  max_upload_bytes=MAX_UPLOAD_BYTES, episode_timeout_seconds=EPISODE_TIMEOUT_SECONDS,
                  passed=False, fuel_step_max=0, wall_step_seconds_max=0., memory_bytes_observed_max=0)
    started = time.perf_counter()
    try:
        if result['bytes'] > MAX_UPLOAD_BYTES:
            raise ValueError('Module exceeds the 128 MiB upload cap')
        # Direct stdio avoids Python callback lifetime issues when a trapped
        # WASI instance is destroyed during CLI shutdown.
        sandbox = Sandbox(module, limits=Limits(**limits), inherit_stdio=True)
        width, height = sandbox.resolution
        sandbox.init(default_guest_config(width=width, height=height))
        result.update(width=width, height=height, fuel_init=sandbox.init_fuel,
                      wall_init_seconds_reported=sandbox.init_wall_seconds)
        result['memory_bytes_observed_max'] = sandbox.memory.data_len(sandbox.store)
        rng = np.random.default_rng(1701)
        yy, xx = np.indices((height, width))
        for k in range(frames):
            if time.perf_counter()-started > EPISODE_TIMEOUT_SECONDS:
                raise TimeoutError('Synthetic episode exceeded 24 minutes')
            gray = ((xx+3*k)^((yy+k)*3)) % 256
            frame = np.stack([gray, (gray+37)%256, (gray+93)%256], axis=-1).astype(np.uint8)
            frame ^= rng.integers(0, 32, frame.shape, dtype=np.uint8)
            timestamps = k/20.+np.arange(1, 11)*.005
            imu = np.column_stack([timestamps, rng.normal(0, .3, (10, 3))+[0, 0, 9.81],
                                   rng.normal(0, .07, (10, 3))])
            step = sandbox.step((k+1)/20., frame, imu)
            result['frames_completed'] += 1
            result['fuel_step_max'] = max(result['fuel_step_max'], step.fuel)
            result['wall_step_seconds_max'] = max(result['wall_step_seconds_max'], step.wall_seconds)
            result['memory_bytes_observed_max'] = max(result['memory_bytes_observed_max'],
                                                      sandbox.memory.data_len(sandbox.store))
            if k == 0 or (k+1)%100 == 0 or k+1 == frames:
                log(f"limits: {k+1}/{frames} frames; max fuel {result['fuel_step_max']/1e9:.3f} G; "
                    f"memory {result['memory_bytes_observed_max']/2**20:.1f} MiB; "
                    f"elapsed {time.perf_counter()-started:.1f}s")
        if time.perf_counter()-started > EPISODE_TIMEOUT_SECONDS:
            raise TimeoutError('Synthetic episode exceeded 24 minutes')
        result['passed'] = True
    except Exception as error:
        result['error'] = str(error)
        log('limits: FAILED: '+str(error))
    finally:
        result['elapsed_seconds'] = time.perf_counter()-started
        if report is not None:
            Path(report).write_text(json.dumps(result, indent=2)+'\n')
    log(f"limits: {'PASS' if result['passed'] else 'FAIL'}; upload {result['bytes']/2**20:.2f} MiB; "
        f"{result['frames_completed']} frames in {result['elapsed_seconds']:.1f}s")
    return result


def main():
    import sys
    sys.path.insert(0, str(REPO))
    from viobench.tools.cli import run_with_large_stack
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('module', type=Path)
    parser.add_argument('--frames', type=int, default=1200, help='1200 frames = 60 simulated seconds at 20 Hz')
    parser.add_argument('--report', type=Path, default=Path('teacher-limits.json'))
    parser.add_argument('--step-fuel',type=int,help='override step cap, e.g. 10000 for a 10K native student')
    args = parser.parse_args()
    result = run_with_large_stack(lambda: verify(args.module, args.frames, args.report,step_fuel=args.step_fuel))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
