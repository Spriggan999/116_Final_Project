"""Run a deployment through the official Python guest shim without WebAssembly."""
import argparse
import importlib.util
import io
from pathlib import Path
import sys
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[5]


def validate_solution(directory):
    directory = Path(directory)
    guest_path = REPO/'viobench/solution/runtime/python/guest/viobench_guest.py'
    spec = importlib.util.spec_from_file_location('_hyperdrone_guest', guest_path)
    guest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and not path.name.startswith('.'):
                archive.writestr(str(path.relative_to(directory)), path.read_bytes())
    saved_solution = sys.modules.pop('solution', None)
    saved_bundle = sys.modules.get('solution_bundle')
    try:
        width = guest.frame_width(buffer.getvalue())
        if width not in (320, 640):
            raise RuntimeError('Guest frame_width failed')
        import json
        config = dict(width=width, height=width*3//4, fx=width*.6, fy=width*.6,
                      cx=width/2-.5, cy=width*3/8-.5,
                      T_cam_imu=[[0,-1,0,.02],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
        status, size = guest.init(json.dumps(config).encode())
        if status != 0:
            raise RuntimeError('Guest init failed')
        rng = np.random.default_rng(17)
        frame = rng.integers(0, 256, size, dtype=np.uint8)
        for k in range(4):
            imu = np.zeros((10, 7), dtype=np.float64)
            imu[:, 0] = k*.05 + np.arange(1, 11)*.005
            imu[:, 3] = 9.81
            output = bytearray(13*8)
            status = guest.step((k+1)*.05, frame.tobytes(), imu.tobytes(), output)
            values = np.frombuffer(output, dtype=np.float64)
            if status != 0 or not np.isfinite(values).all():
                raise RuntimeError('Guest step failed')
            np.testing.assert_allclose(np.linalg.norm(values[3:7]), 1., atol=1e-8)
        # Input buffers are not guaranteed to contain IMU outside normal episodes.
        output = bytearray(13*8)
        if guest.step(.25, frame.tobytes(), b'', output) != 0:
            raise RuntimeError('Guest empty-IMU handling failed')
        print('Official Python guest: import, init and streaming steps passed (host; fuel unmeasured).')
    finally:
        sys.meta_path = [finder for finder in sys.meta_path if not isinstance(finder, guest.BundleImporter)]
        sys.modules.pop('solution', None)
        if saved_solution is not None:
            sys.modules['solution'] = saved_solution
        if saved_bundle is None:
            sys.modules.pop('solution_bundle', None)
        else:
            sys.modules['solution_bundle'] = saved_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('solution', type=Path, nargs='?', default=Path(__file__).resolve().parent/'solution')
    args = parser.parse_args()
    validate_solution(args.solution)


if __name__ == '__main__':
    main()
