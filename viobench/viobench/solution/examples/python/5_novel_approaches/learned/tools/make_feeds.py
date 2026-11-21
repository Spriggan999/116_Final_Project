"""Green-channel replay feeds for host runs (the estimator reads only the green channel).

    python make_feeds.py OUT_DIR episode.zip [...]      -> OUT_DIR/<zip stem>/{frames.g8, imu.f64, gt.f64, ...}
"""
import os, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

VIOBENCH = str(Path(sys.executable).parent / 'viobench')


def one(args):
    out, zip_path = args
    final = out / Path(zip_path).stem
    if (final / 'frames.g8').exists():
        return final
    tmp = out / (Path(zip_path).stem + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run([VIOBENCH, 'feed', str(zip_path), '-o', str(tmp)], check=True, capture_output=True)
    frames, h, w, c, _ = (int(v) for v in (tmp / 'shape.txt').read_text().split())
    rgb = np.memmap(tmp / 'frames.u8', dtype=np.uint8, mode='r', shape=(frames, h, w, c))
    np.ascontiguousarray(rgb[..., 1]).tofile(tmp / 'frames.g8')
    del rgb
    os.remove(tmp / 'frames.u8')
    tmp.rename(final)
    return final


if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(6) as pool:
        for d in pool.map(one, [(out, z) for z in sys.argv[2:]]):
            print(d)
