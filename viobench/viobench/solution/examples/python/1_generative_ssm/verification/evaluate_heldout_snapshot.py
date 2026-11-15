"""Snapshot saved students and evaluate six-second held-out flights in WASM.

Pause the source trainer before running this script. This does not modify the
source run. Baseline table comparisons are proxies for the live website score.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[4]))


def main():
    import numpy as np
    import torch
    from students.model import BudgetStudent, PROFILES
    from students.export import build
    from viobench.tools.abi.config import make_config, pinhole_intrinsics
    from viobench.tools.datasets.episode import Episode
    from viobench.tools.datasets.generate.simulation import mount_to_body_camera
    from viobench.tools.datasets.regularize import regularize
    from viobench.tools.datasets.storage import write_episode, read_episode
    from viobench.tools.geometry import matrix_to_quaternion

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--teacher', type=Path, help='optional exported teacher WASM on the same episodes')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    episodes_dir = args.out/'episodes'
    episodes_dir.mkdir()
    manifest = json.loads((args.validation/'manifest.json').read_text())
    # One scene from each available catalog, without choosing by model scores.
    selected = []
    for row in manifest:
        if row['scene_catalog'] not in {r['scene_catalog'] for r in selected}:
            selected.append(row)
    if len(selected) < 2:
        selected = manifest[:2]
    assert len(selected) == 2 and len({r['scene_index'] for r in selected}) == 2
    source_state = json.loads((args.source/'recovery_state.json').read_text())
    provenance = dict(source=str(args.source.resolve()), completed_epoch=source_state['completed_epochs'],
                      checkpoint_selection='saved global-best student.pt per profile',
                      duration_seconds=6., scenes=[], checkpoints={},
                      baseline_comparison='proxy: website baseline episodes differ',
                      frames='121 RGB frames at 20 Hz, t=0 through t=6; causal hold from 50 Hz captures')
    paths = []
    for row in selected:
        path = args.validation/Path(row['file']).name
        with np.load(path, allow_pickle=False) as archive:
            assert len(archive['state']) >= 1201
            dt = float(archive['dt'])
            assert abs(dt-.005) < 1e-7
            ticks = np.arange(0, 1201, 10)
            camera_ticks = archive['camera_ticks']
            indices = np.searchsorted(camera_ticks, ticks, side='right')-1
            assert (indices >= 0).all()
            frames = archive['camera'][indices].copy()
            state = archive['state'][:1201].copy()
            # Initial recorder IMU row is a placeholder. Extrapolate the first
            # real sample into frame zero's ABI window via regularize().
            imu = np.column_stack([np.arange(1, 1201)/200., archive['imu'][1:1201]])
        parameters = ast.literal_eval(row['parameters'])
        cameras = parameters['cameras']
        camera = cameras.get('front') or next(iter(cameras.values()))
        transform = mount_to_body_camera(camera['mount'])
        name = f"heldout-{row['scene_catalog']}-{row['catalog_index']}-6s"
        config = make_config(width=640, height=480, channels=3, frame_rate_hz=20.,
                             num_frames=len(ticks), T_body_camera=transform,
                             intrinsics=pinhole_intrinsics(640, 480, camera['fov']), episode_name=name)
        truth = np.column_stack([np.arange(1201)/200., state[:, :3],
                                 matrix_to_quaternion(state[:, 3:12].reshape(-1, 3, 3))])
        episode = regularize(Episode(name, config, ticks/200., frames, imu, truth, transform,
                                     meta=dict(scene_index=row['scene_index'], source_rollout=str(path))))
        episode.check_consistency()
        destination = episodes_dir/(name+'.zip')
        write_episode(episode, destination)
        reread = read_episode(destination)
        reread.check_consistency()
        assert abs(reread.duration-6.) < 1e-8 and reread.num_frames == 121
        paths.append(destination.resolve())
        provenance['scenes'].append(dict(scene_index=row['scene_index'], catalog=row['scene_catalog'],
                                        catalog_index=row['catalog_index'], episode_hash=reread.hash,
                                        duration_s=reread.duration, frames=reread.num_frames))
        print(f'Prepared {name}: {reread.num_frames} frames, {reread.duration}s', flush=True)
        del frames, episode, reread
    snapshot = args.out/'students'
    for name in PROFILES:
        if name=='100k' and not (args.source/name).exists():continue
        if name=='1k' and not (args.source/name).exists() and (args.source/'500').exists():name='500'
        directory = snapshot/name
        directory.mkdir(parents=True)
        for filename in ['student.json', 'student.pt']:
            shutil.copy2(args.source/name/filename, directory/filename)
        settings = json.loads((directory/'student.json').read_text())
        model = BudgetStudent(settings)
        model.load_state_dict(torch.load(directory/'student.pt', map_location='cpu', weights_only=True))
        candidate = build(model.eval(), directory/'native')
        assert candidate.stat().st_size <= 128*2**20
        module = candidate.with_name('student.wasm')
        candidate.rename(module)
        provenance['checkpoints'][name] = dict(
            checkpoint_sha256=hashlib.sha256((directory/'student.pt').read_bytes()).hexdigest(),
            wasm_sha256=hashlib.sha256(module.read_bytes()).hexdigest(), fuel_cap=settings['fuel'])
        print(f'Built {name}: {module.stat().st_size} bytes', flush=True)
        del model
    (args.out/'snapshot.json').write_text(json.dumps(provenance, indent=2)+'\n')
    command = [sys.executable, str(HERE/'evaluate_students.py'), '--students', str(snapshot.resolve()),
               '--out', str((args.out/'results').resolve()), '--jobs', str(args.jobs),
               '--episodes', *map(str, paths)]
    if args.teacher:
        command.extend(['--teacher', str(args.teacher.resolve())])
    subprocess.run(command, cwd=HERE, check=True)


if __name__ == '__main__':
    main()
