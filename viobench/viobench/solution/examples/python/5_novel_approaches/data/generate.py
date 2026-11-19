"""Generate 60-second ProcTHOR training flights with the existing gated collector.

Scenes default to ProcTHOR 20-59: disjoint from the public scene (1), the
3_deterministic_hybrid training/validation scenes (2-17, 180-183) and the held-out
evaluation scenes (192-199). Output stays off the quota-limited home disk and
outside the 4_ssm_gated_updates resource directory, whose storage budget a running
sweep enforces. Already prepared scene assets there are reused read-only.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
GATED = HERE.parents[1]/'4_ssm_gated_updates'


def environment(root):
    """Subprocess environment reusing the gated experiment's renderer build and assets."""
    env = dict(os.environ)
    scratch = root/'scratch'
    for name, sub in [('TMPDIR', 'tmp'), ('CONTA_CACHE', 'assets'), ('XDG_CACHE_HOME', 'xdg')]:
        (scratch/sub).mkdir(parents=True, exist_ok=True)
        env[name] = str(scratch/sub)
    # Reuse the renderer build and CUDA cache the gated experiment already made
    # (rebuilding the renderer from scratch fails offline), and its scene assets.
    gated = Path('/var/tmp')/f'gated-updates-{os.getuid()}'
    env['HYPERLUX_CACHE_DIR'] = str(gated/'hyperlux')
    env['CUDA_CACHE_PATH'] = str(gated/'cuda')
    searches = [gated/'assets', Path('/var/tmp')/f'hyperdrone-{os.environ.get("USER", "")}'/'scene-assets',
                Path.home()/'.cache'/'conta']
    env['CONTA_SEARCH_PATHS'] = os.pathsep.join(str(p) for p in searches if p.exists())
    for name in ('CMAKE_BUILD_PARALLEL_LEVEL', 'MAX_JOBS', 'HYPERLUX_BUILD_JOBS'):
        env[name] = '2'
    # hyperlux selects the OptiX renderer only when the environment's CUDA tools
    # are on PATH (as run.sh arranges); otherwise it tries to build WEBGPU.
    env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env.get('PATH', '')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/var/tmp')/f'novel-approaches-{os.getuid()}')
    parser.add_argument('--first-scene', type=int, default=20)
    parser.add_argument('--scenes', type=int, default=40)
    parser.add_argument('--chunk', type=int, default=10, help='scenes per collector process (resumable)')
    parser.add_argument('--episodes', type=int, default=2, help='flights per scene')
    parser.add_argument('--seed', type=int, default=4242)
    args = parser.parse_args()
    held_out = set(range(1, 18)) | set(range(180, 184)) | set(range(192, 200))
    scenes = set(range(args.first_scene, args.first_scene+args.scenes))
    if scenes & held_out:
        parser.error(f'scene range overlaps held-out/public scenes: {sorted(scenes & held_out)}')
    sys.path.insert(0, str(GATED))
    from resource_limits import Limits
    env = environment(args.root)
    for start in range(args.first_scene, args.first_scene+args.scenes, args.chunk):
        count = min(args.chunk, args.first_scene+args.scenes-start)
        out = args.root/'flights'/f'procthor-{start:03d}-{start+count-1:03d}'
        if (out/'manifest.json').exists():
            print('done:', out, flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        spec = dict(instances=1, compression_level=1, scene_catalog='procthor',
                    resource_limits=asdict(Limits()))
        (out/'collection_spec.json').write_text(json.dumps(spec))
        command = [sys.executable, '-u', str(GATED/'gated_collect.py'), '--out', str(out),
                   '--scene-offset', str(start), '--scenes', str(count), '--episodes', str(args.episodes),
                   '--seed', str(args.seed+start), '--steps', '12001', '--width', '640', '--height', '480',
                   '--max-output-mib', str(200*count*args.episodes)]
        print('collecting', out, flush=True)
        subprocess.run(command, cwd=GATED, env=env, check=True)


if __name__ == '__main__':
    main()
