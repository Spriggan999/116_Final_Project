"""Benchmark-format episodes from viobench's own generator (the public-set settings).

Held-out test episodes default to ProcTHOR scenes 192-199, which no model in this
repository trains on. Training episodes use other scenes (see --split).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from generate import environment

SPLITS = {
    'heldout': [f'ProcTHOR-Train-{i}' for i in range(192, 200)],
    'train': [f'ProcTHOR-Train-{i}' for i in range(60, 100)],
    'dev': [f'ProcTHOR-Train-{i}' for i in range(60, 80)],   # variant selection; never a final score
}
FORBIDDEN_FOR_TRAIN = {f'ProcTHOR-Train-{i}' for i in [1, *range(192, 200)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=sorted(SPLITS), default='heldout')
    parser.add_argument('--episodes', type=int, default=2, help='per scene')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--root', type=Path, default=Path('/var/tmp')/f'novel-approaches-{os.getuid()}')
    args = parser.parse_args()
    scenes = SPLITS[args.split]
    if args.split != 'heldout' and set(scenes) & FORBIDDEN_FOR_TRAIN:
        parser.error('training scenes overlap public/held-out scenes')
    out = args.root/'episodes'/args.split
    out.mkdir(parents=True, exist_ok=True)
    (out/'generation.json').write_text(json.dumps(dict(scenes=scenes, episodes_per_scene=args.episodes,
                                                       seed=args.seed), indent=2))
    env = environment(args.root)
    env['VIOBENCH_SPOOL'] = str(args.root/'scratch'/'spool')
    Path(env['VIOBENCH_SPOOL']).mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'viobench', 'simulate', '-o', str(out), '--no-video',
                    '--scene-names', *scenes, '--episodes', str(args.episodes), '--seed', str(args.seed)],
                   env=env, check=True)


if __name__ == '__main__':
    main()
