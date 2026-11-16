"""Benchmark real four-model optimizer updates before selecting a CUDA schedule."""
import argparse
import json
from pathlib import Path
from functools import partial
import legacy
import torch
from gated_models import GatedModel, PROFILES
from gated_schedule import MeasuredPipeline
from gated_training import ground_truth_targets, train_student
from data import EpisodeStore, StateWindows
from torch.utils.data import DataLoader
from performance import adamw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path, help='recorded training NPZ directory')
    parser.add_argument('--out', type=Path, default=legacy.HERE/'scheduling_benchmark.json')
    parser.add_argument('--windows', type=int, nargs='+', default=[2, 1025, 12001],
                        help='sample counts including the initial state; defaults cover 1,1024,12000 IMU intervals')
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA-enabled PyTorch needs an accessible GPU to compare schedules')
    store = EpisodeStore(sorted(args.data.glob('rollout_s*_e*.npz')), 480, 640)
    rows = {}
    for name, settings in PROFILES.items():
        torch.manual_seed(17)
        model = GatedModel(settings).cuda()
        rows[name] = dict(model=model, optimizer=adamw(model.parameters(), lr=3e-4))
    reports = []
    with MeasuredPipeline('cuda', rows) as pipeline:
        for window in args.windows:
            loader = DataLoader(StateWindows(store, window, compact_cameras=True), batch_size=args.batch)
            with pipeline.batches(loader, partial(ground_truth_targets, device='cuda')) as batches:
                prepared = next(batches)
                result = pipeline.autotune(rows, prepared, partial(train_student, device='cuda', weights=(0, 0, 1)))
            result.update(window=window, batch=args.batch)
            reports.append(result)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(reports, indent=2))
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
