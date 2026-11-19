"""Initialization and accumulated-error diagnostics. Never used for selection.

One forward pass per window produces the model's per-tick corrections (and, for
persistent_visual_v2, its capture-time velocity/orientation updates). Neither
architecture feeds integrated state or gravity back into the network, so the
same corrections can be re-integrated under different initial conditions:

  normal          deployable: zero initial velocity, causal accelerometer gravity
  oracle_velocity true initial velocity
  oracle_gravity  true gravity direction
  oracle_both     both oracles
  reset_<N>s      deployable start, state replaced by ground truth every N seconds
                  (teacher-forcing diagnostic; measures accumulated-state error)

Targets always use the corrected convention: absolute velocity and position
relative to the window start, in the initial body frame. Models trained with
legacy delta-velocity targets therefore carry the initial-velocity offset in
"normal"; "oracle_velocity" is their like-for-like comparison.

The "replay_*" fields compare normal re-integration with the model's own
output and should be near zero; large values invalidate the other modes.
"""
import argparse
import json
import math
from pathlib import Path
import legacy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from data import EpisodeStore
from gated_data import GatedStateWindows, validation_files
from gated_models import GatedModel
from gated_training import validation_inputs
from models import so3_exp

WEIGHTS = {'student': 'student.pt', 'stage': 'stage_best.pt', 'latest': 'latest.pt'}


def load_model(run, name, weights, device):
    settings = json.loads((run/name/'student.json').read_text())
    model = GatedModel(settings).to(device).eval()
    if weights == 'latest':
        state = torch.load(run/'latest.pt', map_location='cpu', weights_only=False)['models'][name]['model']
    elif weights == 'stage':
        state = torch.load(run/name/'stage_best.pt', map_location='cpu', weights_only=True)['model']
    else:
        state = torch.load(run/name/'student.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def reintegrate(prediction, imu, dt, gravity, initial_velocity=None, truth=None, reset_every=None):
    """Sequential replay of GatedModel integration in float64.

    Matches models.integrate_corrections and the persistent_visual_v2 capture
    updates (velocity added, orientation right-multiplied after the IMU step).
    """
    corr = prediction['corrections'].double()
    imu, dt, gravity = imu.double(), dt.double()[:, None], gravity.double()
    b, t, _ = imu.shape
    vu = prediction.get('visual_velocity_update')
    ru = prediction.get('visual_orientation_update')
    pos = imu.new_zeros(b, 3)
    vel = initial_velocity.double().clone() if initial_velocity is not None else imu.new_zeros(b, 3)
    rot = torch.eye(3, dtype=torch.float64, device=imu.device).expand(b, 3, 3).clone()
    out_pos, out_vel, out_rot = [pos], [vel], [rot]
    for k in range(1, t):
        gyro = imu[:, k, 3:]+corr[:, k, :3]
        acc = (rot@(imu[:, k, :3]+corr[:, k, 3:6])[..., None]).squeeze(-1)+gravity+corr[:, k, 6:]
        pos = pos+vel*dt+.5*acc*dt.square()
        vel = vel+acc*dt
        rot = rot@so3_exp(gyro*dt)
        if vu is not None:
            vel = vel+vu[:, k].double()
            rot = rot@so3_exp(ru[:, k].double())
        if reset_every and k % reset_every == 0:
            pos, vel, rot = truth['pos'][:, k].double(), truth['vel'][:, k].double(), truth['rot'][:, k].double()
        out_pos.append(pos);out_vel.append(vel);out_rot.append(rot)
    return dict(pos=torch.stack(out_pos, 1), vel=torch.stack(out_vel, 1), rot=torch.stack(out_rot, 1))


def squared_errors(pred, truth):
    """Per-window sums over ticks 1.. (tick 0 is the initialized sample)."""
    position = (pred['pos'][:, 1:]-truth['pos'][:, 1:].double()).square().sum(-1)
    velocity = (pred['vel'][:, 1:]-truth['vel'][:, 1:].double()).square().sum(-1)
    relative = pred['rot'][:, 1:].transpose(-1, -2)@truth['rot'][:, 1:].double()
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)
    angle = torch.rad2deg(torch.arccos(cosine))
    final = (pred['pos'][:, -1]-truth['pos'][:, -1].double()).norm(dim=-1)
    return dict(position=position.sum(1), velocity=velocity.sum(1), orientation=angle.square().sum(1),
                samples=torch.full_like(final, position.shape[1]), final_position=final)


def summarize(rows):
    samples = sum(r['samples'] for r in rows)
    return dict(position_rmse_m=math.sqrt(sum(r['position'] for r in rows)/samples),
                velocity_rmse_mps=math.sqrt(sum(r['velocity'] for r in rows)/samples),
                orientation_rmse_deg=math.sqrt(sum(r['orientation'] for r in rows)/samples),
                mean_final_position_error_m=float(np.mean([r['final_position'] for r in rows])),
                windows=len(rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--profiles', nargs='+')
    parser.add_argument('--weights', choices=sorted(WEIGHTS), default='latest')
    parser.add_argument('--horizon', type=float, default=20.48, help='window length in seconds')
    parser.add_argument('--validation-manifest', type=Path, help='default: the run\'s procthor_data/manifest.json')
    parser.add_argument('--max-windows', type=int, default=48, help='evenly spaced held-out windows')
    parser.add_argument('--reset-seconds', type=float, nargs='*', default=[1.])
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    torch.set_grad_enabled(False)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    run = args.run.resolve()
    metadata = json.loads((run/'distillation.json').read_text())
    if metadata.get('experiment') != '4_ssm_gated_updates':
        parser.error('Not a gated-updates checkpoint set')
    manifest = args.validation_manifest or run/'procthor_data'/'manifest.json'
    try:files = validation_files(manifest)
    except (OSError, ValueError, KeyError) as error:parser.error(str(error))
    length = round(args.horizon*200)+1
    if length < 2 or args.max_windows < 1 or any(s <= 0 for s in args.reset_seconds):
        parser.error('horizon, window count and reset intervals must be positive')
    store = EpisodeStore(files, metadata.get('image_height', 480), metadata.get('image_width', 640))
    store.corrected_supervision = True
    windows = GatedStateWindows(store, length, False, True)
    picks = np.unique(np.linspace(0, len(windows)-1, min(args.max_windows, len(windows))).round().astype(int))
    loader = DataLoader(Subset(windows, picks.tolist()), batch_size=1)
    profiles = args.profiles or metadata['profiles']
    modes = ['normal', 'oracle_velocity', 'oracle_gravity', 'oracle_both']+[f'reset_{s:g}s' for s in args.reset_seconds]
    rows = {name: {mode: [] for mode in modes} for name in profiles}
    replay = {name: dict(position_m=0., velocity_mps=0.) for name in profiles}
    models = {name: load_model(run, name, args.weights, device) for name in profiles}
    for index, batch in enumerate(loader, 1):
        inputs, truth, _ = validation_inputs(batch, device)
        dt = inputs['dt']
        v0 = batch['oracle_initial_velocity'].to(device)
        g_true = batch['oracle_gravity'].to(device)
        for name, model in models.items():
            prediction = model(**inputs)
            normal = reintegrate(prediction, inputs['imu'], dt, inputs['gravity'])
            replay[name]['position_m'] = max(replay[name]['position_m'],
                float((normal['pos']-prediction['pos'].double()).norm(dim=-1).max()))
            replay[name]['velocity_mps'] = max(replay[name]['velocity_mps'],
                float((normal['vel']-prediction['vel'].double()).norm(dim=-1).max()))
            variants = dict(normal=normal,
                oracle_velocity=reintegrate(prediction, inputs['imu'], dt, inputs['gravity'], v0),
                oracle_gravity=reintegrate(prediction, inputs['imu'], dt, g_true),
                oracle_both=reintegrate(prediction, inputs['imu'], dt, g_true, v0))
            for seconds in args.reset_seconds:
                variants[f'reset_{seconds:g}s'] = reintegrate(prediction, inputs['imu'], dt, inputs['gravity'],
                    truth=truth, reset_every=max(1, round(seconds/float(dt[0]))))
            for mode, value in variants.items():
                errors = squared_errors(value, truth)
                rows[name][mode] += [{k: float(v[i]) for k, v in errors.items()} for i in range(len(errors['samples']))]
        print(f'window {index}/{len(picks)}', flush=True)
    report = dict(run=str(run), weights=args.weights, horizon_seconds=(length-1)/200., windows=len(picks),
                  training_policy=metadata.get('training_policy', 'legacy'), manifest=str(Path(manifest).resolve()),
                  note='Diagnostic only; oracle and reset modes use ground truth and are not deployable scores.',
                  replay_max_deviation=replay,
                  results={name: {mode: summarize(values) for mode, values in modes_.items()}
                           for name, modes_ in rows.items()})
    out = args.out or run/'diagnostics'/f'init-{report["horizon_seconds"]:g}s-{args.weights}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for name, results in report['results'].items():
        print(f'\n{name} (replay deviation {replay[name]["position_m"]:.2e} m)')
        print(f'{"mode":<18}{"pos RMSE m":>12}{"vel RMSE m/s":>14}{"rot RMSE deg":>14}{"final pos m":>13}')
        for mode, s in results.items():
            print(f'{mode:<18}{s["position_rmse_m"]:>12.3f}{s["velocity_rmse_mps"]:>14.3f}'
                  f'{s["orientation_rmse_deg"]:>14.3f}{s["mean_final_position_error_m"]:>13.3f}')
    print('\nWrote', out)
    return report


if __name__ == '__main__':
    main()
