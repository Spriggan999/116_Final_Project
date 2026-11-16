"""Independent complete-flight ProcTHOR and public-60s evaluation."""
import argparse
import json
import os
from pathlib import Path
import time
import tempfile
import legacy
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import EpisodeStore, StateEpisodes
from gated_models import GatedModel
from gated_training import validation_inputs, objective
from leaderboard_metrics import LeaderboardATE
from losses import losses
from trajectory_alignment import alignment_objective
from viobench.tools.geometry import matrix_to_quaternion, quaternion_to_matrix
from viobench.tools.runner.metrics import evaluate_trajectory


def load_models(run, device, profiles=None):
    metadata = json.loads((run/'distillation.json').read_text())
    if metadata.get('experiment') != '4_ssm_gated_updates':
        raise ValueError('Not a gated-updates checkpoint set')
    selected = profiles or metadata['profiles']
    models = {}
    for name in selected:
        settings = json.loads((run/name/'student.json').read_text())
        model = GatedModel(settings).to(device).eval()
        model.load_state_dict(torch.load(run/name/'student.pt', map_location=device, weights_only=True))
        models[name] = model
    return models, metadata


def camera_state(prediction, transform):
    """Initial-IMU pose/world velocity -> initial-camera pose/current-camera twist."""
    rotation = prediction['rot'][0].cpu().numpy().astype(float)
    position = prediction['pos'][0].cpu().numpy().astype(float)
    velocity = prediction['vel'][0].cpu().numpy().astype(float)
    omega = prediction['omega'][0].cpu().numpy().astype(float)
    rbc, lever = transform[:3, :3], transform[:3, 3]
    rci = rbc.T
    camera_rotation = rci@rotation@rbc
    camera_position = rci@(position+(rotation-np.eye(3))@lever)
    camera_velocity = rci@(rotation.T@velocity+np.cross(omega, lever))
    return np.concatenate((camera_position, matrix_to_quaternion(camera_rotation), camera_velocity, rci@omega))


def state_dict(states, device):
    return dict(pos=torch.tensor(states[None, :, :3], device=device, dtype=torch.float32),
                rot=torch.tensor(quaternion_to_matrix(states[:, 3:7])[None], device=device, dtype=torch.float32),
                vel=torch.tensor(states[None, :, 7:10], device=device, dtype=torch.float32),
                omega=torch.tensor(states[None, :, 10:13], device=device, dtype=torch.float32))


@torch.no_grad()
def public_flight(model, episode, device, alignment_weight=.1):
    config, frames = episode.adapt((640, 480))
    state = model.stream_init()
    values = episode.imu
    first = float(episode.frame_times[0])
    cursor = int(np.searchsorted(values[:, 0], first+episode.TIME_TOLERANCE, side='right'))
    if cursor == 0:
        raise ValueError('Public episode has no initial causal IMU sample')
    last_imu = values[cursor-1, 1:7]
    acceleration = values[max(0, cursor-10):cursor, 1:4].mean(0)
    gravity = torch.tensor((-9.81*acceleration/max(np.linalg.norm(acceleration), 1e-6))[None],
                           device=device, dtype=torch.float32)
    last_time = first
    estimates = []
    start = time.perf_counter()
    for index, timestamp in enumerate(episode.frame_times):
        image = torch.tensor(np.asarray(frames[index]).copy(), device=device).permute(2, 0, 1)[None].float()/255
        if index == 0:
            prediction, state = model.stream_step(image, torch.tensor(last_imu[None], device=device, dtype=torch.float32),
                                                   0., gravity, state)
        else:
            end = int(np.searchsorted(values[:, 0], timestamp+episode.TIME_TOLERANCE, side='right'))
            while cursor < end:
                sample = values[cursor]
                tick_time = float(sample[0])
                is_last = cursor == end-1 and abs(tick_time-timestamp) <= episode.TIME_TOLERANCE
                prediction, state = model.stream_step(image if is_last else None,
                    torch.tensor(sample[None, 1:7], device=device, dtype=torch.float32),
                    tick_time-last_time, gravity, state)
                last_imu, last_time = sample[1:7], tick_time
                cursor += 1
            if abs(last_time-timestamp) > episode.TIME_TOLERANCE:
                prediction, state = model.stream_step(image,
                    torch.tensor(last_imu[None], device=device, dtype=torch.float32),
                    float(timestamp-last_time), gravity, state)
                last_time = float(timestamp)
        estimates.append(camera_state(prediction, episode.T_body_camera))
    estimates = np.asarray(estimates)
    truth = episode.ground_truth_state()
    scored = evaluate_trajectory(truth, estimates, episode.frame_times)
    pred, target = state_dict(estimates, device), state_dict(truth, device)
    loss, components = losses(pred, target, device, components=True, metrics=False)
    aligned, _ = alignment_objective(pred, target)
    return dict(validation_loss=float(loss+alignment_weight*aligned), sim3_ate_rmse_m=scored.sim3_ate_rmse,
                unaligned_ate_rmse_m=scored.ate_rmse, rotation_rmse_rad=scored.rotation_rmse,
                velocity_rmse_m_s=scored.velocity_rmse, angular_velocity_rmse_rad_s=scored.angular_velocity_rmse,
                inference_seconds=time.perf_counter()-start, completed=True,
                **{key:float(value) for key,value in components.items()}), estimates, truth


@torch.no_grad()
def evaluate(run, suite, device, profiles=None, out=None, wandb_mode='disabled'):
    torch.set_num_threads(2)
    models, metadata = load_models(run, device, profiles)
    out = out or run/suite
    out.mkdir(parents=True, exist_ok=True)
    rows = {name: [] for name in models}
    if suite == 'public60':
        # Download to writable scratch, while keeping pre-existing caches useful.
        previous = os.environ.get('CONTA_CACHE', str(Path.home()/'.cache'/'conta'))
        if 'CONTA_CACHE' not in os.environ:
            cache = Path(tempfile.gettempdir())/f'viobench-{os.getuid()}'/'conta'
            cache.mkdir(parents=True, exist_ok=True)
            os.environ['CONTA_CACHE'] = str(cache)
            searches = os.environ.get('CONTA_SEARCH_PATHS', '').split(os.pathsep)
            os.environ['CONTA_SEARCH_PATHS'] = os.pathsep.join(dict.fromkeys([p for p in searches+[previous] if p]))
        from viobench.episodes import public
        from viobench.tools.datasets.sources import episodes
        manifest = public
        flights = episodes(public)
    else:
        files = sorted((run/'procthor_data').glob('rollout_s*_e*.npz'))
        manifest = json.loads((run/'procthor_data'/'manifest.json').read_text())
        flights = DataLoader(StateEpisodes(EpisodeStore(files, metadata.get('image_height',480),
                                                        metadata.get('image_width',640))), batch_size=1)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    import wandb
    tracking = wandb.init(project='hyperdrone-vio-fuel-aware', mode=wandb_mode,
                          job_type='gated_validation', dir=str(out), name=run.name+'-'+suite,
                          config=dict(source_run=str(run.resolve()), suite=suite, profiles=list(models)))
    try:
        for index, flight in enumerate(flights):
            for name, model in models.items():
                target_dir = out/name
                target_dir.mkdir(exist_ok=True)
                if suite == 'public60':
                    rate = float(flight.config['episode']['frame_rate_hz'])
                    if flight.num_frames != round(60*rate):
                        raise ValueError('Public validation requires complete 60-second episodes')
                    metrics, estimates, truth = public_flight(model, flight, device, metadata['alignment_weight'])
                    metrics.update(episode=flight.key, hash=flight.hash, frames=flight.num_frames)
                    np.savez_compressed(target_dir/(flight.key+'.npz'), estimates=estimates,
                                        truth=truth, timestamps=flight.frame_times)
                else:
                    inputs, truth, batch = validation_inputs(flight, device)
                    prediction = model(**inputs)
                    _, loss, _ = objective(model, prediction, inputs, truth, metadata['embedding_weight'], metadata['alignment_weight'])
                    scorer = LeaderboardATE()
                    scorer.update(prediction, batch)
                    result = scorer.result()
                    metrics = dict(episode=index, validation_loss=float(loss),
                                   sim3_ate_rmse_m=result['mean_ate_m'],
                                   unaligned_ate_rmse_m=result['unaligned_ate_rmse_m'], completed=True)
                    np.savez_compressed(target_dir/f'episode-{index}.npz',
                        **{key:prediction[key].cpu().numpy() for key in ('pos','rot','vel','omega')})
                rows[name].append(metrics)
                (target_dir/'episodes.json').write_text(json.dumps(rows[name], indent=2, allow_nan=False))
                print(f'{suite} {name} episode {index+1}: {metrics}', flush=True)
        expected = len(manifest)
        if any(len(value) != expected for value in rows.values()):
            raise ValueError('Evaluation did not cover the complete selected suite')
        summary = {}
        for name, values in rows.items():
            summary[name] = {key: float(np.mean([row[key] for row in values]))
                             for key in ('validation_loss','sim3_ate_rmse_m','unaligned_ate_rmse_m')}
            summary[name]['episodes'] = len(values)
            prefix = 'public60' if suite == 'public60' else 'procthor_val'
            tracking.log({f'{name}/{prefix}/{key}':value for key,value in summary[name].items()})
        (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        return summary
    finally:
        tracking.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--suite', choices=['procthor', 'public60'], required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--profiles', nargs='+')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--wandb-mode', choices=['online','offline','disabled'], default='disabled')
    args = parser.parse_args(argv)
    evaluate(args.run, args.suite, args.device, args.profiles, args.out, args.wandb_mode)


if __name__ == '__main__':
    main()
