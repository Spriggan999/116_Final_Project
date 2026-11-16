"""Full IMU-rate supervision and camera-rate alignment/JEPA callbacks."""
import legacy
import torch
from data import camera_values
from losses import losses
from trajectory_alignment import alignment_objective
from performance import amp_disabled
from curriculum import photometric_captures
import config as cfg


def frame_inputs(batch, device, with_images=True):
    observed = batch['observed'].bool()
    counts = observed.sum(1)
    if not torch.all(counts == counts[0]):
        raise ValueError('A minibatch must have equal observation counts; use batch size one for irregular flights')
    ticks = observed.nonzero()[:, 1].reshape(len(observed), -1).to(device)
    indices = batch['anchor_img_idx'].gather(1, ticks.cpu())
    # Transfer compact selected captures once, then normalize in CNN chunks.
    cameras = batch['cam_images']
    images = cameras[torch.arange(len(cameras))[:, None], indices].to(device,non_blocking=True)
    if 'photometric' in batch:
        images = photometric_captures(images, batch['photometric'], chunk=cfg.VISUAL_CHUNK)
    inputs = dict(images=images, imu=batch['imu'].to(device), dt=batch['dt'].to(device),
                  gravity=batch['gravity'].to(device), ticks=ticks, observed=observed.to(device))
    truth = {key: batch[key].to(device).clone() for key in ('pos', 'vel', 'rot', 'omega')}
    truth['omega'][:, 0] = 0
    if 'camera_position_in_body' in batch:
        truth['camera_position_in_body'] = batch['camera_position_in_body'].to(device)
    return inputs, truth


def ground_truth_targets(batch, device, with_images=True):
    inputs, truth = frame_inputs(batch, device, with_images)
    return inputs, truth, None, None


def validation_inputs(batch, device, with_images=True):
    inputs, truth = frame_inputs(batch, device, with_images)
    metadata = dict(pos=truth['pos'], rot=truth['rot'], observed=batch['observed'], dt=batch['dt'])
    for key in ('episode_id', 'episode_complete', 'camera_position_in_body', 'camera_calibration_available'):
        if key in batch:
            metadata[key] = batch[key]
    return inputs, truth, metadata


def objective(model, prediction, inputs, truth, embedding_weight=.1, alignment_weight=.1):
    state, terms = losses(prediction, truth, inputs['imu'].device, metrics=False, components=True)
    index = torch.arange(len(inputs['imu']), device=inputs['imu'].device)[:, None]
    ticks = inputs['ticks']
    p = {key: prediction[key][index, ticks] for key in ('pos', 'vel', 'rot', 'omega')}
    target = {key: truth[key][index, ticks] for key in ('pos', 'vel', 'rot', 'omega')}
    if 'camera_position_in_body' in truth:
        target['camera_position_in_body'] = truth['camera_position_in_body']
    alignment, diagnostics = alignment_objective(p, target)
    jepa = model.jepa_loss(prediction, inputs['images'])
    coefficients = dict(embedding_prediction=1., embedding_content=1., embedding_variance=1.,
                        embedding_covariance=.01, auxiliary_prediction=1.)
    embedding = sum(jepa[key]*weight for key, weight in coefficients.items())
    validation = state+alignment_weight*alignment
    total = validation+embedding_weight*embedding
    terms = {'state_'+key: value.detach() if torch.is_tensor(value) else value for key, value in terms.items()}
    terms.update(state_loss=state.detach(), alignment_raw=alignment.detach(),
                 alignment_weight=alignment_weight, alignment_weighted=alignment.detach()*alignment_weight,
                 validation_loss=validation.detach(), total_loss=total.detach(),
                 embedding_loss=embedding.detach(), embedding_loss_weight=embedding_weight,
                 embedding_loss_weighted=embedding.detach()*embedding_weight, **diagnostics)
    for key, value in jepa.items():
        terms[key+'_raw'] = value.detach()
        terms[key+'_weight'] = embedding_weight*coefficients[key]
        terms[key+'_weighted'] = value.detach()*embedding_weight*coefficients[key]
    terms['innovation_update_rms'] = prediction['innovation_update'].detach().square().mean().sqrt()
    return total, validation, terms


def train_student(name, row, payload, device, weights=None, embedding_weight=.1,
                  embedding_ema=.99, alignment_weight=.1):
    inputs, truth = payload[:2]
    model = row['model'].train()
    row['optimizer'].zero_grad(set_to_none=True)
    prediction = model(**inputs)
    total, _, terms = objective(model, prediction, inputs, truth, embedding_weight, alignment_weight)
    if not torch.isfinite(total):
        raise ValueError(f'Nonfinite {name} loss')
    total.backward()
    row['gradient_norm_preclip'] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.).detach()
    terms['gradient_norm_preclip'] = row['gradient_norm_preclip']
    row['optimizer'].step()
    model.update_target(embedding_ema)
    row['loss_terms'] = terms
    return total.detach()


def validate_student(name, row, payload, device):
    with torch.no_grad(), amp_disabled():
        inputs, truth = payload[:2]
        model = row['model'].eval()
        prediction = model(**inputs)
        if len(payload)>2:
            row['validation_metric'].update(prediction, payload[2])
        _, validation, terms = objective(model, prediction, inputs, truth,
                                         row.get('embedding_weight', .1), row.get('alignment_weight', .1))
        row['validation_loss_terms'] = terms
        return validation
