"""Full IMU-rate supervision and camera-rate alignment/JEPA callbacks."""
import legacy
from contextlib import nullcontext
import torch
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
    corrected = model.settings.get('training_policy') == 'corrected_v1'
    pred_state = {key: prediction[key][:, 1:] for key in ('pos', 'vel', 'rot', 'omega')} if corrected else prediction
    truth_state = {key: truth[key][:, 1:] for key in ('pos', 'vel', 'rot', 'omega')} if corrected else truth
    state, terms = losses(pred_state, truth_state, inputs['imu'].device, metrics=False, components=True)
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
    corr = prediction.get('corrections')
    if corr is not None:
        terms['correction_saturation_fraction'] = (corr.detach().abs() >= .2475).float().mean()
    rotation = prediction['rot'].detach()
    eye = torch.eye(3, device=rotation.device)
    terms['rotation_orthogonality_rmse'] = ((rotation.transpose(-1,-2)@rotation-eye).square().mean()).sqrt()
    for key in ('visual_velocity_update', 'visual_orientation_update'):
        if key in prediction:terms[key+'_rms'] = prediction[key].detach().square().mean().sqrt()
    return total, validation, terms


def train_student(name, row, payload, device, weights=None, embedding_weight=.1,
                  embedding_ema=.99, alignment_weight=.1):
    inputs, truth = payload[:2]
    model = row['model'].train()
    optimizer = row['optimizer']
    # Do not let a bad backward pass poison Adam's moments, weights, or EMA.
    # Independent student workers use context-local precision controls.
    for attempt in range(2):
        optimizer.zero_grad(set_to_none=True)
        with (amp_disabled() if attempt else nullcontext()):
            prediction = model(**inputs)
            total, _, terms = objective(model, prediction, inputs, truth, embedding_weight, alignment_weight)
        if torch.isfinite(total):
            with (amp_disabled() if attempt else nullcontext()):
                total.backward()
            for module_name, module in model.named_children():
                grads = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
                if grads:
                    terms['gradient_norm_'+module_name] = torch.stack(grads).norm()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.).detach()
            if torch.isfinite(norm):
                optimizer.step()
                model.update_target(embedding_ema)
                row['gradient_norm_preclip'] = norm
                terms.update(gradient_norm_preclip=norm, gradient_clipped=(norm>1.).float(), numerical_retry=float(attempt),
                             optimizer_update_skipped=0.)
                row['loss_terms'] = terms
                return total.detach()
        # Retry before any mutation of optimizer/EMA state. Release the first
        # graph before allocating the full-precision retry on long horizons.
        finite_loss = bool(torch.isfinite(total))
        result = total.detach()
        del prediction, total
        optimizer.zero_grad(set_to_none=True)
        if attempt == 0:
            print(f'Numerical recovery: {name}: retrying minibatch in FP32', flush=True)
            continue
        if not finite_loss:
            bad = [key for key, value in terms.items()
                   if torch.is_tensor(value) and not torch.isfinite(value).all()]
            raise ValueError(f'Nonfinite {name} loss after FP32 retry; components: {bad}')
        # Even FP32 can produce an invalid backward pass at a singularity.
        # Report the finite evaluated loss, but explicitly log that no update
        # was made. Never report a clipped NaN/Inf norm as a valid measurement.
        print(f'Numerical recovery: {name}: skipped update after nonfinite gradients in FP32', flush=True)
        row['gradient_norm_preclip'] = result.new_zeros(())
        terms.pop('gradient_norm_preclip', None)
        terms.update(numerical_retry=1., optimizer_update_skipped=1.)
        row['loss_terms'] = terms
        return result


def curriculum_squared_errors(prediction,truth):
    """Mean squared physical errors after initialization, independent of loss units."""
    errors={name:(prediction[key][:,1:]-truth[key][:,1:]).square().sum(-1).mean()
            for name,key in (('position_rmse_m','pos'),('velocity_rmse_mps','vel'),
                             ('angular_velocity_rmse_radps','omega'))}
    relative=truth['rot'][:,1:].transpose(-1,-2)@prediction['rot'][:,1:]
    cosine=((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
    errors['orientation_rmse_deg']=torch.rad2deg(torch.acos(cosine)).square().mean()
    return {key:float(value) for key,value in errors.items()}


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
        if len(payload)==2:
            row['curriculum_squared_errors']=curriculum_squared_errors(prediction,truth)
        return validation
