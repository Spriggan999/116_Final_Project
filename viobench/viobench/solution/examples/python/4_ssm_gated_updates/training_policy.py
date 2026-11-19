"""Versioned experiment controls; no changes to legacy experiment defaults."""
import json
import math
from pathlib import Path

LEGACY = 'legacy'
CORRECTED = 'corrected_v1'


def apply_learning_rate(students, resume_lr=None, schedule=None, horizon=None):
    """Call after every restore. Explicit override wins over a horizon schedule."""
    rate = resume_lr
    if rate is None and schedule:
        rate = schedule[0]['lr']
        for row in schedule:
            if horizon >= row['horizon_seconds']:
                rate = row['lr']
    if rate is not None:
        for row in students.values():
            for group in row['optimizer'].param_groups:
                group['lr'] = rate


def read_lr_schedule(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('LR schedule must be a nonempty list of horizon_seconds/lr records')
    previous = -1.
    for row in rows:
        h, lr = row['horizon_seconds'], row['lr']
        if not math.isfinite(h) or h <= previous or h < 0 or not math.isfinite(lr) or lr <= 0:
            raise ValueError('LR schedule horizons must increase; horizons/rates must be finite and positive')
        previous = h
    return rows


def physical_selection_score(accuracy, targets):
    return sum((accuracy[key]/targets[key])**2 for key in targets)/len(targets)


def architecture_settings(profile, architecture, velocity_bound=.5, orientation_bound=.02):
    from gated_models import PROFILES
    return dict(PROFILES[profile], architecture=architecture,
                velocity_update_bound=velocity_bound, orientation_update_bound=orientation_bound)


def load_model_weights(model, state, transfer=False):
    """Explicit v1->v2 transfer only; never silently accept unrelated missing keys."""
    if not transfer:
        model.load_state_dict(state)
        return
    if model.settings.get('architecture') != 'persistent_visual_v2':
        raise ValueError('Architecture transfer only supports legacy -> persistent_visual_v2')
    if any(key.startswith(('visual_gru.', 'velocity_update.', 'orientation_update.')) for key in state):
        raise ValueError('Transfer source must be a legacy architecture')
    result = model.load_state_dict(state, strict=False)
    allowed = ('visual_gru.', 'velocity_update.', 'orientation_update.')
    if result.unexpected_keys or any(not key.startswith(allowed) for key in result.missing_keys):
        raise ValueError(f'Incompatible transfer: {result}')
