"""Atomic latest snapshots, including optimizer and random generators."""
import random
import json
import numpy as np
import torch
from pathlib import Path


def save_weights(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def save_latest(directory, students):
    save_weights(dict(completed_epochs=max(row.get('completed_epoch',0) for row in students.values()),
                      models={name: dict(model=row['model'].state_dict(), optimizer=row['optimizer'].state_dict())
                             for name, row in students.items()},
                      torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                      python_rng=random.getstate(), numpy_rng=np.random.get_state()), directory/'latest.pt')


def restore_latest(directory, students):
    # Local trusted experiment checkpoint contains Python/NumPy RNG tuples.
    state = torch.load(directory/'latest.pt', map_location='cpu', weights_only=False)
    progress = directory/'recovery_state.json'
    if progress.exists() and state['completed_epochs'] != json.loads(progress.read_text())['completed_epochs']:
        raise ValueError('Latest checkpoint and recovery metadata differ; interrupted save cannot be resumed safely')
    for name, row in students.items():
        row['model'].load_state_dict(state['models'][name]['model'])
        row['optimizer'].load_state_dict(state['models'][name]['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    if torch.cuda.is_available() and state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
