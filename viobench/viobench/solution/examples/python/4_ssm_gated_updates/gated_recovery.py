"""Atomic latest snapshots, including optimizer and random generators."""
import random
import json
import numpy as np
import torch
from pathlib import Path
from resource_limits import check_write
from distillation_checkpoints import StageBestCheckpoints, cpu_snapshot


def tensor_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value,(tuple,list)):return sum(tensor_bytes(item) for item in value)
    return 0


def save_weights(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    check_write(path,planned=tensor_bytes(value)+2**20)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class SafeStageBestCheckpoints(StageBestCheckpoints):
    def observe(self, students, scores, epoch):
        import math
        if self.stage is None:raise RuntimeError('Begin a stage before saving it')
        for name,row in students.items():
            score=scores[name]
            if not math.isfinite(score) or score<0:raise ValueError(f'Invalid {name} validation ATE')
            if name in self.best and score>=self.best[name]['validation_mean_ate_m']:continue
            checkpoint=dict(stage=cpu_snapshot(self.stage),stage_index=self.index,epoch=epoch,
                validation_mean_ate_m=float(score),model=cpu_snapshot(row['model'].state_dict()),
                optimizer=cpu_snapshot(row['optimizer'].state_dict()))
            save_weights(checkpoint,row['directory']/'stage_best.pt')
            self.best[name]=dict(epoch=epoch,validation_mean_ate_m=float(score))


def save_latest(directory, students, recovery=None, experiment=None):
    """One atomic file for weights, optimizers, RNG and resume metadata.

    Embedding the recovery/experiment state means an interruption can never
    leave weights and counters from different epochs. JSON round-tripping keeps
    the embedded copy identical to the recovery_state.json mirror.
    """
    state = dict(completed_epochs=max(row.get('completed_epoch',0) for row in students.values()),
                 models={name: dict(model=row['model'].state_dict(), optimizer=row['optimizer'].state_dict())
                        for name, row in students.items()},
                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 python_rng=random.getstate(), numpy_rng=np.random.get_state())
    if recovery is not None:
        state['recovery'] = json.loads(json.dumps(recovery, allow_nan=False))
        state['experiment'] = dict(experiment or {})
    save_weights(state, directory/'latest.pt')


def embedded_recovery(directory):
    """(recovery, experiment) stored in latest.pt, or (None, None) for older runs."""
    path = Path(directory)/'latest.pt'
    if not path.is_file():
        return None, None
    state = torch.load(path, map_location='cpu', weights_only=False)
    return state.get('recovery'), state.get('experiment')


def restore_latest(directory, students):
    # Local trusted experiment checkpoint contains Python/NumPy RNG tuples.
    state = torch.load(directory/'latest.pt', map_location='cpu', weights_only=False)
    progress = directory/'recovery_state.json'
    # Newer checkpoints carry their own metadata; the JSON mirror may lag them.
    if ('recovery' not in state and progress.exists() and
            state['completed_epochs'] != json.loads(progress.read_text())['completed_epochs']):
        raise ValueError('Latest checkpoint and recovery metadata differ; interrupted save cannot be resumed safely')
    for name, row in students.items():
        row['model'].load_state_dict(state['models'][name]['model'])
        row['optimizer'].load_state_dict(state['models'][name]['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    if torch.cuda.is_available() and state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
