"""Bounded stage-best model/optimizer snapshots for curriculum transitions."""
from copy import deepcopy
import math

import torch

from train import save_weights


def cpu_snapshot(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {key:cpu_snapshot(item) for key,item in value.items()}
    if isinstance(value,list):return [cpu_snapshot(item) for item in value]
    if isinstance(value,tuple):return tuple(cpu_snapshot(item) for item in value)
    return deepcopy(value)


class StageBestCheckpoints:
    """Select the best completed validation epoch within the current stage.

    Each profile keeps one atomic checkpoint, overwritten on improvement.
    Global best export weights are managed separately by the distiller.
    """
    def __init__(self):
        self.stage=None
        self.index=0
        self.best={}

    def recover(self,sources):
        if not sources:return
        first=next(iter(sources.values()))
        if any(value['stage']!=first['stage'] or value['stage_index']!=first['stage_index'] for value in sources.values()):
            raise ValueError('Stage recovery checkpoints disagree on their curriculum stage')
        self.stage=deepcopy(first['stage']);self.index=first['stage_index']
        self.best={name:dict(epoch=value['epoch'],validation_mean_ate_m=value['validation_mean_ate_m'])
                   for name,value in sources.items()}

    @staticmethod
    def synchronize(students):
        devices={next(row['model'].parameters()).device for row in students.values()}
        for device in devices:
            if device.type=='cuda':torch.cuda.synchronize(device)

    def begin(self,stage,students,completed_epochs):
        if stage==self.stage:return None
        if self.stage is None:
            self.stage=deepcopy(stage)
            return None
        missing=set(students)-set(self.best)
        if missing:raise RuntimeError(f'Missing current-stage best checkpoints for {sorted(missing)}')
        self.synchronize(students)
        restored={}
        for name,row in students.items():
            path=row['directory']/'stage_best.pt'
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            if checkpoint['stage']!=self.stage or checkpoint['stage_index']!=self.index:
                raise RuntimeError(f'{name} stage checkpoint belongs to a different curriculum stage')
            row['optimizer'].zero_grad(set_to_none=True)
            row['model'].load_state_dict(checkpoint['model'])
            row['optimizer'].load_state_dict(checkpoint['optimizer'])
            restored[name]=dict(epoch=checkpoint['epoch'],validation_mean_ate_m=checkpoint['validation_mean_ate_m'],
                                checkpoint=str(path),optimizer_restored=True)
        # Copies and newly loaded optimizer buffers must be visible on every
        # worker stream before the next update, not just the caller's stream.
        self.synchronize(students)
        report=dict(from_stage_index=self.index,to_stage_index=self.index+1,
                    from_stage=self.stage,to_stage=deepcopy(stage),
                    completed_epochs=completed_epochs,students=restored)
        self.stage=deepcopy(stage);self.index+=1;self.best={}
        return report

    def observe(self,students,scores,epoch):
        if self.stage is None:raise RuntimeError('Begin a curriculum stage before recording its best checkpoint')
        for name,row in students.items():
            score=scores[name]
            if not math.isfinite(score) or score<0:raise ValueError(f'Invalid {name} stage validation ATE')
            if name in self.best and score>=self.best[name]['validation_mean_ate_m']:continue
            checkpoint=dict(stage=deepcopy(self.stage),stage_index=self.index,epoch=epoch,
                validation_mean_ate_m=float(score),model=cpu_snapshot(row['model'].state_dict()),
                optimizer=cpu_snapshot(row['optimizer'].state_dict()))
            save_weights(checkpoint,row['directory']/'stage_best.pt')
            self.best[name]=dict(epoch=epoch,validation_mean_ate_m=float(score))
