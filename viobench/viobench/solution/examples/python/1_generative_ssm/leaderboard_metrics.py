"""Validation Mean Sim(3) ATE, using VIOBench's own per-episode scorer."""
import numpy as np
import torch

from data import state_input_frames
from performance import amp_disabled
from viobench.tools.runner.metrics import evaluate_trajectory,fill_missing_states

ATE_CONFIG=dict(mean_ate_metric='sim3_ate_rmse_mean',mean_ate_aggregation='equal_per_episode',
                mean_ate_sampling='camera_timestamps',mean_ate_scope='complete_validation_recordings',
                mean_ate_inference='pytorch_validation',
                unaligned_mte_metric='mean_euclidean_position_error_m',
                unaligned_mte_aggregation='equal_per_episode',unaligned_mte_alignment='none')


def episode_key(value):
    return str(value.item() if isinstance(value,torch.Tensor) else value)


class LeaderboardATE:
    """Score each complete episode once; never average independently reset windows."""
    def __init__(self):
        self.scores={}
        self.unaligned_mte={}
        self.unaligned_ate_rmse={}
        self.expected=set()
        self.calibrated=set()
        self.counter=0

    @torch.no_grad()
    def update(self,prediction,batch):
        count=len(prediction['pos'])
        names=batch.get('episode_id',[f'batch-{self.counter}-{i}' for i in range(count)])
        names=[episode_key(name) for name in names]
        self.counter+=1
        complete=batch.get('episode_complete',[True]*count)
        self.expected.update(names)
        for i,name in enumerate(names):
            if not bool(complete[i]) or name in self.scores:continue
            observed=batch['observed'][i].detach().cpu().numpy().astype(bool)
            predicted=prediction['pos'][i].detach().double().cpu().numpy()[observed]
            truth=batch['pos'][i].detach().double().cpu().numpy()[observed]
            rotations=prediction['rot'][i].detach().double().cpu().numpy()[observed]
            truth_rot=batch['rot'][i].detach().double().cpu().numpy()[observed]
            offset=batch.get('camera_position_in_body')
            if offset is not None:
                lever=offset[i].detach().double().cpu().numpy()
                predicted=predicted+(rotations-np.eye(3))@lever
                truth=truth+(truth_rot-np.eye(3))@lever
            calibrated=batch.get('camera_calibration_available',[offset is not None]*count)
            if bool(calibrated[i]):self.calibrated.add(name)
            # A common rotation from initial body axes to initial camera axes
            # cannot change a Sim(3)-aligned position RMSE. Preserve lever arms.
            n=len(truth)
            target=np.zeros((n,13),np.float64);estimate=np.zeros_like(target)
            target[:,:3]=truth;estimate[:,:3]=predicted
            target[:,6]=estimate[:,6]=1.
            valid=np.isfinite(predicted).all(-1)&np.isfinite(rotations).all((1,2))
            estimate[~valid]=np.nan  # official scorer fills invalid states with origin/identity
            times=np.flatnonzero(observed)*float(batch['dt'][i])
            scored=evaluate_trajectory(target,estimate,times)
            self.scores[name]=scored.sim3_ate_rmse
            self.unaligned_ate_rmse[name]=scored.ate_rmse
            filled=fill_missing_states(estimate)
            self.unaligned_mte[name]=float(np.linalg.norm(filled[:,:3]-target[:,:3],axis=-1).mean())

    def result(self):
        if not self.scores or self.expected-set(self.scores):
            raise ValueError('Mean Sim(3) ATE requires complete validation episodes')
        value=float(np.mean(list(self.scores.values())))
        return dict(mean_ate_m=value,sim3_ate_rmse_mean=value,
                    unaligned_mte_m=float(np.mean(list(self.unaligned_mte.values()))),
                    unaligned_ate_rmse_m=float(np.mean(list(self.unaligned_ate_rmse.values()))),ate_episodes=len(self.scores),
                    ate_camera_calibrated_episodes=len(self.calibrated))


@torch.no_grad()
def finish_ate(frame_model,student,device,metric,episode_loader=None):
    """Reuse complete validation predictions, evaluating only missing episodes."""
    if episode_loader is not None:
        dataset=getattr(episode_loader,'dataset',None)
        if dataset is not None:
            metric.expected.update(str(i) for i in range(len(dataset.store.entries)))
        if metric.expected and metric.expected<=set(metric.scores):return metric.result()
        with amp_disabled():
            for batch in episode_loader:
                if all(episode_key(name) in metric.scores for name in batch['episode_id']):continue
                frames=state_input_frames(frame_model,batch,device)
                prediction=student(frames,*(batch[k].to(device,non_blocking=True)
                    for k in ('imu','observed','dt','gravity')))
                metric.update(prediction,batch)
    return metric.result()
