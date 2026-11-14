"""Trajectory distance semantics and correctly weighted validation tracking."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from losses import losses,mean_ate
from ppo import validation_loss


def trajectory_batch(distances):
    count=len(distances)
    pos=torch.zeros(count,2,3)
    imu=torch.zeros(count,2,6)
    imu[:,:,0]=torch.tensor(distances)[:,None]
    return dict(pos=pos,vel=torch.zeros_like(pos),omega=torch.zeros_like(pos),
                rot=torch.eye(3).expand(count,2,3,3),imu=imu,
                observed=torch.ones(count,2,dtype=torch.bool),dt=torch.full((count,),.005),
                gravity=torch.zeros(count,3))


class PositionStudent(torch.nn.Module):
    def forward(self,frames,imu,observed,dt,gravity):
        pos=imu[:,:,:3]
        return dict(pos=pos,vel=torch.zeros_like(pos),omega=torch.zeros_like(pos),
                    rot=torch.eye(3).expand(len(pos),2,3,3))


def test_mean_ate_is_mean_euclidean_distance_without_alignment():
    predicted=torch.tensor([[[3.,4.,0.],[0.,0.,0.]],[[0.,0.,12.],[0.,0.,0.]]])
    truth=torch.zeros_like(predicted)
    assert float(mean_ate(predicted,truth))==pytest.approx(17/4)
    # A constant position offset must remain visible rather than being aligned away.
    assert float(mean_ate(truth+torch.tensor([3.,4.,0.]),truth))==5.


def test_state_metric_does_not_change_loss_and_differs_from_coordinate_rmse():
    batch=trajectory_batch([5.,0.])
    prediction=PositionStudent()(batch['pos'],batch['imu'],None,None,None)
    value,metrics=losses(prediction,batch,'cpu')
    without_metrics,empty=losses(prediction,batch,'cpu',metrics=False)
    torch.testing.assert_close(value,without_metrics,rtol=0,atol=0)
    assert empty=={}
    assert metrics['mean_ate_m']==2.5
    assert metrics['pos_rmse']!=metrics['mean_ate_m']


def test_ppo_validation_weights_partial_batch_and_reuses_predictions(monkeypatch):
    import ppo
    calls=[]
    def frames(model,batch,device):
        calls.append(len(batch['pos']))
        return batch['pos']
    monkeypatch.setattr(ppo,'state_input_frames',frames)
    loader=[trajectory_batch([5.,0.]),trajectory_batch([10.])]
    score,metrics=validation_loss(None,PositionStudent(),loader,'cpu',return_metrics=True)
    assert calls==[2,1]  # one forward pass per existing validation batch
    assert metrics['mean_ate_m']==5.  # (5 + 0 + 10) / 3, not average of batch means
    assert validation_loss(None,PositionStudent(),loader,'cpu')==score


def test_ppo_validation_rejects_empty_loader_with_metrics():
    with pytest.raises(ValueError,match='No PPO validation windows'):
        validation_loss(None,PositionStudent(),[],'cpu',return_metrics=True)
