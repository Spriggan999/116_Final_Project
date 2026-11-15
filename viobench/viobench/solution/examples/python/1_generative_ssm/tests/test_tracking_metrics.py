"""Trajectory distance semantics and correctly weighted validation tracking."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from losses import losses
from ppo import validation_loss


def trajectory_batch(distances):
    count=len(distances)
    pos=torch.zeros(count,3,3)
    imu=torch.zeros(count,3,6)
    imu[:,:,0]=torch.tensor(distances)[:,None]
    return dict(pos=pos,vel=torch.zeros_like(pos),omega=torch.zeros_like(pos),
                rot=torch.eye(3).expand(count,3,3,3),imu=imu,
                observed=torch.ones(count,3,dtype=torch.bool),dt=torch.full((count,),.005),
                gravity=torch.zeros(count,3))


class PositionStudent(torch.nn.Module):
    def forward(self,frames,imu,observed,dt,gravity):
        pos=imu[:,:,:3]
        return dict(pos=pos,vel=torch.zeros_like(pos),omega=torch.zeros_like(pos),
                    rot=torch.eye(3).expand(len(pos),pos.shape[1],3,3))


def test_unaligned_mean_ate_is_removed_without_changing_loss():
    batch=trajectory_batch([5.,0.])
    prediction=PositionStudent()(batch['pos'],batch['imu'],None,None,None)
    value,metrics=losses(prediction,batch,'cpu')
    without_metrics,empty=losses(prediction,batch,'cpu',metrics=False)
    torch.testing.assert_close(value,without_metrics,rtol=0,atol=0)
    assert empty=={}
    assert set(metrics)=={'pos_rmse','vel_rmse','rot_deg','omega_rmse'}


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
    # Constant ground-truth positions have zero error after Sim(3) alignment.
    assert metrics['mean_ate_m']==0.
    assert metrics['ate_episodes']==3
    assert validation_loss(None,PositionStudent(),loader,'cpu')==score


def test_ppo_validation_rejects_empty_loader_with_metrics():
    with pytest.raises(ValueError,match='No PPO validation windows'):
        validation_loss(None,PositionStudent(),[],'cpu',return_metrics=True)


def test_unaligned_mte_is_mean_distance_and_distinct_from_rmse():
    from leaderboard_metrics import LeaderboardATE
    batch=trajectory_batch([0.]);pred=PositionStudent()(batch['pos'],batch['imu'],None,None,None)
    pred['pos'][0,:,0]=torch.tensor([0.,3.,9.])
    metric=LeaderboardATE();metric.update(pred,batch);result=metric.result()
    assert result['unaligned_mte_m']==pytest.approx(4.)
    assert result['unaligned_ate_rmse_m']==pytest.approx(30**.5)
    assert result['mean_ate_m']==0.


def test_unaligned_mte_uses_camera_samples_and_equal_episode_weighting():
    from leaderboard_metrics import LeaderboardATE
    metric=LeaderboardATE()
    first=trajectory_batch([3.]);first['episode_id']=['one']
    first['observed'][0,1]=False
    pred=PositionStudent()(first['pos'],first['imu'],None,None,None)
    pred['pos'][0,1,0]=1000.
    metric.update(pred,first);metric.update(pred,first)
    second=trajectory_batch([9.]);second['episode_id']=['two']
    metric.update(PositionStudent()(second['pos'],second['imu'],None,None,None),second)
    assert metric.result()['unaligned_mte_m']==6.
    assert metric.result()['unaligned_ate_rmse_m']==6.


def test_unaligned_mte_includes_camera_lever_arm_and_invalid_state_filling():
    from leaderboard_metrics import LeaderboardATE
    batch=trajectory_batch([0.]);batch['camera_position_in_body']=torch.tensor([[1.,0.,0.]])
    pred=PositionStudent()(batch['pos'],batch['imu'],None,None,None)
    pred['rot']=pred['rot'].clone();pred['rot'][0,1]=torch.diag(torch.tensor([-1.,-1.,1.]))
    pred['pos'][0,2]=float('nan')
    metric=LeaderboardATE();metric.update(pred,batch)
    assert metric.result()['unaligned_mte_m']==pytest.approx(2/3)
