"""Official-score parity, equal episode weighting and complete-trajectory validation."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import StateWindows,StateEpisodes
from leaderboard_metrics import LeaderboardATE,finish_ate
from ppo import validation_loss
from viobench.tools.runner.metrics import evaluate_trajectory


def trajectory_batch(distances):
    count=len(distances)
    pos=torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,1.]])[None].repeat(count,1,1)
    imu=torch.zeros(count,4,6);imu[:,:,:3]=pos
    imu[:,-1,0]+=torch.tensor(distances)
    return dict(pos=pos,vel=torch.zeros_like(pos),omega=torch.zeros_like(pos),
                rot=torch.eye(3).expand(count,4,3,3),imu=imu,
                observed=torch.ones(count,4,dtype=torch.bool),dt=torch.full((count,),.005),
                gravity=torch.zeros(count,3))


class PositionStudent(torch.nn.Module):
    def forward(self,frames,imu,observed,dt,gravity):
        pos=imu[:,:,:3]
        return dict(pos=pos,vel=torch.zeros_like(pos),omega=torch.zeros_like(pos),
                    rot=torch.eye(3).expand(len(pos),pos.shape[1],3,3))


def prediction(batch):
    return PositionStudent()(batch['pos'],batch['imu'],None,None,None)


def official_score(truth,predicted):
    n=len(truth);target=np.zeros((n,13));estimate=np.zeros_like(target)
    target[:,:3]=truth;estimate[:,:3]=predicted
    target[:,6]=estimate[:,6]=1.
    estimate[~np.isfinite(predicted).all(-1)]=np.nan
    return evaluate_trajectory(target,estimate,np.arange(n)*.05).sim3_ate_rmse


@pytest.mark.parametrize('kind',['nonrigid','similarity','constant','invalid'])
def test_mean_ate_matches_official_episode_scorer(kind):
    batch=trajectory_batch([.8]);pred=prediction(batch)
    if kind=='similarity':
        rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        pred['pos']=3*(batch['pos']@rotation.T)+torch.tensor([3.,4.,5.])
    elif kind=='constant':pred['pos'].fill_(7.)
    elif kind=='invalid':pred['pos'][0,2]=float('nan')
    metric=LeaderboardATE();metric.update(pred,batch)
    value=metric.result()['mean_ate_m']
    expected=official_score(batch['pos'][0].numpy(),pred['pos'][0].numpy())
    assert value==pytest.approx(expected,abs=1e-12)
    assert metric.result()['sim3_ate_rmse_mean']==value
    if kind=='similarity':assert value<1e-12
    if kind=='constant':assert value>0


def test_mean_is_equal_per_episode_not_pooled_rmse_or_frame_weighted():
    metric=LeaderboardATE();expected=[]
    for repeat,distance in ((1,.1),(2,1.)):
        batch=trajectory_batch([distance]);pred=prediction(batch)
        for key in ('pos','vel','omega','imu','rot','observed'):
            batch[key]=batch[key].repeat_interleave(repeat,dim=1)
        for key in pred:pred[key]=pred[key].repeat_interleave(repeat,dim=1)
        metric.update(pred,batch)
        expected.append(official_score(batch['pos'][0].numpy(),pred['pos'][0].numpy()))
    result=metric.result()
    assert result['ate_episodes']==2
    assert result['mean_ate_m']==pytest.approx(np.mean(expected),abs=1e-12)
    assert result['mean_ate_m']!=pytest.approx(np.sqrt(np.mean(np.square(expected))))
    assert result['mean_ate_m']!=pytest.approx(np.average(expected,weights=[4,8]))


def test_only_camera_timestamps_contribute_and_full_episodes_are_not_counted_twice():
    batch=trajectory_batch([.8]);batch['episode_id']=['flight-1']
    batch['observed'][0,1]=False
    pred=prediction(batch);pred['pos'][0,1]=1e6
    metric=LeaderboardATE();metric.update(pred,batch);metric.update(pred,batch)
    selected=batch['observed'][0].numpy()
    expected=official_score(batch['pos'][0].numpy()[selected],pred['pos'][0].numpy()[selected])
    assert metric.result()['ate_episodes']==1
    assert metric.result()['mean_ate_m']==pytest.approx(expected,abs=1e-12)


def test_camera_lever_arm_is_included():
    batch=trajectory_batch([.8]);pred=prediction(batch)
    lever=torch.tensor([[.3,-.1,.2]])
    batch['camera_position_in_body']=lever
    pred['rot']=pred['rot'].clone()
    pred['rot'][0,1]=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    metric=LeaderboardATE();metric.update(pred,batch)
    shifted=pred['pos'][0].double().numpy()+(pred['rot'][0].double().numpy()-np.eye(3))@lever[0].double().numpy()
    expected=official_score(batch['pos'][0].numpy(),shifted)
    assert metric.result()['mean_ate_m']==pytest.approx(expected,abs=1e-12)
    assert metric.result()['ate_camera_calibrated_episodes']==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_predictions_match_official_cpu_scoring():
    batch=trajectory_batch([.2,.8]);pred={key:value.cuda() for key,value in prediction(batch).items()}
    metric=LeaderboardATE();metric.update(pred,batch)
    expected=[official_score(batch['pos'][i].numpy(),pred['pos'][i].cpu().numpy()) for i in range(2)]
    assert metric.result()['mean_ate_m']==pytest.approx(np.mean(expected),abs=1e-12)


def test_partial_windows_cannot_be_reported_as_complete_episodes():
    batch=trajectory_batch([.8]);batch['episode_complete']=[False]
    metric=LeaderboardATE();metric.update(prediction(batch),batch)
    with pytest.raises(ValueError,match='complete validation episodes'):metric.result()


def test_ppo_validation_uses_equal_episode_mean_across_partial_batches(monkeypatch):
    import ppo
    calls=[]
    def frames(model,batch,device):
        calls.append(len(batch['pos']));return batch['pos']
    monkeypatch.setattr(ppo,'state_input_frames',frames)
    loader=[trajectory_batch([.1,.2]),trajectory_batch([.4])]
    score,metrics=validation_loss(None,PositionStudent(),loader,'cpu',return_metrics=True)
    assert calls==[2,1]
    expected=[official_score(batch['pos'][i].numpy(),prediction(batch)['pos'][i].numpy())
              for batch in loader for i in range(len(batch['pos']))]
    assert metrics['mean_ate_m']==pytest.approx(np.mean(expected),abs=1e-12)
    assert metrics['ate_episodes']==3
    assert validation_loss(None,PositionStudent(),loader,'cpu')==score


def episode_store(lengths):
    entries=[]
    for i,n in enumerate(lengths):
        state=torch.zeros(n,18);state[:,3:12]=torch.eye(3).flatten()
        state[:,0]=torch.linspace(0,2,n)
        entries.append(dict(file=f'flight-{i}',state=state,imu=torch.zeros(n,6),
            cams=torch.zeros((n-1)//5+1,3,12,16,dtype=torch.uint8),
            cam_ticks=torch.arange(0,n,5),stride=5,dt=.005))
    return SimpleNamespace(entries=entries)


def test_short_validation_windows_score_full_episodes_without_resetting_between_windows(monkeypatch):
    import ppo,leaderboard_metrics
    calls=[]
    def frames(model,batch,device):
        calls.append(batch['imu'].shape[1]);return batch['pos']
    monkeypatch.setattr(ppo,'state_input_frames',frames)
    monkeypatch.setattr(leaderboard_metrics,'state_input_frames',frames)
    store=episode_store([41,61])
    windows=DataLoader(StateWindows(store,length=11,training=False),batch_size=2)
    episodes=DataLoader(StateEpisodes(store),batch_size=1)
    score,metrics=validation_loss(None,PositionStudent(),windows,'cpu',return_metrics=True,episode_loader=episodes)
    assert metrics['ate_episodes']==2 and np.isfinite(metrics['mean_ate_m'])
    assert calls[-2:]==[41,61]
    assert all(n==11 for n in calls[:-2])


def test_complete_validation_predictions_avoid_extra_data_loading_or_inference():
    store=episode_store([41]);dataset=StateEpisodes(store)
    batch=next(iter(DataLoader(dataset,batch_size=1)))
    metric=LeaderboardATE();metric.update(prediction(batch),batch)
    class NoLoading:
        def __init__(self):self.dataset=dataset
        def __iter__(self):raise AssertionError('Complete predictions must be reused')
    result=finish_ate(None,None,'cpu',metric,NoLoading())
    assert result['ate_episodes']==1 and np.isfinite(result['mean_ate_m'])
