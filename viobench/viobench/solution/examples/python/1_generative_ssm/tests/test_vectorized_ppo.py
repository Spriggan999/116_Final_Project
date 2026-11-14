"""Vector PPO replays fresh current-policy actions and preserves bootstrap semantics."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from ppo import advantages,physical_step,vectorized_batch_rollout


@pytest.mark.parametrize('gamma,lam',[(0.,0.),(.995,0.),(.995,.95),(1.,1.)])
def test_parallel_gae_matches_recurrence_and_bootstrap(gamma,lam):
    torch.manual_seed(17)
    rewards=torch.randn(2,1601);values=torch.randn_like(rewards);bootstrap=torch.randn(2)
    expected=advantages(rewards,values,gamma,lam,bootstrap)
    actual=advantages(rewards,values,gamma,lam,bootstrap,parallel=True)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-5)


def batch_and_tokens(device='cpu'):
    torch.manual_seed(1701)
    count,ticks,width=2,31,8
    tokens=torch.randn(count,ticks,width)
    imu=torch.randn(count,ticks,6)*.2;imu[:,:,2]+=9.81
    batch=dict(imu=imu,dt=torch.full((count,),.005),gravity=torch.tensor([[0.,0.,-9.81]]).repeat(count,1),
               pos=torch.zeros(count,ticks,3),vel=torch.zeros(count,ticks,3),omega=imu[:,:,3:6].clone(),
               rot=torch.eye(3)[None,None].repeat(count,ticks,1,1))
    student=SimpleNamespace(out=torch.nn.Linear(width,9))
    critic=torch.nn.Sequential(torch.nn.Linear(width+30,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
    cfg=SimpleNamespace(PPO_STD=.1,PPO_GAMMA=.9995,PPO_GAE_LAMBDA=.95)
    student.out.to(device);critic.to(device)
    return tokens.to(device),{k:v.to(device) for k,v in batch.items()},student,critic,cfg


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_vector_rollout_replays_recorded_actions_with_correct_pre_step_critic_inputs(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    tokens,batch,student,critic,cfg=batch_and_tokens(device)
    result=vectorized_batch_rollout(student,critic,tokens,batch,cfg)
    position=torch.zeros(2,3,device=device);velocity=torch.zeros_like(position);rotation=torch.eye(3,device=device)[None].repeat(2,1,1)
    rewards=[];values=[]
    with torch.no_grad():
        for t in range(1,tokens.shape[1]-1):
            inputs=torch.cat([tokens[:,t],position,velocity,rotation.flatten(1),batch['pos'][:,t],
                              batch['vel'][:,t],batch['rot'][:,t].flatten(1)],-1)
            torch.testing.assert_close(result['critic_inputs'][:,t-1],inputs,atol=2e-6,rtol=1e-5)
            values.append(critic(inputs).squeeze(-1))
            position,velocity,rotation,omega=physical_step(position,velocity,rotation,batch['imu'][:,t],
                result['actions'][:,t-1],batch['dt'],batch['gravity'])
            error=((position-batch['pos'][:,t]).square().sum(-1)+.1*(velocity-batch['vel'][:,t]).square().sum(-1)
                   +.05*(rotation-batch['rot'][:,t]).square().sum((1,2))+.01*(omega-batch['omega'][:,t]).square().sum(-1))
            rewards.append(-error)
        final=torch.cat([tokens[:,-1],position,velocity,rotation.flatten(1),batch['pos'][:,-1],
                         batch['vel'][:,-1],batch['rot'][:,-1].flatten(1)],-1)
        bootstrap=critic(final).squeeze(-1)
    values=torch.stack(values,1);rewards=torch.stack(rewards,1)
    torch.testing.assert_close(result['bootstrap'],bootstrap,atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(result['rewards'],rewards,atol=2e-6,rtol=1e-5)
    expected=advantages(rewards,values,cfg.PPO_GAMMA,cfg.PPO_GAE_LAMBDA,bootstrap)
    for actual,reference in zip((result['advantages'],result['returns']),expected):
        torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
    means=student.out(result['tokens'])
    distribution=torch.distributions.Normal(means,cfg.PPO_STD)
    torch.testing.assert_close(result['log_probs'],distribution.log_prob(result['actions']).sum(-1))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_vector_actor_does_not_observe_privileged_truth(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    tokens,batch,student,critic,cfg=batch_and_tokens(device)
    torch.manual_seed(17);first=vectorized_batch_rollout(student,critic,tokens,batch,cfg)
    changed=dict(batch,pos=batch['pos']+10)
    torch.manual_seed(17);second=vectorized_batch_rollout(student,critic,tokens,changed,cfg)
    torch.testing.assert_close(second['actions'],first['actions'],atol=0,rtol=0)
    assert not torch.equal(first['returns'],second['returns'])
