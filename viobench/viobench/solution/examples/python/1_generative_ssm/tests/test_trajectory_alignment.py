"""Aligned trajectory loss matches the scorer and retains stable gradients."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trajectory_alignment import alignment_objective
from viobench.tools.runner.metrics import evaluate_trajectory


def fixture(kind='general',device='cpu'):
    generator=torch.Generator().manual_seed(17)
    target=torch.randn(2,13,3,generator=generator)
    predicted=target*.4+torch.tensor([2.,-1.,.5])
    if kind=='general':predicted+=torch.randn(predicted.shape,generator=generator)*.1
    elif kind=='constant':predicted[:]=0.
    elif kind=='line':
        target[:,:,1:]=0.;predicted=target*.5+1.
    elif kind=='stationary':target[:]=0.
    predicted=predicted.to(device).requires_grad_();target=target.to(device)
    rotation=torch.eye(3,device=device)[None,None].expand(2,13,3,3)
    return dict(pos=predicted,rot=rotation),dict(pos=target,rot=rotation)


@pytest.mark.parametrize('kind',['general','similarity','constant','line','stationary'])
def test_matches_official_complete_trajectory_scorer_and_has_finite_gradients(kind):
    prediction,truth=fixture(kind)
    loss,terms=alignment_objective(prediction,truth)
    expected=[]
    for p,t in zip(prediction['pos'].detach().numpy(),truth['pos'].numpy()):
        estimated=np.zeros((len(p),13));target=np.zeros_like(estimated)
        estimated[:,:3]=p;target[:,:3]=t;estimated[:,6]=target[:,6]=1.
        expected.append(evaluate_trajectory(target,estimated,np.arange(len(p))*.05).sim3_ate_rmse)
    assert loss.item()==pytest.approx(np.mean(expected),abs=2e-6)
    loss.backward()
    assert torch.isfinite(prediction['pos'].grad).all()
    if kind=='general':assert prediction['pos'].grad.abs().sum()>0
    assert all(torch.isfinite(value) for value in terms.values())


def test_detached_minimizing_fit_matches_finite_difference_gradient():
    prediction,truth=fixture()
    loss,_=alignment_objective(prediction,truth);loss.backward()
    expected=prediction['pos'].grad[0,3,1].item()
    estimates=[]
    for delta in [-.0001,.0001]:
        perturbed=prediction['pos'].detach().clone();perturbed[0,3,1]+=delta
        estimates.append(alignment_objective(dict(prediction,pos=perturbed),truth)[0].item())
    assert expected==pytest.approx((estimates[1]-estimates[0])/.0002,abs=2e-4,rel=.02)


def test_camera_lever_arm_changes_loss_and_short_windows_are_reported():
    prediction,truth=fixture()
    before=alignment_objective(prediction,truth)[0]
    prediction['rot']=prediction['rot'].clone()
    prediction['rot'][0,4]=torch.diag(torch.tensor([-1.,-1.,1.]))
    truth['camera_position_in_body']=torch.tensor([[.3,-.1,.2],[.3,-.1,.2]])
    after=alignment_objective(prediction,truth)[0]
    assert abs(after-before)>1e-4
    p={key:value[:,:2] for key,value in prediction.items()}
    t={key:value[:,:2] if key!='camera_position_in_body' else value for key,value in truth.items()}
    loss,terms=alignment_objective(p,t)
    assert loss==0 and terms['alignment_insufficient_samples_fraction']==1


def test_auxiliary_loss_is_logged_and_reconstructs_total():
    from test_student_embedding import model_inputs
    from distill import train_student
    model,inputs=model_inputs()
    with torch.no_grad():prediction=model(**inputs)
    truth={key:torch.zeros_like(prediction[key]) for key in ['pos','vel','omega']}
    truth['rot']=torch.eye(3)[None,None].expand_as(prediction['rot'])
    row=dict(model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=.001))
    total=train_student('1k',row,(inputs,truth,None,None),'cpu',(0.,0.,1.),alignment_weight=.2)
    parts=row['loss_terms']
    expected=parts['state_loss_weighted']+parts['embedding_loss_weighted']+parts['alignment_weighted']
    torch.testing.assert_close(total.detach(),expected)
    assert parts['alignment_weight']==.2


def test_cuda_loss_and_gradient_agree_with_cpu():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    cpu_pred,cpu_truth=fixture();gpu_pred,gpu_truth=fixture(device='cuda')
    cpu,_=alignment_objective(cpu_pred,cpu_truth);gpu,_=alignment_objective(gpu_pred,gpu_truth)
    cpu.backward();gpu.backward()
    torch.testing.assert_close(cpu,gpu.cpu(),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(cpu_pred['pos'].grad,gpu_pred['pos'].grad.cpu(),atol=2e-6,rtol=2e-4)
