"""Seven-profile contracts, exact sensor integration and website area accounting."""
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import sys

import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from students.model import BudgetStudent,PROFILES,integrate,integrate_parallel,qmatrix
from distill import curriculum_window,train_student
from distillation_pipeline import DistillationPipeline
from evaluate_students import score_portfolio
from benchmark_students import project_times
from performance import amp_disabled


def test_parallel_physics_matches_sequential_outputs_and_gradients():
    torch.manual_seed(42)
    imu=torch.randn(2,103,6,dtype=torch.float64)*.1;imu[:,:,:3]+=torch.tensor([0,0,9.81])
    gravity=torch.tensor([[0.,0.,-9.81]],dtype=torch.float64).expand(2,-1)
    correction=(torch.randn(2,103,9,dtype=torch.float64)*.01).requires_grad_()
    interval=torch.rand(2,103,dtype=torch.float64)*.002+.004
    result=integrate_parallel(imu,correction,interval,gravity)
    p=torch.zeros(2,3,dtype=torch.float64);v=p.clone();q=torch.zeros(2,4,dtype=torch.float64);q[:,3]=1
    pp=[p];vv=[v];rr=[qmatrix(q)];ww=[p]
    for i in range(103):
        p,v,q,w=integrate(p,v,q,imu[:,i],correction[:,i],interval[:,i],gravity)
        pp.append(p);vv.append(v);rr.append(qmatrix(q));ww.append(w)
    expected=dict(pos=torch.stack(pp,1),vel=torch.stack(vv,1),rot=torch.stack(rr,1),omega=torch.stack(ww,1))
    for key,value in expected.items():torch.testing.assert_close(result[key],value,atol=2e-12,rtol=2e-12)
    grad=torch.autograd.grad(sum(x.square().sum() for x in result.values()),correction,retain_graph=True)[0]
    ref=torch.autograd.grad(sum(x.square().sum() for x in expected.values()),correction)[0]
    torch.testing.assert_close(grad,ref,atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('profile',['10k','1m','100m'])
def test_batched_legacy_projections_preserve_streaming_equations(profile):
    torch.manual_seed(17);torch.set_num_threads(1)
    model=BudgetStudent(PROFILES[profile]).eval()
    features=torch.randn(2,13,model.features)*.1
    with torch.no_grad():
        sequence=model.sequence_corrections(features)
        hidden=torch.zeros(2,model.settings['hidden']);values=[]
        for token in features.unbind(1):
            value,hidden=model.update(token,hidden);values.append(value)
    torch.testing.assert_close(sequence,torch.stack(values,1),atol=1e-6,rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Official Mamba-2 CUDA required')
@pytest.mark.parametrize('profile',['1m_ssm','1m_hybrid','100m_hybrid'])
def test_official_sequence_matches_streaming_over_attention_rollover(profile):
    torch.manual_seed(17);torch.set_num_threads(1)
    model=BudgetStudent(PROFILES[profile]).cuda().eval()
    features=torch.randn(2,35,model.features,device='cuda')*.1
    with torch.no_grad(),amp_disabled():
        expected=model.sequence_corrections(features)
        hidden=None;values=[]
        for token in features.unbind(1):
            value,hidden=model.update(token,hidden);values.append(value)
    torch.testing.assert_close(torch.stack(values,1),expected,atol=3e-6,rtol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Official Mamba-2 CUDA required')
def test_frozen_teacher_parallel_physics_matches_serial_twenty_second_trajectory():
    from models import HybridStateStudent
    torch.manual_seed(17);torch.set_num_threads(1)
    model=HybridStateStudent(hidden=64,n_mamba=1,d_state=4,headdim=16,heads=2,visual_embeddings=True).cuda().eval()
    with torch.no_grad():model.out.weight.normal_(0,.01)
    for parameter in model.parameters():parameter.requires_grad_(False)
    frames=torch.randn(1,4001,48,device='cuda')*.1
    imu=torch.randn(1,4001,6,device='cuda')*.01;imu[:,:,2]+=9.81
    observed=torch.arange(4001,device='cuda')[None]%10==0
    dt=torch.tensor([.005],device='cuda');gravity=torch.tensor([[0.,0.,-9.81]],device='cuda')
    with torch.no_grad(),amp_disabled():
        serial=model(frames,imu,observed,dt,gravity,True,parallel_physics=False)
        parallel=model(frames,imu,observed,dt,gravity,True,parallel_physics=True)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    torch.testing.assert_close(serial['corrections'],parallel['corrections'],atol=0,rtol=0)
    # Different FP32 matrix-product association accumulates over 4,000 ticks.
    # Neural labels are identical; bound physical drift in metres/m/s rather
    # than requiring bitwise trajectory sums from different reduction orders.
    for key,atol,rtol in [('pos',.01,.001),('vel',.002,.001),('rot',2e-4,2e-4),('omega',0,0)]:
        torch.testing.assert_close(parallel[key],serial[key],atol=atol,rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_autotune_restores_weights_optimizer_and_rng():
    from test_distillation_pipeline import payload
    model=BudgetStudent(PROFILES['10k']).cuda()
    row=dict(model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4))
    before=deepcopy(model.state_dict());rng=torch.cuda.get_rng_state()
    def move(value):
        if isinstance(value,torch.Tensor):return value.cuda()
        if isinstance(value,dict):return {k:move(v) for k,v in value.items()}
        if isinstance(value,tuple):return tuple(move(v) for v in value)
        return value
    with DistillationPipeline('cuda',['10k']) as pipeline:
        with pipeline.batches([payload('cpu')],move) as batches:
            prepared=next(batches)
            result=pipeline.autotune({'10k':row},prepared,partial(train_student,device='cuda',weights=(1.,.5,1.)))
    assert result['max_concurrent']==1 and not row['optimizer'].state
    for key,value in before.items():torch.testing.assert_close(model.state_dict()[key],value,atol=0,rtol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(),rng,atol=0,rtol=0)


def test_curriculum_reaches_twenty_seconds_and_profiles_remain_separate():
    assert list(PROFILES)==['500','10k','1m','100m','1m_ssm','1m_hybrid','100m_hybrid']
    assert (PROFILES['10k']['grid_width'],PROFILES['10k']['grid_height'])==(6,3)
    windows=[401,801,1601,2401,4001]
    assert curriculum_window(23,windows,6)==2401
    assert curriculum_window(24,windows,6)==4001
    assert PROFILES['1m']['hidden']==192 and PROFILES['100m']['hidden']==1792


def test_area_uses_website_reference_and_does_not_add_overlapping_gain():
    baseline=json.loads((HERE/'website_baselines.json').read_text())['points']
    own=[dict(label='good',fuel_per_frame_max=800_000,sim3_ate_rmse_mean=.5),
         dict(label='dominated',fuel_per_frame_max=900_000,sim3_ate_rmse_mean=.6)]
    result=score_portfolio(own,baseline)
    assert result['fuel_reference']==24_000_000_000
    assert result['error_reference']==pytest.approx(3.234292166482429)
    assert result['points'][0]['on_frontier']
    assert not result['points'][1]['on_frontier']
    assert result['points'][1]['unique_gain_percent']==pytest.approx(0)
    assert result['claimed_percent']==pytest.approx(result['points'][0]['standalone_gain_percent'])
    assert sum(row['standalone_gain_percent'] for row in result['points'])>result['claimed_percent']
    assert score_portfolio(own,[])['claimed_percent'] is None


def test_projection_accounts_for_batch_size_validation_and_collection():
    stage=dict(windows_per_recording=10,batch=4,selected=dict(seconds_per_batch=2))
    result=project_times([stage],3,8,12,5)
    # 20 train batches * 2s + 12 validation recordings * 3s + 5s collection.
    assert result['100']==8100
    assert result['500']==40500
    streaming=project_times([stage],3,8,12,300,overlap=True)
    assert streaming['100']==10000  # 300s generation amortized across three 76s epochs
    assert streaming['earliest_stale_stop_seconds']==2400  # patience needs eight fresh batches
