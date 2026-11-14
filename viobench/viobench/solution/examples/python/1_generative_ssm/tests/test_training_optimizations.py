"""Bounded attention, neural precision and configurable embedding deployment."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from attention import local_attention,observation_ages
from export import make_models,normalize_settings,export_models
from data import state_input_frames
from test_embedding_prediction import causal_batch
import config


@pytest.mark.parametrize('window',[1,16,128])
@pytest.mark.parametrize('bias',[False,True])
def test_local_attention_matches_dense_outputs_and_gradients(window,bias):
    torch.manual_seed(1701)
    original=torch.nn.MultiheadAttention(32,4,batch_first=True,bias=bias)
    bounded=copy.deepcopy(original)
    x=torch.randn(2,79,32,requires_grad=True);y=x.detach().clone().requires_grad_()
    expected=local_attention(original,x,window,0)
    actual=local_attention(bounded,y,window,13)
    torch.testing.assert_close(actual,expected,atol=4e-7,rtol=2e-5)
    expected.square().mean().backward();actual.square().mean().backward()
    torch.testing.assert_close(x.grad,y.grad,atol=1e-8,rtol=2e-4)
    for a,b in zip(original.parameters(),bounded.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-8,rtol=2e-4)


def test_local_attention_masks_stay_bounded_and_causal(monkeypatch):
    import attention
    shapes=[];original=attention.F.scaled_dot_product_attention
    def measured(*args,**kwargs):
        shapes.append(kwargs['attn_mask'].shape)
        return original(*args,**kwargs)
    monkeypatch.setattr(attention.F,'scaled_dot_product_attention',measured)
    module=torch.nn.MultiheadAttention(16,4,batch_first=True).eval()
    x=torch.randn(1,2001,16)
    actual=local_attention(module,x,16,64)
    assert len(shapes)==32 and max(a*b for a,b in shapes)<=64*79
    changed=x.clone();changed[:,1000:]+=10
    later=local_attention(module,changed,16,64)
    torch.testing.assert_close(actual[:,:1000],later[:,:1000],atol=0,rtol=0)
    changed=x.clone();changed[:,:1000]+=10
    older=local_attention(module,changed,16,64)
    torch.testing.assert_close(actual[:,1015:],older[:,1015:],atol=0,rtol=0)


def test_vectorized_observation_age_matches_incremental_clock():
    torch.manual_seed(17)
    observed=torch.rand(3,53)>.8;observed[:,0]=False
    dt=torch.tensor([.005,.004,.006])
    ages=[torch.zeros(3)]
    for t in range(1,53):ages.append(torch.where(observed[:,t],0.,ages[-1]+dt))
    torch.testing.assert_close(observation_ages(observed,dt,torch.float32),torch.stack(ages,1),atol=1e-8,rtol=1e-6)


@pytest.mark.parametrize('dimension',[64,96])
def test_wider_embeddings_reach_state_model_and_export(tmp_path,dimension):
    settings=dict(frame_mode='embedding',backbone='gru',hidden=16,learned_visual=True,
                  width=32,height=24,visual_input_stride=8,embedding_hidden=8,embedding_dim=dimension)
    frame,state=make_models(settings)
    batch=causal_batch()
    batch.update(pos=torch.zeros(2,7,3),gravity=torch.tensor([[0.,0.,-9.81]]).repeat(2,1))
    visual=state_input_frames(frame,batch,'cpu')
    assert visual.shape==(2,7,dimension) and state.feature[0].in_features==2*dimension+8
    prediction=state(visual,batch['imu'],batch['observed'],batch['dt'],batch['gravity'])
    assert torch.isfinite(prediction['pos']).all()
    output=export_models(frame,state,settings,tmp_path/'solution')
    assert json.loads((output/'deployment.json').read_text())['embedding_dim']==dimension
    assert normalize_settings(dict(frame_mode='embedding',backbone='gru'))['embedding_dim']==48


def test_dimension_change_requires_fresh_training(tmp_path):
    source=tmp_path/'old';source.mkdir()
    (source/'training_config.json').write_text(json.dumps(dict(VISUAL_PREDICTION='embedding',
        USE_HYBRID_SSM_TRANSFORMER=False,EMBEDDING_DIM=48)))
    settings=tmp_path/'candidate.json';settings.write_text(json.dumps(dict(EMBEDDING_DIM=96)))
    result=subprocess.run([sys.executable,str(HERE/'train.py'),'--from-checkpoints',str(source),
        '--sl','--config',str(settings),'--out',str(tmp_path/'new'),'--device','cpu'],
        capture_output=True,text=True,timeout=30,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    assert result.returncode==2 and 'changing embedding dimension requires a fresh run' in result.stderr
    assert not (tmp_path/'new').exists()


def test_hybrid_amp_uses_bf16_neural_layers_and_fp32_validation(monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from mamba_helpers import require_mamba
    require_mamba()
    from models import HybridStateStudent
    from performance import amp_disabled
    monkeypatch.setattr(config,'USE_AMP',True)
    monkeypatch.setattr(config,'HYBRID_ATTENTION_CHUNK',32)
    model=HybridStateStudent(hidden=128,n_mamba=1,attn_window=16,visual_embeddings=True).cuda().train()
    with torch.no_grad():model.out.weight.normal_(0,.001)
    x=torch.randn(2,65,48,device='cuda');imu=torch.zeros(2,65,6,device='cuda');imu[:,:,2]=9.81
    observed=torch.arange(65,device='cuda')[None].expand(2,-1)%10==0
    dt=torch.full((2,),.005,device='cuda');gravity=torch.tensor([[0.,0.,-9.81]],device='cuda').repeat(2,1)
    dtypes=[];handle=model.feature.register_forward_hook(lambda m,a,out:dtypes.append(out.dtype))
    try:
        prediction=model(x,imu,observed,dt,gravity,True)
        assert dtypes[-1]==torch.bfloat16 and all(v.dtype==torch.float32 for v in prediction.values())
        prediction['vel'].square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        model.eval()
        with torch.no_grad(),amp_disabled():prediction=model(x,imu,observed,dt,gravity,True)
        assert dtypes[-1]==torch.float32 and all(torch.isfinite(v).all() for v in prediction.values())
    finally:handle.remove()


@pytest.mark.parametrize('length',[1,2,17,401,4001])
def test_parallel_physics_matches_sequential_fp32(length):
    from models import integrate_corrections
    torch.manual_seed(1701)
    corr=torch.randn(2,length,9)*.05
    imu=torch.randn(2,length,6)*.2;imu[:,:,2]+=9.81
    dt=torch.tensor([.005,.006]);gravity=torch.tensor([[0.,0.,-9.81]]).repeat(2,1)
    with torch.no_grad():
        expected=integrate_corrections(corr,imu,dt,gravity)
        actual=integrate_corrections(corr,imu,dt,gravity,parallel=True)
    for key in expected:
        # Float32 reduction order differs; deployment/validation retain the
        # sequential recurrence. Permit 1 cm / 2 mm/s over the stress case's
        # 24-second horizon; gradients are checked separately below.
        atol=dict(pos=.01,vel=.002,rot=.0001,omega=0.,corrections=0.)[key]
        torch.testing.assert_close(actual[key],expected[key],atol=atol,rtol=1e-5)


def test_parallel_physics_gradients_match_sequential_recurrence():
    from models import integrate_corrections
    torch.manual_seed(1701)
    corr=(torch.randn(2,53,9)*.05).requires_grad_()
    other=corr.detach().clone().requires_grad_()
    imu=torch.randn(2,53,6)*.2;imu[:,:,2]+=9.81
    dt=torch.full((2,),.005);gravity=torch.tensor([[0.,0.,-9.81]]).repeat(2,1)
    def objective(result):
        return result['pos'].square().mean()+result['vel'].square().mean()+result['rot'].square().mean()
    objective(integrate_corrections(corr,imu,dt,gravity)).backward()
    objective(integrate_corrections(other,imu,dt,gravity,parallel=True)).backward()
    torch.testing.assert_close(corr.grad,other.grad,atol=2e-8,rtol=2e-4)


def test_only_selected_cameras_are_normalized_and_transferred(tmp_path,monkeypatch):
    import data
    from test_curriculum import recorded_store
    store=recorded_store(tmp_path)
    sample=data.StateWindows(store,length=81)[0]
    batch={key:value[None] for key,value in sample.items()}
    original=data.camera_values;counts=[]
    def measured(images,device=None):
        counts.append(len(images))
        return original(images,device)
    monkeypatch.setattr(data,'camera_values',measured)
    monkeypatch.setattr(config,'VISUAL_CHUNK',3)
    frame,_=make_models(dict(frame_mode='embedding',backbone='gru',hidden=16,learned_visual=False))
    output=data.generate_200hz_embeddings(frame,batch,'cpu')
    assert output.shape==(1,81,48)
    assert sum(counts)==int(batch['observed'].sum())==9
    assert max(counts)<=3 and batch['cam_images'].shape[1]==17


def test_compact_cameras_preserve_timing_targets_and_predicted_embeddings(tmp_path):
    import data
    from test_curriculum import recorded_store
    store=recorded_store(tmp_path)
    native=data.StateWindows(store,length=81)[0]
    compact=data.StateWindows(store,length=81,compact_cameras=True)[0]
    assert native['cam_images'].shape[0]==17 and compact['cam_images'].shape[0]==9
    for name in ('pos','vel','rot','omega','imu','observed','dt','gravity','anchor_tick','prev_span'):
        torch.testing.assert_close(compact[name],native[name],atol=0,rtol=0)
    for index in ('anchor_img_idx','prev_img_idx'):
        torch.testing.assert_close(compact['cam_images'][compact[index]],native['cam_images'][native[index]],atol=0,rtol=0)
    frame,_=make_models(dict(frame_mode='embedding',backbone='gru',hidden=16,learned_visual=False))
    expected=data.generate_200hz_embeddings(frame,{k:v[None] for k,v in native.items()},'cpu')
    actual=data.generate_200hz_embeddings(frame,{k:v[None] for k,v in compact.items()},'cpu')
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
