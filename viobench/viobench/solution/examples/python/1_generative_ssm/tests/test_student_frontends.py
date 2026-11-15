"""Learned encoder/predictor deployment parity, causal inference and gradients."""
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[4]))
from students.model import BudgetStudent,PROFILES,ALL_PROFILES,LEGACY_PROFILES
from students.export import WORKSPACE,write_sources,ordered_weights,build
from distill import state_loss
from performance import amp_disabled
from viobench import Sandbox,Limits
from viobench.tools.abi.config import default_guest_config
from viobench.tools.geometry import matrix_to_quaternion,quaternion_to_matrix


def fixture(profile,frames=12):
    torch.set_num_threads(1);torch.manual_seed(17)
    if ALL_PROFILES[profile].get('kind')=='mamba2' and not torch.cuda.is_available():
        pytest.skip('official Mamba-2 requires CUDA')
    device='cuda' if ALL_PROFILES[profile].get('kind')=='mamba2' else 'cpu'
    model=BudgetStudent(ALL_PROFILES[profile]).to(device).eval()
    with torch.no_grad():
        for parameter in model.parameters():parameter.add_(torch.randn_like(parameter)*.015)
        if profile=='500':model.vision_weight.fill_(.17)
    rng=np.random.default_rng(17)
    images=rng.integers(0,256,(frames,240,320,3),dtype=np.uint8)
    packets=[]
    initial=np.tile([0.,0.,9.81,0.,0.,0.],(10,1))
    for k in range(frames):
        sensor=initial if k==0 else np.column_stack([rng.normal(0,.2,(10,3))+[0,0,9.81],rng.normal(0,.05,(10,3))])
        packets.append(np.column_stack([k*.05+np.arange(-9,1)*.005,sensor]).astype(np.float64))
    imu=np.concatenate([np.zeros((1,6)),*[packet[:,1:] for packet in packets[1:]]])
    inputs=dict(images=torch.from_numpy(images).permute(0,3,1,2).float()[None].to(device)/255.,
                imu=torch.from_numpy(imu).float()[None].to(device),dt=torch.tensor([.005],device=device),
                gravity=torch.tensor([[0.,0.,-9.81]],device=device),ticks=list(range(0,(frames-1)*10+1,10)))
    return model,inputs,images,packets,device


@pytest.mark.parametrize('profile',[*PROFILES,'500'])
def test_full_encoder_forecaster_state_pipeline_matches_native_and_wasm(tmp_path,profile):
    if not shutil.which('cc'):pytest.skip('host compiler unavailable')
    model,inputs,images,packets,device=fixture(profile)
    with torch.no_grad(),amp_disabled():
        expected={key:value.cpu().numpy() for key,value in model(**inputs).items()}
    out=write_sources(model,tmp_path/'native');lib=out/'student.so'
    subprocess.run(['cc','-std=c11','-O2','-DSTUDENT_HOST','-shared','-fPIC',
                    '-I'+str(WORKSPACE/'runtime/native'),str(out/'solution.c'),'-lm','-o',str(lib)],check=True)
    native=ctypes.CDLL(str(lib));fp=ctypes.POINTER(ctypes.c_float);dp=ctypes.POINTER(ctypes.c_double)
    native.student_set_weights.argtypes=[fp]
    native.init.argtypes=[ctypes.c_char_p,ctypes.c_uint32]
    native.step.argtypes=[ctypes.c_double,ctypes.POINTER(ctypes.c_uint8),dp,ctypes.c_int32,dp]
    weights=ordered_weights(model);native.student_set_weights(weights.ctypes.data_as(fp))
    config=default_guest_config();q=np.asarray([.1,-.2,.15,.96]);q/=np.linalg.norm(q)
    Rci=quaternion_to_matrix(q);lever=np.asarray([.12,-.04,.03])
    transform=np.eye(4);transform[:3,:3]=Rci;transform[:3,3]=-Rci@lever
    config['T_cam_imu']=transform.tolist();encoded=json.dumps(config).encode()
    assert native.init(encoded,len(encoded))==0
    sandbox=None
    if (WORKSPACE/'build/toolchain/wasi-sdk/bin/clang').is_file():
        sandbox=Sandbox(build(model,tmp_path/'wasm'),limits=Limits(step_fuel=ALL_PROFILES[profile]['fuel']))
        sandbox.init(config)
    replay=[]
    for k,(frame,packet) in enumerate(zip(images,packets)):
        rotation=expected['rot'][0,k];position=expected['pos'][0,k];velocity=expected['vel'][0,k];omega=expected['omega'][0,k]
        truth=np.concatenate([Rci@(position+(rotation-np.eye(3))@lever),
                              matrix_to_quaternion(Rci@rotation@Rci.T),
                              Rci@(rotation.T@velocity+np.cross(omega,lever)),Rci@omega])
        if k==0:truth=np.asarray([0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.])
        actual=np.zeros(13,np.float64)
        assert native.step(k*.05,frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                           packet.ctypes.data_as(dp),10,actual.ctypes.data_as(dp))==0
        np.testing.assert_allclose(actual,truth,atol=2e-5,rtol=2e-4)
        if sandbox:
            result=sandbox.step(k*.05,frame,packet)
            np.testing.assert_allclose(np.concatenate([result.pose,result.twist]),truth,atol=2e-5,rtol=2e-4)
            assert result.fuel<=ALL_PROFILES[profile]['fuel']
        replay.append(actual.copy())
    assert native.init(encoded,len(encoded))==0
    for k,(frame,packet) in enumerate(zip(images,packets)):
        actual=np.zeros(13,np.float64)
        assert native.step(k*.05,frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),packet.ctypes.data_as(dp),10,actual.ctypes.data_as(dp))==0
        np.testing.assert_array_equal(actual,replay[k])


@pytest.mark.parametrize('profile',list(PROFILES))
def test_ground_truth_gradients_reach_encoder_predictor_and_original_state_family(profile):
    model,inputs,_,_,device=fixture(profile,frames=5)
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    with amp_disabled():
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction=model(**inputs)
            target={key:torch.zeros_like(prediction[key]) for key in ['pos','vel','omega']}
            target['rot']=torch.eye(3,device=device)[None,None].expand_as(prediction['rot'])
            loss=state_loss(prediction,target,device);loss.backward()
            assert torch.isfinite(loss)
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.vision.parameters())
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.embedding_predictor.parameters())
            optimizer.step()
    original=LEGACY_PROFILES.get(profile,{})
    assert model.settings.get('kind')==original.get('kind')


def test_1k_uses_a_spatial_cnn_and_replaces_the_default_500_tier():
    model=BudgetStudent(PROFILES['1k'])
    assert len(PROFILES)==8 and '500' not in PROFILES
    assert PROFILES['1k']['fuel']==6000
    assert isinstance(model.vision.conv1,torch.nn.Conv2d)
    assert model.vision.conv1.weight.shape==(2,1,2,2)
    assert model.vision(torch.randn(1,3,16)).shape==(1,3,2)
    assert sum(p.numel() for p in model.parameters())==169
    assert ALL_PROFILES['500']['fuel']==500


def test_100k_has_learned_visual_prediction_and_recurrent_full_imu_state():
    model=BudgetStudent(PROFILES['100k'])
    assert model.settings['fuel']==100_000 and not model.settings['aggregate']
    assert model.vision.project.out_features==8
    assert model.embedding_predictor[0].in_features==32
    assert model.embedding_predictor[-1].out_features==8
    assert model.input.in_features==24 and model.recurrent.weight.shape==(48,48)
    assert model.head.out_features==9
    assert sum(p.numel() for p in model.parameters())==4417


@pytest.mark.parametrize('profile',['1k','10k','100k','1m','1m_ssm','1m_hybrid','100m_hybrid'])
def test_future_images_and_imu_cannot_change_earlier_predictions(profile):
    model,inputs,_,_,_=fixture(profile,frames=7)
    with torch.no_grad(),amp_disabled():
        baseline=model(**inputs)
        changed=dict(inputs,images=inputs['images'].clone(),imu=inputs['imu'].clone())
        changed['images'][:,4:]=1-changed['images'][:,4:]
        changed['imu'][:,31:]+=3.
        revised=model(**changed)
    for key in ['pos','vel','rot','omega']:
        torch.testing.assert_close(baseline[key][:,:4],revised[key][:,:4],atol=1e-6,rtol=1e-5)


def test_1k_changes_attitude_and_learns_gyro_and_visual_corrections():
    model,inputs,_,_,_=fixture('1k',frames=45)
    with torch.no_grad():
        model.head.weight.zero_();model.head.bias.zero_()
        inputs['imu'][:,:,3:]=torch.tensor([0.,0.,.8])
    prediction=model(**inputs)
    angle=2*44*np.arctan(.8*.025)
    expected=torch.tensor([[np.cos(angle),-np.sin(angle),0.],
                           [np.sin(angle),np.cos(angle),0.],[0.,0.,1.]],dtype=torch.float32)
    torch.testing.assert_close(prediction['rot'][0,-1],expected,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(prediction['omega'][0,-1],torch.tensor([0.,0.,.8]))
    loss=prediction['omega'].square().mean()+prediction['rot'][:,:,:,0].sum()
    loss.backward()
    for parameter in model.head.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum()>0


def test_old_1k_settings_keep_fixed_attitude_and_original_weights(tmp_path):
    settings=dict(PROFILES['1k'],fuel=1000,kind='visual_imu_affine')
    model=BudgetStudent(settings)
    assert sum(p.numel() for p in model.parameters())==64
    assert not hasattr(model,'gyro_gain')
    out=write_sources(model,tmp_path)
    assert 'Fixed attitude' in (out/'solution.c').read_text()
    inputs=dict(imu=torch.ones(1,21,6),ticks=[0,10,20],images=torch.rand(1,3,3,240,320))
    prediction=model.imu_forward(**inputs)
    torch.testing.assert_close(prediction['rot'],torch.eye(3)[None,None].expand(1,3,3,3))
    assert not prediction['omega'].any()


def test_1k_cuda_parallel_attitude_matches_sequential_and_backpropagates():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    model,inputs,_,_,_=fixture('1k',frames=401)
    with torch.no_grad(),amp_disabled():expected=model(**inputs)
    model=model.cuda();inputs={k:v.cuda() if torch.is_tensor(v) else v for k,v in inputs.items()}
    with amp_disabled():
        actual=model(**inputs)
        for key in ['pos','vel','rot','omega']:
            torch.testing.assert_close(actual[key].cpu(),expected[key],atol=2e-4,rtol=2e-4)
        loss=actual['pos'].square().mean()+actual['omega'].square().mean()
        loss.backward()
    for parameter in model.head.parameters():
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0


def test_1k_outputs_and_integrates_all_nine_corrections():
    model,inputs,_,_,_=fixture('1k',frames=2)
    correction=torch.tensor([.02,-.03,.04,.05,.06,-.07,.08,-.09,.10])
    with torch.no_grad():
        model.head.weight.zero_();model.head.bias.copy_(torch.atanh(correction/.25))
        inputs['imu'][:,:,:3]=torch.tensor([0.,0.,9.81]);inputs['imu'][:,:,3:]=0.
    prediction=model(**inputs)
    assert model.head.in_features==12 and model.head.out_features==9
    assert prediction['corrections'].shape==(1,2,9)
    torch.testing.assert_close(prediction['corrections'][0,1],correction)
    acceleration=correction[3:6]+correction[6:]
    torch.testing.assert_close(prediction['vel'][0,1],acceleration*.05,atol=1e-7,rtol=1e-5)
    torch.testing.assert_close(prediction['pos'][0,1],.5*acceleration*.05**2,atol=1e-7,rtol=1e-5)
    torch.testing.assert_close(prediction['omega'][0,1],correction[:3])
    assert not torch.allclose(prediction['rot'][0,1],torch.eye(3))
    prediction['corrections'].square().mean().backward()
    assert (model.head.bias.grad.abs()>0).all()


def test_old_six_output_1k_checkpoint_keeps_orientation_kernel(tmp_path):
    settings=dict(PROFILES['1k'],fuel=1100,kind='visual_imu_affine',orientation_head=True)
    model=BudgetStudent(settings)
    assert sum(p.numel() for p in model.parameters())==82
    out=write_sources(model,tmp_path)
    assert 'W_GYRO_GAIN' in (out/'solution.c').read_text()
    assert model.embedding_predictor[0].in_features==11
