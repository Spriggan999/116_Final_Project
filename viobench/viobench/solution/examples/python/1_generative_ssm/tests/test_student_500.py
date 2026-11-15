"""Focused vision/IMU learning, native parity and 500-fuel deployment check."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from students.model import BudgetStudent,PROFILES
from students.export import export_student
from viobench import Sandbox,Limits
from viobench.tools.abi.config import default_guest_config


def test_500_visual_inputs_learn_and_match_wasm_under_cap(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(17)
    model=BudgetStudent(PROFILES['500']).eval()
    assert len(PROFILES)==7 and '100' not in PROFILES
    assert sum(parameter.numel() for parameter in model.parameters())==18
    with torch.no_grad():
        model.vision_weight.copy_(torch.tensor([[.1,-.1,.02,-.03],[.05,.04,-.02,.01],[.02,-.01,.03,.02]]))
    frames=[]
    for i in range(21):
        frame=np.zeros((240,320,3),np.uint8)
        frame[120,80]=[20+7*i,30+3*i,80]
        frame[120,240]=[150-3*i,90,30+5*i]
        frames.append(frame)
    images=torch.from_numpy(np.stack(frames)).permute(0,3,1,2)[None].float()/255
    imu=torch.tensor([.03,-.02,9.8,0.,0.,0.])[None,None].repeat(1,201,1)
    ticks=list(range(0,201,10))
    prediction=model(images,imu,None,None,ticks)
    changed=images.clone();changed[:,:,:,120,80]=0
    assert not torch.allclose(prediction['accelerations'],model(changed,imu,None,None,ticks)['accelerations'])
    prediction['accelerations'].square().sum().backward()
    assert torch.isfinite(model.vision_weight.grad).all() and model.vision_weight.grad.abs().sum()>0
    assert torch.isfinite(model.gain.grad).all() and model.gain.grad.abs().sum()>0
    wasm=export_student(model,tmp_path/'native',frames=1200)
    assert json.loads((tmp_path/'native/limits.json').read_text())['fuel_step_max']<=500
    config=default_guest_config();R=np.asarray(config['T_cam_imu'])[:3,:3]
    sandbox=Sandbox(wasm,limits=Limits(step_fuel=500));sandbox.init(config)
    for i,frame in enumerate(frames):
        packet=np.empty((0,7),np.float64) if not i else np.column_stack([
            (np.arange(1,11)+10*(i-1))*.005,np.tile(imu[0,0].numpy(),(10,1))])
        actual=sandbox.step(i*.05,frame,packet)
        np.testing.assert_allclose(actual.pose[:3],R@prediction['pos'][0,i].detach().numpy(),atol=3e-6,rtol=3e-5)
        np.testing.assert_allclose(actual.twist[:3],R@prediction['vel'][0,i].detach().numpy(),atol=3e-6,rtol=3e-5)
        assert actual.fuel<=500
    # Resetting the instance must clear the visual history and motion state.
    fresh=Sandbox(wasm,limits=Limits(step_fuel=500));fresh.init(config)
    first=fresh.step(0.,frames[0],np.empty((0,7),np.float64))
    np.testing.assert_array_equal(first.pose,[0,0,0,0,0,0,1])
