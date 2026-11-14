"""Compare a real WASM hybrid guest against official CUDA streaming inference."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[4]))


def check(args):
    from export import normalize_settings
    from models import EmbeddingPredictor,HybridStateStudent
    from performance import amp_disabled
    from viobench import Sandbox
    from viobench.tools.abi.config import default_guest_config
    from viobench.tools.geometry import matrix_to_quaternion
    from viobench.tools.runner.sandbox import Limits
    from verify_limits import LIMIT_VALUES
    settings=normalize_settings(json.loads((args.checkpoints/'student_config.json').read_text()))
    if settings['backbone']!='mamba2_hybrid' or settings['frame_mode']!='embedding':
        raise ValueError('This check requires an embedding-based hybrid checkpoint')
    frame=EmbeddingPredictor(hidden=settings['embedding_hidden'],learned_visual=settings['learned_visual'],
        visual_input_stride=settings['visual_input_stride'],ema=settings['embedding_ema'])
    model=HybridStateStudent(hidden=settings['hidden'],n_mamba=settings['mamba_layers'],
        attn_window=settings['attention_window'],d_state=settings['hybrid_d_state'],headdim=settings['hybrid_head_dim'],
        heads=settings['hybrid_attention_heads'],learned_visual=settings['learned_visual'],
        visual_input_stride=settings['visual_input_stride'],visual_embeddings=True)
    for module,name in ((frame,'frame_student.pt'),(model,'state_student.pt')):
        module.load_state_dict(torch.load(args.checkpoints/name,map_location='cpu',weights_only=True))
        module.cuda().eval().requires_grad_(False)
    from mamba_ssm import Mamba2
    assert all(isinstance(block,Mamba2) for block in model.ssm)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(1)
    sandbox=Sandbox(args.module,limits=Limits(**LIMIT_VALUES),inherit_stdio=True)
    width,height=sandbox.resolution
    config=default_guest_config(width=width,height=height);config['T_cam_imu']=np.eye(4).tolist()
    sandbox.init(config)
    rng=np.random.default_rng(1701)
    gravity=torch.tensor([[0.,0.,-9.81]],device='cuda')
    dt=torch.tensor([.005],device='cuda');previous=anchor=state=None
    worst=np.zeros(13);scaled=0.;fuel=0
    with torch.no_grad(),amp_disabled():
        for index in range(args.frames):
            image=rng.integers(0,256,(height,width,3),dtype=np.uint8)
            packet=np.column_stack([index*.05+np.arange(1,11)*.005,
                rng.normal(0,.2,(10,3))+[0,0,9.81],rng.normal(0,.05,(10,3))])
            if not index:packet[:,1:]=[0,0,9.81,0,0,0]
            pixels=torch.from_numpy(image).permute(2,0,1)[None].cuda().float()/255
            if (height,width)!=(settings['height'],settings['width']):
                raise ValueError('Use a checkpoint with matching ABI/model resolution for this numerical check')
            newest=frame.encoder(pixels)
            if state is None:
                previous=anchor=newest;state=model.stream_init(newest)
                output,state=model.stream_step(newest,torch.from_numpy(packet[-1:,1:]).cuda().float(),
                    torch.ones(1,device='cuda',dtype=torch.bool),dt,gravity,state,initial=True)
            else:
                samples=torch.from_numpy(packet[:,1:]).cuda().float()
                for tick in range(10):
                    observed=torch.tensor([tick==9],device='cuda')
                    span=torch.tensor([[0. if index==1 else .05]],device='cuda')
                    elapsed=torch.tensor([[(tick+1)*.005]],device='cuda')
                    visual=newest if tick==9 else frame.predict_embeddings(previous,anchor,
                        samples[:tick+1].mean(0)[None],elapsed,span)
                    output,state=model.stream_step(visual,samples[tick:tick+1],observed,dt,gravity,state)
                previous,anchor=anchor,newest
            rotation=output['rot'][0].cpu().numpy().astype(np.float64)
            expected=np.concatenate([output['pos'][0].cpu().numpy(),matrix_to_quaternion(rotation),
                rotation.T@output['vel'][0].cpu().numpy(),output['omega'][0].cpu().numpy()])
            actual=sandbox.step((index+1)*.05,image,packet)
            # Match quaternion sign before comparing equivalent orientations.
            estimate=np.concatenate([actual.pose,actual.twist])
            if np.dot(estimate[3:7],expected[3:7])<0:estimate[3:7]*=-1
            error=np.abs(estimate-expected);worst=np.maximum(worst,error)
            scaled=max(scaled,float(np.max(error/(args.atol+args.rtol*np.abs(expected)))))
            np.testing.assert_allclose(estimate,expected,atol=args.atol,rtol=args.rtol)
            fuel=max(fuel,actual.fuel)
            if index==0 or (index+1)%25==0:print(f'CUDA/WASM parity: {index+1}/{args.frames}; max state error={worst.max():.3g}',flush=True)
    return dict(passed=True,frames=args.frames,atol=args.atol,rtol=args.rtol,
        max_abs_error_by_state_component=worst.tolist(),tolerance_ratio_max=scaled,fuel_step_max=fuel,
        checkpoint=str(args.checkpoints.resolve()),module=str(args.module.resolve()),gpu=torch.cuda.get_device_name())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',type=Path,required=True)
    parser.add_argument('--module',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=100)
    parser.add_argument('--atol',type=float,default=5e-4)
    parser.add_argument('--rtol',type=float,default=5e-4)
    args=parser.parse_args()
    from viobench.tools.cli import run_with_large_stack
    result=run_with_large_stack(lambda:check(args))
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
