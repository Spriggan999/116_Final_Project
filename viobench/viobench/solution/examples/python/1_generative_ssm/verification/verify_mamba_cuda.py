"""Compare official Mamba-2 CUDA scan/step and the actual NumPy deployment code."""
import argparse
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))


def guest_for(model,hidden,layers,window,d_state,headdim,heads):
    from export import normalize_settings
    from models import EmbeddingPredictor
    flags=normalize_settings(dict(backbone='mamba2_hybrid',frame_mode='embedding',hidden=hidden,
        mamba_layers=layers,attention_window=window,hybrid_d_state=d_state,hybrid_head_dim=headdim,
        hybrid_attention_heads=heads,learned_visual=False,width=32,height=24))
    frame=EmbeddingPredictor(learned_visual=False).eval()
    files={'deployment.json':json.dumps(flags).encode()}
    for name,module in [('state_student.npz',model),('frame_student.npz',frame)]:
        buffer=io.BytesIO()
        np.savez(buffer,**{key:value.detach().cpu().numpy() for key,value in module.state_dict().items()})
        files[name]=buffer.getvalue()
    bundle=types.ModuleType('solution_bundle')
    bundle.exists=lambda name:name in files
    bundle.read_text=lambda name:files[name].decode()
    bundle.read_bytes=lambda name:files[name]
    bundle.open=lambda name:io.BytesIO(files[name])
    previous=sys.modules.get('solution_bundle')
    sys.modules['solution_bundle']=bundle
    try:
        spec=importlib.util.spec_from_file_location('_mamba_validation_guest',HERE/'solution/solution.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:
        if previous is None:sys.modules.pop('solution_bundle',None)
        else:sys.modules['solution_bundle']=previous
    config=dict(width=320,height=240,T_cam_imu=np.eye(4).tolist())
    return lambda:module.Solution(config)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hidden',type=int,default=512)
    parser.add_argument('--layers',type=int,default=3)
    parser.add_argument('--lengths',type=int,nargs='+',default=[11,101,257,513])
    args=parser.parse_args()
    if not torch.cuda.is_available():parser.error('CUDA required')
    os.environ.setdefault('TRITON_F32_DEFAULT','ieee')
    from mamba_ssm import Mamba2
    from models import HybridStateStudent
    torch.set_num_threads(1);torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    atol,rtol=5e-5,5e-4
    result=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        mamba=importlib.metadata.version('mamba-ssm'),causal_conv1d=importlib.metadata.version('causal-conv1d'),
        dtype='float32',tf32=False,triton_fp32=os.environ['TRITON_F32_DEFAULT'],
        atol=atol,rtol=rtol,hidden=args.hidden,layers=args.layers,checks=[],passed=False)

    def compare(name,actual,expected,**context):
        actual=actual.detach().cpu() if isinstance(actual,torch.Tensor) else torch.from_numpy(np.asarray(actual))
        expected=expected.detach().cpu() if isinstance(expected,torch.Tensor) else torch.from_numpy(np.asarray(expected))
        difference=(actual-expected).abs()
        entry=dict(name=name,**context,max_abs_error=float(difference.max()),
                   rmse=float(difference.square().mean().sqrt()),
                   tolerance_ratio_max=float((difference/(atol+rtol*expected.abs())).max()))
        result['checks'].append(entry)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol,check_dtype=False)
        print(json.dumps(entry),flush=True)

    block=Mamba2(d_model=128,d_state=16,d_conv=4,expand=2,headdim=32).cuda().eval()
    with torch.no_grad():
        for length in args.lengths:
            inputs=torch.randn(2,length,128,device='cuda')*.3
            sequence=block(inputs)
            conv,ssm=block.allocate_inference_cache(2,max_seqlen=1)
            streamed=[]
            for tick in range(length):
                value,conv,ssm=block.step(inputs[:,tick:tick+1],conv,ssm)
                streamed.append(value)
            compare('official_cuda_scan_vs_step',sequence,torch.cat(streamed,1),length=length,batch=2)

    model=HybridStateStudent(hidden=args.hidden,n_mamba=args.layers,attn_window=16,d_state=16,
        headdim=32,heads=4,learned_visual=False,visual_embeddings=True).cuda().eval()
    assert all(isinstance(layer,Mamba2) for layer in model.ssm),'Official Mamba2 was not selected'
    with torch.no_grad():model.out.weight.normal_(0,.01);model.out.bias.normal_(0,.001)
    captured={}
    feature_hook=model.feature.register_forward_hook(lambda module,inputs,output:captured.update(projected=output.detach()))
    output_hook=model.out.register_forward_pre_hook(lambda module,inputs:captured.update(tokens=inputs[0].detach()))
    try:
        make_guest=guest_for(model,args.hidden,args.layers,16,16,32,4)
        for length in args.lengths:
            embeddings=torch.randn(2,length,48,device='cuda')*.3
            imu=torch.randn(2,length,6,device='cuda')*.1;imu[:,:,2]+=9.81
            observed=torch.arange(length,device='cuda')[None].expand(2,-1)%10==0
            dt=torch.full((2,),.005,device='cuda');gravity=torch.tensor([[0.,0.,-9.81]],device='cuda').repeat(2,1)
            with torch.no_grad():prediction=model(embeddings,imu,observed,dt,gravity,True)
            projected=captured['projected'].cpu().numpy();reference=captured['tokens'].cpu()
            actual=[];corrections=[]
            for batch in range(2):
                guest=make_guest();tokens=[];values=[]
                for tick in range(length):
                    token=guest._mamba_update(projected[batch,tick])
                    tokens.append(token.copy())
                    values.append(.25*np.tanh(guest.d['out.weight']@token+guest.d['out.bias']))
                actual.append(np.stack(tokens));corrections.append(np.stack(values))
                assert len(guest.attn_cache)<=16
            compare('official_cuda_hybrid_vs_deployment_tokens',np.stack(actual),reference,length=length,batch=2)
            compare('official_cuda_hybrid_vs_deployment_corrections',np.stack(corrections),prediction['corrections'],length=length,batch=2)
        # A real optimizer update verifies the official fused backward path,
        # followed by a second deployment comparison using the updated weights.
        model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        prediction=model(embeddings,imu,observed,dt,gravity,True)
        loss=prediction['corrections'].square().mean()+prediction['vel'].square().mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step();model.eval()
        with torch.no_grad():prediction=model(embeddings,imu,observed,dt,gravity,True)
        guest=guest_for(model,args.hidden,args.layers,16,16,32,4)()
        projected=captured['projected'][0].cpu().numpy()
        actual=np.stack([guest._mamba_update(token) for token in projected])
        compare('official_cuda_vs_deployment_after_optimizer_step',actual,captured['tokens'][0])
        result.update(passed=True,backward_finite=True)
    finally:
        feature_hook.remove();output_hook.remove()
        args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
