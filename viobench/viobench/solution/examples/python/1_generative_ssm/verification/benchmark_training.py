"""Synthetic CUDA training probe of the legacy RGB option; no accuracy result."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch


def timed(function,units,warmup,steps):
    for _ in range(warmup):function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed=[]
    for _ in range(steps):
        torch.cuda.synchronize()
        started=time.perf_counter()
        function()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-started)
    seconds=sum(elapsed)/len(elapsed)
    return dict(seconds_per_step=seconds,samples_per_second=units/seconds,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)


def frame_probe(args):
    from models import GenerativeFlowPredictor
    from performance import adamw
    frame=GenerativeFlowPredictor(base=33,max_flow_pixels=160,input_stride=8).cuda().train()
    images=torch.rand(args.pairs,3,480,640,device='cuda')
    imu=torch.zeros(args.pairs,6,device='cuda')
    horizon=torch.full((args.pairs,1),.025,device='cuda');span=horizon*2
    optimizer=adamw(frame.parameters(),lr=3e-4)

    def step():
        optimizer.zero_grad(set_to_none=True)
        prediction=frame(images,images,imu,horizon,span)
        loss=(prediction-images).abs().mean()
        loss.backward();optimizer.step()

    return timed(step,args.pairs,args.warmup,args.steps)


def state_probe(args):
    from models import GenerativeFlowPredictor,StateStudent
    from data import state_input_frames
    from performance import adamw
    frame=GenerativeFlowPredictor(base=33,max_flow_pixels=160,input_stride=8).cuda().eval()
    for parameter in frame.parameters():parameter.requires_grad_(False)
    state=StateStudent(hidden=1792,learned_visual=True,visual_input_stride=8).cuda().train()
    b,t=args.batch,101
    ticks=torch.arange(t)[None].expand(b,-1);anchors=ticks//10
    imu=torch.zeros(b,t,6);imu[:,:,2]=9.81
    batch=dict(cam_images=torch.rand(b,11,3,480,640),imu=imu,
               anchor_img_idx=anchors,anchor_tick=anchors*10,
               prev_img_idx=(anchors-1).clamp_min(0),prev_span=torch.full((b,t),.05),
               observed=ticks%10==0,dt=torch.full((b,),.005),
               gravity=torch.tensor([0,0,-9.81]).expand(b,-1))
    optimizer=adamw(state.parameters(),lr=1e-4)

    def step():
        optimizer.zero_grad(set_to_none=True)
        images=state_input_frames(frame,batch,'cuda')
        prediction=state(images,batch['imu'].cuda(),batch['observed'].cuda(),
                         batch['dt'].cuda(),batch['gravity'].cuda(),True)
        loss=(prediction['pos'].square().mean()+prediction['vel'].square().mean()
              +prediction['corrections'].square().mean())
        loss.backward();optimizer.step()

    return timed(step,args.batch,args.warmup,args.steps)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--pairs',type=int,default=1)
    parser.add_argument('--chunk',type=int,default=1)
    parser.add_argument('--amp',action='store_true')
    parser.add_argument('--fused-adamw',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--stage',choices=['frame','state','both'],default='both')
    args=parser.parse_args()
    if min(args.batch,args.pairs,args.chunk,args.warmup,args.steps)<1:parser.error('counts must be positive')
    if not torch.cuda.is_available():parser.error('CUDA is required for this benchmark')
    sys.path.insert(0,str(args.source.resolve()))
    import config as cfg
    cfg.USE_AMP=args.amp;cfg.AMP_DTYPE='bfloat16'
    cfg.FUSED_ADAMW=args.fused_adamw
    cfg.FRAME_GENERATION_CHUNK=args.chunk;cfg.VISUAL_CHUNK=args.chunk
    cfg.STATE_VISUAL_FEATURE_DROPOUT=0.
    torch.set_num_threads(1);torch.manual_seed(17)
    result=dict(gpu=torch.cuda.get_device_name(0),batch=args.batch,pairs=args.pairs,
                chunk=args.chunk,amp=args.amp,fused_adamw=args.fused_adamw,stage=args.stage,warmup=args.warmup,steps=args.steps)
    if args.stage in ('frame','both'):result['frame']=frame_probe(args)
    if args.stage in ('state','both'):
        gc.collect();torch.cuda.empty_cache()
        result['state']=state_probe(args)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
