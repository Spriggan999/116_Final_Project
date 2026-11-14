"""CUDA probe of direct embeddings versus RGB prediction; no accuracy result."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--window',type=int,default=101)
    parser.add_argument('--hidden',type=int,default=1792)
    parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if min(args.batch,args.window,args.hidden,args.steps,args.warmup,args.repeats)<1:parser.error('counts must be positive')
    if not torch.cuda.is_available():parser.error('CUDA required')
    import config as cfg
    from data import state_input_frames
    from export import make_models
    torch.set_num_threads(1);torch.manual_seed(17)
    cfg.USE_GENERATED_FRAMES_FOR_STATE=True;cfg.USE_AMP=True
    b,t=args.batch,args.window
    ticks=torch.arange(t)[None].expand(b,-1);anchors=ticks//10
    imu=torch.zeros(b,t,6);imu[:,:,2]=9.81
    batch=dict(cam_images=torch.rand(b,int(anchors.max())+1,3,480,640),imu=imu,anchor_img_idx=anchors,
        anchor_tick=anchors*10,prev_img_idx=(anchors-1).clamp_min(0),prev_span=(anchors>0).float()*.05,
        observed=ticks%10==0,dt=torch.full((b,),.005),gravity=torch.tensor([[0.,0.,-9.81]]).repeat(b,1))
    batch={key:value.pin_memory() for key,value in batch.items()}

    def probe(mode):
        torch.manual_seed(17)
        frame,state=make_models(dict(frame_mode=mode,backbone='gru',hidden=args.hidden,learned_visual=True,
            width=640,height=480,frame_generator_base=33,frame_input_stride=8,visual_input_stride=8,max_flow_pixels=160.))
        for model in (frame,state):model.cuda().eval().requires_grad_(False)
        imu=batch['imu'].cuda();observed=batch['observed'].cuda();dt=batch['dt'].cuda();gravity=batch['gravity'].cuda()
        shape=None
        @torch.no_grad()
        def step():
            nonlocal shape
            visual=state_input_frames(frame,batch,'cuda');shape=list(visual.shape)
            state(visual,imu,observed,dt,gravity)
        for _ in range(args.warmup):step()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        elapsed=[]
        for _ in range(args.repeats):
            started=time.perf_counter()
            for _ in range(args.steps):step()
            torch.cuda.synchronize();elapsed.append(time.perf_counter()-started)
        count=1
        for size in shape:count*=size
        return dict(seconds=elapsed,windows_per_second=b*args.steps/(sum(elapsed)/len(elapsed)),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,visual_sequence_shape=shape,
            visual_sequence_mib=count*4/2**20,
            real_or_generated_images_encoded_per_step=int(batch['observed'].sum()) if mode=='embedding' else b*t)

    result=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,batch=b,window=t,hidden=args.hidden,
        steps=args.steps,warmup=args.warmup,repeats=args.repeats,
        scope='Synthetic frozen full-size GRU teacher inference, online visual prediction and host transfers. '
              'Excludes simulator, loading, backward, optimizer, export and accuracy evaluation.')
    result['rgb']=probe('flow')
    gc.collect();torch.cuda.empty_cache()
    result['embedding']=probe('embedding')
    result['throughput_ratio']=result['embedding']['windows_per_second']/result['rgb']['windows_per_second']
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
