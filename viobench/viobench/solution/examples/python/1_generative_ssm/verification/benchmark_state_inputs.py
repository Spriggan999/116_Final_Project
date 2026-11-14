"""CUDA state-input transfer/encoding probe; excludes loaders, simulator and state model."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--window',type=int,default=4001)
    parser.add_argument('--steps',type=int,default=3)
    args=parser.parse_args()
    if min(args.window,args.steps)<1:parser.error('counts must be positive')
    if not torch.cuda.is_available():parser.error('CUDA required')
    import config,data
    from models import EmbeddingPredictor
    torch.set_num_threads(2);torch.manual_seed(1701)
    torch.cuda.set_per_process_memory_fraction(.6)
    config.USE_AMP=True;config.USE_GENERATED_FRAMES_FOR_STATE=True
    model=EmbeddingPredictor().cuda().eval().requires_grad_(False)
    ticks=torch.arange(args.window)[None]
    selected=torch.unique_consecutive(torch.arange(0,args.window,10)//4)
    slots=(torch.searchsorted(selected,ticks//4,right=True)-1).clamp_min(0)
    anchors=selected[slots];previous=selected[(slots-1).clamp_min(0)]
    cameras=torch.randint(0,256,(1,1,3,480,640),dtype=torch.uint8).expand(1,(args.window-1)//4+1,3,480,640)
    imu=torch.zeros(1,args.window,6);imu[:,:,2]=9.81
    batch=dict(cam_images=cameras,imu=imu,observed=ticks==anchors*4,
               anchor_img_idx=anchors,prev_img_idx=previous,anchor_tick=anchors*4,
               prev_span=(anchors-previous).float()*.02,dt=torch.tensor([.005]))
    original=data.observed_camera_features
    def old(model,b,observed,device):
        count=observed.shape[1]
        real=observed.reshape(-1).nonzero(as_tuple=True)[0].to(device)
        indices=b['anchor_img_idx'].to(device)
        all_cameras=data.camera_values(b['cam_images'],device)
        return model.encoder(all_cameras[real//count,indices[real//count,real%count]])
    result=dict(scope=__doc__,gpu=torch.cuda.get_device_name(),window=args.window,batch=1,
                native_cameras=cameras.shape[1],selected_cameras=int(batch['observed'].sum()),
                source='Synthetic expanded uint8 frames; no filesystem/IPC measurement.',cases={})
    reference=None
    try:
        for label,function in (('previous',old),('selected_and_chunked',original)):
            data.observed_camera_features=function
            gc.collect();torch.cuda.empty_cache()
            with torch.no_grad():data.generate_200hz_embeddings(model,batch,'cuda')
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            with torch.no_grad():
                for _ in range(args.steps):output=data.generate_200hz_embeddings(model,batch,'cuda')
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            result['cases'][label]=dict(seconds_per_window=seconds/args.steps,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            if reference is None:reference=output.cpu()
            else:
                actual=output.cpu()
                result['max_embedding_error']=float((actual-reference).abs().max())
                torch.testing.assert_close(actual,reference,atol=2e-6,rtol=2e-5)
            del output
    finally:data.observed_camera_features=original
    result['passed']=True
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
