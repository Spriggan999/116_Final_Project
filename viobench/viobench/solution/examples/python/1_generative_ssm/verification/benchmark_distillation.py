"""Compare serial/parallel CUDA distillation on synthetic full-size windows."""
import argparse
from functools import partial
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
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--fused-adamw',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--visual-prediction',choices=['embedding','rgb'],default='embedding')
    args=parser.parse_args()
    if min(args.batch,args.warmup,args.steps,args.repeats)<1 or args.window<11:parser.error('invalid counts')
    if not torch.cuda.is_available():parser.error('CUDA is required')
    import config as cfg
    from performance import adamw
    from distill import frame_targets,train_student
    from distillation_pipeline import DistillationPipeline
    from models import GenerativeFlowPredictor,StateStudent,EmbeddingPredictor
    from students.model import BudgetStudent,PROFILES
    cfg.USE_AMP=args.amp;cfg.USE_GENERATED_FRAMES_FOR_STATE=True;cfg.FUSED_ADAMW=args.fused_adamw
    torch.set_num_threads(1);torch.manual_seed(17)
    embedding=args.visual_prediction=='embedding'
    frame=(EmbeddingPredictor(hidden=256,learned_visual=True,visual_input_stride=8) if embedding else
           GenerativeFlowPredictor(base=33,max_flow_pixels=160,input_stride=8)).cuda().eval()
    teacher=StateStudent(hidden=1792,learned_visual=True,visual_input_stride=8,visual_embeddings=embedding).cuda().eval()
    for model in (frame,teacher):
        for parameter in model.parameters():parameter.requires_grad_(False)
    b,t=args.batch,args.window
    ticks=torch.arange(t)[None].expand(b,-1);anchors=ticks//10
    imu=torch.zeros(b,t,6);imu[:,:,2]=9.81
    batch=dict(cam_images=torch.rand(b,int(anchors.max())+1,3,480,640),imu=imu,
        anchor_img_idx=anchors,anchor_tick=anchors*10,prev_img_idx=(anchors-1).clamp_min(0),
        prev_span=torch.full((b,t),.05),observed=ticks%10==0,dt=torch.full((b,),.005),
        gravity=torch.tensor([[0.,0.,-9.81]]).repeat(b,1),pos=torch.zeros(b,t,3),
        vel=torch.zeros(b,t,3),rot=torch.eye(3)[None,None].repeat(b,t,1,1),omega=torch.zeros(b,t,3))
    batch={key:value.pin_memory() for key,value in batch.items()}
    prepare=partial(frame_targets,teacher_frame=frame,teacher_state=teacher,device='cuda')
    update=partial(train_student,device='cuda',weights=(1.,.5,1.))

    def probe(parallel):
        torch.manual_seed(31)
        students={}
        for name,spec in PROFILES.items():
            model=BudgetStudent(spec).cuda()
            students[name]=dict(model=model,optimizer=adamw(model.parameters(),lr=3e-4))
        def passes(pipeline,count):
            with pipeline.batches([batch]*count,prepare) as batches:
                for prepared in batches:pipeline.run(students,prepared,update)
        with DistillationPipeline('cuda',students,parallel) as pipeline:
            passes(pipeline,args.warmup)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            elapsed=[]
            for _ in range(args.repeats):
                started=time.perf_counter();passes(pipeline,args.steps);torch.cuda.synchronize()
                elapsed.append(time.perf_counter()-started)
            seconds=sum(elapsed)/len(elapsed)
            return dict(seconds=elapsed,windows_per_second=args.steps*b/seconds,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)

    result=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,batch=b,window=t,
        amp=args.amp,fused_adamw=args.fused_adamw,visual_prediction=args.visual_prediction,steps=args.steps,warmup=args.warmup,repeats=args.repeats,
        profiles=list(PROFILES),scope='Synthetic GRU teacher and selected default students; shared target generation, transfers, backward and AdamW. '
              'Excludes simulator, DataLoader, validation, checkpoint/export and accuracy evaluation.')
    result['serial']=probe(False)
    gc.collect();torch.cuda.empty_cache()
    result['parallel']=probe(True)
    result['speedup']=result['parallel']['windows_per_second']/result['serial']['windows_per_second']
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
