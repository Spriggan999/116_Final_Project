"""Bounded CUDA probes of embedding batches and hybrid AMP/local attention.

Synthetic compute/transfer measurements, excluding simulator/data loading and
accuracy evaluation. No running training checkpoints or datasets are modified.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def measured(step,units,warmup,steps):
    for _ in range(warmup):step()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for _ in range(steps):step()
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    return dict(seconds_per_step=seconds/steps,samples_per_second=units*steps/seconds,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)


def embedding_probe(batch,dimension,chunk,args):
    import config
    from models import EmbeddingPredictor
    from embedding_training import embedding_objective
    from performance import adamw
    config.DEVICE='cuda';config.USE_AMP=True;config.VISUAL_CHUNK=chunk
    model=EmbeddingPredictor(hidden=config.EMBEDDING_HIDDEN,embedding_dim=dimension).cuda().train()
    values={key:torch.randint(0,256,(batch,3,480,640),dtype=torch.uint8).pin_memory()
            for key in ('previous','anchor','target')}
    values.update(imu=torch.zeros(batch,6).pin_memory(),elapsed=torch.full((batch,1),.025).pin_memory(),
                  prev_elapsed=torch.full((batch,1),.05).pin_memory(),
                  photometric=torch.tensor([[1.1,1.1,.5]]).repeat(batch,1).pin_memory())
    optimizer=adamw((p for p in model.parameters() if p.requires_grad),lr=config.LR_FRAME)
    def step():
        optimizer.zero_grad(set_to_none=True)
        loss,_=embedding_objective(model,values,config)
        if not torch.isfinite(loss):raise ValueError('Nonfinite embedding loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();model.update_target()
    return measured(step,batch,args.warmup,args.steps)


def hybrid_probe(batch,length,dimension,amp,chunk,parallel,args):
    import config
    from models import HybridStateStudent
    from performance import adamw,amp_disabled
    config.USE_AMP=amp;config.HYBRID_ATTENTION_CHUNK=chunk;config.PARALLEL_TRAINING_PHYSICS=parallel
    model=HybridStateStudent(hidden=args.hidden,visual_embeddings=True,embedding_dim=dimension).cuda().train()
    with torch.no_grad():model.out.weight.normal_(0,.001)
    visual=torch.randn(batch,length,dimension,device='cuda')*.1
    imu=torch.zeros(batch,length,6,device='cuda');imu[:,:,2]=9.81
    observed=torch.arange(length,device='cuda')[None].expand(batch,-1)%10==0
    dt=torch.full((batch,),.005,device='cuda');gravity=torch.tensor([[0.,0.,-9.81]],device='cuda').repeat(batch,1)
    optimizer=adamw(model.parameters(),lr=config.LR_STUDENT)
    dtypes=[];handle=model.feature.register_forward_hook(lambda m,a,out:dtypes.append(str(out.dtype)))
    def step():
        optimizer.zero_grad(set_to_none=True)
        prediction=model(visual,imu,observed,dt,gravity,True)
        loss=prediction['pos'].square().mean()+.1*prediction['vel'].square().mean()+prediction['corrections'].square().mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite hybrid loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    try:
        result=measured(step,batch,args.warmup,args.steps)
        result['training_feature_dtype']=dtypes[-1]
        model.eval()
        with torch.no_grad(),amp_disabled():prediction=model(visual,imu,observed,dt,gravity,True)
        result['validation_feature_dtype']=dtypes[-1]
        result['physics_fp32']=all(v.dtype==torch.float32 for v in prediction.values())
        result['validation_finite']=all(bool(torch.isfinite(v).all()) for v in prediction.values())
        return result
    finally:handle.remove()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['embedding','hybrid','all'],default='all')
    parser.add_argument('--dimensions',type=int,nargs='+',default=[48,96])
    parser.add_argument('--batches',type=int,nargs='+',default=[64,128])
    parser.add_argument('--windows',type=int,nargs='+',default=[401,1601,4001])
    parser.add_argument('--state-batch',type=int,default=2)
    parser.add_argument('--hidden',type=int,default=896)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--memory-fraction',type=float,default=.6)
    parser.add_argument('--reserve-gib',type=float,default=2.)
    parser.add_argument('--variants',nargs='+',choices=['legacy','bounded','amp','parallel'],
                        default=['legacy','bounded','amp','parallel'])
    args=parser.parse_args()
    if min(args.dimensions+args.batches+args.windows+[args.state_batch,args.hidden,args.warmup,args.steps])<1:
        parser.error('counts must be positive')
    if not 0<args.memory_fraction<=.8 or args.reserve_gib<1:parser.error('invalid memory guard')
    if not torch.cuda.is_available():parser.error('CUDA required')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    result=dict(scope=__doc__,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                settings=vars(args)|{'output':str(args.output)},cases=[])
    def run_case(kind,settings,function):
        gc.collect();torch.cuda.empty_cache()
        free,total=torch.cuda.mem_get_info()
        row=dict(stage=kind,**settings,free_mib_before=free/2**20)
        if free < args.reserve_gib*2**30+min(total*args.memory_fraction,4*2**30):
            row.update(skipped='Insufficient free memory for a bounded probe')
        else:
            torch.manual_seed(1701)
            try:row.update(function(),passed=True)
            except torch.cuda.OutOfMemoryError as error:row.update(passed=False,error='Probe memory limit: '+str(error))
        result['cases'].append(row)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    for dimension in args.dimensions:
        if args.stage in ('embedding','all'):
            for batch in args.batches:
                run_case('embedding',dict(dimension=dimension,batch=batch,visual_chunk=16),
                         lambda:embedding_probe(batch,dimension,16,args))
        if args.stage in ('hybrid','all'):
            for length in args.windows:
                batch=min(args.state_batch,max(1,3208//length))
                variants=dict(legacy=(False,0,False),bounded=(False,512,False),
                              amp=(True,512,False),parallel=(True,512,True))
                for variant in args.variants:
                    amp,chunk,parallel=variants[variant]
                    run_case('hybrid',dict(dimension=dimension,batch=batch,window=length,amp=amp,
                             attention_chunk=chunk,parallel_physics=parallel),
                             lambda:hybrid_probe(batch,length,dimension,amp,chunk,parallel,args))


if __name__=='__main__':main()
