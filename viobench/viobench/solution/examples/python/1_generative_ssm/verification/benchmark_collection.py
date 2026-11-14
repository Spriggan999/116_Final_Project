"""Measure long-recording RAM and real CUDA training steps; no accuracy claims."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available():parser.error('CUDA required')
    import config as cfg
    from data import EpisodeStore,FramePairs,StateWindows,state_input_frames
    from embedding_training import embedding_objective
    from losses import losses
    from models import EmbeddingPredictor,HybridStateStudent
    from performance import adamw,loader_options
    from streaming import TrainingData
    from torch.utils.data import DataLoader
    torch.set_num_threads(4);torch.manual_seed(1701)
    started=time.perf_counter()
    train=EpisodeStore(sorted(args.train.glob('rollout_s*_e*.npz')),cfg.H,cfg.W)
    validation=EpisodeStore(sorted(args.validation.glob('rollout_s*_e*.npz')),cfg.H,cfg.W)
    entries=train.entries+validation.entries
    camera_bytes=sum(e['cams'].numel()*e['cams'].element_size() for e in entries)
    assert all(e['cams'].dtype==torch.uint8 for e in entries)
    result=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        stored_camera_bytes=camera_bytes,float32_camera_bytes_equivalent=camera_bytes*4,
        load_seconds=time.perf_counter()-started,passed=False,checks={})
    for name,directory,store in [('train',args.train,train),('validation',args.validation,validation)]:
        manifest=json.loads((directory/'manifest.json').read_text())
        result[name]=dict(episodes=len(store.entries),scenes=sorted({r['scene_index'] for r in manifest}),
            ticks=[len(e['imu']) for e in store.entries],captures=sum(len(e['cams']) for e in store.entries),
            recording_bytes=sum(p.stat().st_size for p in directory.iterdir() if p.is_file()),
            collection=json.loads((directory/'collection_stats.json').read_text()))
    frame=EmbeddingPredictor(hidden=cfg.EMBEDDING_HIDDEN,visual_input_stride=cfg.VISUAL_INPUT_STRIDE).cuda()
    pairs=DataLoader(FramePairs(train,max_samples=cfg.MAX_FRAME_TRAIN_PAIRS),batch_size=cfg.FRAME_PAIR_BATCH,
        **loader_options(cfg,frame_pairs=True))
    optimizer=adamw((p for p in frame.parameters() if p.requires_grad),lr=cfg.LR_FRAME)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    loss,_=embedding_objective(frame,next(iter(pairs)),cfg)
    loss.backward();assert torch.isfinite(loss)
    optimizer.step();frame.update_target();torch.cuda.synchronize()
    result['checks']['embedding_step']=dict(loss=float(loss.detach()),seconds=time.perf_counter()-start,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    del optimizer;frame.eval().requires_grad_(False)
    model=HybridStateStudent(hidden=cfg.STUDENT_HIDDEN,n_mamba=cfg.HYBRID_NUM_MAMBA_LAYERS,
        attn_window=cfg.HYBRID_ATTENTION_WINDOW,d_state=cfg.HYBRID_D_STATE,headdim=cfg.HYBRID_HEAD_DIM,
        learned_visual=True,visual_input_stride=cfg.VISUAL_INPUT_STRIDE,visual_embeddings=True).cuda()
    with torch.no_grad():model.out.weight.normal_(0,.001)
    options=loader_options(cfg)
    if options['num_workers']:options['multiprocessing_context']='spawn'
    loader=DataLoader(StateWindows(train,length=cfg.WINDOW),batch_size=cfg.BATCH,**options)
    optimizer=adamw(model.parameters(),lr=cfg.LR_STUDENT,weight_decay=1e-3)
    try:
        batch=next(iter(loader));torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        embeddings=state_input_frames(frame,batch,'cuda')
        prediction=model(embeddings,*[batch[key].cuda(non_blocking=True) for key in ['imu','observed','dt','gravity']])
        loss,_=losses(prediction,batch,'cuda',metrics=False);assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step();torch.cuda.synchronize()
        result['checks']['state_step']=dict(loss=float(loss.detach()),seconds=time.perf_counter()-start,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,loader_workers=options['num_workers'],
            window=cfg.WINDOW,batch=cfg.BATCH)
    finally:TrainingData._close_loader(loader)
    result.update(passed=True,process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  elapsed_seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
