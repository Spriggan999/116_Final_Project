"""Bounded seven-student pilot with a real frozen teacher and recorded sensors.

Timings are measurements; convergence times remain conditional projections.
No W&B run, full training run, simulator launch, or website upload is started.
"""
import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))


def project_times(stages,validation_seconds,training_recordings,validation_recordings,collection_seconds=0,*,overlap=False,max_passes=3):
    """Epoch budgets use measured shared throughput, not a sum of solo costs."""
    costs=[]
    for row in stages:
        batches=math.ceil(row['windows_per_recording']*training_recordings/row['batch'])
        cost=batches*row['selected']['seconds_per_batch']+validation_seconds*validation_recordings
        # With bounded reuse, a collector-limited run can amortize one fresh
        # batch across at most max_passes epochs. Decode/startup remain extra.
        costs.append(max(cost,collection_seconds/max_passes) if overlap else cost+collection_seconds)
    result={}
    for epochs in (100,300,500):
        intro=sum(6*cost for cost in costs[:-1])
        result[str(epochs)]=intro+max(0,epochs-6*(len(costs)-1))*costs[-1]
    result['curriculum_seconds']=sum(6*cost for cost in costs[:-1])
    result['earliest_stale_stop_seconds']=result['curriculum_seconds']+8*costs[-1]
    result['earliest_stale_stop_with_3_pass_reuse_seconds']=result['curriculum_seconds']+24*costs[-1]
    if overlap:
        row=stages[-1]
        base=math.ceil(row['windows_per_recording']*training_recordings/row['batch'])*row['selected']['seconds_per_batch']+validation_seconds*validation_recordings
        result['earliest_stale_stop_seconds']=result['curriculum_seconds']+8*max(base,collection_seconds)
        result['earliest_stale_stop_with_3_pass_reuse_seconds']=result['curriculum_seconds']+8*max(max_passes*base,collection_seconds)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--validation-recording',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--windows',type=int,nargs='+',default=[401,801,1601,2401,4001])
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--max-seconds',type=float,default=1200)
    parser.add_argument('--training-recordings',type=int,default=8)
    parser.add_argument('--validation-recordings',type=int,default=12)
    parser.add_argument('--collection-seconds-per-batch',type=float,default=0)
    parser.add_argument('--collection-stats',type=Path,help='collection_stats.json for an ideal-overlap streaming forecast')
    parser.add_argument('--verify-frames',type=int,default=1200)
    parser.add_argument('--parallel-teacher-physics',action=argparse.BooleanOptionalAction,default=True)
    args=parser.parse_args()
    if min(args.batch,args.steps,args.training_recordings,args.validation_recordings,args.verify_frames)<1 or min(args.windows)<11:
        parser.error('invalid counts')
    if args.windows!=sorted(set(args.windows)):parser.error('--windows must be increasing and distinct')
    if not math.isfinite(args.max_seconds) or args.max_seconds<=0:parser.error('--max-seconds must be positive and finite')
    if not math.isfinite(args.collection_seconds_per_batch) or args.collection_seconds_per_batch<0:
        parser.error('--collection-seconds-per-batch must be finite and nonnegative')
    if not torch.cuda.is_available():parser.error('CUDA required')
    args.out.mkdir(parents=True,exist_ok=False)
    import config as cfg
    from performance import configure,adamw,amp_disabled
    saved=json.loads((args.teacher/'training_config.json').read_text())
    for key,value in saved.items():
        if key.isupper() and hasattr(cfg,key):setattr(cfg,key,value)
    cfg.DEVICE='cuda';cfg.BATCH=args.batch;cfg.USE_AMP=True
    configure(cfg);torch.manual_seed(17)
    from distill import load_teacher,frame_targets,train_student,validation_inputs,select_architecture
    from students.model import BudgetStudent,PROFILES
    from students.export import export_student
    from distillation_pipeline import DistillationPipeline
    from data import EpisodeStore,StateWindows,StateEpisodes
    from torch.utils.data import DataLoader
    frame,teacher,settings=load_teacher(args.teacher,'cuda')
    teacher_hash=hashlib.sha256((args.teacher/'state_student.pt').read_bytes()).hexdigest()
    cfg.H,cfg.W=settings['height'],settings['width']
    cfg.USE_GENERATED_FRAMES_FOR_STATE=settings['use_generated_frames'];cfg.VISUAL_PREDICTION=settings['visual_prediction']
    cfg.CACHE_GENERATED_WINDOWS=False
    store=EpisodeStore([args.recording],cfg.H,cfg.W)
    val_store=EpisodeStore([args.validation_recording],cfg.H,cfg.W) if args.validation_recording else store
    if any(int(row['cam_ticks'][-1])+1<args.windows[-1] for row in val_store.entries):
        parser.error('validation recordings must cover the final calibration horizon')
    result=dict(gpu=torch.cuda.get_device_name(),teacher=str(args.teacher.resolve()),teacher_sha256=teacher_hash,
        recording=str(args.recording.resolve()),validation_recording=str(args.validation_recording or args.recording),
        provisional=True,architectures={},resources={},stages=[],
        parallel_teacher_physics=args.parallel_teacher_physics,
        assumptions=dict(training_recordings=args.training_recordings,validation_recordings=args.validation_recordings,
            collection_seconds_per_batch=args.collection_seconds_per_batch),
        limitations=['A bounded timing pilot cannot predict epochs to convergence.',
            'Epoch estimates extrapolate one recording; flight lengths and window counts must match.',
            'Steady-state timings reuse one collated minibatch; exclude ongoing loader/collation and NPZ loading.',
            'Collection is not launched; supply measured collection time for an inclusive projection.',
            'Export probes and compilation are reported separately from steady-state GPU time.'])
    # Resource probes live under --out (use /tmp); only small reports need copying into the repo.
    students={};probe_start=time.perf_counter()
    for name,spec in PROFILES.items():
        directory=args.out/name;directory.mkdir()
        if spec.get('kind')=='mamba2':spec=select_architecture(name,directory,frames=args.verify_frames)
        else:spec=dict(spec,profile=name)
        model=BudgetStudent(spec)
        export_student(model.eval(),directory/'native',frames=args.verify_frames)
        result['resources'][name]=json.loads((directory/'native/limits.json').read_text())
        result['architectures'][name]=spec
        model=model.cuda()
        students[name]=dict(model=model,optimizer=adamw(model.parameters(),lr=3e-4))
    result['resource_probe_seconds']=time.perf_counter()-probe_start
    (args.out/'architectures.json').write_text(json.dumps(result['architectures'],indent=2)+'\n')
    prepare=partial(frame_targets,teacher_frame=frame,teacher_state=teacher,device='cuda',
                    parallel_teacher_physics=args.parallel_teacher_physics)
    update=partial(train_student,device='cuda',weights=(1.,.5,1.))
    def timed(function,steps=None):
        function()  # warmup, excluded
        samples=[]
        for _ in range(steps or args.steps):
            torch.cuda.synchronize();start=time.perf_counter();function();torch.cuda.synchronize()
            samples.append(time.perf_counter()-start)
        return dict(seconds=sum(samples)/len(samples),samples=samples)
    started=time.perf_counter()
    for length in args.windows:
        if time.perf_counter()-started>args.max_seconds:break
        dataset=StateWindows(store,length=length,training=True,compact_cameras=True)
        batch_size=min(args.batch,max(1,3208//length))
        batch=next(iter(DataLoader(dataset,batch_size=batch_size,pin_memory=True)))
        actual_batch=len(batch['imu'])
        teacher_timing=timed(lambda:prepare(batch))
        payload=prepare(batch);torch.cuda.synchronize()
        if args.parallel_teacher_physics and length==args.windows[-1]:
            reference=prepare(batch,parallel_teacher_physics=False)
            result['teacher_target_parity']=dict(
                correction_max_abs=float((payload[3]-reference[3]).abs().max()),
                state_max_abs={key:float((payload[2][key]-reference[2][key]).abs().max()) for key in payload[2]})
            torch.testing.assert_close(payload[3],reference[3],atol=3e-6,rtol=3e-5)
            reference=None
        solo={name:timed(lambda name=name,row=row:update(name,row,payload)) for name,row in students.items()}
        payload=None
        schedules=[]
        for mode,count in [('serial',1),('grouped',2),('streams',len(students))]:
            with DistillationPipeline('cuda',students,mode!='serial',count) as pipeline:
                def step():
                    with pipeline.batches([batch]*args.steps,prepare) as batches:
                        for prepared in batches:pipeline.run(students,prepared,update)
                torch.cuda.reset_peak_memory_stats()
                try:
                    timing=timed(step,steps=2)
                    torch.cuda.empty_cache();free,total=torch.cuda.mem_get_info()
                    peak=torch.cuda.max_memory_allocated();external=max(0,total-free-torch.cuda.memory_allocated())
                    schedules.append(dict(mode=mode,max_concurrent=count,seconds_per_batch=timing['seconds']/args.steps,
                        free_mib=free/2**20,peak_allocated_mib=peak/2**20,safe=peak+external<=.85*total))
                except torch.OutOfMemoryError:
                    raise RuntimeError('Pilot exceeded VRAM; rerun with a smaller --batch. Production updates were not started.') from None
        safe=[row for row in schedules if row['safe']]
        if not safe:raise RuntimeError('No pilot schedule leaves 15% VRAM free; reduce --batch')
        selected=min(safe,key=lambda row:row['seconds_per_batch'])
        row=dict(window=length,batch=actual_batch,windows_per_recording=len(dataset),teacher=teacher_timing,
                 students=solo,schedules=schedules,selected=selected)
        result['stages'].append(row)
        print(f"Pilot {length}: {selected['mode']} {selected['seconds_per_batch']:.4f}s/shared batch",flush=True)
        (args.out/'timing.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['stages']:raise RuntimeError('No timing stages completed')
    batch=next(iter(DataLoader(StateEpisodes(val_store),batch_size=1)))
    inputs,truth,metadata=validation_inputs(batch,'cuda')
    validation={}
    with torch.no_grad(),amp_disabled():
        for name,row in students.items():
            row['model'].eval()
            validation[name]=timed(lambda row=row:row['model'](**inputs),steps=2)['seconds']
    result['validation_seconds_per_recording']=validation
    result['portfolio_projection_seconds']=project_times(result['stages'],sum(validation.values()),args.training_recordings,
        args.validation_recordings,args.collection_seconds_per_batch)
    result['student_projection_seconds']={}
    for name in students:
        solo_stages=[dict(row,selected=dict(seconds_per_batch=row['teacher']['seconds']+row['students'][name]['seconds'])) for row in result['stages']]
        result['student_projection_seconds'][name]=project_times(solo_stages,validation[name],args.training_recordings,
            args.validation_recordings,args.collection_seconds_per_batch)
        # Provisional pilot weights, not converged checkpoints or deployable trained exports.
        torch.save({key:value.detach().cpu() for key,value in students[name]['model'].state_dict().items()},args.out/name/'pilot_student.pt')
        (args.out/name/'student.json').write_text(json.dumps(result['architectures'][name],indent=2)+'\n')
    result['complete_horizon_calibration']=len(result['stages'])==len(args.windows)
    if args.collection_stats and result['complete_horizon_calibration']:
        stats=json.loads(args.collection_stats.read_text())
        if stats['episodes']<1 or stats['seconds']<=0:raise ValueError('Collection stats need positive episodes and duration')
        seconds=stats['seconds']*args.training_recordings/stats['episodes']
        result['collection_measurement']=dict(source=str(args.collection_stats),stats=stats,estimated_seconds_per_training_batch=seconds)
        result['streaming_projection_seconds']=project_times(result['stages'],sum(validation.values()),args.training_recordings,
            args.validation_recordings,seconds,overlap=True)
    if not result['complete_horizon_calibration']:
        result['portfolio_projection_seconds']=None;result['student_projection_seconds']={}
    result['pilot_wall_seconds']=time.perf_counter()-started
    if hashlib.sha256((args.teacher/'state_student.pt').read_bytes()).hexdigest()!=teacher_hash:
        raise RuntimeError('Teacher file changed during pilot')
    (args.out/'timing.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Conditional timing projections','',
           'Measured with a real frozen teacher and recordings. Epoch budgets are not predictions of convergence.',
           'Compute projections reuse a collated batch and exclude CPU loading, validation preprocessing and initial collection/setup.',
           f"Assumptions: {args.training_recordings} training recordings, {args.validation_recordings} validation recordings per epoch; collection allowance {args.collection_seconds_per_batch:g} s/epoch.",'',
           '| Profile | Measured max fuel/frame | Solo 100 epochs (h) | Solo 300 epochs (h) | Solo 500 epochs (h) |',
           '|---|---:|---:|---:|---:|']
    for name,times in result['student_projection_seconds'].items():
        lines.append(f"| {name} | {result['resources'][name]['fuel_step_max']:,} | {times['100']/3600:.2f} | {times['300']/3600:.2f} | {times['500']/3600:.2f} |")
    if result['portfolio_projection_seconds']:
        times=result['portfolio_projection_seconds']
        lines+=['',f"Shared-teacher portfolio: {times['100']/3600:.2f} / {times['300']/3600:.2f} / {times['500']/3600:.2f} hours for 100 / 300 / 500 epochs with all seven active.",
                f"Earliest stale-stop scenario after the curriculum: {times['earliest_stale_stop_seconds']/3600:.2f}–{times['earliest_stale_stop_with_3_pass_reuse_seconds']/3600:.2f} h, assuming no further qualifying improvements."]
    else:lines+=['','Pilot budget ended before every horizon; full-run projections are unavailable.']
    if 'streaming_projection_seconds' in result:
        times=result['streaming_projection_seconds'];seconds=result['collection_measurement']['estimated_seconds_per_training_batch']
        lines+=['',f"Measured collection scales to {seconds:.1f} s per training batch. With at most three passes and ideal generation/training overlap, the portfolio projection is {times['100']/3600:.2f} / {times['300']/3600:.2f} / {times['500']/3600:.2f} h for 100 / 300 / 500 epochs.",
                'This extrapolates one scene and excludes initial validation collection, dataset decoding, and GPU interference between rendering and training.']
    (args.out/'timing.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['portfolio_projection_seconds'],indent=2))


if __name__=='__main__':main()
