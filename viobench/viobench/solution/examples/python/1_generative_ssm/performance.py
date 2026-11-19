"""Host-only precision, bounded data loading and throughput helpers."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import time

import torch
import config as cfg

_allow_amp=ContextVar('hyperdrone_allow_amp',default=True)
PERFORMANCE_SETTINGS=frozenset(('BATCH','FRAME_PAIR_BATCH','FRAME_GENERATION_CHUNK','VISUAL_CHUNK',
    'COLLECT_INSTANCES','USE_AMP','AMP_DTYPE','LOADER_WORKERS','FRAME_LOADER_WORKERS','LOADER_PREFETCH','PIN_MEMORY',
    'PERSISTENT_WORKERS','FUSED_ADAMW','EMBEDDING_PREDICTION_CHUNK','FRAME_PAIRS_UINT8','TORCH_CPU_THREADS','FRAME_VAL_PAIR_BATCH',
    'HYBRID_ATTENTION_CHUNK','PARALLEL_TRAINING_PHYSICS','COMPACT_STATE_CAMERAS','VECTORIZED_PPO_ROLLOUT'))


def neural_autocast(tensor):
    if (tensor.device.type!='cuda' or not getattr(cfg,'USE_AMP',True)
            or not _allow_amp.get() or torch.jit.is_tracing()):
        return nullcontext()
    return torch.autocast('cuda',dtype=torch.bfloat16)


@contextmanager
def amp_disabled():
    token=_allow_amp.set(False)
    try:yield
    finally:_allow_amp.reset(token)


def fp32_physics(function):
    """Disable ambient autocast; neural subregions can explicitly opt back in."""
    @wraps(function)
    def wrapped(*args,**kwargs):
        candidates=[]
        for a in args:candidates.extend(a.values() if isinstance(a,dict) else [a])
        tensor=next((a for a in candidates if isinstance(a,torch.Tensor)),None)
        context=(torch.autocast(tensor.device.type,enabled=False)
                 if tensor is not None and tensor.device.type in ('cpu','cuda') else nullcontext())
        with context:return function(*args,**kwargs)
    return wrapped


def configure(config):
    if getattr(config,'AMP_DTYPE','bfloat16')!='bfloat16':
        raise ValueError('AMP_DTYPE must be bfloat16; disable USE_AMP for full FP32')
    for key in ('BATCH','FRAME_PAIR_BATCH','FRAME_GENERATION_CHUNK','VISUAL_CHUNK','COLLECT_INSTANCES','LOADER_PREFETCH','EMBEDDING_PREDICTION_CHUNK'):
        if getattr(config,key,1)<1:raise ValueError(f'{key} must be positive')
    if getattr(config,'FRAME_VAL_PAIR_BATCH',8)<1:raise ValueError('FRAME_VAL_PAIR_BATCH must be positive')
    chunk=getattr(config,'HYBRID_ATTENTION_CHUNK',512)
    if type(chunk) is not int or chunk<0:raise ValueError('HYBRID_ATTENTION_CHUNK must be a nonnegative integer')
    for key,default in (('PARALLEL_TRAINING_PHYSICS',False),('COMPACT_STATE_CAMERAS',True),('VECTORIZED_PPO_ROLLOUT',True)):
        if type(getattr(config,key,default)) is not bool:raise ValueError(f'{key} must be a boolean')
    for key in ('LOADER_WORKERS','FRAME_LOADER_WORKERS'):
        if getattr(config,key,0)<0:raise ValueError(f'{key} must be nonnegative')
    if str(config.DEVICE).startswith('cuda'):
        threads=getattr(config,'TORCH_CPU_THREADS',2)
        if type(threads) is not int or threads<1:raise ValueError('TORCH_CPU_THREADS must be a positive integer')
        torch.set_num_threads(threads)
        if config.USE_AMP and not torch.cuda.is_bf16_supported():
            raise ValueError('This GPU does not support BF16; use --no-amp or USE_AMP=false')
    print(f'Performance: BF16 neural layers={config.USE_AMP and str(config.DEVICE).startswith("cuda")}; '
          f'physics/validation FP32; batch={config.BATCH}, pairs={config.FRAME_PAIR_BATCH}, '
          f'generation/visual chunks={config.FRAME_GENERATION_CHUNK}/{config.VISUAL_CHUNK}, '
          f'collector instances={config.COLLECT_INSTANCES}; '
          f'recording compression level={getattr(config,"COLLECT_COMPRESSION_LEVEL",6)}',flush=True)


def loader_options(config,frame_pairs=False):
    key='FRAME_LOADER_WORKERS' if frame_pairs else 'LOADER_WORKERS'
    workers=getattr(config,key,0) if str(config.DEVICE).startswith('cuda') else 0
    result=dict(num_workers=workers,pin_memory=bool(getattr(config,'PIN_MEMORY',True) and str(config.DEVICE).startswith('cuda')))
    if workers:result.update(prefetch_factor=getattr(config,'LOADER_PREFETCH',1),
                             persistent_workers=bool(getattr(config,'PERSISTENT_WORKERS',True)))
    return result


def adamw(parameters,**options):
    """Fuse CUDA parameter updates; retain the ordinary optimizer on CPU."""
    parameters=tuple(parameters)
    if parameters and isinstance(parameters[0],dict):
        # Materialize group generators once so checking device placement does
        # not consume parameters before AdamW constructs its groups.
        parameters=tuple(dict(group,params=tuple(group['params'])) for group in parameters)
        flat=tuple(parameter for group in parameters for parameter in group['params'])
    else:flat=parameters
    options.setdefault('fused',bool(getattr(cfg,'FUSED_ADAMW',True) and flat
                                   and all(parameter.is_cuda for parameter in flat)))
    return torch.optim.AdamW(parameters,**options)


class EpochTimer:
    def __init__(self,device):
        self.device=device;self.samples=0;self.loader_wait_seconds=0.
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        self.start=time.perf_counter()

    def add(self,n):self.samples+=n

    def result(self):
        if str(self.device).startswith('cuda'):torch.cuda.synchronize(self.device)
        seconds=time.perf_counter()-self.start
        result=dict(seconds=seconds,samples=self.samples,samples_per_second=self.samples/max(seconds,1e-9),
                    loader_wait_seconds=self.loader_wait_seconds)
        if str(self.device).startswith('cuda'):
            result['peak_allocated_mib']=torch.cuda.max_memory_allocated(self.device)/2**20
        return result


def timed_batches(loader,timer):
    """Measure main-process waits for input separately from model work."""
    start=time.perf_counter();iterator=iter(loader)
    timer.loader_wait_seconds+=time.perf_counter()-start
    while True:
        start=time.perf_counter()
        try:batch=next(iterator)
        except StopIteration:return
        timer.loader_wait_seconds+=time.perf_counter()-start
        yield batch


def record_epoch(directory,stage,epoch,measurement,data,*,loader=None):
    """Keep useful throughput diagnostics even with W&B disabled."""
    import json
    row=dict(stage=stage,epoch=epoch,**measurement,recording_batch=data.batch_id,
             fresh_data_wait_seconds=data.rolling.last_wait_seconds if data.rolling else 0.,
             recording_pass=data.rolling.passes_on_current if data.rolling else 1)
    if loader is not None:
        # The loader's batch limit includes the curriculum's sequence-length cap;
        # a final partial batch can contain fewer examples.
        row['training_batch_size']=loader.batch_size
        length=getattr(loader.dataset,'length',None)
        if length is not None:row['window_samples']=length
    if data.rolling and getattr(data.rolling,'current',None):
        stats=data.rolling.current/'collection_stats.json'
        if stats.is_file():
            collection=json.loads(stats.read_text())
            row.update(collection_seconds=collection.get('seconds'),
                       collection_build_seconds=collection.get('build_seconds'),
                       collection_rollout_seconds=collection.get('rollout_seconds'),
                       collection_compression_seconds=collection.get('compression_seconds'),
                       collection_write_seconds=collection.get('write_seconds'),
                       collection_recording_bytes=collection.get('recording_bytes'),
                       collection_compression_level=collection.get('compression_level'))
    with (directory/'performance.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    return row


def pipeline_log(measurement):
    return {'pipeline/'+key:measurement[key] for key in (
        'fresh_data_wait_seconds','loader_wait_seconds','collection_seconds',
        'collection_build_seconds','collection_rollout_seconds','training_batch_size',
        'collection_compression_seconds','collection_write_seconds','collection_recording_bytes',
        'collection_compression_level',
        'window_samples','samples_per_second','peak_allocated_mib')
        if measurement.get(key) is not None}
