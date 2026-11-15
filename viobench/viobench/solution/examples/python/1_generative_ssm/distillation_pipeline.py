"""Bounded teacher prefetch and independent student CUDA streams."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import time

import torch


def scheduling_candidates(names,costs):
    """Compare serial, uniform concurrency and selective heavy-model isolation."""
    names=list(names);n=len(names);plans=[];seen=set()
    def add(mode,groups):
        key=(mode,tuple(tuple(group) for group in groups))
        if key in seen:return
        seen.add(key)
        plans.append(dict(mode=mode,groups=groups,max_concurrent=max(map(len,groups))))
    add('serial_students',[[name] for name in names])
    for count in sorted({1,min(2,n),min(3,n),min(4,n),n}):
        add('streams',[names[start:start+count] for start in range(0,n,count)])
    ordered=sorted(names,key=lambda name:costs[name],reverse=True)
    if n>2:
        remaining=list(ordered);groups=[]
        while remaining:
            group=[remaining.pop(0)]
            if remaining:group.append(remaining.pop())
            groups.append(group)
        add('streams',groups)  # overlap expensive models with cheap models
        add('streams',[[ordered[0]],ordered[1:]])
        if n>3:add('streams',[[ordered[0]],[ordered[1]],ordered[2:]])
    return plans


@contextmanager
def prefetched_batches(loader, prepare, pool):
    """Prepare at most one batch ahead; join the producer even on interruption."""
    pending = None

    def batches():
        nonlocal pending
        iterator = iter(loader)
        end=object()

        def following():
            # Fetch/collate on the producer too, so a slow next(loader) cannot
            # delay launching students on the already prepared current batch.
            batch=next(iterator,end)
            return end if batch is end else prepare(batch)

        pending=pool.submit(following)
        while pending is not None:
            current = pending.result()
            pending = None
            if current is end:break
            pending=pool.submit(following)
            yield current

    iterator = batches()
    try:
        yield iterator
    finally:
        iterator.close()
        if pending is not None:
            pending.cancel()
            # Preserve the original consumer exception, if any. A producer
            # exception is raised by result() when its batch is consumed.
            if not pending.cancelled():
                pending.exception()


def record_tensors(value, stream):
    """Protect shared labels from allocator reuse on consumer streams."""
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            value.record_stream(stream)
    elif isinstance(value, dict):
        for item in value.values():
            record_tensors(item, stream)
    elif isinstance(value, (list, tuple)):
        for item in value:
            record_tensors(item, stream)


class DistillationPipeline:
    """One teacher and one worker/stream per profile; CPU remains synchronous.

    Each step joins all student updates before advancing, so one model never
    has concurrent optimizer steps. Only read-only inputs/targets are shared.
    """
    def __init__(self, device, profiles, parallel=True, max_concurrent=None):
        self.device = torch.device(device)
        self.parallel = parallel and self.device.type == 'cuda'
        self.profiles = tuple(profiles)
        self.pool = None
        self.streams = {}
        self.max_concurrent=(max_concurrent or len(self.profiles)) if self.parallel else 1
        self.groups=None
        self.serial_students=False
        self.tuning=[]

    def __enter__(self):
        if self.parallel:
            with torch.cuda.device(self.device):
                self.teacher_stream = torch.cuda.Stream(device=self.device)
                self.streams = {name: torch.cuda.Stream(device=self.device) for name in self.profiles}
                # Models and optimizer parameters were initialized on this stream.
                initial = torch.cuda.current_stream(self.device)
                for stream in (self.teacher_stream, *self.streams.values()):
                    stream.wait_stream(initial)
            self.pool = ThreadPoolExecutor(max_workers=len(self.profiles)+1,
                                           thread_name_prefix='distillation')
        return self

    @contextmanager
    def batches(self, loader, prepare):
        if not self.parallel:
            yield ((prepare(batch), None) for batch in loader)
            return

        def produce(batch):
            with torch.cuda.device(self.device), torch.cuda.stream(self.teacher_stream), torch.no_grad():
                payload = prepare(batch)
                ready = torch.cuda.Event()
                ready.record(self.teacher_stream)
                return payload, ready

        with prefetched_batches(loader, produce, self.pool) as batches:
            yield batches

    def run(self, active, prepared, operation):
        payload, ready = prepared
        if not self.parallel or self.serial_students:
            if self.parallel:
                stream=torch.cuda.current_stream(self.device)
                stream.wait_event(ready)
                record_tensors(payload,stream)
            return {name: float(operation(name, row, payload).detach()) for name, row in active.items()}

        def work(name, row):
            stream = self.streams[name]
            with torch.cuda.device(self.device), torch.cuda.stream(stream):
                stream.wait_event(ready)
                record_tensors(payload, stream)
                result = operation(name, row, payload).detach()
                # Join this stream only. Other students and teacher prefetch keep
                # running; metrics/checkpointing can safely use completed updates.
                stream.synchronize()
                return float(result)

        names=list(active);results={}
        groups=self.groups or [names[start:start+self.max_concurrent] for start in range(0,len(names),self.max_concurrent)]
        for group in groups:
            futures={name:self.pool.submit(work,name,active[name]) for name in group if name in active}
            errors=[]
            for name,future in futures.items():
                try:results[name]=future.result()
                except BaseException as error:errors.append(error)
            # All workers must finish before errors propagate or weights reload.
            if errors:raise errors[0]
        return results

    def autotune(self,active,prepared,operation,headroom=.15):
        """Probe real updates, then restore weights/optimizer/RNG exactly.

        OOM retries happen only here, never after a production optimizer step.
        Teacher prefetch remains enabled even for one-student groups.
        """
        if not self.parallel:return {'max_concurrent':1,'candidates':[]}
        def cpu_copy(value):
            if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
            if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
            if isinstance(value,list):return [cpu_copy(v) for v in value]
            if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
            return deepcopy(value)
        torch.cuda.synchronize(self.device)
        snapshots={name:(cpu_copy(row['model'].state_dict()),cpu_copy(row['optimizer'].state_dict()),row['model'].training)
                   for name,row in active.items()}
        rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state(self.device)
        candidates=[];costs={}
        original=(self.max_concurrent,self.groups,self.serial_students)
        selected=False
        def restore():
            torch.cuda.synchronize(self.device)
            for name,row in active.items():
                weights,optimizer,training=snapshots[name]
                row['optimizer'].zero_grad(set_to_none=True)
                row['model'].load_state_dict(weights);row['optimizer'].load_state_dict(optimizer)
                row['model'].train(training)
            torch.set_rng_state(rng);torch.cuda.set_rng_state(cuda_rng,self.device)
            torch.cuda.empty_cache()
        try:
            self.max_concurrent=1;self.groups=None;self.serial_students=False
            for name,row in active.items():
                try:
                    self.run({name:row},prepared,operation)
                    start=time.perf_counter();self.run({name:row},prepared,operation)
                    costs[name]=time.perf_counter()-start
                finally:restore()
            for plan in scheduling_candidates(active,costs):
                try:
                    self.max_concurrent=plan['max_concurrent'];self.groups=plan['groups']
                    self.serial_students=plan['mode']=='serial_students'
                    self.run(active,prepared,operation)  # compilation/warmup
                    torch.cuda.reset_peak_memory_stats(self.device)
                    start=time.perf_counter()
                    for _ in range(2):self.run(active,prepared,operation)
                    elapsed=(time.perf_counter()-start)/2
                    torch.cuda.empty_cache()
                    free,total=torch.cuda.mem_get_info(self.device)
                    peak=torch.cuda.max_memory_allocated(self.device)
                    external=max(0,total-free-torch.cuda.memory_allocated(self.device))
                    candidates.append(dict(plan,seconds=elapsed,
                        peak_allocated_mib=peak/2**20,estimated_peak_total_mib=(peak+external)/2**20,
                        free_mib=free/2**20,safe=peak+external<=(1-headroom)*total))
                except torch.OutOfMemoryError:
                    candidates.append(dict(plan,safe=False,error='CUDA out of memory'))
                finally:restore()
            safe=[row for row in candidates if row['safe']]
            if not safe:raise RuntimeError('No distillation schedule leaves 15% VRAM headroom; reduce --batch or recording concurrency')
            best=min(safe,key=lambda row:row['seconds'])
            self.max_concurrent=best['max_concurrent'];self.groups=best['groups']
            self.serial_students=best['mode']=='serial_students';selected=True
            result=dict(mode=best['mode'],groups=self.groups,max_concurrent=self.max_concurrent,
                        student_seconds=costs,candidates=candidates)
            self.tuning.append(result)
            return result
        finally:
            if not selected:self.max_concurrent,self.groups,self.serial_students=original
            snapshots.clear()

    def __exit__(self, *_):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            for stream in (self.teacher_stream, *self.streams.values()):
                stream.synchronize()
