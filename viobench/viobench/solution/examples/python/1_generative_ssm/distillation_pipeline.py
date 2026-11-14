"""Bounded teacher prefetch and independent student CUDA streams."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch


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
    def __init__(self, device, profiles, parallel=True):
        self.device = torch.device(device)
        self.parallel = parallel and self.device.type == 'cuda'
        self.profiles = tuple(profiles)
        self.pool = None
        self.streams = {}

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
        if not self.parallel:
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

        futures = {name: self.pool.submit(work, name, row) for name, row in active.items()}
        return {name: future.result() for name, future in futures.items()}

    def __exit__(self, *_):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            for stream in (self.teacher_stream, *self.streams.values()):
                stream.synchronize()
