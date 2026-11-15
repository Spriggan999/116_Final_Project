"""Let collector OOM/OptiX compilation retries reserve the training GPU."""
from contextlib import contextmanager
import threading


class GpuCollectionGate:
    """Concurrent training leases, with a fair exclusive collector retry.

    GPU launches may remain asynchronous after a lease exits. An exclusive
    retry blocks new launches, synchronizes them, then releases unused cache.
    """
    def __init__(self,device='cuda'):
        self.device=device
        self.condition=threading.Condition()
        self.local=threading.local()
        self.readers=0
        self.waiting=0
        self.reserved=False
        self.stop=None

    @contextmanager
    def training(self):
        depth=getattr(self.local,'depth',0)
        if depth:
            self.local.depth=depth+1
            try:yield
            finally:self.local.depth=depth
            return
        with self.condition:
            while self.reserved or self.waiting:self.condition.wait()
            self.readers+=1
        self.local.depth=1
        try:yield
        finally:
            self.local.depth=0
            with self.condition:
                self.readers-=1;self.condition.notify_all()

    def release_cache(self):
        import torch
        device=torch.device(self.device)
        if device.type=='cuda':
            with torch.cuda.device(device):
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()

    @contextmanager
    def exclusive(self,stop=None):
        with self.condition:
            self.waiting+=1
            self.condition.notify_all()
            try:
                while self.readers or self.reserved:
                    if stop is not None and stop.is_set():raise RuntimeError('Collection cancelled')
                    self.condition.wait(.1)
                if stop is not None and stop.is_set():raise RuntimeError('Collection cancelled')
                self.reserved=True
            finally:
                self.waiting-=1;self.condition.notify_all()
        try:
            self.release_cache()
            yield
        finally:
            with self.condition:
                self.reserved=False;self.condition.notify_all()

    def cancel(self):
        if self.stop is not None:self.stop.set()
