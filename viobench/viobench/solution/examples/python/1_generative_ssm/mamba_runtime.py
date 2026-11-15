"""Thread-safe host launch of the official Mamba Triton kernels.

Triton's Autotuner stores per-call arguments on its shared instance. Concurrent
first-use calls from teacher/student threads can overwrite/clear those arguments.
Protect individual launchers, not model forwards or GPU streams: cached launches
remain asynchronous and their GPU kernels can execute concurrently.
"""
from functools import wraps
import sys
from threading import RLock

_registry_lock=RLock()


def protect_official_launchers():
    from triton.runtime.autotuner import Autotuner
    with _registry_lock:
        for name,module in tuple(sys.modules.items()):
            if not name.startswith('mamba_ssm.') or module is None:continue
            for value in tuple(vars(module).values()):
                if not isinstance(value,Autotuner) or getattr(value,'_hyperdrone_launch_lock',None) is not None:continue
                lock=RLock();original=value.run
                @wraps(original)
                def run(*args,_original=original,_lock=lock,**kwargs):
                    with _lock:return _original(*args,**kwargs)
                value.run=run
                value._hyperdrone_launch_lock=lock
