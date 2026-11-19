"""Experiment-owned scratch configuration and collector cancellation."""
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import getpass
import legacy
from streaming import RollingBatches
from resource_limits import ResourceLimitError, filesystem, existing_parent


@contextmanager
def scratch_environment(root):
    root=Path(root)
    mappings=dict(TMPDIR=root/'tmp',CONTA_CACHE=root/'assets',WANDB_CACHE_DIR=root/'wandb-cache',
                  WANDB_DATA_DIR=root/'wandb-data',TORCHINDUCTOR_CACHE_DIR=root/'inductor',
                  TRITON_CACHE_DIR=root/'triton',TORCH_EXTENSIONS_DIR=root/'torch-extensions',
                  HYPERLUX_CACHE_DIR=root/'hyperlux',XDG_CACHE_HOME=root/'xdg',CUDA_CACHE_PATH=root/'cuda')
    previous={name:os.environ.get(name) for name in mappings}
    previous_search=os.environ.get('CONTA_SEARCH_PATHS')
    old_temp=tempfile.tempdir
    parallel={name:os.environ.get(name) for name in ('CMAKE_BUILD_PARALLEL_LEVEL','MAX_JOBS','HYPERLUX_BUILD_JOBS')}
    old_cache=previous['CONTA_CACHE'] or str(Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'conta')
    searches=[p for p in (previous_search or '').split(os.pathsep) if p]
    legacy_cache=Path('/var/tmp')/('hyperdrone-'+getpass.getuser())/'scene-assets'
    if legacy_cache.exists():searches.append(str(legacy_cache))
    try:
        for name,path in mappings.items():
            path.mkdir(parents=True,exist_ok=True)
            os.environ[name]=str(path)
        os.environ['CONTA_SEARCH_PATHS']=os.pathsep.join(dict.fromkeys(searches+[old_cache]))
        tempfile.tempdir=str(mappings['TMPDIR'])
        for name,value in parallel.items():
            try:count=min(2,max(1,int(value or '2')))
            except ValueError:count=2
            os.environ[name]=str(count)
        yield
    finally:
        tempfile.tempdir=old_temp
        for name,value in previous.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value
        for name,value in parallel.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value
        if previous_search is None:os.environ.pop('CONTA_SEARCH_PATHS',None)
        else:os.environ['CONTA_SEARCH_PATHS']=previous_search


class SafeRollingBatches(RollingBatches):
    def __init__(self,root,cfg,guard,skip_validation=False,**kwargs):
        self.skip_validation=skip_validation
        root=Path(root).resolve()
        if root!=guard.root and guard.root not in root.parents:
            raise ResourceLimitError('--stream-dir must be inside the watched --resource-dir')
        if filesystem(existing_parent(root))[0] in ('tmpfs','ramfs'):
            raise ResourceLimitError('Recording --stream-dir must be disk-backed, not /tmp or /dev/shm on this host')
        if cfg.STREAM_MAX_MIB > guard.limits.max_scratch_mib:
            raise ResourceLimitError('Recording budget exceeds --max-scratch-mib')
        self.guard=guard
        super().__init__(root,cfg,collector=Path(__file__).with_name('gated_collect.py'),**kwargs)
        guard.cancellers.append(self.stop.set)

    def _generate(self,name,index):
        # A fixed --validation-manifest replaces the per-run held-out flights;
        # collecting ~0.9 GB that is never read would only cost time and disk.
        if index<0 and self.skip_validation:
            directory=self.owned/name
            directory.mkdir()
            return directory
        return super()._generate(name,index)

    def _subprocess(self,directory,spec,stop):
        self.guard.check()
        (directory/'collection_spec.json').write_text(json.dumps(dict(spec,resource_limits=asdict(self.guard.limits))))
        return super()._subprocess(directory,spec,stop)
