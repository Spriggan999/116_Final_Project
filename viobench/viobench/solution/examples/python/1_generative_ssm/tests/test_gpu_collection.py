"""Collector GPU recovery drains training work and retries only once."""
from pathlib import Path
import sys
import threading

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gpu_collection import GpuCollectionGate
from streaming import RollingBatches,CollectorProcessError
from distillation_pipeline import DistillationPipeline
from test_streaming_ppo import stream_cfg


def test_training_remains_concurrent_but_exclusive_retry_drains_and_blocks_launches():
    gate=GpuCollectionGate('cpu')
    entered=[threading.Event(),threading.Event()];release=[threading.Event(),threading.Event()]
    nested=threading.Event();retry_entered=threading.Event();retry_release=threading.Event();late=threading.Event()
    def train(index):
        with gate.training():
            entered[index].set()
            assert release[index].wait(5)
            if index==0:
                with gate.training():nested.set()  # reentrant autotune/run
    def retry():
        with gate.exclusive():
            assert gate.readers==0
            retry_entered.set();assert retry_release.wait(5)
    def following():
        with gate.training():late.set()
    threads=[threading.Thread(target=train,args=(index,)) for index in range(2)]
    for thread in threads:thread.start()
    assert all(event.wait(5) for event in entered)
    writer=threading.Thread(target=retry);writer.start()
    with gate.condition:assert gate.condition.wait_for(lambda:gate.waiting==1,timeout=5)
    last=threading.Thread(target=following);last.start()
    release[0].set();assert nested.wait(5)
    assert not retry_entered.is_set() and not late.is_set()
    release[1].set();assert retry_entered.wait(5)
    assert not late.is_set()
    retry_release.set()
    for thread in [*threads,writer,last]:thread.join(5);assert not thread.is_alive()
    assert late.is_set() and gate.readers==0 and not gate.reserved


def test_cancelled_retry_does_not_leave_a_pending_writer():
    gate=GpuCollectionGate('cpu');stop=threading.Event();errors=[]
    def retry():
        try:
            with gate.exclusive(stop):pytest.fail('Cancelled retry acquired GPU')
        except RuntimeError as error:errors.append(str(error))
    with gate.training():
        writer=threading.Thread(target=retry);writer.start()
        with gate.condition:assert gate.condition.wait_for(lambda:gate.waiting==1,timeout=5)
        stop.set();writer.join(5)
        assert not writer.is_alive() and gate.waiting==0
    assert errors==['Collection cancelled']
    with gate.training():assert gate.readers==1


def test_pipeline_preparation_and_updates_use_gpu_leases_without_serializing_normal_work():
    gate=GpuCollectionGate('cpu')
    def prepare(batch):assert gate.readers==1;return batch
    def update(name,row,payload):assert gate.readers==1;return torch.tensor(1.)
    # Exercise lease wiring without requiring CUDA kernels in this unit test.
    with DistillationPipeline('cuda',['student'],parallel=False,gpu_gate=gate) as pipeline:
        with pipeline.batches([1],prepare) as batches:
            assert pipeline.run({'student':{}},next(batches),update)=={'student':1.}
    assert gate.readers==0


@pytest.mark.parametrize('diagnostic,counter',[
    ('cudaMalloc failed: out of memory','oom'),
    ('optixModuleCreate failed with code 7999\nCOMPILE ERROR: Module compilation failed','optix_compile'),
])
def test_retry_is_exclusive_single_flight_and_does_not_publish_partial_files(tmp_path,diagnostic,counter):
    gate=GpuCollectionGate('cpu');attempts=[];cleared=[]
    gate.release_cache=lambda:cleared.append(True)
    def collect(directory,spec,stop):
        if directory.name=='batch-000001':
            attempts.append(spec['instances'])
            if len(attempts)==1:
                (directory/'partial.npz').write_bytes(b'partial')
                raise CollectorProcessError(2,diagnostic)
            assert gate.reserved and gate.readers==0 and not (directory/'partial.npz').exists()
        (directory/'rollout_s000_e000.npz').write_bytes(b'complete')
    with RollingBatches(tmp_path,stream_cfg(COLLECT_INSTANCES=2),collect=collect,gpu_gate=gate) as batches:
        current=batches.current
        batches.future.result(timeout=5)
        assert current.exists()
        assert (batches.validation/'rollout_s000_e000.npz').read_bytes()==b'complete'
        batches.next_epoch()
        assert attempts==[2,1] and cleared==[True]
        assert getattr(batches,counter+'_retries')==1 and getattr(batches,counter+'_recoveries')==1
        other='oom' if counter=='optix_compile' else 'optix_compile'
        assert getattr(batches,other+'_retries')==0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('diagnostic',[
    'CUDA_ERROR_OUT_OF_MEMORY',
    'optixModuleCreate failed with code 7999\nCOMPILE ERROR: Module compilation failed',
])
def test_persistent_gpu_failure_has_only_one_retry(tmp_path,diagnostic):
    attempts=[];gate=GpuCollectionGate('cpu')
    def collect(directory,spec,stop):
        if directory.name=='batch-000001':
            attempts.append(spec['instances'])
            raise CollectorProcessError(2,diagnostic)
        (directory/'rollout_s000_e000.npz').write_bytes(b'complete')
    with pytest.raises(CollectorProcessError):
        with RollingBatches(tmp_path,stream_cfg(COLLECT_INSTANCES=2),collect=collect,gpu_gate=gate) as batches:
            batches.next_epoch()
            batches.future.exception(timeout=5)
            batches.next_epoch()
    assert attempts==[2,1] and not list(tmp_path.iterdir())


def test_unrelated_compiler_error_is_not_retried(tmp_path):
    attempts=[]
    def collect(directory,spec,stop):
        attempts.append(True)
        raise CollectorProcessError(2,'COMPILE ERROR: invalid source')
    with pytest.raises(CollectorProcessError,match='invalid source'):
        with RollingBatches(tmp_path,stream_cfg(),collect=collect,gpu_gate=GpuCollectionGate('cpu')):pass
    assert len(attempts)==1 and not list(tmp_path.iterdir())


@pytest.mark.parametrize('diagnostic,counter',[
    ('cudaMalloc failed: out of memory','out_of_memory'),
    ('optixModuleCreate failed with code 7999\nCOMPILE ERROR: Module compilation failed','optix_compile_failure'),
])
def test_subprocess_retains_failure_signature_before_long_compiler_tail(tmp_path,diagnostic,counter):
    script=tmp_path/'failed.py'
    script.write_text(f'import sys\nprint({diagnostic!r})\nfor i in range(300): print("Info",i)\nsys.exit(2)\n')
    batches=RollingBatches(tmp_path/'managed',stream_cfg(),collector=script)
    try:
        with pytest.raises(CollectorProcessError) as failure:
            batches._subprocess(tmp_path,dict(scene_offset=0,scenes=1,episodes=1,seed=1,
                steps=41,width=32,height=24,max_bytes=1024),threading.Event())
        assert getattr(failure.value,counter)
        assert diagnostic in failure.value.output
        assert len(failure.value.output.splitlines())<=84
        if counter=='optix_compile_failure':assert not failure.value.out_of_memory
    finally:batches.close()


def test_subprocess_error_retains_bounded_native_diagnostics(tmp_path):
    script=tmp_path/'failed.py'
    script.write_text('import sys\nfor i in range(100): print("context",i)\nprint("cudaMalloc failed: out of memory")\nsys.exit(2)\n')
    batches=RollingBatches(tmp_path/'managed',stream_cfg(),collector=script)
    try:
        with pytest.raises(CollectorProcessError) as failure:
            batches._subprocess(tmp_path,dict(scene_offset=0,scenes=1,episodes=1,seed=1,
                steps=41,width=32,height=24,max_bytes=1024),threading.Event())
        assert failure.value.out_of_memory
        assert 'cudaMalloc failed: out of memory' in str(failure.value)
        assert len(failure.value.output.splitlines())<=80
        assert not CollectorProcessError(2,'invalid mesh').out_of_memory
    finally:batches.close()
