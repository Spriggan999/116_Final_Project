"""Bounded prefetch lifecycle and serial/parallel student-update equivalence."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from pathlib import Path
import sys
import threading

import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import config as cfg
from distillation_pipeline import DistillationPipeline,prefetched_batches
from distill import train_student,validate_student
from students.model import BudgetStudent,PROFILES


def test_prefetch_overlaps_consumer_and_holds_only_one_following_batch():
    started=threading.Event();release=threading.Event();consumed=[];produced=[]
    def source():
        for index in range(5):
            consumed.append(index)
            yield index
    def prepare(index):
        produced.append(index)
        if index==1:
            started.set()
            assert release.wait(5)
        return index*2
    with ThreadPoolExecutor(max_workers=1) as pool:
        with prefetched_batches(source(),prepare,pool) as batches:
            assert next(batches)==0
            assert started.wait(5)  # preparation runs while consumer owns batch 0
            assert consumed==[0,1] and produced==[0,1]
            release.set()
            assert list(batches)==[2,4,6,8]
    assert produced==list(range(5))


def test_prefetch_joins_writer_on_consumer_failure():
    started=threading.Event();stopped=threading.Event();release=threading.Event()
    def prepare(index):
        if index:
            started.set()
            assert release.wait(5)
            stopped.set()
        return index
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError,match='consumer failed'):
            with prefetched_batches(range(2),prepare,pool) as batches:
                assert next(batches)==0
                assert started.wait(5)
                release.set()
                raise RuntimeError('consumer failed')
        assert stopped.is_set()


def test_slow_next_loader_does_not_delay_current_student_batch():
    started=threading.Event();release=threading.Event()
    def source():
        yield 0
        started.set()
        assert release.wait(5)
        yield 1
    with ThreadPoolExecutor(max_workers=1) as pool:
        with prefetched_batches(source(),lambda value:value,pool) as batches:
            try:
                assert next(batches)==0
                assert started.wait(5)
            finally:
                release.set()
            assert list(batches)==[1]


def test_prefetch_propagates_teacher_failure_and_handles_empty_loader():
    def prepare(index):
        if index:raise ValueError('teacher failed')
        return index
    with ThreadPoolExecutor(max_workers=1) as pool:
        with prefetched_batches([],prepare,pool) as batches:assert list(batches)==[]
        with pytest.raises(ValueError,match='teacher failed'):
            with prefetched_batches(range(3),prepare,pool) as batches:list(batches)


def payload(device):
    imu=torch.randn(2,21,6,device=device)*.1;imu[:,:,2]+=9.81
    inputs=dict(images=torch.rand(2,3,3,24,32,device=device),imu=imu,
                dt=torch.full((2,),.005,device=device),gravity=torch.tensor([[0.,0.,-9.81]],device=device).repeat(2,1),
                ticks=[0,10,20])
    truth=dict(pos=torch.zeros(2,3,3,device=device),vel=torch.zeros(2,3,3,device=device),
               rot=torch.eye(3,device=device)[None,None].repeat(2,3,1,1),omega=torch.zeros(2,3,3,device=device))
    return inputs,truth,{key:value.clone() for key,value in truth.items()},torch.zeros(2,2,9,device=device)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_pipeline_matches_serial_updates_and_independent_profile_stopping(monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1);torch.manual_seed(17)
    monkeypatch.setattr(cfg,'USE_AMP',False)
    profiles=list(PROFILES) if device=='cuda' else ['500','10k','1m']
    initial={name:BudgetStudent(PROFILES[name]) for name in profiles}
    # Prepare copies on the producer stream to exercise cross-stream tensor ownership.
    sources=[payload('cpu') for _ in range(3)]
    snapshots=[deepcopy(source) for source in sources]
    def prepare(source):
        def move(value):
            if isinstance(value,torch.Tensor):return value.to(device).clone()
            if isinstance(value,dict):return {key:move(item) for key,item in value.items()}
            if isinstance(value,tuple):return tuple(move(item) for item in value)
            return value
        return move(source)
    def run(parallel):
        students={}
        for name,original in initial.items():
            model=deepcopy(original).to(device)
            students[name]=dict(model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4))
        losses=[];frozen=None
        with DistillationPipeline(device,profiles,parallel) as pipeline:
            assert pipeline.parallel==(parallel and device=='cuda')
            if pipeline.parallel:
                pipeline.groups=[['100m'],[name for name in profiles if name!='100m']]
            with pipeline.batches(sources,prepare) as batches:
                for step,prepared in enumerate(batches):
                    active=students if step<2 else {name:row for name,row in students.items() if name!='500'}
                    losses.append(pipeline.run(active,prepared,partial(train_student,device=device,weights=(1.,.5,1.))))
                    if step==1:frozen=deepcopy(students['500']['model'].state_dict())
            for key,value in frozen.items():torch.testing.assert_close(students['500']['model'].state_dict()[key],value,rtol=0,atol=0)
            with pipeline.batches(sources,lambda source:prepare(source)[:2]) as batches:
                validation=[pipeline.run(students,batch,partial(validate_student,device=device)) for batch in batches]
        return students,losses,validation
    serial,serial_losses,serial_validation=run(False)
    parallel,parallel_losses,parallel_validation=run(True)
    for name in profiles:
        for key,value in serial[name]['model'].state_dict().items():
            # Official scan backward uses floating-point parallel reductions;
            # Adam can amplify their rounding when a gradient is close to zero.
            tolerance=1e-5 if PROFILES[name].get('kind')=='mamba2' else 2e-6
            torch.testing.assert_close(parallel[name]['model'].state_dict()[key],value,atol=tolerance,rtol=2e-5)
    for expected,actual in zip(serial_losses+serial_validation,parallel_losses+parallel_validation):
        assert actual==pytest.approx(expected,rel=2e-5,abs=2e-6)
    for before,after in zip(snapshots,sources):
        for key in before[1]:torch.testing.assert_close(before[1][key],after[1][key],atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_student_bf16_neural_layers_keep_integrator_and_validation_fp32(monkeypatch):
    if not torch.cuda.is_bf16_supported():pytest.skip('BF16 unavailable')
    monkeypatch.setattr(cfg,'USE_AMP',True)
    from performance import amp_disabled
    model=BudgetStudent(PROFILES['1m']).cuda()
    dtypes=[]
    hook=model.head.register_forward_hook(lambda module,args,result:dtypes.append(result.dtype))
    inputs=payload('cuda')[0]
    prediction=model(**inputs)
    assert set(dtypes)=={torch.bfloat16}
    assert all(value.dtype==torch.float32 for value in prediction.values())
    dtypes.clear()
    with torch.no_grad(),amp_disabled():model(**inputs)
    assert set(dtypes)=={torch.float32}
    hook.remove()
