"""Memory fallback must finish before any irreversible student update."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from distill import training_batches,distillation_batch_size
from student_curriculum import DEFAULT_STUDENT_WINDOWS
from distillation_pipeline import memory_headroom,ScheduleMemoryError


def test_distiller_bounds_batches_independently_of_teacher_curriculum():
    assert [distillation_batch_size(8,length) for length in DEFAULT_STUDENT_WINDOWS]==[8,4,2,2,1,1,1,1,1,1]
    assert distillation_batch_size(2,401)==2
    assert distillation_batch_size(8,8001)==1


def test_memory_budget_does_not_count_allocator_cache_as_external_memory():
    # 8 live + 4 cached + 1 external; an earlier 12-unit tensor peak fits
    # within an 85%-of-16 budget. Counting cache as external rejects it.
    report=memory_headroom(16,3,8,12,12)
    assert report['safe']
    assert report['estimated_peak_total_mib']==13/2**20
    assert not memory_headroom(16,1,8,14,12)['safe']  # current usage also matters
    assert not memory_headroom(16,3,8,8,12)['safe']  # genuinely external usage


class FakePipeline:
    parallel=True
    device=torch.device('cuda')
    target_prefetch=True
    def __init__(self):self.tuning=[];self.attempts=[];self.preparations=0
    @contextmanager
    def batches(self,loader,prepare):
        yield ((prepare(batch),None) for batch in loader)
    def autotune(self,active,prepared,update):
        self.attempts.append((len(prepared[0]),self.target_prefetch))
        if self.target_prefetch:raise ScheduleMemoryError([dict(safe=False)])
        result=dict(mode='streams',groups=[list(active)],max_concurrent=1)
        self.tuning.append(result)
        return result


def setup(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *args:None)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    sizes=[]
    def loader(length,batch_size=None):
        sizes.append(batch_size or 4)
        return DataLoader(torch.arange(4),batch_size=batch_size or 4)
    data=SimpleNamespace(ppo_loader=loader)
    run=SimpleNamespace(summary={})
    tracking=SimpleNamespace(phase=lambda name:None)
    return data,run,tracking,sizes


def test_fallback_reduces_batch_then_disables_teacher_prefetch_before_updates(tmp_path,monkeypatch):
    data,run,tracking,sizes=setup(monkeypatch)
    pipeline=FakePipeline();limits={};tuned=set();updates=[]
    with training_batches(pipeline,data,801,{'student':{}},lambda x:x,None,
                          limits,tuned,tracking,run,tmp_path) as (batches,total):
        for batch,_ in batches:updates.extend(batch.tolist())
    assert updates==[0,1,2,3] and total==4
    assert pipeline.attempts==[(4,True),(2,True),(1,True),(1,False)]
    assert sizes==[4,2,1,1] and limits=={801:1}
    assert (801,('student',)) in tuned
    records=[json.loads(line) for line in (tmp_path/'memory_recovery.jsonl').read_text().splitlines()]
    assert [row['action'] for row in records]==['reduce_batch','reduce_batch','disable_target_prefetch']
    assert not json.loads((tmp_path/'scheduling.json').read_text())[0]['target_prefetch']


def test_production_oom_is_never_replayed(tmp_path,monkeypatch):
    data,run,tracking,sizes=setup(monkeypatch)
    pipeline=FakePipeline();pipeline.target_prefetch=False
    with pytest.raises(torch.OutOfMemoryError,match='partial production update'):
        with training_batches(pipeline,data,801,{'student':{}},lambda x:x,None,
                              {},set(),tracking,run,tmp_path) as (batches,total):
            next(batches)
            raise torch.OutOfMemoryError('partial production update')
    assert sizes==[4]
    assert not (tmp_path/'memory_recovery.jsonl').exists()


def test_target_preparation_oom_retries_before_any_student_update(tmp_path,monkeypatch):
    data,run,tracking,sizes=setup(monkeypatch)
    pipeline=FakePipeline();pipeline.target_prefetch=False
    def prepare(batch):
        if len(batch)>1:raise torch.OutOfMemoryError('teacher preparation')
        return batch
    with training_batches(pipeline,data,801,{'student':{}},prepare,None,
                          {},set(),tracking,run,tmp_path) as (batches,total):
        assert [int(batch[0]) for batch,_ in batches]==[0,1,2,3]
    assert sizes==[4,2,1]


def test_autotune_repeats_when_a_previously_seen_training_group_resumes(tmp_path,monkeypatch):
    data,run,tracking,sizes=setup(monkeypatch)
    pipeline=FakePipeline();pipeline.target_prefetch=False;tuned=set()
    for names in (('student','peer'),('peer',),('student','peer')):
        active=dict.fromkeys(names,{})
        with training_batches(pipeline,data,801,active,lambda x:x,None,
                              {},tuned,tracking,run,tmp_path) as (batches,total):
            next(batches)
    assert len(pipeline.attempts)==3 and len(tuned)==2


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('fused',[False,True])
def test_failed_cuda_probe_restores_existing_optimizer_without_target_prefetch(fused):
    from copy import deepcopy
    from functools import partial
    from distill import train_student
    from distillation_pipeline import DistillationPipeline
    from students.model import BudgetStudent,PROFILES
    from test_distillation_pipeline import payload
    torch.set_num_threads(1)
    model=BudgetStudent(PROFILES['10k']).cuda()
    row=dict(model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,fused=fused))
    update=partial(train_student,device='cuda',weights=(1.,.5,1.))
    source=payload('cuda')
    with DistillationPipeline('cuda',['10k'],target_prefetch=False) as pipeline:
        with pipeline.batches([source,source],lambda value:value) as batches:
            for prepared in batches:pipeline.run({'10k':row},prepared,update)
        weights=deepcopy(model.state_dict());optimizer=deepcopy(row['optimizer'].state_dict())
        rng=torch.cuda.get_rng_state()
        with pipeline.batches([source],lambda value:value) as batches:
            with pytest.raises(ScheduleMemoryError) as failure:
                pipeline.autotune({'10k':row},next(batches),update,headroom=1.)
        assert failure.value.candidates and all(not item['safe'] for item in failure.value.candidates)
        for key,value in weights.items():torch.testing.assert_close(model.state_dict()[key],value,atol=0,rtol=0)
        actual=row['optimizer'].state_dict()
        assert actual['param_groups']==optimizer['param_groups']
        for index,state in optimizer['state'].items():
            for key,value in state.items():torch.testing.assert_close(actual['state'][index][key],value,atol=0,rtol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(),rng,atol=0,rtol=0)
