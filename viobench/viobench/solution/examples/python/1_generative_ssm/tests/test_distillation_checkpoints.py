"""Stage-local selection restores matching model weights and Adam state."""
from copy import deepcopy
from functools import partial
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from distillation_checkpoints import StageBestCheckpoints,cpu_snapshot
from distillation_pipeline import DistillationPipeline
from distillation_tracking import DistillationLogger
from distill import train_student
from students.model import BudgetStudent,ALL_PROFILES as PROFILES
from test_distillation_pipeline import payload
from test_distillation_tracking import FakeRun


def equal_state(actual,expected):
    if isinstance(expected,torch.Tensor):torch.testing.assert_close(actual.cpu(),expected.cpu(),atol=0,rtol=0)
    elif isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key in expected:equal_state(actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected)
        for value,reference in zip(actual,expected):equal_state(value,reference)
    else:assert actual==expected


@pytest.mark.parametrize('device,fused',[('cpu',False),('cuda',False),('cuda',True)])
def test_restores_best_stage_epoch_and_matching_optimizer_before_next_update(tmp_path,device,fused):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1);torch.manual_seed(17)
    model=BudgetStudent(PROFILES['10k']).to(device)
    row=dict(model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,fused=fused),directory=tmp_path)
    students={'10k':row};source=payload(device)
    operation=partial(train_student,device=device,weights=(1.,.5,1.))
    checkpoints=StageBestCheckpoints()
    first=dict(window=11,levels=dict(visual=0.,imu=0.,motion=0.))
    second=dict(window=11,levels=dict(visual=.5,imu=0.,motion=0.))
    assert checkpoints.begin(first,students,0) is None
    with DistillationPipeline(device,students) as pipeline:
        with pipeline.batches([source]*3,lambda value:value) as batches:
            for epoch,prepared in enumerate(batches,1):
                pipeline.run(students,prepared,operation)
                checkpoints.observe(students,{'10k':{1:.8,2:.5,3:.9}[epoch]},epoch)
                if epoch==2:
                    weights=cpu_snapshot(model.state_dict());optimizer=cpu_snapshot(row['optimizer'].state_dict())
        row['optimizer'].param_groups[0]['lr']=.1
        report=checkpoints.begin(second,students,3)
        assert report['students']['10k']['epoch']==2
        equal_state(model.state_dict(),weights);equal_state(row['optimizer'].state_dict(),optimizer)
        assert not checkpoints.best
        # A new stage must select its own completed epochs, even when its best
        # score is worse than the previous stage's score.
        checkpoints.observe(students,{'10k':1.2},4)
        saved=torch.load(tmp_path/'stage_best.pt',map_location='cpu',weights_only=True)
        assert saved['stage']==second and saved['epoch']==4 and saved['validation_mean_ate_m']==1.2
        reference=BudgetStudent(PROFILES['10k']).to(device);reference.load_state_dict(weights)
        reference_row=dict(model=reference,optimizer=torch.optim.AdamW(reference.parameters(),lr=3e-4,fused=fused))
        reference_row['optimizer'].load_state_dict(deepcopy(optimizer))
        operation('reference',reference_row,source)
        with pipeline.batches([source],lambda value:value) as batches:
            pipeline.run(students,next(batches),operation)
        equal_state(model.state_dict(),reference.state_dict())
        equal_state(row['optimizer'].state_dict(),reference_row['optimizer'].state_dict())


def test_failed_atomic_stage_save_preserves_previous_checkpoint_and_selection(tmp_path,monkeypatch):
    model=BudgetStudent(PROFILES['500'])
    row=dict(model=model,optimizer=torch.optim.AdamW(model.parameters()),directory=tmp_path)
    checkpoints=StageBestCheckpoints();checkpoints.begin({'window':11}, {'500':row},0)
    checkpoints.observe({'500':row},{'500':.5},1)
    original=(tmp_path/'stage_best.pt').read_bytes()
    def fail(value,path):
        path.write_bytes(b'incomplete')
        raise OSError('simulated interrupted save')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError,match='interrupted save'):
        checkpoints.observe({'500':row},{'500':.4},2)
    assert (tmp_path/'stage_best.pt').read_bytes()==original
    assert checkpoints.best['500']['epoch']==1


def test_stage_restore_is_visible_in_wandb():
    run=FakeRun()
    report=dict(to_stage_index=1,students={'500':dict(epoch=2,validation_mean_ate_m=.5,optimizer_restored=True)})
    with DistillationLogger(run,['500']) as logger:
        logger.stage_restore(7,report)
        assert run.history[-1]['curriculum/500/restored_epoch']==2
        assert run.history[-1]['curriculum/500/optimizer_restored']
        assert run.summary['distillation/last_stage_restore']==report
