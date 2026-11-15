"""Five completed frozen-peer epochs permit one shared-phase holdout to advance."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from student_curriculum import StagedStudentCurriculum
from test_curriculum import settings


def begin_wait(names=('small','medium','large')):
    cur=StagedStudentCurriculum(settings(),[11,21],stage_batches=3,min_batches=1,plateau_patience=2)
    for epoch in range(1,4):
        frozen=set(cur.paused)
        scores={name:(.8**epoch if name=='large' else 1.) for name in names}
        event=cur.update(f'fresh-{epoch}',{'motion':scores},scores)
        assert not event['promoted']
        assert cur.finish_epoch(epoch,names,frozen,fresh_batch=True) is None
    assert cur.paused==set(names)-{'large'} and cur.sole_holdout_wait_epochs==0
    return cur


def test_exactly_five_completed_epochs_and_same_recording_reuse_counts():
    names=['small','medium','large'];cur=begin_wait()
    gates=deepcopy(cur.state_dict()['student_gates'])
    for epoch in range(4,9):
        assert cur.update('fresh-3',{}, {}) is None
        event=cur.finish_epoch(epoch,names,cur.paused)
        if epoch<8:
            assert event is None and cur.sole_holdout_wait_epochs==epoch-3
            assert cur.state_dict()['student_gates']==gates
            assert cur.finish_epoch(epoch,names,cur.paused) is None
            assert cur.sole_holdout_wait_epochs==epoch-3
    assert event['forced_promotion'] and event['forced_student']=='large'
    assert event['sole_holdout_wait_epochs']==5 and not event['fresh_batch']
    assert event['reasons']['motion']['large']=='sole_holdout_timeout'
    assert event['promoted']==['motion'] and cur.schedule.stages['motion']==1
    assert not cur.paused and cur.sole_holdout_wait_epochs==0


def test_fresh_gate_checks_and_sole_holdout_can_force_while_still_improving():
    cur=begin_wait();event=None
    for epoch in range(4,9):
        frozen=set(cur.paused);scores=dict(small=1.,medium=1.,large=.8**epoch)
        normal=cur.update(f'fresh-{epoch}',{'motion':scores},scores)
        assert not normal['promoted']
        event=cur.finish_epoch(epoch,scores,frozen,fresh_batch=True)
    assert event['forced_promotion'] and event['fresh_batch']


def test_regression_or_holdout_identity_changes_reset_wait():
    cur=begin_wait();names=['small','medium','large']
    for epoch in [4,5]:assert cur.finish_epoch(epoch,names,cur.paused) is None
    frozen=set(cur.paused)
    scores=dict(small=2.,medium=1.,large=.1)
    cur.update('regression',{'motion':scores},scores)
    assert cur.finish_epoch(6,names,frozen) is None
    assert cur.sole_holdout_wait_epochs==0
    cur.paused={'small','large'}
    assert cur.finish_epoch(7,names,{'small','large'}) is None
    assert cur.sole_holdout_student=='medium' and cur.sole_holdout_wait_epochs==1


def test_two_holdouts_single_model_final_stage_and_nonconsecutive_epochs_do_not_force():
    cur=begin_wait();names=['small','medium','large']
    cur.paused={'small'}
    for epoch in range(4,11):assert cur.finish_epoch(epoch,names,{'small'}) is None
    assert cur.sole_holdout_wait_epochs==0
    assert cur.finish_epoch(11,['large'],set()) is None
    cur.paused={'small','medium'}
    assert cur.finish_epoch(12,names,cur.paused) is None
    assert cur.finish_epoch(14,names,cur.paused) is None
    assert cur.sole_holdout_wait_epochs==1
    cur.window_stage=1;cur.schedule.stages=dict(motion=9,imu=9,visual=9)
    assert cur.finish_epoch(15,names,cur.paused) is None
    assert cur.sole_holdout_wait_epochs==0


def test_recovery_keeps_wait_and_group_changes_reset_it():
    cur=begin_wait();names=['small','medium','large']
    for epoch in [4,5,6]:cur.finish_epoch(epoch,names,cur.paused)
    saved=json.loads(json.dumps(cur.state_dict()))
    resumed=StagedStudentCurriculum(settings(),[11,21])
    resumed.load_state_dict(saved,preserve_progress=True,students=names)
    assert resumed.state_dict()==saved
    assert resumed.finish_epoch(7,names,resumed.paused) is None
    assert resumed.finish_epoch(8,names,resumed.paused)['forced_promotion']
    subset=StagedStudentCurriculum(settings(),[11,21])
    subset.load_state_dict(saved,preserve_progress=True,students=['small','large'])
    assert subset.sole_holdout_wait_epochs==0
    older=deepcopy(saved);older.pop('sole_holdout')
    resumed.load_state_dict(older,preserve_progress=True,students=names)
    assert resumed.sole_holdout_wait_epochs==0


def test_timeout_at_horizon_boundary_resets_motion_and_restores_best_checkpoint(tmp_path):
    from students.model import BudgetStudent,PROFILES
    from distillation_checkpoints import StageBestCheckpoints
    cur=begin_wait(('small','large'));cur.schedule.stages['motion']=9
    rows={}
    for name in ['small','large']:
        model=BudgetStudent(PROFILES['1k']);directory=tmp_path/name;directory.mkdir()
        rows[name]=dict(model=model,optimizer=torch.optim.AdamW(model.parameters()),directory=directory)
    checkpoint=StageBestCheckpoints();stage=dict(window=cur.length,levels=cur.schedule.snapshot())
    checkpoint.begin(stage,rows,3);checkpoint.observe(rows,dict(small=.5,large=.8),3)
    expected={name:deepcopy(row['model'].state_dict()) for name,row in rows.items()}
    with torch.no_grad():
        for row in rows.values():next(row['model'].parameters()).add_(1.)
    for epoch in range(4,9):event=cur.finish_epoch(epoch,rows,cur.paused)
    assert event['window_promoted'] and cur.length==21 and cur.schedule.stages['motion']==0
    restored=checkpoint.begin(dict(window=cur.length,levels=cur.schedule.snapshot()),rows,8)
    assert set(restored['students'])==set(rows)
    for name,row in rows.items():
        for key,value in row['model'].state_dict().items():torch.testing.assert_close(value,expected[name][key])


def test_wandb_reports_wait_counter_and_forced_promotion():
    from distillation_tracking import DistillationLogger
    from test_distillation_tracking import FakeRun
    cur=begin_wait();names=['small','medium','large'];run=FakeRun()
    with DistillationLogger(run,names,namespace='supervised') as logger:
        for epoch in range(4,9):
            event=cur.finish_epoch(epoch,names,cur.paused)
            if event:event['student_probe_losses']={}
            logger.curriculum(epoch,cur,event)
        assert run.history[-1]['curriculum/forced_promotion']
        assert run.history[-1]['curriculum/forced_student']=='large'
        assert run.history[-1]['curriculum/sole_holdout_completed_wait_epochs']==5
        assert not run.history[-1]['curriculum/fresh_batch']
        assert run.summary['curriculum/last_forced_promotion']['epoch']==8
