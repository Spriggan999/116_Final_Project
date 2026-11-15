"""Two-second horizon steps and preservation of older five-horizon progress."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from student_curriculum import StagedStudentCurriculum,DEFAULT_STUDENT_WINDOWS,LEGACY_STUDENT_WINDOWS
from test_curriculum import settings


def test_all_ten_horizons_run_motion_before_horizon_and_reset_motion():
    assert DEFAULT_STUDENT_WINDOWS==[401,801,1201,1601,2001,2401,2801,3201,3601,4001]
    cur=StagedStudentCurriculum(settings(),DEFAULT_STUDENT_WINDOWS,stage_batches=3,min_batches=1,plateau_patience=2)
    horizons=[cur.length]
    for epoch in range(2000):
        event=cur.update(epoch,{axis:{'student':1.} for axis in cur.probe_axes},{'student':1.})
        if event['window_promoted']:
            assert event['levels']['motion']==1.
            assert cur.schedule.stages['motion']==0
            assert cur.schedule.stages['imu']==cur.schedule.stages['visual']==0
            horizons.append(cur.length)
        if cur.window_stage<9:assert cur.schedule.stages['imu']==cur.schedule.stages['visual']==0
        if cur.complete:break
    assert cur.complete and horizons==DEFAULT_STUDENT_WINDOWS
    assert [round((n-1)*.005) for n in horizons]==list(range(2,21,2))


@pytest.mark.parametrize('index',range(5))
def test_five_horizon_checkpoint_keeps_current_length_and_difficulty(index):
    old=StagedStudentCurriculum(settings(),LEGACY_STUDENT_WINDOWS)
    old.window_stage=index;old.schedule.stages['motion']=5
    snapshot=json.loads(json.dumps(old.state_dict()))
    new=StagedStudentCurriculum(settings(),DEFAULT_STUDENT_WINDOWS)
    new.load_state_dict(snapshot,preserve_progress=True,students=['student'])
    assert new.length==LEGACY_STUDENT_WINDOWS[index]
    assert new.schedule.stages['motion']==5
    assert new.windows==DEFAULT_STUDENT_WINDOWS
    assert new.window_stage==DEFAULT_STUDENT_WINDOWS.index(old.length)
    assert new.horizon_migration['preserved_horizon']==old.length
    assert snapshot['windows']==LEGACY_STUDENT_WINDOWS
    saved=json.loads(json.dumps(new.state_dict()))
    restored=StagedStudentCurriculum(settings(),DEFAULT_STUDENT_WINDOWS)
    restored.load_state_dict(saved,preserve_progress=True,students=['student'])
    assert restored.state_dict()==saved


def test_migration_preserves_frozen_peers_plateau_history_and_holdout_timeout():
    old=StagedStudentCurriculum(settings(),LEGACY_STUDENT_WINDOWS)
    old.window_stage=2;old.schedule.stages['motion']=5
    for epoch in range(1,9):
        frozen=set(old.paused);scores={'small':1.,'large':.8**epoch}
        old.update(epoch,{'motion':scores},scores)
        old.finish_epoch(epoch,scores,frozen)
    assert old.sole_holdout_wait_epochs==2
    saved=deepcopy(old.state_dict())
    new=StagedStudentCurriculum(settings(),DEFAULT_STUDENT_WINDOWS)
    new.load_state_dict(saved,preserve_progress=True,students=['small','large'])
    assert new.length==1601 and new.window_stage==3
    assert new.paused=={'small'} and new.sole_holdout_wait_epochs==2
    assert new.state_dict()['student_gates']==saved['student_gates']
    for epoch in range(9,12):event=new.finish_epoch(epoch,['small','large'],new.paused)
    assert event['forced_promotion'] and new.schedule.stages['motion']==6


def test_custom_incompatible_horizon_schedule_still_rejected():
    old=StagedStudentCurriculum(settings(),LEGACY_STUDENT_WINDOWS)
    new=StagedStudentCurriculum(settings(),[401,1001,4001])
    with pytest.raises(ValueError,match='windows/levels differ'):new.load_state_dict(old.state_dict(),preserve_progress=True)
