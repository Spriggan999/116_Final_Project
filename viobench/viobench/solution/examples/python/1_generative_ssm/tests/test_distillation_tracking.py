"""Live progress, consolidated epoch records and successful/error run lifecycle."""
from pathlib import Path
import sys

import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from distillation_tracking import DistillationLogger


class FakeRun:
    def __init__(self):
        self.summary={};self.history=[];self.metrics=[];self.finished=[]

    def define_metric(self,name,**kwargs):self.metrics.append((name,kwargs))
    def log(self,values,**kwargs):
        assert 'step' not in kwargs  # live updates must not make epoch logs stale
        self.history.append(dict(values))
    def finish(self,exit_code):self.finished.append(exit_code)


def records():
    return {name:dict(train_loss=.2,validation_state_loss=.3,validation_mean_ate_m=.4,
        best_validation_mean_ate_m=.35,active=name!='500',learning_rate=.0003,
        non_improving_data_cycles=8 if name=='500' else 0) for name in
        ('500','10k','1m','100m','1m_ssm','1m_hybrid','100m_hybrid')}


def test_live_updates_are_throttled_and_all_students_share_one_epoch_record():
    run=FakeRun();now=[0.]
    rows=records()
    with DistillationLogger(run,rows,clock=lambda:now[0]) as logger:
        logger.progress(1,1,4,401,{'500':.2,'10k':.4},2)
        count=len(run.history)
        now[0]=1.;logger.progress(1,2,4,401,{'500':.1},2)
        assert len(run.history)==count
        now[0]=12.;logger.progress(1,3,4,401,{'500':.1},2)
        now[0]=13.;logger.progress(1,4,4,401,{'500':.1},2)
        assert run.history[-1]['progress/epoch_percent']==100
        logger.epoch(1,rows,dict(seconds=4.,samples_per_second=20.,peak_allocated_mib=800.,max_concurrent=2),1.5)
        event=run.history[-1]
        assert event['distillation/fresh_data_wait_seconds']==1.5
        assert all(name+'/validation_mean_ate_m' in event for name in rows)
        assert run.summary['500/status']=='early_stopped'
        logger.phase('exporting:500')
        assert not run.finished
    assert run.finished==[0] and run.summary['distillation/status']=='completed'


@pytest.mark.parametrize('error,code,status',[(RuntimeError('export failed'),1,'failed'),(KeyboardInterrupt(),130,'interrupted')])
def test_failure_or_interrupt_does_not_report_a_successful_run(error,code,status):
    run=FakeRun()
    with pytest.raises(type(error)):
        with DistillationLogger(run,['500']) as logger:
            logger.phase('exporting:500')
            raise error
    assert run.finished==[code] and run.summary['distillation/status']==status


def test_real_offline_wandb_persists_live_and_all_student_epoch_metrics(tmp_path):
    wandb=pytest.importorskip('wandb')
    run=wandb.init(project='hyperdrone-distillation-log-test',dir=str(tmp_path),mode='offline',
                   settings=wandb.Settings(x_disable_stats=True,quiet=True))
    rows=records()
    with DistillationLogger(run,rows) as logger:
        logger.phase('training')
        logger.progress(1,1,1,401,{'500':.2},2)
        logger.epoch(1,rows,dict(seconds=2.,samples_per_second=12.,max_concurrent=2))
        logger.phase('exporting:500')
    files=list(tmp_path.glob('wandb/offline-run-*/run-*.wandb'))
    assert len(files)==1
    # The SDK journals uncompressed protobuf records. Check persisted keys
    # without depending on its removed legacy Python datastore reader.
    saved=files[0].read_bytes()
    assert all((name+'/validation_mean_ate_m').encode() in saved for name in rows)
    assert b'progress/epoch_percent' in saved
    assert b'completed' in saved and b'distillation/status' in saved
