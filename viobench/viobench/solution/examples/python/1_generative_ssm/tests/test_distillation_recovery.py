"""Stage recovery preserves each student's curriculum and stopping progress."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from distillation_curriculum import DistillationCurriculum
from distillation_checkpoints import StageBestCheckpoints
from distillation_recovery import load_recovery,save_recovery,records
from streaming import FreshBatchPatience,RollingBatches
from test_curriculum import settings
from test_distillation_freezing import paused_curriculum
from test_streaming_ppo import stream_cfg
from distillation_tracking import DistillationLogger
from test_distillation_tracking import FakeRun


def test_recovery_keeps_individual_gates_readiness_pause_and_exposure_exactly():
    original=paused_curriculum()
    state=deepcopy(original.state_dict());state['final_cycles']=2
    recovered=DistillationCurriculum(settings(),[11],stage_batches=3,min_batches=1)
    recovered.load_state_dict(state,preserve_progress=True,students=('500','10k'))
    actual=recovered.state_dict()
    for key in ('stages','window_stage','student_gates','readiness','plateaued','paused','normalizers','final_cycles'):
        assert actual[key]==state[key]
    assert recovered.seen==original.seen and recovered.paused=={'500'}
    assert set(recovered.training_students({'500':{},'10k':{}}))=={'10k'}
    run=FakeRun()
    with DistillationLogger(run,['500','10k']) as logger:
        logger.curriculum(9,recovered,None)
        assert run.history[-1]['curriculum/500/paused']
        assert run.history[-1]['curriculum/500/motion_ready']
        assert run.history[-1]['curriculum/10k/motion_checks']==original.gate('motion','10k').age
    recovered.gate('motion','10k').age+=1
    assert recovered.gate('motion','10k').age!=original.gate('motion','10k').age


def test_stage_best_index_and_selection_survive_recovery():
    stage=dict(window=801,levels=dict(motion=.5,visual=.5,imu=.5))
    sources={name:dict(stage=stage,stage_index=1,epoch=epoch,validation_mean_ate_m=.7)
             for name,epoch in (('500',8),('10k',7))}
    checkpoints=StageBestCheckpoints();checkpoints.recover(sources)
    assert checkpoints.index==1 and checkpoints.stage==stage
    assert checkpoints.best['500']['epoch']==8 and checkpoints.best['10k']['epoch']==7
    assert checkpoints.begin(stage,{'500':{},'10k':{}},9) is None


def test_legacy_nine_epoch_run_recovers_counters_and_next_scene_batch(tmp_path):
    (tmp_path/'distillation.json').write_text(json.dumps(dict(curriculum_settings=dict(CURRICULUM_EARLY_STOP_GRACE=3))))
    curriculum=paused_curriculum().state_dict()
    (tmp_path/'curriculum_state.json').write_text(json.dumps(curriculum))
    event=dict(epoch=6,promoted=['motion'],window_promoted=False,recording_batch='/old/batch-000005')
    (tmp_path/'curriculum_metrics.jsonl').write_text(json.dumps(event)+'\n')
    for name in ('500','10k'):
        folder=tmp_path/name;folder.mkdir()
        rows=[dict(epoch=epoch,active=True,best_validation_mean_ate_m=.5,
                   non_improving_data_cycles=1 if name=='10k' else 0,recording_batch=f'/old/batch-{batch:06d}')
              for epoch,batch in ((6,5),(7,5),(8,6),(9,7))]
        (folder/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (tmp_path/'performance.jsonl').write_text(json.dumps(dict(epoch=9,recording_batch='/old/batch-000007'))+'\n')
    recovered=load_recovery(tmp_path,('500','10k'))
    assert recovered['completed_epochs']==9 and recovered['collection_next_index']==8
    assert recovered['curriculum']==curriculum and recovered['legacy_migration']
    assert recovered['students']['10k']['patience']==dict(bad_cycles=1,last_batch='/old/batch-000007',grace_remaining=1)


def test_atomic_recovery_metadata_keeps_exact_stopping_state(tmp_path):
    curriculum=paused_curriculum();students={}
    for name in ('500','10k'):
        patience=FreshBatchPatience();patience.bad_cycles=4;patience.last_batch='old-batch';patience.grace_remaining=2
        students[name]=dict(active=name!='10k',best=.5,patience=patience)
    checkpoint=StageBestCheckpoints();checkpoint.index=2
    saved=save_recovery(tmp_path,17,curriculum,students,checkpoint,SimpleNamespace(current=Path('/old/batch-000013')))
    actual=load_recovery(tmp_path,('500','10k'))
    assert actual==saved and actual['curriculum_stage_index']==2
    assert actual['collection_next_index']==14 and not actual['students']['10k']['active']
    assert actual['students']['500']['patience']==dict(bad_cycles=4,last_batch='old-batch',grace_remaining=2)
    assert not (tmp_path/'recovery_state.json.tmp').exists()


def test_collection_resumes_at_saved_index_with_same_scene_and_seed_rules(tmp_path):
    specs=[]
    def collect(directory,spec,stop):
        specs.append((directory.name,spec))
        (directory/'rollout_s000_e000.npz').write_bytes(b'complete')
    with RollingBatches(tmp_path,stream_cfg(NUM_SCENES=313,STREAM_SCENES=4),collect=collect,initial_index=8) as batches:
        assert batches.current.name=='batch-000008'
        batches.future.result(timeout=5)
    current=next(spec for name,spec in specs if name=='batch-000008')
    assert current['scene_offset']==32 and current['seed']==17+1009*8


def test_legacy_metrics_ignore_an_incomplete_last_record(tmp_path):
    path=tmp_path/'metrics.jsonl';path.write_text('{"epoch":9}\n{"epoch":')
    assert records(path)==[dict(epoch=9)]
