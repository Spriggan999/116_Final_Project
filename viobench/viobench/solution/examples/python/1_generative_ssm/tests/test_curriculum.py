"""Curriculum consistency, progression, sampling, and complete CPU training."""
import copy
from contextlib import ExitStack
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import config
from curriculum import (AugmentedDataset,DifficultyCurriculum,LearningProgress,ProgressGate,
                        curriculum_sampler,difficult_images,noisy_imu,validate_config,state_batch_size)
from data import EpisodeStore,FramePairs,StateWindows,frame_pair_images
from test_streaming_ppo import FAKE_COLLECTOR
from streaming import TrainingData


def settings(**overrides):
    values={key:copy.deepcopy(value) for key,value in vars(config).items() if key.isupper()}
    values.update(USE_CURRICULUM=True,CURRICULUM_MIN_EPOCHS=1,CURRICULUM_PATIENCE=1)
    values.update(overrides)
    return SimpleNamespace(**values)


def recorded_store(tmp_path):
    files=[]
    for scene,speed in enumerate((0.,1.,4.)):
        n=81
        state=np.zeros((n,18),np.float32);state[:,3:12]=np.eye(3).ravel()
        state[:,12]=speed;state[:,0]=np.arange(n)*.005*speed
        imu=np.zeros((n,6),np.float32);imu[1:,2]=9.81
        path=tmp_path/f'rollout_s{scene:03d}_e000.npz'
        camera=np.random.default_rng(scene).integers(30,220,(17,24,32,3),dtype=np.uint8)
        np.savez(path,camera=camera,camera_ticks=np.arange(0,n,5),state=state,imu=imu,
                 dt=np.float32(.005),frame_stride=5,scene_id=scene)
        files.append(path)
    return EpisodeStore(files,24,32)


def test_axes_advance_independently_on_fixed_probe_improvements():
    schedule=DifficultyCurriculum(settings())
    assert schedule.update(dict(motion=10.,visual=10.,imu=10.))==[]
    assert schedule.update(dict(motion=9.,visual=10.1,imu=10.1))==['motion']
    assert schedule.snapshot()==dict(motion=.5,visual=0.,imu=0.)
    assert schedule.update(dict(motion=9.,visual=9.,imu=10.1))==['visual']
    assert schedule.snapshot()==dict(motion=.5,visual=.5,imu=0.)


def test_noisy_windows_share_flight_errors_without_changing_truth_or_timing(tmp_path):
    store=recorded_store(tmp_path)
    originals=[{key:entry[key].clone() for key in ('imu','state','cams')} for entry in store.entries]
    cfg=settings(CURRICULUM_MIX=[1.,0.,0.])
    schedule=DifficultyCurriculum(cfg);schedule.stages=dict(motion=2,visual=2,imu=2)
    windows=StateWindows(store,length=31)
    view=AugmentedDataset(windows,cfg,schedule,123)
    first,overlapping=view[0],view[1]
    assert windows.items[:2]==[(0,0),(0,20)]
    torch.testing.assert_close(first['imu'][20:],overlapping['imu'][:11],atol=0,rtol=0)
    assert not torch.equal(first['imu'],windows[0]['imu'])
    assert not torch.equal(first['cam_images'],windows[0]['cam_images'])
    assert first['cam_images'].dtype==torch.uint8
    for key in ('pos','vel','rot','omega','observed','anchor_tick','anchor_img_idx','dt'):
        torch.testing.assert_close(first[key],windows[0][key],atol=0,rtol=0)
    gravity=-first['imu'][1:11,:3].mean(0).clamp(-20,20)
    torch.testing.assert_close(first['gravity'],9.81*gravity/gravity.norm())
    for entry,original in zip(store.entries,originals):
        for key,value in original.items():torch.testing.assert_close(entry[key],value,atol=0,rtol=0)


def test_frame_pairs_and_state_windows_receive_the_same_augmented_imu(tmp_path):
    store=recorded_store(tmp_path)
    cfg=settings(CURRICULUM_MIX=[1.,0.,0.])
    schedule=DifficultyCurriculum(cfg);schedule.stages['imu']=2
    pairs=FramePairs(store)
    pair_view=AugmentedDataset(pairs,cfg,schedule,123)
    state_view=AugmentedDataset(StateWindows(store,length=31),cfg,schedule,123)
    index,previous,anchor,target=pairs.items[0]
    ticks=store.entries[index]['cam_ticks']
    expected=state_view.flight_imu(index)[int(ticks[anchor])+1:int(ticks[target])+1].mean(0)
    torch.testing.assert_close(pair_view[0]['imu'],expected,atol=0,rtol=0)


def test_training_view_with_module_config_can_be_sent_to_spawn_workers(tmp_path):
    base=StateWindows(recorded_store(tmp_path),length=31)
    schedule=DifficultyCurriculum(config);schedule.stages['imu']=2
    view=AugmentedDataset(base,config,schedule,123)
    restored=pickle.loads(pickle.dumps(view))
    for key in ('imu','cam_images','pos','observed'):
        torch.testing.assert_close(restored[0][key],view[0][key],atol=0,rtol=0)


@pytest.mark.parametrize('compact',[None,False,True])
def test_state_windows_in_spawn_worker_preserve_legacy_camera_layout(tmp_path,compact):
    from torch.utils.data import DataLoader
    base=StateWindows(recorded_store(tmp_path),length=31,compact_cameras=bool(compact))
    if compact is None:
        # Reproduce an instance created before the compact-camera field existed.
        del base.compact_cameras
    view=AugmentedDataset(base,settings(),DifficultyCurriculum(config),123)
    batch=next(iter(DataLoader(view,batch_size=1,num_workers=1,multiprocessing_context='spawn',timeout=30)))
    reference=StateWindows(base.store,length=31,compact_cameras=bool(compact))[0]
    assert batch['cam_images'].shape[1]==reference['cam_images'].shape[0]
    for key in ('cam_images','anchor_img_idx','prev_img_idx','observed','pos','imu'):
        torch.testing.assert_close(batch[key][0],reference[key],atol=0,rtol=0)


def test_photometric_changes_are_deterministic_and_shared_across_frames():
    cfg=settings()
    images=torch.full((5,3,12,16),120,dtype=torch.uint8)
    changed=difficult_images(images,1.,cfg,torch.Generator().manual_seed(17))
    assert changed.shape==images.shape and changed.dtype==images.dtype
    assert not torch.equal(changed,images)
    assert torch.equal(changed,changed[:1].expand_as(changed))
    assert torch.equal(changed,difficult_images(images,1.,cfg,torch.Generator().manual_seed(17)))
    assert difficult_images(images,0.,cfg,torch.Generator()) is images


def test_extra_sensor_noise_can_be_disabled_and_has_distinct_random_flight_biases():
    cfg=settings(CURRICULUM_ACCEL_NOISE=0.,CURRICULUM_GYRO_NOISE=0.,
                 CURRICULUM_ACCEL_WALK=0.,CURRICULUM_GYRO_WALK=0.)
    imu=torch.zeros(101,6)
    assert noisy_imu(imu,.005,0.,cfg,torch.Generator()) is imu
    a=noisy_imu(imu,.005,1.,cfg,torch.Generator().manual_seed(17))
    b=noisy_imu(imu,.005,1.,cfg,torch.Generator().manual_seed(18))
    torch.testing.assert_close(a[1:],a[1:2].expand_as(a[1:]),atol=0,rtol=0)
    assert not torch.equal(a[1],b[1]) and torch.equal(a[0],imu[0])
    assert torch.equal(imu,torch.zeros_like(imu))


def test_motion_sampling_retains_easier_and_harder_segments(tmp_path):
    windows=StateWindows(recorded_store(tmp_path),length=31)
    schedule=DifficultyCurriculum(settings());schedule.stages['motion']=1
    sampler=curriculum_sampler(windows,schedule,None,17)
    masses=[float(sampler.weights[[i for i,item in enumerate(windows.items) if item[0]==scene]].sum())
            for scene in range(3)]
    np.testing.assert_allclose(masses,[.2,.7,.1],atol=1e-10)
    assert list(sampler)==list(curriculum_sampler(windows,schedule,None,17))


def test_priority_rewards_learning_progress_instead_of_large_error():
    progress=LearningProgress(limit=2)
    progress.update(['improves','stuck'],torch.tensor([1.,100.]))
    assert progress.priority('improves')==progress.priority('stuck')==1
    progress.update(['improves','stuck'],torch.tensor([.5,100.]))
    assert progress.priority('improves')>progress.priority('stuck')
    progress.update(['new'],torch.tensor([1.]))
    assert len(progress.history)==2


def test_sl_promotions_rebuild_windows_and_keep_validation_probes_fixed(tmp_path):
    recorded_store(tmp_path)
    cfg=settings(STREAM_DATA=False,WORKDIR=tmp_path,H=24,W=32,WINDOW=11,BATCH=1,
                 FRAME_PAIR_BATCH=1,MAX_FRAME_TRAIN_PAIRS=4,MAX_FRAME_VAL_PAIRS=4,
                 SL_CURRICULUM_WINDOWS=[11,21,31],DEVICE='cpu')
    args=SimpleNamespace(out=tmp_path,collector=None,stream_dir=None)
    with ExitStack() as stack:
        data=TrainingData(args,cfg,stack)
        data.begin_phase('state')
        validation=data.state_validation_loader()
        assert validation.dataset.length==31
        _,first=data.next_epoch()
        assert first.dataset.length==11
        probe=next(iter(data._probes('imu')))
        native=next(iter(validation))
        assert not data.finish_curriculum_epoch(10.,lambda loader:10.)
        assert data.curriculum_log()['curriculum/promoted']==0
        repeated_probe=next(iter(data._probes('imu')))
        torch.testing.assert_close(probe['imu'],repeated_probe['imu'],atol=0,rtol=0)
        assert data.finish_curriculum_epoch(9.,lambda loader:9.)
        logged=data.curriculum_log()
        assert logged['curriculum/promoted']==1 and logged['curriculum/promotion_count']==4
        for axis in ('motion','visual','imu','window'):
            assert logged[f'curriculum/{axis}_promoted']==1
            assert logged[f'curriculum/{axis}_promotion_reason']=='improved'
            assert logged[f'curriculum/{axis}_promoted_improved']==1
            assert logged[f'curriculum/{axis}_promoted_plateau']==0
        _,following=data.next_epoch()
        assert following.dataset.length==21 and following is not first
        assert data.state_validation_loader() is validation
        later_probe=next(iter(data._probes('imu')))
        later_native=next(iter(validation))
        assert probe['imu'].shape[1]==11 and later_probe['imu'].shape[1]==21
        for key in ('imu','cam_images','observed','pos'):
            torch.testing.assert_close(native[key],later_native[key],atol=0,rtol=0)


@pytest.mark.parametrize('overrides',[
    dict(CURRICULUM_LEVELS=[0.,1.,.5]),dict(CURRICULUM_MIX=[.7,.2,.2]),
    dict(CURRICULUM_ACCEL_NOISE=-1),dict(CURRICULUM_BLUR=1.1),
    dict(SL_CURRICULUM_WINDOWS=[31,21]),dict(CURRICULUM_PROBE_BATCHES=0),
    dict(CURRICULUM_REQUIRE_COMPLETION=1),dict(CURRICULUM_FINAL_STAGE_CYCLES=-1),
    dict(CURRICULUM_FINAL_STAGE_CYCLES=True),dict(CURRICULUM_FINAL_STAGE_CYCLES=1.5)])
def test_invalid_curriculum_settings_fail_before_collection(overrides):
    with pytest.raises(ValueError):validate_config(settings(**overrides))


@pytest.mark.parametrize('grace',[-1,1.5,True])
def test_invalid_early_stopping_grace_is_rejected(grace):
    with pytest.raises(ValueError,match='CURRICULUM_EARLY_STOP_GRACE'):
        validate_config(settings(CURRICULUM_EARLY_STOP_GRACE=grace))


def test_window_promotion_alone_requests_a_patience_reset(tmp_path):
    recorded_store(tmp_path)
    cfg=settings(STREAM_DATA=False,WORKDIR=tmp_path,H=24,W=32,WINDOW=11,BATCH=1,
        FRAME_PAIR_BATCH=1,MAX_FRAME_TRAIN_PAIRS=4,MAX_FRAME_VAL_PAIRS=4,
        SL_CURRICULUM_WINDOWS=[11,21,31],DEVICE='cpu')
    with ExitStack() as stack:
        data=TrainingData(SimpleNamespace(out=tmp_path,collector=None,stream_dir=None),cfg,stack)
        data.begin_phase('state');data.next_epoch()
        assert not data.finish_curriculum_epoch(10.,lambda loader:10.)
        assert data.finish_curriculum_epoch(99.,lambda loader:9. if loader.dataset.dataset.probe=='window' else 10.)
        row=json.loads((tmp_path/'curriculum_metrics.jsonl').read_text().splitlines()[-1])
        assert row['promoted']==[] and row['window_promoted']
        assert row['early_stopping_reset'] and row['early_stopping_grace_cycles']==3
        assert row['validation_loss']==99. and row['progress_validation_loss']==9.
        logged=data.curriculum_log()
        assert logged['curriculum/window_promoted']==1
        assert logged['curriculum/motion_promoted']==0
        assert logged['curriculum/motion_promotion_reason']=='none'


def test_ppo_horizon_promotion_is_in_shared_curriculum_log(tmp_path):
    recorded_store(tmp_path)
    cfg=settings(STREAM_DATA=False,WORKDIR=tmp_path,H=24,W=32,WINDOW=11,BATCH=1,
        FRAME_PAIR_BATCH=1,MAX_FRAME_TRAIN_PAIRS=4,MAX_FRAME_VAL_PAIRS=4,DEVICE='cpu')
    with ExitStack() as stack:
        data=TrainingData(SimpleNamespace(out=tmp_path,collector=None,stream_dir=None),cfg,stack)
        data.begin_phase('ppo');data.next_epoch();data.ppo_probe_length=11
        assert not data.finish_curriculum_epoch(10.,lambda loader:10.,window_event=dict(
            window_samples=11,next_window_samples=21,promoted=True,reason='plateau',
            progress_validation_loss=10.5))  # PPO never requests SL early-stop patience.
        logged=data.curriculum_log()
        assert logged['curriculum/promoted']==1 and logged['curriculum/promotion_count']==1
        assert logged['curriculum/window_samples']==11 and logged['curriculum/next_window_samples']==21
        assert logged['curriculum/window_promotion_reason']=='plateau'
        assert logged['curriculum/window_promoted_plateau']==1
        assert logged['curriculum/window_promoted_improved']==0
        assert logged['curriculum/progress_loss']==10.5
        row=json.loads((tmp_path/'curriculum_metrics.jsonl').read_text())
        assert row['window_promoted'] and row['promotion_reasons']=={'window':'plateau'}


def test_plateau_can_advance_but_regression_cannot():
    gate=ProgressGate(minimum=1,patience=2,maximum=3,tolerance=.1)
    assert not gate.update(10.)
    assert not gate.update(10.2)
    assert not gate.update(10.1)
    assert gate.update(10.5) and gate.last_reason=='plateau'  # worse than the best is still a plateau
    gate=ProgressGate(minimum=1,patience=2,maximum=3,tolerance=.1)
    for score in (10.,12.,12.,12.):assert not gate.update(score)


@pytest.mark.parametrize('prefix',['SL','PPO'])
def test_window_plateau_allows_small_deterioration_but_rejects_large_regression(prefix):
    from ppo import WindowCurriculum
    cfg=settings(SL_CURRICULUM_WINDOWS=[11,21],PPO_CURRICULUM_WINDOWS=[11,21],
                 CURRICULUM_MAX_STAGE_EPOCHS=3)
    for final,expected in ((11.,True),(11.01,False)):
        schedule=WindowCurriculum(cfg,prefix=prefix)
        for score in (10.,10.2,10.1):assert not schedule.update(score)
        assert schedule.update(final)==expected
        assert schedule.length==(21 if expected else 11)
        if expected:assert schedule.last_reason=='plateau'


def test_window_progress_restarts_its_baseline_at_each_horizon():
    from ppo import WindowCurriculum
    cfg=settings(SL_CURRICULUM_WINDOWS=[11,21,31],CURRICULUM_MAX_STAGE_EPOCHS=2)
    schedule=WindowCurriculum(cfg,prefix='SL')
    assert not schedule.update(1.)
    assert schedule.update(.9) and schedule.length==21
    assert schedule.reference is None
    assert not schedule.update(50.)  # different horizon, not a regression from .9
    assert not schedule.update(50.)
    assert schedule.update(50.) and schedule.last_reason=='plateau' and schedule.length==31


def test_length_sensitive_batch_budget_never_exceeds_configured_batch():
    cfg=settings(BATCH=4,CURRICULUM_BATCH_TOKENS=1604)
    assert [state_batch_size(cfg,n) for n in (101,401,801,1601,2401,4001)]==[4,4,2,1,1,1]


def test_native_bytes_and_deferred_augmentation_match_float_images(tmp_path):
    store=recorded_store(tmp_path)
    cfg=settings(CURRICULUM_MIX=[1.,0.,0.],CURRICULUM_DEVICE_PHOTOMETRIC=True)
    schedule=DifficultyCurriculum(cfg);schedule.stages['visual']=2
    raw=AugmentedDataset(FramePairs(store,raw_images=True),cfg,schedule,17)[0]
    cfg.CURRICULUM_DEVICE_PHOTOMETRIC=False
    floating=AugmentedDataset(FramePairs(store),cfg,schedule,17)[0]
    batch={key:value[None] for key,value in raw.items() if isinstance(value,torch.Tensor)}
    actual=frame_pair_images(batch,'cpu')
    for image,key in zip(actual,('previous','anchor','target')):
        assert raw[key].dtype==torch.uint8
        torch.testing.assert_close(image[0],floating[key],atol=1e-7,rtol=1e-6)


def test_minimum_sl_exposure_does_not_change_checkpoint_improvement_rules(tmp_path):
    from streaming import FreshBatchPatience
    recorded_store(tmp_path)
    cfg=settings(STREAM_DATA=False,WORKDIR=tmp_path,H=24,W=32,WINDOW=11,BATCH=1,
        FRAME_PAIR_BATCH=1,MAX_FRAME_TRAIN_PAIRS=4,MAX_FRAME_VAL_PAIRS=4,
        SL_CURRICULUM_WINDOWS=[11,21],CURRICULUM_MIN_STATE_EPOCHS=3,
        CURRICULUM_REQUIRE_COMPLETION=False,DEVICE='cpu')
    with ExitStack() as stack:
        data=TrainingData(SimpleNamespace(out=tmp_path,collector=None,stream_dir=None),cfg,stack)
        data.begin_phase('state');patience=FreshBatchPatience()
        for _ in range(3):
            data.next_epoch()
            assert data.update_patience(patience,False,False)==0
        data.next_epoch()
        assert data.update_patience(patience,False,False)==1


def test_early_stopping_waits_for_final_window_and_fresh_final_stage_batches(tmp_path):
    from streaming import FreshBatchPatience
    recorded_store(tmp_path)
    cfg=settings(STREAM_DATA=False,WORKDIR=tmp_path,H=24,W=32,WINDOW=11,BATCH=1,
        FRAME_PAIR_BATCH=1,MAX_FRAME_TRAIN_PAIRS=4,MAX_FRAME_VAL_PAIRS=4,
        SL_CURRICULUM_WINDOWS=[11,21],CURRICULUM_FINAL_STAGE_CYCLES=2,
        CURRICULUM_EARLY_STOP_GRACE=0,DEVICE='cpu')
    with ExitStack() as stack:
        data=TrainingData(SimpleNamespace(out=tmp_path,collector=None,stream_dir=None),cfg,stack)
        data.begin_phase('state');patience=FreshBatchPatience()
        # Axes alone cannot enable stopping: the longest horizon must be trained.
        data.curriculum.stages=dict.fromkeys(data.curriculum.axes,2)
        for i,batch in enumerate(('a','a','a','b','b','c','d','d','e')):
            data.next_epoch();data.batch_id=batch
            promoted=data.finish_curriculum_epoch(10.,lambda loader:10. if i==0 else 9.)
            bad=data.update_patience(patience,False,promoted)
            row=data.last_curriculum_record
            if i==0:
                assert not row['curriculum_complete'] and not row['early_stopping_eligible']
            else:assert row['curriculum_complete']
            assert row['final_stage_cycles']==(0,0,0,1,1,2,3,3,4)[i]
            assert bad==(0,0,0,0,0,0,1,1,2)[i]
        assert data.curriculum_log()['curriculum/early_stopping_eligible']==1
        data.begin_phase('teacher')
        assert data.final_stage_cycles==0 and not data.curriculum_complete()


def test_large_regressions_do_not_force_promotion_or_bypass_completion(tmp_path):
    from streaming import FreshBatchPatience
    recorded_store(tmp_path)
    cfg=settings(STREAM_DATA=False,WORKDIR=tmp_path,H=24,W=32,WINDOW=11,BATCH=1,
        FRAME_PAIR_BATCH=1,MAX_FRAME_TRAIN_PAIRS=4,MAX_FRAME_VAL_PAIRS=4,
        SL_CURRICULUM_WINDOWS=[11,21],CURRICULUM_MAX_STAGE_EPOCHS=1,DEVICE='cpu')
    with ExitStack() as stack:
        data=TrainingData(SimpleNamespace(out=tmp_path,collector=None,stream_dir=None),cfg,stack)
        data.begin_phase('state');patience=FreshBatchPatience()
        for score in (10.,12.,12.,12.):
            data.next_epoch()
            promoted=data.finish_curriculum_epoch(score,lambda loader:score)
            assert not promoted and data.update_patience(patience,False,promoted)==0
        assert data.curriculum.snapshot()==dict(motion=0.,visual=0.,imu=0.)
        assert data.window_schedule.length==11


def test_scheduler_uses_stage_progress_and_restarts_when_horizon_changes():
    data=TrainingData.__new__(TrainingData)
    parameter=torch.nn.Parameter(torch.zeros(1))
    optimizer=torch.optim.SGD([parameter],lr=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=0,factor=.5)
    data.last_curriculum_record=dict(window_promoted=False,progress_validation_loss=1.)
    data.step_scheduler(scheduler,100.)
    data.last_curriculum_record['progress_validation_loss']=.9
    data.step_scheduler(scheduler,200.)
    assert optimizer.param_groups[0]['lr']==.01
    data.last_curriculum_record['window_promoted']=True
    data.step_scheduler(scheduler,300.)
    data.last_curriculum_record=dict(window_promoted=False,progress_validation_loss=30.)
    data.step_scheduler(scheduler,400.)
    assert optimizer.param_groups[0]['lr']==.01  # new horizon establishes a new scale


def test_priority_bookkeeping_flushes_once_after_batches(tmp_path):
    recorded_store(tmp_path)
    cfg=settings(STREAM_DATA=False,WORKDIR=tmp_path,H=24,W=32,WINDOW=11,BATCH=1,
        FRAME_PAIR_BATCH=1,MAX_FRAME_TRAIN_PAIRS=4,MAX_FRAME_VAL_PAIRS=4,DEVICE='cpu')
    with ExitStack() as stack:
        data=TrainingData(SimpleNamespace(out=tmp_path,collector=None,stream_dir=None),cfg,stack)
        data.begin_phase('state')
        data.observe_errors({'curriculum_key':['a']},torch.tensor([1.]))
        data.observe_errors({'curriculum_key':['a']},torch.tensor([.5]))
        assert data.progress.history=={}
        data.flush_priorities()
        assert data.progress.priority('a')>1 and data.pending_priority==[]


@pytest.mark.parametrize('visual_mode',['rgb','embedding'])
@pytest.mark.parametrize('require_completion',[False,True])
def test_sl_early_stopping_allows_adaptation_then_stops_at_a_plateau(tmp_path,visual_mode,require_completion):
    collector=tmp_path/'fake_collect.py';collector.write_text(FAKE_COLLECTOR)
    overrides=dict(H=24,W=32,WINDOW=11,BATCH=1,STUDENT_HIDDEN=16,
        VISUAL_PREDICTION=visual_mode,EMBEDDING_HIDDEN=8,USE_GENERATIVE_FRAME_MODEL=False,
        USE_LEARNED_VISUAL_ENCODER=False,USE_HYBRID_SSM_TRANSFORMER=False,WANDB_MODE='disabled',
        NUM_SCENES=2,STREAM_SCENES=1,STREAM_VALIDATION_SCENES=1,
        STREAM_EPISODES_PER_SCENE=1,STREAM_VALIDATION_EPISODES=1,COLLECT_STEPS=41,
        MAX_FRAME_TRAIN_PAIRS=2,MAX_FRAME_VAL_PAIRS=2,FRAME_PAIR_BATCH=2,
        USE_CURRICULUM=True,SL_CURRICULUM_WINDOWS=[11],CURRICULUM_PROBE_BATCHES=1,
        CURRICULUM_MIN_EPOCHS=1,CURRICULUM_PATIENCE=1,CURRICULUM_EARLY_STOP_GRACE=2,
        CURRICULUM_REQUIRE_COMPLETION=require_completion,CURRICULUM_FINAL_STAGE_CYCLES=3,
        CURRICULUM_MAX_STAGE_EPOCHS=1 if require_completion else 6,
        STREAM_MAX_PASSES_PER_BATCH=1,PATIENCE_FRAME=1,PATIENCE_STUDENT=1,
        EPOCHS_FRAME=8,EPOCHS_STUDENT=8,LR_FRAME=0.,LR_STUDENT=0.)
    path=tmp_path/'settings.json';path.write_text(json.dumps(overrides))
    output=tmp_path/'checkpoints'
    # Hold model weights fixed to produce a true main-validation plateau. Feed a
    # Controlled probes improve once, then slightly deteriorate. Ordinary
    # patience=1 would stop at epoch 2 while difficulty is still incomplete.
    script='''
import train, streaming
original=streaming.TrainingData.finish_curriculum_epoch
counts={}
def probe_curve(self, score, evaluate):
    phase=self.curriculum_phase
    counts[phase]=counts.get(phase,0)+1
    value=10. if counts[phase]==1 else (9. if counts[phase]==2 else 9.5)
    return original(self,score,lambda loader:value)
streaming.TrainingData.finish_curriculum_epoch=probe_curve
train.main()
'''
    result=subprocess.run([sys.executable,'-c',script,'--stream-data','--collector',str(collector),
        '--config',str(path),'--device','cpu','--out',str(output),'--max-data-mib','3',
        '--no-ppo','--skip-export'],cwd=HERE,capture_output=True,text=True,timeout=120,
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    assert result.returncode==0,result.stdout+result.stderr
    rows=[json.loads(line) for line in (output/'curriculum_metrics.jsonl').read_text().splitlines()]
    for phase in ('frame','state'):
        epochs=[row for row in rows if row['phase']==phase]
        assert len(epochs)==(7 if require_completion else 5)
        expected_resets=[False,True,True,False,False,False,False] if require_completion else [False,True,False,False,False]
        assert [row['early_stopping_reset'] for row in epochs]==expected_resets
        assert epochs[1]['early_stopping_grace_cycles']==2
        if require_completion:
            assert epochs[2]['promotion_reasons']==dict(motion='plateau',visual='plateau',imu='plateau')
            assert all(value==1. for value in epochs[-1]['levels'].values())
            assert [row['early_stopping_eligible'] for row in epochs]==[False]*6+[True]
            assert epochs[-1]['final_stage_cycles']==4
    assert (output/'post_sl/state_student.pt').is_file()
    assert f'Student early stopped at epoch {7 if require_completion else 5}' in result.stdout
    if visual_mode=='embedding':
        metrics=[json.loads(line) for line in (output/'embedding_metrics.jsonl').read_text().splitlines()]
        assert [row['early_stopping_grace_remaining'] for row in metrics]==([0,2,2,1,0,0,0] if require_completion else [0,2,1,0,0])
        assert all(row['best_validation_loss']==metrics[0]['best_validation_loss'] for row in metrics)


@pytest.mark.parametrize('visual_mode,dimension',[('rgb',48),('embedding',48),('embedding',96)])
def test_full_curriculum_runs_during_sl_and_ppo(tmp_path,visual_mode,dimension):
    collector=tmp_path/'fake_collect.py';collector.write_text(FAKE_COLLECTOR)
    overrides=dict(H=24,W=32,WINDOW=11,BATCH=1,STUDENT_HIDDEN=16,
        VISUAL_PREDICTION=visual_mode,EMBEDDING_HIDDEN=8,USE_GENERATIVE_FRAME_MODEL=False,
        USE_LEARNED_VISUAL_ENCODER=dimension!=48,EMBEDDING_DIM=dimension,
        USE_HYBRID_SSM_TRANSFORMER=False,WANDB_MODE='disabled',
        NUM_SCENES=2,STREAM_SCENES=1,STREAM_VALIDATION_SCENES=1,
        STREAM_EPISODES_PER_SCENE=1,STREAM_VALIDATION_EPISODES=1,COLLECT_STEPS=41,
        MAX_FRAME_TRAIN_PAIRS=4,MAX_FRAME_VAL_PAIRS=4,FRAME_PAIR_BATCH=2,
        USE_CURRICULUM=True,SL_CURRICULUM_WINDOWS=[11,21,31],
        PPO_CURRICULUM_WINDOWS=[11,21,31],CURRICULUM_PROBE_BATCHES=1,
        CURRICULUM_MIN_EPOCHS=1,CURRICULUM_PATIENCE=1,
        USE_TEACHER=visual_mode=='embedding',TEACHER_HIDDEN=16,EPOCHS_TEACHER=1,EPOCHS_DISTILL=1,
        PPO_ITERATIONS=2,PPO_ROLLOUT_BATCHES=1,PPO_UPDATE_EPOCHS=1,PPO_MINIBATCH=16)
    path=tmp_path/'settings.json';path.write_text(json.dumps(overrides))
    output=tmp_path/'checkpoints';managed=tmp_path/'recordings'
    captured=tmp_path/'wandb_rows.jsonl'
    program=f'''
import json, wandb, train
original_init=wandb.init
def capturing_init(*args,**kwargs):
    run=original_init(*args,**kwargs)
    original_log=run.log
    def log(row,*args,**kwargs):
        with open({str(captured)!r},'a') as stream:
            stream.write(json.dumps(row,default=str)+'\\n')
        return original_log(row,*args,**kwargs)
    run.log=log
    return run
wandb.init=capturing_init
train.main()
'''
    command=[sys.executable,'-c',program,'--stream-data','--stream-dir',str(managed),
        '--collector',str(collector),'--config',str(path),'--epochs-frame','2','--epochs-state','2',
        '--device','cpu','--out',str(output),'--max-data-mib','3','--skip-export']
    result=subprocess.run(command,cwd=HERE,capture_output=True,text=True,timeout=120,
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads((output/'student_config.json').read_text())['embedding_dim']==dimension
    rows=[json.loads(line) for line in (output/'curriculum_metrics.jsonl').read_text().splitlines()]
    phases=['frame','frame','state','state']
    if visual_mode=='embedding':phases+=['teacher','distill']
    assert [row['phase'] for row in rows]==phases+['ppo','ppo']
    assert all(set(row['probe_losses'])=={'motion','visual','imu'} for row in rows)
    assert all(np.isfinite(list(row['probe_losses'].values())).all() for row in rows)
    assert rows[2]['window_samples']==11 and rows[2]['priority_examples']>0
    assert rows[-2]['levels']==rows[-3]['next_levels']  # PPO keeps the preceding SL mixture
    logged=[json.loads(line) for line in captured.read_text().splitlines()]
    prefixes=['student','ppo']+(['teacher','distill'] if visual_mode=='embedding' else [])
    for prefix in prefixes:
        points=[row for row in logged if f'{prefix}/val_mean_ate_m' in row]
        assert points and all(np.isfinite(row[f'{prefix}/val_mean_ate_m']) and
                              row[f'{prefix}/val_mean_ate_m']>=0 for row in points)
        assert all(row[f'{prefix}/val_mean_ate_m']==row[f'{prefix}/val_sim3_ate_rmse_mean'] for row in points)
        assert all(row[f'{prefix}/val_ate_episodes']==1 for row in points)
        assert all(not any('unaligned_mean_ate' in key for key in row) for row in points)
    curriculum_points=[row for row in logged if 'curriculum/phase_epoch' in row]
    assert len(curriculum_points)==len(rows)
    for row in curriculum_points:
        assert 'curriculum/promoted' in row
        assert all(f'curriculum/{axis}_promoted' in row for axis in ('motion','visual','imu','window'))
    ppo_rows=[json.loads(line) for line in (output/'ppo_metrics.jsonl').read_text().splitlines()]
    ppo_logged=[row for row in logged if 'ppo/val_mean_ate_m' in row]
    for row,point in zip(ppo_rows,ppo_logged,strict=True):
        assert row['val_mean_ate_m']==point['ppo/val_mean_ate_m']
        assert point['curriculum/window_samples']==row['window_samples']
        assert point['curriculum/next_window_samples']==row['next_window_samples']
        assert point['curriculum/window_promoted']==int(row['curriculum_promoted'])
        assert point['curriculum/window_promotion_reason']==row['curriculum_promotion_reason']
    performance=[json.loads(line) for line in (output/'performance.jsonl').read_text().splitlines()]
    assert [row['stage'] for row in performance]==[
        'embedding' if phase=='frame' and visual_mode=='embedding' else phase for phase in phases+['ppo','ppo']]
    for row in performance:
        pairs=row['stage'] in ('frame','embedding')
        assert row['training_batch_size']==(2 if pairs else 1)
        assert row.get('window_samples')==(None if pairs else 11)
        assert row['samples']>0 and row['samples_per_second']>0
        assert row['loader_wait_seconds']>=0
    assert (output/'post_sl/state_student.pt').is_file()
    assert not list(managed.iterdir())
    before=(output/'state_student.pt').read_bytes()
    later=tmp_path/'ppo-only'
    result=subprocess.run([sys.executable,str(HERE/'train.py'),'--from-checkpoints',str(output),
        '--stream-data','--collector',str(collector),'--config',str(path),'--out',str(later),
        '--max-data-mib','3','--device','cpu','--wandb-mode','disabled','--skip-export'],
        capture_output=True,text=True,timeout=120,
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    assert result.returncode==0,result.stdout+result.stderr
    later_rows=[json.loads(line) for line in (later/'curriculum_metrics.jsonl').read_text().splitlines()]
    assert len(later_rows)==2 and all(row['phase']=='ppo' for row in later_rows)
    assert (output/'state_student.pt').read_bytes()==before
