"""Bounded recording reuse, patience accounting, and fresh PPO policy samples."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import config
import streaming
from export import make_models
from ppo import collect_rollout
from streaming import FreshBatchPatience, RollingBatches, TrainingData
from test_streaming_ppo import small_batch, stream_cfg


@pytest.mark.parametrize('limit',[1,3])
def test_slow_collector_obeys_reuse_limit_then_waits_and_swaps(tmp_path,limit):
    started,release=threading.Event(),threading.Event()
    def producer(directory,spec,stop):
        if directory.name == 'batch-000001':
            started.set()
            assert release.wait(5)
        (directory/'rollout_s000_e000.npz').write_bytes(b'recording')
    with RollingBatches(tmp_path/'managed',stream_cfg(STREAM_MAX_PASSES_PER_BATCH=limit),
                        collect=producer) as batches:
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                first=batches.next_epoch()
                assert started.wait(5)
                for _ in range(limit-1):
                    assert batches.next_epoch() == first
                assert batches.passes_on_current == limit and first.exists()
                entered=threading.Event()
                def next_pass():
                    entered.set()
                    return batches.next_epoch()
                waiting=pool.submit(next_pass)
                assert entered.wait(5)
                assert not waiting.done()  # reuse cap forces waiting for the producer
                assert first.exists()
                release.set()
                following=waiting.result(timeout=5)
                assert following != first and following.exists() and not first.exists()
                assert batches.passes_on_current == 1
            finally:
                release.set()
    assert not list((tmp_path/'managed').iterdir())


def test_collector_error_interrupts_reuse_before_limit(tmp_path):
    release=threading.Event()
    def producer(directory,spec,stop):
        if directory.name == 'batch-000001':
            assert release.wait(5)
            raise RuntimeError('collection failed')
        (directory/'rollout_s000_e000.npz').write_bytes(b'recording')
    with RollingBatches(tmp_path/'managed',stream_cfg(STREAM_MAX_PASSES_PER_BATCH=3),
                        collect=producer) as batches:
        try:
            batches.next_epoch()
            batches.next_epoch()
            assert batches.passes_on_current == 2
            release.set()
            batches.future.exception(timeout=5)
            with pytest.raises(RuntimeError,match='collection failed'):
                batches.next_epoch()
        finally:
            release.set()


@pytest.mark.parametrize('limit',[0,-1,1.5])
def test_invalid_reuse_limits_are_rejected(tmp_path,limit):
    with pytest.raises(ValueError,match='positive integer'):
        RollingBatches(tmp_path,stream_cfg(STREAM_MAX_PASSES_PER_BATCH=limit))


def test_patience_counts_fresh_batches_and_resets_on_reused_improvement():
    patience=FreshBatchPatience()
    assert patience.update(True,'batch0') == 0
    assert patience.update(False,'batch1') == 1
    assert patience.update(False,'batch1') == 1
    assert patience.update(False,'batch1') == 1
    assert patience.update(False,'batch2') == 2
    assert patience.update(True,'batch2') == 0
    assert patience.update(False,'batch2') == 0
    assert patience.update(False,'batch3') == 1


def test_fixed_data_patience_remains_per_epoch():
    patience=FreshBatchPatience()
    assert patience.update(True,1) == 0
    assert [patience.update(False,epoch) for epoch in range(2,8)] == [1,2,3,4,5,6]


def test_promotion_resets_patience_and_grace_counts_only_fresh_batches():
    patience=FreshBatchPatience()
    assert patience.update(False,'a')==1
    assert patience.update(False,'b')==2
    assert patience.update(False,'b',promoted=True,adaptation_cycles=2)==0
    assert patience.grace_remaining==2  # promotion validation used the previous difficulty
    assert patience.update(False,'b')==0 and patience.grace_remaining==2
    assert patience.update(False,'c')==0 and patience.grace_remaining==1
    assert patience.update(False,'c')==0 and patience.grace_remaining==1
    assert patience.update(False,'d')==0 and patience.grace_remaining==0
    assert patience.update(False,'d')==0  # a reused adaptation batch cannot be charged
    assert patience.update(False,'e')==1
    assert patience.update(False,'e')==1
    assert patience.update(False,'f')==2


def test_fixed_data_adaptation_uses_epochs_and_improvements_do_not_extend_grace():
    patience=FreshBatchPatience()
    assert patience.update(True,1)==0
    assert patience.update(False,2,promoted=True,adaptation_cycles=2)==0
    assert patience.update(True,3)==0 and patience.grace_remaining==1
    assert patience.update(False,4)==0 and patience.grace_remaining==0
    assert patience.update(False,5)==1
    assert patience.update(True,6)==0


def test_new_promotion_restarts_grace_without_stacking_old_cycles():
    patience=FreshBatchPatience()
    patience.update(False,1,promoted=True,adaptation_cycles=3)
    patience.update(False,2)
    assert patience.grace_remaining==2
    assert patience.update(False,3,promoted=True,adaptation_cycles=3)==0
    assert patience.grace_remaining==3


def test_zero_grace_still_resets_patience_on_promotion():
    patience=FreshBatchPatience()
    assert patience.update(False,1)==1
    assert patience.update(False,2,promoted=True,adaptation_cycles=0)==0
    assert patience.update(False,3)==1


@pytest.mark.parametrize('cycles',[-1,1.5,True])
def test_invalid_adaptation_cycles_are_rejected(cycles):
    with pytest.raises(ValueError,match='nonnegative integer'):
        FreshBatchPatience().update(False,1,promoted=True,adaptation_cycles=cycles)


@pytest.mark.parametrize('persistent',[False,True])
def test_reuse_retains_loaded_training_and_fixed_validation(tmp_path,monkeypatch,persistent):
    if persistent:
        from multiprocessing.connection import Listener
        try:
            with Listener():pass
        except PermissionError:
            pytest.skip('sandbox disallows local sockets required for tensor sharing')
    release=threading.Event()
    def producer(directory,spec,stop):
        if directory.name == 'batch-000001':
            assert release.wait(20)
        n=41
        state=np.zeros((n,18),np.float32)
        state[:,3:12]=np.eye(3).ravel()
        imu=np.zeros((n,6),np.float32); imu[:,2]=9.81
        np.savez(directory/f'rollout_s{spec["scene_offset"]:03d}_e000.npz',
                 camera=np.zeros((9,24,32,3),np.uint8),camera_ticks=np.arange(0,n,5),
                 state=state,imu=imu,dt=np.float32(.005),frame_stride=5,
                 scene_id=spec['scene_offset'])
    monkeypatch.setattr(streaming,'RollingBatches',
        lambda root,cfg,collector:RollingBatches(root,cfg,collect=producer))
    cfg=SimpleNamespace(**{k:v for k,v in vars(config).items() if k.isupper()})
    cfg.__dict__.update(vars(stream_cfg(STREAM_DATA=True,STREAM_MAX_PASSES_PER_BATCH=3)))
    cfg.H,cfg.W,cfg.WINDOW=24,32,11
    cfg.DEVICE='cpu'
    if persistent:
        from performance import loader_options
        monkeypatch.setattr('performance.loader_options',lambda cfg,frame_pairs=False:
            dict(num_workers=1,persistent_workers=True,multiprocessing_context='spawn') if not frame_pairs
            else loader_options(cfg,frame_pairs=True))
    args=SimpleNamespace(stream_dir=tmp_path/'managed',out=tmp_path,collector=None)
    with ExitStack() as stack:
        try:
            data=TrainingData(args,cfg,stack)
            first=data.next_epoch()
            if persistent:
                list(first[1]);list(data.val_state_loader)
                old_workers=first[1]._iterator._workers
                validation_iterator=data.val_state_loader._iterator
            first_id=data.batch_id
            validation=data.val_state_loader
            old=data.loaded_path
            second=data.next_epoch()
            assert all(a is b for a,b in zip(first,second))
            assert data.batch_id == first_id
            release.set()
            data.rolling.future.result(timeout=5)
            third=data.next_epoch()
            assert all(a is not b for a,b in zip(first,third))
            assert data.batch_id != first_id and not old.exists()
            assert data.val_state_loader is validation
            if persistent:
                assert all(not worker.is_alive() for worker in old_workers)
                assert validation._iterator is validation_iterator
                list(third[1])
                final_workers=[*third[1]._iterator._workers,*validation._iterator._workers]
        finally:
            release.set()
    if persistent:assert all(not worker.is_alive() for worker in final_workers)


@pytest.mark.parametrize('vectorized',[False,True])
def test_ppo_on_reused_recordings_resamples_from_current_policy(vectorized):
    torch.set_num_threads(1)
    frame,student=make_models(dict(backbone='gru',frame_mode='repeat',hidden=16,
                                  learned_visual=False,width=32,height=24))
    student.eval()
    batch=small_batch()
    cfg=SimpleNamespace(DEVICE='cpu',PPO_ROLLOUT_BATCHES=1,PPO_STD=.1,
                        PPO_GAMMA=.995,PPO_GAE_LAMBDA=.95,VECTORIZED_PPO_ROLLOUT=vectorized)
    critic=torch.nn.Sequential(torch.nn.Linear(46,1))
    torch.manual_seed(17)
    first,_=collect_rollout(frame,student,critic,[batch],cfg)
    with torch.no_grad():
        student.out.bias.add_(.2)
    # Identical noise isolates the change of sampling policy on the same inputs.
    torch.manual_seed(17)
    second,_=collect_rollout(frame,student,critic,[batch],cfg)
    torch.testing.assert_close(second['actions'],first['actions']+.2)
    expected=torch.distributions.Normal(student.out(second['tokens']),.1)
    torch.testing.assert_close(second['log_probs'],expected.log_prob(second['actions']).sum(-1))
    assert not torch.equal(first['returns'],second['returns'])
