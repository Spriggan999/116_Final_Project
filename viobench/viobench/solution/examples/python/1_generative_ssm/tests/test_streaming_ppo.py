"""Rolling-data lifecycle, bounded writes, and on-policy refinement regressions."""
from pathlib import Path
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from collect import write_episode
from streaming import RollingBatches, TrainingData
from ppo import WindowCurriculum, advantages, collect_rollout, frozen_tokens, ppo_update
from export import make_models


def stream_cfg(**overrides):
    cfg = dict(STREAM_MAX_MIB=3, STREAM_SCENES=1, NUM_SCENES=2, SCENE_OFFSET=0,
               STREAM_VALIDATION_EPISODES=1, STREAM_EPISODES_PER_SCENE=1,
               SEED=17,COLLECT_STEPS=41,SOURCE_WIDTH=32,SOURCE_HEIGHT=24,
               STREAM_COLLECT_TIMEOUT=30)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_rolling_pipeline_keeps_current_until_next_is_ready(tmp_path):
    started, release = threading.Event(), threading.Event()
    specs = []
    def producer(directory, spec, stop):
        specs.append(spec)
        if directory.name == 'batch-000001':
            started.set()
            assert release.wait(5)
        (directory/'rollout_s000_e000.npz').write_bytes(b'recording')
    keep = tmp_path/'my-original-data'
    keep.mkdir()
    (keep/'keep.npz').write_text('original')
    root = tmp_path/'managed'
    with RollingBatches(root,stream_cfg(),collect=producer) as batches:
        first = batches.next_epoch()
        assert started.wait(5)
        assert first.exists()  # producer is writing; active training is safe
        val_bytes = list(batches.validation.iterdir())[0].read_bytes()
        release.set()
        batches.future.result(timeout=5)
        second = batches.next_epoch()
        assert second != first and second.exists() and not first.exists()
        assert list(batches.validation.iterdir())[0].read_bytes() == val_bytes
        batches.future.result(timeout=5)
        assert len(list(batches.owned.glob('batch-*'))) == 2
    assert not list(root.iterdir())
    assert (keep/'keep.npz').read_text() == 'original'
    assert specs[0]['scene_offset'] == 2
    assert {s['scene_offset'] for s in specs[1:]} <= {0,1}
    assert len({s['seed'] for s in specs}) == len(specs)


def test_collector_failure_is_propagated_and_cleaned(tmp_path):
    def producer(directory, spec, stop):
        if directory.name == 'batch-000001':
            raise RuntimeError('simulator crashed')
        (directory/'rollout_s000_e000.npz').write_bytes(b'recording')
    with pytest.raises(RuntimeError, match='simulator crashed'):
        with RollingBatches(tmp_path/'managed',stream_cfg(),collect=producer) as batches:
            batches.next_epoch()
            batches.future.exception(timeout=5)
            batches.next_epoch()
    assert not list((tmp_path/'managed').iterdir())


def test_cleanup_waits_for_background_writer(tmp_path):
    started = threading.Event()
    ended = threading.Event()
    def producer(directory, spec, stop):
        if directory.name == 'batch-000001':
            started.set()
            stop.wait(5)
            ended.set()
        (directory/'rollout_s000_e000.npz').write_bytes(b'recording')
    with pytest.raises(KeyboardInterrupt):
        with RollingBatches(tmp_path/'managed',stream_cfg(),collect=producer):
            assert started.wait(5)
            raise KeyboardInterrupt
    assert ended.is_set()
    assert not list((tmp_path/'managed').iterdir())


def test_output_budget_checked_before_write(tmp_path):
    target = tmp_path/'rollout.npz'
    data = dict(camera=np.random.default_rng(0).integers(0,256,(8,24,32,3),dtype=np.uint8))
    assert not write_episode(target,data,max_bytes=100)
    assert not list(tmp_path.iterdir())
    assert write_episode(target,data,max_bytes=100_000)
    before = target.read_bytes()
    assert not write_episode(tmp_path/'second.npz',data,max_bytes=len(before)+1)
    assert target.read_bytes() == before


def test_gae_terminal_hand_calculation():
    adv, returns = advantages(torch.tensor([[1.,2.]]),torch.tensor([[.5,.25]]),1.,1.)
    torch.testing.assert_close(returns,torch.tensor([[3.,2.]]))
    torch.testing.assert_close(adv,torch.tensor([[2.5,1.75]]))


def test_gae_time_cutoff_uses_next_state_value():
    adv, returns=advantages(torch.tensor([[1.,2.]]),torch.tensor([[.5,.25]]),
                            1.,1.,bootstrap_value=torch.tensor([4.]))
    torch.testing.assert_close(returns,torch.tensor([[7.,6.]]))
    torch.testing.assert_close(adv,torch.tensor([[6.5,5.75]]))


def test_curriculum_requires_sustained_validation_progress_and_minimum_age():
    curriculum=WindowCurriculum(SimpleNamespace(WINDOW=101,PPO_CURRICULUM_WINDOWS=[401,1001,4001],
        PPO_CURRICULUM_MIN_ITERATIONS=3,PPO_CURRICULUM_PATIENCE=2,
        PPO_CURRICULUM_REL_IMPROVEMENT=.02))
    curriculum.reference=10.
    assert not curriculum.update(9.7) and curriculum.length==401
    assert not curriculum.update(10.1)  # regression resets the streak
    assert not curriculum.update(9.7)
    assert curriculum.update(9.6) and curriculum.length==1001
    assert not curriculum.update(9.5)  # the new stage has a new reference
    assert not curriculum.update(9.3)
    assert curriculum.update(9.2) and curriculum.length==4001
    assert not curriculum.update(8.) and curriculum.length==4001


@pytest.mark.parametrize('windows',[[2],[101,101],[101,51],[4.5]])
def test_invalid_curriculum_windows_are_rejected(windows):
    with pytest.raises(ValueError,match='strictly increasing'):
        WindowCurriculum(SimpleNamespace(WINDOW=101,PPO_CURRICULUM_WINDOWS=windows))


def curriculum_data(lengths):
    data=TrainingData.__new__(TrainingData)
    data.cfg=SimpleNamespace(WINDOW=11,BATCH=1)
    data.loader_options=dict(num_workers=0)
    entries=[dict(state=torch.zeros(n,18),cam_ticks=torch.arange(0,n,5),stride=5,file=f'flight-{i}')
             for i,n in enumerate(lengths)]
    data.train_data=data.val_data=SimpleNamespace(entries=entries)
    data.train_state_loader=object();data.val_state_loader=object()
    return data


def test_ppo_window_loaders_preserve_sl_and_rebuild_only_when_needed():
    data=curriculum_data([41])
    assert data.ppo_loader(11) is data.train_state_loader
    assert data.ppo_loader(11,validation=True) is data.val_state_loader
    short=data.ppo_loader(21)
    assert data.ppo_loader(21) is short
    long=data.ppo_loader(31)
    assert long is not short and long.dataset.length==31
    fixed=data.ppo_loader(31,validation=True)
    data.ppo_loader(21)
    assert data.ppo_loader(31,validation=True) is fixed
    data.close()


def test_ppo_rejects_missing_long_windows_instead_of_shortening_horizon():
    with pytest.raises(ValueError,match='No 81-sample PPO training windows'):
        curriculum_data([41]).ppo_loader(81)
    with pytest.raises(ValueError,match='validation recordings lack 31-sample windows'):
        curriculum_data([21,41]).ppo_loader(31,validation=True)


def small_batch():
    b,t,h,w = 2,11,24,32
    imu = torch.zeros(b,t,6); imu[:,:,2] = 9.81
    return dict(imu=imu,observed=torch.ones(b,t,dtype=torch.bool),dt=torch.full((b,),.005),
                gravity=torch.tensor([[0.,0.,-9.81]]).repeat(b,1),
                pos=torch.zeros(b,t,3),vel=torch.zeros(b,t,3),omega=torch.zeros(b,t,3),
                rot=torch.eye(3)[None,None].repeat(b,t,1,1),
                frames_cached=torch.randint(0,255,(b,t,3,h,w),dtype=torch.uint8))


@pytest.mark.parametrize('backbone',['gru','mamba2_hybrid'])
@pytest.mark.parametrize('vectorized',[False,True])
def test_ppo_tokens_actions_and_frozen_backbone(backbone,vectorized):
    device='cpu'
    if backbone=='mamba2_hybrid':
        from mamba_helpers import require_mamba
        require_mamba();device='cuda'
    flags=dict(backbone=backbone,frame_mode='repeat',learned_visual=False,hidden=64 if backbone=='mamba2_hybrid' else 16,
               width=32,height=24,mamba_layers=1,attention_window=4,
               hybrid_d_state=4,hybrid_head_dim=16,hybrid_attention_heads=4)
    frame, student = make_models(flags)
    frame.to(device);student.to(device)
    for p in student.parameters(): p.requires_grad_(False)
    for p in student.out.parameters(): p.requires_grad_(True)
    student.eval()
    batch = {key:value.to(device) for key,value in small_batch().items()}
    frames = batch['frames_cached'].float()/255
    tokens = frozen_tokens(student,frames,batch,device)
    with torch.no_grad():
        reference = student(frames,batch['imu'],batch['observed'],batch['dt'],batch['gravity'],True)['corrections']
    torch.testing.assert_close(.25*torch.tanh(student.out(tokens)),reference)
    cfg=SimpleNamespace(DEVICE=device,PPO_ROLLOUT_BATCHES=1,PPO_STD=.1,PPO_GAMMA=.995,
                        PPO_GAE_LAMBDA=.95,PPO_UPDATE_EPOCHS=2,PPO_MINIBATCH=8,
                        PPO_TARGET_KL=100.,PPO_CLIP=.2,PPO_ANCHOR_COEF=.01,VECTORIZED_PPO_ROLLOUT=vectorized)
    critic=torch.nn.Sequential(torch.nn.Linear(student.hidden+30,32),torch.nn.Tanh(),torch.nn.Linear(32,1)).to(device)
    before = {k:v.clone() for k,v in student.state_dict().items()}
    anchor = __import__('copy').deepcopy(student.out)
    optimizer=torch.optim.Adam(list(student.out.parameters())+list(critic.parameters()),lr=1e-3)
    torch.manual_seed(17)
    rollout,_=collect_rollout(frame,student,critic,[batch],cfg)
    assert not rollout['actions'].requires_grad
    assert rollout['actions'].shape == (18,9)  # initial + final bootstrap tokens are reserved
    result=ppo_update(student,critic,anchor,optimizer,rollout,cfg)
    assert result['updates'] > 0
    assert any(not torch.equal(v,before[k]) for k,v in student.state_dict().items() if k.startswith('out.'))
    assert all(torch.equal(v,before[k]) for k,v in student.state_dict().items() if not k.startswith('out.'))


FAKE_COLLECTOR = '''
import argparse, pathlib, numpy as np
p=argparse.ArgumentParser()
for name in ('out','scenes','episodes','scene-offset','seed','steps','width','height','max-output-mib'):
 p.add_argument('--'+name, required=True)
a=p.parse_args(); root=pathlib.Path(a.out)
rng=np.random.default_rng(int(a.seed)); n=int(a.steps); stride=5
truth=np.zeros((n,18),np.float32); truth[:,3:12]=np.eye(3).ravel()
imu=np.zeros((n,6),np.float32); imu[1:,2]=9.81
for j in range(int(a.scenes)):
 scene=int(a.scene_offset)+j
 np.savez(root/f'rollout_s{scene:03d}_e000.npz',
  camera=rng.integers(0,256,((n-1)//stride+1,24,32,3),dtype=np.uint8),
  camera_ticks=np.arange(0,n,stride,dtype=np.int32),imu=imu,state=truth,
  dt=np.float32(.005),frame_stride=stride,scene_id=scene)
'''


def test_streaming_sl_ppo_and_later_ppo_only(tmp_path):
    collector=tmp_path/'fake_collect.py'; collector.write_text(FAKE_COLLECTOR)
    config=tmp_path/'settings.json'
    config.write_text(json.dumps(dict(H=24,W=32,WINDOW=21,BATCH=1,STUDENT_HIDDEN=16,
        USE_GENERATIVE_FRAME_MODEL=False,USE_LEARNED_VISUAL_ENCODER=False,USE_HYBRID_SSM_TRANSFORMER=False,WANDB_MODE='disabled',
        NUM_SCENES=2,STREAM_SCENES=1,STREAM_EPISODES_PER_SCENE=1,STREAM_VALIDATION_EPISODES=1,
        COLLECT_STEPS=41,MAX_FRAME_TRAIN_PAIRS=2,MAX_FRAME_VAL_PAIRS=2,
        STREAM_MAX_PASSES_PER_BATCH=1,
        PPO_ITERATIONS=2,PPO_ROLLOUT_BATCHES=1,PPO_UPDATE_EPOCHS=2,PPO_MINIBATCH=16,
        PPO_CURRICULUM_WINDOWS=[11,21,31])))
    root=tmp_path/'batches'; checkpoints=tmp_path/'checkpoints'
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,str(HERE/'train.py'),'--stream-data','--stream-dir',str(root),
             '--collector',str(collector),'--config',str(config),'--epochs-frame','2',
             '--epochs-state','2','--device','cpu','--out',str(checkpoints),
             '--solution-out',str(tmp_path/'solution'),'--max-data-mib','3',
             '--max-passes-per-batch','3']
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    assert not list(root.iterdir())
    assert 'Training stages: SL + PPO' in result.stdout
    assert 'cache' in result.stdout.lower()
    assert not list(tmp_path.rglob('frame_cache*.npy'))
    assert json.loads((checkpoints/'training_config.json').read_text())['STREAM_MAX_PASSES_PER_BATCH'] == 3
    rows=[json.loads(line) for line in (checkpoints/'ppo_metrics.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and rows[-1]['best_validation_loss'] <= rows[0]['sl_validation_loss']
    assert rows[0]['window_samples']==11
    assert all(row['validation_window_samples']==31 for row in rows)
    assert all(1 <= row['recording_pass'] <= 3 and row['recording_batch'] for row in rows)
    for name in ['state_student_sl.pt','state_student_ppo.pt','state_student.pt','ppo_training.pt']:
        assert (checkpoints/name).is_file()
    snapshot=checkpoints/'post_sl'
    for name in ('frame_student.pt','state_student.pt','student_config.json','training_config.json','stage.json'):
        assert (snapshot/name).is_file()
    sl_weights=torch.load(snapshot/'state_student.pt',weights_only=True)
    assert all(torch.equal(value,sl_weights[key]) for key,value in
               torch.load(checkpoints/'state_student_sl.pt',weights_only=True).items())
    assert (snapshot/'frame_student.pt').read_bytes() == (checkpoints/'frame_student.pt').read_bytes()
    exported_sl=tmp_path/'sl-solution'
    result=subprocess.run([sys.executable,str(HERE/'export.py'),'--checkpoints',str(snapshot),
        '--out',str(exported_sl)],env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode == 0,result.stdout+result.stderr
    np.testing.assert_array_equal(np.load(exported_sl/'state_student.npz')['out.weight'],
                                  sl_weights['out.weight'].numpy())
    selected=torch.load(checkpoints/'state_student.pt',weights_only=True)
    exported=np.load(tmp_path/'solution/state_student.npz')
    np.testing.assert_array_equal(exported['out.weight'],selected['out.weight'].numpy())
    # A later RL-only invocation must not train SL or overwrite source checkpoints.
    original=(checkpoints/'state_student.pt').read_bytes()
    later=tmp_path/'later'
    result=subprocess.run([sys.executable,str(HERE/'train.py'),'--from-checkpoints',str(checkpoints),
        '--stream-data','--collector',str(collector),'--out',str(later),
        '--solution-out',str(tmp_path/'later-solution'),'--wandb-mode','disabled'],env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'Training stages: PPO only' in result.stdout
    assert 'Frame 1/' not in result.stdout and 'Student 1/' not in result.stdout
    assert (checkpoints/'state_student.pt').read_bytes() == original
    assert not list((later/'stream-data').iterdir())


def test_background_subprocess_is_stopped_on_cancel(tmp_path):
    script=tmp_path/'sleeping.py'
    script.write_text('import time\nprint("collector started",flush=True)\ntime.sleep(120)\n')
    batches=RollingBatches(tmp_path/'managed',stream_cfg(),collector=script)
    stopped=threading.Event()
    # Exercise the actual child-process polling/termination path with cancellation.
    timer=threading.Timer(.2,stopped.set)
    timer.start()
    try:
        with pytest.raises(RuntimeError,match='cancelled'):
            batches._subprocess(tmp_path,dict(scene_offset=0,scenes=1,episodes=1,seed=1,
                steps=41,width=32,height=24,max_bytes=1024),stopped)
    finally:
        timer.join()
        batches.close()


def test_oversized_batch_is_rejected_and_cleaned(tmp_path):
    def producer(directory,spec,stop):
        (directory/'rollout_s000_e000.npz').write_bytes(b'x'*(1024*1024))
    with pytest.raises(RuntimeError,match='exceeded'):
        with RollingBatches(tmp_path/'managed',stream_cfg(STREAM_MAX_MIB=1),collect=producer):
            pass
    assert not list((tmp_path/'managed').iterdir())
