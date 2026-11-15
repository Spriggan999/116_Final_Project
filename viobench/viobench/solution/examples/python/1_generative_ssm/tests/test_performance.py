"""Batched flights, causal missing-only generation, and precision boundaries."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import config as cfg
from collect import collect_round,instance_frames,instance_rows
from data import generate_200hz_frames
from performance import EpochTimer,amp_disabled,configure,fp32_physics,loader_options,neural_autocast,record_epoch,pipeline_log,timed_batches


class Flights:
    total_instances=2
    frame_stride=4
    dt=.005
    episode_step_limit=100

    def reset(self):self.tick=0
    def step(self):self.tick+=1
    def render(self):pass
    def terminated(self):return np.array([self.tick>=30,False])
    def observe_privileged(self):
        state=np.zeros((2,18),np.float32)
        state[:,3:12]=np.eye(3).ravel()
        state[:,0]=[self.tick,100+self.tick]
        if self.tick>=30:state[0]=np.nan
        return state
    def observe_imu(self):
        imu=np.zeros((2,6),np.float32)
        imu[:,0]=[self.tick,100+self.tick]
        if self.tick>=30:imu[0]=np.nan
        return imu
    def frames(self):
        x=np.empty((2,1,8,12,3),np.float32)
        x[0]=self.tick/255.;x[1]=(100+self.tick)/255.
        if self.tick>=30:x[0]=np.nan
        return x


def test_flights_terminate_independently_without_mixing_observations():
    rows=collect_round(Flights(),41)
    assert [len(row['state']) for row in rows] == [30,41]
    for slot,row in enumerate(rows):
        ticks=row['camera_ticks']
        np.testing.assert_array_equal(row['state'][:,0],np.arange(len(row['state']))+100*slot)
        np.testing.assert_array_equal(row['imu'][1:,0],np.arange(1,len(row['state']))+100*slot)
        np.testing.assert_array_equal(row['camera'][:,0,0,0],ticks+100*slot)
        assert np.isfinite(row['state']).all() and np.isfinite(row['imu']).all()
        assert ticks[-1]<len(row['state'])


def test_flights_can_exceed_informational_limit_and_still_stop_crashes():
    env=Flights()
    env.episode_step_limit=40
    rows=collect_round(env,81)
    assert [len(row['state']) for row in rows] == [30,81]


def test_collector_records_per_flight_camera_positions():
    class MountedFlights(Flights):
        def instance_parameters(self):
            return [dict(cameras=dict(front=dict(mount=dict(pose=dict(position=position)))))
                    for position in ([.1,.2,.3],[.4,.5,.6])]
    rows=collect_round(MountedFlights(),41)
    np.testing.assert_allclose(rows[0]['camera_position_in_body'],[.1,.2,.3])
    np.testing.assert_allclose(rows[1]['camera_position_in_body'],[.4,.5,.6])


@pytest.mark.parametrize('channels_first',[False,True])
def test_frame_layouts_preserve_instance_and_first_camera(channels_first):
    x=np.zeros((2,2,8,12,4),np.uint8)
    x[0,0,...,:3]=23;x[1,0,...,:3]=91;x[:,1]=180
    if channels_first:x=x.transpose(0,1,4,2,3)
    frames=instance_frames(x,2)
    assert [int(f[0,0,0]) for f in frames]==[23,91]
    assert all(f.shape==(8,12,3) for f in frames)
    with pytest.raises(ValueError):instance_rows(np.ones((3,6)),2,6)


def test_generation_never_calls_network_for_observed_frames(monkeypatch):
    class Generator(torch.nn.Module):
        def __init__(self):super().__init__();self.samples=0
        def forward(self,previous,anchor,imu,elapsed,prev_elapsed):
            self.samples+=len(anchor)
            assert len(anchor)<=2
            return anchor+elapsed[:,:,None,None]
    monkeypatch.setattr(cfg,'FRAME_GENERATION_CHUNK',2)
    generator=Generator();b,t=2,5
    cam=torch.rand(b,2,3,8,12)
    observed=torch.tensor([[True,False,False,True,False],[True,False,True,False,False]])
    idx=torch.tensor([[0,0,0,1,1],[0,0,1,1,1]])
    ticks=torch.tensor([[0,0,0,3,3],[0,0,2,2,2]])
    batch=dict(cam_images=cam,imu=torch.zeros(b,t,6),anchor_img_idx=idx,anchor_tick=ticks,
               prev_img_idx=torch.zeros_like(idx),prev_span=torch.zeros(b,t),observed=observed,dt=torch.full((b,),.005))
    actual=generate_200hz_frames(generator,batch,'cpu')
    assert generator.samples==int((~observed).sum())
    for i in range(b):
        for j in range(t):
            expected=cam[i,idx[i,j]] if observed[i,j] else cam[i,idx[i,j]]+(j-ticks[i,j])*.005
            torch.testing.assert_close(actual[i,j],expected)
    batch['observed']=torch.ones_like(observed)
    generator.samples=0
    actual=generate_200hz_frames(generator,batch,'cpu')
    assert generator.samples==0
    torch.testing.assert_close(actual,cam[torch.arange(b)[:,None],idx])


def test_fp32_physics_and_neural_cpu_fallback():
    @fp32_physics
    def physics(x):return x@x.T
    x=torch.randn(3,3)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        assert physics(x).dtype==torch.float32
    with amp_disabled(),neural_autocast(x):
        assert (x@x.T).dtype==torch.float32


def test_loader_prefetch_is_bounded_and_cpu_stays_simple():
    config=SimpleNamespace(DEVICE='cpu',LOADER_WORKERS=2,PIN_MEMORY=True,LOADER_PREFETCH=1)
    assert loader_options(config)==dict(num_workers=0,pin_memory=False)
    config.DEVICE='cuda'
    assert loader_options(config,frame_pairs=True)==dict(num_workers=0,pin_memory=True)
    assert loader_options(config)==dict(num_workers=2,pin_memory=True,prefetch_factor=1,persistent_workers=True)
    config.PERSISTENT_WORKERS=False
    assert not loader_options(config)['persistent_workers']


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fused_adamw_matches_ordinary_updates_and_supports_fallback(monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from performance import adamw
    monkeypatch.setattr(cfg,'FUSED_ADAMW',True)
    torch.manual_seed(17)
    parameters=[torch.nn.Parameter(torch.randn(13,7,device=device)),torch.nn.Parameter(torch.randn(7,device=device))]
    reference=[torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    optimizer=adamw(parameters,lr=3e-4,weight_decay=.01)
    ordinary=adamw(reference,lr=3e-4,weight_decay=.01,fused=False)
    assert optimizer.defaults['fused']==(device=='cuda')
    for _ in range(4):
        for actual,expected in zip(parameters,reference):
            gradient=torch.randn_like(actual)
            actual.grad=gradient;expected.grad=gradient.clone()
        optimizer.step();ordinary.step()
    for actual,expected in zip(parameters,reference):
        torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
    monkeypatch.setattr(cfg,'FUSED_ADAMW',False)
    assert not adamw(parameters).defaults['fused']


def test_throughput_report_includes_data_wait(tmp_path):
    timer=EpochTimer('cpu');timer.add(3)
    result=timer.result()
    data=SimpleNamespace(batch_id='batch-000002',rolling=SimpleNamespace(last_wait_seconds=.25,passes_on_current=2))
    record_epoch(tmp_path,'state',1,result,data)
    row=json.loads((tmp_path/'performance.jsonl').read_text())
    assert row['samples']==3 and row['samples_per_second']>0
    assert row['fresh_data_wait_seconds']==.25 and row['recording_pass']==2


@pytest.mark.parametrize('length',[None,401])
def test_throughput_logs_effective_loader_batch_despite_partial_tail(tmp_path,length):
    from torch.utils.data import DataLoader,TensorDataset
    dataset=TensorDataset(torch.zeros(9,1))
    if length is not None:dataset.length=length
    loader=DataLoader(dataset,batch_size=8)
    data=SimpleNamespace(batch_id=1,rolling=None)
    timer=EpochTimer('cpu')
    counts=[]
    for (batch,) in timed_batches(loader,timer):
        counts.append(len(batch));timer.add(len(batch))
    row=record_epoch(tmp_path,'state' if length else 'embedding',1,timer.result(),data,loader=loader)
    assert counts==[8,1] and row['samples']==9
    assert row['training_batch_size']==8
    assert row.get('window_samples')==length
    metrics=pipeline_log(row)
    assert metrics['pipeline/training_batch_size']==8
    assert metrics['pipeline/samples_per_second']>0
    assert metrics['pipeline/loader_wait_seconds']>=0
    assert metrics.get('pipeline/window_samples')==length


@pytest.mark.parametrize('key,value',[('BATCH',0),('FRAME_GENERATION_CHUNK',-1),('COLLECT_INSTANCES',0),('LOADER_WORKERS',-1),('AMP_DTYPE','float16')])
def test_invalid_performance_settings_fail_before_training(monkeypatch,key,value):
    monkeypatch.setattr(cfg,'DEVICE','cpu');monkeypatch.setattr(cfg,key,value)
    with pytest.raises(ValueError):configure(cfg)


@pytest.mark.parametrize('flags,expected_batch,expected_amp',[
    ([],4,True),(['--batch','2','--no-amp','--collect-instances','1'],2,False)])
def test_warm_start_uses_new_performance_defaults_unless_overridden(tmp_path,flags,expected_batch,expected_amp):
    import os
    import subprocess
    source=tmp_path/'old';source.mkdir()
    (source/'training_config.json').write_text(json.dumps(dict(BATCH=1,FRAME_PAIR_BATCH=1,
        FRAME_GENERATION_CHUNK=1,VISUAL_CHUNK=1,COLLECT_INSTANCES=1,USE_AMP=False,LOADER_WORKERS=0,
        NUM_SCENES=6,COLLECT_STEPS=401,STREAM_SCENES=2,STREAM_VALIDATION_SCENES=1,STREAM_MAX_MIB=512,
        COLLECT_COMPRESSION_LEVEL=9)))
    for name in ('frame_student.pt','state_student.pt'):(source/name).touch()
    output=tmp_path/'new'
    result=subprocess.run([sys.executable,'-c','import train; train.run_training=lambda *a:None; train.main()',
        '--from-checkpoints',str(source),'--out',str(output),'--device','cpu','--skip-export',*flags],
        cwd=HERE,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    settings=json.loads((output/'training_config.json').read_text())
    assert settings['BATCH']==expected_batch and settings['USE_AMP']==expected_amp
    assert settings['FRAME_PAIR_BATCH']==8 and settings['FRAME_GENERATION_CHUNK']==16
    assert settings['EPOCHS_FRAME']==200 and settings['EPOCHS_STUDENT']==300
    assert (settings['NUM_SCENES'],settings['COLLECT_STEPS'],settings['STREAM_SCENES'],
            settings['STREAM_VALIDATION_SCENES'],settings['STREAM_MAX_MIB'])==(313,2000,4,12,3072)
    assert settings['SCENE_CATALOG']=='mixed'
    assert settings['COLLECT_COMPRESSION_LEVEL']==1


@pytest.mark.parametrize('warm_start',[False,True])
@pytest.mark.parametrize('cli_override',[False,True])
def test_curriculum_preset_applies_to_fresh_and_resumed_training(tmp_path,warm_start,cli_override):
    import os
    import subprocess
    from curriculum import state_batch_size
    flags=['--config',str(HERE/'curriculum.json')]
    if warm_start:
        source=tmp_path/'old';source.mkdir()
        (source/'training_config.json').write_text(json.dumps(dict(BATCH=1,FRAME_PAIR_BATCH=1,
            CURRICULUM_BATCH_TOKENS=401,EPOCHS_FRAME=40,EPOCHS_STUDENT=50,EPOCHS_DISTILL=25)))
        for name in ('frame_student.pt','state_student.pt'):(source/name).touch()
        flags+=['--from-checkpoints',str(source)]
    if cli_override:
        flags+=['--batch','2','--frame-pair-batch','16','--epochs-frame','3','--epochs-state','5']
    output=tmp_path/'new'
    # Exercise production config precedence without running CUDA-only kernels.
    # Numerical training is covered separately with small CPU-capable models.
    script='import train, models; models.require_official_mamba=lambda *a:None; train.run_training=lambda *a:None; train.main()'
    result=subprocess.run([sys.executable,'-c',script,
        '--out',str(output),'--device','cpu','--skip-export',*flags],cwd=HERE,
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    settings=json.loads((output/'training_config.json').read_text())
    config=SimpleNamespace(**settings)
    assert (config.FRAME_PAIR_BATCH,config.BATCH)==((16,2) if cli_override else (64,8))
    assert (config.EPOCHS_FRAME,config.EPOCHS_STUDENT)==((3,5) if cli_override else (400,600))
    assert (config.EPOCHS_TEACHER,config.EPOCHS_DISTILL)==(60,100)
    assert config.FRAME_VAL_PAIR_BATCH==8
    assert [state_batch_size(config,n) for n in config.SL_CURRICULUM_WINDOWS]==([2,2,2,1,1] if cli_override else [8,4,2,1,1])
    assert [state_batch_size(config,n) for n in config.PPO_CURRICULUM_WINDOWS]==([2,2,1,1] if cli_override else [8,3,1,1])
    assert config.PPO_ITERATIONS==60 and config.PPO_MINIBATCH==cfg.PPO_MINIBATCH
    assert config.COLLECT_INSTANCES==2
