"""Default stage selection and SL snapshot survival across PPO/interruption."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
ENV=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')


@pytest.mark.parametrize('existing,flags,expected',[
    (False,[],'SL + PPO'),(True,[],'PPO only'),
    (True,['--sl'],'SL + PPO'),(True,['--sl','--no-ppo'],'SL only'),
    (False,['--no-ppo'],'SL only'),(True,['--no-ppo'],'SL only'),
    (True,['--ppo-only'],'PPO only'),
    (False,['--no-ppo','--wandb-mode','disabled'],'SL only'),
    (True,['--wandb-mode','offline'],'PPO only')])
def test_default_and_explicit_stages(tmp_path,existing,flags,expected):
    source=tmp_path/'source'; source.mkdir()
    # Old USE_PPO=False must not disable the newly requested default behavior.
    (source/'training_config.json').write_text(json.dumps(dict(USE_PPO=False,STUDENT_HIDDEN=16,
        MAMBA_TRAIN_BACKEND='pure_torch',USE_PURE_TORCH_MAMBA_INFERENCE=True,EXPORT_PURE_MAMBA_ONNX=False,
        COLLECT=False,EPISODES_PER_SCENE=8,WANDB_MODE='disabled')))
    for name in ('frame_student.pt','state_student.pt'):
        (source/name).touch()  # stage resolution is tested before loading tensors
    command=[sys.executable,'-c','import train; train.run_training=lambda *a:None; train.main()',
             '--out',str(tmp_path/'new'),'--device','cpu']
    if existing: command+=['--from-checkpoints',str(source)]
    else: command+=['--config',str(HERE/'gru.json')]
    result=subprocess.run(command+flags,cwd=HERE,env=ENV,capture_output=True,text=True,timeout=30)
    assert result.returncode == 0,result.stdout+result.stderr
    assert f'Training stages: {expected}' in result.stdout
    resolved=json.loads((tmp_path/'new/training_config.json').read_text())
    assert resolved['USE_HYBRID_SSM_TRANSFORMER'] is False
    assert resolved['WANDB_MODE']==(flags[flags.index('--wandb-mode')+1] if '--wandb-mode' in flags else 'online')
    assert not any(key in resolved for key in ('MAMBA_TRAIN_BACKEND','USE_PURE_TORCH_MAMBA_INFERENCE','EXPORT_PURE_MAMBA_ONNX',
                                             'COLLECT','EPISODES_PER_SCENE'))


@pytest.mark.parametrize('setting',['MAMBA_TRAIN_BACKEND','USE_PURE_TORCH_MAMBA_INFERENCE','EXPORT_PURE_MAMBA_ONNX'])
def test_removed_mamba_settings_rejected_before_collection(tmp_path,setting):
    config=tmp_path/'config.json';config.write_text(json.dumps({setting:True}))
    result=subprocess.run([sys.executable,str(HERE/'train.py'),'--config',str(config),
        '--out',str(tmp_path/'run'),'--stream-data'],env=ENV,capture_output=True,text=True,timeout=30)
    assert result.returncode != 0 and 'was removed: official Mamba-2' in result.stderr
    assert not (tmp_path/'run').exists()


def test_hybrid_cpu_rejected_before_collection(tmp_path):
    result=subprocess.run([sys.executable,str(HERE/'train.py'),
        '--device','cpu','--out',str(tmp_path/'run'),'--stream-data'],env=ENV,capture_output=True,text=True,timeout=30)
    assert result.returncode != 0 and 'official Mamba-2 CUDA kernels' in result.stderr
    assert not (tmp_path/'run').exists()


def test_incomplete_checkpoint_requires_sl_before_ppo(tmp_path):
    source=tmp_path/'source'; source.mkdir()
    (source/'training_config.json').write_text('{}')
    (source/'frame_student.pt').touch()
    result=subprocess.run([sys.executable,str(HERE/'train.py'),'--from-checkpoints',str(source),
        '--stream-data','--out',str(tmp_path/'new')],env=ENV,capture_output=True,text=True,timeout=30)
    assert result.returncode != 0
    assert 'missing state_student.pt' in result.stderr and 'Use --sl' in result.stderr
    assert not (tmp_path/'new').exists()  # no collection or training started


@pytest.mark.parametrize('existing,flags',[
    (False,['--no-sl']),(False,['--ppo-only']),
    (True,['--ppo-only','--no-ppo']),(True,['--ppo-only','--sl'])])
def test_incompatible_stage_requests_rejected_before_training(tmp_path,existing,flags):
    source=tmp_path/'source'; source.mkdir()
    (source/'training_config.json').write_text('{}')
    command=[sys.executable,str(HERE/'train.py'),'--out',str(tmp_path/'new')]+flags
    if existing: command+=['--from-checkpoints',str(source)]
    result=subprocess.run(command,env=ENV,capture_output=True,text=True,timeout=30)
    assert result.returncode != 0 and 'error:' in result.stderr
    assert not (tmp_path/'new').exists()


def recordings(tmp_path):
    data=tmp_path/'data'; data.mkdir()
    for scene in (0,1):
        n=21
        state=np.zeros((n,18),np.float32); state[:,3:12]=np.eye(3).ravel()
        imu=np.zeros((n,6),np.float32); imu[:,2]=9.81
        np.savez(data/f'rollout_s{scene:03d}_e000.npz',camera=np.full((5,24,32,3),127,np.uint8),
            camera_ticks=np.arange(0,n,5),state=state,imu=imu,dt=np.float32(.005),
            frame_stride=5,scene_id=scene)
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(H=24,W=32,WINDOW=11,BATCH=1,STUDENT_HIDDEN=16,
        USE_GENERATIVE_FRAME_MODEL=False,USE_LEARNED_VISUAL_ENCODER=False,USE_HYBRID_SSM_TRANSFORMER=False,WANDB_MODE='disabled',
        MAX_FRAME_TRAIN_PAIRS=1,MAX_FRAME_VAL_PAIRS=1,PPO_ITERATIONS=1,
        PPO_ROLLOUT_BATCHES=1,PPO_UPDATE_EPOCHS=1)))
    return data,config


def test_post_sl_snapshot_survives_ppo_interruption_and_exports(tmp_path):
    data,config=recordings(tmp_path)
    output=tmp_path/'run'
    program='''
import torch, ppo, train
def interrupted(frame, student, data, cfg, output, save):
    assert (output/'post_sl'/'stage.json').is_file()
    with torch.no_grad():
        student.out.weight.fill_(99.)
    save(student.state_dict(),cfg.STATE_CHECKPOINT)
    raise RuntimeError('simulated PPO interruption')
ppo.refine_ppo=interrupted
train.main()
'''
    result=subprocess.run([sys.executable,'-c',program,str(data),'--config',str(config),
        '--out',str(output),'--epochs-frame','1','--epochs-state','1','--device','cpu','--skip-export'],
        cwd=HERE,env=ENV,capture_output=True,text=True,timeout=90)
    assert result.returncode != 0 and 'simulated PPO interruption' in result.stderr
    snapshot=output/'post_sl'
    sl=torch.load(snapshot/'state_student.pt',weights_only=True)
    root=torch.load(output/'state_student.pt',weights_only=True)
    assert not torch.equal(sl['out.weight'],root['out.weight'])
    result=subprocess.run([sys.executable,str(HERE/'export.py'),'--checkpoints',str(snapshot),
        '--out',str(tmp_path/'sl-solution')],env=ENV,capture_output=True,text=True,timeout=90)
    assert result.returncode == 0,result.stdout+result.stderr
    np.testing.assert_array_equal(np.load(tmp_path/'sl-solution/state_student.npz')['out.weight'],
                                  sl['out.weight'].numpy())


def test_explicit_sl_recovers_frame_only_checkpoint(tmp_path):
    data,config=recordings(tmp_path)
    output=tmp_path/'first'
    base=[sys.executable,str(HERE/'train.py'),str(data),'--config',str(config),
          '--epochs-frame','1','--epochs-state','1','--device','cpu','--skip-export','--no-ppo']
    result=subprocess.run(base+['--out',str(output)],env=ENV,capture_output=True,text=True,timeout=90)
    assert result.returncode == 0,result.stdout+result.stderr
    (output/'state_student.pt').unlink()
    recovered=tmp_path/'recovered'
    result=subprocess.run(base+['--from-checkpoints',str(output),'--sl','--out',str(recovered)],
        env=ENV,capture_output=True,text=True,timeout=90)
    assert result.returncode == 0,result.stdout+result.stderr
    assert 'State checkpoint missing; initializing' in result.stdout
    assert (recovered/'post_sl/state_student.pt').is_file()


@pytest.mark.parametrize('dimension,ppo_enabled',[(48,True),(96,True),(48,False)])
def test_skip_embedding_trains_state_and_optional_ppo_with_unchanged_embedding(tmp_path,dimension,ppo_enabled):
    data,config=recordings(tmp_path)
    settings=json.loads(config.read_text())
    settings.update(VISUAL_PREDICTION='embedding',EMBEDDING_DIM=dimension,EMBEDDING_HIDDEN=8,
        USE_LEARNED_VISUAL_ENCODER=True,VISUAL_INPUT_STRIDE=2,USE_TEACHER=False,
        USE_CURRICULUM=True,SL_CURRICULUM_WINDOWS=[11,21],PPO_CURRICULUM_WINDOWS=[11,21],
        CURRICULUM_PROBE_BATCHES=1,FRAME_LOADER_WORKERS=0,LOADER_WORKERS=0)
    config.write_text(json.dumps(settings))
    source=tmp_path/'embedding-complete';source.mkdir()
    (source/'training_config.json').write_text(json.dumps(settings))
    # Construct a frame-only source, just like a run stopped before state SL.
    prepare='''
import sys,torch
from models import EmbeddingPredictor
torch.manual_seed(17)
model=EmbeddingPredictor(hidden=8,visual_input_stride=2,embedding_dim=int(sys.argv[2]))
torch.save(model.state_dict(),sys.argv[1])
'''
    result=subprocess.run([sys.executable,'-c',prepare,str(source/'frame_student.pt'),str(dimension)],
        cwd=HERE,env=ENV,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    before=(source/'frame_student.pt').read_bytes()
    output=tmp_path/'state-and-ppo'
    program='''
import embedding_training,data,train
def forbidden(*args,**kwargs):
    raise AssertionError('Embedding training must not be called')
embedding_training.train_embeddings=forbidden
original_input=data.state_input_frames
def checked_input(model,batch,device):
    assert not model.training
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    return original_input(model,batch,device)
data.state_input_frames=checked_input
train.main()
'''
    command=[sys.executable,'-c',program,str(data),'--from-checkpoints',str(source),
        '--skip-embedding-training','--config',str(config),'--wandb-mode','disabled',
        '--out',str(output),'--epochs-state','2','--device','cpu','--skip-export']
    if not ppo_enabled:command.append('--no-ppo')
    result=subprocess.run(command,cwd=HERE,env=ENV,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'Embedding SL skipped' in result.stdout
    assert 'State checkpoint missing; initializing' in result.stdout
    assert f'Training stages: {"SL + PPO" if ppo_enabled else "SL only"}' in result.stdout
    assert not (output/'embedding_metrics.jsonl').exists()
    assert not (source/'state_student.pt').exists()
    assert (source/'frame_student.pt').read_bytes()==before
    expected=torch.load(source/'frame_student.pt',weights_only=True)
    for directory in (output,output/'post_sl'):
        actual=torch.load(directory/'frame_student.pt',weights_only=True)
        assert expected.keys()==actual.keys()
        assert all(torch.equal(value,actual[key]) for key,value in expected.items())
        assert (directory/'state_student.pt').is_file()
    phases=[json.loads(line)['phase'] for line in (output/'curriculum_metrics.jsonl').read_text().splitlines()]
    assert phases==['state','state']+(['ppo'] if ppo_enabled else [])
    if ppo_enabled:
        metrics=[json.loads(line) for line in (output/'ppo_metrics.jsonl').read_text().splitlines()]
        assert len(metrics)==1 and np.isfinite(metrics[0]['val_mean_ate_m'])
        assert (output/'state_student_ppo.pt').is_file()
    else:
        assert not (output/'ppo_metrics.jsonl').exists()


@pytest.mark.parametrize('kind,flags,message',[
    ('none',[],'requires --from-checkpoints'),
    ('missing',[],'requires --from-checkpoints'),
    ('rgb',[],'requires an embedding-mode checkpoint'),
    ('embedding',['--ppo-only'],'cannot use --ppo-only or --no-sl'),
    ('embedding',['--no-sl'],'cannot use --ppo-only or --no-sl')])
def test_invalid_skip_embedding_requests_fail_before_collection(tmp_path,kind,flags,message):
    source=tmp_path/'source';source.mkdir()
    (source/'training_config.json').write_text(json.dumps(dict(
        VISUAL_PREDICTION='rgb' if kind=='rgb' else 'embedding',USE_HYBRID_SSM_TRANSFORMER=False)))
    if kind not in ('none','missing'):(source/'frame_student.pt').touch()
    command=[sys.executable,str(HERE/'train.py'),'--skip-embedding-training',
        '--out',str(tmp_path/'new'),'--stream-data','--device','cpu']
    if kind!='none':command+=['--from-checkpoints',str(source)]
    result=subprocess.run(command+flags,env=ENV,capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and message in result.stderr
    assert not (tmp_path/'new').exists()
