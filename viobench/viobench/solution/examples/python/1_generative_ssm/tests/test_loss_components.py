"""Loss accounting reconstructs objectives without changing their gradients."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from loss_tracking import LossAverages,record_components,rgb_objective
from losses import losses
from distill import train_student
from test_student_embedding import model_inputs


def test_state_terms_reconstruct_loss_and_preserve_gradients():
    model,inputs=model_inputs();pred=model(**inputs)
    truth={key:torch.zeros_like(pred[key]) for key in ['pos','vel','omega']}
    truth['rot']=torch.eye(3)[None,None].expand_as(pred['rot'])
    old,_=losses(pred,truth,'cpu',metrics=False)
    total,parts=losses(pred,truth,'cpu',metrics=False,components=True)
    assert set(parts)=={name+'_'+suffix for name in ['position','velocity','rotation','angular_velocity'] for suffix in ['raw','weight','weighted']}
    torch.testing.assert_close(total,sum(value for key,value in parts.items() if key.endswith('_weighted')))
    old_grads=torch.autograd.grad(old,tuple(model.parameters()),retain_graph=True,allow_unused=True)
    new_grads=torch.autograd.grad(total,tuple(model.parameters()),allow_unused=True)
    for old_grad,new_grad in zip(old_grads,new_grads):
        if old_grad is not None:torch.testing.assert_close(old_grad,new_grad,atol=0,rtol=0)
    assert all(not value.requires_grad for value in parts.values() if torch.is_tensor(value))


@pytest.mark.parametrize('distill',[False,True])
def test_student_nested_components_reconstruct_full_objective(distill):
    model,inputs=model_inputs()
    with torch.no_grad():pred=model(**inputs)
    truth={key:torch.zeros_like(pred[key]) for key in ['pos','vel','omega']}
    truth['rot']=torch.eye(3)[None,None].expand_as(pred['rot'])
    row=dict(model=model,optimizer=torch.optim.Adam(model.parameters(),lr=.001))
    teacher={key:value+.001 for key,value in truth.items()}
    teacher['rot']=truth['rot']
    corrections=torch.randn(1,4,9)*.01
    weights=(.8,.4,.7) if distill else (0.,0.,1.)
    total=train_student('1k',row,(inputs,truth,teacher,corrections),'cpu',weights,embedding_weight=.23)
    parts=row['loss_terms']
    leaf=sum(parts['state_'+key+'_weighted'] for key in ['position','velocity','rotation','angular_velocity'])
    leaf+=sum(parts['embedding_'+key+'_weighted'] for key in ['prediction','content','variance','covariance'])
    if distill:
        leaf+=sum(parts['teacher_state_'+key+'_weighted'] for key in ['position','velocity','rotation','angular_velocity'])
        kd=sum(parts['kd_'+key+'_weighted'] for key in ['gyro','body_acceleration','world_acceleration'])
        torch.testing.assert_close(kd,parts['correction_kd_weighted'])
        leaf+=kd
    torch.testing.assert_close(total.detach(),leaf)
    assert parts['embedding_covariance_weight']==pytest.approx(.23*.01)
    assert parts['state_position_weight']==pytest.approx(3*weights[2])


@pytest.mark.parametrize('enabled',[False,True])
def test_rgb_components_reconstruct_original_objective_and_gradients(enabled):
    pred=torch.rand(2,3,8,10,requires_grad=True);target=torch.rand_like(pred)
    flow=torch.rand(2,2,8,10,requires_grad=True)
    cfg=SimpleNamespace(USE_FRAME_EDGE_LOSS=enabled,FRAME_EDGE_LOSS_WEIGHT=.05,
        USE_GENERATIVE_FRAME_MODEL=True,USE_FLOW_SMOOTHNESS_LOSS=enabled,FRAME_FLOW_SMOOTHNESS_WEIGHT=.0001)
    total,parts=rgb_objective(pred,target,cfg,flow)
    original=F.l1_loss(pred,target)
    if enabled:
        dx=pred[:,:,:,1:]-pred[:,:,:,:-1]-target[:,:,:,1:]+target[:,:,:,:-1]
        dy=pred[:,:,1:,:]-pred[:,:,:-1,:]-target[:,:,1:,:]+target[:,:,:-1,:]
        original=original+.05*(dx.abs().mean()+dy.abs().mean())
        original=original+.0001*((flow[:,:,:,1:]-flow[:,:,:,:-1]).abs().mean()+(flow[:,:,1:,:]-flow[:,:,:-1,:]).abs().mean())
    torch.testing.assert_close(total,original)
    torch.testing.assert_close(total,sum(value for key,value in parts.items() if key.endswith('_weighted')))
    for a,b in zip(torch.autograd.grad(total,(pred,flow),retain_graph=True,allow_unused=True),torch.autograd.grad(original,(pred,flow),allow_unused=True)):
        if a is not None:torch.testing.assert_close(a,b)
    if not enabled:assert parts['edge_x_weight']==parts['flow_x_weight']==0


def test_averages_weight_samples_and_write_wandb_payload(tmp_path):
    values=LossAverages();values.update({'position_raw':torch.tensor(1.,requires_grad=True)},3)
    values.update({'position_raw':torch.tensor(5.)},1)
    logs=record_components(tmp_path,'student',7,values)
    assert logs['student/train_position_raw']==2
    saved=json.loads((tmp_path/'loss_components.jsonl').read_text())
    assert saved['epoch']==7 and saved['train']['position_raw']==2
    assert not values.sums['position_raw'].requires_grad


def test_teacher_embedding_state_and_internal_distillation_log_components(tmp_path):
    from test_streaming_ppo import FAKE_COLLECTOR
    collector=tmp_path/'collect.py';collector.write_text(FAKE_COLLECTOR)
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(H=24,W=32,WINDOW=21,BATCH=1,
        USE_HYBRID_SSM_TRANSFORMER=False,STUDENT_HIDDEN=16,TEACHER_BACKBONE='gru',TEACHER_HIDDEN=16,
        USE_TEACHER=True,EPOCHS_FRAME=1,EPOCHS_STUDENT=1,EPOCHS_TEACHER=1,EPOCHS_DISTILL=1,
        WANDB_MODE='disabled',VISUAL_PREDICTION='embedding',EMBEDDING_HIDDEN=16,
        USE_LEARNED_VISUAL_ENCODER=True,VISUAL_INPUT_STRIDE=2,NUM_SCENES=2,STREAM_SCENES=1,
        STREAM_VALIDATION_SCENES=1,COLLECT_STEPS=41,MAX_FRAME_TRAIN_PAIRS=2,MAX_FRAME_VAL_PAIRS=2,
        LOADER_WORKERS=0,USE_CURRICULUM=False)))
    output=tmp_path/'run'
    result=subprocess.run([sys.executable,str(HERE/'train.py'),'--config',str(config),'--stream-data',
        '--collector',str(collector),'--out',str(output),'--device','cpu','--no-ppo','--skip-export',
        '--max-data-mib','3'],env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
        capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    rows={row['phase']:row for row in map(json.loads,(output/'loss_components.jsonl').read_text().splitlines())}
    assert set(rows)=={'embedding','student','teacher','distill'}
    assert 'prediction_weighted' in rows['embedding']['train']
    for phase in ['student','teacher','distill']:
        assert rows[phase]['train']['position_weight']==3
        assert 'velocity_raw' in rows[phase]['validation']
    assert rows['distill']['train']['correction_kd_weight']==.2
