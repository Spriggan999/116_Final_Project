"""JEPA target isolation, checkpoint safety, gradients and zero export overhead."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from students.model import BudgetStudent,PROFILES
from students.export import ordered_weights,write_sources
from student_embedding import objective,update_target,target_key
from test_student_frontends import fixture
from distill import train_student


def model_inputs():
    base,inputs,_,_,_=fixture('1k',frames=5)
    model=BudgetStudent(dict(PROFILES['1k'],jepa_target=True))
    model.load_state_dict(base.state_dict())
    return model,inputs


def test_prediction_gradients_ema_and_deployment_unchanged(tmp_path):
    model,inputs=model_inputs()
    base=BudgetStudent(PROFILES['1k']);base.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('_jepa_target_')})
    np.testing.assert_array_equal(ordered_weights(base),ordered_weights(model))
    plain=write_sources(base,tmp_path/'plain');jepa=write_sources(model,tmp_path/'jepa')
    assert (plain/'student_weights.h').read_bytes()==(jepa/'student_weights.h').read_bytes()
    assert sum(p.numel() for p in model.parameters())==169
    loss,terms=objective(model,inputs);assert set(terms)=={'embedding_prediction','embedding_content','embedding_variance','embedding_covariance','embedding_loss'}
    loss.backward()
    for module in (model.vision,model.embedding_predictor):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    assert all(not value.requires_grad and value.grad is None for value in model.buffers())
    name=next(iter(model.vision.state_dict()));key=target_key(name)
    before=getattr(model,key).clone()
    with torch.no_grad():next(model.vision.parameters()).add_(.2)
    update_target(model,.99)
    torch.testing.assert_close(getattr(model,key),.99*before+.01*model.vision.state_dict()[name])
    restored=BudgetStudent(model.settings);restored.load_state_dict(model.state_dict())
    for key,value in model.state_dict().items():torch.testing.assert_close(restored.state_dict()[key],value)


def test_collapse_penalty_and_target_buffers_do_not_affect_inference():
    model,inputs=model_inputs()
    with torch.no_grad():
        for parameter in model.vision.parameters():parameter.zero_()
        update_target(model,0.)
    loss,terms=objective(model,inputs)
    assert torch.isfinite(loss) and terms['embedding_variance']>.08
    assert terms['embedding_content']==0
    # Changing a target encoder cannot alter the deployed trajectory.
    with torch.no_grad():before=model(**inputs)
    for buffer in model.buffers():buffer.add_(10.)
    with torch.no_grad():after=model(**inputs)
    for key in ['pos','vel','rot','omega']:torch.testing.assert_close(before[key],after[key])


def test_optimizer_step_updates_ema_only_when_auxiliary_enabled():
    model,inputs=model_inputs()
    with torch.no_grad():truth={key:value.detach()+.01 for key,value in model(**inputs).items() if key in ['pos','vel','rot','omega']}
    truth['rot']=torch.eye(3)[None,None].expand_as(truth['rot'])
    row=dict(model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=.003))
    initial={name:value.clone() for name,value in model.named_buffers()}
    loss=train_student('1k',row,(inputs,truth,None,None),'cpu',(0.,0.,1.),embedding_weight=.1)
    assert torch.isfinite(loss) and 'embedding_prediction' in row['loss_terms']
    assert any(not torch.equal(initial[name],value) for name,value in model.named_buffers())
    initial={name:value.clone() for name,value in model.named_buffers()}
    train_student('1k',row,(inputs,truth,None,None),'cpu',(0.,0.,1.),embedding_weight=0.)
    assert all(torch.equal(initial[name],value) for name,value in model.named_buffers())
    checkpoint=deepcopy(model.state_dict())
    loss,_=objective(model,inputs);loss.backward();update_target(model,0.)
    model.load_state_dict(checkpoint)
    assert all(torch.equal(checkpoint[name],value) for name,value in model.state_dict().items())


def test_all_profiles_cuda_jepa_gradients_and_ema():
    import pytest
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from performance import amp_disabled
    for name in PROFILES:
        _,inputs,_,_,_=fixture('1k',frames=4)
        inputs={key:value.cuda() if torch.is_tensor(value) else value for key,value in inputs.items()}
        model=BudgetStudent(dict(PROFILES[name],jepa_target=True)).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
        with amp_disabled():
            prediction=model(**inputs)
            truth={key:torch.zeros_like(prediction[key]) for key in ['pos','vel','omega']}
            truth['rot']=torch.eye(3,device='cuda')[None,None].expand_as(prediction['rot'])
            row=dict(model=model,optimizer=optimizer)
            loss=train_student(name,row,(inputs,truth,None,None),'cuda',(0.,0.,1.),alignment_weight=.1)
        assert torch.isfinite(loss)
        assert 'embedding_prediction' in row['loss_terms']
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.embedding_predictor.parameters())
        assert all(torch.isfinite(value).all() for value in model.buffers())


def test_cuda_scheduling_probe_restores_ema_buffers():
    import pytest
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from functools import partial
    from distillation_pipeline import DistillationPipeline
    model,inputs=model_inputs();model=model.cuda()
    inputs={key:value.cuda() if torch.is_tensor(value) else value for key,value in inputs.items()}
    truth=dict(pos=torch.zeros(1,5,3,device='cuda'),vel=torch.zeros(1,5,3,device='cuda'),
               omega=torch.zeros(1,5,3,device='cuda'),rot=torch.eye(3,device='cuda')[None,None].expand(1,5,3,3))
    row=dict(model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=.003))
    before=deepcopy(model.state_dict())
    operation=partial(train_student,device='cuda',weights=(0.,0.,1.),alignment_weight=.1)
    with DistillationPipeline('cuda',['1k']) as pipeline:
        with pipeline.batches([0],lambda _: (inputs,truth,None,None)) as batches:
            prepared=next(batches)
            pipeline.autotune({'1k':row},prepared,operation)
    for key,value in before.items():torch.testing.assert_close(model.state_dict()[key],value,atol=0,rtol=0)
