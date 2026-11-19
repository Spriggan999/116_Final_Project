"""Protect matched-seed comparisons, optimizer partitioning and recovery."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lr_sweep import student_parameter_groups
from lr_sweep_windows import WindowSweep,common_seed_ranking,horizon,embedding_boundary_rates
from performance import adamw
from students.model import BudgetStudent,PROFILES


def test_separate_embedding_group_preserves_equal_rate_updates():
    torch.manual_seed(17)
    plain=BudgetStudent(dict(PROFILES['1k'],jepa_target=True))
    grouped=deepcopy(plain)
    groups=student_parameter_groups(grouped,3e-4,3e-4)
    ids=[id(value) for group in groups for value in group['params']]
    assert len(ids)==len(set(ids))==len(list(grouped.parameters()))
    a=torch.optim.AdamW(plain.parameters(),lr=3e-4)
    b=adamw(groups,lr=3e-4)
    for x,y in zip(plain.parameters(),grouped.parameters()):
        x.grad=torch.ones_like(x);y.grad=torch.ones_like(y)
    a.step();b.step()
    for x,y in zip(plain.parameters(),grouped.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
    changed=student_parameter_groups(grouped,1e-5,1e-3)
    assert [group['lr'] for group in changed]==[1e-5,1e-3]
    assert {id(value) for value in changed[1]['params']}=={
        id(value) for module in (grouped.vision,grouped.embedding_predictor) for value in module.parameters()}


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_adamw_group_generators_keep_learning_rates_and_fused_updates(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    a=torch.nn.Parameter(torch.ones(3,device=device));b=torch.nn.Parameter(torch.ones(3,device=device))
    optimizer=adamw([dict(params=(value for value in [a]),lr=1e-5),
                     dict(params=(value for value in [b]),lr=1e-3)],weight_decay=0)
    a.grad=torch.ones_like(a);b.grad=torch.ones_like(b);optimizer.step()
    assert [group['lr'] for group in optimizer.param_groups]==[1e-5,1e-3]
    assert (1-b).mean()>(1-a).mean()*50
    torch.testing.assert_close(a,torch.full_like(a,1-1e-5))
    torch.testing.assert_close(b,torch.full_like(b,1-1e-3))


def test_ranking_requires_both_seeds_and_excludes_failed_candidates():
    rows=[dict(seed=seed,lr=rate,status='complete',score=score) for seed,rate,score in
          [(17,1e-4,10.),(17,3e-4,11.),(17,1e-3,1.),(29,1e-4,100.),(29,3e-4,200.)]]
    rows.append(dict(seed=29,lr=1e-3,status='failed',score=None))
    ranking,scores=common_seed_ranking(rows)
    assert ranking==[1e-4,3e-4]
    assert scores[1e-4]==1.
    assert scores[3e-4]==pytest.approx(1.55)
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError,match='Duplicate'):common_seed_ranking(rows)


def test_embedding_edges_expand_both_metrics_with_finite_bounds():
    rows=[dict(status='complete',embedding_lr=rate,score=state,embedding_score=jepa)
          for rate,state,jepa in [(1e-5,1.,3.),(3e-4,2.,2.),(1e-3,3.,1.)]]
    assert embedding_boundary_rates(rows)==[1e-6,3e-6,3e-3,1e-2]
    rows=[dict(status='complete',embedding_lr=rate,score=loss,embedding_score=loss)
          for rate,loss in [(1e-7,1.),(1e-2,2.)]]
    assert embedding_boundary_rates(rows)==[]


def test_state_resume_finishes_missing_confirmations(tmp_path):
    # A completed seed-29 candidate must not make an interrupted window look
    # finished while the remaining shortlist is missing.
    sweep=object.__new__(WindowSweep);sweep.parent=tmp_path
    (tmp_path/'confirmation').mkdir()
    (tmp_path/'confirmation/recommendations.json').write_text('{"1k":{"best_lr_by_horizon":{"2":0.0003}}}')
    rows=[dict(model='1k',phase='window-4s',seed=17,lr=rate,status='complete',score=score)
          for rate,score in [(1e-5,5.),(3e-5,4.),(1e-4,1.),(3e-4,3.),(1e-3,2.)]]
    rows.append(dict(model='1k',phase='window-4s',seed=29,lr=1e-4,status='complete',score=1.))
    calls=[]
    sweep.group=lambda *args,**kwargs:rows
    def trial(name,seconds,seed,rate,*unused,**unused_options):
        calls.append((seed,rate))
        if not any(row['seed']==seed and row['lr']==rate for row in rows):
            rows.append(dict(model=name,phase=f'window-{seconds}s',seed=seed,lr=rate,status='complete',score=2.))
    sweep.get_trial=trial
    result=sweep.state_window('1k',4,None,None)
    assert (29,3e-4) in calls and (29,1e-3) in calls
    assert result['lr']==1e-4


@pytest.mark.parametrize('phase,seconds',[('screen-2s',2),('seed29-20s',20),('window-18s',18)])
def test_historical_and_new_window_identifiers(phase,seconds):
    assert horizon(dict(phase=phase))==seconds
