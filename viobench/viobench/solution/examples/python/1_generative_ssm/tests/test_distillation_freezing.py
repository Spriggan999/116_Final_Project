"""All-student progression barriers and temporary optimizer freezing."""
from copy import deepcopy
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from distillation_curriculum import DistillationCurriculum
from distillation_pipeline import DistillationPipeline
from distillation_tracking import DistillationLogger
from distill import train_student,validate_student
from students.model import BudgetStudent,ALL_PROFILES as PROFILES
from test_curriculum import settings,recorded_store
from test_distillation_pipeline import payload
from test_distillation_tracking import FakeRun,records


def update(curriculum,index,large=1.3,small=1.):
    errors={'500':small,'10k':large}
    return curriculum.update(index,{axis:dict(errors) for axis in curriculum.schedule.axes},errors)


def paused_curriculum():
    curriculum=DistillationCurriculum(settings(),[11],stage_batches=3,min_batches=1)
    update(curriculum,0,large=1.)
    update(curriculum,1)
    event=update(curriculum,2)
    assert curriculum.paused=={'500'} and not event['promoted']
    return curriculum


def test_plateaued_student_waits_for_larger_student_then_resumes_on_promotion():
    curriculum=paused_curriculum()
    assert set(curriculum.training_students({'500':{},'10k':{}}))=={'10k'}
    assert update(curriculum,2) is None  # reused recordings do not consume checks
    assert curriculum.paused=={'500'}
    assert update(curriculum,3)['paused_students']==['500']
    event=update(curriculum,4,large=1.)
    assert set(event['promoted'])=={'motion','visual','imu'}
    assert not curriculum.paused and not curriculum.readiness and not curriculum.student_gates
    assert set(curriculum.training_students({'500':{},'10k':{}}))=={'500','10k'}


def test_validation_regression_revokes_pause_and_warm_start_rechecks_readiness():
    curriculum=paused_curriculum()
    state=json.loads(json.dumps(curriculum.state_dict()))
    assert state['paused']==['500'] and state['progression_policy']=='all_students_ready'
    restored=DistillationCurriculum(settings(),[11],stage_batches=3,min_batches=1)
    restored.load_state_dict(state)
    assert not restored.paused and not restored.student_gates
    event=update(curriculum,3,small=1.3)
    assert not curriculum.paused and event['readiness']['motion']['500'] is None


def test_memory_batch_reduction_preserves_the_readiness_probe_examples(tmp_path):
    store=recorded_store(tmp_path)
    curriculum=DistillationCurriculum(settings(BATCH=4),[31],probe_batches=2)
    for axis in curriculum.schedule.axes:
        large=curriculum.probe_loader(store,axis,4)
        small=curriculum.probe_loader(store,axis,1)
        assert large.batch_size==4 and small.batch_size==1
        assert large.dataset.indices==small.dataset.indices


def test_frozen_weights_and_optimizer_unchanged_while_peer_trains_and_validation_runs():
    torch.set_num_threads(1);torch.manual_seed(17)
    students={}
    for name in ('500','10k'):
        model=BudgetStudent(PROFILES[name])
        students[name]=dict(model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4))
    source=payload('cpu');operation=partial(train_student,device='cpu',weights=(1.,.5,1.))
    for name,row in students.items():operation(name,row,source)
    curriculum=paused_curriculum()
    weights=deepcopy(students['500']['model'].state_dict())
    optimizer=deepcopy(students['500']['optimizer'].state_dict())
    peer=deepcopy(students['10k']['model'].state_dict())
    with DistillationPipeline('cpu',students) as pipeline:
        for _ in range(2):
            pipeline.run(curriculum.training_students(students),(source,None),operation)
            values=pipeline.run(students,(source[:2],None),partial(validate_student,device='cpu'))
            assert set(values)==set(students)
    for key,value in weights.items():torch.testing.assert_close(students['500']['model'].state_dict()[key],value,atol=0,rtol=0)
    actual=students['500']['optimizer'].state_dict()
    assert actual['param_groups']==optimizer['param_groups']
    for index,state in optimizer['state'].items():
        for key,value in state.items():torch.testing.assert_close(actual['state'][index][key],value,atol=0,rtol=0)
    assert any(not torch.equal(students['10k']['model'].state_dict()[key],value) for key,value in peer.items())
    update(curriculum,3,large=1.)
    assert not curriculum.paused
    operation('500',curriculum.training_students(students)['500'],source)
    assert any(not torch.equal(students['500']['model'].state_dict()[key],value) for key,value in weights.items())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_schedule_for_training_subset_keeps_validation_and_resumed_models():
    torch.set_num_threads(1)
    students={}
    for name in ('500','10k'):
        model=BudgetStudent(PROFILES[name]).cuda()
        students[name]=dict(model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4))
    source=payload('cuda');operation=partial(train_student,device='cuda',weights=(1.,.5,1.))
    with DistillationPipeline('cuda',students) as pipeline:
        pipeline.groups=[['10k']];pipeline.max_concurrent=1
        with pipeline.batches([source],lambda value:value) as batches:
            assert set(pipeline.run({'10k':students['10k']},next(batches),operation))=={'10k'}
        with pipeline.batches([source[:2]],lambda value:value) as batches:
            values=pipeline.run(students,next(batches),partial(validate_student,device='cuda'))
            assert set(values)==set(students)
        with pipeline.batches([source],lambda value:value) as batches:
            assert set(pipeline.run(students,next(batches),operation))==set(students)


def test_wandb_reports_paused_students_without_inventing_zero_training_loss():
    run=FakeRun();rows=records()
    rows['500'].update(active=True,train_loss=None,training_paused=True,paused_waiting_for_curriculum=True)
    with DistillationLogger(run,rows) as logger:
        logger.progress(2,1,1,401,{'10k':.3},1,paused_students=('500',))
        assert run.history[-1]['progress/active_students']==2
        assert run.history[-1]['progress/paused_students']==1
        logger.epoch(2,rows,{},0.)
        assert '500/train_loss' not in run.history[-1]
        assert run.summary['500/status']=='paused_waiting_for_curriculum'


def test_cli_skips_paused_updates_validates_every_epoch_and_resumes(tmp_path):
    from export import make_models
    teacher=tmp_path/'teacher';teacher.mkdir()
    architecture=dict(backbone='gru',frame_mode='affine',frame_student='affine',hidden=16,
                      learned_visual=False,width=32,height=24,frame_input_stride=1,visual_input_stride=2)
    frame,state=make_models(architecture)
    torch.save(frame.state_dict(),teacher/'frame_student.pt');torch.save(state.state_dict(),teacher/'state_student.pt')
    (teacher/'student_config.json').write_text(json.dumps(architecture))
    (teacher/'training_config.json').write_text(json.dumps(dict(H=24,W=32,BATCH=1,WINDOW=11,
        USE_GENERATIVE_FRAME_MODEL=False,USE_LEARNED_VISUAL_ENCODER=False,STUDENT_HIDDEN=16,
        NUM_SCENES=3,MAX_FRAME_TRAIN_PAIRS=1,MAX_FRAME_VAL_PAIRS=1)))
    data=tmp_path/'data';data.mkdir();recorded_store(data)
    wrapper=tmp_path/'controlled_probes.py'
    wrapper.write_text(f'''import sys
sys.path.insert(0,{str(ROOT)!r})
import distillation_curriculum as dc
original=dc.DistillationCurriculum.update
def controlled(self,batch_id,axis_scores,ate_scores):
    self.checks=getattr(self,'checks',0)+1
    values={{name:1.3 if name=='10k' and 2<=self.checks<=4 else 1. for name in ate_scores}}
    return original(self,batch_id,{{axis:dict(values) for axis in axis_scores}},values)
dc.DistillationCurriculum.update=controlled
from distill import main
main()
''')
    output=tmp_path/'students'
    result=subprocess.run([sys.executable,str(wrapper),str(data),'--teacher',str(teacher),
        '--out',str(output),'--profiles','500','10k','--device','cpu','--window','11','--epochs','6',
        '--stage-epochs','3','--curriculum-min-batches','1','--curriculum-probe-batches','1',
        '--wandb-mode','disabled','--skip-export'],env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
        capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    small=[json.loads(line) for line in (output/'500/metrics.jsonl').read_text().splitlines()]
    large=[json.loads(line) for line in (output/'10k/metrics.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in small if row['training_paused']]==[4,5]
    assert all(row['train_loss'] is None and row['training_samples']==0 for row in small if row['training_paused'])
    assert all(row['validation_state_loss']>=0 for row in small)
    assert small[5]['training_samples']>0 and not small[5]['training_paused']
    assert all(row['training_samples']>0 for row in large)
    assert 'train_loss=paused' in result.stdout and 'paused_waiting_for_group' in result.stdout
    restores=[json.loads(line) for line in (output/'stage_restores.jsonl').read_text().splitlines()]
    assert len(restores)==1 and restores[0]['from_stage_index']==0 and restores[0]['to_stage_index']==1
    for name,rows in (('500',small),('10k',large)):
        best=min(rows[:5],key=lambda row:row['validation_mean_ate_m'])
        restored=restores[0]['students'][name]
        assert restored['epoch']==best['epoch'] and restored['optimizer_restored']
        assert restored['validation_mean_ate_m']==best['validation_mean_ate_m']
        assert rows[5]['curriculum_stage']==1 and rows[5]['stage_best_epoch']==6
        snapshot=torch.load(output/name/'stage_best.pt',map_location='cpu',weights_only=True)
        assert snapshot['stage_index']==1 and snapshot['epoch']==6
    assert 'Distillation stage restore:' in result.stdout
