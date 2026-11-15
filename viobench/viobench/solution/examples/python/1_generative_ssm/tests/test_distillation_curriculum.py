"""Shared augmentation labels, fresh-batch progression and fixed validation."""
import json
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from curriculum import AugmentedDataset,difficult_images,photometric_parameters,photometric_captures
from data import StateWindows,camera_values
from distill import frame_targets,validation_inputs
from distillation_curriculum import DistillationCurriculum
from students.model import camera_frames
from test_curriculum import settings,recorded_store
from test_distillation_tracking import FakeRun
from distillation_tracking import DistillationLogger


def scores(value):return {axis:{'500':value,'1m':value,'100m':value} for axis in ('motion','visual','imu')}
def ates(value):return dict.fromkeys(('500','1m','100m'),value)
def reasons(event):return {reason for students in event['reasons'].values() for reason in students.values()}


def test_plateau_advances_only_after_fresh_exposure_and_holds_on_regression():
    curriculum=DistillationCurriculum(settings(),[11],stage_batches=3,min_batches=1)
    assert curriculum.update('a',scores(1.),ates(1.))['promoted']==[]
    for _ in range(6):assert curriculum.update('a',scores(.5),ates(.5)) is None
    assert curriculum.schedule.stages==dict(motion=0,visual=0,imu=0)
    assert curriculum.update('b',scores(1.2),ates(1.))['promoted']==[]
    event=curriculum.update('c',scores(1.),ates(1.))
    assert set(event['promoted'])=={'motion','visual','imu'}
    assert reasons(event)=={'plateau'}


def test_smaller_students_cannot_promote_before_largest_student_is_ready():
    curriculum=DistillationCurriculum(settings(),[11],stage_batches=6,min_batches=1)
    curriculum.update('a',scores(1.),ates(1.))
    values={axis:{'500':.8,'1m':.8,'100m':1.3} for axis in curriculum.schedule.axes}
    event=curriculum.update('b',values,ates(1.))
    assert not event['promoted']
    assert event['readiness']['motion']['500']=='improved'
    assert event['readiness']['motion']['100m'] is None
    event=curriculum.update('c',scores(.8),ates(.8))
    assert set(event['promoted'])=={'motion','visual','imu'}
    assert reasons(event)=={'improved'}


def test_default_minimum_exposure_and_two_improvement_checks_are_required():
    curriculum=DistillationCurriculum(settings(CURRICULUM_PATIENCE=2),[11],stage_batches=6,min_batches=3)
    assert not curriculum.update('a',scores(1.),ates(1.))['promoted']
    assert not curriculum.update('b',scores(.95),ates(.95))['promoted']
    assert set(curriculum.update('c',scores(.9),ates(.9))['promoted'])=={'motion','visual','imu'}


def test_full_difficulty_exposure_precedes_early_stopping_and_state_survives_warm_start():
    cfg=settings(CURRICULUM_FINAL_STAGE_CYCLES=3)
    curriculum=DistillationCurriculum(cfg,[11,21],stage_batches=3,min_batches=1)
    for index in range(20):
        curriculum.update(index,scores(1.),ates(1.))
        if curriculum.complete:break
        assert not curriculum.early_stopping_eligible
    assert curriculum.complete and curriculum.length==21 and curriculum.final_cycles==0
    for index in range(3):
        curriculum.update(f'final-{index}',{},ates(1.))
        assert curriculum.early_stopping_eligible==(index==2)
    restored=DistillationCurriculum(cfg,[11,21],stage_batches=3,min_batches=1)
    restored.load_state_dict(json.loads(json.dumps(curriculum.state_dict())))
    assert restored.complete and restored.schedule.stages==curriculum.schedule.stages
    assert restored.normalizers==curriculum.normalizers and restored.seen==curriculum.seen
    assert not restored.early_stopping_eligible  # warm-started weights need fresh final exposure


def test_deferred_appearance_matches_existing_transform_without_mutating_captures():
    cfg=settings();images=torch.randint(0,256,(2,5,3,12,16),dtype=torch.uint8)
    original=images.clone()
    parameters=[];expected=[]
    for index in range(2):
        generator=torch.Generator().manual_seed(index)
        parameters.append(photometric_parameters(.5,cfg,generator))
        expected.append(difficult_images(images[index],.5,cfg,torch.Generator().manual_seed(index)))
    actual=photometric_captures(images,torch.tensor(parameters),chunk=2)
    torch.testing.assert_close(actual,torch.stack(expected),atol=0,rtol=0)
    torch.testing.assert_close(images,original,atol=0,rtol=0)
    assert photometric_captures(images,torch.tensor([[1.,1.,0.]]).expand(2,-1)) is images


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_bounded_gpu_appearance_matches_cpu_and_preserves_identity_rows():
    images=torch.randint(0,256,(2,5,3,12,16),dtype=torch.uint8)
    parameters=torch.tensor([[1.,1.,0.],[1.1,.9,.2]])
    expected=photometric_captures(images,parameters,chunk=2)
    actual=photometric_captures(images.cuda(),parameters,chunk=2).cpu()
    torch.testing.assert_close(actual,expected,atol=1,rtol=0)
    torch.testing.assert_close(actual[0],images[0],atol=0,rtol=0)


def test_teacher_and_student_share_same_noisy_imu_and_single_appearance_transform(tmp_path,monkeypatch):
    import config as cfg
    monkeypatch.setattr(cfg,'USE_GENERATED_FRAMES_FOR_STATE',False)
    store=recorded_store(tmp_path)
    base=StateWindows(store,length=31,compact_cameras=True)
    original=base[0]
    curriculum=DistillationCurriculum(settings(CURRICULUM_MIX=[1.,0.,0.]),[31])
    curriculum.schedule.stages=dict(motion=2,visual=2,imu=2)
    view=AugmentedDataset(base,curriculum.cfg,curriculum.schedule,123)
    batch=next(iter(DataLoader(view,batch_size=1)))
    expected=photometric_captures(batch['cam_images'],batch['photometric'])
    class Teacher:
        def __call__(self,frames,imu,observed,dt,gravity,return_corrections):
            self.frames=frames;self.imu=imu
            b,t=imu.shape[:2]
            return dict(pos=torch.zeros(b,t,3),vel=torch.zeros(b,t,3),omega=torch.zeros(b,t,3),
                        rot=torch.eye(3).expand(b,t,3,3),corrections=torch.zeros(b,t,9))
    teacher=Teacher()
    inputs,truth,target,correction=frame_targets(batch,None,teacher,'cpu')
    assert not torch.equal(inputs['imu'][0],original['imu'])
    torch.testing.assert_close(teacher.imu,inputs['imu'],atol=0,rtol=0)
    torch.testing.assert_close(teacher.frames,camera_values(expected)[0,batch['anchor_img_idx'][0]][None],atol=0,rtol=0)
    ticks=inputs['ticks'];anchors=batch['anchor_img_idx'][:,ticks]
    student_images=camera_frames(camera_values(expected)[torch.arange(len(expected))[:,None],anchors])
    torch.testing.assert_close(inputs['images'],student_images,atol=0,rtol=0)
    for key in ('pos','vel','rot'):
        torch.testing.assert_close(truth[key][0],original[key][ticks],atol=0,rtol=0)
    torch.testing.assert_close(base[0]['imu'],original['imu'],atol=0,rtol=0)


def test_fixed_axis_probes_leave_main_validation_clean_and_log_curriculum(tmp_path):
    store=recorded_store(tmp_path)
    curriculum=DistillationCurriculum(settings(),[31],probe_batches=1,min_batches=1)
    clean=StateWindows(store,length=31,training=False,compact_cameras=True)[0]
    for axis in curriculum.schedule.axes:
        loader=curriculum.probe_loader(store,axis,1)
        assert loader is curriculum.probe_loader(store,axis,1)
        batch=next(iter(loader))
        inputs,truth,metadata=validation_inputs(batch,'cpu')
        assert torch.isfinite(inputs['imu']).all()
    torch.testing.assert_close(clean['cam_images'],StateWindows(store,length=31,training=False,compact_cameras=True)[0]['cam_images'],atol=0,rtol=0)
    run=FakeRun()
    with DistillationLogger(run,['500']) as logger:
        event=curriculum.update('a',scores(1.),ates(1.))
        event['student_probe_losses']=scores(1.)
        logger.curriculum(1,curriculum,event)
        assert run.history[-1]['curriculum/window_samples']==31
        assert 'curriculum/500/imu_probe_loss' in run.history[-1]
        assert 'distillation/curriculum' in run.summary
        event=curriculum.update('b',scores(.8),ates(.8))
        event['student_probe_losses']=scores(.8)
        logger.curriculum(2,curriculum,event)
        assert run.history[-1]['curriculum/motion_level']==0.
        assert run.history[-1]['curriculum/next_motion_level']==.5


def test_cli_reaches_full_difficulty_before_early_stopping_on_plateau(tmp_path):
    import os
    import subprocess
    from export import make_models
    root=Path(__file__).resolve().parents[1]
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
    output=tmp_path/'students'
    result=subprocess.run([sys.executable,str(root/'distill.py'),str(data),'--teacher',str(teacher),
        '--out',str(output),'--profiles','500','--device','cpu','--window','11','--epochs','20',
        '--lr','1e-30','--patience','1','--stage-epochs','2','--curriculum-min-batches','1',
        '--curriculum-probe-batches','1','--wandb-mode','disabled','--skip-export'],
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    curriculum=json.loads((output/'curriculum_state.json').read_text())
    assert all(stage==2 for stage in curriculum['stages'].values())
    events=[json.loads(line) for line in (output/'curriculum_metrics.jsonl').read_text().splitlines()]
    assert any('plateau' in reasons(event) for event in events)
    assert events[-1]['early_stopping_eligible'] and events[-1]['final_cycles']>=3
    rows=[json.loads(line) for line in (output/'500/metrics.jsonl').read_text().splitlines()]
    assert not rows[-1]['active'] and rows[-1]['epoch']<20
    assert 'val_loss=' in result.stdout and 'Mean Sim(3)-aligned ATE=' in result.stdout
