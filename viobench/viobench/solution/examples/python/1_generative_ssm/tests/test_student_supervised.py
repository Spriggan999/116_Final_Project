"""Direct student supervision, ordered plateau progression, and sequential runs."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from student_curriculum import PlateauGate,StagedStudentCurriculum
from test_curriculum import settings,recorded_store


def test_supervised_preset_excludes_hssd_and_keeps_scenes_disjoint():
    from scenes import scene_specs
    preset=json.loads((HERE/'supervised_procthor.json').read_text())
    scenes=scene_specs(preset['SCENE_CATALOG'])
    start=preset['SCENE_OFFSET'];end=start+preset['NUM_SCENES']
    training=scenes[start:end]
    validation=scenes[end:end+preset['STREAM_VALIDATION_SCENES']]
    assert len(training)==192 and len(validation)==8
    assert all(catalog=='procthor' for catalog,_,_ in training+validation)
    assert not {sid for _,_,sid in training}&{sid for _,_,sid in validation}


def test_continuing_improvement_does_not_trigger_plateau():
    gate=PlateauGate(minimum=3,patience=3)
    for index in range(30):assert not gate.update(.95**index)
    assert gate.last_reason is None


def test_plateau_requires_stability_near_best_not_just_old_age():
    gate=PlateauGate(minimum=3,patience=3)
    for score in [10.,8.,*([9.]*15)]:assert not gate.update(score)
    assert not gate.update(8.01)
    assert not gate.update(8.02)
    assert gate.update(8.01) and gate.last_reason=='plateau'


def step(cur,cycle,values=None,ates=None):
    values=values or {'1k':1.}
    return cur.update(cycle,{axis:values for axis in cur.probe_axes},ates or values)


def test_ordered_motion_horizon_resets_and_late_augmentation_then_early_stopping():
    cur=StagedStudentCurriculum(settings(),[11,21],stage_batches=3,plateau_patience=2)
    transitions=[]
    assert cur.schedule.levels==[i/9 for i in range(10)]
    for cycle in range(200):
        event=step(cur,cycle)
        if event['promoted'] or event['window_promoted']:
            transitions.append((event['phase'],cur.length,cur.schedule.snapshot()))
        if cur.complete:break
        if cur.length==11 or cur.schedule.stages['motion']<9:
            assert cur.schedule.stages['visual']==cur.schedule.stages['imu']==0
        if cur.schedule.stages['imu']<9:assert cur.schedule.stages['visual']==0
    assert [phase for phase,_,_ in transitions]==['motion']*9+['window']+['motion']*9+['imu']*9+['visual']*9
    assert transitions[9][1:]==(21,dict(motion=0.,visual=0.,imu=0.))
    assert cur.complete and not cur.early_stopping_eligible
    for index in range(3):
        step(cur,f'final-{index}')
        assert cur.early_stopping_eligible==(index==2)


def test_reused_batches_do_not_age_gates_and_waiting_students_freeze():
    cur=StagedStudentCurriculum(settings(),[11],stage_batches=3,plateau_patience=2)
    for cycle in range(5):
        event=step(cur,cycle,{'1k':1.,'1m':.9**cycle})
        before=deepcopy(cur.state_dict())
        assert step(cur,cycle,{'1k':.1,'1m':.1}) is None
        assert cur.state_dict()==before
        assert not event['promoted']
    assert cur.paused=={'1k'}
    for cycle in range(5,10):
        event=step(cur,cycle,{'1k':1.,'1m':.9**4})
        if event['promoted']:break
    assert event['promoted']==['motion'] and not cur.paused


def test_staged_recovery_retains_plateau_history_and_phase():
    cur=StagedStudentCurriculum(settings(),[11,21],stage_batches=6)
    step(cur,'a');step(cur,'b')
    state=json.loads(json.dumps(cur.state_dict()))
    restored=StagedStudentCurriculum(settings(),[11,21])
    restored.load_state_dict(state,preserve_progress=True,students=['1k'])
    assert restored.state_dict()==cur.state_dict()
    assert step(restored,'c')==step(cur,'c')


def test_staged_training_does_not_add_disabled_augmentations(tmp_path):
    from curriculum import AugmentedDataset
    from data import StateWindows
    store=recorded_store(tmp_path)
    cur=StagedStudentCurriculum(settings(),[31])
    base=StateWindows(store,length=31,training=True,compact_cameras=True)
    for seed in range(20):
        view=AugmentedDataset(base,cur.cfg,cur.schedule,seed)
        assert view.level('imu',store.entries[0])==view.level('visual',store.entries[0])==0.
    cur.schedule.stages.update(motion=9,imu=5)
    view=AugmentedDataset(base,cur.cfg,cur.schedule,17)
    assert view.level('visual',store.entries[0])==0.
    # Fixed probe examples are independent of chosen memory batch size.
    a=cur.probe_loader(store,'imu',1);b=cur.probe_loader(store,'imu',2)
    assert a.dataset.indices==b.dataset.indices
    probe=a.dataset.dataset
    assert all(probe.level('imu',entry)==cur.schedule.levels[5] for entry in store.entries)
    assert probe.mix==[1.,0.,0.]
    # Sparse motion bins must never fall forward to a harder validation bin.
    from curriculum import motion_scores
    cur.schedule.stages['motion']=2
    selected=cur.probe_loader(store,'motion',1).dataset
    scores=motion_scores(selected.dataset.base)
    cuts=torch.quantile(scores,torch.linspace(0,1,11,dtype=torch.float64)[1:-1])
    bins=torch.bucketize(scores,cuts,right=False)
    assert all(int(bins[index])<=2 for index in selected.indices)


def test_supervised_loss_never_needs_teacher_targets():
    from distill import train_student,state_loss
    from students.model import BudgetStudent,PROFILES
    from test_distillation_pipeline import payload
    torch.manual_seed(17)
    model=BudgetStudent(PROFILES['1k'])
    inputs,truth=payload('cpu')[:2]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    expected=state_loss(model(**inputs),truth,'cpu').detach()
    before=deepcopy(model.state_dict())
    loss=train_student('1k',dict(model=model,optimizer=optimizer),(inputs,truth,None,None),'cpu',(0.,0.,1.))
    torch.testing.assert_close(loss.detach(),expected)
    assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items())


def test_all_earlier_motion_bins_share_the_replay_budget_without_harder_samples(monkeypatch):
    import curriculum
    cur=StagedStudentCurriculum(settings(),[11])
    cur.schedule.stages['motion']=8
    scores=torch.arange(100,dtype=torch.float64)
    monkeypatch.setattr(curriculum,'motion_scores',lambda dataset:scores)
    sampler=curriculum.curriculum_sampler(range(100),cur.schedule,None,17)
    masses=sampler.weights.reshape(10,10).sum(1)
    torch.testing.assert_close(masses[:8],torch.full((8,),.25/8,dtype=torch.float64))
    assert masses[8]==pytest.approx(.75) and masses[9]==0
    cur.schedule.stages['motion']=0
    weights=curriculum.curriculum_sampler(range(100),cur.schedule,None,17).weights
    assert weights[:10].sum()==pytest.approx(1.) and weights[10:].sum()==0


def test_augmentation_replay_can_select_any_previous_level():
    from curriculum import mixed_level
    generator=torch.Generator().manual_seed(17)
    levels=[i/9 for i in range(10)]
    samples=[mixed_level(8,levels,[.75,.25,0.],generator,all_easier=True) for _ in range(4000)]
    assert set(samples)==set(levels[:9])
    assert samples.count(levels[8])/len(samples)==pytest.approx(.75,abs=.025)
    counts=[samples.count(value)/len(samples) for value in levels[:8]]
    assert all(abs(count-.25/8)<.012 for count in counts)


def test_imu_noise_amplitude_and_blur_vary_with_seed_but_remain_bounded():
    from curriculum import noisy_imu,photometric_parameters
    cfg=settings(CURRICULUM_RANDOM_ERROR_STRENGTH=True,CURRICULUM_ACCEL_BIAS=0.,
                 CURRICULUM_GYRO_BIAS=0.,CURRICULUM_ACCEL_WALK=0.,CURRICULUM_GYRO_WALK=0.)
    imu=torch.zeros(20000,6)
    a=noisy_imu(imu,.005,1.,cfg,torch.Generator().manual_seed(17))
    b=noisy_imu(imu,.005,1.,cfg,torch.Generator().manual_seed(17))
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    deviations=[];blurs=[]
    for seed in range(20):
        result=noisy_imu(imu,.005,1.,cfg,torch.Generator().manual_seed(seed))
        deviations.append(float(result[1:,0].std()))
        brightness,contrast,blur=photometric_parameters(1.,cfg,torch.Generator().manual_seed(seed))
        assert .6<=brightness<=1.4 and .6<=contrast<=1.4 and 0<=blur<=.5
        blurs.append(blur)
    assert min(deviations)>=.065 and max(deviations)<=.147
    assert max(deviations)-min(deviations)>.025
    assert max(blurs)-min(blurs)>.3
    assert photometric_parameters(0.,cfg,torch.Generator())==(1.,1.,0.)


def test_supervised_wandb_uses_supervised_namespace_and_plateau_metrics():
    from distillation_tracking import DistillationLogger
    from test_distillation_tracking import FakeRun
    run=FakeRun();cur=StagedStudentCurriculum(settings(),[11])
    with DistillationLogger(run,['1k'],namespace='supervised') as logger:
        event=step(cur,'a');event['student_probe_losses']={'motion':{'1k':1.}}
        logger.curriculum(1,cur,event)
        assert run.history[-1]['supervised/epoch']==1
        assert run.history[-1]['curriculum/phase']=='motion'
        assert run.history[-1]['curriculum/1k/motion_plateau_wait']==0
        assert 'supervised/curriculum' in run.summary


def test_supervised_cli_completes_all_stages_before_early_stopping(tmp_path):
    data=tmp_path/'data';data.mkdir();recorded_store(data)
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(H=24,W=32,BATCH=1,LOADER_WORKERS=0,MAX_FRAME_TRAIN_PAIRS=1,MAX_FRAME_VAL_PAIRS=1)))
    output=tmp_path/'complete'
    command=[sys.executable,str(HERE/'train_students.py'),str(data),'--config',str(config),
        '--out',str(output),'--profiles','1k','--device','cpu','--window','11','--epochs','110',
        '--lr','1e-30','--patience','1','--stage-epochs','2','--curriculum-min-batches','1',
        '--plateau-patience','2','--batch','1','--skip-export','--wandb-mode','disabled']
    result=subprocess.run(command,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
                          capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    recovery=json.loads((output/'recovery_state.json').read_text())
    assert all(v==9 for v in recovery['curriculum']['stages'].values())
    assert not recovery['students']['1k']['active'] and recovery['completed_epochs']<110
    events=[json.loads(line) for line in (output/'curriculum_metrics.jsonl').read_text().splitlines()]
    assert [e['promoted'][0] for e in events if e['promoted']]==['motion']*9+['imu']*9+['visual']*9
    assert events[-1]['early_stopping_eligible']
    assert (output/'stage_restores.jsonl').is_file()


def test_teacher_free_cli_fresh_training_recovery_and_wandb_logging(tmp_path):
    data=tmp_path/'data';data.mkdir();recorded_store(data)
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(H=24,W=32,BATCH=1,LOADER_WORKERS=0,MAX_FRAME_TRAIN_PAIRS=1,MAX_FRAME_VAL_PAIRS=1)))
    output=tmp_path/'fresh'
    common=[sys.executable,str(HERE/'train_students.py'),str(data),'--config',str(config),
            '--profiles','1k','--device','cpu','--window','21','--batch','1',
            '--wandb-mode','disabled','--skip-export','--no-calibrate-architectures']
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run([*common,'--out',str(output),'--epochs','2'],env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    meta=json.loads((output/'distillation.json').read_text())
    assert meta['ground_truth_only'] and meta['teacher'] is None and meta['warm_start'] is None
    assert meta['kd_weight']==meta['teacher_state_weight']==0.
    assert meta['embedding_weight']==.1 and meta['embedding_ema']==.99
    assert meta['alignment_weight']==.1
    metrics=json.loads((output/'1k/metrics.jsonl').read_text().splitlines()[-1])
    for key in ['train_state_loss','train_embedding_loss','train_embedding_prediction','validation_embedding_loss',
                'train_state_position_raw','train_state_position_weighted','train_state_velocity_raw',
                'validation_state_angular_velocity_weighted','train_embedding_covariance_weighted',
                'validation_unaligned_mte_m','validation_unaligned_ate_rmse_m',
                'train_alignment_raw','train_alignment_weighted','validation_alignment_raw']:
        assert key in metrics and metrics[key]>=0
    assert metrics['train_alignment_insufficient_samples_fraction']==0.
    assert metrics['train_alignment_weight']==pytest.approx(.1)
    checkpoint=torch.load(output/'1k/student.pt',map_location='cpu',weights_only=True)
    assert any(key.startswith('_jepa_target_') for key in checkpoint)
    assert meta['curriculum_order']=='staged'
    assert len(meta['curriculum_settings']['CURRICULUM_LEVELS'])==10
    assert meta['curriculum_settings']['CURRICULUM_MIX']==[.75,.25,0.]
    assert meta['collection']['SCENE_CATALOG']=='procthor'
    assert meta['collection']['NUM_SCENES']==192 and meta['collection']['STREAM_VALIDATION_SCENES']==8
    assert (output/'1k/student.pt').is_file()
    state=json.loads((output/'recovery_state.json').read_text())
    assert state['completed_epochs']==2
    assert state['curriculum']['student_gates']['motion']['1k']['age']==2
    continued=tmp_path/'continued'
    resume_common=list(common);index=resume_common.index('--profiles');del resume_common[index:index+2]
    result=subprocess.run([*resume_common,'--out',str(continued),'--epochs','3',
                          '--warm-start',str(output),'--warm-start-stage'],env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    state=json.loads((continued/'recovery_state.json').read_text())
    assert state['completed_epochs']==3 and state['curriculum']['student_gates']['motion']['1k']['age']==3
    assert 'val_loss=' in result.stdout and 'Mean Sim(3)-aligned ATE=' in result.stdout


@pytest.mark.parametrize('code',[0,7])
def test_sequential_launcher_orders_profiles_and_stops_on_failure(tmp_path,code):
    arguments=tmp_path/'arguments.jsonl';fake=tmp_path/'fake-python'
    fake.write_text(f'#!{sys.executable}\nimport json,sys\nwith open({str(arguments)!r},"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nprint("fake supervised run")\nsys.exit({code})\n')
    fake.chmod(0o755);output=tmp_path/'seven'
    result=subprocess.run(['bash',str(HERE/'train_students_sequential.sh'),'--out-root',str(output),
                           '--epochs','1'],env=dict(os.environ,HD_STUDENT_PYTHON=str(fake)),capture_output=True,text=True)
    assert result.returncode==code,result.stdout+result.stderr
    rows=[json.loads(line) for line in arguments.read_text().splitlines()]
    names=[r[r.index('--profiles')+1] for r in rows]
    assert names==(['1k'] if code else ['1k','10k','100k','1m_ssm','1m','1m_hybrid','100m','100m_hybrid'])
    assert all('--teacher' not in row and '--warm-start' not in row for row in rows)
    assert all(Path(row[row.index('--collection-config')+1]).name=='supervised_procthor.json' for row in rows)
    assert (output/'runs/1k.log').is_file()
