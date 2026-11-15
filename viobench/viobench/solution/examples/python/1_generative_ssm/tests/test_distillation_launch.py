"""Automatic mixed schedules and the tmux-friendly launcher."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from distillation_pipeline import scheduling_candidates


def test_schedule_search_includes_serial_parallel_and_heavy_student_isolation():
    names=['500','10k','1m','100m','1m_ssm','1m_hybrid','100m_hybrid']
    costs={name:i+1 for i,name in enumerate(names)}
    costs['100m']=100
    plans=scheduling_candidates(names,costs)
    assert any(plan['mode']=='serial_students' for plan in plans)
    assert any(plan['groups']==[names] for plan in plans)
    assert any(plan['groups']==[['100m'],[name for name in sorted(names,key=costs.get,reverse=True) if name!='100m']] for plan in plans)
    for plan in plans:
        flattened=[name for group in plan['groups'] for name in group]
        assert sorted(flattened)==sorted(names) and len(flattened)==len(set(flattened))


def test_cuda_autotune_selects_fastest_safe_plan_and_restores_all_students():
    import torch
    from copy import deepcopy
    from functools import partial
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from students.model import BudgetStudent,PROFILES
    from distill import train_student
    from distillation_pipeline import DistillationPipeline
    from test_distillation_pipeline import payload
    torch.set_num_threads(1);torch.manual_seed(17)
    students={}
    for name in ('500','10k','1m'):
        model=BudgetStudent(PROFILES[name]).cuda()
        students[name]=dict(model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4))
    original={name:deepcopy(row['model'].state_dict()) for name,row in students.items()}
    source=payload('cuda');rng=torch.cuda.get_rng_state()
    with DistillationPipeline('cuda',students) as pipeline:
        with pipeline.batches([source],lambda batch:batch) as batches:
            result=pipeline.autotune(students,next(batches),partial(train_student,device='cuda',weights=(1.,.5,1.)))
        best=min((row for row in result['candidates'] if row['safe']),key=lambda row:row['seconds'])
        assert result['mode']==best['mode'] and result['groups']==best['groups']
        assert len(result['student_seconds'])==3
    for name,row in students.items():
        assert not row['optimizer'].state
        for key,value in original[name].items():torch.testing.assert_close(row['model'].state_dict()[key],value,atol=0,rtol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(),rng,atol=0,rtol=0)


@pytest.mark.parametrize('code',[0,7])
def test_launcher_logs_output_preserves_exit_status_and_does_not_launch_real_training(tmp_path,code):
    teacher=tmp_path/'teacher';teacher.mkdir()
    for name in ('frame_student.pt','state_student.pt','student_config.json','training_config.json'):
        (teacher/name).touch()
    fake=tmp_path/'fake-python'
    arguments=tmp_path/'arguments.json'
    fake.write_text(f'#!{sys.executable}\nimport json,sys\nopen({str(arguments)!r},"w").write(json.dumps(sys.argv[1:]))\nprint("fake training output")\nsys.exit({code})\n')
    fake.chmod(0o755)
    output=tmp_path/'students'
    result=subprocess.run(['bash',str(HERE/'distill_students.sh'),'--teacher',str(teacher),'--out',str(output),'--epochs','1'],
        env=dict(os.environ,HD_DISTILL_PYTHON=str(fake)),capture_output=True,text=True)
    assert result.returncode==code,result.stdout+result.stderr
    args=json.loads(arguments.read_text())
    assert args[args.index('--parallel-mode')+1]=='auto'
    assert args[args.index('--wandb-mode')+1]=='online'
    assert args[-2:]==['--epochs','1']
    assert 'fake training output' in Path(str(output)+'.log').read_text()
    assert not output.exists()  # distill.py, not tee, must create the run folder
