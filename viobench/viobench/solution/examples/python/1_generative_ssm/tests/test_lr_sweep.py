"""Fixed-data LR trials preserve real student input preprocessing and initialization."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lr_sweep import sparse_gray,CachedStudent,state_hash
from students.model import BudgetStudent,PROFILES


@pytest.mark.parametrize('grid',[(4,4),(6,3),(8,6),(32,24)])
def test_sparse_cache_matches_actual_abi_rounding_and_gray_inputs(grid):
    torch.set_num_threads(1);torch.manual_seed(17)
    images=torch.randint(0,256,(2,3,480,640),dtype=torch.uint8)
    spec=dict(PROFILES['1k'],grid_width=grid[0],grid_height=grid[1])
    model=BudgetStudent(spec)
    expected=model.image_features(images[None].float()/255)[0]
    actual=sparse_gray(images,*grid)
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)


def test_cached_student_predictions_and_gradients_match_original():
    from test_student_frontends import fixture
    original,inputs,_,_,_=fixture('1k',frames=4)
    cached=CachedStudent(original.settings);cached.load_state_dict(deepcopy(original.state_dict()))
    revised=dict(inputs,images=original.image_features(inputs['images']))
    expected=original(**inputs);actual=cached(**revised)
    for key in ['pos','vel','rot','omega','corrections']:torch.testing.assert_close(actual[key],expected[key])
    expected['pos'].square().mean().backward();actual['pos'].square().mean().backward()
    for a,b in zip(original.parameters(),cached.parameters()):
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad)
    assert state_hash(original)==state_hash(cached)


def test_fresh_multi_student_cli_uses_per_profile_learning_rates(tmp_path):
    import json,os,subprocess
    from test_curriculum import recorded_store
    here=Path(__file__).resolve().parents[1]
    data=tmp_path/'data';data.mkdir();recorded_store(data)
    settings=tmp_path/'config.json';settings.write_text(json.dumps(dict(H=24,W=32,BATCH=1,LOADER_WORKERS=0)))
    rates=tmp_path/'rates.json';rates.write_text(json.dumps({'1k':.00003,'10k':.0001}))
    output=tmp_path/'run'
    completed=subprocess.run([sys.executable,str(here/'train_students.py'),str(data),'--config',str(settings),
        '--out',str(output),'--profiles','1k','10k','--device','cpu','--window','21','--epochs','1',
        '--lr-config',str(rates),'--wandb-mode','disabled','--skip-export','--no-calibrate-architectures'],
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=90)
    assert completed.returncode==0,completed.stdout+completed.stderr
    metadata=json.loads((output/'distillation.json').read_text())
    assert metadata['learning_rates']=={'1k':.00003,'10k':.0001}
    for name,expected in metadata['learning_rates'].items():
        row=json.loads((output/name/'metrics.jsonl').read_text().splitlines()[-1])
        assert row['learning_rate']==expected
