"""Resource verifier rejects oversized modules and episode timeout failures."""
from pathlib import Path
import json
import sys

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[4]))
import verify_limits


def test_oversized_upload_rejected_without_instantiation(tmp_path,monkeypatch):
    module=tmp_path/'large.wasm'; module.write_bytes(b'too large')
    report=tmp_path/'report.json'
    monkeypatch.setattr(verify_limits,'MAX_UPLOAD_BYTES',1)
    result=verify_limits.verify(module,frames=1,report=report,log=lambda line:None)
    assert not result['passed']
    assert '128 MiB' in result['error']
    assert result['frames_completed'] == 0
    assert json.loads(report.read_text())['passed'] is False


def test_episode_timeout_and_store_caps(tmp_path,monkeypatch):
    import viobench
    seen=[]
    class FakeSandbox:
        resolution=(640,480)
        memory=type('Memory',(),{'data_len':lambda self,store:64*2**20})()
        store=None
        init_fuel=1000
        init_wall_seconds=.05
        def __init__(self,module,limits,**kwargs):
            seen.append(limits)
        def init(self,config):
            pass
    monkeypatch.setattr(viobench,'Sandbox',FakeSandbox)
    tick=[-1441.]
    def now():
        tick[0]+=1441.
        return tick[0]
    monkeypatch.setattr(verify_limits.time,'perf_counter',now)
    module=tmp_path/'module.wasm'; module.write_bytes(b'wasm placeholder')
    result=verify_limits.verify(module,frames=1,log=lambda line:None)
    assert not result['passed'] and '24 minutes' in result['error']
    assert seen[0].memory_bytes == 2*2**30
    assert seen[0].table_elements == 1_048_576
    assert seen[0].step_fuel == 20_000_000_000
    assert seen[0].init_fuel == 200_000_000_000


def test_training_can_bundle_and_verify_selected_checkpoint(tmp_path):
    import os
    import subprocess
    import numpy as np
    import pytest
    runtime=HERE.parents[2]/'build/runtime/dist/runtime.wasm'
    if not runtime.is_file():
        pytest.skip('prebuilt runtime required; this test never builds or downloads it')
    data=tmp_path/'data'; data.mkdir()
    for scene in (0,1):
        truth=np.zeros((41,18),np.float32); truth[:,3:12]=np.eye(3).ravel()
        imu=np.zeros((41,6),np.float32); imu[1:,2]=9.81
        np.savez(data/f'rollout_s{scene:03d}_e000.npz',
            camera=np.full((9,24,32,3),127,np.uint8),camera_ticks=np.arange(0,41,5),
            state=truth,imu=imu,dt=np.float32(.005),frame_stride=5,scene_id=scene)
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(H=24,W=32,STUDENT_HIDDEN=16,WINDOW=11,
        USE_GENERATIVE_FRAME_MODEL=False,USE_LEARNED_VISUAL_ENCODER=False,USE_HYBRID_SSM_TRANSFORMER=False,WANDB_MODE='disabled',
        MAX_FRAME_TRAIN_PAIRS=1,MAX_FRAME_VAL_PAIRS=1)))
    output=tmp_path/'checkpoints'
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        XDG_CACHE_HOME=str(tmp_path/'cache'),VIOBENCH_RUNTIME=str(runtime),
        PYTHONPATH=os.pathsep.join([str(HERE.parents[4]),os.environ.get('PYTHONPATH','')]))
    result=subprocess.run([sys.executable,str(HERE/'train.py'),str(data),'--config',str(config),
        '--epochs-frame','1','--epochs-state','1','--device','cpu',
        '--out',str(output),'--solution-out',str(tmp_path/'solution'),
        '--verify-limits','--verify-frames','2','--no-ppo'],env=environment,
        capture_output=True,text=True,timeout=90)
    assert result.returncode == 0,result.stdout+result.stderr
    report=json.loads((output/'teacher-limits.json').read_text())
    assert report['passed'] and report['frames_completed'] == 2
    assert (output/'teacher.wasm').read_bytes()[:4] == b'\0asm'
