"""Native ABI parity, exact fuel caps, and teacher-to-student training/export."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[4]))
from students.model import BudgetStudent,PROFILES,LEGACY_PROFILES,camera_frames,integrate,qmatrix
from students.export import WORKSPACE,write_sources,ordered_weights,export_student,build
from viobench.tools.abi.config import default_guest_config
from viobench.tools.geometry import matrix_to_quaternion


@pytest.mark.parametrize('profile',[name for name,spec in LEGACY_PROFILES.items() if spec.get('kind') not in ('imu_affine','visual_imu_affine')])
def test_native_math_matches_torch_with_extrinsics_and_lever_arm(tmp_path,profile):
    if not shutil.which('cc'):pytest.skip('host C compiler not installed')
    if LEGACY_PROFILES[profile].get('kind')=='mamba2' and not torch.cuda.is_available():pytest.skip('official Mamba-2 needs CUDA')
    torch.set_num_threads(1);torch.manual_seed(17)
    student=BudgetStudent(LEGACY_PROFILES[profile]).eval()
    device='cpu'
    if student.is_ssm:
        if not torch.cuda.is_available():pytest.skip('official Mamba-2 needs CUDA')
        device='cuda';student.to(device)
    output=write_sources(student,tmp_path/'native')
    lib=output/'student.so'
    subprocess.run(['cc','-std=c11','-O2','-DSTUDENT_HOST','-shared','-fPIC',
        '-I'+str(WORKSPACE/'runtime/native'),str(output/'solution.c'),'-lm','-o',str(lib)],check=True)
    native=ctypes.CDLL(str(lib))
    fp=ctypes.POINTER(ctypes.c_float);dp=ctypes.POINTER(ctypes.c_double)
    native.student_set_weights.argtypes=[fp]
    native.init.argtypes=[ctypes.c_char_p,ctypes.c_uint32]
    native.step.argtypes=[ctypes.c_double,ctypes.POINTER(ctypes.c_uint8),dp,ctypes.c_int32,dp]
    weights=ordered_weights(student);native.student_set_weights(weights.ctypes.data_as(fp))
    config=default_guest_config();config['T_cam_imu'][0][3]=.12;config['T_cam_imu'][1][3]=-.04
    encoded=json.dumps(config).encode();assert native.init(encoded,len(encoded)) == 0
    sandbox=None
    if (WORKSPACE/'build/toolchain/wasi-sdk/bin/clang').is_file():
        from viobench import Sandbox,Limits
        sandbox=Sandbox(build(student,tmp_path/'wasm'),limits=Limits(step_fuel=PROFILES[profile]['fuel']))
        sandbox.init(config)
    Rci=np.asarray(config['T_cam_imu'])[:3,:3]
    lever=-Rci.T@np.asarray(config['T_cam_imu'])[:3,3]
    rng=np.random.default_rng(17)
    p=torch.zeros(1,3,dtype=torch.float64);v=p.clone();q=torch.tensor([[0.,0.,0.,1.]],dtype=torch.float64)
    hidden=torch.zeros(1,student.settings['hidden']);previous=None;gravity=None
    replay=[]
    for k in range(20):
        frame=rng.integers(0,256,(240,320,3),dtype=np.uint8)
        imu=np.column_stack([k*.05+np.arange(1,11)*.005,
            rng.normal(0,.1,(10,3))+[0,0,9.81],rng.normal(0,.1,(10,3))]).astype(np.float64)
        raw=torch.from_numpy(imu[:,1:]);mean=raw.mean(0)
        if gravity is None:
            gravity=-9.81*mean[:3].clamp(-20,20)/torch.linalg.vector_norm(mean[:3].clamp(-20,20))
        image=torch.from_numpy(frame).permute(2,0,1)[None,None].float()/255
        with torch.no_grad():
            gray=student.image_features(image)[:,0]
            if previous is None:previous=gray
            features=torch.cat([gray,gray-previous,(mean/10).float()[None],torch.tensor([[1. if k else 0.]])],-1)
            correction,hidden=student.update(features.to(device),hidden)
            correction=correction.cpu()
            previous=gray
            if k:
                if student.settings['aggregate']:
                    p,v,q,w=integrate(p,v,q,mean[None],correction,torch.tensor([.05],dtype=torch.float64),gravity[None])
                else:
                    for row in raw:
                        p,v,q,w=integrate(p,v,q,row[None],correction,torch.tensor([.005],dtype=torch.float64),gravity[None])
                R=qmatrix(q)[0].numpy()
                pose_rotation=Rci@R@Rci.T
                expected=np.concatenate([Rci@(p[0].numpy()+(R-np.eye(3))@lever),
                    matrix_to_quaternion(pose_rotation),
                    Rci@(R.T@v[0].numpy()+np.cross(w[0].numpy(),lever)),Rci@w[0].numpy()])
            else:
                expected=np.zeros(13);expected[6]=1
        actual=np.zeros(13,np.float64)
        assert native.step((k+1)*.05,frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            imu.ctypes.data_as(dp),10,actual.ctypes.data_as(dp)) == 0
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-5)
        if sandbox is not None:
            result=sandbox.step((k+1)*.05,frame,imu)
            np.testing.assert_allclose(np.concatenate([result.pose,result.twist]),expected,atol=3e-6,rtol=3e-5)
        if k<2:replay.append((frame.copy(),imu.copy(),actual.copy()))
        np.testing.assert_allclose(np.linalg.norm(actual[3:7]),1,atol=1e-12)
    # Empty IMU is allowed; the existing physical pose and velocity remain finite.
    assert native.step(1.05,frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        imu.ctypes.data_as(dp),0,actual.ctypes.data_as(dp)) == 0
    assert np.isfinite(actual).all()
    assert native.init(encoded,len(encoded))==0
    for k,(frame,imu,expected) in enumerate(replay):
        assert native.step((k+1)*.05,frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),imu.ctypes.data_as(dp),10,actual.ctypes.data_as(dp))==0
        np.testing.assert_allclose(actual,expected,atol=0,rtol=0)


def test_student_export_rejects_over_budget_candidate(tmp_path):
    if not (WORKSPACE/'build/toolchain/wasi-sdk/bin/clang').is_file():pytest.skip('wasi-sdk missing')
    model=BudgetStudent(dict(PROFILES['10k'],fuel=1))
    with pytest.raises(RuntimeError,match='exceeded its 1 fuel/frame tier'):
        export_student(model,tmp_path/'native',frames=2)
    assert not (tmp_path/'native/student.wasm').exists()
    result=json.loads((tmp_path/'native/limits.json').read_text())
    assert not result['passed'] and result['limits']['step_fuel']==1


def test_literal_100_fuel_student_is_sensor_aware_image_free_and_calibrated(tmp_path):
    if not shutil.which('cc'):pytest.skip('host C compiler missing')
    model=BudgetStudent(dict(fuel=100,grid_width=0,grid_height=0,hidden=0,aggregate=True,kind='imu_affine',profile='100')).eval()
    with torch.no_grad():
        model.gain.copy_(torch.tensor([.913,1.037,1.081]));model.bias.copy_(torch.tensor([.013,-.007,-9.67]))
    output=write_sources(model,tmp_path/'native');lib=output/'student.so'
    subprocess.run(['cc','-std=c11','-O3','-DSTUDENT_HOST','-shared','-fPIC',
        '-I'+str(WORKSPACE/'runtime/native'),str(output/'solution.c'),'-lm','-o',str(lib)],check=True)
    native=ctypes.CDLL(str(lib));fp=ctypes.POINTER(ctypes.c_float);dp=ctypes.POINTER(ctypes.c_double)
    native.student_set_weights.argtypes=[fp]
    native.init.argtypes=[ctypes.c_char_p,ctypes.c_uint32]
    native.step.argtypes=[ctypes.c_double,ctypes.POINTER(ctypes.c_uint8),dp,ctypes.c_int32,dp]
    weights=ordered_weights(model);native.student_set_weights(weights.ctypes.data_as(fp))
    from viobench.tools.geometry import quaternion_to_matrix
    config=default_guest_config()
    quaternion=np.array([.1,-.2,.15,.96]);quaternion/=np.linalg.norm(quaternion)
    Rci=quaternion_to_matrix(quaternion)
    config['T_cam_imu']=np.eye(4).tolist()
    for i in range(3):config['T_cam_imu'][i][:3]=Rci[i].tolist()
    encoded=json.dumps(config).encode();assert native.init(encoded,len(encoded))==0
    initial=np.zeros(13,np.float64)
    assert native.step(0.,None,None,0,initial.ctypes.data_as(dp))==0
    np.testing.assert_array_equal(initial,[0,0,0,0,0,0,1,0,0,0,0,0,0])
    assert native.init(encoded,len(encoded))==0
    rng=np.random.default_rng(100);packets=[];actual=[]
    for k in range(8):
        imu=np.column_stack([k*.05+np.arange(1,11)*.005,
            rng.normal(0,.2,(10,3))+[0,0,9.81],rng.normal(0,.2,(10,3))]).astype(np.float64)
        packets.append(imu)
        result=np.zeros(13,np.float64)
        # A null frame pointer proves this kernel never reads image memory.
        assert native.step((k+1)*.05,None,imu.ctypes.data_as(dp),10,result.ctypes.data_as(dp))==0
        actual.append(result)
    sequence=np.concatenate([packets[0][-1:,1:],*[p[:,1:] for p in packets[1:]]])
    with torch.no_grad():
        prediction=model(None,torch.from_numpy(sequence).float()[None],None,None,list(range(0,71,10)))
    for k,state in enumerate(actual):
        np.testing.assert_allclose(state[:3],Rci@prediction['pos'][0,k].numpy(),atol=2e-6,rtol=1e-5)
        np.testing.assert_allclose(state[7:10],Rci@prediction['vel'][0,k].numpy(),atol=2e-6,rtol=1e-5)
        np.testing.assert_array_equal(state[3:7],[0,0,0,1])
    assert np.linalg.norm(actual[-1][:3])>1e-4
    if (WORKSPACE/'build/toolchain/wasi-sdk/bin/clang').is_file():
        export_student(model,tmp_path/'wasm',frames=1200)
        report=json.loads((tmp_path/'wasm/limits.json').read_text())
        assert report['fuel_step_max']<=100 and report['limits']['step_fuel']==100
        from viobench import Sandbox,Limits
        sandbox=Sandbox(tmp_path/'wasm/student.wasm',limits=Limits(step_fuel=100));sandbox.init(config)
        image=np.zeros((240,320,3),np.uint8)
        first=sandbox.step(0.,image,np.empty((0,7),np.float64))
        np.testing.assert_array_equal(first.pose,[0,0,0,0,0,0,1])
        following=sandbox.step(.05,image,packets[0])
        assert following.fuel<=100


def test_training_camera_downsample_matches_abi_area_rounding():
    import cv2
    image=np.random.default_rng(17).integers(0,256,(480,640,3),dtype=np.uint8)
    actual=camera_frames(torch.from_numpy(image).permute(2,0,1)[None,None].float()/255)
    expected=cv2.resize(image,(320,240),interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal((actual[0,0].permute(1,2,0).numpy()*255).round().astype(np.uint8),expected)


@pytest.mark.parametrize('streaming',[False,True])
@pytest.mark.parametrize('visual_prediction',['rgb','embedding'])
def test_distill_all_tiers_preserves_teacher_and_exports_checkpoints(tmp_path,streaming,visual_prediction):
    from export import make_models
    teacher=tmp_path/'teacher';teacher.mkdir()
    settings=dict(backbone='gru',frame_mode='affine',frame_student='affine',hidden=16,
        learned_visual=False,width=32,height=24,frame_input_stride=1,visual_input_stride=2)
    if visual_prediction=='embedding':settings.update(frame_mode='embedding',frame_student='embedding_predictor',embedding_hidden=8)
    frame,state=make_models(settings)
    torch.save(frame.state_dict(),teacher/'frame_student.pt');torch.save(state.state_dict(),teacher/'state_student.pt')
    (teacher/'student_config.json').write_text(json.dumps(settings))
    (teacher/'training_config.json').write_text(json.dumps(dict(H=24,W=32,BATCH=1,WINDOW=11,
        USE_GENERATIVE_FRAME_MODEL=False,USE_LEARNED_VISUAL_ENCODER=False,STUDENT_HIDDEN=16,
        NUM_SCENES=2,STREAM_SCENES=1,STREAM_EPISODES_PER_SCENE=1,STREAM_VALIDATION_EPISODES=1,COLLECT_STEPS=21,
        MAX_FRAME_TRAIN_PAIRS=1,MAX_FRAME_VAL_PAIRS=1)))
    original={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in teacher.glob('*.pt')}
    data=tmp_path/'data';data.mkdir()
    rng=np.random.default_rng(17)
    for scene in (0,1):
        n=21;truth=np.zeros((n,18),np.float32);truth[:,3:12]=np.eye(3).ravel()
        imu=np.zeros((n,6),np.float32);imu[:,2]=9.81
        np.savez(data/f'rollout_s{scene:03d}_e000.npz',camera=rng.integers(0,256,(5,24,32,3),dtype=np.uint8),
            camera_ticks=np.arange(0,n,5),state=truth,imu=imu,dt=np.float32(.005),frame_stride=5,scene_id=scene)
    output=tmp_path/'students'
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    extra=['--profiles','1k','10k','100k','1m','100m']
    if streaming:
        from test_streaming_ppo import FAKE_COLLECTOR
        collector=tmp_path/'collect_fake.py';collector.write_text(FAKE_COLLECTOR)
        collection=tmp_path/'collection.json'
        collection.write_text(json.dumps(dict(NUM_SCENES=2,STREAM_SCENES=1,
            STREAM_EPISODES_PER_SCENE=1,STREAM_VALIDATION_SCENES=1,STREAM_VALIDATION_EPISODES=1,COLLECT_STEPS=21)))
        extra=['--stream-data','--collector',str(collector),'--max-data-mib','3',
               '--collection-config',str(collection),'--stream-dir',str(tmp_path/'managed'),'--profiles','1k']
        if visual_prediction=='embedding':
            extra=['--stream-data','--collector',str(collector),'--collection-config',str(collection),'--profiles','1k']
    result=subprocess.run([sys.executable,str(HERE/'distill.py'),str(data),'--teacher',str(teacher),
        '--out',str(output),'--epochs','1','--window','11','--device','cpu','--skip-export','--wandb-mode','disabled',*extra],
        env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode == 0,result.stdout+result.stderr
    performance=json.loads((output/'performance.jsonl').read_text())
    assert performance['stage']=='distillation' and performance['samples']>0
    assert performance['samples_per_second']>0 and not performance['parallel']
    assert performance['student_updates']==performance['samples']*len(performance['active_profiles'])
    if streaming:
        assert not Path(performance['recording_batch']).parent.exists()
        if visual_prediction=='embedding':
            provenance=json.loads((output/'distillation.json').read_text())
            import tempfile
            assert provenance['max_data_mib']==8192
            assert Path(provenance['stream_dir']).parent==Path(tempfile.gettempdir())
        else:assert not list((tmp_path/'managed').iterdir())
    for name,spec in PROFILES.items():
        if spec.get('kind')=='mamba2':continue
        if streaming and name!='1k':continue
        directory=output/name
        assert (directory/'student.pt').is_file()
        saved=json.loads((directory/'student.json').read_text())
        assert saved['fuel'] == spec['fuel']
        model=BudgetStudent(saved)
        model.load_state_dict(torch.load(directory/'student.pt',map_location='cpu',weights_only=True))
        assert all(torch.isfinite(p).all() for p in model.parameters())
        if name in ('1k','10k') and (WORKSPACE/'build/toolchain/wasi-sdk/bin/clang').is_file():
            result=subprocess.run([sys.executable,'-m','students.export','--checkpoints',str(directory),
                '--verify-frames','3'],cwd=HERE,env=env,capture_output=True,text=True,timeout=90)
            assert result.returncode == 0,result.stdout+result.stderr
            assert (directory/'native/student.wasm').read_bytes()[:4]==b'\0asm'
            assert json.loads((directory/'native/limits.json').read_text())['limits']['step_fuel']==spec['fuel']
    assert original=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in teacher.glob('*.pt')}
    if not streaming and visual_prediction=='embedding':
        recovered=tmp_path/'recovered'
        before=hashlib.sha256((output/'1k/student.pt').read_bytes()).hexdigest()
        result=subprocess.run([sys.executable,str(HERE/'distill.py'),str(data),'--teacher',str(teacher),
            '--out',str(recovered),'--warm-start',str(output),'--start-epoch','1','--epochs','2',
            '--window','11','--profiles','1k','--device','cpu','--skip-export','--wandb-mode','disabled'],
            env=env,capture_output=True,text=True,timeout=90)
        assert result.returncode==0,result.stdout+result.stderr
        provenance=json.loads((recovered/'distillation.json').read_text())
        assert provenance['warm_start']==str(output) and provenance['start_epoch']==1
        metrics=json.loads((recovered/'1k/metrics.jsonl').read_text())
        assert metrics['epoch']==2
        assert json.loads((recovered/'1k/student.json').read_text())==json.loads((output/'1k/student.json').read_text())
        assert before==hashlib.sha256((output/'1k/student.pt').read_bytes()).hexdigest()
        stage_run=tmp_path/'stage-recovered';trace=tmp_path/'restored-stage-state.pt'
        wrapper=tmp_path/'stage-recovery.py'
        wrapper.write_text(f'''import sys
from pathlib import Path
sys.path.insert(0,{str(HERE)!r})
import torch,distill
from distillation_checkpoints import cpu_snapshot
original=distill.train_student
from distillation_curriculum import DistillationCurriculum
load_original=DistillationCurriculum.load_state_dict
def loaded(self,state,**options):
    result=load_original(self,state,**options)
    Path({str(tmp_path/'restored-curriculum.json')!r}).write_text(__import__('json').dumps(self.state_dict()))
    return result
DistillationCurriculum.load_state_dict=loaded
def checked(name,row,payload,**options):
    if not Path({str(trace)!r}).exists():
        torch.save(cpu_snapshot(dict(model=row['model'].state_dict(),optimizer=row['optimizer'].state_dict(),patience=vars(row['patience']))),{str(trace)!r})
    return original(name,row,payload,**options)
distill.train_student=checked
distill.main()
''')
        result=subprocess.run([sys.executable,str(wrapper),str(data),'--teacher',str(teacher),
            '--out',str(stage_run),'--warm-start',str(output),'--warm-start-stage','--epochs','2',
            '--window','11','--profiles','1k','--device','cpu','--skip-export','--wandb-mode','disabled'],
            env=env,capture_output=True,text=True,timeout=90)
        assert result.returncode==0,result.stdout+result.stderr
        from test_distillation_checkpoints import equal_state
        expected=torch.load(output/'1k/stage_best.pt',map_location='cpu',weights_only=True)
        actual=torch.load(trace,map_location='cpu',weights_only=True)
        equal_state(actual['model'],expected['model']);equal_state(actual['optimizer'],expected['optimizer'])
        provenance=json.loads((stage_run/'distillation.json').read_text())
        assert provenance['warm_start_stage'] and provenance['start_epoch']==1
        recovered_curriculum=json.loads((tmp_path/'restored-curriculum.json').read_text())
        original_curriculum=json.loads((output/'curriculum_state.json').read_text())
        for axis,gates in recovered_curriculum['student_gates'].items():
            assert gates['1k']==original_curriculum['student_gates'][axis]['1k']
        assert actual['patience']==json.loads((output/'recovery_state.json').read_text())['students']['1k']['patience']
