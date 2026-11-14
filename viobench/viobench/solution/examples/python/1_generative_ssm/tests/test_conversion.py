"""Numerical and ABI regressions for the HyperDrone notebook conversion."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from export import make_models, export_models

torch.set_num_threads(1)

CONFIG = dict(width=320, height=240, fx=190.7, fy=190.7, cx=159.5, cy=119.5,
              T_cam_imu=[[0,-1,0,.02],[0,0,-1,.03],[1,0,0,0],[0,0,0,1]])


def settings(**changes):
    result = dict(backbone='gru', frame_mode='repeat', use_generated_frames=False,
                  learned_visual=False, hidden=32, width=96, height=72,
                  frame_generator_base=8, mamba_layers=2, attention_window=4,
                  hybrid_d_state=4, hybrid_head_dim=16, hybrid_attention_heads=2)
    result.update(changes)
    if result['backbone']=='mamba2_hybrid' and 'hidden' not in changes:result['hidden']=64
    return result


def estimator(monkeypatch, flags, state=None, frame=None):
    if state is None:
        frame, state = make_models(flags)
    files = {'deployment.json': json.dumps(flags).encode()}
    for name, model in [('state_student.npz', state), ('frame_student.npz', frame)]:
        if model is not None:
            buffer = io.BytesIO()
            np.savez(buffer, **{k: v.detach().cpu().numpy() for k,v in model.state_dict().items()})
            files[name] = buffer.getvalue()
    bundle = types.ModuleType('solution_bundle')
    bundle.exists = lambda name: name in files
    bundle.read_text = lambda name: files[name].decode()
    bundle.read_bytes = lambda name: files[name]
    bundle.open = lambda name: io.BytesIO(files[name])
    monkeypatch.setitem(sys.modules, 'solution_bundle', bundle)
    spec = importlib.util.spec_from_file_location('hyperdrone_test_solution', HERE/'solution/solution.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Solution(CONFIG), module, state


def test_gru_recurrence_matches_pytorch(monkeypatch):
    torch.manual_seed(17)
    guest, _, model = estimator(monkeypatch, settings())
    hidden = torch.zeros(1, 32)
    for token in torch.randn(20, 32):
        with torch.no_grad():
            hidden = model.gru(token[None], hidden)
        np.testing.assert_allclose(guest._gru_update(token.numpy()), hidden.numpy()[0], atol=2e-6, rtol=2e-5)


def test_mamba_and_bounded_attention_match_pytorch(monkeypatch):
    from mamba_helpers import require_mamba
    official=require_mamba()
    torch.manual_seed(17)
    guest, _, model = estimator(monkeypatch, settings(backbone='mamba2_hybrid'))
    model.cuda()
    assert all(isinstance(layer,official) for layer in model.ssm)
    caches = [layer.allocate_inference_cache(1,max_seqlen=1) for layer in model.ssm]
    history = []
    for token in torch.randn(15, 64):
        with torch.no_grad():
            x = token[None, None].cuda()
            for i, (block, norm) in enumerate(zip(model.ssm, model.ssm_norm)):
                y, conv, ssm = block.step(x, *caches[i])
                caches[i] = conv, ssm
                x = norm(x+y)
            history.append(x)
            history = history[-4:]
            memory = torch.cat(history, dim=1)
            attended, _ = model.attn(x, memory, memory, need_weights=False)
            reference = model.attn_norm(x+attended)[0, 0].cpu().numpy()
        np.testing.assert_allclose(guest._mamba_update(token.numpy()), reference, atol=4e-6, rtol=3e-5)
        assert len(guest.attn_cache) <= 4


@pytest.mark.parametrize('shape,target', [((72,96),(8,8)), ((72,96),(6,8)), ((17,21),(6,8))])
def test_pooling_matches_training(monkeypatch, shape, target):
    _, module, _ = estimator(monkeypatch, settings())
    image = np.random.default_rng(17).random(shape, dtype=np.float32)
    expected = F.adaptive_avg_pool2d(torch.from_numpy(image)[None, None], target).numpy()[0,0]
    np.testing.assert_allclose(module.adaptive_average_pool(image, *target), expected, atol=2e-7)


def test_camera_rotation_and_lever_arm(monkeypatch):
    guest, module, _ = estimator(monkeypatch, settings())
    frame = np.zeros((240, 320, 3), np.uint8)
    imu = np.zeros((10,7))
    imu[:,0] = np.arange(1,11)*.005
    imu[:,3] = 9.81
    first = guest.step(.05, frame, imu)
    np.testing.assert_allclose(first, [0,0,0,0,0,0,1,0,0,0,0,0,0], atol=1e-12)
    imu[:,0] += .05
    imu[:,6] = 1.
    actual = guest.step(.10, frame, imu)
    rotation = module.so3_exp(np.array([0.,0.,.05]))
    ci = np.asarray(CONFIG['T_cam_imu'])[:3,:3]
    lever = -ci.T @ np.asarray(CONFIG['T_cam_imu'])[:3,3]
    np.testing.assert_allclose(actual[:3], ci @ ((rotation-np.eye(3)) @ lever), atol=1e-10)
    np.testing.assert_allclose(actual[3:7], module.matrix_to_quat(ci@rotation@ci.T), atol=1e-10)
    np.testing.assert_allclose(actual[7:10], ci@np.cross([0,0,1], lever), atol=1e-10)
    np.testing.assert_allclose(actual[10:], ci@[0,0,1], atol=1e-10)


def test_empty_imu_and_timestamp_validation(monkeypatch):
    guest, _, _ = estimator(monkeypatch, settings())
    frame = np.zeros((240,320,3), np.uint8)
    assert np.isfinite(guest.step(.05, frame, np.empty((0,7)))).all()
    assert np.isfinite(guest.step(.10, frame, np.empty((0,7)))).all()
    with pytest.raises(ValueError, match='Frame timestamps'):
        guest.step(.10, frame, np.empty((0,7)))


@pytest.mark.parametrize('backbone,mode,visual', [
    ('gru','repeat',True), ('gru','affine',False), ('gru','flow',True),
    ('mamba2_hybrid','repeat',False), ('mamba2_hybrid','flow',True),
    ('gru','embedding',False),('gru','embedding',True),('mamba2_hybrid','embedding',True)])
def test_export_and_official_guest(tmp_path, backbone, mode, visual):
    if backbone=='mamba2_hybrid':
        from mamba_helpers import require_mamba
        require_mamba(cuda=False)
    torch.manual_seed(17)
    flags = settings(backbone=backbone, frame_mode=mode,
                     use_generated_frames=mode!='repeat', learned_visual=visual)
    frame, model = make_models(flags)
    with torch.no_grad():
        model.out.weight.normal_(0,.01)
        if mode == 'flow':
            frame.head.weight.normal_(0,.01)
            frame.head.bias.normal_(0,.01)
    output = export_models(frame, model, flags, tmp_path/'solution')
    assert (output/'deployment.json').is_file()
    with np.load(output/'state_student.npz', allow_pickle=False) as weights:
        assert 'out.weight' in weights
    with __import__('zipfile').ZipFile(output/'state_student.npz') as archive:
        assert all(info.compress_type == 0 for info in archive.infolist())


@pytest.mark.parametrize('hybrid', [False, True])
def test_training_checkpoints_can_be_reexported(tmp_path, hybrid):
    """Train briefly on synthetic recordings; all outputs stay in tmp_path."""
    import os
    import subprocess
    if hybrid:
        from mamba_helpers import require_mamba
        require_mamba()
    data = tmp_path/'data'
    data.mkdir()
    rng = np.random.default_rng(17)
    for scene in (0, 1):
        truth = np.zeros((41, 18), np.float32)
        truth[:,3:12] = np.eye(3).ravel()
        imu = np.zeros((41, 6), np.float32)
        imu[1:,2] = 9.81
        np.savez(data/f'rollout_s{scene:03d}_e000.npz',
                 camera=rng.integers(0,256,(9,48,64,3),dtype=np.uint8),
                 camera_ticks=np.arange(0,41,5,dtype=np.int32), imu=imu, state=truth,
                 dt=np.float32(.005), frame_stride=5, scene_id=scene)
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(H=24, W=32, WINDOW=21, BATCH=1,
        USE_GENERATIVE_FRAME_MODEL=not hybrid, USE_LEARNED_VISUAL_ENCODER=not hybrid,
        USE_HYBRID_SSM_TRANSFORMER=hybrid,
        HYBRID_NUM_MAMBA_LAYERS=1, HYBRID_D_STATE=4, HYBRID_HEAD_DIM=16,
        FRAME_GENERATOR_BASE=8, STUDENT_HIDDEN=64 if hybrid else 16, MAX_FRAME_TRAIN_PAIRS=2,
        MAX_FRAME_VAL_PAIRS=2, FRAME_CACHE_BATCH=1)))
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    checkpoints = tmp_path/'checkpoints'
    command = [sys.executable, str(HERE/'train.py'), str(data), '--config', str(config),
               '--epochs-frame', '1', '--epochs-state', '1', '--device', 'cuda' if hybrid else 'cpu',
               '--out', str(checkpoints), '--solution-out', str(tmp_path/'solution'),'--no-ppo','--no-amp','--loader-workers','0','--wandb-mode','disabled']
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (checkpoints/'frame_student.pt').is_file()
    assert (checkpoints/'state_student.pt').is_file()
    assert (checkpoints/'training_config.json').is_file()
    saved = json.loads((checkpoints/'student_config.json').read_text())
    assert saved['backbone'] == ('mamba2_hybrid' if hybrid else 'gru')
    result = subprocess.run([sys.executable, str(HERE/'export.py'), '--checkpoints', str(checkpoints),
                             '--out', str(tmp_path/'restored')], env=environment,
                            capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_array_equal(np.load(tmp_path/'solution/state_student.npz')['out.weight'],
                                  np.load(tmp_path/'restored/state_student.npz')['out.weight'])
