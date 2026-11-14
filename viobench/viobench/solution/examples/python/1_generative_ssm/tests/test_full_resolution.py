"""Full-resolution collection/training/export and bounded sequence memory."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import config as cfg
from data import EpisodeStore, generate_200hz_frames
from export import export_models, make_models
from models import GenerativeFlowPredictor, VisualFeatures


def test_default_teacher_and_collection_are_full_resolution():
    assert (cfg.H,cfg.W,cfg.SOURCE_HEIGHT,cfg.SOURCE_WIDTH) == (480,640,480,640)
    assert (cfg.BATCH,cfg.FRAME_PAIR_BATCH,cfg.FRAME_GENERATION_CHUNK,cfg.VISUAL_CHUNK) == (4,8,16,16)
    assert cfg.COLLECT_INSTANCES == 2 and cfg.USE_AMP
    assert not cfg.CACHE_GENERATED_WINDOWS
    assert cfg.USE_HYBRID_SSM_TRANSFORMER and cfg.STUDENT_HIDDEN==896
    assert cfg.SCENE_CATALOG == 'mixed'
    assert (cfg.NUM_SCENES,cfg.STREAM_VALIDATION_SCENES,cfg.COLLECT_STEPS)==(313,12,2000)
    assert cfg.NUM_SCENES+cfg.STREAM_VALIDATION_SCENES==325 and cfg.STREAM_MAX_MIB==3072
    from export import notebook_settings
    resolved=notebook_settings(cfg)
    assert resolved['backbone']=='mamba2_hybrid' and resolved['mamba_layers']==3
    preset=json.loads((HERE/'full_resolution.json').read_text())
    assert preset['USE_HYBRID_SSM_TRANSFORMER'] and preset['STUDENT_HIDDEN']==896


@pytest.mark.parametrize('hybrid',[True,False])
def test_training_cli_records_large_teacher_defaults(tmp_path,hybrid):
    """Check the default hybrid and explicit GRU keep their profiled widths."""
    if hybrid:
        from mamba_helpers import require_mamba
        require_mamba()
    import os
    import subprocess
    output=tmp_path/'run'
    result=subprocess.run([sys.executable,'-c',
        'import train; train.run_training = lambda *args: None; train.main()',
        '--out',str(output),'--device','cuda' if hybrid else 'cpu','--skip-export',
        *([] if hybrid else ['--config',str(HERE/'gru.json')])],
        cwd=HERE,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
        capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    settings=json.loads((output/'student_config.json').read_text())
    assert settings['hidden'] == (896 if hybrid else 1792)
    assert settings['frame_generator_base'] == 33
    assert settings['frame_input_stride'] == settings['visual_input_stride'] == 8
    assert settings['backbone'] == ('mamba2_hybrid' if hybrid else 'gru') and settings['frame_mode'] == 'embedding'
    assert settings['visual_prediction']=='embedding' and settings['embedding_hidden']==256
    frame,state=make_models(settings)
    from models import EmbeddingPredictor
    assert isinstance(frame,EmbeddingPredictor) and isinstance(state.visual,torch.nn.Identity)
    assert sum(p.numel() for model in (frame,state) for p in model.parameters()) < 20_428_876


def test_small_recordings_cannot_silently_be_upsampled(tmp_path):
    path=tmp_path/'old.npz'
    np.savez(path,camera=np.zeros((2,120,160,3),np.uint8))
    with pytest.raises(ValueError,match='requires 640x480 recordings'):
        EpisodeStore([path],480,640)


def test_full_resolution_visual_backward_matches_unchunked(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(cfg,'VISUAL_CHUNK',1)
    monkeypatch.setattr(cfg,'STATE_VISUAL_FEATURE_DROPOUT',0.)
    torch.manual_seed(17)
    visual=VisualFeatures(learned=True).train()
    reference=copy.deepcopy(visual.cnn).train()
    inputs=torch.rand(2,3,480,640)
    seen=[]
    hook=visual.cnn[0].register_forward_pre_hook(lambda m,args:seen.append(tuple(args[0].shape)))
    actual=visual(inputs)
    actual.sum().backward()
    hook.remove()
    expected=reference(inputs)
    expected.sum().backward()
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    assert all(shape == (1,3,480,640) for shape in seen)
    for a,b in zip(visual.cnn.parameters(),reference.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=2e-5,rtol=1e-4)


def test_frame_generation_chunk_gathers_preserve_causal_inputs(monkeypatch):
    monkeypatch.setattr(cfg,'FRAME_GENERATION_CHUNK',2)
    class Generator(torch.nn.Module):
        def forward(self,previous,anchor,imu,elapsed,prev_elapsed):
            assert len(anchor) <= 2
            return anchor+.1*previous+imu[:,0,None,None,None]+elapsed[:,:,None,None]+prev_elapsed[:,:,None,None]
    b,t=2,7
    cams=torch.rand(b,3,3,12,16)
    anchor=torch.tensor([[0,0,0,1,1,1,2]]).repeat(b,1)
    ticks=torch.tensor([[0,0,0,3,3,3,6]]).repeat(b,1)
    previous=(anchor-1).clamp_min(0)
    imu=torch.rand(b,t,6)
    batch=dict(cam_images=cams,imu=imu,anchor_img_idx=anchor,anchor_tick=ticks,
               prev_img_idx=previous,prev_span=(anchor-previous).float()*.015,
               observed=torch.arange(t)[None] == ticks,dt=torch.full((b,),.005))
    actual=generate_200hz_frames(Generator(),batch,'cpu')
    for i in range(b):
        for k in range(t):
            a=cams[i,anchor[i,k]]
            if batch['observed'][i,k]:
                expected=a
            else:
                avg=imu[i,ticks[i,k]+1:k+1,0].mean()
                elapsed=(k-ticks[i,k])*.005
                expected=a+.1*cams[i,previous[i,k]]+avg+elapsed+batch['prev_span'][i,k]
            torch.testing.assert_close(actual[i,k],expected)


def test_full_resolution_flow_preserves_output_size():
    torch.set_num_threads(1)
    model=GenerativeFlowPredictor(base=4,max_flow_pixels=160).eval()
    image=torch.ones(1,3,480,640)*.5
    with torch.no_grad():
        result=model(image,image,torch.zeros(1,6),torch.tensor([[.025]]),torch.tensor([[.05]]))
    assert result.shape == (1,3,480,640)
    torch.testing.assert_close(result,image,atol=1e-5,rtol=1e-5)


def test_full_resolution_export_requests_and_encodes_full_frames(tmp_path):
    import onnx
    torch.set_num_threads(1)
    settings=dict(backbone='gru',frame_mode='repeat',learned_visual=True,
                  hidden=16,width=640,height=480,mamba_layers=0,max_flow_pixels=160.)
    frame,state=make_models(settings)
    output=export_models(frame,state,settings,tmp_path/'teacher')
    metadata=json.loads((output/'deployment.json').read_text())
    assert metadata['frame_width'] == metadata['width'] == 640
    assert metadata['height'] == 480
    assert 'frame_width = 640' in (output/'solution.py').read_text()
    graph=onnx.load(str(output/'visual.onnx'))
    shape=[d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
    assert shape == [1,3,480,640]


def test_full_resolution_teacher_can_train_without_export(tmp_path):
    import os
    import subprocess
    data=tmp_path/'data'; data.mkdir()
    for scene in (0,1):
        n=41
        truth=np.zeros((n,18),np.float32); truth[:,3:12]=np.eye(3).ravel()
        imu=np.zeros((n,6),np.float32); imu[1:,2]=9.81
        np.savez(data/f'rollout_s{scene:03d}_e000.npz',
            camera=np.full((9,480,640,3),127,np.uint8),camera_ticks=np.arange(0,n,5),
            state=truth,imu=imu,dt=np.float32(.005),frame_stride=5,scene_id=scene)
    config=tmp_path/'tiny-teacher.json'
    config.write_text(json.dumps(dict(STUDENT_HIDDEN=16,FRAME_GENERATOR_BASE=4,WINDOW=11,USE_HYBRID_SSM_TRANSFORMER=False,WANDB_MODE='disabled',
        MAX_FRAME_TRAIN_PAIRS=1,MAX_FRAME_VAL_PAIRS=1)))
    output=tmp_path/'checkpoints'; solution=tmp_path/'not-exported'
    result=subprocess.run([sys.executable,str(HERE/'train.py'),str(data),'--config',str(config),
        '--epochs-frame','1','--epochs-state','1','--device','cpu','--skip-export',
        '--out',str(output),'--solution-out',str(solution),'--no-ppo'],
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
        capture_output=True,text=True,timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    assert (output/'frame_student.pt').is_file() and (output/'state_student.pt').is_file()
    assert not solution.exists()
    metadata=json.loads((output/'student_config.json').read_text())
    assert (metadata['width'],metadata['height'],metadata['frame_width']) == (640,480,640)
    assert metadata['max_flow_pixels'] == 160.


def test_learned_patch_projections_consume_every_full_resolution_pixel():
    torch.set_num_threads(1)
    flow=GenerativeFlowPredictor(base=4,input_stride=8).enc1[0]
    visual=VisualFeatures(learned=True,input_stride=8).cnn[0]
    for layer,channels in ((flow,11),(visual,3)):
        with torch.no_grad():
            layer.weight.fill_(1.)
        image=torch.zeros(1,channels,480,640,requires_grad=True)
        layer(image).sum().backward()
        assert image.grad.shape == image.shape
        assert torch.all(image.grad > 0)


def test_new_patch_stems_require_sl_before_ppo_only(tmp_path):
    from train import load_weights,save_weights
    old=GenerativeFlowPredictor(base=4,input_stride=1)
    new=GenerativeFlowPredictor(base=4,input_stride=8)
    path=tmp_path/'frame.pt'
    save_weights(old.state_dict(),path)
    with pytest.raises(ValueError,match='run supervised training'):
        load_weights(new,path,'cpu')
    stem=new.enc1[0].weight.detach().clone()
    load_weights(new,path,'cpu',allow_new_patch_stems=True)
    torch.testing.assert_close(new.enc1[0].weight,stem)
    torch.testing.assert_close(new.enc2[0].weight,old.enc2[0].weight)
    assert path.is_file()


@pytest.mark.parametrize('component',['frame','state'])
def test_warm_start_changed_width_requires_fresh_training(tmp_path,component):
    from models import StateStudent
    from train import load_weights,save_weights
    if component == 'frame':
        old,new=GenerativeFlowPredictor(base=4),GenerativeFlowPredictor(base=5)
    else:
        old,new=StateStudent(hidden=16),StateStudent(hidden=32)
    path=tmp_path/'weights.pt'
    save_weights(old.state_dict(),path)
    with pytest.raises(ValueError,match='fresh training run without --from-checkpoints'):
        load_weights(new,path,'cpu',allow_new_patch_stems=True)
