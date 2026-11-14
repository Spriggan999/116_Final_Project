"""Direct visual prediction: gradient targets, causality and deployment parity."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import config as cfg
from data import generate_200hz_embeddings,state_input_frames
from embedding_training import embedding_objective
from export import make_models,normalize_settings
from models import EmbeddingPredictor


def test_prediction_learns_online_encoder_and_keeps_ema_targets_frozen():
    torch.set_num_threads(1);torch.manual_seed(17)
    model=EmbeddingPredictor(hidden=16,learned_visual=True,visual_input_stride=2,ema=.9).train()
    config=SimpleNamespace(DEVICE='cpu',EMBEDDING_STD_TARGET=.1,EMBEDDING_VARIANCE_WEIGHT=1.,EMBEDDING_COVARIANCE_WEIGHT=.01)
    batch={key:torch.rand(2,3,24,32) for key in ('previous','anchor','target')}
    batch.update(imu=torch.rand(2,6),elapsed=torch.full((2,1),.025),prev_elapsed=torch.full((2,1),.05))
    loss,metrics=embedding_objective(model,batch,config)
    assert torch.isfinite(loss) and all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum()>0 for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum()>0 for parameter in model.predictor.parameters())
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.target_encoder.parameters())
    before=[parameter.clone() for parameter in model.target_encoder.parameters()]
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-3)
    optimizer.step();model.update_target()
    for previous,target,online in zip(before,model.target_encoder.parameters(),model.encoder.parameters()):
        torch.testing.assert_close(target,previous*.9+online*.1,atol=1e-7,rtol=1e-6)
    assert not model.target_encoder.training
    assert model.encoder.feature_dropout==model.target_encoder.feature_dropout==0


def test_constant_embedding_has_nonzero_collapse_penalty():
    model=EmbeddingPredictor(hidden=8,learned_visual=False)
    for parameter in model.predictor.parameters():torch.nn.init.zeros_(parameter)
    config=SimpleNamespace(DEVICE='cpu',EMBEDDING_STD_TARGET=.1,EMBEDDING_VARIANCE_WEIGHT=1.,EMBEDDING_COVARIANCE_WEIGHT=.01)
    batch={key:torch.zeros(1,3,24,32) for key in ('previous','anchor','target')}
    batch.update(imu=torch.zeros(1,6),elapsed=torch.full((1,1),.025),prev_elapsed=torch.zeros(1,1))
    loss,metrics=embedding_objective(model,batch,config)
    assert metrics['prediction']==0 and metrics['variance']>0 and loss>0


def causal_batch():
    b,t=2,7
    anchors=torch.tensor([[0,0,0,1,1,1,2]]).repeat(b,1)
    ticks=anchors*3
    return dict(cam_images=torch.rand(b,3,3,12,16),imu=torch.rand(b,t,6),
                anchor_img_idx=anchors,anchor_tick=ticks,prev_img_idx=(anchors-1).clamp_min(0),
                prev_span=(anchors-(anchors-1).clamp_min(0)).float()*.015,
                observed=torch.arange(t)[None].expand(b,-1)==ticks,dt=torch.full((b,),.005))


def test_only_real_captures_are_encoded_and_missing_features_are_causal(monkeypatch):
    monkeypatch.setattr(cfg,'EMBEDDING_PREDICTION_CHUNK',2)
    monkeypatch.setattr(cfg,'USE_GENERATED_FRAMES_FOR_STATE',True)
    class Encoder(torch.nn.Module):
        def __init__(self):super().__init__();self.count=0
        def forward(self,images):
            self.count+=len(images)
            return images.mean((1,2,3))[:,None].expand(-1,48)
    class Predictor(EmbeddingPredictor):
        def __init__(self):
            super().__init__(hidden=8,learned_visual=False)
            self.encoder=Encoder();self.count=0
        def predict_embeddings(self,previous,anchor,imu,elapsed,prev_elapsed):
            assert len(anchor)<=2
            self.count+=len(anchor)
            return anchor+.1*previous+imu[:,:1]+elapsed+prev_elapsed
    model=Predictor().eval();batch=causal_batch()
    actual=state_input_frames(model,batch,'cpu')
    assert actual.shape==(2,7,48) and model.encoder.count==6 and model.count==8
    for b in range(2):
        for tick in range(7):
            anchor=batch['anchor_img_idx'][b,tick]
            expected=batch['cam_images'][b,anchor].mean()
            if not batch['observed'][b,tick]:
                start=batch['anchor_tick'][b,tick]
                expected=(expected+.1*batch['cam_images'][b,max(0,anchor-1)].mean()
                          +batch['imu'][b,start+1:tick+1,0].mean()+float(tick-start)*.005
                          +batch['prev_span'][b,tick])
            torch.testing.assert_close(actual[b,tick],expected.expand(48))
    batch['cam_images'][:,1:]+=10;batch['imu'][:,3:]+=10
    changed=generate_200hz_embeddings(model,batch,'cpu')
    torch.testing.assert_close(changed[:,:3],actual[:,:3],atol=0,rtol=0)
    monkeypatch.setattr(cfg,'USE_GENERATED_FRAMES_FOR_STATE',False)
    model.count=0
    repeated=state_input_frames(model,batch,'cpu')
    assert model.count==0
    torch.testing.assert_close(repeated[:,1:3],repeated[:,:1].expand(-1,2,-1))


@pytest.mark.parametrize('backbone',['gru','mamba2_hybrid'])
def test_embedding_state_sequence_matches_streaming(backbone):
    device='cpu'
    if backbone=='mamba2_hybrid':
        from mamba_helpers import require_mamba
        require_mamba();device='cuda'
    torch.set_num_threads(1);torch.manual_seed(17)
    frame,model=make_models(dict(frame_mode='embedding',backbone=backbone,learned_visual=False,
        embedding_hidden=8,hidden=64 if backbone=='mamba2_hybrid' else 16,width=32,height=24,mamba_layers=1,attention_window=4,
        hybrid_d_state=4,hybrid_head_dim=16,hybrid_attention_heads=4))
    with torch.no_grad():model.out.weight.normal_(0,.01)
    model.to(device)
    embeddings=torch.rand(2,7,48,device=device);imu=torch.rand(2,7,6,device=device)
    observed=causal_batch()['observed'].to(device);dt=torch.full((2,),.005,device=device);gravity=torch.tensor([[0.,0.,-9.81]],device=device).repeat(2,1)
    with torch.no_grad():
        sequence=model(embeddings,imu,observed,dt,gravity,True)
        state=model.stream_init(embeddings[:,0])
        for tick in range(7):
            output,state=model.stream_step(embeddings[:,tick],imu[:,tick],observed[:,tick],dt,gravity,state,initial=tick==0)
            for key in ('pos','vel','rot','omega','corrections'):
                torch.testing.assert_close(output[key],sequence[key][:,tick],atol=2e-6,rtol=2e-5)
    assert isinstance(model.visual,torch.nn.Identity)


def test_numpy_guest_embedding_predictor_and_real_capture_cadence(monkeypatch):
    from test_conversion import estimator,settings
    flags=settings(frame_mode='embedding',use_generated_frames=True,learned_visual=False,embedding_hidden=8,width=32,height=24)
    frame,model=make_models(flags)
    guest,_,_=estimator(monkeypatch,flags,state=model,frame=frame)
    rng=np.random.default_rng(17)
    guest.prev_real=rng.random(48,dtype=np.float32);guest.last_real=rng.random(48,dtype=np.float32)
    imu=rng.random(6,dtype=np.float32)
    with torch.no_grad():
        expected=frame.predict_embeddings(torch.from_numpy(guest.prev_real)[None],torch.from_numpy(guest.last_real)[None],
            torch.from_numpy(imu)[None],torch.tensor([[.025]]),torch.tensor([[.05]])).numpy()[0]
    np.testing.assert_allclose(guest.synthetic(imu,.025,.05),expected,atol=1e-7,rtol=1e-6)
    guest.last_real=None;guest.prev_real=None
    original=guest.features;encoded=[]
    def features(value):
        if value.ndim==3:encoded.append(value.shape)
        return original(value)
    guest.features=features
    for index in range(4):
        packet=np.zeros((10,7));packet[:,0]=index*.05+np.arange(1,11)*.005;packet[:,3]=9.81
        result=guest.step((index+1)*.05,rng.integers(0,256,(240,320,3),dtype=np.uint8),packet)
        assert np.isfinite(result).all()
    assert len(encoded)==4
    assert guest.last_real.shape==guest.prev_real.shape==(48,)


def test_legacy_settings_keep_rgb_and_embedding_repeat_is_explicit():
    assert normalize_settings(dict(frame_mode='flow'))['visual_prediction']=='rgb'
    settings=normalize_settings(dict(frame_mode='embedding',use_generated_frames=False))
    assert settings['visual_prediction']=='embedding' and not settings['use_generated_frames']


@pytest.mark.parametrize('mode',['embedding','rgb'])
def test_training_cli_defaults_to_embeddings_and_can_select_rgb(tmp_path,mode):
    import json,os,subprocess
    from test_training_stages import recordings
    data,config=recordings(tmp_path)
    output=tmp_path/'run'
    flags=[] if mode=='embedding' else ['--visual-prediction','rgb']
    result=subprocess.run([sys.executable,str(HERE/'train.py'),str(data),'--config',str(config),
        '--epochs-frame','1','--epochs-state','1','--device','cpu','--no-ppo','--skip-export','--out',str(output),*flags],
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    saved=json.loads((output/'student_config.json').read_text())
    assert saved['visual_prediction']==mode
    frame,state=make_models(saved)
    assert isinstance(frame,EmbeddingPredictor)==(mode=='embedding')
    frame.load_state_dict(torch.load(output/'frame_student.pt',weights_only=True))
    state.load_state_dict(torch.load(output/'state_student.pt',weights_only=True))
    assert (output/'embedding_metrics.jsonl').exists()==(mode=='embedding')


def test_old_checkpoint_defaults_to_rgb_and_rejects_representation_change(tmp_path):
    import json,os,subprocess
    source=tmp_path/'source';source.mkdir()
    (source/'training_config.json').write_text('{}')
    for name in ('frame_student.pt','state_student.pt'):(source/name).touch()
    base=[sys.executable,'-c','import train; train.run_training=lambda *a:None; train.main()',
          '--from-checkpoints',str(source),'--device','cpu']
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run(base+['--out',str(tmp_path/'legacy')],cwd=HERE,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads((tmp_path/'legacy/student_config.json').read_text())['visual_prediction']=='rgb'
    result=subprocess.run(base+['--out',str(tmp_path/'changed'),'--visual-prediction','embedding'],
        cwd=HERE,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'requires a fresh run' in result.stderr
    assert not (tmp_path/'changed').exists()


def test_embedding_patch_stem_changes_require_sl_and_preserve_both_new_stems(tmp_path):
    from train import load_weights
    old=EmbeddingPredictor(hidden=8,visual_input_stride=2)
    new=EmbeddingPredictor(hidden=8,visual_input_stride=8)
    path=tmp_path/'embedding.pt';torch.save(old.state_dict(),path)
    with pytest.raises(ValueError,match='run supervised training'):load_weights(new,path,'cpu')
    online=new.encoder.cnn[0].weight.detach().clone()
    target=new.target_encoder.cnn[0].weight.detach().clone()
    load_weights(new,path,'cpu',allow_new_patch_stems=True)
    torch.testing.assert_close(new.encoder.cnn[0].weight,online)
    torch.testing.assert_close(new.target_encoder.cnn[0].weight,target)
    torch.testing.assert_close(new.encoder.cnn[2].weight,old.encoder.cnn[2].weight)
    torch.testing.assert_close(new.predictor[0].weight,old.predictor[0].weight)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_embedding_training_and_state_precision(monkeypatch):
    if not torch.cuda.is_bf16_supported():pytest.skip('BF16 unavailable')
    from performance import adamw,amp_disabled
    monkeypatch.setattr(cfg,'USE_AMP',True)
    monkeypatch.setattr(cfg,'USE_GENERATED_FRAMES_FOR_STATE',True)
    model=EmbeddingPredictor(hidden=16,learned_visual=True,visual_input_stride=2).cuda().train()
    config=SimpleNamespace(DEVICE='cuda',EMBEDDING_STD_TARGET=.1,EMBEDDING_VARIANCE_WEIGHT=1.,EMBEDDING_COVARIANCE_WEIGHT=.01)
    batch={key:torch.rand(2,3,24,32) for key in ('previous','anchor','target')}
    batch.update(imu=torch.rand(2,6),elapsed=torch.full((2,1),.025),prev_elapsed=torch.full((2,1),.05))
    optimizer=adamw(p for p in model.parameters() if p.requires_grad)
    loss,_=embedding_objective(model,batch,config)
    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();model.update_target();model.eval()
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    from models import StateStudent
    state=StateStudent(hidden=16,visual_embeddings=True).cuda().eval()
    observations=causal_batch()
    embeddings=state_input_frames(model,observations,'cuda')
    assert embeddings.dtype==torch.float32 and embeddings.shape==(2,7,48)
    imu=observations['imu'].cuda();dt=observations['dt'].cuda();observed=observations['observed'].cuda()
    gravity=torch.tensor([[0.,0.,-9.81]],device='cuda').repeat(2,1)
    with torch.no_grad(),amp_disabled():result=state(embeddings,imu,observed,dt,gravity,True)
    assert all(value.dtype==torch.float32 and torch.isfinite(value).all() for value in result.values())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
