"""Long recordings stay compact and training/validation scenes remain disjoint."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from data import EpisodeStore,FramePairs,StateWindows,camera_values,state_input_frames
from models import EmbeddingPredictor
from streaming import RollingBatches
from scenes import scene_specs, scene_references
from test_streaming_ppo import stream_cfg


def test_recordings_keep_uint8_and_sampled_pixels_match_float_pipeline(tmp_path,monkeypatch):
    import config
    from distill import frame_inputs
    rng=np.random.default_rng(17)
    camera=rng.integers(0,256,(9,24,32,3),dtype=np.uint8)
    state=np.zeros((41,18),np.float32);state[:,3:12]=np.eye(3).ravel()
    imu=np.zeros((41,6),np.float32);imu[:,2]=9.81
    path=tmp_path/'rollout.npz'
    np.savez(path,camera=camera,camera_ticks=np.arange(0,41,5),state=state,imu=imu,
             dt=np.float32(.005),frame_stride=5,scene_id=0)
    store=EpisodeStore([path],24,32)
    assert store.entries[0]['cams'].dtype==torch.uint8
    assert store.entries[0]['cams'].untyped_storage().nbytes()==camera.nbytes
    pairs=FramePairs(store);sample=pairs[0];_,previous,anchor,target=pairs.items[0]
    for key,index in [('previous',previous),('anchor',anchor),('target',target)]:
        torch.testing.assert_close(sample[key],torch.from_numpy(camera[index]).permute(2,0,1).float()/255.,atol=0,rtol=0)
    window=StateWindows(store,length=21)[0]
    batch={key:value[None] for key,value in window.items()}
    floating=dict(batch,cam_images=camera_values(batch['cam_images']))
    model=EmbeddingPredictor(hidden=8,learned_visual=False).eval()
    monkeypatch.setattr(config,'USE_GENERATED_FRAMES_FOR_STATE',True)
    torch.testing.assert_close(state_input_frames(model,batch,'cpu'),state_input_frames(model,floating,'cpu'),atol=0,rtol=0)
    actual,_=frame_inputs(batch,'cpu');expected,_=frame_inputs(floating,'cpu')
    torch.testing.assert_close(actual['images'],expected['images'],atol=0,rtol=0)
    monkeypatch.setattr(config,'USE_GENERATED_FRAMES_FOR_STATE',False)
    torch.testing.assert_close(state_input_frames(None,batch,'cpu'),state_input_frames(None,floating,'cpu'),atol=0,rtol=0)


@pytest.mark.parametrize('batch_scenes',[2,4])
def test_all_training_scenes_rotate_without_touching_validation(tmp_path,batch_scenes):
    specs=[]
    def producer(directory,spec,stop):
        specs.append(dict(spec))
        rows=[]
        for scene in range(spec['scene_offset'],spec['scene_offset']+spec['scenes']):
            name=f'rollout_s{scene:03d}_e000.npz';(directory/name).write_bytes(b'recording')
            rows.append(dict(scene_index=scene))
        (directory/'manifest.json').write_text(json.dumps(rows))
    cfg=stream_cfg(NUM_SCENES=192,STREAM_SCENES=batch_scenes,STREAM_VALIDATION_SCENES=8,
                   STREAM_EPISODES_PER_SCENE=2,STREAM_VALIDATION_EPISODES=1,COLLECT_STEPS=2000)
    with RollingBatches(tmp_path/'managed',cfg,collect=producer) as batches:
        for _ in range(192//batch_scenes):
            batches.next_epoch();batches.future.result(timeout=5)
    validation=specs[0];training=specs[1:1+192//batch_scenes]
    heldout=set(range(validation['scene_offset'],validation['scene_offset']+validation['scenes']))
    seen={scene for spec in training for scene in range(spec['scene_offset'],spec['scene_offset']+spec['scenes'])}
    assert seen==set(range(192)) and heldout==set(range(192,200)) and seen.isdisjoint(heldout)
    assert all(spec['steps']==2000 for spec in specs)


def test_mixed_catalog_has_stable_ids_and_holds_out_both_sources():
    specs=scene_specs()
    training,validation=specs[:313],specs[313:]
    assert len(specs)==325 and {row[2] for row in specs}==set(range(325))
    assert sum(row[0]=='procthor' for row in training)==192
    assert sum(row[0]=='hssd' for row in training)==121
    assert sum(row[0]=='procthor' for row in validation)==8
    assert sum(row[0]=='hssd' for row in validation)==4
    assert set(training).isdisjoint(validation)
    assert {row[0] for row in training[:4]}=={'procthor','hssd'}
    assert all({row[0] for row in training[start:start+4]}=={'procthor','hssd'} for start in range(0,312,4))
    assert scene_specs('hssd')==[('hssd',i,200+i) for i in range(125)]
    assert scene_specs('procthor')==[('procthor',i,i) for i in range(200)]


def test_catalog_loader_uses_native_hssd_and_procthor_references(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules,'hyperlux.datasets.procthor',SimpleNamespace(
        Train200=lambda:SimpleNamespace(references=[f'proc-{i}' for i in range(200)])))
    monkeypatch.setitem(sys.modules,'hyperlux.datasets.hssd',SimpleNamespace(
        Train125=lambda:SimpleNamespace(references=[f'hssd-{i}' for i in range(125)])))
    rows=scene_references()
    assert rows[1]==('hssd',0,200,'hssd-0')
    assert rows[313]==('procthor',192,192,'proc-192')
    assert rows[-1]==('hssd',124,324,'hssd-124')


def test_mixed_rotation_covers_remainder_batch_without_validation_leakage(tmp_path):
    specs=[]
    def producer(directory,spec,stop):
        specs.append(dict(spec))
        (directory/'rollout_s000_e000.npz').write_bytes(b'recording')
    with RollingBatches(tmp_path/'managed',stream_cfg(NUM_SCENES=313,STREAM_SCENES=4,
            STREAM_VALIDATION_SCENES=12,SCENE_CATALOG='mixed'),collect=producer) as batches:
        for _ in range(79):
            batches.next_epoch();batches.future.result(timeout=5)
    training=specs[1:80]
    seen=[i for spec in training for i in range(spec['scene_offset'],spec['scene_offset']+spec['scenes'])]
    assert sorted(seen)==list(range(313))
    assert training[-1]['scene_offset']==312 and training[-1]['scenes']==1
    assert specs[0]['scene_offset']==313 and specs[0]['scenes']==12
    assert all(spec['scene_catalog']=='mixed' for spec in specs)


def test_incomplete_heldout_coverage_is_rejected_and_cleaned(tmp_path):
    def producer(directory,spec,stop):
        (directory/'rollout_s002_e000.npz').write_bytes(b'recording')
        (directory/'manifest.json').write_text(json.dumps([dict(scene_index=spec['scene_offset'])]))
    with pytest.raises(RuntimeError,match='Validation collection missed scenes'):
        with RollingBatches(tmp_path/'managed',stream_cfg(STREAM_VALIDATION_SCENES=3),collect=producer):pass
    assert not list((tmp_path/'managed').iterdir())


@pytest.mark.parametrize('scenes',[0,-1,1.5])
def test_invalid_validation_scene_count_fails_before_collection(tmp_path,scenes):
    with pytest.raises(ValueError,match='STREAM_VALIDATION_SCENES must be a positive integer'):
        RollingBatches(tmp_path,stream_cfg(STREAM_VALIDATION_SCENES=scenes))
