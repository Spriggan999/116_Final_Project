"""Lossless fast recording writes, disk caps and collection telemetry."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import collect
from collect import write_episode
from performance import record_epoch,pipeline_log
from streaming import RollingBatches
from test_streaming_ppo import stream_cfg


def arrays():
    camera=np.random.default_rng(17).integers(0,256,(9,8,12,3),dtype=np.uint8)
    return dict(camera=camera[:,:,::-1,:],camera_ticks=np.arange(0,41,5,dtype=np.int32),
                state=np.asfortranarray(np.arange(41*18,dtype=np.float32).reshape(41,18)),
                imu=np.zeros((41,6),np.float32),dt=np.array(.005,np.float32),
                frame_stride=np.array(5,np.int32),scene_id=np.array(200,np.int64))


@pytest.mark.parametrize('level',[0,1,6,9])
def test_npz_levels_roundtrip_exact_arrays_and_preserve_legacy_readability(tmp_path,level):
    values=arrays();path=tmp_path/'recording.npz';timing={}
    assert write_episode(path,values,compression_level=level,timing=timing)
    with np.load(path,allow_pickle=False) as data:
        assert set(data.files)==set(values)
        for name,array in values.items():
            assert data[name].dtype==array.dtype
            np.testing.assert_array_equal(data[name],array)
    assert timing['compression_seconds']>0 and timing['write_seconds']>=0
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('level',[1,6])
def test_fast_write_accounts_for_metadata_before_touching_disk(tmp_path,level):
    values=arrays();probe=tmp_path/'probe';probe.mkdir()
    reference=probe/'test.npz'
    assert write_episode(reference,values,compression_level=level)
    directory=tmp_path/'bounded';directory.mkdir()
    metadata=directory/'manifest.json';metadata.write_text('[]\n')
    limit=reference.stat().st_size+metadata.stat().st_size
    target=directory/'recording.npz'
    assert not write_episode(target,values,limit-1,compression_level=level)
    assert list(directory.iterdir())==[metadata]
    assert write_episode(target,values,limit,compression_level=level)
    assert sum(p.stat().st_size for p in directory.iterdir())==limit


def test_serialization_failure_keeps_previous_recording_atomic(tmp_path,monkeypatch):
    target=tmp_path/'recording.npz'
    assert write_episode(target,arrays())
    before=target.read_bytes()
    def fail(*args,**kwargs):raise OSError('serialization failed')
    monkeypatch.setattr(np.lib.format,'write_array',fail)
    with pytest.raises(OSError,match='serialization failed'):
        write_episode(target,arrays(),compression_level=1)
    assert target.read_bytes()==before and not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('level',[-1,10,1.5,True])
def test_invalid_levels_rejected_before_starting_collection(tmp_path,level):
    with pytest.raises(ValueError,match='compression_level'):
        write_episode(tmp_path/'bad.npz',arrays(),compression_level=level)
    with pytest.raises(ValueError,match='COLLECT_COMPRESSION_LEVEL'):
        RollingBatches(tmp_path,stream_cfg(COLLECT_COMPRESSION_LEVEL=level))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('custom',[False,True])
def test_only_builtin_collector_receives_compression_flag(tmp_path,monkeypatch,custom):
    import streaming
    commands=[]
    class Child:
        returncode=0
        def __init__(self,command,**kwargs):commands.append(command)
        def poll(self):return 0
    monkeypatch.setattr(streaming.subprocess,'Popen',Child)
    batches=RollingBatches(tmp_path,stream_cfg(COLLECT_COMPRESSION_LEVEL=1),
                           collector=tmp_path/'custom.py' if custom else None)
    spec=dict(scene_offset=0,scenes=1,episodes=2,seed=17,steps=41,width=12,height=8,
              max_bytes=100_000,compression_level=1)
    try:batches._subprocess(tmp_path,spec,batches.stop)
    finally:batches.close()
    command=commands[0]
    assert ('--compression-level' in command)==(not custom)
    if not custom:assert command[command.index('--compression-level')+1]=='1'


@pytest.mark.parametrize('level',[None,1])
def test_collector_emits_io_timings_and_manifest_compression(tmp_path,monkeypatch,level):
    from test_performance import Flights
    class Env(Flights):
        def close(self):pass
    monkeypatch.setitem(sys.modules,'hyperdrone',SimpleNamespace(Config=lambda **kwargs:kwargs,
        HyperDrone=lambda *args,**kwargs:Env()))
    monkeypatch.setattr(collect,'scene_references',lambda catalog:[('procthor',0,0,'scene')])
    import scene_assets
    monkeypatch.setattr(scene_assets,'prepare_scene',lambda reference,directory:(reference,0))
    flags=['collect.py','--out',str(tmp_path),'--scene-catalog','procthor',
        '--scenes','1','--episodes','2','--steps','41']
    if level is not None:flags+=['--compression-level',str(level)]
    monkeypatch.setattr(sys,'argv',flags)
    collect.main()
    expected=6 if level is None else level
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    stats=json.loads((tmp_path/'collection_stats.json').read_text())
    assert len(manifest)==stats['episodes']==2
    assert all(row['compression_level']==expected for row in manifest)
    assert stats['compression_level']==expected and stats['compression_seconds']>0 and stats['write_seconds']>=0
    assert stats['recording_bytes']==sum(p.stat().st_size for p in tmp_path.glob('rollout*.npz'))
    data=SimpleNamespace(batch_id='batch',rolling=SimpleNamespace(
        current=tmp_path,last_wait_seconds=0.,passes_on_current=1))
    row=record_epoch(tmp_path,'state',1,dict(seconds=1.,samples=2),data)
    metrics=pipeline_log(row)
    assert metrics['pipeline/collection_compression_seconds']==stats['compression_seconds']
    assert metrics['pipeline/collection_write_seconds']==stats['write_seconds']
    assert metrics['pipeline/collection_compression_level']==expected
    assert metrics['pipeline/collection_recording_bytes']==stats['recording_bytes']


def test_prefix_benchmark_reads_exact_subset_and_mixes_catalogs(tmp_path):
    from verification.benchmark_recording_io import recording_prefix,select_recordings
    values=arrays()
    paths=[tmp_path/f'rollout_s{i:03d}_e000.npz' for i in (0,1,200,201)]
    for path in paths:assert write_episode(path,values,compression_level=1)
    rows=[dict(file=str(path),scene_catalog='procthor' if i<2 else 'hssd') for i,path in enumerate(paths)]
    (tmp_path/'manifest.json').write_text(json.dumps(rows))
    assert select_recordings(tmp_path,4)==[paths[0],paths[2],paths[1],paths[3]]
    subset=recording_prefix(paths[0],5)
    np.testing.assert_array_equal(subset['camera'],values['camera'][:5])
    np.testing.assert_array_equal(subset['camera_ticks'],values['camera_ticks'][:5])
    np.testing.assert_array_equal(subset['state'],values['state'][:21])
    assert len(subset['imu'])==21
