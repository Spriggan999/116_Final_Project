"""CPU NPZ I/O on bounded real-recording prefixes; excludes collection/training."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from collect import write_episode


def recording_prefix(path,frames):
    """Decompress only a camera prefix, avoiding a full-flight RAM allocation."""
    with zipfile.ZipFile(path) as archive,archive.open('camera.npy') as member:
        version=np.lib.format.read_magic(member)
        readers={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}
        if version not in readers:raise ValueError(f'Unsupported NPY header {version}')
        shape,fortran,dtype=readers[version](member)
        if fortran or len(shape)!=4 or dtype!=np.uint8:
            raise ValueError('Benchmark needs C-contiguous uint8 camera recordings')
        count=min(frames,shape[0])
        raw=member.read(count*int(np.prod(shape[1:]))*dtype.itemsize)
        camera=np.frombuffer(raw,dtype=dtype).reshape((count,*shape[1:]))
    with np.load(path,allow_pickle=False) as data:
        arrays={name:data[name] for name in data.files if name!='camera'}
    arrays['camera']=camera
    arrays['camera_ticks']=arrays['camera_ticks'][:count]
    ticks=int(arrays['camera_ticks'][-1])+1
    for name in ('state','imu'):arrays[name]=arrays[name][:ticks]
    return arrays


def select_recordings(directory,count):
    manifest=directory/'manifest.json'
    if manifest.exists():
        groups={}
        for row in json.loads(manifest.read_text()):
            path=directory/Path(row['file']).name
            if path.exists():groups.setdefault(row.get('scene_catalog','unknown'),[]).append(path)
        if groups:
            return [group[i] for i in range(max(map(len,groups.values())))
                    for group in groups.values() if i<len(group)][:count]
    return sorted(directory.glob('rollout_s*_e*.npz'))[:count]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=32)
    parser.add_argument('--samples',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    if min(args.frames,args.samples,args.repeats)<1:parser.error('counts must be positive')
    paths=select_recordings(args.recordings,args.samples)
    if not paths:parser.error('no completed recordings found')
    rows=[]
    with tempfile.TemporaryDirectory(prefix='hyperdrone-recording-io-') as temporary:
        root=Path(temporary)
        for path in paths:
            arrays=recording_prefix(path,args.frames)
            levels={1:[],6:[]}
            for repeat in range(args.repeats):
                # Alternate order to reduce bias from cold execution.
                for level in ((6,1) if repeat%2==0 else (1,6)):
                    destination=root/f'level-{level}.npz'
                    timing={};started=time.perf_counter()
                    assert write_episode(destination,arrays,compression_level=level,timing=timing)
                    seconds=time.perf_counter()-started
                    started=time.perf_counter()
                    with np.load(destination,allow_pickle=False) as data:
                        actual={name:data[name] for name in data.files}
                    read_seconds=time.perf_counter()-started
                    assert set(actual)==set(arrays)
                    for name,value in arrays.items():
                        assert actual[name].dtype==value.dtype and actual[name].shape==value.shape
                        assert np.array_equal(actual[name],value),f'Round-trip mismatch: {name}'
                    levels[level].append(dict(seconds=seconds,read_seconds=read_seconds,
                                              bytes=destination.stat().st_size,**timing))
            row=dict(file=path.name,frames=len(arrays['camera']),
                     raw_bytes=sum(a.nbytes for a in arrays.values()),
                     levels={str(level):{key:statistics.median(r[key] for r in values)
                                        for key in values[0]} for level,values in levels.items()})
            rows.append(row)
    old={key:sum(row['levels']['6'][key] for row in rows) for key in ('seconds','compression_seconds','bytes')}
    new={key:sum(row['levels']['1'][key] for row in rows) for key in old}
    result=dict(scope=__doc__,frames_per_sample=args.frames,repeats=args.repeats,
                samples=rows,roundtrip_exact=True,
                write_path_speedup=old['seconds']/new['seconds'],
                compression_speedup=old['compression_seconds']/new['compression_seconds'],
                size_ratio=new['bytes']/old['bytes'])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='samples'},indent=2))


if __name__=='__main__':main()
