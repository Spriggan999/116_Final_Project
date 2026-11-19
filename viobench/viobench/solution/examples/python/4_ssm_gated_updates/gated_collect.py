"""Memory-bounded 20 Hz recording and disk-streamed NPZ compression."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import zipfile
import legacy
import numpy as np
import collect as recorder
from resource_limits import Limits, ResourceLimitError, available_memory, disk_check, MIB

LIMITS = Limits()


def collect_round(env, steps):
    count, stride, dt = int(env.total_instances), 10, float(env.dt)
    if abs(dt-.005) > 1e-6:
        raise ValueError('The gated collector expects 200 Hz IMU')
    env.reset();env.render()
    parameters = env.instance_parameters() if hasattr(env,'instance_parameters') else None
    positions=[]
    for index in range(count):
        cameras=parameters[index]['cameras'] if parameters is not None else {}
        front=cameras.get('front') or next(iter(cameras.values()),None)
        positions.append(np.asarray(front['mount']['pose']['position'] if front else [0,0,0],np.float32))
    initial=recorder.instance_rows(env.observe_privileged(),count,18)
    recorder.validate_truth(initial)
    frames=recorder.instance_frames(env.frames(),count)
    rows=[dict(state=[initial[i,:18].copy()],imu=[np.zeros(6,np.float32)],camera=[frames[i]],ticks=[0]) for i in range(count)]
    active=np.ones(count,dtype=bool)
    for tick in range(1,steps):
        env.step()
        active &= ~np.asarray(env.terminated(),dtype=bool).reshape(count)
        if not active.any():
            break
        imu=recorder.instance_rows(env.observe_imu(),count,6,active)
        truth=recorder.instance_rows(env.observe_privileged(),count,18,active)
        recorder.validate_truth(truth[active])
        for i in np.flatnonzero(active):
            rows[i]['imu'].append(imu[i,:6].copy());rows[i]['state'].append(truth[i,:18].copy())
        if tick % stride == 0:
            if available_memory() < LIMITS.min_ram_free_mib*MIB:
                raise ResourceLimitError('Collector stopped because available RAM is below its reserve')
            env.render()
            frames=recorder.instance_frames(env.frames(),count,active)
            for i in np.flatnonzero(active):
                rows[i]['camera'].append(frames[i]);rows[i]['ticks'].append(tick)
    result=[]
    for index,row in enumerate(rows):
        if len(row['state'])<max(25,stride*4):
            result.append(None);continue
        # Check before stacking, which briefly duplicates the frame list.
        planned=sum(frame.nbytes for frame in row['camera'])
        if available_memory()-planned < LIMITS.min_ram_free_mib*MIB:
            raise ResourceLimitError('Not enough RAM headroom to assemble the collected flight')
        result.append(dict(camera=np.stack(row['camera']),camera_ticks=np.asarray(row['ticks'],np.int32),
            imu=np.stack(row['imu']),state=np.stack(row['state']),dt=np.array(dt,np.float32),
            frame_stride=np.array(stride,np.int32),camera_position_in_body=positions[index]))
        row.clear()
    return result


class FileBudgetExceeded(Exception):
    pass


class BoundedFile:
    def __init__(self, stream, maximum, path):
        self.stream,self.maximum,self.path=stream,maximum,path

    def __getattr__(self, name):
        return getattr(self.stream,name)

    def write(self, value):
        if self.maximum is not None and self.stream.tell()+len(value)>self.maximum:
            raise FileBudgetExceeded()
        disk_check(self.path,LIMITS,planned=len(value))
        return self.stream.write(value)


def write_episode(path, arrays, max_bytes=None, *, compression_level=1, timing=None):
    """Never construct an entire compressed episode in a BytesIO RAM buffer."""
    used=sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())
    remaining=None if max_bytes is None else max_bytes-used
    if remaining is not None and remaining<=0:
        return False
    temporary=path.with_suffix('.npz.tmp')
    started=time.perf_counter()
    try:
        with temporary.open('w+b') as stream:
            with zipfile.ZipFile(BoundedFile(stream,remaining,path),'w',compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=compression_level) as archive:
                for name,array in arrays.items():
                    with archive.open(name+'.npy','w',force_zip64=True) as member:
                        np.lib.format.write_array(member,np.asanyarray(array),allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        return True
    except FileBudgetExceeded:
        return False
    finally:
        temporary.unlink(missing_ok=True)
        if timing is not None:
            timing['compression_seconds']=timing.get('compression_seconds',0)+time.perf_counter()-started


def main():
    global LIMITS
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out',type=Path,required=True)
    args,_=parser.parse_known_args()
    spec=json.loads((args.out/'collection_spec.json').read_text())
    LIMITS=Limits(**spec.pop('resource_limits'))
    # RollingBatches passes common flags; its spec supplies these custom ones.
    sys.argv += ['--instances',str(spec['instances']),'--compression-level',str(spec['compression_level']),
                 '--scene-catalog',spec['scene_catalog']]
    recorder.collect_round=collect_round
    recorder.write_episode=write_episode
    recorder.main()
    manifest=args.out/'manifest.json'
    records=json.loads(manifest.read_text())
    for record in records:
        record['native_frame_stride']=record.get('frame_stride')
        record['frame_stride']=10
        record['stored_camera_hz']=20
    manifest.write_text(json.dumps(records,indent=2))


if __name__ == '__main__':
    main()
