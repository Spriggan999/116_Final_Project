"""Standalone HyperDrone recorder; run as a separate process, as in Colab."""
import argparse
from pathlib import Path
from scenes import CATALOGS, scene_references, scene_specs


def write_episode(path, arrays, max_bytes=None, *, compression_level=6, timing=None):
    """Compress in memory, enforcing the cap before writing a new recording."""
    import io,time,zipfile
    import numpy as np
    if type(compression_level) is not int or not 0<=compression_level<=9:
        raise ValueError('compression_level must be an integer from 0 to 9')
    buffer = io.BytesIO()
    started=time.perf_counter()
    # Standard DEFLATE NPZ stays readable by np.load, including older Python.
    # Streaming NPY members avoids another full camera-array byte copy.
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=compression_level) as archive:
        for name,array in arrays.items():
            with archive.open(name+'.npy','w',force_zip64=True) as member:
                np.lib.format.write_array(member,np.asanyarray(array),allow_pickle=False)
    if timing is not None:
        timing['compression_seconds']=timing.get('compression_seconds',0.)+time.perf_counter()-started
    size = buffer.tell()
    used = sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())
    if max_bytes is not None and used + size > max_bytes:
        return False
    temporary = path.with_suffix('.npz.tmp')
    started=time.perf_counter()
    try:
        temporary.write_bytes(buffer.getbuffer())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        if timing is not None:
            timing['write_seconds']=timing.get('write_seconds',0.)+time.perf_counter()-started
    return True


def instance_rows(raw, count, channels,active=None):
    import numpy as np
    x=np.asarray(raw,dtype=np.float32)
    if count==1:x=x.reshape(1,-1)
    elif x.ndim<2 or x.shape[0]!=count:
        raise ValueError(f'Expected {count} instance rows, got {x.shape}')
    else:x=x.reshape(count,-1)
    if x.shape[1]<channels or not np.isfinite(x if active is None else x[active]).all():
        raise ValueError(f'Invalid batched observation shape/values: {x.shape}')
    return x


def instance_frames(raw,count,active=None):
    import numpy as np
    x=np.asarray(raw)
    if count==1 and x.ndim==3:x=x[None]
    if x.shape[0]!=count:raise ValueError(f'Expected {count} frame instances, got {x.shape}')
    result=[]
    for slot in range(count):
        if active is not None and not active[slot]:result.append(None);continue
        image=x[slot]
        while image.ndim>3:image=image[0]  # first camera, preserving the flight axis
        if image.ndim!=3:raise ValueError(f'Unrecognized frame shape: {x.shape}')
        if image.shape[0] in (3,4) and image.shape[-1] not in (3,4):image=image.transpose(1,2,0)
        if image.shape[-1] not in (3,4):raise ValueError(f'Unrecognized frame shape: {x.shape}')
        image=image[...,:3]
        if image.dtype!=np.uint8:
            if not np.isfinite(image).all():raise ValueError('Nonfinite RGB')
            if image.min()>=0 and image.max()<=1.01:image=image*255
            image=np.clip(np.rint(image),0,255).astype(np.uint8)
        result.append(image.copy())
    return result


def validate_truth(states):
    import numpy as np
    R=states[:,3:12].reshape(-1,3,3)
    if not np.allclose(R.transpose(0,2,1)@R,np.eye(3),atol=.12) or not np.allclose(np.linalg.det(R),1,atol=.12):
        raise ValueError('Privileged channels must begin with [position, R(row-major), velocity, omega]')


def collect_round(env,steps):
    """One vectorized reset/flight round; crashed slots stop independently."""
    import numpy as np
    count=int(env.total_instances);stride=int(env.frame_stride);dt=float(env.dt)
    env.reset();env.render()
    # Preserve each reset's camera lever arm for camera-centre trajectory scoring.
    parameters=env.instance_parameters() if hasattr(env,'instance_parameters') else None
    camera_positions=[]
    for i in range(count):
        cameras=parameters[i]['cameras'] if parameters is not None else {}
        front=cameras.get('front') or next(iter(cameras.values()),None)
        camera_positions.append(np.asarray(front['mount']['pose']['position'] if front else [0.,0.,0.],np.float32))
    initial=instance_rows(env.observe_privileged(),count,18);validate_truth(initial)
    frames=instance_frames(env.frames(),count)
    records=[dict(state=[initial[i].copy()],imu=[np.zeros(6,np.float32)],camera=[frames[i]],camera_ticks=[0]) for i in range(count)]
    active=np.ones(count,dtype=bool)
    # The raw simulator's episode_step_limit is informational. Unlike its Gym
    # wrapper, it does not truncate at that limit; terminated() still ends crashes.
    for tick in range(1,steps):
        env.step()
        active &= ~np.asarray(env.terminated(),dtype=bool).reshape(count)
        if not active.any():break
        imu=instance_rows(env.observe_imu(),count,6,active)
        state=instance_rows(env.observe_privileged(),count,18,active)
        validate_truth(state[active])
        for i in np.flatnonzero(active):
            records[i]['imu'].append(imu[i,:6].copy());records[i]['state'].append(state[i].copy())
        if tick%stride==0:
            env.render();frames=instance_frames(env.frames(),count,active)
            for i in np.flatnonzero(active):
                records[i]['camera'].append(frames[i]);records[i]['camera_ticks'].append(tick)
    result=[]
    for i,row in enumerate(records):
        if len(row['state'])<max(25,stride*4):result.append(None);continue
        result.append(dict(camera=np.stack(row['camera']),camera_ticks=np.asarray(row['camera_ticks'],np.int32),
            imu=np.stack(row['imu']),state=np.stack(row['state']),dt=np.array(dt,np.float32),frame_stride=np.array(stride,np.int32),
            camera_position_in_body=camera_positions[i]))
    return result


def main():
    import time,json,faulthandler,tempfile
    from scene_assets import prepare_scene
    parser=argparse.ArgumentParser(description='Collect vectorized HyperDrone RGB/IMU/state rollouts.')
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'data')
    parser.add_argument('--scene-catalog',choices=CATALOGS,default='mixed',
                        help='mixed: 313 training scenes followed by 12 held-out scenes')
    parser.add_argument('--scenes',type=int,default=313)
    parser.add_argument('--episodes',type=int,default=2)
    parser.add_argument('--instances',type=int,default=2,help='parallel flights per scene; default 2')
    parser.add_argument('--compression-level',type=int,choices=range(10),default=6,
                        help='lossless NPZ DEFLATE level; streaming passes config (default 1); standalone default 6')
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--scene-offset',type=int,default=0)
    parser.add_argument('--width',type=int,default=640);parser.add_argument('--height',type=int,default=480)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--max-output-mib',type=float,default=3072,
                        help='recording directory budget; default 3072 MiB; may stop before every scene')
    args=parser.parse_args()
    if min(args.scenes,args.episodes,args.instances,args.steps,args.width,args.height)<1 or args.scene_offset<0:
        parser.error('counts/dimensions must be positive and scene offset nonnegative')
    if args.max_output_mib is not None and args.max_output_mib<=0:parser.error('--max-output-mib must be positive')
    available=len(scene_specs(args.scene_catalog))
    if args.scene_offset+args.scenes>available:
        parser.error(f'Requested scene range exceeds the {available} available {args.scene_catalog} scenes')
    from hyperdrone import Config,HyperDrone
    import numpy as np
    faulthandler.enable(all_threads=True)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    refs=scene_references(args.scene_catalog)
    refs=refs[args.scene_offset:args.scene_offset+args.scenes]
    manifest=[];full=False;started=time.perf_counter();captures=0
    build_seconds=0.;rollout_seconds=0.;simulated_captures=0
    write_timing=dict(compression_seconds=0.,write_seconds=0.)
    repaired_empty_primitives=0
    instances=min(args.instances,args.episodes)
    for scene_index,(catalog,catalog_index,scene_id,scene) in enumerate(refs):
        env=None
        scratch=tempfile.TemporaryDirectory(prefix='hyperdrone-scene-')
        try:
            config=Config(num_worlds=1,instances_per_world=instances,width=args.width,height=args.height,
                          preset='x500_fpv_imu',task='visual_inertial_localization')
            print(f'Build scene {args.scene_offset+scene_index} ({catalog}:{catalog_index}): '
                  f'{instances} parallel flights',flush=True)
            build_start=time.perf_counter()
            scene_path,removed=prepare_scene(scene,scratch.name)
            repaired_empty_primitives+=removed
            if removed:print(f'Scene cleanup: removed {removed} empty primitives from {catalog}:{catalog_index}; valid geometry retained.',flush=True)
            env=HyperDrone([scene_path],config=config,seed=args.seed+100*scene_index)
            build_seconds+=time.perf_counter()-build_start
            if abs(float(env.dt)-.005)>.001:raise ValueError(f'Unexpected IMU period {env.dt}')
            print(f'Flight limit: {args.steps} samples; '
                  f'native informational limit {int(env.episode_step_limit)}; dt={float(env.dt):.3f}s',flush=True)
            info=dict(dt=float(env.dt),frame_stride=int(env.frame_stride),scene_index=args.scene_offset+scene_index,
                scene_id=scene_id,scene_catalog=catalog,catalog_index=catalog_index,
                instances_per_world=instances,seed=args.seed+100*scene_index,
                imu_layout=str(getattr(env,'observation_layout_imu','')),
                observation_layout=str(getattr(env,'observation_layout','')),
                parameters=str(getattr(env,'parameters','')),backend=str(getattr(env,'render_backend','')),
                requested_steps=args.steps,episode_step_limit=int(env.episode_step_limit))
            info['compression_level']=args.compression_level
            info['removed_empty_primitives']=removed
            for first in range(0,args.episodes,instances):
                rollout_start=time.perf_counter()
                rows=collect_round(env,args.steps)
                rollout_seconds+=time.perf_counter()-rollout_start
                simulated_captures+=sum(len(row['camera']) for row in rows if row is not None)
                for slot,arrays in enumerate(rows[:args.episodes-first]):
                    if arrays is None:
                        print(f'Skipping short flight scene={scene_index}, episode={first+slot}',flush=True);continue
                    episode=first+slot
                    arrays['scene_id']=np.array(scene_id)
                    path=out/f'rollout_s{scene_id:03d}_e{episode:03d}.npz'
                    record=dict(file=str(path),n_ticks=len(arrays['state']),n_cameras=len(arrays['camera']),
                                instance_slot=slot,round=first//instances,**info)
                    text=json.dumps(manifest+[record],indent=2)
                    limit=None if args.max_output_mib is None else int(args.max_output_mib*2**20)-len(text.encode())-4096
                    if not write_episode(path,arrays,limit,compression_level=args.compression_level,timing=write_timing):
                        full=True;print('Data budget reached; returning completed episodes.',flush=True);break
                    manifest.append(record);captures+=len(arrays['camera'])
                    (out/'manifest.json').write_text(text)
                    print(f'SAVED {path}: ticks={len(arrays["state"])}, frames={len(arrays["camera"])}',flush=True)
                if full:break
        finally:
            try:
                if env is not None:env.close()
            finally:scratch.cleanup()
        if full:break
    if not manifest:raise RuntimeError('No usable HyperDrone episodes were captured')
    elapsed=time.perf_counter()-started
    (out/'collection_stats.json').write_text(json.dumps(dict(episodes=len(manifest),captures=captures,
        seconds=elapsed,captures_per_second=captures/elapsed,instances=instances,
        build_seconds=build_seconds,rollout_seconds=rollout_seconds,
        compression_level=args.compression_level,recording_bytes=sum(p.stat().st_size for p in out.glob('rollout_s*_e*.npz')),
        repaired_empty_primitives=repaired_empty_primitives,
        **write_timing,
        simulated_captures=simulated_captures,
        rollout_captures_per_second=simulated_captures/max(rollout_seconds,1e-9)),indent=2)+'\n')
    # Stats also live inside the caller's bounded directory. Reserve their bytes
    # before collecting by leaving a small metadata allowance in the producer.
    print(f'DONE {len(manifest)} episodes in {elapsed:.2f}s; {captures/elapsed:.1f} captures/s',flush=True)
    print(f'Collection timing: build={build_seconds:.2f}s, rollout={rollout_seconds:.2f}s, '
          f'compression={write_timing["compression_seconds"]:.2f}s (level {args.compression_level}), '
          f'write={write_timing["write_seconds"]:.2f}s',flush=True)


if __name__=='__main__':main()
