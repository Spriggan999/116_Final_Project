"""Recorded HyperDrone datasets and causal frame generation (host only)."""
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import config as cfg


def camera_values(images,device=None):
    """Normalize sampled uint8 captures; preserve existing float inputs."""
    if images.dtype==torch.uint8:
        return images.to(device=device,dtype=torch.float32,non_blocking=True)/255.
    return images.to(device=device,non_blocking=True)


def frame_pair_images(batch,device):
    """Normalize after transfer; optional batched photometric work runs on-device."""
    images=[camera_values(batch[key],device) for key in ('previous','anchor','target')]
    if 'photometric' not in batch:
        return images
    parameters=batch['photometric'].to(device=device,non_blocking=True)
    count=len(images[0]);x=torch.cat(images,0)
    brightness,contrast,blur=parameters.repeat(3,1).unbind(-1)
    x=((x-.5)*contrast[:,None,None,None]+.5)*brightness[:,None,None,None]
    # No host read of CUDA scalars and no per-image host thread pools.
    blurred=F.avg_pool2d(x,3,stride=1,padding=1,count_include_pad=False)
    x=((1-blur[:,None,None,None])*x+blur[:,None,None,None]*blurred).clamp(0,1)
    return x.split(count,0)


class EpisodeStore:
    def __init__(self, files, h, w):
        self.entries=[]
        for f in files:
            with np.load(f,allow_pickle=False) as n:
                cams=torch.from_numpy(n['camera'].copy())
                if cams.ndim!=4 or cams.shape[-1]!=3:
                    raise ValueError(f'{f}: expected uint8 frames [N,H,W,3]')
                if (h,w) == (480,640) and tuple(cams.shape[1:3]) != (480,640):
                    raise ValueError(f'{f}: full-resolution training requires 640x480 recordings; '
                                     'recollect with --stream-data or collect.py --width 640 --height 480')
                source_size = tuple(cams.shape[1:3])
                # Keep long recordings compact; convert sampled camera windows
                # and frame pairs only. Legacy resizing retains its float result.
                cams = cams.permute(0,3,1,2)
                if source_size != (h,w):
                    cams=F.interpolate(camera_values(cams),size=(h,w),mode='bilinear',align_corners=False)
                state=torch.from_numpy(n['state'][:,:18].copy()).float()
                imu=torch.from_numpy(n['imu'][:,:6].copy()).float()
                ticks=torch.from_numpy(n['camera_ticks'].copy()).long()
                stride=int(n['frame_stride']); dt=float(n['dt'])
                assert ticks[0]==0 and torch.all(ticks[1:] - ticks[:-1]==stride), (f,ticks)
                assert state.shape[0]==imu.shape[0] and ticks[-1]<state.shape[0]
                assert abs(dt-.005)<.001
                assert torch.isfinite(state).all() and torch.isfinite(imu).all()
                self.entries.append(dict(cams=cams,cam_ticks=ticks,imu=imu,state=state,
                    stride=stride,dt=dt,file=str(f),scene_id=int(n['scene_id']) if 'scene_id' in n else str(f)))
        if not self.entries: raise ValueError('No recorded files. Run the collector first.')

class FramePairs(Dataset):
    def __init__(self,store,max_gap_s=.08,max_samples=None,seed=17,raw_images=False):
        self.store=store; self.items=[];self.raw_images=raw_images
        for eidx,e in enumerate(store.entries):
            ticks=e['cam_ticks']; stride=e['stride']; dt=e['dt']
            maxlag=max(1,int(round(max_gap_s/(stride*dt))))
            for j in range(2,len(ticks)):
                for lag in range(1,min(j-1,maxlag)+1):
                    a=j-lag
                    for previous_gap in (1,2,3):
                        prev=a-previous_gap
                        if prev>=0:
                            self.items.append((eidx,prev,a,j))
        if max_samples is not None and len(self.items)>max_samples:
            self.items=random.Random(seed).sample(self.items,max_samples)
        if not self.items: raise RuntimeError('No pairs for frame training')
    def __len__(self): return len(self.items)
    def __getitem__(self,idx):
        eidx,prev,a,b=self.items[idx];e=self.store.entries[eidx]
        t_prev=int(e['cam_ticks'][prev]);t0=int(e['cam_ticks'][a]);t1=int(e['cam_ticks'][b])
        imu=e['imu'][t0+1:t1+1].mean(0)
        convert=(lambda image:image) if self.raw_images else camera_values
        return dict(previous=convert(e['cams'][prev]),anchor=convert(e['cams'][a]),target=convert(e['cams'][b]),
                    imu=imu,elapsed=torch.tensor([(t1-t0)*e['dt']],dtype=torch.float32),
                    prev_elapsed=torch.tensor([(t0-t_prev)*e['dt']],dtype=torch.float32))

class StateWindows(Dataset):
    def __init__(self,store,length=101,training=True,compact_cameras=False):
        self.store=store;self.length=length;self.items=[]
        self.cached_frames=None
        self.compact_cameras=compact_cameras
        for eidx,e in enumerate(store.entries):
            s=e['stride']
            # Every window starts on a real camera capture.
            starts=range(0,e['state'].shape[0]-length+1, s*(4 if training else 5))
            for start in starts:
                if int(e['cam_ticks'][-1])>=start+(length-1):
                    self.items.append((eidx,start))
        if not self.items: raise RuntimeError('No state windows. Increase HD_STEPS or reduce WINDOW.')
    def __len__(self):return len(self.items)
    def __getitem__(self,idx):
        eidx,start=self.items[idx];e=self.store.entries[eidx];T=self.length
        stride=e['stride']
        raw=e['state'][start:start+T]
        p=raw[:,:3];R=raw[:,3:12].reshape(-1,3,3);v=raw[:,12:15];omega=raw[:,15:18]
        R0=R[0];R0T=R0.T
        rot=R0T.unsqueeze(0)@R
        pos=(R0T@(p-p[0]).T).T
        # The VIO recurrent state initializes velocity to zero, so train *delta velocity*
        # relative to the window's initial velocity (which is unknown at inference).
        vel=(R0T@(v-v[0]).T).T
        # Initial gravity is inferred from the accelerometer, not simulator truth.
        imu=e['imu'][start:start+T]
        g_hat=-imu[1:min(T,11),:3].mean(0)
        g_hat=g_hat.clamp(-20,20)
        g_hat=9.81*g_hat/torch.linalg.vector_norm(g_hat).clamp_min(1e-5)
        cam_start=start//stride
        n_cam=(T-1)//stride+1
        images=e['cams'][cam_start:cam_start+n_cam]
        assert images.shape[0]==n_cam
        # Select available native captures at ~20 Hz without ever accessing future RGB.
        # When camera stride=4 (50 Hz), desired ticks 0,10,20,... map to
        # observed ticks 0,8,20,28,40,... (variable gaps, 20 Hz on average).
        ts=torch.arange(T)
        desired_ticks=torch.arange(0,T,10)
        selected_idx=torch.unique_consecutive(torch.div(desired_ticks,stride,rounding_mode='floor'))
        capture_idx=torch.div(ts,stride,rounding_mode='floor')
        selected_slot=torch.searchsorted(selected_idx,capture_idx,right=True)-1
        selected_slot=selected_slot.clamp_min(0)
        anchor_img_idx=selected_idx[selected_slot]
        anchor_tick=anchor_img_idx*stride
        observed=(ts==anchor_tick)
        previous_slot=(selected_slot-1).clamp_min(0)
        prev_img_idx=selected_idx[previous_slot]
        prev_span=(anchor_img_idx-prev_img_idx)*stride*e['dt']
        # A trainer started before compact camera support can send an older
        # dataset instance to a newly spawned worker importing this module.
        # Its camera indices still address the original, full capture array.
        if getattr(self,'compact_cameras',False):
            # Retain precisely the same available observations and timing while
            # avoiding augmentation/IPC of native captures the model cannot see.
            images=images[selected_idx]
            anchor_img_idx=selected_slot
            prev_img_idx=previous_slot
        # gt targets expressed relative to the initial BODY frame.
        result=dict(cam_images=images,anchor_img_idx=anchor_img_idx,
                    prev_img_idx=prev_img_idx,prev_span=prev_span.float(),
                    anchor_tick=anchor_tick,observed=observed,
                    imu=imu,pos=pos,vel=vel,rot=rot,omega=omega,
                    gravity=g_hat,dt=torch.tensor(e['dt'],dtype=torch.float32))
        if self.cached_frames is not None:
            # A read-only .npy mmap. Copy to a writable tensor for DataLoader.
            result['frames_cached']=torch.from_numpy(
                np.asarray(self.cached_frames[idx]).copy())
            for k in ('cam_images','anchor_img_idx','prev_img_idx',
                      'prev_span','anchor_tick'):
                result.pop(k,None)
        return result

@torch.no_grad()
def generate_200hz_frames(model,b,device):
    """Synthesize RGB in small chunks without materializing all anchor copies."""
    cam=camera_values(b['cam_images'],device); imu=b['imu'].to(device,non_blocking=True)
    anchor_idx=b['anchor_img_idx'].long().to(device,non_blocking=True)
    anchor_tick=b['anchor_tick'].long().to(device,non_blocking=True)
    prev_img_idx=b['prev_img_idx'].long().to(device,non_blocking=True)
    prev_span=b['prev_span'].to(device,non_blocking=True)
    # Plan on CPU once. Never read a CUDA boolean inside the generation loop.
    observed=b['observed'].detach().cpu().reshape(-1)
    real_indices=observed.nonzero(as_tuple=True)[0].to(device,non_blocking=True)
    missing_indices=(~observed).nonzero(as_tuple=True)[0].to(device,non_blocking=True)
    B,T=anchor_idx.shape; _,_,C,H,W=cam.shape
    sums=torch.cat([imu.new_zeros(B,1,6),imu.cumsum(1)],1)
    dt=b['dt'].to(device,non_blocking=True)
    chunk=max(1,int(cfg.FRAME_GENERATION_CHUNK))
    output=cam.new_empty(B*T,C,H,W)
    real_batch,real_tick=real_indices//T,real_indices%T
    output[real_indices]=cam[real_batch,anchor_idx[real_batch,real_tick]]
    for lo in range(0,len(missing_indices),chunk):
        indices=missing_indices[lo:lo+chunk]
        batch_idx,tick=indices//T,indices%T
        anchors=cam[batch_idx,anchor_idx[batch_idx,tick]]
        previous=cam[batch_idx,prev_img_idx[batch_idx,tick]]
        start=anchor_tick[batch_idx,tick]
        duration=(tick-start).clamp_min(1)
        avg=(sums[batch_idx,tick+1]-sums[batch_idx,start+1])/duration[:,None]
        elapsed=((tick-start)*dt[batch_idx])[:,None]
        output[indices]=model(previous,anchors,avg,elapsed,prev_span[batch_idx,tick,None])
    return output.reshape(B,T,C,H,W)

def observed_camera_features(model,b,observed,device):
    """Select real observations before bounded transfers/normalization."""
    ticks=observed.shape[1]
    real=observed.reshape(-1).nonzero(as_tuple=True)[0]
    anchors=b['anchor_img_idx'].detach().cpu()
    cameras=b['cam_images']
    parts=[];chunk=max(1,int(cfg.VISUAL_CHUNK))
    for start in range(0,len(real),chunk):
        indices=real[start:start+chunk]
        rows=indices//ticks
        images=anchors[rows,indices%ticks]
        if cameras.device.type!='cpu':rows=rows.to(cameras.device);images=images.to(cameras.device)
        selected=cameras[rows,images]
        if cameras.device.type=='cpu' and cameras.is_pinned():selected=selected.pin_memory()
        parts.append(model.encoder(camera_values(selected,device)))
    return torch.cat(parts,0)


@torch.no_grad()
def generate_200hz_embeddings(model,b,device):
    """Encode selected real captures once; predict missing ticks in feature space."""
    observed=b['observed'].detach().cpu()
    B,T=observed.shape
    if not observed[:,0].all():raise ValueError('Embedding windows must start on a real capture')
    real=observed.reshape(-1).nonzero(as_tuple=True)[0].to(device,non_blocking=True)
    features=observed_camera_features(model,b,observed,device)
    # Slots refer to selected past observations, rather than all native captures.
    slots=observed.long().cumsum(1)-1
    offsets=observed.sum(1).cumsum(0)-observed.sum(1)
    anchors=(slots+offsets[:,None]).to(device,non_blocking=True)
    previous=((slots-1).clamp_min(0)+offsets[:,None]).to(device,non_blocking=True)
    if not cfg.USE_GENERATED_FRAMES_FOR_STATE:return features[anchors]
    dimension=features.shape[-1]
    output=features.new_empty(B*T,dimension)
    output[real]=features
    missing=(~observed).reshape(-1).nonzero(as_tuple=True)[0].to(device,non_blocking=True)
    imu=b['imu'].to(device,non_blocking=True)
    sums=torch.cat([imu.new_zeros(B,1,6),imu.cumsum(1)],1)
    anchor_tick=b['anchor_tick'].long().to(device,non_blocking=True)
    span=b['prev_span'].to(device,non_blocking=True)
    dt=b['dt'].to(device,non_blocking=True)
    chunk=max(1,int(cfg.EMBEDDING_PREDICTION_CHUNK))
    for lo in range(0,len(missing),chunk):
        ids=missing[lo:lo+chunk];batch,tick=ids//T,ids%T
        start=anchor_tick[batch,tick]
        mean=(sums[batch,tick+1]-sums[batch,start+1])/(tick-start).clamp_min(1)[:,None]
        output[ids]=model.predict_embeddings(features[previous[batch,tick]],features[anchors[batch,tick]],
            mean,((tick-start)*dt[batch])[:,None],span[batch,tick,None])
    return output.reshape(B,T,dimension)


def state_input_frames(model,b,device):
    """Causal RGB windows or direct embeddings for state/PPO/distillation callers."""
    from models import EmbeddingPredictor
    if isinstance(model,EmbeddingPredictor):return generate_200hz_embeddings(model,b,device)
    if 'frames_cached' in b:
        return b['frames_cached'].to(device=device,dtype=torch.float32,non_blocking=True).div_(255.)
    if cfg.USE_GENERATED_FRAMES_FOR_STATE:
        return generate_200hz_frames(model,b,device)
    cams=camera_values(b['cam_images'],device)
    idx=b['anchor_img_idx'].long().to(device,non_blocking=True)
    batch_idx=torch.arange(cams.shape[0],device=device)[:,None]
    return cams[batch_idx,idx]
