"""Collect baseline traces on non-public recordings; train/export five tiny networks.

The public episodes are never loaded here. Training runs on CPU.
"""
import argparse
import ast
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch
from torch import nn
from viobench.tools.abi.config import guest_config, pinhole_intrinsics
from viobench.tools.datasets.generate.simulation import mount_to_body_camera
from prepare import HERE, NAMES, generate

ABI=HERE.parents[2]/'runtime/native'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def collect(paths, out):
    generate()
    library=out/'collector.so'
    subprocess.run(['g++','-shared','-fPIC','-O2','-std=c++17','-I'+str(ABI),
                    str(HERE/'common/collect.cpp'),'-o',str(library)],check=True)
    lib=ct.CDLL(str(library.resolve()))
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.init.argtypes=[ct.c_char_p,ct.c_uint32]
    lib.step.argtypes=[ct.c_double,ct.c_void_p,ptr,ct.c_int32,ptr]
    lib.hybrid_patches.argtypes=[ct.c_void_p];lib.hybrid_window.argtypes=[ct.c_void_p]
    patches=[];windows=[];manifest=[]
    for file_index,path in enumerate(paths):
        # Full calibration is required: do not silently assume a lens or mount.
        records=json.loads((path.parent/'manifest.json').read_text())
        info=next(r for r in records if Path(r['file']).name==path.name)
        params=ast.literal_eval(info['parameters'])
        front=params['cameras']['front']
        T=mount_to_body_camera(front['mount'])
        cfg=guest_config(320,240,**pinhole_intrinsics(320,240,front['fov']),T_cam_imu=np.linalg.inv(T))
        raw=json.dumps(cfg).encode();lib.init(raw,len(raw))
        with np.load(path) as data:
            scene=int(data['scene_id'])
            if not (2<=scene<192): raise ValueError(f'Refusing public or held-out scene {scene}')
            dt=float(data['dt']);ticks=data['camera_ticks'];state=data['state'];imu=data['imu']
            # Select actual captures nearest 20 Hz; retain their actual timestamps.
            selected=np.unique(np.abs(ticks[:,None]*dt-np.arange(0,min(20,float(ticks[-1]*dt)),.05)[None,:]).argmin(axis=0))
            camera=data['camera']
            first_R=state[0,3:12].reshape(3,3)@T[:3,:3]
            positions=state[:,:3]+np.einsum('nij,j->ni',state[:,3:12].reshape(-1,3,3),T[:3,3])
            relative=(positions-positions[0])@first_R
            before=-1;patch_count=window_count=0
            for ci in selected:
                tick=int(ticks[ci]);start=max(1,before+1)
                indices=np.arange(start,tick+1)
                rows=np.ascontiguousarray(np.column_stack([indices*dt,imu[indices,:6]]),dtype=np.float64)
                frame=np.ascontiguousarray(cv2.resize(camera[ci],(320,240),interpolation=cv2.INTER_AREA))
                output=np.empty(13,np.float64)
                lib.step(tick*dt,frame.ctypes.data,rows,len(rows),output)
                before=tick
                records_buf=np.empty((20,138),np.float32)
                n=lib.hybrid_patches(records_buf.ctypes.data)
                for r in records_buf[:n]:patches.append((scene,r.copy()));patch_count+=1
                record=np.empty(18,np.float64)
                if lib.hybrid_window(record.ctypes.data):
                    t0,t1=record[12:14]
                    truth0=np.array([np.interp(t0,np.arange(len(state))*dt,relative[:,k]) for k in range(3)])
                    truth1=np.array([np.interp(t1,np.arange(len(state))*dt,relative[:,k]) for k in range(3)])
                    windows.append((scene,np.r_[record,truth1-truth0]));window_count+=1
            manifest.append(dict(path=str(path),sha256=sha(path),scene_id=scene,patches=patch_count,windows=window_count,
                                 fov_degrees=front['fov'],actual_capture_times=True))
            print(f'Collected {path.name}: {patch_count} patch pairs, {window_count} windows',flush=True)
    np.savez_compressed(out/'training_data.npz',
                        patch_scene=np.array([s for s,r in patches]),patch=np.array([r for s,r in patches]),
                        window_scene=np.array([s for s,r in windows]),window=np.array([r for s,r in windows]))
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2))

class MLP(nn.Module):
    def __init__(self,inputs,outputs,hidden=8):
        super().__init__();self.layers=nn.Sequential(nn.Linear(inputs,hidden),nn.ReLU(),nn.Linear(hidden,outputs))
    def forward(self,x):return self.layers(x)

class Tracker(nn.Module):
    def __init__(self,hidden=8,channels=4):
        super().__init__();self.conv=nn.Conv2d(2,channels,3,stride=2);self.head=MLP(channels*9,2,hidden)
    def forward(self,x):return self.head(torch.relu(self.conv(x.reshape(-1,2,8,8))).flatten(1))

def array(name,value):
    flat=np.asarray(value,dtype=np.float32).ravel()
    values=','.join(format(float(v),'.9g')+'f' if '.' in format(float(v),'.9g') or 'e' in format(float(v),'.9g') else format(float(v),'.9g')+'.0f' for v in flat)
    return f'static const float {name}[{len(flat)}]={{'+values+'};\n'

def export(name,model,mean,std,metadata,model_root=None):
    directory=(Path(model_root) if model_root is not None else HERE/'models')/name
    directory.mkdir(parents=True,exist_ok=True)
    head=model.head if name=='patch_tracker' else model
    l1,l2=head.layers[0],head.layers[2]
    text='#pragma once\n// Trained parameters; provenance: training.json\n'
    text+=f'constexpr int NET_IN={l1.in_features},NET_H={l1.out_features},NET_OUT={l2.out_features};\n'
    for key,val in [('net_mean',mean),('net_invstd',1/std),('net_w1',l1.weight.detach()),('net_b1',l1.bias.detach()),('net_w2',l2.weight.detach()),('net_b2',l2.bias.detach())]:text+=array(key,val)
    if name=='patch_tracker':
        text+=array('conv_w',model.conv.weight.detach())+array('conv_b',model.conv.bias.detach())
    (directory/'weights.h').write_text(text)
    torch.save(dict(state_dict=model.state_dict(),mean=mean,std=std),directory/'checkpoint.pt')
    (directory/'training.json').write_text(json.dumps(metadata,indent=2))

def fit(name,x,y,scene,task,epochs,seed,*,validation_scenes=None,
        patience=40,min_delta=1e-4,learning_rate=.001,model_root=None,
        warm_start=True,augment=False,hidden_size=8,tracker_channels=4,
        tracker_shift_scale=3.,variance_bounds=(.25,16.),scale_bounds=(.5,2.)):
    """Train with a fixed scene split, LR decay, and validation-best retention.

    An incumbent is scored on the SAME validation samples before any update.
    Re-normalize its first layer algebraically so warm starts preserve inference.
    """
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    x=np.asarray(x,dtype=np.float32);y=np.asarray(y);scene=np.asarray(scene)
    if len(x)==0:raise ValueError(f'No samples for {name}')
    finite=np.isfinite(x).all(axis=1)&np.isfinite(y.reshape(len(y),-1)).all(axis=1)
    x,y,scene=x[finite],y[finite],scene[finite]
    scenes=sorted(set(scene.tolist()))
    val_scenes=sorted(validation_scenes) if validation_scenes is not None else [scenes[-1]]
    val=np.isin(scene,val_scenes);train=~val
    if not train.any() or not val.any():raise ValueError(f'{name}: empty train/validation split')
    if name=='patch_tracker':mean=np.zeros(tracker_channels*9,np.float32);std=np.ones(tracker_channels*9,np.float32)
    else:mean=x[train].mean(0);std=np.maximum(x[train].std(0),.05)
    net=Tracker(hidden=hidden_size,channels=tracker_channels) if name=='patch_tracker' else MLP(x.shape[1],3 if task=='class' else y.shape[1],hidden=hidden_size)
    directory=(Path(model_root) if model_root is not None else HERE/'models')/name
    checkpoint=directory/'checkpoint.pt'
    resumed=False
    if warm_start and checkpoint.exists():
        old=torch.load(checkpoint,map_location='cpu',weights_only=False)
        net.load_state_dict(old['state_dict']);resumed=True
        if name!='patch_tracker':
            layer=net.layers[0]
            with torch.no_grad():
                weight=layer.weight.clone()
                layer.bias.add_(weight@torch.tensor((mean-old['mean'])/old['std']))
                layer.weight.mul_(torch.tensor(std/old['std'])[None,:])
    tx=torch.tensor(x if name=='patch_tracker' else (x-mean)/std,dtype=torch.float32)
    ty=torch.tensor(y,dtype=torch.long if task=='class' else torch.float32)
    indices=np.flatnonzero(train);vi=np.flatnonzero(val)
    opt=torch.optim.AdamW(net.parameters(),lr=learning_rate,weight_decay=1e-3)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,
        patience=max(3,patience//4),threshold=min_delta,threshold_mode='abs',min_lr=1e-6)
    def loss_fn(pred,target):
        if task=='class':return nn.functional.cross_entropy(pred,target)
        if task=='binary':return nn.functional.binary_cross_entropy_with_logits(pred,target)
        if task=='variance':
            pred=pred.clamp(np.log(variance_bounds[0]),np.log(variance_bounds[1]))
            return .5*(pred+target*torch.exp(-pred)).mean()
        if name=='translation_scale':pred=pred.clamp(np.log(scale_bounds[0]),np.log(scale_bounds[1]))
        elif name=='patch_tracker':pred=pred.clamp(-1,1)
        return nn.functional.smooth_l1_loss(pred,target)
    def evaluate():
        net.eval();total=0.
        with torch.no_grad():
            for batch in np.array_split(vi,max(1,int(np.ceil(len(vi)/1024)))):
                total+=float(loss_fn(net(tx[batch]),ty[batch]))*len(batch)
        return total/len(vi)
    initial=evaluate();best=initial;significant=initial;best_epoch=0;stale=0
    best_state={k:v.clone() for k,v in net.state_dict().items()}
    history=[];training_history=[];lr_history=[];stop_reason='max_epochs'
    for epoch in range(epochs):
        net.train();train_total=0.
        for batch in np.array_split(rng.permutation(indices),max(1,int(np.ceil(len(indices)/256)))):
            bx=tx[batch];by=ty[batch]
            if augment and name=='patch_tracker':
                # Flips preserve translation labels; augment ONLY training pairs.
                image=bx.reshape(-1,2,8,8).clone();by=by.clone()
                horizontal=torch.rand(len(batch))<.5;vertical=torch.rand(len(batch))<.5
                image[horizontal]=image[horizontal].flip(-1);by[horizontal,0]*=-1
                image[vertical]=image[vertical].flip(-2);by[vertical,1]*=-1
                gain=.85+.3*torch.rand(len(batch),1,1,1)
                bx=(image*gain).clamp(-2,2).flatten(1)
            opt.zero_grad();loss=loss_fn(net(bx),by);loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
            train_total+=float(loss.detach())*len(batch)
        value=evaluate();history.append(value);training_history.append(train_total/len(indices))
        lr_history.append(float(opt.param_groups[0]['lr']));scheduler.step(value)
        if value<best:
            best=value;best_epoch=epoch+1;best_state={k:v.clone() for k,v in net.state_dict().items()}
        if value<significant-min_delta:significant=value;stale=0
        else:stale+=1
        if (epoch+1)%25==0:
            print(f'  {name} epoch {epoch+1}: val={value:.6f}, best={best:.6f}, lr={opt.param_groups[0]["lr"]:.2g}',flush=True)
        if stale>=patience:stop_reason='validation_plateau';break
    net.load_state_dict(best_state);net.eval()
    with torch.no_grad():pred=net(tx[vi])
    extra={}
    if task=='class':extra['validation_accuracy']=float((pred.argmax(1)==ty[vi]).float().mean())
    elif name=='patch_tracker':extra['validation_offset_rmse_px']=float(((pred.clamp(-1,1)-ty[vi])*tracker_shift_scale).square().mean().sqrt())
    metadata=dict(seed=seed,epochs=len(history),max_epochs=epochs,best_epoch=best_epoch,
                  stop_reason=stop_reason,patience=patience,min_delta=min_delta,
                  train_samples=int(train.sum()),validation_samples=int(val.sum()),
                  training_scenes=sorted(set(scene[train].tolist())),validation_scene=val_scenes[0],
                  validation_scenes=val_scenes,initial_validation_loss=initial,validation_loss=best,
                  warm_started=resumed,label_type=task,hidden_size=hidden_size,
                  tracker_channels=tracker_channels,tracker_shift_scale=tracker_shift_scale,
                  variance_bounds=list(variance_bounds),scale_bounds=list(scale_bounds),
                  validation_history=history,
                  training_history=training_history,learning_rate_history=lr_history,**extra)
    export(name,net,mean,std,metadata,model_root=model_root)
    print(f'Trained {name}: {metadata["train_samples"]} train / {metadata["validation_samples"]} validation; '
          f'{initial:.5f} -> {best:.5f}; best epoch {best_epoch}; {stop_reason}',flush=True)
    return metadata

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--recordings',type=Path,required=True)
    p.add_argument('--out',type=Path,default=HERE/'training')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--seed',type=int,default=116)
    p.add_argument('--max-recordings',type=int,default=10)
    p.add_argument('--reuse-data',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    if not args.reuse_data:
        # One recording per scene, excluding public scene 1 and held-out 192+.
        chosen={}
        for path in sorted(args.recordings.rglob('rollout_s*_e*.npz')):
            scene=int(path.name.split('_s')[1].split('_')[0])
            if 2<=scene<192 and scene not in chosen:chosen[scene]=path
        paths=list(chosen.values())[:args.max_recordings]
        if len(paths)<2:raise ValueError('Need recordings from two or more non-public training scenes')
        collect(paths,args.out)
    with np.load(args.out/'training_data.npz') as d:
        patches=d['patch'];ps=d['patch_scene'];windows=d['window'];ws=d['window_scene']
    if len(windows)<20 or len(patches)<100:raise ValueError('Insufficient usable training traces')
    results={}
    results['patch_quality']=fit('patch_quality',patches[:,:6],patches[:,6:7],ps,'binary',args.epochs,args.seed)
    ok=patches[:,6]>.5
    results['patch_tracker']=fit('patch_tracker',patches[ok,10:],patches[ok,7:9],ps[ok],'regression',args.epochs,args.seed)
    f=windows[:,:8].astype(np.float32);direction=windows[:,8:11];distance=windows[:,11];truth=windows[:,18:21]
    length=np.linalg.norm(truth,axis=1);along=(truth*direction).sum(1)
    perpendicular=np.linalg.norm(truth-direction*along[:,None],axis=1)
    # Squared standardized residuals train log variance via Gaussian NLL.
    angle_error=np.arctan2(perpendicular,np.maximum(along,1e-6))
    err2=np.column_stack([(angle_error/np.maximum(windows[:,14],.02))**2,
                           ((along-distance)/np.maximum(windows[:,15],.01))**2]).clip(0,100).astype(np.float32)
    results['measurement_confidence']=fit('measurement_confidence',f,err2,ws,'variance',args.epochs,args.seed)
    # Distill a conservative budget heuristic for the *next* visual window.
    # This is an imitation target, not a claim of an optimal compute policy.
    x=[];y=[];scene_ids=[]
    for scene in sorted(set(ws)):
        ids=np.flatnonzero(ws==scene)
        for before,current in zip(ids[:-1],ids[1:]):
            row=f[current]
            action=0 if row[2]>.85 and row[1]<.25 and row[6]<.2 and row[0]<1 else 1
            if row[2]<.65 or row[1]>.6 or row[6]>.6:action=2
            x.append(f[before]);y.append(action);scene_ids.append(scene)
    results['compute_controller']=fit('compute_controller',np.array(x),np.array(y),np.array(scene_ids),'class',args.epochs,args.seed)
    valid=(along>.005)&(distance>.005)&(length>.01)&((along/np.maximum(length,1e-9))>.7)
    target=np.log(np.maximum(along[valid],1e-6)/distance[valid]).clip(-np.log(2),np.log(2)).astype(np.float32)[:,None]
    results['translation_scale']=fit('translation_scale',f[valid],target,ws[valid],'regression',args.epochs,args.seed)
    manifest=json.loads((args.out/'data_manifest.json').read_text())
    for name in NAMES:
        path=HERE/'models'/name/'training.json';meta=json.loads(path.read_text())
        meta.update(base_sha256=sha(HERE/'base/solution.cpp'),data=manifest,
                    public_episodes_used=False,training_data_sha256=sha(args.out/'training_data.npz'))
        path.write_text(json.dumps(meta,indent=2))
    (args.out/'summary.json').write_text(json.dumps(results,indent=2))
    print('All five trained weight headers exported. Run benchmark.py for full 60-second public evaluation.')

if __name__=='__main__':main()
