"""Frame-rate recurrent correction networks and their differentiable integrator."""
import torch
from torch import nn
import torch.nn.functional as F
from performance import neural_autocast

PROFILES={
    '500': dict(fuel=500,grid_width=2,grid_height=1,hidden=0,aggregate=True,kind='visual_imu_affine'),
    '10k': dict(fuel=10_000,grid_width=6,grid_height=3,hidden=0,aggregate=True),
    '1m': dict(fuel=1_000_000,grid_width=8,grid_height=6,hidden=192,aggregate=False),
    '100m': dict(fuel=100_000_000,grid_width=32,grid_height=24,hidden=1792,aggregate=False),
    '1m_ssm': dict(fuel=1_000_000,grid_width=8,grid_height=6,hidden=64,aggregate=False,
                   kind='mamba2',layers=1,d_state=8,headdim=16,attention_dim=0,heads=2,window=8),
    '1m_hybrid': dict(fuel=1_000_000,grid_width=8,grid_height=6,hidden=64,aggregate=False,
                      kind='mamba2',layers=1,d_state=8,headdim=16,attention_dim=64,heads=2,window=8),
    '100m_hybrid': dict(fuel=100_000_000,grid_width=32,grid_height=24,hidden=384,aggregate=False,
                        kind='mamba2',layers=3,d_state=16,headdim=32,attention_dim=384,heads=4,window=16),
}


def architecture_candidates(profile):
    """Ordered, bounded fuel search; never change legacy RNN layouts."""
    spec=dict(PROFILES[profile],profile=profile)
    if spec.get('kind')!='mamba2':return [spec]
    if profile=='1m_ssm':changes=[{},dict(d_state=4)]
    elif profile=='1m_hybrid':
        changes=[{},dict(d_state=4),dict(d_state=4,attention_dim=32),dict(d_state=4,attention_dim=32,window=4)]
    else:changes=[{},dict(layers=2),dict(hidden=256,attention_dim=256),dict(hidden=256,attention_dim=256,layers=2)]
    return [dict(spec,**change) for change in changes]


def camera_frames(frames):
    """Match the uint8 ABI image before extracting sparse pixel features."""
    b,k,c,h,w=frames.shape
    if (h,w)==(240,320):return frames
    x=frames.reshape(b*k,c,h,w)
    if (h,w)==(480,640):
        x=F.avg_pool2d((x*255).round(),2)
    else:
        x=F.interpolate(x,size=(240,320),mode='bilinear',align_corners=False)*255
    return (torch.floor(x+.5).clamp(0,255)/255).reshape(b,k,c,240,320)


def qmul(a,b):
    av,aw=a[...,:3],a[...,3:]
    bv,bw=b[...,:3],b[...,3:]
    return torch.cat([aw*bv+bw*av+torch.linalg.cross(av,bv),aw*bw-(av*bv).sum(-1,keepdim=True)],-1)


def qrotate(q,v):
    t=2*torch.linalg.cross(q[...,:3],v)
    return v+q[...,3:]*t+torch.linalg.cross(q[...,:3],t)


def qmatrix(q):
    eye=torch.eye(3,device=q.device,dtype=q.dtype).expand(*q.shape[:-1],3,3)
    return torch.stack([qrotate(q,eye[...,i,:]) for i in range(3)],-1)


def integrate(p,v,q,imu,correction,dt,gravity):
    omega=imu[:,3:]+correction[:,:3]
    acceleration=qrotate(q,imu[:,:3]+correction[:,3:6])+gravity+correction[:,6:]
    interval=dt[:,None]
    position=p+v*interval+.5*acceleration*interval.square()
    velocity=v+acceleration*interval
    dq=torch.cat([omega*interval*.5,torch.ones_like(interval)],-1)
    rotation=F.normalize(qmul(q,dq),dim=-1)
    return position,velocity,rotation,omega


def integrate_parallel(imu,correction,interval,gravity):
    """Same normalized Euler quaternion integrator, using associative prefixes.

    Used on CUDA to avoid thousands of Python/autograd launches per student.
    No SO(3) exponential substitution: deployment keeps the original equations.
    """
    b,n,_=imu.shape
    omega=imu[:,:,3:]+correction[:,:,:3]
    dq=F.normalize(torch.cat([omega*interval[:,:,None]*.5,torch.ones_like(interval[:,:,None])],-1),dim=-1)
    offset=1
    while offset<n:
        dq=torch.cat([dq[:,:offset],F.normalize(qmul(dq[:,:-offset],dq[:,offset:]),dim=-1)],1)
        offset*=2
    initial=imu.new_zeros(b,1,4);initial[:,:,3]=1
    q=torch.cat([initial,dq],1)
    acceleration=qrotate(q[:,:-1],imu[:,:,:3]+correction[:,:,3:6])+gravity[:,None]+correction[:,:,6:]
    zero=imu.new_zeros(b,1,3)
    v=torch.cat([zero,(acceleration*interval[:,:,None]).cumsum(1)],1)
    p=torch.cat([zero,(v[:,:-1]*interval[:,:,None]+.5*acceleration*interval[:,:,None].square()).cumsum(1)],1)
    return dict(pos=p,vel=v,rot=qmatrix(q),omega=torch.cat([zero,omega],1))


class BudgetStudent(nn.Module):
    def __init__(self,settings):
        super().__init__()
        self.settings=dict(settings)
        if settings.get('kind') in ('imu_affine','visual_imu_affine'):
            self.features=7 if settings['kind']=='visual_imu_affine' else 0
            self.gain=nn.Parameter(torch.ones(3))
            self.bias=nn.Parameter(torch.tensor([0.,0.,-9.81]))
            if settings['kind']=='visual_imu_affine':self.vision_weight=nn.Parameter(torch.zeros(3,4))
            return
        n=settings['grid_width']*settings['grid_height']
        self.features=2*n+7
        h=settings['hidden']
        self.is_ssm=settings.get('kind')=='mamba2'
        if self.is_ssm:
            from models import _make_mamba2
            self.input=nn.Linear(self.features,h)
            self.ssm=nn.ModuleList([_make_mamba2(h,settings['d_state'],settings['headdim']) for _ in range(settings['layers'])])
            self.ssm_norm=nn.ModuleList([nn.LayerNorm(h) for _ in self.ssm])
            a=settings['attention_dim']
            if a:
                self.attn_input=nn.Linear(h,a) if a!=h else nn.Identity()
                self.attn=nn.MultiheadAttention(a,settings['heads'],batch_first=True)
                self.attn_output=nn.Linear(a,h) if a!=h else nn.Identity()
                self.attn_norm=nn.LayerNorm(h)
            self.head=nn.Linear(h,9)
            nn.init.normal_(self.head.weight,std=.01);nn.init.zeros_(self.head.bias)
            return
        if h:
            self.input=nn.Linear(self.features,h)
            self.recurrent=nn.Linear(h,h,bias=False)
        self.head=nn.Linear(h or self.features,9)
        nn.init.normal_(self.head.weight,std=.01)
        nn.init.zeros_(self.head.bias)

    def update(self,features,hidden):
        if self.is_ssm:
            if self.training:raise ValueError('Mamba training requires sequence forward, not streaming updates')
            if not isinstance(hidden,dict):
                hidden=dict(caches=[block.allocate_inference_cache(len(features),1) for block in self.ssm],history=[])
            x=torch.tanh(self.input(features))[:,None]
            for i,(block,norm) in enumerate(zip(self.ssm,self.ssm_norm)):
                value,conv,state=block.step(x,*hidden['caches'][i])
                hidden['caches'][i]=(conv,state);x=norm(x+value)
            if self.settings['attention_dim']:
                token=self.attn_input(x);hidden['history']=(hidden['history']+[token])[-self.settings['window']:]
                history=torch.cat(hidden['history'],1)
                value=self.attn(token,history,history,need_weights=False)[0]
                x=self.attn_norm(x+self.attn_output(value))
            return (.25*torch.tanh(self.head(x[:,0]))).float(),hidden
        with neural_autocast(features):
            if self.settings['hidden']:
                hidden=torch.tanh(self.input(features)+self.recurrent(hidden))
                value=hidden
            else:
                value=features
            correction=.25*torch.tanh(self.head(value))
        return correction.float(),hidden.float()

    def sequence_corrections(self,features):
        with neural_autocast(features):
            if not self.is_ssm:
                if self.settings['hidden']:
                    projected=self.input(features)
                    hidden=features.new_zeros(len(features),self.settings['hidden']);states=[]
                    for token in projected.unbind(1):
                        hidden=torch.tanh(token+self.recurrent(hidden)).float();states.append(hidden)
                    value=torch.stack(states,1)
                else:value=features
                return (.25*torch.tanh(self.head(value))).float()
            from attention import local_attention
            x=torch.tanh(self.input(features))
            for block,norm in zip(self.ssm,self.ssm_norm):x=norm(x+block(x))
            if self.settings['attention_dim']:
                value=local_attention(self.attn,self.attn_input(x),self.settings['window'],128)
                x=self.attn_norm(x+self.attn_output(value))
            return (.25*torch.tanh(self.head(x))).float()

    def image_features(self,frames):
        """Native runtime samples the same RGB pixel centres from 320x240."""
        b,k,c=frames.shape[:3]
        x=camera_frames(frames).reshape(b*k,c,240,320)
        nx,ny=self.settings['grid_width'],self.settings['grid_height']
        yy=((2*torch.arange(ny,device=x.device)+1)*240//(2*ny)).long()
        xx=((2*torch.arange(nx,device=x.device)+1)*320//(2*nx)).long()
        return x[:,:,yy[:,None],xx[None,:]].mean(1).reshape(b,k,nx*ny)

    def forward(self,images,imu,dt,gravity,ticks):
        if self.settings.get('kind') in ('imu_affine','visual_imu_affine'):
            return self.imu_forward(imu,ticks,images)
        b=images.shape[0]
        gray=self.image_features(images)
        p=images.new_zeros(b,3);v=p.clone();q=images.new_zeros(b,4);q[:,3]=1
        positions=[p];velocities=[v];rotations=[qmatrix(q)];omegas=[images.new_zeros(b,3)]
        spans=[hi-lo for lo,hi in zip(ticks[:-1],ticks[1:])]
        counts=torch.tensor(spans,device=imu.device)
        # Batched packet means avoid cumulative-sum cancellation over long
        # flights. Project all inputs and output heads as large GEMMs; only
        # the RNN hidden-to-hidden recurrence remains sequential.
        if spans and all(span==spans[0] for span in spans):
            means=imu[:,ticks[0]+1:ticks[-1]+1].reshape(b,len(spans),spans[0],6).mean(2)
        else:means=torch.stack([imu[:,lo+1:hi+1].mean(1) for lo,hi in zip(ticks[:-1],ticks[1:])],1)
        means=torch.cat([imu[:,ticks[0]:ticks[0]+1],means],1)
        interval=torch.cat([dt.new_zeros(b,1),counts[None]*dt[:,None]],1)
        previous=torch.cat([gray[:,:1],gray[:,:-1]],1)
        features=torch.cat([gray,gray-previous,means/10.,(interval*20).clamp(max=4)[:,:,None]],-1)
        corrections=self.sequence_corrections(features)
        if images.is_cuda:
            if self.settings['aggregate']:
                sensor=means[:,1:];corr=corrections[:,1:]
                interval=counts[None]*dt[:,None];selected=None
            else:
                sensor=imu[:,ticks[0]+1:ticks[-1]+1]
                corr=torch.repeat_interleave(corrections[:,1:],counts,dim=1,output_size=ticks[-1]-ticks[0])
                interval=dt[:,None].expand(-1,sensor.shape[1]);selected=torch.cat([counts.new_zeros(1),counts.cumsum(0)])
            result=integrate_parallel(sensor,corr,interval,gravity)
            if selected is not None:result={key:value[:,selected] for key,value in result.items()}
            result['corrections']=corrections
            return result
        for index,tick in enumerate(ticks[1:],1):
            start=ticks[index-1]
            correction=corrections[:,index]
            if self.settings['aggregate']:
                p,v,q,omega=integrate(p,v,q,means[:,index],correction,(tick-start)*dt,gravity)
            else:
                for j in range(start+1,tick+1):
                    p,v,q,omega=integrate(p,v,q,imu[:,j],correction,dt,gravity)
            positions.append(p);velocities.append(v);rotations.append(qmatrix(q));omegas.append(omega)
        return dict(pos=torch.stack(positions,1),vel=torch.stack(velocities,1),
                    rot=torch.stack(rotations,1),omega=torch.stack(omegas,1),
                    corrections=corrections)

    def imu_forward(self,imu,ticks,images=None):
        """Tiny affine acceleration model: 3 IMU axes + optional 2-pixel vision.

        Semi-implicit Euler and float32 match the SIMD native kernel exactly.
        Legacy imu_affine checkpoints retain their image-free equations. The
        500 profile learns from two lumas and their temporal differences.
        """
        b=imu.shape[0]
        accelerations=imu[:,ticks[1:],:3]*self.gain+self.bias
        if self.settings.get('kind')=='visual_imu_affine':
            gray=self.image_features(images)
            vision=torch.cat([gray[:,1:],gray[:,1:]-gray[:,:-1]],-1)
            accelerations=accelerations+F.linear(vision,self.vision_weight)
        if imu.is_cuda:
            zero=imu.new_zeros(b,1,3)
            v=torch.cat([zero,(accelerations*.05).cumsum(1)],1)
            p=torch.cat([zero,(v[:,1:]*.05).cumsum(1)],1)
            k=len(ticks)
            return dict(pos=p,vel=v,rot=torch.eye(3,device=imu.device,dtype=imu.dtype)[None,None].expand(b,k,3,3),
                        omega=imu.new_zeros(b,k,3),accelerations=accelerations)
        p=imu.new_zeros(b,3);v=p.clone()
        positions=[p];velocities=[v]
        for acceleration in accelerations.unbind(1):
            v=v+acceleration*.05
            p=p+v*.05
            positions.append(p);velocities.append(v)
        k=len(ticks)
        return dict(pos=torch.stack(positions,1),vel=torch.stack(velocities,1),
                    rot=torch.eye(3,device=imu.device,dtype=imu.dtype)[None,None].expand(b,k,3,3),
                    omega=imu.new_zeros(b,k,3),accelerations=accelerations)
