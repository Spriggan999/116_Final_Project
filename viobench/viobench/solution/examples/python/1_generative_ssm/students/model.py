"""Frame-rate recurrent correction networks and their differentiable integrator."""
import torch
from torch import nn
import torch.nn.functional as F
from performance import neural_autocast

PROFILES={
    '100': dict(fuel=100,grid_width=0,grid_height=0,hidden=0,aggregate=True,kind='imu_affine'),
    '10k': dict(fuel=10_000,grid_width=4,grid_height=3,hidden=0,aggregate=True),
    '1m': dict(fuel=1_000_000,grid_width=8,grid_height=6,hidden=192,aggregate=False),
    '100m': dict(fuel=100_000_000,grid_width=32,grid_height=24,hidden=1792,aggregate=False),
}


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


class BudgetStudent(nn.Module):
    def __init__(self,settings):
        super().__init__()
        self.settings=dict(settings)
        if settings.get('kind')=='imu_affine':
            self.features=0
            self.gain=nn.Parameter(torch.ones(3))
            self.bias=nn.Parameter(torch.tensor([0.,0.,-9.81]))
            return
        n=settings['grid_width']*settings['grid_height']
        self.features=2*n+7
        h=settings['hidden']
        if h:
            self.input=nn.Linear(self.features,h)
            self.recurrent=nn.Linear(h,h,bias=False)
        self.head=nn.Linear(h or self.features,9)
        nn.init.normal_(self.head.weight,std=.01)
        nn.init.zeros_(self.head.bias)

    def update(self,features,hidden):
        with neural_autocast(features):
            if self.settings['hidden']:
                hidden=torch.tanh(self.input(features)+self.recurrent(hidden))
                value=hidden
            else:
                value=features
            correction=.25*torch.tanh(self.head(value))
        return correction.float(),hidden.float()

    def image_features(self,frames):
        """Native runtime samples the same RGB pixel centres from 320x240."""
        b,k,c=frames.shape[:3]
        x=camera_frames(frames).reshape(b*k,c,240,320)
        nx,ny=self.settings['grid_width'],self.settings['grid_height']
        yy=((2*torch.arange(ny,device=x.device)+1)*240//(2*ny)).long()
        xx=((2*torch.arange(nx,device=x.device)+1)*320//(2*nx)).long()
        return x[:,:,yy[:,None],xx[None,:]].mean(1).reshape(b,k,nx*ny)

    def forward(self,images,imu,dt,gravity,ticks):
        if self.settings.get('kind')=='imu_affine':
            return self.imu_forward(imu,ticks)
        b=images.shape[0]
        gray=self.image_features(images)
        p=images.new_zeros(b,3);v=p.clone();q=images.new_zeros(b,4);q[:,3]=1
        hidden=images.new_zeros(b,self.settings['hidden'])
        positions=[p];velocities=[v];rotations=[qmatrix(q)];omegas=[images.new_zeros(b,3)];corrections=[]
        means=[]
        # Neural recurrence depends only on sensor features, so compute its
        # corrections before integration. One scope caches BF16 weight casts
        # across camera ticks; the following physics loop remains outside it.
        with neural_autocast(images):
            for index,tick in enumerate(ticks):
                start=ticks[index-1] if index else tick
                mean=imu[:,start+1:tick+1].mean(1) if index else imu[:,tick]
                means.append(mean)
                interval=(tick-start)*dt
                previous=gray[:,max(0,index-1)]
                features=torch.cat([gray[:,index],gray[:,index]-previous,mean/10.,
                                    (interval*20).clamp(max=4)[:,None]],-1)
                correction,hidden=self.update(features,hidden)
                corrections.append(correction)
        for index,tick in enumerate(ticks[1:],1):
            start=ticks[index-1]
            correction=corrections[index]
            if self.settings['aggregate']:
                p,v,q,omega=integrate(p,v,q,means[index],correction,(tick-start)*dt,gravity)
            else:
                for j in range(start+1,tick+1):
                    p,v,q,omega=integrate(p,v,q,imu[:,j],correction,dt,gravity)
            positions.append(p);velocities.append(v);rotations.append(qmatrix(q));omegas.append(omega)
        return dict(pos=torch.stack(positions,1),vel=torch.stack(velocities,1),
                    rot=torch.stack(rotations,1),omega=torch.stack(omegas,1),
                    corrections=torch.stack(corrections,1))

    def imu_forward(self,imu,ticks):
        """100-fuel prior: last accelerometer sample, fixed 20 Hz, fixed attitude.

        Semi-implicit Euler and float32 match the SIMD native kernel exactly.
        Gyro, RGB and gravity initialization are deliberately absent at this tier.
        """
        b=imu.shape[0]
        p=imu.new_zeros(b,3);v=p.clone()
        positions=[p];velocities=[v];accelerations=[]
        for tick in ticks[1:]:
            acceleration=imu[:,tick,:3]*self.gain+self.bias
            v=v+acceleration*.05
            p=p+v*.05
            positions.append(p);velocities.append(v);accelerations.append(acceleration)
        k=len(ticks)
        return dict(pos=torch.stack(positions,1),vel=torch.stack(velocities,1),
                    rot=torch.eye(3,device=imu.device,dtype=imu.dtype)[None,None].expand(b,k,3,3),
                    omega=imu.new_zeros(b,k,3),accelerations=torch.stack(accelerations,1))
