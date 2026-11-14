"""Training models with the official CUDA Mamba-2 implementation."""
import math
import torch
from torch import nn
import torch.nn.functional as F
import config as cfg
from performance import neural_autocast,fp32_physics


RETIRED_MAMBA_SETTINGS = frozenset((
    'MAMBA_TRAIN_BACKEND', 'USE_PURE_TORCH_MAMBA_INFERENCE', 'EXPORT_PURE_MAMBA_ONNX'))


def require_official_mamba(device=None):
    """Lazy dependency check: GRU workflows do not need the CUDA package."""
    if device is not None and torch.device(device).type != 'cuda':
        raise ValueError('The hybrid uses official Mamba-2 CUDA kernels; select --device cuda')
    # PyTorch's TF32 flag does not control Triton's dot products. BF16 training
    # remains available; IEEE FP32 keeps validation close to the WASM recurrence.
    import os
    os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
    try:
        from mamba_ssm import Mamba2
    except ImportError as error:
        raise ImportError('Official Mamba-2 is required for the hybrid. Install '
                          'requirements-mamba.txt in a compatible CUDA environment.') from error
    return Mamba2


def _make_mamba2(d_model,d_state,headdim):
    if (min(d_model,d_state,headdim)<1 or (2*d_model) % headdim
            or ((2*d_model)//headdim) % 8 or d_state % 4):
        raise ValueError('Official fused Mamba-2 requires an inner head count divisible '
                         'by 8 and d_state divisible by 4; use hidden multiples of 128 '
                         'with headdim=32')
    return require_official_mamba()(d_model=d_model,d_state=d_state,d_conv=4,
                                    expand=2,headdim=headdim)


class MotionWarp(nn.Module):
    """Two-frame, causal motion-conditioned affine warp. No heavy full-res CNN decoder.
    Previous and current are both genuinely observed frames; output is synthesized.
    """
    def __init__(self,hidden=24):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(128+14,hidden),nn.Tanh(),
                               nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    @fp32_physics
    def forward(self,previous,anchor,imu_mean,elapsed,prev_elapsed):
        old=F.adaptive_avg_pool2d(previous.mean(1,keepdim=True),(8,8)).flatten(1)
        now=F.adaptive_avg_pool2d(anchor.mean(1,keepdim=True),(8,8)).flatten(1)
        inp=torch.cat([old,now,imu_mean/10.,imu_mean/10.*elapsed,
                       elapsed,prev_elapsed],dim=-1)
        with neural_autocast(inp): params=self.net(inp).float()
        dxdy=torch.tanh(params[:,:2])*elapsed*3.
        angle=torch.tanh(params[:,2:3])*elapsed*3.
        c=torch.cos(angle).squeeze(-1);s=torch.sin(angle).squeeze(-1)
        B=len(anchor)
        theta=anchor.new_zeros(B,2,3)
        theta[:,0,0]=c;theta[:,0,1]=-s;theta[:,1,0]=s;theta[:,1,1]=c
        theta[:,:,2]=dxdy
        grid=F.affine_grid(theta,anchor.size(),align_corners=False)
        return F.grid_sample(anchor,grid,mode='bilinear',
                             padding_mode='border',align_corners=False)

class GenerativeFlowPredictor(nn.Module):
    """Causal, geometry-guided RGB predictor (flow warp + generative refinement).

    Inputs: previous real frame, latest real frame, IMU average since latest
    frame, target time offset, and preceding camera-frame spacing.
    Output: predicted RGB for any positive horizon; no future observed frame.
    Trained on future *native* camera captures, not unavailable 200Hz RGB truth.
    """
    def __init__(self, base=24, max_flow_pixels=24., residual_scale=.25, input_stride=1):
        super().__init__()
        self.max_flow_pixels=float(max_flow_pixels)
        self.residual_scale=float(residual_scale)
        if input_stride not in (1,2,4,8):
            raise ValueError('flow input_stride must be 1, 2, 4 or 8')
        self.input_stride=input_stride

        def conv_block(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                nn.GroupNorm(math.gcd(8,cout), cout), nn.SiLU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.GroupNorm(math.gcd(8,cout), cout), nn.SiLU(),
            )

        # previous RGB + anchor RGB + change RGB + normalized XY coordinates
        self.enc1=conv_block(11,base)
        if input_stride > 1:
            self.enc1[0]=nn.Conv2d(11,base,input_stride,stride=input_stride)
        self.enc2=conv_block(base,2*base,stride=2)
        self.enc3=conv_block(2*base,4*base,stride=2)
        self.bottleneck=conv_block(4*base,4*base)
        self.imu_condition=nn.Sequential(
            nn.Linear(8,4*base),nn.SiLU(),nn.Linear(4*base,4*base))
        self.dec2=conv_block(4*base+2*base,2*base)
        self.dec1=conv_block(2*base+base,base)
        # flow_x, flow_y, generator/blend mask, RGB residual
        self.head=nn.Conv2d(base,6,3,padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @fp32_physics
    def forward(self,previous,anchor,imu_mean,elapsed,prev_elapsed,
                return_aux=False):
        B,C,H,W=anchor.shape
        if previous.shape != anchor.shape or C != 3:
            raise ValueError('Expected previous and anchor to be [B,3,H,W]')
        # align_corners=False: pixel-center normalized coordinates
        yy,xx=torch.meshgrid(
            (torch.arange(H,device=anchor.device,dtype=anchor.dtype)+.5)*2/H-1,
            (torch.arange(W,device=anchor.device,dtype=anchor.dtype)+.5)*2/W-1,
            indexing='ij')
        base_grid=torch.stack([xx,yy],dim=-1)[None].expand(B,-1,-1,-1)
        coords=base_grid.permute(0,3,1,2)
        x=torch.cat([previous,anchor,anchor-previous,coords],dim=1)
        with neural_autocast(x):
            a=self.enc1(x)
            b=self.enc2(a)
            c=self.enc3(b)
            cond=torch.cat([imu_mean/10.,elapsed,prev_elapsed],dim=1)
            z=self.bottleneck(c)+self.imu_condition(cond)[:,:,None,None]
            d2=self.dec2(torch.cat([F.interpolate(z,size=b.shape[-2:],mode='bilinear',
                                                  align_corners=False),b],dim=1))
            d1=self.dec1(torch.cat([F.interpolate(d2,size=a.shape[-2:],mode='bilinear',
                                                  align_corners=False),a],dim=1))
            raw=self.head(d1)
        raw=raw.float()
        if self.input_stride > 1:
            raw=F.interpolate(raw,size=(H,W),mode='bilinear',align_corners=False)
        # The bound is in pixels of the *internal* training resolution.
        time_scale=(elapsed/0.05).clamp(0.,2.)[:,:,None,None]
        flow_px=self.max_flow_pixels*torch.tanh(raw[:,:2])*time_scale
        dx=flow_px[:,0]* (2./W)
        dy=flow_px[:,1]* (2./H)
        offset=torch.stack([dx,dy],dim=-1)
        warped=F.grid_sample(anchor,base_grid+offset,mode='bilinear',
                             padding_mode='border',align_corners=False)
        # The generative branch is deliberately initially conservative.
        generator_mask=torch.sigmoid(raw[:,2:3]-2.)
        generated=(anchor+self.residual_scale*torch.tanh(raw[:,3:6])).clamp(0.,1.)
        image=(1.-generator_mask)*warped+generator_mask*generated
        if return_aux:
            return image,dict(flow_pixels=flow_px,mask=generator_mask,
                              warped=warped,generated=generated)
        return image

def so3_exp(omega):
    # Stable Rodrigues formula. Input axis-angle [B,3].
    theta2=(omega*omega).sum(-1,keepdim=True).clamp_min(1e-12)
    theta=torch.sqrt(theta2)
    a=torch.where(theta2<1e-6,1-theta2/6+theta2*theta2/120,torch.sin(theta)/theta)
    b=torch.where(theta2<1e-6,.5-theta2/24+theta2*theta2/720,(1-torch.cos(theta))/theta2)
    x,y,z=omega.unbind(-1);zero=torch.zeros_like(x)
    K=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],dim=-1).reshape(-1,3,3)
    I=torch.eye(3,device=omega.device,dtype=omega.dtype).unsqueeze(0)
    return I+a[:,:,None]*K+b[:,:,None]*(K@K)

class VisualFeatures(nn.Module):
    """Legacy 48-value gray pooling or a learned CNN with configurable output width.
    Direct embedding models carry the selected width through temporal fusion.
    """
    def __init__(self, learned=False, input_stride=2, feature_dim=48):
        super().__init__()
        self.learned=learned
        self.feature_dropout=cfg.STATE_VISUAL_FEATURE_DROPOUT
        if type(feature_dim) is not int or feature_dim<1 or (not learned and feature_dim!=48):
            raise ValueError('Non-48 visual dimensions require the learned visual encoder')
        self.feature_dim=feature_dim
        if input_stride not in (2,4,8):
            raise ValueError('visual input_stride must be 2, 4 or 8')
        if learned:
            # The default 48D interface preserves existing checkpoint shapes.
            self.cnn=nn.Sequential(
                nn.Conv2d(3,16,3,stride=2,padding=1),nn.ReLU(),
                nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU(),
                nn.Conv2d(32,48,3,stride=2,padding=1),nn.ReLU(),
                nn.Conv2d(48,64,3,stride=2,padding=1),nn.ReLU(),
                nn.AdaptiveAvgPool2d((3,4)),nn.Flatten(),
                nn.Linear(64*3*4,feature_dim),nn.Tanh())
            if input_stride > 2:
                self.cnn[0]=nn.Conv2d(3,16,input_stride,stride=input_stride)
    def encode(self,images):
        with neural_autocast(images):return self.cnn(images).float()

    def forward(self,frame):
        if self.learned:
            # At full resolution a B*T sequence can contain hundreds of images.
            # Bound inference temporaries, and recompute activations during
            # backward instead of retaining the entire sequence's CNN graph.
            if torch.jit.is_tracing():
                # Export has a static batch of one; keep its graph direct.
                features=self.cnn(frame)
            else:
                from torch.utils.checkpoint import checkpoint
                chunk = max(1, int(cfg.VISUAL_CHUNK))
                parts = []
                for start in range(0, len(frame), chunk):
                    images = frame[start:start+chunk]
                    if self.training and torch.is_grad_enabled():
                        parts.append(checkpoint(self.encode, images, use_reentrant=False))
                    else:
                        parts.append(self.encode(images))
                features=torch.cat(parts, dim=0)
        else:
            features=F.adaptive_avg_pool2d(frame.mean(1,keepdim=True),(6,8)).flatten(1)
        return F.dropout(features,p=self.feature_dropout,training=self.training)

class EmbeddingPredictor(nn.Module):
    """Causal IMU-conditioned visual prediction with a stop-gradient EMA encoder."""
    def __init__(self,hidden=256,learned_visual=True,visual_input_stride=8,ema=.99,embedding_dim=48):
        super().__init__()
        import copy
        self.embedding_dim=embedding_dim
        self.encoder=VisualFeatures(learned_visual,visual_input_stride,feature_dim=embedding_dim)
        self.encoder.feature_dropout=0.
        self.target_encoder=copy.deepcopy(self.encoder).eval()
        self.target_encoder.requires_grad_(False)
        self.ema=ema
        self.predictor=nn.Sequential(nn.Linear(3*embedding_dim+8,hidden),nn.Tanh(),nn.Linear(hidden,embedding_dim))
        nn.init.normal_(self.predictor[-1].weight,std=.01)
        nn.init.zeros_(self.predictor[-1].bias)

    def train(self,mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def predict_embeddings(self,previous,anchor,imu_mean,elapsed,prev_elapsed):
        features=torch.cat([previous,anchor,anchor-previous,imu_mean/10.,elapsed,prev_elapsed],-1)
        with neural_autocast(features):
            residual=self.predictor(features).float()
        return anchor+.25*torch.tanh(residual)*(elapsed/.05).clamp(0.,2.)

    def forward(self,previous,anchor,imu_mean,elapsed,prev_elapsed):
        encoded=self.encoder(torch.cat([previous,anchor],0))
        a,b=encoded.chunk(2,0)
        return self.predict_embeddings(a,b,imu_mean,elapsed,prev_elapsed)

    @torch.no_grad()
    def update_target(self):
        for target,online in zip(self.target_encoder.parameters(),self.encoder.parameters()):
            target.lerp_(online,1-self.ema)
        for target,online in zip(self.target_encoder.buffers(),self.encoder.buffers()):
            target.copy_(online)


class StateStudent(nn.Module):
    """Streaming, causal estimator. Visual features + 200 Hz IMU + GRUCell.
    Accepts [B,T,D] direct embeddings, or legacy [B,T,3,H,W] RGB with its
    own visual encoder. Real captures retain the configured training resolution.
    """
    def __init__(self,hidden=32,learned_visual=False,visual_input_stride=2,visual_embeddings=False,embedding_dim=48):
        super().__init__()
        self.hidden=hidden
        self.visual_embeddings=visual_embeddings
        self.embedding_dim=embedding_dim if visual_embeddings else 48
        self.visual=nn.Identity() if visual_embeddings else VisualFeatures(learned_visual,visual_input_stride)
        self.feature=nn.Sequential(nn.Linear(2*self.embedding_dim+8,hidden),nn.Tanh())
        self.gru=nn.GRUCell(hidden,hidden)
        self.out=nn.Linear(hidden,9)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    def update(self,frame,prev_frame,imu,age,observed,h,g=None,prev=None):
        if g is None: g=self.visual(frame)
        if prev is None: prev=self.visual(prev_frame)
        inp=torch.cat([g,g-prev,imu/10.,age,observed],1) # 96 + 6 + 1 + 1 = 104
        with neural_autocast(inp):
            h=self.gru(self.feature(inp),h)
            out=.25*torch.tanh(self.out(h))
        return out.float(),h.float()
    def stream_init(self,first_frame):
        """Persistent on-device state; call stream_step once per IMU timestamp."""
        B=first_frame.shape[0]
        return dict(h=first_frame.new_zeros(B,self.hidden),
                    prev=first_frame,
                    age=first_frame.new_zeros(B),
                    pos=first_frame.new_zeros(B,3),
                    vel=first_frame.new_zeros(B,3),
                    R=torch.eye(3,device=first_frame.device,dtype=first_frame.dtype)[None].expand(B,-1,-1).clone())

    def stream_step(self,frame,imu,observed,dt,gravity,state,initial=False):
        """O(1)-state incremental inference; no replay of previous observations."""
        age=torch.where(observed,torch.zeros_like(state['age']),state['age']+dt)
        out,h=self.update(frame,state['prev'],imu,age[:,None],observed.float()[:,None],state['h'])
        if initial:
            pos,vel,R=state['pos'],state['vel'],state['R']
            omega=imu[:,3:6]
        else:
            omega=imu[:,3:6]+out[:,:3]
            acc_body=imu[:,:3]+out[:,3:6]
            acc_w=(state['R']@acc_body[:,:,None]).squeeze(-1)+gravity
            dt1=dt[:,None]
            R=state['R']@so3_exp(omega*dt1)
            acc_corrected=acc_w+out[:,6:9]
            vel=state['vel']+acc_corrected*dt1
            pos=state['pos']+state['vel']*dt1+.5*acc_corrected*dt1.square()
        next_state=dict(h=h,prev=frame,age=age,pos=pos,vel=vel,R=R)
        return dict(pos=pos,vel=vel,rot=R,omega=omega,corrections=out),next_state

    @fp32_physics
    def forward(self,frames,imu,observed,dt,gravity,return_corrections=False):
        B,T=frames.shape[:2]
        h=frames.new_zeros(B,self.hidden)
        corrections=[]
        last_real=frames.new_zeros(B)
        # Hold one autocast scope across ticks so trainable weight casts are cached.
        with neural_autocast(frames):
            visual=frames if self.visual_embeddings else self.visual(frames.flatten(0,1)).reshape(B,T,self.embedding_dim)
            for t in range(T):
                last_real=torch.where(observed[:,t],torch.zeros_like(last_real),last_real+dt)
                out,h=self.update(frames[:,t],frames[:,max(0,t-1)],imu[:,t],last_real[:,None],
                                  observed[:,t].float()[:,None],h,g=visual[:,t],prev=visual[:,max(0,t-1)])
                corrections.append(out)
        result=integrate_corrections(torch.stack(corrections,1).float(),imu.float(),dt.float(),gravity.float(),
                                    parallel=self.training and frames.is_cuda and getattr(cfg,'PARALLEL_TRAINING_PHYSICS',False))
        if not return_corrections:result.pop('corrections')
        return result

class HybridStateStudent(nn.Module):
    """Trainable Mamba-2 + local causal attention student.

    Three Mamba-2 layers and one attention block; Mamba layers carry constant-sized
    recurrent state and attention stores only the last `attn_window` feature vectors.
    `forward` trains on sequences, `stream_step` is persistent one-tick inference.
    """
    def __init__(self, hidden=128, n_mamba=3, attn_window=16,
                 d_state=16, headdim=32, heads=4, learned_visual=False,
                 visual_input_stride=2,visual_embeddings=False,embedding_dim=48):
        super().__init__()
        if min(hidden,n_mamba,attn_window,d_state,headdim,heads)<1 or hidden % heads or (hidden * 2) % headdim:
            raise ValueError('Hybrid dimensions must be positive; heads must divide hidden and headdim must divide 2*hidden')
        self.hidden=hidden
        self.attn_window=attn_window
        self.visual_embeddings=visual_embeddings
        self.embedding_dim=embedding_dim if visual_embeddings else 48
        self.visual=nn.Identity() if visual_embeddings else VisualFeatures(learned_visual,visual_input_stride)
        self.feature=nn.Sequential(nn.Linear(2*self.embedding_dim+8,hidden),nn.Tanh())
        self.ssm=nn.ModuleList([
            _make_mamba2(d_model=hidden,d_state=d_state,headdim=headdim)
            for _ in range(n_mamba)
        ])
        self.ssm_norm=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_mamba)])
        self.attn=nn.MultiheadAttention(hidden,heads,batch_first=True)
        self.attn_norm=nn.LayerNorm(hidden)
        self.out=nn.Linear(hidden,9)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _visual_pair(self,frame,previous):
        g=self.visual(frame)
        gp=self.visual(previous)
        return g,g-gp

    def encode_tokens(self,frames,imu,observed,dt):
        """Causal backbone tokens without integrating unused physical outputs."""
        B,T=frames.shape[:2]
        from attention import local_attention,observation_ages
        g=frames if self.visual_embeddings else self.visual(frames.flatten(0,1)).reshape(B,T,self.embedding_dim)
        prior=torch.cat([g[:,:1],g[:,:-1]],dim=1)
        ages=observation_ages(observed,dt,frames.dtype)
        feats=torch.cat([g,g-prior,imu/10.,ages.unsqueeze(-1),
                         observed.to(frames.dtype).unsqueeze(-1)],dim=-1)
        # Training neural work honors AMP. Validation/PPO use amp_disabled;
        # deployment recurrence and physical integration remain FP32.
        from contextlib import nullcontext
        with neural_autocast(feats) if self.training else nullcontext():
            x=self.feature(feats)
            for block,norm in zip(self.ssm,self.ssm_norm):
                x=norm(x+block(x))
            attended=local_attention(self.attn,x,self.attn_window,getattr(cfg,'HYBRID_ATTENTION_CHUNK',512))
            return self.attn_norm(x+attended)

    def forward(self,frames,imu,observed,dt,gravity,return_corrections=False):
        from contextlib import nullcontext
        tokens=self.encode_tokens(frames,imu,observed,dt)
        with neural_autocast(tokens) if self.training else nullcontext():
            corrections=(.25*torch.tanh(self.out(tokens))).float()
        # The same physical integrator as the GRU student.
        result=integrate_corrections(corrections,imu,dt,gravity,
                                    parallel=self.training and frames.is_cuda and getattr(cfg,'PARALLEL_TRAINING_PHYSICS',False))
        if not return_corrections:
            result.pop('corrections')
        return result

    def stream_init(self,first_frame):
        """Allocate native Mamba recurrence buffers plus fixed-window attention memory.

        Do this once per new episode; stream_step must be called once per 200 Hz tick.
        Official CUDA caches update in place.
        Streaming is intended for evaluation/inference only.
        """
        B=first_frame.shape[0]
        caches=[block.allocate_inference_cache(B,max_seqlen=1) for block in self.ssm]
        return dict(ssm_caches=caches,
                    attn_cache=first_frame.new_zeros(B,0,self.hidden),
                    prev=first_frame, age=first_frame.new_zeros(B),
                    pos=first_frame.new_zeros(B,3),vel=first_frame.new_zeros(B,3),
                    R=torch.eye(3,device=first_frame.device,dtype=first_frame.dtype)[None].expand(B,-1,-1).clone())

    @torch.no_grad()
    def stream_step(self,frame,imu,observed,dt,gravity,state,initial=False):
        """Exactly one official Mamba token; attend to the latest local window."""
        age=torch.where(observed,torch.zeros_like(state['age']),state['age']+dt)
        g,diff=self._visual_pair(frame,state['prev'])
        feats=torch.cat([g,diff,imu/10.,age[:,None],observed.float()[:,None]],dim=1)
        x=self.feature(feats).unsqueeze(1)  # [B,1,H]
        next_caches=[]
        for block,norm,(conv_state,ssm_state) in zip(self.ssm,self.ssm_norm,state['ssm_caches']):
            m,next_conv,next_ssm=block.step(x,conv_state,ssm_state)
            next_caches.append((next_conv,next_ssm))
            x=norm(x+m)
        history=torch.cat([state['attn_cache'],x],dim=1)[:,-self.attn_window:]
        attended,_=self.attn(x,history,history,need_weights=False)
        out=.25*torch.tanh(self.out(self.attn_norm(x+attended).squeeze(1)))
        if initial:
            pos,vel,R=state['pos'],state['vel'],state['R']
            omega=imu[:,3:6]
        else:
            omega=imu[:,3:6]+out[:,:3]
            acc_body=imu[:,:3]+out[:,3:6]
            acc_w=(state['R']@acc_body[:,:,None]).squeeze(-1)+gravity
            dt1=dt[:,None]
            R=state['R']@so3_exp(omega*dt1)
            acc_corrected=acc_w+out[:,6:9]
            vel=state['vel']+acc_corrected*dt1
            pos=state['pos']+state['vel']*dt1+.5*acc_corrected*dt1.square()
        nxt=dict(ssm_caches=next_caches,attn_cache=history,
                 prev=frame,age=age,pos=pos,vel=vel,R=R)
        return dict(pos=pos,vel=vel,rot=R,omega=omega,corrections=out),nxt

class StateTeacher(nn.Module):
    """More expressive causal training-time teacher. Does not ship to WASM."""
    def __init__(self,hidden=128,heads=4,backbone='gru',learned_visual=False,visual_input_stride=2,visual_embeddings=False,embedding_dim=48):
        super().__init__()
        self.visual_embeddings=visual_embeddings
        self.embedding_dim=embedding_dim if visual_embeddings else 48
        self.visual=nn.Identity() if visual_embeddings else VisualFeatures(learned_visual,visual_input_stride)
        self.features=nn.Sequential(nn.Linear(2*self.embedding_dim+8,hidden),nn.GELU())
        self.backbone=backbone
        if backbone=='gru':
            self.gru=nn.GRU(hidden,hidden,num_layers=2,batch_first=True)
        elif backbone=='mamba_hybrid':
            # Official parallel CUDA scans; this auxiliary teacher stays on the host.
            self.mamba=nn.ModuleList([
                _make_mamba2(d_model=hidden,d_state=64,headdim=32) for _ in range(3)])
            self.mamba_norm=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        else:
            raise ValueError('backbone must be gru or mamba_hybrid')
        self.attn=nn.MultiheadAttention(hidden,heads,batch_first=True)
        self.layernorm=nn.LayerNorm(hidden)
        self.out=nn.Linear(hidden,9)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,frames,imu,observed,dt,gravity):
        B,T=frames.shape[:2]
        pooled=frames if self.visual_embeddings else self.visual(frames.flatten(0,1)).reshape(B,T,self.embedding_dim)
        prev=torch.cat([pooled[:,:1],pooled[:,:-1]],dim=1)
        age=torch.zeros_like(observed,dtype=frames.dtype)
        for t in range(1,T):
            age[:,t]=torch.where(observed[:,t],torch.zeros_like(age[:,t]),age[:,t-1]+dt)
        x=torch.cat([pooled,pooled-prev,imu/10.,age.unsqueeze(-1),observed.float().unsqueeze(-1)],-1)
        x=self.features(x)
        if self.backbone=='gru':
            x,_=self.gru(x)
        else:
            for block,norm in zip(self.mamba,self.mamba_norm):
                x=norm(x+block(x))
        causal_mask=torch.ones(T,T,device=frames.device,dtype=torch.bool).triu(1)
        a,_=self.attn(x,x,x,attn_mask=causal_mask,need_weights=False)
        corr=.25*torch.tanh(self.out(self.layernorm(x+a)))
        # Integrate outputs of teacher through the same physical model.
        return integrate_corrections(corr,imu,dt,gravity,
                                    parallel=self.training and frames.is_cuda and getattr(cfg,'PARALLEL_TRAINING_PHYSICS',False))

@fp32_physics
def integrate_corrections(corr,imu,dt,gravity,*,parallel=False):
    corr,imu,dt,gravity=corr.float(),imu.float(),dt.float(),gravity.float()
    B,T,_=corr.shape
    if parallel and T>1:
        # Differentiable associative prefix products: chronological SO(3)
        # multiplication is unchanged, with a different FP32 reduction order.
        gyro=imu[:,1:,3:6]+corr[:,1:,:3]
        rotations=so3_exp((gyro*dt[:,None,None]).reshape(-1,3)).reshape(B,T-1,3,3)
        offset=1
        while offset<T-1:
            rotations=torch.cat([rotations[:,:offset],rotations[:,:-offset]@rotations[:,offset:]],1)
            offset*=2
        identity=torch.eye(3,device=corr.device,dtype=corr.dtype)[None,None].expand(B,1,3,3)
        rotation=torch.cat([identity,rotations],1)
        acceleration=(rotation[:,:-1]@(imu[:,1:,:3]+corr[:,1:,3:6])[...,None]).squeeze(-1)
        acceleration=acceleration+gravity[:,None]+corr[:,1:,6:9]
        zero=corr.new_zeros(B,1,3)
        velocity=torch.cat([zero,(acceleration*dt[:,None,None]).cumsum(1)],1)
        increments=velocity[:,:-1]*dt[:,None,None]+.5*acceleration*dt[:,None,None].square()
        position=torch.cat([zero,increments.cumsum(1)],1)
        return dict(pos=position,vel=velocity,rot=rotation,
                    omega=torch.cat([imu[:,:1,3:6],gyro],1),corrections=corr)
    pos=corr.new_zeros(B,3);vel=corr.new_zeros(B,3)
    R=torch.eye(3,device=corr.device,dtype=corr.dtype)[None].expand(B,-1,-1).clone()
    pp=[pos];vv=[vel];rr=[R];ww=[imu[:,0,3:6]]
    for t in range(1,T):
        gyro=imu[:,t,3:6]+corr[:,t,:3]
        acc=(R@(imu[:,t,:3]+corr[:,t,3:6])[:,:,None]).squeeze(-1)+gravity
        dt1=dt[:,None]
        Rnext=R@so3_exp(gyro*dt1)
        acc_corrected=acc+corr[:,t,6:9]
        vnext=vel+acc_corrected*dt1
        pnext=pos+vel*dt1+0.5*acc_corrected*dt1.square()
        R,pos,vel=Rnext,pnext,vnext
        pp.append(pos);vv.append(vel);rr.append(R);ww.append(gyro)
    return dict(pos=torch.stack(pp,1),vel=torch.stack(vv,1),rot=torch.stack(rr,1),
                omega=torch.stack(ww,1),corrections=corr)
