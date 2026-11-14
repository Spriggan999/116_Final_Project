"""VIOBench Python guest v1. Bundled by the official `viobench bundle` command.

Host deps: numpy and cv2, both included in VIOBench's Python runtime.
No torch, scipy, openpyxl, zlib, filesystem access or external inference services.

Integrates learned corrections in IMU axes and converts pose and twist to the
first/current camera frames, including the camera lever arm. Training and camera
appearance/calibration still differ from the evaluation distribution.
"""
import math
import cv2
import numpy as np
import solution_bundle as bundle


def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def layer_norm(x, w, b, eps=1e-5):
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / np.sqrt((x*x).mean(axis=-1, keepdims=True) + eps)
    return x * w + b


def matrix_to_quat(R):
    """Return xyzw; avoids scipy, normalizes to handle numeric drift."""
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(max(1e-12, 1.0 + trace)) * 2.0
        q = [(R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s,
             (R[1,0]-R[0,1])/s, 0.25*s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i+1)%3, (i+2)%3
        s = math.sqrt(max(1e-12, 1.0 + R[i,i]-R[j,j]-R[k,k]))*2.0
        qq = [0.,0.,0.,0.]
        qq[i] = 0.25*s
        qq[j] = (R[i,j]+R[j,i])/s
        qq[k] = (R[i,k]+R[k,i])/s
        qq[3] = (R[k,j]-R[j,k])/s
        q = qq
    q = np.asarray(q, np.float64)
    return q / max(np.linalg.norm(q), 1e-12)


def so3_exp(w):
    theta2 = float(np.dot(w, w))
    theta = math.sqrt(theta2)
    if theta2 < 1e-6:
        a = 1.0 - theta2/6.0 + theta2*theta2/120.0
        b = 0.5 - theta2/24.0 + theta2*theta2/720.0
    else:
        a = math.sin(theta)/theta
        b = (1.0 - math.cos(theta))/theta2
    x,y,z = w
    K=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]],np.float64)
    return np.eye(3)+a*K+b*(K@K)


def adaptive_average_pool(gray, height, width):
    """PyTorch adaptive_avg_pool2d bins, including overlapping edge bins."""
    h, w = gray.shape
    if h % height == 0 and w % width == 0:
        return gray.reshape(height, h//height, width, w//width).mean(axis=(1, 3))
    result = np.empty((height, width), dtype=np.float32)
    for i in range(height):
        y0, y1 = i*h//height, ((i+1)*h+height-1)//height
        for j in range(width):
            x0, x1 = j*w//width, ((j+1)*w+width-1)//width
            result[i, j] = gray[y0:y1, x0:x1].mean()
    return result


class Solution:
    # Full-resolution teacher. export.py substitutes saved legacy resolutions.
    frame_width = 640

    def __init__(self, config):
        import json
        required = ("deployment.json", "state_student.npz")
        missing = [name for name in required if not bundle.exists(name)]
        if missing:
            raise FileNotFoundError(
                "Missing HyperDrone deployment assets: " + ", ".join(missing) +
                ". Run 1_generative_ssm/train.py or export.py first.")
        with np.load(bundle.open('state_student.npz'), allow_pickle=False) as archive:
            self.d = dict(archive)
        flags = json.loads(bundle.read_text('deployment.json'))
        if flags['backbone'] not in ('gru', 'mamba2_hybrid'):
            raise ValueError("Unsupported state backbone: " + str(flags['backbone']))
        if flags['frame_mode'] not in ('repeat', 'affine', 'flow','embedding'):
            raise ValueError("Unsupported frame mode: " + str(flags['frame_mode']))
        self.generator_mode = flags['frame_mode']
        self.embedding_dim = int(flags.get('embedding_dim',48)) if self.generator_mode=='embedding' else 48
        self.use_generated = bool(flags['use_generated_frames'])
        self.backbone = flags['backbone']
        self.learned_visual = bool(flags['learned_visual'])
        self.hidden = int(flags['hidden'])
        self.w = int(flags['width']); self.h = int(flags['height'])
        self.max_flow_pixels = float(flags.get('max_flow_pixels',24.))
        self.num_mamba = int(flags['mamba_layers'])
        self.attn_window = int(flags['attention_window'])
        self.d_state = int(flags['hybrid_d_state'])
        self.headdim = int(flags['hybrid_head_dim'])
        self.attention_heads = int(flags.get('hybrid_attention_heads', 4))
        self.nheads = (2*self.hidden)//self.headdim if self.backbone == 'mamba2_hybrid' else 0

        # Model 1: direct embedding predictor, optional raw-field CNN or affine MLP.
        self.frame_d = {}
        if self.generator_mode in ('affine','embedding'):
            with np.load(bundle.open('frame_student.npz'), allow_pickle=False) as archive:
                self.frame_d = dict(archive)
        self.frame_net = None
        if self.generator_mode == 'flow':
            self.frame_net = cv2.dnn.readNetFromONNX(
                np.frombuffer(bundle.read_bytes('frame_raw.onnx'),dtype=np.uint8))
        self.visual_net = None
        if self.learned_visual:
            self.visual_net = cv2.dnn.readNetFromONNX(
                np.frombuffer(bundle.read_bytes('visual.onnx'),dtype=np.uint8))

        extrinsic = np.asarray(config['T_cam_imu'],dtype=np.float64)
        self.R_ci = extrinsic[:3,:3]  # IMU frame -> camera frame
        self.r_ic = -self.R_ci.T @ extrinsic[:3,3]  # camera centre in IMU axes
        self.omega = np.zeros(3, dtype=np.float64)
        self.gravity_ready = False
        self.gravity = np.zeros(3,dtype=np.float64)
        self.last_real = None
        self.prev_real = None
        self.last_real_t = None
        self.prev_real_t = None
        self.prev_visual = None
        self.age = 0.
        self.pos = np.zeros(3,dtype=np.float64)
        self.vel = np.zeros(3,dtype=np.float64)
        self.R = np.eye(3,dtype=np.float64)
        self.hstate = np.zeros(self.hidden,np.float32)
        self.mamba_states=[]
        self.attn_cache=[]
        if self.backbone == 'mamba2_hybrid':
            for _ in range(self.num_mamba):
                d_ssm=2*self.hidden
                self.mamba_states.append([
                    np.zeros((d_ssm+2*self.d_state,4),np.float32),
                    np.zeros((self.nheads,self.headdim,self.d_state),np.float32)])

    def wgt(self, key):
        return self.d[key]

    def prepare_frame(self, frame):
        # [3,H,W] float32 in [0,1], same ordering as training.
        img = frame if frame.shape[:2] == (self.h,self.w) else cv2.resize(
            frame,(self.w,self.h),interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(img.transpose(2,0,1).astype(np.float32)/255.)

    def features(self, frame):
        if self.generator_mode=='embedding' and frame.ndim==1:
            return frame
        if self.visual_net is not None:
            self.visual_net.setInput(frame[None])
            return np.asarray(self.visual_net.forward(),np.float32).reshape(self.embedding_dim)
        return adaptive_average_pool(frame.mean(axis=0), 6, 8).reshape(-1)

    def _frame_affine(self, previous, anchor, imu, elapsed, prev_elapsed):
        # Affine model baseline: previous/anchor 8x8 mean features.
        a=adaptive_average_pool(previous.mean(axis=0), 8, 8).reshape(-1)
        b=adaptive_average_pool(anchor.mean(axis=0), 8, 8).reshape(-1)
        x=np.concatenate([a,b,imu/10,imu/10*elapsed,[elapsed,prev_elapsed]]).astype(np.float32)
        for i in (0,2,4):
            x=self.frame_d[f'net.{i}.weight']@x + self.frame_d[f'net.{i}.bias']
            if i!=4: x=np.tanh(x)
        dxdy=np.tanh(x[:2])*elapsed*3.
        theta=float(np.tanh(x[2])*elapsed*3.)
        c,s=math.cos(theta),math.sin(theta)
        X,Y=np.meshgrid(np.arange(self.w,dtype=np.float32),np.arange(self.h,dtype=np.float32))
        xx=(X+.5)*2/self.w-1
        yy=(Y+.5)*2/self.h-1
        sx=(c*xx-s*yy+dxdy[0]+1)*self.w/2-.5
        sy=(s*xx+c*yy+dxdy[1]+1)*self.h/2-.5
        out=cv2.remap(anchor.transpose(1,2,0),sx.astype('float32'),sy.astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        return np.ascontiguousarray(out.transpose(2,0,1))

    def _frame_flow(self, previous, anchor, imu, elapsed, prev_elapsed):
        yy,xx=np.meshgrid((np.arange(self.h)+.5)*2/self.h-1,(np.arange(self.w)+.5)*2/self.w-1,indexing='ij')
        coords=np.stack([xx,yy]).astype(np.float32)
        inp=np.concatenate([previous,anchor,anchor-previous,coords],axis=0)[None]
        cond=np.concatenate([imu/10,[elapsed,prev_elapsed]]).astype(np.float32)[None]
        self.frame_net.setInput(np.ascontiguousarray(inp),'image')
        self.frame_net.setInput(np.ascontiguousarray(cond),'condition')
        raw=np.asarray(self.frame_net.forward(),np.float32).reshape(6,self.h,self.w)
        ts=min(2.,max(0.,elapsed/.05))
        flow=self.max_flow_pixels*np.tanh(raw[:2])*ts
        X,Y=np.meshgrid(np.arange(self.w,dtype=np.float32),np.arange(self.h,dtype=np.float32))
        warped=cv2.remap(anchor.transpose(1,2,0),X+flow[0],Y+flow[1],
                         cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).transpose(2,0,1)
        mask=sigmoid(raw[2]-2.)[None]
        generated=np.clip(anchor+.25*np.tanh(raw[3:6]),0.,1.)
        return np.clip((1-mask)*warped+mask*generated,0.,1.).astype(np.float32)

    def synthetic(self, imu_mean, elapsed, prev_elapsed):
        if not self.use_generated or self.prev_real is None:
            return self.last_real
        if self.generator_mode=='flow':
            return self._frame_flow(self.prev_real,self.last_real,imu_mean,elapsed,prev_elapsed)
        if self.generator_mode=='affine':
            return self._frame_affine(self.prev_real,self.last_real,imu_mean,elapsed,prev_elapsed)
        if self.generator_mode=='embedding':
            x=np.concatenate([self.prev_real,self.last_real,self.last_real-self.prev_real,
                              imu_mean/10.,[elapsed,prev_elapsed]]).astype(np.float32)
            x=np.tanh(self.frame_d['predictor.0.weight']@x+self.frame_d['predictor.0.bias'])
            residual=self.frame_d['predictor.2.weight']@x+self.frame_d['predictor.2.bias']
            return (self.last_real+.25*np.tanh(residual)*np.clip(elapsed/.05,0.,2.)).astype(np.float32)
        return self.last_real

    def _gru_update(self, x):
        # This is the exact GRUCell update (PyTorch gate order r,z,n).
        Wih=self.wgt('gru.weight_ih'); Whh=self.wgt('gru.weight_hh')
        v=Wih@x+self.wgt('gru.bias_ih')
        q=Whh@self.hstate+self.wgt('gru.bias_hh')
        r=sigmoid(v[:self.hidden]+q[:self.hidden])
        z=sigmoid(v[self.hidden:2*self.hidden]+q[self.hidden:2*self.hidden])
        n=np.tanh(v[2*self.hidden:]+r*q[2*self.hidden:])
        self.hstate=((1-z)*n+z*self.hstate).astype(np.float32)
        return self.hstate

    def _mamba_update(self, token):
        H=self.hidden
        for i,(conv,ssm) in enumerate(self.mamba_states):
            prefix=f'ssm.{i}.'
            x0=self.wgt(prefix+'in_proj.weight')@token
            ns=self.d_state; nh=self.nheads; inner=2*H; cd=inner+2*ns
            z=x0[:inner]; xbc=x0[inner:inner+cd]; dt=x0[inner+cd:]
            conv[:,:-1]=conv[:,1:]
            conv[:,-1]=xbc
            xbc=silu((conv*self.wgt(prefix+'conv1d.weight')[:,0,:]).sum(axis=-1)+
                     self.wgt(prefix+'conv1d.bias'))
            xx=xbc[:inner].reshape(nh,self.headdim)
            B=xbc[inner:inner+ns]; C=xbc[inner+ns:]
            dt=softplus(dt+self.wgt(prefix+'dt_bias'))
            a=-np.exp(self.wgt(prefix+'A_log'))
            ssm*=np.exp(dt*a)[:,None,None]
            ssm+=(dt[:,None,None]*xx[:,:,None]*B[None,None,:]).astype(np.float32)
            y=np.sum(ssm*C[None,None,:],axis=-1)+self.wgt(prefix+'D')[:,None]*xx
            y=y.reshape(-1)*silu(z)
            y=y / math.sqrt(max(1e-12,float(np.mean(y*y))+1e-5))
            y=y*self.wgt(prefix+'norm.weight')
            y=self.wgt(prefix+'out_proj.weight')@y
            token=layer_norm(token+y, self.wgt(f'ssm_norm.{i}.weight'),self.wgt(f'ssm_norm.{i}.bias'))
        self.attn_cache.append(token.copy())
        self.attn_cache=self.attn_cache[-self.attn_window:]
        history=np.stack(self.attn_cache)
        # MultiheadAttention: q from current token, keys/values from history.
        pW=self.wgt('attn.in_proj_weight'); pb=self.wgt('attn.in_proj_bias')
        q=pW[:H]@token+pb[:H]
        k=history@pW[H:2*H].T+pb[H:2*H]
        v=history@pW[2*H:].T+pb[2*H:]
        nheads=self.attention_heads; d=H//nheads
        Q=q.reshape(nheads,d);K=k.reshape(-1,nheads,d).transpose(1,0,2)
        V=v.reshape(-1,nheads,d).transpose(1,0,2)
        scores=np.einsum('hd,hld->hl',Q,K)/math.sqrt(d)
        scores-=scores.max(axis=-1,keepdims=True)
        weights=np.exp(scores);weights/=weights.sum(axis=-1,keepdims=True)
        out=np.einsum('hl,hld->hd',weights,V).reshape(H)
        out=self.wgt('attn.out_proj.weight')@out+self.wgt('attn.out_proj.bias')
        return layer_norm(token+out,self.wgt('attn_norm.weight'),self.wgt('attn_norm.bias'))

    def tick(self, frame, imu_raw, dt, observed, initial):
        # frame: 48D embedding or legacy [3,H,W]; imu_raw: 6 IMU-axis values.
        g=self.features(frame).astype(np.float32)
        if self.prev_visual is None: self.prev_visual=g.copy()
        self.age=0. if observed else self.age+dt
        features=np.concatenate([g,g-self.prev_visual,imu_raw.astype(np.float32)/10,
                                 [self.age,1. if observed else 0.]]).astype(np.float32)
        x=np.tanh(self.wgt('feature.0.weight')@features + self.wgt('feature.0.bias'))
        if self.backbone=='gru':
            hidden=self._gru_update(x)
        else:
            hidden=self._mamba_update(x)
        correction=.25*np.tanh(self.wgt('out.weight')@hidden+self.wgt('out.bias'))
        self.prev_visual=g
        if initial:
            return np.zeros(3),np.zeros(3)
        # The notebook learns corrections in body/IMU axes. Integrate there,
        # then transform the complete pose/twist to the ABI camera frame.
        gyro=imu_raw[3:]+correction[:3]
        acc=imu_raw[:3]+correction[3:6]
        a_world=self.R@acc + self.gravity + correction[6:9]
        p_new=self.pos+self.vel*dt+.5*a_world*dt*dt
        self.vel=self.vel+a_world*dt
        self.pos=p_new
        self.R=self.R@so3_exp(gyro*dt)
        return gyro,self.vel

    def _state(self):
        rotation = self.R_ci @ self.R @ self.R_ci.T
        position = self.R_ci @ (self.pos + (self.R - np.eye(3)) @ self.r_ic)
        velocity = self.R_ci @ (self.R.T @ self.vel + np.cross(self.omega, self.r_ic))
        omega = self.R_ci @ self.omega
        state = np.concatenate([position, matrix_to_quat(rotation), velocity, omega])
        if state.shape != (13,) or not np.isfinite(state).all():
            raise ValueError('Nonfinite or incorrectly sized camera state')
        return state

    def _gravity(self, data):
        if not self.gravity_ready and len(data):
            # Same magnitude/direction convention as StateWindows; assumes a
            # near-static start. Moving-start gravity remains an estimator limit.
            acceleration = np.clip(data[:, 1:4].mean(axis=0), -20., 20.)
            norm = np.linalg.norm(acceleration)
            if norm > 1e-5:
                self.gravity = -9.81 * acceleration / norm
                self.gravity_ready = True

    def step(self, t, frame, imu):
        data = np.asarray(imu, dtype=np.float64).reshape(-1, 7)
        t = float(t)
        if not np.isfinite(t) or not np.isfinite(data).all():
            raise ValueError('Nonfinite timestamp or IMU sample')
        newest = self.prepare_frame(frame)
        if self.generator_mode=='embedding':newest=self.features(newest).astype(np.float32)
        if self.last_real is None:
            self.last_real = newest
            self.prev_real = newest.copy()
            self.prev_real_t = t if self.generator_mode=='embedding' else t-.05
            self.last_real_t = t
            self._gravity(data)
            sample = data[-1, 1:] if len(data) else np.zeros(6)
            self.tick(newest, sample, 0., True, True)
            return self._state()
        if t <= self.last_real_t:
            raise ValueError('Frame timestamps must increase')
        if len(data) and (np.any(np.diff(data[:, 0]) <= 0) or
                          data[0, 0] <= self.last_real_t or data[-1, 0] > t + 1e-8):
            raise ValueError('IMU timestamps must increase within the frame interval')
        self._gravity(data)
        prev_dt = max(0. if self.generator_mode=='embedding' else 1e-3,self.last_real_t-self.prev_real_t)
        prior_t = self.last_real_t
        imu_sum = np.zeros(6, np.float64)
        for i, row in enumerate(data):
            ti = float(row[0])
            dt = ti - prior_t
            prior_t = ti
            imu_sum += row[1:]
            observed = i == len(data)-1 and abs(t-ti) < 1e-8
            virtual = newest if observed else self.synthetic(
                imu_sum/(i+1), ti-self.last_real_t, prev_dt)
            self.omega, _ = self.tick(virtual, row[1:], dt, observed, False)
        if len(data) and t-prior_t > 1e-8:
            self.omega, _ = self.tick(newest, data[-1, 1:], t-prior_t, True, False)
        if not len(data):
            # Keep the last physical estimate; still retain this real frame.
            self.prev_visual = self.features(newest).astype(np.float32)
            self.age = 0.
        self.prev_real, self.last_real = self.last_real, newest
        self.prev_real_t, self.last_real_t = self.last_real_t, t
        return self._state()
