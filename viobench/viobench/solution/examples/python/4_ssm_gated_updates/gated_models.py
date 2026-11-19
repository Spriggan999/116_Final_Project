"""IMU-only GRU recurrence, sparse vision, and four controlled JEPA ablations."""
import copy
import legacy
import torch
from torch import nn
import torch.nn.functional as F
from models import VisualFeatures, integrate_corrections, so3_exp
from performance import neural_autocast, amp_disabled
import config as cfg

NAMES = ('baseline', 'aux_jepa', 'innovation', 'combined')
PROFILES = {name: dict(profile=name, hidden=256, embedding_dim=48,
                      kind='gated_gru', fuel=0, teacher_style=False,
                      auxiliary=name in ('aux_jepa', 'combined'),
                      innovation=name in ('innovation', 'combined')) for name in NAMES}
ALL_PROFILES = PROFILES


class GatedModel(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.settings = dict(settings)
        self.architecture = settings.get('architecture', 'legacy')
        if self.architecture not in ('legacy','persistent_visual_v2'):raise ValueError('Unknown gated architecture')
        h, e = settings['hidden'], settings['embedding_dim']
        self.vision = VisualFeatures(True, 8, e)
        self.vision.feature_dropout = 0.
        self.target = copy.deepcopy(self.vision).requires_grad_(False).eval()
        self.gru = nn.GRU(6, h, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(h+e, h), nn.Tanh(), nn.Linear(h, 9))
        self.predictor = nn.Sequential(nn.Linear(3*e+8, 256), nn.Tanh(), nn.Linear(256, e))
        self.auxiliary = (nn.Sequential(nn.Linear(h, 256), nn.Tanh(), nn.Linear(256, e))
                          if settings['auxiliary'] else None)
        self.innovation = (nn.Sequential(nn.Linear(h+e, h), nn.Tanh(), nn.Linear(h, 9))
                           if settings['innovation'] else None)
        if self.innovation is not None:
            nn.init.zeros_(self.innovation[-1].weight)
            nn.init.zeros_(self.innovation[-1].bias)

        if self.architecture == 'persistent_visual_v2':
            self.visual_gru = nn.GRUCell(3*e+1, h)
            self.velocity_update = nn.Linear(h, 3)
            self.orientation_update = nn.Linear(h, 3)
            for head in (self.velocity_update, self.orientation_update):
                nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def train(self, mode=True):
        super().train(mode)
        self.target.eval()
        return self

    @torch.no_grad()
    def update_target(self, decay=.99):
        for target, online in zip(self.target.parameters(), self.vision.parameters()):
            target.lerp_(online, 1-decay)

    def predict(self, previous, anchor, imu_mean, elapsed, previous_span):
        x = torch.cat((previous, anchor, anchor-previous, imu_mean/10,
                       elapsed, previous_span), -1)
        return anchor + .25*torch.tanh(self.predictor(x))*(elapsed/.05).clamp(0, 2)

    def encode_captures(self, encoder, images):
        flat = images.flatten(0,1)
        if flat.dtype != torch.uint8:
            return encoder(flat)
        from torch.utils.checkpoint import checkpoint
        def encode(chunk):
            return encoder(chunk.float()/255.)
        parts=[]
        for chunk in flat.split(max(1,int(cfg.VISUAL_CHUNK))):
            if encoder.training and torch.is_grad_enabled():
                # Retain shared uint8 inputs, including during backward.
                parts.append(checkpoint(encode,chunk,use_reentrant=False))
            else:
                parts.append(encode(chunk))
        return torch.cat(parts)

    def forward(self, images, imu, dt, gravity, ticks, observed, **unused):
        if self.training:
            return self._forward(images, imu, dt, gravity, ticks, observed)
        # Evaluation must agree with persistent FP32 streaming inference.
        with amp_disabled():
            return self._forward(images, imu, dt, gravity, ticks, observed)

    def _forward(self, images, imu, dt, gravity, ticks, observed):
        if self.architecture == "persistent_visual_v2":
            return self._forward_persistent(images, imu, dt, gravity, ticks, observed)
        b, t, _ = imu.shape
        k = images.shape[1]
        e = self.settings['embedding_dim']
        online = self.encode_captures(self.vision,images).reshape(b, k, e)
        with neural_autocast(imu):
            hidden, _ = self.gru(imu/10)
            # Prefix sums give interval IMU means without a Python tick loop.
            sums = torch.cat((imu.new_zeros(b, 1, 6), imu.cumsum(1)), 1)
            prior_ticks = torch.cat((ticks[:, :1], ticks[:, :-1]), 1)
            delta = (ticks-prior_ticks).clamp_min(1)
            index = torch.arange(b, device=imu.device)[:, None]
            means = (sums[index, ticks+1]-sums[index, prior_ticks+1])/delta[..., None]
            elapsed = (ticks-prior_ticks).float()*dt[:, None]
            spans = torch.cat((elapsed[:, :1]*0, elapsed[:, :-1]), 1)
            anchor = torch.cat((online[:, :1], online[:, :-1]), 1)
            previous = torch.cat((anchor[:, :1], anchor[:, :-1]), 1)
            predicted = self.predict(previous, anchor, means, elapsed[..., None], spans[..., None])
            valid = torch.ones((b, k), device=imu.device, dtype=torch.bool)
            valid[:, 0] = False
            sparse = online.new_zeros(b, t, e).scatter(1, ticks[..., None].expand(-1, -1, e), online)
            raw = self.fusion(torch.cat((hidden, sparse), -1))
            innovation = online-predicted
            residual = raw.new_zeros(b, t, 9)
            if self.innovation is not None:
                real_hidden = hidden[index, ticks]
                update = self.innovation(torch.cat((real_hidden, innovation), -1))*valid[..., None]
                residual = residual.scatter(1, ticks[..., None].expand(-1, -1, 9), update)
                raw = raw+residual
            corrections = (.25*torch.tanh(raw)).float()
            auxiliary = self.auxiliary(hidden[index, ticks]).float() if self.auxiliary is not None else None
        result = integrate_corrections(corrections, imu, dt, gravity,
                                      parallel=imu.is_cuda)
        # Initial state is identity pose and zero twist, as in the benchmark.
        result['omega'] = torch.cat((imu.new_zeros(b, 1, 3), result['omega'][:, 1:]), 1)
        result.update(online=online, predicted=predicted.float(), auxiliary=auxiliary,
                      valid=valid, sparse_visual=sparse, innovation_update=residual.float())
        return result

    def jepa_loss(self, prediction, images):
        b, k = images.shape[:2]
        with torch.no_grad(), amp_disabled():
            target = self.encode_captures(self.target,images).reshape(b, k, -1)
        mask = prediction['valid']
        online = prediction['online'].float()
        zero = online.sum()*0
        pred = F.smooth_l1_loss(prediction['predicted'][mask], target[mask], beta=.1) if mask.any() else zero
        content = F.smooth_l1_loss(online, target, beta=.1)
        values = online.flatten(0, 1)
        centered = values-values.mean(0)
        variance = F.relu(.1-torch.sqrt(centered.square().mean(0)+1e-4)).mean()
        covariance = centered.T@centered/max(1, len(values)-1)
        covariance = (covariance-torch.diag_embed(covariance.diagonal())).square().sum()/values.shape[-1]
        auxiliary = (F.smooth_l1_loss(prediction['auxiliary'][mask], target[mask], beta=.1)
                     if self.auxiliary is not None and mask.any() else zero)
        return dict(embedding_prediction=pred, embedding_content=content,
                    embedding_variance=variance, embedding_covariance=covariance,
                    auxiliary_prediction=auxiliary)

    def stream_init(self, device=None):
        device = device or next(self.parameters()).device
        h, e = self.settings['hidden'], self.settings['embedding_dim']
        z = torch.zeros(1, 3, device=device)
        return dict(h=torch.zeros(1, 1, h, device=device), pos=z.clone(), vel=z.clone(),
                    rot=torch.eye(3, device=device)[None], anchor=torch.zeros(1, e, device=device),
                    previous=torch.zeros(1, e, device=device), imu_sum=torch.zeros(1, 6, device=device),
                    count=0, age=0., span=0., captures=0, initial=True)

    @torch.no_grad()
    @amp_disabled()
    def stream_step(self, image, imu, dt, gravity, state):
        """One tick; image is [1,3,H,W] or None. No state reset at camera gaps."""
        if self.architecture == "persistent_visual_v2":
            return self._persistent_step(image, imu, dt, gravity, state)
        s = dict(state)
        hidden, s['h'] = self.gru(imu[:, None]/10, s['h'])
        hidden = hidden[:, 0]
        visual = torch.zeros_like(s['anchor'])
        update = imu.new_zeros(1, 9)
        if not s['initial']:
            s['imu_sum'] = s['imu_sum']+imu
            s['count'] += 1
            s['age'] += float(dt)
        if image is not None:
            visual = self.vision(image)
            if s['captures'] and self.innovation is not None:
                predicted = self.predict(s['previous'], s['anchor'], s['imu_sum']/max(1, s['count']),
                                         imu.new_tensor([[s['age']]]), imu.new_tensor([[s['span']]]))
                update = self.innovation(torch.cat((hidden, visual-predicted), -1))
            s['previous'] = s['anchor'] if s['captures'] else visual
            s['anchor'] = visual
            s['span'], s['age'], s['count'] = s['age'], 0., 0
            s['imu_sum'] = torch.zeros_like(imu)
            s['captures'] += 1
        correction = .25*torch.tanh(self.fusion(torch.cat((hidden, visual), -1))+update)
        omega = imu[:, 3:]+correction[:, :3]
        if not s['initial']:
            acceleration = (s['rot']@(imu[:, :3]+correction[:, 3:6])[..., None]).squeeze(-1)+gravity+correction[:, 6:]
            s['pos'] = s['pos']+s['vel']*dt+.5*acceleration*dt**2
            s['vel'] = s['vel']+acceleration*dt
            s['rot'] = s['rot']@so3_exp(omega*dt)
        else:
            omega = torch.zeros_like(omega)
        s['initial'] = False
        return dict(pos=s['pos'], vel=s['vel'], rot=s['rot'], omega=omega,
                    corrections=correction), s

    def _visual_segment(self, imu, hidden, visual, anchor, previous, mean, elapsed, span, capture, bootstrap):
        """Neural work between captures; safe to checkpoint as one small segment."""
        with neural_autocast(imu):
            sequence, hidden = self.gru(imu/10, hidden)
            # IMU-only state at this tick, before the current capture is
            # assimilated; the auxiliary JEPA head must not see its own target.
            imu_hidden = hidden[0]
            predicted = self.predict(previous, anchor, mean, elapsed, span)
            vu = imu.new_zeros(len(imu), 3)
            ru = vu.clone()
            innovation = imu.new_zeros(len(imu), imu.shape[1], 9)
            sparse = imu.new_zeros(len(imu), imu.shape[1], visual.shape[-1])
            if capture:
                sparse = torch.cat((sparse[:, :-1], visual[:, None]), 1)
                if not bootstrap:
                    features = torch.cat((visual, anchor, visual-anchor, elapsed), -1)
                    assimilated = self.visual_gru(features, hidden[0])
                    hidden = assimilated[None]
                    sequence = torch.cat((sequence[:, :-1], assimilated[:, None]), 1)
                    vu = self.settings.get('velocity_update_bound', .5)*torch.tanh(self.velocity_update(assimilated))
                    ru = self.settings.get('orientation_update_bound', .02)*torch.tanh(self.orientation_update(assimilated))
                    if self.innovation is not None:
                        residual = self.innovation(torch.cat((assimilated, visual-predicted), -1))
                        innovation = torch.cat((innovation[:, :-1], residual[:, None]), 1)
            raw = self.fusion(torch.cat((sequence, sparse), -1))+innovation
            corr = .25*torch.tanh(raw)
            aux = self.auxiliary(imu_hidden) if self.auxiliary is not None else visual*0
        return tuple(v.float() for v in (corr, hidden, predicted, aux, vu, ru, innovation))

    def _forward_persistent(self, images, imu, dt, gravity, ticks, observed):
        # Different capture grids are valid; process each grid separately.
        if len(imu)>1 and not torch.equal(ticks, ticks[:1].expand_as(ticks)):
            rows=[self._forward_persistent(images[i:i+1],imu[i:i+1],dt[i:i+1],gravity[i:i+1],
                                          ticks[i:i+1],observed[i:i+1]) for i in range(len(imu))]
            return {key: None if rows[0][key] is None else torch.cat([r[key] for r in rows],0) for key in rows[0]}
        from torch.utils.checkpoint import checkpoint
        from functools import partial
        b,t,_=imu.shape;k=images.shape[1];h=self.settings['hidden'];e=self.settings['embedding_dim']
        capture_ticks=ticks[0].tolist()
        if not capture_ticks or capture_ticks[0]!=0:raise ValueError('Persistent windows require an initial capture')
        online=self.encode_captures(self.vision, images).reshape(b,k,e).float()
        hidden=imu.new_zeros(1,b,h);pos=imu.new_zeros(b,3);vel=pos.clone()
        rotation=torch.eye(3,device=imu.device).expand(b,3,3)
        pieces={key:[] for key in ('pos','vel','rot','omega','corrections','innovation_update',
                                   'visual_velocity_update','visual_orientation_update')}
        predicted=[];auxiliary=[];start=0;previous_tick=0;previous_span=imu.new_zeros(b,1)
        anchor=online[:,0];previous=anchor
        boundaries=[(tick,slot) for slot,tick in enumerate(capture_ticks)]
        if capture_ticks[-1]<t-1:boundaries.append((t-1,None))
        for end,slot in boundaries:
            capture=slot is not None;bootstrap=start==0
            visual=online[:,slot] if capture else anchor*0
            elapsed=(end-previous_tick)*dt[:,None]
            mean=imu[:,previous_tick+1:end+1].mean(1)/10 if end>previous_tick else imu[:,0]*0
            operation=partial(self._visual_segment,capture=capture,bootstrap=bootstrap)
            arguments=(imu[:,start:end+1],hidden,visual,anchor,previous,mean,elapsed,previous_span)
            values=checkpoint(operation,*arguments,use_reentrant=False) if self.training and torch.is_grad_enabled() else operation(*arguments)
            corr,hidden,pred,aux,vu,ru,innovation=values
            if bootstrap:
                pp=pos[:,None];vv=vel[:,None];rr=rotation[:,None];ww=imu.new_zeros(b,1,3)
            else:
                # The relative integrator skips its initialized sample.
                segment_imu=torch.cat((imu[:,start:start+1],imu[:,start:end+1]),1)
                segment_corr=torch.cat((corr[:,:1]*0,corr),1)
                segment_corr=torch.cat((segment_corr[...,:6],(rotation[:,None].transpose(-1,-2)@segment_corr[...,6:,None]).squeeze(-1)), -1)
                local_gravity=(rotation.transpose(-1,-2)@gravity[...,None]).squeeze(-1)
                local=integrate_corrections(segment_corr,segment_imu,dt,local_gravity,parallel=imu.is_cuda)
                times=torch.arange(1,end-start+2,device=imu.device)[None,:,None]*dt[:,None,None]
                pp=pos[:,None]+vel[:,None]*times+(rotation[:,None]@local['pos'][:,1:,...,None]).squeeze(-1)
                vv=vel[:,None]+(rotation[:,None]@local['vel'][:,1:,...,None]).squeeze(-1)
                rr=rotation[:,None]@local['rot'][:,1:]
                ww=local['omega'][:,1:]
                if capture:
                    vv=torch.cat((vv[:,:-1],(vv[:,-1]+vu)[:,None]),1)
                    rr=torch.cat((rr[:,:-1],(rr[:,-1]@so3_exp(ru))[:,None]),1)
            for key,value in (('pos',pp),('vel',vv),('rot',rr),('omega',ww),('corrections',corr),('innovation_update',innovation)):
                pieces[key].append(value)
            for key,value in (('visual_velocity_update',vu),('visual_orientation_update',ru)):
                zeros=imu.new_zeros(b,end-start+1,3)
                pieces[key].append(torch.cat((zeros[:,:-1],value[:,None]),1))
            pos,vel,rotation=pp[:,-1],vv[:,-1],rr[:,-1]
            if capture:
                predicted.append(pred[:,None]);auxiliary.append(aux[:,None])
                previous,anchor=anchor,visual
                previous_span=elapsed;previous_tick=end
            start=end+1
        result={key:torch.cat(value,1) for key,value in pieces.items()}
        valid=torch.ones(b,k,device=imu.device,dtype=torch.bool);valid[:,0]=False
        sparse=online.new_zeros(b,t,e).scatter(1,ticks[...,None].expand(-1,-1,e),online)
        result.update(online=online,predicted=torch.cat(predicted,1),
                      auxiliary=torch.cat(auxiliary,1) if self.auxiliary is not None else None,
                      valid=valid,sparse_visual=sparse)
        return result

    def _persistent_step(self, image, imu, dt, gravity, state):
        s=dict(state);bootstrap=s['initial'];capture=image is not None
        if not bootstrap:
            s['imu_sum']=s['imu_sum']+imu;s['count']+=1;s['age']+=float(dt)
        visual=self.vision(image).float() if capture else s['anchor']*0
        values=self._visual_segment(imu[:,None],s['h'],visual,s['anchor'],s['previous'],
            s['imu_sum']/max(1,s['count'])/10,imu.new_tensor([[s['age']]]),imu.new_tensor([[s['span']]]),
            capture,bootstrap)
        corr,s['h'],_,_,vu,ru,_=values;corr=corr[:,0]
        omega=imu[:,3:]+corr[:,:3]
        if not bootstrap:
            acceleration=(s['rot']@(imu[:,:3]+corr[:,3:6])[...,None]).squeeze(-1)+gravity+corr[:,6:]
            s['pos']=s['pos']+s['vel']*dt+.5*acceleration*dt**2
            s['vel']=s['vel']+acceleration*dt
            s['rot']=s['rot']@so3_exp(omega*dt)
            if capture:s['vel']=s['vel']+vu;s['rot']=s['rot']@so3_exp(ru)
        else:omega=omega*0
        if capture:
            # The bootstrap capture is both anchor and previous, as in batching.
            s['previous'],s['anchor']=(visual if bootstrap else s['anchor']),visual
            s['span'],s['age'],s['count']=s['age'],0.,0
            s['imu_sum']=s['imu_sum']*0;s['captures']+=1
        s['initial']=False
        return dict(pos=s['pos'],vel=s['vel'],rot=s['rot'],omega=omega,corrections=corr),s
