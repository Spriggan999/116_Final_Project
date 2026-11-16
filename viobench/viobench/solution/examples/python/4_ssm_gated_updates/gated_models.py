"""IMU-only GRU recurrence, sparse vision, and four controlled JEPA ablations."""
import copy
import legacy
import torch
from torch import nn
import torch.nn.functional as F
from models import VisualFeatures, integrate_corrections, so3_exp
from performance import neural_autocast, amp_disabled

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

    def forward(self, images, imu, dt, gravity, ticks, observed, **unused):
        if self.training:
            return self._forward(images, imu, dt, gravity, ticks, observed)
        # Evaluation must agree with persistent FP32 streaming inference.
        with amp_disabled():
            return self._forward(images, imu, dt, gravity, ticks, observed)

    def _forward(self, images, imu, dt, gravity, ticks, observed):
        b, t, _ = imu.shape
        k = images.shape[1]
        e = self.settings['embedding_dim']
        online = self.vision(images.flatten(0, 1)).reshape(b, k, e)
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
            target = self.target(images.flatten(0, 1).float()).reshape(b, k, -1)
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
