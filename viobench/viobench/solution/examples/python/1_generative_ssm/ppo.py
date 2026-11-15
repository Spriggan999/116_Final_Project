"""On-policy PPO for the existing bounded motion-correction head.

External camera/IMU recordings are exogenous: each rollout samples NEW actions
from the current actor and re-integrates the estimator state. Old actions and
log probabilities are discarded after an iteration. The SL encoder/recurrent
backbone is frozen; the privileged training critic is never exported.
"""
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal

from data import state_input_frames
from losses import losses
from performance import EpochTimer,amp_disabled,fp32_physics,record_epoch,timed_batches,pipeline_log
from models import so3_exp


def frozen_tokens(student, frames, batch, device):
    """Capture inputs to the existing output head, supporting GRU and Mamba."""
    if hasattr(student,'encode_tokens'):
        with torch.no_grad():
            return student.encode_tokens(frames,batch['imu'].to(device,non_blocking=True),
                batch['observed'].to(device,non_blocking=True),batch['dt'].to(device,non_blocking=True)).detach().float()
    captured = []
    hook = student.out.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().float()))
    try:
        with torch.no_grad():
            student(frames, batch['imu'].to(device,non_blocking=True), batch['observed'].to(device,non_blocking=True),
                    batch['dt'].to(device,non_blocking=True), batch['gravity'].to(device,non_blocking=True))
    finally:
        hook.remove()
    return captured[0] if captured[0].ndim == 3 else torch.stack(captured, dim=1)


def advantages(rewards, values, gamma, lam, bootstrap_value=None,*,parallel=False):
    """GAE over B x T windows; bootstrap when a window is time-truncated."""
    if parallel:
        tail=torch.zeros_like(values[:,-1]) if bootstrap_value is None else bootstrap_value
        next_values=torch.cat([values[:,1:],tail[:,None]],1)
        result=rewards+gamma*next_values-values
        discount=gamma*lam;offset=1
        while discount and offset<rewards.shape[1]:
            result=result+(discount**offset)*torch.cat([result[:,offset:],torch.zeros_like(result[:,:offset])],1)
            offset*=2
        return result,result+values
    result = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])
    next_value = torch.zeros_like(running) if bootstrap_value is None else bootstrap_value
    for t in reversed(range(rewards.shape[1])):
        delta = rewards[:, t] + gamma*next_value - values[:, t]
        running = delta + gamma*lam*running
        result[:, t] = running
        next_value = values[:, t]
    return result, result+values


@fp32_physics
def physical_step(position, velocity, rotation, imu, latent, dt, gravity):
    correction = .25*torch.tanh(latent)
    omega = imu[:, 3:6]+correction[:, :3]
    acceleration = (rotation @ (imu[:, :3]+correction[:, 3:6])[:, :, None]).squeeze(-1)
    acceleration = acceleration+gravity+correction[:, 6:9]
    interval = dt[:, None]
    return (position+velocity*interval+.5*acceleration*interval.square(),
            velocity+acceleration*interval,
            rotation @ so3_exp(omega*interval), omega)


def collect_rollout(frame_model, student, critic, loader, cfg, observe_errors=None,*,timer=None):
    if getattr(cfg,'VECTORIZED_PPO_ROLLOUT',False):
        return _collect_vectorized(frame_model,student,critic,loader,cfg,observe_errors,timer)
    fields = {name: [] for name in ('tokens','critic_inputs','actions','log_probs','values','advantages','returns')}
    reward_means = []
    batches=timed_batches(loader,timer) if timer is not None else loader
    for index, batch in enumerate(batches):
        if index >= cfg.PPO_ROLLOUT_BATCHES:
            break
        with torch.no_grad():
            frames = state_input_frames(frame_model, batch, cfg.DEVICE)
            tokens = frozen_tokens(student, frames, batch, cfg.DEVICE)
            del frames
            imu, dt, gravity = (batch[k].to(cfg.DEVICE,non_blocking=True) for k in ('imu','dt','gravity'))
            truth_p, truth_v, truth_r, truth_w = (batch[k].to(cfg.DEVICE,non_blocking=True) for k in ('pos','vel','rot','omega'))
            count, ticks, _ = tokens.shape
            if timer is not None:timer.add(count)
            position = tokens.new_zeros(count, 3)
            velocity = torch.zeros_like(position)
            rotation = torch.eye(3, device=tokens.device)[None].repeat(count,1,1)
            samples = {key: [] for key in ('tokens','critic_inputs','actions','log_probs','values','rewards')}
            if ticks < 3:
                raise ValueError('PPO windows need at least three samples, including a bootstrap state')
            # Reserve the last token for the next-state value: a window boundary
            # is a time truncation, not an absorbing failure of the estimator.
            for t in range(1, ticks-1):
                token = tokens[:, t]
                # Ground truth enters only reward/critic, never the exported actor.
                critic_input = torch.cat([token, position, velocity, rotation.flatten(1),
                                          truth_p[:,t], truth_v[:,t], truth_r[:,t].flatten(1)], -1)
                distribution = Normal(student.out(token), cfg.PPO_STD)
                latent = distribution.sample()
                value = critic(critic_input).squeeze(-1)
                position, velocity, rotation, omega = physical_step(
                    position, velocity, rotation, imu[:,t], latent, dt, gravity)
                error = ((position-truth_p[:,t]).square().sum(-1)
                         + .1*(velocity-truth_v[:,t]).square().sum(-1)
                         + .05*(rotation-truth_r[:,t]).square().sum((1,2))
                         + .01*(omega-truth_w[:,t]).square().sum(-1))
                samples['tokens'].append(token)
                samples['critic_inputs'].append(critic_input)
                samples['actions'].append(latent)
                # Latent-action log probabilities are sufficient: the fixed
                # tanh transform's Jacobian cancels in the PPO density ratio.
                samples['log_probs'].append(distribution.log_prob(latent).sum(-1))
                samples['values'].append(value)
                samples['rewards'].append(-error)
            samples = {k: torch.stack(v, dim=1) for k,v in samples.items()}
            if observe_errors is not None:
                observe_errors(batch,-samples['rewards'].mean(1))
            final_input = torch.cat([tokens[:,-1], position, velocity, rotation.flatten(1),
                                     truth_p[:,-1], truth_v[:,-1], truth_r[:,-1].flatten(1)], -1)
            bootstrap = critic(final_input).squeeze(-1)
            adv, returns = advantages(samples['rewards'], samples['values'], cfg.PPO_GAMMA,
                                      cfg.PPO_GAE_LAMBDA, bootstrap)
            samples.update(advantages=adv, returns=returns)
            reward_means.append(float(samples['rewards'].mean()))
            for name in fields:
                value = samples[name]
                fields[name].append(value.reshape(-1, *value.shape[2:]))
    return _rollout_result(fields,reward_means)


def _rollout_result(fields,reward_means):
    if not reward_means:
        raise ValueError('PPO has no rollout windows')
    result = {k: torch.cat(v).detach() for k,v in fields.items()}
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite PPO rollout')
    return result, sum(reward_means)/len(reward_means)


@torch.no_grad()
def vectorized_batch_rollout(student,critic,tokens,batch,cfg):
    """Actions depend only on frozen causal tokens, so sample all NEW actions first."""
    from models import integrate_corrections
    count,ticks,width=tokens.shape
    if ticks<3:raise ValueError('PPO windows need at least three samples, including a bootstrap state')
    policy_tokens=tokens[:,1:-1]
    means=student.out(policy_tokens.reshape(-1,width)).reshape(count,ticks-2,9)
    distribution=Normal(means,cfg.PPO_STD)
    actions=distribution.sample()
    corrections=torch.cat([tokens.new_zeros(count,1,9),.25*torch.tanh(actions)],1)
    imu,dt,gravity=(batch[k].to(tokens.device,non_blocking=True) for k in ('imu','dt','gravity'))
    truth_p,truth_v,truth_r,truth_w=(batch[k].to(tokens.device,non_blocking=True) for k in ('pos','vel','rot','omega'))
    state=integrate_corrections(corrections,imu[:,:-1],dt,gravity,parallel=True)
    inputs=torch.cat([policy_tokens,state['pos'][:,:-1],state['vel'][:,:-1],state['rot'][:,:-1].flatten(2),
                      truth_p[:,1:-1],truth_v[:,1:-1],truth_r[:,1:-1].flatten(2)],-1)
    values=critic(inputs.reshape(-1,inputs.shape[-1])).reshape(count,ticks-2)
    error=((state['pos'][:,1:]-truth_p[:,1:-1]).square().sum(-1)
           +.1*(state['vel'][:,1:]-truth_v[:,1:-1]).square().sum(-1)
           +.05*(state['rot'][:,1:]-truth_r[:,1:-1]).square().sum((2,3))
           +.01*(state['omega'][:,1:]-truth_w[:,1:-1]).square().sum(-1))
    final=torch.cat([tokens[:,-1],state['pos'][:,-1],state['vel'][:,-1],state['rot'][:,-1].flatten(1),
                     truth_p[:,-1],truth_v[:,-1],truth_r[:,-1].flatten(1)],-1)
    bootstrap=critic(final).squeeze(-1)
    adv,returns=advantages(-error,values,cfg.PPO_GAMMA,cfg.PPO_GAE_LAMBDA,bootstrap,parallel=True)
    return dict(tokens=policy_tokens,critic_inputs=inputs,actions=actions,log_probs=distribution.log_prob(actions).sum(-1),
                values=values,rewards=-error,advantages=adv,returns=returns,bootstrap=bootstrap)


def _collect_vectorized(frame_model,student,critic,loader,cfg,observe_errors,timer):
    fields={name:[] for name in ('tokens','critic_inputs','actions','log_probs','values','advantages','returns')}
    rewards=[];batches=timed_batches(loader,timer) if timer is not None else loader
    for index,batch in enumerate(batches):
        if index>=cfg.PPO_ROLLOUT_BATCHES:break
        with torch.no_grad():
            frames=state_input_frames(frame_model,batch,cfg.DEVICE)
            tokens=frozen_tokens(student,frames,batch,cfg.DEVICE)
            del frames
            samples=vectorized_batch_rollout(student,critic,tokens,batch,cfg)
            if timer is not None:timer.add(len(tokens))
            if observe_errors is not None:observe_errors(batch,-samples['rewards'].mean(1))
            rewards.append(float(samples['rewards'].mean()))
            for name in fields:
                value=samples[name];fields[name].append(value.reshape(-1,*value.shape[2:]))
    return _rollout_result(fields,rewards)


def ppo_update(student, critic, reference_head, optimizer, rollout, cfg):
    adv = rollout['advantages']
    adv = (adv-adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
    n = len(adv)
    last_kl, updates = 0., 0
    for _ in range(cfg.PPO_UPDATE_EPOCHS):
        order = torch.randperm(n, device=adv.device)
        for start in range(0,n,cfg.PPO_MINIBATCH):
            idx = order[start:start+cfg.PPO_MINIBATCH]
            means = student.out(rollout['tokens'][idx])
            distribution = Normal(means, cfg.PPO_STD)
            log_probs = distribution.log_prob(rollout['actions'][idx]).sum(-1)
            log_ratio = log_probs-rollout['log_probs'][idx]
            ratio = log_ratio.clamp(-20,20).exp()
            last_kl = float(((ratio-1)-log_ratio).mean().detach())
            if last_kl > cfg.PPO_TARGET_KL:
                return dict(approx_kl=last_kl, updates=updates)
            unclipped = ratio*adv[idx]
            clipped = ratio.clamp(1-cfg.PPO_CLIP,1+cfg.PPO_CLIP)*adv[idx]
            actor_loss = -torch.minimum(unclipped,clipped).mean()
            value = critic(rollout['critic_inputs'][idx]).squeeze(-1)
            value_loss = .5*(value-rollout['returns'][idx]).square().mean()
            with torch.no_grad():
                reference = reference_head(rollout['tokens'][idx])
            anchor = .5*((means-reference)/cfg.PPO_STD).square().sum(-1).mean()
            objective = actor_loss+value_loss+cfg.PPO_ANCHOR_COEF*anchor
            if not torch.isfinite(objective):
                raise ValueError('Nonfinite PPO objective')
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(list(student.out.parameters())+list(critic.parameters()), .5)
            optimizer.step()
            updates += 1
    return dict(approx_kl=last_kl, updates=updates)


def validation_loss(frame_model, student, loader, device, *, return_metrics=False,episode_loader=None):
    total, count = 0., 0
    if return_metrics:
        from leaderboard_metrics import LeaderboardATE,finish_ate
        ate=LeaderboardATE()
    student.eval()
    with torch.no_grad(),amp_disabled():
        for batch in loader:
            frames = state_input_frames(frame_model,batch,device)
            prediction = student(frames,batch['imu'].to(device,non_blocking=True),batch['observed'].to(device,non_blocking=True),
                                 batch['dt'].to(device,non_blocking=True),batch['gravity'].to(device,non_blocking=True))
            value,_ = losses(prediction,batch,device,metrics=False)
            n = len(frames)
            if return_metrics:
                ate.update(prediction,batch)
            total += float(value)*n
            count += n
    if not count:
        raise ValueError('No PPO validation windows')
    result = total/count
    if not math.isfinite(result):
        raise ValueError('Nonfinite PPO validation loss')
    if return_metrics:
        return result,finish_ate(frame_model,student,device,ate,episode_loader)
    return result


class WindowCurriculum:
    """Advance after sustained improvement on one fixed validation horizon."""
    def __init__(self, cfg, prefix='PPO'):
        self.windows = list(getattr(cfg, prefix+'_CURRICULUM_WINDOWS', [])) or [cfg.WINDOW]
        if (any(type(n) is not int or n < 3 for n in self.windows)
                or any(b <= a for a, b in zip(self.windows, self.windows[1:]))):
            raise ValueError(prefix+'_CURRICULUM_WINDOWS must be strictly increasing integers >= 3')
        self.minimum = getattr(cfg, 'PPO_CURRICULUM_MIN_ITERATIONS' if prefix=='PPO' else 'CURRICULUM_MIN_EPOCHS', 3)
        self.patience = getattr(cfg, 'PPO_CURRICULUM_PATIENCE' if prefix=='PPO' else 'CURRICULUM_PATIENCE', 2)
        self.improvement = getattr(cfg, 'PPO_CURRICULUM_REL_IMPROVEMENT' if prefix=='PPO' else 'CURRICULUM_REL_IMPROVEMENT', .02)
        if (type(self.minimum) is not int or self.minimum < 1
                or type(self.patience) is not int or self.patience < 1
                or not 0 < self.improvement < 1):
            raise ValueError('Invalid PPO curriculum promotion settings')
        self.stage = self.age = self.streak = 0
        self.reference = None
        self.best=None
        self.stage_scores=getattr(cfg,'USE_CURRICULUM',False)
        self.maximum=getattr(cfg,'CURRICULUM_MAX_STAGE_EPOCHS',0) if self.stage_scores else 0
        self.tolerance=getattr(cfg,'CURRICULUM_PLATEAU_TOLERANCE',.1)
        self.last_reason=None

    @property
    def length(self):
        return self.windows[self.stage]

    def update(self, score):
        self.last_reason=None
        if not math.isfinite(score):raise ValueError('Nonfinite curriculum progress loss')
        if self.reference is None:
            if not self.stage_scores:raise ValueError('Set curriculum reference to the initial validation loss')
            self.reference=self.best=score
            return False
        self.age += 1
        self.best=score if self.best is None else min(self.best,score)
        good = self.reference-score > self.improvement*max(abs(self.reference), 1e-12)
        self.streak = self.streak+1 if good else 0
        improved=self.age >= self.minimum and self.streak >= self.patience
        plateau=bool(self.maximum and self.age>=self.maximum and score<=self.best+self.tolerance*max(abs(self.best),1e-8))
        if (improved or plateau) and self.stage+1 < len(self.windows):
            self.last_reason='improved' if improved else 'plateau'
            self.stage += 1
            self.age = self.streak = 0
            self.reference = None if self.stage_scores else score
            self.best=None
            return True
        return False


def refine_ppo(frame_model, student, data, cfg, output, save_weights):
    import wandb
    from leaderboard_metrics import ATE_CONFIG
    with wandb.init(project=cfg.WANDB_PROJECT,dir=str(output),job_type='ppo_refinement',mode=cfg.WANDB_MODE,
                    config=dict(iterations=cfg.PPO_ITERATIONS,rollout_batches=cfg.PPO_ROLLOUT_BATCHES,
                                lr=cfg.PPO_LR,windows=cfg.PPO_CURRICULUM_WINDOWS,**ATE_CONFIG)) as tracker:
        return _refine_ppo(frame_model,student,data,cfg,output,save_weights,tracker)


def _refine_ppo(frame_model, student, data, cfg, output, save_weights, tracker):
    if min(cfg.PPO_ITERATIONS,cfg.PPO_ROLLOUT_BATCHES,cfg.PPO_UPDATE_EPOCHS,cfg.PPO_MINIBATCH) < 1:
        raise ValueError('PPO iteration, rollout and minibatch counts must be positive')
    if (cfg.PPO_STD <= 0 or cfg.PPO_LR <= 0 or not 0 < cfg.PPO_CLIP < 1
            or not 0 <= cfg.PPO_GAMMA <= 1 or not 0 <= cfg.PPO_GAE_LAMBDA <= 1
            or cfg.PPO_TARGET_KL <= 0 or cfg.PPO_ANCHOR_COEF < 0):
        raise ValueError('Invalid PPO distribution/optimization settings')
    output = Path(output)
    curriculum = WindowCurriculum(cfg)
    data.begin_phase('ppo')
    frame_model.eval()
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in student.out.parameters():
        parameter.requires_grad_(True)
    reference_head = copy.deepcopy(student.out).eval()
    for parameter in reference_head.parameters():
        parameter.requires_grad_(False)
    critic = nn.Sequential(nn.Linear(student.hidden+30,64),nn.Tanh(),nn.Linear(64,1)).to(cfg.DEVICE,non_blocking=True)
    optimizer = torch.optim.Adam(list(student.out.parameters())+list(critic.parameters()),lr=cfg.PPO_LR)
    validation_loader = data.ppo_loader(curriculum.windows[-1], validation=True)
    if data.curriculum is not None:
        data.curriculum_validation_loader=validation_loader
    best,baseline_metrics = validation_loss(frame_model,student,validation_loader,cfg.DEVICE,return_metrics=True,
                                           episode_loader=getattr(data,'val_episode_loader',None))
    selected_mean_ate=baseline_metrics['mean_ate_m']
    baseline = best
    curriculum.reference = None if curriculum.stage_scores else baseline
    best_path = output/'state_student_ppo.pt'
    save_weights(student.state_dict(),best_path)
    print(f'PPO: frozen encoder/backbone, trainable correction head; SL validation={baseline:.6g}',flush=True)
    for iteration in range(cfg.PPO_ITERATIONS):
        data.next_epoch()
        length = curriculum.length
        if data.curriculum is not None:data.ppo_probe_length=length
        loader = data.ppo_training_loader(length)
        timer=EpochTimer(cfg.DEVICE)
        # Recording reuse is allowed because actions do not control the drone.
        # Each iteration still needs NEW policy samples and estimator rollouts.
        rollout,reward = collect_rollout(frame_model,student,critic,loader,cfg,data.observe_errors,timer=timer)
        diagnostics = ppo_update(student,critic,reference_head,optimizer,rollout,cfg)
        del rollout
        performance=record_epoch(output,'ppo',iteration+1,timer.result(),data,loader=loader)
        score,validation_metrics = validation_loss(frame_model,student,validation_loader,cfg.DEVICE,return_metrics=True,
                                                  episode_loader=getattr(data,'val_episode_loader',None))
        selected = score < best
        if selected:
            best = score
            selected_mean_ate=validation_metrics['mean_ate_m']
            save_weights(student.state_dict(),best_path)
        progress_score=data.progress_score(length,
            lambda loader:validation_loss(frame_model,student,loader,cfg.DEVICE)) if data.curriculum is not None else score
        promoted = curriculum.update(progress_score)
        data.finish_curriculum_epoch(score,
            lambda loader:validation_loss(frame_model,student,loader,cfg.DEVICE),
            window_event=dict(window_samples=length,next_window_samples=curriculum.length,
                              promoted=promoted,reason=curriculum.last_reason,
                              progress_validation_loss=progress_score))
        record = dict(iteration=iteration+1,mean_reward=reward,validation_loss=score,
                      val_mean_ate_m=validation_metrics['mean_ate_m'],
                      val_sim3_ate_rmse_mean=validation_metrics['sim3_ate_rmse_mean'],
                      val_ate_episodes=validation_metrics['ate_episodes'],
                      val_ate_camera_calibrated_episodes=validation_metrics['ate_camera_calibrated_episodes'],
                      sl_val_mean_ate_m=baseline_metrics['mean_ate_m'],
                      selected_val_mean_ate_m=selected_mean_ate,
                      window_samples=length,next_window_samples=curriculum.length,
                      progress_validation_loss=progress_score,
                      validation_window_samples=curriculum.windows[-1],curriculum_promoted=promoted,
                      curriculum_promotion_reason=curriculum.last_reason or 'none',
                      best_validation_loss=best,sl_validation_loss=baseline,selected=selected,
                      recording_batch=data.batch_id,
                      recording_pass=data.rolling.passes_on_current if data.rolling else 1,
                      **diagnostics)
        with (output/'ppo_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        tracker.log({'ppo/'+key:value for key,value in record.items()
                     if key!='recording_batch'}|data.curriculum_log()|pipeline_log(performance))
        save_weights(dict(actor=student.out.state_dict(),critic=critic.state_dict(),
                          optimizer=optimizer.state_dict(),iteration=iteration+1,
                          curriculum=dict(windows=curriculum.windows,stage=curriculum.stage,
                                          age=curriculum.age,streak=curriculum.streak,
                                          reference=curriculum.reference)),output/'ppo_training.pt')
        print(f'PPO {iteration+1}/{cfg.PPO_ITERATIONS}: window={length}, reward={reward:.6g}, '
              f'val={score:.6g}, selected={selected}, next_window={curriculum.length}',flush=True)
    student.load_state_dict(torch.load(best_path,map_location=cfg.DEVICE,weights_only=True))
    student.eval()
    tracker.summary['best_validation_loss']=best
    tracker.summary['initial_validation_loss']=baseline
    tracker.summary['selected_val_mean_ate_m']=selected_mean_ate
    tracker.summary['initial_val_mean_ate_m']=baseline_metrics['mean_ate_m']
    print(f'PPO selected validation={best:.6g}; SL baseline={baseline:.6g}',flush=True)
