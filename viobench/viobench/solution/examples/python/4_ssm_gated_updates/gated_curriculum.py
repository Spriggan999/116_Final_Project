"""Curriculum-matched validation with bounded, competence-based promotion."""
from copy import deepcopy
import math
import legacy
from student_curriculum import StagedStudentCurriculum

DEFAULT_HORIZON_STEPS = [1 << level for level in range(14)] + [12_000]
DEFAULT_GATED_WINDOWS = [steps + 1 for steps in DEFAULT_HORIZON_STEPS]
HORIZON_POLICY = 'imu_intervals_powers_of_two_to_12000_v1'
VALIDATION_POLICY = 'current_stage_hybrid_accuracy_budget_v2'
# [base tolerance, tolerance per second]. Provisional, configurable physical targets.
DEFAULT_ACCURACY_TARGETS = {
    'position_rmse_m': [.02, .02],
    'velocity_rmse_mps': [.10, .01],
    'orientation_rmse_deg': [2., .10],
    'angular_velocity_rmse_radps': [.05, .001],
}


def accuracy_targets(settings=None):
    targets=deepcopy(DEFAULT_ACCURACY_TARGETS)
    if settings is not None:
        if not isinstance(settings,dict) or set(settings)-set(targets):
            raise ValueError('Unknown curriculum accuracy targets')
        targets.update(settings)
    for key,pair in targets.items():
        if (not isinstance(pair,(list,tuple)) or len(pair)!=2 or
            any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in pair) or pair[0]<=0):
            raise ValueError(f'{key} must be [positive base, nonnegative per-second tolerance]')
    return targets


class GatedStudentCurriculum(StagedStudentCurriculum):
    """Train all peers; advance on sustained competence or a bounded stage budget.

    Full-flight metrics never determine promotion. The inherited plateau gate is
    diagnostic only. Budgets count fresh recording checks, not reused epochs.
    """
    policy='staged_hybrid_accuracy_budget_10_levels_v1'

    def __init__(self,*args,max_stage_checks=18,accuracy_patience=3,targets=None,
                 budget_action="advance", fixed_horizon=False, **kwargs):
        super().__init__(*args,**kwargs)
        if budget_action not in ("advance", "stop"):raise ValueError("Invalid curriculum budget action")
        self.budget_action=budget_action;self.fixed_horizon=fixed_horizon
        self.max_stage_checks=max_stage_checks
        self.accuracy_patience=accuracy_patience
        self.accuracy_target_settings=accuracy_targets(targets)
        self.validate_budgets()
        self.stage_checks=0;self.hybrid_history={}
        self.validation_migration=None

    def validate_budgets(self):
        if (type(self.max_stage_checks) is not int or type(self.accuracy_patience) is not int or
            not 1<=self.accuracy_patience<=self.minimum<=self.max_stage_checks):
            raise ValueError('Require 1 <= accuracy patience <= minimum checks <= maximum checks')

    @property
    def probe_axes(self):
        return ['motion' if self.phase in ('window','final') else self.phase]

    def probe_loader(self,store,axis,batch_size):
        if not getattr(store, 'corrected_supervision', False):
            return super().probe_loader(store,'motion',batch_size)
        from gated_data import GatedStateWindows
        from curriculum import motion_scores, AugmentedDataset
        from torch.utils.data import DataLoader, Subset
        import torch
        key=(axis,self.length,tuple(self.schedule.stages.values()),batch_size,'corrected')
        if key not in self.probes:
            base=GatedStateWindows(store,self.length,False,True)
            scores=motion_scores(base)
            cuts=torch.quantile(scores,torch.linspace(0,1,len(self.schedule.levels)+1,dtype=torch.float64)[1:-1])
            bins=torch.bucketize(scores,cuts,right=False)
            available=sorted(set(bins.tolist()))
            selected=max((n for n in available if n<=self.schedule.stages['motion']),default=min(available))
            candidates=torch.arange(len(base))[bins==selected]
            count=min(len(candidates),self.probe_batches)
            indices=candidates[torch.linspace(0,len(candidates)-1,count).round().long()].tolist()
            fixed=deepcopy(self.schedule);fixed.mix=[1.,0.,0.]
            view=AugmentedDataset(base,self.cfg,fixed,self.cfg.SEED+70_001)
            self.probes[key]=DataLoader(Subset(view,indices),batch_size=batch_size,num_workers=0)
        return self.probes[key]

    def current_accuracy_targets(self):
        duration=(self.length-1)/200.
        levels=self.schedule.snapshot()
        scale=1.+.5*levels['motion']+levels['imu']+levels['visual']
        return {key:(base+rate*duration)*scale for key,(base,rate) in self.accuracy_target_settings.items()}

    def advance_phase(self,phase):
        result=super().advance_phase(phase)
        self.stage_checks=0;self.hybrid_history={}
        return result

    def finish_epoch(self,*args,**kwargs):
        # Everyone continues learning; the fresh-check cap replaces the old
        # frozen-peer/sole-holdout timeout.
        return None

    def update(self,batch_id,axis_scores,ate_scores,accuracy_scores=None):
        if not self.fresh(batch_id):return None
        losses=axis_scores[self.probe_axes[0]]
        if not losses or set(losses)!=set(ate_scores) or accuracy_scores is None or set(losses)!=set(accuracy_scores):
            raise ValueError('Each active model needs stage loss and physical accuracy metrics')
        targets=self.current_accuracy_targets()
        for name,score in losses.items():
            if not math.isfinite(score) or score<0:raise ValueError('Invalid curriculum stage loss')
            if set(accuracy_scores[name])!=set(targets) or any(not math.isfinite(x) or x<0 for x in accuracy_scores[name].values()):
                raise ValueError('Curriculum accuracy must be complete, finite and nonnegative')
        self.seen.add(str(batch_id));self.stage_checks+=1
        used=self.schedule.snapshot();used_window=self.length;phase=self.phase;was_complete=self.complete
        checks=self.stage_checks;ready={};diagnostics={}
        for name,loss in losses.items():
            plateau=self.gate('fit',name).update(loss)
            history=self.hybrid_history.setdefault(name,[])
            history.append(dict(accuracy_scores[name]))
            del history[:-self.accuracy_patience]
            competent=(checks>=self.minimum and len(history)==self.accuracy_patience and
                       all(all(row[key]<=target for key,target in targets.items()) for row in history))
            reason='competence' if competent else ('budget_exhausted' if checks>=self.max_stage_checks else None)
            if reason:ready[name]=reason
            diagnostics[name]=dict(stage_loss=loss,accuracy=accuracy_scores[name],competent=competent,
                                   plateau_diagnostic=plateau,
                                   target_ratios={key:accuracy_scores[name][key]/target for key,target in targets.items()})
        scores={phase:self.relative_score(phase,losses)}
        self.readiness={phase:dict(ready)};self.paused.clear();self.plateaued.clear()
        promoted=[];window_promoted=False;reasons={}
        advance=phase!='final' and len(ready)==len(losses)
        budget_exhausted=checks>=self.max_stage_checks and any(not row['competent'] for row in diagnostics.values())
        stop=not self.fixed_horizon and self.budget_action=='stop' and budget_exhausted
        advance=advance and not self.fixed_horizon and not stop
        if advance:
            reasons={phase:dict(ready)}
            promoted,window_promoted=self.advance_phase(phase)
        if was_complete:self.final_cycles+=1
        elif self.complete:self.final_cycles=0
        return dict(stop_requested=stop,budget_exhausted=budget_exhausted,levels=used,next_levels=self.schedule.snapshot(),window=used_window,next_window=self.length,
                    phase=phase,next_phase=self.phase,promoted=promoted,window_promoted=window_promoted,
                    reasons=reasons,probe_scores=scores,readiness={phase:{name:ready.get(name) for name in losses}},
                    paused_before=[],paused_students=[],complete=self.complete,final_cycles=self.final_cycles,
                    early_stopping_eligible=self.early_stopping_eligible,fresh_batch=True,
                    forced_promotion=advance and 'budget_exhausted' in ready.values(),
                    validation_policy=VALIDATION_POLICY,fit_metric='current_stage_physical_accuracy',
                    validation_horizon_imu_steps=used_window-1,stage_checks=checks,
                    minimum_stage_checks=self.minimum,maximum_stage_checks=self.max_stage_checks,
                    accuracy_patience=self.accuracy_patience,accuracy_targets=targets,students=diagnostics)

    def state_dict(self):
        state=super().state_dict()
        state.update(budget_action=self.budget_action,fixed_horizon=self.fixed_horizon,validation_policy=VALIDATION_POLICY,hybrid=dict(
            maximum_stage_checks=self.max_stage_checks,accuracy_patience=self.accuracy_patience,
            accuracy_targets=self.accuracy_target_settings,stage_checks=self.stage_checks,
            history=deepcopy(self.hybrid_history)))
        return state

    def load_state_dict(self,state,**options):
        changed=state.get('validation_policy')!=VALIDATION_POLICY
        previous=state.get('validation_policy','full_flight_ate_fit_gate')
        state=deepcopy(state)
        if changed:
            if state.get('progression_policy') not in ('staged_plateau_only_10_levels',self.policy):
                raise ValueError('Cannot migrate an unsupported curriculum policy')
            state['progression_policy']=self.policy
        super().load_state_dict(state,**options)
        self.validation_migration=None
        saved=state.get('hybrid')
        if not changed and saved:
            self.max_stage_checks=saved['maximum_stage_checks'];self.accuracy_patience=saved['accuracy_patience']
            self.accuracy_target_settings=accuracy_targets(saved['accuracy_targets'])
        self.validate_budgets()
        if options.get('preserve_progress') and not changed and saved:
            selected=options.get('students')
            self.stage_checks=saved['stage_checks']
            self.hybrid_history={name:history for name,history in saved['history'].items() if selected is None or name in selected}
        else:
            self.stage_checks=0;self.hybrid_history={}
            self.student_gates={};self.readiness={};self.normalizers={};self.probes={};self.final_cycles=0
        self.paused.clear();self.plateaued.clear();self.clear_holdout();self.sole_holdout_last_epoch=0
        if changed:
            self.validation_migration=dict(previous=previous,current=VALIDATION_POLICY,
                plateau_counters_reset=True,horizon_and_difficulty_preserved=True,
                minimum_budget_restarts=True)
