"""Plateau-first curriculum with a bounded sole-holdout freeze timeout."""
from copy import deepcopy
import math
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset

from curriculum import AugmentedDataset, motion_scores
from data import StateWindows
from distillation_curriculum import DistillationCurriculum


DEFAULT_STUDENT_WINDOWS=list(range(401,4002,400))  # 200 Hz: (ticks-1)*0.005 = 2,4,...,20 s
LEGACY_STUDENT_WINDOWS=[401,801,1601,2401,4001]


class PlateauGate:
    """Require a stable near-best plateau; improvements restart patience."""
    def __init__(self, minimum=6, patience=3, improvement=.005, tolerance=.05):
        self.minimum=minimum;self.patience=patience
        self.improvement=improvement;self.tolerance=tolerance
        self.reference=self.best=None
        self.age=self.streak=self.bad_cycles=0
        self.recent=[];self.last_reason=None

    def update(self, score):
        if not math.isfinite(score) or score<0:raise ValueError('Invalid plateau validation score')
        self.last_reason=None;self.age+=1
        if self.best is None:
            self.reference=self.best=score
        else:
            self.best=min(self.best,score)
            if score<self.reference-self.improvement*max(abs(self.reference),1e-8):
                self.reference=score;self.bad_cycles=0
            else:self.bad_cycles+=1
        self.recent=(self.recent+[score])[-self.patience:]
        scale=max(abs(self.best),1e-8)
        stable=(len(self.recent)==self.patience and
                max(self.recent)-min(self.recent)<=self.tolerance*scale)
        ready=(self.age>=self.minimum and self.bad_cycles>=self.patience and stable and
               max(self.recent)<=self.best+self.tolerance*scale)
        if ready:self.last_reason='plateau'
        return ready


class StagedStudentCurriculum(DistillationCurriculum):
    """Each selected student must plateau before the current phase advances.

    Sequential runs select one profile, so their progression is independent.
    Multi-profile callers freeze qualified peers; a sole holdout may time out
    after five fully frozen peer epochs. Single-profile runs never force advance.
    """
    policy='staged_plateau_only_10_levels'

    def __init__(self,cfg,windows,start_epoch=0,stage_batches=6,min_batches=3,probe_batches=12,
                 plateau_patience=3,tolerance=.05,improvement=.005):
        cfg=SimpleNamespace(**{key:deepcopy(value) for key,value in vars(cfg).items()
                              if key.startswith('CURRICULUM_') or key in ('SEED','BATCH')})
        cfg.CURRICULUM_LEVELS=[i/9 for i in range(10)]
        cfg.CURRICULUM_MIX=[.75,.25,0.]
        cfg.CURRICULUM_ALL_EASIER=True
        cfg.CURRICULUM_RANDOM_ERROR_STRENGTH=True
        super().__init__(cfg,windows,0,stage_batches,min_batches,probe_batches)
        self.minimum=max(stage_batches,min_batches)
        self.plateau_patience=plateau_patience;self.tolerance=tolerance;self.improvement=improvement
        self.horizon_migration=None
        self.sole_holdout_timeout_epochs=5
        self.sole_holdout_student=None;self.sole_holdout_group=[]
        self.sole_holdout_wait_epochs=0;self.sole_holdout_last_epoch=0

    @property
    def phase(self):
        last=len(self.schedule.levels)-1
        if self.schedule.stages['motion']<last:return 'motion'
        if self.window_stage<len(self.windows)-1:return 'window'
        if self.schedule.stages['imu']<last:return 'imu'
        if self.schedule.stages['visual']<last:return 'visual'
        return 'final'

    @property
    def probe_axes(self):
        if self.phase=='final':return []
        return ['motion' if self.phase=='window' else self.phase]

    def gate(self,axis,name):
        gates=self.student_gates.setdefault(axis,{})
        if name not in gates:
            gates[name]=PlateauGate(self.minimum,self.plateau_patience,self.improvement,self.tolerance)
        return gates[name]

    def probe_loader(self,store,axis,batch_size):
        # Evaluate the actual current distribution, including previously enabled
        # augmentation axes. Fixed indices/seeds, independent of memory batching.
        key=(axis,self.length,tuple(self.schedule.stages.values()),batch_size)
        if key not in self.probes:
            base=StateWindows(store,length=self.length,training=False,compact_cameras=True)
            candidates=torch.arange(len(base))
            if axis=='motion':
                scores=motion_scores(base)
                cuts=torch.quantile(scores,torch.linspace(0,1,len(self.schedule.levels)+1,dtype=torch.float64)[1:-1])
                bins=torch.bucketize(scores,cuts,right=False)
                available=sorted(set(bins.tolist()))
                selected=max((n for n in available if n<=self.schedule.stages['motion']),default=min(available))
                candidates=candidates[bins==selected]
            # A sample budget rather than a minibatch count prevents a horizon
            # change from silently reducing the probe to only two windows.
            count=min(len(candidates),self.probe_batches)
            indices=candidates[torch.linspace(0,len(candidates)-1,count).round().long()].tolist()
            fixed=deepcopy(self.schedule);fixed.mix=[1.,0.,0.]
            view=AugmentedDataset(base,self.cfg,fixed,self.cfg.SEED+70_001)
            self.probes[key]=DataLoader(Subset(view,indices),batch_size=batch_size,num_workers=0)
        return self.probes[key]

    def clear_holdout(self):
        self.sole_holdout_student=None;self.sole_holdout_group=[]
        self.sole_holdout_wait_epochs=0

    def advance_phase(self,phase):
        promoted=[];window_promoted=False
        if phase=='window':
            self.window_stage+=1;self.schedule.stages['motion']=0
            window_promoted=True
        else:self.schedule.stages[phase]+=1;promoted=[phase]
        self.student_gates={};self.readiness={};self.plateaued.clear();self.paused.clear()
        self.normalizers={};self.probes.clear();self.clear_holdout()
        return promoted,window_promoted

    def finish_epoch(self,epoch,students,frozen_students,*,fresh_batch=False):
        """Count completed epochs, including reused recordings, after validation.

        Peers must be frozen both throughout this epoch and still qualified at
        its end. Becoming qualified now does not count as an already frozen epoch.
        """
        if type(epoch) is not int or epoch<1:raise ValueError('Epoch must be a positive integer')
        if epoch<=self.sole_holdout_last_epoch:return None
        consecutive=epoch==self.sole_holdout_last_epoch+1
        self.sole_holdout_last_epoch=epoch
        group=set(students);frozen=set(frozen_students)
        remaining=group-self.paused
        eligible=(not self.complete and len(group)>1 and len(remaining)==1 and
                  self.paused==group-remaining and frozen==self.paused)
        if not eligible:
            self.clear_holdout();return None
        student=next(iter(remaining));members=sorted(group)
        same=(self.sole_holdout_student==student and self.sole_holdout_group==members)
        self.sole_holdout_wait_epochs=self.sole_holdout_wait_epochs+1 if same and consecutive else 1
        self.sole_holdout_student=student;self.sole_holdout_group=members
        if self.sole_holdout_wait_epochs<self.sole_holdout_timeout_epochs:return None
        phase=self.phase;used=self.schedule.snapshot();used_window=self.length
        waited=self.sole_holdout_wait_epochs;paused_before=sorted(self.paused)
        reasons={name:('sole_holdout_timeout' if name==student else 'plateau') for name in members}
        promoted,window_promoted=self.advance_phase(phase)
        if self.complete:self.final_cycles=0
        return dict(levels=used,next_levels=self.schedule.snapshot(),window=used_window,next_window=self.length,
                    phase=phase,next_phase=self.phase,promoted=promoted,window_promoted=window_promoted,
                    reasons={phase:reasons},probe_scores={},readiness={phase:reasons},
                    paused_before=paused_before,paused_students=[],complete=self.complete,final_cycles=self.final_cycles,
                    early_stopping_eligible=self.early_stopping_eligible,fresh_batch=fresh_batch,
                    forced_promotion=True,forced_student=student,sole_holdout_wait_epochs=waited,
                    sole_holdout_timeout_epochs=self.sole_holdout_timeout_epochs)

    def update(self,batch_id,axis_scores,ate_scores):
        if not self.fresh(batch_id):return None
        if not ate_scores:raise ValueError('Curriculum needs active students')
        self.seen.add(str(batch_id))
        used=self.schedule.snapshot();used_window=self.length;phase=self.phase
        was_complete=self.complete
        probe='motion' if phase=='window' else phase
        values=axis_scores[probe] if phase!='final' else ate_scores
        if set(values)!=set(ate_scores):raise ValueError('Missing student probe scores')
        ready=self.readiness.setdefault(phase,{})
        self.plateaued.clear()
        for name,score in ate_scores.items():
            fit_ready=self.gate('fit',name).update(score)
            phase_ready=self.gate(phase,name).update(values[name]) if phase!='final' else fit_ready
            if fit_ready:self.plateaued.add(name)
            # Both the phase loss and full-trajectory ATE must be stable.
            if phase_ready and fit_ready:ready[name]='plateau'
            else:ready.pop(name,None)
        readiness={phase:{name:ready.get(name) for name in ate_scores}}
        advance=phase!='final' and all(name in ready for name in ate_scores)
        promoted=[];window_promoted=False;reasons={}
        paused_before=sorted(self.paused)
        scores={phase:self.relative_score(phase,values)}
        if advance:
            reasons[phase]=dict(ready)
            promoted,window_promoted=self.advance_phase(phase)
        elif not self.complete:self.paused=set(ready)
        else:self.paused.clear()
        if was_complete:self.final_cycles+=1
        elif self.complete:self.final_cycles=0
        return dict(levels=used,next_levels=self.schedule.snapshot(),window=used_window,next_window=self.length,
                    phase=phase,next_phase=self.phase,promoted=promoted,window_promoted=window_promoted,
                    reasons=reasons,probe_scores=scores,readiness=readiness,paused_before=paused_before,
                    paused_students=sorted(self.paused),complete=self.complete,final_cycles=self.final_cycles,
                    early_stopping_eligible=self.early_stopping_eligible,fresh_batch=True,forced_promotion=False)

    def state_dict(self):
        state=super().state_dict()
        state.update(progression_policy=self.policy,phase=self.phase,horizon_migration=self.horizon_migration,
                     sole_holdout=dict(student=self.sole_holdout_student,group=list(self.sole_holdout_group),
                                       wait_epochs=self.sole_holdout_wait_epochs,last_epoch=self.sole_holdout_last_epoch,
                                       timeout_epochs=self.sole_holdout_timeout_epochs),
                     plateau_settings=dict(minimum=self.minimum,patience=self.plateau_patience,
                                           improvement=self.improvement,tolerance=self.tolerance))
        return state

    def load_state_dict(self,state,**options):
        if state.get('progression_policy')!=self.policy:
            raise ValueError('Cannot restore a different curriculum policy into staged training')
        self.horizon_migration=deepcopy(state.get('horizon_migration'))
        if state['windows']==LEGACY_STUDENT_WINDOWS and self.windows==DEFAULT_STUDENT_WINDOWS:
            state=deepcopy(state)
            current=state['windows'][state['window_stage']]
            self.horizon_migration=dict(previous_windows=list(state['windows']),windows=list(self.windows),
                                        preserved_horizon=current)
            state['window_stage']=self.windows.index(current)
            state['windows']=list(self.windows)
        settings=state['plateau_settings']
        self.minimum=settings['minimum'];self.plateau_patience=settings['patience']
        self.improvement=settings['improvement'];self.tolerance=settings['tolerance']
        super().load_state_dict(state,**options)
        self.clear_holdout();self.sole_holdout_last_epoch=0
        saved=state.get('sole_holdout',{})
        selected=options.get('students')
        same_group=selected is None or set(selected)==set(saved.get('group',[]))
        if options.get('preserve_progress') and saved and same_group:
            self.sole_holdout_timeout_epochs=saved.get('timeout_epochs',5)
            self.sole_holdout_last_epoch=saved.get('last_epoch',0)
            self.sole_holdout_student=saved.get('student')
            self.sole_holdout_group=list(saved.get('group',[]))
            self.sole_holdout_wait_epochs=saved.get('wait_epochs',0)
