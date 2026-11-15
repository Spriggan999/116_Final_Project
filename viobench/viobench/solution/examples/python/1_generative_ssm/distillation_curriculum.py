"""Shared, fresh-data-gated curriculum with fixed per-axis validation probes."""
from copy import deepcopy
import math
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader,Subset

from curriculum import (AugmentedDataset,DifficultyCurriculum,ProgressGate,
                        curriculum_sampler,motion_scores,seed_for)
from data import StateWindows


class DistillationCurriculum:
    """One shared observation distribution preserves one teacher pass per batch.

    Each student has its own readiness gates. An axis advances only when every
    active student qualifies. Ready students with a stable trajectory plateau
    pause their optimizer updates until the shared difficulty changes.
    """
    def __init__(self,cfg,windows,start_epoch=0,stage_batches=6,min_batches=3,probe_batches=2):
        self.cfg=SimpleNamespace(**{key:deepcopy(value) for key,value in vars(cfg).items()
                                   if key.startswith('CURRICULUM_') or key=='SEED'})
        self.cfg.CURRICULUM_MIN_EPOCHS=min_batches
        self.cfg.CURRICULUM_MAX_STAGE_EPOCHS=stage_batches
        self.cfg.CURRICULUM_STATE_DEVICE_PHOTOMETRIC=True
        self.schedule=DifficultyCurriculum(self.cfg)
        self.windows=list(windows)
        self.window_stage=min(start_epoch//stage_batches,len(windows)-1)
        self.probe_batches=probe_batches
        self.batch_ceiling=getattr(cfg,'BATCH',8)
        self.normalizers={}
        self.seen=set()
        self.final_cycles=0
        self.probes={}
        self.student_gates={}
        self.readiness={}
        self.plateaued=set()
        self.paused=set()

    def training_students(self,active):
        return {name:row for name,row in active.items() if name not in self.paused}

    def gate(self,axis,name):
        gates=self.student_gates.setdefault(axis,{})
        if name not in gates:
            gates[name]=ProgressGate(self.cfg.CURRICULUM_MIN_EPOCHS,self.cfg.CURRICULUM_PATIENCE,
                self.cfg.CURRICULUM_REL_IMPROVEMENT,self.cfg.CURRICULUM_MAX_STAGE_EPOCHS,
                self.cfg.CURRICULUM_PLATEAU_TOLERANCE)
        return gates[name]

    @property
    def length(self):return self.windows[self.window_stage]

    @property
    def probe_axes(self):
        return [axis for axis in self.schedule.axes if self.schedule.stages[axis]<len(self.schedule.levels)-1]

    @property
    def complete(self):
        return (self.window_stage==len(self.windows)-1 and
                all(stage==len(self.schedule.levels)-1 for stage in self.schedule.stages.values()))

    @property
    def early_stopping_eligible(self):
        return self.complete and self.final_cycles>=self.cfg.CURRICULUM_FINAL_STAGE_CYCLES

    def fresh(self,batch_id):return str(batch_id) not in self.seen

    def training_loader(self,base,options,epoch):
        seed=self.cfg.SEED+100_003*epoch
        view=AugmentedDataset(base.dataset,self.cfg,self.schedule,seed)
        sampler=curriculum_sampler(base.dataset,self.schedule,None,seed)
        return DataLoader(view,batch_size=base.batch_size,sampler=sampler,**options)

    def probe_loader(self,store,axis,batch_size):
        key=(axis,self.length,batch_size)
        if key not in self.probes:
            base=StateWindows(store,length=self.length,training=False,compact_cameras=True)
            candidates=torch.arange(len(base))
            if axis=='motion':
                scores=motion_scores(base)
                candidates=candidates[scores>=torch.quantile(scores,.67)]
            budget=getattr(self.cfg,'CURRICULUM_BATCH_TOKENS',0) or 3208
            probe_batch=min(self.batch_ceiling,max(1,budget//self.length))
            # Memory recovery can change minibatch size. Keep the actual probe
            # examples fixed so this cannot manufacture readiness improvements.
            count=min(len(candidates),probe_batch*self.probe_batches)
            indices=candidates[torch.linspace(0,len(candidates)-1,count).round().long()].tolist()
            view=AugmentedDataset(base,self.cfg,self.schedule,self.cfg.SEED+70_001,probe=axis)
            self.probes[key]=DataLoader(Subset(view,indices),batch_size=batch_size,num_workers=0,
                generator=torch.Generator().manual_seed(seed_for(self.cfg.SEED,'distill-probe',axis,self.length)))
        return self.probes[key]

    def relative_score(self,axis,values):
        if not values or any(not math.isfinite(value) or value<0 for value in values.values()):
            raise ValueError('Curriculum needs finite nonnegative per-student probe errors')
        references=self.normalizers.setdefault(axis,{})
        # Diagnostic only; promotion uses every student's own gate below.
        return max(value/max(references.setdefault(name,value),1e-8) for name,value in values.items())

    @staticmethod
    def advance(gate,score):
        if gate.best is not None and score>gate.best+gate.tolerance*max(abs(gate.best),1e-8):
            gate.age+=1;gate.streak=0;gate.last_reason=None
            return False
        first=gate.reference is None
        promoted=gate.update(score)
        if first:gate.age=1
        return promoted

    def update(self,batch_id,axis_scores,ate_scores):
        if not self.fresh(batch_id):return None
        self.seen.add(str(batch_id))
        used=self.schedule.snapshot();used_window=self.length;was_complete=self.complete
        students=tuple(ate_scores)
        required=[axis for axis in self.schedule.axes if self.schedule.stages[axis]<len(self.schedule.levels)-1]
        if self.window_stage<len(self.windows)-1:required.append('window')
        if not students:raise ValueError('Curriculum needs at least one active student')
        values={axis:(ate_scores if axis=='window' else axis_scores[axis]) for axis in required}
        if any(set(errors)!=set(students) for errors in values.values()):
            raise ValueError('Each curriculum axis must include every active student')
        scores={axis:self.relative_score(axis,errors) for axis,errors in values.items()}
        for axis,errors in values.items():
            ready=self.readiness.setdefault(axis,{})
            for name,score in errors.items():
                gate=self.gate(axis,name)
                regressed=gate.best is not None and score>gate.best+gate.tolerance*max(abs(gate.best),1e-8)
                if regressed:ready.pop(name,None)
                if ready.get(name)=='plateau' and not regressed:continue
                if self.advance(gate,score):ready[name]=gate.last_reason
        # Pausing requires a trajectory plateau as well as readiness on every
        # unfinished axis. Improvement readiness alone does not freeze learning.
        for name,score in ate_scores.items():
            if not math.isfinite(score) or score<0:raise ValueError('Nonfinite curriculum ATE')
            gate=self.gate('fit',name)
            if gate.best is not None and score>gate.best+gate.tolerance*max(abs(gate.best),1e-8):
                self.plateaued.discard(name)
            if name not in self.plateaued and self.advance(gate,score) and gate.last_reason=='plateau':
                self.plateaued.add(name)
        qualified={axis:all(name in self.readiness.get(axis,{}) for name in students) for axis in required}
        reasons={axis:dict(self.readiness[axis]) for axis in required if qualified[axis]}
        readiness={axis:{name:self.readiness.get(axis,{}).get(name) for name in students} for axis in required}
        promoted=[axis for axis in self.schedule.axes if qualified.get(axis,False)]
        for axis in promoted:self.schedule.stages[axis]+=1
        window_promoted=qualified.get('window',False)
        if window_promoted:self.window_stage+=1
        paused_before=sorted(self.paused)
        if promoted or window_promoted:
            # Every model must adapt to the new joint input distribution. Old
            # readiness, even on other axes, must not trigger another promotion.
            self.student_gates={};self.readiness={};self.plateaued.clear();self.paused.clear()
            self.normalizers={};self.probes.clear()
        elif not self.complete:
            self.paused={name for name in students if name in self.plateaued and
                         all(name in self.readiness.get(axis,{}) for axis in required)}
        else:self.paused.clear()
        if self.paused==set(students):raise RuntimeError('All students paused without curriculum promotion')
        if was_complete:self.final_cycles+=1
        elif self.complete:self.final_cycles=0
        return dict(levels=used,next_levels=self.schedule.snapshot(),window=used_window,next_window=self.length,
                    promoted=promoted,window_promoted=window_promoted,reasons=reasons,probe_scores=scores,
                    readiness=readiness,paused_before=paused_before,paused_students=sorted(self.paused),
                    complete=self.complete,final_cycles=self.final_cycles,
                    early_stopping_eligible=self.early_stopping_eligible)

    def state_dict(self):
        return dict(windows=self.windows,levels=self.schedule.levels,window_stage=self.window_stage,
                    stages=self.schedule.stages,normalizers=self.normalizers,seen=sorted(self.seen),
                    progression_policy='all_students_ready',
                    student_gates={axis:{name:vars(gate) for name,gate in gates.items()} for axis,gates in self.student_gates.items()},
                    readiness=self.readiness,plateaued=sorted(self.plateaued),paused=sorted(self.paused),
                    final_cycles=self.final_cycles)

    def load_state_dict(self,state,*,preserve_progress=False,students=None):
        if state['windows']!=self.windows or state['levels']!=self.schedule.levels:
            raise ValueError('Warm-start curriculum windows/levels differ from the saved run')
        self.window_stage=state['window_stage'];self.schedule.stages=dict(state['stages'])
        self.normalizers=deepcopy(state['normalizers']);self.seen=set(state['seen'])
        self.student_gates={};self.readiness={};self.plateaued.clear();self.paused.clear()
        self.final_cycles=0
        if preserve_progress:
            selected=set(students) if students is not None else None
            for axis,gates in state.get('student_gates',{}).items():
                for name,values in gates.items():
                    if selected is None or name in selected:vars(self.gate(axis,name)).update(deepcopy(values))
            self.readiness={axis:{name:reason for name,reason in ready.items() if selected is None or name in selected}
                            for axis,ready in state.get('readiness',{}).items()}
            self.plateaued=set(state.get('plateaued',()))
            self.paused=set(state.get('paused',()))
            if selected is not None:
                self.plateaued.intersection_update(selected);self.paused.intersection_update(selected)
                # A selected subset may have been waiting only for models that
                # the user has now omitted. Re-evaluate it without erasing gates.
                if selected and self.paused==selected:self.paused.clear()
            self.final_cycles=state.get('final_cycles',0)
