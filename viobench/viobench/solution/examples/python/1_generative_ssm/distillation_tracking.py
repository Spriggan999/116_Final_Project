"""Live W&B progress for the shared distiller; all logging stays on its owner thread."""
import time


class DistillationLogger:
    def __init__(self,run,profiles,interval=10.,clock=time.perf_counter,namespace='distillation'):
        self.run=run;self.clock=clock;self.interval=interval;self.namespace=namespace
        self.profiles=tuple(profiles)
        self.started=clock();self.last_log=self.started;self.updates=0
        run.define_metric('progress/update',hidden=True)
        run.define_metric('progress/*',step_metric='progress/update')
        run.define_metric(f'{self.namespace}/epoch')
        run.define_metric(f'{self.namespace}/*',step_metric=f'{self.namespace}/epoch')
        run.define_metric('curriculum/*',step_metric=f'{self.namespace}/epoch')
        for name in self.profiles:run.define_metric(name+'/*',step_metric=f'{self.namespace}/epoch')

    def __enter__(self):
        self.phase('initializing_students')
        return self

    def phase(self,name):
        self.run.summary[f'{self.namespace}/status']=name
        self.run.log({'progress/update':self.updates,'progress/phase':name,
                      'progress/elapsed_seconds':self.clock()-self.started})

    def progress(self,epoch,batch,total,window,losses,max_concurrent,paused_students=()):
        self.updates+=1;now=self.clock()
        if batch!=1 and batch!=total and now-self.last_log<self.interval:return
        self.last_log=now
        values={'progress/update':self.updates,'progress/epoch':epoch,'progress/batch':batch,
                'progress/total_batches':total,'progress/epoch_percent':100*batch/total,
                'progress/window':window,'progress/active_students':len(losses)+len(paused_students),
                'progress/training_students':len(losses),'progress/paused_students':len(paused_students),
                'progress/max_concurrent':max_concurrent,'progress/elapsed_seconds':now-self.started}
        values.update({f'progress/{name}/train_loss':float(loss) for name,loss in losses.items()})
        self.run.log(values)

    def epoch(self,epoch,records,measurement,fresh_data_wait_seconds=0.):
        # One history row per epoch includes every active student. W&B's own
        # step remains monotonic while custom axes separate batch/epoch plots.
        values={f'{self.namespace}/epoch':epoch,f'{self.namespace}/fresh_data_wait_seconds':fresh_data_wait_seconds,
                f'{self.namespace}/active_students':len(records)}
        values.update({f'{self.namespace}/{key}':value for key,value in measurement.items() if isinstance(value,(int,float,bool))})
        for name,record in records.items():
            values.update({f'{name}/{key}':value for key,value in record.items() if isinstance(value,(int,float,bool))})
            self.run.summary[name+'/best_validation_mean_ate_m']=record['best_validation_mean_ate_m']
            self.run.summary[name+'/completed_epochs']=epoch
            self.run.summary[name+'/status']=('early_stopped' if not record['active'] else
                'paused_waiting_for_curriculum' if record.get('paused_waiting_for_curriculum') else 'training')
        self.run.log(values)

    def curriculum(self,epoch,curriculum,event):
        values={f'{self.namespace}/epoch':epoch,'curriculum/window_samples':event['window'] if event else curriculum.length,
                'curriculum/next_window_samples':curriculum.length,
                'curriculum/complete':curriculum.complete,'curriculum/final_cycles':curriculum.final_cycles,
                'curriculum/early_stopping_eligible':curriculum.early_stopping_eligible,
                'curriculum/paused_students':len(curriculum.paused),
                'curriculum/fresh_batch':event.get('fresh_batch',True) if event else False,
                'curriculum/sole_holdout_wait_epochs':getattr(curriculum,'sole_holdout_wait_epochs',0),
                'curriculum/sole_holdout_student':getattr(curriculum,'sole_holdout_student',None) or 'none',
                'curriculum/sole_holdout_timeout_epochs':getattr(curriculum,'sole_holdout_timeout_epochs',0),
                'curriculum/forced_promotion':bool(event and event.get('forced_promotion',False))}
        for axis,level in curriculum.schedule.snapshot().items():
            values[f'curriculum/{axis}_level']=event['levels'][axis] if event else level
            values[f'curriculum/next_{axis}_level']=level
        if event:
            if event.get('forced_promotion'):
                values['curriculum/forced_student']=event['forced_student']
                values['curriculum/sole_holdout_completed_wait_epochs']=event['sole_holdout_wait_epochs']
                self.run.summary['curriculum/last_forced_promotion']=dict(event,epoch=epoch)
            values['curriculum/window_promoted']=event['window_promoted']
            for axis in curriculum.schedule.axes:
                values[f'curriculum/{axis}_promoted']=axis in event['promoted']
            for axis,score in event['probe_scores'].items():values[f'curriculum/{axis}_relative_probe_loss']=score
            for axis,reasons in event['reasons'].items():
                kinds=set(reasons.values())
                values[f'curriculum/{axis}_promotion_reason']=next(iter(kinds)) if len(kinds)==1 else 'mixed'
            for axis,scores in event['student_probe_losses'].items():
                for name,loss in scores.items():values[f'curriculum/{name}/{axis}_probe_loss']=loss
        readiness=event['readiness'] if event else curriculum.readiness
        for axis,students in readiness.items():
            for name in self.profiles:
                reason=students.get(name)
                values[f'curriculum/{name}/{axis}_ready']=reason is not None
                if reason:values[f'curriculum/{name}/{axis}_ready_reason']=reason
        for axis,gates in curriculum.student_gates.items():
            for name,gate in gates.items():
                values[f'curriculum/{name}/{axis}_checks']=gate.age
                values[f'curriculum/{name}/{axis}_improvement_streak']=gate.streak
                if hasattr(gate,'bad_cycles'):
                    values[f'curriculum/{name}/{axis}_plateau_wait']=gate.bad_cycles
                    if gate.best is not None:values[f'curriculum/{name}/{axis}_best_probe_score']=gate.best
        if hasattr(curriculum,'phase'):values['curriculum/phase']=curriculum.phase
        for name in self.profiles:values[f'curriculum/{name}/paused']=name in curriculum.paused
        self.run.summary[f'{self.namespace}/curriculum']=curriculum.state_dict()
        self.run.log(values)

    def stage_restore(self,epoch,report):
        values={f'{self.namespace}/epoch':epoch,'curriculum/stage_index':report['to_stage_index'],
                'curriculum/stage_best_restored':True}
        for name,metadata in report['students'].items():
            values[f'curriculum/{name}/restored_epoch']=metadata['epoch']
            values[f'curriculum/{name}/restored_validation_mean_ate_m']=metadata['validation_mean_ate_m']
            values[f'curriculum/{name}/optimizer_restored']=True
        self.run.summary[f'{self.namespace}/last_stage_restore']=report
        self.run.log(values)

    def __exit__(self,exc_type,exc,tb):
        if exc_type is None:
            status='completed';code=0
        else:
            interrupted=issubclass(exc_type,KeyboardInterrupt)
            status='interrupted' if interrupted else 'failed';code=130 if interrupted else 1
            self.run.summary[f'{self.namespace}/error']=str(exc)
        try:self.phase(status)
        finally:self.run.finish(exit_code=code)
