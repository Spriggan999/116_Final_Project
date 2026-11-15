"""Live W&B progress for the shared distiller; all logging stays on its owner thread."""
import time


class DistillationLogger:
    def __init__(self,run,profiles,interval=10.,clock=time.perf_counter):
        self.run=run;self.clock=clock;self.interval=interval
        self.started=clock();self.last_log=self.started;self.updates=0
        run.define_metric('progress/update',hidden=True)
        run.define_metric('progress/*',step_metric='progress/update')
        run.define_metric('distillation/epoch')
        run.define_metric('distillation/*',step_metric='distillation/epoch')
        for name in profiles:run.define_metric(name+'/*',step_metric='distillation/epoch')

    def __enter__(self):
        self.phase('initializing_students')
        return self

    def phase(self,name):
        self.run.summary['distillation/status']=name
        self.run.log({'progress/update':self.updates,'progress/phase':name,
                      'progress/elapsed_seconds':self.clock()-self.started})

    def progress(self,epoch,batch,total,window,losses,max_concurrent):
        self.updates+=1;now=self.clock()
        if batch!=1 and batch!=total and now-self.last_log<self.interval:return
        self.last_log=now
        values={'progress/update':self.updates,'progress/epoch':epoch,'progress/batch':batch,
                'progress/total_batches':total,'progress/epoch_percent':100*batch/total,
                'progress/window':window,'progress/active_students':len(losses),
                'progress/max_concurrent':max_concurrent,'progress/elapsed_seconds':now-self.started}
        values.update({f'progress/{name}/train_loss':float(loss) for name,loss in losses.items()})
        self.run.log(values)

    def epoch(self,epoch,records,measurement,fresh_data_wait_seconds=0.):
        # One history row per epoch includes every active student. W&B's own
        # step remains monotonic while custom axes separate batch/epoch plots.
        values={'distillation/epoch':epoch,'distillation/fresh_data_wait_seconds':fresh_data_wait_seconds,
                'distillation/active_students':len(records)}
        values.update({f'distillation/{key}':value for key,value in measurement.items() if isinstance(value,(int,float,bool))})
        for name,record in records.items():
            values.update({f'{name}/{key}':value for key,value in record.items() if isinstance(value,(int,float,bool))})
            self.run.summary[name+'/best_validation_mean_ate_m']=record['best_validation_mean_ate_m']
            self.run.summary[name+'/completed_epochs']=epoch
            self.run.summary[name+'/status']='training' if record['active'] else 'early_stopped'
        self.run.log(values)

    def __exit__(self,exc_type,exc,tb):
        if exc_type is None:
            status='completed';code=0
        else:
            interrupted=issubclass(exc_type,KeyboardInterrupt)
            status='interrupted' if interrupted else 'failed';code=130 if interrupted else 1
            self.run.summary['distillation/error']=str(exc)
        try:self.phase(status)
        finally:self.run.finish(exit_code=code)
