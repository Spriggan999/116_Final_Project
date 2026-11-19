"""Existing W&B progress logger with hybrid-specific promotion telemetry."""
import legacy
from distillation_tracking import DistillationLogger


class GatedLogger(DistillationLogger):
    def curriculum(self,epoch,curriculum,event):
        values={f'{self.namespace}/epoch':epoch,
                'curriculum/window_samples':event['window'] if event else curriculum.length,
                'curriculum/next_window_samples':curriculum.length,
                'curriculum/complete':curriculum.complete,'curriculum/final_cycles':curriculum.final_cycles,
                'curriculum/early_stopping_eligible':curriculum.early_stopping_eligible,
                'curriculum/paused_students':0,'curriculum/fresh_batch':event is not None,
                'curriculum/forced_promotion':bool(event and event['forced_promotion']),
                'curriculum/phase':event['phase'] if event else curriculum.phase,
                'curriculum/next_phase':curriculum.phase,
                'curriculum/stage_checks':event['stage_checks'] if event else curriculum.stage_checks,
                'curriculum/next_stage_checks':curriculum.stage_checks,
                'curriculum/minimum_stage_checks':curriculum.minimum,
                'curriculum/maximum_stage_checks':curriculum.max_stage_checks,
                'curriculum/accuracy_patience':curriculum.accuracy_patience}
        for axis,level in curriculum.schedule.snapshot().items():
            values[f'curriculum/{axis}_level']=event['levels'][axis] if event else level
            values[f'curriculum/next_{axis}_level']=level
            values[f'curriculum/{axis}_promoted']=bool(event and axis in event['promoted'])
        if event:
            values['curriculum/window_promoted']=event['window_promoted']
            for axis,reasons in event['reasons'].items():
                kinds=set(reasons.values())
                values[f'curriculum/{axis}_promotion_reason']=next(iter(kinds)) if len(kinds)==1 else 'mixed'
            for axis,scores in event['student_probe_losses'].items():
                for name,loss in scores.items():values[f'curriculum/{name}/{axis}_probe_loss']=loss
            for name,metrics in event['students'].items():
                prefix=f'curriculum/{name}/'
                values[prefix+'competent']=metrics['competent']
                values[prefix+'plateau_diagnostic']=metrics['plateau_diagnostic']
                reason=event['readiness'][event['phase']][name]
                values[prefix+'ready']=reason is not None
                values[prefix+'ready_reason']=reason or 'waiting'
                for key,ratio in metrics['target_ratios'].items():values[prefix+key+'_target_ratio']=ratio
            if event['promoted'] or event['window_promoted']:
                self.run.summary['curriculum/last_promotion']=event
            if event['forced_promotion']:
                self.run.summary['curriculum/last_forced_promotion']=event
        self.run.summary[f'{self.namespace}/curriculum']=curriculum.state_dict()
        self.run.log(values)
