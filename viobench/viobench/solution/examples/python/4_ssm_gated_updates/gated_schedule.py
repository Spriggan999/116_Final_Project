"""Measure safe concurrent update schedules, restoring all training state."""
import statistics
import time
import legacy
import torch
from distillation_checkpoints import cpu_snapshot
from distillation_pipeline import DistillationPipeline, scheduling_candidates, memory_headroom, ScheduleMemoryError


class MeasuredPipeline(DistillationPipeline):
    warmup_steps = 2
    measure_steps = 5
    repeats = 3

    def autotune(self, active, prepared, operation, headroom=.15):
        if not self.parallel:
            return dict(mode='serial', max_concurrent=1, candidates=[])
        torch.cuda.synchronize(self.device)
        snapshots = {name: (cpu_snapshot(row['model'].state_dict()),
                            cpu_snapshot(row['optimizer'].state_dict()), row['model'].training)
                     for name, row in active.items()}
        rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        original = self.max_concurrent, self.groups, self.serial_students
        selected = False

        def restore():
            torch.cuda.synchronize(self.device)
            for name, row in active.items():
                model, optimizer, training = snapshots[name]
                row['optimizer'].zero_grad(set_to_none=True)
                row['model'].load_state_dict(model)
                row['optimizer'].load_state_dict(cpu_snapshot(optimizer))
                row['model'].train(training)
            torch.set_rng_state(rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

        candidates = []
        plans = scheduling_candidates(active, {name: 1 for name in active})
        try:
            for plan in plans:
                print(f"Probing schedule: {plan['mode']} {plan['groups']}", flush=True)
                self.max_concurrent = plan['max_concurrent']
                self.groups = plan['groups']
                self.serial_students = plan['mode'] == 'serial_students'
                try:
                    for _ in range(self.warmup_steps):
                        self.run(active, prepared, operation)
                    torch.cuda.synchronize(self.device)
                    torch.cuda.reset_peak_memory_stats(self.device)
                    samples = []
                    for _ in range(self.repeats):
                        start = time.perf_counter()
                        for _ in range(self.measure_steps):
                            self.run(active, prepared, operation)
                        torch.cuda.synchronize(self.device)
                        samples.append((time.perf_counter()-start)/self.measure_steps)
                    free, total = torch.cuda.mem_get_info(self.device)
                    memory = memory_headroom(total, free, torch.cuda.memory_allocated(self.device),
                                             torch.cuda.memory_reserved(self.device),
                                             torch.cuda.max_memory_allocated(self.device), headroom)
                    candidates.append(dict(plan, seconds=statistics.median(samples), timings=samples, **memory))
                    print(f"  {statistics.median(samples):.4f} s/all-model update; safe={memory['safe']}", flush=True)
                except torch.OutOfMemoryError:
                    candidates.append(dict(plan, safe=False, error='CUDA out of memory'))
                finally:
                    restore()
            safe = [row for row in candidates if row['safe']]
            if not safe:
                raise ScheduleMemoryError(candidates)
            fastest = min(row['seconds'] for row in safe)
            best = min((row for row in safe if row['seconds'] <= fastest*1.03),
                       key=lambda row: (row['max_concurrent'], row['seconds']))
            self.max_concurrent, self.groups = best['max_concurrent'], best['groups']
            self.serial_students = best['mode'] == 'serial_students'
            selected = True
            result = dict(mode=best['mode'], groups=self.groups, max_concurrent=self.max_concurrent,
                          candidates=candidates, seconds=best['seconds'], device=torch.cuda.get_device_name(self.device),
                          scope='shared prepared batch: all active forward/backward/optimizer/EMA updates',
                          headroom=headroom, warmup_steps=self.warmup_steps,
                          measure_steps=self.measure_steps, repeats=self.repeats)
            self.tuning.append(result)
            return result
        finally:
            if not selected:
                self.max_concurrent, self.groups, self.serial_students = original
