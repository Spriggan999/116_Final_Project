"""Bounded, process-isolated collection with a single prefetched batch."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque


class CollectorProcessError(RuntimeError):
    def __init__(self,code,output):
        self.output=output
        lowered=output.lower()
        self.out_of_memory='out of memory' in lowered or 'cuda_error_out_of_memory' in lowered
        self.optix_compile_failure='optixmodulecreate' in lowered and 'module compilation failed' in lowered
        super().__init__(f'Background collector exited with status {code}\n{output}')


COLLECTION_SETTINGS=frozenset(('SCENE_CATALOG','NUM_SCENES','SCENE_OFFSET','COLLECT_STEPS','STREAM_SCENES',
    'STREAM_EPISODES_PER_SCENE','STREAM_VALIDATION_SCENES','STREAM_VALIDATION_EPISODES','STREAM_MAX_MIB',
    'COLLECT_COMPRESSION_LEVEL'))


class FreshBatchPatience:
    """Charge at most one unsuccessful validation epoch to each recording batch.

    A qualifying improvement always resets patience, even on a reused batch.
    Fixed-data callers supply a new epoch ID each time to retain epoch patience.
    Curriculum promotions reset patience and protect the following adaptation
    cycles. A cycle is a fresh recording batch, or an epoch for fixed data.
    """
    def __init__(self):
        self.bad_cycles = 0
        self.last_batch = None
        self.grace_remaining = 0

    def update(self, improved, batch_id, *, promoted=False, adaptation_cycles=0):
        if promoted:
            if type(adaptation_cycles) is not int or adaptation_cycles < 0:
                raise ValueError('adaptation_cycles must be a nonnegative integer')
            self.bad_cycles = 0
            self.grace_remaining = adaptation_cycles
            self.last_batch = batch_id
            return self.bad_cycles
        if self.grace_remaining:
            if batch_id != self.last_batch:
                self.grace_remaining -= 1
            self.last_batch = batch_id
            return self.bad_cycles
        if improved:
            self.bad_cycles = 0
        elif batch_id != self.last_batch:
            self.bad_cycles += 1
        self.last_batch = batch_id
        return self.bad_cycles


class RollingBatches:
    """Own only a freshly created run directory; never prune caller datasets.

    collect is an optional injectable producer for tests. A successful producer
    must honor its byte limit and return a directory containing rollout NPZs.
    """
    def __init__(self, root, cfg, collector=None, collect=None,gpu_gate=None,initial_index=0):
        self.root = Path(root)
        self.cfg = cfg
        self.collector = Path(collector or Path(__file__).with_name('collect.py')).resolve()
        self.producer = collect or self._subprocess
        self.stop = threading.Event()
        self.gpu_gate=gpu_gate
        if gpu_gate is not None:gpu_gate.stop=self.stop
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rollout-producer')
        self.future = None
        self.current = None
        self.counter = 0
        if type(initial_index) is not int or initial_index<0:
            self.pool.shutdown()
            raise ValueError('Initial collection index must be nonnegative')
        self.initial_index=initial_index
        self.passes_on_current = 0
        self.max_passes = getattr(cfg, 'STREAM_MAX_PASSES_PER_BATCH', 3)
        # Old checkpoints and injected test collectors retain the former split.
        self.validation_scenes = getattr(cfg, 'STREAM_VALIDATION_SCENES', 1)
        self.last_wait_seconds=0.
        self.oom_retries=0
        self.oom_recoveries=0
        self.optix_compile_retries=0
        self.optix_compile_recoveries=0
        compression=getattr(cfg,'COLLECT_COMPRESSION_LEVEL',6)
        if type(compression) is not int or not 0<=compression<=9:
            self.pool.shutdown()
            raise ValueError('COLLECT_COMPRESSION_LEVEL must be an integer from 0 to 9')
        if not isinstance(self.max_passes, int) or self.max_passes < 1:
            self.pool.shutdown()
            raise ValueError('STREAM_MAX_PASSES_PER_BATCH must be a positive integer')
        self.slot_bytes = (int(cfg.STREAM_MAX_MIB * 2**20)-64*1024) // 3
        if self.slot_bytes < 128*1024:
            self.pool.shutdown()
            raise ValueError('STREAM_MAX_MIB is too small for three data slots')
        if not 1 <= cfg.STREAM_SCENES <= cfg.NUM_SCENES:
            self.pool.shutdown()
            raise ValueError('STREAM_SCENES must be between 1 and NUM_SCENES')
        if not isinstance(self.validation_scenes,int) or self.validation_scenes < 1:
            self.pool.shutdown()
            raise ValueError('STREAM_VALIDATION_SCENES must be a positive integer')
        if self.producer == self._subprocess and self.collector == Path(__file__).with_name('collect.py').resolve():
            from scenes import scene_specs
            available = len(scene_specs(getattr(cfg, 'SCENE_CATALOG', 'mixed')))
            if cfg.SCENE_OFFSET < 0 or cfg.SCENE_OFFSET + cfg.NUM_SCENES + self.validation_scenes > available:
                self.pool.shutdown()
                raise ValueError('Training and validation scene ranges exceed the selected scene catalog')
        self.owned = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.owned = self.root / ('run-' + uuid.uuid4().hex)
        self.owned.mkdir()
        (self.owned / '.owned-by-training').write_text('rolling training data\n')
        try:
            self.validation = self._generate('validation', -1)
            self.current = self._generate(f'batch-{self.initial_index:06d}', self.initial_index)
            self.counter = self.initial_index+1
            self._prefetch()
            return self
        except BaseException:
            self.close()
            raise

    def _generate(self, name, index):
        if self.stop.is_set():
            raise RuntimeError('Collection cancelled')
        directory = self.owned / name
        directory.mkdir()
        validation = index < 0
        if validation:
            offset = self.cfg.SCENE_OFFSET + self.cfg.NUM_SCENES
            scenes = self.validation_scenes
        else:
            cycle = (self.cfg.NUM_SCENES+self.cfg.STREAM_SCENES-1)//self.cfg.STREAM_SCENES
            start = (index % cycle)*self.cfg.STREAM_SCENES
            offset = self.cfg.SCENE_OFFSET + start
            scenes = min(self.cfg.STREAM_SCENES, self.cfg.NUM_SCENES-start)
        spec = dict(scene_offset=offset, scenes=scenes,
                    scene_catalog=getattr(self.cfg,'SCENE_CATALOG','mixed'),
                    episodes=self.cfg.STREAM_VALIDATION_EPISODES if validation else self.cfg.STREAM_EPISODES_PER_SCENE,
                    seed=self.cfg.SEED + (10_000_000 if validation else 1009*index),
                    instances=1 if validation else getattr(self.cfg,'COLLECT_INSTANCES',2),
                    compression_level=getattr(self.cfg,'COLLECT_COMPRESSION_LEVEL',6),
                    steps=self.cfg.COLLECT_STEPS, width=self.cfg.SOURCE_WIDTH,
                    height=self.cfg.SOURCE_HEIGHT, max_bytes=self.slot_bytes-64*1024)
        try:
            self.producer(directory, spec, self.stop)
        except CollectorProcessError as error:
            if not (error.out_of_memory or error.optix_compile_failure) or self.gpu_gate is None or self.stop.is_set():raise
            if error.out_of_memory:
                kind='GPU OOM';counter='oom'
            else:
                # This signature does not establish an OOM. Try a fresh native
                # context with less GPU pressure once; persistent errors fail.
                kind='OptiX module compilation failure';counter='optix_compile'
            setattr(self,counter+'_retries',getattr(self,counter+'_retries')+1)
            print(f'Collector {kind} in {name}; pausing training GPU work, releasing unused cache, '
                  'and retrying this batch once with one flight at a time.',flush=True)
            # This is the owned, not-yet-published slot; no training loader can
            # be reading its partial files. Keep current/validation untouched.
            shutil.rmtree(directory);directory.mkdir()
            spec=dict(spec,instances=1)
            with self.gpu_gate.exclusive(self.stop):
                self.producer(directory,spec,self.stop)
            setattr(self,counter+'_recoveries',getattr(self,counter+'_recoveries')+1)
            print(f'Collector {kind} recovery completed for {name}; parallel training resumes.',flush=True)
        if self.stop.is_set():
            raise RuntimeError('Collection cancelled')
        if not list(directory.glob('rollout_s*_e*.npz')):
            raise RuntimeError(f'Collector produced no episodes in {directory}')
        total = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
        if total > self.slot_bytes:
            raise RuntimeError(f'Collector exceeded its data-slot budget ({total} bytes)')
        manifest=directory/'manifest.json'
        if validation and manifest.is_file():
            import json
            seen={row['scene_index'] for row in json.loads(manifest.read_text())}
            missing=set(range(offset,offset+self.validation_scenes))-seen
            if missing:
                raise RuntimeError(f'Validation collection missed scenes {sorted(missing)}; '
                                   'increase --max-data-mib or reduce the validation recording size')
        return directory

    def _subprocess(self, directory, spec, stop):
        command = [sys.executable, '-u', str(self.collector), '--out', str(directory)]
        for flag, key in [('scene-offset','scene_offset'), ('scenes','scenes'), ('episodes','episodes'),
                          ('seed','seed'), ('steps','steps'), ('width','width'), ('height','height')]:
            command.extend(['--'+flag, str(spec[key])])
        command.extend(['--max-output-mib', str(spec['max_bytes']/2**20)])
        if self.collector==Path(__file__).with_name('collect.py').resolve():
            command.extend(['--instances',str(spec.get('instances',2))])
            command.extend(['--scene-catalog',spec.get('scene_catalog','mixed')])
            command.extend(['--compression-level',str(spec.get('compression_level',6))])
        # Stream output instead of accumulating an unbounded subprocess log.
        child = subprocess.Popen(command, start_new_session=os.name == 'posix',
                                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1)
        tail=deque(maxlen=80)
        diagnostics={}
        def forward_output():
            for line in child.stdout:
                print(line,end='',flush=True)
                tail.append(line[-4096:])
                # Native compilers can print many informational lines after
                # the actual failure. Retain each signature outside the tail.
                for marker in ('out of memory','cuda_error_out_of_memory','optixmodulecreate','module compilation failed'):
                    if marker in line.lower():diagnostics[marker]=line[-4096:]
        reader=threading.Thread(target=forward_output,name='collector-output',daemon=True)
        reader.start()
        started = time.monotonic()
        try:
            while child.poll() is None:
                if stop.wait(.1):
                    raise RuntimeError('Collection cancelled')
                if time.monotonic()-started > self.cfg.STREAM_COLLECT_TIMEOUT:
                    raise TimeoutError('Background collector exceeded STREAM_COLLECT_TIMEOUT')
            if child.returncode:
                reader.join(timeout=5)
                retained=list(tail)
                diagnostic_lines=list(dict.fromkeys(diagnostics.values()))
                output=''.join(line for line in diagnostic_lines if line not in retained)+''.join(retained)
                raise CollectorProcessError(child.returncode,output)
        finally:
            if child.poll() is None:
                if os.name == 'posix':
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                else:
                    child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == 'posix':
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        child.kill()
                    child.wait()
            reader.join(timeout=5)
            if not reader.is_alive():child.stdout.close()

    def _prefetch(self):
        index = self.counter
        self.counter += 1
        self.future = self.pool.submit(self._generate, f'batch-{index:06d}', index)

    def next_epoch(self):
        self.last_wait_seconds=0.
        if self.passes_on_current:
            if not self.future.done() and self.passes_on_current < self.max_passes:
                self.passes_on_current += 1
                return self.current
            if not self.future.done():
                print(f'Training batch {self.current.name}: {self.max_passes} passes used; '
                      'waiting for fresh recordings.', flush=True)
            # The caller has finished its old DataLoader iteration. Its NPZ files
            # were fully loaded and closed by EpisodeStore, with no active mmap.
            waiting=time.perf_counter()
            following = self.future.result()
            self.last_wait_seconds=time.perf_counter()-waiting
            old = self.current
            self.current = following
            self.future = None
            shutil.rmtree(old)
            self.passes_on_current = 0
            self._prefetch()
        self.passes_on_current += 1
        return self.current

    def close(self):
        self.stop.set()
        if self.future is not None:
            self.future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.owned is not None:
            # Delete only the unique directory this instance created, after its
            # worker has exited. Parent and other runs are never traversed.
            shutil.rmtree(self.owned)
            self.owned = None

    def __exit__(self, *_):
        self.close()


class TrainingData:
    """Fixed validation and bounded reuse while the next batch is collected."""
    def __init__(self, args, cfg, stack):
        import numpy as np
        self.cfg = cfg
        self.output = args.out
        self.curriculum = None
        self.curriculum_phase = None
        self.window_schedule = None
        self.probe_loaders = {}
        if getattr(cfg,'USE_CURRICULUM',False):
            cfg.CACHE_GENERATED_WINDOWS = False
            print('Curriculum: generated-window caches disabled; camera timing stays fixed.',flush=True)
        self.rolling = None
        self.epoch = 0
        self.batch_id = None
        if cfg.STREAM_DATA:
            if cfg.CACHE_GENERATED_WINDOWS:
                print('Rolling mode: synthetic-window disk caches disabled.', flush=True)
            cfg.CACHE_GENERATED_WINDOWS = False
            root = args.stream_dir or args.out/'stream-data'
            self.rolling = stack.enter_context(RollingBatches(root, cfg, args.collector,
                                               gpu_gate=getattr(args,'gpu_collection_gate',None),
                                               initial_index=getattr(args,'resume_collection_index',0)))
            self.loaded_path = self.rolling.current
            train_files = sorted(self.rolling.current.glob('rollout_s*_e*.npz'))
            val_files = sorted(self.rolling.validation.glob('rollout_s*_e*.npz'))
        else:
            records = []
            for path in sorted(cfg.WORKDIR.glob('rollout_s*_e*.npz')):
                with np.load(path, allow_pickle=False) as data:
                    records.append((path, int(data['scene_id'])))
            if len({scene for _, scene in records}) < 2:
                raise ValueError('Need at least two scenes in the recorded data')
            heldout = max(scene for _, scene in records)
            train_files = [p for p, scene in records if scene != heldout]
            val_files = [p for p, scene in records if scene == heldout]
        from data import EpisodeStore, FramePairs, StateWindows, StateEpisodes
        from torch.utils.data import DataLoader
        from performance import loader_options
        self.loader_options=loader_options(cfg)
        self.frame_loader_options=loader_options(cfg,frame_pairs=True)
        stack.callback(self.close)
        self.val_data = EpisodeStore(val_files, cfg.H, cfg.W)
        self.val_pairs = FramePairs(self.val_data, max_samples=cfg.MAX_FRAME_VAL_PAIRS, seed=cfg.SEED+1,
                                    raw_images=getattr(cfg,'FRAME_PAIRS_UINT8',False))
        self.val_windows = StateWindows(self.val_data, length=cfg.WINDOW, training=False,
                                       compact_cameras=getattr(cfg,'COMPACT_STATE_CAMERAS',True))
        self.val_pair_loader = DataLoader(self.val_pairs,batch_size=getattr(cfg,'FRAME_VAL_PAIR_BATCH',cfg.FRAME_PAIR_BATCH),**self.frame_loader_options)
        self.val_state_loader = DataLoader(self.val_windows, batch_size=cfg.BATCH,**self.loader_options)
        self.val_episode_loader=DataLoader(StateEpisodes(self.val_data,
            compact_cameras=getattr(cfg,'COMPACT_STATE_CAMERAS',True)),batch_size=1,
            num_workers=0,pin_memory=self.loader_options.get('pin_memory',False))
        self._load_train(train_files)

    def _load_train(self, paths):
        from data import EpisodeStore, FramePairs, StateWindows
        from torch.utils.data import DataLoader
        cfg = self.cfg
        # Persistent workers must never keep an earlier rolling dataset alive.
        for name in ('train_pair_loader','train_state_loader','ppo_state_loader',
                     'curriculum_pair_loader','curriculum_state_loader'):
            self._close_loader(getattr(self,name,None))
        self.ppo_state_loader = None
        self.train_data = EpisodeStore(paths, cfg.H, cfg.W)
        self.train_pairs = FramePairs(self.train_data, max_samples=cfg.MAX_FRAME_TRAIN_PAIRS,
                                      seed=cfg.SEED + (self.rolling.counter if self.rolling else 0),
                                      raw_images=getattr(cfg,'FRAME_PAIRS_UINT8',False))
        self.train_windows = StateWindows(self.train_data, length=cfg.WINDOW, training=True,
                                         compact_cameras=getattr(cfg,'COMPACT_STATE_CAMERAS',True))
        self.train_pair_loader = DataLoader(self.train_pairs, batch_size=cfg.FRAME_PAIR_BATCH, shuffle=True,**self.frame_loader_options)
        self.train_state_loader = DataLoader(self.train_windows, batch_size=cfg.BATCH, shuffle=True,**self.loader_options)

    def begin_phase(self, phase):
        if not getattr(self.cfg,'USE_CURRICULUM',False):
            return
        from curriculum import DifficultyCurriculum, LearningProgress
        from ppo import WindowCurriculum
        for name in ('curriculum_pair_loader','curriculum_state_loader'):
            self._close_loader(getattr(self,name,None))
            setattr(self,name,None)
        for loader in self.probe_loaders.values():
            self._close_loader(loader)
        self.probe_loaders = {}
        self.curriculum_phase = phase
        self.phase_epochs=0
        self.last_curriculum_record=None
        self.final_stage_cycles=0
        self.final_stage_last_batch=None
        previous_stages = dict(self.curriculum.stages) if self.curriculum is not None else None
        self.curriculum = DifficultyCurriculum(self.cfg)
        if phase=='ppo' and previous_stages is not None:
            self.curriculum.stages.update(previous_stages)
        self.progress = LearningProgress() if self.cfg.CURRICULUM_PRIORITIZE else None
        self.pending_priority=[]
        self.window_schedule = WindowCurriculum(self.cfg, prefix='SL') if phase in ('state','teacher','distill') else None
        if self.window_schedule:
            # One fixed long horizon for all supervised checkpoint comparisons.
            self.curriculum_validation_loader = self.ppo_loader(self.window_schedule.windows[-1], validation=True)
        print(f'Curriculum phase {phase}: independent motion/visual/IMU gates enabled.',flush=True)

    def state_validation_loader(self):
        return getattr(self,'curriculum_validation_loader',self.val_state_loader)

    def _curriculum_loader(self, base, pairs=False):
        from curriculum import AugmentedDataset, curriculum_sampler, state_batch_size
        from torch.utils.data import DataLoader
        name = 'curriculum_pair_loader' if pairs else 'curriculum_state_loader'
        self._close_loader(getattr(self,name,None))
        seed = self.cfg.SEED + 100_003*self.epoch
        augmented = AugmentedDataset(base,self.cfg,self.curriculum,seed)
        sampler = curriculum_sampler(base,self.curriculum,self.progress,seed)
        loader = DataLoader(augmented, batch_size=self.cfg.FRAME_PAIR_BATCH if pairs else state_batch_size(self.cfg,base.length),
                            sampler=sampler, **(self.frame_loader_options if pairs else self.loader_options))
        setattr(self,name,loader)
        return loader

    def ppo_training_loader(self, length):
        loader = self.ppo_loader(length)
        return self._curriculum_loader(loader.dataset) if self.curriculum is not None else loader

    def observe_errors(self, batch, errors):
        if self.curriculum is not None and self.progress is not None and 'curriculum_key' in batch:
            # Do not synchronize the GPU for tiny diagnostic transfers every batch.
            self.pending_priority.append((list(batch['curriculum_key']),errors.detach()))

    def flush_priorities(self):
        if not self.pending_priority:return
        import torch
        keys=[key for batch_keys,_ in self.pending_priority for key in batch_keys]
        values=torch.cat([errors for _,errors in self.pending_priority])
        self.progress.update(keys,values)
        self.pending_priority=[]

    def observe_prediction(self, batch, prediction):
        import torch
        if self.curriculum is None:
            return
        with torch.no_grad():
            errors = (prediction['pos']-batch['pos'].to(prediction['pos'].device)).square().mean((1,2))
        self.observe_errors(batch,errors)

    def _probes(self, axis, length=None):
        pairs = self.curriculum_phase=='frame'
        if not pairs and length is None:
            length=self.window_schedule.length if self.window_schedule else getattr(self,'ppo_probe_length',None)
        key=(axis,length)
        if key in self.probe_loaders:
            return self.probe_loaders[key]
        import torch
        from torch.utils.data import DataLoader,Subset
        from curriculum import AugmentedDataset,motion_scores,seed_for,state_batch_size
        from data import StateWindows
        base = self.val_pairs if pairs else (StateWindows(self.val_data,length=length,training=False,
                                              compact_cameras=getattr(self.cfg,'COMPACT_STATE_CAMERAS',True))
                                             if length is not None else self.state_validation_loader().dataset)
        count = getattr(self.cfg,'FRAME_VAL_PAIR_BATCH',self.cfg.FRAME_PAIR_BATCH) if pairs else state_batch_size(self.cfg,base.length)
        candidates = torch.arange(len(base))
        if axis=='motion':
            scores = motion_scores(base)
            candidates = candidates[scores >= torch.quantile(scores,.67)]
        n = min(len(candidates),count*self.cfg.CURRICULUM_PROBE_BATCHES)
        positions = torch.linspace(0,len(candidates)-1,n).round().long()
        indices = candidates[positions].tolist()
        view = AugmentedDataset(base,self.cfg,self.curriculum,self.cfg.SEED+70_001,probe=axis)
        # Fixed sample indices and augmentation seeds, independent of training RNG.
        loader = DataLoader(Subset(view,indices),batch_size=count,
                            generator=torch.Generator().manual_seed(seed_for(self.cfg.SEED,'probe',self.curriculum_phase,axis)),
                            num_workers=0,pin_memory=self.loader_options.get('pin_memory',False))
        self.probe_loaders[key] = loader
        return loader

    def progress_score(self,length,evaluate):
        return float(evaluate(self._probes('window',length=length)))

    def minimum_phase_epochs(self):
        if self.curriculum is None or self.curriculum_phase=='ppo':return 0
        key='CURRICULUM_MIN_FRAME_EPOCHS' if self.curriculum_phase=='frame' else 'CURRICULUM_MIN_STATE_EPOCHS'
        return getattr(self.cfg,key,0)

    def curriculum_complete(self):
        if self.curriculum is None or self.curriculum_phase=='ppo':return False
        axes_complete=all(stage==len(self.curriculum.levels)-1 for stage in self.curriculum.stages.values())
        window_complete=self.window_schedule is None or self.window_schedule.stage==len(self.window_schedule.windows)-1
        return axes_complete and window_complete

    def early_stopping_eligible(self):
        if self.curriculum is None or self.curriculum_phase=='ppo':return True
        if self.phase_epochs<=self.minimum_phase_epochs():return False
        if not getattr(self.cfg,'CURRICULUM_REQUIRE_COMPLETION',True):return True
        # Complete the requested final-stage batches before starting ordinary
        # patience on the next fresh batch. Promotion epochs cannot count here.
        return (self.curriculum_complete() and
                self.final_stage_cycles>getattr(self.cfg,'CURRICULUM_FINAL_STAGE_CYCLES',3))

    def update_patience(self,patience,improved,promoted):
        protected=not self.early_stopping_eligible()
        return patience.update(improved or protected,self.batch_id,promoted=promoted,
                               adaptation_cycles=self.cfg.CURRICULUM_EARLY_STOP_GRACE)

    def curriculum_log(self):
        row=getattr(self,'last_curriculum_record',None)
        if row is None:return {}
        metrics={'curriculum/phase_epoch':row['phase_epoch'],
                 'curriculum/promoted':int(bool(row['promoted'] or row['window_promoted'])),
                 'curriculum/promotion_count':len(row['promoted'])+int(row['window_promoted']),
                 'curriculum/minimum_epochs_remaining':row['minimum_epochs_remaining'],
                 'curriculum/complete':int(row['curriculum_complete']),
                 'curriculum/final_stage_cycles':row['final_stage_cycles'],
                 'curriculum/final_stage_cycles_remaining':row['final_stage_cycles_remaining'],
                 'curriculum/early_stopping_eligible':int(row['early_stopping_eligible'])}
        for axis,value in row['levels'].items():metrics['curriculum/'+axis+'_level']=value
        for axis,value in row['next_levels'].items():metrics['curriculum/next_'+axis+'_level']=value
        for axis,value in row['probe_losses'].items():metrics['curriculum/'+axis+'_probe_loss']=value
        for axis in (*row['levels'],'window'):
            promoted=row['window_promoted'] if axis=='window' else axis in row['promoted']
            metrics['curriculum/'+axis+'_promoted']=int(promoted)
            reason=row['promotion_reasons'].get(axis,'none')
            metrics['curriculum/'+axis+'_promotion_reason']=reason
            for cause in ('improved','plateau'):
                metrics['curriculum/'+axis+'_promoted_'+cause]=int(promoted and reason==cause)
        if row['progress_validation_loss'] is not None:metrics['curriculum/progress_loss']=row['progress_validation_loss']
        if row['probe_window_samples'] is not None:metrics['curriculum/window_samples']=row['probe_window_samples']
        if row['next_window_samples'] is not None:metrics['curriculum/next_window_samples']=row['next_window_samples']
        return metrics

    def step_scheduler(self,scheduler,score):
        row=getattr(self,'last_curriculum_record',None)
        if row and row['window_promoted']:
            # The next horizon has a different loss scale. Preserve the current
            # LR but restart plateau tracking rather than comparing to old units.
            state=scheduler.state_dict()
            state.update(best=float('inf'),num_bad_epochs=0,cooldown_counter=0)
            scheduler.load_state_dict(state)
        else:
            progress=row.get('progress_validation_loss') if row else None
            scheduler.step(score if progress is None else progress)

    def finish_curriculum_epoch(self, score, evaluate, *, window_event=None):
        if self.curriculum is None:
            return False
        import json,math
        if not math.isfinite(score):raise ValueError('Nonfinite checkpoint validation loss')
        self.flush_priorities()
        was_complete=self.curriculum_complete()
        used = self.curriculum.snapshot()
        scores = {axis:float(evaluate(self._probes(axis))) for axis in self.curriculum.axes}
        promoted = self.curriculum.update(scores)
        window = self.window_schedule.length if self.window_schedule else None
        window_promoted = False
        stage_score=None
        reasons={axis:self.curriculum.gates[axis].last_reason for axis in promoted}
        if self.window_schedule:
            stage_score=self.progress_score(window,evaluate)
            window_promoted = self.window_schedule.update(stage_score)
            if window_promoted:
                reasons['window']=self.window_schedule.last_reason
                self.curriculum.reset_references()
        elif window_event is not None:
            # PPO owns its horizon schedule; include its event in the shared
            # record so all stage dashboards expose the same promotion keys.
            window=window_event['window_samples']
            stage_score=window_event['progress_validation_loss']
            window_promoted=window_event['promoted']
            if window_promoted:
                reasons['window']=window_event['reason']
                self.curriculum.reset_references()
        complete=self.curriculum_complete()
        if complete:
            if not was_complete:
                # This batch trained the previous stage. Its promotion and any
                # reuse of the same recordings do not satisfy final exposure.
                self.final_stage_cycles=0
                self.final_stage_last_batch=self.batch_id
            elif self.batch_id!=self.final_stage_last_batch:
                self.final_stage_cycles+=1
                self.final_stage_last_batch=self.batch_id
        reset_patience = bool(promoted or window_promoted) and self.curriculum_phase!='ppo'
        grace = self.cfg.CURRICULUM_EARLY_STOP_GRACE if reset_patience else 0
        row = dict(phase=self.curriculum_phase,epoch=self.epoch,validation_loss=score,
                   phase_epoch=self.phase_epochs,minimum_epochs_remaining=max(0,self.minimum_phase_epochs()-self.phase_epochs),
                   curriculum_complete=complete,final_stage_cycles=self.final_stage_cycles,
                   final_stage_cycles_remaining=max(0,getattr(self.cfg,'CURRICULUM_FINAL_STAGE_CYCLES',3)-self.final_stage_cycles),
                   early_stopping_eligible=self.early_stopping_eligible(),
                   progress_validation_loss=stage_score,probe_window_samples=window or getattr(self,'ppo_probe_length',None),
                   promotion_reasons=reasons,
                   probe_losses=scores,levels=used,next_levels=self.curriculum.snapshot(),
                   promoted=promoted,window_samples=window,
                   next_window_samples=(self.window_schedule.length if self.window_schedule else
                                        window_event['next_window_samples'] if window_event is not None else None),
                   window_promoted=window_promoted,recording_batch=self.batch_id,
                   early_stopping_reset=reset_patience,early_stopping_grace_cycles=grace,
                   priority_examples=len(self.progress.history) if self.progress is not None else 0)
        self.last_curriculum_record=row
        with (self.output/'curriculum_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        state = dict(phase=self.curriculum_phase,levels=self.curriculum.stages,
                     curriculum_complete=complete,final_stage_cycles=self.final_stage_cycles,
                     early_stopping_eligible=row['early_stopping_eligible'],
                     gates={axis:vars(gate) for axis,gate in self.curriculum.gates.items()},
                     window=vars(self.window_schedule) if self.window_schedule else None)
        (self.output/'curriculum_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(f'Curriculum {self.curriculum_phase}: levels={row["next_levels"]}; '
              f'next_window={row["next_window_samples"]}; promoted={promoted}',flush=True)
        if reset_patience:
            print(f'Early-stopping patience reset; adapting for {grace} fresh data cycles.',flush=True)
        return reset_patience

    def ppo_loader(self, length, validation=False, batch_size=None):
        """Separate PPO windows leave SL and fixed validation datasets intact."""
        from data import StateWindows
        from torch.utils.data import DataLoader
        from curriculum import state_batch_size
        if length == self.cfg.WINDOW and batch_size is None:
            return self.val_state_loader if validation else self.train_state_loader
        name = 'ppo_val_loader' if validation else 'ppo_state_loader'
        loader = getattr(self, name, None)
        count=state_batch_size(self.cfg,length)
        if batch_size is not None:count=min(count,batch_size)
        if loader is None or loader.dataset.length != length or loader.batch_size != count:
            self._close_loader(loader)
            store = self.val_data if validation else self.train_data
            try:
                windows = StateWindows(store, length=length, training=not validation,
                                       compact_cameras=getattr(self.cfg,'COMPACT_STATE_CAMERAS',True))
            except RuntimeError as error:
                raise ValueError(f'No {length}-sample PPO {"validation" if validation else "training"} windows; '
                                 'increase COLLECT_STEPS and the recording budget, or shorten the curriculum. '
                                 'Crashes can also shorten recordings.') from error
            if validation:
                covered = {index for index, _ in windows.items}
                missing = [entry['file'] for index, entry in enumerate(store.entries) if index not in covered]
                if missing:
                    raise ValueError(f'PPO validation recordings lack {length}-sample windows: {missing}; '
                                     'recollect longer collision-free validation flights or shorten the curriculum')
            loader = DataLoader(windows, batch_size=count, shuffle=not validation, **self.loader_options)
            setattr(self, name, loader)
        return loader

    @staticmethod
    def _close_loader(loader):
        iterator=getattr(loader,'_iterator',None)
        if iterator is not None:
            # PyTorch DataLoader has no public close() for persistent workers.
            # Use its iterator's teardown before dropping references to data.
            iterator._shutdown_workers()
            loader._iterator=None

    def close(self):
        for name in ('train_pair_loader','train_state_loader','val_pair_loader','val_state_loader','val_episode_loader',
                     'ppo_state_loader','ppo_val_loader','curriculum_pair_loader','curriculum_state_loader'):
            self._close_loader(getattr(self,name,None))
        for loader in getattr(self,'probe_loaders',{}).values():
            self._close_loader(loader)

    def next_epoch(self):
        self.epoch += 1
        if self.curriculum is not None:self.phase_epochs+=1
        if self.rolling:
            path = self.rolling.next_epoch()
            reused = path == self.loaded_path and self.rolling.passes_on_current > 1
            if path != self.loaded_path:
                self._load_train(sorted(path.glob('rollout_s*_e*.npz')))
                self.loaded_path = path
            self.batch_id = str(path)
            print(f'Training batch: {path.name}; pass {self.rolling.passes_on_current}/'
                  f'{self.rolling.max_passes} ({"reused" if reused else "fresh"}); '
                  f'fresh-data wait={self.rolling.last_wait_seconds:.3f}s; next batch collecting in background.', flush=True)
        else:
            # An ordinary fixed-data epoch retains the previous stopping policy.
            self.batch_id = self.epoch
        if self.curriculum is not None:
            if self.curriculum_phase=='frame':
                return self._curriculum_loader(self.train_pairs,pairs=True), self.train_state_loader
            if self.curriculum_phase in ('state','teacher','distill'):
                from data import StateWindows
                try:
                    base = StateWindows(self.train_data,length=self.window_schedule.length,training=True,
                                        compact_cameras=getattr(self.cfg,'COMPACT_STATE_CAMERAS',True))
                except RuntimeError as error:
                    raise ValueError(f'No {self.window_schedule.length}-sample SL curriculum windows; '
                                     'increase COLLECT_STEPS or shorten SL_CURRICULUM_WINDOWS') from error
                return self.train_pair_loader, self._curriculum_loader(base)
        return self.train_pair_loader, self.train_state_loader
