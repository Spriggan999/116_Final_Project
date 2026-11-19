"""Bounded rolling data without copying validation flights into the home quota."""
import json
from pathlib import Path
import numpy as np
import legacy
import resource_limits
from streaming import TrainingData
from safe_storage import SafeRollingBatches
from data import EpisodeStore, FramePairs, StateWindows, StateEpisodes
from performance import loader_options
from torch.utils.data import DataLoader


GRAVITY_SAMPLES = 10


def causal_gravity(imu, start):
    """Initial-body gravity from accelerometer samples at or before ``start``.

    Mirrors public-flight inference (evaluate.public_flight): mean of up to ten
    samples ending at the first frame. Recordings store an all-zero placeholder
    at tick 0, so placeholder rows are ignored. A window starting at tick 0 has
    no earlier sample; it uses the first real one (one IMU tick later).
    """
    history = imu[max(0, start-GRAVITY_SAMPLES+1):start+1, :3]
    history = history[history.norm(dim=-1) > 1e-6]
    if not len(history):
        later = imu[start:, :3]
        real = (later.norm(dim=-1) > 1e-6).nonzero().flatten()
        if not len(real):
            raise ValueError('Recording has no nonzero accelerometer samples')
        history = later[real[:1]]
    acceleration = history.float().mean(0)
    return -9.81*acceleration/acceleration.norm().clamp_min(1e-5)


class GatedStateWindows(StateWindows):
    """Consistent absolute velocity targets and causal gravity for corrected runs."""
    def initial_gravity(self, flight_imu, start, length):
        if getattr(self.store, 'corrected_supervision', False):
            return causal_gravity(flight_imu, start)
        return super().initial_gravity(flight_imu, start, length)

    def window(self, item, length, cache_index=None):
        result = super().window(item, length, cache_index)
        if not getattr(self.store, 'corrected_supervision', False):
            return result
        entry_index, start = item
        entry = self.store.entries[entry_index]
        raw = entry['state'][start:start+length]
        r0 = raw[0, 3:12].reshape(3, 3)
        result['vel'] = (r0.T @ raw[:, 12:15].T).T
        # Diagnostics only: frame_inputs never forwards these to the model.
        result['oracle_initial_velocity'] = r0.T @ raw[0, 12:15]
        result['oracle_gravity'] = r0.T @ raw.new_tensor([0., 0., -9.81])
        return result


class GatedStateEpisodes(GatedStateWindows, StateEpisodes):
    __init__ = StateEpisodes.__init__
    __getitem__ = StateEpisodes.__getitem__


def validation_files(manifest):
    path = Path(manifest)
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError('Validation manifest must contain recording records')
    files = []
    scenes = set()
    for record in records:
        f = Path(record['file'])
        if not f.is_absolute():
            f = path.parent/f
        f = f.resolve()
        with np.load(f, allow_pickle=False) as archive:
            scene = int(archive['scene_id'])
        if not 192 <= scene < 200 or scene != record['scene_id'] or f in files:
            raise ValueError(f'Invalid held-out recording: {f}')
        scenes.add(scene)
        files.append(f)
    if scenes != set(range(192, 200)):
        raise ValueError('Fixed validation must cover all scenes 192–199')
    return files


class GatedTrainingData(TrainingData):
    def __init__(self, args, cfg, stack):
        self.cfg, self.output = cfg, args.out
        self.corrected = getattr(args, "training_policy", "legacy") == "corrected_v1"
        fixed_validation = validation_files(args.validation_manifest) if getattr(args, "validation_manifest", None) else None
        self.curriculum = self.curriculum_phase = self.window_schedule = None
        self.probe_loaders = {}
        self.rolling = None
        self.epoch = 0
        self.batch_id = None
        self.loader_options = loader_options(cfg)
        self.frame_loader_options = loader_options(cfg, frame_pairs=True)
        if cfg.STREAM_DATA:
            if resource_limits.ACTIVE is None:
                raise RuntimeError('Start through train.py to install storage and memory safeguards')
            if args.collector is not None:
                raise ValueError('Custom collectors bypass resource controls; use the bounded gated collector')
            ring = SafeRollingBatches(args.stream_dir,cfg,resource_limits.ACTIVE,
                gpu_gate=getattr(args,'gpu_collection_gate',None),
                initial_index=getattr(args,'resume_collection_index',0),
                skip_validation=fixed_validation is not None)
            self.rolling = stack.enter_context(ring)
            stack.callback(self.close)
            self.loaded_path = ring.current
            train_files = sorted(ring.current.glob('rollout_s*_e*.npz'))
            val_files = []
            if fixed_validation is None:
                # Preserve exactly one validation set on the same disk. Never copy
                # gigabytes into the quota-limited checkpoint directory.
                retained = ring.owned.parent/(ring.owned.name+'-validation')
                ring.validation.rename(retained)
                (retained/'.gated-validation').write_text(str(args.out.resolve())+'\n')
                ring.validation = retained
                val_files = sorted(retained.glob('rollout_s*_e*.npz'))
        else:
            stack.callback(self.close)
            import numpy as np
            records = []
            for path in sorted(cfg.WORKDIR.glob('rollout_s*_e*.npz')):
                with np.load(path,allow_pickle=False) as archive:
                    records.append((path,int(archive['scene_id'])))
            train_files = [p for p,scene in records if 0<=scene<192]
            val_files = [p for p,scene in records if 192<=scene<200]
            if not train_files or not (val_files or fixed_validation):
                raise ValueError('Recorded data needs training scenes 0–191 and held-out scenes 192–199')
        self.val_data = EpisodeStore(fixed_validation or val_files,cfg.H,cfg.W)
        self.val_data.corrected_supervision = self.corrected
        self.val_pairs = FramePairs(self.val_data,max_samples=cfg.MAX_FRAME_VAL_PAIRS,raw_images=True)
        self.val_windows = GatedStateWindows(self.val_data,cfg.WINDOW,False,True)
        self.val_state_loader = DataLoader(self.val_windows,batch_size=1)
        self.val_pair_loader = DataLoader(self.val_pairs,batch_size=1)
        self.val_episode_loader = DataLoader(GatedStateEpisodes(self.val_data),batch_size=1)
        self._load_train(train_files)
        destination = args.out/'procthor_data'
        destination.mkdir()
        manifest = [dict(file=str(Path(entry['file']).resolve()),scene_id=entry['scene_id'])
                    for entry in self.val_data.entries]
        (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        resource_limits.check_write(args.out)

    def _load_train(self, paths):
        super()._load_train(paths)
        self.train_data.corrected_supervision = self.corrected
        self.ppo_val_loader = None

    def ppo_loader(self, length, validation=False, batch_size=None):
        if not self.corrected:
            return super().ppo_loader(length, validation, batch_size)
        from curriculum import state_batch_size
        count = min(state_batch_size(self.cfg, length), batch_size or self.cfg.BATCH)
        name = 'ppo_val_loader' if validation else 'ppo_state_loader'
        loader = getattr(self, name, None)
        if loader is None or loader.dataset.length != length or loader.batch_size != count:
            self._close_loader(loader)
            store = self.val_data if validation else self.train_data
            windows = GatedStateWindows(store, length, not validation, True)
            if validation and {i for i, _ in windows.items} != set(range(len(store.entries))):
                raise ValueError(f'Validation recordings lack {length}-sample windows')
            loader = DataLoader(windows, batch_size=count, shuffle=not validation, **self.loader_options)
            setattr(self, name, loader)
        return loader
