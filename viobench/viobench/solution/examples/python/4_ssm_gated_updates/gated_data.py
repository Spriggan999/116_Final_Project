"""Same recorded curriculum datasets; explicit ProcTHOR split and durable validation."""
import json
import os
import shutil
import legacy
from streaming import TrainingData
from data import EpisodeStore, FramePairs, StateWindows, StateEpisodes
from performance import loader_options
from torch.utils.data import DataLoader


class GatedTrainingData(TrainingData):
    def __init__(self, args, cfg, stack):
        if cfg.STREAM_DATA:
            super().__init__(args, cfg, stack)
        else:
            import numpy as np
            self.cfg, self.output = cfg, args.out
            self.curriculum = self.curriculum_phase = self.window_schedule = None
            self.probe_loaders = {}
            self.rolling = None
            self.epoch = 0
            self.batch_id = None
            self.loader_options = loader_options(cfg)
            self.frame_loader_options = loader_options(cfg, frame_pairs=True)
            records = []
            for path in sorted(cfg.WORKDIR.glob('rollout_s*_e*.npz')):
                with np.load(path, allow_pickle=False) as archive:
                    records.append((path, int(archive['scene_id'])))
            train_files = [p for p, scene in records if 0 <= scene < 192]
            val_files = [p for p, scene in records if 192 <= scene < 200]
            if not train_files or not val_files:
                raise ValueError('Recorded data needs training scenes 0–191 and held-out scenes 192–199')
            self.val_data = EpisodeStore(val_files, cfg.H, cfg.W)
            self.val_pairs = FramePairs(self.val_data, max_samples=cfg.MAX_FRAME_VAL_PAIRS)
            self.val_windows = StateWindows(self.val_data, cfg.WINDOW, False, True)
            self.val_state_loader = DataLoader(self.val_windows, batch_size=1)
            self.val_pair_loader = DataLoader(self.val_pairs, batch_size=1)
            self.val_episode_loader = DataLoader(StateEpisodes(self.val_data), batch_size=1)
            self._load_train(train_files)
            stack.callback(self.close)
        destination = args.out/'procthor_data'
        destination.mkdir()
        manifest = []
        for entry in self.val_data.entries:
            source = legacy.Path(entry['file'])
            target = destination/source.name
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
            manifest.append(dict(file=target.name, scene_id=entry['scene_id']))
        (destination/'manifest.json').write_text(json.dumps(manifest, indent=2))
