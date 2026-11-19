"""Training-only difficulty schedules, consistent augmentation and sampling."""
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler


def seed_for(seed, *parts):
    digest = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return (int.from_bytes(digest, 'little') + seed) % (2**63-1)


def sample_key(dataset, item):
    return f'{dataset.store.entries[item[0]]["file"]}:{getattr(dataset,"length",None)}:{item[1:]}'


def flight_identity(entry):
    return entry.get('scene_id'), Path(entry['file']).name


class ProgressGate:
    def __init__(self, minimum=3, patience=2, improvement=.02, maximum=0, tolerance=.1):
        if (type(minimum) is not int or minimum < 1 or type(patience) is not int
                or patience < 1 or not 0 < improvement < 1):
            raise ValueError('Invalid curriculum promotion settings')
        self.minimum, self.patience, self.improvement = minimum, patience, improvement
        if type(maximum) is not int or maximum < 0 or (maximum and maximum < minimum) or not 0 <= tolerance <= 1:
            raise ValueError('Invalid curriculum plateau advancement settings')
        self.maximum,self.tolerance=maximum,tolerance
        self.reference = None
        self.best = None
        self.age = self.streak = 0
        self.last_reason = None

    def update(self, score):
        self.last_reason=None
        if not math.isfinite(score):
            raise ValueError('Nonfinite curriculum validation score')
        if self.reference is None:
            self.reference = self.best = score
            return False
        self.age += 1
        self.best=min(self.best,score)
        good = self.reference-score > self.improvement*max(abs(self.reference), 1e-12)
        self.streak = self.streak+1 if good else 0
        improved=self.age >= self.minimum and self.streak >= self.patience
        plateau=bool(self.maximum and self.age >= self.maximum and score <= self.best+self.tolerance*max(abs(self.best),1e-8))
        if improved or plateau:
            self.last_reason='improved' if improved else 'plateau'
            self.reference = score
            self.best = score
            self.age = self.streak = 0
            return True
        return False


class DifficultyCurriculum:
    axes = ('motion', 'visual', 'imu')

    def __init__(self, cfg):
        self.levels = list(cfg.CURRICULUM_LEVELS)
        if (len(self.levels) < 2 or self.levels[0] != 0 or self.levels[-1] != 1
                or any(not math.isfinite(n) for n in self.levels)
                or any(b <= a for a, b in zip(self.levels, self.levels[1:]))):
            raise ValueError('CURRICULUM_LEVELS must increase from 0 to 1')
        self.mix = list(cfg.CURRICULUM_MIX)
        self.all_easier=getattr(cfg,'CURRICULUM_ALL_EASIER',False)
        if (len(self.mix) != 3 or any(not math.isfinite(n) or n < 0 for n in self.mix)
                or abs(sum(self.mix)-1) > 1e-6):
            raise ValueError('CURRICULUM_MIX must contain current/easier/harder probabilities summing to 1')
        self.stages = dict.fromkeys(self.axes, 0)
        self.gates = {axis: ProgressGate(cfg.CURRICULUM_MIN_EPOCHS, cfg.CURRICULUM_PATIENCE,
                                       cfg.CURRICULUM_REL_IMPROVEMENT,
                                       getattr(cfg,'CURRICULUM_MAX_STAGE_EPOCHS',0),
                                       getattr(cfg,'CURRICULUM_PLATEAU_TOLERANCE',.1)) for axis in self.axes}

    def update(self, scores):
        promoted = []
        for axis, gate in self.gates.items():
            if self.stages[axis]+1 < len(self.levels) and gate.update(scores[axis]):
                self.stages[axis] += 1
                promoted.append(axis)
        return promoted

    def snapshot(self):
        return {axis: self.levels[index] for axis, index in self.stages.items()}

    def reset_references(self):
        for gate in self.gates.values():
            gate.reference=gate.best=None
            gate.age=gate.streak=0


def state_batch_size(cfg,length):
    budget=getattr(cfg,'CURRICULUM_BATCH_TOKENS',0) if getattr(cfg,'USE_CURRICULUM',False) else 0
    return min(cfg.BATCH,max(1,budget//length)) if budget else cfg.BATCH


def mixed_level(stage, levels, mix, generator, all_easier=False):
    choice = int(torch.multinomial(torch.tensor(mix), 1, generator=generator))
    if choice==1 and all_easier and stage>0:
        return levels[int(torch.randint(stage,(1,),generator=generator))]
    index = (stage, max(0, stage-1), min(len(levels)-1, stage+1))[choice]
    return levels[index]


def noisy_imu(imu, dt, strength, cfg, generator):
    """Add sensor errors once per flight; preserve native noise and stored truth."""
    if strength == 0:
        return imu
    white = imu.new_tensor([cfg.CURRICULUM_ACCEL_NOISE]*3 + [cfg.CURRICULUM_GYRO_NOISE]*3)
    bias = imu.new_tensor([cfg.CURRICULUM_ACCEL_BIAS]*3 + [cfg.CURRICULUM_GYRO_BIAS]*3)
    walk = imu.new_tensor([cfg.CURRICULUM_ACCEL_WALK]*3 + [cfg.CURRICULUM_GYRO_WALK]*3)
    if getattr(cfg,'CURRICULUM_RANDOM_ERROR_STRENGTH',False):
        # Per-flight, per-channel amplitude draws. Overlapping windows retain
        # identical sensor errors; the normal innovations remain random too.
        amplitudes=.5+.5*torch.rand((3,6),generator=generator)
        white=white*amplitudes[0].to(imu.device)
        bias=bias*amplitudes[1].to(imu.device)
        walk=walk*amplitudes[2].to(imu.device)
    offset = (2*torch.rand(6, generator=generator)-1)*bias
    drift = torch.randn(imu.shape, generator=generator).cumsum(0)*walk*math.sqrt(dt)
    result = imu + strength*(offset + drift + torch.randn(imu.shape, generator=generator)*white)
    # Tick zero in recordings is a placeholder, not a measured IMU sample.
    result[0] = imu[0]
    return result


def photometric_parameters(strength,cfg,generator):
    if strength==0:return (1.,1.,0.)
    brightness=1+strength*cfg.CURRICULUM_BRIGHTNESS*(2*float(torch.rand((),generator=generator))-1)
    contrast=1+strength*cfg.CURRICULUM_CONTRAST*(2*float(torch.rand((),generator=generator))-1)
    blur=strength*cfg.CURRICULUM_BLUR
    if getattr(cfg,'CURRICULUM_RANDOM_ERROR_STRENGTH',False):
        blur*=float(torch.rand((),generator=generator))
    return brightness,contrast,blur


def difficult_images(images, strength, cfg, generator):
    """One photometric transform per sample, shared by all its images; no warps."""
    if strength == 0:
        return images
    byte = images.dtype == torch.uint8
    output = torch.empty_like(images)
    brightness,contrast,blur=photometric_parameters(strength,cfg,generator)
    for start in range(0, len(images), 8):
        x = images[start:start+8].float()/(255. if byte else 1.)
        x = ((x-.5)*contrast+.5)*brightness
        if blur:
            x = (1-blur)*x + blur*F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
        x = x.clamp(0, 1)
        output[start:start+8] = (x*255).round().to(torch.uint8) if byte else x
    return output


@torch.no_grad()
def photometric_captures(images,parameters,chunk=16):
    """Apply per-flight appearance parameters in bounded device-side chunks."""
    host_parameters=parameters.device.type=='cpu'
    if host_parameters and torch.equal(parameters,parameters.new_tensor([1.,1.,0.]).expand_as(parameters)):
        return images
    byte=images.dtype==torch.uint8
    result=torch.empty_like(images)
    for index in range(len(images)):
        brightness,contrast,blur=(parameters[index].tolist() if host_parameters
                                  else parameters[index].to(images.device).unbind())
        if host_parameters and (brightness,contrast,blur)==(1.,1.,0.):
            result[index].copy_(images[index])
            continue
        for start in range(0,images.shape[1],chunk):
            source=images[index,start:start+chunk]
            x=source.float()/(255. if byte else 1.)
            x=((x-.5)*contrast+.5)*brightness
            if not host_parameters or blur:
                x=(1-blur)*x+blur*F.avg_pool2d(x,3,stride=1,padding=1,count_include_pad=False)
            x=x.clamp(0,1)
            result[index,start:start+chunk]=(x*255).round().to(torch.uint8) if byte else x
    return result


class AugmentedDataset(Dataset):
    """Epoch-local view. Workers own cached noisy IMU; base stores stay immutable."""
    def __init__(self, base, cfg, schedule, seed, probe=None):
        self.base, self.seed, self.probe = base, seed, probe
        # config.py is a module, which spawn/forkserver workers cannot pickle.
        self.cfg = SimpleNamespace(**{key:value for key,value in vars(cfg).items()
                                      if key.startswith('CURRICULUM_')})
        self.items, self.store = base.items, base.store
        self.length = getattr(base, 'length', None)
        self.levels, self.mix = schedule.levels, schedule.mix
        self.all_easier=getattr(schedule,'all_easier',False)
        self.stages = dict(schedule.stages)
        self.imu_cache = {}

    def __len__(self):
        return len(self.base)

    def level(self, axis, entry):
        if self.probe is not None:
            return float(self.probe == axis)
        generator = torch.Generator().manual_seed(seed_for(self.seed, flight_identity(entry), axis))
        return mixed_level(self.stages[axis], self.levels, self.mix, generator,self.all_easier)

    def flight_imu(self, eidx):
        if eidx not in self.imu_cache:
            entry = self.store.entries[eidx]
            generator = torch.Generator().manual_seed(seed_for(self.seed, flight_identity(entry), 'sensor'))
            self.imu_cache[eidx] = noisy_imu(entry['imu'], entry['dt'], self.level('imu', entry),
                                              self.cfg, generator)
        return self.imu_cache[eidx]

    def __getitem__(self, index):
        result = dict(self.base[index])
        item = self.items[index]
        entry = self.store.entries[item[0]]
        generator = torch.Generator().manual_seed(seed_for(self.seed, flight_identity(entry), 'appearance'))
        strength = self.level('visual', entry)
        if self.length is None:
            _, previous, anchor, target = item
            if getattr(self.cfg,'CURRICULUM_DEVICE_PHOTOMETRIC',False):
                result['photometric']=torch.tensor(photometric_parameters(strength,self.cfg,generator))
            else:
                images = torch.stack([result[key] for key in ('previous', 'anchor', 'target')])
                images = difficult_images(images, strength, self.cfg, generator)
                for key, image in zip(('previous', 'anchor', 'target'), images):
                    result[key] = image
            start, end = int(entry['cam_ticks'][anchor]), int(entry['cam_ticks'][target])
            result['imu'] = self.flight_imu(item[0])[start+1:end+1].mean(0)
        else:
            start = item[1]
            if 'frames_cached' in result:
                raise ValueError('Curriculum augmentation requires uncached camera/IMU inputs')
            if getattr(self.cfg,'CURRICULUM_STATE_DEVICE_PHOTOMETRIC',False):
                result['photometric']=torch.tensor(photometric_parameters(strength,self.cfg,generator))
            else:
                result['cam_images'] = difficult_images(result['cam_images'], strength, self.cfg, generator)
            result['imu'] = self.flight_imu(item[0])[start:start+self.length]
            result['gravity'] = self.base.initial_gravity(self.flight_imu(item[0]), start, self.length)
        # Keys identify recordings, rather than transient indices in rolling batches.
        result['curriculum_key'] = sample_key(self.base,item)
        return result


def motion_scores(dataset):
    """Physical motion complexity: speed + angular rate + acceleration magnitude."""
    sums = []
    for entry in dataset.store.entries:
        state = entry['state']
        velocity = state[:, 12:15]
        acceleration = torch.cat([torch.zeros_like(velocity[:1]), velocity[1:]-velocity[:-1]])/entry['dt']
        score = velocity.norm(dim=-1) + state[:, 15:18].norm(dim=-1) + .1*acceleration.norm(dim=-1)
        sums.append(torch.cat([score.new_zeros(1), score.cumsum(0)]))
    result = []
    length = getattr(dataset, 'length', None)
    for item in dataset.items:
        if length is None:
            entry = dataset.store.entries[item[0]]
            start, end = int(entry['cam_ticks'][item[1]]), int(entry['cam_ticks'][item[3]])+1
        else:
            start, end = item[1], item[1]+length
        result.append((sums[item[0]][end]-sums[item[0]][start])/(end-start))
    return torch.stack(result).double()


class LearningProgress:
    """Bounded EMA priorities; high error alone does not receive a high priority."""
    def __init__(self, limit=100_000):
        self.history = {}
        self.limit = limit

    def update(self, keys, errors):
        for key, value in zip(keys, errors.detach().cpu().tolist()):
            if not math.isfinite(value):
                continue
            if key in self.history:
                previous, progress = self.history[key]
                gain = max(0., (previous-value)/max(abs(previous), 1e-6))
                self.history[key] = (.8*previous+.2*value, .8*progress+.2*min(gain, 1.))
            else:
                self.history[key] = (value, 0.)
        while len(self.history) > self.limit:
            self.history.pop(next(iter(self.history)))

    def priority(self, key):
        return 1 + 4*self.history.get(key, (0., 0.))[1]


def curriculum_sampler(dataset, schedule, progress, seed):
    scores = motion_scores(dataset)
    cuts = torch.quantile(scores, torch.linspace(0, 1, len(schedule.levels)+1, dtype=torch.float64)[1:-1])
    bins = torch.bucketize(scores, cuts, right=False)
    stage = schedule.stages['motion']
    weights = torch.zeros(len(dataset), dtype=torch.float64)
    available = sorted(set(bins.tolist()))
    if getattr(schedule,'all_easier',False):
        current=max((n for n in available if n<=stage),default=min(available))
        # Relative to `current`, which falls back below `stage` when that bin is
        # empty; relative to `stage` it would also be counted as easier.
        easier=[n for n in available if n<current]
        allocations=[(current,schedule.mix[0] if easier else 1.)]
        if easier:allocations.extend((n,schedule.mix[1]/len(easier)) for n in easier)
        for selected,probability in allocations:
            mask=bins==selected
            weights[mask]+=probability/int(mask.sum())
    else:
        for probability, desired in zip(schedule.mix, (stage, max(0, stage-1), min(len(schedule.levels)-1, stage+1))):
            selected = min(available, key=lambda n: abs(n-desired))
            mask = bins == selected
            weights[mask] += probability/int(mask.sum())
    if progress is not None:
        for index, item in enumerate(dataset.items):
            weights[index] *= progress.priority(sample_key(dataset,item))
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)


def validate_config(cfg):
    if not getattr(cfg, 'USE_CURRICULUM', False):
        return
    DifficultyCurriculum(cfg)
    for name in ('CURRICULUM_MAX_STAGE_EPOCHS','CURRICULUM_MIN_FRAME_EPOCHS',
                 'CURRICULUM_MIN_STATE_EPOCHS','CURRICULUM_BATCH_TOKENS','CURRICULUM_FINAL_STAGE_CYCLES'):
        value=getattr(cfg,name,0)
        if type(value) is not int or value<0:raise ValueError(f'{name} must be a nonnegative integer')
    if type(getattr(cfg,'CURRICULUM_REQUIRE_COMPLETION',True)) is not bool:
        raise ValueError('CURRICULUM_REQUIRE_COMPLETION must be a boolean')
    if type(cfg.CURRICULUM_EARLY_STOP_GRACE) is not int or cfg.CURRICULUM_EARLY_STOP_GRACE < 0:
        raise ValueError('CURRICULUM_EARLY_STOP_GRACE must be a nonnegative integer')
    for name in ('CURRICULUM_ACCEL_NOISE', 'CURRICULUM_GYRO_NOISE', 'CURRICULUM_ACCEL_BIAS',
                 'CURRICULUM_GYRO_BIAS', 'CURRICULUM_ACCEL_WALK', 'CURRICULUM_GYRO_WALK',
                 'CURRICULUM_BRIGHTNESS', 'CURRICULUM_CONTRAST'):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not 0 <= cfg.CURRICULUM_BLUR <= 1:
        raise ValueError('CURRICULUM_BLUR must be between 0 and 1')
    if type(cfg.CURRICULUM_PROBE_BATCHES) is not int or cfg.CURRICULUM_PROBE_BATCHES < 1:
        raise ValueError('CURRICULUM_PROBE_BATCHES must be a positive integer')
    for name in ('SL_CURRICULUM_WINDOWS', 'PPO_CURRICULUM_WINDOWS'):
        windows = getattr(cfg, name)
        if (not isinstance(windows, list) or any(type(n) is not int or n < 3 for n in windows)
                or any(b <= a for a, b in zip(windows, windows[1:]))):
            raise ValueError(f'{name} must be a list of increasing integers >= 3')
