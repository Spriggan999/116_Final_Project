"""Disk, quota and host-memory headroom for this training process and children."""
from dataclasses import dataclass, asdict
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time

MIB = 2**20
ACTIVE = None


class ResourceLimitError(RuntimeError):
    pass


def filesystem(path):
    path = Path(path).resolve()
    matches = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        cut = fields.index('-')
        mount = Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1],8)), fields[4]))
        if path == mount or mount in path.parents:
            matches.append((len(mount.parts), fields[cut+1], fields[cut+2]))
    _, kind, source = max(matches, default=(0,'unknown','unknown'))
    return kind, source


def existing_parent(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def directory_bytes(path):
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(root)/d).is_symlink()]
        for name in files:
            try:
                item = Path(root)/name
                if not item.is_symlink():
                    total += item.stat().st_size
            except FileNotFoundError:
                pass  # The recording ring retires files concurrently.
    return total


def quota_limits(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields)<4:
            continue
        try:
            used, soft, hard = (int(value.rstrip('*'))*1024 for value in fields[1:4])
        except ValueError:
            continue
        limits = [value for value in (soft,hard) if value]
        if limits:
            result[fields[0]] = max(0,min(limits)-used)
    return result


def available_memory():
    values = dict((key.rstrip(':'),int(value)*1024) for key,value,*_ in
                  (line.split() for line in Path('/proc/meminfo').read_text().splitlines()))
    return values.get('MemAvailable', values['MemFree'])


def process_tree_rss():
    records = {}
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            fields = dict(line.split(':',1) for line in path.read_text().splitlines() if ':' in line)
            records[int(path.parent.name)] = (int(fields['PPid']),int(fields.get('VmRSS','0 kB').split()[0])*1024)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    selected = {os.getpid()}
    while True:
        children = {pid for pid,(parent,_) in records.items() if parent in selected}
        if children <= selected:
            break
        selected |= children
    return sum(records.get(pid,(0,0))[1] for pid in selected)


@dataclass
class Limits:
    min_disk_free_mib: int = 10240
    min_quota_free_mib: int = 1024
    min_ram_free_mib: int = 8192
    max_host_memory_mib: int = 24576
    max_scratch_mib: int = 16384
    max_run_mib: int = 1024

    def validate(self):
        if any(type(value) is not int or value<=0 for value in asdict(self).values()):
            raise ValueError('Resource limits must be positive integer MiB values')


def disk_check(path, limits, planned=0, quotas=None):
    parent = existing_parent(path)
    free = shutil.disk_usage(parent).free
    if free-planned < limits.min_disk_free_mib*MIB:
        raise ResourceLimitError(f'Low disk space at {parent}: keep {limits.min_disk_free_mib} MiB free; refusing further writes')
    _, source = filesystem(parent)
    matches = [(len(key),value) for key,value in (quotas or {}).items()
               if source==key or source.startswith(key+'/')]
    if matches and max(matches)[1]-planned < limits.min_quota_free_mib*MIB:
        raise ResourceLimitError(f'Home/filesystem quota headroom exhausted at {parent}; keep {limits.min_quota_free_mib} MiB free')


class ResourceGuard:
    def __init__(self, run, root, limits=None, interval=1.):
        self.run = Path(run).resolve()
        self.root = Path(root).resolve()
        self.limits = limits or Limits()
        self.limits.validate()
        self.interval = interval
        self.done = threading.Event()
        self.cancellers = []
        self.failure = None
        self.quotas = {}
        self.quota_checked = self.sizes_checked = float('-inf')
        self.lock = threading.RLock()

    def check(self, planned=0, path=None):
        with self.lock:
            now = time.monotonic()
            if now-self.quota_checked > 30:
                try:
                    output = subprocess.run(['quota','-w'], capture_output=True,text=True,timeout=3)
                    self.quotas = quota_limits(output.stdout)
                except (OSError,subprocess.TimeoutExpired):
                    pass
                self.quota_checked = now
            paths = {existing_parent(self.root),existing_parent(self.run)}
            for parent in paths:
                disk_check(parent,self.limits,planned if path is None or existing_parent(path)==parent else 0,self.quotas)
            if path is not None:
                disk_check(path,self.limits,planned,self.quotas)
            if available_memory() < self.limits.min_ram_free_mib*MIB:
                raise ResourceLimitError('Low available RAM: stopping before the machine starts swapping heavily')
            if process_tree_rss() > self.limits.max_host_memory_mib*MIB:
                raise ResourceLimitError(f'Training/collector RSS exceeded {self.limits.max_host_memory_mib} MiB')
            if now-self.sizes_checked > 5 or planned:
                for directory, maximum in [(self.root,self.limits.max_scratch_mib),(self.run,self.limits.max_run_mib)]:
                    target=Path(path).resolve() if path is not None else None
                    growth=planned if target is None or target==directory or directory in target.parents else 0
                    if directory_bytes(directory)+growth > maximum*MIB:
                        raise ResourceLimitError(f'Storage budget exceeded for {directory}: {maximum} MiB; existing checkpoints are retained')
                self.sizes_checked = now

    def _interrupt(self, signum, frame):
        if self.failure:
            raise ResourceLimitError(str(self.failure))
        if callable(self.old_signal):
            return self.old_signal(signum,frame)
        signal.default_int_handler(signum,frame)

    def _watch(self):
        while not self.done.wait(self.interval):
            try:
                self.check()
            except (ResourceLimitError,OSError) as error:
                self.failure = error
                print(f'\nRESOURCE LIMIT: {error}. Stopping collection and training; retaining completed checkpoints.',flush=True)
                for cancel in self.cancellers:
                    cancel()
                os.kill(os.getpid(),signal.SIGINT)
                return

    def __enter__(self):
        global ACTIVE
        if filesystem(existing_parent(self.root))[0] in ('tmpfs','ramfs'):
            raise ResourceLimitError(f'{self.root} is RAM-backed; choose disk-backed --resource-dir (for example /var/tmp)')
        self.check()
        self.root.mkdir(parents=True,exist_ok=True)
        self.old_signal = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT,self._interrupt)
        ACTIVE = self
        self.thread = threading.Thread(target=self._watch,name='resource-guard',daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*args):
        global ACTIVE
        self.done.set()
        self.thread.join(timeout=4)
        signal.signal(signal.SIGINT,self.old_signal)
        ACTIVE = None


def check_write(path, planned=0):
    if ACTIVE is not None:
        ACTIVE.check(planned=planned,path=path)
