"""Run fresh curriculum training without a time limit; mirror on exit.

Primary checkpoints live in the project, while scene assets and recordings use
local disk. Interrupts stop the trainer gracefully before the final mirror.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo


PACIFIC = ZoneInfo('America/Los_Angeles')


def positive(value):
    result = float(value)
    if not 0 < result < float('inf'):
        raise argparse.ArgumentTypeError('must be finite and positive')
    return result


def run(args, trainer=None):
    """trainer is injectable for orchestration checks without starting the GPU."""
    project = Path(__file__).resolve().parent
    source=getattr(args,'from_checkpoints',None)
    skip_embedding=getattr(args,'skip_embedding_training',False)
    if skip_embedding and (source is None or not (Path(source).expanduser()/'frame_student.pt').is_file()):
        raise ValueError('--skip-embedding-training requires --from-checkpoints with frame_student.pt')
    config=Path(getattr(args,'config',project/'curriculum.json')).expanduser().resolve()
    if not config.is_file():raise FileNotFoundError(f'Training config not found: {config}')
    if shutil.which('rsync') is None:
        raise RuntimeError('rsync is required for the final checkpoint mirror')
    local = args.local_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(PACIFIC).strftime('%Y%m%d-%H%M%S')
    run_dir = Path(tempfile.mkdtemp(prefix=stamp+'-curriculum-', dir=checkpoint_root))
    backup = local/'checkpoint-backups'/run_dir.name
    backup.mkdir(parents=True)
    directories = {name:local/name for name in ('conta', 'recordings', 'cache', 'triton', 'cuda', 'wandb-cache')}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    deadline = time.time()+args.seconds if args.seconds is not None else None
    environment = dict(os.environ, CONTA_CACHE=str(directories['conta']),
                       CONTA_SEARCH_PATHS='', XDG_CACHE_HOME=str(directories['cache']),
                       TRITON_CACHE_DIR=str(directories['triton']), CUDA_CACHE_PATH=str(directories['cuda']),
                       WANDB_CACHE_DIR=str(directories['wandb-cache']))
    command = [sys.executable, '-u', str(trainer or project/'train.py'),
               '--stream-data', '--stream-dir', str(directories['recordings']),
               '--config', str(config), '--max-data-mib', str(args.max_data_mib),
               '--out', str(run_dir), '--solution-out', str(run_dir/'solution'), '--skip-export']
    if source is not None:
        command.extend(['--from-checkpoints',str(Path(source).expanduser().resolve()),'--sl'])
    if skip_embedding:
        command.append('--skip-embedding-training')
    stop_reason = None
    stopped_at = None
    received_signal = None
    child = None
    old_handlers = {}
    returncode = None
    failure = None
    backup_ok = False
    forced = False

    def stop(reason, signum=None):
        nonlocal stop_reason, stopped_at, received_signal
        if stop_reason is not None:
            return
        stop_reason, received_signal, stopped_at = reason, signum, time.monotonic()
        print(f'\nStopping training ({reason}); allowing {args.stop_grace:g}s for cleanup.', flush=True)
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    def handle(signum, frame):
        stop(signal.Signals(signum).name, signum)

    print(f'Primary checkpoints and log: {run_dir}', flush=True)
    print(f'Final local mirror: {backup}', flush=True)
    if deadline is None:
        print('No automatic time limit; training runs until completion or a manual stop.', flush=True)
    else:
        print(f'Stop at: {datetime.fromtimestamp(deadline, PACIFIC):%Y-%m-%d %H:%M:%S %Z}', flush=True)
    print('Model checkpoints are saved during validation; the final mirror copies completed files.', flush=True)
    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            old_handlers[signum] = signal.signal(signum, handle)
        log_path = run_dir/'train.log'
        with log_path.open('wb') as log, log_path.open('rb') as reader:
            child = subprocess.Popen(command, cwd=project, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if stop_reason is not None:
                # A signal may arrive between installing handlers and spawning.
                try:
                    os.killpg(child.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            while True:
                chunk = reader.read(64*1024)
                if chunk:
                    try:
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                    except BrokenPipeError:
                        pass  # retain the on-disk log if the viewing pipe closes
                returncode = child.poll()
                if returncode is not None:
                    # The child has closed its log; drain bytes written after read().
                    rest = reader.read()
                    if rest:
                        try:
                            sys.stdout.buffer.write(rest)
                            sys.stdout.buffer.flush()
                        except BrokenPipeError:
                            pass
                    break
                if deadline is not None and time.time() >= deadline:
                    stop('deadline')
                if stopped_at is not None and time.monotonic()-stopped_at >= args.stop_grace:
                    forced = True
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
                time.sleep(.1)
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        print(f'Launcher error: {failure}', file=sys.stderr, flush=True)
    finally:
        # Stop/wait before copying, including failures in the monitor itself.
        if child is not None and child.poll() is None:
            stop('launcher error')
            try:
                child.wait(timeout=args.stop_grace)
            except subprocess.TimeoutExpired:
                forced = True
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        if child is not None:
            returncode = child.returncode
        # Repeated terminal signals must not interrupt the final mirror.
        for signum in old_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            weights = sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob('*.pt')
                             if path.is_file() and not any(part.startswith('.post-sl-') for part in path.relative_to(run_dir).parts))
            status = dict(child_returncode=returncode,stop_reason=stop_reason,forced_stop=forced,
                          error=failure,checkpoint_files=weights,primary=str(run_dir),local_mirror=str(backup))
            try:
                (run_dir/'run_status.json').write_text(json.dumps(status,indent=2)+'\n')
            except OSError as error:
                print(f'Could not write run status: {error}', file=sys.stderr, flush=True)
            try:
                result = subprocess.run(['rsync','-aH','--exclude=*.tmp','--exclude=.post-sl-*',
                                         str(run_dir)+'/',str(backup)+'/'], check=False)
                backup_ok = result.returncode == 0
                backup_error = f'rsync {result.returncode}'
            except OSError as error:
                backup_error = str(error)
            if backup_ok:
                print(f'BACKUP COMPLETE: {backup}', flush=True)
            else:
                print(f'BACKUP FAILED ({backup_error}); primary checkpoints remain at {run_dir}',
                      file=sys.stderr, flush=True)
            if weights:
                print(f'Saved model files: {", ".join(weights)}', flush=True)
            else:
                print('No model checkpoint was saved before training stopped; logs/configs were retained.',
                      file=sys.stderr, flush=True)
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
    if not backup_ok or failure or not weights:
        return 1
    if stop_reason == 'deadline':
        return 0
    if received_signal is not None:
        return 128+received_signal
    return 0 if returncode == 0 else (returncode if returncode and returncode > 0 else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parent
    parser.add_argument('--local-root',type=Path,default=Path('/var/tmp')/f'hyperdrone-{os.getuid()}',
                        help='local scene cache, recordings and final mirror')
    parser.add_argument('--checkpoint-root',type=Path,default=project/'checkpoints',
                        help='persistent primary checkpoint parent')
    parser.add_argument('--config',type=Path,default=project/'curriculum.json',help='training JSON preset')
    parser.add_argument('--from-checkpoints',type=Path,help='continue SL then PPO in a new run, preserving source weights')
    parser.add_argument('--skip-embedding-training',action='store_true',
                        help='keep the saved embedding model fixed; train state SL then PPO')
    parser.add_argument('--max-data-mib',type=positive,default=6144)
    parser.add_argument('--seconds',type=positive,help='optional time limit in seconds; default: no time limit')
    parser.add_argument('--stop-grace',type=positive,default=120,help='cleanup seconds before forced termination')
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
