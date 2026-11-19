"""Launch four supervised ablations; measured scheduling precedes real updates."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import os
import legacy
from resource_limits import ResourceGuard, Limits
from safe_storage import scratch_environment


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--public60', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resource-dir',type=Path,default=Path('/var/tmp')/f'gated-updates-{os.getuid()}')
    for name,value in vars(Limits()).items():
        parser.add_argument('--'+name.replace('_','-'),type=int,default=value)
    args, remaining = parser.parse_known_args(argv)
    # Resolve the run before handing off so post-training public validation can
    # select exactly the checkpoints produced by this invocation.
    if '--out' not in remaining and not any(x.startswith('--out=') for x in remaining):
        remaining += ['--out', str(legacy.HERE/'checkpoints'/datetime.now().strftime('%Y%m%d-%H%M%S-%f'))]
    from gated_engine import main as train
    if '--help' in remaining or '-h' in remaining:
        parser.print_help()
        return train(remaining)
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument('--out',type=Path)
    options.add_argument('--device',default=None)
    options.add_argument('--wandb-mode',default='online')
    options.add_argument('--stream-dir',type=Path)
    parsed,_ = options.parse_known_args(remaining)
    if parsed.stream_dir is None:
        remaining += ['--stream-dir',str(args.resource_dir/'recordings')]
    limits=Limits(**{name:getattr(args,name) for name in vars(Limits())})
    with ResourceGuard(parsed.out,args.resource_dir,limits),scratch_environment(args.resource_dir):
        train(remaining)
        if args.public60:
            from evaluate import main as evaluate
            command = ['--run',str(parsed.out),'--suite','public60','--wandb-mode',parsed.wandb_mode]
            if parsed.device:
                command += ['--device',parsed.device]
            evaluate(command)


if __name__ == '__main__':
    main(sys.argv[1:])
