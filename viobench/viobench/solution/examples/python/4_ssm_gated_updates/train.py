"""Launch four supervised ablations; measured scheduling precedes real updates."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import legacy


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--public60', action=argparse.BooleanOptionalAction, default=True)
    args, remaining = parser.parse_known_args(argv)
    # Resolve the run before handing off so post-training public validation can
    # select exactly the checkpoints produced by this invocation.
    if '--out' not in remaining and not any(x.startswith('--out=') for x in remaining):
        remaining += ['--out', str(legacy.HERE/'checkpoints'/datetime.now().strftime('%Y%m%d-%H%M%S-%f'))]
    from gated_engine import main as train
    train(remaining)
    if args.public60 and '--help' not in remaining and '-h' not in remaining:
        options = argparse.ArgumentParser(add_help=False)
        options.add_argument('--out', type=Path)
        options.add_argument('--device', default=None)
        options.add_argument('--wandb-mode', default='online')
        parsed, _ = options.parse_known_args(remaining)
        from evaluate import main as evaluate
        command = ['--run', str(parsed.out), '--suite', 'public60', '--wandb-mode', parsed.wandb_mode]
        if parsed.device:
            command += ['--device', parsed.device]
        evaluate(command)


if __name__ == '__main__':
    main(sys.argv[1:])
