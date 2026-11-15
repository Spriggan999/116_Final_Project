"""Train budget students directly from ground truth, without a teacher."""
import sys
from pathlib import Path

from distill import main


if __name__=='__main__':
    # Defaults precede user options so the normal parser remains the one source
    # of validation/help. This entry point always disables teacher supervision.
    main(['--curriculum-order','staged','--curriculum-probe-batches','12',
          '--collection-config',str(Path(__file__).with_name('supervised_procthor.json')),
          *sys.argv[1:],'--ground-truth-only'])
