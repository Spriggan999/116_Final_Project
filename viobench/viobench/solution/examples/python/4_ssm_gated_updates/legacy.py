"""Explicit access to the existing, tested data/curriculum/training utilities."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / '1_generative_ssm'
ROOT = HERE.parents[4]
for path in (SOURCE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
