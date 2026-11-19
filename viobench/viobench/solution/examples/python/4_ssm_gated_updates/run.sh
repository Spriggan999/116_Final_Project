#!/usr/bin/env bash
set -euo pipefail
GATED_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GATED_PYTHON="${GATED_PYTHON:-/home/cc/eecs116/fa26/class/eecs116-aaq/.conda/envs/myenv/bin/python}"
GATED_PYTHON="$(command -v -- "$GATED_PYTHON")"
export PATH="$(dirname -- "$GATED_PYTHON"):$PATH"
exec "$GATED_PYTHON" -u "$GATED_DIR/train.py" \
  --device cuda --stream-data --parallel-mode auto --wandb-mode online "$@"
