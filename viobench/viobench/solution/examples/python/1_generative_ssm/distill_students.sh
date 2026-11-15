#!/usr/bin/env bash
# Run inside tmux after activating the CUDA Conda environment.
set -euo pipefail
HD_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HD_SCRIPT_DIR"
HD_TEACHER="${HD_DISTILL_TEACHER:-checkpoints/20261113-192121-curriculum-tujmpo0u}"
HD_OUT="checkpoints/distill-$(TZ=America/Los_Angeles date +%Y%m%d-%H%M%S)-$$"
HD_EXTRA=()
while (($#)); do
    case "$1" in
        --teacher|--out)
            if (($#<2)); then printf 'Missing value for %s\n' "$1" >&2; exit 2; fi
            if [[ "$1" == --teacher ]]; then HD_TEACHER="$2"; else HD_OUT="$2"; fi
            shift 2 ;;
        --help|-h)
            printf 'Usage: bash distill_students.sh [--teacher DIR] [--out DIR] [distill.py options]\nActivate myenv first. Defaults: all seven students, automatic scheduling, W&B online.\n'
            exit 0 ;;
        *) HD_EXTRA+=("$1"); shift ;;
    esac
done
for HD_FILE in frame_student.pt state_student.pt student_config.json training_config.json; do
    if [[ ! -f "$HD_TEACHER/$HD_FILE" ]]; then
        printf 'Incomplete teacher: %s/%s. Pass --teacher DIR.\n' "$HD_TEACHER" "$HD_FILE" >&2
        exit 2
    fi
done
if [[ -e "$HD_OUT" || -e "$HD_OUT.log" ]]; then
    printf 'Choose a fresh --out; output/log already exists: %s\n' "$HD_OUT" >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$HD_OUT")"
HD_PYTHON="${HD_DISTILL_PYTHON:-python}"
HD_COMMAND=("$HD_PYTHON" -u "$HD_SCRIPT_DIR/distill.py" --teacher "$HD_TEACHER" --out "$HD_OUT"
            --device cuda --stream-data --parallel-mode auto --wandb-mode online "${HD_EXTRA[@]}")
printf 'Checkpoints: %s\nConsole log: %s.log\n' "$HD_OUT" "$HD_OUT"
set +e
"${HD_COMMAND[@]}" 2>&1 | tee "$HD_OUT.log"
HD_CODES=("${PIPESTATUS[@]}")
set -e
HD_EXIT="${HD_CODES[0]}"
if [[ "$HD_EXIT" == 0 && "${HD_CODES[1]}" != 0 ]]; then HD_EXIT="${HD_CODES[1]}"; fi
printf 'Distillation exit code: %s\nConsole log: %s.log\n' "$HD_EXIT" "$HD_OUT"
exit "$HD_EXIT"
