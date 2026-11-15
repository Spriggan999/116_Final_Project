#!/usr/bin/env bash
# Fresh, independent supervised runs. Activate the CUDA environment first.
set -euo pipefail
HD_STUDENT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HD_STUDENT_SCRIPT_DIR"
HD_STUDENT_ROOT="checkpoints/supervised-$(TZ=America/Los_Angeles date +%Y%m%d-%H%M%S)-$$"
HD_STUDENT_EXTRA=()
while (($#)); do
    case "$1" in
        --out-root)
            if (($#<2)); then printf 'Missing --out-root value\n' >&2; exit 2; fi
            HD_STUDENT_ROOT="$2"; shift 2 ;;
        --warm-start|--warm-start=*|--warm-start-stage|--start-epoch|--start-epoch=*|--profiles|--profiles=*|--out|--out=*|--teacher|--teacher=*|--curriculum-order|--curriculum-order=*|--no-difficulty-curriculum)
            printf 'Sequential fresh training manages %s; use train_students.py for custom runs.\n' "$1" >&2
            exit 2 ;;
        --help|-h)
            printf 'Usage: bash train_students_sequential.sh [--out-root NEW_DIR] [train_students.py options]\nRuns 1k, 10k, 100k, 1m_ssm, 1m, 1m_hybrid, 100m, 100m_hybrid in ascending fuel/frame order, from random weights.\n'
            exit 0 ;;
        *) HD_STUDENT_EXTRA+=("$1"); shift ;;
    esac
done
if [[ -e "$HD_STUDENT_ROOT" ]]; then
    printf 'Choose a fresh --out-root; already exists: %s\n' "$HD_STUDENT_ROOT" >&2; exit 2
fi
mkdir -p -- "$HD_STUDENT_ROOT/runs" "$HD_STUDENT_ROOT/students"
HD_STUDENT_PYTHON="${HD_STUDENT_PYTHON:-python}"
# Within each fuel cap, order architectures by measured deployed fuel/frame.
for HD_STUDENT_PROFILE in 1k 10k 100k 1m_ssm 1m 1m_hybrid 100m 100m_hybrid; do
    HD_STUDENT_OUT="$HD_STUDENT_ROOT/runs/$HD_STUDENT_PROFILE"
    printf '\nTraining %s from scratch; output: %s\n' "$HD_STUDENT_PROFILE" "$HD_STUDENT_OUT"
    set +e
    "$HD_STUDENT_PYTHON" -u "$HD_STUDENT_SCRIPT_DIR/train_students.py" \
        --profiles "$HD_STUDENT_PROFILE" --out "$HD_STUDENT_OUT" \
        --device cuda --stream-data --parallel-mode auto --wandb-mode online \
        --collection-config "$HD_STUDENT_SCRIPT_DIR/supervised_procthor.json" \
        --epochs 1500 "${HD_STUDENT_EXTRA[@]}" 2>&1 | tee "$HD_STUDENT_OUT.log"
    HD_STUDENT_CODES=("${PIPESTATUS[@]}")
    set -e
    HD_STUDENT_EXIT="${HD_STUDENT_CODES[0]}"
    if [[ "$HD_STUDENT_EXIT" == 0 && "${HD_STUDENT_CODES[1]}" != 0 ]]; then HD_STUDENT_EXIT="${HD_STUDENT_CODES[1]}"; fi
    if [[ "$HD_STUDENT_EXIT" != 0 ]]; then
        printf 'Stopped at %s (exit %s); completed models remain in %s/students.\n' "$HD_STUDENT_PROFILE" "$HD_STUDENT_EXIT" "$HD_STUDENT_ROOT" >&2
        exit "$HD_STUDENT_EXIT"
    fi
    ln -s -- "../runs/$HD_STUDENT_PROFILE/$HD_STUDENT_PROFILE" "$HD_STUDENT_ROOT/students/$HD_STUDENT_PROFILE"
done
printf '\nAll eight supervised runs finished. Checkpoints and WASMs: %s/students/PROFILE\n' "$HD_STUDENT_ROOT"
