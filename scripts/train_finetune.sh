#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
student_checkpoint="$1"
shift
python ultra/run_distill.py --task UltraDistillObjV3RL \
  --cfg_env ultra/data/cfg/g1_student_finetune.yaml \
  --cfg_train ultra/data/cfg/train/rlg/g1_student_finetune.yaml \
  --resume_from "$student_checkpoint" \
  --headless \
  --output_path output/finetune \
  "$@"
