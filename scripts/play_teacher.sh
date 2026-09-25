#!/bin/bash
# Roll out a trained teacher in the IsaacGym viewer.
# Usage: scripts/play_teacher.sh checkpoints/g1_teacher.pth [num_envs] [extra args...]
set -e
cd "$(dirname "$0")/.."
CKPT="${1:-checkpoints/g1_teacher.pth}"; shift || true
NUM_ENVS="${1:-4}"; shift || true
python ultra/run.py --task UltraG1Retarget \
    --cfg_env ultra/data/cfg/g1_teacher.yaml \
    --cfg_train ultra/data/cfg/train/rlg/g1_teacher.yaml \
    --test \
    --num_envs "${NUM_ENVS}" \
    --checkpoint "${CKPT}" \
    "$@"
