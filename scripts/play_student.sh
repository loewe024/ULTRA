#!/bin/bash
# Roll out the VAE student in the IsaacGym viewer under one observation preset.
# Usage: scripts/play_student.sh <ckpt.pth> [num_envs] [task_mode] [obj_obs] [extra run_distill.py args...]
#   task_mode: full_track | sparse_track | object_obs   (omit or "" for training-style random masks)
#   obj_obs  : points | pos | none                       (only used with object_obs)
set -e
cd "$(dirname "$0")/.."
CKPT="${1:-checkpoints/g1_student_vae.pth}"; shift || true
NUM_ENVS="${1:-1}"; shift || true
TASK_MODE="${1:-}"; shift || true
OBJ_OBS="${1:-points}"; shift || true
EXTRA_ARGS=()
if [ -n "${TASK_MODE}" ]; then
  EXTRA_ARGS+=(--task_mode "${TASK_MODE}" --obj_obs "${OBJ_OBS}")
fi
python ultra/run_distill.py --task UltraDistillObjV2Point \
    --cfg_env ultra/data/cfg/g1_student_vae.yaml \
    --cfg_train ultra/data/cfg/train/rlg/g1_student_vae.yaml \
    --test \
    --num_envs "${NUM_ENVS}" \
    --checkpoint "${CKPT}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
