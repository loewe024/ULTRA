#!/bin/bash
# Replay the reference dataset (no policy) in the student environment to inspect motions and object placement.
set -e
cd "$(dirname "$0")/.."
python ultra/run_distill.py --task UltraDistillObjV2Point \
    --cfg_env ultra/data/cfg/g1_student_vae.yaml \
    --cfg_train ultra/data/cfg/train/rlg/g1_student_vae.yaml \
    --num_envs 1 \
    --test \
    --play_dataset \
    "$@"
