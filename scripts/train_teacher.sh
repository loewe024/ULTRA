#!/bin/bash
# Stage 2: train the privileged dense-tracking teacher on the retargeted G1 dataset (Isaac Lab).
# Data: set env.motion_file in ultra/data/cfg/g1_teacher.yaml (default InterAct/OMOMO_retarget_aug).
set -e
cd "$(dirname "$0")/.."
python ultra/run.py --task UltraG1Retarget \
    --cfg_env ultra/data/cfg/g1_teacher.yaml \
    --cfg_train ultra/data/cfg/train/rlg/g1_teacher.yaml \
    --headless \
    --output_path output/teacher \
    "$@"
