#!/bin/bash
# Stage 1 (reference implementation): RL neural retargeting from SMPL-X human-object mocap to the G1.
# Data: env.motion_file in ultra/data/cfg/g1_retarget_smplx.yaml (InterAct/OMOMO_retarget, SMPL-X format).
# NOTE: the rollout exporter that writes the retargeted G1 dataset is not part of this release yet (see README).
set -e
cd "$(dirname "$0")/.."
python ultra/run.py --task UltraG1 \
    --cfg_env ultra/data/cfg/g1_retarget_smplx.yaml \
    --cfg_train ultra/data/cfg/train/rlg/g1_retarget_smplx.yaml \
    --headless \
    --output_path output/retarget_smplx \
    "$@"
