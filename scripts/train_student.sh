#!/bin/bash
# Stage 3: distill the teacher into the multimodal VAE student (single GPU).
# Requires env.teacherPolicy in ultra/data/cfg/g1_student_vae.yaml to point at a trained teacher.
set -e
cd "$(dirname "$0")/.."
python ultra/run_distill.py --task UltraDistillObjV2Point \
    --cfg_env ultra/data/cfg/g1_student_vae.yaml \
    --cfg_train ultra/data/cfg/train/rlg/g1_student_vae.yaml \
    --headless \
    --output_path output/student \
    "$@"
