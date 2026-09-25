#!/bin/bash
# Stage 3 on several GPUs of one node. Usage: NUM_GPUS=4 scripts/train_student_multigpu.sh
# Keep minibatch_size x NUM_GPUS consistent with the single-GPU setting when comparing runs.
set -e
cd "$(dirname "$0")/.."
NUM_GPUS="${NUM_GPUS:-2}"
torchrun \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    ultra/run_distill.py --task UltraDistillObjV2Point \
    --cfg_env ultra/data/cfg/g1_student_vae.yaml \
    --cfg_train ultra/data/cfg/train/rlg/g1_student_vae.yaml \
    --headless \
    --output_path output/student \
    --multi_gpu \
    "$@"
