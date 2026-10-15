#!/usr/bin/env bash
# Stage 1: train the SMPLX-to-G1 neural retargeting policy with position targets.
# Several GPUs of one node: NUM_GPUS=4 scripts/train_retarget_smplx.sh (one process per GPU,
# --num_envs applies per GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
RAW_MOTION_DIR="${RAW_MOTION_DIR:-InterAct/OMOMO_retarget}"
ASSET_SCALE="${ASSET_SCALE:-080_080_080}"
PREPARED_MOTION_DIR="${PREPARED_MOTION_DIR:-InterAct/OMOMO_retarget_supported_${ASSET_SCALE}}"
python scripts/prepare_retarget_smplx.py \
  --input-dir "$RAW_MOTION_DIR" \
  --output-dir "$PREPARED_MOTION_DIR" \
  --asset-scale "$ASSET_SCALE"
NUM_GPUS="${NUM_GPUS:-1}"
if [ "$NUM_GPUS" -gt 1 ]; then
  LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" ultra/run.py --multi_gpu)
else
  LAUNCH=(python ultra/run.py)
fi
"${LAUNCH[@]}" --task UltraG1 \
  --cfg_env ultra/data/cfg/g1_retarget_smplx.yaml \
  --cfg_train ultra/data/cfg/train/rlg/g1_retarget_smplx.yaml \
  --headless \
  --motion_file "$PREPARED_MOTION_DIR" \
  --output_path output/retarget_smplx \
  "$@"
