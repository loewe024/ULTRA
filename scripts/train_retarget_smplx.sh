#!/usr/bin/env bash
# Stage 1: train the SMPLX-to-G1 neural retargeting policy with position targets.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW_MOTION_DIR="${RAW_MOTION_DIR:-InterAct/OMOMO_retarget}"
ASSET_SCALE="${ASSET_SCALE:-080_080_080}"
PREPARED_MOTION_DIR="${PREPARED_MOTION_DIR:-InterAct/OMOMO_retarget_supported_${ASSET_SCALE}}"
python scripts/prepare_retarget_smplx.py \
  --input-dir "$RAW_MOTION_DIR" \
  --output-dir "$PREPARED_MOTION_DIR" \
  --asset-scale "$ASSET_SCALE"
python ultra/run.py --task UltraG1 \
  --cfg_env ultra/data/cfg/g1_retarget_smplx.yaml \
  --cfg_train ultra/data/cfg/train/rlg/g1_retarget_smplx.yaml \
  --headless \
  --motion_file "$PREPARED_MOTION_DIR" \
  --output_path output/retarget_smplx \
  "$@"
