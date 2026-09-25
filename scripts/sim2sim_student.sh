#!/bin/bash
# Sim2sim (MuJoCo) rollout of the VAE student for every motion in a directory, one video per motion.
# Usage: scripts/sim2sim_student.sh <ckpt.pth|jit.pt> <motion_dir> [task_mode] [obj_obs] [extra sim2sim_vae.py args...]
#   task_mode: full_track | sparse_track | object_obs ; obj_obs: points | pos | none
#   NOTE: the MuJoCo full_track preset hides the object modalities and the command goal (pure humanoid
#   reference tracking); the IsaacGym full_track preset keeps everything visible.
# Add --use_jit when passing a JIT model exported with ultra/export_jit.py (auto-detected for .pt files).
set -e
cd "$(dirname "$0")/.."
CKPT="${1:?checkpoint path required}"; shift
MOTION_DIR="${1:?motion directory required}"; shift
TASK_MODE="${1:-full_track}"; shift || true
OBJ_OBS="${1:-points}"; shift || true
shopt -s nullglob
motions=("${MOTION_DIR}"/*.pt)
[ "${#motions[@]}" -gt 0 ] || { echo "no .pt motions found in ${MOTION_DIR}" >&2; exit 1; }
OUT_DIR="sim2sim_videos/$(basename "${CKPT%.*}")_${TASK_MODE}"
mkdir -p "${OUT_DIR}"
for motion in "${motions[@]}"; do
    name=$(basename "${motion}" .pt)
    echo "[sim2sim] ${TASK_MODE} ${name}"
    python ultra/sim2sim_vae.py \
        --ckpt "${CKPT}" \
        --motion_path "${motion}" \
        --record_video \
        --goal_phase_dim 4 \
        --task_mode "${TASK_MODE}" \
        --obj_obs "${OBJ_OBS}" \
        "$@"
    if [ -f out.mp4 ]; then mv out.mp4 "${OUT_DIR}/${name}.mp4"; fi
done
echo "videos written to ${OUT_DIR}"
