#!/bin/bash
# Sim2sim (MuJoCo) rollout of the teacher for every motion in a directory, one video per motion.
# Usage: scripts/sim2sim_teacher.sh <ckpt.pth> <motion_dir> [extra sim2sim_teacher.py args...]
set -e
cd "$(dirname "$0")/.."
CKPT="${1:?checkpoint path required}"; shift
MOTION_DIR="${1:?motion directory required}"; shift
shopt -s nullglob
motions=("${MOTION_DIR}"/*.pt)
[ "${#motions[@]}" -gt 0 ] || { echo "no .pt motions found in ${MOTION_DIR}" >&2; exit 1; }
OUT_DIR="sim2sim_videos/$(basename "${CKPT%.*}")_teacher"
mkdir -p "${OUT_DIR}"
for motion in "${motions[@]}"; do
    name=$(basename "${motion}" .pt)
    echo "[sim2sim teacher] ${name}"
    python ultra/sim2sim_teacher.py --ckpt "${CKPT}" --motion_path "${motion}" --record_video "$@"
    if [ -f out.mp4 ]; then mv out.mp4 "${OUT_DIR}/${name}.mp4"; fi
done
echo "videos written to ${OUT_DIR}"
