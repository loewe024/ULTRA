#!/bin/bash
# Roll out a trained teacher on all motions of a rollout config and render the videos.
# Usage: scripts/eval_teacher.sh [config yaml]   (default ultra/data/cfg/teacher_rollout.yaml, settings described there)
#   1. ultra/run_teacher_inference.py: all clips without rollout in one Isaac Sim process (one env per clip),
#      output <output_dir>/<clip>.pt, env.yaml, logs/
#   2. scripts/render_rollout.py: missing videos in <output_dir>/videos (only with videos: true)
# Settings that differ from an existing <output_dir>/env.yaml stop the run before anything is rolled out or rendered.
cd "$(dirname "$0")/.."
CFG="${1:-ultra/data/cfg/teacher_rollout.yaml}"
export OMNI_KIT_ACCEPT_EULA=YES
export MUJOCO_GL="${MUJOCO_GL:-egl}"

python ultra/run_teacher_inference.py --cfg "${CFG}"
status=$?
if [ ${status} -eq 2 ]; then
    exit 2  # settings differ from env.yaml (or an invalid config)
fi
python scripts/render_rollout.py --cfg "${CFG}" || status=1
exit ${status}
