#!/bin/bash
# Interactive MuJoCo sim2sim with keyboard goal commands for the VAE student.
# Usage: scripts/sim2sim_keyboard.sh <ckpt.pth|jit.pt> <motion.pt> [--task_mode sparse_track|object_obs|full_track] [extra args...]
set -e
cd "$(dirname "$0")/.."
CKPT="${1:?checkpoint path required}"; shift
MOTION="${1:?reference motion .pt required}"; shift
python ultra/sim2sim_vae_keyboard.py --ckpt "${CKPT}" --motion_path "${MOTION}" --goal_phase_dim 4 "$@"
