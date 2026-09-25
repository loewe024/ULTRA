#!/bin/bash
# Export a trained VAE student (.pth) to a TorchScript module (.pt) for deployment / sim2sim --use_jit.
# Usage: scripts/export_jit.sh checkpoints/g1_student_vae.pth [out.pt]
# The released config trains with normalize_input: False; the exporter then reads the observation size
# (numObsStudent) from ultra/data/cfg/g1_student_vae.yaml and uses identity normalization.
set -e
cd "$(dirname "$0")/.."
CKPT="${1:?checkpoint path required}"
OUT="${2:-${CKPT%.*}_jit.pt}"
python ultra/export_jit.py --ckpt "${CKPT}" --out "${OUT}"
