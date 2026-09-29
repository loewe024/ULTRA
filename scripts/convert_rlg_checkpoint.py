"""Convert an rl-games 1.1.4 checkpoint (IsaacGym version of ULTRA) to the rl-games 1.6 layout.

rl-games 1.6 keeps the observation/value normalizers inside the model, so ``running_mean_std`` (and
``reward_mean_std``) move into ``checkpoint['model']``. ULTRA's loaders convert old checkpoints on the fly as well;
this script writes the converted file once, e.g. for other tools.

Usage (from the repository root):
    python scripts/convert_rlg_checkpoint.py old.pth new.pth
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ultra"))

from learning.ultra_models import upgrade_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    before = set(checkpoint.get("model", {}))
    upgrade_checkpoint(checkpoint)
    added = sorted(set(checkpoint["model"]) - before)
    torch.save(checkpoint, args.output)
    print(f"wrote {args.output}; added model entries: {added if added else 'none (already converted)'}")


if __name__ == "__main__":
    main()
