"""Compare the final object pose of source, retargeted and teacher-rollout motions.

For every clip that exists in all three folders, the object pose of the last frame is compared for
source -> retarget, source -> rollout and retarget -> rollout. Reported per pair: the share of clips whose
final object pose lies within the position/orientation thresholds, and the mean/median final errors.

Usage: python scripts/compare_motions.py [--source-dir ...] [--retarget-dir ...] [--rollout-dir ...]
       [--pos-thresh 0.1] [--rot-thresh 15] [--csv per_clip.csv]
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch

OBJ_POS_SRC, OBJ_ROT_SRC = slice(318, 321), slice(321, 325)  # [T, 591] source layout
OBJ_POS_G1, OBJ_ROT_G1 = slice(71, 74), slice(74, 78)        # [T, 630] G1 layout (also first 630 of observed_hoi)
PAIRS = [("source", "retarget"), ("source", "rollout"), ("retarget", "rollout")]


def load(path):
    return torch.load(path, map_location="cpu")


def to_numpy(x):
    return x.detach().cpu().numpy().astype(np.float64)


def xyz_multiplier(name):
    """Per-axis multiplier encoded in the export name, e.g. _xyz1p000000-1p000000-1p000000_."""
    m = re.search(r"_xyz([0-9p.\-]+?)_", name)
    if m is None:
        return np.ones(3)
    return np.array([float(v.replace("p", ".")) for v in m.group(1).split("-")])


def final_pose(pos, rot):
    q = rot[-1] / np.linalg.norm(rot[-1])
    return pos[-1], q


def pose_error(a, b):
    pos_err = np.linalg.norm(a[0] - b[0])
    dot = np.clip(abs(np.dot(a[1], b[1])), 0.0, 1.0)
    rot_err = np.degrees(2.0 * np.arccos(dot))
    return pos_err, rot_err


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=Path("InterAct/all_motions"))
    parser.add_argument("--retarget-dir", type=Path, default=Path("output/retarget_export_080"))
    parser.add_argument("--rollout-dir", type=Path, default=Path("output/teacher_videos"),
                        help="folder with one <clip>/rollout.pt per clip (scripts/eval_teacher.sh output)")
    parser.add_argument("--scale", type=float, default=0.8, help="env scaling applied to the source positions")
    parser.add_argument("--pos-thresh", type=float, default=0.5, help="goal position tolerance [m]")
    parser.add_argument("--rot-thresh", type=float, default=45.0, help="goal orientation tolerance [deg]")
    parser.add_argument("--csv", type=Path, help="optional per-clip output")
    args = parser.parse_args()

    clips = []
    for rollout_file in sorted(args.rollout_dir.glob("*/rollout.pt")):
        name = rollout_file.parent.name
        retarget_file = args.retarget_dir / f"{name}.pt"
        source_file = args.source_dir / f"{name.split('_xyz')[0]}.pt"
        if retarget_file.is_file() and source_file.is_file():
            clips.append((name, source_file, retarget_file, rollout_file))
    if not clips:
        raise SystemExit("no clip found in all three folders")

    rows, short = [], []
    for name, source_file, retarget_file, rollout_file in clips:
        source = to_numpy(load(source_file))[1:]  # frame 0 is all zeros
        retarget = to_numpy(load(retarget_file))
        rollout = to_numpy(load(rollout_file)["observed_hoi"])[:, :630]
        # observed_hoi[i] belongs to retarget frame i + 1, so a complete rollout has len(retarget) - 1 frames
        if len(rollout) != len(retarget) - 1:
            short.append(f"{name} ({len(rollout) + 1}/{len(retarget)} frames)")

        # the source is upsampled 30 -> 60 Hz for the retarget, but its last frame stays the last frame
        poses = {
            "source": final_pose(source[:, OBJ_POS_SRC] * args.scale * xyz_multiplier(name), source[:, OBJ_ROT_SRC]),
            "retarget": final_pose(retarget[:, OBJ_POS_G1], retarget[:, OBJ_ROT_G1]),
            "rollout": final_pose(rollout[:, OBJ_POS_G1], rollout[:, OBJ_ROT_G1]),
        }
        row = {"clip": name}
        for a, b in PAIRS:
            pos_err, rot_err = pose_error(poses[a], poses[b])
            row[f"{a}-{b}_pos"] = pos_err
            row[f"{a}-{b}_rot"] = rot_err
        rows.append(row)

    print(f"{len(rows)} clips with source, retarget and rollout "
          f"(goal: pos < {args.pos_thresh:.3f} m and rot < {args.rot_thresh:.1f} deg)")
    if short:
        print(f"warning: {len(short)} rollouts end early, their last frame is used: {', '.join(short)}")
    header = f"{'pair':<20} {'reached':>8} {'pos ok':>8} {'rot ok':>8} " \
             f"{'pos mean':>9} {'pos med':>9} {'rot mean':>9} {'rot med':>9}"
    print(header)
    print("-" * len(header))
    for a, b in PAIRS:
        pos = np.array([r[f"{a}-{b}_pos"] for r in rows])
        rot = np.array([r[f"{a}-{b}_rot"] for r in rows])
        pos_ok, rot_ok = pos < args.pos_thresh, rot < args.rot_thresh
        print(f"{a + ' -> ' + b:<20} {100 * (pos_ok & rot_ok).mean():7.1f}% {100 * pos_ok.mean():7.1f}% "
              f"{100 * rot_ok.mean():7.1f}% {pos.mean():8.3f}m {np.median(pos):8.3f}m "
              f"{rot.mean():8.1f}° {np.median(rot):8.1f}°")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"per-clip errors in {args.csv}")


if __name__ == "__main__":
    main()
