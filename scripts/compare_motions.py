"""Compare the final object pose of source, retargeted and teacher-rollout motions.

For every clip that exists in all three folders, the object pose of the last frame is compared for
source -> retarget, source -> rollout and retarget -> rollout. Reported per pair: the share of clips whose
final object pose lies within the position/orientation thresholds, and the mean/median final errors.

The folders are set below (SOURCE_DIR, RETARGET_DIR, ROLLOUT_DIR); paths are relative to the repository root.

A plot of the distribution of the final position and orientation errors (share of clips with an error up to x,
per pair) is saved to --plot (default output/graphics/final_errors.png).

Usage: python scripts/compare_motions.py [--pos-thresh 0.5] [--rot-thresh 45] [--csv per_clip.csv] [--plot png]
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
# Folders to compare
SOURCE_DIR = REPO / "InterAct/all_motions"                                    # original [T, 591] motions
RETARGET_DIR = REPO / "output/retarget_export_080_largebox"                   # retargeted [T, 630] G1 motions
ROLLOUT_DIR = REPO / "output/teacher_rollout_motions/quality_scaling_14000"        # <clip>.pt teacher rollouts

OBJ_POS_SRC, OBJ_ROT_SRC = slice(318, 321), slice(321, 325)  # [T, 591] source layout
OBJ_POS_G1, OBJ_ROT_G1 = slice(71, 74), slice(74, 78)        # [T, 630] G1 layout (also first 630 of observed_hoi)
PAIRS = [("source", "retarget"), ("source", "rollout"), ("retarget", "rollout")]
PAIR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


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


def plot_errors(rows, pos_thresh, rot_thresh, path):
    """Empirical CDF of the final position / orientation error per pair: share of clips with an error <= x."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = [("pos", "final object position error [m]", pos_thresh, "{:.2f} m"),
              ("rot", "final object orientation error [deg]", rot_thresh, "{:.0f}°")]
    for ax, (key, xlabel, thresh, fmt) in zip(axes, panels):
        xmax = 0.0
        for (a, b), color in zip(PAIRS, PAIR_COLORS):
            err = np.sort([r[f"{a}-{b}_{key}"] for r in rows])
            share = 100 * np.arange(1, len(err) + 1) / len(err)
            ax.step(np.concatenate([[0], err]), np.concatenate([[0], share]), where="post", lw=2, color=color,
                    label=f"{a} -> {b} ({100 * (err < thresh).mean():.0f}% < {fmt.format(thresh)})")
            xmax = max(xmax, np.percentile(err, 98))
        ax.axvline(thresh, color=MUTED, lw=1, ls="--")
        ax.text(thresh, 2, f" threshold {fmt.format(thresh)}", color=MUTED, fontsize=9, va="bottom")
        ax.set_xlim(0, max(xmax, 1.5 * thresh))  # the long tail beyond the 98th percentile is cut off
        ax.set_ylim(0, 100)
        ax.set_xlabel(xlabel, color=INK)
        ax.set_ylabel("clips with error <= x [%]", color=INK)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(f"Final object pose error, {len(rows)} clips ({ROLLOUT_DIR.name})", color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"error distribution plot in {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=float, default=0.8, help="env scaling applied to the source positions")
    parser.add_argument("--pos-thresh", type=float, default=0.2, help="goal position tolerance [m]")
    parser.add_argument("--rot-thresh", type=float, default=20.0, help="goal orientation tolerance [deg]")
    parser.add_argument("--csv", type=Path, help="optional per-clip output")
    parser.add_argument("--plot", type=Path, default=REPO / "output/graphics/final_errors.png",
                        help="plot of the error distributions")
    args = parser.parse_args()
    args.source_dir, args.retarget_dir, args.rollout_dir = SOURCE_DIR, RETARGET_DIR, ROLLOUT_DIR
    # describe the rollouts from the env.yaml run_teacher_inference.py wrote next to them
    env_file = args.rollout_dir / "env.yaml"
    if env_file.is_file():
        env = yaml.safe_load(env_file.read_text())
        progress = env["env"].get("objCurriculumProgress") if env["env"].get("motionSourceDir") else None
        print(f"rollouts {args.rollout_dir}: checkpoint {env.get('eval', {}).get('checkpoint', '?')}, object "
              f"trajectory {'source' if progress == 1 else 'retargeted' if not progress else f'blend {progress}'}")

    clips = []
    for rollout_file in sorted(args.rollout_dir.glob("*.pt")):
        name = rollout_file.stem
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

    plot_errors(rows, args.pos_thresh, args.rot_thresh, args.plot)


if __name__ == "__main__":
    main()
