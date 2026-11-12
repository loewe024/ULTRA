"""Compare the final object pose of source, retargeted and teacher-rollout motions.

For every clip that exists in all three folders, the object pose is compared for source -> retarget,
source -> rollout and retarget -> rollout. Three tables with the same columns are printed, each with one error per
clip: the error of the final frame, the median error over the trajectory and the mean error over the trajectory.
Reported per pair: the share of clips whose error lies within the position/orientation thresholds, and the
mean/median of the per-clip errors.

The trajectory errors use the frames of the rollout (retarget frames 1 .. len(rollout)); the 30 Hz source is
aligned to them like UltraG1Retarget._load_source_obj (frame 0 dropped, upsampled x2 linearly).

The folders are set below (SOURCE_DIR, RETARGET_DIR, ROLLOUT_DIR); paths are relative to the repository root.

A plot of the distribution of the position and orientation errors (share of clips with an error up to x, per
pair; one row each for the final, median and mean error) is saved to --plot (default
output/graphics/final_errors.png).

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
ROLLOUT_DIR = REPO / "output/teacher_rollout_motions/quality_world_obj_16000"        # <clip>.pt teacher rollouts
#quality_all_from_0, quality_scaling_rop_19250, more_rop_with_curr_source, quality_small_objGain_14250, quality_world_obj_16000, quality_world_obj_12250
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


def upsample_x2(x):
    """Linear x2 upsampling with aligned corners: N frames -> 2N - 1 (as interp_time_series(factor=2))."""
    t = np.arange(2 * len(x) - 1) / 2.0
    return np.stack([np.interp(t, np.arange(len(x)), x[:, i]) for i in range(x.shape[1])], axis=1)


def pose_errors(pos_a, rot_a, pos_b, rot_b):
    """Per-frame position error [m] and orientation error [deg] of two [T, 3] / [T, 4] trajectories."""
    pos_err = np.linalg.norm(pos_a - pos_b, axis=-1)
    qa = rot_a / np.linalg.norm(rot_a, axis=-1, keepdims=True)
    qb = rot_b / np.linalg.norm(rot_b, axis=-1, keepdims=True)
    dot = np.clip(np.abs((qa * qb).sum(-1)), 0.0, 1.0)
    return pos_err, np.degrees(2.0 * np.arccos(dot))


def print_table(rows, suffix, title, pos_thresh, rot_thresh):
    """One table over all clips; per clip the error <pair>_pos<suffix> / <pair>_rot<suffix> is used."""
    print(f"\n{title}")
    header = f"{'pair':<20} {'reached':>8} {'pos ok':>8} {'rot ok':>8} " \
             f"{'pos mean':>9} {'pos med':>9} {'rot mean':>9} {'rot med':>9}"
    print(header)
    print("-" * len(header))
    for a, b in PAIRS:
        pos = np.array([r[f"{a}-{b}_pos{suffix}"] for r in rows])
        rot = np.array([r[f"{a}-{b}_rot{suffix}"] for r in rows])
        pos_ok, rot_ok = pos < pos_thresh, rot < rot_thresh
        print(f"{a + ' -> ' + b:<20} {100 * (pos_ok & rot_ok).mean():7.1f}% {100 * pos_ok.mean():7.1f}% "
              f"{100 * rot_ok.mean():7.1f}% {pos.mean():8.3f}m {np.median(pos):8.3f}m "
              f"{rot.mean():8.1f}° {np.median(rot):8.1f}°")


ERROR_KINDS = (("", "final"), ("_median", "trajectory median"), ("_mean", "trajectory mean"))


def plot_errors(rows, pos_thresh, rot_thresh, path):
    """Empirical CDF of the object position / orientation error per pair (share of clips with an error <= x), one
    row per error kind: final frame, median and mean over the trajectory."""
    fig, axes = plt.subplots(len(ERROR_KINDS), 2, figsize=(12, 4.2 * len(ERROR_KINDS)), squeeze=False)
    panels = [(ax, f"{key}{suffix}", f"{kind} object {what} error [{unit}]", thresh, fmt)
              for row_axes, (suffix, kind) in zip(axes, ERROR_KINDS)
              for ax, (key, what, unit, thresh, fmt) in zip(row_axes, (
                  ("pos", "position", "m", pos_thresh, "{:.2f} m"),
                  ("rot", "orientation", "deg", rot_thresh, "{:.0f}°")))]
    for ax, key, xlabel, thresh, fmt in panels:
        xmax = 0.0
        for (a, b), color in zip(PAIRS, PAIR_COLORS):
            err = np.sort([r[f"{a}-{b}_{key}"] for r in rows])
            share = 100 * np.arange(1, len(err) + 1) / len(err)
            ax.step(np.concatenate([[0], err]), np.concatenate([[0], share]), where="post", lw=2, color=color,
                    label=f"{a} -> {b} ({100 * (err < thresh).mean():.0f}% < {fmt.format(thresh)})")
            xmax = max(xmax, np.percentile(err, 98))
        ax.axvline(thresh, color=MUTED, lw=1, ls="--", label=f"threshold {fmt.format(thresh)}")
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
    fig.suptitle(f"Object pose error, {len(rows)} clips ({ROLLOUT_DIR.name})", color=INK)
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

    rows, short, misaligned = [], [], []
    for name, source_file, retarget_file, rollout_file in clips:
        source = to_numpy(load(source_file))[1:]  # frame 0 is all zeros
        retarget = to_numpy(load(retarget_file))
        rollout = to_numpy(load(rollout_file)["observed_hoi"])[:, :630]
        # observed_hoi[i] belongs to retarget frame i + 1, so a complete rollout has len(retarget) - 1 frames
        if len(rollout) != len(retarget) - 1:
            short.append(f"{name} ({len(rollout) + 1}/{len(retarget)} frames)")
        if 2 * len(source) - 1 != len(retarget):
            misaligned.append(f"{name} ({len(source) + 1} source / {len(retarget)} retarget frames)")
            continue

        # the source upsampled 30 -> 60 Hz, frame-aligned with the retarget; its last frame stays the last frame
        src_pos = upsample_x2(source[:, OBJ_POS_SRC]) * args.scale * xyz_multiplier(name)
        src_rot = upsample_x2(source[:, OBJ_ROT_SRC])
        frames = slice(1, len(rollout) + 1)  # retarget frames covered by the rollout
        traj = {
            "source": (src_pos[frames], src_rot[frames]),
            "retarget": (retarget[frames, OBJ_POS_G1], retarget[frames, OBJ_ROT_G1]),
            "rollout": (rollout[:, OBJ_POS_G1], rollout[:, OBJ_ROT_G1]),
        }
        final = {
            "source": (src_pos[-1:], src_rot[-1:]),
            "retarget": (retarget[-1:, OBJ_POS_G1], retarget[-1:, OBJ_ROT_G1]),
            "rollout": (rollout[-1:, OBJ_POS_G1], rollout[-1:, OBJ_ROT_G1]),
        }
        row = {"clip": name}
        for a, b in PAIRS:
            pos_err, rot_err = pose_errors(*final[a], *final[b])
            row[f"{a}-{b}_pos"], row[f"{a}-{b}_rot"] = pos_err[0], rot_err[0]
            pos_err, rot_err = pose_errors(*traj[a], *traj[b])
            row[f"{a}-{b}_pos_median"], row[f"{a}-{b}_rot_median"] = np.median(pos_err), np.median(rot_err)
            row[f"{a}-{b}_pos_mean"], row[f"{a}-{b}_rot_mean"] = pos_err.mean(), rot_err.mean()
        rows.append(row)
    if not rows:
        raise SystemExit("no clip with matching source and retarget length")

    print(f"{len(rows)} clips with source, retarget and rollout "
          f"(goal: pos < {args.pos_thresh:.3f} m and rot < {args.rot_thresh:.1f} deg)")
    if short:
        print(f"warning: {len(short)} rollouts end early, their last frame is used: {', '.join(short)}")
    if misaligned:
        print(f"warning: {len(misaligned)} clips skipped, source and retarget lengths do not match: "
              f"{', '.join(misaligned)}")
    for suffix, kind in ERROR_KINDS:
        print_table(rows, suffix, f"{kind} error", args.pos_thresh, args.rot_thresh)

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
