#!/usr/bin/env python3
"""Export G1 rollouts from prepared OMOMO SMPL-X references."""
import argparse
import copy
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml


def scale_tag(xyz):
    return "xyz" + "-".join(f"{value:.6f}".replace(".", "p") for value in xyz)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-scale", choices=("080_080_080", "100_100_100"), default="080_080_080")
    parser.add_argument("--xyz", type=float, nargs=3, action="append", metavar=("X", "Y", "Z"))
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    source_dir = args.input_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    variants = args.xyz or [[1.0, 1.0, 1.0]]
    env_template = yaml.safe_load((repo / "ultra/data/cfg/g1_retarget_smplx.yaml").read_text())
    train_template = yaml.safe_load((repo / "ultra/data/cfg/train/rlg/g1_retarget_smplx.yaml").read_text())
    pattern = re.compile(r"^sub[0-9]+_([a-z0-9]+)_[0-9]+$")
    sources = []
    for source in sorted(source_dir.glob("*.pt")):
        match = pattern.fullmatch(source.stem)
        if match is None:
            continue
        asset_name = f"{match.group(1)}_{args.asset_scale}"
        asset_dir = repo / "ultra/data/assets/objects/diverse" / asset_name
        if not (asset_dir / f"{asset_name}.urdf").is_file():
            continue
        if not (asset_dir / f"{asset_name}.obj").is_file():
            continue
        sources.append(source)
    if not sources:
        raise RuntimeError("no supported OMOMO motions found")

    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = output_dir.parent / f"{output_dir.name}_work"
    exported = 0
    for source in sources:
        for xyz in variants:
            tag = scale_tag(xyz)
            destination = output_dir / f"{source.stem}_{tag}_{args.asset_scale}.pt"
            if destination.is_file():
                print("existing:", destination, flush=True)
                continue

            work = work_root / source.stem / args.asset_scale / tag
            motion_dir = work / "motion"
            motion_dir.mkdir(parents=True, exist_ok=True)
            link = motion_dir / f"{source.stem}_{args.asset_scale}.pt"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(source)

            env_cfg = copy.deepcopy(env_template)
            env_cfg["env"].update({
                "numEnvs": 1,
                "motion_file": str(motion_dir),
                "stateInit": "Start",
                "enableEarlyTermination": False,
                "rolloutLength": 100000,
                "sparseXYZMultiplier": list(xyz),
                "retargetExportPath": str(destination),
            })
            env_path = work / "env.yaml"
            env_path.write_text(yaml.safe_dump(env_cfg, sort_keys=False))
            train_cfg = copy.deepcopy(train_template)
            train_cfg["params"]["config"]["player"] = {
                "games_num": 1, "n_game_life": 1, "determenistic": True, "render": False
            }
            train_cfg["params"]["config"]["name"] = "g1_retarget_export"
            train_cfg["params"]["config"]["full_experiment_name"] = "g1_retarget_export"
            train_path = work / "train.yaml"
            train_path.write_text(yaml.safe_dump(train_cfg, sort_keys=False))
            command = [
                sys.executable, "ultra/run.py", "--task", "UltraG1",
                "--cfg_env", str(env_path), "--cfg_train", str(train_path),
                "--headless", "--test", "--checkpoint", str(checkpoint),
                "--num_envs", "1", "--seed", str(args.seed),
                "--output_path", str(work / "eval"),
            ]
            log_path = work / "inference.log"
            environment = os.environ.copy()
            environment["WANDB_DISABLED"] = "true"
            environment["WANDB_MODE"] = "disabled"
            print("exporting:", source.name, tag, args.asset_scale, flush=True)
            with log_path.open("w") as log:
                result = subprocess.run(
                    command, cwd=repo, env=environment,
                    stdout=log, stderr=subprocess.STDOUT
                )
            if result.returncode != 0 or not destination.is_file():
                raise RuntimeError(f"export failed; see {log_path}")
            exported += 1
    print(f"exported {exported} G1 rollouts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
