"""Run the included tracking teacher on one G1 motion."""
import argparse
import os
from pathlib import Path
import runpy
import sys

import yaml

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "ultra"))

# see ultra/run.py: TorchScript must be disabled before torch is imported
os.environ.setdefault("PYTORCH_JIT", "0")

import torch
from learning import ultra_players


def run_rollout(self):
    task = self.env.task
    source = Path(task.motion_file[0])
    source_frames = int(task.max_episode_length[0]) - 20  # loader pads the last frame
    task._enable_early_termination = False
    task.rollout_length = int(task.max_episode_length[0]) + 1

    obs = self.env_reset()
    self.get_batch_size(obs["obs"], 1)
    observed, reference, actions = [], [], []
    for _ in range(source_frames - 1):
        action = self.get_action(obs, True)
        next_obs, _, done, _ = self.env_step(self.env, action)
        obs = next_obs if isinstance(next_obs, dict) else {"obs": next_obs}
        observed.append(task._curr_obs[0].detach().cpu().clone())
        reference.append(task._curr_ref_obs[0].detach().cpu().clone())
        actions.append(action[0].detach().cpu().clone())
        if bool(done[0]):
            break

    output = Path(os.environ["ULTRA_TEACHER_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    result = output / "rollout.pt"
    torch.save({"observed_hoi": torch.stack(observed),
                "reference_hoi": torch.stack(reference),
                "actions": torch.stack(actions),
                "motion_file": str(source)}, result)
    print(f"Saved {len(actions)} teacher actions to {result}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", type=Path, required=True,
                        help="directory containing one released [T,630] G1 motion")
    parser.add_argument("--checkpoint", type=Path,
                        default=repo / "ultra/weights/teacher_ultra_inference.pth")
    parser.add_argument("--output-dir", type=Path, default=repo / "output/teacher_inference")
    parser.add_argument("--seed", type=int, default=42)
    # Added: inputs of teachers trained with the retarget quality observation and/or the object curriculum
    parser.add_argument("--quality-dir", type=Path,
                        help="retarget quality folder (<motion>_quality); required for teachers trained with "
                             "env.motionQualityDir (numObs 4054)")
    parser.add_argument("--quality-column", type=int, default=0,
                        help="env.motionQualityColumn the teacher was trained with")
    parser.add_argument("--source-dir", type=Path,
                        help="folder of the original [T,591] source motions (env.motionSourceDir); the object "
                             "reference is blended towards the source trajectory with --obj-curriculum")
    parser.add_argument("--obj-curriculum", type=float, default=1.0,
                        help="fixed object curriculum progress with --source-dir: 0 = retargeted, 1 = source "
                             "object trajectory")
    args = parser.parse_args()
    motion_dir = args.motion_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    motion_files = sorted(motion_dir.glob("*.pt"))
    if len(motion_files) != 1:
        parser.error("--motion-dir must contain exactly one .pt motion")
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")

    # Added: env config = g1_teacher_no_dr.yaml plus the optional quality / source inputs, written next to the output
    cfg_env = yaml.safe_load((repo / "ultra/data/cfg/g1_teacher_no_dr.yaml").read_text())
    if args.quality_dir is not None:
        if not args.quality_dir.is_dir():
            parser.error(f"--quality-dir not found: {args.quality_dir}")
        cfg_env["env"]["motionQualityDir"] = str(args.quality_dir.resolve())
        cfg_env["env"]["motionQualityColumn"] = args.quality_column
        cfg_env["env"]["numObs"] += 2  # quality at t+1 and t+16
    if args.source_dir is not None:
        if not args.source_dir.is_dir():
            parser.error(f"--source-dir not found: {args.source_dir}")
        cfg_env["env"]["motionSourceDir"] = str(args.source_dir.resolve())
        cfg_env["env"]["objCurriculumProgress"] = args.obj_curriculum
    # Added: check the observation size against the checkpoint before starting Isaac Sim
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    ckpt_obs = model["running_mean_std.running_mean"].shape[0] if "running_mean_std.running_mean" in model \
        else model["a2c_network.actor_mlp.0.weight"].shape[1]
    del model
    if ckpt_obs != cfg_env["env"]["numObs"]:
        hint = " (trained with the retarget quality observation: pass --quality-dir)" \
            if ckpt_obs == cfg_env["env"]["numObs"] + 2 else ""
        parser.error(f"{checkpoint} expects {ckpt_obs} observations, this configuration gives "
                     f"{cfg_env['env']['numObs']}{hint}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_env_path = output_dir / "env.yaml"
    cfg_env_path.write_text(yaml.safe_dump(cfg_env, sort_keys=False))

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["ULTRA_TEACHER_OUTPUT_DIR"] = str(output_dir)
    os.chdir(repo)
    ultra_players.UltraPlayerContinuous.run = run_rollout
    sys.argv = [str(repo / "ultra/run.py"),
                "--task", "UltraG1Retarget",
                "--cfg_env", str(cfg_env_path),
                "--cfg_train", "ultra/data/cfg/train/rlg/g1_teacher.yaml",
                "--test", "--headless", "--num_envs", "1", "--seed", str(args.seed),
                "--checkpoint", str(checkpoint), "--motion_file", str(motion_dir),
                "--output_path", str(output_dir / "rlgames")]
    runpy.run_path(str(repo / "ultra/run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
