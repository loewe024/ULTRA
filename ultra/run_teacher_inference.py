"""Roll out a trained tracking teacher on the G1 motions of a rollout config (ultra/data/cfg/teacher_rollout.yaml).

All clips of motion_dir that have no rollout in output_dir yet run in parallel in one Isaac Sim process, one env
per clip (env i tracks motion i, see UltraG1Retarget._reset_ref_state_init). Output in output_dir:
  <clip>.pt          rollout: observed_hoi / reference_hoi [T-1, 747], actions [T-1, 29], motion_file
  env.yaml           env config of the run (g1_teacher_no_dr.yaml + quality / source object + the reward weights of
                     g1_teacher.yaml + motion folder and checkpoint); a rerun with other settings stops (exit code 2)
  logs/inference_<time>.log   output of Isaac Sim

Usage: python ultra/run_teacher_inference.py [--cfg ultra/data/cfg/teacher_rollout.yaml]
"""
import argparse
import difflib
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

import yaml

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "ultra"))

# see ultra/run.py: TorchScript must be disabled before torch is imported
os.environ.setdefault("PYTORCH_JIT", "0")

import torch
from learning import ultra_players
from utils.rollout_cfg import DEFAULT_CFG, config_error, load_rollout_cfg

ENV_CFG_HEADER = "# Manually check and if necessary correct the training weights.\n"
PAD_FRAMES = 20  # UltraG1Retarget._load_motion appends 20 copies of the last frame
_run = {}        # output folder and terminal of the running rollout, set by main()


def report(message):
    """Print to the terminal; stdout / stderr of this process go to the log file during the rollout."""
    _run["terminal"].write(message + "\n")
    _run["terminal"].flush()


def run_rollout(self):
    """Replaces UltraPlayerContinuous.run: deterministic rollout of all envs, one rollout file per clip."""
    task = self.env.task
    num_envs = task.num_envs
    # steps per clip: one per reference frame, without the padded end frames (as for a single clip before)
    steps = (task.max_episode_length.cpu() - PAD_FRAMES - 1).tolist()
    if len(steps) != num_envs:
        raise RuntimeError(f"{len(steps)} motions for {num_envs} envs")
    task._enable_early_termination = False
    task.rollout_length = max(steps) + 2

    obs = self.env_reset()
    self.get_batch_size(obs["obs"], num_envs)
    # the reset assigns the motions (env i -> motion i % num_motions, UltraG1Retarget._reset_ref_state_init)
    data_id = task.data_id.cpu()
    if not torch.equal(data_id, torch.arange(num_envs)):
        raise RuntimeError(f"env -> motion assignment is not env i -> motion i: {data_id.tolist()}")
    observed, reference, actions = [], [], []
    for step in range(max(steps)):
        action = self.get_action(obs, True)
        next_obs, _, _, _ = self.env_step(self.env, action)
        obs = next_obs if isinstance(next_obs, dict) else {"obs": next_obs}
        observed.append(task._curr_obs.detach().cpu().clone())
        reference.append(task._curr_ref_obs.detach().cpu().clone())
        actions.append(action.detach().cpu().clone())
        if (step + 1) % 100 == 0:
            report(f"  step {step + 1}/{max(steps)}")
    observed, reference, actions = torch.stack(observed), torch.stack(reference), torch.stack(actions)

    output_dir = _run["output_dir"]
    for i in range(num_envs):
        motion = Path(task.motion_file[i]).resolve()  # the symlink in .work/ resolved to the real motion
        n = steps[i]
        torch.save({"observed_hoi": observed[:n, i].clone(),
                    "reference_hoi": reference[:n, i].clone(),
                    "actions": actions[:n, i].clone(),
                    "motion_file": str(motion)}, output_dir / f"{motion.stem}.pt")
    report(f"saved {num_envs} rollouts to {output_dir}")
    # here, not in main(): Isaac Sim ends the process when ultra/run.py returns
    shutil.rmtree(output_dir / ".work", ignore_errors=True)


def build_env_cfg(rcfg):
    """Env config of the run: g1_teacher_no_dr.yaml (no randomization, start at frame 0) plus the settings of the
    rollout config, the reward weights of g1_teacher.yaml and the motion folder / checkpoint as log."""
    cfg_env = yaml.safe_load((repo / "ultra/data/cfg/g1_teacher_no_dr.yaml").read_text())
    # The current g1_teacher.yaml need not be the one the checkpoint was trained with, hence ENV_CFG_HEADER.
    train_cfg = repo / "ultra/data/cfg/g1_teacher.yaml"
    cfg_train_env = yaml.safe_load(train_cfg.read_text())
    quality_scale = cfg_train_env["env"].get("rewardQualityScale")
    if quality_scale is not None:
        # its q_eff is the quality observation (instead of quality_dir) and needs the source object; with the
        # retargeted object trajectory c = 0, so q_eff = 1 as at the start of the training curriculum
        if rcfg.quality_dir is not None:
            config_error(f"{train_cfg} has rewardQualityScale, which replaces the quality observation: "
                         f"set quality_dir: null in {rcfg.path}")
        cfg_env["env"]["rewardQualityScale"] = quality_scale
        cfg_env["env"]["numObs"] += 2  # q_eff at t+1 and t+16
        cfg_env["env"]["motionSourceDir"] = str(rcfg.source_dir)
        cfg_env["env"]["objCurriculumProgress"] = 1.0 if rcfg.use_source_object else 0.0
    if rcfg.quality_dir is not None:
        cfg_env["env"]["motionQualityDir"] = str(rcfg.quality_dir)
        cfg_env["env"]["motionQualityColumn"] = rcfg.quality_column
        cfg_env["env"]["numObs"] += 2  # quality at t+1 and t+16
    if rcfg.use_source_object:
        cfg_env["env"]["motionSourceDir"] = str(rcfg.source_dir)
        cfg_env["env"]["objCurriculumProgress"] = 1.0
    cfg_env["env"]["rewardWeights"] = cfg_train_env["env"]["rewardWeights"]
    cfg_env["smooth_rewards"] = cfg_train_env["smooth_rewards"]
    # only a record: the clips of the run are passed with --motion_file
    cfg_env["env"]["motion_file"] = str(rcfg.motion_dir)
    cfg_env["eval"] = {"checkpoint": str(rcfg.checkpoint), "seed": rcfg.seed, "reward_source": str(train_cfg)}
    return cfg_env


def check_obs_size(rcfg, cfg_env):
    model = torch.load(rcfg.checkpoint, map_location="cpu", weights_only=False)["model"]
    ckpt_obs = model["running_mean_std.running_mean"].shape[0] if "running_mean_std.running_mean" in model \
        else model["a2c_network.actor_mlp.0.weight"].shape[1]
    if ckpt_obs != cfg_env["env"]["numObs"]:
        hint = " (trained with a quality observation: set quality_dir or rewardQualityScale in g1_teacher.yaml)" \
            if ckpt_obs == cfg_env["env"]["numObs"] + 2 else ""
        config_error(f"{rcfg.checkpoint} expects {ckpt_obs} observations, {rcfg.path} gives "
                         f"{cfg_env['env']['numObs']}{hint}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG, help="rollout config")
    args = parser.parse_args()
    rcfg = load_rollout_cfg(args.cfg)
    output_dir = rcfg.output_dir

    cfg_env = build_env_cfg(rcfg)
    check_obs_size(rcfg, cfg_env)
    # one env.yaml per output folder: written by the first run, later runs must have the same settings
    env_text = ENV_CFG_HEADER + yaml.safe_dump(cfg_env, sort_keys=False)
    env_path = output_dir / "env.yaml"
    if env_path.is_file() and env_path.read_text() != env_text:
        diff = difflib.unified_diff(env_path.read_text().splitlines(), env_text.splitlines(),
                                    str(env_path), "this run", lineterm="")
        print("\n".join(diff), file=sys.stderr)
        print(f"settings differ from {env_path} (other checkpoint / options?), use another output_dir",
              file=sys.stderr)
        sys.exit(2)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_path.write_text(env_text)

    motions = sorted(rcfg.motion_dir.glob("*.pt"))
    if not motions:
        config_error(f"no .pt motions in {rcfg.motion_dir}")
    todo = [m for m in motions if not (output_dir / m.name).is_file()]
    print(f"{len(motions)} clips in {rcfg.motion_dir}, {len(motions) - len(todo)} already rolled out, "
          f"{len(todo)} to do", flush=True)
    if not todo:
        return
    # the task reads motion_file as a folder (ultra.py), so the clips of this run are linked into one
    work = output_dir / ".work"
    shutil.rmtree(work, ignore_errors=True)
    (work / "motion").mkdir(parents=True)
    for m in todo:
        (work / "motion" / m.name).symlink_to(m.resolve())

    log = output_dir / "logs" / f"inference_{time.strftime('%Y%m%d-%H%M%S')}.log"
    log.parent.mkdir(exist_ok=True)
    print(f"rolling out {len(todo)} clips in parallel, Isaac Sim log: {log}", flush=True)
    _run["output_dir"] = output_dir
    _run["terminal"] = os.fdopen(os.dup(1), "w")
    log_fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)

    os.environ["WANDB_DISABLED"] = "true"
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.chdir(repo)
    ultra_players.UltraPlayerContinuous.run = run_rollout
    sys.argv = [str(repo / "ultra/run.py"),
                "--task", "UltraG1Retarget",
                "--cfg_env", str(env_path),
                "--cfg_train", "ultra/data/cfg/train/rlg/g1_teacher.yaml",
                "--test", "--headless", "--num_envs", str(len(todo)), "--seed", str(rcfg.seed),
                "--checkpoint", str(rcfg.checkpoint), "--motion_file", str(work / "motion"),
                "--output_path", str(work / "rlgames")]
    try:
        runpy.run_path(str(repo / "ultra/run.py"), run_name="__main__")
    finally:  # reached on errors; after a successful rollout Isaac Sim may end the process before
        if not all((output_dir / m.name).is_file() for m in todo):
            report(f"rollout failed, see {log}")
            sys.exit(1)


if __name__ == "__main__":
    main()
