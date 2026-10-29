"""Reproduce the PhysX blow-ups in the first policy step after a reset (see output/blowups/*.pt).

For every case (clip, start frame) all environments are reset to exactly that reference frame and one policy step
is simulated; an environment counts as exploded when a robot body is faster than --body-vel or the object faster
than --obj-vel after that step. The robot and object velocities after each of the 17 physics substeps of that step
are recorded, so the substep where the explosion starts is visible. Two modes per case:
  after_episode: before the reset the environments run a normal episode (random clip/frame, --prior-steps policy
                 steps), as in training
  after_episode_double: as after_episode, but the reset state is written twice before the step
  after_episode_lift: as after_episode, but robot and object are lifted 2 m and simulated for one physics substep
                 before the reset, so no contact of the previous episode survives
  repeat:        the environments are reset to the same frame twice in a row (state before the reset is the frame
                 itself), so a stale previous state cannot matter
Usage: python scripts/repro_reset_blowup.py [--cases clip:frame ...] [--hulls 10] [--num-envs 64] [--reps 20]
Without --cases, the cases of output/blowups/*.pt are used.
"""
import argparse
import importlib.abc
import importlib.util
import os
from pathlib import Path
import runpy
import sys
import tempfile

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "ultra"))
os.environ.setdefault("PYTORCH_JIT", "0")

import torch
from learning import ultra_players

ARGS = None


class _HullOverride(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Patch isaac.scene_cfg.object_usd_path after import so the object uses --hulls convex hulls."""

    def find_spec(self, name, path, target=None):
        if name != "isaac.scene_cfg":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        self._loader = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self._loader.exec_module(module)
        original = module.object_usd_path
        module.object_usd_path = lambda name, max_convex_hulls: original(name, ARGS.hulls)


def reset_to_frame(task, env_ids, data_id, frame):
    """Hybrid/random reference-state init (UltraG1Retarget._reset_hybrid_state_init) with a fixed clip and frame."""
    n = len(env_ids)
    task.data_id[env_ids] = data_id
    task.ref_index[env_ids] = 0
    task.progress_buf[env_ids] = frame
    task.start_times[env_ids] = frame
    task.is_stand_still[env_ids] = False
    task.stand_still_frame[env_ids] = frame
    task._hist_obs[env_ids] = 0
    task.contact_reset[env_ids] = 0
    # the clean-up humanoid.post_physics_step does for environments with reset_buf
    for buf in (task.last_actions, task.last_dof_vel, task.last_dof_pos, task.actions, task.action_history_buf,
                task.last_contact_forces, task.obs_history_buf):
        buf[env_ids] = 0
    ref = task.hoi_refs[data_id, 0, frame].unsqueeze(0).repeat(n, 1)
    ref[:, 2] += ARGS.root_lift  # optional: spawn the robot higher than the reference
    task._set_env_state(env_ids=env_ids, root_pos=ref[:, 0:3], root_rot=ref[:, 3:7], dof_pos=ref[:, 13:42],
                        root_vel=ref[:, 7:10], root_ang_vel=ref[:, 10:13], dof_vel=ref[:, 42:71])
    task._reset_target(env_ids)
    task._reset_env_tensors(env_ids)
    task._refresh_sim_tensors()
    task._compute_observations(env_ids)


def run_experiment(self):
    task = self.env.task
    task._enable_early_termination = False
    # the training-time detector still resets exploded environments, but must not print or dump into output/blowups
    task.cfg["env"]["blowup"] = {**task.cfg["env"].get("blowup", {}), "maxPrintsPerStep": 0}
    device = task.device
    all_ids = torch.arange(task.num_envs, device=device)
    clips = [os.path.basename(f) for f in task.motion_file]

    # record robot/object speed after every physics substep while `substeps` is a list
    substeps = None
    first_vel = {}
    refresh = task._refresh_sim_tensors

    def refresh_and_record():
        refresh()
        if substeps is not None:
            if not substeps:
                first_vel["v"] = task._rigid_body_vel.clone().cpu()  # body velocities after the first substep
            substeps.append(torch.stack((task._rigid_body_vel.norm(dim=-1).max(dim=-1).values,
                                         task._target_states[:, 7:10].norm(dim=-1)), dim=-1).cpu())
    task._refresh_sim_tensors = refresh_and_record

    obs = self.env_reset()
    self.get_batch_size(obs["obs"], 1)

    def step():
        nonlocal obs
        action = self.get_action(obs, True)
        next_obs, _, _, _ = self.env_step(self.env, action)
        obs = next_obs if isinstance(next_obs, dict) else {"obs": next_obs}

    results = []
    for clip, frame in ARGS.case_list:
        data_id = clips.index(clip)
        for mode in ARGS.modes:
            exploded, first_substeps, prior, flags, directions = 0, [], [], [], []
            for _ in range(ARGS.reps):
                if mode.startswith("after_episode"):
                    self.env.task.reset(all_ids)  # random clip and frame, as in training
                    obs = {"obs": task.obs_buf.clone()}
                    for _ in range(ARGS.prior_steps):
                        step()
                    if mode == "after_episode_lift":
                        # lift robot and object 2 m and simulate one physics substep, so that every contact of the
                        # previous episode is lost before the reset
                        task._humanoid_root_states[:, 2] += 2.0
                        task._target_states[:, 2] += 2.0
                        task._set_actor_root_state_all()
                        task._simulate()
                        task._refresh_sim_tensors()
                else:
                    reset_to_frame(task, all_ids, data_id, frame)
                    obs = {"obs": task.obs_buf.clone()}
                    step()
                # state before the reset, to see which property of it makes the reset explode
                root = task._humanoid_root_states
                target = task.hoi_refs[data_id, 0, frame]
                prior.append(torch.stack((
                    root[:, 2],                                                    # root height
                    (task._contact_forces.norm(dim=-1) > 1.0).sum(-1).float(),     # bodies in contact
                    task._rigid_body_vel.norm(dim=-1).max(-1).values,              # max body speed
                    (root[:, 0:3] - target[0:3]).norm(dim=-1),                     # teleport distance
                    (task._target_states[:, 0:3] - target[71:74]).norm(dim=-1),    # object teleport distance
                    task._rigid_body_pos[:, :, 2].min(-1).values,                  # lowest body
                ), dim=-1).cpu())
                prior_body_pos = task._rigid_body_pos[:, :task.num_bodies].clone().cpu()
                reset_to_frame(task, all_ids, data_id, frame)
                if mode == "after_episode_double":
                    reset_to_frame(task, all_ids, data_id, frame)  # write the reset state a second time
                obs = {"obs": task.obs_buf.clone()}
                substeps = []
                step()
                trace = torch.stack(substeps)  # [substeps, envs, (body speed, object speed)]
                substeps = None
                bad = (trace[-1, :, 0] > ARGS.body_vel) | (trace[-1, :, 1] > ARGS.obj_vel)
                flags.append(bad)
                exploded += int(bad.sum())
                for e in torch.where(bad)[0].tolist()[:2]:
                    first_substeps.append(trace[:, e].tolist())
                # direction of the fastest body after the first substep vs. where that body was before the reset
                ref_body_pos = task.hoi_data[data_id, frame, 84:201].view(-1, 3).cpu()
                for e in torch.where(bad)[0].tolist():
                    v = first_vel["v"][e, :task.num_bodies]
                    b = int(v.norm(dim=-1).argmax())
                    d = prior_body_pos[e, b] - ref_body_pos[b]
                    cos = float(torch.dot(v[b], d) / (v[b].norm() * d.norm() + 1e-9))
                    directions.append((b, cos, float(d.norm()), float(v[b].norm())))
            total = ARGS.reps * task.num_envs
            results.append((clip, frame, mode, exploded, total))
            print(f"[repro] {clip} frame {frame} {mode}: {exploded}/{total} exploded", flush=True)
            for tr in first_substeps[:3]:
                print("    body/object speed per substep: "
                      + " ".join(f"{b:.1f}/{o:.1f}" for b, o in tr), flush=True)
            for b, cos, dist, speed in directions[:8]:
                print(f"    fastest body {b} after substep 1: {speed:.0f} m/s, cos to its pre-reset position "
                      f"{cos:+.2f} ({dist:.2f} m away)", flush=True)
            prior, flags = torch.cat(prior), torch.cat(flags)
            names = ("root_z", "n_contacts", "max_body_vel", "root_jump", "obj_jump", "min_body_z")
            for i, name in enumerate(names):
                ok = prior[~flags, i]
                line = f"    prior {name:<13} ok: mean {ok.mean():.3f} [{ok.min():.3f}, {ok.max():.3f}]"
                if flags.any():
                    bad = prior[flags, i]
                    line += f" | exploded: mean {bad.mean():.3f} [{bad.min():.3f}, {bad.max():.3f}]"
                print(line, flush=True)

    print("\n[repro] summary (hulls %d)" % ARGS.hulls)
    for clip, frame, mode, exploded, total in results:
        print(f"[repro] {clip:<70} {frame:>4} {mode:<14} {exploded:>5}/{total}")


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="*", help="clip:frame, e.g. sub8_largebox_031_xyz..._080_080_080.pt:52")
    parser.add_argument("--motion-dir", type=Path, default=repo / "output/retarget_export_080_largebox")
    parser.add_argument("--checkpoint", type=Path, default=repo / "output/teacher/g1_teacher/nn/g1_teacher.pth")
    parser.add_argument("--hulls", type=int, default=10, help="object convex hulls (teacher 10, retarget 5)")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--prior-steps", type=int, default=20)
    parser.add_argument("--body-vel", type=float, default=50.0)
    parser.add_argument("--obj-vel", type=float, default=50.0)
    parser.add_argument("--root-lift", type=float, default=0.0, help="spawn the robot this much higher [m], on top of env.resetRootLift")
    parser.add_argument("--modes", nargs="+", default=["after_episode", "repeat"],
                        choices=["after_episode", "after_episode_double", "after_episode_lift", "repeat"])
    ARGS = parser.parse_args()

    if ARGS.cases:
        ARGS.case_list = [(c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1])) for c in ARGS.cases]
    else:
        ARGS.case_list = []
        for f in sorted((repo / "output/blowups").glob("*.pt")):
            d = torch.load(f)
            ARGS.case_list.append((d["clip"], int(d["start_time"])))
        ARGS.case_list = list(dict.fromkeys(ARGS.case_list))
    if not ARGS.case_list:
        parser.error("no cases: pass --cases or put blow-up dumps into output/blowups")

    # motion folder with the case clips plus a few others for the random episodes before the reset
    motion_dir = Path(tempfile.mkdtemp(prefix="repro_motions_"))
    clips = {c for c, _ in ARGS.case_list} | {p.name for p in sorted(ARGS.motion_dir.glob("*.pt"))[:20]}
    for c in clips:
        (motion_dir / c).symlink_to((ARGS.motion_dir / c).resolve())

    sys.meta_path.insert(0, _HullOverride())
    os.environ["WANDB_DISABLED"] = "true"
    os.chdir(repo)
    ultra_players.UltraPlayerContinuous.run = run_experiment
    sys.argv = [str(repo / "ultra/run.py"), "--task", "UltraG1Retarget",
                "--cfg_env", "ultra/data/cfg/g1_teacher.yaml",
                "--cfg_train", "ultra/data/cfg/train/rlg/g1_teacher.yaml",
                "--test", "--headless", "--num_envs", str(ARGS.num_envs),
                "--checkpoint", str(ARGS.checkpoint), "--motion_file", str(motion_dir),
                "--output_path", tempfile.mkdtemp(prefix="repro_out_")]
    runpy.run_path(str(repo / "ultra/run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
