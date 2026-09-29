"""Check the legacy-layout mapping of the Isaac Lab environment against MuJoCo forward kinematics.

Random joint positions and root poses are written through the environment's legacy interface (xyzw quaternions,
LEGACY_DOF_NAMES order, env-local positions). The body poses read back (LEGACY_BODY_NAMES order) must match
MuJoCo's forward kinematics of ultra/data/assets/g1/g1_29dof.xml. Object root poses must round-trip as well.

Usage (from the repository root):
    python scripts/check_layout.py [--num_envs 4]
"""

import argparse
import os
import sys
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--tolerance", type=float, default=2e-3, help="max position / rotation error (m / rad)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "ultra"))
os.chdir(REPO)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from isaac.legacy_layout import LEGACY_BODY_NAMES, LEGACY_DOF_NAMES  # noqa: E402
from utils.parse_task import parse_task  # noqa: E402


def write_standing_motion(directory):
    """Minimal [T, 630] reference so the teacher task can be created without the datasets."""
    frames = torch.zeros(10, 630)
    frames[:, 2] = 0.8
    frames[:, 6] = 1.0
    frames[:, 71:74] = torch.tensor([0.6, 0.0, 0.25])
    frames[:, 77] = 1.0
    frames[:, 201:357] = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(39)
    torch.save(frames, os.path.join(directory, "sub1_largebox_000_080_080_080.pt"))


def random_quat_xyzw(n, rng):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def mujoco_body_poses(model, data, root_pos, root_quat_xyzw, dof_pos):
    data.qpos[:] = 0
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = np.roll(root_quat_xyzw, 1)
    for name, q in zip(LEGACY_DOF_NAMES, dof_pos):
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = q
    mujoco.mj_kinematics(model, data)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in LEGACY_BODY_NAMES]
    return data.xpos[ids].copy(), np.roll(data.xquat[ids], -1, axis=1)


def rotation_error(q1, q2):
    """Angle between two sets of (x, y, z, w) quaternions."""
    dot = np.abs(np.sum(q1 * q2, axis=-1)).clip(0.0, 1.0)
    return 2.0 * np.arccos(dot)


def main():
    with tempfile.TemporaryDirectory() as motion_dir:
        write_standing_motion(motion_dir)
        with open("ultra/data/cfg/g1_teacher_no_dr.yaml") as f:
            cfg = yaml.safe_load(f)
        cfg["env"]["numEnvs"] = args.num_envs
        cfg["env"]["motion_file"] = motion_dir
        cfg_train = {"params": {"seed": 0}}
        run_args = argparse.Namespace(task="UltraG1Retarget", headless=True, device=args.device, rl_device=args.device)
        task, _ = parse_task(run_args, cfg, cfg_train)

    rng = np.random.default_rng(0)
    n = task.num_envs
    lower = task.dof_limits_lower.cpu().numpy()
    upper = task.dof_limits_upper.cpu().numpy()
    dof_pos = np.clip(rng.uniform(-0.3, 0.3, size=(n, task.num_dof)), lower, upper)
    root_pos = np.column_stack([rng.uniform(-0.5, 0.5, n), rng.uniform(-0.5, 0.5, n), np.full(n, 1.5)])
    root_rot = random_quat_xyzw(n, rng)
    obj_pos = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), np.full(n, 2.5)])
    obj_rot = random_quat_xyzw(n, rng)

    device = task.device
    env_ids = torch.arange(n, device=device)
    task._humanoid_root_states[:] = 0
    task._humanoid_root_states[:, 0:3] = torch.tensor(root_pos, device=device, dtype=torch.float)
    task._humanoid_root_states[:, 3:7] = torch.tensor(root_rot, device=device, dtype=torch.float)
    task._dof_pos[:] = torch.tensor(dof_pos, device=device, dtype=torch.float)
    task._dof_vel[:] = 0
    task._target_states[:] = 0
    task._target_states[:, 0:3] = torch.tensor(obj_pos, device=device, dtype=torch.float)
    task._target_states[:, 3:7] = torch.tensor(obj_rot, device=device, dtype=torch.float)
    task._set_actor_root_state_indexed(task._humanoid_actor_ids[env_ids])
    task._set_dof_state_indexed(task._humanoid_actor_ids[env_ids])
    task._set_actor_root_state_indexed(task._tar_actor_ids[env_ids])
    # one 1 ms physics step (no gravity, no actuation, no contacts at this height) to update the link poses
    task._set_gravity([0.0, 0.0, 0.0])
    task._apply_dof_efforts(torch.zeros(n, task.num_dof, device=device))
    task._simulate()
    task._refresh_sim_tensors()

    # The physics step may move the joints slightly (self-collisions, limits), so the kinematics are checked
    # against the state read back; the round trip of the written state is reported separately.
    dof_roundtrip = float(np.abs(task._dof_pos.cpu().numpy() - dof_pos).max())
    print(f"[check_layout] joint positions: max round-trip deviation after one step {dof_roundtrip:.2e} rad")
    dof_pos = task._dof_pos.cpu().numpy()
    root_pos = task._humanoid_root_states[:, 0:3].cpu().numpy()
    root_rot = task._humanoid_root_states[:, 3:7].cpu().numpy()

    model = mujoco.MjModel.from_xml_path("ultra/data/assets/g1/g1_29dof.xml")
    data = mujoco.MjData(model)
    body_pos = task._rigid_body_pos.cpu().numpy()
    body_rot = task._rigid_body_rot.cpu().numpy()
    pos_err = np.zeros((n, len(LEGACY_BODY_NAMES)))
    rot_err = np.zeros((n, len(LEGACY_BODY_NAMES)))
    for i in range(n):
        ref_pos, ref_rot = mujoco_body_poses(model, data, root_pos[i], root_rot[i], dof_pos[i])
        pos_err[i] = np.abs(body_pos[i] - ref_pos).max(axis=-1)
        rot_err[i] = rotation_error(body_rot[i], ref_rot)
    max_pos, max_rot = float(pos_err.max()), float(rot_err.max())
    worst = LEGACY_BODY_NAMES[int(np.argmax(rot_err.max(axis=0)))]
    obj_pos_err = float(np.abs(task._target_states[:, 0:3].cpu().numpy() - obj_pos).max())
    obj_rot_err = float(rotation_error(task._target_states[:, 3:7].cpu().numpy(), obj_rot).max())

    print(f"[check_layout] bodies: max position error {max_pos:.2e} m, max rotation error {max_rot:.2e} rad ({worst})")
    print(f"[check_layout] object: max position error {obj_pos_err:.2e} m, max rotation error {obj_rot_err:.2e} rad")
    ok = max(max_pos, max_rot, obj_pos_err, obj_rot_err) < args.tolerance
    print("[check_layout] PASSED" if ok else "[check_layout] FAILED")
    task.close()
    return ok


if __name__ == "__main__":
    passed = main()
    simulation_app.close()
    sys.exit(0 if passed else 1)
