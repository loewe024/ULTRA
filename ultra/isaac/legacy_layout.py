"""The ULTRA ("legacy") data layout of the G1 and the mapping to Isaac Lab's ordering.

ULTRA's datasets, observations, policies, the MuJoCo sim2sim code and the real-robot deployment all use the body
and joint order that IsaacGym produced from ``g1_29dof.urdf``: a depth-first traversal with children sorted by name.
The joint order equals the URDF (and Unitree motor) order. Isaac Lab orders bodies and joints differently, so the
simulation backend permutes every tensor it reads or writes with the index maps built here.

This module has no Isaac Lab dependency so it can also be used by the MuJoCo tools.
"""

import torch

LEGACY_BODY_NAMES = [
    "pelvis", "imu_in_pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link",
    "left_ankle_pitch_link", "left_ankle_roll_link",
    "pelvis_contour_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link",
    "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link", "d435_link", "head_link", "imu_in_torso",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
    "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link", "left_rubber_hand",
    "logo_link", "mid360_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link",
    "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link", "right_rubber_hand",
]  # fmt: skip

LEGACY_DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip

# Per-DOF drive parameters of the released policies, in LEGACY_DOF_NAMES order.
G1_STIFFNESS = [40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
                40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
                40.179238, 28.501246, 28.501246,
                14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327,
                14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327]  # fmt: skip
G1_DAMPING = [2.557890, 6.308802, 2.557890, 6.308802, 1.814446, 1.814446,
              2.557890, 6.308802, 2.557890, 6.308802, 1.814446, 1.814446,
              2.557890, 1.814446, 1.814446,
              0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142,
              0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142]  # fmt: skip
# NOTE: feet + waist roll/pitch are doubled vs ARMATURE_5020
G1_ARMATURE = [0.010178, 0.025102, 0.010178, 0.025102, 0.007219, 0.007219,
               0.010178, 0.025102, 0.010178, 0.025102, 0.007219, 0.007219,
               0.010178, 0.007219, 0.007219,
               0.003610, 0.003610, 0.003610, 0.003610, 0.003610, 0.004250, 0.004250,
               0.003610, 0.003610, 0.003610, 0.003610, 0.003610, 0.004250, 0.004250]  # fmt: skip
G1_EFFORT = [88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
             88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
             88.0, 50.0, 50.0,
             25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
             25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0]  # fmt: skip


def per_joint(values):
    """Map a LEGACY_DOF_NAMES-ordered list to the ``{joint_name: value}`` dict Isaac Lab actuator configs expect."""
    return {name: float(v) for name, v in zip(LEGACY_DOF_NAMES, values)}


def legacy_to_sim_index(sim_names, legacy_names):
    """Index tensor ``idx`` with ``sim_tensor[..., idx] == legacy_tensor``, i.e. ``idx[i]`` is the simulator index
    of ``legacy_names[i]``."""
    missing = [n for n in legacy_names if n not in sim_names]
    if missing:
        raise ValueError(f"Simulator is missing {missing}; available: {list(sim_names)}")
    return torch.tensor([list(sim_names).index(n) for n in legacy_names], dtype=torch.long)


def xyzw_to_wxyz(q):
    return torch.roll(q, 1, dims=-1)


def wxyz_to_xyzw(q):
    return torch.roll(q, -1, dims=-1)
