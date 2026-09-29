"""Isaac Lab environment configuration for the ULTRA tasks.

The task parameters (rewards, observations, domain randomization, ...) still live in the YAML files under
``ultra/data/cfg``. ``make_env_cfg`` turns such a YAML dict into an :class:`UltraEnvCfg` with the simulation
settings of its ``sim`` block; the environment then adds the scene (G1 articulation, per-environment object, contact
sensors, ground) with ``finalize_env_cfg`` once it knows its objects and observation size.
"""

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaac.legacy_layout import G1_ARMATURE, G1_DAMPING, G1_EFFORT, G1_STIFFNESS, per_joint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
USD_ROOT = os.path.join(REPO_ROOT, "ultra", "data", "assets", "usd")
SIM_DT = 1.0 / 1000.0
# IsaacGym default rigid-body damping, which the released policies were trained with.
IG_ANGULAR_DAMPING = 0.5
IG_MAX_ANGULAR_VELOCITY = 64.0
IG_MAX_LINEAR_VELOCITY = 1000.0


@configclass
class UltraSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    object: RigidObjectCfg = MISSING
    robot_contact: ContactSensorCfg = MISSING
    object_contact: ContactSensorCfg = MISSING
    terrain: TerrainImporterCfg = MISSING


@configclass
class UltraEnvCfg(DirectRLEnvCfg):
    """Isaac Lab config of an ULTRA task.

    ``task`` holds the task YAML as a dict. Item access (``cfg["env"]``, ``cfg.get(...)``) is forwarded to it so the
    ULTRA task code, which was written against that dict, can keep reading its parameters from ``self.cfg``.
    """

    task: dict = MISSING
    object_names: list = MISSING
    decimation: int = MISSING
    episode_length_s: float = 1.0e6  # episodes end through the task's own termination logic
    observation_space: int = MISSING
    action_space: int = MISSING
    sim: SimulationCfg = MISSING
    scene: UltraSceneCfg = MISSING
    viewer: ViewerCfg = ViewerCfg(eye=(0.0, -3.0, 1.0), lookat=(0.0, 0.0, 1.0), origin_type="env", env_index=0)

    def __getitem__(self, key):
        return self.task[key]

    def __setitem__(self, key, value):
        self.task[key] = value

    def __contains__(self, key):
        return key in self.task

    def get(self, key, default=None):
        return self.task.get(key, default)


def robot_usd_path(robot_type):
    """``g1/g1_29dof.urdf`` (the YAML ``robotType``) -> converted USD (see ``scripts/convert_assets.py``)."""
    name = os.path.splitext(os.path.basename(robot_type))[0]
    return os.path.join(USD_ROOT, os.path.dirname(robot_type), f"{name}.usd")


def object_usd_path(object_name, max_convex_hulls):
    return os.path.join(USD_ROOT, "objects", object_name, f"{object_name}_h{max_convex_hulls}.usd")


def _require(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found. Convert the assets first: python scripts/convert_assets.py")
    return path


def build_sim_cfg(task, device, decimation):
    # Solver settings come from the YAML ``sim.physx`` block. The IsaacGym GPU buffer sizes
    # (max_gpu_contact_pairs, default_buffer_size_multiplier) do not translate to PhysX 5 and would allocate
    # several GB; Isaac Lab's defaults are sized for thousands of humanoids.
    physx = task.get("sim", {}).get("physx", {})
    physx_cfg = PhysxCfg(
        solver_type=physx.get("solver_type", 1),
        min_position_iteration_count=1,
        min_velocity_iteration_count=0,
        bounce_threshold_velocity=physx.get("bounce_threshold_velocity", 0.2),
        # convex-decomposed G1 + object: ~100 contact patches per environment
        gpu_max_rigid_patch_count=2**20,
    )
    return SimulationCfg(dt=SIM_DT, render_interval=decimation, device=device, physx=physx_cfg)


def build_robot_cfg(task):
    env = task["env"]
    physx = task.get("sim", {}).get("physx", {})
    position_control = env.get("retargetPositionControl", False)
    actuator = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        # effort mode: the task computes PD torques itself every physics step
        stiffness=per_joint(G1_STIFFNESS) if position_control else 0.0,
        damping=per_joint(G1_DAMPING) if position_control else 0.0,
        armature=per_joint(G1_ARMATURE),
        effort_limit_sim=per_joint(G1_EFFORT),
        friction=0.0,
    )
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_require(robot_usd_path(env["robotType"])),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.0,
                angular_damping=IG_ANGULAR_DAMPING,
                max_linear_velocity=IG_MAX_LINEAR_VELOCITY,
                max_angular_velocity=IG_MAX_ANGULAR_VELOCITY,
                max_depenetration_velocity=physx.get("max_depenetration_velocity", 1.0),
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=physx.get("num_position_iterations", 4),
                solver_velocity_iteration_count=physx.get("num_velocity_iterations", 1),
            ),
            # collision offsets (sim.physx contact_offset / rest_offset) are baked into the USD by
            # scripts/convert_assets.py: the G1 colliders are instanced and cannot be modified here
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.89), joint_pos={".*": 0.0}),
        actuators={"body": actuator},
    )


def build_object_cfg(task, object_names):
    env = task["env"]
    physx = task.get("sim", {}).get("physx", {})
    max_convex_hulls = 5 if env.get("retargetPositionControl", False) else 10
    scale = float(env.get("ballSize", 1.0))
    assets = [
        sim_utils.UsdFileCfg(usd_path=_require(object_usd_path(name, max_convex_hulls)), scale=(scale, scale, scale))
        for name in object_names
    ]
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=assets,
            # environment i gets object i % len(object_names), as in IsaacGym
            random_choice=False,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.0,
                angular_damping=IG_ANGULAR_DAMPING,
                max_linear_velocity=IG_MAX_LINEAR_VELOCITY,
                max_angular_velocity=IG_MAX_ANGULAR_VELOCITY,
                max_depenetration_velocity=physx.get("max_depenetration_velocity", 1.0),
                solver_position_iteration_count=physx.get("num_position_iterations", 4),
                solver_velocity_iteration_count=physx.get("num_velocity_iterations", 1),
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=float(env["objectDensity"])),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=physx.get("contact_offset", 0.02), rest_offset=physx.get("rest_offset", 0.0)
            ),
        ),
        # away from the robot until the first reset places it (overlapping start poses flood the contact buffers)
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.5)),
    )


def make_env_cfg(task, device="cuda:0"):
    """Create the :class:`UltraEnvCfg` of a task from its YAML dict.

    The scene and the observation/action sizes depend on the loaded motions, so the environment completes the
    config with :func:`finalize_env_cfg` before the simulation is created.
    """
    decimation = int(task["env"].get("controlFrequencyInv", 1))
    seed = task.get("seed", -1)
    return UltraEnvCfg(
        task=task,
        decimation=decimation,
        sim=build_sim_cfg(task, device, decimation),
        seed=None if seed is None or seed == -1 else int(seed),
    )


def finalize_env_cfg(cfg: UltraEnvCfg, object_names, num_obs, num_actions):
    task = cfg.task
    env = task["env"]
    plane = env["plane"]
    cfg.object_names = list(object_names)
    cfg.observation_space = int(num_obs)
    cfg.action_space = int(num_actions)
    cfg.scene = UltraSceneCfg(
        num_envs=int(env["numEnvs"]),
        env_spacing=float(env["envSpacing"]),
        # heterogeneous objects per environment
        replicate_physics=False,
        robot=build_robot_cfg(task),
        object=build_object_cfg(task, object_names),
        robot_contact=ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", update_period=0.0),
        object_contact=ContactSensorCfg(prim_path="/World/envs/env_.*/Object", update_period=0.0),
        terrain=TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=plane["staticFriction"],
                dynamic_friction=plane["dynamicFriction"],
                restitution=plane["restitution"],
            ),
        ),
    )
    return cfg
