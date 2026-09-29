"""Isaac Lab base environment of the ULTRA tasks.

:class:`UltraBaseEnv` owns the Isaac Lab scene (G1 articulation, one object per environment, contact sensors,
ground plane) and exposes the simulation state in ULTRA's legacy layout, which the task, reward and observation
code as well as the datasets and policies are written for:

* ``_root_states`` ``[num_envs * 2, 13]``: per environment the humanoid root, then the object root
  (``pos, quat (x, y, z, w), lin_vel, ang_vel``), positions relative to the environment origin.
* ``_dof_state`` ``[num_envs * 29, 2]``: joint position and velocity in ``LEGACY_DOF_NAMES`` order.
* ``_rigid_body_state`` ``[num_envs * 40, 13]``: the 39 G1 bodies in ``LEGACY_BODY_NAMES`` order, then the object.
* ``_contact_force_state`` ``[num_envs * 40, 3]``: net contact force per body, same layout.

Tasks modify these tensors in place and push them with :meth:`_set_actor_root_state_indexed` /
:meth:`_set_dof_state_indexed`, which take the legacy actor ids (``env_id * 2`` for the humanoid, ``+ 1`` for the
object). All permutation and quaternion conversion happens in this class.

Resets are not automatic: the rl-games agents reset finished environments before the next step so that the
terminal observation of a time-out can still be used for value bootstrapping.
"""

import sys

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv

from isaac.legacy_layout import LEGACY_BODY_NAMES, LEGACY_DOF_NAMES, legacy_to_sim_index, wxyz_to_xyzw, xyzw_to_wxyz
from isaac.scene_cfg import UltraEnvCfg, finalize_env_cfg
from isaac.viewer import UltraViewer

NUM_ACTORS = 2  # humanoid, object


class UltraBaseEnv(DirectRLEnv):
    cfg: UltraEnvCfg
    # Per-motion episode lengths set by the tasks; shadows DirectRLEnv's read-only property.
    max_episode_length = None

    def __init__(self, cfg: UltraEnvCfg, render_mode=None, **kwargs):
        self.headless = cfg.task.get("headless", True)
        finalize_env_cfg(cfg, self._scene_object_names(), self.get_obs_size(), self.get_action_size())
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.object = self.scene["object"]
        self._robot_contact = self.scene["robot_contact"]
        self._object_contact = self.scene["object_contact"]
        self._build_index_maps()
        self._allocate_state_tensors()
        self._allocate_task_buffers()
        self._check_object_assignment()

        self.viewer = None
        if not self.headless or cfg.task["env"].get("saveImages", False):
            self.viewer = UltraViewer(self)

        self._setup_env_properties()
        self._refresh_sim_tensors()

    # -- hooks for the tasks ------------------------------------------------------------------------------------------
    def _scene_object_names(self):
        """Object (asset) names; environment ``i`` spawns ``names[i % len(names)]``."""
        return self.object_name

    def _setup_env_properties(self):
        """Called once the simulation runs: derive asset properties and apply start-up randomization."""
        pass

    def get_obs_size(self):
        raise NotImplementedError

    def get_action_size(self):
        raise NotImplementedError

    def pre_physics_step(self, actions):
        raise NotImplementedError

    def _physics_step(self):
        raise NotImplementedError

    def post_physics_step(self):
        raise NotImplementedError

    # -- DirectRLEnv --------------------------------------------------------------------------------------------------
    def _setup_scene(self):
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def step(self, actions):
        """One policy step. Unlike :meth:`DirectRLEnv.step`, finished environments are *not* reset here."""
        self.pre_physics_step(actions)
        self._physics_step()
        self.post_physics_step()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # The DirectRLEnv hooks below are bypassed by :meth:`step`.
    def _pre_physics_step(self, actions):
        raise NotImplementedError

    def _apply_action(self):
        raise NotImplementedError

    def _get_observations(self):
        raise NotImplementedError

    def _get_rewards(self):
        raise NotImplementedError

    def _get_dones(self):
        raise NotImplementedError

    # -- setup --------------------------------------------------------------------------------------------------------
    def _build_index_maps(self):
        device = self.device
        self._dof_sim_ids = legacy_to_sim_index(self.robot.joint_names, LEGACY_DOF_NAMES).to(device)
        self._body_sim_ids = legacy_to_sim_index(self.robot.body_names, LEGACY_BODY_NAMES).to(device)
        self._contact_sim_ids = legacy_to_sim_index(self._robot_contact.body_names, LEGACY_BODY_NAMES).to(device)
        self.num_bodies = len(LEGACY_BODY_NAMES)
        self.num_dof = len(LEGACY_DOF_NAMES)
        if self.robot.num_bodies != self.num_bodies or self.robot.num_joints != self.num_dof:
            raise RuntimeError(
                f"G1 USD has {self.robot.num_bodies} bodies / {self.robot.num_joints} joints, expected"
                f" {self.num_bodies} / {self.num_dof}. Re-run scripts/convert_assets.py --force."
            )

    def _allocate_state_tensors(self):
        n, device = self.num_envs, self.device
        self._root_states = torch.zeros(n * NUM_ACTORS, 13, device=device)
        self._dof_state = torch.zeros(n * self.num_dof, 2, device=device)
        self._rigid_body_state = torch.zeros(n * (self.num_bodies + 1), 13, device=device)
        self._contact_force_state = torch.zeros(n * (self.num_bodies + 1), 3, device=device)
        self.dof_force_tensor = torch.zeros(n, self.num_dof, device=device)
        self._all_env_ids_cpu = torch.arange(n, dtype=torch.long)

    def _allocate_task_buffers(self):
        """The per-environment buffers of the former IsaacGym ``BaseTask``."""
        n, device = self.num_envs, self.device
        self.num_obs = self.cfg.observation_space
        self.num_states = 0
        self.num_actions = self.cfg.action_space
        self.control_freq_inv = self.cfg.decimation
        self.obs_buf = torch.zeros(n, self.num_obs, device=device, dtype=torch.float)
        self.states_buf = torch.zeros(n, self.num_states, device=device, dtype=torch.float)
        self.rew_buf = torch.zeros(n, device=device, dtype=torch.float)
        self.reset_buf = torch.ones(n, device=device, dtype=torch.long)
        self.progress_buf = torch.zeros(n, device=device, dtype=torch.long)
        self.start_times = torch.zeros(n, device=device, dtype=torch.long)
        self.randomize_buf = torch.zeros(n, device=device, dtype=torch.long)
        self.data_id = torch.zeros(n, device=device, dtype=torch.long)
        self.contact_reset = torch.zeros(n, 2, device=device, dtype=torch.float)
        self.extras = {}

    def _check_object_assignment(self):
        """Tasks assume environment ``i`` holds object ``i % len(object_names)`` (as in IsaacGym)."""
        names = self.cfg.object_names
        for env_id, prim_path in enumerate(self.object.root_physx_view.prim_paths[: self.num_envs]):
            expected = f"/World/envs/env_{env_id}/Object"
            if not prim_path.startswith(expected):
                raise RuntimeError(f"Object view order differs from environment order: {prim_path} != {expected}")
        self.env_object_ids = torch.arange(self.num_envs, device=self.device) % len(names)

    # -- state exchange -----------------------------------------------------------------------------------------------
    def _refresh_sim_tensors(self):
        """Copy the simulation state into the legacy tensors (in place, so views stay valid)."""
        n, nb = self.num_envs, self.num_bodies
        origins = self.scene.env_origins

        roots = self._root_states.view(n, NUM_ACTORS, 13)
        self._write_legacy_state(roots[:, 0], self.robot.data.root_state_w, origins)
        self._write_legacy_state(roots[:, 1], self.object.data.root_state_w, origins)

        dofs = self._dof_state.view(n, self.num_dof, 2)
        dofs[..., 0] = self.robot.data.joint_pos[:, self._dof_sim_ids]
        dofs[..., 1] = self.robot.data.joint_vel[:, self._dof_sim_ids]
        self.dof_force_tensor[:] = self.robot.data.applied_torque[:, self._dof_sim_ids]

        bodies = self._rigid_body_state.view(n, nb + 1, 13)
        self._write_legacy_state(bodies[:, :nb], self.robot.data.body_state_w[:, self._body_sim_ids], origins[:, None])
        bodies[:, nb] = roots[:, 1]

        contacts = self._contact_force_state.view(n, nb + 1, 3)
        contacts[:, :nb] = self._robot_contact.data.net_forces_w[:, self._contact_sim_ids]
        contacts[:, nb] = self._object_contact.data.net_forces_w[:, 0]

    @staticmethod
    def _write_legacy_state(dst, src, origins):
        dst[..., 0:3] = src[..., 0:3] - origins
        dst[..., 3:7] = wxyz_to_xyzw(src[..., 3:7])
        dst[..., 7:13] = src[..., 7:13]

    def _sim_state(self, legacy_states, env_ids):
        state = legacy_states.clone()
        state[:, 0:3] += self.scene.env_origins[env_ids]
        state[:, 3:7] = xyzw_to_wxyz(state[:, 3:7])
        return state

    def _set_actor_root_state_indexed(self, actor_ids):
        """Push ``_root_states`` of the given legacy actor ids to the simulation."""
        actor_ids = actor_ids.long()
        env_ids, kinds = actor_ids // NUM_ACTORS, actor_ids % NUM_ACTORS
        roots = self._root_states.view(self.num_envs, NUM_ACTORS, 13)
        for kind, asset in ((0, self.robot), (1, self.object)):
            ids = env_ids[kinds == kind]
            if len(ids) > 0:
                asset.write_root_state_to_sim(self._sim_state(roots[ids, kind], ids), env_ids=ids)

    def _set_actor_root_state_all(self):
        n = self.num_envs
        self._set_actor_root_state_indexed(torch.arange(n * NUM_ACTORS, device=self.device))

    def _set_dof_state_indexed(self, actor_ids):
        """Push ``_dof_pos`` / ``_dof_vel`` of the humanoids behind the given legacy actor ids."""
        env_ids = actor_ids.long() // NUM_ACTORS
        dofs = self._dof_state.view(self.num_envs, self.num_dof, 2)[env_ids]
        pos = torch.empty_like(dofs[..., 0])
        vel = torch.empty_like(dofs[..., 1])
        pos[:, self._dof_sim_ids] = dofs[..., 0]
        vel[:, self._dof_sim_ids] = dofs[..., 1]
        self.robot.write_joint_state_to_sim(pos, vel, env_ids=env_ids)

    def _to_sim_dofs(self, legacy):
        sim = torch.empty_like(legacy)
        sim[:, self._dof_sim_ids] = legacy
        return sim

    def _apply_dof_efforts(self, torques):
        self.robot.set_joint_effort_target(self._to_sim_dofs(torques))

    def _apply_dof_position_targets(self, targets):
        self.robot.set_joint_position_target(self._to_sim_dofs(targets))

    def _simulate(self):
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

    # -- physics properties -------------------------------------------------------------------------------------------
    def _dof_limits(self):
        """Joint position limits ``(lower, upper)`` in legacy order."""
        limits = self.robot.data.joint_pos_limits[0, self._dof_sim_ids]
        return limits[:, 0].clone(), limits[:, 1].clone()

    def _set_shape_materials(self, asset, friction, restitution=None, env_ids=None):
        """Set static and dynamic friction (IsaacGym has a single coefficient) of all shapes of ``asset``.

        ``friction`` / ``restitution``: scalar or per-environment tensor.
        """
        env_ids = self._all_env_ids_cpu if env_ids is None else env_ids.cpu()
        materials = asset.root_physx_view.get_material_properties()
        friction = torch.as_tensor(friction, dtype=materials.dtype).reshape(-1, 1).expand(len(env_ids), -1)
        materials[env_ids, :, 0] = friction
        materials[env_ids, :, 1] = friction
        if restitution is not None:
            restitution = torch.as_tensor(restitution, dtype=materials.dtype).reshape(-1, 1).expand(len(env_ids), -1)
            materials[env_ids, :, 2] = restitution
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def _set_object_color(self, env_id, rgb):
        """Replacement for ``gym.set_rigid_body_color`` on the object (visual only)."""
        if not hasattr(self, "_color_materials"):
            self._color_materials = {}
        key = tuple(float(c) for c in rgb)
        path = self._color_materials.get(key)
        if path is None:
            path = f"/World/Looks/UltraColor{len(self._color_materials)}"
            material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=key)
            material_cfg.func(path, material_cfg)
            self._color_materials[key] = path
        sim_utils.bind_visual_material(f"/World/envs/env_{int(env_id)}/Object", path)

    def _set_gravity(self, gravity):
        import carb

        self.sim.physics_sim_view.set_gravity(carb.Float3(*[float(g) for g in gravity]))

    # -- viewer -------------------------------------------------------------------------------------------------------
    def render(self, sync_frame_time=False):
        if self.viewer is None:
            return
        import omni.kit.app

        if not omni.kit.app.get_app().is_running():
            sys.exit()
        self.sim.render()

    def close(self):
        super().close()
