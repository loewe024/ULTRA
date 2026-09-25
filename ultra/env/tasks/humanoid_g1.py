import torch
import os

from isaacgym import gymapi
from isaacgym.torch_utils import *
import torch.nn.functional as F
from utils import torch_utils
from env.tasks.humanoid import *


class Humanoid_G1(Humanoid_SMPLX):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._key_body_ids_gt = to_torch(cfg["env"]["keyIndex"], device="cuda:"+str(device_id), dtype=torch.long)
        self._contact_body_ids_gt = to_torch(cfg["env"]["contactIndex"], device="cuda:"+str(device_id), dtype=torch.long)
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.last_actions = torch.zeros((self.num_envs, 29), device=self.device, dtype=torch.float)
        self.last_dof_vel = torch.zeros((self.num_envs, 29), device=self.device, dtype=torch.float)
        self.actions = torch.zeros((self.num_envs, 29), device=self.device, dtype=torch.float)
        self.torques = torch.zeros((self.num_envs, 29), device=self.device, dtype=torch.float)
        self.last_dof_pos = torch.zeros((self.num_envs, 29), device=self.device, dtype=torch.float)
        self.last_dof_vel = torch.zeros((self.num_envs, 29), device=self.device, dtype=torch.float)
        self.cfg = cfg
        if cfg['smooth_rewards']['scales'] is not None:
            import env.tasks.smooth_rewards as smooth_rewards
            self.smooth_rewards = smooth_rewards.SmoothRewards(self)
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, 2), device=self.device
        ).repeat((self.num_envs, 1))
        return

    def _setup_character_props(self, key_bodies):
        self._dof_obs_size = self.cfg["env"]["numDoF"]
        self._num_actions = self.cfg["env"]["numDoF"]
        self._num_actions_hand = self.cfg["env"]["numDoFHand"]
        self._num_actions_wrist = self.cfg["env"]["numDoFWrist"]
        self._num_obs = self.cfg["env"]["numObs"]
        return

    def _build_termination_heights(self):
        super()._build_termination_heights()
        self._termination_heights_init = 0.5
        self._termination_heights_init = to_torch(self._termination_heights_init, device=self.device)
        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.robot_type

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.max_convex_hulls = 5
        asset_options.vhacd_params.max_num_vertices_per_ch = 16
        asset_options.vhacd_params.resolution = 60000
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT

        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_ankle_roll_link")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_ankle_roll_link")
        
        self.feet_indices = torch.zeros(
            2, dtype=torch.long, device=self.device, requires_grad=False
        )
        penalized_contact_names = []
        penalize_contacts_on = ["shoulder", "elbow", "hip"]
        body_names = self.gym.get_asset_rigid_body_names(humanoid_asset)
        for name in penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        self.penalized_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_asset_rigid_body_index(
                humanoid_asset, penalized_contact_names[i]
            )
        self.feet_indices[0] = left_foot_idx
        self.feet_indices[1] = right_foot_idx
        knee_names = [s for s in body_names if 'knee' in s]
        self.knee_indices = torch.zeros(
            len(knee_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_asset_rigid_body_index(
                humanoid_asset, knee_names[i]
            )
        self.num_humanoid_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_humanoid_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)

        self.torso_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "torso_link")
        
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        max_agg_bodies = self.num_humanoid_bodies + 2
        max_agg_shapes = self.num_humanoid_shapes + 65
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            self._build_env(i, env_ptr, humanoid_asset)

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return
    
    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        # NOTE: The default friction is all set to 1.0
        if self.cfg['domain_rand']['randomize_friction'] and self.cfg['domain_rand']['domain_rand_general']:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg['domain_rand']['friction_range']
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0], friction_range[1], (num_buckets, 1), device="cpu"
                )
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props
    
    def _process_rigid_body_props(self, props, env_id):
        # No need to use tensors as only called upon env creation
        if self.cfg['domain_rand']['randomize_base_mass'] and self.cfg['domain_rand']['domain_rand_general']:
            rng_mass = self.cfg['domain_rand']['added_mass_range']
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1,))
            props[self.torso_idx].mass += rand_mass
        else:
            rand_mass = np.zeros((1,))
        if self.cfg['domain_rand']['randomize_base_com'] and self.cfg['domain_rand']['domain_rand_general']:
            rng_com = self.cfg['domain_rand']['added_com_range']
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3,))
            props[self.torso_idx].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        mass_params = np.concatenate([rand_mass, rand_com])
        return props, mass_params
        
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        asset_file = self.robot_type
        char_h = 0.89

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(humanoid_asset)
        rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, env_id)
        self.gym.set_asset_rigid_shape_properties(humanoid_asset, rigid_shape_props)


        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter, segmentation_id)

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_EFFORT 
            # stiffness = [
            #     150, 150, 
            #     200, 200,
            #     20, 20,
            #     150, 150,
            #     200, 200,
            #     20, 20,
            #     200, 200, 200,
            #     40 ,40 ,40 ,40, 20 ,20 ,20,
            #     40 ,40 ,40 ,40, 20 ,20 ,20,
            # ]
            # damping = [
            #     5, 5, 5, 5,
            #     4, 4,
            #     5, 5, 5, 5,
            #     4, 4,
            #     5, 5, 5,
            #     10, 10, 10, 10, 0.5, 0.5, 0.5,
            #     10, 10, 10, 10, 0.5, 0.5, 0.5,    
            # ]
            # Kp (stiffness) per DOF
            stiffness = [40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
                        40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
                        40.179238, 28.501246, 28.501246,
                        14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327,
                        14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327]

            # Kd (damping) per DOF
            damping = [2.557890, 6.308802, 2.557890, 6.308802, 1.814446, 1.814446,
                    2.557890, 6.308802, 2.557890, 6.308802, 1.814446, 1.814446,
                    2.557890, 1.814446, 1.814446,
                    0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142,
                    0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142]

            # armature per DOF (NOTE: feet + waist roll/pitch are doubled vs ARMATURE_5020)
            armature = [0.010178, 0.025102, 0.010178, 0.025102, 0.007219, 0.007219,
                        0.010178, 0.025102, 0.010178, 0.025102, 0.007219, 0.007219,
                        0.010178, 0.007219, 0.007219,
                        0.003610, 0.003610, 0.003610, 0.003610, 0.003610, 0.004250, 0.004250,
                        0.003610, 0.003610, 0.003610, 0.003610, 0.003610, 0.004250, 0.004250]

            # effort limits per DOF (used for dof_prop["effort"] and for torque_limits)
            effort = [88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
                    88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
                    88.0, 50.0, 50.0,
                    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
                    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0]
            action_scale = [e/k for e, k in zip(effort, stiffness)]
            dof_prop["effort"] = effort
            # dof_armature_29 = [0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597] * 2 + [0.0103] * 3 + [0.003597] * 14       # 8 (original small joints) + 4 (extra wrist DoF)
            dof_prop["armature"] = armature
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)
            body_props = self.gym.get_actor_rigid_body_properties(env_ptr, humanoid_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, env_id)
            self.gym.set_actor_rigid_body_properties(
                env_ptr, humanoid_handle, body_props, recomputeInertia=True
            )

            self.p_gains = torch.tensor(stiffness, device=self.device, dtype=torch.float32)
            self.d_gains = torch.tensor(damping, device=self.device, dtype=torch.float32)
            self.torque_limits = torch.tensor(dof_prop["effort"], device=self.device, dtype=torch.float32) * 0.8
            self.action_scale = torch.tensor(action_scale, device=self.device, dtype=torch.float32)
                    
        # fetch all the data
        shape_props        = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
        body_names         = self.gym.get_actor_rigid_body_names(env_ptr, humanoid_handle)
        body_shape_indices = self.gym.get_actor_rigid_body_shape_indices(env_ptr, humanoid_handle)

        # for each body, modify the filter on every shape in its range
        for body_idx, idx_range in enumerate(body_shape_indices):
            name = body_names[body_idx]
            start, count = idx_range.start, idx_range.count
            # print(name, start, count)
            for si in range(start, start + count):
                sp = shape_props[si]
                if 'right' in name:
                    if 'ankle' in name:
                        sp.filter = 2
                    elif 'knee' in name:
                        sp.filter = 6
                    elif 'hip' in name:
                        sp.filter = 12
                if 'left' in name:
                    if 'ankle' in name:
                        sp.filter = 16
                    elif 'knee' in name:
                        sp.filter = 48
                    elif 'hip' in name:
                        sp.filter = 96
                # print(name, si, sp.filter)

        # write them back
        self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, shape_props)
        self.humanoid_handles.append(humanoid_handle)

        return

    def _build_pd_action_offset_scale(self):
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)
        return

    def _get_humanoid_collision_filter(self):
        return 0

    def _compute_reward(self, actions):
        hist_dof_vel = self._hist_obs[:,160:160+self.num_actions]

        local_vel = (self._curr_obs[:,160:160+self.num_actions] - hist_dof_vel)*self.fps_data
        dof_diffacc = (local_vel.view(-1, self.num_actions)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)).clone()

        hist_obj_vel = self._hist_obs[:,320:323]
        obj_diffacc = (self._curr_obs[:,320:323] - hist_obj_vel)*self.fps_data
        obj_diffacc = obj_diffacc*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)

        hist_obj_rot_vel = self._hist_obs[:,323:326]
        local_vel = (self._curr_obs[:,323:326] - hist_obj_rot_vel)*self.fps_data
        obj_rot_diffacc = local_vel.view(-1, 3)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)
        self.rew_buf[:], ig_reset, contact_reset, kinematic_reset, metric_1, metric_2 = compute_humanoid_reward(
                                                  self._curr_ref_obs,
                                                  self._curr_obs,
                                                  self._contact_forces,
                                                  self._tar_contact_forces,
                                                  self._key_body_ids,
                                                  self.reward_weights,
                                                  dof_diffacc,
                                                  self.object_points[self.object_id[self.data_id]],
                                                  obj_diffacc,
                                                  obj_rot_diffacc,
                                                  self.init_dof,
                                                  self._num_actions_hand + self._num_actions_wrist,
                                                  self._contact_body_ids_gt,
                                                  self._contact_body_ids,
                                                  self.progress_buf,
                                                  self.scaling
                                                  )
        smooth_rewards = self.smooth_rewards._compute_smooth_reward()
        smooth_rewards = torch.exp(smooth_rewards / 10) * (self.progress_buf < 30) + torch.exp(smooth_rewards / 15) * (self.progress_buf >= 30)
        # print(smooth_rewards)
        self.rew_buf[:] *= smooth_rewards
        self.contact_reset = (self.contact_reset + contact_reset) * contact_reset
        self._reset_ig = torch.logical_or(ig_reset, kinematic_reset)

        return
    
    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf, self.obs_buf,
                                                   self._contact_forces,
                                                   self._rigid_body_pos, self.max_episode_length[self.data_id],
                                                   self._enable_early_termination, self._termination_heights, self._termination_heights_init, self._curr_ref_obs, self._curr_obs, self.start_times, self.rollout_length, self._reset_ig, torch.any(self.contact_reset > 10, dim=-1)
                                                   )
        return


    def _compute_observations(self, env_ids=None):
        if (env_ids is None):
            self.obs_buf[:] = torch.cat((self._compute_observations_iter(None, 1), self._compute_observations_iter(None, 16), (self.progress_buf >= 5).float().unsqueeze(1)), dim=-1)

        else:
            self.obs_buf[env_ids] = torch.cat((self._compute_observations_iter(env_ids, 1), self._compute_observations_iter(env_ids, 16), (self.progress_buf[env_ids] >= 5).float().unsqueeze(1)), dim=-1)
        return

    def _compute_humanoid_obs(self, env_ids=None, ref_obs=None, next_ts=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            contact_forces = self._contact_forces
            actions = self.actions
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            torques = self.torques
            last_dof_pos = self.last_dof_pos
            last_dof_vel = self.last_dof_vel
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            contact_forces = self._contact_forces[env_ids]
            actions = self.actions[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            torques = self.torques[env_ids]
            last_dof_pos = self.last_dof_pos[env_ids]
            last_dof_vel = self.last_dof_vel[env_ids]
        
        obs = compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                self._root_height_obs,
                                                contact_forces, self._contact_body_ids, ref_obs, self._key_body_ids, 
                                                self._key_body_ids_gt, self._contact_body_ids_gt, actions, dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel, self.scaling)

        return obs


@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, contact_body_ids, ref_obs, key_body_ids, key_body_ids_gt, contact_body_ids_gt, actions, dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel, scale):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    len_keypos = len(key_body_ids)
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand_2 = heading_rot_expand.repeat((1, len_keypos, 1))
    flat_heading_rot_2 = heading_rot_expand_2.reshape(heading_rot_expand_2.shape[0] * heading_rot_expand_2.shape[1], 
                                               heading_rot_expand_2.shape[2])
    
    heading_rot_expand = heading_rot_expand.repeat((1, len_keypos, 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])

    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
    heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, len_keypos, 1))
    flat_heading_inv_rot = heading_inv_rot_expand.reshape(heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                               heading_inv_rot_expand.shape[2])
    
    _ref_body_pos = ref_obs[:,326:326+len_keypos*3].view(-1, len_keypos, 3)
    _body_pos = body_pos[:, key_body_ids, :]

    diff_global_body_pos = _ref_body_pos * scale - _body_pos
    diff_local_body_pos_flat = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_body_pos.view(-1, 3)).view(-1, len_keypos * 3)

    local_ref_body_pos = _body_pos - root_pos.unsqueeze(1)  # preserves the body position
    local_ref_body_pos = torch_utils.quat_rotate(flat_heading_rot_2, local_ref_body_pos.view(-1, 3)).view(-1, len_keypos * 3)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos[:, key_body_ids, :] - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot[:, key_body_ids, :].reshape(body_rot.shape[0] * len_keypos, body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], len_keypos * flat_local_body_rot_obs.shape[1])
    
    ref_body_rot = ref_obs[:, 326+len_keypos*3+1+52+len_keypos*3: 326+len_keypos*3+1+52+len_keypos*3+52*4].view(-1, 52, 4)
    ref_body_rot_no_hand = ref_body_rot[:, key_body_ids_gt, :]
    body_rot_no_hand = body_rot[:, key_body_ids]

    diff_global_body_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)), body_rot_no_hand.reshape(-1, 4))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(flat_heading_rot, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot)
    diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
    diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot_no_hand.shape[0], body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1])

    local_ref_body_rot = torch_utils.quat_mul(flat_heading_rot, ref_body_rot_no_hand.reshape(-1, 4))
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot_no_hand.shape[0], -1)

    ref_body_vel = ref_obs[:, 326+len_keypos*3+1+52+len_keypos*3+52*4:326+len_keypos*3+1+52+len_keypos*3+52*4+len_keypos*3].view(-1, len_keypos, 3)
    _body_vel = body_vel[:, key_body_ids, :]
    diff_global_vel = ref_body_vel * scale - _body_vel
    diff_local_vel = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, len_keypos * 3)

    ref_body_ang_vel = ref_obs[:, 326+len_keypos*3+1+52+len_keypos*3+52*4+len_keypos*3:326+len_keypos*3+1+52+len_keypos*3+52*4+len_keypos*3+52*3]
    ref_body_ang_vel_no_hand = ref_body_ang_vel.view(-1, 52, 3)[:, key_body_ids_gt]
    body_ang_vel_no_hand = body_ang_vel[:, key_body_ids]
    diff_global_ang_vel = ref_body_ang_vel_no_hand - body_ang_vel_no_hand
    diff_local_ang_vel = torch_utils.quat_rotate(flat_heading_rot, diff_global_ang_vel.view(-1, 3)).view(-1, len_keypos * 3)

    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel[:, key_body_ids, :].reshape(body_vel.shape[0] * len_keypos, body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], len_keypos * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel[:, key_body_ids, :].reshape(body_ang_vel.shape[0] * len_keypos, body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], len_keypos * body_ang_vel.shape[2])

    body_contact_buf = contact_forces[:, contact_body_ids, :].clone() #.view(contact_forces.shape[0],-1)
    contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()
    ref_body_contact = ref_obs[:,326+len_keypos*3+1:326+len_keypos*3+1+52][:, contact_body_ids_gt]
    diff_body_contact = ref_body_contact * ((ref_body_contact + 1) / 2 - contact)
    # print(actions.shape, root_h_obs.shape, local_body_pos.shape, local_body_rot_obs.shape, local_body_vel.shape, local_body_ang_vel.shape, contact.shape, diff_local_body_pos_flat.shape, diff_local_body_rot_obs.shape, diff_body_contact.shape, local_ref_body_pos.shape, local_ref_body_rot.shape, diff_local_vel.shape, diff_local_ang_vel.shape)
    # local_ref_body_rot[:, :29] = actions
    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, contact, diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact, local_ref_body_pos, local_ref_body_rot, diff_local_vel, diff_local_ang_vel, actions, dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel), dim=-1)
    return obs

def huber_loss(diff, sigma=1.0):
    beta = 1. / (sigma ** 2)
    diff = torch.abs(diff)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss

def compute_humanoid_reward(hoi_ref, hoi_obs, contact_buf, tar_contact_forces, key_body_ids, w, actions, object_points, object_pos_action, object_rot_action, init_dof, num_dof_hand, _contact_body_ids_gt, _contact_body_ids, t, scale):
    len_keypos = len(key_body_ids)
    root_pos = hoi_obs[:,:3]
    root_rot = hoi_obs[:,3:3+4]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    num_dof = init_dof.shape[0]
    dof_pos = hoi_obs[:,7:7+num_dof]
    dof_vel = hoi_obs[:,7+153:7+153+num_dof]
    rec_dof_pos = init_dof
    
    ## stage I: standing
    reward_dof = torch.exp(-20 * ((dof_pos - rec_dof_pos.unsqueeze(0))*(dof_pos - rec_dof_pos.unsqueeze(0))).mean(dim=-1))
    
    left_hand_dof_pos = dof_pos[:, 19:22]
    right_hand_dof_pos = dof_pos[:, 26:29]
    obj_pos = hoi_obs[:,313:313+3]
    obj_rot = hoi_obs[:,316:316+4]

    local_obj_pos = obj_pos - root_pos
    local_obj_pos[..., -1] = obj_pos[..., -1]
    local_obj_pos = quat_rotate(heading_rot, local_obj_pos)

    local_obj_rot = quat_mul(heading_rot, obj_rot)

    obj_rot_extend = obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    object_points_extend = object_points.view(-1, 3)
    obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + obj_pos.unsqueeze(1)

    obj_pos_vel = hoi_obs[:,320:320+3]
    obj_rot_vel = hoi_obs[:,323:323+3]
    key_pos = hoi_obs[:,326:326+len_keypos*3].view(-1, len_keypos, 3)
    body_pos_vel = hoi_obs[:, 326+len_keypos*3+1+52+len_keypos*3+52*4:326+len_keypos*3+1+52+len_keypos*3+52*4+len_keypos*3] #.view(-1, len_keypos, 3)


    #   keyBodies: ['left_hip_yaw_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_roll_link', 
    #               'torso_link', 'mid360_link', 
    #               'left_shoulder_yaw_link', 'left_elbow_link', 'left_hand_palm_link', 
    #               'right_shoulder_yaw_link', 'right_elbow_link', 'right_hand_palm_link', ]
    #   keyIndex: [ 1,  2,  3,  5,  6,  7,  
    #               9, 13, 
    #               15,  16,  17,  
    #               34, 35, 36, ]
  
    ## calculate link vectors
    rot_id = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], 
              [7, 8], [8, 9],
              [7, 11], [11, 12]]
    local_key_rot = [(key_pos[:, idx[0]] - key_pos[:, idx[1]]).unsqueeze(1) for idx in rot_id]

    local_key_rot = torch.cat(local_key_rot, dim=1)
    local_key_rot = (local_key_rot / (local_key_rot.norm(dim=-1, keepdim=True) + 1e-5))
    left_hand_key_pos = key_pos[:, [10], :]
    right_hand_key_pos = key_pos[:, [13], :]

    left_hand_ig = (left_hand_key_pos.unsqueeze(2) - obj_points.unsqueeze(1)).norm(dim=-1).min(dim=-1)[0]
    right_hand_ig = (right_hand_key_pos.unsqueeze(2) - obj_points.unsqueeze(1)).norm(dim=-1).min(dim=-1)[0]

    local_key_pos = key_pos[:, [2, 5], :] ## feet
    ig = key_pos[:, [10, 13], :].unsqueeze(2) - obj_points.unsqueeze(1)

    ref_root_pos = hoi_ref[:,:3]
    ref_root_rot = hoi_ref[:,3:3+4]

    ref_heading_rot = torch_utils.calc_heading_quat_inv(ref_root_rot)

    ref_obj_pos = hoi_ref[:,313:313+3]
    ref_obj_rot = hoi_ref[:,316:316+4]

    ref_body_pos_vel = hoi_ref[:, 326+len_keypos*3+1+52+len_keypos*3+52*4:326+len_keypos*3+1+52+len_keypos*3+52*4+len_keypos*3] #.view(-1, len_keypos, 3)
    
    ref_local_obj_pos = ref_obj_pos - ref_root_pos
    ref_local_obj_pos[..., -1] = ref_obj_pos[..., -1]
    ref_local_obj_pos = quat_rotate(ref_heading_rot, ref_local_obj_pos)

    ref_local_obj_rot = quat_mul(ref_heading_rot, ref_obj_rot)


    ref_obj_rot_extend = ref_obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    ref_obj_points = torch_utils.quat_rotate(ref_obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + ref_obj_pos.unsqueeze(1)

    ref_obj_pos_vel = hoi_ref[:,320:320+3]
    ref_obj_rot_vel = hoi_ref[:,323:323+3]
    ref_key_pos = hoi_ref[:,326:326+len_keypos*3].view(-1, len_keypos, 3)
    ref_local_key_rot = [(ref_key_pos[:, idx[0]] - ref_key_pos[:, idx[1]]).unsqueeze(1) for idx in rot_id]
    ref_local_key_rot = torch.cat(ref_local_key_rot, dim=1)

    ref_local_key_rot = (ref_local_key_rot / (ref_local_key_rot.norm(dim=-1, keepdim=True) + 1e-5))

    contact_buf = contact_buf[:, _contact_body_ids]
    ref_human_contact = hoi_ref[:,326+len_keypos*3+1:326+len_keypos*3+1+52][:, _contact_body_ids_gt]

    ref_local_key_pos = ref_key_pos[:, [2, 5], :]

    ref_ig = ref_key_pos[:, [10, 13], :].unsqueeze(2) - ref_obj_points.unsqueeze(1)

    human_reset = (ref_key_pos[:, [2, 5, 10, 13], :] * scale - key_pos[:, [2, 5, 10, 13], :]).view(-1, 4, 3)[:, :2, :].norm(dim=-1).mean(dim=-1) > 0.5
    object_reset = (obj_points[:, :2] - ref_obj_points[:, :2] * scale).norm(dim=-1).mean(dim=-1) > 0.5
    kinematic_reset = torch.logical_or(human_reset, object_reset)
    
    w_ig = 5

    weight_1 = (1 / torch.clamp((ig**2).sum(dim=-1), min=0.2))
    weight_1 = weight_1 / weight_1.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
    weight_2 = (1 / torch.clamp((ref_ig**2).sum(dim=-1), min=0.2))
    weight_2 = weight_2 / weight_2.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

    eig = ((ig - ref_ig)**2).sum(dim=-1) * (weight_1 + weight_2)  

    rig = torch.exp(-w_ig * (eig.sum(dim=-1).sum(dim=-1) * 0.5))
    
    reset_ig_1 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp(((ref_ig)**2).sum(dim=-1).sqrt(), min=0.3)).max(dim=-1)[0].max(dim=-1)[0] > 1.0
    reset_ig_2 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ig**2).sum(dim=-1).sqrt(), min=0.3)).max(dim=-1)[0].max(dim=-1)[0] > 1.0
    reset_ig = torch.logical_or(reset_ig_1, reset_ig_2)
    
    ep = torch.mean(((ref_local_key_pos * scale - local_key_pos)**2).sum(dim=-1),dim=-1)
    rp = torch.exp(-ep*w['p'])

    er = torch.mean(((ref_local_key_rot - local_key_rot)**2).sum(dim=-1),dim=-1)
    rr = torch.exp(-er*w['r'])
    
    # body pos vel reward
    epv = torch.mean((ref_body_pos_vel * scale - body_pos_vel)**2,dim=-1)
    # epv = torch.mean(pos_vel ,dim=-1) # torch.zeros_like(ep)
    rpv = torch.exp(-epv*w['pv'])

    energy = actions.pow(2).mean(dim=-1).mul(-w['eg1']).exp()

    rb = rp*rr*rpv*energy

    # object pos reward
    eop = torch.mean(((ref_local_obj_pos - local_obj_pos)**2),dim=-1)
    rop = torch.exp(-eop*0)

    # object rot reward
    diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_local_obj_rot), local_obj_rot)
    diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
    diff = diff_angle.view(-1, 1)
    
    eor = torch.mean(huber_loss(diff, 3),dim=-1)
    ror = torch.exp(-eor*w['or'])

    # object pos vel reward
    eopv = torch.mean((ref_obj_pos_vel * scale - obj_pos_vel)**2,dim=-1)
    ropv = torch.exp(-eopv*w['opv'])

    # object rot vel reward
    eorv = torch.mean((ref_obj_rot_vel - obj_rot_vel)**2,dim=-1)
    rorv = torch.exp(-eorv*w['orv'])
    obj_energy = (object_pos_action.pow(2).mean(dim=-1).mul(-w['eg2']).exp()) * (object_rot_action.pow(2).mean(dim=-1).mul(-w['eg2']).exp())
    ro = rop*ror*ropv*rorv*obj_energy


    contact_thres = 0.1
    
    er_left_hand = left_hand_dof_pos.abs().sum(dim=-1)
    rr_left_hand = torch.exp(-er_left_hand * 0.5)

    rr_left_hand_ig = torch.exp(-F.relu(left_hand_ig.abs() - 0.02).sum(dim=-1))

    left_contact_hand_ids = [10]
    ref_left_contact_hand = ref_human_contact[:, left_contact_hand_ids]
    ref_left_contact_hand_any = torch.any(ref_left_contact_hand > contact_thres, dim=-1).float()
    left_hand_contact_buf = contact_buf[:, left_contact_hand_ids, :].clone()
    left_hand_contact = torch.any(torch.abs(left_hand_contact_buf) > contact_thres, dim=-1).float()
    left_hand_contact_any = torch.any(left_hand_contact > contact_thres, dim=-1, keepdim=True).float()
    
    w_cg_left = 0.5
    ecg_left = (((ref_left_contact_hand > contact_thres) * torch.abs(left_hand_contact - ref_left_contact_hand)).sum(dim=-1))
    rcg_left = 0.5 * (rr_left_hand + torch.exp(-w_cg_left * ecg_left)) * (ref_left_contact_hand_any) + (1 - ref_left_contact_hand_any) * rr_left_hand

    er_right_hand = right_hand_dof_pos.abs().sum(dim=-1)
    rr_right_hand = torch.exp(-er_right_hand * 0.5)

    rr_right_hand_ig = torch.exp(-F.relu(right_hand_ig.abs() - 0.02).sum(dim=-1))

    right_contact_hand_ids = [13]
    ref_right_contact_hand = ref_human_contact[:, right_contact_hand_ids]
    ref_right_contact_hand_any = torch.any(ref_right_contact_hand > contact_thres, dim=-1).float()
    right_hand_contact_buf = contact_buf[:, right_contact_hand_ids, :].clone()
    right_hand_contact = torch.any(torch.abs(right_hand_contact_buf) > contact_thres, dim=-1).float()
    right_hand_contact_any = torch.any(right_hand_contact > contact_thres, dim=-1, keepdim=True).float()

    w_cg_right = 0.5
    ecg_right = (((ref_right_contact_hand > contact_thres) * torch.abs(right_hand_contact - ref_right_contact_hand)).sum(dim=-1))
    rcg_right = 0.5 * (rr_right_hand + torch.exp(-w_cg_right * ecg_right)) * (ref_right_contact_hand_any) + (1 - ref_right_contact_hand_any) * rr_right_hand

    rcg3 = rcg_left * rcg_right

    # ref_other_contact = ref_human_contact
    # other_contact_buf = contact_buf.clone()
    # other_contact = torch.any(torch.abs(other_contact_buf) > contact_thres, dim=-1).float()
    # w_4 = torch.ones_like(other_contact).to(other_contact.device).float()
    # w_cg4 = 1.0
    # ecg4 = ((torch.abs(other_contact - ref_other_contact) * (ref_other_contact > contact_thres))).mean(dim=-1)
    # rcg4 = torch.exp(-ecg4*w_cg4)
    
    ## stage II
    reward_dof_2 = torch.exp(-1 * ((dof_pos - rec_dof_pos.unsqueeze(0))*(dof_pos - rec_dof_pos.unsqueeze(0)))[..., [1, 2, 4, 5, 7, 8, 10, 11, 19, 20, 21, 26, 27, 28]].mean(dim=-1))
    reward_dof_vel_2 = torch.exp(-0.0001 * ((dof_vel)*(dof_vel)).mean(dim=-1))

    contact_reset = torch.cat([ 
                               torch.abs(ref_left_contact_hand_any.unsqueeze(-1) - left_hand_contact_any) * ref_left_contact_hand_any.unsqueeze(-1), 
                               torch.abs(ref_right_contact_hand_any.unsqueeze(-1) - right_hand_contact_any) * ref_right_contact_hand_any.unsqueeze(-1),
                               ], dim=-1)
    # w_cg5 = 0.5
    # ref_all_contact = ref_human_contact[:, [2, 5]]
    # all_contact_buf = contact_buf[:, [2, 5], :].clone()
    # no_contact = torch.all(torch.abs(all_contact_buf) < contact_thres, dim=-1).float()

    # ecg5 = (torch.abs(no_contact + ref_all_contact) * (ref_all_contact < -contact_thres)).mean(dim=-1)
    # rcg5 = torch.exp(-ecg5*w_cg5)

    energy_ids = [_ for _ in range(14) if _ != 2 and _ != 5] # not counting the contact of feet
    contact_all = contact_buf[:,energy_ids, :].clone().abs().sum(dim=-1).max(dim=-1)[0]
    contact_energy = contact_all.pow(2).mul(-w['eg3']).exp()

    rcg = rcg3*contact_energy

    # I: standing; II: approaching the first state; III: tracking
    # w1 ~ 1 for t≪5, →0 for t≫5
    w1 = torch.sigmoid((10.0 - t) / 5)
    # w3 ~ 0 until t≫15, →1 beyond
    w3 = torch.sigmoid((t - 20.0) / 2)
    # middle weight is whatever’s left
    w2 = 1.0 - w1 - w3
    
    reward = reward_dof*contact_energy*energy*obj_energy * w1 + (rb*ro*rig) * w2 + rb*ro*rig*rcg*w3

    metric_1 = (ref_key_pos - key_pos)[:, [2,5], :].norm(dim=-1).mean(dim=-1)
    metric_2 = (obj_points - ref_obj_points).norm(dim=-1).mean(dim=-1)
    return reward, reset_ig, contact_reset, kinematic_reset, metric_1, metric_2

def compute_humanoid_reset(reset_buf, progress_buf, obs_buf, contact_buf, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, termination_heights_init, hoi_ref, hoi_obs, start_times, rollout_length, reset_ig, contact_reset):
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        body_height = rigid_body_pos[:, 0, 2] # root height

        body_fall = body_height < termination_heights# [4096] 
        has_failed = body_fall.clone()
        has_failed *= (progress_buf > 1)

        body_fail_init = body_height < termination_heights_init
        has_failed_init = torch.logical_and(body_fail_init, progress_buf < 10)
        has_failed = torch.logical_or(has_failed, has_failed_init)
        reset_ig *= (progress_buf > 30 + start_times)
        contact_reset *= (progress_buf > 30 + start_times)
        invalid_obs = ~torch.isfinite(obs_buf)
        invalid_batches = torch.any(invalid_obs, dim=1)
        if torch.any(invalid_obs):
            raise Exception("invalid observation")
        terminated = torch.where(torch.logical_or(invalid_batches, torch.logical_or(has_failed, torch.logical_or(reset_ig, contact_reset))), torch.ones_like(reset_buf), terminated)
    reset = torch.where(torch.logical_or(progress_buf >= max_episode_length-1, progress_buf - start_times >= rollout_length-1), torch.ones_like(reset_buf), terminated)

    return reset, terminated
