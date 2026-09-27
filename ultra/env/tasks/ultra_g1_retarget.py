import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import *

from utils import torch_utils
import torch.nn.functional as F
from env.tasks.humanoid_g1 import *
from env.tasks.ultra import Ultra

_DEFAULT_NOISE_STD = {
    "root_pos"     : 0.05,  # m
    "root_rot"     : 0.04,  # rad (axis–angle magnitude)
    "dof_pos"      : 0.15,  # rad or m
    "root_vel"     : 0.05,  # m/s
    "root_ang_vel" : 0.05,  # rad/s
    "dof_vel"      : 0.05,  # rad/s or m/s
    "target_pos"   : 0.04,  # m
    "target_rot"   : 0.08,  # rad
}
def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:, 0]
    y = quat_angle[:, 1]
    z = quat_angle[:, 2]
    w = quat_angle[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians
def _randn_like(x, std):
    """Add zero-mean Gaussian noise with per-element std."""
    if std == 0:
        return x
    return x + torch.randn_like(x) * std


def _perturb_quat(q, std):
    """Apply small-angle axis-angle noise then renormalise."""
    if std == 0:
        return q
    # random unit axes
    axis = torch.randn_like(q[..., :3])
    axis = axis / torch.norm(axis, dim=-1, keepdim=True).clamp_(min=1e-6)
    angle = torch.randn(q.shape[0], 1, device=q.device) * std  # radians
    half  = 0.5 * angle
    delta = torch.cat([axis * torch.sin(half), torch.cos(half)], dim=-1)
    qn    = quat_mul(delta, q)
    return qn / torch.norm(qn, dim=-1, keepdim=True)

class UltraG1Retarget(Humanoid_G1, Ultra):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.init_dof = torch.cat([to_torch([-0.1, 0, 0.0, 0.3, -0.2, 0, -0.1, 0, 0.0, 0.3, -0.2, 0, 0, 0, 0, 
                                             0, 0, 0, 0.5], device=self.device, dtype=torch.float),
                                   to_torch([0] * (self._num_actions_hand + self._num_actions_wrist), device=self.device, dtype=torch.float),
                                   to_torch([0, 0, 0, 0.5], device=self.device, dtype=torch.float),
                                   to_torch([0] * (self._num_actions_hand + self._num_actions_wrist), device=self.device, dtype=torch.float)])
        self.ref_hoi_obs_size = 747
        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self.is_distill = self.cfg['env'].get('distillation', False)
        if self.cfg['env']['history_len'] > 0:
            self.obs_history_buf = torch.zeros(
                self.num_envs,
                self.cfg['env']['history_len'],
                self.cfg['env']['numObsProprio'],
                device=self.device,
                dtype=torch.float,
            )
        self.headless = headless
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.long_term_t = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)

        # Track which environments should "stand still" (10% of environments)
        # These environments will have a fixed reference frame to train the policy to stand still
        self.stand_still_ratio = 0.01 if cfg['domain_rand']['domain_rand_general'] else 0.0
        self.is_stand_still = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.stand_still_frame = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Track steps since last push for each environment
        # Used to temporarily disable tracking termination after pushes
        self.steps_since_push = torch.full((self.num_envs,), 999, device=self.device, dtype=torch.long)
        # Observation masking configuration
        self.obs_task_keep_prob = cfg['env'].get('obs_task_keep_prob', 0.3)  # Probability to keep task_obs
        self.obs_ig_keep_prob = cfg['env'].get('obs_ig_keep_prob', 0.1)  # Probability to keep IG features

        # Per-environment flags for which observations to keep (set at episode start)
        self.keep_task_obs_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.keep_ig_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        return

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        return
    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(166, device=self.device, dtype=torch.float)
        if not self.cfg['noise']['add_noise']:
            return noise_vec
        noise_scales = self.cfg['noise']['noise_scales']
        noise_level = self.cfg['noise']['noise_level']
        # noise_vec[:3] = noise_scales['ang_vel'] * noise_level
        # noise_vec[3:5] = noise_scales['imu'] * noise_level
        # noise_vec[5 : 5 + self._num_actions] = (
        #     noise_scales['dof_pos'] * noise_level
        # )
        # noise_vec[5 + self._num_actions : 5 + self._num_actions * 2] = (
        #     noise_scales['dof_vel'] * noise_level
        # )

        noise_vec[61: 64] = noise_scales['root_pos'] * noise_level
        noise_vec[67: 70] = noise_scales['ang_vel'] * noise_level
        noise_vec[73: 76] = noise_scales['imu'] * noise_level
        noise_vec[79: 108] = (
            noise_scales['dof_pos'] * noise_level
        )
        noise_vec[108: 137] = (
            noise_scales['dof_vel'] * noise_level
        )


        return noise_vec
    
        # obs_buf = torch.cat(
        #     (
        #         root_ang_vel, 
        #         ref_root_ang_vel, 
        #         diff_root_ang_vel,
        #         imu_obs,  # 2 dims
        #         imu_obs_ref,
        #         diff_imu_obs_ref,
        #         self.reindex((dof_pos)),
        #         self.reindex(dof_vel),
        #         actions[:, -1, :],
        #     ),
        #     dim=-1,
        # )

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        return

    def _load_motion(self, motion_file):
        self.hoi_data_dict = []
        hoi_datas = []
        hoi_refs = []
        if type(motion_file) != type([]):
            motion_file = [motion_file]
        self.max_episode_length = []
        for idx, data_path in enumerate(motion_file):
            loaded_dict = {}
            hoi_data = torch.load(data_path)
            loaded_dict['hoi_data'] = hoi_data.detach().cpu() # .to(self.device)
            last_frame = loaded_dict['hoi_data'][-1:].clone()
            repeated_last_frame = last_frame.repeat(20, 1)  # [20, feature_dim]
            loaded_dict['hoi_data'] = torch.cat([loaded_dict['hoi_data'], repeated_last_frame], dim=0)

            self.max_episode_length.append(loaded_dict['hoi_data'].shape[0])
            self.fps_data = new_fps = 60.

            loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
            loaded_dict['root_pos_vel'] = loaded_dict['hoi_data'][:, 7:10].clone()

            loaded_dict['root_rot'] = loaded_dict['hoi_data'][:, 3:7]

            loaded_dict['root_rot_vel'] = loaded_dict['hoi_data'][:, 10:13]

            loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 13:42]

            loaded_dict['dof_pos_vel'] = loaded_dict['hoi_data'][:, 42:71]

            loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 84:201]
            loaded_dict['body_pos_vel'] = loaded_dict['hoi_data'][:, 357:474]

            loaded_dict['obj_pos'] = loaded_dict['hoi_data'][:, 71:74]

            loaded_dict['obj_pos_vel'] = loaded_dict['hoi_data'][:, 78:81]

            loaded_dict['obj_rot'] = loaded_dict['hoi_data'][:, 74:78]
            loaded_dict['obj_rot_vel'] = loaded_dict['hoi_data'][:, 81:84]
            object_points = self.object_points[self.object_id[idx]].cpu()
            obj_rot_extend = loaded_dict['obj_rot'].unsqueeze(1).repeat(1, object_points.shape[0], 1).view(-1, 4)
            object_points_extend = object_points.unsqueeze(0).repeat(loaded_dict['obj_rot'].shape[0], 1, 1).view(-1, 3)
            obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(loaded_dict['obj_rot'].shape[0], object_points.shape[0], 3) + loaded_dict['obj_pos'].unsqueeze(1)
            key_body_pose = loaded_dict['body_pos'][:,:].clone()
            ref_ig = compute_sdf(key_body_pose.view(loaded_dict['obj_rot'].shape[0],-1,3), obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(loaded_dict['root_rot'])
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1] // 3, 1).view(-1, 4)
            ref_ig = quat_rotate(heading_rot_extend, ref_ig).view(loaded_dict['obj_rot'].shape[0], -1)    
            loaded_dict['contact_parts'] = loaded_dict['hoi_data'][:, 591:630].clone()

            loaded_dict['human_rot'] = loaded_dict['hoi_data'][:, 201:357]
            loaded_dict['human_rot_vel'] = loaded_dict['hoi_data'][:, 474:591]
            loaded_dict['hoi_data'] = torch.cat([loaded_dict['hoi_data'], ref_ig], dim=-1)

            self.hoi_data_dict.append(loaded_dict)
            hoi_datas.append(loaded_dict['hoi_data'])
            
            hoi_ref = torch.cat((
                                loaded_dict['root_pos'].clone(), # 0:3
                                loaded_dict['root_rot'].clone(), # 3:7
                                loaded_dict['root_pos_vel'].clone(), # 7:10
                                loaded_dict['root_rot_vel'].clone(), # 10:13
                                loaded_dict['dof_pos'].clone(), # 13:42
                                loaded_dict['dof_pos_vel'].clone(), # 42:71
                                loaded_dict['obj_pos'].clone(), # 71:74
                                loaded_dict['obj_rot'].clone(), # 74:78
                                loaded_dict['obj_pos_vel'].clone(), # 78:81
                                loaded_dict['obj_rot_vel'].clone(), # 81:84
                                ),dim=-1)
            assert self.max_episode_length[-1] == loaded_dict['root_pos'].clone().shape[0]
            hoi_refs.append(hoi_ref)
        max_length = max(self.max_episode_length)
        self.num_motions = len(hoi_refs)
        self.max_episode_length = to_torch(self.max_episode_length, dtype=torch.long, device=self.device)
        print('total_motion_time: ', self.max_episode_length.sum())
        self.hoi_data = []
        self.hoi_refs = []
        for i, data in enumerate(hoi_datas):
            pad_size = (0, 0, 0, max_length - data.size(0))
            padded_data = F.pad(data, pad_size, "constant", 0)
            self.hoi_data.append(padded_data)
            self.hoi_refs.append(F.pad(hoi_refs[i], pad_size, "constant", 0))
        self.hoi_data = torch.stack(self.hoi_data, dim=0).to(self.device)
        self.hoi_refs = torch.stack(self.hoi_refs, dim=0).unsqueeze(1).repeat(1, 1, 1, 1).to(self.device)
        # self.ref_reward = torch.zeros((self.hoi_refs.shape[0], self.hoi_refs.shape[1], self.hoi_refs.shape[2])).to(self.hoi_refs.device)
        # self.ref_reward[:, 0, :] = 1.0
        self.ref_index = torch.zeros((self.num_envs, )).long().to(self.hoi_refs.device)
        return


    def _create_envs(self, num_envs, spacing, num_per_row):

        self._target_handles = []
        self._load_target_asset()
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_target(env_id, env_ptr)
        return   

    def _reset_target(self, env_ids):
        noise = _DEFAULT_NOISE_STD if self.cfg['domain_rand']['domain_rand_general'] else dict.fromkeys(_DEFAULT_NOISE_STD, 0.0)

        obj_pos      = self.hoi_refs[self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids], 71:74]
        obj_rot      = self.hoi_refs[self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids], 74:78]
        obj_pos_vel  = self.hoi_refs[self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids], 78:81]
        obj_rot_vel  = self.hoi_refs[self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids], 81:84]

        # add noise only to position/rotation (vel noise handled above if desired)
        obj_pos_new  = _randn_like(obj_pos, noise["target_pos"]) * (self.progress_buf[env_ids, None] < 0.5) + obj_pos * (self.progress_buf[env_ids, None] > 0.5)
        obj_pos_new[..., 2:3] = (obj_pos[..., 2:3] + 0.1) * (self.progress_buf[env_ids, None] < 0.5) + (obj_pos[..., 2:3] + 0.01) * (self.progress_buf[env_ids, None] > 0.5)
        obj_rot_new  = _perturb_quat(obj_rot, noise["target_rot"]) * (self.progress_buf[env_ids, None] < 0.5) + obj_rot * (self.progress_buf[env_ids, None] > 0.5)
        self._target_states[env_ids, :3]     = obj_pos_new
        self._target_states[env_ids, 3:7]    = obj_rot_new
        self._target_states[env_ids, 7:10]   = obj_pos_vel
        self._target_states[env_ids, 10:13]  = obj_rot_vel
        return

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        
        target_handle = self.gym.create_actor(env_ptr, self._target_asset[env_id % len(self.object_name)], default_pose, self.object_name[env_id % len(self.object_name)], col_group, col_filter, segmentation_id)
        env_cfg = self.cfg.get("env", {})
        dr_cfg = self.cfg.get("domain_rand", {})

        props = self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        for p_idx in range(len(props)):
            props[p_idx].restitution = 0.05
            props[p_idx].friction = 0.6
            props[p_idx].rolling_friction = 0.01
            props[p_idx].torsion_friction = 0.01
            # if self.object_name[env_id % len(self.object_name)] == 'plasticbox' or self.object_name[env_id % len(self.object_name)] == 'trashcan':
            #     props[p_idx].rest_offset = 0.015
        self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, props)

        self.randomize_physical_properties(env_ptr, target_handle)

        self._target_handles.append(target_handle)
        self.gym.set_actor_scale(env_ptr, target_handle, self.ball_size)
        
        props = self.gym.get_actor_rigid_body_properties(env_ptr, target_handle)
        for rbp in props:
            orig_mass = rbp.mass
            orig_com = rbp.com
            orig_inertia_diag = gymapi.Vec3(
                rbp.inertia.x.x,
                rbp.inertia.y.y,
                rbp.inertia.z.z
            )
            mass_min, mass_max = dr_cfg.get("obj_mass_range", [0.15, 1.5])
            rare_mass_min, rare_mass_max = dr_cfg.get("obj_mass_rare_range", [0.001, 0.01])
            rare_every = int(dr_cfg.get("obj_mass_rare_every", 10))
            mass_scale = np.random.uniform(mass_min, mass_max)
            if rare_every > 0 and env_id % rare_every == 0:
                mass_scale = np.random.uniform(rare_mass_min, rare_mass_max)
            rbp.mass = orig_mass * mass_scale
            com_delta = dr_cfg.get("obj_com_range", [-0.05, 0.05])
            rbp.com = gymapi.Vec3(
                orig_com.x + np.random.uniform(com_delta[0], com_delta[1]),
                orig_com.y + np.random.uniform(com_delta[0], com_delta[1]),
                orig_com.z + np.random.uniform(com_delta[0], com_delta[1]),
            )

            inertia_min, inertia_max = dr_cfg.get("obj_inertia_range", [0.5, 2.0])
            inertia_scale = mass_scale * np.random.uniform(inertia_min, inertia_max)
            rand_inertia = gymapi.Vec3(
                orig_inertia_diag.x * inertia_scale,
                orig_inertia_diag.y * inertia_scale,
                orig_inertia_diag.z * inertia_scale,
            )
            self.set_random_diagonal_inertia(rbp.inertia, rand_inertia)
        self.gym.set_actor_rigid_body_properties(env_ptr, target_handle, props, recomputeInertia=True)
        return

    def set_random_diagonal_inertia(self, mat33, vals):
        mat33.x.x = vals.x
        mat33.x.y = 0.0
        mat33.x.z = 0.0

        mat33.y.x = 0.0
        mat33.y.y = vals.y
        mat33.y.z = 0.0

        mat33.z.x = 0.0
        mat33.z.y = 0.0
        mat33.z.z = vals.z
    
    def randomize_physical_properties(self, env_ptr, target_handle):
        if not self.cfg['domain_rand']['domain_rand_general']:
            return
        # Sample new properties
        env_cfg = self.cfg.get("env", {})
        dr_cfg = self.cfg.get("domain_rand", {})
        friction_min, friction_max = dr_cfg.get("obj_friction_range", [0.2, 1.2])
        restitution_min, restitution_max = dr_cfg.get("obj_restitution_range", [0.0, 0.3])
        rolling_min, rolling_max = dr_cfg.get("obj_rolling_friction_range", [0.0, 0.05])
        torsion_min, torsion_max = dr_cfg.get("obj_torsion_friction_range", [0.0, 0.05])
        compliance_min, compliance_max = dr_cfg.get("obj_compliance_range", [0.0, 1.0])
        rest_min, rest_max = dr_cfg.get("obj_rest_offset_range", [0.0, 0.01])
        contact_min, contact_max = dr_cfg.get("obj_contact_offset_range", [0.01, 0.03])

        friction = np.random.uniform(friction_min, friction_max)
        restitution = np.random.uniform(restitution_min, restitution_max)
        rest_offset = np.random.uniform(rest_min, rest_max)
        contact_offset = np.random.uniform(contact_min, contact_max)
        if contact_offset < rest_offset + 1e-4:
            contact_offset = rest_offset + 1e-4

        # Retrieve existing properties
        rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        # Modify properties (applies to all shapes of this actor)
        for shape_prop in rigid_shape_props:
            shape_prop.friction = friction
            shape_prop.restitution = restitution
            shape_prop.rolling_friction = np.random.uniform(rolling_min, rolling_max)
            shape_prop.torsion_friction = np.random.uniform(torsion_min, torsion_max)
            shape_prop.compliance = np.random.uniform(compliance_min, compliance_max)
            if hasattr(shape_prop, "rest_offset"):
                shape_prop.rest_offset = rest_offset
            if hasattr(shape_prop, "contact_offset"):
                shape_prop.contact_offset = contact_offset


    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)


        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        for eid in env_ids:
            self.randomize_physical_properties(self.envs[eid], self._target_handles[eid])

        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)

        return    
    
    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        i = to_torch([_ % self.num_motions for _ in env_ids], device=self.device, dtype=torch.long)

        if (self._state_init == Ultra.StateInit.Random
            or self._state_init == Ultra.StateInit.Hybrid):
            if self.is_distill:
                motion_times = torch.cat([torch.randint(0, max(1, self.max_episode_length[i[e]]-2), (1,), device=self.device, dtype=torch.long) for e in range(num_envs)])
            else:
                motion_times = torch.cat([torch.randint(0, max(1, self.max_episode_length[i[e]]-self.rollout_length), (1,), device=self.device, dtype=torch.long) for e in range(num_envs)])
        elif (self._state_init == Ultra.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device, dtype=torch.long)#.int()

        # Randomly assign 10% of these environments to "stand still" mode
        # where reference frame stays constant throughout the episode
        rand_vals = torch.rand(num_envs, device=self.device)
        stand_still_mask = rand_vals < self.stand_still_ratio
        self.is_stand_still[env_ids] = stand_still_mask
        self.stand_still_frame[env_ids] = motion_times.clone()

        # Randomly determine which privileged observations to keep for this episode
        # This adds diversity to training: some episodes train with task_obs, some with IG, some with neither
        self.keep_task_obs_mask[env_ids] = torch.rand(num_envs, device=self.device) < self.obs_task_keep_prob
        self.keep_ig_mask[env_ids] = torch.rand(num_envs, device=self.device) < self.obs_ig_keep_prob


        # ref_reward = self.ref_reward[i, :, motion_times]
        # prob = ref_reward / ref_reward.sum(1, keepdim=True)

        # cdf = torch.cumsum(prob, dim=1)
        # idx = torch.searchsorted(cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)).squeeze(1)
        self.ref_index[env_ids] = 0
        self.progress_buf[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        self.data_id[env_ids] = i
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0
        self._set_env_state(env_ids=env_ids,
                            root_pos=self.hoi_refs[i, self.ref_index[env_ids], motion_times, 0:3],
                            root_rot=self.hoi_refs[i, self.ref_index[env_ids], motion_times, 3:7],
                            dof_pos=self.hoi_refs[i, self.ref_index[env_ids], motion_times, 13:42],
                            root_vel=self.hoi_refs[i, self.ref_index[env_ids], motion_times, 7:10],
                            root_ang_vel=self.hoi_refs[i, self.ref_index[env_ids], motion_times, 10:13],
                            dof_vel=self.hoi_refs[i, self.ref_index[env_ids], motion_times, 42:71],
                            )

        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        noise = _DEFAULT_NOISE_STD if self.cfg['domain_rand']['domain_rand_general'] else dict.fromkeys(_DEFAULT_NOISE_STD, 0.0)
        root_pos_new = _randn_like(root_pos, noise["root_pos"]) * (self.progress_buf[env_ids, None] < 0.5) + root_pos * (self.progress_buf[env_ids, None] > 0.5)
        # root_pos_new[..., 2:3] = (root_pos[..., 2:3] + 0.1) * (self.progress_buf[env_ids, None] < 0.5) + (root_pos[..., 2:3] + 0.01) * (self.progress_buf[env_ids, None] > 0.5)
        root_rot_new = _perturb_quat(root_rot, noise["root_rot"]) * (self.progress_buf[env_ids, None] < 0.5) + root_rot * (self.progress_buf[env_ids, None] > 0.5)
        root_vel_new = _randn_like(root_vel, noise["root_vel"]) * (self.progress_buf[env_ids, None] < 0.5) + root_vel * (self.progress_buf[env_ids, None] > 0.5)
        root_ang_vel_new = _randn_like(root_ang_vel, noise["root_ang_vel"]) * (self.progress_buf[env_ids, None] < 0.5) + root_ang_vel * (self.progress_buf[env_ids, None] > 0.5)
        
        dof_pos_new = _randn_like(dof_pos, noise["dof_pos"]) * (self.progress_buf[env_ids, None] < 0.5) + dof_pos * (self.progress_buf[env_ids, None] > 0.5)
        dof_vel_new = _randn_like(dof_vel, noise["dof_vel"]) * (self.progress_buf[env_ids, None] < 0.5) + dof_vel * (self.progress_buf[env_ids, None] > 0.5)
        dof_pos_new = torch.clamp(dof_pos_new, min=self.dof_limits_lower, max=self.dof_limits_upper)
        self._humanoid_root_states[env_ids, 0:3] = root_pos_new
        self._humanoid_root_states[env_ids, 3:7] = root_rot_new
        self._humanoid_root_states[env_ids, 7:10] = root_vel_new
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel_new
        
        self._dof_pos[env_ids] = dof_pos_new
        self._dof_vel[env_ids] = dof_vel_new
        return
    
    def cal_cdf(self, i, e):
        rewards = self.ref_reward[i[e], :, :max(1, self.max_episode_length[i[e]]-self.rollout_length)].clone() 
        ref_reward_sum = 1 / (rewards.sum(dim=0)) 
        prob = ref_reward_sum / ref_reward_sum.sum()
        cdf = torch.cumsum(prob, 0)
        return cdf

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        i = to_torch([torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())] for i in env_ids], device=self.device, dtype=torch.long)
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]

        if self.is_distill:
            motion_times = torch.cat([torch.randint(0, max(1, self.max_episode_length[i[e]]-2), (1,), device=self.device, dtype=torch.long) for e in range(num_envs)])
        else:
            motion_times = torch.cat([torch.randint(0, max(1, self.max_episode_length[i[e]]-self.rollout_length), (1,), device=self.device, dtype=torch.long) for e in range(num_envs)])

        # Randomly assign 10% of these environments to "stand still" mode
        # where reference frame stays constant throughout the episode
        rand_vals = torch.rand(num_envs, device=self.device)
        stand_still_mask = rand_vals < self.stand_still_ratio
        self.is_stand_still[env_ids] = stand_still_mask
        self.stand_still_frame[env_ids] = motion_times.clone()

        # Randomly determine which privileged observations to keep for this episode
        # This adds diversity to training: some episodes train with task_obs, some with IG, some with neither
        self.keep_task_obs_mask[env_ids] = torch.rand(num_envs, device=self.device) < self.obs_task_keep_prob
        self.keep_ig_mask[env_ids] = torch.rand(num_envs, device=self.device) < self.obs_ig_keep_prob

        # ref_reward = self.ref_reward[i, :, motion_times]
        # prob = ref_reward / ref_reward.sum(1, keepdim=True)

        # cdf = torch.cumsum(prob, dim=1)
        # idx = torch.searchsorted(cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)).squeeze(1)
        self.ref_index[env_ids] = 0
        self.progress_buf[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        self.data_id[env_ids] = i
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0
        self._set_env_state(env_ids=env_ids,
                            root_pos=self.hoi_refs[i, self.ref_index[env_ids], motion_times, 0:3],
                            root_rot=self.hoi_refs[i, self.ref_index[env_ids], motion_times, 3:7],
                            dof_pos=self.hoi_refs[i, self.ref_index[env_ids], motion_times, 13:42],
                            root_vel=self.hoi_refs[i, self.ref_index[env_ids], motion_times, 7:10],
                            root_ang_vel=self.hoi_refs[i, self.ref_index[env_ids], motion_times, 10:13],
                            dof_vel=self.hoi_refs[i, self.ref_index[env_ids], motion_times, 42:71],
                            )
        return


    def _compute_reward(self, actions):
        hist_dof_vel = self._hist_obs[:,42:71]

        local_vel = (self._curr_obs[:,42:71] - hist_dof_vel)*self.fps_data
        dof_diffacc = (local_vel.view(-1, self.num_actions)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)).clone()

        hist_obj_vel = self._hist_obs[:,78:81]
        obj_diffacc = (self._curr_obs[:,78:81] - hist_obj_vel)*self.fps_data
        obj_diffacc = obj_diffacc*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)

        hist_obj_rot_vel = self._hist_obs[:,81:84]
        local_vel = (self._curr_obs[:,81:84] - hist_obj_rot_vel)*self.fps_data
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
                                                  )
        self.contact_reset = (self.contact_reset + contact_reset) * contact_reset
        self._reset_ig = torch.logical_or(ig_reset, kinematic_reset)

        # Temporarily disable tracking termination after pushes to allow recovery
        # Curriculum learning: gradually reduce the grace period
        if self.cfg['domain_rand']['push_robots']:
            curriculum_steps = 1000 * 32
            curriculum_progress = min(self.common_step_counter / curriculum_steps, 1.0)

            # Gradually reduce grace period from 60 steps to 20 steps
            max_grace_steps = 60
            min_grace_steps = 20
            current_grace_steps = int(max_grace_steps - (max_grace_steps - min_grace_steps) * curriculum_progress)

            # Disable reset_ig for environments recently pushed
            recently_pushed = self.steps_since_push < current_grace_steps
            self._reset_ig = self._reset_ig & ~recently_pushed
            # self.contact_reset = self.contact_reset & ~recently_pushed

        self._compute_reset()
        smooth_rewards = self.smooth_rewards._compute_smooth_reward()

        # Curriculum learning: gradually increase smooth reward impact
        # Start with minimal impact and increase to full strength
        curriculum_steps = 5000 * 32  # Number of steps to reach full strength (5000 iterations * 24 steps/iter)
        curriculum_progress = min(self.common_step_counter / curriculum_steps, 1.0)

        # Scale the smooth reward magnitude by curriculum progress
        # At curriculum_progress=0: smooth_rewards is close to 0, so exp(0/3) ≈ 1 (no penalty)
        # At curriculum_progress=1: smooth_rewards is full strength
        self.extras['smooth_reward_names'] = self.smooth_rewards.reward_names
        self.extras['smooth_reward'] = self.smooth_rewards.reward
        self.extras['smooth_reward_buf'] = self.smooth_rewards.reward_buf
        # print(smooth_rewards)
        self.rew_buf[:] += smooth_rewards * curriculum_progress
        return
    
    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf, self.obs_buf,
                                                   self._contact_forces,
                                                   self._rigid_body_pos, self.max_episode_length[self.data_id],
                                                   self._enable_early_termination, self._termination_heights, self._termination_heights_init, self._curr_ref_obs, self._curr_obs, self.start_times, self.rollout_length, self._reset_ig, torch.any(self.contact_reset > 20, dim=-1), torch.logical_or(self.is_stand_still, self.progress_buf >= self.max_episode_length[self.data_id] - 20)
                                                   )
        self.obs_history_buf[self.reset_buf > 0] *= 0
        return
    
    def _compute_humanoid_obs(self, env_ids=None, ref_obs=None, next_ts=None, student_obs=False):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            contact_forces = self._contact_forces
            actions = self.action_history_buf
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            torques = self.torques
            last_dof_pos = self.last_dof_pos
            last_dof_vel = self.last_dof_vel
            humanoid_root_states = self._humanoid_root_states
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            contact_forces = self._contact_forces[env_ids]
            actions = self.action_history_buf[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            torques = self.torques[env_ids]
            last_dof_pos = self.last_dof_pos[env_ids]
            last_dof_vel = self.last_dof_vel[env_ids]
            humanoid_root_states = self._humanoid_root_states[env_ids]


        obs = self.compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                self._root_height_obs,
                                                contact_forces, self._contact_body_ids, ref_obs, self._key_body_ids, 
                                                self._key_body_ids_gt, self._contact_body_ids_gt, actions, dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel, humanoid_root_states, env_ids, student_obs=student_obs)

        return obs



    def compute_humanoid_observations_max(self, body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, contact_body_ids, ref_obs, key_body_ids, key_body_ids_gt, contact_body_ids_gt, actions, dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel, humanoid_root_states, env_ids, student_obs=False):
        root_pos = self._rand_vec(body_pos[:, 0, :], 0.0)
        root_rot = self._rand_vec(body_rot[:, 0, :], 0.)

        root_h = root_pos[:, 2:3]
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        if (not root_height_obs):
            root_h_obs = torch.zeros_like(root_h)
        else:
            root_h_obs = root_h

        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand_2 = heading_rot_expand.repeat((1, 39, 1))
        flat_heading_rot_2 = heading_rot_expand_2.reshape(heading_rot_expand_2.shape[0] * heading_rot_expand_2.shape[1], 
                                                heading_rot_expand_2.shape[2])
        
        heading_rot_expand = heading_rot_expand.repeat((1, 39, 1))
        flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                heading_rot_expand.shape[2])

        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, 39, 1))
        flat_heading_inv_rot = heading_inv_rot_expand.reshape(heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                heading_inv_rot_expand.shape[2])
        
        _ref_body_pos = ref_obs[:,84:201].view(-1, 39, 3)
        _body_pos = self._rand_vec(body_pos, 0.0)

        diff_global_body_pos = _ref_body_pos - _body_pos
        diff_local_body_pos_flat = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_body_pos.view(-1, 3)).view(-1, 39 * 3)

        local_ref_body_pos = _ref_body_pos - root_pos.unsqueeze(1)  # preserves the body position
        local_ref_body_pos = torch_utils.quat_rotate(flat_heading_rot_2, local_ref_body_pos.view(-1, 3)).view(-1, 39 * 3)

        root_pos_expand = root_pos.unsqueeze(-2)
        local_body_pos = self._rand_vec(body_pos, 0.0) - root_pos_expand
        flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
        flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
        local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
        local_body_pos = local_body_pos[..., 3:] # remove root pos

        flat_body_rot = self._rand_vec(body_rot.reshape(body_rot.shape[0] * 39, body_rot.shape[2]), 0.)
        flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
        flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], 39 * flat_local_body_rot_obs.shape[1])
        
        ref_body_rot = ref_obs[:, 201:357].view(-1, 39, 4)
        ref_body_rot_no_hand = ref_body_rot
        body_rot_no_hand = self._rand_vec(body_rot, 0.)

        diff_global_body_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)), body_rot_no_hand.reshape(-1, 4))
        diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(flat_heading_rot, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot)
        diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
        diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot_no_hand.shape[0], body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1])

        local_ref_body_rot = torch_utils.quat_mul(flat_heading_rot, ref_body_rot_no_hand.reshape(-1, 4))
        local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot_no_hand.shape[0], -1)

        ref_body_vel = ref_obs[:, 357:474].view(-1, 39, 3)
        # body_vel = self._rand_vec(body_vel, 0.1)
        diff_global_vel = ref_body_vel - body_vel
        diff_local_vel = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, 39 * 3)

        ref_body_ang_vel = ref_obs[:, 474:591]
        ref_body_ang_vel_no_hand = ref_body_ang_vel.view(-1, 39, 3)
        body_ang_vel_no_hand = body_ang_vel
        diff_global_ang_vel = ref_body_ang_vel_no_hand - body_ang_vel_no_hand
        diff_local_ang_vel = torch_utils.quat_rotate(flat_heading_rot, diff_global_ang_vel.view(-1, 3)).view(-1, 39 * 3)

        if (local_root_obs):
            root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
            local_body_rot_obs[..., 0:6] = root_rot_obs

        flat_body_vel = body_vel.reshape(body_vel.shape[0] * 39, body_vel.shape[2])
        flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
        local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], 39 * body_vel.shape[2])
        
        flat_body_ang_vel = body_ang_vel_no_hand.reshape(body_ang_vel.shape[0] * 39, body_ang_vel.shape[2])
        flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
        local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], 39 * body_ang_vel.shape[2])

        body_contact_buf = contact_forces.clone() #.view(contact_forces.shape[0],-1)
        contact = (torch.abs(body_contact_buf).sum(dim=-1) > 0.1).float()
        ref_body_contact = ref_obs[:,591:630]
        diff_body_contact = ref_body_contact - contact
        contact_new = torch.zeros_like(contact).to(contact.device)
        diff_body_contact_new = torch.zeros_like(diff_body_contact).to(contact.device)
        contact_new[:, [6, 7, 13, 14, 28, 38]] = contact[:, [6, 7, 13, 14, 28, 38]]
        diff_body_contact_new[:, [6, 7, 13, 14, 28, 38]] = diff_body_contact[:, [6, 7, 13, 14, 28, 38]]

        if student_obs:
            # obs_prop shape: [nuv_envs, 320] (1771 - 320 = 1451)
            root_ang_vel = humanoid_root_states[:, 10:13]
            ref_root_ang_vel = ref_body_ang_vel[:, 0:3]
            diff_root_ang_vel = ref_root_ang_vel - root_ang_vel
            base_quat = humanoid_root_states[:, 3:7]
            ref_base_quat = ref_body_rot[:, 0]
            roll, pitch, yaw = euler_from_quaternion(base_quat)
            roll_ref, pitch_ref, yaw_ref = euler_from_quaternion(ref_base_quat)
            imu_obs = torch.stack((roll, pitch), dim=1)
            imu_obs_all = torch.stack((roll, pitch, yaw), dim=1)
            imu_obs_ref_all = torch.stack((roll_ref, pitch_ref, yaw_ref), dim=1)
            diff_imu_obs_all = imu_obs_ref_all - imu_obs_all
            # imu_obs_ref = torch.stack((roll_ref, pitch_ref), dim=1)
            # diff_imu_obs_ref = imu_obs_ref - imu_obs
            dof_pos = self._rand_vec(dof_pos, 0.0)
            ref_dof_pos = ref_obs[:, 13:42]
            diff_dof_pos = dof_pos - ref_dof_pos
            _dof_vel = dof_vel.clone()
            _dof_vel[..., [12, 13, 14]] *= 0.05
            root_pos = _body_pos[:, 0, :].clone()
            ref_root_pos = _ref_body_pos[:, 0, :].clone()
            diff_root_pos = root_pos - ref_root_pos 
            # ref_root_ang_vel.zero_()
            # diff_root_ang_vel.zero_()
            # imu_obs_ref.zero_()
            # diff_imu_obs_ref.zero_()
            # diff_local_body_pos_flat.zero_()
            if self.cfg['env']['history_len'] > 0:
                # # obs shape: [nuv_envs, 2450]
                # print(root_ang_vel.shape[1] + ref_root_ang_vel.shape[1] + diff_root_ang_vel.shape[1] + imu_obs.shape[1] + imu_obs_ref.shape[1] + diff_imu_obs_ref.shape[1])
                obs_prop = torch.cat((ref_dof_pos, diff_dof_pos, ref_root_pos, root_pos, diff_root_pos, root_ang_vel, imu_obs_ref_all, imu_obs_all, diff_imu_obs_all, dof_pos, _dof_vel, actions[:, -1, :], self.obs_history_buf.view(self.num_envs, -1)[env_ids]), dim=-1)
                # 405=29+29+29+3+2+29+29 + 92 * 25
                obs_buf = torch.cat(
                    (
                        ref_dof_pos,  # 29 [0:29]
                        diff_dof_pos, # 29 | 58 [29 :58]
                        ref_root_pos, # 3 | 61 [58:61]
                        root_pos, # 3 | 64 [61:64]
                        diff_root_pos, # 3 | 67 [64:67]
                        root_ang_vel, # 3 | 70 [67:70]
                        imu_obs_ref_all,  # 3 dims | 73 [70:73]
                        imu_obs_all,  # 3 dims | 76 [73:76]
                        diff_imu_obs_all, # 3 dims | 79 [76:79]
                        self.reindex((dof_pos)),
                        self.reindex(_dof_vel),
                        actions[:, -1, :],
                    ),
                    dim=-1,
                )
                if self.cfg['noise']['add_noise'] and self.headless:
                    noise = (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * min(
                        self.common_step_counter / (self.cfg['noise']['noise_increasing_steps'] * 32), 1.0
                    )
                    obs_buf += noise
                    obs_buf[:, 29:58] += noise[:, 79:79+29]
                    obs_buf[:, 64:67] += noise[:, 61:64]
                    obs_buf[:, 76:79] -= noise[:, 73:76]
                    obs_prop[:, :166] += noise
                    obs_prop[:, 29:58] += noise[:, 79:79+29]
                    obs_prop[:, 64:67] += noise[:, 61:64]
                    obs_prop[:, 76:79] -= noise[:, 73:76]
                elif self.cfg['noise']['add_noise'] and not self.headless:
                    noise = (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec
                    obs_buf += noise
                    obs_buf[:, 29:58] += noise[:, 79:79+29]
                    obs_buf[:, 64:67] += noise[:, 61:64]
                    obs_buf[:, 76:79] -= noise[:, 73:76]
                    obs_prop[:, :166] += noise
                    obs_prop[:, 29:58] += noise[:, 79:79+29]
                    obs_prop[:, 64:67] += noise[:, 61:64]
                    obs_prop[:, 76:79] -= noise[:, 73:76]
                else:
                    obs_buf += 0.0

                if env_ids.shape[0] == self.num_envs:
                    self.obs_history_buf = torch.where(
                        (self.episode_length_buf <= 1)[:, None, None],
                        torch.stack([obs_buf] * self.cfg['env']['history_len'], dim=1),
                        torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1),
                    )
                else:
                    self.obs_history_buf[env_ids] = torch.stack([obs_buf] * self.cfg['env']['history_len'], dim=1)
                return obs_prop
            else:
                obs_prop = torch.cat((ref_dof_pos, diff_dof_pos, root_ang_vel, imu_obs, dof_pos, _dof_vel, actions[:, -1, :]), dim=-1)
                return obs_prop
        else:
            # # obs shape: [nuv_envs, 1771]
            # # local_body_vel = local_body_vel * 0
            # # local_body_vel = local_body_vel * 0
            # local_body_ang_vel = local_body_ang_vel * 0
            # # diff_local_vel = diff_local_vel * 0
            # diff_local_ang_vel = diff_local_ang_vel * 0
            # # actions = actions * 0
            # # dof_vel = dof_vel * 0
            torques = torques * 0
            # # last_dof_vel = last_dof_vel * 0
            # print(root_h_obs.shape[1] + local_body_pos.shape[1] + local_body_rot_obs.shape[1] + local_body_vel.shape[1] + local_body_ang_vel.shape[1] + contact_new.shape[1] + diff_local_body_pos_flat.shape[1] + diff_local_body_rot_obs.shape[1] + diff_body_contact_new.shape[1] + local_ref_body_pos.shape[1] + local_ref_body_rot.shape[1] + diff_local_vel.shape[1] + diff_local_ang_vel.shape[1] + actions.shape[1])
            obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, contact_new, diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact_new, local_ref_body_pos, local_ref_body_rot, diff_local_vel, diff_local_ang_vel, actions[:, -1, :], self._rand_vec(dof_pos, 0.0), self._rand_vec(dof_vel, 0.), torques, self._rand_vec(last_dof_pos, 0.0), self._rand_vec(last_dof_vel, 0.)), dim=-1)
            return obs

    def reindex(self, vec):
        return vec

    def _compute_hoi_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos
        key_body_vel = self._rigid_body_vel
        key_body_rot = self._rigid_body_rot
        key_body_ang_vel = self._rigid_body_ang_vel
        if (env_ids is None):
            self._curr_obs[:] = build_hoi_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._target_states,
                                                               self._tar_contact_forces,
                                                               self._contact_forces,
                                                               self.object_points[self.object_id[self.data_id]],
                                                               key_body_rot,
                                                               key_body_vel,
                                                               key_body_ang_vel,
                                                               self._key_body_ids_gt,
                                                               self._contact_body_ids_gt,
                                                               )
        else:
            self._curr_obs[env_ids] = build_hoi_observations(self._rigid_body_pos[env_ids][:, 0, :],
                                                                   self._rigid_body_rot[env_ids][:, 0, :],
                                                                   self._rigid_body_vel[env_ids][:, 0, :],
                                                                   self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._target_states[env_ids],
                                                                   self._tar_contact_forces[env_ids],
                                                                   self._contact_forces[env_ids],
                                                                   self.object_points[self.object_id[self.data_id[env_ids]]],
                                                                   key_body_rot[env_ids],
                                                                   key_body_vel[env_ids],
                                                                   key_body_ang_vel[env_ids],
                                                                   self._key_body_ids_gt,
                                                                   self._contact_body_ids_gt,
                                                                   ).float()
        return
    
    def _compute_observations_iter(self, env_ids=None, delta_t=1, student_obs=False):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
            ts = self.progress_buf.clone()
            self._curr_ref_obs = self.hoi_data[self.data_id[env_ids], ts].clone()
            next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]]-1)

            # For "stand still" environments, use the fixed frame instead of advancing
            next_ts = torch.where(self.is_stand_still, self.stand_still_frame, next_ts)
            self._curr_ref_obs[next_ts<=ts] = self.hoi_data[self.data_id[next_ts<=ts], next_ts[next_ts<=ts]].clone()
            ref_obs = self.hoi_data[self.data_id[env_ids], next_ts].clone()
            obs = self._compute_humanoid_obs(env_ids, ref_obs, next_ts, student_obs)
            obs_shape = obs.shape[-1]
            task_obs, obj_points = self._compute_task_obs(env_ids, ref_obs)
            obs = torch.cat([obs, task_obs], dim=-1)
            key_body_pose = self._rigid_body_pos.clone() 
            if student_obs:
                key_body_pose = self._rand_vec(key_body_pose, 0.01)
            ig = compute_sdf(key_body_pose, obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(self._rand_vec(self._rigid_body_rot[:, 0, :], 0.1))
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1], 1).view(-1, 4)
            ig = quat_rotate(heading_rot_extend, ig).view(env_ids.shape[0], -1, 3)
            ig_norm = ig.norm(dim=-1, keepdim=True)
            ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
            ig = ig_all.view(env_ids.shape[0], -1)
            ig_all = ig_all.view(env_ids.shape[0], -1)    
            ref_ig = ref_obs[:, 630:].view(env_ids.shape[0], 39, 3)
            ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
            ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()  
            ref_ig = ref_ig.view(env_ids.shape[0], -1)          
            obs = torch.cat((obs,ig_all,ref_ig-ig),dim=-1)
            if student_obs:
                # Use per-environment masks to decide which observations to keep
                # Split the privileged observations into task_obs and ig features
                task_obs_size = task_obs.shape[-1]  # Object-related observations
                # Zero out task observations for environments where keep_task_obs_mask is False
                if task_obs_size > 0:
                    task_start = obs_shape
                    task_end = obs_shape + task_obs_size
                    obs[~self.keep_task_obs_mask[env_ids], task_start:task_end] = 0

                # Zero out IG features for environments where keep_ig_mask is False
                obs[~self.keep_ig_mask[env_ids], task_end:] = 0
            return obs

        else:
            ts = self.progress_buf[env_ids].clone()
            self._curr_ref_obs[env_ids] = self.hoi_data[self.data_id[env_ids], ts].clone()
            next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]]-1)

            # For "stand still" environments, use the fixed frame instead of advancing
            next_ts = torch.where(self.is_stand_still[env_ids], self.stand_still_frame[env_ids], next_ts)
            self._curr_ref_obs[env_ids[next_ts<=ts]] = self.hoi_data[self.data_id[env_ids[next_ts<=ts]], next_ts[next_ts<=ts]].clone()
            ref_obs = self.hoi_data[self.data_id[env_ids], next_ts].clone()
            obs = self._compute_humanoid_obs(env_ids, ref_obs, next_ts, student_obs)
            obs_shape = obs.shape[-1]
            task_obs, obj_points = self._compute_task_obs(env_ids, ref_obs)
            obs = torch.cat([obs, task_obs], dim=-1)
            key_body_pose = self._rigid_body_pos[env_ids].clone() 
            if student_obs:
                key_body_pose = self._rand_vec(key_body_pose, 0.01)
            ig = compute_sdf(key_body_pose, obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(self._rand_vec(self._rigid_body_rot[env_ids][:, 0, :], 0.1))
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1], 1).view(-1, 4)
            ig = quat_rotate(heading_rot_extend, ig).view(env_ids.shape[0], -1, 3)
            ig_norm = ig.norm(dim=-1, keepdim=True)
            ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
            ig = ig_all.view(env_ids.shape[0], -1)
            ig_all = ig_all.view(env_ids.shape[0], -1)  
            ref_ig = ref_obs[:, 630:].view(env_ids.shape[0], 39, 3)
            ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
            ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()  
            ref_ig = ref_ig.view(env_ids.shape[0], -1)          
            obs = torch.cat((obs,ig_all,ref_ig-ig),dim=-1)
            if student_obs:
                # Use per-environment masks to decide which observations to keep
                # Split the privileged observations into task_obs and ig features
                task_obs_size = task_obs.shape[-1]  # Object-related observations
                # Zero out task observations for environments where keep_task_obs_mask is False
                if task_obs_size > 0:
                    task_start = obs_shape
                    task_end = obs_shape + task_obs_size
                    obs[~self.keep_task_obs_mask[env_ids], task_start:task_end] = 0

                obs[~self.keep_ig_mask[env_ids], task_end:] = 0
            return obs
            

        return

    def _compute_task_obs(self, env_ids=None, ref_obs=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_states = self._target_states
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target_states[env_ids]
        
        obs, obj_points = compute_obj_observations(root_states, tar_states, self.object_points[self.object_id[self.data_id[env_ids]]], ref_obs)
        return obs, obj_points

    def _compute_observations(self, env_ids=None):
        if (env_ids is None):
            self.obs_buf[:] = torch.cat((self._compute_observations_iter(None, 1), self._compute_observations_iter(None, 16)), dim=-1)
            # self.obs_buf[:] += (2 * torch.rand_like(self.obs_buf[:]) - 1) * 0.1

        else:
            self.obs_buf[env_ids] = torch.cat((self._compute_observations_iter(env_ids, 1), self._compute_observations_iter(env_ids, 16)), dim=-1)
            # self.obs_buf[env_ids] += (2 * torch.rand_like(self.obs_buf[env_ids]) - 1) * 0.1

        return 

    def _rand_vec(self, vec, scale=0.1):
        if not self.cfg['domain_rand']['domain_rand_general']:
            return vec
        return vec + (2 * torch.rand_like(vec) - 1) * scale
    
    def play_dataset_step(self, time):

        t = time
        if t == 0:
            self.data_id = to_torch([torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())] for i in range(self.num_envs)], device=self.device, dtype=torch.long)
        env_ids = to_torch([i for i in range(self.num_envs) if t < self.max_episode_length[self.data_id[i]]], device=self.device, dtype=torch.long)

        ### update object ###
        self._target_states[env_ids, :3] = self.hoi_refs[self.data_id[env_ids], 0, t, 71:74]
        self._target_states[env_ids, 3:7] = self.hoi_refs[self.data_id[env_ids], 0, t, 74:78]
        self._target_states[env_ids, 7:10] = self.hoi_refs[self.data_id[env_ids], 0, t, 78:81]# self.hoi_refs[self.data_id[env_ids], 0, t, 326:329]
        self._target_states[env_ids, 10:13] = self.hoi_refs[self.data_id[env_ids], 0, t, 81:84]# self.hoi_refs[self.data_id[env_ids], 0, t, 329:332]

        ### update subject ###   
        _humanoid_root_pos = self.hoi_refs[self.data_id[env_ids], 0, t, 0:3]
        _humanoid_root_rot = self.hoi_refs[self.data_id[env_ids], 0, t, 3:7]
        self._humanoid_root_states[env_ids, 0:3] = _humanoid_root_pos
        self._humanoid_root_states[env_ids, 3:7] = _humanoid_root_rot
        self._humanoid_root_states[env_ids, 7:10] = self.hoi_refs[self.data_id[env_ids], 0, t, 7:10]
        self._humanoid_root_states[env_ids, 10:13] = self.hoi_refs[self.data_id[env_ids], 0, t, 10:13]
        
        self._dof_pos[env_ids] = self.hoi_refs[self.data_id[env_ids], 0, t, 13:42]
        self._dof_vel[env_ids] = self.hoi_refs[self.data_id[env_ids], 0, t, 42:71]


        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self._refresh_sim_tensors()
        self.actions = self.hoi_refs[:, 0, t, 13:42]
        self.step(self.actions)
        return

def compute_humanoid_reward(hoi_ref, hoi_obs, contact_buf, tar_contact_forces, key_body_ids, w, actions, object_points, object_pos_action, object_rot_action, init_dof, num_dof_hand, _contact_body_ids_gt, _contact_body_ids, t):
    root_pos = hoi_obs[:,:3]
    root_rot = hoi_obs[:,3:3+4]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    dof_pos = hoi_obs[:,13:42]
    dof_vel = hoi_obs[:,42:71]

    obj_pos = hoi_obs[:,71:74]
    obj_rot = hoi_obs[:,74:78]

    local_obj_pos = obj_pos - root_pos
    local_obj_pos[..., -1] = obj_pos[..., -1]
    local_obj_pos = quat_rotate(heading_rot, local_obj_pos)

    local_obj_rot = quat_mul(heading_rot, obj_rot)

    obj_rot_extend = obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    object_points_extend = object_points.view(-1, 3)
    obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + obj_pos.unsqueeze(1)

    obj_pos_vel = hoi_obs[:,78:81]
    obj_rot_vel = hoi_obs[:,81:84]
    key_pos = hoi_obs[:,84:201].view(-1, 39, 3)
    body_pos_vel = hoi_obs[:, 357:474] #.view(-1, len_keypos, 3)
    human_contact = hoi_obs[:, 591:630]

    body_rot = hoi_obs[:, 201:357].view(-1, 39*4)# torch.cat([ref_obs[:, 3:7], torch_utils.exp_map_to_quat(ref_obs[:,7:7+51*3].reshape(-1, 3)).view(-1, 51 * 4)], dim=-1)
    ig = key_pos.view(-1,39,3).unsqueeze(2) - obj_points.unsqueeze(1)

    ref_root_pos = hoi_ref[:,:3]
    ref_root_rot = hoi_ref[:,3:3+4]

    ref_heading_rot = torch_utils.calc_heading_quat_inv(ref_root_rot)


    ref_dof_pos = hoi_ref[:,13:42]
    ref_dof_vel = hoi_ref[:,42:71]

    ref_obj_pos = hoi_ref[:,71:74]
    ref_obj_rot = hoi_ref[:,74:78]

    ref_local_obj_pos = ref_obj_pos - ref_root_pos
    ref_local_obj_pos[..., -1] = ref_obj_pos[..., -1]
    ref_local_obj_pos = quat_rotate(ref_heading_rot, ref_local_obj_pos)

    ref_local_obj_rot = quat_mul(ref_heading_rot, ref_obj_rot)

    ref_obj_rot_extend = ref_obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    ref_obj_points = torch_utils.quat_rotate(ref_obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + ref_obj_pos.unsqueeze(1)

    ref_obj_pos_vel = hoi_ref[:,78:81]
    ref_obj_rot_vel = hoi_ref[:,81:84]
    ref_key_pos = hoi_ref[:,84:201].view(-1, 39, 3)
    ref_body_pos_vel = hoi_ref[:, 357:474] #.view(-1, len_keypos, 3)

    # Amplify foot height based on foot velocity to encourage higher lifting during swing
    # Indices 2 and 5 are left and right feet (ankle)
    left_foot_idx = 2
    right_foot_idx = 5

    # Reshape velocity to access individual body velocities
    ref_body_pos_vel_reshaped = ref_body_pos_vel.view(-1, 39, 3)

    # Get foot velocities (use norm of 3D velocity as indicator of swing phase)
    left_foot_vel_norm = ref_body_pos_vel_reshaped[:, left_foot_idx, :2].norm(dim=-1)  # [batch]
    right_foot_vel_norm = ref_body_pos_vel_reshaped[:, right_foot_idx, :2].norm(dim=-1)  # [batch]

    # Coefficient to scale velocity to height offset
    vel_to_height_coef = 0.1  # Adjust this to control how much height is added

    # Add height offset proportional to foot velocity
    ref_key_pos = ref_key_pos.clone()
    ref_key_pos[:, left_foot_idx, 2] = ref_key_pos[:, left_foot_idx, 2] + left_foot_vel_norm * vel_to_height_coef
    ref_key_pos[:, right_foot_idx, 2] = ref_key_pos[:, right_foot_idx, 2] + right_foot_vel_norm * vel_to_height_coef
    ref_human_contact = hoi_ref[:, 591:630]


    ref_body_rot = hoi_ref[:, 201:357].view(-1, 39*4)# torch.cat([ref_obs[:, 3:7], torch_utils.exp_map_to_quat(ref_obs[:,7:7+51*3].reshape(-1, 3)).view(-1, 51 * 4)], dim=-1)


    ref_ig = ref_key_pos.view(-1,39,3).unsqueeze(2) - ref_obj_points.unsqueeze(1)

    human_reset = (ref_key_pos - key_pos).view(-1, 39, 3).norm(dim=-1).max(dim=-1)[0] > 0.5
    object_reset = (obj_points - ref_obj_points).norm(dim=-1).max(dim=-1)[0] > 0.5
    kinematic_reset = torch.logical_or(human_reset, object_reset)
    
    w_ig = 1

    weight_1 = (1 / torch.clamp((ig**2).sum(dim=-1), min=0.2))
    weight_1 = weight_1 / weight_1.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
    weight_2 = (1 / torch.clamp((ref_ig**2).sum(dim=-1), min=0.2))
    weight_2 = weight_2 / weight_2.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

    eig = ((ig - ref_ig)**2).sum(dim=-1) * (weight_1 + weight_2)  

    rig = torch.exp(-w_ig * (eig.sum(dim=-1).sum(dim=-1) * 0.5))
    
    reset_ig_1 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp(((ref_ig)**2).sum(dim=-1).sqrt(), min=0.5)).max(dim=-1)[0].max(dim=-1)[0] > 1.0
    reset_ig_2 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ig**2).sum(dim=-1).sqrt(), min=0.5)).max(dim=-1)[0].max(dim=-1)[0] > 1.0
    reset_ig = torch.logical_or(reset_ig_1, reset_ig_2)
    
    ep = torch.mean(((ref_key_pos - key_pos)**2).sum(dim=-1),dim=-1)
    rp = torch.exp(-ep*w['p'])
    weight = torch.ones_like(ref_dof_pos).to(ref_dof_pos.device)
    # weight[:, :11] = 0.5 
    er = torch.mean(((ref_dof_pos - dof_pos)**2) * weight,dim=-1)

    rr = torch.exp(-er*w['r'])
    
    # body pos vel reward
    epv = torch.mean((ref_body_pos_vel - body_pos_vel)**2,dim=-1)
    # print(ref_dof_vel, dof_vel)
    # epv = torch.mean(pos_vel ,dim=-1) # torch.zeros_like(ep)
    rpv = torch.exp(-epv*w['pv'])
    
    erv = torch.mean((ref_dof_vel - dof_vel)**2,dim=-1)
    rrv = torch.exp(-erv*w['rv'])

    rb = rp*rr*rpv*rrv
    # print('rb', rp, rr, rpv, rrv, rig)

    # object pos reward
    eop = torch.mean(((ref_local_obj_pos - local_obj_pos)**2),dim=-1) # * (1 - weight_h.max(dim=-1)[0])
    rop = torch.exp(-eop*w['op'])

    # object rot reward
    diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_local_obj_rot), local_obj_rot)
    diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
    diff = diff_angle.view(-1, 1)
    
    eor = torch.mean(huber_loss(diff, 3),dim=-1)
    ror = torch.exp(-eor*w['or'])

    # object pos vel reward
    eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel)**2,dim=-1)
    ropv = torch.exp(-eopv*w['opv'])

    # object rot vel reward
    eorv = torch.mean((ref_obj_rot_vel - obj_rot_vel)**2,dim=-1)
    rorv = torch.exp(-eorv*w['orv'])
    obj_energy = (object_pos_action.pow(2).mean(dim=-1).mul(-w['eg2']).exp()) * (object_rot_action.pow(2).mean(dim=-1).mul(-w['eg2']).exp())
    ro = rop*ror*ropv*rorv*obj_energy
    # print('ro', rop, ror, ropv, rorv, obj_energy)

    ref_contact = ref_human_contact
    ref_left_contact_hand_any = (ref_contact[:, -11] > 0.1).float()
    ref_right_contact_hand_any = (ref_contact[:, -1] > 0.1).float()

    contact_buf = contact_buf.clone()
    contact = (torch.abs(contact_buf).sum(dim=-1) > 0.1).float()
    left_contact_hand_any = (contact[:, -11] > 0.1).float()
    right_contact_hand_any = (contact[:, -1] > 0.1).float()
    # print(left_contact_hand_any.shape, ref_left_contact_hand_any.shape)
    # w_4[:, 0:13] = 0.1
    w_cg = 1.0

    # Compute contact error with special handling for feet (indices 2 and 5)
    # For feet: only penalize when ref_contact=0 but actual contact=1 (false positive)
    # For other body parts: penalize any mismatch
    contact_error = torch.abs(contact - ref_contact)  # [batch, num_bodies]

    # Feet indices: 2 (left ankle) and 5 (right ankle)
    feet_indices = [2, 5]

    # For feet: mask out errors when ref_contact=1 (allow contact when reference has it)
    for foot_idx in feet_indices:
        # Only count error when ref_contact is 0 (no contact expected)
        contact_error[:, foot_idx] = contact_error[:, foot_idx] * (1.0 - ref_contact[:, foot_idx])

    ecg = contact_error.mean(dim=-1)
    # print(contact, ref_contact)
    rcg = torch.exp(-ecg*w_cg)
    contact_reset = torch.stack([ 
                               torch.abs(ref_left_contact_hand_any - left_contact_hand_any) * ref_left_contact_hand_any, 
                               torch.abs(ref_right_contact_hand_any - right_contact_hand_any) * ref_right_contact_hand_any,
                               ], dim=-1)
    # print(contact_reset.shape)
    energy_ids = [_ for _ in range(14) if _ != 2 and _ != 5] # not counting the contact of feet
    contact_all = contact_buf[:,energy_ids, :].clone().abs().sum(dim=-1).max(dim=-1)[0]
    contact_energy = contact_all.pow(2).mul(-w['eg3']).exp()

    rcg = rcg*contact_energy
    # print('rcg', rcg, contact_energy)

    reward = rb*ro*rig*rcg
    # print(rb, ro, rig, rcg)

    metric_1 = (ref_key_pos - key_pos).norm(dim=-1).mean(dim=-1)
    metric_2 = (obj_points - ref_obj_points).norm(dim=-1).mean(dim=-1)
    tracking_reward = reward * 1.6
    return tracking_reward, reset_ig, contact_reset, kinematic_reset, metric_1, metric_2

def build_hoi_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, target_states, target_contact_buf, contact_buf, object_points, body_rot, body_vel, body_rot_vel, _key_body_ids_gt, _contact_body_ids_gt):

    contact = torch.any(torch.abs(contact_buf) > 0.1, dim=-1).float()
    target_contact = torch.any(torch.abs(target_contact_buf) > 0.1, dim=-1).float().unsqueeze(1)

    tar_pos = target_states[:, 0:3]
    tar_rot = target_states[:, 3:7]
    obj_rot_extend = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    object_points_extend = object_points.view(-1, 3)
    obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)
    ig = compute_sdf(key_body_pos, obj_points).view(-1, 3)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pos.shape[1], 1).view(-1, 4)
    ig = quat_rotate(heading_rot_extend, ig).view(tar_pos.shape[0], -1)    
    obs = torch.cat((root_pos, 
                     root_rot,
                     root_vel,
                     root_ang_vel, 
                     dof_pos, 
                     dof_vel, 
                     target_states, 
                     key_body_pos.contiguous().view(-1,key_body_pos.shape[1]*key_body_pos.shape[2]), 
                     body_rot.view(-1, 39*4), 
                     body_vel.view(-1,key_body_pos.shape[1]*key_body_pos.shape[2]), 
                     body_rot_vel.view(-1, 39*3),
                     contact,
                     ig), dim=-1)
    return obs

def compute_humanoid_reset(reset_buf, progress_buf, obs_buf, contact_buf, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, termination_heights_init, hoi_ref, hoi_obs, start_times, rollout_length, reset_ig, contact_reset, is_stand_still):
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        body_height = rigid_body_pos[:, 0, 2] # root height

        body_fall = body_height < termination_heights# [4096] 
        has_failed = body_fall.clone()
        has_failed *= (progress_buf > 20 + start_times)
        reset_ig *= (progress_buf > 20 + start_times)
        contact_reset *= (progress_buf > 20 + start_times)
        invalid_obs = ~torch.isfinite(obs_buf)
        invalid_batches = torch.any(invalid_obs, dim=1)
        if torch.any(invalid_obs):
            raise Exception("invalid observation")

        # For stand-still tasks, only terminate on falls and invalid observations
        # For tracking tasks, also terminate on reset_ig and contact_reset
        termination_conditions = torch.logical_or(invalid_batches, has_failed)
        termination_conditions = torch.where(is_stand_still,
                                             termination_conditions,
                                             torch.logical_or(termination_conditions, torch.logical_or(reset_ig, contact_reset)))
        terminated = torch.where(termination_conditions, torch.ones_like(reset_buf), terminated)
    reset = torch.where(torch.logical_or(progress_buf >= max_episode_length-1, progress_buf - start_times >= rollout_length-1), torch.ones_like(reset_buf), terminated)

    return reset, terminated

def _rand_vec(vec, scale=0.1):
    return vec + (2 * torch.rand_like(vec) - 1) * scale


def compute_obj_observations(root_states, tar_states, object_points, ref_obs):
    root_pos = _rand_vec(root_states[:, 0:3], 0.)
    root_rot = _rand_vec(root_states[:, 3:7], 0.)

    tar_pos = _rand_vec(tar_states[:, 0:3], 0.0)
    tar_rot = _rand_vec(tar_states[:, 3:7], 0.)
    tar_vel = _rand_vec(tar_states[:, 7:10], 0.)
    tar_ang_vel = tar_states[:, 10:13]

    obj_rot_extend = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    object_points_extend = object_points.view(-1, 3)
    obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    _ref_obj_pos = ref_obs[:,71:74]
    diff_global_obj_pos = _ref_obj_pos - tar_pos
    diff_local_obj_pos_flat = torch_utils.quat_rotate(heading_rot, diff_global_obj_pos)

    local_ref_obj_pos = _ref_obj_pos - root_pos  # preserves the body position
    local_ref_obj_pos = torch_utils.quat_rotate(heading_rot, local_ref_obj_pos)

    ref_obj_rot = ref_obs[:,74:78]
    diff_global_obj_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_obj_rot), tar_rot)
    diff_local_obj_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)), heading_inv_rot)  # Need to be change of basis
    diff_local_obj_rot_obs = torch_utils.quat_to_tan_norm(diff_local_obj_rot_flat)

    local_ref_obj_rot = torch_utils.quat_mul(heading_rot, ref_obj_rot)
    local_ref_obj_rot = torch_utils.quat_to_tan_norm(local_ref_obj_rot)

    ref_obj_vel = ref_obs[:,78:81]
    diff_global_vel = ref_obj_vel - tar_vel
    diff_local_vel = torch_utils.quat_rotate(heading_rot, diff_global_vel)

    ref_obj_ang_vel = ref_obs[:,81:84]
    diff_global_ang_vel = ref_obj_ang_vel - tar_ang_vel
    diff_local_ang_vel = torch_utils.quat_rotate(heading_rot, diff_global_ang_vel)

    # local_tar_ang_vel = local_tar_ang_vel * 0
    # # local_tar_vel = local_tar_vel * 0
    # # diff_local_vel = diff_local_vel * 0
    # diff_local_ang_vel = diff_local_ang_vel * 0
    obs = torch.cat([local_tar_vel, local_tar_ang_vel, diff_local_obj_pos_flat, diff_local_obj_rot_obs, diff_local_vel, diff_local_ang_vel], dim=-1)
    return obs, obj_points
