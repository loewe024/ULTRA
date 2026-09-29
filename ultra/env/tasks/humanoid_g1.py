import torch
import os

from utils.gym_torch_utils import *
import torch.nn.functional as F
from utils import torch_utils
from env.tasks.humanoid import *
from isaac.legacy_layout import G1_DAMPING, G1_EFFORT, G1_STIFFNESS


class Humanoid_G1(Humanoid_SMPLX):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self._key_body_ids_gt = to_torch(cfg["env"]["keyIndex"], device=cfg.sim.device, dtype=torch.long)
        self._contact_body_ids_gt = to_torch(cfg["env"]["contactIndex"], device=cfg.sim.device, dtype=torch.long)
        super().__init__(cfg, render_mode, **kwargs)
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
    
    def _setup_env_properties(self):
        body_names = LEGACY_BODY_NAMES
        right_foot_idx = self._find_body_index("right_ankle_roll_link")
        left_foot_idx = self._find_body_index("left_ankle_roll_link")
        
        self.feet_indices = torch.zeros(
            2, dtype=torch.long, device=self.device, requires_grad=False
        )
        penalized_contact_names = []
        penalize_contacts_on = ["shoulder", "elbow", "hip"]
        for name in penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        self.penalized_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self._find_body_index(penalized_contact_names[i])
        self.feet_indices[0] = left_foot_idx
        self.feet_indices[1] = right_foot_idx
        knee_names = [s for s in body_names if 'knee' in s]
        self.knee_indices = torch.zeros(
            len(knee_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(knee_names)):
            self.knee_indices[i] = self._find_body_index(knee_names[i])
        self.num_humanoid_bodies = self.num_bodies

        self.torso_idx = self._find_body_index("torso_link")

        lower, upper = self._dof_limits()
        self.dof_limits_lower = torch.minimum(lower, upper)
        self.dof_limits_upper = torch.maximum(lower, upper)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

            # Stiffness/damping/armature/effort limits of the drives are set in isaac/scene_cfg.py.
            action_scale = [e/k for e, k in zip(G1_EFFORT, G1_STIFFNESS)]
            self.p_gains = torch.tensor(G1_STIFFNESS, device=self.device, dtype=torch.float32)
            self.d_gains = torch.tensor(G1_DAMPING, device=self.device, dtype=torch.float32)
            self.torque_limits = torch.tensor(G1_EFFORT, device=self.device, dtype=torch.float32) * 0.8
            self.action_scale = torch.tensor(action_scale, device=self.device, dtype=torch.float32)

        self._randomize_rigid_shape_props()
        self._randomize_rigid_body_props()
        return
    
    def _randomize_rigid_shape_props(self):
        """Friction of all humanoid shapes, one value per environment drawn from 64 buckets.

        NOTE: The default friction is 1.0 (the IsaacGym default the policies were trained with).
        """
        friction = torch.ones(self.num_envs)
        if self.cfg['domain_rand']['randomize_friction'] and self.cfg['domain_rand']['domain_rand_general']:
            # prepare friction randomization
            friction_range = self.cfg['domain_rand']['friction_range']
            num_buckets = 64
            bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
            friction_buckets = torch_rand_float(
                friction_range[0], friction_range[1], (num_buckets, 1), device="cpu"
            )
            self.friction_coeffs = friction_buckets[bucket_ids]
            friction = self.friction_coeffs.view(-1)
        self._set_shape_materials(self.robot, friction)
        return
    
    def _randomize_rigid_body_props(self):
        """Torso mass and center-of-mass randomization (once, at start-up)."""
        dr = self.cfg['domain_rand']
        randomize_mass = dr['randomize_base_mass'] and dr['domain_rand_general']
        randomize_com = dr['randomize_base_com'] and dr['domain_rand_general']
        if not (randomize_mass or randomize_com):
            return
        view = self.robot.root_physx_view
        torso = int(self._body_sim_ids[self.torso_idx])
        env_ids = self._all_env_ids_cpu
        if randomize_mass:
            rng_mass = dr['added_mass_range']
            masses = view.get_masses()
            default_mass = masses[:, torso].clone()
            masses[:, torso] += torch.empty(self.num_envs).uniform_(rng_mass[0], rng_mass[1])
            view.set_masses(masses, env_ids)
            # IsaacGym recomputed the inertia for the new mass (recomputeInertia=True)
            inertias = view.get_inertias()
            inertias[:, torso] *= (masses[:, torso] / default_mass)[:, None]
            view.set_inertias(inertias, env_ids)
        if randomize_com:
            rng_com = dr['added_com_range']
            coms = view.get_coms().clone()
            coms[:, torso, :3] += torch.empty(self.num_envs, 3).uniform_(rng_com[0], rng_com[1])
            view.set_coms(coms, env_ids)
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
