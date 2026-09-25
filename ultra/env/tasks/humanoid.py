# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np
import torch
import os

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils
from env.tasks.base_task import BaseTask



class Humanoid_SMPLX(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.ref_hoi_obs_size = 324 + len(self.cfg["env"]["keyBodies"])*9 + 52 + 3 + 52 * 4 + 52 * 3
        self.ref_contact_obs_size = len(self.cfg["env"]["contactBodies"])*3
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        
        key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(key_bodies)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
         
        super().__init__(cfg=self.cfg)
        
        self.dt = self.control_freq_inv * sim_params.dt
        
        self.common_step_counter = 10000 * 32
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        str_rng = self.cfg['domain_rand']['motor_strength_range']
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(
            2,
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        ) + str_rng[0]
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)


        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()
        
        self._humanoid_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:] = 0
        self._initial_humanoid_root_states[:, 6:7] = 1

        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]
        
        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self._rigid_body_pos = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.action_history_buf = torch.zeros(
            self.num_envs,
            self.cfg['domain_rand']['action_buf_len'],
            self._num_actions,
            device=self.device,
            dtype=torch.float,
        )

        self._build_termination_heights()
        
        contact_bodies = self.cfg["env"]["contactBodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)
        self.prev_pd_tar = None
        self.prev_torque = None
        if self.viewer != None:
            self._init_camera()
            
        return

    def get_obs_size(self):
        return self._num_obs

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        return

    def reset(self, env_ids=None):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)
        return

    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
        return

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _setup_character_props(self, key_bodies):
        self._dof_obs_size = (51)*3
        self._num_actions = (51)*3
        self._num_obs = self.cfg["env"]["numObs"]

        return

    def _build_termination_heights(self):
        self._termination_heights = 0.15
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return

    def get_num_amp_obs(self):
        return self.ref_hoi_obs_size
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.robot_type

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_humanoid_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_humanoid_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
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
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        asset_file = self.robot_type
        char_h = 0.89

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter, segmentation_id)

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.75, 0.54, 0.3))

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        self.humanoid_handles.append(humanoid_handle)

        return

    def _build_pd_action_offset_scale(self):
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = np.maximum(lim_high, -lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)
        self._pd_action_offset[self._pd_action_offset.abs() > 1e-5] = 0.
        return

    def _get_humanoid_collision_filter(self):
        return 1

    def _compute_reward(self, actions):
        return
    
    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf, self.obs_buf,
                                                   self._contact_forces,
                                                   self._rigid_body_pos, self.max_episode_length[self.data_id],
                                                   self._enable_early_termination, self._termination_heights, self._curr_ref_obs, self._curr_obs, self.start_times, self.rollout_length, self._reset_ig, torch.any(self.contact_reset > 10, dim=-1)
                                                   )
        return

    def _refresh_sim_tensors(self):
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
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
    
    def _compute_observations_iter(self, env_ids=None, delta_t=1):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
            ts = self.progress_buf.clone()
            self._curr_ref_obs = self.hoi_data[self.data_id[env_ids], ts].clone() 
            next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]]-1)
            ref_obs = self.hoi_data[self.data_id[env_ids], next_ts].clone()
            obs = self._compute_humanoid_obs(env_ids, ref_obs, next_ts)
            task_obs, obj_points = self._compute_task_obs(env_ids, ref_obs)
            obs = torch.cat([obs, task_obs], dim=-1)
            key_body_pose = self._rigid_body_pos.clone() 
            ig = compute_sdf(key_body_pose, obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(self._rigid_body_rot[:, 0, :])
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1], 1).view(-1, 4)
            ig = quat_rotate(heading_rot_extend, ig).view(env_ids.shape[0], -1, 3)
            ig_norm = ig.norm(dim=-1, keepdim=True)
            ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
            ig = ig_all[:, self._key_body_ids, :].view(env_ids.shape[0], -1)
            ig_all = ig_all.view(env_ids.shape[0], -1)    
            len_key_body_ids = len(self._key_body_ids)
            ref_ig = ref_obs[:, 326+len_key_body_ids*3+1+52:326+len_key_body_ids*3+1+52+len_key_body_ids*3].view(env_ids.shape[0], len_key_body_ids, 3)
            ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
            ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()  
            ref_ig = ref_ig.view(env_ids.shape[0], -1)          
            return torch.cat((obs,ig_all,ref_ig-ig),dim=-1)

        else:
            ts = self.progress_buf[env_ids].clone()
            self._curr_ref_obs[env_ids] = self.hoi_data[self.data_id[env_ids], ts].clone() 
            next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]]-1)
            ref_obs = self.hoi_data[self.data_id[env_ids], next_ts].clone()
            obs = self._compute_humanoid_obs(env_ids, ref_obs, next_ts)
            task_obs, obj_points = self._compute_task_obs(env_ids, ref_obs)
            obs = torch.cat([obs, task_obs], dim=-1)
            key_body_pose = self._rigid_body_pos[env_ids].clone() 
            ig = compute_sdf(key_body_pose, obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(self._rigid_body_rot[env_ids][:, 0, :])
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1], 1).view(-1, 4)
            ig = quat_rotate(heading_rot_extend, ig).view(env_ids.shape[0], -1, 3)
            ig_norm = ig.norm(dim=-1, keepdim=True)
            ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
            ig = ig_all[:, self._key_body_ids, :].view(env_ids.shape[0], -1)
            ig_all = ig_all.view(env_ids.shape[0], -1)    
            len_key_body_ids = len(self._key_body_ids)
            ref_ig = ref_obs[:, 326+len_key_body_ids*3+1+52:326+len_key_body_ids*3+1+52+len_key_body_ids*3].view(env_ids.shape[0], len_key_body_ids, 3)
            ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
            ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()  
            ref_ig = ref_ig.view(env_ids.shape[0], -1)          
            return torch.cat((obs,ig_all,ref_ig-ig),dim=-1)
            

        return

    def _compute_observations(self, env_ids=None):
        if (env_ids is None):
            self.obs_buf[:] = torch.cat((self._compute_observations_iter(None, 1), self._compute_observations_iter(None, 16)), dim=-1)

        else:
            self.obs_buf[env_ids] = torch.cat((self._compute_observations_iter(env_ids, 1), self._compute_observations_iter(env_ids, 16)), dim=-1)
            
        # assert torch.isnan(self.obs_buf).any() == False
        return

    def _compute_humanoid_obs(self, env_ids=None, ref_obs=None, next_ts=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            contact_forces = self._contact_forces
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            contact_forces = self._contact_forces[env_ids]
        
        obs = compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                self._root_height_obs,
                                                contact_forces, self._contact_body_ids, ref_obs, self._key_body_ids)

        return obs

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg["control"]["action_scale"]
        # print(f"dof_vel: {self._dof_vel[0]}")
        # actions_scaled = self._action_to_pd_targets(actions)
        if self.cfg["control"]["control_type"] == "P":
            if not (self.cfg["domain_rand"]["randomize_motor"] and self.cfg["domain_rand"]["domain_rand_general"]):  # TODO add strength to gain directly
                torques = (
                    self.p_gains * (actions_scaled + self._initial_dof_pos - self._dof_pos)
                    - self.d_gains * self._dof_vel
                )
            else:
                torques = (
                    self.motor_strength[0]
                    * self.p_gains
                    * (actions_scaled + self._initial_dof_pos - self._dof_pos)
                    - self.motor_strength[1] * self.d_gains * self._dof_vel
                )

        elif self.cfg["control"]["control_type"] == "V":
            torques = (
                self.p_gains * (actions_scaled - self._dof_vel)
                - self.d_gains * (self._dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif self.cfg["control"]["control_type"] == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {self.cfg['control']['control_type']}")
        # return torques
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        # if self.prev_torque is not None:
        #     torques = 0.1 * torques + 0.9 * self.prev_torque
        # self.prev_torque = torques
        return torques

    def _reset_actors(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids] 
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids] 
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        return

    def pre_physics_step(self, actions):
        self.last_actions = self.actions.clone() if hasattr(self, 'actions') else torch.zeros_like(actions)
        self.last_dof_vel = self._dof_vel.clone() if hasattr(self, '_dof_vel') else torch.zeros_like(self._dof_vel)
        self.last_dof_pos = self._dof_pos.clone() if hasattr(self, '_dof_pos') else torch.zeros_like(self._dof_pos)
        self.last_contact_forces = self._contact_forces.clone() if hasattr(self, '_contact_forces') else torch.zeros_like(self._contact_forces)
        self.actions = actions.to(self.device).clone()
        self.action_history_buf = torch.cat(
            [self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1
        )
        if self.cfg['domain_rand']['action_delay'] and self.cfg['domain_rand']['domain_rand_general']:
            if self.common_step_counter <= 1000 * 32:
                self.delay = torch.tensor(0, device=self.device, dtype=torch.float)
            else:
                self.delay = torch.tensor(
                    np.random.randint(2), device=self.device, dtype=torch.float
                )
            indices = -self.delay - 1
            self.actions = self.action_history_buf[:, indices.long()].to(self.device).clone()
        # self.actions = 0.1 * actions.to(self.device).clone() + 0.9 * last_actions if last_actions is not None else actions.to(self.device).clone()
        # if (self._pd_control):
        #     pd_tar = self._action_to_pd_targets(self.actions)
        #     pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
        #     self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        # else:
        #     forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
        #     force_tensor = gymtorch.unwrap_tensor(forces)
        #     self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)       

        return
    
    def _physics_step(self):
        self.render()
        if hasattr(self, 'dof_vel_list'):
            self.dof_vel_list = [self.dof_vel_list[-1]]
        else:
            self.dof_vel_list = []
        if hasattr(self, 'torques_list'):
            self.torques_list = [self.torques_list[-1]]
        else:
            self.torques_list = []
        if hasattr(self, 'ang_vel_list'):
            self.ang_vel_list = [self.ang_vel_list[-1]]
        else:
            self.ang_vel_list = []
        for i in range(self.control_freq_inv):
            self.torques = self._compute_torques(self.actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self._refresh_sim_tensors()
            self.dof_vel_list.append(self._dof_vel.clone())
            self.ang_vel_list.append(self._rigid_body_ang_vel[:,0,:].clone())
            self.torques_list.append(self.torques.clone())
        return
    
    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_vel = self.cfg['domain_rand']['max_push_vel_xy']
        max_ang = self.cfg['domain_rand'].get('max_push_ang_vel', max_vel)

        # Curriculum learning: gradually increase push magnitude
        curriculum_steps = 1000 * 32  # Number of steps to reach full push strength
        curriculum_progress = min(self.common_step_counter / curriculum_steps, 1.0)
        max_vel_scaled = max_vel * curriculum_progress
        max_ang_scaled = max_ang * curriculum_progress

        self._humanoid_root_states[:, 7:9] = torch_rand_float(
            -max_vel_scaled, max_vel_scaled, (self.num_envs, 2), device=self.device
        )  # lin vel x/y
        if hasattr(self, "_target_states"):
            self._target_states[:, 7:9] = torch_rand_float(
                -max_vel_scaled, max_vel_scaled, (self.num_envs, 2), device=self.device
            )  # obj lin vel x/y
            self._target_states[:, 10:13] = torch_rand_float(
                -max_ang_scaled, max_ang_scaled, (self.num_envs, 3), device=self.device
            )  # obj ang vel xyz
        # self.root_states[:, 7:9] = -3.5 + 0*torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        # self.root_states[:, 8] = 0
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))

        # Reset the steps_since_push counter for all environments
        if hasattr(self, 'steps_since_push'):
            self.steps_since_push[:] = 0

    def _randomize_gravity(self, external_force=None):
        if self.cfg['domain_rand']['randomize_gravity'] and external_force is None:
            min_gravity, max_gravity = self.cfg['domain_rand']['gravity_range']
            external_force = (
                torch.rand(3, dtype=torch.float, device=self.device, requires_grad=False)
                * (max_gravity - min_gravity)
                + min_gravity
            )

        sim_params = self.gym.get_sim_params(self.sim)
        if external_force is None:
            gravity = torch.Tensor([0, 0, -9.81]).to(self.device)
        else:
            gravity = external_force + torch.Tensor([0, 0, -9.81]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)

    def post_physics_step(self):
        self.progress_buf += 1
        self.common_step_counter += 1
        self.episode_length_buf += 1

        # Increment steps_since_push counter
        if hasattr(self, 'steps_since_push'):
            self.steps_since_push += 1

        self._refresh_sim_tensors()
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        #extra calc of self._curr_hoi_obs_buf, for correct calculate of imitation reward
        if self.cfg['domain_rand']['push_robots'] and self.cfg['domain_rand']['domain_rand_general'] and (
            self.common_step_counter % (self.cfg['domain_rand']['push_interval_s'] // self.dt) == 0
        ):
            self._push_robots()

        if self.cfg['domain_rand']['randomize_gravity'] and self.cfg['domain_rand']['domain_rand_general'] and (
            self.common_step_counter % (self.cfg['domain_rand']['gravity_rand_interval_s'] // self.dt) == 0
        ):
            self._randomize_gravity()
        
        self._compute_hoi_observations(env_ids)

        self._compute_observations(env_ids)
        self._compute_reward(self.actions)
        self._compute_reset()
      
        self.last_actions[self.reset_buf > 0] = 0.0
        self.last_dof_vel[self.reset_buf > 0] = 0.0
        self.last_dof_pos[self.reset_buf > 0] = 0.0
        self.actions[self.reset_buf > 0] = 0.0
        self.action_history_buf[self.reset_buf > 0,: ,: ] = 0.0
        self.episode_length_buf[self.reset_buf > 0] = 0
        self.last_contact_forces[self.reset_buf > 0] = torch.zeros_like(self._contact_forces[self.reset_buf > 0])
        # if hasattr(self, '_prev_contact'):
        #     # Initialize on first call
        #     self._prev_contact[self.reset_buf > 0] = torch.zeros_like(self._contact_forces[self.reset_buf > 0])
        
        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self, sync_frame_time=False):
        if self.viewer:
            self._update_camera()

        super().render(sync_frame_time)
        return

    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
                              self._cam_prev_char_pos[1] - 3.0, 
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        return
    

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return

def _rand_vec(vec, scale=0.1):
    return vec + (2 * torch.rand_like(vec) - 1) * scale

def compute_obj_observations(root_states, tar_states, object_points, ref_obs):
    root_pos = _rand_vec(root_states[:, 0:3], 0.01)
    root_rot = _rand_vec(root_states[:, 3:7], 0.1)

    tar_pos = _rand_vec(tar_states[:, 0:3], 0.01)
    tar_rot = _rand_vec(tar_states[:, 3:7], 0.1)
    tar_vel = _rand_vec(tar_states[:, 7:10], 0.1)
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

    _ref_obj_pos = ref_obs[:,313:316]
    diff_global_obj_pos = _ref_obj_pos - tar_pos
    diff_local_obj_pos_flat = torch_utils.quat_rotate(heading_rot, diff_global_obj_pos)

    local_ref_obj_pos = _ref_obj_pos - root_pos  # preserves the body position
    local_ref_obj_pos = torch_utils.quat_rotate(heading_rot, local_ref_obj_pos)

    ref_obj_rot = ref_obs[:,316:320]
    diff_global_obj_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_obj_rot), tar_rot)
    diff_local_obj_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)), heading_inv_rot)  # Need to be change of basis
    diff_local_obj_rot_obs = torch_utils.quat_to_tan_norm(diff_local_obj_rot_flat)

    local_ref_obj_rot = torch_utils.quat_mul(heading_rot, ref_obj_rot)
    local_ref_obj_rot = torch_utils.quat_to_tan_norm(local_ref_obj_rot)

    ref_obj_vel = ref_obs[:, 320:323]
    diff_global_vel = ref_obj_vel - tar_vel
    diff_local_vel = torch_utils.quat_rotate(heading_rot, diff_global_vel)

    ref_obj_ang_vel = ref_obs[:, 323:326]
    diff_global_ang_vel = ref_obj_ang_vel - tar_ang_vel
    diff_local_ang_vel = torch_utils.quat_rotate(heading_rot, diff_global_ang_vel)

    # local_tar_ang_vel = local_tar_ang_vel * 0
    # # local_tar_vel = local_tar_vel * 0
    # # diff_local_vel = diff_local_vel * 0
    # diff_local_ang_vel = diff_local_ang_vel * 0
    obs = torch.cat([local_tar_vel, local_tar_ang_vel, diff_local_obj_pos_flat, diff_local_obj_rot_obs, diff_local_vel, diff_local_ang_vel], dim=-1)
    return obs, obj_points



@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, contact_body_ids, ref_obs, key_body_ids):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor, Tensor, Tensor) -> Tensor
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
    
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand_no_hand = heading_rot_expand.repeat((1, 22, 1))
    flat_heading_rot_no_hand = heading_rot_expand_no_hand.reshape(heading_rot_expand_no_hand.shape[0] * heading_rot_expand_no_hand.shape[1], 
                                               heading_rot_expand_no_hand.shape[2])

    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
    heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_inv_rot = heading_inv_rot_expand.reshape(heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                               heading_inv_rot_expand.shape[2])

    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
    heading_inv_rot_expand_no_hand = heading_inv_rot_expand.repeat((1, 22, 1))
    flat_heading_inv_rot_no_hand = heading_inv_rot_expand_no_hand.reshape(heading_inv_rot_expand_no_hand.shape[0] * heading_inv_rot_expand_no_hand.shape[1], 
                                               heading_inv_rot_expand_no_hand.shape[2])
    
    _ref_body_pos = ref_obs[:,326:326+len_keypos*3].view(-1, len_keypos, 3)
    _body_pos = body_pos[:, key_body_ids, :]

    diff_global_body_pos = _ref_body_pos - _body_pos
    diff_local_body_pos_flat = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_body_pos.view(-1, 3)).view(-1, len_keypos * 3)
    
    local_ref_body_pos = _body_pos - root_pos.unsqueeze(1)  # preserves the body position
    local_ref_body_pos = torch_utils.quat_rotate(flat_heading_rot_2, local_ref_body_pos.view(-1, 3)).view(-1, len_keypos * 3)
   
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    ref_body_rot = ref_obs[:, 326+len_keypos*3+1+52+len_keypos*3: 326+len_keypos*3+1+52+len_keypos*3+52*4]
    ref_body_rot_no_hand = torch.cat((ref_body_rot[:, :18*4], ref_body_rot[:, 33*4:37*4]), dim=-1) 
    body_rot_no_hand = torch.cat((body_rot[:, :18], body_rot[:, 33:37]), dim=1)
    diff_global_body_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)), body_rot_no_hand.reshape(-1, 4))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(flat_heading_rot_no_hand, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot_no_hand)
    diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
    diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot_no_hand.shape[0], body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1])

    local_ref_body_rot = torch_utils.quat_mul(flat_heading_rot_no_hand, ref_body_rot_no_hand.reshape(-1, 4))
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot_no_hand.shape[0], -1)

    ref_body_vel = ref_obs[:, 326+len_keypos*3+1+52+len_keypos*3+52*4:326+len_keypos*3+1+52+len_keypos*3+52*4+len_keypos*3].view(-1, len_keypos, 3)
    _body_vel = body_vel[:, key_body_ids, :]
    diff_global_vel = ref_body_vel - _body_vel
    diff_local_vel = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, len_keypos * 3)

    ref_body_ang_vel = ref_obs[:, 326+len_keypos*3+1+52+len_keypos*3+52*4+len_keypos*3:326+len_keypos*3+1+52+len_keypos*3+52*4+len_keypos*3+52*3]
    ref_body_ang_vel_no_hand = torch.cat((ref_body_ang_vel[:, :18*3], ref_body_ang_vel[:, 33*3:37*3]), dim=-1)
    body_ang_vel_no_hand = torch.cat((body_ang_vel[:, :18], body_ang_vel[:, 33:37]), dim=1)
    diff_global_ang_vel = ref_body_ang_vel_no_hand.view(-1, 22, 3) - body_ang_vel_no_hand
    diff_local_ang_vel = torch_utils.quat_rotate(flat_heading_rot_no_hand, diff_global_ang_vel.view(-1, 3)).view(-1, 22 * 3)

    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    body_contact_buf = contact_forces[:, contact_body_ids, :].clone() #.view(contact_forces.shape[0],-1)
    contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()
    ref_body_contact = ref_obs[:,326+len_keypos*3+1:326+len_keypos*3+1+52][:, contact_body_ids]
    diff_body_contact = ref_body_contact * ((ref_body_contact + 1) / 2 - contact)

    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, contact, diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact, local_ref_body_pos, local_ref_body_rot, diff_local_vel, diff_local_ang_vel), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, obs_buf, contact_buf, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, hoi_ref, hoi_obs, start_times, rollout_length, reset_ig, contact_reset):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        body_height = rigid_body_pos[:, 0, 2] # root height
        body_fall = body_height < termination_heights# [4096] 
        has_failed = body_fall.clone()
        has_failed *= (progress_buf > 1)
        reset_ig *= (progress_buf > 1 + start_times)
        contact_reset *= (progress_buf > 1 + start_times)
        invalid_obs = ~torch.isfinite(obs_buf)  # True where obs is NaN or infinite
        invalid_batches = torch.any(invalid_obs, dim=1)  # Check if any invalid number in each batch (B, N)
        if torch.any(invalid_obs):
            print("invalid observation")
            raise Exception("invalid observation")
            
        terminated = torch.where(torch.logical_or(invalid_batches, torch.logical_or(has_failed, torch.logical_or(reset_ig, contact_reset))), torch.ones_like(reset_buf), terminated)
    reset = torch.where(torch.logical_or(progress_buf >= max_episode_length-1, progress_buf - start_times >= rollout_length-1), torch.ones_like(reset_buf), terminated)

    return reset, terminated

@torch.jit.script
def compute_sdf(points1, points2):
    # type: (Tensor, Tensor) -> Tensor
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)
    B_indices, N_indices = torch.meshgrid(torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing='ij')
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    return min_dis_mat
