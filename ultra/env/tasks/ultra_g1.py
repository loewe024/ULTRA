import torch
from pathlib import Path

from utils.gym_torch_utils import *

from utils import torch_utils
import torch.nn.functional as F
from env.tasks.humanoid_g1 import *
from env.tasks.ultra import Ultra


class UltraG1(Humanoid_G1, Ultra):

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.scaling = cfg['env']['scaling']
        self.init_root_height = cfg['env']['initRootHeight']
        self.init_dof = torch.cat([to_torch([-0.1, 0, 0.0, 0.3, -0.2, 0, -0.1, 0, 0.0, 0.3, -0.2, 0, 0, 0, 0, 
                                             0, 0, 0, 0.5], device=self.device, dtype=torch.float),
                                   to_torch([0] * (self._num_actions_hand + self._num_actions_wrist), device=self.device, dtype=torch.float),
                                   to_torch([0, 0, 0, 0.5], device=self.device, dtype=torch.float),
                                   to_torch([0] * (self._num_actions_hand + self._num_actions_wrist), device=self.device, dtype=torch.float)])
        return

    def _physics_step(self):
        if self.cfg["env"].get("retargetExportPath"):
            if not hasattr(self, "_export_frames"):
                self._export_frames = []
            if not self._export_frames:
                self._capture_retarget_frame()
        super()._physics_step()
        if self.cfg["env"].get("retargetExportPath"):
            self._capture_retarget_frame()

    def post_physics_step(self):
        super().post_physics_step()
        if self.cfg["env"].get("retargetExportPath") and bool(self.reset_buf[0]):
            self._save_retarget_rollout()

    def _capture_retarget_frame(self):
        contact = torch.any(torch.abs(self._contact_forces[0]) > 0.1, dim=-1).float()
        frame = torch.cat((
            self._humanoid_root_states[0],
            self._dof_pos[0], self._dof_vel[0],
            self._target_states[0],
            self._rigid_body_pos[0].reshape(-1),
            self._rigid_body_rot[0].reshape(-1),
            self._rigid_body_vel[0].reshape(-1),
            self._rigid_body_ang_vel[0].reshape(-1),
            contact,
        )).detach().cpu()
        self._export_frames.append(frame)

    def _save_retarget_rollout(self):
        # UltraG1 prepends 30 standing frames to each reference.
        frames = torch.stack(self._export_frames[30:])
        destination = Path(self.cfg["env"]["retargetExportPath"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(frames, destination)
        print("ULTRA_RETARGET_EXPORT", destination, tuple(frames.shape), flush=True)
        self._export_frames = []

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        return

    def _compute_reset(self):
        super()._compute_reset()


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
            hoi_data = torch.load(data_path)[1:]
            loaded_dict['hoi_data'] = hoi_data.detach().to('cuda')

            self.max_episode_length.append(loaded_dict['hoi_data'].shape[0])
            self.fps_data = new_fps = 60.

            loaded_dict['root_pos'] = interp_time_series(loaded_dict['hoi_data'][:, 0:3].clone(), factor=2, mode='linear')
            xyz = torch.tensor(self.cfg['env'].get('sparseXYZMultiplier', [1., 1., 1.]), device=self.device, dtype=torch.float32)
            loaded_dict['root_pos'] *= xyz
            loaded_dict['root_pos_vel'] = (loaded_dict['root_pos'][1:,:].clone() - loaded_dict['root_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['root_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_pos_vel'].shape[-1])).to('cuda'),loaded_dict['root_pos_vel']),dim=0)

            loaded_dict['root_rot'] = interp_time_series(loaded_dict['hoi_data'][:, 3:7].clone(), factor=2, mode='linear')

            root_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['root_rot'])
            loaded_dict['root_rot_vel'] = (root_rot_exp_map[1:,:].clone() - root_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['root_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_rot_vel'].shape[-1])).to('cuda'),loaded_dict['root_rot_vel']),dim=0)

            loaded_dict['dof_pos'] = interp_time_series(loaded_dict['hoi_data'][:, 9:9+153].clone(), factor=2, mode='linear')

            loaded_dict['dof_pos_vel'] = []

            loaded_dict['dof_pos_vel'] = (loaded_dict['dof_pos'][1:,:].clone() - loaded_dict['dof_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['dof_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['dof_pos_vel'].shape[-1])).to('cuda'),loaded_dict['dof_pos_vel']),dim=0)

            body_pos_raw = loaded_dict['hoi_data'][:, 162: 162+52*3].clone().view(self.max_episode_length[-1], 52, 3)
            loaded_dict['body_pos'] = interp_time_series(body_pos_raw, factor=2, mode='linear')

            loaded_dict['key_body_pos'] = loaded_dict['body_pos'][:, self._key_body_ids_gt, :].reshape(loaded_dict['body_pos'].shape[0],-1).clone()

            loaded_dict['key_body_pos'] = (loaded_dict['key_body_pos'].reshape(-1, len(self._key_body_ids_gt), 3) * xyz).reshape(-1, len(self._key_body_ids_gt) * 3)
            loaded_dict['key_body_pos_vel'] = (loaded_dict['key_body_pos'][1:,:].clone() - loaded_dict['key_body_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['key_body_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['key_body_pos_vel'].shape[-1])).to('cuda'),loaded_dict['key_body_pos_vel']),dim=0)

            loaded_dict['obj_pos'] = interp_time_series(loaded_dict['hoi_data'][:, 318:321].clone(), factor=2, mode='linear')

            loaded_dict['obj_pos'] *= xyz
            loaded_dict['obj_pos_vel'] = (loaded_dict['obj_pos'][1:,:].clone() - loaded_dict['obj_pos'][:-1,:].clone())*self.fps_data
            if self.init_vel:
                loaded_dict['obj_pos_vel'] = torch.cat((loaded_dict['obj_pos_vel'][:1],loaded_dict['obj_pos_vel']),dim=0)
            else:
                loaded_dict['obj_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_pos_vel'].shape[-1])).to('cuda'),loaded_dict['obj_pos_vel']),dim=0)


            loaded_dict['obj_rot'] = interp_time_series(loaded_dict['hoi_data'][:, 321:325].clone(), factor=2, mode='linear')
            obj_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['obj_rot'])
            loaded_dict['obj_rot_vel'] = (obj_rot_exp_map[1:,:].clone() - obj_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['obj_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_rot_vel'].shape[-1])).to('cuda'),loaded_dict['obj_rot_vel']),dim=0)

            obj_rot_extend = loaded_dict['obj_rot'].unsqueeze(1).repeat(1, self.object_points[self.object_id[idx]].shape[0], 1).view(-1, 4)
            object_points_extend = self.object_points[self.object_id[idx]].unsqueeze(0).repeat(loaded_dict['obj_rot'].shape[0], 1, 1).view(-1, 3)
            obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(loaded_dict['obj_rot'].shape[0], self.object_points[self.object_id[idx]].shape[0], 3) + loaded_dict['obj_pos'].unsqueeze(1)
            key_body_pose = loaded_dict['key_body_pos'][:,:].clone()
            ref_ig = compute_sdf(key_body_pose.view(loaded_dict['obj_rot'].shape[0],-1,3), obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(loaded_dict['root_rot'])
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1] // 3, 1).view(-1, 4)
            ref_ig = quat_rotate(heading_rot_extend, ref_ig).view(loaded_dict['obj_rot'].shape[0], -1)    
            loaded_dict['contact'] = interp_time_series(loaded_dict['hoi_data'][:, 330:331].clone(), factor=2, mode='linear')
            loaded_dict['contact_parts'] = torch.round(loaded_dict['hoi_data'][:, 331:331+52].clone())
            loaded_dict['contact'] = torch.round(loaded_dict['contact'])
            loaded_dict['contact_parts'] = interp_time_series(loaded_dict['hoi_data'][:, 331:331+52].clone(), factor=2, mode='linear')
            loaded_dict['contact_parts'] = torch.round(loaded_dict['contact_parts'])

            loaded_dict['human_rot'] = interp_time_series(loaded_dict['hoi_data'][:, 331+52:331+52+52*4].clone(), factor=2, mode='linear')
            human_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['human_rot'].reshape(-1, 4)).view(-1, 52*3)
            loaded_dict['human_rot_vel'] = (human_rot_exp_map[1:,:].clone() - human_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['human_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['human_rot_vel'].shape[-1])).to('cuda'),loaded_dict['human_rot_vel']),dim=0)

            loaded_dict['hoi_data'] = torch.cat((
                                                    loaded_dict['root_pos'].clone(), # 0:3
                                                    loaded_dict['root_rot'].clone(), # 3:7
                                                    loaded_dict['dof_pos'].clone(), # 7:7+153=160
                                                    loaded_dict['dof_pos_vel'].clone(), # 160:160+153=313
                                                    loaded_dict['obj_pos'].clone(), # 313:316
                                                    loaded_dict['obj_rot'].clone(), # 316:320
                                                    loaded_dict['obj_pos_vel'].clone(), # 320:323
                                                    loaded_dict['obj_rot_vel'].clone(), # 323:326
                                                    loaded_dict['key_body_pos'][:,:].clone(), # 323:326
                                                    loaded_dict['contact'].clone(),
                                                    loaded_dict['contact_parts'].clone(),
                                                    ref_ig.clone(),
                                                    loaded_dict['human_rot'].clone(),
                                                    loaded_dict['key_body_pos_vel'].clone(),
                                                    loaded_dict['human_rot_vel'],
                                                    ),dim=-1)
            # print(self.ref_hoi_obs_size, loaded_dict['hoi_data'].shape[-1])
            assert(self.ref_hoi_obs_size == loaded_dict['hoi_data'].shape[-1])
            self.hoi_data_dict.append(loaded_dict)
            loaded_dict['hoi_data'] = torch.cat([loaded_dict['hoi_data'][0:1] for _ in range(30)]+[loaded_dict['hoi_data']], dim=0)
            hoi_datas.append(loaded_dict['hoi_data'])
            
            hoi_ref = torch.cat((
                                loaded_dict['root_pos'].clone(), # 0:3
                                loaded_dict['root_rot'].clone(), # 3:7
                                loaded_dict['dof_pos'].clone(), # 7:7+153=160
                                loaded_dict['dof_pos_vel'].clone(), # 160:160+153=313
                                loaded_dict['root_pos_vel'].clone(), # 313:316
                                loaded_dict['root_rot_vel'].clone(), # 316:319
                                loaded_dict['obj_pos'].clone(), # 319:322
                                loaded_dict['obj_rot'].clone(), # 322:326
                                loaded_dict['obj_pos_vel'].clone(), # 326:329
                                loaded_dict['obj_rot_vel'].clone(), # 329:332
                                ),dim=-1)
            hoi_ref = torch.cat([hoi_ref[0:1] for _ in range(30)]+[hoi_ref], dim=0)
            self.max_episode_length[-1] = (self.max_episode_length[-1]-1) * 2+1
            assert self.max_episode_length[-1] == loaded_dict['root_pos'].clone().shape[0]
            hoi_refs.append(hoi_ref)
        max_length = max(self.max_episode_length) + 30
        self.num_motions = len(hoi_refs)
        self.max_episode_length = to_torch(self.max_episode_length, dtype=torch.long) + 30
        self.hoi_data = []
        self.hoi_refs = []
        for i, data in enumerate(hoi_datas):
            pad_size = (0, 0, 0, max_length - data.size(0))
            padded_data = F.pad(data, pad_size, "constant", 0)
            self.hoi_data.append(padded_data)
            self.hoi_refs.append(F.pad(hoi_refs[i], pad_size, "constant", 0))
        self.hoi_data = torch.stack(self.hoi_data, dim=0)
        self.hoi_refs = torch.stack(self.hoi_refs, dim=0).unsqueeze(1).repeat(1, 1, 1, 1)
        self.ref_reward = torch.zeros((self.hoi_refs.shape[0], self.hoi_refs.shape[1], self.hoi_refs.shape[2])).to(self.hoi_refs.device)
        self.ref_reward[:, 0, :] = 1.0
        self.ref_index = torch.zeros((self.num_envs, )).long().to(self.hoi_refs.device)
        return


    def _setup_env_properties(self):
        self._load_target_asset()
        super()._setup_env_properties()
        self._setup_target_properties()
        return

    def _reset_target(self, env_ids):
        super()._reset_target(env_ids)
        self._target_states[env_ids, 0:2] = self._target_states[env_ids, 0:2] * self.scaling
        return


    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)


        env_ids_int32 = self._tar_actor_ids[env_ids]
        self._set_actor_root_state_indexed(env_ids_int32)
    
        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super()._reset_envs(env_ids)

        return    

    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos * self.scaling
        self._humanoid_root_states[env_ids, 2:3] = self.init_root_height
        self._humanoid_root_states[env_ids, 3:5] = 0
        self._humanoid_root_states[env_ids, 5:6] = 1
        self._humanoid_root_states[env_ids, 6:7] = -1
        self._humanoid_root_states[env_ids, 7:10] = 0
        self._humanoid_root_states[env_ids, 10:13] = 0
        
        self._dof_pos[env_ids] = self.init_dof
        self._dof_vel[env_ids] = 0
        return

    
    def _compute_hoi_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        key_body_vel = self._rigid_body_vel[:, self._key_body_ids, :]
        key_body_rot = self._rigid_body_rot[:, self._key_body_ids, :]
        key_body_ang_vel = self._rigid_body_ang_vel[:, self._key_body_ids, :]
        if (env_ids is None):
            self._curr_obs[:] = build_hoi_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._target_states,
                                                               self._tar_contact_forces,
                                                               self._contact_forces[:, self._contact_body_ids, :],
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
                                                                   self._contact_forces[env_ids][:, self._contact_body_ids, :],
                                                                   self.object_points[self.object_id[self.data_id[env_ids]]],
                                                                   key_body_rot[env_ids],
                                                                   key_body_vel[env_ids],
                                                                   key_body_ang_vel[env_ids],
                                                                   self._key_body_ids_gt,
                                                                   self._contact_body_ids_gt,
                                                                   ).float()
        return

    
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
    dof_pos_new = torch.zeros((root_pos.shape[0], 153), device=root_pos.device)
    dof_vel_new = torch.zeros((root_pos.shape[0], 153), device=root_pos.device)    
    dof_pos_new[:, :dof_pos.shape[1]] = dof_pos        
    dof_vel_new[:, :dof_vel.shape[1]] = dof_vel    
    contact_new = torch.zeros((root_pos.shape[0], 52), device=root_pos.device) 
    contact_new[:, _contact_body_ids_gt] = contact
    body_rot_new = torch.zeros((root_pos.shape[0], 52, 4), device=root_pos.device) 
    body_rot_vel_new = torch.zeros((root_pos.shape[0], 52, 3), device=root_pos.device) 
    body_rot_new[:, _key_body_ids_gt] = body_rot
    body_rot_vel_new[:, _key_body_ids_gt] = body_rot_vel
    obs = torch.cat((root_pos, 
                     root_rot, 
                     dof_pos_new, 
                     dof_vel_new, 
                     target_states, 
                     key_body_pos.contiguous().view(-1,key_body_pos.shape[1]*key_body_pos.shape[2]), 
                     target_contact, 
                     contact_new, 
                     ig, 
                     body_rot_new.view(-1, 52*4), 
                     body_vel.view(-1,key_body_pos.shape[1]*key_body_pos.shape[2]), 
                     body_rot_vel_new.view(-1, 52*3)), dim=-1)
    return obs


def interp_time_series(x, factor=2, mode='linear'):
    """
    Interpolates a time series tensor along dimension 0.
    For a tensor x of shape (T, ...) the new length will be (T-1)*factor + 1.
    
    Args:
        x: input tensor with shape (T, ...) where T is the time dimension.
        factor: interpolation factor (2 for going from 30Hz to 60Hz).
        mode: interpolation mode; use 'linear' for continuous signals or 'nearest' for discrete signals.
        
    Returns:
        Interpolated tensor with shape ((T-1)*factor + 1, ...).
    """
    T = x.shape[0]
    new_T = (T - 1) * factor + 1
    # Flatten all dimensions except time.
    flat = x.view(T, -1).transpose(0, 1).unsqueeze(0)  # shape: (1, features, T)
    flat_intp = F.interpolate(flat, size=new_T, mode=mode, align_corners=True)
    flat_intp = flat_intp.squeeze(0).transpose(0, 1)
    return flat_intp.view(new_T, *x.shape[1:])
