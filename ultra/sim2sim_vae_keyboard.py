import argparse
from collections import OrderedDict

import numpy as np
import torch
import mujoco as mj
import mujoco.viewer

from sim2sim_vae import HumanoidEnv as BaseHumanoidEnv
from utils import torch_utils_mujoco
from utils.obs_vae import MujocoObs, compute_sdf


def _as_goal_tensor(goal_pos, ref_tensor):
    return torch.as_tensor(goal_pos, device=ref_tensor.device, dtype=ref_tensor.dtype).view(1, 3)

def _normalize_xy(vec):
    vec = vec.copy()
    vec[2] = 0.0
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


class KeyboardMujocoObs(MujocoObs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyboard_goal_pos = None
        self.time_offset = 0
        self.last_curr_t = 0
        self.manual_command_goal = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
        self.cooldown_enabled = False

    def set_keyboard_goal(self, goal_pos):
        if goal_pos is None:
            self.keyboard_goal_pos = None
        else:
            self.keyboard_goal_pos = np.asarray(goal_pos, dtype=np.float32).copy()


    def toggle_command_goal(self, mode):
        if self.manual_command_goal is None:
            self.manual_command_goal = torch.zeros((1, 3), dtype=torch.float32)
        if mode == "stand":
            self.manual_command_goal[:, 0] = 0.0 if self.manual_command_goal[:, 0].item() > 0.5 else 1.0
        elif mode == "approach_leave":
            self.manual_command_goal[:, 0] = 0.0
            approaching = self.manual_command_goal[:, 1].item() > 0.5
            leaving = self.manual_command_goal[:, 2].item() > 0.5
            if approaching and not leaving:
                self.manual_command_goal[:, 1] = 0.0
                self.manual_command_goal[:, 2] = 1.0
            elif leaving and not approaching:
                self.manual_command_goal[:, 1] = 1.0
                self.manual_command_goal[:, 2] = 0.0
            else:
                self.manual_command_goal[:, 1] = 1.0
                self.manual_command_goal[:, 2] = 0.0
            new_len = min(120, max(1, int(self.goal_phase_max_len)))
            if hasattr(self, "long_term_t"):
                self.long_term_t = torch.full_like(self.long_term_t, new_len)
        elif mode == "reset_time":
            new_len = min(180, max(1, int(self.goal_phase_max_len)))
            if hasattr(self, "long_term_t"):
                self.long_term_t = torch.full_like(self.long_term_t, new_len)
        else:
            raise ValueError(f"Unknown command goal mode: {mode}")

    def get_command_goal(self):
        if self.manual_command_goal is None:
            return self.command_goal
        goal = self.command_goal.clone()
        goal[:, :3] = self.manual_command_goal.to(goal.device, goal.dtype)
        # if goal.shape[-1] >= 4 and self.cooldown_enabled:
        goal[:, 3] = self.long_term_t / self.goal_phase_max_len
        return goal

    def toggle_cooldown(self):
        self.cooldown_enabled = not self.cooldown_enabled
        if self.cooldown_enabled and self.manual_command_goal is not None:
            self.manual_command_goal[:, 0] = 1.0
            self.manual_command_goal[:, 1] = 0.0
            self.manual_command_goal[:, 2] = 1.0
            if hasattr(self, "long_term_t"):
                max_len = int(self.goal_phase_max_len) if self.goal_phase_max_len is not None else 240
                half_len = max(1, max_len // 4)
                self.long_term_t = torch.full_like(self.long_term_t, half_len)
            self.cooldown_history_reset_pending = True

    def disable_cooldown_and_reset_time(self):
        self.cooldown_enabled = False
        if hasattr(self, "long_term_t"):
            max_len = int(self.goal_phase_max_len) if self.goal_phase_max_len is not None else 240
            self.long_term_t = torch.full_like(self.long_term_t, max_len)
    
    def reset_time_offset(self, current_ts=None):
        if current_ts is None:
            current_ts = self.last_curr_t
        self.time_offset = int(current_ts or 0)

    def _check_goal_achieved(self, data, goal_ts):
        if self.keyboard_goal_pos is None:
            return super()._check_goal_achieved(data, goal_ts)

        # When driving goal from keyboard, don't stall progress on goal checks.
        return True

    def _compute_observations_iter(self, data, curr_t, delta_t=1, actions=None, torques=None, student_obs=False, episode_length=0):
        self.last_curr_t = int(curr_t)
        if self.time_offset:
            curr_t = max(int(curr_t) - self.time_offset, 0)
            episode_length = max(int(episode_length) - self.time_offset, 0)
        ts = curr_t
        if student_obs:
            need_new_goal = False
            is_frozen = False

            if episode_length <= 0:
                need_new_goal = True
                self.current_goal_ts = None
                self.frozen_ts = None
            elif self.long_term_t.item() <= 1:
                if self.goal_achievement_enabled and self.current_goal_ts is not None:
                    goal_achieved = self._check_goal_achieved(data, self.current_goal_ts)
                    if goal_achieved:
                        need_new_goal = True
                        self.frozen_ts = None
                        print("Goal achieved at ts", ts)
                    else:
                        is_frozen = True
                        if self.frozen_ts is None:
                            self.frozen_ts = ts
                else:
                    need_new_goal = True

            effective_ts = self.frozen_ts if is_frozen and self.frozen_ts is not None else ts

            if need_new_goal:
                curr_ref_obs = self.hoi_data[None, min(effective_ts, self.max_episode_length - 1)].clone()
                ref_vel = curr_ref_obs[:, 78:81]
                ref_speed = torch.norm(ref_vel, dim=-1)
                use_long = ref_speed < self.long_term_speed_threshold
                long_horizon = torch.randint(
                    120, 240, (1,), device=self.long_term_t.device, dtype=self.long_term_t.dtype
                )
                short_horizon = torch.randint(
                    60, 120, (1,), device=self.long_term_t.device, dtype=self.long_term_t.dtype
                )
                self.long_term_t = torch.where(use_long, long_horizon, short_horizon)
                self.current_goal_ts = min(effective_ts + self.long_term_t.item(), self.max_episode_length - 1)

            if self.current_goal_ts is not None:
                next_ts = self.current_goal_ts
            else:
                next_ts = min(effective_ts + self.long_term_t.item(), self.max_episode_length - 1)
                self.current_goal_ts = next_ts

            if not is_frozen:
                if not self.goal_achievement_enabled or need_new_goal:
                    self.long_term_t = torch.clamp(self.long_term_t - 1, min=0)
                else:
                    goal_achieved = self._check_goal_achieved(data, self.current_goal_ts)
                    if goal_achieved:
                        self.long_term_t = torch.clamp(self.long_term_t - 1, min=0)
        else:
            next_ts = min(ts + delta_t, self.max_episode_length - 1)
        next_ts_local = min(ts + 1, self.max_episode_length - 1)

        ref_obs = self.hoi_data[None, next_ts].clone()
        ref_obs_local = self.hoi_data[None, next_ts_local].clone()
        next_ts_16 = min(ts + 16, self.max_episode_length - 1)
        ref_obs_16 = self.hoi_data[None, next_ts_16].clone()

        if self.keyboard_goal_pos is not None:
            goal = _as_goal_tensor(self.keyboard_goal_pos, ref_obs)
            ref_obs[:, 71:74] = goal
            ref_obs_local[:, 71:74] = goal
            ref_obs_16[:, 71:74] = goal

        obs, key_body_pose, key_body_rot, obs_dict = self._compute_humanoid_obs(
            data,
            ref_obs,
            ref_obs_16,
            actions,
            torques,
            episode_length,
            student_obs=student_obs,
            local_ref_obs=ref_obs_local,
        )
        if self.cooldown_enabled and obs.shape[1] >= 3:
            obs[:, :3] = 0.0
        # if self.cooldown_enabled and obs.shape[1] >= 34:
        #     obs[:, 3:34] = 0.0
        obs_shape = obs.shape[-1]
        task_obs, obj_points, obj_obs_dict = self._compute_task_obs(data, ref_obs, is_student=student_obs)
        if self.cooldown_enabled and task_obs is not None and task_obs.shape[-1] >= 9:
            task_obs = task_obs.clone()
            task_obs[:, 0:3] = 0.0
            task_obs[:, 3:9] = 0.0
        if not student_obs:
            task_obs = task_obs * 0
        obs = torch.cat([obs, task_obs], dim=-1)
        obs_dict.update(obj_obs_dict)
        obj_points_ig = obj_obs_dict.get("obj_points_world", obj_points)  # SDF always on world-frame points
        if student_obs:
            ig = compute_sdf(key_body_pose, obj_points_ig).view(1, -1, 3)
            ig_norm = ig.norm(dim=-1, keepdim=True)
            goal_phase = self._update_goal_phase(ig_norm, episode_length)
            if self.keyboard_goal_pos is not None and goal_phase.shape[-1] >= 1:
                goal_phase = goal_phase.clone()
                goal_phase[:, 0] = 0.0
            self.command_goal = goal_phase
            if self.manual_command_goal is not None:
                self.command_goal = self.get_command_goal()
            task_obs_size = task_obs.shape[-1] if task_obs is not None else 0
            obs_final = self._finalize_student_obs(
                obs,
                obs_shape=obs_shape,
                task_obs_size=task_obs_size,
                force_global_goal_mask=torch.tensor(True) if self.cooldown_enabled else None,
            )
            self._append_goal_viz(obs_dict, ref_obs)
            return obs_final, obs_dict

        ig = compute_sdf(key_body_pose, obj_points_ig).view(-1, 3)
        heading_rot = torch_utils_mujoco.calc_heading_quat_inv(key_body_rot[:, 0, :])
        heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1], 1).view(-1, 4)
        ig = torch_utils_mujoco.quat_rotate(heading_rot_extend, ig).view(1, -1, 3)
        ig_norm = ig.norm(dim=-1, keepdim=True)
        ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
        ig = ig_all.view(1, -1)
        ig_all = ig_all.view(1, -1)
        ref_ig = ref_obs[:, 630:].view(1, 39, 3)
        ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
        ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()
        ref_ig = ref_ig.view(1, -1)
        ref_ig *= 0
        ig_all *= 0
        ig *= 0
        obs_dict.update(OrderedDict([("ig", ig_all), ("diff_ig", ref_ig - ig)]))
        return torch.cat((obs, ig_all, ref_ig - ig), dim=-1), obs_dict


class HumanoidEnvKeyboard(BaseHumanoidEnv):
    def __init__(
        self,
        policy_path,
        robot_type="g1_29dof",
        device="cuda",
        record_video=False,
        motion_path=None,
        use_jit=False,
        obj_rot_keep_prob=1.0,
        obj_trans_keep_prob=1.0,
        obj_point_keep_prob=1.0,
        obj_pos_keep_prob=1.0,
        human_move_keep_prob=1.0,
        human_global_keep_prob=None,
        human_local_keep_prob=None,
        human_goal_keep_prob=1.0,
        mask_flip_prob=0.0,
        point_noise_std=0.0,
        point_dropout_prob=0.0,
        point_outlier_prob=0.0,
        point_outlier_scale=0.5,
        point_depth_noise_scale=0.0,
        point_density_min=1.0,
        point_density_max=1.0,
        point_cluster_noise_std=0.0,
        point_scale_min=1.0,
        point_scale_max=1.0,
        point_translation_noise=0.0,
        point_occlusion_prob=0.0,
        camera_rot_noise=0.0,
        camera_pos_noise=0.0,
        point_fixed_grid_sampling=True,
        point_surface_grid_resolution=None,
        render_camera="tracking",
        object_alpha=0.2,
        point_viz_scale=8.0,
        point_viz_offset=0.0,
        point_viz_sites=64,
        point_viz_site_size=0.02,
        goal_achievement_enabled=False,
        goal_pos_threshold=0.3,
        obj_goal_decouple="hard",
        obj_goal_z_threshold=0.15,
        goal_phase_dim=4,
        keyboard_goal_z=0.5,
        keyboard_step=0.02,
    ):
        self.keyboard_goal_z = float(keyboard_goal_z)
        self.keyboard_step = float(keyboard_step)
        self.keyboard_goal_pos = None
        self._keyboard_goal_enabled = False
        self._obs_builder_kwargs = dict(
            obj_rot_keep_prob=obj_rot_keep_prob,
            obj_trans_keep_prob=obj_trans_keep_prob,
            obj_point_keep_prob=obj_point_keep_prob,
            obj_pos_keep_prob=obj_pos_keep_prob,
            human_move_keep_prob=human_move_keep_prob,
            human_global_keep_prob=human_global_keep_prob,
            human_local_keep_prob=human_local_keep_prob,
            human_goal_keep_prob=human_goal_keep_prob,
            mask_flip_prob=mask_flip_prob,
            point_noise_std=point_noise_std,
            point_dropout_prob=point_dropout_prob,
            point_outlier_prob=point_outlier_prob,
            point_outlier_scale=point_outlier_scale,
            point_depth_noise_scale=point_depth_noise_scale,
            point_density_min=point_density_min,
            point_density_max=point_density_max,
            point_cluster_noise_std=point_cluster_noise_std,
            point_scale_min=point_scale_min,
            point_scale_max=point_scale_max,
            point_translation_noise=point_translation_noise,
            point_occlusion_prob=point_occlusion_prob,
            camera_rot_noise=camera_rot_noise,
            camera_pos_noise=camera_pos_noise,
            point_fixed_grid_sampling=point_fixed_grid_sampling,
            point_surface_grid_resolution=point_surface_grid_resolution,
            goal_achievement_enabled=goal_achievement_enabled,
            goal_pos_threshold=goal_pos_threshold,
            goal_phase_dim=goal_phase_dim,
            obj_goal_decouple=obj_goal_decouple,
            obj_goal_z_threshold=obj_goal_z_threshold,
        )

        super().__init__(
            policy_path=policy_path,
            robot_type=robot_type,
            device=device,
            record_video=record_video,
            motion_path=motion_path,
            use_jit=use_jit,
            obj_rot_keep_prob=obj_rot_keep_prob,
            obj_trans_keep_prob=obj_trans_keep_prob,
            obj_point_keep_prob=obj_point_keep_prob,
            obj_pos_keep_prob=obj_pos_keep_prob,
            human_move_keep_prob=human_move_keep_prob,
            human_global_keep_prob=human_global_keep_prob,
            human_local_keep_prob=human_local_keep_prob,
            human_goal_keep_prob=human_goal_keep_prob,
            mask_flip_prob=mask_flip_prob,
            point_noise_std=point_noise_std,
            point_dropout_prob=point_dropout_prob,
            point_outlier_prob=point_outlier_prob,
            point_outlier_scale=point_outlier_scale,
            point_depth_noise_scale=point_depth_noise_scale,
            point_density_min=point_density_min,
            point_density_max=point_density_max,
            point_cluster_noise_std=point_cluster_noise_std,
            point_scale_min=point_scale_min,
            point_scale_max=point_scale_max,
            point_translation_noise=point_translation_noise,
            point_occlusion_prob=point_occlusion_prob,
            camera_rot_noise=camera_rot_noise,
            camera_pos_noise=camera_pos_noise,
            point_fixed_grid_sampling=point_fixed_grid_sampling,
            point_surface_grid_resolution=point_surface_grid_resolution,
            render_camera=render_camera,
            object_alpha=object_alpha,
            point_viz_scale=point_viz_scale,
            point_viz_offset=point_viz_offset,
            point_viz_sites=point_viz_sites,
            point_viz_site_size=point_viz_site_size,
            goal_achievement_enabled=goal_achievement_enabled,
            goal_pos_threshold=goal_pos_threshold,
            goal_phase_dim=goal_phase_dim,
            obj_goal_decouple=obj_goal_decouple,
            obj_goal_z_threshold=obj_goal_z_threshold,
        )

        self._rebuild_obs_builder()
        self._extend_trajectory(mult=100)
        self._reset_sim()
        self.obs_builder.toggle_cooldown()
        if not self.record_video:
            self._reset_viewer_with_keyboard()

    def _rebuild_obs_builder(self):
        self.obs_builder = KeyboardMujocoObs(
            self.model,
            self.object_name,
            self.max_episode_length,
            self.hoi_data,
            self.object_points,
            self.object_corners,
            history_step=10,
            **self._obs_builder_kwargs,
        )
        self.obs_builder.set_command_goal(self.commands)

    def _extend_trajectory(self, mult=100):
        if mult is None or mult <= 1:
            return
        if not isinstance(self.hoi_data, torch.Tensor):
            self.hoi_data = torch.as_tensor(self.hoi_data)
        if not isinstance(self.hoi_ref, torch.Tensor):
            self.hoi_ref = torch.as_tensor(self.hoi_ref)

        self.hoi_data = torch.cat([self.hoi_data] * mult, dim=0)
        self.hoi_ref = torch.cat([self.hoi_ref] * mult, dim=0)
        self.max_episode_length = self.hoi_data.shape[0]
        self.sim_duration *= float(mult)

        self.obs_builder.hoi_data = self.hoi_data
        self.obs_builder.max_episode_length = self.max_episode_length

    def _reset_viewer_with_keyboard(self):
        try:
            self.viewer.close()
        except Exception:
            pass

        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self._on_key,
        )
        self.viewer.cam.distance = 5.0
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 140
        self.viewer.opt.sitegroup[:] = 1
        self.viewer.user_scn.ngeom = 0
        print("Keyboard goal control: arrows = X/Y, Q/E = Z, R = reset")

    def _get_object_pos(self):
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.object_name)
        if body_id < 0:
            return None
        return self.data.xpos[body_id].copy()

    def _get_heading_axes(self):
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        left = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return forward, left

    def _init_keyboard_goal(self):
        base = None
        if isinstance(self.hoi_ref, torch.Tensor) and self.hoi_ref.shape[0] > 0:
            base = self.hoi_ref[0, 71:74].detach().cpu().numpy()
        if base is None:
            base = self._get_object_pos()
        # base = self._get_object_pos()
        if base is None:
            base = np.zeros(3, dtype=np.float32)
        base = np.asarray(base, dtype=np.float32).copy()
        # base[2] = base[2] + 0.6
        base[1] = base[1] - 1.0
        self.keyboard_goal_pos = base
        self._keyboard_goal_enabled = True
        self.obs_builder.set_keyboard_goal(self.keyboard_goal_pos)
        self._reset_goal_progress()
        print(f"Keyboard goal initialized at {self.keyboard_goal_pos}")

    def _reset_goal_progress(self):
        if not hasattr(self, "obs_builder"):
            return
        goal_len = getattr(self.obs_builder, "goal_phase_max_len", 240.0)
        if goal_len is None:
            goal_len = 240.0
        goal_len = int(goal_len)
        if hasattr(self.obs_builder, "long_term_t"):
            self.obs_builder.long_term_t = torch.full_like(self.obs_builder.long_term_t, goal_len)
        self.obs_builder.current_goal_ts = None
        self.obs_builder.frozen_ts = None
        if hasattr(self.obs_builder, "prev_hand_ig_valid"):
            self.obs_builder.prev_hand_ig_valid[:] = False

    def _reset_sim(self):
        current_ts = getattr(self.obs_builder, "last_curr_t", 0)
        self.reset_human_and_object_from_ref()
        self.last_action = torch.zeros((1, 29))
        self.last_torque = torch.zeros((1, 29))
        self.obs_builder.reset_time_offset(current_ts)
        self._init_keyboard_goal()

    def _on_key(self, key):
        glfw = mujoco.viewer.glfw
        if key == glfw.KEY_R:
            self._reset_sim()
            return
        if key == glfw.KEY_V:
            self.obs_builder.toggle_cooldown()
            print(f"Cooldown: {self.obs_builder.cooldown_enabled}")
            # if self.obs_builder.cooldown_enabled:
            #     print(f"Cooldown: {self.obs_builder.cooldown_enabled}")
            # else:
            #     self.obs_builder.disable_cooldown_and_reset_time()
            #     self.obs_builder.toggle_command_goal("stand")
            #     print(f"Command goal: {self.obs_builder.get_command_goal().cpu().numpy()}")
            return
        if key == glfw.KEY_B:
            self.obs_builder.disable_cooldown_and_reset_time()
            self.obs_builder.toggle_command_goal("approach_leave")
            print(f"Command goal: {self.obs_builder.get_command_goal().cpu().numpy()}")
            return
        if key == glfw.KEY_N:
            self.obs_builder.disable_cooldown_and_reset_time()
            self.obs_builder.toggle_command_goal("reset_time")
            print(f"Command goal: {self.obs_builder.get_command_goal().cpu().numpy()}")
            return
        if self.keyboard_goal_pos is None:
            self._init_keyboard_goal()

        delta = np.zeros(3, dtype=np.float32)
        forward, left = self._get_heading_axes()
        if key == glfw.KEY_UP:
            delta += forward * self.keyboard_step
        elif key == glfw.KEY_DOWN:
            delta -= forward * self.keyboard_step
        elif key == glfw.KEY_LEFT:
            delta += left * self.keyboard_step
        elif key == glfw.KEY_RIGHT:
            delta -= left * self.keyboard_step
        elif key == glfw.KEY_Q:
            delta[2] += self.keyboard_step
        elif key == glfw.KEY_E:
            delta[2] -= self.keyboard_step
        else:
            return

        self.keyboard_goal_pos = self.keyboard_goal_pos + delta
        self._keyboard_goal_enabled = True
        self.obs_builder.set_keyboard_goal(self.keyboard_goal_pos)
        self._reset_goal_progress()
        print(f"Keyboard goal: {self.keyboard_goal_pos}")

    def reset_human_and_object_from_ref(self, *args, **kwargs):
        super().reset_human_and_object_from_ref(*args, **kwargs)
        if not self._keyboard_goal_enabled:
            self._init_keyboard_goal()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Student checkpoint (.pth) or JIT model (.pt with --use_jit)')
    parser.add_argument('--robot', type=str, default='g1_29dof')
    parser.add_argument('--record_video', action='store_true')
    parser.add_argument('--motion_path', type=str, required=True, help='Reference motion .pt used to place the object and initialize the robot')
    parser.add_argument('--sim_duration', type=float, default=60)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_jit', action='store_true', help='Use JIT (.pt) model instead of checkpoint (.pth)')
    parser.add_argument('--render_camera', type=str, default='tracking', choices=['tracking', 'first_person'])
    parser.add_argument('--object_alpha', type=float, default=0.2, help='Alpha value for object transparency')
    parser.add_argument('--point_viz_scale', type=float, default=8.0, help='Scale factor for point cloud positions')
    parser.add_argument('--point_viz_offset', type=float, default=0.0, help='Offset applied to point cloud visualization')
    parser.add_argument('--point_viz_sites', type=int, default=64, help='Number of MJCF sites to use for point visualization')
    parser.add_argument('--point_viz_site_size', type=float, default=0.02, help='Size of point visualization sites')
    parser.add_argument('--task_mode', type=str, default=None,
                        choices=['full_track', 'sparse_track', 'object_obs'],
                        help='Preset observation masking mode (omit for training-style random masks)')
    parser.add_argument('--obj_obs', type=str, default='points', choices=['points', 'pos', 'none'],
                        help='Object observation source for object_obs mode')
    parser.add_argument('--obs_obj_rot_keep_prob', type=float, default=1.0,
                        help='Keep probability for object rotation observations')
    parser.add_argument('--obs_obj_trans_keep_prob', type=float, default=1.0,
                        help='Keep probability for object translation observations')
    parser.add_argument('--obs_obj_point_keep_prob', type=float, default=1.0,
                        help='Keep probability for object point observations')
    parser.add_argument('--obs_obj_pos_keep_prob', type=float, default=1.0,
                        help='Keep probability for object position observations')
    parser.add_argument('--obs_human_move_keep_prob', type=float, default=1.0,
                        help='Keep probability for human movement observations')
    parser.add_argument('--obs_human_global_keep_prob', type=float, default=None,
                        help='Keep probability for human global goal observations')
    parser.add_argument('--obs_human_local_keep_prob', type=float, default=None,
                        help='Keep probability for human local goal observations')
    parser.add_argument('--obs_human_goal_keep_prob', type=float, default=1.0,
                        help='Keep probability for command goal observations')
    parser.add_argument('--obs_mask_flip_prob', type=float, default=0.0,
                        help='Probability of flipping a keep mask per step')

    parser.add_argument('--point_noise_std', type=float, default=0.02,
                        help='Standard deviation of Gaussian noise added to points')
    parser.add_argument('--point_dropout_prob', type=float, default=0.15,
                        help='Probability of dropping out individual points')
    parser.add_argument('--point_outlier_prob', type=float, default=0.05,
                        help='Probability of injecting outlier points')
    parser.add_argument('--point_outlier_scale', type=float, default=0.5,
                        help='Scale of outlier displacement (meters)')
    parser.add_argument('--point_depth_noise_scale', type=float, default=0.01,
                        help='Scale of depth-dependent noise')
    parser.add_argument('--point_density_min', type=float, default=0.5,
                        help='Minimum density factor for point sampling')
    parser.add_argument('--point_density_max', type=float, default=1.0,
                        help='Maximum density factor for point sampling')
    parser.add_argument('--point_cluster_noise_std', type=float, default=0.005,
                        help='Standard deviation of cluster-based noise')
    parser.add_argument('--point_scale_min', type=float, default=0.95,
                        help='Minimum scale factor for point cloud')
    parser.add_argument('--point_scale_max', type=float, default=1.05,
                        help='Maximum scale factor for point cloud')
    parser.add_argument('--point_translation_noise', type=float, default=0.02,
                        help='Standard deviation of random translation noise')
    parser.add_argument('--point_occlusion_prob', type=float, default=0.1,
                        help='Probability of random occlusion')
    parser.add_argument('--camera_rot_noise', type=float, default=0.05,
                        help='Camera rotation noise (radians)')
    parser.add_argument('--camera_pos_noise', type=float, default=0.02,
                        help='Camera position noise (meters)')

    parser.add_argument('--no_point_fixed_grid_sampling', action='store_true',
                        help='Disable fixed UV grid for point sampling (default: enabled)')
    parser.add_argument('--point_surface_grid_resolution', type=int, default=None,
                        help='Grid resolution for fixed surface sampling (default: auto-compute)')

    parser.add_argument('--goal_achievement_enabled', action='store_true',
                        help='Enable goal achievement checking for sparse tracking (only progress when goal is reached)')
    parser.add_argument('--goal_pos_threshold', type=float, default=0.1,
                        help='Position threshold (meters) for goal achievement')
    parser.add_argument('--goal_phase_dim', type=int, default=4, choices=[3, 4],
                        help='Goal phase command dimension (3 or 4)')
    parser.add_argument('--obj_goal_decouple', type=str, default="hard",
                        choices=["off", "hard", "soft"],
                        help='Decouple object translation goal: vertical first then horizontal (sim2sim only)')
    parser.add_argument('--obj_goal_z_threshold', type=float, default=0.15,
                        help='Z threshold (meters) for object goal decoupling')

    parser.add_argument('--keyboard_goal_z', type=float, default=0.5,
                        help='Initial keyboard goal height (meters)')
    parser.add_argument('--keyboard_step', type=float, default=0.02,
                        help='Keyboard goal step size (meters)')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.use_jit and args.ckpt.endswith('.pt'):
        print("Detected .pt file extension, automatically enabling --use_jit")
        args.use_jit = True

    env = HumanoidEnvKeyboard(
        policy_path=args.ckpt,
        robot_type=args.robot,
        device=device,
        record_video=args.record_video,
        motion_path=args.motion_path,
        use_jit=args.use_jit,
        obj_rot_keep_prob=args.obs_obj_rot_keep_prob,
        obj_trans_keep_prob=args.obs_obj_trans_keep_prob,
        obj_point_keep_prob=args.obs_obj_point_keep_prob,
        obj_pos_keep_prob=args.obs_obj_pos_keep_prob,
        human_move_keep_prob=args.obs_human_move_keep_prob,
        human_global_keep_prob=args.obs_human_global_keep_prob,
        human_local_keep_prob=args.obs_human_local_keep_prob,
        human_goal_keep_prob=args.obs_human_goal_keep_prob,
        mask_flip_prob=args.obs_mask_flip_prob,
        point_noise_std=args.point_noise_std,
        point_dropout_prob=args.point_dropout_prob,
        point_outlier_prob=args.point_outlier_prob,
        point_outlier_scale=args.point_outlier_scale,
        point_depth_noise_scale=args.point_depth_noise_scale,
        point_density_min=args.point_density_min,
        point_density_max=args.point_density_max,
        point_cluster_noise_std=args.point_cluster_noise_std,
        point_scale_min=args.point_scale_min,
        point_scale_max=args.point_scale_max,
        point_translation_noise=args.point_translation_noise,
        point_occlusion_prob=args.point_occlusion_prob,
        camera_rot_noise=args.camera_rot_noise,
        camera_pos_noise=args.camera_pos_noise,
        point_fixed_grid_sampling=not args.no_point_fixed_grid_sampling,
        point_surface_grid_resolution=args.point_surface_grid_resolution,
        render_camera=args.render_camera,
        object_alpha=args.object_alpha,
        point_viz_scale=args.point_viz_scale,
        point_viz_offset=args.point_viz_offset,
        point_viz_sites=args.point_viz_sites,
        point_viz_site_size=args.point_viz_site_size,
        goal_achievement_enabled=args.goal_achievement_enabled,
        goal_pos_threshold=args.goal_pos_threshold,
        goal_phase_dim=args.goal_phase_dim,
        obj_goal_decouple=args.obj_goal_decouple,
        obj_goal_z_threshold=args.obj_goal_z_threshold,
        keyboard_goal_z=args.keyboard_goal_z,
        keyboard_step=args.keyboard_step,
    )
    if args.task_mode is not None:
        env.obs_builder.configure_task_mode(args.task_mode, obj_obs=args.obj_obs)
    env.run()
