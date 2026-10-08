import torch
from pathlib import Path

from utils.gym_torch_utils import *

from utils import torch_utils
import torch.nn.functional as F
from env.tasks.humanoid_g1 import *
from env.tasks.ultra import Ultra


class UltraG1(Humanoid_G1, Ultra):

    """
    Print of the cfg that was given
    UltraEnvCfg(
        viewer=ViewerCfg(
            eye=(0.0, -3.0, 1.0),
            lookat=(0.0, 0.0, 1.0),
            cam_prim_path='/OmniverseKit_Persp',
            resolution=(1280, 720),
            origin_type='env',
            env_index=0,
            asset_name=None,
            body_name=None
        ),
        sim=SimulationCfg(
            physics_prim_path='/physicsScene',
            device='cuda:0',
            dt=0.001,
            render_interval=17,
            gravity=(0.0, 0.0, -9.81),
            enable_scene_query_support=False,
            use_fabric=True,
            physx=PhysxCfg(
                solver_type=1,
                solve_articulation_contact_last=False,
                min_position_iteration_count=1,
                max_position_iteration_count=255,
                min_velocity_iteration_count=0,
                max_velocity_iteration_count=255,
                enable_ccd=False,
                enable_stabilization=False,
                enable_external_forces_every_iteration=False,
                enable_enhanced_determinism=False,
                bounce_threshold_velocity=0.2,
                friction_offset_threshold=0.04,
                friction_correlation_distance=0.025,
                gpu_max_rigid_contact_count=8388608,
                gpu_max_rigid_patch_count=1048576,
                gpu_found_lost_pairs_capacity=2097152,
                gpu_found_lost_aggregate_pairs_capacity=33554432,
                gpu_total_aggregate_pairs_capacity=2097152,
                gpu_collision_stack_size=67108864,
                gpu_heap_capacity=67108864,
                gpu_temp_buffer_capacity=16777216,
                gpu_max_num_partitions=8,
                gpu_max_soft_body_contacts=1048576,
                gpu_max_particle_contacts=1048576
            ),
            physics_material=RigidBodyMaterialCfg(
                func=<function spawn_rigid_body_material at 0x75b62177c040>,
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.0,
                friction_combine_mode='average',
                restitution_combine_mode='average',
                compliant_contact_stiffness=0.0,
                compliant_contact_damping=0.0
            ),
            render=RenderCfg(
                enable_translucency=None,
                enable_reflections=None,
                enable_global_illumination=None,
                antialiasing_mode=None,
                enable_dlssg=None,
                enable_dl_denoiser=None,
                dlss_mode=None,
                enable_direct_lighting=None,
                samples_per_pixel=None,
                enable_shadows=None,
                enable_ambient_occlusion=None,
                dome_light_upper_lower_strategy=None,
                carb_settings=None,
                rendering_mode=None
            ),
            create_stage_in_memory=False,
            logging_level='WARNING',
            save_logs_to_file=True,
            log_dir=None
        ),
        ui_window_class_type=<class 'isaaclab.envs.ui.base_env_window.BaseEnvWindow'>,
        seed=9483,
        decimation=17,
        is_finite_horizon=False,
        episode_length_s=1000000.0,
        scene=<dataclasses._MISSING_TYPE object at 0x75b56e304dd0>,
        events=None,
        observation_space=<dataclasses._MISSING_TYPE object at 0x75b56e304dd0>,
        num_observations=None,
        state_space=None,
        num_states=None,
        observation_noise_model=None,
        action_space=<dataclasses._MISSING_TYPE object at 0x75b56e304dd0>,
        num_actions=None,
        num_observations=None,
        observation_noise_model=None,
        action_space=<dataclasses._MISSING_TYPE object at 0x75b56e304dd0>,
        num_actions=None,
        num_observations=None,
        observation_noise_model=None,
        action_space=<dataclasses._MISSING_TYPE object at 0x75b56e304dd0>,
        num_actions=None,
        rerender_on_reset=False,
        num_rerenders_on_reset=0,
        wait_for_textures=True,
        xr=None,
        log_dir=None,

        task={
            'env': {
                'numEnvs': 2048,
                'envSpacing': 1,
                'episodeLength': 1000,
                'isFlagrun': False,
                'enableDebugVis': False,
                'playdataset': False,
                'projtype': 'None',
                'saveImages': False,
                'initVel': False,
                'moreRigid': False,
                'pdControl': True,
                'retargetPositionControl': True,
                'powerScale': 1.0,
                'controlFrequencyInv': 17,
                'stateInit': 'Start',
                'rolloutLength': 1000,
                'hybridInitProb': 0.5,
                'dataFPS': 30,
                'dataFramesScale': 1,
                'ballSize': 1.0,
                'numObs': 1853,
                'numDoF': 29,
                'numDoFWrist': 3,
                'numDoFHand': 0,
                'motion_file': 'InterAct/OMOMO_retarget_supported_080_080_080',
                'robotType': 'g1/g1_29dof.urdf',
                'objectDensity': 25,
                'localRootObs': False,

                'keyBodies': [
                    'left_hip_yaw_link',
                    'left_knee_link',
                    'left_ankle_roll_link',
                    'right_hip_yaw_link',
                    'right_knee_link',
                    'right_ankle_roll_link',
                    'torso_link',
                    'mid360_link',
                    'left_shoulder_yaw_link',
                    'left_elbow_link',
                    'left_rubber_hand',
                    'right_shoulder_yaw_link',
                    'right_elbow_link',
                    'right_rubber_hand'
                ],

                'keyIndex': [
                    1, 2, 3, 5, 6, 7, 9,
                    13, 15, 16, 17, 34, 35, 36
                ],

                'contactBodies': [
                    'left_hip_yaw_link',
                    'left_knee_link',
                    'left_ankle_roll_link',
                    'right_hip_yaw_link',
                    'right_knee_link',
                    'right_ankle_roll_link',
                    'torso_link',
                    'mid360_link',
                    'left_shoulder_yaw_link',
                    'left_elbow_link',
                    'left_rubber_hand',
                    'right_shoulder_yaw_link',
                    'right_elbow_link',
                    'right_rubber_hand'
                ],

                'contactIndex': [
                    1, 2, 3, 5, 6, 7, 9,
                    13, 15, 16, 17, 34, 35, 36
                ],

                'terminationHeight': 0.15,
                'enableEarlyTermination': True,
                'scaling': 0.8,
                'initRootHeight': 0.95,

                'asset': {
                    'assetRoot': 'ultra/data/assets'
                },

                'plane': {
                    'staticFriction': 1.0,
                    'dynamicFriction': 1.0,
                    'restitution': 0
                },

                'rewardWeights': {
                    'p': 10.0,
                    'r': 2.0,
                    'pv': 1.0,
                    'rv': 0.0,
                    'op': 0.0,
                    'or': 0.1,
                    'opv': 0.01,
                    'orv': 0.0,
                    'ig': 20.0,
                    'cg1': 5.0,
                    'cg2': 5.0,
                    'eg1': 0.0,
                    'eg2': 1e-06,
                    'eg3': 1e-10
                },

                'seed': 9483
            },

            'smooth_rewards': {
                'scales': {
                    'base_lin_vel': -0.1,
                    'ang_vel': -0.1,
                    'dof_vel': -0.001,
                    'action_rate': -0.1,
                    'torques': -0.003,
                    'dof_pos_limits': -5,
                    'dof_torque_limits': -5,
                    'energy': -0.0001,
                    'dof_acc': -1e-07,
                    'dof_acc_all': -0.0,
                    'torques_vel_all': -0.0
                },
                'soft_torque_limit': 0.95
            },

            'sim': {
                'substeps': 1,

                'physx': {
                    'num_threads': 8,
                    'solver_type': 1,
                    'num_position_iterations': 4,
                    'num_velocity_iterations': 1,
                    'contact_offset': 0.02,
                    'rest_offset': 0.0,
                    'bounce_threshold_velocity': 0.2,
                    'max_depenetration_velocity': 1.0
                },

                'flex': {
                    'num_inner_iterations': 10,
                    'warm_start': 0.25
                }
            },

            'domain_rand': {
                'domain_rand_general': False,
                'randomize_gravity': False,
                'gravity_rand_interval_s': 4,
                'gravity_range': [-0.1, 0.1],
                'randomize_friction': False,
                'friction_range': [0.5, 2.0],
                'randomize_base_mass': False,
                'added_mass_range': [-3.0, 3.0],
                'randomize_base_com': False,
                'added_com_range': [-0.05, 0.05],
                'push_robots': False,
                'push_interval_s': 4,
                'max_push_vel_xy': 1.0,
                'randomize_motor': False,
                'motor_strength_range': [0.8, 1.2],
                'action_delay': False,
                'action_buf_len': 8
            },

            'noise': {
                'add_noise': False,
                'noise_level': 1.0,
                'noise_increasing_steps': 3000,

                'noise_scales': {
                    'dof_pos': 0.01,
                    'dof_vel': 0.1,
                    'ang_vel': 0.05,
                    'imu': 0.05,
                    'root_pos': 0.02
                }
            },

            'control': {
                'stiffness': {
                    'hip_yaw': 150,
                    'hip_roll': 150,
                    'hip_pitch': 200,
                    'knee': 200,
                    'ankle': 20,
                    'shoulder': 40,
                    'elbow': 40,
                    'waist': 200
                },

                'damping': {
                    'hip_yaw': 5,
                    'hip_roll': 5,
                    'hip_pitch': 5,
                    'knee': 5,
                    'ankle': 4,
                    'shoulder': 10,
                    'elbow': 10,
                    'waist': 5
                },

                'action_scale': 3,
                'control_type': 'P'
            },

            'name': 'UltraG1',
            'headless': True,

            'task': {
                'randomize': False
            },

            'seed': 9483,

            'args': Namespace(
                test=False,
                play=False,
                resume=0,
                checkpoint=None,
                resume_from='',
                multi_gpu=False,
                task='UltraG1',
                projtype='None',
                task_type='Python',
                rl_device='cuda:0',
                logdir='logs/',
                experiment='Base',
                metadata=False,
                cfg_env='ultra/data/cfg/g1_retarget_smplx.yaml',
                cfg_train='ultra/data/cfg/train/rlg/g1_retarget_smplx.yaml',
                motion_file='InterAct/OMOMO_retarget_supported_080_080_080',
                play_dataset=False,
                init_vel=False,
                save_images=False,
                num_envs=2048,
                task_mode=None,
                obj_obs='points',
                episode_length=0,
                seed=None,
                frames_scale=0.0,
                ball_size=0.0,
                cg1=-1.0,
                cg2=-1.0,
                ig=-1.0,
                op=-1.0,
                max_iterations=0,
                horizon_length=-1,
                minibatch_size=-1,
                randomize=False,
                torch_deterministic=False,
                output_path='output/retarget_smplx',
                llc_checkpoint='',
                enable_cameras=False,
                xr=False,
                device='cuda:0',
                cpu=False,
                verbose=False,
                info=False,
                rendering_mode=None,
                kit_args='',
                anim_recording_enabled=False,
                anim_recording_start_time=0,
                anim_recording_stop_time=10,
                distributed=False,
                headless=True,
                hide_ui=True,
                create_new_stage=False,
                physics_gpu=0,
                active_gpu=0,
                train=True,
                sigma=None
            ),

            'rank': 0,
            'rl_device': 'cuda:0'
        }
    )

    """


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
            loaded_dict['hoi_data'] = torch.cat([loaded_dict['hoi_data'][0:1] for _ in range(30)]+[loaded_dict['hoi_data']], dim=0)
            # Per-clip buffers are kept on the CPU: on the GPU they and the padding copies below exceed 16 GB
            # next to the simulation (~800 OMOMO clips).
            self.hoi_data_dict.append({k: v.cpu() for k, v in loaded_dict.items()})
            hoi_datas.append(loaded_dict['hoi_data'].cpu())
            
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
            hoi_refs.append(hoi_ref.cpu())
        max_length = max(self.max_episode_length) + 30
        self.num_motions = len(hoi_refs)
        self.max_episode_length = to_torch(self.max_episode_length, dtype=torch.long) + 30
        # zero-padded to the longest motion; assembled on the CPU, then copied to the GPU once
        hoi_data = torch.zeros(self.num_motions, max_length, hoi_datas[0].shape[-1])
        hoi_refs_padded = torch.zeros(self.num_motions, max_length, hoi_refs[0].shape[-1])
        for i, (data, ref) in enumerate(zip(hoi_datas, hoi_refs)):
            hoi_data[i, :data.shape[0]] = data
            hoi_refs_padded[i, :ref.shape[0]] = ref
        self.hoi_data = hoi_data.to(self.device)
        self.hoi_refs = hoi_refs_padded.to(self.device).unsqueeze(1)
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
        self._humanoid_root_states[env_ids, 5:6] = 1 / 1.41421356237
        self._humanoid_root_states[env_ids, 6:7] = -1 / 1.41421356237
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
