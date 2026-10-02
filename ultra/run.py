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

"""Train or play ULTRA policies (all stages) with Isaac Lab and rl-games.

    python ultra/run.py --task UltraG1Retarget --cfg_env ... --cfg_train ... [--headless] [--test --checkpoint ...]

``--task`` takes the ULTRA task name (``UltraG1``, ``UltraG1Retarget``, ``UltraDistillObjV2Point``,
``UltraDistillObjV3RL``) or its gymnasium id (``Ultra-G1-RetargetSMPLX-v0``, ``Ultra-G1-Teacher-v0``,
``Ultra-G1-Student-v0``, ``Ultra-G1-Finetune-v0``). Isaac Lab's AppLauncher options (``--headless``,
``--device``, ``--enable_cameras``, ...) are accepted as well.
"""

import os

# Run the @torch.jit.script helpers eagerly. With torch 2.7 TorchScript re-specializes them for every new batch
# size, and the observation function then takes ~3 minutes per partial reset (eager: milliseconds). Must be set
# before torch is imported; PYTORCH_JIT=1 restores scripting.
os.environ.setdefault("PYTORCH_JIT", "0")

from isaaclab.app import AppLauncher

from utils.config import finalize_args, get_args_parser

parser = get_args_parser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# one process per GPU (torchrun); AppLauncher binds each process to cuda:LOCAL_RANK
args.distributed = args.multi_gpu
if args.save_images:
    # frames are read from the viewport camera, which needs rendering in headless mode
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
args = finalize_args(args)

from utils.config import set_np_formatting, set_seed, load_cfg  # noqa: E402
from utils.parse_task import is_distill_task, parse_task, resolve_task  # noqa: E402

from rl_games.algos_torch import model_builder, torch_ext  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import AlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import torch  # noqa: E402

from isaac.vec_task import UltraDAggerVecTask  # noqa: E402
from learning import ultra_models  # noqa: E402

cfg = None
cfg_train = None
created_envs = []


def create_rlgpu_env(**kwargs):
    """
    Works for:
      - Single GPU (python ultra/run.py ...)
      - Multi-GPU via torchrun (one process per GPU, --multi_gpu)
    """
    rank = app_launcher.global_rank if args.distributed else 0
    local_rank = app_launcher.local_rank if args.distributed else 0

    # ----- per-rank seeding & device binding -----
    # mirror the old behavior: seed += rank
    cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        args.device = f'cuda:{local_rank}'
        args.rl_device = f'cuda:{local_rank}'
    else:
        args.device = 'cpu'
        args.rl_device = 'cpu'
    cfg['rank'] = rank
    cfg['rl_device'] = args.rl_device

    # ----- build sim & env -----
    task, env = parse_task(args, cfg, cfg_train)
    created_envs.append(task)

    print('rank:', rank, ' local_rank:', local_rank)
    print('num_envs: {:d}'.format(env.num_envs))
    print('num_actions: {:d}'.format(env.num_actions))
    print('num_obs: {:d}'.format(env.num_obs))
    print('num_states: {:d}'.format(env.num_states))
    return env


class RLGPUAlgoObserver(AlgoObserver):
    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0 and self.writer is not None:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
        return


class RLGPUEnv(vecenv.IVecEnv):
    """rl-games vector env; the student (DAgger) environments additionally return the teacher's outputs."""

    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)
        self.with_expert = isinstance(self.env, UltraDAggerVecTask)

        self.full_state = {}
        self.reset()
        return

    def step(self, action):
        result = self.env.step(action)
        next_obs, reward, is_done, info = result[:4]

        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            obs = self.full_state
        else:
            obs = self.full_state["obs"]
        if self.with_expert:
            return obs, reward, is_done, info, result[4]
        return obs, reward, is_done, info

    def reset(self, env_ids=None):
        result = self.env.reset(env_ids)
        if self.with_expert:
            self.full_state["obs"], expert = result
        else:
            self.full_state["obs"] = result
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            obs = self.full_state
        else:
            obs = self.full_state["obs"]
        if self.with_expert:
            return obs, expert
        return obs

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['amp_observation_space'] = self.env.amp_observation_space

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info

    def get_env_state(self):
        return None

    def set_env_state(self, env_state):
        pass


vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU'})


def build_alg_runner(algo_observer):
    task = resolve_task(args.task)
    if task == 'UltraDistillObjV3RL':
        from learning import ultra_agent_distill_vae_rl, ultra_network_builder_obj_v3, ultra_players_distill
        agent = ultra_agent_distill_vae_rl.UltraAgentDistill
        player = ultra_players_distill.UltraPlayerContinuousDistill
        builder = ultra_network_builder_obj_v3.UltraBuilder
    elif is_distill_task(task):
        from learning import ultra_agent_distill_vae, ultra_network_builder_obj_v2, ultra_players_distill
        agent = ultra_agent_distill_vae.UltraAgentDistill
        player = ultra_players_distill.UltraPlayerContinuousDistill
        builder = ultra_network_builder_obj_v2.UltraBuilder
    else:
        from learning import ultra_agent, ultra_network_builder, ultra_players
        agent = ultra_agent.UltraAgent
        player = ultra_players.UltraPlayerContinuous
        builder = ultra_network_builder.UltraBuilder

    runner = Runner(algo_observer)
    runner.algo_factory.register_builder('ultra', lambda **kwargs: agent(**kwargs))
    runner.player_factory.register_builder('ultra', lambda **kwargs: player(**kwargs))
    model_builder.register_model('ultra', ultra_models.ModelUltraContinuous)
    model_builder.register_network('ultra', builder)

    return runner


def main():
    global cfg
    global cfg_train

    set_np_formatting()
    cfg, cfg_train, logdir = load_cfg(args)

    if args.resume_from:
        cfg_train['params']['config']['resume_from'] = args.resume_from

    cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

    # Force multi_gpu setting from command-line argument to override YAML config
    cfg_train['params']['config']['multi_gpu'] = args.multi_gpu

    if args.horizon_length != -1:
        cfg_train['params']['config']['horizon_length'] = args.horizon_length

    if args.minibatch_size != -1:
        cfg_train['params']['config']['minibatch_size'] = args.minibatch_size
        
    if args.motion_file:
        cfg['env']['motion_file'] = args.motion_file

    if args.play_dataset:
        cfg['env']['playdataset'] = True

    if args.task_mode:
        cfg['env']['task_mode'] = args.task_mode
        cfg['env']['obj_obs'] = args.obj_obs

    if args.projtype:
        cfg['env']['projtype'] = args.projtype

    if args.cg1 != -1.:
        cfg['env']['rewardWeights']['cg1'] = args.cg1

    if args.cg2 != -1.:
        cfg['env']['rewardWeights']['cg2'] = args.cg2

    if args.ig != -1.:
        cfg['env']['rewardWeights']['ig'] = args.ig

    if args.op != -1.:
        cfg['env']['rewardWeights']['op'] = args.op

    if args.save_images:
        cfg['env']['saveImages'] = True
    
    if args.init_vel:
        cfg['env']['initVel'] = True

    if args.frames_scale != 0.:
        cfg['env']['dataFramesScale'] = args.frames_scale

    if args.ball_size != 0.:
        cfg['env']['ballSize'] = args.ball_size
    
    # Create default directories for weights and statistics
    cfg_train['params']['config']['train_dir'] = args.output_path

    # Weights & Biases logging is optional: set WANDB_DISABLED=true to turn it off,
    # and WANDB_ENTITY / WANDB_PROJECT (or wandb_entity / wandb_project in the train yaml) to route it.
    if os.getenv("WANDB_DISABLED", "false").lower() not in ("1", "true"):
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", cfg_train['params']['config'].get('wandb_project', 'ultra')),
            name=cfg_train['params']['config'].get('name', 'ultra_run'),
            entity=os.getenv("WANDB_ENTITY", cfg_train['params']['config'].get('wandb_entity', None)),
            sync_tensorboard=True,
        )

    vargs = vars(args)
    vargs['checkpoint'] = None if args.checkpoint in (None, '', 'Base') else args.checkpoint
    vargs['sigma'] = None

    algo_observer = RLGPUAlgoObserver()

    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    runner.run(vargs)

    # the simulation app only shuts down cleanly once the environment is closed
    for task in created_envs:
        task.close()
    return


if __name__ == '__main__':
    main()
    simulation_app.close()
