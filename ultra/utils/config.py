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

import os
import yaml

import argparse

import numpy as np
import random
import torch

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]")


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def load_cfg(args):
    with open(os.path.join(os.getcwd(), args.cfg_train), 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        cfg["env"]["numEnvs"] = args.num_envs

    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    cfg["name"] = args.task
    cfg["headless"] = args.headless

    # Set physics domain randomization
    if "task" in cfg:
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}

    logdir = args.logdir
    # Set deterministic mode
    if args.torch_deterministic:
        cfg_train["params"]["torch_deterministic"] = True

    exp_name = cfg_train["params"]["config"]['name']

    if args.experiment != 'Base':
        if args.metadata:
            exp_name = "{}_{}_{}".format(args.experiment, args.task_type, args.device)

            if cfg["task"]["randomize"]:
                exp_name += "_DR"
        else:
             exp_name = args.experiment

    # Override config name
    cfg_train["params"]["config"]['name'] = exp_name

    if args.resume > 0:
        cfg_train["params"]["load_checkpoint"] = True

    if args.checkpoint != "Base":
        cfg_train["params"]["load_path"] = args.checkpoint
        
    if args.llc_checkpoint != "":
        cfg_train["params"]["config"]["llc_checkpoint"] = args.llc_checkpoint

    # Set maximum number of training iterations (epochs)
    if args.max_iterations > 0:
        cfg_train["params"]["config"]['max_epochs'] = args.max_iterations

    cfg_train["params"]["config"]["num_actors"] = cfg["env"]["numEnvs"]

    seed = cfg_train["params"].get("seed", -1)
    if args.seed is not None:
        seed = args.seed
    cfg["seed"] = seed
    cfg_train["params"]["seed"] = seed

    cfg["args"] = args

    return cfg, cfg_train, logdir


def get_args_parser(benchmark=False):
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test, can be used only by rl_games RL library"},
        {"name": "--resume", "type": int, "default": 0,
            "help": "Resume training or start testing from a checkpoint"},
        {"name": "--checkpoint", "type": str, "default": "Base",
            "help": "Path to the saved weights, only for rl_games RL library"},
        {"name": "--resume_from", "type": str, "default": "",
            "help": "Initialize training from a student checkpoint"},
        {"name": "--multi_gpu", "action": "store_true", "default": False,
            "help": "Enable torch.distributed multi-GPU training; launch with torchrun (see scripts/train_student_multigpu.sh)"},
        {"name": "--task", "type": str, "default": "Humanoid",
            "help": "Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity"},
        {"name": "--projtype", "type": str, "default": "None",
            "help": "Can be None, Auto, Mouse"},
        {"name": "--task_type", "type": str,
            "default": "Python", "help": "Choose Python or C++"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
            "help": "Choose CPU or GPU device for inferencing policy network"},
        {"name": "--logdir", "type": str, "default": "logs/"},
        {"name": "--experiment", "type": str, "default": "Base",
            "help": "Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name"},
        {"name": "--metadata", "action": "store_true", "default": False,
            "help": "Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by user"},
        {"name": "--cfg_env", "type": str, "default": "Base", "help": "Environment configuration file (.yaml)"},
        {"name": "--cfg_train", "type": str, "default": "Base", "help": "Training configuration file (.yaml)"},
        {"name": "--motion_file", "type": str,
            "default": "", "help": "Specify reference motion file"},
        {"name": "--play_dataset", "action": "store_true", "default": False,
            "help": "Display the dataset"},
        {"name": "--init_vel", "action": "store_true", "default": False,
            "help": "Init the object velocity at the first frame"},
        {"name": "--save_images", "action": "store_true", "default": False,
            "help": "save images for viewer"},
        {"name": "--num_envs", "type": int, "default": 0,
            "help": "Number of environments to create - override config file"},
        {"name": "--task_mode", "type": str, "default": None,
            "help": "Student observation preset for playback: full_track, sparse_track, object_obs"},
        {"name": "--obj_obs", "type": str, "default": "points",
            "help": "Object observation used when task_mode=object_obs: points, pos, none"},
        {"name": "--episode_length", "type": int, "default": 0,
            "help": "Episode length, by default is read from yaml config"},
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--frames_scale", "type": float, "default": 0.,
            "help": "Set the fps scale for the reference HOI dataset"},
        {"name": "--ball_size", "type": float, "default": 0.,
            "help": "Set the ball size scale"},
        {"name": "--cg1", "type": float, "default": -1.,
            "help": "Set the contact graph reward weight 1"},
        {"name": "--cg2", "type": float, "default": -1.,
            "help": "Set the contact graph reward weight 2"},
        {"name": "--ig", "type": float, "default": -1.,
            "help": "Set the interaction graph reward weight"},
        {"name": "--op", "type": float, "default": -1.,
            "help": "Set the object position reward weight"},
        {"name": "--max_iterations", "type": int, "default": 0,
            "help": "Set a maximum number of training iterations"},
        {"name": "--horizon_length", "type": int, "default": -1,
            "help": "Set number of simulation steps per 1 PPO iteration. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--minibatch_size", "type": int, "default": -1,
            "help": "Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--randomize", "action": "store_true", "default": False,
            "help": "Apply physics domain randomization"},
        {"name": "--torch_deterministic", "action": "store_true", "default": False,
            "help": "Apply additional PyTorch settings for more deterministic behaviour"},
        {"name": "--output_path", "type": str, "default": "output/", "help": "Specify output directory"},
        {"name": "--llc_checkpoint", "type": str, "default": "",
            "help": "Path to the saved weights for the low-level controller of an HRL agent."}]

    if benchmark:
        custom_parameters += [{"name": "--num_proc", "type": int, "default": 1, "help": "Number of child processes to launch"},
                              {"name": "--random_actions", "action": "store_true",
                                  "help": "Run benchmark with random actions instead of inferencing"},
                              {"name": "--bench_len", "type": int, "default": 10,
                                  "help": "Number of timing reports"},
                              {"name": "--bench_file", "action": "store", "help": "Filename to store benchmark results"}]

    parser = argparse.ArgumentParser(description="RL Policy")
    for param in custom_parameters:
        kwargs = {k: v for k, v in param.items() if k != "name"}
        parser.add_argument(param["name"], **kwargs)
    return parser


def finalize_args(args):
    """Derive the legacy train/play flags after parsing (``get_args_parser`` + Isaac Lab AppLauncher args)."""
    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args
