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

import gymnasium as gym
import numpy as np

from isaac.scene_cfg import make_env_cfg
from isaac.vec_task import UltraDAggerVecTask, UltraVecTask

# legacy task name -> (Isaac Lab / gymnasium id, entry point, student/DAgger task)
TASKS = {
    "UltraG1": ("Ultra-G1-RetargetSMPLX-v0", "env.tasks.ultra_g1:UltraG1", False),
    "UltraG1Retarget": ("Ultra-G1-Teacher-v0", "env.tasks.ultra_g1_retarget:UltraG1Retarget", False),
    "UltraDistillObjV2Point": ("Ultra-G1-Student-v0", "env.tasks.ultra_g1_distill_obj_v2vae:UltraDistillObjV2Point", True),
    "UltraDistillObjV3RL": ("Ultra-G1-Finetune-v0", "env.tasks.ultra_g1_distill_obj_v3rl:UltraDistillObjV3RL", True),
}


def register_tasks():
    for gym_id, entry_point, _ in TASKS.values():
        if gym_id not in gym.registry:
            # resets and episode bookkeeping are done by the ULTRA agents, not by gymnasium wrappers
            gym.register(id=gym_id, entry_point=entry_point, disable_env_checker=True, order_enforce=False)


def resolve_task(name):
    """Accept the legacy class name (``UltraG1Retarget``) or the gymnasium id (``Ultra-G1-Teacher-v0``)."""
    if name in TASKS:
        return name
    for legacy_name, (gym_id, _, _) in TASKS.items():
        if name == gym_id:
            return legacy_name
    raise Exception(f"Unrecognized task {name}!\nTask should be one of: {list(TASKS)} or {[t[0] for t in TASKS.values()]}")


def is_distill_task(name):
    return TASKS[resolve_task(name)][2]


def parse_task(args, cfg, cfg_train):
    """Create the Isaac Lab environment of ``args.task`` and wrap it for rl-games."""
    task_name = resolve_task(args.task)
    gym_id, _, distill = TASKS[task_name]
    register_tasks()

    cfg["seed"] = cfg_train["params"].get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]
    cfg["headless"] = args.headless

    env_cfg = make_env_cfg(cfg, device=args.device)
    task = gym.make(gym_id, cfg=env_cfg).unwrapped
    wrapper_cls = UltraDAggerVecTask if distill else UltraVecTask
    env = wrapper_cls(task, args.rl_device, cfg_train.get("clip_observations", np.inf), cfg_train.get("clip_actions", 1.0))

    return task, env
