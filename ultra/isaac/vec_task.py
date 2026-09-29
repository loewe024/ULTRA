"""rl-games facing wrappers around the ULTRA Isaac Lab environments.

rl-games 1.6 still works with ``gym`` (not ``gymnasium``) spaces, so the spaces are built with ``gym`` here.
Resets are driven by the agents (``reset(env_ids)``), see :class:`isaac.base_env.UltraBaseEnv`.
"""

import numpy as np
import torch
from gym import spaces


class UltraVecTask:
    def __init__(self, task, rl_device, clip_observations=np.inf, clip_actions=1.0):
        self.task = task

        self.num_environments = task.num_envs
        self.num_agents = 1  # used for multi-agent environments
        self.num_observations = task.num_obs
        self.num_states = task.num_states
        self.num_actions = task.num_actions

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.inf, np.ones(self.num_obs) * np.inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.inf, np.ones(self.num_states) * np.inf)
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0)
        self._amp_obs_space = spaces.Box(np.ones(task.get_num_amp_obs()) * -np.inf, np.ones(task.get_num_amp_obs()) * np.inf)

        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.rl_device = rl_device
        print("RL device: ", rl_device)

    def get_number_of_agents(self):
        return self.num_agents

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations

    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def step(self, actions):
        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.task.step(actions_tensor)
        return (
            torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device),
            self.task.rew_buf.to(self.rl_device),
            self.task.reset_buf.to(self.rl_device),
            self.task.extras,
        )

    def reset(self, env_ids=None):
        self.task.reset(env_ids)
        return torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)

    def get_env_state(self):
        return None

    def set_env_state(self, env_state):
        pass

    def render(self, mode="human"):
        self.task.render()

    def close(self):
        self.task.close()


class UltraDAggerVecTask(UltraVecTask):
    """Student environments: also return the teacher's actions and (normalized) observation."""

    def _expert(self):
        task = self.task
        curr_obs = (task.obs_buf - task.running_mean.float().to(task.device)) / torch.sqrt(
            task.running_var.float().to(task.device) + 1e-05
        )
        curr_obs = torch.clamp(curr_obs, min=-5.0, max=5.0)
        return {"actions": task.action_buf, "mus": task.mu_buf, "teacher_obs": curr_obs}

    def reset(self, env_ids=None):
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.to(self.task.device)
        self.task.reset(env_ids)
        return torch.clamp(self.task.obs_buf_student, -self.clip_obs, self.clip_obs).to(self.rl_device), self._expert()

    def step(self, actions):
        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.task.step(actions_tensor)
        return (
            torch.clamp(self.task.obs_buf_student, -self.clip_obs, self.clip_obs).to(self.rl_device),
            self.task.rew_buf.to(self.rl_device),
            self.task.reset_buf.to(self.rl_device),
            self.task.extras,
            self._expert(),
        )
