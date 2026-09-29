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

import torch
import numpy as np
import learning.ultra_players as ultra_players


class UltraPlayerContinuousDistill(ultra_players.UltraPlayerContinuous):
    """Player for the multimodal VAE student.

    At inference the latent is sampled from the prior only (no privileged encoder).
    One noise vector per environment is kept for the whole episode and resampled on reset,
    so the latent does not jitter from step to step.
    """

    def __init__(self, params):
        super().__init__(params)
        self._last_expert = None
        self._vae_noise = None
        return

    def env_reset(self, env_ids=None):
        obs, expert = self.env.reset(env_ids)
        self._last_expert = expert
        obs = self.obs_to_torch(obs)
        if self._vae_noise is not None:
            if env_ids is None:
                self._vae_noise = None
            else:
                self._vae_noise[env_ids] = torch.randn_like(self._vae_noise[env_ids])
        return obs

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos, expert = env.step(actions)
        self._last_expert = expert

        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]

        if self._vae_noise is not None:
            if self.is_tensor_obses:
                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            else:
                done_indices = torch.as_tensor(np.where(dones)[0], device=self._vae_noise.device)
            if done_indices.numel() > 0:
                self._vae_noise[done_indices] = torch.randn_like(self._vae_noise[done_indices])

        if self.is_tensor_obses:
            return obs, rewards.to(self.device), dones.to(self.device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def get_action(self, obs_dict, is_deterministic=False):
        obs = obs_dict['obs']
        obs = self._preproc_obs(obs)

        if self._vae_noise is None or self._vae_noise.shape[0] != obs.shape[0]:
            vae_dim = getattr(self.model.a2c_network, 'vae_dim', self.config.get('vae_dim', 64))
            self._vae_noise = torch.randn(obs.shape[0], vae_dim, device=self.device)

        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states,
            'with_encoder': False,
            'with_vae': False,
            'vae_noise': self._vae_noise,
            'skip_critic': True,
        }
        if self._last_expert is not None and 'teacher_obs' in self._last_expert:
            input_dict['teacher_obs'] = self._last_expert['teacher_obs'].to(self.device)

        self.model.eval()
        with torch.no_grad():
            res_dict = self.model(input_dict)

        action = res_dict['mus'] if is_deterministic else res_dict['actions']
        return torch.clamp(action, -1.0, 1.0)
