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

"""ULTRA policy model on top of rl-games 1.6 (input/value normalization live inside the model).

Checkpoints written by the IsaacGym version (rl-games 1.1.4) keep the observation normalizer next to the model
(``checkpoint['running_mean_std']``); :func:`upgrade_checkpoint` moves it into the model state so both formats load.
"""

import torch

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.models import ModelA2CContinuousLogStd


def upgrade_checkpoint(checkpoint):
    """Convert an rl-games 1.1.4 checkpoint (normalizers stored beside the model) to the 1.6 layout in place."""
    model_state = checkpoint.get('model', {})
    for legacy_key, prefix in (('running_mean_std', 'running_mean_std.'), ('reward_mean_std', 'value_mean_std.')):
        stats = checkpoint.get(legacy_key)
        if stats is None or any(k.startswith(prefix) for k in model_state):
            continue
        for name, value in stats.items():
            model_state[prefix + name] = value
    return checkpoint


def load_checkpoint(path):
    return upgrade_checkpoint(torch_ext.load_checkpoint(path))


def input_normalizer(checkpoint):
    """``(running_mean, running_var)`` of the observation normalizer, or None (``normalize_input: False``)."""
    state = upgrade_checkpoint(checkpoint)['model']
    if 'running_mean_std.running_mean' not in state:
        return None
    return state['running_mean_std.running_mean'], state['running_mean_std.running_var']


def load_model_state(model, checkpoint):
    """Load ``checkpoint['model']`` into ``model``, dropping normalizer entries the model was built without."""
    state = dict(checkpoint['model'])
    own = model.state_dict()
    for prefix in ('running_mean_std.', 'value_mean_std.'):
        if not any(k.startswith(prefix) for k in own):
            state = {k: v for k, v in state.items() if not k.startswith(prefix)}
    model.load_state_dict(state)


class ModelUltraContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('ultra', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelUltraContinuous.Network(
            net,
            obs_shape=config['input_shape'],
            normalize_value=config.get('normalize_value', False),
            normalize_input=config.get('normalize_input', False),
            value_size=config.get('value_size', 1),
        )

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            return

        def forward(self, input_dict):
            # Same as rl-games, but the ULTRA student networks may skip the critic (value is None).
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                return {
                    'prev_neglogp': torch.squeeze(prev_neglogp),
                    'values': value,
                    'entropy': entropy,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma,
                }
            selected_action = distr.sample()
            neglogp = self.neglogp(selected_action, mu, sigma, logstd)
            return {
                'neglogpacs': torch.squeeze(neglogp),
                'values': None if value is None else self.denorm_value(value),
                'actions': selected_action,
                'rnn_states': states,
                'mus': mu,
                'sigmas': sigma,
            }


def load_teacher_policy(model_path, network_params, obs_dim, actions_num, num_seqs, device):
    """Frozen tracking teacher used inside the student environments.

    Returns ``(model, running_mean, running_var)``; the environment normalizes the teacher observation itself,
    so the model is built without an input normalizer.
    """
    from learning import ultra_network_builder

    builder = ultra_network_builder.UltraBuilder()
    builder.load(network_params)
    model = ModelUltraContinuous(builder).build({
        'actions_num': actions_num,
        'input_shape': (obs_dim,),
        'num_seqs': num_seqs,
        'value_size': 1,
    })
    checkpoint = load_checkpoint(model_path)
    running_mean, running_var = input_normalizer(checkpoint)
    load_model_state(model, checkpoint)
    model.to(device)
    model.eval()
    return model, running_mean, running_var
