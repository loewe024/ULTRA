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
from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import central_value
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common

import torch
from torch import optim

import learning.amp_datasets as amp_datasets
from learning import custom_schedulers
from learning.ultra_models import load_checkpoint


class CommonAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        config = params['config']
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)

        self._load_config_params(config)

        self._setup_action_space()
        self._save_intermediate = config.get('save_intermediate', False)
        self._setup_custom_scheduler(config)

        net_config = self._build_net_config()
        self.model = self.network.build(net_config)
        self.model.to(self.ppo_device)
        self.states = None

        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)

        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape,
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length, 
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_length' : self.seq_length,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
                'zero_rnn_on_done' : self.zero_rnn_on_done
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = amp_datasets.AMPDataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std
        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)
        
        return

    def _setup_custom_scheduler(self, config):
        """Learning-rate schedules added by ULTRA on top of the rl-games ones."""
        lr_schedule = config.get('lr_schedule')
        schedulers = {'curriculum': custom_schedulers.CurriculumScheduler, 'cosine': custom_schedulers.CosineScheduler}
        if lr_schedule not in schedulers:
            return
        initial_lr = float(config['learning_rate'])
        self.scheduler = schedulers[lr_schedule](
            initial_lr=initial_lr,
            final_lr=config.get('lr_final', initial_lr * 0.25),  # Default: 25% of initial
            warmup_end=config.get('lr_warmup_end', 500),
            decay_end=config.get('lr_decay_end', 5500),
            apply_to_entropy=config.get('schedule_entropy', False),
            start_entropy_coef=config.get('entropy_coef')
        )
        return

    def init_tensors(self):
        super().init_tensors()
        self.experience_buffer.tensor_dict['next_obses'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        self.experience_buffer.tensor_dict['next_values'] = torch.zeros_like(self.experience_buffer.tensor_dict['values'])

        self.tensor_list += ['next_obses']
        return

    def set_full_state_weights(self, weights, set_epoch=True):
        self.set_weights(weights)
        if set_epoch:
            self.epoch_num = weights.get('epoch', 0)
            self.epoch_num_start = weights.get('epoch', 0)
        if self.has_central_value and 'assymetric_vf_nets' in weights:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])
        if 'optimizer' in weights:
            self.optimizer.load_state_dict(weights['optimizer'])
        if set_epoch:
            self.frame = weights.get('frame', 0)
        self.last_mean_rewards = weights.get('last_mean_rewards', -100500)

        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

        return

    def restore(self, fn, set_epoch=True):
        checkpoint = load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)
        
        advantages = self._calc_advs(batch_dict)

        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

        return

    def discount_values(self, mb_fdones, mb_values, mb_rewards, mb_next_values):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam

        return mb_advs

    def env_reset(self, env_ids=None):
        obs = self.vec_env.reset(env_ids)
        obs = self.obs_to_tensors(obs)
        return obs

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def _get_mean_rewards(self):
        return self.game_rewards.get_mean()

    def _load_config_params(self, config):
        self.last_lr = config['learning_rate']
        return

    def _build_net_config(self):
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : self.obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        return config

    def _setup_action_space(self):
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)
        return

    def _init_train(self):
        return

    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs = obs_dict['obs']
        processed_obs = self.model.norm_obs(self._preproc_obs(obs))
        value = self.model.a2c_network.eval_critic(processed_obs)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def _actor_loss(self, old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip):
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip,
                                    1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)

        clipped = torch.abs(ratio - 1.0) > curr_e_clip
        clipped = clipped.detach()
        
        info = {
            'actor_loss': a_loss,
            'actor_clipped': clipped.detach()
        }
        return info

    def _critic_loss(self, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
        if clip_value:
            value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
            value_losses = (values - return_batch)**2
            value_losses_clipped = (value_pred_clipped - return_batch)**2
            c_loss = torch.max(value_losses, value_losses_clipped)
        else:
            c_loss = (return_batch - values)**2

        info = {
            'critic_loss': c_loss
        }
        return info
    
    def _calc_advs(self, batch_dict):
        returns = batch_dict['returns']
        values = batch_dict['values']

        advantages = returns - values
        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def _record_train_batch_info(self, batch_dict, train_info):
        return

    def _log_train_info(self, train_info, frame):
        self.writer.add_scalar('performance/update_time', train_info['update_time'], frame)
        self.writer.add_scalar('performance/play_time', train_info['play_time'], frame)
        self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(train_info['actor_loss']).item(), frame)
        self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(train_info['critic_loss']).item(), frame)
        
        self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(train_info['b_loss']).item(), frame)
        self.writer.add_scalar('losses/entropy', torch_ext.mean_list(train_info['entropy']).item(), frame)
        self.writer.add_scalar('info/last_lr', train_info['last_lr'][-1] * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/lr_mul', train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/e_clip', self.e_clip * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/clip_frac', torch_ext.mean_list(train_info['actor_clip_frac']).item(), frame)
        self.writer.add_scalar('info/kl', torch_ext.mean_list(train_info['kl']).item(), frame)
        return
