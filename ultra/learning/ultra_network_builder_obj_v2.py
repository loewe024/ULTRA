import copy
import numpy as np
import torch
import torch.nn as nn

from rl_games.algos_torch import network_builder


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for transformer tokens."""

    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        if self.pe.shape[1] < x.shape[1]:
            raise ValueError(
                f"PositionalEncoding max_len={self.pe.shape[1]} is smaller than sequence {x.shape[1]}"
            )
        return x + self.pe[:, : x.shape[1]]


class SimplePointEncoder(nn.Module):
    """Permutation-invariant encoder that keeps coarse object geometry."""

    def __init__(self, point_dim=3, hidden_dim=128, output_dim=128):
        super().__init__()
        self.point_dim = point_dim
        self.output_dim = output_dim

        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        global_dim = hidden_dim * 2 + 3 + 3 + 9
        global_hidden = max(hidden_dim, output_dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, global_hidden),
            nn.ReLU(),
            nn.Linear(global_hidden, output_dim),
            nn.ReLU(),
        )

    def forward(self, points):
        """
        Args:
            points: [batch, num_points * point_dim]
        """
        batch_size = points.shape[0]
        num_points = max(1, points.shape[1] // max(1, self.point_dim))
        if num_points == 0:
            return torch.zeros(
                batch_size, self.output_dim, device=points.device, dtype=points.dtype
            )

        points = points.reshape(batch_size, num_points, self.point_dim)
        centroid = points.mean(dim=1)
        rel_points = points - centroid.unsqueeze(1)

        point_features = self.point_mlp(rel_points)
        global_max = torch.max(point_features, dim=1).values
        global_mean = torch.mean(point_features, dim=1)

        axis_extents = torch.max(points, dim=1).values - torch.min(points, dim=1).values
        covariance = torch.bmm(rel_points.transpose(1, 2), rel_points) / float(num_points)
        covariance = covariance.reshape(batch_size, -1)

        global_features = torch.cat(
            [centroid, axis_extents, covariance, global_max, global_mean], dim=1
        )
        return self.global_mlp(global_features)


class UltraBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, point_cloud_cfg=None, **kwargs):
            self.point_cloud_cfg = point_cloud_cfg or {}
            self._input_shape = kwargs.get("input_shape")
            params_local = copy.deepcopy(params)
            if params_local.get("disable_double_critic", True):
                params_local["separate"] = False
            super().__init__(params_local, **kwargs)

            self.obs_dim = int(np.prod(self._input_shape))
            self.modal_cfg = params_local.get("modal_config", {})
            self.vae_dim = params_local.get("vae_dim", self.modal_cfg.get("vae_dim", 64))
            self.debug_modal_layout = self.modal_cfg.get("debug_modal_layout", False)
            self.mask_routing = self.modal_cfg.get("mask_routing", True)

            self.token_dim = self.modal_cfg.get("token_dim", 192)
            self.modal_hidden = self.modal_cfg.get("fusion_hidden", 384)
            self.transformer_heads = self.modal_cfg.get("transformer_heads", 2)
            self.transformer_layers = self.modal_cfg.get("transformer_layers", 2)
            self.transformer_ffn = self.modal_cfg.get(
                "transformer_ffn_dim", self.token_dim * 4
            )
            self.transformer_dropout = self.modal_cfg.get("transformer_dropout", 0.0)
            self.film_scale = self.modal_cfg.get("film_scale", 0.1)
            self.teacher_obs_dim = self.modal_cfg.get("teacher_obs_dim", 4052)

            self._init_point_encoder()
            self._init_modal_dimensions()
            self._build_modal_encoders()
            self._build_prior_heads()
            self._build_actor_trunk()
            self._build_critic_trunk()

        def forward(self, obs_dict):
            if obs_dict.get("with_vae", False):
                actor_outputs, states = self.get_action_and_vae_outputs(obs_dict)
            else:
                actor_outputs = self.act(obs_dict)
                states = obs_dict.get("rnn_states", None)

            obs = obs_dict["obs"]
            if obs_dict.get("skip_critic", False):
                value = torch.zeros(
                    obs.shape[0], 1, device=obs.device, dtype=obs.dtype
                )
            else:
                value = self.eval_critic(obs)
            return actor_outputs + (value, states)

        # ------------------------------------------------------------------ #
        # Initialization helpers
        # ------------------------------------------------------------------ #
        def _init_point_encoder(self):
            cfg = self.point_cloud_cfg
            self.point_encoder_enabled = cfg.get("enabled", True)
            self.point_dim = cfg.get("point_dim", 3)
            self.num_points = cfg.get("num_points", max(1, cfg.get("raw_obs_dim", 0) // max(1, self.point_dim)))
            default_raw_dim = self.num_points * self.point_dim
            self.raw_point_obs_dim = cfg.get("raw_obs_dim", default_raw_dim)
            encoder_hidden = cfg.get("hidden_dim", 128)
            encoder_output = cfg.get("output_dim", 128)

            if self.point_encoder_enabled and self.raw_point_obs_dim > 0:
                self.point_encoder = SimplePointEncoder(
                    point_dim=self.point_dim,
                    hidden_dim=encoder_hidden,
                    output_dim=encoder_output,
                )
                self.point_token_proj = nn.Sequential(
                    nn.Linear(encoder_output, encoder_hidden),
                    nn.ReLU(),
                    nn.Linear(encoder_hidden, self.token_dim),
                )
            else:
                self.point_encoder = None
                self.point_token_proj = (
                    nn.Linear(self.raw_point_obs_dim, self.token_dim)
                    if self.raw_point_obs_dim > 0
                    else None
                )

        def _init_modal_dimensions(self):
            cfg = self.modal_cfg
            self.goal_dim = cfg.get("goal_dim", 3)
            self.command_dim = cfg.get("command_dim", 3)
            self.local_goal_dim = cfg.get("local_goal_dim", 31)
            self.obj_trans_dim = cfg.get("obj_trans_dim", 3)
            self.obj_rot_dim = cfg.get("obj_rot_dim", 6)
            self.obj_pos_dim = cfg.get("obj_pos_dim", 3)

            default_task = (
                self.obj_trans_dim
                + self.obj_rot_dim
                + self.obj_pos_dim
                + self.raw_point_obs_dim
            )
            
            self.task_obs_dim = cfg.get("task_obs_dim", max(0, default_task))

            default_mask = (
                self.goal_dim
                + self.local_goal_dim
                + self.obj_trans_dim
                + self.obj_rot_dim
                + self.obj_pos_dim
                + self.raw_point_obs_dim
                + self.command_dim
            )
            self.mask_dim = cfg.get("mask_dim", max(0, default_mask))
            self.expected_mask_dim = max(0, default_mask)

            reserved = self.goal_dim + self.command_dim + self.task_obs_dim + self.mask_dim
            reserved += self.local_goal_dim
            self.body_dim = cfg.get("proprio_dim", self.obs_dim - reserved)
            if self.body_dim <= 0:
                self.body_dim = max(1, self.obs_dim - (self.goal_dim + self.task_obs_dim))

            consumed = (
                self.goal_dim
                + self.command_dim
                + self.local_goal_dim
                + self.body_dim
                + self.task_obs_dim
                + self.mask_dim
            )
            if consumed < self.obs_dim:
                self.body_dim += self.obs_dim - consumed

            tokens = 0
            tokens += int(self.body_dim > 0)
            tokens += int(self.command_dim > 0)
            tokens += int(self.goal_dim > 0)
            tokens += int(self.local_goal_dim > 0)
            tokens += int(self.obj_trans_dim > 0)
            tokens += int(self.obj_rot_dim > 0)
            tokens += int(self.obj_pos_dim > 0)
            tokens += int(self.raw_point_obs_dim > 0)
            tokens += int(self.mask_dim > 0)
            self.num_modal_tokens = max(1, tokens)
            self.total_tokens = self.num_modal_tokens + 1  # add context token
            self.flat_token_dim = self.num_modal_tokens * self.token_dim
            if self.debug_modal_layout:
                expected = (
                    self.goal_dim
                    + self.command_dim
                    + self.local_goal_dim
                    + self.body_dim
                    + self.task_obs_dim
                    + self.mask_dim
                )
                print(
                    "[UltraBuilder] obs_dim={} expected_total={} goal_dim={} command_dim={} local_goal_dim={} "
                    "body_dim={} task_obs_dim={} mask_dim={} raw_point_obs_dim={} obj_trans_dim={} obj_rot_dim={} "
                    "obj_pos_dim={} tokens={}".format(
                        self.obs_dim,
                        expected,
                        self.goal_dim,
                        self.command_dim,
                        self.local_goal_dim,
                        self.body_dim,
                        self.task_obs_dim,
                        self.mask_dim,
                        self.raw_point_obs_dim,
                        self.obj_trans_dim,
                        self.obj_rot_dim,
                        self.obj_pos_dim,
                        self.num_modal_tokens,
                    )
                )
                print("default_task", default_task)

        def _build_modal_encoders(self):
            hidden = max(self.token_dim, self.modal_hidden)
            activation = nn.GELU()

            self.goal_proj = (
                nn.Sequential(
                    nn.Linear(self.goal_dim, hidden),
                    activation,
                    nn.Linear(hidden, self.token_dim),
                )
                if self.goal_dim > 0
                else None
            )

            self.command_proj = (
                nn.Sequential(
                    nn.Linear(self.command_dim, hidden),
                    activation,
                    nn.Linear(hidden, self.token_dim),
                )
                if self.command_dim > 0
                else None
            )

            self.body_proj = nn.Sequential(
                nn.Linear(self.body_dim, hidden),
                activation,
                nn.Linear(hidden, self.token_dim),
            )
            self.local_goal_proj = (
                nn.Sequential(
                    nn.Linear(self.local_goal_dim, hidden),
                    activation,
                    nn.Linear(hidden, self.token_dim),
                )
                if self.local_goal_dim > 0
                else None
            )
            self.obj_trans_proj = (
                nn.Sequential(
                    nn.Linear(self.obj_trans_dim, hidden),
                    activation,
                    nn.Linear(hidden, self.token_dim),
                )
                if self.obj_trans_dim > 0
                else None
            )
            self.obj_rot_proj = (
                nn.Sequential(
                    nn.Linear(self.obj_rot_dim, hidden),
                    activation,
                    nn.Linear(hidden, self.token_dim),
                )
                if self.obj_rot_dim > 0
                else None
            )
            self.obj_pos_proj = (
                nn.Sequential(
                    nn.Linear(self.obj_pos_dim, hidden),
                    activation,
                    nn.Linear(hidden, self.token_dim),
                )
                if self.obj_pos_dim > 0
                else None
            )
            self.mask_proj = (
                nn.Sequential(
                    nn.Linear(self.mask_dim, hidden),
                    activation,
                    nn.Linear(hidden, self.token_dim),
                )
                if self.mask_dim > 0
                else None
            )

            self.context_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
            self.pos_encoding = PositionalEncoding(
                d_model=self.token_dim, max_len=self.total_tokens + 1
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.token_dim,
                nhead=self.transformer_heads,
                dim_feedforward=self.transformer_ffn,
                dropout=self.transformer_dropout,
                activation="gelu",
                batch_first=True,
            )
            self.modal_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=self.transformer_layers
            )
            self.modal_projection = nn.Sequential(
                nn.Linear(self.flat_token_dim, self.modal_hidden),
                nn.LayerNorm(self.modal_hidden),
                activation,
                nn.Linear(self.modal_hidden, self.modal_hidden),
                activation,
            )
            self.mask_context_proj = nn.Linear(self.mask_dim, self.token_dim)

        def _build_prior_heads(self):
            hidden = max(128, self.token_dim)
            self.prior_mu_head = nn.Sequential(
                nn.Linear(self.token_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.vae_dim),
            )
            self.prior_logvar_head = nn.Sequential(
                nn.Linear(self.token_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.vae_dim),
            )
            self.latent_decoder = nn.Sequential(
                nn.Linear(self.vae_dim, 256),
                nn.ReLU(),
                nn.Linear(
                    256,
                    self.command_dim
                    + self.goal_dim
                    + self.obj_trans_dim
                    + self.obj_rot_dim,
                ),
            )

            self.encoder_input_dim = (
                self.teacher_obs_dim + self.obs_dim + self.mask_dim
            )
            self.encoder_net = nn.Sequential(
                nn.Linear(self.encoder_input_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
            )
            self.encoder_mu_head = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.vae_dim)
            )
            self.encoder_logvar_head = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.vae_dim)
            )

        def _build_actor_trunk(self):
            self.actor_hidden_dim = self.mu.in_features
            self.actor_mlp_input_dim = self._infer_mlp_input_dim(
                self.actor_mlp, self.actor_hidden_dim
            )

            self.actor_encoder_input = self.body_dim + self.vae_dim
            self.actor_encoder = nn.Sequential(
                nn.Linear(self.actor_encoder_input, self.actor_mlp_input_dim),
                nn.ReLU(),
                nn.Linear(self.actor_mlp_input_dim, self.actor_mlp_input_dim),
                nn.ReLU(),
            )
            self.h_norm = nn.LayerNorm(self.actor_mlp_input_dim)
            self.z_norm = nn.LayerNorm(self.vae_dim, elementwise_affine=False)
            self.film = nn.Linear(self.vae_dim, 2 * self.actor_mlp_input_dim)
            self.local_goal_skip_scale = self.modal_cfg.get("local_goal_skip_scale", 0.1)
            if self.local_goal_dim > 0:
                self.local_goal_skip = nn.Linear(self.local_goal_dim, self.actor_mlp_input_dim)
                nn.init.zeros_(self.local_goal_skip.weight)
                nn.init.zeros_(self.local_goal_skip.bias)
                self.local_goal_pred_head = nn.Sequential(
                    nn.Linear(self.actor_encoder_input, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.local_goal_dim),
                )
            else:
                self.local_goal_skip = None
                self.local_goal_pred_head = None

        def _build_critic_trunk(self):
            self.critic_hidden_dim = self.value.in_features
            self.critic_encoder = nn.Sequential(
                nn.Linear(self.modal_hidden, self.critic_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.critic_hidden_dim, self.critic_hidden_dim),
                nn.ReLU(),
            )

        def _infer_mlp_input_dim(self, module, default_dim):
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
            return default_dim

        # ------------------------------------------------------------------ #
        # Observation processing
        # ------------------------------------------------------------------ #
        def _split_modalities(self, obs):
            cursor = 0
            goal = (
                obs[:, cursor : cursor + self.goal_dim] if self.goal_dim > 0 else None
            )
            cursor += self.goal_dim

            command = (
                obs[:, cursor : cursor + self.command_dim]
                if self.command_dim > 0
                else None
            )
            cursor += self.command_dim

            local_goal = (
                obs[:, cursor : cursor + self.local_goal_dim]
                if self.local_goal_dim > 0
                else None
            )
            cursor += self.local_goal_dim

            body = obs[:, cursor : cursor + self.body_dim]
            cursor += self.body_dim

            task = obs[:, cursor : cursor + self.task_obs_dim]
            cursor += self.task_obs_dim

            mask = (
                obs[:, cursor : cursor + self.mask_dim] if self.mask_dim > 0 else None
            )
            cursor += self.mask_dim

            if cursor < obs.shape[1]:
                # Safety: append any leftover dims to body representation.
                raise RuntimeError("Leftover observation dimensions detected.")
                remainder = obs[:, cursor:]
                body = torch.cat([body, remainder], dim=-1)

            obj_trans = (
                task[:, : self.obj_trans_dim] if self.obj_trans_dim > 0 else None
            )
            obj_rot = (
                task[:, self.obj_trans_dim : self.obj_trans_dim + self.obj_rot_dim]
                if self.obj_rot_dim > 0
                else None
            )
            obj_pos = (
                task[:, 
                    self.obj_trans_dim
                    + self.obj_rot_dim : self.obj_trans_dim
                    + self.obj_rot_dim
                    + self.obj_pos_dim
                ]
                if self.obj_pos_dim > 0
                else None
            )
            obj_points = (
                task[:, 
                    self.obj_trans_dim
                    + self.obj_rot_dim
                    + self.obj_pos_dim :
                ]
                if self.raw_point_obs_dim > 0
                else None
            )

            return {
                "goal": goal,
                "command": command,
                "local_goal": local_goal,
                "body": body,
                "obj_trans": obj_trans,
                "obj_rot": obj_rot,
                "obj_pos": obj_pos,
                "obj_points": obj_points,
                "mask": mask,
            }

        def _split_mask_segments(self, mask):
            segments = {}
            if mask is None or mask.numel() == 0:
                return segments
            cursor = 0

            def take(length):
                nonlocal cursor
                if length <= 0 or cursor >= mask.shape[1]:
                    return None
                end = cursor + length
                if end > mask.shape[1]:
                    end = mask.shape[1]
                segment = mask[:, cursor:end]
                cursor = end
                return segment

            segments["goal"] = take(self.goal_dim)
            segments["local_goal"] = take(self.local_goal_dim)
            segments["obj_trans"] = take(self.obj_trans_dim)
            segments["obj_rot"] = take(self.obj_rot_dim)
            segments["obj_pos"] = take(self.obj_pos_dim)
            segments["obj_points"] = take(self.raw_point_obs_dim)
            segments["command"] = take(self.command_dim)
            return segments

        def _tokens_from_split(self, split, mask_segments=None, apply_mask=False):
            tokens = []
            mask_segments = mask_segments or {}

            def gate(name, tensor):
                if not apply_mask or tensor is None:
                    return tensor
                mask = mask_segments.get(name)
                if mask is None:
                    return tensor
                return tensor * mask

            def route(token, name):
                if not apply_mask or not self.mask_routing or token is None:
                    return token
                mask = mask_segments.get(name)
                if mask is None:
                    return token
                scale = mask.mean(dim=-1, keepdim=True)
                return token * scale

            body_token = self.body_proj(split["body"])
            tokens.append(body_token.unsqueeze(1))

            if self.command_proj is not None and split["command"] is not None:
                command_feat = gate("command", split["command"])
                command_token = self.command_proj(command_feat)
                tokens.append(route(command_token, "command").unsqueeze(1))

            if self.goal_proj is not None and split["goal"] is not None:
                goal_feat = gate("goal", split["goal"])
                goal_token = self.goal_proj(goal_feat)
                tokens.append(route(goal_token, "goal").unsqueeze(1))

            if self.local_goal_proj is not None and split["local_goal"] is not None:
                local_feat = gate("local_goal", split["local_goal"])
                local_token = self.local_goal_proj(local_feat)
                tokens.append(route(local_token, "local_goal").unsqueeze(1))

            if self.obj_trans_proj is not None and split["obj_trans"] is not None:
                trans_feat = gate("obj_trans", split["obj_trans"])
                trans_token = self.obj_trans_proj(trans_feat)
                tokens.append(route(trans_token, "obj_trans").unsqueeze(1))

            if self.obj_rot_proj is not None and split["obj_rot"] is not None:
                rot_feat = gate("obj_rot", split["obj_rot"])
                rot_token = self.obj_rot_proj(rot_feat)
                tokens.append(route(rot_token, "obj_rot").unsqueeze(1))

            if self.obj_pos_proj is not None and split["obj_pos"] is not None:
                pos_feat = gate("obj_pos", split["obj_pos"])
                pos_token = self.obj_pos_proj(pos_feat)
                tokens.append(route(pos_token, "obj_pos").unsqueeze(1))

            if self.raw_point_obs_dim > 0 and split["obj_points"] is not None:
                point_input = gate("obj_points", split["obj_points"])
                if self.point_encoder is not None:
                    point_features = self.point_encoder(point_input)
                else:
                    point_features = point_input

                point_token = (
                    self.point_token_proj(point_features)
                    if self.point_token_proj is not None
                    else point_features
                )
                tokens.append(route(point_token, "obj_points").unsqueeze(1))

            if self.mask_proj is not None and split["mask"] is not None:
                tokens.append(self.mask_proj(split["mask"]).unsqueeze(1))

            return torch.cat(tokens, dim=1)

        def _encode_modalities(self, obs, apply_mask=False):
            split = self._split_modalities(obs)
            mask_segments = self._split_mask_segments(split["mask"])
            tokens = self._tokens_from_split(split, mask_segments, apply_mask=apply_mask)
            batch = obs.shape[0]
            context = self.context_token.expand(batch, -1, -1)
            if apply_mask:
                if split["mask"] is None:
                    mask_vec = torch.zeros(batch, self.mask_dim, device=obs.device, dtype=obs.dtype)
                else:
                    pad = max(0, self.mask_dim - split["mask"].shape[1])
                    mask_vec = split["mask"]
                    if pad > 0:
                        mask_vec = torch.cat(
                            [mask_vec, torch.zeros(batch, pad, device=obs.device, dtype=obs.dtype)],
                            dim=-1,
                        )
                    elif pad < 0:
                        mask_vec = mask_vec[:, : self.mask_dim]
                context = context + self.mask_context_proj(mask_vec).unsqueeze(1)
            seq = torch.cat([context, tokens], dim=1)
            seq = self.pos_encoding(seq)
            encoded = self.modal_transformer(seq)
            context_vec = encoded[:, 0]
            modal_tokens = encoded[:, 1:]
            flat_tokens = modal_tokens.reshape(batch, -1)
            fused = self.modal_projection(flat_tokens)
            return {"context": context_vec, "tokens": modal_tokens, "fused": fused}

        # ------------------------------------------------------------------ #
        # VAE components
        # ------------------------------------------------------------------ #
        def _prior(self, input_dict):
            encoded = self._encode_modalities(input_dict["obs"], apply_mask=True)
            context = encoded["context"]
            mu = self.prior_mu_head(context)
            logvar = self.prior_logvar_head(context)
            return {"mu": mu, "logvar": logvar}

        def _encoder(self, input_dict):
            teacher_obs = input_dict["teacher_obs"]
            student_obs = input_dict["obs"]
            split = self._split_modalities(student_obs)
            if split["mask"] is None or split["mask"].shape[1] == 0:
                mask = torch.zeros(
                    student_obs.shape[0],
                    self.mask_dim,
                    device=student_obs.device,
                    dtype=student_obs.dtype,
                )
            else:
                pad = self.mask_dim - split["mask"].shape[1]
                mask = split["mask"]
                if pad > 0:
                    mask = torch.cat(
                        [
                            mask,
                            torch.zeros(
                                student_obs.shape[0],
                                pad,
                                device=student_obs.device,
                                dtype=student_obs.dtype,
                            ),
                        ],
                        dim=-1,
                    )
                elif pad < 0:
                    mask = mask[:, : self.mask_dim]
            encoder_input = torch.cat([teacher_obs, student_obs, mask], dim=-1)
            embedding = self.encoder_net(encoder_input)
            mu = self.encoder_mu_head(embedding)
            logvar = self.encoder_logvar_head(embedding)
            return {"mu": mu, "logvar": logvar}

        def _trunk(self, input_dict):
            z = input_dict["vae_latent"]
            obs = input_dict["obs"]
            split = self._split_modalities(obs)
            body_obs = split["body"]
            actor_in = torch.cat([body_obs, z], dim=-1)
            h = self.actor_encoder(actor_in)
            h = self.h_norm(h)
            local_goal_pred = None
            if self.local_goal_pred_head is not None:
                local_goal_pred = self.local_goal_pred_head(actor_in)
                input_dict["local_goal_pred"] = local_goal_pred

            if self.local_goal_skip is not None and split["local_goal"] is not None:
                local_goal = split["local_goal"]
                mask_scale = None
                if split.get("mask") is not None:
                    mask_segments = self._split_mask_segments(split["mask"])
                    mask = mask_segments.get("local_goal")
                    if mask is not None and mask.numel() > 0:
                        mask_scale = mask.mean(dim=-1, keepdim=True)
                if mask_scale is not None and local_goal_pred is not None:
                    keep = (mask_scale > 0.5)
                    local_goal = torch.where(keep, local_goal, local_goal_pred)
                skip = self.local_goal_skip(local_goal)
                h = h + self.local_goal_skip_scale * skip

            z_norm = self.z_norm(z)
            gamma, beta = self.film(z_norm).chunk(2, dim=-1)
            gamma = 1.0 + self.film_scale * torch.tanh(gamma)
            beta = self.film_scale * torch.tanh(beta)
            h = gamma * h + beta

            a_h = self.actor_mlp(h)

            mu = self.mu_act(self.mu(a_h))
            if self.space_config["fixed_sigma"]:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(a_h))

            return mu, sigma

        def eval_critic(self, obs):
            modal_encoding = self._encode_modalities(obs, apply_mask=False)
            value_features = self.critic_encoder(modal_encoding["fused"])
            value = self.value_act(self.value(value_features))
            return value

        def reparameterization(self, mean, std, vae_noise):
            return mean + std * vae_noise

        def act(self, input_dict):
            prior_out = self._prior(input_dict)
            if input_dict.get("with_encoder", False):
                encoder_out = self._encoder(input_dict)
                mu = prior_out["mu"] + encoder_out["mu"]
                logvar = encoder_out["logvar"]
            else:
                mu = prior_out["mu"]
                logvar = prior_out["logvar"]

            if "vae_noise" not in input_dict:
                input_dict["vae_noise"] = torch.randn_like(mu)

            z = self.reparameterization(
                mu,
                torch.exp(0.5 * logvar),
                input_dict["vae_noise"],
            )
            input_dict["vae_latent"] = z
            action = self._trunk(input_dict)
            return action

        def get_action_and_vae_outputs(self, input_dict):
            prior_out = self._prior(input_dict)
            encoder_out = self._encoder(input_dict)

            mu = prior_out["mu"] + encoder_out["mu"]
            logvar = encoder_out["logvar"]

            if "vae_noise" not in input_dict:
                input_dict["vae_noise"] = torch.randn_like(mu)

            z = self.reparameterization(
                mu,
                torch.exp(0.5 * logvar),
                input_dict["vae_noise"],
            )

            input_dict["vae_latent"] = z
            action = self._trunk(input_dict)
            aux_pred = self.latent_decoder(z)
            local_goal_pred = input_dict.get("local_goal_pred")
            rnn_states = {
                "prior_out": prior_out,
                "encoder_out": encoder_out,
                "aux_pred": aux_pred,
                "local_goal_pred": local_goal_pred,
            }
            return action, rnn_states

    def build(self, name, **kwargs):
        point_cfg = self.params.get("point_cloud", {})
        return UltraBuilder.Network(self.params, point_cloud_cfg=point_cfg, **kwargs)
