"""
export_jit.py

Export a trained multimodal VAE student to a TorchScript module mapping observations to actions.
Observation normalization is baked in when the checkpoint carries running statistics
(normalize_input: True); the released config trains with normalize_input: False, in which case the
observation size is taken from the env config (numObsStudent) and identity normalization is used.
"""

import argparse
import copy
import os
from pathlib import Path

import torch
import torch.nn as nn

from learning import ultra_network_builder_obj_v2, ultra_models

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "cfg"
    / "train"
    / "rlg"
    / "g1_student_vae.yaml"
)
_DEFAULT_ENV_CONFIG_PATH = Path(__file__).resolve().parent / "data" / "cfg" / "g1_student_vae.yaml"


def _load_network_config_from_yaml(cfg_path):
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r") as f:
        import yaml
        raw_cfg = yaml.safe_load(f) or {}
    params = raw_cfg.get("params", {})
    network_cfg = copy.deepcopy(params.get("network"))
    if not network_cfg:
        raise ValueError(f"'network' section missing from {cfg_path}")
    if "vae_dim" not in network_cfg:
        vae_dim = params.get("config", {}).get("vae_dim")
        if vae_dim is not None:
            network_cfg["vae_dim"] = vae_dim
    return network_cfg


class ObsToActionModule(nn.Module):
    def __init__(self, policy, running_mean, running_var, vae_dim):
        super().__init__()
        self.register_buffer("running_mean", running_mean.float())
        self.register_buffer("running_var", running_var.float())
        self.policy = policy
        self.vae_dim = int(vae_dim)

    def _forward_impl(self, obs, vae_noise):
        normalized_obs = (obs - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        normalized_obs = torch.clamp(normalized_obs, min=-5.0, max=5.0)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": normalized_obs,
            "rnn_states": None,
            "with_vae": False,
            "with_encoder": False,
            "vae_noise": vae_noise,
            "skip_critic": True,
        }
        action = self.policy.a2c_network.act(input_dict)
        if isinstance(action, (tuple, list)):
            action = action[0]
        return torch.clamp(action, min=-1.0, max=1.0)

    def forward(self, obs):
        # Derive the zero latent noise from obs so the traced graph follows the runtime device/dtype
        # (torch.zeros(..., device=obs.device) would bake the tracing device into the TorchScript).
        vae_noise = torch.zeros_like(obs[:, : self.vae_dim])
        return self._forward_impl(obs, vae_noise)

    def forward_with_noise(self, obs, vae_noise):
        return self._forward_impl(obs, vae_noise)


def export_jit(ckpt_path, save_path, cfg_path=None, device="cpu", obs_dim=None):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = ultra_models.load_checkpoint(ckpt_path)
    running_stats = ultra_models.input_normalizer(ckpt)
    if running_stats is None:
        if obs_dim is None:
            # normalize_input: False (released config) -> no running stats; read the observation size from the env cfg.
            import yaml
            with _DEFAULT_ENV_CONFIG_PATH.open("r") as f:
                env_cfg = yaml.safe_load(f) or {}
            obs_dim = int(env_cfg.get("env", {}).get("numObsStudent", 1496))
            print(f"running_mean_std missing (normalize_input: False); using identity normalization with obs_dim={obs_dim} from {_DEFAULT_ENV_CONFIG_PATH.name}")
        running_mean = torch.zeros(int(obs_dim), dtype=torch.float32)
        running_var = torch.ones(int(obs_dim), dtype=torch.float32)
    else:
        running_mean, running_var = running_stats

    cfg_path = Path(cfg_path) if cfg_path else _DEFAULT_CONFIG_PATH
    network_cfg = _load_network_config_from_yaml(cfg_path)

    config = {
        "actions_num": 29,
        "input_shape": (int(running_mean.shape[0]),),
        "num_seqs": 1,
        "value_size": 1,
    }

    network_builder = ultra_network_builder_obj_v2.UltraBuilder()
    network_builder.load(network_cfg)
    model_wrapper = ultra_models.ModelUltraContinuous(network_builder)
    policy = model_wrapper.build(config)
    ultra_models.load_model_state(policy, ckpt)
    policy.to(device)
    policy.eval()

    vae_dim = getattr(policy.a2c_network, "vae_dim", network_cfg.get("vae_dim", 64))
    jit_module = ObsToActionModule(policy, running_mean, running_var, vae_dim)
    jit_module.to(device)
    jit_module.eval()

    example_obs = torch.randn(1, running_mean.shape[0], device=device)
    example_noise = torch.zeros(1, vae_dim, device=device)
    with torch.no_grad():
        test_out = jit_module(example_obs)

    # Disable trace checking to avoid false positives from internal transformer ops.
    traced = torch.jit.trace_module(
        jit_module,
        {"forward": (example_obs,), "forward_with_noise": (example_obs, example_noise)},
        check_trace=False,
    )

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    torch.jit.save(traced, save_path)

    loaded = torch.jit.load(save_path)
    loaded.to(device)
    with torch.no_grad():
        _ = loaded(example_obs)

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert obj_v2 student checkpoint to JIT model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file (.pth)")
    parser.add_argument("--out", type=str, default=None, help="Output path for JIT model (.pt)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config yaml (default: ultra/data/cfg/train/rlg/g1_student_vae.yaml)",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--obs_dim", type=int, default=None,
                        help="Observation dimension (required if checkpoint lacks running_mean_std)")
    args = parser.parse_args()

    if args.out is None:
        base_dir = os.path.dirname(args.ckpt)
        base_name = os.path.basename(args.ckpt).replace(".pth", "_objv2_jit.pt")
        args.out = os.path.join(base_dir, base_name)

    export_jit(args.ckpt, args.out, cfg_path=args.config, device=args.device, obs_dim=args.obs_dim)
