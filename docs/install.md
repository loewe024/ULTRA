# Installation

Targets Python 3.11, Isaac Sim 5.1, Isaac Lab 2.3.x (our fork based on 2.3.2), PyTorch 2.7.0 + CUDA 12.8 and the
rl-games version pinned by the Isaac Lab fork. Tested on Ubuntu 24.04 with an RTX 5070 Ti.

## 0. Requirements

| Component | Version |
| --- | --- |
| OS | Ubuntu 22.04 / 24.04 (glibc ≥ 2.35) |
| NVIDIA driver | ≥ 580.65 (`nvidia-smi`) |
| Python | 3.11 (required by Isaac Sim 5.x) |
| PyTorch | 2.7.0 + cu128 |

RTX 50xx GPUs (Blackwell, compute capability 12.0) need a PyTorch build with CUDA ≥ 12.8. Older builds
(cu118/cu121) fail with `no kernel image is available for execution on the device`.

## 1. Conda environment

```bash
conda create -n ultra python=3.11 -y
conda activate ultra
pip install --upgrade pip
```

## 2. Isaac Sim 5.1 and PyTorch

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

The first `isaacsim` launch asks you to accept the NVIDIA EULA. You can accept it non-interactively with
`export OMNI_KIT_ACCEPT_EULA=YES`.

## 3. Isaac Lab (fork) and rl-games

ULTRA uses a fork of Isaac Lab based on 2.3.2 that adds support for custom RL frameworks. rl-games is **not**
pinned in `requirements.txt`, so install it through the fork to get the version the fork expects:

```bash
git clone <isaaclab-fork-url> IsaacLab
cd IsaacLab
./isaaclab.sh --install rl_games
cd -
```

Do not install `rl-games` from PyPI afterwards; that would replace the fork's version.

## 4. ULTRA dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

`requirements.txt` keeps `numpy<2` because Isaac Sim 5.1 needs numpy 1.26.x. If pip upgrades torch or numpy,
repeat the torch line from step 2.

## 5. Sanity check

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability()); print(torch.ones(1, device='cuda'))"
python -c "import isaacsim, isaaclab, rl_games; print('ok')"
# headless Isaac Lab smoke test (run inside the IsaacLab checkout)
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

For an RTX 5070 Ti, the first line should print `2.7.0+cu128 (12, 0)` and then a CUDA tensor.

## 6. MuJoCo (sim2sim only)

`mujoco` is installed from `requirements.txt` and is enough for `ultra/sim2sim_*.py`. The merged robot+object
scene is generated at runtime into `ultra/data/assets/merge/` (git-ignored).

## 7. Data and checkpoints

See the *Data* and *Checkpoints* sections of the top-level README. Datasets go under `InterAct/`, checkpoints under
`checkpoints/` (both git-ignored). All commands are run from the repository root.
