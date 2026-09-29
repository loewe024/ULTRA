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
git clone git@github.com:ethz-mrl/IsaacLab-MRL.git IsaacLab
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

## 5. Convert the assets

Isaac Lab loads USD. Convert the G1 URDF and the object meshes once (writes `ultra/data/assets/usd/`, git-ignored):

```bash
python scripts/convert_assets.py
```

All 39 G1 links are kept as bodies (the datasets and policies use them), and the leg self-collision filters of the
IsaacGym version are written into the USD. Re-run with `--force` after changing the URDF or the object meshes.

## 6. Sanity check

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability()); print(torch.ones(1, device='cuda'))"
python -c "import isaacsim, isaaclab, rl_games; print('ok')"
# ULTRA's Isaac Lab environment against MuJoCo forward kinematics (body/joint order, quaternions, env origins)
python scripts/check_layout.py
```

For an RTX 5070 Ti, the first line should print `2.7.0+cu128 (12, 0)` and then a CUDA tensor, and
`check_layout.py` should end with `PASSED`.

## 7. GPU memory

The task configs use 4096 environments. With the 4052-dimensional teacher observation this needs more than 16 GB;
on a 16 GB GPU (e.g. RTX 5070 Ti) pass `--num_envs 2048` (about 15 GB for teacher training). Keep
`horizon_length * num_envs` divisible by `minibatch_size`.

## 8. MuJoCo (sim2sim only)

`mujoco` is installed from `requirements.txt` and is enough for `ultra/sim2sim_*.py`. The merged robot+object
scene is generated at runtime into `ultra/data/assets/merge/` (git-ignored).

## 9. Data and checkpoints

See the *Data* and *Checkpoints* sections of the top-level README. Datasets go under `InterAct/`, checkpoints under
`checkpoints/` (both git-ignored). All commands are run from the repository root.

Checkpoints of the IsaacGym version (rl-games 1.1.4, e.g. `ultra/weights/teacher_ultra_inference.pth`) load
directly; `scripts/convert_rlg_checkpoint.py` rewrites them in the rl-games 1.6 layout if another tool needs that.

## Differences to the IsaacGym version

The policies, datasets and the MuJoCo sim2sim code keep the IsaacGym conventions: (x, y, z, w) quaternions and the
IsaacGym body/joint order (`ultra/isaac/legacy_layout.py`). The simulation itself is PhysX 5, so results differ
somewhat from IsaacGym; the retargeting and tracking policies may need further training. Specifically:

- Object rolling and torsional friction and shape compliance have no PhysX 5 counterpart and are not simulated.
- Convex decomposition (VHACD in IsaacGym) uses PhysX 5's decomposition with the same hull budgets; with domain
  randomization the objects use 10 hulls instead of a random 1–10.
- The IsaacGym GPU buffer settings (`max_gpu_contact_pairs`, `default_buffer_size_multiplier`) are ignored; Isaac
  Lab's buffer sizes are used.
- Contact forces come from Isaac Lab contact sensors; the measured joint torques of position-controlled
  retargeting are the implicit PD torques (`applied_torque`).
