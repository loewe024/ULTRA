# Installation

Tested with Python 3.8, CUDA 11.x, IsaacGym Preview 4 and `rl-games==1.1.4`.

## 1. Conda environment

```bash
conda create -n ultra python=3.8 -y
conda activate ultra
# pick the torch build matching your CUDA driver, e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

`environment.yml` is a full linux-64 export of a working environment and can be used instead of the two `pip` lines
(`conda env create -f environment.yml`), but it contains many unrelated pins and does not include `wandb`
(either `pip install wandb` or run with `WANDB_DISABLED=true`). The pipeline was verified with torch 2.4.1+cu121 and numpy 1.23.5; the two `pip` lines above are the recommended path.

## 2. IsaacGym

Download [IsaacGym Preview 4](https://developer.nvidia.com/isaac-gym) and install it into the same environment:

```bash
cd isaacgym/python && pip install -e .
# sanity check (needs a display)
python examples/joint_monkey.py
```

IsaacGym Preview 4 uses the `np.float` alias that NumPy removed in 1.24, so keep `numpy<1.24` (pinned in
`requirements.txt`; `pip install -r requirements.txt` after installing IsaacGym if pip upgraded NumPy).

If you hit `ImportError: libpython3.8...` when importing `isaacgym`, add the conda library path:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## 3. MuJoCo (sim2sim only)

`pip install mujoco` is enough for `ultra/sim2sim_*.py`. The merged robot+object scene is generated at runtime
into `ultra/data/assets/merge/` (git-ignored).

## 4. Data and checkpoints

See the *Data* and *Checkpoints* sections of the top-level README. Datasets go under `InterAct/`, checkpoints under
`checkpoints/` (both git-ignored). All commands are run from the repository root.
