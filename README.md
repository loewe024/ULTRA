<p align="center">
  <h1 align="center"><strong>ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation</strong></h1>
  <p align="center">
    <a href="https://xialin-he.github.io/" target="_blank">Xialin He</a><sup>*</sup>&emsp;
    <a href="https://sirui-xu.github.io/" target="_blank">Sirui Xu</a><sup>*</sup>&emsp;
    <a href="https://lixinyao11.github.io/" target="_blank">Xinyao Li</a>&emsp;
    <a href="https://runpeidong.web.illinois.edu/" target="_blank">Runpei Dong</a>&emsp;
    <a href="https://scholar.google.com/citations?user=J2z-0lgAAAAJ&hl=en" target="_blank">Liuyu Bian</a>&emsp;
    <a href="https://yxw.cs.illinois.edu/" target="_blank">Yu-Xiong Wang</a><sup>&dagger;</sup>&emsp;
    <a href="https://lgui.cs.illinois.edu/" target="_blank">Liang-Yan Gui</a><sup>&dagger;</sup>
    <br>
    University of Illinois Urbana-Champaign
    <br>
    <sup>*</sup>Equal contribution&emsp;<sup>&dagger;</sup>Equal advising
    <br>
    <strong>IROS 2026 &middot; Best Paper Award on Mobile Manipulation &middot; Finalist 🏆</strong>
  </p>
</p>

<p align="center">
  <a href='https://arxiv.org/abs/2603.03279'>
    <img src='https://img.shields.io/badge/Arxiv-2603.03279-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
  <a href='https://arxiv.org/pdf/2603.03279'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a>
  <a href='https://ultra-humanoid.github.io/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
  <a href='https://github.com/MuLinjiu/ULTRA-locomanip'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
</p>

## 🏠 Overview

<div align="center">
  <img src="assets/ultra_demo.gif" width="100%" alt="ULTRA demo"/>
</div>

> **ULTRA** is a single multimodal controller for humanoid whole-body loco-manipulation: it tracks a motion reference when one is available, and acts from egocentric perception and a sparse goal when it is not — one policy, one set of weights, on a real Unitree G1.

This repository provides the training code: a privileged dense-tracking teacher, distilled into a single multimodal student with a compact latent space, plus IsaacGym playback and MuJoCo sim2sim.

## 📝 TODO

- [ ] Release code on retargeting and augmentation
- [ ] Release code for finetuning

The complete retargeted and augmented dataset is already available in the [Data](#-data) section. The first TODO is for the code used to produce it.

## ⚙️ Installation

See [docs/install.md](docs/install.md): Python 3.8, `pip install -r requirements.txt`, IsaacGym Preview 4, and
`mujoco` for sim2sim. Run every command below from the repository root.

## 📦 Data

Download the retargeted G1 rollouts from [Google Drive](https://drive.google.com/file/d/1g6OzxXGJczZ4klVqKCpTzy-orKkllCr4/view?usp=sharing), unzip, and place them under `InterAct/OMOMO_retarget_aug/` (git-ignored), one `.pt` per clip. The G1
and object assets are already in `ultra/data/assets/`.

## 🚀 Training

```bash
# Teacher — dense full-body tracking (PPO, 4096 envs). Output: output/teacher/g1_teacher/nn/
scripts/train_teacher.sh

# Student — multimodal distillation from the teacher. First set env.teacherPolicy in
# ultra/data/cfg/g1_student_vae.yaml to the teacher checkpoint (default: checkpoints/g1_teacher.pth).
scripts/train_student.sh
NUM_GPUS=4 scripts/train_student_multigpu.sh          # torchrun, single node
```

Every script forwards extra arguments to Python, e.g. `--num_envs N`, `--motion_file <dir>`, `--headless`,
`--output_path <dir>`. With few environments also pass a smaller `--minibatch_size` (rl_games requires
`minibatch_size <= num_envs * horizon_length`). Weights & Biases is optional: `WANDB_DISABLED=true` turns it off.

## 🎮 Inference

```bash
scripts/play_student.sh <student.pth> 1 full_track      # IsaacGym playback (also: sparse_track, object_obs points)
scripts/sim2sim_student.sh <student.pth> InterAct/OMOMO_retarget_aug full_track   # MuJoCo
scripts/export_jit.sh <student.pth> student_jit.pt      # TorchScript export
```

## 📄 License

ULTRA's own code is released under Apache-2.0 (top-level `LICENSE`); the InterMimic-derived simulation and learning
stack is under MIT (`LICENSE-InterMimic`). `ultra/env/tasks/base_task.py` and `vec_task.py` are adapted from
NVIDIA's IsaacGym examples and keep NVIDIA's header. Robot assets are from Unitree; motion data derives from OMOMO
through InterAct.

## 📖 Citation

```bibtex
@inproceedings{he2026ultra,
  title     = {ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation},
  author    = {He, Xialin and Xu, Sirui and Li, Xinyao and Dong, Runpei and Bian, Liuyu and Wang, Yu-Xiong and Gui, Liang-Yan},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```
