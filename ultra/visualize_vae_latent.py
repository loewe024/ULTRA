"""Visualize VAE latent space to check for multimodality.

Resets MuJoCo to reference-motion states, builds student observations,
and passes them through the prior network to collect mu / logvar.
No physics simulation or PD control needed.

Generates:
  1. PCA  – 2-D projection of prior mu coloured by object / motion / timestep
  2. t-SNE – same, with t-SNE
  3. Per-dimension histograms & KDE
  4. Temporal trajectories of latent mu for individual motions
  5. Variance analysis – which latent dims are most active
  6. Multimodality score per dimension (bimodality coefficient)
  7. mu vs sampled-z comparison
  8. Latent-dim correlation matrix

Usage (from the repository root):
    python ultra/visualize_vae_latent.py \
        --ckpt  checkpoints/YOUR_CHECKPOINT.pth \
        --motion_dir InterAct/OMOMO_retarget_aug \
        --output_dir latent_vis \
        --num_motions 10
"""

import argparse
import csv
import glob as glob_module
import os
import random
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------- project imports (run from the repository root) ----------
import mujoco
import trimesh
from learning import ultra_network_builder_obj_v2, ultra_models
from utils.obs_vae import MujocoObs, compute_sdf
from utils import torch_utils_mujoco

from sim2sim_vae import (
    _load_network_config_from_yaml,
    merge_mjcf,
    _set_freejoint_state,
    _write_hinge_qpos_qvel_by_names,
    JOINT_NAMES_29,
    to_torch,
)

# ================================================================== #
#  Action-label CSV loading
# ================================================================== #

def load_action_labels(csv_path):
    """Load omomo.csv → dict mapping motion name (no .pt) to action label."""
    label_map = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Input.video_url"].strip()
            action = row["Answer.action"].strip()
            label_map[name] = action
    return label_map


# ================================================================== #
#  Text description loading & encoding
# ================================================================== #

def load_text_descriptions(text_dir, mask_object=False):
    """Load .txt files from text_dir → dict: base_key → description string.

    File format: description#pos_tagged#float#float  (we take field 0).
    If *mask_object* is True, replace the object name (from filename) with
    "object" so the embedding captures action semantics only.
    """
    desc_map = {}
    for fname in sorted(os.listdir(text_dir)):
        if not fname.endswith(".txt"):
            continue
        key = os.path.splitext(fname)[0]
        with open(os.path.join(text_dir, fname)) as f:
            line = f.read().strip()
        desc = line.split("#")[0].strip()
        if desc:
            if mask_object:
                # key = sub10_clothesstand_000 → object name = clothesstand
                obj_name = key.split("_")[1]
                desc = desc.replace(obj_name, "object")
            desc_map[key] = desc
    return desc_map


def build_text_info(desc_map, model_name="all-MiniLM-L6-v2", n_clusters=None):
    """Encode descriptions with sentence-transformers and k-means cluster them.

    Returns dict with:
      keys, sentences, embeddings, cluster_labels, key_to_cluster,
      cluster_names (id→short representative description), n_clusters.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_distances

    keys = sorted(desc_map.keys())
    sentences = [desc_map[k] for k in keys]

    print(f"  Encoding {len(sentences)} descriptions with {model_name} ...")
    st_model = SentenceTransformer(model_name)
    embeddings = np.array(
        st_model.encode(sentences, show_progress_bar=True, batch_size=64)
    )

    n = len(embeddings)
    if n_clusters is None:
        best_k, best_score = 2, -1
        for k in range(2, min(16, n)):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labs = km.fit_predict(embeddings)
            sc = silhouette_score(embeddings, labs)
            if sc > best_score:
                best_k, best_score = k, sc
        n_clusters = best_k
        print(f"  Auto k={n_clusters} (silhouette={best_score:.3f})")

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    key_to_cluster = {keys[i]: int(labels[i]) for i in range(n)}

    # representative short description per cluster (closest to centroid)
    cluster_names = {}
    for ci in range(n_clusters):
        idx = np.where(labels == ci)[0]
        dists = cosine_distances(
            embeddings[idx], km.cluster_centers_[ci : ci + 1]
        ).ravel()
        closest = idx[np.argmin(dists)]
        desc = sentences[closest]
        if len(desc) > 50:
            desc = desc[:47] + "..."
        cluster_names[ci] = f"C{ci}: {desc}"

    return {
        "keys": keys,
        "sentences": sentences,
        "embeddings": embeddings,
        "cluster_labels": labels,
        "key_to_cluster": key_to_cluster,
        "cluster_names": cluster_names,
        "n_clusters": n_clusters,
    }


# ================================================================== #
#  Model loading
# ================================================================== #

def load_model(ckpt_path, device="cuda", goal_phase_dim=4):
    obs_dim = 1494 + 2 * max(0, goal_phase_dim - 3)
    config = {
        "actions_num": 29,
        "input_shape": (obs_dim,),
        "num_seqs": 1,
        "value_size": 1,
    }
    network = ultra_network_builder_obj_v2.UltraBuilder()
    network_params = _load_network_config_from_yaml()
    network.load(network_params)
    network = ultra_models.ModelUltraContinuous(network)
    ck = ultra_models.load_checkpoint(ckpt_path)
    policy = network.build(config)
    policy.to(device)
    state = {k: v for k, v in ck["model"].items() if not k.startswith(("running_mean_std.", "value_mean_std."))}
    policy.load_state_dict(state, strict=False)
    policy.eval()
    return policy


# ================================================================== #
#  Minimal MuJoCo env – reference tracking only, no physics step
# ================================================================== #

class RefEnv:
    """Set MuJoCo to reference states and build observations. No PD / sim."""

    def __init__(self, motion_path, goal_phase_dim=4):
        object_name = os.path.basename(motion_path).split("_")[1]
        object_path = f"ultra/data/assets/objects/{object_name}.xml"
        human_path  = "ultra/data/assets/g1/g1_29dof.xml"

        self._load_object(object_name, object_path)
        self.object_name = object_name
        self._load_motion(motion_path)

        os.makedirs("ultra/data/assets/merge/", exist_ok=True)
        model_xml = f"ultra/data/assets/merge/{object_name}.xml"
        merge_mjcf(human_path, object_path, model_xml, viz_sites=0)

        self.mj_model = mujoco.MjModel.from_xml_path(model_xml)
        self.mj_model.opt.timestep = 1.0 / (60.0 * 17.0)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        mujoco.mj_step(self.mj_model, self.mj_data)

        self.commands = np.zeros(goal_phase_dim, dtype=np.float32)
        self.obs_builder = MujocoObs(
            self.mj_model,
            object_name,
            self.max_episode_length,
            self.hoi_data,
            self.object_points,
            self.object_corners,
            history_step=10,
            goal_phase_dim=goal_phase_dim,
        )
        self.obs_builder.set_command_goal(self.commands)

    # -- object mesh --
    def _load_object(self, object_name, object_path):
        obj_file = (
            f"ultra/data/assets/objects/objects/{object_name}/{object_name}.obj"
        )
        mesh_obj = trimesh.load(obj_file, process=False, force="mesh")
        corners = mesh_obj.bounding_box_oriented.vertices
        center = np.mean(mesh_obj.vertices, 0)
        pts, _ = trimesh.sample.sample_surface_even(mesh_obj, count=256, seed=2024)
        pts = to_torch(pts - center)
        corners = to_torch(corners - center)
        while pts.shape[0] < 256:
            pts = torch.cat([pts, pts[: 256 - pts.shape[0]]], dim=0)
        self.object_points = to_torch(pts)
        self.object_corners = to_torch(corners)

    # -- reference motion --
    def _load_motion(self, data_path):
        hoi_data = torch.load(data_path, weights_only=False, map_location="cpu")
        hoi_expand = torch.zeros(
            (hoi_data.shape[0] * 2, hoi_data.shape[1]), device=hoi_data.device
        )
        hoi_expand[: hoi_data.shape[0]] = hoi_data
        hoi_expand[hoi_data.shape[0] :] = hoi_data[-1]
        hoi_data = hoi_expand
        self.max_episode_length = hoi_data.shape[0]

        root_pos = hoi_data[:, 0:3].clone()
        root_rot = hoi_data[:, 3:7]
        obj_pos  = hoi_data[:, 71:74]
        obj_rot  = hoi_data[:, 74:78]
        body_pos = hoi_data[:, 84:201]

        obj_rot_ext = obj_rot.unsqueeze(1).repeat(1, self.object_points.shape[0], 1).view(-1, 4)
        pts_ext = self.object_points.unsqueeze(0).repeat(obj_rot.shape[0], 1, 1).view(-1, 3)
        obj_pts = (
            torch_utils_mujoco.quat_rotate(obj_rot_ext, pts_ext)
            .view(obj_rot.shape[0], self.object_points.shape[0], 3)
            + obj_pos.unsqueeze(1)
        )
        ref_ig = compute_sdf(
            body_pos.clone().view(obj_rot.shape[0], -1, 3), obj_pts
        ).view(-1, 3)
        heading = torch_utils_mujoco.calc_heading_quat_inv(root_rot)
        heading_ext = heading.unsqueeze(1).repeat(
            1, body_pos.shape[1] // 3, 1
        ).view(-1, 4)
        ref_ig = torch_utils_mujoco.quat_rotate(heading_ext, ref_ig).view(
            obj_rot.shape[0], -1
        )
        self.hoi_data = torch.cat([hoi_data.detach().cpu(), ref_ig], dim=-1)
        self.hoi_ref = torch.cat(
            (
                root_pos,
                hoi_data[:, 3:7],   # root_rot
                hoi_data[:, 7:10],  # root_pos_vel
                hoi_data[:, 10:13], # root_rot_vel
                hoi_data[:, 13:42], # dof_pos
                hoi_data[:, 42:71], # dof_vel
                hoi_data[:, 71:74], # obj_pos
                hoi_data[:, 74:78], # obj_rot
                hoi_data[:, 78:81], # obj_pos_vel
                hoi_data[:, 81:84], # obj_rot_vel
            ),
            dim=-1,
        )

    # -- set MuJoCo to reference frame --
    def reset_to_ref(self, t=0):
        key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, key_id)
        else:
            mujoco.mj_resetData(self.mj_model, self.mj_data)

        ref = self.hoi_ref[t].to("cpu", torch.float32)
        _set_freejoint_state(
            self.mj_model, self.mj_data, "pelvis",
            ref[0:3], ref[3:7],
            torch.zeros(3), torch.zeros(3),
        )
        _write_hinge_qpos_qvel_by_names(
            self.mj_model, self.mj_data, JOINT_NAMES_29,
            ref[13:42], torch.zeros(29),
        )
        _set_freejoint_state(
            self.mj_model, self.mj_data, f"{self.object_name}_free",
            ref[71:74], ref[74:78],
            torch.zeros(3), torch.zeros(3),
        )
        mujoco.mj_forward(self.mj_model, self.mj_data)

    # -- build one observation --
    def get_obs(self, t):
        """Reset to ref-frame *t*, build and return student obs tensor."""
        self.reset_to_ref(t)
        dummy_action = torch.zeros((1, 29))
        dummy_torque = torch.zeros((1, 29))
        obs, _ = self.obs_builder._compute_observations(
            self.mj_data, t, dummy_action, dummy_torque,
            student_obs=True, episode_length=t, return_dict=True,
        )
        return obs  # (1, obs_dim)


# ================================================================== #
#  Data collection
# ================================================================== #

def collect_latent_data(
    policy, motion_paths, device="cuda", goal_phase_dim=4, max_steps=300,
):
    vae_net = policy.a2c_network
    vae_dim = vae_net.vae_dim

    mu_list, logvar_list, z_list = [], [], []
    mid_list, obj_list, ts_list, fname_list = [], [], [], []

    for mi, mpath in enumerate(motion_paths):
        fname = os.path.basename(mpath)
        obj_name = fname.split("_")[1]
        print(f"[{mi + 1}/{len(motion_paths)}] {fname}  (object={obj_name})")

        env = RefEnv(mpath, goal_phase_dim=goal_phase_dim)
        n_steps = min(max_steps, env.max_episode_length - 1)

        for t in range(n_steps):
            obs = env.get_obs(t).to(device)

            with torch.no_grad():
                prior = vae_net._prior({"obs": obs})
                mu = prior["mu"].cpu().numpy().squeeze(0)
                logvar = prior["logvar"].cpu().numpy().squeeze(0)

            std = np.exp(0.5 * logvar)
            z = mu + std * np.random.randn(*mu.shape)

            mu_list.append(mu)
            logvar_list.append(logvar)
            z_list.append(z)
            mid_list.append(mi)
            obj_list.append(obj_name)
            ts_list.append(t)
            fname_list.append(fname)

    return {
        "mu": np.array(mu_list),
        "logvar": np.array(logvar_list),
        "z_samples": np.array(z_list),
        "motion_ids": np.array(mid_list),
        "object_names": obj_list,
        "timesteps": np.array(ts_list),
        "motion_filenames": fname_list,
    }


# ================================================================== #
#  Mask-conditioned analysis
# ================================================================== #

# Mask layout (242 dims total):
#   goal_mask(3) | local_goal_mask(31) | obj_trans_mask(3) |
#   obj_rot_mask(6) | obj_pos_mask(3) | obj_points_mask(192) | command_mask(4)

MASK_DIM = 242
MASK_SLICES = {
    "goal":       (0,   3),
    "local_goal": (3,   34),
    "obj_trans":  (34,  37),
    "obj_rot":    (37,  43),
    "obj_pos":    (43,  46),
    "obj_points": (46,  238),
    "command":    (238, 242),
}


def _make_mask(parts_on, total=MASK_DIM):
    """Create a (1, MASK_DIM) mask tensor. *parts_on* lists which parts are 1."""
    m = torch.zeros(1, total)
    for name in parts_on:
        s, e = MASK_SLICES[name]
        m[:, s:e] = 1.0
    return m


def define_mask_modes():
    """Return dict of mode_name -> (1, 242) mask tensor.

    Mirrors the task-mode presets in MujocoObs.configure_task_mode():
      full_track   – human tracking only (goal + local_goal, no object, no command)
      sparse_track – global goal + obj_trans + command (no local_goal, no obj detail)
      object_obs / points – obj_trans + obj_points + command
      object_obs / pos    – obj_trans + obj_pos + command
      object_obs / none   – obj_trans + command only
      all_visible  – everything on (baseline)
    """
    modes = {
        # ---------- actual sim2sim task modes ----------
        "full_track":       _make_mask(["goal", "local_goal"]),
        "obj_obs_points":   _make_mask(["obj_trans", "obj_points", "command"]),
        "obj_obs_pos":      _make_mask(["obj_trans", "obj_pos", "command"]),
        "obj_obs_none":     _make_mask(["obj_trans", "command"]),
        # ---------- reference baseline ----------
        "all_visible":      _make_mask(list(MASK_SLICES.keys())),
    }
    return modes


# Paper-friendly display names for each mask mode
MODE_DISPLAY_NAMES = {
    "full_track":     "Motion Tracking",
    "sparse_track":   "Root Tracking",
    "obj_obs_points": "Object (Points)",
    "obj_obs_pos":    "Object (Position)",
    "obj_obs_none":   "Object (None)",
    "all_visible":    "Full Observation",
}


def _display(mode):
    """Return paper-friendly display name for a mask mode."""
    return MODE_DISPLAY_NAMES.get(mode, mode)


def collect_mask_conditioned(
    policy, motion_paths, device="cuda", goal_phase_dim=4,
    max_steps=300, step_stride=5,
):
    """For each obs, override the mask portion and collect prior mu under each mode."""
    vae_net = policy.a2c_network
    modes = define_mask_modes()
    obs_dim = None

    mu_all, mode_all, mid_all, ts_all, obj_all = [], [], [], [], []

    for mi, mpath in enumerate(motion_paths):
        fname = os.path.basename(mpath)
        obj_name = fname.split("_")[1]
        print(f"[mask-cond {mi+1}/{len(motion_paths)}] {fname}")

        env = RefEnv(mpath, goal_phase_dim=goal_phase_dim)
        n_steps = min(max_steps, env.max_episode_length - 1)

        for t in range(0, n_steps, step_stride):
            base_obs = env.get_obs(t)  # (1, obs_dim) on cpu
            if obs_dim is None:
                obs_dim = base_obs.shape[-1]

            for mode_name, mask_vec in modes.items():
                obs = base_obs.clone()
                obs[:, -MASK_DIM:] = mask_vec
                obs = obs.to(device)

                with torch.no_grad():
                    prior = vae_net._prior({"obs": obs})
                    mu = prior["mu"].cpu().numpy().squeeze(0)

                mu_all.append(mu)
                mode_all.append(mode_name)
                mid_all.append(mi)
                ts_all.append(t)
                obj_all.append(obj_name)

    return {
        "mu": np.array(mu_all),
        "mode_names": mode_all,
        "motion_ids": np.array(mid_all),
        "timesteps": np.array(ts_all),
        "object_names": obj_all,
        "mask_modes": list(modes.keys()),
    }


# -- mask-conditioned plots --

def plot_mask_pca(mdata, out):
    from sklearn.decomposition import PCA
    mu = mdata["mu"]
    modes = mdata["mode_names"]
    pca = PCA(n_components=2)
    proj = pca.fit_transform(mu)
    unique_modes = mdata["mask_modes"]

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = cm.get_cmap("Set1", max(len(unique_modes), 1))
    for i, mode in enumerate(unique_modes):
        idx = [j for j, m in enumerate(modes) if m == mode]
        ax.scatter(proj[idx, 0], proj[idx, 1], s=6, alpha=0.4,
                   color=cmap(i), label=mode)
    ax.set_title("PCA of prior mu (by mask mode)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(markerscale=4, fontsize=8)
    plt.tight_layout()
    p = os.path.join(out, "mask_pca.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_mask_tsne(mdata, out, perplexity=30, max_points=5000):
    from sklearn.manifold import TSNE
    mu = mdata["mu"]
    modes = mdata["mode_names"]
    n = mu.shape[0]
    idx = np.random.choice(n, min(n, max_points), replace=False) if n > max_points else np.arange(n)
    mu_sub = mu[idx]
    modes_sub = [modes[i] for i in idx]

    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(mu_sub) - 1),
                random_state=42, init="pca", learning_rate="auto")
    proj = tsne.fit_transform(mu_sub)

    unique_modes = mdata["mask_modes"]
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = cm.get_cmap("Set1", max(len(unique_modes), 1))
    for i, mode in enumerate(unique_modes):
        mask = [j for j, m in enumerate(modes_sub) if m == mode]
        if mask:
            ax.scatter(proj[mask, 0], proj[mask, 1], s=8, alpha=0.4,
                       color=cmap(i), label=mode)
    ax.set_title("t-SNE of prior mu (by mask mode)")
    ax.legend(markerscale=4, fontsize=8)
    plt.tight_layout()
    p = os.path.join(out, "mask_tsne.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_mask_per_mode_pca(mdata, out):
    """Separate PCA subplot for each mask mode, coloured by motion."""
    from sklearn.decomposition import PCA
    mu = mdata["mu"]
    modes = mdata["mode_names"]
    mids = mdata["motion_ids"]
    unique_modes = mdata["mask_modes"]
    n_modes = len(unique_modes)
    ncols = min(4, n_modes)
    nrows = (n_modes + ncols - 1) // ncols

    pca = PCA(n_components=2)
    proj = pca.fit_transform(mu)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n_modes == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    n_m = int(mids.max()) + 1
    for ai, mode in enumerate(unique_modes):
        ax = axes[ai]
        idx = np.array([j for j, m in enumerate(modes) if m == mode])
        sc = ax.scatter(proj[idx, 0], proj[idx, 1], s=6, alpha=0.5,
                        c=mids[idx], cmap=cm.get_cmap("tab20", max(n_m, 1)),
                        vmin=0, vmax=max(n_m - 1, 1))
        ax.set_title(mode, fontsize=10)
    for j in range(n_modes, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("PCA per mask mode (coloured by motion)", fontsize=12)
    plt.tight_layout()
    p = os.path.join(out, "mask_per_mode_pca.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_mask_mu_shift(mdata, out):
    """Bar chart: mean L2 distance of mu from 'all_visible' baseline per mode."""
    mu = mdata["mu"]
    modes = mdata["mode_names"]
    mids = mdata["motion_ids"]
    ts = mdata["timesteps"]
    unique_modes = mdata["mask_modes"]

    baseline_idx = np.array([i for i, m in enumerate(modes) if m == "all_visible"])
    baseline_mu = {(mids[i], ts[i]): mu[i] for i in baseline_idx}

    shifts = {m: [] for m in unique_modes if m != "all_visible"}
    for i, m in enumerate(modes):
        if m == "all_visible":
            continue
        key = (mids[i], ts[i])
        if key in baseline_mu:
            shifts[m].append(np.linalg.norm(mu[i] - baseline_mu[key]))

    names = list(shifts.keys())
    means = [np.mean(shifts[n]) for n in names]
    stds = [np.std(shifts[n]) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, color="steelblue", alpha=0.7, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("L2 distance from all_visible mu")
    ax.set_title("Prior mu shift per mask mode (vs all_visible baseline)")
    plt.tight_layout()
    p = os.path.join(out, "mask_mu_shift.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_mask_dim_comparison(mdata, out, top_k=8):
    """For top-variance dims, overlay histograms of mu under each mask mode."""
    mu = mdata["mu"]
    modes = mdata["mode_names"]
    unique_modes = mdata["mask_modes"]
    var = mu.var(axis=0)
    top = np.argsort(var)[::-1][:top_k]

    nrows = 2; ncols = (top_k + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()
    cmap = cm.get_cmap("Set1", max(len(unique_modes), 1))

    for pi, dim in enumerate(top):
        ax = axes[pi]
        for mi, mode in enumerate(unique_modes):
            idx = [j for j, m in enumerate(modes) if m == mode]
            vals = mu[idx, dim]
            ax.hist(vals, bins=40, density=True, alpha=0.3, color=cmap(mi), label=mode)
        ax.set_title(f"dim {dim} (var={var[dim]:.2f})", fontsize=9)
        if pi == 0:
            ax.legend(fontsize=5, ncol=2)
    for j in range(pi + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Prior mu histograms per mask mode (top-variance dims)", fontsize=11)
    plt.tight_layout()
    p = os.path.join(out, "mask_dim_comparison.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


# -- controlled-variable visualisations --

def collect_controlled(
    policy, motion_paths, device="cuda", goal_phase_dim=4,
    max_steps=300, sample_timesteps=None,
):
    """For a few fixed (motion, timestep) pairs, collect mu AND action under each mask mode."""
    vae_net = policy.a2c_network
    modes = define_mask_modes()

    if sample_timesteps is None:
        sample_timesteps = [0, 30, 60, 100, 150, 200]

    records = []  # list of dicts

    for mi, mpath in enumerate(motion_paths):
        fname = os.path.basename(mpath)
        obj_name = fname.split("_")[1]
        print(f"[controlled {mi+1}/{len(motion_paths)}] {fname}")

        env = RefEnv(mpath, goal_phase_dim=goal_phase_dim)
        n_frames = min(max_steps, env.max_episode_length - 1)

        for t in sample_timesteps:
            if t >= n_frames:
                continue
            base_obs = env.get_obs(t)  # (1, obs_dim) cpu

            for mode_name, mask_vec in modes.items():
                obs = base_obs.clone()
                obs[:, -MASK_DIM:] = mask_vec
                obs_gpu = obs.to(device)

                with torch.no_grad():
                    prior = vae_net._prior({"obs": obs_gpu})
                    mu = prior["mu"]
                    logvar = prior["logvar"]
                    z = mu  # since prior variance is tiny, z ≈ mu

                    # get action through trunk
                    trunk_in = {"obs": obs_gpu, "vae_latent": z}
                    action_mu, action_sigma = vae_net._trunk(trunk_in)

                records.append({
                    "motion_id": mi,
                    "timestep": t,
                    "fname": fname,
                    "obj": obj_name,
                    "mode": mode_name,
                    "mu": mu.cpu().numpy().squeeze(0),
                    "logvar": logvar.cpu().numpy().squeeze(0),
                    "action": action_mu.cpu().numpy().squeeze(0),
                })

    return records, list(modes.keys())


def plot_fixed_obs_heatmap(records, mode_list, out, n_examples=6):
    """Heatmap: rows=latent dim, columns=mask mode, for a few fixed obs."""
    import itertools
    obs_keys = list(dict.fromkeys(
        (r["motion_id"], r["timestep"]) for r in records
    ))[:n_examples]

    fig, axes = plt.subplots(len(obs_keys), 2, figsize=(18, 3.5 * len(obs_keys)),
                             gridspec_kw={"width_ratios": [2, 1]})
    if len(obs_keys) == 1:
        axes = axes[np.newaxis, :]

    for ri, (mid, t) in enumerate(obs_keys):
        sub = [r for r in records if r["motion_id"] == mid and r["timestep"] == t]
        sub_sorted = sorted(sub, key=lambda r: mode_list.index(r["mode"]))
        mu_mat = np.array([r["mu"] for r in sub_sorted])       # (n_modes, 64)
        act_mat = np.array([r["action"] for r in sub_sorted])  # (n_modes, 29)
        labels = [r["mode"] for r in sub_sorted]

        ax = axes[ri, 0]
        im = ax.imshow(mu_mat.T, aspect="auto", cmap="RdBu_r",
                        vmin=-np.abs(mu_mat).max(), vmax=np.abs(mu_mat).max())
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("latent dim")
        ax.set_title(f"mu | motion {mid}, t={t} ({sub_sorted[0]['fname'][:30]}..)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.02)

        ax2 = axes[ri, 1]
        im2 = ax2.imshow(act_mat.T, aspect="auto", cmap="RdBu_r",
                          vmin=-np.abs(act_mat).max(), vmax=np.abs(act_mat).max())
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("action dim (29 DoF)")
        ax2.set_title(f"action | motion {mid}, t={t}", fontsize=9)
        plt.colorbar(im2, ax=ax2, fraction=0.03)

    plt.suptitle("Fixed obs: latent mu & action across mask modes", fontsize=12)
    plt.tight_layout()
    p = os.path.join(out, "ctrl_fixed_obs_heatmap.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_action_distance(records, mode_list, out):
    """For each mask mode pair, show mean action L2 distance."""
    from itertools import combinations
    obs_keys = list(dict.fromkeys(
        (r["motion_id"], r["timestep"]) for r in records
    ))
    lookup = {}
    for r in records:
        lookup[(r["motion_id"], r["timestep"], r["mode"])] = r

    pair_names, pair_dists_mu, pair_dists_act = [], [], []
    for m1, m2 in combinations(mode_list, 2):
        dists_mu, dists_act = [], []
        for (mid, t) in obs_keys:
            r1 = lookup.get((mid, t, m1))
            r2 = lookup.get((mid, t, m2))
            if r1 is None or r2 is None:
                continue
            dists_mu.append(np.linalg.norm(r1["mu"] - r2["mu"]))
            dists_act.append(np.linalg.norm(r1["action"] - r2["action"]))
        if dists_mu:
            pair_names.append(f"{m1}\nvs\n{m2}")
            pair_dists_mu.append(np.mean(dists_mu))
            pair_dists_act.append(np.mean(dists_act))

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(pair_names) * 1.2), 6))
    x = np.arange(len(pair_names))

    axes[0].bar(x, pair_dists_mu, color="steelblue", alpha=0.7)
    axes[0].set_xticks(x); axes[0].set_xticklabels(pair_names, fontsize=6)
    axes[0].set_ylabel("Mean L2 distance"); axes[0].set_title("Latent mu distance (pairwise)")

    axes[1].bar(x, pair_dists_act, color="coral", alpha=0.7)
    axes[1].set_xticks(x); axes[1].set_xticklabels(pair_names, fontsize=6)
    axes[1].set_ylabel("Mean L2 distance"); axes[1].set_title("Action distance (pairwise)")

    plt.suptitle("Pairwise distance between mask modes (same obs)", fontsize=12)
    plt.tight_layout()
    p = os.path.join(out, "ctrl_action_distance.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_temporal_mask_heatmap(records, mode_list, out, n_motions=3, top_k=8):
    """For n motions, show heatmap: x=timestep, y=mask mode, color=mu value for top dims."""
    motion_ids = sorted(set(r["motion_id"] for r in records))[:n_motions]

    # get top-variance dims from all_visible baseline
    all_vis = [r for r in records if r["mode"] == "all_visible"]
    mu_all = np.array([r["mu"] for r in all_vis])
    var = mu_all.var(axis=0)
    top = np.argsort(var)[::-1][:top_k]

    for mid in motion_ids:
        sub = [r for r in records if r["motion_id"] == mid]
        timesteps = sorted(set(r["timestep"] for r in sub))
        fname = sub[0]["fname"]

        fig, axes = plt.subplots(top_k, 1, figsize=(max(10, len(timesteps) * 0.6), 2.0 * top_k))
        if top_k == 1:
            axes = [axes]

        for di, dim in enumerate(top):
            mat = np.zeros((len(mode_list), len(timesteps)))
            for ti, t in enumerate(timesteps):
                for mi, mode in enumerate(mode_list):
                    r = next((r for r in sub if r["timestep"] == t and r["mode"] == mode), None)
                    if r is not None:
                        mat[mi, ti] = r["mu"][dim]

            ax = axes[di]
            vmax = np.abs(mat).max()
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_yticks(range(len(mode_list)))
            ax.set_yticklabels(mode_list, fontsize=7)
            ax.set_xticks(range(len(timesteps)))
            ax.set_xticklabels(timesteps, fontsize=6)
            ax.set_title(f"dim {dim} (var={var[dim]:.2f})", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.015)

        plt.suptitle(f"mu(time, mask) — motion {mid}: {fname[:40]}..", fontsize=11)
        plt.tight_layout()
        p = os.path.join(out, f"ctrl_temporal_mask_m{mid}.png")
        plt.savefig(p, dpi=150); plt.close()
        print(f"  Saved {p}")


def plot_delta_structure(records, mode_list, out, ref_mode="all_visible"):
    """PCA of delta_mu vectors (mu(mode) - mu(ref)) to see if mask shifts are structured."""
    from sklearn.decomposition import PCA
    obs_keys = list(dict.fromkeys(
        (r["motion_id"], r["timestep"]) for r in records
    ))
    lookup = {}
    for r in records:
        lookup[(r["motion_id"], r["timestep"], r["mode"])] = r

    delta_mu, delta_act, delta_mode = [], [], []
    for (mid, t) in obs_keys:
        ref = lookup.get((mid, t, ref_mode))
        if ref is None:
            continue
        for mode in mode_list:
            if mode == ref_mode:
                continue
            r = lookup.get((mid, t, mode))
            if r is None:
                continue
            delta_mu.append(r["mu"] - ref["mu"])
            delta_act.append(r["action"] - ref["action"])
            delta_mode.append(mode)

    if not delta_mu:
        return

    delta_mu = np.array(delta_mu)
    delta_act = np.array(delta_act)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = cm.get_cmap("Set1", max(len(mode_list), 1))
    mode_colors = {m: cmap(i) for i, m in enumerate(mode_list) if m != ref_mode}

    for ax, data, title in [(axes[0], delta_mu, "delta mu"), (axes[1], delta_act, "delta action")]:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(data)
        for mode in [m for m in mode_list if m != ref_mode]:
            idx = [i for i, m in enumerate(delta_mode) if m == mode]
            if idx:
                ax.scatter(proj[idx, 0], proj[idx, 1], s=10, alpha=0.5,
                           color=mode_colors[mode], label=mode)
        ax.set_title(f"PCA of {title} (vs {ref_mode})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(markerscale=3, fontsize=8)
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.axvline(0, color="gray", ls=":", lw=0.5)

    plt.suptitle(f"Structured shift: delta from {ref_mode}", fontsize=12)
    plt.tight_layout()
    p = os.path.join(out, "ctrl_delta_structure.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_delta_multi_proj(records, mode_list, out, ref_mode="all_visible"):
    """Try multiple projection methods on delta_mu to find best cluster separation.

    Methods: PCA (1-2), PCA (3-4), t-SNE, UMAP, LDA (supervised by mask mode).
    """
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import LabelEncoder

    obs_keys = list(dict.fromkeys(
        (r["motion_id"], r["timestep"]) for r in records
    ))
    lookup = {}
    for r in records:
        lookup[(r["motion_id"], r["timestep"], r["mode"])] = r

    delta_mu, delta_mode = [], []
    for (mid, t) in obs_keys:
        ref = lookup.get((mid, t, ref_mode))
        if ref is None:
            continue
        for mode in mode_list:
            if mode == ref_mode:
                continue
            r = lookup.get((mid, t, mode))
            if r is None:
                continue
            delta_mu.append(r["mu"] - ref["mu"])
            delta_mode.append(mode)

    if not delta_mu:
        return

    delta_mu = np.array(delta_mu)
    active_modes = [m for m in mode_list if m != ref_mode]
    cmap = cm.get_cmap("Set1", max(len(mode_list), 1))
    mode_colors = {m: cmap(i) for i, m in enumerate(mode_list) if m != ref_mode}

    # -- build projections --
    projections = {}

    # PCA 1-2
    pca = PCA(n_components=min(4, delta_mu.shape[1]))
    pca_all = pca.fit_transform(delta_mu)
    projections["PCA (PC1-2)"] = (
        pca_all[:, :2],
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
    )
    # PCA 3-4
    if pca_all.shape[1] >= 4:
        projections["PCA (PC3-4)"] = (
            pca_all[:, 2:4],
            f"PC3 ({pca.explained_variance_ratio_[2]:.1%})",
            f"PC4 ({pca.explained_variance_ratio_[3]:.1%})",
        )

    # t-SNE
    n = delta_mu.shape[0]
    max_pts = 8000
    if n > max_pts:
        tsne_idx = np.random.choice(n, max_pts, replace=False)
    else:
        tsne_idx = np.arange(n)
    tsne = TSNE(n_components=2, perplexity=min(50, len(tsne_idx) - 1),
                random_state=42, init="pca", learning_rate="auto")
    tsne_proj_sub = tsne.fit_transform(delta_mu[tsne_idx])
    # map back
    tsne_proj = np.full((n, 2), np.nan)
    tsne_proj[tsne_idx] = tsne_proj_sub
    projections["t-SNE"] = (tsne_proj, "t-SNE 1", "t-SNE 2")

    # UMAP
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
        umap_proj = reducer.fit_transform(delta_mu)
        projections["UMAP"] = (umap_proj, "UMAP 1", "UMAP 2")
    except ImportError:
        print("  [skip UMAP — not installed]")

    # LDA (supervised)
    le = LabelEncoder()
    labels = le.fit_transform(delta_mode)
    n_classes = len(active_modes)
    lda_dim = min(n_classes - 1, 2)
    if lda_dim >= 2:
        lda = LDA(n_components=2)
        lda_proj = lda.fit_transform(delta_mu, labels)
        projections["LDA (supervised)"] = (
            lda_proj,
            f"LD1 ({lda.explained_variance_ratio_[0]:.1%})",
            f"LD2 ({lda.explained_variance_ratio_[1]:.1%})",
        )
    elif lda_dim == 1:
        lda = LDA(n_components=1)
        lda_1d = lda.fit_transform(delta_mu, labels).ravel()
        lda_proj = np.column_stack([lda_1d, np.zeros_like(lda_1d)])
        projections["LDA (supervised, 1D)"] = (
            lda_proj,
            f"LD1 ({lda.explained_variance_ratio_[0]:.1%})",
            "(no LD2)",
        )

    # -- plot --
    n_proj = len(projections)
    ncols = min(3, n_proj)
    nrows = (n_proj + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = np.atleast_2d(axes).reshape(-1) if n_proj > 1 else [axes]

    for ai, (name, (proj, xl, yl)) in enumerate(projections.items()):
        ax = axes[ai]
        for mode in active_modes:
            idx = np.array([i for i, m in enumerate(delta_mode) if m == mode])
            valid = ~np.isnan(proj[idx, 0])
            idx_v = idx[valid]
            if len(idx_v):
                ax.scatter(proj[idx_v, 0], proj[idx_v, 1], s=8, alpha=0.45,
                           color=mode_colors[mode], label=mode)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.legend(markerscale=3, fontsize=7)
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.axvline(0, color="gray", ls=":", lw=0.5)

    for j in range(n_proj, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Delta mu — multiple projections (vs {ref_mode})", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out, "ctrl_delta_multi_proj.png")
    plt.savefig(p, dpi=200); plt.close()
    print(f"  Saved {p}")

    # ---- Paper-quality individual plots (PCA, t-SNE, LDA) ----
    paper_keys = ["PCA (PC1-2)", "t-SNE", "LDA (supervised)"]
    paper_names = {"PCA (PC1-2)": "pca", "t-SNE": "tsne", "LDA (supervised)": "lda"}

    for key in paper_keys:
        if key not in projections:
            continue
        proj, xl, yl = projections[key]

        fig, ax = plt.subplots(figsize=(7, 6))
        for mode in active_modes:
            idx = np.array([i for i, m in enumerate(delta_mode) if m == mode])
            valid = ~np.isnan(proj[idx, 0])
            idx_v = idx[valid]
            if len(idx_v):
                ax.scatter(proj[idx_v, 0], proj[idx_v, 1], s=8, alpha=0.45,
                           color=mode_colors[mode], label=_display(mode))
        ax.set_xlabel(xl, fontsize=18)
        ax.set_ylabel(yl, fontsize=18)
        ax.tick_params(labelsize=15)
        ax.legend(markerscale=3, fontsize=15, loc="best")
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.axvline(0, color="gray", ls=":", lw=0.5)
        plt.tight_layout()
        p = os.path.join(out, f"paper_delta_{paper_names[key]}.pdf")
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved {p}")


def plot_delta_by_motion(records, mode_list, out, ref_mode="all_visible"):
    """Same delta PCA as plot_delta_structure, but coloured by motion (filename).

    Produces two figures:
      1. A combined plot with all mask modes, points coloured by motion.
      2. Per-mask-mode subplots, each coloured by motion.
    This reveals whether different motions cluster separately in the delta space.
    """
    from sklearn.decomposition import PCA

    obs_keys = list(dict.fromkeys(
        (r["motion_id"], r["timestep"]) for r in records
    ))
    lookup = {}
    for r in records:
        lookup[(r["motion_id"], r["timestep"], r["mode"])] = r

    # -- build per-record id mapping: motion_id -> short label --
    mid_to_fname = {}
    for r in records:
        if r["motion_id"] not in mid_to_fname:
            # truncate filename for legend readability
            fname = r["fname"].replace(".pt", "")
            if len(fname) > 35:
                fname = fname[:32] + "..."
            mid_to_fname[r["motion_id"]] = fname

    # -- compute deltas --
    delta_mu, delta_act = [], []
    delta_mode, delta_mid = [], []
    for (mid, t) in obs_keys:
        ref = lookup.get((mid, t, ref_mode))
        if ref is None:
            continue
        for mode in mode_list:
            if mode == ref_mode:
                continue
            r = lookup.get((mid, t, mode))
            if r is None:
                continue
            delta_mu.append(r["mu"] - ref["mu"])
            delta_act.append(r["action"] - ref["action"])
            delta_mode.append(mode)
            delta_mid.append(mid)

    if not delta_mu:
        return

    delta_mu = np.array(delta_mu)
    delta_act = np.array(delta_act)
    delta_mid = np.array(delta_mid)

    unique_mids = sorted(set(delta_mid))
    n_motions = len(unique_mids)

    # Use a large colormap for many motions
    if n_motions <= 20:
        cmap_name = "tab20"
    else:
        cmap_name = "nipy_spectral"
    motion_cmap = cm.get_cmap(cmap_name, max(n_motions, 1))

    # ---- Figure 1: combined plot (all modes), coloured by motion ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, data, title in [(axes[0], delta_mu, "delta mu"),
                             (axes[1], delta_act, "delta action")]:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(data)
        for ci, mid in enumerate(unique_mids):
            idx = np.where(delta_mid == mid)[0]
            label = f"m{mid}: {mid_to_fname.get(mid, str(mid))}"
            ax.scatter(proj[idx, 0], proj[idx, 1], s=10, alpha=0.45,
                       color=motion_cmap(ci), label=label)
        ax.set_title(f"PCA of {title} (vs {ref_mode})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        # Only show legend if manageable number of motions
        if n_motions <= 20:
            ax.legend(markerscale=3, fontsize=5, ncol=2, loc="best")
        else:
            # use colorbar instead
            sm = plt.cm.ScalarMappable(cmap=motion_cmap,
                                       norm=plt.Normalize(0, n_motions - 1))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="motion index")
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.axvline(0, color="gray", ls=":", lw=0.5)

    plt.suptitle(f"Delta structure coloured by motion (vs {ref_mode})", fontsize=12)
    plt.tight_layout()
    p = os.path.join(out, "ctrl_delta_by_motion.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")

    # ---- Figure 2: per-mask-mode subplots, coloured by motion ----
    active_modes = [m for m in mode_list if m != ref_mode]
    n_modes = len(active_modes)
    ncols = min(3, n_modes)
    nrows = (n_modes + ncols - 1) // ncols

    # only do delta_mu for per-mode subplots (the more informative one)
    pca_mu = PCA(n_components=2)
    proj_mu = pca_mu.fit_transform(delta_mu)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_modes == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).reshape(-1)

    for ai, mode in enumerate(active_modes):
        ax = axes[ai]
        mode_idx = np.array([i for i, m in enumerate(delta_mode) if m == mode])
        for ci, mid in enumerate(unique_mids):
            idx = mode_idx[delta_mid[mode_idx] == mid]
            if len(idx) == 0:
                continue
            label = f"m{mid}" if n_motions <= 20 else None
            ax.scatter(proj_mu[idx, 0], proj_mu[idx, 1], s=12, alpha=0.5,
                       color=motion_cmap(ci), label=label)
        ax.set_title(f"{mode}", fontsize=10)
        ax.set_xlabel(f"PC1 ({pca_mu.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca_mu.explained_variance_ratio_[1]:.1%})")
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.axvline(0, color="gray", ls=":", lw=0.5)

    for j in range(n_modes, len(axes)):
        axes[j].set_visible(False)

    if n_motions > 20:
        sm = plt.cm.ScalarMappable(cmap=motion_cmap,
                                   norm=plt.Normalize(0, n_motions - 1))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[:n_modes].tolist(), label="motion index",
                     shrink=0.6, pad=0.02)

    plt.suptitle(f"Delta mu per mask mode, coloured by motion (vs {ref_mode})", fontsize=12)
    plt.tight_layout()
    p = os.path.join(out, "ctrl_delta_by_motion_per_mode.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


# ================================================================== #
#  Vis helpers
# ================================================================== #

def _unique_objects(names):
    seen = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen

def _obj_cmap(names):
    u = _unique_objects(names)
    cmap = cm.get_cmap("tab10", max(len(u), 1))
    return {n: cmap(i) for i, n in enumerate(u)}


# ================================================================== #
#  Plot functions
# ================================================================== #

def plot_pca(data, out):
    from sklearn.decomposition import PCA
    mu = data["mu"]
    pca = PCA(n_components=2)
    proj = pca.fit_transform(mu)
    obj_names = data["object_names"]
    ids = data["motion_ids"]
    ts = data["timesteps"]
    has_actions = "action_labels" in data

    ncols = 4 if has_actions else 3
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))

    ax = axes[0]
    for obj in _unique_objects(obj_names):
        mask = [i for i, n in enumerate(obj_names) if n == obj]
        ax.scatter(proj[mask, 0], proj[mask, 1], s=4, alpha=0.5, label=obj)
    ax.set_title("PCA of prior mu (by object)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(markerscale=4, fontsize=8)

    col = 1
    if has_actions:
        ax = axes[col]; col += 1
        action_labels = data["action_labels"]
        unique_actions = _unique_objects(action_labels)  # preserve order
        action_cmap = cm.get_cmap("tab10", max(len(unique_actions), 1))
        for ai, act in enumerate(unique_actions):
            mask = [i for i, a in enumerate(action_labels) if a == act]
            ax.scatter(proj[mask, 0], proj[mask, 1], s=4, alpha=0.5,
                       label=act, color=action_cmap(ai))
        ax.set_title("PCA of prior mu (by action)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(markerscale=4, fontsize=8)

    ax = axes[col]; col += 1
    n_m = int(ids.max()) + 1
    sc = ax.scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.5,
                    c=ids, cmap=cm.get_cmap("tab20", max(n_m, 1)))
    ax.set_title("PCA of prior mu (by motion)")
    plt.colorbar(sc, ax=ax, label="motion id")

    ax = axes[col]
    sc = ax.scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.5, c=ts, cmap="viridis")
    ax.set_title("PCA of prior mu (by timestep)")
    plt.colorbar(sc, ax=ax, label="timestep")

    plt.tight_layout()
    p = os.path.join(out, "pca_prior_mu.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_action_multi_proj(data, out, max_points=8000):
    """Multi-projection comparison of prior mu, all coloured by action label.

    Methods: PCA, t-SNE, UMAP, LDA (supervised by action).
    Generates one combined grid figure + individual paper-quality PDFs.
    """
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import LabelEncoder

    if "action_labels" not in data:
        return

    mu = data["mu"]
    action_labels = data["action_labels"]
    n = mu.shape[0]

    # subsample for speed
    if n > max_points:
        idx = np.random.RandomState(42).choice(n, max_points, replace=False)
    else:
        idx = np.arange(n)
    mu_sub = mu[idx]
    labels_sub = [action_labels[i] for i in idx]

    unique_actions = _unique_objects(action_labels)
    action_cmap = cm.get_cmap("tab10", max(len(unique_actions), 1))
    action_colors = {a: action_cmap(i) for i, a in enumerate(unique_actions)}

    # encode labels for LDA
    le = LabelEncoder()
    le.fit(unique_actions)
    labels_enc = le.transform(labels_sub)
    n_classes = len(unique_actions)

    # ---- build projections ----
    projections = {}

    # PCA
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(mu_sub)
    projections["PCA"] = (
        pca_proj,
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
    )

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(50, len(mu_sub) - 1),
                random_state=42, init="pca", learning_rate="auto")
    tsne_proj = tsne.fit_transform(mu_sub)
    projections["t-SNE"] = (tsne_proj, "t-SNE 1", "t-SNE 2")

    # UMAP
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                            random_state=42)
        umap_proj = reducer.fit_transform(mu_sub)
        projections["UMAP"] = (umap_proj, "UMAP 1", "UMAP 2")
    except ImportError:
        print("  [skip UMAP — not installed]")

    # LDA (supervised by action)
    lda_dim = min(n_classes - 1, 2)
    if lda_dim >= 2:
        lda = LDA(n_components=2)
        lda_proj = lda.fit_transform(mu_sub, labels_enc)
        projections["LDA"] = (
            lda_proj,
            f"LD1 ({lda.explained_variance_ratio_[0]:.1%})",
            f"LD2 ({lda.explained_variance_ratio_[1]:.1%})",
        )
    elif lda_dim == 1:
        lda = LDA(n_components=1)
        lda_1d = lda.fit_transform(mu_sub, labels_enc).ravel()
        lda_proj = np.column_stack([lda_1d, np.zeros_like(lda_1d)])
        projections["LDA (1D)"] = (
            lda_proj,
            f"LD1 ({lda.explained_variance_ratio_[0]:.1%})",
            "(no LD2)",
        )

    # ---- combined grid figure ----
    n_proj = len(projections)
    ncols = min(4, n_proj)
    nrows = (n_proj + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = np.atleast_2d(axes).reshape(-1) if n_proj > 1 else [axes]

    for ai, (name, (proj, xl, yl)) in enumerate(projections.items()):
        ax = axes[ai]
        for act in unique_actions:
            mask = [i for i, a in enumerate(labels_sub) if a == act]
            if mask:
                ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.45,
                           color=action_colors[act], label=act)
        ax.set_title(name, fontsize=13)
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.legend(markerscale=3, fontsize=8)

    for j in range(n_proj, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Prior mu — multiple projections (by action label)", fontsize=14)
    plt.tight_layout()
    p = os.path.join(out, "action_multi_proj.png")
    plt.savefig(p, dpi=200); plt.close()
    print(f"  Saved {p}")

    # ---- individual paper-quality PDFs ----
    for name, (proj, xl, yl) in projections.items():
        fig, ax = plt.subplots(figsize=(7, 6))
        for act in unique_actions:
            mask = [i for i, a in enumerate(labels_sub) if a == act]
            if mask:
                ax.scatter(proj[mask, 0], proj[mask, 1], s=8, alpha=0.45,
                           color=action_colors[act], label=act)
        ax.set_xlabel(xl, fontsize=18)
        ax.set_ylabel(yl, fontsize=18)
        ax.tick_params(labelsize=15)
        ax.legend(markerscale=3, fontsize=14, loc="best")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        p = os.path.join(out, f"paper_action_{safe_name}.pdf")
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved {p}")


# ================================================================== #
#  Text-description-based visualisation
# ================================================================== #

def plot_text_multi_proj(data, text_info, out, max_points=8000):
    """PCA / t-SNE / LDA of prior mu, coloured by text-embedding cluster."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import LabelEncoder

    if "text_clusters" not in data:
        return

    mu = data["mu"]
    cluster_labels = data["text_clusters"]
    n = mu.shape[0]

    if n > max_points:
        idx = np.random.RandomState(42).choice(n, max_points, replace=False)
    else:
        idx = np.arange(n)
    mu_sub = mu[idx]
    labels_sub = [cluster_labels[i] for i in idx]

    unique_clusters = sorted(set(cluster_labels))
    if "unknown" in unique_clusters:
        unique_clusters.remove("unknown")
        unique_clusters.append("unknown")
    n_cl = len(unique_clusters)
    # Use high-contrast colors for small number of clusters
    if n_cl <= 10:
        _distinct = [
            "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
            "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
        ]
        cluster_colors = {c: _distinct[i % len(_distinct)] for i, c in enumerate(unique_clusters)}
    else:
        cmap_c = cm.get_cmap("tab20", max(n_cl, 1))
        cluster_colors = {c: cmap_c(i) for i, c in enumerate(unique_clusters)}

    le = LabelEncoder()
    le.fit(unique_clusters)
    labels_enc = le.transform(labels_sub)
    n_classes = n_cl

    projections = {}

    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(mu_sub)
    projections["PCA"] = (
        pca_proj,
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
    )

    tsne = TSNE(n_components=2, perplexity=min(50, len(mu_sub) - 1),
                random_state=42, init="pca", learning_rate="auto")
    tsne_proj = tsne.fit_transform(mu_sub)
    projections["t-SNE"] = (tsne_proj, "t-SNE 1", "t-SNE 2")

    lda_dim = min(n_classes - 1, 2)
    if lda_dim >= 2:
        lda = LDA(n_components=2)
        lda_proj = lda.fit_transform(mu_sub, labels_enc)
        projections["LDA"] = (
            lda_proj,
            f"LD1 ({lda.explained_variance_ratio_[0]:.1%})",
            f"LD2 ({lda.explained_variance_ratio_[1]:.1%})",
        )
    elif lda_dim == 1:
        lda = LDA(n_components=1)
        lda_1d = lda.fit_transform(mu_sub, labels_enc).ravel()
        lda_proj = np.column_stack([lda_1d, np.zeros_like(lda_1d)])
        projections["LDA (1D)"] = (
            lda_proj,
            f"LD1 ({lda.explained_variance_ratio_[0]:.1%})",
            "(no LD2)",
        )

    # combined grid
    n_proj = len(projections)
    ncols = min(3, n_proj)
    nrows = (n_proj + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = np.atleast_2d(axes).reshape(-1) if n_proj > 1 else [axes]

    for ai, (name, (proj, xl, yl)) in enumerate(projections.items()):
        ax = axes[ai]
        for cl in unique_clusters:
            mask = [i for i, c in enumerate(labels_sub) if c == cl]
            if mask:
                ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.45,
                           color=cluster_colors[cl], label=cl)
        ax.set_title(name, fontsize=13)
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.legend(markerscale=3, fontsize=6, ncol=2)

    for j in range(n_proj, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Prior mu — by text description cluster", fontsize=14)
    plt.tight_layout()
    p = os.path.join(out, "text_cluster_multi_proj.png")
    plt.savefig(p, dpi=200); plt.close()
    print(f"  Saved {p}")

    # short legend labels for paper PDFs: "C0: Lift the object, move..." → "C0"
    _short = {}
    for cl in unique_clusters:
        if cl == "unknown":
            _short[cl] = "unk"
        else:
            _short[cl] = cl.split(":")[0]  # e.g. "C3"

    # individual paper PDFs
    for name, (proj, xl, yl) in projections.items():
        fig, ax = plt.subplots(figsize=(7, 6))
        for cl in unique_clusters:
            mask = [i for i, c in enumerate(labels_sub) if c == cl]
            if mask:
                ax.scatter(proj[mask, 0], proj[mask, 1], s=8, alpha=0.45,
                           color=cluster_colors[cl], label=_short[cl])
        ax.set_xlabel(xl, fontsize=18)
        ax.set_ylabel(yl, fontsize=18)
        ax.tick_params(labelsize=15)
        ax.legend(markerscale=3, fontsize=15, loc="best")
        plt.tight_layout()
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        p = os.path.join(out, f"paper_text_{safe}.pdf")
        plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
        print(f"  Saved {p}")


def plot_text_embedding_space(text_info, out):
    """PCA of text embeddings showing cluster structure."""
    from sklearn.decomposition import PCA

    emb = text_info["embeddings"]
    labels = text_info["cluster_labels"]
    n_clusters = text_info["n_clusters"]
    cluster_names = text_info["cluster_names"]

    pca = PCA(n_components=2)
    proj = pca.fit_transform(emb)

    cmap_c = cm.get_cmap("tab20", max(n_clusters, 1))
    fig, ax = plt.subplots(figsize=(12, 9))
    for ci in range(n_clusters):
        idx = np.where(labels == ci)[0]
        ax.scatter(proj[idx, 0], proj[idx, 1], s=15, alpha=0.6,
                   color=cmap_c(ci), label=cluster_names[ci])
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)
    ax.set_title("Text embedding space (PCA, coloured by cluster)")
    ax.legend(fontsize=7, ncol=2, loc="best")
    plt.tight_layout()
    p = os.path.join(out, "text_embedding_pca.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def _motion_key(fname):
    """Extract base clip key from an augmented .pt filename."""
    base = os.path.splitext(fname)[0]
    parts = base.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else base


def plot_text_cca(data, text_info, out):
    """CCA between per-motion average latent mu and text embedding."""
    from sklearn.cross_decomposition import CCA

    mu = data["mu"]
    mids = data["motion_ids"]
    fnames = data["motion_filenames"]

    unique_mids = sorted(set(mids))
    mid_to_key = {}
    for i, mid in enumerate(mids):
        if mid not in mid_to_key:
            mid_to_key[mid] = _motion_key(fnames[i])

    key_to_emb = {text_info["keys"][i]: text_info["embeddings"][i]
                  for i in range(len(text_info["keys"]))}

    mu_avg, text_embs, matched_keys = [], [], []
    for mid in unique_mids:
        key = mid_to_key[mid]
        if key not in key_to_emb:
            continue
        mask = mids == mid
        mu_avg.append(mu[mask].mean(axis=0))
        text_embs.append(key_to_emb[key])
        matched_keys.append(key)

    if len(mu_avg) < 3:
        print("  [skip CCA — too few matched motions]")
        return

    mu_avg = np.array(mu_avg)
    text_embs = np.array(text_embs)

    n_comp = min(10, mu_avg.shape[0] - 1, mu_avg.shape[1], text_embs.shape[1])
    cca = CCA(n_components=n_comp)
    X_c, Y_c = cca.fit_transform(mu_avg, text_embs)
    corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_comp)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(range(n_comp), corrs, color="teal", alpha=0.7)
    axes[0].set_xlabel("CCA component")
    axes[0].set_ylabel("Canonical correlation")
    axes[0].set_title(f"CCA: latent mu vs text emb ({len(mu_avg)} motions)")
    axes[0].set_ylim(0, 1)

    ax = axes[1]
    ax.scatter(X_c[:, 0], Y_c[:, 0], s=30, alpha=0.6, c="steelblue")
    ax.set_xlabel("Latent CCA1")
    ax.set_ylabel("Text CCA1")
    ax.set_title(f"CCA dim 1 (r={corrs[0]:.3f})")
    z = np.polyfit(X_c[:, 0], Y_c[:, 0], 1)
    xs = np.linspace(X_c[:, 0].min(), X_c[:, 0].max(), 100)
    ax.plot(xs, np.polyval(z, xs), "r--", lw=1, alpha=0.7)

    plt.tight_layout()
    p = os.path.join(out, "text_cca.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_text_sim_comparison(data, text_info, out):
    """Compare pairwise text similarity with pairwise latent similarity."""
    from sklearn.metrics.pairwise import cosine_similarity

    mu = data["mu"]
    mids = data["motion_ids"]
    fnames = data["motion_filenames"]

    unique_mids = sorted(set(mids))
    mid_to_key = {}
    for i, mid in enumerate(mids):
        if mid not in mid_to_key:
            mid_to_key[mid] = _motion_key(fnames[i])

    key_to_emb = {text_info["keys"][i]: text_info["embeddings"][i]
                  for i in range(len(text_info["keys"]))}

    mu_avg, text_embs = [], []
    for mid in unique_mids:
        key = mid_to_key[mid]
        if key not in key_to_emb:
            continue
        mask = mids == mid
        mu_avg.append(mu[mask].mean(axis=0))
        text_embs.append(key_to_emb[key])

    if len(mu_avg) < 3:
        return

    mu_avg = np.array(mu_avg)
    text_embs = np.array(text_embs)
    n = len(mu_avg)

    text_sim = cosine_similarity(text_embs)
    latent_sim = cosine_similarity(mu_avg)

    triu_idx = np.triu_indices(n, k=1)
    text_pairs = text_sim[triu_idx]
    latent_pairs = latent_sim[triu_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(text_pairs, latent_pairs, s=8, alpha=0.3, color="steelblue")
    r = np.corrcoef(text_pairs, latent_pairs)[0, 1]
    ax.set_xlabel("Text cosine similarity")
    ax.set_ylabel("Latent mu cosine similarity")
    ax.set_title(f"Pairwise similarity (r={r:.3f})")

    ax = axes[1]
    im = ax.imshow(text_sim, cmap="viridis", aspect="auto")
    ax.set_title("Text embedding similarity")
    plt.colorbar(im, ax=ax, fraction=0.03)

    ax = axes[2]
    im = ax.imshow(latent_sim, cmap="viridis", aspect="auto")
    ax.set_title("Latent mu similarity")
    plt.colorbar(im, ax=ax, fraction=0.03)

    plt.suptitle(f"Text vs latent similarity ({n} motions)", fontsize=12)
    plt.tight_layout()
    p = os.path.join(out, "text_sim_comparison.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_tsne(data, out, perplexity=30, max_points=5000):
    from sklearn.manifold import TSNE
    mu = data["mu"]
    n = mu.shape[0]
    idx = np.random.choice(n, min(n, max_points), replace=False) if n > max_points else np.arange(n)
    mu_sub = mu[idx]
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(mu_sub) - 1),
                random_state=42, init="pca", learning_rate="auto")
    proj = tsne.fit_transform(mu_sub)
    obj_sub = [data["object_names"][i] for i in idx]
    ids_sub = data["motion_ids"][idx]
    ts_sub  = data["timesteps"][idx]
    has_actions = "action_labels" in data

    ncols = 4 if has_actions else 3
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    ax = axes[0]
    for obj in _unique_objects(data["object_names"]):
        mask = [i for i, n in enumerate(obj_sub) if n == obj]
        if mask:
            ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.5, label=obj)
    ax.set_title("t-SNE of prior mu (by object)"); ax.legend(markerscale=4, fontsize=8)

    col = 1
    if has_actions:
        ax = axes[col]; col += 1
        action_sub = [data["action_labels"][i] for i in idx]
        unique_actions = _unique_objects(data["action_labels"])
        action_cmap = cm.get_cmap("tab10", max(len(unique_actions), 1))
        for ai, act in enumerate(unique_actions):
            mask = [i for i, a in enumerate(action_sub) if a == act]
            if mask:
                ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.5,
                           label=act, color=action_cmap(ai))
        ax.set_title("t-SNE of prior mu (by action)"); ax.legend(markerscale=4, fontsize=8)

    ax = axes[col]; col += 1
    sc = ax.scatter(proj[:, 0], proj[:, 1], s=6, alpha=0.5,
                    c=ids_sub, cmap=cm.get_cmap("tab20", max(int(data["motion_ids"].max())+1,1)))
    ax.set_title("t-SNE of prior mu (by motion)"); plt.colorbar(sc, ax=ax, label="motion id")

    ax = axes[col]
    sc = ax.scatter(proj[:, 0], proj[:, 1], s=6, alpha=0.5, c=ts_sub, cmap="viridis")
    ax.set_title("t-SNE of prior mu (by timestep)"); plt.colorbar(sc, ax=ax, label="timestep")

    plt.tight_layout()
    p = os.path.join(out, "tsne_prior_mu.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_dim_histograms(data, out, top_k=16):
    from scipy.stats import gaussian_kde
    mu = data["mu"]
    var = mu.var(axis=0)
    top = np.argsort(var)[::-1][:top_k]
    nrows, ncols = 4, (top_k + 3) // 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
    axes = axes.flatten()
    cmap = _obj_cmap(data["object_names"])
    for pi, dim in enumerate(top):
        ax = axes[pi]; vals = mu[:, dim]
        ax.hist(vals, bins=60, density=True, alpha=0.3, color="gray", label="all")
        for obj in _unique_objects(data["object_names"]):
            v = vals[np.array([n == obj for n in data["object_names"]])]
            if len(v) < 5: continue
            try:
                kde = gaussian_kde(v)
                xs = np.linspace(v.min(), v.max(), 200)
                ax.plot(xs, kde(xs), lw=1.5, label=obj, color=cmap[obj])
            except np.linalg.LinAlgError:
                pass
        ax.set_title(f"dim {dim} (var={var[dim]:.3f})", fontsize=9)
        if pi == 0: ax.legend(fontsize=6)
    for j in range(pi + 1, len(axes)): axes[j].set_visible(False)
    plt.suptitle("Prior mu per-dimension (top variance dims)", fontsize=12)
    plt.tight_layout()
    p = os.path.join(out, "dim_histograms.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_all_dim_histograms(data, out):
    mu = data["mu"]; nd = mu.shape[1]
    ncols = 8; nrows = (nd + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2))
    axes = axes.flatten()
    for d in range(nd):
        axes[d].hist(mu[:, d], bins=50, density=True, alpha=0.6, color="steelblue")
        axes[d].set_title(f"d{d}", fontsize=7); axes[d].tick_params(labelsize=5)
    for j in range(nd, len(axes)): axes[j].set_visible(False)
    plt.suptitle("Prior mu – all 64 dims", fontsize=11)
    plt.tight_layout()
    p = os.path.join(out, "dim_histograms_all64.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_temporal(data, out, max_motions=6, top_k=8):
    mu = data["mu"]; mids = data["motion_ids"]; ts = data["timesteps"]
    var = mu.var(axis=0); top = np.argsort(var)[::-1][:top_k]
    uniq = sorted(set(mids))[:max_motions]
    fig, axes = plt.subplots(len(uniq), 1, figsize=(14, 3 * len(uniq)), sharex=False)
    if len(uniq) == 1: axes = [axes]
    cmp = cm.get_cmap("tab10", max(top_k, 1))
    for ax, mid in zip(axes, uniq):
        mask = mids == mid; t = ts[mask]; si = np.argsort(t)
        for k, d in enumerate(top):
            ax.plot(t[si], mu[mask][si, d], lw=1, alpha=0.8, color=cmp(k), label=f"dim {d}")
        ax.set_title(f"Motion {mid}: {data['motion_filenames'][np.where(mask)[0][0]]}", fontsize=9)
        ax.set_ylabel("mu")
        if mid == uniq[0]: ax.legend(fontsize=6, ncol=top_k, loc="upper right")
    axes[-1].set_xlabel("timestep")
    plt.suptitle("Temporal prior mu (top-variance dims)", fontsize=11)
    plt.tight_layout()
    p = os.path.join(out, "temporal_mu.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_variance_analysis(data, out):
    mu = data["mu"]; logvar = data["logvar"]
    mu_var = mu.var(axis=0); mean_lv = logvar.mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(range(len(mu_var)), mu_var, color="steelblue", alpha=0.7)
    axes[0].set_title("Var(mu) per dim"); axes[0].set_xlabel("dim"); axes[0].set_ylabel("variance")

    axes[1].bar(range(len(mean_lv)), mean_lv, color="coral", alpha=0.7)
    axes[1].set_title("Mean(logvar) per dim"); axes[1].set_xlabel("dim")

    snr = mu_var / (np.exp(mean_lv) + 1e-8)
    axes[2].bar(range(len(snr)), snr, color="seagreen", alpha=0.7)
    axes[2].axhline(1.0, color="red", ls="--", lw=0.8, label="SNR=1")
    axes[2].set_title("SNR: Var(mu)/E[exp(logvar)]"); axes[2].set_xlabel("dim"); axes[2].legend()

    plt.tight_layout()
    p = os.path.join(out, "variance_analysis.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_multimodality(data, out):
    from scipy.stats import skew, kurtosis as krt
    mu = data["mu"]; nd = mu.shape[1]
    bc = np.array([(skew(mu[:, d])**2 + 1) / max(krt(mu[:, d], fisher=False), 1e-8) for d in range(nd)])

    obj_bc = {}
    for obj in _unique_objects(data["object_names"]):
        mask = np.array([n == obj for n in data["object_names"]])
        obj_bc[obj] = np.array([
            (skew(mu[mask, d])**2 + 1) / max(krt(mu[mask, d], fisher=False), 1e-8)
            if mask.sum() >= 5 else 0.0
            for d in range(nd)
        ])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].bar(range(nd), bc, color="mediumpurple", alpha=0.7)
    axes[0].axhline(0.555, color="red", ls="--", lw=1, label="BC=0.555")
    axes[0].set_title("Bimodality Coefficient (all)"); axes[0].set_xlabel("dim"); axes[0].legend()

    cmap = _obj_cmap(data["object_names"]); objs = _unique_objects(data["object_names"])
    w = 0.8 / max(len(objs), 1)
    for oi, obj in enumerate(objs):
        axes[1].bar(np.arange(nd) + oi * w - 0.4, obj_bc[obj], width=w, alpha=0.6, label=obj, color=cmap[obj])
    axes[1].axhline(0.555, color="red", ls="--", lw=1)
    axes[1].set_title("Bimodality Coefficient (per object)"); axes[1].set_xlabel("dim"); axes[1].legend(fontsize=7)

    plt.tight_layout()
    p = os.path.join(out, "multimodality_scores.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_z_vs_mu(data, out):
    from sklearn.decomposition import PCA
    mu = data["mu"]; z = data["z_samples"]
    pca = PCA(n_components=2)
    proj = pca.fit_transform(np.vstack([mu, z]))
    pm, pz = proj[:len(mu)], proj[len(mu):]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(pm[:, 0], pm[:, 1], s=3, alpha=0.3, color="blue", label="mu")
    ax.scatter(pz[:, 0], pz[:, 1], s=3, alpha=0.3, color="red", label="z (sampled)")
    ax.set_title("PCA: mu vs sampled z"); ax.legend(markerscale=5)
    plt.tight_layout()
    p = os.path.join(out, "mu_vs_z.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


def plot_correlation(data, out):
    corr = np.corrcoef(data["mu"].T)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title("Latent dim correlation (prior mu)")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    p = os.path.join(out, "latent_correlation.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


# ================================================================== #
#  Multi-sample per observation
# ================================================================== #

def collect_multi_sample(policy, motion_paths, device, goal_phase_dim,
                         n_samples=50, every=30, max_steps=300):
    vae = policy.a2c_network; vd = vae.vae_dim
    zs_list, mu_list, labels = [], [], []
    for mi, mp in enumerate(motion_paths[:3]):
        fn = os.path.basename(mp); obj = fn.split("_")[1]
        print(f"  multi-sample: {fn}")
        env = RefEnv(mp, goal_phase_dim)
        for t in range(0, min(max_steps, env.max_episode_length - 1), every):
            obs = env.get_obs(t).to(device)
            with torch.no_grad():
                pr = vae._prior({"obs": obs})
                mu = pr["mu"].cpu().numpy().squeeze(0)
                lv = pr["logvar"].cpu().numpy().squeeze(0)
            std = np.exp(0.5 * lv)
            zs_list.append(mu[None] + std[None] * np.random.randn(n_samples, vd))
            mu_list.append(mu); labels.append((mi, t, obj))
    return {"obs_z_list": zs_list, "obs_mu_list": mu_list, "labels": labels}


def plot_multi_sample(ms, out):
    from sklearn.decomposition import PCA
    if not ms["obs_z_list"]: return
    combined = np.vstack(ms["obs_z_list"])
    pca = PCA(n_components=2); proj = pca.fit_transform(combined)
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = cm.get_cmap("tab20", max(len(ms["obs_z_list"]), 1))
    off = 0
    for i, (zs, lbl) in enumerate(zip(ms["obs_z_list"], ms["labels"])):
        n = zs.shape[0]; p = proj[off:off+n]
        ax.scatter(p[:, 0], p[:, 1], s=8, alpha=0.4, color=cmap(i),
                   label=f"m{lbl[0]}_t{lbl[1]}_{lbl[2]}")
        mp = pca.transform(ms["obs_mu_list"][i][None])
        ax.scatter(mp[0, 0], mp[0, 1], s=80, marker="x", color=cmap(i), lw=2)
        off += n
    ax.set_title("Multi-sample z per obs (x = mu)"); ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    p = os.path.join(out, "multi_sample_z.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved {p}")


# ================================================================== #
#  Main
# ================================================================== #

def main():
    pa = argparse.ArgumentParser(description="VAE latent space visualisation")
    pa.add_argument("--ckpt", required=True)
    pa.add_argument("--motion_dir", default="InterAct/OMOMO_retarget_aug")
    pa.add_argument("--output_dir", default="latent_vis")
    pa.add_argument("--num_motions", type=int, default=10)
    pa.add_argument("--max_steps", type=int, default=300)
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--goal_phase_dim", type=int, default=4)
    pa.add_argument("--skip_tsne", action="store_true")
    pa.add_argument("--multi_sample", action="store_true")
    pa.add_argument("--mask_analysis", action="store_true",
                    help="Run mask-conditioned latent analysis")
    pa.add_argument("--mask_only", action="store_true",
                    help="Only run mask-conditioned analysis (skip base plots)")
    pa.add_argument("--controlled", action="store_true",
                    help="Run controlled-variable analysis (fixed obs, varying mask)")
    pa.add_argument("--controlled_only", action="store_true",
                    help="Only run controlled-variable analysis")
    pa.add_argument("--step_stride", type=int, default=5,
                    help="Step stride for mask-conditioned analysis")
    pa.add_argument("--seed", type=int, default=42,
                    help="Random seed for motion sampling")
    pa.add_argument("--label_csv", type=str, default=None,
                    help="CSV with action labels (e.g. InterAct/omomo.csv)")
    pa.add_argument("--text_dir", type=str, default=None,
                    help="Directory with text descriptions (e.g. extracted_texts)")
    pa.add_argument("--text_n_clusters", type=int, default=None,
                    help="Number of text clusters (auto-select via silhouette if not set)")
    pa.add_argument("--mask_object", action="store_true",
                    help="Replace object names in text descriptions with 'object'")
    args = pa.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model ...")
    policy = load_model(args.ckpt, args.device, args.goal_phase_dim)
    print(f"  VAE latent dim = {policy.a2c_network.vae_dim}")

    motion_paths = sorted(glob_module.glob(os.path.join(args.motion_dir, "*.pt")))
    if not motion_paths:
        sys.exit(f"No .pt files in {args.motion_dir}")
    if args.num_motions > 0 and args.num_motions < len(motion_paths):
        # Group by base clip key so we sample diverse clips, not duplicate augs
        from collections import defaultdict
        clip_groups = defaultdict(list)
        for p in motion_paths:
            key = _motion_key(os.path.basename(p))
            clip_groups[key].append(p)
        rng = random.Random(args.seed)
        unique_keys = list(clip_groups.keys())
        n_pick = min(args.num_motions, len(unique_keys))
        sampled_keys = rng.sample(unique_keys, n_pick)
        # one random augmentation per sampled base clip
        motion_paths = [rng.choice(clip_groups[k]) for k in sampled_keys]
        rng.shuffle(motion_paths)  # random order
        print(f"  Sampled {n_pick} unique base clips from {len(unique_keys)} total")
    print(f"Processing {len(motion_paths)} motions (reference-only, no physics)\n")

    # ---- load action labels if provided ----
    action_label_map = None
    if args.label_csv:
        action_label_map = load_action_labels(args.label_csv)
        print(f"Loaded {len(action_label_map)} action labels from {args.label_csv}")
        uniq = sorted(set(action_label_map.values()))
        print(f"  Action types: {uniq}")

    # ---- load text descriptions if provided ----
    # NOTE: clustering is deferred until after motion sampling so that only
    # descriptions matching the sampled motions are used.
    text_desc_map_full = None
    text_info = None
    if args.text_dir:
        print("\n=== Loading text descriptions ===")
        text_desc_map_full = load_text_descriptions(args.text_dir, mask_object=args.mask_object)
        print(f"  Loaded {len(text_desc_map_full)} descriptions (full)")
        # filter to sampled motion keys only
        sampled_base_keys = set(_motion_key(os.path.basename(p)) for p in motion_paths)
        desc_map = {k: v for k, v in text_desc_map_full.items() if k in sampled_base_keys}
        print(f"  Kept {len(desc_map)} descriptions matching sampled motions "
              f"({len(set(desc_map.values()))} unique)")
        text_info = build_text_info(desc_map, n_clusters=args.text_n_clusters)
        print(f"  {text_info['n_clusters']} clusters:")
        for ci, name in sorted(text_info["cluster_names"].items()):
            n_in = int((text_info["cluster_labels"] == ci).sum())
            print(f"    {name}  ({n_in} motions)")

    if not args.mask_only and not args.controlled_only:
        data = collect_latent_data(policy, motion_paths, args.device,
                                   args.goal_phase_dim, args.max_steps)

        # attach action labels to data
        if action_label_map is not None:
            labels = []
            for fname in data["motion_filenames"]:
                # .pt filenames are augmented: sub10_largebox_000_063_070_076_100_100_100.pt
                # CSV keys are base clips: sub10_largebox_000
                # extract first 3 underscore-separated parts to match
                base = os.path.splitext(fname)[0]
                parts = base.split("_")
                key = "_".join(parts[:3]) if len(parts) >= 3 else base
                labels.append(action_label_map.get(key, "unknown"))
            data["action_labels"] = labels
            n_labelled = sum(1 for l in labels if l != "unknown")
            print(f"  Matched {n_labelled}/{len(labels)} samples to action labels")

        # attach text clusters to data
        if text_info is not None:
            clusters = []
            for fname in data["motion_filenames"]:
                key = _motion_key(fname)
                cid = text_info["key_to_cluster"].get(key)
                if cid is not None:
                    clusters.append(text_info["cluster_names"][cid])
                else:
                    clusters.append("unknown")
            data["text_clusters"] = clusters
            n_matched = sum(1 for c in clusters if c != "unknown")
            print(f"  Matched {n_matched}/{len(clusters)} samples to text clusters")

        np.savez(os.path.join(args.output_dir, "latent_data.npz"),
                 mu=data["mu"], logvar=data["logvar"], z_samples=data["z_samples"],
                 motion_ids=data["motion_ids"], timesteps=data["timesteps"])
        print(f"\nCollected {data['mu'].shape[0]} vectors (shape={data['mu'].shape})")

        print("\nGenerating plots ...")
        plot_pca(data, args.output_dir)
        if "action_labels" in data:
            plot_action_multi_proj(data, args.output_dir)
        if "text_clusters" in data:
            print("\nText-description plots ...")
            plot_text_multi_proj(data, text_info, args.output_dir)
            plot_text_embedding_space(text_info, args.output_dir)
            plot_text_cca(data, text_info, args.output_dir)
            plot_text_sim_comparison(data, text_info, args.output_dir)
        if not args.skip_tsne:
            plot_tsne(data, args.output_dir)
        plot_dim_histograms(data, args.output_dir)
        plot_all_dim_histograms(data, args.output_dir)
        plot_temporal(data, args.output_dir)
        plot_variance_analysis(data, args.output_dir)
        plot_multimodality(data, args.output_dir)
        plot_z_vs_mu(data, args.output_dir)
        plot_correlation(data, args.output_dir)

        if args.multi_sample:
            print("\nMulti-sample analysis ...")
            ms = collect_multi_sample(policy, motion_paths, args.device, args.goal_phase_dim)
            plot_multi_sample(ms, args.output_dir)

    # ---- mask-conditioned analysis (aggregate) ----
    if args.mask_analysis or args.mask_only:
        print("\n=== Mask-conditioned analysis ===")
        print(f"  Mask modes: {list(define_mask_modes().keys())}")
        print(f"  Step stride: {args.step_stride}")
        mdata = collect_mask_conditioned(
            policy, motion_paths, args.device, args.goal_phase_dim,
            args.max_steps, step_stride=args.step_stride,
        )
        print(f"\nCollected {mdata['mu'].shape[0]} mask-conditioned vectors")
        print("\nGenerating mask-conditioned plots ...")
        plot_mask_pca(mdata, args.output_dir)
        if not args.skip_tsne:
            plot_mask_tsne(mdata, args.output_dir)
        plot_mask_per_mode_pca(mdata, args.output_dir)
        plot_mask_mu_shift(mdata, args.output_dir)
        plot_mask_dim_comparison(mdata, args.output_dir)

    # ---- controlled-variable analysis (fixed obs, varying mask) ----
    if args.controlled or args.controlled_only:
        print("\n=== Controlled-variable analysis ===")
        mode_list = list(define_mask_modes().keys())
        print(f"  Mask modes: {mode_list}")
        records, mode_list = collect_controlled(
            policy, motion_paths, args.device, args.goal_phase_dim,
            args.max_steps,
        )
        print(f"\nCollected {len(records)} controlled records")
        print("\nGenerating controlled-variable plots ...")
        plot_fixed_obs_heatmap(records, mode_list, args.output_dir)
        plot_action_distance(records, mode_list, args.output_dir)
        plot_temporal_mask_heatmap(records, mode_list, args.output_dir)
        plot_delta_structure(records, mode_list, args.output_dir)
        plot_delta_multi_proj(records, mode_list, args.output_dir)
        plot_delta_by_motion(records, mode_list, args.output_dir)

    print(f"\nDone – all plots in {args.output_dir}/")


if __name__ == "__main__":
    main()
