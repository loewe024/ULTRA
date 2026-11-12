import os
import sys
import torch

# utils lives in ultra/utils; add ultra/ to the import path like `python ultra/run.py` does implicitly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ultra'))

from utils import torch_utils
import matplotlib.pyplot as plt
import numpy as np

# Load Tensors
clip = "043"
subject = "sub8_largebox"
motion_name = f"{subject}_{clip}"

quality_data = torch.load(f"./output/retarget_export_080_largebox_quality/{motion_name}_xyz1p000000-1p000000-1p000000_080_080_080.pt")
original_motion = torch.load(f"./InterAct/all_motions/{motion_name}.pt")
retargeted_motion = torch.load(f"./output/retarget_export_080_largebox/{motion_name}_xyz1p000000-1p000000-1p000000_080_080_080.pt")
#rollout_motion = torch.load(f"./output/teacher_rollout_motions/more_rop_with_curr_source/{motion_name}_xyz1p000000-1p000000-1p000000_080_080_080.pt")
#rollout_motion = torch.load(f"./output/teacher_rollout_motions/no_curr_retarget/{motion_name}_xyz1p000000-1p000000-1p000000_080_080_080.pt")
#rollout_motion = torch.load(f"./output/teacher_rollout_motions/quality_all_from_0/{motion_name}_xyz1p000000-1p000000-1p000000_080_080_080.pt")
#rollout_motion = torch.load(f"./output/teacher_rollout_motions/quality_scaling_rop_19250/{motion_name}_xyz1p000000-1p000000-1p000000_080_080_080.pt")
rollout_motion = torch.load(f"./output/teacher_rollout_motions/quality_world_obj_16000/{motion_name}_xyz1p000000-1p000000-1p000000_080_080_080.pt")
#quality_world_obj_16000, quality_small_objGain_14250
"""
Quality tensor cols:
0           reward
1           Foot error
2           object error
3           rig
4:10        hand positions of rubber hands first left xyz, then right xyz
10:16       ref hand positions
16:784      obj points (256 points), xyz point for point
784:1552    ref obj points (256 points)   
"""



# Convert to numpy
def source_at_60hz(raw):                     # raw: [T, 591]
    x = raw[1:]                               # drop the zero frame 0
    t = np.arange(2 * len(x) - 1) / 2.0       # 2T-3 time stamps
    return np.stack([np.interp(t, np.arange(len(x)), x[:, k]) for k in range(x.shape[1])], 1)

original_motion = original_motion.detach().numpy()
original_motion = source_at_60hz(original_motion)
retargeted_motion = retargeted_motion.detach().numpy()
rollout_motion = rollout_motion["observed_hoi"].detach().cpu().numpy()[:, :630]
rollout_motion = np.concatenate([rollout_motion[:1], rollout_motion], axis=0)
quality_data = quality_data.detach().numpy()
ref_object_pos = 0.8 * original_motion[:, 318:321]
retarget_object_pos = retargeted_motion[:, 71:74]
rollout_object_pos = rollout_motion[:, 71:74]
retarget_rollout_error = np.linalg.norm(rollout_object_pos-retarget_object_pos, axis=1)
original_retarget_error = np.linalg.norm(ref_object_pos-retarget_object_pos, axis=1)
original_rollout_error = np.linalg.norm(rollout_object_pos-ref_object_pos, axis=1)

#Different possible quality metrics

q1 = original_motion[:, 321:325] / np.maximum(np.linalg.norm(original_motion[:, 321:325], axis=-1, keepdims=True), 1e-8)
q2 = retargeted_motion[:, 74:78] / np.maximum(np.linalg.norm(retargeted_motion[:, 74:78], axis=-1, keepdims=True), 1e-8)
dot = np.sum(q1 * q2, axis=-1)
dot = np.abs(dot)
dot = np.clip(dot, 0.0, 1.0)
orig_ret_obj_orient_err = 2.0 * np.arccos(dot)# * 180/3.141

# Hand positions: (T, 2, 3)
hand_pos = quality_data[:, 4:10].reshape(-1, 2, 3)
ref_hand_pos = quality_data[:, 10:16].reshape(-1, 2, 3)

# Object points: (T, 256, 3)
obj_points = quality_data[:, 16:784].reshape(-1, 256, 3)
ref_obj_points = quality_data[:, 784:1552].reshape(-1, 256, 3)

# Distances from each hand to each object point: (T, 2, 256)
dist = np.linalg.norm(hand_pos[:, :, None, :] - obj_points[:, None, :, :], axis=-1)

ref_dist = np.linalg.norm(ref_hand_pos[:, :, None, :] - ref_obj_points[:, None, :, :], axis=-1)

# Absolute distance errors
errors = np.abs(dist - ref_dist)

# Count points exceeding threshold for each hand
counts = np.sum(errors > 0.01, axis=-1)
mean_errors = np.mean(errors, axis=-1)  # (T, 2)
mean_tot = np.mean(mean_errors, axis=1)
#print(mean_tot)

# Plot differences in object positions
fig1 = plt.figure(figsize=(12, 5))
plt.plot(quality_data[:, 0], label="Reward")
#plt.plot(quality_data[:, 1], label="Foot Error")
#plt.plot(quality_data[:, 2], label="Object Error")
#plt.plot(retarget_rollout_error, label="retarget_rollout_error")
plt.plot(original_retarget_error, label="original_retarget_error")
#plt.plot(original_rollout_error, label="original_rollout_error")
plt.plot(orig_ret_obj_orient_err, label="orig_ret_obj_orient_err")
#plt.plot(mean_errors[:, 0], label="ig_error_left")
#plt.plot(mean_errors[:, 1], label="ig_error_right")
plt.plot(mean_tot, label="ig_error_total")

plt.xlabel("Frame")
plt.ylabel("Value")
plt.title("Difference in object position")
plt.legend()
plt.grid(True)
plt.tight_layout()
fig1.savefig("output/graphics/quality.png", dpi=150)
#plt.show()



fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Original trajectory
ax.plot(
    ref_object_pos[:, 0],
    ref_object_pos[:, 1],
    ref_object_pos[:, 2],
    label="Original motion",
)

# Retargeted trajectory
ax.plot(
    retarget_object_pos[:, 0],
    retarget_object_pos[:, 1],
    retarget_object_pos[:, 2],
    label="Retargeted motion",
)

# Rollout trajectory
ax.plot(
    rollout_object_pos[:, 0],
    rollout_object_pos[:, 1],
    rollout_object_pos[:, 2],
    label="Rollout motion",
)

# Mark start and end positions
for positions, label in [
    (ref_object_pos, "Original"),
    (retarget_object_pos, "Retargeted"),
    #(rollout_object_pos, "Rollout"),
]:
    ax.scatter(*positions[0], marker="o", s=60, label=f"{label} start")
    ax.scatter(*positions[-1], marker="X", s=60, label=f"{label} end")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Object Trajectories")
ax.legend()
plt.tight_layout()
#fig.savefig("output/graphics/object_traj.png", dpi=150)
plt.show()