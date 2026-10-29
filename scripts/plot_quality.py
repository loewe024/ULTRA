import torch
import matplotlib.pyplot as plt
import numpy as np

# Load Tensors
quality_data = torch.load("./output/retarget_export_080_quality/sub10_largebox_043_xyz1p000000-1p000000-1p000000_080_080_080.pt")  # Shape: (T, 3)
original_motion = torch.load("./InterAct/all_motions/sub10_largebox_043.pt")
retargeted_motion = torch.load("./output/retarget_export_080/sub10_largebox_043_xyz1p000000-1p000000-1p000000_080_080_080.pt")
rollout_motion = torch.load("./output/teacher_videos/sub10_largebox_043_xyz1p000000-1p000000-1p000000_080_080_080/rollout.pt")


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


# Plot differences in object positions
plt.figure(figsize=(12, 5))
#plt.plot(quality_data[:, 0], label="Reward")
#plt.plot(quality_data[:, 1], label="Foot Error")
plt.plot(quality_data[:, 2], label="Object Error")
plt.plot(retarget_rollout_error, label="retarget_rollout_error")
plt.plot(original_retarget_error, label="original_retarget_error")
plt.plot(original_rollout_error, label="original_rollout_error")

plt.xlabel("Frame")
plt.ylabel("Value")
plt.title("Difference in object position")
plt.legend()
plt.grid(True)
plt.tight_layout()
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
    (rollout_object_pos, "Rollout"),
]:
    ax.scatter(*positions[0], marker="o", s=60, label=f"{label} start")
    ax.scatter(*positions[-1], marker="X", s=60, label=f"{label} end")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Object Trajectories")
ax.legend()
plt.tight_layout()
plt.show()