# Deployment notes

The real-robot stack (ROS node, Unitree G1 low-level control, OptiTrack object tracking, egocentric depth
point-cloud extraction, keyboard goal interface) is **not part of this repository**. What this repository provides
for deployment is:

1. `ultra/export_jit.py` — exports a trained VAE student checkpoint to a TorchScript module. `forward(obs)` takes
   the raw 1496-d student observation (batch x 1496) and returns the 29-d action in [-1, 1] with the VAE latent noise
   set to zero (prior mean); `forward_with_noise(obs, noise)` accepts an explicit (batch x 64) latent sample.
   Observation normalization is baked in when the checkpoint carries running statistics (`normalize_input: True`);
   the released config trains with `normalize_input: False`, so identity normalization is used. The action is not a
   joint target: the PD target is `action * 3.0` (the `control.action_scale` used in training) in the G1 29-DoF joint
   order, see the `scaled_actions` block in `ultra/sim2sim_vae.py`.
2. `ultra/utils/obs_vae.py` (`MujocoObs`) — the reference implementation of the student observation builder
   outside Isaac Lab (proprioception, history buffer, modality masks, object point cloud / pose, goal commands).
   `ultra/sim2sim_vae.py` drives it at 60 Hz with 17 MuJoCo substeps of 1/(60*17) s (~0.98 ms); Isaac Lab training
   uses `controlFrequencyInv: 17` over a 1 ms PhysX step (17 ms, ~58.8 Hz).
3. `ultra/sim2sim_vae_keyboard.py` — the keyboard goal interface used in the sim2sim demos.

The PD gains, armature and effort limits used by the released policies are defined in
`ultra/isaac/legacy_layout.py` (`G1_STIFFNESS`, `G1_DAMPING`, `G1_ARMATURE`, `G1_EFFORT`, in the G1 29-DoF joint
order); in the yaml `control:` block only `stiffness` / `damping` are legacy and unused, while `action_scale` and
`control_type` are read by `Humanoid_SMPLX._compute_torques`.
