# Retargeting-reward quality observation for the teacher: implementation guide

## 1. Idea and data flow

The teacher (stage 3, task `UltraG1Retarget`) tracks G1 trajectories that the retargeting policy produced (stage 2 export, `[T, 630]` at 60 Hz). Some frames are retargeted badly: the feet or hands drift from the human reference, or the object pose or contacts don't match. The teacher has no way to tell these frames from good ones.

First experiment: use the per-frame reward of the retargeting env (`UltraG1`) as a quality signal and give it to the teacher as an extra observation.

```
Stage 1  UltraG1 training           (unchanged)
Stage 2  UltraG1 --test export      NEW: also record rew_buf[0] per frame
            ├─ OUTPUT/<clip>.pt                [T, 630]   (unchanged)
            └─ OUTPUT_quality/<clip>.pt        [T] or [T, 3]   NEW
Stage 3  UltraG1Retarget            NEW: load OUTPUT_quality/<clip>.pt,
                                         append quality(t+1), quality(t+16) to obs
Stage 4/5 students                  inherit automatically (teacher obs)
```

Why a **separate folder**:
- `Ultra.__init__` (`ultra/env/tasks/ultra.py:37`) loads every file in `motion_file` with `os.listdir`, so a sidecar file in the same folder would be treated as a motion.
- Column 630 can't be used either: `UltraG1Retarget._load_motion` appends `ref_ig` at `630:`, and other consumers (`render_rollout.py`, sim2sim, `play_dataset`) expect exactly 630 columns.

---

## 2. Relevant existing code

| Where | What it does |
|---|---|
| `ultra/env/tasks/humanoid_g1.py:152` `Humanoid_G1._compute_reward` | Calls `compute_humanoid_reward`, multiplies by `SmoothRewards` and writes `self.rew_buf`. `metric_1` and `metric_2` are returned and then **discarded**. |
| `ultra/env/tasks/humanoid_g1.py:345` `compute_humanoid_reward` | The retargeting reward: `reward = standing*w1 + rb*ro*rig*w2 + rb*ro*rig*rcg*w3`. `metric_1` is the foot keypoint error (m) and `metric_2` the object surface-point error (m). |
| `ultra/env/tasks/ultra_g1.py:415` `UltraG1._physics_step` | Export: one frame before the first step, then one frame after every step. |
| `ultra/env/tasks/ultra_g1.py:425` `UltraG1.post_physics_step` | `super()` computes obs, reward and reset. It then saves the rollout when `reset_buf[0]` is set. |
| `ultra/env/tasks/ultra_g1.py:430` `_capture_retarget_frame` | Builds one 630-dim frame of env 0. |
| `ultra/env/tasks/ultra_g1.py:444` `_save_retarget_rollout` | Drops the first 30 standing frames and calls `torch.save`. |
| `scripts/export_retarget_smplx.py:76-95` | Writes `env.yaml` per clip and variant (`retargetExportPath`, …) and starts `run.py --test`. |
| `ultra/env/tasks/ultra_g1_retarget.py:159` `UltraG1Retarget._load_motion` | Loads `[T, 630]`, appends 20 copies of the last frame and `ref_ig`, pads to `max_length` and builds `self.hoi_data` / `self.hoi_refs`. |
| `ultra/env/tasks/ultra_g1_retarget.py:800` `_compute_observations_iter` | Reference index `next_ts = clamp(progress_buf + delta_t, max=len-1)`, with the stand-still override `next_ts = where(is_stand_still, stand_still_frame, next_ts)`. |
| `ultra/env/tasks/ultra_g1_retarget.py:901` `_compute_observations` | `obs_buf = cat(iter(·,1), iter(·,16))`, which gives 4052 dims. |
| `ultra/learning/ultra_network_builder.py` | Plain MLP on the full, normalised obs. Extra dims at the end don't need any network changes. |
| `ultra_g1_distill_obj_v2vae.py`, `_v3rl.py` | Subclass `UltraG1Retarget` and do **not** override `_compute_observations` or `_load_motion`. The teacher obs passed to `env.teacherPolicy` are `self.obs_buf`, so the new dims come along automatically. |

---

## 3. Stage 2: record the reward during export

### 3.1 `Humanoid_G1._compute_reward` (`humanoid_g1.py`), small change
Keep the metrics instead of throwing them away, so you can switch quality signals later without re-exporting:

```python
self.rew_buf[:], ig_reset, contact_reset, kinematic_reset, metric_1, metric_2 = compute_humanoid_reward(...)
...
self.rew_buf[:] *= smooth_rewards
self._retarget_metrics = torch.stack((metric_1, metric_2), dim=-1)   # NEW, [N, 2]
```

(Optional: also return and store the tracking-only reward `rb*ro*rig*rcg`, without `energy`/`obj_energy`/`SmoothRewards`. See section 6.)

### 3.2 `UltraG1` (`ultra_g1.py`): change three methods, add one

**`_physics_step`**: when the first frame is captured, also start the reward list:
```python
if not self._export_frames:
    self._export_rewards = []
    self._capture_retarget_frame()
```

**`post_physics_step`**: record the reward after `super()`, which is where `_compute_reward` ran:
```python
def post_physics_step(self):
    super().post_physics_step()
    if self.cfg["env"].get("retargetExportPath"):
        self._capture_retarget_quality()          # NEW
        if bool(self.reset_buf[0]):
            self._save_retarget_rollout()
```

**NEW `_capture_retarget_quality`**:
```python
def _capture_retarget_quality(self):
    q = torch.cat((self.rew_buf[0:1], self._retarget_metrics[0])).detach().cpu()   # [3]
    self._export_rewards.append(q)
```

**`_save_retarget_rollout`**: align and save:
- Frame 0 of `_export_frames` is the initial state, and reward k belongs to the state after step k, which is frame k+1.
- Prepend one copy of the first reward so that `len(rewards) == len(frames)`, then drop the first 30, just like the frames.
```python
quality_path = self.cfg["env"].get("retargetQualityPath")
if quality_path:
    rewards = torch.stack([self._export_rewards[0]] + self._export_rewards)[30:]   # [T, 3]
    assert rewards.shape[0] == frames.shape[0]
    Path(quality_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(rewards, quality_path)
self._export_rewards = []
```
Store `[T, 3]` = (reward, foot err, object err). The teacher picks a column with a config option.

### 3.3 `scripts/export_retarget_smplx.py`
- New argument: `parser.add_argument("--quality-dir", type=Path)`, default `output_dir.parent / f"{output_dir.name}_quality"`.
- Per variant: `quality = quality_dir / destination.name`, and add `"retargetQualityPath": str(quality)` to `env_cfg["env"].update({...})`.
- Skip check: skip only when **both** `destination` and `quality` exist. Otherwise remove a stale `destination` and re-export, because `_save_retarget_rollout` writes both together.
- Success check: `if result.returncode != 0 or not destination.is_file() or not quality.is_file()`.

> Existing exports and the released `OMOMO_retarget_aug` have no quality file. They have to be re-exported with a stage 1 checkpoint, using the same `--xyz` variants.

---

## 4. Stage 3: load and observe in the teacher (`ultra_g1_retarget.py`)

### 4.1 New config keys in `ultra/data/cfg/g1_teacher.yaml` (`env:`)
```yaml
numObs: 4054                 # 4052 + 2 (quality at t+1, t+16); keep 4052 when the feature is off
motionQualityDir: null       # e.g. output/retarget_export_080_quality ; null = off
motionQualityColumn: 0       # 0 = retarget reward, 1 = foot err, 2 = object err
```

### 4.2 `UltraG1Retarget.__init__`: read the keys **before** `super().__init__`
`_load_motion` runs inside `super().__init__` (through `Ultra.__init__`), so the attributes have to exist beforehand:
```python
self._quality_dir = cfg["env"].get("motionQualityDir")
self._quality_col = cfg["env"].get("motionQualityColumn", 0)
```

### 4.3 NEW helper `_load_motion_quality(self, data_path, T)`
```python
def _load_motion_quality(self, data_path, T):
    path = os.path.join(self._quality_dir, os.path.basename(data_path))
    q = torch.load(path).float()
    if q.dim() == 2:
        q = q[:, self._quality_col]
    assert q.shape[0] == T, f"{path}: {q.shape[0]} != {T}"
    return torch.cat([q, q[-1:].repeat(20)])        # same +20 as hoi_data
```

### 4.4 `_load_motion`: change
- In the per-clip loop, **before** appending the 20 frames: `T = hoi_data.shape[0]`, and if `self._quality_dir`, append `self._load_motion_quality(data_path, T)` to a new list `qualities`.
- After the loop, pad like `hoi_data` (`F.pad(q, (0, max_length - q.size(0)))`) and stack: `self.ref_quality = torch.stack(...).to(self.device)` with shape `[num_motions, max_length]`.
- If the feature is off, set `self.ref_quality = None`.

### 4.5 NEW helper `_next_ref_ts(self, env_ids, delta_t)`
`_compute_observations_iter` computes `next_ts` twice, inline. Factor it out so that obs and quality use exactly the same frame:
```python
def _next_ref_ts(self, env_ids, delta_t):
    ts = self.progress_buf[env_ids]
    next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]] - 1)
    return torch.where(self.is_stand_still[env_ids], self.stand_still_frame[env_ids], next_ts)
```
(Optional: use it inside both branches of `_compute_observations_iter` too. The student subclasses override `_compute_observations_iter`, so they wouldn't benefit. That's fine.)

### 4.6 NEW helper `_compute_quality_obs(self, env_ids)`
```python
def _compute_quality_obs(self, env_ids):
    d = self.data_id[env_ids]
    return torch.stack((self.ref_quality[d, self._next_ref_ts(env_ids, 1)],
                        self.ref_quality[d, self._next_ref_ts(env_ids, 16)]), dim=-1)   # [N, 2]
```

### 4.7 `_compute_observations`: change (both branches)
```python
parts = [self._compute_observations_iter(ids, 1), self._compute_observations_iter(ids, 16)]
if self.ref_quality is not None:
    parts.append(self._compute_quality_obs(ids_tensor))   # all envs → torch.arange(num_envs)
self.obs_buf[...] = torch.cat(parts, dim=-1)
```
Appending at the **end** keeps every existing slice valid: `_teacher_task_slice`, `_teacher_ig_slice` and the noise indices. With `motionQualityDir: null` the obs stays 4052-dim and the released `teacher_ultra_inference.pth` keeps working.

### 4.8 Optional ablation: dropout of the quality signal
Add a config key `qualityObsDropProb`. Per reset (in `_reset_hybrid_state_init`, next to `keep_task_obs_mask`), sample `self.keep_quality_mask[env_ids]` and zero the quality dims for masked envs. This makes the teacher robust when the signal isn't available, and you can measure how much it relies on it.

---

## 5. Downstream things that have to follow

| File | Change |
|---|---|
| `ultra/data/cfg/g1_student_vae.yaml`, `g1_student_finetune.yaml` | Only if distilling from the new teacher: `numObs: 4054`, `motionQualityDir: …`. `numObsStudent` doesn't change, because the student doesn't see the quality signal. |
| `ultra/sim2sim_teacher.py`, `ultra/utils/obs.py` | The MuJoCo teacher builds the 4052 obs by hand. A 4054 teacher needs the quality file loaded and the same 2 values appended. Not needed for the training experiment. |
| `ultra/run_teacher_inference.py`, `scripts/play_teacher.sh` | They use the env config, so they work as long as `numObs` and `motionQualityDir` match the checkpoint. |
| `docs/PIPELINE_ABLAUF.md` sections 7 and 8 | Document the new output folder and obs size. |

---

## 6. Design notes for the experiment

- **What the raw reward measures.** `rew_buf` also contains action-energy terms (`energy`, `obj_energy`, `contact_energy`) and the `SmoothRewards` multiplier. These say how smooth the retargeting policy was, not how faithful the trajectory is. If the signal is noisy, try in this order:
  1. the tracking-only product `rb*ro*rig*rcg`,
  2. `metric_1` / `metric_2` (errors in metres, more interpretable),
  3. a combination.
  Saving `[T, 3]` now keeps these options open without another export.
- **Scale.** The reward is in [0, 1]. Obs normalisation (running mean and std) handles the rest. The metrics are in metres; maybe apply `exp(-k·err)` to bound them.
- **Look-ahead.** The quality at t+1 and t+16 matches the two reference windows the teacher already gets. One option is a short window (min over t..t+16), which warns earlier about bad sections.
- **Stand-still envs.** They use `stand_still_frame`, and the helper in 4.5 handles that the same way as the reference obs.
- **Reward shaping (optional, separate experiment).** You could also down-weight the teacher's tracking reward or relax the early-termination thresholds on low-quality frames. Keep this separate from the obs experiment so you can tell the two effects apart.
- **Baseline.** Same data, seed and number of iterations: `motionQualityDir: null` (4052) vs. set (4054). Compare tracking error and the termination rate, both overall and split by quality bins.

---

## 7. Verification

1. **Export one clip.** Run `scripts/export_retarget_smplx.py` with a single-clip input dir. `OUTPUT_quality/<name>.pt` must have the same `T` as `OUTPUT/<name>.pt`. Plot the reward over time and cross-check the dips against `scripts/render_rollout.py`.
2. **Teacher smoke test.** Run `scripts/train_teacher.sh --num_envs 64 --motion_file <export dir> --max_iterations 2` with `numObs: 4054` and `motionQualityDir` set. It must start without shape errors, and `obs_buf[:, -2:]` should look plausible.
3. **Regression.** With `motionQualityDir: null` and `numObs: 4052`, `scripts/play_teacher.sh` must still run with `ultra/weights/teacher_ultra_inference.pth`.
4. **Full runs.** 2048 envs, with the quality obs vs. the baseline.
