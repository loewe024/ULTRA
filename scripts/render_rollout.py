"""Replay motions kinematically in MuJoCo (no physics, no Isaac Sim, no GPU needed).

Accepts
  * a rollout of ultra/run_teacher_inference.py (dict with observed_hoi / reference_hoi / motion_file):
    G1 + object, the reference as translucent spheres (green: G1 bodies, orange: object centre).
    With --human the source clip of the tracked motion is drawn on top, as for a [T, 630] motion;
  * a [T, 630] G1 motion (retargeting export, InterAct/OMOMO_retarget_aug), 60 Hz.
    With --human <[T, 591] source clip> the human skeleton is drawn on top, interpolated to 60 Hz and scaled
    like the retargeting target (0.8 x the --xyz factors encoded in the file name);
  * a [T, 591] human motion (InterAct/OMOMO_retarget, before retargeting), 30 Hz:
    skeleton of the 52 joints + object; bodies with a contact label are drawn red.

G1 layout: root pos 0:3, root quat (x, y, z, w) 3:7, joints 13:42 in LEGACY_DOF_NAMES order, object pos 71:74,
object quat 74:78, body positions 84:201. Human layout: root 0:3, joint positions 162:318 (52 x 3), object pos
318:321, object quat 321:325, contact labels 331:383.

Usage (from the repository root):
    python scripts/render_rollout.py output/teacher_inference/rollout.pt --out rollout.mp4
    python scripts/render_rollout.py InterAct/OMOMO_retarget/sub10_largebox_003.pt --view
    python scripts/render_rollout.py output/retarget_export_080/sub10_largebox_003_xyz..._080_080_080.pt \\
        --human InterAct/OMOMO_retarget/sub10_largebox_003.pt --out compare.mp4

Folder mode: give a folder of motions and an output folder; one .mp4 per motion is written (existing videos are
skipped unless --overwrite). With --human <folder>, each G1 motion gets the skeleton of its source clip, found by
name (``sub10_largebox_003_xyz..._080_080_080.pt`` -> ``sub10_largebox_003.pt``):
    python scripts/render_rollout.py output/retarget_export_080 --human InterAct/OMOMO_retarget \\
        --out videos/retarget_export_080
Without a display, rendering to a file needs MUJOCO_GL=egl.
"""

import argparse
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "ultra"))

from isaac.legacy_layout import LEGACY_DOF_NAMES  # noqa: E402

SCALE_RE = re.compile(r"_(\d{3})_(\d{3})_(\d{3})$")
XYZ_RE = re.compile(r"_xyz(\d+p\d+)-(\d+p\d+)-(\d+p\d+)")
HUMAN_SCALING = 0.8  # env.scaling of g1_retarget_smplx.yaml

# Parent of each of the 52 human joints (InterMimic / SMPL-X order: legs, spine, left arm + 15 finger joints,
# right arm + 15 finger joints). -1: root.
HUMAN_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16]
for base in (17,):
    for finger in range(5):
        HUMAN_PARENTS += [base, 18 + 3 * finger, 19 + 3 * finger]
HUMAN_PARENTS += [11, 33, 34, 35]
for finger in range(5):
    HUMAN_PARENTS += [36, 37 + 3 * finger, 38 + 3 * finger]
assert len(HUMAN_PARENTS) == 52


def load(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        return ("rollout", data["observed_hoi"].detach().numpy(), data["reference_hoi"].detach().numpy(),
                data["motion_file"])
    data = data.detach().float().numpy()
    kind = {591: "human", 630: "g1"}.get(data.shape[1])
    if kind is None:
        raise ValueError(f"{path}: expected [T, 591] (human) or [T, 630] (G1), got {list(data.shape)}")
    return kind, data, None, path


def object_from_motion(motion_file):
    """``sub10_largebox_000_..._080_080_080.pt`` -> ('largebox', 0.8); no suffix -> original size."""
    stem = os.path.splitext(os.path.basename(motion_file))[0]
    match = SCALE_RE.search(stem)
    return stem.split("_")[1], (int(match.group(1)) / 100.0 if match else 1.0)


def xyz_from_motion(motion_file):
    """Augmentation factors of an export (``..._xyz1p050000-1p000000-0p950000_...``), default 1."""
    match = XYZ_RE.search(os.path.basename(motion_file))
    return np.array([float(g.replace("p", ".")) for g in match.groups()]) if match else np.ones(3)


def build_model(object_name, object_scale, with_robot=True):
    """g1_29dof.xml (floor, light, robot) merged with the object's MuJoCo XML; the robot is removed if not needed."""
    robot_xml = os.path.join(REPO, "ultra/data/assets/g1/g1_29dof.xml")
    object_xml = os.path.join(REPO, f"ultra/data/assets/objects/{object_name}.xml")
    if not os.path.isfile(object_xml):
        raise FileNotFoundError(f"{object_xml} not found (MuJoCo object description, see ultra/data/assets/objects/)")
    root, obj = ET.parse(robot_xml).getroot(), ET.parse(object_xml).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", os.path.join(os.path.dirname(robot_xml), compiler.get("meshdir", ".")))
    world = root.find("worldbody")
    if not with_robot:
        world.remove(world.find("body[@name='pelvis']"))
        for tag in ("actuator", "sensor", "contact", "equality"):
            for element in root.findall(tag):
                root.remove(element)
    for tag in ("asset", "default", "worldbody"):
        dst = root.find(tag)
        for child in list(obj.find(tag)):
            if tag == "asset" and child.tag == "mesh":
                child.set("file", os.path.join(os.path.dirname(object_xml), child.get("file")))
                child.set("scale", " ".join([str(object_scale)] * 3))
            dst.append(child)
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def qpos_indices(model, with_robot):
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not with_robot:
        return None, None, model.jnt_qposadr[free[0]]
    dof = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in LEGACY_DOF_NAMES]
    root = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis")]
    obj = [model.jnt_qposadr[j] for j in free if model.jnt_qposadr[j] != root][0]
    return root, np.array(dof), obj


def set_robot(data, frame, idx):
    root, dof, obj = idx
    data.qpos[root:root + 3] = frame[0:3]
    data.qpos[root + 3:root + 7] = np.roll(frame[3:7], 1)       # xyzw -> wxyz
    data.qpos[dof] = frame[13:42]
    data.qpos[obj:obj + 3] = frame[71:74]
    data.qpos[obj + 3:obj + 7] = np.roll(frame[74:78], 1)
    data.qvel[:] = 0


def set_object(data, pos, quat_xyzw, obj_adr):
    data.qpos[obj_adr:obj_adr + 3] = pos
    data.qpos[obj_adr + 3:obj_adr + 7] = np.roll(quat_xyzw, 1)
    data.qvel[:] = 0


def _sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                        np.asarray(pos, dtype=np.float64), np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def _bone(scene, a, b, width, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, dtype=np.float64),
                         np.asarray(b, dtype=np.float64))
    scene.ngeom += 1


def draw_g1_reference(scene, ref_frame):
    for p in ref_frame[84:201].reshape(39, 3):
        _sphere(scene, p, 0.03, (0.1, 0.9, 0.2, 0.5))
    _sphere(scene, ref_frame[71:74], 0.06, (1.0, 0.5, 0.0, 0.5))


def draw_skeleton(scene, joints, contact, alpha=1.0):
    """joints [52, 3]; contact [52] (1: in contact)."""
    bone_rgba = (0.25, 0.45, 0.95, alpha)
    for j, parent in enumerate(HUMAN_PARENTS):
        if parent >= 0:
            fingers = j >= 18 and not 33 <= j <= 36
            _bone(scene, joints[parent], joints[j], 0.006 if fingers else 0.015, bone_rgba)
    for j, p in enumerate(joints):
        fingers = j >= 18 and not 33 <= j <= 36
        rgba = (0.95, 0.15, 0.1, alpha) if contact[j] > 0.5 else (0.9, 0.9, 0.95, alpha)
        _sphere(scene, p, 0.01 if fingers else 0.025, rgba)


def human_track(raw, scale, to_60hz):
    """Joint positions [T, 52, 3] and contact labels [T, 52]; optionally as the retargeting target at 60 Hz."""
    joints = raw[:, 162:318].reshape(-1, 52, 3)
    contact = np.round(raw[:, 331:383])
    if to_60hz:  # as UltraG1._load_motion: drop frame 0, interpolate 30 -> 60 Hz
        joints, contact = joints[1:], contact[1:]
        t_new = np.arange((len(joints) - 1) * 2 + 1) / 2.0
        idx = np.arange(len(joints))
        joints = np.stack([np.stack([np.interp(t_new, idx, joints[:, j, k]) for k in range(3)], -1)
                           for j in range(52)], 1)
        contact = contact[np.round(t_new).astype(int).clip(max=len(contact) - 1)]
    return joints * scale, contact


def render(path, human, out, view, args):
    """Render one motion to ``out`` and/or show it in the viewer."""
    kind, states, reference, motion_file = load(path)
    if human and kind == "human":
        raise ValueError("--human only works with a [T, 630] G1 motion or a teacher rollout")
    with_robot = kind != "human"
    object_name, object_scale = object_from_motion(motion_file)
    model = build_model(object_name, object_scale, with_robot)
    data = mujoco.MjData(model)
    idx = qpos_indices(model, with_robot)
    fps = 30 if kind == "human" else 60

    overlays = []  # per-frame drawing on top of the MuJoCo scene
    if kind == "rollout" and not args.no_reference:
        overlays.append(lambda scene, t: draw_g1_reference(scene, reference[t]))
    if kind == "human":
        states = states[1:]  # frame 0 of the InterMimic clips is all zeros (ULTRA drops it as well)
        joints, contact = human_track(states, 1.0, to_60hz=False)
        overlays.append(lambda scene, t: draw_skeleton(scene, joints[t], contact[t]))
    elif human:
        raw = torch.load(human, map_location="cpu", weights_only=False).detach().float().numpy()
        joints, contact = human_track(raw, HUMAN_SCALING * xyz_from_motion(motion_file), to_60hz=True)
        if kind == "rollout":
            # rollout step t is recorded after the first step, i.e. at frame t + 1 of the tracked motion
            joints, contact = joints[1:], contact[1:]
        if len(joints) != len(states):
            print(f"note: {len(joints)} human frames vs {len(states)} G1 frames, using the shorter one")
        n = min(len(joints), len(states))
        states = states[:n]
        overlays.append(lambda scene, t: draw_skeleton(scene, joints[t], contact[t], alpha=0.6))
    overlay = (lambda scene, t: [draw(scene, t) for draw in overlays]) if overlays else None  # noqa: E731

    def apply(t):
        if with_robot:
            set_robot(data, states[t], idx)
        else:
            set_object(data, states[t, 318:321], states[t, 321:325], idx[2])
        mujoco.mj_forward(model, data)

    def configure(cam, t):
        if with_robot:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = states[t, 0:3]
        if t == 0:
            cam.distance, cam.elevation, cam.azimuth = 3.0, -15.0, 135.0

    print(f"{len(states)} frames at {fps} Hz ({kind}), object {object_name} x{object_scale}, "
          f"overlay {'yes' if overlay else 'no'}")

    if out:
        import imageio

        width, height = args.size
        model.vis.global_.offwidth, model.vis.global_.offheight = max(width, 640), max(height, 480)
        renderer = mujoco.Renderer(model, height=height, width=width)
        cam = mujoco.MjvCamera()
        with imageio.get_writer(out, fps=fps, macro_block_size=1) as writer:
            for t in range(len(states)):
                apply(t)
                configure(cam, t)
                renderer.update_scene(data, camera=cam)
                if overlay:
                    overlay(renderer.scene, t)
                writer.append_data(renderer.render())
        renderer.close()
        print(f"wrote {out}")

    if view:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            configure(viewer.cam, 0)
            while viewer.is_running():
                for t in range(len(states)):
                    if not viewer.is_running():
                        break
                    apply(t)
                    with viewer.lock():
                        if not with_robot:
                            viewer.cam.lookat[:] = states[t, 0:3]
                        viewer.user_scn.ngeom = 0
                        if overlay:
                            overlay(viewer.user_scn, t)
                    viewer.sync()
                    time.sleep(1.0 / (fps * args.speed))




def source_for(motion_path, human_dir):
    """Source clip of an exported motion: ``sub10_largebox_003_xyz..._080_080_080.pt`` -> ``sub10_largebox_003.pt``.
    For a teacher rollout the motion it tracked is used."""
    data = torch.load(motion_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        motion_path = data["motion_file"]
    stem = os.path.splitext(os.path.basename(motion_path))[0]
    stem = XYZ_RE.split(stem)[0] if XYZ_RE.search(stem) else SCALE_RE.sub("", stem)
    candidate = os.path.join(human_dir, stem + ".pt")
    return candidate if os.path.isfile(candidate) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="rollout.pt, [T, 630] G1 motion, [T, 591] human motion, or a folder of them")
    parser.add_argument("--human", help="[T, 591] source clip, or a folder of source clips, to overlay on G1 motions "
                                        "or teacher rollouts")
    parser.add_argument("--out", help="video file (.mp4); for a folder of motions: output folder, one .mp4 per motion")
    parser.add_argument("--view", action="store_true", help="open the interactive MuJoCo viewer (single motion only)")
    parser.add_argument("--overwrite", action="store_true", help="folder mode: re-render videos that already exist")
    parser.add_argument("--no-reference", action="store_true", help="do not draw the reference of a rollout")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed in the viewer")
    parser.add_argument("--size", type=int, nargs=2, default=(960, 720), metavar=("W", "H"))
    args = parser.parse_args()
    video_endings = (".mp4", ".mov", ".mkv", ".avi", ".gif")

    if os.path.isdir(args.path):
        if args.view or not args.out:
            parser.error("for a folder of motions use --out <output folder> (no --view)")
        if os.path.splitext(args.out)[1].lower() in video_endings:
            parser.error(f"--out {args.out}: for a folder of motions --out must be a folder")
        if args.human and not os.path.isdir(args.human):
            parser.error("--human must be a folder of source clips when the motions are a folder")
        motions = sorted(f for f in os.listdir(args.path) if f.endswith(".pt"))
        if not motions:
            parser.error(f"no .pt files in {args.path}")
        os.makedirs(args.out, exist_ok=True)
        done, skipped, failed, no_source = 0, 0, [], []
        for i, name in enumerate(motions, 1):
            path = os.path.join(args.path, name)
            out = os.path.join(args.out, os.path.splitext(name)[0] + ".mp4")
            if os.path.exists(out) and not args.overwrite:
                skipped += 1
                continue
            human = source_for(path, args.human) if args.human else None
            if args.human and human is None:
                no_source.append(name)
            print(f"[{i}/{len(motions)}] {name}" + (f"  (source {os.path.basename(human)})" if human else ""))
            try:
                render(path, human, out, False, args)
                done += 1
            except Exception as error:  # keep going with the other clips
                failed.append((name, error))
                print(f"    failed: {error}")
        print(f"\n{done} videos written to {args.out}, {skipped} already existed (--overwrite to redo), "
              f"{len(failed)} failed")
        if no_source:
            print(f"{len(no_source)} motions without a matching source clip in {args.human} (rendered without "
                  f"skeleton), e.g. {no_source[0]}")
        for name, error in failed:
            print(f"  {name}: {error}")
        return

    if not args.out and not args.view:
        parser.error("choose --out video.mp4 and/or --view")
    if args.out and os.path.splitext(args.out)[1].lower() not in video_endings:
        parser.error(f"--out {args.out}: use a video file ending (.mp4, .mov, .mkv, .avi or .gif)")
    human = args.human
    if human and os.path.isdir(human):
        human = source_for(args.path, human)
        if human is None:
            parser.error(f"no source clip for {os.path.basename(args.path)} in {args.human}")
    render(args.path, human, args.out, args.view, args)


if __name__ == "__main__":
    main()
