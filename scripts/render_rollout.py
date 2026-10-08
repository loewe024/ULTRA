"""Replay G1 + object states kinematically in MuJoCo (no physics, no Isaac Sim, no GPU needed).

Accepts
  * a rollout of ultra/run_teacher_inference.py (dict with observed_hoi / reference_hoi / motion_file), or
  * a [T, 630] G1 motion (retargeting export, InterAct/OMOMO_retarget_aug).
Both use the legacy layout: root pos 0:3, root quat (x, y, z, w) 3:7, joints 13:42 in LEGACY_DOF_NAMES order,
object pos 71:74, object quat 74:78, body positions 84:201 (39 bodies). Frames are 60 Hz.

With a rollout, the reference is drawn as translucent spheres (green: G1 bodies, orange: object centre).

Usage (from the repository root):
    python scripts/render_rollout.py output/teacher_inference/rollout.pt --out rollout.mp4
    python scripts/render_rollout.py output/teacher_inference/rollout.pt --view
    python scripts/render_rollout.py InterAct/OMOMO_retarget_aug/sub10_largebox_000_....pt --out ref.mp4
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

FPS = 60
SCALE_RE = re.compile(r"_(\d{3})_(\d{3})_(\d{3})$")


def load(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        return data["observed_hoi"].numpy(), data["reference_hoi"].numpy(), data["motion_file"]
    return data.numpy(), None, path


def object_from_motion(motion_file):
    """``sub10_largebox_000_..._080_080_080.pt`` -> ('largebox', 0.8)."""
    stem = os.path.splitext(os.path.basename(motion_file))[0]
    name = stem.split("_")[1]
    match = SCALE_RE.search(stem)
    scale = int(match.group(1)) / 100.0 if match else 1.0
    return name, scale


def build_model(object_name, object_scale):
    """Merge g1_29dof.xml with the object's MuJoCo XML (meshes in original size, scaled here)."""
    robot_xml = os.path.join(REPO, "ultra/data/assets/g1/g1_29dof.xml")
    object_xml = os.path.join(REPO, f"ultra/data/assets/objects/{object_name}.xml")
    if not os.path.isfile(object_xml):
        raise FileNotFoundError(f"{object_xml} not found (MuJoCo object description, see ultra/data/assets/objects/)")
    robot, obj = ET.parse(robot_xml).getroot(), ET.parse(object_xml).getroot()
    compiler = robot.find("compiler")
    compiler.set("meshdir", os.path.join(os.path.dirname(robot_xml), compiler.get("meshdir", ".")))
    for tag in ("asset", "default", "worldbody"):
        dst = robot.find(tag)
        for child in list(obj.find(tag)):
            if tag == "asset" and child.tag == "mesh":
                child.set("file", os.path.join(os.path.dirname(object_xml), child.get("file")))
                child.set("scale", " ".join([str(object_scale)] * 3))
            dst.append(child)
    # brighter floor and light for recordings
    world = robot.find("worldbody")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    model = mujoco.MjModel.from_xml_string(ET.tostring(robot, encoding="unicode"))
    return model


def qpos_indices(model):
    dof = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in LEGACY_DOF_NAMES]
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    root = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis")]
    obj = [model.jnt_qposadr[j] for j in free if model.jnt_qposadr[j] != root][0]
    return root, np.array(dof), obj


def set_state(data, frame, idx):
    root, dof, obj = idx
    data.qpos[root:root + 3] = frame[0:3]
    data.qpos[root + 3:root + 7] = np.roll(frame[3:7], 1)       # xyzw -> wxyz
    data.qpos[dof] = frame[13:42]
    data.qpos[obj:obj + 3] = frame[71:74]
    data.qpos[obj + 3:obj + 7] = np.roll(frame[74:78], 1)
    data.qvel[:] = 0


def add_reference(scene, ref_frame):
    """Translucent spheres at the reference body positions and the reference object position."""
    def sphere(pos, radius, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                            np.asarray(pos, dtype=np.float64), np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
        scene.ngeom += 1

    for p in ref_frame[84:201].reshape(39, 3):
        sphere(p, 0.03, (0.1, 0.9, 0.2, 0.5))
    sphere(ref_frame[71:74], 0.06, (1.0, 0.5, 0.0, 0.5))


def make_camera(model):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.distance, cam.elevation, cam.azimuth = 3.0, -15.0, 135.0
    return cam


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="rollout.pt or [T, 630] motion file")
    parser.add_argument("--out", help="write an MP4 video to this path")
    parser.add_argument("--view", action="store_true", help="open the interactive MuJoCo viewer")
    parser.add_argument("--no-reference", action="store_true", help="do not draw the reference")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed in the viewer")
    parser.add_argument("--size", type=int, nargs=2, default=(960, 720), metavar=("W", "H"))
    args = parser.parse_args()
    if not args.out and not args.view:
        parser.error("choose --out video.mp4 and/or --view")

    states, reference, motion_file = load(args.path)
    if args.no_reference:
        reference = None
    object_name, object_scale = object_from_motion(motion_file)
    model = build_model(object_name, object_scale)
    data = mujoco.MjData(model)
    idx = qpos_indices(model)
    print(f"{len(states)} frames, object {object_name} x{object_scale}, reference {'yes' if reference is not None else 'no'}")

    if args.out:
        import imageio

        width, height = args.size
        model.vis.global_.offwidth, model.vis.global_.offheight = max(width, 640), max(height, 480)
        renderer = mujoco.Renderer(model, height=height, width=width)
        cam = make_camera(model)
        with imageio.get_writer(args.out, fps=FPS, macro_block_size=1) as writer:
            for t, frame in enumerate(states):
                set_state(data, frame, idx)
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                if reference is not None:
                    add_reference(renderer.scene, reference[t])
                writer.append_data(renderer.render())
        renderer.close()
        print(f"wrote {args.out}")

    if args.view:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.0, -15.0, 135.0
            while viewer.is_running():
                for t, frame in enumerate(states):
                    if not viewer.is_running():
                        break
                    set_state(data, frame, idx)
                    mujoco.mj_forward(model, data)
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        if reference is not None:
                            add_reference(viewer.user_scn, reference[t])
                    viewer.sync()
                    time.sleep(1.0 / (FPS * args.speed))


if __name__ == "__main__":
    main()
