"""Prepare the ULTRA robot and object assets for Isaac Lab: scaled object meshes, then URDF/OBJ -> USD.

1. Scaled object meshes: every ``objects/objects/<name>/<name>.obj`` (original size, as written by InterAct) is
   scaled per ``--object-scales`` tag ``SSS_SSS_SSS`` (x, y, z in percent, scaled about the mesh origin) to
   ``objects/diverse/<name>_<tag>/<name>_<tag>.obj``. The motion file names carry the same tag
   (``sub10_largebox_000_080_080_080.pt``). Existing meshes are kept unless ``--force`` is given.
2. USD conversion. The G1 keeps every URDF link as a rigid body (fixed joints are not merged) so the 39-body layout
   used by the datasets and policies is preserved. The IsaacGym self-collision filter bits of the leg links
   (see ``LEG_COLLISION_FILTERS``) are reproduced with ``UsdPhysics.FilteredPairsAPI``. Objects are converted per
   object/scale and per convex-decomposition budget (5 hulls for position-controlled retargeting, 10 hulls
   otherwise). Density and scale are applied when spawning, not baked into the USD.

Usage (from the repository root):
    python scripts/convert_assets.py [--force] [--object-scales 080_080_080 100_100_100] [--objects-only]
"""

import argparse
import glob
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_ROOT = os.path.join(REPO, "ultra", "data", "assets")
USD_ROOT = os.path.join(ASSET_ROOT, "usd")
# Original-size meshes (InterAct layout <name>/<name>.obj) and the scaled variants the tasks load.
OBJECT_SOURCE = os.path.join(ASSET_ROOT, "objects", "objects")
OBJECT_DIVERSE = os.path.join(ASSET_ROOT, "objects", "diverse")
# Asset scales of the released datasets (``--asset-scale`` of prepare_retarget_smplx.py / export_retarget_smplx.py).
DEFAULT_OBJECT_SCALES = ("080_080_080", "100_100_100")
SCALE_TAG = re.compile(r"^(\d{3})_(\d{3})_(\d{3})$")


def scale_factors(tag):
    """``"080_080_080"`` -> ``(0.8, 0.8, 0.8)``."""
    match = SCALE_TAG.fullmatch(tag)
    if match is None:
        raise ValueError(f"invalid object scale {tag!r}, expected three 3-digit percentages like 080_080_080")
    return tuple(int(g) / 100.0 for g in match.groups())


def prepare_scaled_objects(source_dir, scales, force):
    """Write ``diverse/<name>_<tag>/<name>_<tag>.obj`` for every source mesh and scale tag; returns the new files.

    Loaded without processing and exported with trimesh, which reproduces the released meshes byte for byte.
    """
    import trimesh

    sources = sorted(p for p in glob.glob(os.path.join(source_dir, "*", "*.obj"))
                     if os.path.splitext(os.path.basename(p))[0] == os.path.basename(os.path.dirname(p)))
    if not sources:
        raise FileNotFoundError(f"no object meshes <name>/<name>.obj in {source_dir}")
    written = []
    for src in sources:
        name = os.path.splitext(os.path.basename(src))[0]
        for tag in scales:
            dst = os.path.join(OBJECT_DIVERSE, f"{name}_{tag}", f"{name}_{tag}.obj")
            if os.path.isfile(dst) and not force:
                continue
            mesh = trimesh.load(src, process=False, force="mesh")
            factors = scale_factors(tag)
            mesh.apply_scale(factors[0] if len(set(factors)) == 1 else factors)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            mesh.export(dst)
            written.append(dst)
    print(f"[convert_assets] scaled objects: {len(sources)} meshes x {len(scales)} scales, {len(written)} written")
    return written


parser = argparse.ArgumentParser(description="Prepare the ULTRA assets (scaled object meshes, USD) for Isaac Lab.")
parser.add_argument("--force", action="store_true", help="Re-create the scaled meshes and re-convert the USD files.")
parser.add_argument("--object-scales", nargs="+", default=list(DEFAULT_OBJECT_SCALES), metavar="SSS_SSS_SSS",
                    help="object scales to create (x, y, z in percent); default: %(default)s")
parser.add_argument("--object-source", default=OBJECT_SOURCE,
                    help="directory with the original-size meshes <name>/<name>.obj; default: %(default)s")
parser.add_argument("--objects-only", action="store_true",
                    help="only create the scaled object meshes, without starting Isaac Sim")
args, _ = parser.parse_known_args()
for _tag in args.object_scales:
    scale_factors(_tag)
if args.objects_only:
    prepare_scaled_objects(args.object_source, args.object_scales, args.force)
    raise SystemExit(0)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg, UrdfConverter, UrdfConverterCfg  # noqa: E402

# IsaacGym shape filter bits of Humanoid_G1._build_env: two shapes do not collide if (filter_a & filter_b) != 0.
LEG_COLLISION_FILTERS = {
    ("right", "ankle"): 2,
    ("right", "knee"): 6,
    ("right", "hip"): 12,
    ("left", "ankle"): 16,
    ("left", "knee"): 48,
    ("left", "hip"): 96,
}
# Additional pairs that must not collide. The convex hulls of the pelvis cover overlap the space the thigh mounts
# sweep through when the hips flex: without this filter, hip pitch stops at about -0.35 rad (the motions need about
# -1.6 rad to bend down to an object), and the released teacher falls over on every pick-up motion.
EXTRA_FILTERED_PAIRS = {"pelvis_contour_link": ("left_hip_pitch_link", "right_hip_pitch_link")}
# URDF links without <inertial>; IsaacGym treated them as (almost) massless frames.
MASSLESS_LINK_MASS = 1e-3
MASSLESS_LINK_INERTIA = 1e-6
# Revolute joints of g1_29dof.urdf; every one must get a force drive.
G1_NUM_DOFS = 29
# IsaacGym VHACD settings of the G1 (Humanoid_G1._create_envs).
G1_DECOMPOSITION = dict(max_convex_hulls=5, hull_vertex_limit=16, voxel_resolution=60000)
# IsaacGym VHACD settings of the objects (Ultra._load_target_asset); keyed by the hull budget.
OBJECT_DECOMPOSITION = {n: dict(max_convex_hulls=n, hull_vertex_limit=64, voxel_resolution=300000) for n in (5, 10)}
# Collision offsets of the ``sim.physx`` block of all ULTRA task configs. They are baked into the USD because the
# G1 colliders are instanced and cannot be modified when spawning.
CONTACT_OFFSET = 0.02
REST_OFFSET = 0.0
# Default object density (``env.objectDensity``); the environment re-applies the configured value when spawning.
OBJECT_DENSITY = 25.0


def leg_filter(link_name):
    for (side, part), bits in LEG_COLLISION_FILTERS.items():
        if side in link_name and part in link_name:
            return bits
    return 0


def set_decomposition(prim, max_convex_hulls, hull_vertex_limit, voxel_resolution):
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexDecomposition")
    api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
    api.CreateMaxConvexHullsAttr().Set(max_convex_hulls)
    api.CreateHullVertexLimitAttr().Set(hull_vertex_limit)
    api.CreateVoxelResolutionAttr().Set(voxel_resolution)


def set_collision_offsets(prim):
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    api.CreateContactOffsetAttr().Set(CONTACT_OFFSET)
    api.CreateRestOffsetAttr().Set(REST_OFFSET)


def remove_empty_visual_refs(usd_path, links_without_visuals):
    """Clear the ``visuals`` references the importer writes for links without any visual geometry.

    They point to ``/visuals/<link>``, which is never created, and USD warns once per link and environment
    ("Unresolved reference prim path") whenever the scene is composed.
    """
    stem = os.path.splitext(os.path.basename(usd_path))[0]
    base_layer = Sdf.Layer.FindOrOpen(os.path.join(os.path.dirname(usd_path), "configuration", f"{stem}_base.usd"))
    cleared = []
    for root in base_layer.rootPrims:
        for name in links_without_visuals:
            spec = base_layer.GetPrimAtPath(root.path.AppendChild(name).AppendChild("visuals"))
            if spec is not None and spec.hasReferences:
                spec.referenceList.ClearEdits()
                cleared.append(name)
    if cleared:
        base_layer.Save()
    return sorted(cleared)


def postprocess_g1(usd_path, urdf_path):
    import xml.etree.ElementTree as ET

    links = ET.parse(urdf_path).getroot().findall("link")
    massless = {l.get("name") for l in links if l.find("inertial") is None}
    cleared = remove_empty_visual_refs(usd_path, [l.get("name") for l in links if l.find("visual") is None])

    stage = Usd.Stage.Open(usd_path)
    bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    missing = massless - bodies.keys()
    if missing:
        raise RuntimeError(f"Converted G1 is missing links {sorted(missing)}; fixed joints were merged?")

    for name in massless:
        mass_api = UsdPhysics.MassAPI.Apply(bodies[name])
        mass_api.CreateMassAttr().Set(MASSLESS_LINK_MASS)
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(MASSLESS_LINK_INERTIA))

    # The Isaac Sim 5.1 importer keeps the colliders as instanced prototypes in the physics layer.
    stem = os.path.splitext(os.path.basename(usd_path))[0]
    physics_path = os.path.join(os.path.dirname(usd_path), "configuration", f"{stem}_physics.usd")
    physics_stage = Usd.Stage.Open(physics_path)
    num_colliders = 0
    drive_types = {}
    for prim in physics_stage.Traverse():
        drive_type = prim.GetAttribute("drive:angular:physics:type")
        if drive_type and drive_type.HasAuthoredValue():
            drive_types[prim.GetName()] = drive_type.Get()
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            set_collision_offsets(prim)
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            set_decomposition(prim, **G1_DECOMPOSITION)
            num_colliders += 1
    physics_stage.GetRootLayer().Save()
    wrong = sorted(n for n, t in drive_types.items() if t != "force")
    if len(drive_types) != G1_NUM_DOFS or wrong:
        raise RuntimeError(
            f"Converted G1 has {len(drive_types)} angular drives (expected {G1_NUM_DOFS}), non-force drives: {wrong}"
        )

    filters = {name: leg_filter(name) for name in bodies}
    names = sorted(n for n, f in filters.items() if f)
    num_pairs = 0
    for i, a in enumerate(names):
        targets = [bodies[b].GetPath() for b in names[i + 1:] if filters[a] & filters[b]]
        if targets:
            UsdPhysics.FilteredPairsAPI.Apply(bodies[a]).CreateFilteredPairsRel().SetTargets(targets)
            num_pairs += len(targets)
    for a, others in EXTRA_FILTERED_PAIRS.items():
        rel = UsdPhysics.FilteredPairsAPI.Apply(bodies[a]).CreateFilteredPairsRel()
        rel.SetTargets(sorted(set(rel.GetTargets()) | {bodies[b].GetPath() for b in others}))
        num_pairs += len(others)
    stage.GetRootLayer().Save()
    print(
        f"[convert_assets] G1: {len(bodies)} bodies, {num_colliders} mesh colliders, {len(massless)} massless links,"
        f" {num_pairs} filtered pairs, {len(drive_types)} force drives,"
        f" empty visual references cleared: {cleared or 'none'}"
    )


def convert_g1(force):
    urdf_path = os.path.join(ASSET_ROOT, "g1", "g1_29dof.urdf")
    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.join(USD_ROOT, "g1"),
        usd_file_name="g1_29dof.usd",
        fix_base=False,
        merge_fixed_joints=False,
        collider_type="convex_decomposition",
        self_collision=True,
        # Force drives as in IsaacGym (DOF_MODE_POS): stiffness/damping in Nm/rad and Nm*s/rad. Without a
        # joint_drive the importer writes acceleration drives, which scale the gains by the joint inertia.
        # The gains themselves are set when spawning (ImplicitActuatorCfg in isaac/scene_cfg.py).
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        make_instanceable=False,
        force_usd_conversion=force,
    )
    converter = UrdfConverter(cfg)
    postprocess_g1(converter.usd_path, urdf_path)
    return converter.usd_path


def convert_objects(force):
    paths = []
    for obj_file in sorted(glob.glob(os.path.join(OBJECT_DIVERSE, "*", "*.obj"))):
        name = os.path.splitext(os.path.basename(obj_file))[0]
        for hulls, decomposition in OBJECT_DECOMPOSITION.items():
            cfg = MeshConverterCfg(
                asset_path=obj_file,
                usd_dir=os.path.join(USD_ROOT, "objects", name),
                usd_file_name=f"{name}_h{hulls}.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(density=OBJECT_DENSITY),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, contact_offset=CONTACT_OFFSET, rest_offset=REST_OFFSET
                ),
                mesh_collision_props=sim_utils.ConvexDecompositionPropertiesCfg(**decomposition),
                make_instanceable=False,
                force_usd_conversion=force,
            )
            paths.append(MeshConverter(cfg).usd_path)
    print(f"[convert_assets] objects: {len(paths)} USD files")
    return paths


def main():
    prepare_scaled_objects(args.object_source, args.object_scales, args.force)
    print("[convert_assets] G1 ->", convert_g1(args.force))
    convert_objects(args.force)


if __name__ == "__main__":
    main()
    simulation_app.close()
