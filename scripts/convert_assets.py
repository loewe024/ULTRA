"""Convert the ULTRA robot and object assets (URDF/OBJ) to USD for Isaac Lab.

The G1 keeps every URDF link as a rigid body (fixed joints are not merged) so the 39-body layout used by the
datasets and policies is preserved. The IsaacGym self-collision filter bits of the leg links
(see ``LEG_COLLISION_FILTERS``) are reproduced with ``UsdPhysics.FilteredPairsAPI``.

Objects are converted per object/scale and per convex-decomposition budget (5 hulls for position-controlled
retargeting, 10 hulls otherwise). Density and scale are applied when spawning, not baked into the USD.

Usage (from the repository root):
    python scripts/convert_assets.py [--force]
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert ULTRA assets to USD.")
parser.add_argument("--force", action="store_true", help="Re-convert even if the USD files already exist.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import glob  # noqa: E402

from pxr import Gf, PhysxSchema, Usd, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg, UrdfConverter, UrdfConverterCfg  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_ROOT = os.path.join(REPO, "ultra", "data", "assets")
USD_ROOT = os.path.join(ASSET_ROOT, "usd")

# IsaacGym shape filter bits of Humanoid_G1._build_env: two shapes do not collide if (filter_a & filter_b) != 0.
LEG_COLLISION_FILTERS = {
    ("right", "ankle"): 2,
    ("right", "knee"): 6,
    ("right", "hip"): 12,
    ("left", "ankle"): 16,
    ("left", "knee"): 48,
    ("left", "hip"): 96,
}
# URDF links without <inertial>; IsaacGym treated them as (almost) massless frames.
MASSLESS_LINK_MASS = 1e-3
MASSLESS_LINK_INERTIA = 1e-6
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


def postprocess_g1(usd_path, urdf_path):
    import xml.etree.ElementTree as ET

    massless = {l.get("name") for l in ET.parse(urdf_path).getroot().findall("link") if l.find("inertial") is None}

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
    for prim in physics_stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            set_collision_offsets(prim)
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            set_decomposition(prim, **G1_DECOMPOSITION)
            num_colliders += 1
    physics_stage.GetRootLayer().Save()

    filters = {name: leg_filter(name) for name in bodies}
    names = sorted(n for n, f in filters.items() if f)
    num_pairs = 0
    for i, a in enumerate(names):
        targets = [bodies[b].GetPath() for b in names[i + 1:] if filters[a] & filters[b]]
        if targets:
            UsdPhysics.FilteredPairsAPI.Apply(bodies[a]).CreateFilteredPairsRel().SetTargets(targets)
            num_pairs += len(targets)
    stage.GetRootLayer().Save()
    print(
        f"[convert_assets] G1: {len(bodies)} bodies, {num_colliders} mesh colliders, {len(massless)} massless links,"
        f" {num_pairs} filtered pairs"
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
        joint_drive=None,
        make_instanceable=False,
        force_usd_conversion=force,
    )
    converter = UrdfConverter(cfg)
    postprocess_g1(converter.usd_path, urdf_path)
    return converter.usd_path


def convert_objects(force):
    paths = []
    for obj_file in sorted(glob.glob(os.path.join(ASSET_ROOT, "objects", "diverse", "*", "*.obj"))):
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
    print("[convert_assets] G1 ->", convert_g1(args.force))
    convert_objects(args.force)


if __name__ == "__main__":
    main()
    simulation_app.close()
