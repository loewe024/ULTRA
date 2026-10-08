#!/usr/bin/env python3
"""
Script to generate MuJoCo XML files from the object meshes ``objects/<name>/<name>.obj``

Usage: python gxml.py [--output-dir DIR]   (default: next to this script)
"""
import argparse
from pathlib import Path
import trimesh
import numpy as np

MESH_SCALE = '1.0 1.0 1.0'
RGBA = '0.7 0.8 0.9 1'

def parse_obj(obj_path, density=10.0):
    """Collect the information for the MuJoCo XML of ``objects/<name>/<name>.obj``"""
    name = obj_path.stem
    mesh_filename = f'objects/{name}/{obj_path.name}'
    mesh_scale = MESH_SCALE
    rgba = RGBA

    # Use IsaacGym friction settings
    # IsaacGym base: friction=0.5, rolling_friction=0.01, torsion_friction=0.5
    # Training uses friction randomization in range [0.5, 2.0], mean=1.25
    # Using 1.0 as a balanced middle ground for sim2sim
    lateral_friction = 0.6
    torsion_friction = 0.01
    rolling_friction = 0.01

    # Load mesh and calculate mass from density
    mesh_path = obj_path

    try:
        # Load the mesh
        mesh_obj = trimesh.load(str(mesh_path), process=False, force='mesh')

        # Parse scale
        scale = [float(s) for s in mesh_scale.split()]
        mesh_obj.apply_scale(scale)

        # Calculate volume and mass
        volume = mesh_obj.volume
        mass = density * volume

        # Get center of mass (COM)
        com = mesh_obj.center_mass

        # Calculate inertia (approximate as uniform density)
        inertia = mesh_obj.moment_inertia
        diag_inertia = np.diag(inertia)

        print(f"  {name}: volume={volume:.6f} m³, mass={mass:.3f} kg, com={com}, inertia={diag_inertia}")

    except Exception as e:
        print(f"  Warning: Could not load mesh {mesh_path}: {e}")
        print(f"  Using default mass=1.6 kg, com=[0,0,0]")
        mass = 1.6
        com = np.array([0.0, 0.0, 0.0])
        diag_inertia = [0.25, 0.25, 0.25]

    return {
        'name': name,
        'mesh_file': mesh_filename,
        'mesh_scale': mesh_scale,
        'rgba': rgba,
        'friction': f"{lateral_friction} {torsion_friction} {rolling_friction}",
        'mass': mass,
        'com': com,
        'inertia': diag_inertia,
    }

def generate_mujoco_xml(obj_info, output_path):
    """Generate MuJoCo XML file from the object info"""
    name = obj_info['name']
    mesh_file = obj_info['mesh_file']
    mesh_scale = obj_info['mesh_scale'].replace(' ', ' ')
    rgba = obj_info['rgba']
    friction = obj_info['friction']
    mass = obj_info['mass']
    com = obj_info['com']
    inertia = obj_info['inertia']

    # Format COM and inertia as strings
    com_str = ' '.join([f'{c:.6f}' for c in com])
    inertia_str = ' '.join([f'{i:.6f}' for i in inertia])

    xml_content = f'''<mujoco model="{name}">
  <!-- Note: compiler and option settings are inherited from robot XML during merge -->

  <asset>
    <mesh name="{name}_mesh" file="{mesh_file}" scale="{mesh_scale}"/>
    <material name="{name}_mat" rgba="{rgba}"/>
  </asset>

  <default>
    <default class="{name}_phys">
      <geom
        contype="1" conaffinity="1"
        friction="{friction}"
        solimp="0.99 0.99 0.01"
        solref="0.01 1"
        density="25"
      />
    </default>
  </default>

  <worldbody>
    <body name="{name}" pos="0 0 0.9" quat="1 0 0 0">
      <freejoint name="{name}_free"/>

      <geom class="{name}_phys"
            name="{name}_geom"
            type="mesh" mesh="{name}_mesh"
            material="{name}_mat"/>

      <!-- MuJoCo will compute mass, COM, and inertia from mesh geometry and density=25 -->
      <!-- Expected values: mass={mass:.3f} kg, com={com_str}, inertia={inertia_str} -->
    </body>
  </worldbody>
</mujoco>
'''

    with open(output_path, 'w') as f:
        f.write(xml_content)

    print(f"Generated: {output_path}")

def main():
    # Get the directory of this script
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description='Generate MuJoCo XML files from the object meshes.')
    parser.add_argument('--output-dir', type=Path, default=script_dir)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # One mesh per object: objects/<name>/<name>.obj
    obj_files = sorted(p for p in script_dir.glob('objects/*/*.obj') if p.stem == p.parent.name)

    print(f"Found {len(obj_files)} object meshes in {script_dir / 'objects'}")

    for obj_file in obj_files:
        try:
            # Parse mesh
            obj_info = parse_obj(obj_file)

            # Generate output path
            output_path = args.output_dir / f"{obj_info['name']}.xml"

            # Generate MuJoCo XML
            generate_mujoco_xml(obj_info, output_path)

        except Exception as e:
            print(f"Error processing {obj_file}: {e}")

    print(f"\nSuccessfully generated XML files for {len(obj_files)} objects")

if __name__ == '__main__':
    main()
