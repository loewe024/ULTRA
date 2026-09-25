#!/usr/bin/env python3
"""
Script to generate MuJoCo XML files from URDF files
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import trimesh
import numpy as np

def parse_urdf(urdf_path, density=10.0):
    """Parse URDF file and extract necessary information"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Get robot name
    robot_name = root.get('name', 'unknown').replace('.urdf', '')

    # Get mesh filename from visual
    link = root.find('.//link')
    visual = link.find('visual')
    geometry = visual.find('geometry')
    mesh = geometry.find('mesh')
    mesh_filename = mesh.get('filename')
    mesh_scale = mesh.get('scale', '1.0 1.0 1.0')

    # Get material color
    material = visual.find('material')
    color = material.find('color')
    rgba = color.get('rgba', '0.7 0.8 0.9 1')

    # Use IsaacGym friction settings (ignore URDF values)
    # IsaacGym base: friction=0.5, rolling_friction=0.01, torsion_friction=0.5
    # Training uses friction randomization in range [0.5, 2.0], mean=1.25
    # Using 1.0 as a balanced middle ground for sim2sim
    lateral_friction = 0.6
    torsion_friction = 0.01
    rolling_friction = 0.01

    # Load mesh and calculate mass from density
    urdf_dir = urdf_path.parent
    mesh_path = urdf_dir / mesh_filename

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

        print(f"  {robot_name}: volume={volume:.6f} m³, mass={mass:.3f} kg, com={com}, inertia={diag_inertia}")

    except Exception as e:
        print(f"  Warning: Could not load mesh {mesh_path}: {e}")
        print(f"  Using default mass=1.6 kg, com=[0,0,0]")
        mass = 1.6
        com = np.array([0.0, 0.0, 0.0])
        diag_inertia = [0.25, 0.25, 0.25]

    return {
        'name': robot_name,
        'mesh_file': mesh_filename,
        'mesh_scale': mesh_scale,
        'rgba': rgba,
        'friction': f"{lateral_friction} {torsion_friction} {rolling_friction}",
        'mass': mass,
        'com': com,
        'inertia': diag_inertia,
    }

def generate_mujoco_xml(urdf_info, output_path):
    """Generate MuJoCo XML file from URDF info"""
    name = urdf_info['name']
    mesh_file = urdf_info['mesh_file']
    mesh_scale = urdf_info['mesh_scale'].replace(' ', ' ')
    rgba = urdf_info['rgba']
    friction = urdf_info['friction']
    mass = urdf_info['mass']
    com = urdf_info['com']
    inertia = urdf_info['inertia']

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

    # Find all URDF files in the current directory (not in subdirectories)
    urdf_files = list(script_dir.glob('*.urdf'))

    print(f"Found {len(urdf_files)} URDF files in {script_dir}")

    for urdf_file in urdf_files:
        try:
            # Parse URDF
            urdf_info = parse_urdf(urdf_file)

            # Generate output path
            output_path = urdf_file.with_suffix('.xml')

            # Generate MuJoCo XML
            generate_mujoco_xml(urdf_info, output_path)

        except Exception as e:
            print(f"Error processing {urdf_file}: {e}")

    print(f"\nSuccessfully generated XML files for {len(urdf_files)} objects")

if __name__ == '__main__':
    main()