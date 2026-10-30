"""Make a review-only copy with static drawer collision matching its assumed visual."""
import argparse
import json
import math
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from video2urdf.assessment import assess  # noqa: E402
from video2urdf.physics import apply  # noqa: E402


def create_copy(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise ValueError('destination exists; choose a new directory')
    if source == destination or source in destination.parents:
        raise ValueError('destination must be outside source')
    shape = json.loads((source / 'shape_inference.json').read_text())
    if not ((source / 'static.json').exists() and shape.get('active') is True
            and shape.get('category') == 'drawer' and shape.get('static_drawer_depth_assumed_mm')):
        raise ValueError('requires an accepted static drawer with assumed visual depth')
    geometry = source / 'meshes/static_drawer_infer.stl'
    if not geometry.is_file():
        raise ValueError('inferred drawer mesh is missing')
    original = json.loads((source / 'physics.json').read_text())
    info = original.get('links', {}).get('base', {})
    mass, friction = info.get('mass_kg'), info.get('friction')
    if (not isinstance(mass, (int, float)) or not math.isfinite(mass) or mass <= 0
            or not isinstance(friction, (int, float)) or not math.isfinite(friction) or friction < 0):
        raise ValueError('a positive estimated mass and nonnegative friction are required')
    for name in ('model.xml', 'model.nophysics.xml', 'model.urdf', 'model.nophysics.urdf'):
        if not (source / name).is_file():
            raise ValueError(f'missing {name}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.drawer-collision-', dir=destination.parent) as tmp:
        stage = Path(tmp) / 'result'
        shutil.copytree(source, stage)
        shutil.copy2(stage / 'physics.json', stage / 'physics.original.json')
        for name in ('model.xml', 'model.nophysics.xml'):
            path = stage / name; tree = ET.parse(path)
            mesh = tree.getroot().find("./asset/mesh[@name='base_0']")
            if mesh is None or mesh.get('file') != 'meshes/static_object.stl':
                raise ValueError(f'{name} does not have the expected fitted collision mesh')
            mesh.set('file', 'meshes/static_drawer_infer.stl')
            ET.indent(tree, '  '); tree.write(path)
        for name in ('model.urdf', 'model.nophysics.urdf'):
            path = stage / name; tree = ET.parse(path)
            mesh = tree.getroot().find("./link[@name='base']/collision/geometry/mesh")
            if mesh is None or mesh.get('filename') != 'meshes/static_object.stl':
                raise ValueError(f'{name} does not have the expected fitted collision mesh')
            mesh.set('filename', 'meshes/static_drawer_infer.stl')
            ET.indent(tree, '  '); tree.write(path, xml_declaration=True, encoding='unicode')
        physics = apply(stage, {'base': {'material': info.get('material', 'unknown'),
                                          'mass_kg': mass, 'friction': friction}}, {}, 30.)
        result = {'state': 'experiment_only', 'source': str(source),
                  'depth_mm': shape['static_drawer_depth_assumed_mm'],
                  'retained_estimated_mass_kg': mass, 'implied_density_kg_m3': physics['links']['base']['implied_density_kg_m3'],
                  'warning': 'Collision depth is inferred from a visual assumption; retained mass, uniform-density inertia '
                             'and friction are estimates. This is not validated physical reconstruction.'}
        (stage / 'experimental_drawer_collision.json').write_text(json.dumps(result, indent=2) + '\n')
        for name in ('physics_demo.mp4', 'robot_demo.mp4'):
            (stage / name).unlink(missing_ok=True)
        assess(stage)
        stage.rename(destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(create_copy(args.source, args.destination), indent=2))


if __name__ == '__main__':
    main()
