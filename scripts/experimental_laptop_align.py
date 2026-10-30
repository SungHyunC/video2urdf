"""Create a review-only laptop visual alignment experiment from an existing fit.

This does not produce a physics-ready model: the fitted collision meshes and inertia
remain unchanged, while the visual meshes and replay angles are hypothetical.
"""
import argparse
import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from video2urdf.appearance import complete_laptop_details  # noqa: E402
from video2urdf.shape import prism, write_stl  # noqa: E402
from video2urdf.texture import read_stl  # noqa: E402


def align_copy(source: Path, destination: Path, coherent_collision: bool = False) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise ValueError('destination already exists; choose a new directory')
    if source == destination or source in destination.parents:
        raise ValueError('destination must not be inside the source')
    appearance = json.loads((source / 'appearance.json').read_text())
    if appearance.get('category') != 'laptop':
        raise ValueError('requires an existing laptop classification')
    root = ET.parse(source / 'model.xml').getroot()
    files = {m.get('name'): m.get('file') for m in root.findall('./asset/mesh')}
    expected = {'base_0_infer': 'meshes/base_infer.stl', 'lid_0_infer': 'meshes/lid_infer.stl',
                'lid_1_infer': 'meshes/screen_infer.stl'}
    if any(files.get(k) != v for k, v in expected.items()):
        raise ValueError('requires untextured inferred base, lid and screen meshes')
    base = root.find(".//body[@name='base']")
    lid = base.find("body[@name='lid']") if base is not None else None
    joint = lid.find('joint') if lid is not None else None
    axis = np.fromstring(joint.get('axis', '0 0 1'), sep=' ') if joint is not None else np.zeros(3)
    lid_pos = np.fromstring(lid.get('pos', '0 0 0'), sep=' ') if lid is not None else np.ones(3)
    joint_pos = np.fromstring(joint.get('pos', '0 0 0'), sep=' ') if joint is not None else np.ones(3)
    if (joint is None or joint.get('name') != 'lid_joint' or axis.size != 3
            or not np.allclose(axis, [1, 0, 0]) or lid_pos.size != 3 or joint_pos.size != 3
            or not np.allclose(lid_pos, 0) or not np.allclose(joint_pos, 0)
            or any(lid.get(k) for k in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis'))):
        raise ValueError('requires an unrotated lid at the base origin with an x-axis hinge')
    if coherent_collision and ((source / 'physics.json').exists() or any(root.iter('inertial'))):
        raise ValueError('coherent collision experiment requires a fit without assigned physics or inertials')
    mesh_dir = source / 'meshes'
    base_pts = read_stl(mesh_dir / 'base_infer.stl').reshape(-1, 3)
    lid_pts = read_stl(mesh_dir / 'lid_infer.stl').reshape(-1, 3)
    lo, hi = base_pts.min(axis=0), base_pts.max(axis=0)
    if (not np.isfinite(base_pts).all() or not np.isfinite(lid_pts).all()
            or min(hi[0] - lo[0], hi[1] - lo[1]) < .05
            or not 0 <= lo[1] <= .04 or not -.04 <= hi[2] <= .04):
        raise ValueError('base bounds are incompatible with the experimental alignment')
    yz = lid_pts[:, 1:]
    _, sv, axes = np.linalg.svd(yz - yz.mean(axis=0), full_matrices=False)
    if sv[0] <= 1e-6 or sv[0] < 2 * sv[1]:
        raise ValueError('lid direction is ambiguous')
    along = axes[0]
    if along @ yz.mean(axis=0) < 0:
        along = -along
    closed = float(-np.arctan2(along[1], along[0]))
    if not -np.pi < closed < -.2:
        raise ValueError('fitted lid direction is incompatible with this alignment')
    with np.load(source / 'replay3d.npz') as rec:
        data = {k: rec[k].copy() for k in rec.files}
    q = data.get('q_lid_joint')
    if q is None or q.ndim != 1 or not np.isfinite(q).any():
        raise ValueError('no usable lid replay trajectory')
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.laptop-align-', dir=parent) as tmp:
        staged = Path(tmp) / 'result'
        shutil.copytree(source, staged)
        mesh_dir = staged / 'meshes'
        # A visual hypothesis derived from the fitted footprint and hinge, not a measurement.
        lo[1], hi[2] = 0., 0.
        lo[2] = hi[2] - min(.015, .08 * min(hi[0] - lo[0], hi[1] - lo[1]))
        def rectangle(margin):
            return np.array([[lo[0]+margin, lo[1]+margin], [hi[0]-margin, lo[1]+margin],
                             [hi[0]-margin, hi[1]-margin], [lo[0]+margin, hi[1]-margin]])
        v, f = prism(rectangle(0), np.zeros(3), *np.eye(3), lo[2], hi[2])
        write_stl(mesh_dir / 'base_infer.stl', v, f)
        c, s = np.cos(closed), np.sin(closed)
        rotation = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
        margin = min(.012, .08 * min(hi[0] - lo[0], hi[1] - lo[1]))
        for name, inset, z0, z1 in [('lid_infer.stl', 0., .001, .006),
                                     ('screen_infer.stl', margin, .0005, .001)]:
            v, f = prism(rectangle(inset), np.zeros(3), *np.eye(3), z0, z1)
            write_stl(mesh_dir / name, v @ rotation, f)
        shutil.copy2(staged / 'replay3d.npz', staged / 'replay3d.original.npz')
        data['q_lid_joint'] = np.maximum(q, closed)
        shift = 0.
        if coherent_collision:
            for name in ('model.xml', 'model.nophysics.xml'):
                path = staged / name
                if not path.exists():
                    continue
                tree = ET.parse(path)
                model = tree.getroot()
                body = model.find(".//body[@name='base']")
                pos = np.fromstring(body.get('pos', '0 0 0'), sep=' ')
                if pos.size != 3:
                    raise ValueError(f'invalid base position in {name}')
                dz = -lo[2] - pos[2]
                if name == 'model.xml':
                    shift = float(dz)
                elif not np.isclose(dz, shift):
                    raise ValueError('model variants disagree on base position')
                pos[2] += dz
                body.set('pos', ' '.join(f'{v:.9f}' for v in pos))
                hinge = model.find(".//joint[@name='lid_joint']")
                limits = np.fromstring(hinge.get('range', ''), sep=' ')
                if limits.size != 2 or closed >= limits[1]:
                    raise ValueError(f'invalid hinge limits in {name}')
                hinge.set('range', f'{closed:.9f} {limits[1]:.9f}')
                option = model.find('option')
                timestep = float(option.get('timestep', '.002')) if option is not None else .002
                if not np.isfinite(timestep) or timestep <= 0:
                    raise ValueError(f'invalid timestep in {name}')
                # MuJoCo's limits are soft; a shorter stable time constant reduces
                # visible over-rotation. Other URDF simulators have their own solver.
                hinge.set('solreflimit', f'{max(.004, 2 * timestep):.6f} 1')
                for asset, file in [('base_0', 'meshes/base_infer.stl'), ('lid_0', 'meshes/lid_infer.stl')]:
                    mesh = model.find(f"./asset/mesh[@name='{asset}']")
                    if mesh is None:
                        raise ValueError(f'missing {asset} in {name}')
                    mesh.set('file', file)
                ET.indent(tree, '  ')
                tree.write(path)
            for name in ('model.urdf', 'model.nophysics.urdf'):
                path = staged / name
                if not path.exists():
                    continue
                tree = ET.parse(path)
                model = tree.getroot()
                if any(model.iter('inertial')):
                    raise ValueError(f'{name} contains assigned inertial values')
                for link, file in [('base', 'meshes/base_infer.stl'), ('lid', 'meshes/lid_infer.stl')]:
                    mesh = model.find(f"./link[@name='{link}']/collision/geometry/mesh")
                    if mesh is None:
                        raise ValueError(f'missing {link} collision in {name}')
                    mesh.set('filename', file)
                limit = model.find("./joint[@name='lid_joint']/limit")
                if limit is None:
                    raise ValueError(f'missing lid joint limit in {name}')
                limit.set('lower', f'{closed:.9f}')
                ET.indent(tree, '  ')
                tree.write(path, encoding='unicode', xml_declaration=True)
            data['cam_to_world'][:, 2, 3] += shift
        np.savez(staged / 'replay3d.npz', **data)
        complete_laptop_details(staged)
        # Old media and quality evidence describe the unmodified model.
        for name in ('replay3d.mp4', 'replay.mp4', 'turntable.mp4', 'quality.json'):
            (staged / name).unlink(missing_ok=True)
        report = {'state': 'experiment_only', 'closed_angle_deg': float(np.rad2deg(closed)),
                  'frames_clamped': int(np.count_nonzero(np.isfinite(q) & (q < closed))),
                  'source': str(source), 'coherent_collision': coherent_collision, 'world_z_shift_m': shift,
                  'warning': ('Visuals, replay angles, collision and joint limit are hypothetical; mass/inertia and '
                              'contact have not been validated. Do not treat this as verified physics.' if coherent_collision else
                              'Visual and replay angles are hypothetical; collision, inertia and joint limits retain the original fit. '
                              'Do not use this experiment for physics or as a verified export.')}
        (staged / 'experimental_alignment.json').write_text(json.dumps(report, indent=2) + '\n')
        quality = {'status': 'review', 'warnings': [report['warning'],
                   'Stored fit coverage and shape comparisons refer to the original model, not this experimental alignment.'],
                   'note': 'Experimental alignment is not independently validated.'}
        (staged / 'quality.json').write_text(json.dumps(quality, indent=2) + '\n')
        staged.rename(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--coherent-collision', action='store_true',
                        help='experimentally align collision meshes, joint limit and floor; physics remains unverified')
    args = parser.parse_args()
    print(json.dumps(align_copy(args.source, args.destination, args.coherent_collision), indent=2))


if __name__ == '__main__':
    main()
