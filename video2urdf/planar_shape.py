"""Conservative appearance completion for planar articulated stands."""
import copy
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from .planar_check import compare_planar
from .shape import extrude, write_stl
from .texture import read_stl


def complete_stand_base(track_dir: Path, video: Path, appearance: dict) -> dict:
    """Compare a straight-sided base visual; retain collision and original on failure."""
    report = {'active': False, 'inferred_visuals': [],
              'shape': 'Planar silhouette model; cross-view thickness and hidden surfaces are assumed.'}
    if 'stand' not in appearance.get('object', '').lower() or not (track_dir / 'planar_replay.npz').exists():
        return report
    root = ET.parse(track_dir / 'model.xml').getroot()
    body = root.find('./worldbody/body')
    geoms = body.findall('geom') if body is not None else []
    if body is None or len(geoms) != 1 or not geoms[0].get('mesh'):
        return report
    name, mesh_name = body.get('name'), geoms[0].get('mesh')
    asset = root.find(f"./asset/mesh[@name='{mesh_name}']")
    file = asset.get('file', '') if asset is not None else ''
    relative = Path(file)
    if not file.endswith('.stl') or relative.is_absolute() or '..' in relative.parts:
        return report
    pts = read_stl(track_dir / relative).reshape(-1, 3)
    if len(pts) < 3 or not np.isfinite(pts).all():
        return report
    depth = float(np.ptp(pts[:, 1]))
    points = np.unique(pts[:, [0, 2]], axis=0).astype(np.float32)
    if len(points) < 3 or depth <= 0 or not np.allclose([pts[:, 1].min(), pts[:, 1].max()], [-depth / 2, depth / 2], atol=1e-5):
        return report
    outline = cv2.convexHull(points).reshape(-1, 2).astype(float)
    if len(outline) < 3:
        return report
    vertices, faces = extrude(outline, depth)
    new_file = relative.with_name(relative.stem + '_infer.stl')
    with tempfile.TemporaryDirectory(prefix='.planar-shape-', dir=track_dir.parent) as tmp:
        stage = Path(tmp) / 'candidate'
        shutil.copytree(track_dir, stage)
        write_stl(stage / new_file, vertices, faces)
        changed = []
        for filename in ('model.xml', 'model.nophysics.xml'):
            path = stage / filename
            if not path.exists():
                continue
            tree = ET.parse(path)
            model = tree.getroot()
            old_asset = model.find(f"./asset/mesh[@name='{mesh_name}']")
            old_body = next((b for b in model.iter('body') if b.get('name') == name), None)
            old_geom = next((g for g in old_body.findall('geom') if g.get('mesh') == mesh_name), None) if old_body is not None else None
            if old_asset is None or old_geom is None:
                raise ValueError('planar model variants do not match')
            visual_asset = copy.deepcopy(old_asset)
            visual_asset.set('name', mesh_name + '_infer')
            visual_asset.set('file', str(new_file))
            model.find('asset').append(visual_asset)
            visual = copy.deepcopy(old_geom)
            visual.set('mesh', mesh_name + '_infer')
            visual.set('contype', '0'); visual.set('conaffinity', '0'); visual.set('mass', '0')
            old_geom.set('group', '3')
            old_body.append(visual)
            tree.write(path)
            changed.append(filename)
        for filename in ('model.urdf', 'model.nophysics.urdf'):
            path = stage / filename
            if not path.exists():
                continue
            tree = ET.parse(path)
            link = tree.getroot().find(f"./link[@name='{name}']")
            visuals = list(link.findall('visual')) if link is not None else []
            found = False
            for visual in visuals:
                mesh = visual.find('./geometry/mesh')
                if mesh is not None and mesh.get('filename') == file:
                    mesh.set('filename', str(new_file)); found = True
            if not found:
                raise ValueError('planar URDF visual does not match the fitted base')
            tree.write(path, encoding='unicode', xml_declaration=True)
            changed.append(filename)
        comparison = compare_planar(track_dir, stage, video, [name])
        score = comparison['links'][name]
        report['visual_coverage_check'] = comparison
        if (score['frames'] < 10 or score['baseline_iou'] is None or score['inferred_iou'] is None
                or not np.isfinite([score['baseline_iou'], score['inferred_iou']]).all()
                or min(score['baseline_iou'], score['inferred_iou']) < .5
                or score['baseline_iou'] - score['inferred_iou'] > .05):
            report['shape'] += ' Straight-sided base candidate did not pass the sampled mask comparison.'
            return report
        for filename in changed:
            shutil.copy2(stage / filename, track_dir / filename)
        shutil.copy2(stage / new_file, track_dir / new_file)
    report.update(active=True, inferred_visuals=[{'link': name, 'source': file, 'generated': str(new_file),
                                                   'basis': 'convex outline of the fitted planar silhouette'}])
    report['shape'] += ' The base visual uses a simplified convex outline; collision remains fitted.'
    return report
