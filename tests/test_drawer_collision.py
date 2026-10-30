import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from scripts.experimental_drawer_collision import create_copy
from video2urdf.appearance import complete_geometry
from video2urdf.emit3d import Link3D, Visual, mjcf, urdf
from video2urdf.physics import apply, link_sizes
from video2urdf.shape import prism, write_stl


class DrawerCollisionTest(unittest.TestCase):
    def test_copy_recomputes_collision_and_retains_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; (source / 'meshes').mkdir(parents=True)
            rect = np.array([[-.2, -.006], [.2, -.006], [.2, .006], [-.2, .006]])
            vertices, faces = prism(rect, np.zeros(3), *np.eye(3), -.3, .3)
            write_stl(source / 'meshes/static_object.stl', vertices, faces)
            links = [Link3D('base', [Visual('meshes/static_object.stl', (.4, .3, .2))])]
            (source / 'model.xml').write_text(mjcf('test', links))
            (source / 'model.urdf').write_text(urdf('test', links))
            (source / 'joints.json').write_text('{"joints": []}')
            (source / 'model3d.json').write_text('{"static_envelope": true}')
            (source / 'static.json').write_text('{"shape": "box", "part": "object"}')
            cams = np.repeat(np.eye(4)[None], 10, axis=0); cams[:, 1, 3] = -1
            np.savez(source / 'replay3d.npz', cam_to_world=cams)
            apply(source, {'base': {'mass_kg': 1., 'material': 'wood', 'friction': .4}}, {}, 30., log=lambda _: None)
            shape = complete_geometry(source, {'category': 'drawer', 'drawer_count': 3}, log=lambda _: None)
            shape['active'] = True
            (source / 'shape_inference.json').write_text(json.dumps(shape))
            result = create_copy(source, root / 'output')
            self.assertEqual(result['state'], 'experiment_only')
            self.assertAlmostEqual(link_sizes(source / 'model.xml')['base']['size_mm'][1], 12., places=1)
            self.assertAlmostEqual(link_sizes(root / 'output/model.xml')['base']['size_mm'][1], 260., places=1)
            self.assertAlmostEqual(float(mujoco.MjModel.from_xml_path(str(root / 'output/model.xml')).body('base').mass[0]), 1.)
            self.assertTrue((root / 'output/physics.original.json').exists())
            with self.assertRaisesRegex(ValueError, 'exists'):
                create_copy(source, root / 'output')
