import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from scripts.experimental_laptop_align import align_copy
from video2urdf.shape import prism, write_stl


class LaptopAlignmentTest(unittest.TestCase):
    def test_copy_preserves_original_trajectory_and_removes_stale_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'source', Path(tmp) / 'output'
            meshes = source / 'meshes'
            meshes.mkdir(parents=True)
            rect = np.array([[-.15, .01], [.15, .01], [.15, .2], [-.15, .2]])
            v, f = prism(rect, np.zeros(3), *np.eye(3), -.02, -.005)
            write_stl(meshes / 'base_infer.stl', v, f)
            rect = np.array([[-.15, 0], [.15, 0], [.15, .2], [-.15, .2]])
            v, f = prism(rect, np.zeros(3), np.eye(3)[0], np.eye(3)[2], -np.eye(3)[1], -.003, .003)
            write_stl(meshes / 'lid_infer.stl', v, f)
            write_stl(meshes / 'screen_infer.stl', v, f)
            (source / 'appearance.json').write_text(json.dumps({'category': 'laptop'}))
            (source / 'model.xml').write_text('<mujoco><asset>'
                '<mesh name="base_0" file="meshes/base_infer.stl"/>'
                '<mesh name="lid_0" file="meshes/lid_infer.stl"/>'
                '<mesh name="base_0_infer" file="meshes/base_infer.stl"/>'
                '<mesh name="lid_0_infer" file="meshes/lid_infer.stl"/>'
                '<mesh name="lid_1_infer" file="meshes/screen_infer.stl"/>'
                '</asset><worldbody><body name="base" pos="0 0 .05"><geom mesh="base_0_infer"/>'
                '<body name="lid"><joint name="lid_joint" type="hinge" axis="1 0 0" range="-2.1 .5"/>'
                '<geom mesh="lid_0_infer"/></body></body></worldbody></mujoco>')
            q = np.array([-2., -1., 0.])
            np.savez(source / 'replay3d.npz', q_lid_joint=q, cam_to_world=np.repeat(np.eye(4)[None], 3, axis=0))
            (source / 'replay3d.mp4').write_bytes(b'old')
            result = align_copy(source, output)
            self.assertEqual(result['state'], 'experiment_only')
            self.assertGreater(result['frames_clamped'], 0)
            np.testing.assert_array_equal(np.load(source / 'replay3d.npz')['q_lid_joint'], q)
            np.testing.assert_array_equal(np.load(output / 'replay3d.original.npz')['q_lid_joint'], q)
            self.assertFalse((output / 'replay3d.mp4').exists())
            quality = json.loads((output / 'quality.json').read_text())
            self.assertEqual(quality['status'], 'review')
            self.assertTrue(any('not this experimental alignment' in w for w in quality['warnings']))
            with self.assertRaisesRegex(ValueError, 'already exists'):
                align_copy(source, output)
            root = Path(tmp)
            coherent = root / 'coherent'
            result = align_copy(source, coherent, coherent_collision=True)
            self.assertTrue(result['coherent_collision'])
            hinge = ET.parse(coherent / 'model.xml').getroot().find(".//joint[@name='lid_joint']")
            self.assertEqual(hinge.get('solreflimit'), '0.004000 1')
            self.assertAlmostEqual(result['world_z_shift_m'], -.035, places=5)
            with np.load(coherent / 'replay3d.npz') as data:
                self.assertTrue(np.allclose(data['cam_to_world'][:, 2, 3], -.035, atol=1e-5))
            (source / 'physics.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'without assigned physics'):
                align_copy(source, root / 'refused', coherent_collision=True)
