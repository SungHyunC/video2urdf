import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from video2urdf.planar_shape import complete_stand_base
from video2urdf.shape import extrude, write_stl


class PlanarShapeTest(unittest.TestCase):
    def test_guard_preserves_collision_and_rejects_worse_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'run'; p.mkdir(); (p / 'meshes').mkdir()
            vertices, faces = extrude(np.array([[0., 0.], [.2, 0.], [.2, .03], [0., .03]]), .1)
            write_stl(p / 'meshes/base.stl', vertices, faces)
            (p / 'model.xml').write_text('<mujoco><asset><mesh name="base_mesh" file="meshes/base.stl"/></asset><worldbody><body name="base"><geom mesh="base_mesh"/></body></worldbody></mujoco>')
            (p / 'model.urdf').write_text('<robot><link name="base"><visual><geometry><mesh filename="meshes/base.stl"/></geometry></visual><collision><geometry><mesh filename="meshes/base.stl"/></geometry></collision></link></robot>')
            (p / 'planar_replay.npz').write_bytes(b'placeholder')
            before = (p / 'model.xml').read_text()
            def scores(value):
                return {'links': {'base': {'frames': 20, 'baseline_iou': .8, 'inferred_iou': value}}}
            with patch('video2urdf.planar_shape.compare_planar', return_value=scores(.6)):
                result = complete_stand_base(p, p / 'video.mp4', {'object': 'articulated stand'})
            self.assertFalse(result['active'])
            self.assertEqual((p / 'model.xml').read_text(), before)
            weak = {'links': {'base': {'frames': 20, 'baseline_iou': .3, 'inferred_iou': .31}}}
            with patch('video2urdf.planar_shape.compare_planar', return_value=weak):
                self.assertFalse(complete_stand_base(p, p / 'video.mp4', {'object': 'articulated stand'})['active'])
            invalid = {'links': {'base': {'frames': 20, 'baseline_iou': .8, 'inferred_iou': float('nan')}}}
            with patch('video2urdf.planar_shape.compare_planar', return_value=invalid):
                self.assertFalse(complete_stand_base(p, p / 'video.mp4', {'object': 'articulated stand'})['active'])
            with patch('video2urdf.planar_shape.compare_planar', return_value=scores(.78)):
                result = complete_stand_base(p, p / 'video.mp4', {'object': 'articulated stand'})
            self.assertTrue(result['active'])
            urdf = (p / 'model.urdf').read_text()
            self.assertIn('base_infer.stl', urdf)
            self.assertIn('base.stl', urdf)


if __name__ == '__main__':
    unittest.main()
