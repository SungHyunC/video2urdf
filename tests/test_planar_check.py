import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from video2urdf.planar_check import compare_planar


class PlanarComparisonTest(unittest.TestCase):
    def test_identical_models_have_identical_sampled_scores(self):
        xml = '''<mujoco><visual><global offwidth="32" offheight="32"/></visual>
          <worldbody><camera name="video" pos="0 -5 0" xyaxes="1 0 0 0 0 1"/>
          <body name="base"><geom type="box" size=".1 .01 .1"/></body></worldbody></mujoco>'''
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for name in ('original', 'candidate'):
                (p / name).mkdir()
                (p / name / 'model.xml').write_text(xml)
            (p / 'original' / 'joints.json').write_text(json.dumps({'joints': []}))
            np.savez(p / 'original' / 'planar_replay.npz', root_poses=np.repeat(np.eye(3)[None], 3, axis=0))
            det = SimpleNamespace(usable=lambda w, h: True, full_mask=lambda h, w: np.ones((h, w), bool))
            tracks = {'base': {i: det for i in range(3)}}
            with patch('video2urdf.planar_check.video_info', return_value=((32, 32), 3, 30)), \
                 patch('video2urdf.planar_check.load_parts', return_value=(['base'], tracks)):
                result = compare_planar(p / 'original', p / 'candidate', p / 'video.mp4', ['base'])
            scores = result['links']['base']
            self.assertEqual(scores['frames'], 3)
            self.assertAlmostEqual(scores['baseline_iou'], scores['inferred_iou'])
            self.assertGreater(scores['baseline_iou'], 0)


if __name__ == '__main__':
    unittest.main()
