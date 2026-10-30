import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from unittest.mock import patch

from video2urdf.lidar import front_normal, front_travel
from video2urdf.track_checks import conflicting_drawer_tracks, drawer_prompt_warnings
from video2urdf.sam import Detection
from types import SimpleNamespace


class DrawerTrackingTest(unittest.TestCase):
    def test_same_seed_description_prompts_review_not_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = {'0': {'name': 'side_panel', 'prompt': 'Wooden side panel'},
                     '1': {'name': 'drawer_front', 'prompt': 'wooden side panel'}}
            (root / 'parts.json').write_text(json.dumps(parts))
            warning = drawer_prompt_warnings(root, ['side_panel'], ['drawer_front'])
            self.assertEqual(len(warning), 1)
            self.assertIn('masks may still be different', warning[0])
            parts['1']['prompt'] = 'open drawer front'
            (root / 'parts.json').write_text(json.dumps(parts))
            self.assertEqual(drawer_prompt_warnings(root, ['side_panel'], ['drawer_front']), [])

    def test_duplicate_base_and_drawer_track_is_reported(self):
        def det(t, x):
            return Detection(t, 0, (x, 0, x + 7, 7), np.ones((8, 8), dtype=bool))
        moving = {t: det(t, 0) for t in range(10)}
        base = {t: det(t, 0 if t < 4 else 12) for t in range(10)}
        c = SimpleNamespace(by={'base': base, 'drawer': moving}, w=24, h=16)
        self.assertTrue(conflicting_drawer_tracks(c, ['base'], [('drawer', (0, 9))]))
        self.assertFalse(conflicting_drawer_tracks(c, ['base'], [('drawer', (5, 9))]))

    def test_front_normal_tries_other_observed_frames(self):
        class Detection:
            def __init__(self, area): self.area = area
        class Capture:
            n = 2
            by = {'drawer': {0: Detection(1), 1: Detection(2)}}
            def has(self, part, t): return True
            def cloud(self, part, t):
                return np.tile([float(t), 0., 0.] if part == 'drawer' else [float(t), 1., 0.], (250, 1))
        def planes(points, rng):
            return [] if points[0, 0] == 1 else [(np.array([1., 0., 0.]), 0.)]
        with patch('video2urdf.lidar._planes', side_effect=planes):
            result = front_normal(Capture(), 'base', 'drawer', (0, 1), np.array([0., 0., 1.]))
        self.assertIsNotNone(result)
        self.assertEqual(result[1], 0)

    def test_occluded_drawer_is_unknown_not_shut(self):
        class Detection:
            area = 100
        class Capture:
            n = 3
            by = {'drawer': {0: Detection(), 2: Detection()}}
            def has(self, part, t):
                return part == 'base' or t in self.by['drawer']
            def cloud(self, part, t):
                x = (.1 if part == 'drawer' else 0.)
                return np.tile([x, 0., 0.], (250, 1))
        result = front_travel(Capture(), 'base', 'drawer', (0, 2), np.array([1., 0., 0.]))
        np.testing.assert_allclose(result['q'][[0, 2]], [.1, .1])
        self.assertTrue(np.isnan(result['q'][1]))

    def test_missing_observations_report_retracking(self):
        class Capture:
            n = 2
            by = {'drawer': {}}
            def has(self, part, t):
                return part == 'base'
        with self.assertRaisesRegex(ValueError, 'retrack'):
            front_travel(Capture(), 'base', 'drawer', (0, 1), np.array([1., 0., 0.]))


if __name__ == '__main__':
    unittest.main()
