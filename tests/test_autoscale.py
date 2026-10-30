import unittest

import numpy as np

from video2urdf.autoscale import card_edge_mm


def card_points(long_dir, short_dir, center, size=(0.0856, 0.054), n=3000, rng=None):
    rng = rng or np.random.default_rng(0)
    u, v = rng.uniform(-0.5, 0.5, (2, n))
    return center + np.outer(u * size[0], long_dir) + np.outer(v * size[1], short_dir)


class AutoScaleTest(unittest.TestCase):
    def test_card_lying_on_a_table_seen_obliquely(self):
        # camera looks along +z; the table tilts 60 deg away; the long edge runs across the view (x)
        tilt = np.deg2rad(60)
        along = np.array([0, np.cos(tilt), np.sin(tilt)])  # away from the camera, in the table
        pts = card_points(np.array([1.0, 0, 0]), along, np.array([0, 0.1, 0.4]))
        e = card_edge_mm(pts, np.random.default_rng(0))
        self.assertAlmostEqual(e["across_mm"], 85.6, delta=1.5)
        self.assertAlmostEqual(e["along_mm"], 54.0, delta=1.5)
        self.assertLess(e["cos_across"], 0.1)
        self.assertGreater(e["cos_along"], 0.5)

    def test_squeezed_depth_only_changes_the_side_along_the_view(self):
        tilt = np.deg2rad(60)
        along = np.array([0, np.cos(tilt), np.sin(tilt)])
        pts = card_points(np.array([1.0, 0, 0]), along, np.array([0, 0.1, 0.4]))
        pts[:, 2] = 0.4 + 0.6 * (pts[:, 2] - 0.4)  # estimated depth compressed toward the card's centre
        e = card_edge_mm(pts, np.random.default_rng(0))
        self.assertAlmostEqual(e["across_mm"], 85.6, delta=1.5)
        self.assertLess(e["along_mm"], 50)


if __name__ == "__main__":
    unittest.main()
