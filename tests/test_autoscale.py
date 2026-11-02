import unittest

import numpy as np

from video2urdf.autoscale import card_edge_mm, largest_extent
from video2urdf.lift3d import Intrinsics
from video2urdf.stray import Capture


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


class LargestExtentTest(unittest.TestCase):
    """A 300 x 200 mm flat object 0.5 m in front of the camera, facing it."""

    f, w, h, z = 500.0, 640, 480, 0.5

    def capture(self, frames):
        depth = np.full((frames, self.h, self.w), self.z, np.float32)
        return Capture(None, depth, Intrinsics(self.f, self.w / 2, self.h / 2))

    def mask(self, cx=320, cy=240, long_px=None):
        long_px = long_px or 0.3 * self.f / self.z  # 300 mm -> 300 px
        m = np.zeros((self.h, self.w), bool)
        m[int(cy - 0.2 * self.f / self.z / 2):int(cy + 0.2 * self.f / self.z / 2),
          int(cx - long_px / 2):int(cx + long_px / 2)] = True
        return m

    def test_length_across_the_view(self):
        found = largest_extent(self.capture(3), {0: self.mask(), 1: self.mask(), 2: self.mask()}, log=lambda s: None)
        self.assertIsNotNone(found)
        length, views, depth = found
        self.assertAlmostEqual(length, 0.96 * 296, delta=4)  # 2..98th percentile of the mask eroded by 2 px a side
        self.assertEqual(views, 3)
        self.assertAlmostEqual(depth, self.z, places=3)

    def test_object_cut_off_by_the_frame_is_not_used(self):
        cut = self.mask(cx=60)  # runs off the left edge
        found = largest_extent(self.capture(3), {0: cut, 1: self.mask(), 2: cut}, log=lambda s: None)
        self.assertIsNone(found)  # one usable view is not enough


if __name__ == "__main__":
    unittest.main()
