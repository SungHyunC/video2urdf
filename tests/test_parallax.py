import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from video2urdf.parallax import plane_heights


def project(k, x):
    p = x @ k.T
    return p[:, :2] / p[:, 2:3]


class PlaneHeightsTest(unittest.TestCase):
    """A floor 0.5 m below a camera that looks down at 50 deg, a 12 mm slab on it, and a second
    view 8 cm to the side, turned a little: the slab's height comes back from parallax."""

    def scene(self, focal=1400.0, height=0.012, noise=0.0, seed=0):
        rng = np.random.default_rng(seed)
        k = np.array([[focal, 0, 960], [0, focal, 720], [0, 0, 1.0]])
        # world: floor z = 0, up +z; camera 1 at height 0.5 m looking down 50 deg toward +y
        pitch = np.deg2rad(50)
        look = np.array([0, np.cos(pitch), -np.sin(pitch)])
        right = np.array([1.0, 0, 0])
        down = np.cross(look, right)
        r_w2c = np.stack([right, down, look])  # rows: camera axes in world
        c1 = np.array([0, 0, 0.5])
        floor = np.column_stack([rng.uniform(-0.35, 0.35, 600), rng.uniform(0.1, 0.9, 600), np.zeros(600)])
        slab = np.column_stack([rng.uniform(-0.1, 0.1, 200), rng.uniform(0.35, 0.55, 200), np.full(200, height)])
        turn = Rotation.from_rotvec([0.02, -0.05, 0.03]).as_matrix()
        r2 = turn @ r_w2c
        c2 = c1 + np.array([0.08, -0.02, 0.01])
        pts = np.vstack([floor, slab])
        x1 = project(k, (pts - c1) @ r_w2c.T)
        x2 = project(k, (pts - c2) @ r2.T) + rng.normal(0, noise, (len(pts), 2))
        on_plane = np.r_[np.ones(600, bool), np.zeros(200, bool)]
        hint = r2 @ r_w2c.T  # camera 1 -> camera 2 rotation
        return k, x1, x2, on_plane, hint, c1[2]

    def test_slab_height_from_two_views(self):
        k, x1, x2, on_plane, hint, distance_up = self.scene()
        heights, inliers = plane_heights(x1, x2, on_plane, k, hint)
        # heights are in units of camera 1's distance to the floor (0.5 m along the floor normal)
        self.assertAlmostEqual(float(np.median(heights[~on_plane])) * distance_up, 0.012, delta=0.0005)
        self.assertLess(float(np.abs(heights[on_plane]).max()) * distance_up, 0.0005)
        self.assertGreater(inliers.mean(), 0.99)

    def test_tracking_noise_and_a_wrong_focal(self):
        k, x1, x2, on_plane, hint, distance_up = self.scene(noise=0.3, seed=1)
        heights, _ = plane_heights(x1, x2, on_plane, k, hint)
        self.assertAlmostEqual(float(np.median(heights[~on_plane])) * distance_up, 0.012, delta=0.0015)
        # a focal 40% long (an RGB-only estimate) reads the same parallax far off: the
        # measurement needs a known focal length
        long_k = k.copy()
        long_k[0, 0] = long_k[1, 1] = 1.4 * k[0, 0]
        heights, _ = plane_heights(x1, x2, on_plane, long_k, hint)
        self.assertGreater(abs(float(np.median(heights[~on_plane])) * distance_up - 0.012), 0.004)

    def test_too_few_plane_points(self):
        k, x1, x2, on_plane, hint, _ = self.scene()
        keep = np.r_[np.arange(30), np.arange(600, 800)]
        self.assertIsNone(plane_heights(x1[keep], x2[keep], on_plane[keep], k, hint))


if __name__ == "__main__":
    unittest.main()
