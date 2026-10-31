"""A rod chain whose base can't be measured is anchored to the static scene: the pure pieces."""

import unittest

import numpy as np

from video2urdf.chain3d import (BELOW_M, MIN_BASE_POINTS, RodLine, base_points, chain_reference, drop_to_floor,
                                floor_plane, model_rotation, stand_on_floor, toward_free_end)


class Det:
    def __init__(self, area=1000, box=(0, 0, 10, 10)):
        self.area, self.box = area, box


class FakeCapture:
    """Tracks as {part: {frame: Det}} and a point count (or points) per part and frame."""

    def __init__(self, by, points, n=100):
        self.by, self.points, self.n = by, points, n

    def has(self, part, t):
        return part is not None and t in self.by.get(part, {})

    def cloud(self, part, t):
        p = self.points[part].get(t, 0)
        return np.zeros((p, 3)) if isinstance(p, int) else p


class BasePointsTest(unittest.TestCase):
    def test_nearest_frame_kept_when_it_has_enough(self):
        # the lamp's case: the base is well seen next to the reference, so nothing changes
        c = FakeCapture({"base": {9: Det(10), 0: Det(500)}}, {"base": {9: 40, 0: 900}})
        pts, t, n_near = base_points(c, "base", ref=10)
        self.assertEqual((len(pts), t, n_near), (40, 9, 40))

    def test_best_seen_frame_when_the_reference_cuts_it(self):
        # the phone stand: a sliver at the picture's edge on the reference, whole on frame 0
        c = FakeCapture({"base": {9: Det(10), 0: Det(500), 5: Det(100)}}, {"base": {9: 0, 0: 1031, 5: 60}})
        pts, t, n_near = base_points(c, "base", ref=10)
        self.assertEqual((len(pts), t, n_near), (1031, 0, 0))

    def test_too_few_anywhere_is_reported(self):
        c = FakeCapture({"pole": {t: Det(50) for t in range(20)}}, {"pole": {t: 11 for t in range(20)}})
        pts, t, n_near = base_points(c, "pole", ref=3)
        self.assertLess(len(pts), MIN_BASE_POINTS)

    def test_untracked_base(self):
        pts, t, n_near = base_points(FakeCapture({}, {}), "base", ref=0)
        self.assertEqual((len(pts), t), (0, -1))


class FloorPlaneTest(unittest.TestCase):
    def plane_points(self, normal, d, n=400, seed=0):
        rng = np.random.default_rng(seed)
        normal = normal / np.linalg.norm(normal)
        a = np.cross(normal, [1.0, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(normal, a)
        uv = rng.uniform(-0.5, 0.5, (n, 2))
        return d * normal + uv[:, :1] * a + uv[:, 1:] * b

    def test_nearest_frame_with_depth_and_normal_along_up(self):
        up = np.array([0, -1.0, 0])  # camera y points down
        pts = self.plane_points(np.array([0, 1.0, 0]), 0.8)  # fitted normal may come out either way
        c = FakeCapture({"floor": {3: Det(), 40: Det(), 90: Det()}}, {"floor": {3: pts, 40: 20, 90: pts}})
        n, d, t = floor_plane(c, "floor", ref=50, up=up, rng=np.random.default_rng(0))
        self.assertEqual(t, 90)  # frame 40 is nearer but has no depth
        self.assertGreater(n @ up, 0.999)
        self.assertAlmostEqual(d, -0.8, places=3)

    def test_tilted_surface_is_not_the_floor(self):
        tilted = np.array([0, -1.0, 0.5])
        c = FakeCapture({"floor": {0: Det()}}, {"floor": {0: self.plane_points(tilted, 0.5)}})
        self.assertIsNone(floor_plane(c, "floor", 0, np.array([0, -1.0, 0]), np.random.default_rng(0)))

    def test_no_floor(self):
        self.assertIsNone(floor_plane(FakeCapture({}, {}), None, 0, np.array([0, 0, 1.0]), np.random.default_rng(0)))


class AnchorGeometryTest(unittest.TestCase):
    def test_lower_end_starts_an_anchored_rod(self):
        up = np.array([0.0, 0, 1])
        line = RodLine(np.array([0.2, 0.1, 0.4]), np.array([0, 0.6, -0.8]), -0.2, 0.25)  # drawn pointing down
        out = toward_free_end(line, -BELOW_M * up)
        start, end = out.center + out.lo * out.direction, out.center + out.hi * out.direction
        self.assertLess(start @ up, end @ up)
        self.assertAlmostEqual(out.hi - out.lo, 0.45)
        # and a line already pointing up is kept
        self.assertIs(toward_free_end(out, -BELOW_M * up), out)

    def test_upright_pole_followed_down_to_the_floor(self):
        n, d = np.array([0, 0, 1.0]), 0.02  # floor at z = 0.02
        direction = np.array([np.sin(0.1), 0, np.cos(0.1)])
        low = np.array([0.1, 0.0, 0.0]) + 0.3 * direction  # lowest point seen, 0.3 m up the rod
        low[2] += d
        foot = stand_on_floor(low, direction, n, d)
        self.assertAlmostEqual(foot @ n, d)
        self.assertAlmostEqual(float((low - foot) @ direction), 0.3)
        np.testing.assert_allclose(np.cross(low - foot, direction), 0, atol=1e-12)  # along the rod's own line

    def test_leaning_rod_is_not_extended(self):
        direction = np.array([np.sin(np.deg2rad(50)), 0, np.cos(np.deg2rad(50))])
        self.assertIsNone(stand_on_floor(np.array([0, 0, 0.3]), direction, np.array([0, 0, 1.0]), 0.0))

    def test_estimated_pole_stands_under_its_hinge(self):
        n, d = np.array([0.0, -1.0, 0]), -0.9  # camera coordinates: floor 0.9 m below (y down)
        hinge = np.array([0.1, 0.35, 1.2])  # 0.55 m above it
        foot = drop_to_floor(hinge, n, d)
        self.assertAlmostEqual(foot @ n, d)
        np.testing.assert_allclose(np.cross(hinge - foot, n), 0, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(hinge - foot), 0.55)

    def test_model_axes(self):
        up = np.array([0, -1.0, 0.1])
        rot = model_rotation(up, np.array([1.0, 0.3, 0]), np.array([0, 0, 1.0]))
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rot), 1.0)
        np.testing.assert_allclose(rot[2], up / np.linalg.norm(up))
        self.assertAlmostEqual(rot[0] @ up, 0.0)
        self.assertGreater(rot[0] @ np.array([1.0, 0, 0]), 0.9)
        # a hinge about the vertical: x from the fallback's horizontal part
        rot = model_rotation(np.array([0, 0, 1.0]), np.array([0, 0.1, 1.0]), np.array([0, 1.0, 1.0]))
        np.testing.assert_allclose(rot[0], [0, 1.0, 0], atol=1e-12)
        # both vertical: still a right-handed frame
        rot = model_rotation(np.array([0, 0, 1.0]), np.array([0, 0, 1.0]), np.array([0, 0, -1.0]))
        self.assertAlmostEqual(np.linalg.det(rot), 1.0)


class ChainReferenceTest(unittest.TestCase):
    def test_anchored_chain_needs_no_base_in_view(self):
        rods = {"pole": {t: Det(300, (0, 0, 100, 10)) for t in range(10)},
                "arm": {t: Det(200 + 10 * t, (200, 0, 300, 10)) for t in range(10)}}
        c = FakeCapture({"base": {0: Det()}, **rods}, {}, n=10)
        self.assertEqual(chain_reference(c, "base", ["pole", "arm"]), 0)  # the only frame with the base
        self.assertEqual(chain_reference(c, None, ["pole", "arm"]), 9)  # the rods largest


if __name__ == "__main__":
    unittest.main()
