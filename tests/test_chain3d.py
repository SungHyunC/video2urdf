import unittest

import numpy as np

from video2urdf.chain3d import RodLine, _unturn, fit_line, fit_rod_joint, rotation


def rod(start, direction, length=0.3, n=200, noise=0.002, rng=None):
    rng = rng or np.random.default_rng(0)
    s = rng.uniform(0, length, n)
    return start + np.outer(s, direction) + rng.normal(0, noise, (n, 3))


class Chain3DTest(unittest.TestCase):
    def test_line_through_noisy_rod_with_outliers(self):
        rng = np.random.default_rng(1)
        d = np.array([1.0, 2.0, 0.5]) / np.linalg.norm([1.0, 2.0, 0.5])
        pts = np.vstack([rod(np.zeros(3), d, rng=rng), rng.uniform(-0.5, 0.5, (40, 3))])  # mixed edge pixels
        line, frac = fit_line(pts, rng)
        self.assertLess(np.rad2deg(np.arccos(abs(line.direction @ d))), 1.0)
        self.assertAlmostEqual(line.hi - line.lo, 0.3, delta=0.03)

    def test_hinge_axis_pivot_and_angle(self):
        axis = np.array([0.2, 0.1, 1.0]) / np.linalg.norm([0.2, 0.1, 1.0])
        pivot = np.array([0.1, -0.05, 0.6])
        d0 = np.cross(axis, [1.0, 0, 0])
        d0 /= np.linalg.norm(d0)
        angles = np.deg2rad(np.linspace(-40, 50, 60))
        lines = {}
        for t, a in enumerate(angles):
            d = rotation(axis, a) @ d0
            lines[t] = RodLine(pivot + 0.15 * d, d, -0.15, 0.15)  # the rod starts at the hinge
        j = fit_rod_joint(lines, ref=20, n=60, near=pivot)
        self.assertGreater(abs(j.axis @ axis), np.cos(np.deg2rad(0.5)))
        off = j.point - pivot
        self.assertLess(np.linalg.norm(off - (off @ axis) * axis), 0.002)
        q = np.sign(j.axis @ axis) * j.q
        np.testing.assert_allclose(q, angles - angles[20], atol=1e-6)

    def test_child_measured_relative_to_parent(self):
        axis, pivot = np.array([0, 0, 1.0]), np.zeros(3)
        parent = fit_rod_joint({t: RodLine(np.array([0.1, 0, 0]) @ rotation(axis, a).T, rotation(axis, a) @ [1.0, 0, 0], -0.1, 0.1)
                                for t, a in enumerate(np.deg2rad(np.linspace(0, 30, 40)))}, 0, 40, pivot)
        # a child that turns with the parent only: no relative motion once unturned
        child = RodLine(np.array([0.3, 0.05, 0]), np.array([0, 1.0, 0]), -0.05, 0.05)
        moved = [RodLine(rotation(axis, a) @ child.center, rotation(axis, a) @ child.direction, -0.05, 0.05)
                 for a in np.deg2rad(np.linspace(0, 30, 40))]
        back = [_unturn(l, parent, t) for t, l in enumerate(moved)]
        for b in back:
            self.assertGreater(abs(b.direction @ child.direction), 0.9999)


if __name__ == "__main__":
    unittest.main()
