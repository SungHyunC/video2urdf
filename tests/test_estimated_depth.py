"""Estimated-depth (MapAnything) captures: no gravity, focal source, opening and drawer travel
measured from the shut state, and the floor's plane as up for rod chains."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from video2urdf.chain3d import floor_plane
from video2urdf.lidar import Capture, openings_from_shut, slide_from_shut


def stray_folder(root: Path, mapanything: dict | None) -> Path:
    """A one-frame Stray-layout capture; with ``mapanything`` it is an estimated one."""
    (root / "depth").mkdir(parents=True)
    (root / "confidence").mkdir()
    (root / "rgb.mp4").touch()
    np.savetxt(root / "camera_matrix.csv", np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]), delimiter=",")
    cv2.imwrite(str(root / "depth/000000.png"), np.full((4, 4), 500, np.uint16))
    cv2.imwrite(str(root / "confidence/000000.png"), np.full((4, 4), 2, np.uint8))
    # camera-to-world turned 90 deg about x: ARKit's world up (+y) is the camera's -z
    (root / "odometry.csv").write_text("timestamp, frame, x, y, z, qx, qy, qz, qw\n"
                                       "0,000000,0,0,0,0.70710678,0,0,0.70710678\n"
                                       "0.03,000001,0,0,0,0.70710678,0,0,0.70710678\n")
    if mapanything is not None:
        (root / "mapanything.json").write_text(json.dumps(mapanything))
    return root


class EstimatedCaptureTest(unittest.TestCase):
    def capture(self, root: Path) -> Capture:
        c = Capture.__new__(Capture)  # only what up / focal_known read
        c.source, c.ref = root, 0
        c.estimated_depth = (root / "mapanything.json").exists()
        return c

    def test_lidar_capture_keeps_arkit_gravity(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.capture(stray_folder(Path(tmp), None))
            np.testing.assert_allclose(c.up, [0, 0, -1], atol=1e-6)
            self.assertFalse(c.focal_known)

    def test_estimated_capture_has_no_gravity(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.capture(stray_folder(Path(tmp), {"focal_given": True}))
            self.assertIsNone(c.up)
            self.assertTrue(c.focal_known)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(self.capture(stray_folder(Path(tmp), {"focal_given": False})).focal_known)


class ShutZeroTest(unittest.TestCase):
    def test_opening_measured_from_the_shut_frames(self):
        q = np.full(100, np.nan)
        q[:40] = 0.0  # the reference pose (open), q = 0 there
        q[50:90] = np.deg2rad(-130) + np.deg2rad(np.linspace(-2, 2, 40))  # shut, 130 deg away
        out = openings_from_shut(q, set(range(50, 90)), ref=0)
        self.assertEqual(out["shut_frames"], 40)
        self.assertAlmostEqual(out["deg"][0], 130, delta=0.1)  # the reference reads open, positive
        self.assertAlmostEqual(np.median([out["deg"][t] for t in range(50, 90)]), 0, delta=0.1)
        self.assertNotIn(45, out["deg"])  # frames without an angle stay without one

    def test_too_few_shut_frames(self):
        q = np.zeros(30)
        self.assertIsNone(openings_from_shut(q, {1, 2, 3}, ref=0))


class SlideShutTest(unittest.TestCase):
    def test_travel_measured_from_the_shut_reading(self):
        q = np.full(100, np.nan)
        travel = np.concatenate([np.zeros(10), np.linspace(0, 0.25, 30), np.full(20, 0.25), np.linspace(0.25, 0, 30)])
        q[10:100] = 0.165 + 0.6 * travel  # shut reads 165 mm out, the motion squeezed
        q[50] = np.nan
        out, shut = slide_from_shut(q, (10, 99))
        self.assertAlmostEqual(shut, 0.165, places=3)
        self.assertAlmostEqual(float(np.nanmax(out)), 0.15, places=3)  # the squeeze stays: it can't be seen
        self.assertTrue(np.isnan(out[50]) and np.isnan(out[5]))  # unmeasured frames stay unmeasured
        self.assertGreaterEqual(float(np.nanmin(out[10:100])), 0.0)

    def test_a_drawer_that_reads_shut_is_left_alone(self):
        q = np.full(40, np.nan)
        q[5:35] = np.linspace(0, 0.2, 30)
        out, shut = slide_from_shut(q, (5, 34))
        self.assertEqual(shut, 0.0)
        np.testing.assert_array_equal(out[5:35], q[5:35])


class FloorPlaneTest(unittest.TestCase):
    def test_without_gravity_the_floor_normal_points_to_the_camera(self):
        rng = np.random.default_rng(0)
        # a floor 0.6 m below the camera along +y (image down), seen from above
        pts = np.column_stack([rng.uniform(-0.3, 0.3, 500), np.full(500, 0.6), rng.uniform(0.3, 1.0, 500)])
        c = SimpleNamespace(by={"floor": {0: None}}, world=np.eye(4)[None], cloud=lambda part, t: pts)
        n, d, t = floor_plane(c, "floor", 0, None, rng)
        np.testing.assert_allclose(n, [0, -1, 0], atol=1e-6)  # up = toward the camera, image up
        self.assertAlmostEqual(d, -0.6, places=6)
        self.assertEqual(t, 0)


if __name__ == "__main__":
    unittest.main()
