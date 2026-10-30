import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from video2urdf.static_shape import transform_reference_up, drawer_handle_up
from video2urdf.sam import Detection
from types import SimpleNamespace
from video2urdf.lift3d import Intrinsics
from video2urdf.static_shape import upright_box, observed_track_shift, static_profile, observed_circle
from video2urdf.texture import link_masks
from scripts.export_viewer import joint_tracks
from video2urdf.shape import prism, write_stl
from video2urdf.emit3d import Link3D, Visual, mjcf
from video2urdf.appearance import complete_geometry
import mujoco

class StaticShapeTest(unittest.TestCase):
    def test_handle_upright_uses_floor_for_sign_and_rejects_ambiguity(self):
        w, h = 200, 160
        obj = np.zeros((h, w), bool); obj[20:140, 10:125] = True
        floor = np.zeros((h, w), bool); floor[:, 135:] = True
        def detection(mask):
            return Detection(0, 0, (0, 0, w - 1, h - 1), mask)
        parts = {'cabinet': {0: detection(obj)}, 'floor': {0: detection(floor)}}
        capture = SimpleNamespace(h=h, w=w, by=parts, world=np.eye(4)[None],
                                  cap=SimpleNamespace(depth=np.full((1, h, w), 2., np.float32),
                                                      intrinsics=Intrinsics(f=200., cx=100., cy=80.)))
        capture.has = lambda name, frame: frame in parts.get(name, {})
        handles = []
        for x in (30, 70, 110):
            mask = np.zeros((h, w), bool); mask[70:80, x:x+5] = True
            handles.append(detection(mask))
        up = drawer_handle_up(capture, 'cabinet', 'floor', 0, handles)
        self.assertIsNotNone(up)
        self.assertLess(up[0], -.99)
        # Camera motion must rotate a frame-local hint back into the reference frame.
        capture.world[0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        rotated = drawer_handle_up(capture, 'cabinet', 'floor', 0, handles)
        np.testing.assert_allclose(rotated, [0, 1, 0], atol=.02)
        capture.world[0] = np.eye(4)
        self.assertIsNone(drawer_handle_up(capture, 'cabinet', 'floor', 0, handles[:2]))
        parts['floor'][0] = detection(obj)
        self.assertIsNone(drawer_handle_up(capture, 'cabinet', 'floor', 0, handles))

    def test_reference_up_rotates_with_camera_and_rejects_invalid(self):
        transform = np.eye(4)
        transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        np.testing.assert_allclose(transform_reference_up([2, 0, 0], transform), [0, 1, 0])
        for value in ([0, 0, 0], [float('nan'), 0, 1], [1, 2]):
            with self.assertRaises(ValueError):
                transform_reference_up(value, transform)

    def test_curve_fit_uses_visible_arc_and_rejects_flat_surface(self):
        angle = np.linspace(-1, 1, 200)
        arc = np.column_stack([.06 * np.sin(angle), .06 * np.cos(angle), np.zeros(len(angle))])
        circle = observed_circle(arc, np.eye(3), .11)
        self.assertIsNotNone(circle)
        self.assertAlmostEqual(circle[1], .06, places=6)
        noisy = arc.copy()
        noisy[:, 1] += np.random.default_rng(10).normal(0, .003, len(noisy))
        estimate = observed_circle(noisy, np.eye(3), .11)
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate[1], .06, delta=.006)
        flat = np.column_stack([np.linspace(-.1, .1, 200), np.zeros(200), np.zeros(200)])
        self.assertIsNone(observed_circle(flat, np.eye(3), .2))

    def test_cylinder_profile_is_round_and_uses_observed_width(self):
        profile = static_profile([.10, .12, .3], 'cylinder')
        self.assertEqual(profile.shape, (32, 2))
        np.testing.assert_allclose(np.linalg.norm(profile, axis=1), .06)
        with self.assertRaisesRegex(ValueError, 'box or cylinder'):
            static_profile([.1, .1, .3], 'unknown')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v, f = prism(profile, np.zeros(3), *np.eye(3), -.15, .15)
            write_stl(root / 'cylinder.stl', v, f)
            xml = mjcf('cylinder', [Link3D('base', [Visual('cylinder.stl', (.5, .5, .5))])])
            (root / 'model.xml').write_text(xml)
            model = mujoco.MjModel.from_xml_path(str(root / 'model.xml'))
            self.assertEqual(model.njnt, 0)
            self.assertEqual(model.nmesh, 1)

    def test_shift_checks_object_in_common_frame(self):
        class Capture:
            n = 4
            by = {'object': {i: object() for i in range(4)}}
            def cloud(self, part, frame):
                return np.tile([frame * .03, 0, 0], (100, 1))
        shift, count = observed_track_shift(Capture(), 'object', 0)
        self.assertEqual(count, 4)
        self.assertGreater(shift, .05)

    def test_static_roles_and_viewer_do_not_require_joint_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'static.json').write_text(json.dumps({'part': 'object'}))
            self.assertEqual(link_masks(root), {'base': ['object']})
            self.assertEqual(joint_tracks(root), {})

    def test_box_classification_and_cylinder_fit_require_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'static.json').write_text(json.dumps({'part': 'object', 'shape': 'cylinder'}))
            (root / 'model.xml').write_text('<mujoco><worldbody><body name="base"/></worldbody></mujoco>')
            result = complete_geometry(root, {'category': 'box'}, log=lambda _: None)
            self.assertIn('cylindrical', result['classification_mismatch'])

    def test_rotated_box_recovers_straight_bounds(self):
        rng = np.random.default_rng(42)
        points = rng.uniform(-.5, .5, (20000, 3)) * [0.4, 0.2, 0.1]
        angle = .7
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        result = upright_box(points @ rotation.T + [1, 2, 3], [0, 0, 1])
        np.testing.assert_allclose(result['size'], [.384, .192, .096], atol=.008)
        np.testing.assert_allclose(result['center'], [1, 2, 3], atol=.005)
        np.testing.assert_allclose(result['axes'] @ result['axes'].T, np.eye(3), atol=1e-9)
    def test_incomplete_depth_is_rejected(self):
        points = np.random.default_rng(2).uniform(0, 1, (100, 3))
        points[:, 2] = 0
        with self.assertRaisesRegex(ValueError, 'too thin'):
            upright_box(points, [0, 0, 1])

if __name__ == '__main__': unittest.main()
