"""Capture check numbers and path choice (no model calls)."""
import unittest

import cv2
import numpy as np

from video2urdf.capture_check import assess, camera_motion, moving_area
from video2urdf.route import choose_path, parse_look, upright_turn


def texture(h=270, w=480, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (h // 6, w // 6), dtype=np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(img, (3, 3), 0)


def shifted(img, dx, dy, turn=0.0):
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), turn, 1.0)
    m[:, 2] += (dx, dy)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


class CameraMotionTest(unittest.TestCase):
    def test_tripod_is_fixed_even_with_a_moving_part(self):
        bg = texture()
        frames = []
        for k in range(12):
            f = bg.copy()
            cv2.rectangle(f, (200 + 5 * k, 100), (260 + 5 * k, 160), 255, -1)  # a part moving across
            frames.append(f)
        out = camera_motion(frames)
        self.assertEqual(out["camera"], "fixed")
        area = moving_area(frames, out["transforms"])
        self.assertGreater(area["fraction"], 0.005)

    def test_hand_held_drift_is_steady(self):
        bg = texture(seed=1)
        frames = [shifted(bg, 4 * k, 2 * k, 0.3 * k) for k in range(12)]  # drifts ~12% of the frame
        self.assertEqual(camera_motion(frames)["camera"], "steady")

    def test_new_views_are_moving(self):
        frames = [texture(seed=0)]
        for k in range(1, 12):  # the view changes completely half-way through
            frames.append(shifted(texture(seed=0), 3 * k, 0) if k < 4 else texture(seed=100 + k // 3))
        out = camera_motion(frames)
        self.assertIn(out["camera"], ("moving", "unknown"))

    def test_blank_frames_are_unknown(self):
        frames = [np.full((270, 480), 128, np.uint8) for _ in range(6)]
        self.assertEqual(camera_motion(frames)["camera"], "unknown")


class AssessTest(unittest.TestCase):
    def metrics(self, **kw):
        m = {"decodable": True, "frames": 300, "fps": 30.0, "seconds": 10.0, "width": 1080, "height": 1920,
             "exposure": {"mean": 130, "dark_fraction": 0.05, "clipped_fraction": 0.01, "contrast": 50},
             "sharpness": 200, "camera": {"camera": "fixed"}, "moving_area": {"fraction": 0.2}}
        m.update(kw)
        return m

    def test_good_capture_passes_quietly(self):
        a = assess(self.metrics(), has_depth=False)
        self.assertTrue(a["usable"])
        self.assertEqual(a["warnings"], [])

    def test_unusable_inputs(self):
        self.assertFalse(assess(self.metrics(frames=5, seconds=0.2), False)["usable"])
        self.assertFalse(assess(self.metrics(width=100, height=80), False)["usable"])
        black = {"mean": 2, "dark_fraction": 1.0, "clipped_fraction": 0, "contrast": 0.5}
        self.assertFalse(assess(self.metrics(exposure=black), False)["usable"])
        self.assertFalse(assess({"decodable": False}, False)["usable"])

    def test_problems_become_tips(self):
        dark = {"mean": 30, "dark_fraction": 0.6, "clipped_fraction": 0, "contrast": 20}
        a = assess(self.metrics(seconds=2.0, frames=60, fps=8.0, exposure=dark, sharpness=5,
                                moving_area={"fraction": 0.0001}), has_depth=False)
        text = " ".join(a["warnings"])
        for word in ("short", "frame rate", "dark", "blurry", "nothing seems to move"):
            self.assertIn(word, text)
        self.assertEqual(len(a["tips"]), len(a["warnings"]))

    def test_photo_rate_is_not_judged(self):
        a = assess(self.metrics(fps=10.0), has_depth=False, assumed_fps=True)
        self.assertFalse(any("frame rate" in w for w in a["warnings"]))

    def test_look_adds_tips(self):
        look = {"whole_object_in_frame": False, "card_visible": False, "anything_moves": True,
                "lighting": "backlit", "problems": ["keep the hinge visible"]}
        a = assess(self.metrics(), has_depth=False, look=look)
        self.assertTrue(any("leaves the frame" in w for w in a["warnings"]))
        self.assertTrue(any("card" in t for t in a["tips"]))
        self.assertIn("keep the hinge visible", a["tips"])
        self.assertFalse(any("card" in t for t in assess(self.metrics(), True, look=look)["tips"]))  # LiDAR: no card
        look["problems"] = ["no scale card visible - place a card next to object", "open the lid fully"]
        tips = assess(self.metrics(), True, look=look)["tips"]
        self.assertFalse(any("card" in t for t in tips))  # Spark's own card tip is dropped for depth captures
        self.assertIn("open the lid fully", tips)
        tips = assess(self.metrics(), False, look=look)["tips"]  # plain video: our card tip, not Spark's too
        self.assertEqual(sum("card" in t for t in tips), 1)

    def test_lidar_depth_coverage(self):
        a = assess(self.metrics(depth_coverage=0.1), has_depth=True)
        self.assertTrue(any("LiDAR depth" in w for w in a["warnings"]))


class LookTest(unittest.TestCase):
    def test_parse_wrapped_json_and_bad_fields(self):
        text = 'Sure:\n```json\n{"object": "laptop", "up": "RIGHT", "up_confidence": "high", "view": "sideways", ' \
               '"camera": "fixed", "anything_moves": "true", "whole_object_in_frame": false, "problems": ["x"]}\n```'
        look = parse_look(text)
        self.assertEqual(look["object"], "laptop")
        self.assertEqual(look["up"], "right")
        self.assertIsNone(look["view"])  # not one of the allowed views
        self.assertIs(look["anything_moves"], True)
        self.assertIs(look["whole_object_in_frame"], False)
        self.assertIsNone(parse_look("no json here"))
        self.assertIsNone(parse_look('{"broken": '))

    def test_upright_turn(self):
        self.assertEqual(upright_turn({"up": "right", "up_confidence": "high"}), 270)
        self.assertEqual(upright_turn({"up": "left", "up_confidence": "high"}), 90)
        self.assertEqual(upright_turn({"up": "bottom", "up_confidence": "high"}), 180)
        self.assertEqual(upright_turn({"up": "right", "up_confidence": "medium"}), 0)
        self.assertEqual(upright_turn({"up": "top", "up_confidence": "high"}), 0)
        self.assertEqual(upright_turn(None), 0)


class ChoosePathTest(unittest.TestCase):
    def path(self, kind="video", camera="fixed", look=None, forced=None):
        return choose_path(kind, {"camera": camera, "why": "test"}, look, forced)[0]

    def test_flag_wins(self):
        self.assertEqual(self.path(kind="stray", forced="planar"), "planar")
        self.assertEqual(self.path(camera="fixed", look={"view": "side"}, forced="3d"), "3d")

    def test_depth_inputs_take_the_3d_path(self):
        for kind in ("stray", "mapanything", "rgbd"):
            self.assertEqual(self.path(kind=kind, camera="fixed", look={"view": "side"}), "3d")

    def test_plain_video(self):
        self.assertEqual(self.path(camera="fixed", look={"view": "side"}), "planar")
        self.assertEqual(self.path(camera="steady", look={"view": "side", "camera": "shaky"}), "planar")
        self.assertEqual(self.path(camera="fixed", look={"view": "oblique"}), "3d")
        self.assertEqual(self.path(camera="fixed", look={"view": "toward"}), "3d")
        self.assertEqual(self.path(camera="moving", look={"view": "side"}), "3d")
        self.assertEqual(self.path(camera="unsteady", look={"view": "side", "camera": "moving"}), "3d")
        self.assertEqual(self.path(camera="fixed", look={"view": "side", "anything_moves": False}), "3d")

    def test_without_a_look(self):
        self.assertEqual(self.path(camera="fixed"), "planar")
        self.assertEqual(self.path(camera="unsteady"), "planar")
        self.assertEqual(self.path(camera="unknown"), "3d")
        self.assertEqual(self.path(camera="moving"), "3d")


if __name__ == "__main__":
    unittest.main()
