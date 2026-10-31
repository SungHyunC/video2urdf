import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from video2urdf import completion as C
from video2urdf.lift3d import Intrinsics
from video2urdf.shape import prism, write_stl

FRONT = C.Front(np.zeros(3), np.array([0.0, -1.0, 0.0]))  # front faces -y: u = +x, depth v = +y
BOX = ([0.0, 0.0, 0.0], [0.4, 0.3, 0.25])  # a 400 x 300 x 250 mm body (u, v, z)
INTR = Intrinsics(800.0, 480.0, 360.0)
SIZE = (960, 720)


def camera(pos, look_at) -> np.ndarray:
    """OpenCV camera (x right, y down, z forward) -> model frame."""
    f = np.asarray(look_at, float) - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    m = np.eye(4)
    m[:3, :3] = np.stack([r, d, f], axis=1)
    m[:3, 3] = pos
    return m


def projector(c2m):
    return C.FrameView(0, c2m, {}).project


def bounds_from(c2m, edge_px=3):
    project = projector(c2m)
    mask = C.silhouette(project(INTR, C.box_corners(FRONT, *BOX)), SIZE)
    depths = np.arange(0, 1.0, 0.01)
    return C.frame_depth_bounds(lambda d: C.box_corners(FRONT, [0, 0, 0], [0.4, d, 0.25]),
                                lambda p: project(INTR, p), mask, depths, edge_px)


class DepthBoundTest(unittest.TestCase):
    def test_a_camera_above_the_top_brackets_the_true_depth(self):
        b = bounds_from(camera(np.array([0.25, -0.6, 0.55]), [0.2, 0.15, 0.1]))
        self.assertIsNotNone(b["upper_m"])
        self.assertLessEqual(b["lower_m"], 0.30 + 1e-9)
        self.assertGreaterEqual(b["lower_m"], 0.25)
        self.assertGreaterEqual(b["upper_m"], 0.30 - 1e-9)
        self.assertLessEqual(b["upper_m"], 0.36)

    def test_a_camera_that_never_sees_past_the_front_gives_no_bound(self):
        # level with the middle of the front, straight ahead: no top or side face shows
        b = bounds_from(camera(np.array([0.2, -0.8, 0.125]), [0.2, 0.0, 0.125]))
        self.assertIsNone(b["upper_m"])

    def test_frames_that_disagree_keep_only_the_upper_bound(self):
        per = {1: {"upper_m": 0.44, "lower_m": 0.33}, 2: {"upper_m": 1.0, "lower_m": 0.64}}
        out = C.combine_bounds(per, seen_depth=0.02)
        self.assertAlmostEqual(out["upper_m"], 0.44)
        self.assertEqual(out["upper_frames"], [1])
        self.assertFalse(out["frames_agree"])
        self.assertAlmostEqual(out["lower_m"], 0.02)  # the depth points' least depth, not the silhouettes'

    def test_most_frames_agreeing_give_a_lower_bound_and_name_the_outlier(self):
        per = {5: {"upper_m": 0.34, "lower_m": 0.26}, 1126: {"upper_m": 0.44, "lower_m": 0.33},
               1151: {"upper_m": 0.39, "lower_m": 0.30}, 1161: {"upper_m": 0.32, "lower_m": 0.25},
               1176: {"upper_m": 0.39, "lower_m": 0.29}, 1261: {"upper_m": 0.33, "lower_m": 0.26}}
        out = C.combine_bounds(per)
        self.assertAlmostEqual(out["upper_m"], 0.32)
        self.assertTrue(out["frames_agree"])
        self.assertAlmostEqual(out["lower_m"], 0.30)
        self.assertEqual(out["lower_frames"], [1151])
        self.assertEqual(out["outside_consensus"], [1126])

    def test_frames_that_agree_bracket_the_depth(self):
        per = {1: {"upper_m": 0.44, "lower_m": 0.33}, 2: {"upper_m": 0.5, "lower_m": 0.36}}
        out = C.combine_bounds(per, seen_depth=None)
        self.assertAlmostEqual(out["upper_m"], 0.44)
        self.assertAlmostEqual(out["lower_m"], 0.36)

    def test_a_mask_without_the_top_is_dropped(self):
        per = {1: {"upper_m": 0.0, "lower_m": 0.0}, 2: {"upper_m": 0.5, "lower_m": 0.2}}
        out = C.combine_bounds(per)
        self.assertIn("1", out["dropped_frames"])
        self.assertAlmostEqual(out["upper_m"], 0.5)


class LevellingTest(unittest.TestCase):
    def test_a_leaning_front_is_turned_onto_the_pooled_plane(self):
        rng = np.random.default_rng(0)
        uz = rng.uniform([0, 0], [0.4, 0.25], (2000, 2))
        tilt = np.deg2rad(5)  # the frame's front leans 5 deg and sits 12 mm behind the pooled plane
        pts = np.stack([uz[:, 0], 0.012 + uz[:, 1] * np.sin(tilt), uz[:, 1] * np.cos(tilt)], axis=1)
        m, info = C.front_correction(pts, FRONT, rng)
        self.assertTrue(info["aligned"])
        v = FRONT.uvz(C._apply(m, pts))[:, 1]
        self.assertLess(np.abs(v).max(), 1e-3)
        self.assertAlmostEqual(info["turn_deg"], 5.0, places=1)

    def test_support_roll_levels_side_to_side_and_keeps_the_front(self):
        rng = np.random.default_rng(0)
        xy = rng.uniform([0, -0.3], [0.4, -0.02], (2000, 2))
        roll = np.deg2rad(3)  # the support tilts along the front (about its normal), 8 mm high
        pts = np.stack([xy[:, 0], xy[:, 1], 0.008 + xy[:, 0] * np.tan(roll)], axis=1)
        m, info = C.support_roll(pts, FRONT, 0.0, rng)
        out = C._apply(m, pts)
        self.assertLess(np.ptp(out[:, 2]), 1e-3)
        self.assertAlmostEqual(float(np.median(out[:, 2])), 0.0, places=3)
        np.testing.assert_allclose(m[:3, :3] @ FRONT.n, FRONT.n, atol=1e-9)  # turned about the front's normal


class ShutStateTest(unittest.TestCase):
    def test_shut_is_the_end_away_from_the_open_frames(self):
        q = np.deg2rad(np.r_[np.full(10, 12.0), np.linspace(10, -40, 10), np.full(10, 11.0)])
        open_frames = range(12, 20)
        q_shut, shut = C.shut_state(q, open_frames)
        self.assertAlmostEqual(np.rad2deg(q_shut), 12.0, delta=0.6)
        self.assertTrue(set(range(10)) <= set(shut.tolist()))
        self.assertFalse(set(shut.tolist()) & set(open_frames))


GEOMETRY = {"width_m": 0.48, "height_m": 0.27, "pocket_u": [-0.005, 0.40], "pocket_z": [0.01, 0.27], "pocket_v": 0.015}


class ResolveDimsTest(unittest.TestCase):
    def test_spark_depth_is_clamped_to_the_video_bound_and_labelled(self):
        answer = {"object": "microwave oven", "outer_depth_mm": {"value": 500, "range": [450, 520], "reason": "typical"},
                  "cavity_width_mm": {"value": 330, "reason": "23 L"}, "cavity_height_mm": {"value": 210, "reason": "23 L"},
                  "cavity_depth_mm": {"value": 330, "reason": "23 L"}, "wall_thickness_mm": {"value": 25, "reason": "case"},
                  "mass_kg": {"value": 11.5, "reason": "spec"}}
        dims = C.resolve_dims(GEOMETRY, {"upper_m": 0.38, "upper_why": "wall", "lower_m": 0.02}, answer, "spark",
                              *C.category_of("microwave oven"))
        d = dims["outer_depth"]
        self.assertAlmostEqual(d["value_m"], 0.38)
        self.assertEqual(d["source"], "Muse Spark")
        self.assertAlmostEqual(d["clamped"]["from_m"], 0.5)
        self.assertEqual(d["clamped"]["by"], "wall")
        self.assertEqual(d["range_m"], [0.45, 0.52])
        # the cavity fits the depth that's left behind the pocket, with a back wall
        self.assertLessEqual(dims["cavity_depth"]["value_m"], 0.38 - 0.015 - C.MIN_WALL_M + 1e-9)
        self.assertAlmostEqual(dims["mass"]["value_kg"], 11.5)

    def test_no_answer_falls_back_to_the_category_default(self):
        dims = C.resolve_dims(GEOMETRY, {"upper_m": None, "lower_m": None}, None, None, *C.category_of("microwave"))
        self.assertEqual(dims["outer_depth"]["source"], "microwave default")
        self.assertAlmostEqual(dims["outer_depth"]["value_m"], 0.78 * 0.48)
        self.assertGreater(dims["mass"]["value_kg"], 0)
        self.assertEqual(dims["mass"]["source"], "microwave default")
        for k in ("cavity_width", "cavity_height", "cavity_depth"):
            self.assertGreaterEqual(dims[k]["value_m"], C.MIN_CAVITY_M)

    def test_an_unknown_object_uses_the_generic_default(self):
        cat, defaults = C.category_of("thing")
        self.assertIsNone(cat)
        dims = C.resolve_dims(GEOMETRY, {}, {"outer_depth_mm": "deep"}, None, cat, defaults)
        self.assertEqual(dims["outer_depth"]["source"], "generic default")

    def test_a_cavity_wider_than_the_door_is_clamped_behind_it(self):
        answer = {"cavity_width_mm": {"value": 900}, "cavity_height_mm": {"value": 900}, "outer_depth_mm": {"value": 300}}
        dims = C.resolve_dims(GEOMETRY, {}, answer, "spark", *C.category_of("microwave"))
        open_w = (GEOMETRY["pocket_u"][1] - GEOMETRY["pocket_u"][0]) - 2 * C.FRAME_MARGIN_M
        self.assertAlmostEqual(dims["cavity_width"]["value_m"], open_w)
        self.assertIn("clamped", dims["cavity_height"])

    def test_a_measured_cavity_depth_wins_over_the_estimate(self):
        answer = {"outer_depth_mm": {"value": 370}, "cavity_depth_mm": {"value": 250}}
        dims = C.resolve_dims(GEOMETRY, {"upper_m": 0.39, "upper_why": "wall"}, answer, "spark",
                              *C.category_of("microwave"), measured={"cavity_depth": (0.30, "seen through the door")})
        cd = dims["cavity_depth"]
        self.assertEqual(cd["source"], "measured")
        self.assertAlmostEqual(cd["value_m"], 0.30)
        self.assertAlmostEqual(cd["estimate_m"], 0.25)  # Spark's estimate is kept in the record
        # a measured value that can't fit the body is still clamped to it
        dims = C.resolve_dims(GEOMETRY, {"upper_m": 0.2, "upper_why": "silhouette"}, answer, "spark",
                              *C.category_of("microwave"), measured={"cavity_depth": (0.30, "seen")})
        self.assertLessEqual(dims["cavity_depth"]["value_m"], 0.2 - GEOMETRY["pocket_v"])

    def test_a_lower_bound_above_the_upper_keeps_the_lower(self):
        dims = C.resolve_dims(GEOMETRY, {"upper_m": 0.1, "upper_why": "silhouette", "lower_m": 0.2, "lower_why": "seen"},
                              {"outer_depth_mm": {"value": 300}}, "spark", *C.category_of("microwave"))
        self.assertAlmostEqual(dims["outer_depth"]["value_m"], 0.2)


class CarcassLayoutTest(unittest.TestCase):
    outer_u, outer_z, depth = (0.0, 0.48), (0.0, 0.27), 0.37
    pocket = {"u": [0.0, 0.39], "z": [0.01, 0.265], "v": 0.013}
    cavity = {"u": [0.03, 0.36], "z": [0.03, 0.24], "depth": 0.30}

    def volume(self, boxes):
        return sum(np.prod(np.array(b["hi"]) - np.array(b["lo"])) for b in boxes)

    def test_boxes_partition_the_body_minus_the_pocket_and_the_cavity(self):
        boxes = C.carcass_boxes(self.outer_u, self.outer_z, self.depth, self.pocket, self.cavity)
        self.assertEqual(C.overlaps(boxes), [])
        outer = 0.48 * 0.27 * 0.37
        pocket = 0.39 * 0.255 * 0.013
        cavity = 0.33 * 0.21 * 0.30
        self.assertAlmostEqual(self.volume(boxes), outer - pocket - cavity, places=9)

    def test_the_cavity_and_the_pocket_are_empty_and_the_back_is_closed(self):
        boxes = C.carcass_boxes(self.outer_u, self.outer_z, self.depth, self.pocket, self.cavity)
        rng = np.random.default_rng(0)
        cav = rng.uniform([0.031, 0.014, 0.031], [0.359, 0.312, 0.239], (500, 3))
        pocket = rng.uniform([0.001, 0.0005, 0.011], [0.389, 0.0125, 0.264], (500, 3))
        self.assertEqual(C.penetration(cav, boxes)[0], 0.0)
        self.assertEqual(C.penetration(pocket, boxes)[0], 0.0)
        behind = np.array([[0.2, 0.35, 0.13]])  # between the cavity and the back
        self.assertGreater(C.penetration(behind, boxes)[0], 0.0)

    def test_a_door_poking_past_its_pocket_is_caught(self):
        boxes = C.carcass_boxes(self.outer_u, self.outer_z, self.depth, self.pocket, self.cavity)
        door = np.array([[0.2, 0.025, 0.13]])  # 25 mm behind the front, beside the cavity's opening? no: in the frame
        self.assertEqual(C.penetration(door, boxes)[0], 0.0)  # inside the cavity's opening: free
        door = np.array([[0.015, 0.025, 0.13]])  # behind the pocket, in the wall beside the cavity
        self.assertGreater(C.penetration(door, boxes)[0], 0.0)

    def test_no_cavity_fills_behind_the_door(self):
        boxes = C.carcass_boxes(self.outer_u, self.outer_z, self.depth, self.pocket, None)
        self.assertEqual(C.overlaps(boxes), [])
        self.assertAlmostEqual(self.volume(boxes), 0.48 * 0.27 * 0.37 - 0.39 * 0.255 * 0.013, places=9)

    def test_no_pocket_is_a_solid_box(self):
        boxes = C.carcass_boxes(self.outer_u, self.outer_z, self.depth, None, None)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(self.volume(boxes), 0.48 * 0.27 * 0.37, places=9)

    def test_door_pocket_is_where_the_door_goes_behind_the_front(self):
        self.assertIsNone(C.door_pocket(np.array([[0.1, -0.02, 0.1], [0.2, -0.05, 0.2]])))
        p = C.door_pocket(np.array([[0.0, 0.01, 0.02], [0.38, 0.008, 0.26], [0.2, -0.3, 0.1]]))
        self.assertAlmostEqual(p["u"][0], -C.DOOR_GAP_M)
        self.assertAlmostEqual(p["u"][1], 0.38 + C.DOOR_GAP_M)
        self.assertAlmostEqual(p["v"], 0.01 + C.DOOR_GAP_M)

    def test_boxes_land_where_the_front_says(self):
        front = C.Front(np.array([0.1, 0.0, 0.0]), np.array([np.sin(0.3), -np.cos(0.3), 0.0]))
        c, half, yaw = C.box_pose(front, {"lo": [0, 0, 0], "hi": [0.4, 0.3, 0.2]})
        corners = C.box_corners(front, [0, 0, 0], [0.4, 0.3, 0.2])
        np.testing.assert_allclose(c, corners.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(half, [0.2, 0.15, 0.1])
        r = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        np.testing.assert_allclose(r @ [1, 0, 0], front.u, atol=1e-12)
        np.testing.assert_allclose(r @ [0, 1, 0], -front.n, atol=1e-12)


class TriggerTest(unittest.TestCase):
    def run_dir(self, tmp: Path, hinge_vertical: bool, body_depth: float) -> Path:
        (tmp / "meshes").mkdir()
        (tmp / "model3d.json").write_text(json.dumps({"hinge_vertical": hinge_vertical}))
        # a door 0.4 m wide along x, 0.25 m tall, thin along y; a body body_depth deep along y
        v, f = prism(np.array([[0, 0], [0.4, 0], [0.4, 0.25], [0, 0.25]]), np.zeros(3), np.array([1.0, 0, 0]),
                     np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), -0.01, 0.01)
        write_stl(tmp / "meshes" / "lid.stl", v, f)
        v, f = prism(np.array([[0.41, 0], [0.48, 0], [0.48, body_depth], [0.41, body_depth]]), np.zeros(3),
                     np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), 0.0, 0.13)
        write_stl(tmp / "meshes" / "base.stl", v, f)
        return tmp

    def test_door_with_a_fragment_body_is_partial(self):
        with tempfile.TemporaryDirectory() as d:
            partial, why = C.needs_completion(self.run_dir(Path(d), True, 0.011))
            self.assertTrue(partial, why)

    def test_door_with_a_deep_body_is_not(self):
        with tempfile.TemporaryDirectory() as d:
            partial, why = C.needs_completion(self.run_dir(Path(d), True, 0.35))
            self.assertFalse(partial, why)

    def test_laptops_drawers_and_lamps_are_not(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(C.needs_completion(self.run_dir(Path(d), False, 0.011))[0])
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(C.needs_completion(Path(d))[0])  # no model3d.json (a chain or static run)


if __name__ == "__main__":
    unittest.main()
