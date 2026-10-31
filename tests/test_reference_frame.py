import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from video2urdf.lidar import Capture
from video2urdf.lidar import Roles
from video2urdf.lidar import can_use_closed_fallback, require_reference_angle
from video2urdf.lift3d import clean_angles
from video2urdf.agent import Agent
from video2urdf.track_checks import (alternate_plate_hint, named_base_warning, plate_covers_base_warning,
                                     slid_plate_lines, suggested_open_anchors)


class Mask:
    def __init__(self, mask):
        self.mask = mask
        self.area = int(mask.sum())

    def full_mask(self, height, width):
        return self.mask


class ReferenceFrameTest(unittest.TestCase):
    def test_closed_overlap_does_not_trigger_open_track_drift_warning(self):
        c = object.__new__(Capture)
        c.h = c.w = 20
        c.n = 10
        a = np.zeros((20, 20), bool)
        a[:10, :] = True
        b = ~a
        c.by = {"base": {i: Mask(b) for i in range(10)},
                "plate": {i: Mask(a if i < 5 else b) for i in range(10)},
                "keyboard": {i: Mask(b) for i in range(5)}}
        warnings = Agent.mask_checks(object.__new__(Agent), c, Roles(None, "base", "plate", "base", "keyboard"), {})
        self.assertFalse(any("drifted" in w or "whole object" in w for w in warnings))

    def test_plate_on_shut_lid_back_is_informational(self):
        # A laptop: shut on 40-99 (no keyboard), the screen track slid onto the lid's back there.
        indicator = set(range(40)) | set(range(100, 140))
        lines = slid_plate_lines(range(40, 100), 140, indicator, [])
        self.assertEqual(len(lines), 1)
        self.assertIn("60 shut", lines[0])
        self.assertFalse(lines[0].startswith("WARNING"))
        # plate_closed's track followed the raised lid on open frames: measured, not a warning
        lines = slid_plate_lines(range(0, 100), 140, indicator, range(0, 30))
        self.assertFalse(any(line.startswith("WARNING") for line in lines))
        self.assertIn("30 open where plate_closed's track followed the raised plate", lines[0])
        # a few unraised open frames as the lid shuts are normal
        lines = slid_plate_lines(range(30, 100), 140, indicator, [])
        self.assertFalse(any(line.startswith("WARNING") for line in lines))

    def test_plate_on_closed_surface_while_open_warns(self):
        indicator = set(range(40)) | set(range(100, 140))
        lines = slid_plate_lines(range(0, 140), 140, indicator, range(100, 110))
        self.assertTrue(lines[-1].startswith("WARNING"))
        self.assertIn("70 of the 80 open frames", lines[-1])
        self.assertIn("0-39", lines[-1])
        self.assertEqual(slid_plate_lines([], 140, indicator, []), [])

    def test_slid_frames_without_indicator_keep_the_share_rule(self):
        self.assertFalse(any(l.startswith("WARNING") for l in slid_plate_lines(range(20), 100, None, [])))
        self.assertTrue(slid_plate_lines(range(40), 100, None, [])[-1].startswith("WARNING"))

    def test_dropped_plate_frames_are_recorded(self):
        c = object.__new__(Capture)
        c.h = c.w = 10
        lid = np.zeros((10, 10), bool)
        lid[:4] = True
        other = np.zeros((10, 10), bool)
        other[6:] = True
        c.by = {"plate": {0: Mask(lid), 1: Mask(lid), 2: Mask(lid)},
                "closed": {1: Mask(lid), 2: Mask(other)}}
        self.assertEqual(c.drop_where_same("plate", "closed"), 1)
        self.assertEqual(c.slid_frames, [1])
        self.assertEqual(set(c.by["plate"]), {0, 2})
        self.assertEqual(c.drop_where_same("plate", None), 0)
        self.assertEqual(c.slid_frames, [])

    def test_reanchor_suggestions_span_visible_open_intervals(self):
        c = object.__new__(Capture)
        c.by = {"keyboard": {t: Mask(np.ones((2, 2), bool)) for t in (0, 10, 20, 80, 90, 100)}}
        roles = Roles(None, "base", "plate", open_indicator="keyboard")
        frames = suggested_open_anchors(c, roles, 4)
        self.assertEqual(frames, [0, 10, 80, 100])
        self.assertEqual(suggested_open_anchors(c, Roles(None, "base", "plate")), [])

    def test_failed_refit_does_not_mark_old_fit_current_or_replace_roles(self):
        class CaptureStub:
            names = ["floor", "base", "plate"]
            by = {}
            world = np.eye(4)[None, :, :]
            ref, moving_camera, n = 0, True, 1
            def drop_where_same(self, *args): return 0
            def drop_open_base_overlap(self, *args): return 0
            def ref_with(self, *args): return 0
            def level_with(self, *args): return {"shift_mm_max": 0}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = '{"base": "previous"}'
            (root / "roles.json").write_text(old)
            agent = object.__new__(Agent)
            agent.capture, agent.tracked, agent.track_dir, agent.video = root, True, root, root / 'video.mp4'
            agent.last_fit = {"old": True}
            with patch("video2urdf.agent.lidar.open_capture", return_value=CaptureStub()), \
                 patch("video2urdf.agent.lidar.fit_hinge", side_effect=ValueError("bad reference")):
                with self.assertRaisesRegex(ValueError, "bad reference"):
                    agent.tool_fit_lidar("floor", "base", "plate")
            self.assertIsNone(agent.last_fit)
            self.assertEqual((root / "roles.json").read_text(), old)

    def test_reference_failure_suggests_specific_retracking_frames(self):
        class CaptureStub:
            names = ["floor", "base", "plate", "keyboard"]
            ref, moving_camera, n = 0, True, 4
            by = {"keyboard": {0: None, 1: None, 3: None}}
            world = np.eye(4)[None, :, :]
            def drop_where_same(self, *args): return 0
            def drop_open_base_overlap(self, *args): return 0
            def ref_with(self, *args): return 0
            def level_with(self, *args): return {"shift_mm_max": 0}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = object.__new__(Agent)
            agent.capture, agent.tracked, agent.track_dir, agent.video = root, True, root, root / "video.mp4"
            with patch("video2urdf.agent.lidar.open_capture", return_value=CaptureStub()), \
                 patch("video2urdf.agent.lidar.fit_hinge", side_effect=ValueError("angle on reference frame rejected")):
                message = agent.tool_fit_lidar("floor", "base", "plate", open_indicator="keyboard")
            self.assertIn("[0, 1, 3]", message)
            self.assertIn("retrack_part", message)
            self.assertIsNone(agent.last_fit)


    def test_plate_track_on_open_base_is_removed_but_closed_frame_retained(self):
        c = object.__new__(Capture)
        c.h = c.w = 20
        base = np.zeros((20, 20), bool)
        base[10:19, 1:19] = True
        plate = np.zeros((20, 20), bool)
        plate[1:8, 1:19] = True
        c.by = {"base": {0: Mask(base), 1: Mask(base), 2: Mask(base)},
                "plate": {0: Mask(base), 1: Mask(base), 2: Mask(plate)},
                "keyboard": {0: Mask(base), 2: Mask(base)}}
        roles = Roles(None, "base", "plate", "base", "keyboard")
        self.assertEqual(c.drop_open_base_overlap(roles), 1)
        self.assertEqual(set(c.by["plate"]), {1, 2})

    def test_open_base_is_not_used_as_a_missing_lid(self):
        class CaptureStub:
            def has(self, part, t):
                return part == "keyboard" and t == 1
        roles = Roles(None, "base", "plate", "base", "keyboard")
        self.assertFalse(can_use_closed_fallback(CaptureStub(), roles, 1))
        self.assertTrue(can_use_closed_fallback(CaptureStub(), roles, 2))
        self.assertTrue(can_use_closed_fallback(CaptureStub(), Roles(None, "base", "plate", "lid_back", "keyboard"), 1))

    def test_separate_back_track_that_drifted_onto_base_is_not_a_lid(self):
        c = object.__new__(Capture)
        c.h = c.w = 10
        base = np.zeros((10, 10), bool)
        base[5:, :] = True
        lid = ~base
        c.by = {"base": {0: Mask(base)}, "back": {0: Mask(base)}, "keyboard": {0: Mask(base)}}
        roles = Roles(None, "base", "plate", "back", "keyboard")
        self.assertFalse(can_use_closed_fallback(c, roles, 0))
        c.by["back"][0] = Mask(lid)
        self.assertTrue(can_use_closed_fallback(c, roles, 0))

    def test_hinge_retries_distinct_open_reference_after_angle_failure(self):
        from types import SimpleNamespace
        from video2urdf.lidar import fit_hinge_with_reference_retries
        class Stub:
            n, ref = 101, 90
            by = {"plate": {20: None, 80: None}, "base": {20: None, 80: None},
                  "keyboard": {20: None, 80: None}}
            def has(self, name, frame): return frame in self.by[name]
            def set_ref(self, frame): self.ref = frame
        c = Stub()
        roles = SimpleNamespace(plate="plate", base="base", open_indicator="keyboard")
        def fit(capture, roles, log):
            if capture.ref == 90:
                raise ValueError("angle on reference frame rejected")
            return {"reference": capture.ref}
        with patch("video2urdf.lidar.fit_hinge", side_effect=fit):
            result, attempts = fit_hinge_with_reference_retries(c, roles, log=lambda _: None)
        self.assertEqual(attempts, [90, 20])
        self.assertEqual(result["reference"], 20)
        self.assertEqual(c.ref, 20)
        c.ref = 90
        prepared = []
        def prepare(capture, frame):
            prepared.append(frame)
            capture.set_ref(frame)
        with patch("video2urdf.lidar.fit_hinge", side_effect=fit):
            result, attempts = fit_hinge_with_reference_retries(c, roles, log=lambda _: None,
                                                                 prepare_reference=prepare)
        self.assertEqual(prepared, [20])
        self.assertEqual(result["reference"], 20)

    def test_hinge_requires_actual_plate_mask_at_reference(self):
        from types import SimpleNamespace
        from video2urdf.lidar import fit_hinge
        capture = SimpleNamespace(ref=150, has=lambda part, frame: False)
        roles = SimpleNamespace(plate="screen", base="base")
        with self.assertRaisesRegex(ValueError, "plate.*reference frame 150"):
            fit_hinge(capture, roles)

    def test_reference_angle_must_not_be_replaced_by_a_neighbour(self):
        angles = clean_angles(np.array([-1.8] * 15 + [0.0]))
        self.assertTrue(np.isnan(angles[-1]))
        with self.assertRaisesRegex(ValueError, "reference frame 15"):
            require_reference_angle({15: np.zeros(3), 14: np.ones(3)}, angles, 15)
        require_reference_angle({0: np.zeros(3)}, np.array([0.0]), 0)

    def test_suggests_testing_swapped_tracks_only_when_alternate_is_distinct(self):
        c = object.__new__(Capture)
        c.h = c.w = 20
        c.ref = 0
        base = np.zeros((20, 20), bool)
        base[10:18, 3:17] = True
        alternate = np.zeros((20, 20), bool)
        alternate[1:8, 3:17] = True
        c.by = {"base": {0: Mask(base)}, "plate": {0: Mask(base)}, "back": {0: Mask(alternate)}}
        roles = Roles(None, "base", "plate", "back")
        self.assertIn("swapping", alternate_plate_hint(c, roles))
        c.by["back"][0] = Mask(base)
        self.assertIsNone(alternate_plate_hint(c, roles))

    def test_plate_that_covers_base_is_flagged(self):
        c = object.__new__(Capture)
        c.h = c.w = 20
        base = np.zeros((20, 20), bool)
        base[10:15, 5:15] = True
        plate = np.zeros((20, 20), bool)
        plate[8:19, 2:18] = True
        c.by = {"base": {0: Mask(base)}, "plate": {0: Mask(plate)}}
        self.assertTrue(plate_covers_base_warning(c, Roles(None, "base", "plate")))
        c.by["plate"][0] = Mask(np.roll(plate, 10, axis=0))
        self.assertFalse(plate_covers_base_warning(c, Roles(None, "base", "plate")))
        c.by["base"] = {i: Mask(base) for i in range(5)}
        c.by["plate"] = {i: Mask(plate if i == 3 else np.roll(plate, 10, axis=0)) for i in range(5)}
        c.ref = 3
        self.assertIn("reference frame 3", plate_covers_base_warning(c, Roles(None, "base", "plate"))[0])

    def test_laptop_component_seed_warns_even_if_base_is_indicator(self):
        c = object.__new__(Capture)
        with tempfile.TemporaryDirectory() as tmp:
            c.track_dir = Path(tmp)
            (c.track_dir / "parts.json").write_text(json.dumps({"0": {"name": "base", "prompt": "laptop keyboard"}}))
            roles = Roles(None, "base", "plate", open_indicator="base")
            self.assertTrue(named_base_warning(c, roles))
            (c.track_dir / "parts.json").write_text(json.dumps({"0": {"name": "base", "prompt": "silver laptop base"}}))
            self.assertFalse(named_base_warning(c, roles))

    def test_reference_requires_open_indicator_and_suitable_masks(self):
        c = object.__new__(Capture)
        c.h = c.w = 10
        c.moving_camera = True
        base = np.zeros((10, 10), bool)
        base[5:, :] = True
        plate = np.zeros((10, 10), bool)
        plate[:4, :] = True
        larger_plate = np.ones((10, 10), bool)
        c.by = {"base": {0: Mask(base), 1: Mask(base)},
                "plate": {0: Mask(larger_plate), 1: Mask(plate)},
                "indicator": {1: Mask(base)}}
        self.assertEqual(c.ref_with("base", "plate", open_indicator="indicator"), 1)
        c.by["indicator"] = {}
        with self.assertRaisesRegex(ValueError, "No frame"):
            c.ref_with("base", "plate", open_indicator="indicator")
        c.by["indicator"] = {1: Mask(base)}
        c.by["closed"] = {1: Mask(plate)}
        with self.assertRaisesRegex(ValueError, "No open-indicator frame"):
            c.ref_with("base", "plate", "closed", "indicator")

    def test_reference_rejects_plate_that_contains_base(self):
        c = object.__new__(Capture)
        c.h = c.w = 20
        c.moving_camera = True
        base = np.zeros((20, 20), bool)
        base[10:15, 5:15] = True
        wrong = np.zeros((20, 20), bool)
        wrong[8:19, 2:18] = True
        good = np.zeros((20, 20), bool)
        good[1:8, 3:17] = True
        c.by = {"base": {0: Mask(base), 1: Mask(base)},
                "plate": {0: Mask(wrong), 1: Mask(good)},
                "indicator": {0: Mask(base), 1: Mask(base)}}
        self.assertEqual(c.ref_with("base", "plate", open_indicator="indicator"), 1)

    def test_warns_when_base_is_only_a_fragment(self):
        c = object.__new__(Capture)
        c.h = c.w = 20
        c.n = 6
        base = np.zeros((20, 20), bool)
        base[10:15, 0:5] = True
        indicator = np.zeros((20, 20), bool)
        indicator[10:15, 10:15] = True
        plate = np.zeros((20, 20), bool)
        plate[0:5, 0:5] = True
        c.by = {"base": {i: Mask(base) for i in range(6)},
                "plate": {i: Mask(plate) for i in range(6)},
                "keyboard": {i: Mask(indicator) for i in range(6)}}
        agent = object.__new__(Agent)
        warnings = agent.mask_checks(c, Roles(None, "base", "plate", open_indicator="keyboard"), {})
        self.assertTrue(any("fragment" in w for w in warnings))
        with tempfile.TemporaryDirectory() as tmp:
            c.track_dir = Path(tmp)
            (c.track_dir / "parts.json").write_text(json.dumps({"1": {"name": "base", "anchors": [[0, []]]}}))
            # Later tracking drift should not turn a good anchored candidate into a false warning.
            c.by["base"][0] = Mask(base | indicator)
            warnings = agent.mask_checks(c, Roles(None, "base", "plate", open_indicator="keyboard"), {})
            self.assertFalse(any("fragment" in w for w in warnings))

    def test_avoids_small_base_and_mask_that_includes_plate(self):
        def rect(y0, y1, x0, x1):
            mask = np.zeros((20, 20), bool)
            mask[y0:y1, x0:x1] = True
            return Mask(mask)
        c = object.__new__(Capture)
        c.h = c.w = 20
        c.by = {
            "base": {0: rect(10, 20, 0, 10), 1: rect(17, 20, 0, 10),
                     2: rect(0, 20, 0, 10), 3: rect(10, 20, 0, 9)},
            "plate": {0: rect(0, 5, 0, 10), 1: rect(0, 9, 0, 10),
                      2: rect(0, 10, 0, 10), 3: rect(0, 8, 0, 10)},
            "closed": {i: rect(0, 1, 19, 20) for i in range(4)},
        }
        self.assertEqual(c.ref_with("base", "plate", "closed"), 3)

    def test_retry_starts_from_the_largest_plate_whose_angle_was_measured(self):
        from types import SimpleNamespace
        from video2urdf.lidar import ReferenceAngleError, fit_hinge_with_reference_retries

        class Stub:
            n, ref = 101, 90
            by = {"plate": {t: SimpleNamespace(area=a) for t, a in ((20, 50), (55, 300), (70, 900), (90, 1000))},
                  "base": {t: None for t in (20, 55, 70, 90)}, "keyboard": {t: None for t in (20, 55, 90)}}
            def has(self, name, frame): return frame in self.by[name]
            def set_ref(self, frame): self.ref = frame

        def fit(capture, roles, log):
            if capture.ref == 90:  # 70 has the larger plate but no open indicator; 20 wasn't measured
                raise ReferenceAngleError("angle on reference frame 90 rejected", valid=[55, 70, 80])
            return {"reference": capture.ref}
        roles = SimpleNamespace(plate="plate", base="base", open_indicator="keyboard")
        with patch("video2urdf.lidar.fit_hinge", side_effect=fit):
            result, attempts = fit_hinge_with_reference_retries(Stub(), roles, log=lambda _: None)
        self.assertEqual(attempts, [90, 55])
        self.assertEqual(result["reference"], 55)

    def test_reference_angle_error_lists_measured_frames(self):
        from video2urdf.lidar import ReferenceAngleError
        angles = np.array([0.1, np.nan, 0.3, np.nan])
        with self.assertRaises(ReferenceAngleError) as caught:
            require_reference_angle({0: 0, 2: 0, 3: 0}, angles, 3)
        self.assertEqual(sorted(caught.exception.valid), [0, 2])


def door_clouds(hinge=np.array([0.1, 0.0, 0.6]), width=0.36, closed=range(0, 40), opened=range(40, 100)):
    """A vertical door (y is up, as in camera coordinates) hinged at ``hinge``: shut frames see its
    whole face, open frames only a narrow strip at its free edge (a glossy door's handle)."""
    rng = np.random.default_rng(3)
    face = np.array([0.8, 0.0, -0.6])  # along the shut door, from the hinge to its free edge

    def door(angle, r_lo):
        r = rng.uniform(r_lo, width, 800)
        y = rng.uniform(-0.15, 0.15, 800)
        c, s = np.cos(angle), np.sin(angle)
        d = np.array([c * face[0] + s * face[2], 0.0, -s * face[0] + c * face[2]])
        return hinge + np.outer(r, d) + np.outer(y, [0, 1.0, 0]) + rng.normal(0, 0.001, (800, 3))

    clouds = {t: door(0.0, 0.0) for t in closed}
    angles = np.linspace(np.deg2rad(15), np.deg2rad(100), len(opened))
    clouds.update({t: door(a, width - 0.02) for t, a in zip(opened, angles)})
    return clouds, hinge


class TrackCandidatesTest(unittest.TestCase):
    def test_candidate_and_candidates_are_both_anchors(self):
        from types import SimpleNamespace

        class Captured(Exception):
            pass

        class Sam:
            def track_boxes(self, video, size, anchors):
                raise Captured(anchors)
        agent = object.__new__(Agent)
        agent.video, agent.size, agent.sam = Path("v.mp4"), (20, 10), Sam()
        box = {"s1.0": (1, 1, 5, 5), "s2.0": (2, 2, 6, 6), "s3.0": (3, 3, 7, 7)}
        agent.candidates = {"s1.0": ("microwave door", SimpleNamespace(box=box["s1.0"]), 427),
                            "s2.0": ("closed microwave door", SimpleNamespace(box=box["s2.0"]), 0),
                            "s3.0": ("closed microwave door", SimpleNamespace(box=box["s3.0"]), 854)}
        with self.assertRaises(Captured) as caught:  # Spark's form: the main candidate plus more anchors
            agent.tool_track([{"name": "plate", "candidate": "s1.0", "candidates": ["s2.0", "s3.0"]}])
        self.assertEqual(caught.exception.args[0], {0: [(0, box["s2.0"]), (427, box["s1.0"]), (854, box["s3.0"])]})
        with self.assertRaises(Captured) as caught:
            agent.tool_track([{"name": "plate", "candidates": ["s1.0", "s1.0"]}])
        self.assertEqual(caught.exception.args[0], {0: [(427, box["s1.0"])]})
        agent.candidates["s4.0"] = ("microwave door", SimpleNamespace(box=(4, 4, 8, 8)), 0)
        self.assertIn("one candidate per frame",
                      agent.tool_track([{"name": "plate", "candidate": "s4.0", "candidates": ["s2.0"]}]))
        self.assertIn("give a segment candidate", agent.tool_track([{"name": "plate"}]))


class DoorHingeTest(unittest.TestCase):
    def test_door_hinge_on_the_closed_face_where_open_points_stay_equidistant(self):
        from video2urdf.lift3d import fit_planar_hinge
        clouds, hinge = door_clouds()
        up = np.array([0, -1.0, 0])
        fit = fit_planar_hinge(clouds, 100, 70, pin=hinge, base_normal=up, closed_frames=set(range(40)))
        self.assertEqual(fit.placed_by, "door closed face")
        gap = fit.point - hinge
        gap -= (gap @ fit.axis) * fit.axis
        self.assertLess(np.linalg.norm(gap), 0.01)
        q = np.rad2deg(fit.q)
        self.assertAlmostEqual(abs(q[0] - q[99]), 100, delta=3)  # shut vs widest open, either sign
        # without shut frames the door still gets a hinge (a circle through its centroids)
        fit = fit_planar_hinge(clouds, 100, 70, pin=hinge, base_normal=up)
        self.assertEqual(fit.placed_by, "door circle")

    def test_door_test_on_plate_normals(self):
        from video2urdf.lift3d import is_door
        up = np.array([0, -1.0, 0])
        door = [np.array([np.cos(a), 0, np.sin(a)]) for a in np.linspace(0, 1.7, 20)]
        lid = [np.array([0, -np.cos(a), np.sin(a)]) for a in np.linspace(0, 1.7, 20)]  # turns about x
        self.assertTrue(is_door(np.array(door), up))
        self.assertFalse(is_door(np.array(lid), up))
        self.assertFalse(is_door(np.array(door), None))

    def test_door_told_from_the_plate_track_before_the_lid_fallback(self):
        from video2urdf.lidar import looks_like_door
        clouds, _ = door_clouds()

        class Stub:
            by = {"plate": {t: None for t in clouds}}
            def cloud(self, part, t): return clouds[t]
        up = np.array([0, -1.0, 0])
        self.assertTrue(looks_like_door(Stub(), Roles(None, "base", "plate"), up))
        # a laptop lid turning about a horizontal hinge on the base's top is not a door
        rng = np.random.default_rng(0)
        lid = {}
        for t, a in enumerate(np.linspace(0, 1.9, 40)):
            r, x = rng.uniform(0, 0.2, 600), rng.uniform(-0.15, 0.15, 600)
            lid[t] = np.column_stack([x, -r * np.sin(a), 0.5 - r * np.cos(a)])
        Stub.by = {"plate": {t: None for t in lid}}
        Stub.cloud = lambda self, part, t: lid[t]
        self.assertFalse(looks_like_door(Stub(), Roles(None, "base", "plate"), up))

    def test_door_angles_past_the_shut_position_are_dropped(self):
        from video2urdf.lift3d import drop_overclosed
        q = np.deg2rad(np.r_[[-15.0] * 10, np.linspace(-15, 35, 20), [35.0] * 10, [-160, -158], [-12.0] * 10])
        out = drop_overclosed(q, shut=list(range(10)) + list(range(42, 52)))
        self.assertTrue(np.isnan(out[40]) and np.isnan(out[41]))
        self.assertEqual(int(np.isnan(out).sum()), 2)
        # mirrored: a door opening the other way
        out = drop_overclosed(-q, shut=list(range(10)) + list(range(42, 52)))
        self.assertEqual(int(np.isnan(out).sum()), 2)
        self.assertTrue(np.array_equal(drop_overclosed(q, shut=[0, 1]), q))  # too few shut frames: unchanged


if __name__ == "__main__":
    unittest.main()
