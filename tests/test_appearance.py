import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
import sys

import mujoco
import numpy as np

from video2urdf.appearance import bevel_mesh, book_cover_mesh, complete_geometry, laptop_base_visual, laptop_base_width_hint, laptop_lid_height_hint, laptop_screen_visual, screen_needs_bezel, category_joint_mismatch, complete_laptop_details, remove_laptop_details, complete_laptop_hinge
from video2urdf.appearance_check import observed_mask, open_frame_available, rejected_links
from video2urdf.sam import Detection
from video2urdf.emit3d import Link3D, Visual, mjcf, urdf
from video2urdf.shape import prism, write_stl
from video2urdf.texture import mask_rectified_cell

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sam3d_meshes import add_visuals, align_to_depth, model_to_link, quality_passes, select_frame, retry_frames, surface_samples, write_obj
from export_viewer import front


class AppearanceTest(unittest.TestCase):
    def test_static_drawer_thin_fit_gets_hypothetical_visual_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'meshes').mkdir()
            rect = np.array([[-.2, -.006], [.2, -.006], [.2, .006], [-.2, .006]])
            v, f = prism(rect, np.zeros(3), *np.eye(3), -.3, .3)
            write_stl(root / 'meshes/static_object.stl', v, f)
            links = [Link3D('base', [Visual('meshes/static_object.stl', (.4, .3, .2))])]
            (root / 'model.xml').write_text(mjcf('test', links))
            (root / 'model.urdf').write_text(urdf('test', links))
            (root / 'static.json').write_text(json.dumps({'shape': 'box', 'part': 'object'}))
            cams = np.repeat(np.eye(4)[None], 10, axis=0); cams[:, 1, 3] = -1
            np.savez(root / 'replay3d.npz', cam_to_world=cams)
            before = mujoco.MjModel.from_xml_path(str(root / 'model.xml'))
            shape = complete_geometry(root, {'category': 'drawer', 'drawer_count': 3}, log=lambda _: None)
            after = mujoco.MjModel.from_xml_path(str(root / 'model.xml'))
            self.assertAlmostEqual(shape['static_drawer_depth_assumed_mm'], 260., places=3)
            self.assertEqual(len(shape['inferred_visuals']), 4)
            np.testing.assert_allclose(before.body_mass, after.body_mass)
            model = ET.parse(root / 'model.urdf').getroot()
            self.assertEqual(model.find("./link[@name='base']/collision/geometry/mesh").get('filename'),
                             'meshes/static_object.stl')

    def test_static_drawer_adds_fronts_to_consistently_viewed_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'meshes').mkdir()
            rect = np.array([[-.2, -.15], [.2, -.15], [.2, .15], [-.2, .15]])
            v, f = prism(rect, np.zeros(3), *np.eye(3), -.3, .3)
            write_stl(root / 'meshes/static_object.stl', v, f)
            links = [Link3D('base', [Visual('meshes/static_object.stl', (.4, .3, .2))])]
            (root / 'model.xml').write_text(mjcf('test', links))
            (root / 'model.urdf').write_text(urdf('test', links))
            (root / 'static.json').write_text(json.dumps({'shape': 'box', 'part': 'object'}))
            cams = np.repeat(np.eye(4)[None], 10, axis=0)
            cams[:, 1, 3] = -1
            np.savez(root / 'replay3d.npz', cam_to_world=cams)
            before = mujoco.MjModel.from_xml_path(str(root / 'model.xml'))
            shape = complete_geometry(root, {'category': 'drawer', 'drawer_count': 3,
                                             'drawer_has_handles': True}, log=lambda _: None)
            after = mujoco.MjModel.from_xml_path(str(root / 'model.xml'))
            self.assertEqual(len(shape['inferred_visuals']), 6)
            self.assertTrue(shape['static_drawer_side_assumed'])
            handle = next(item for item in shape['inferred_visuals'] if item['name'] == 'inferred_drawer_handle_1')
            self.assertAlmostEqual(handle['size_m'][0], .4 * .25, places=5)
            shape['active'] = True
            (root / 'shape_inference.json').write_text(json.dumps(shape))
            self.assertEqual(front(root / 'model.xml'), [0., -1., 0.])
            shape['active'] = False
            (root / 'shape_inference.json').write_text(json.dumps(shape))
            self.assertIsNone(front(root / 'model.xml'))
            np.testing.assert_allclose(before.body_mass, after.body_mass)
            self.assertTrue(all(after.geom_contype[i] == 0 for i in range(after.ngeom)
                                if after.geom(i).name.startswith('inferred_')))
            self.assertIsNotNone(ET.parse(root / 'model.urdf').getroot().find("./link[@name='base']/visual[@name='inferred_drawer_front_1']"))
            cams[:5, 1, 3] = 1
            np.savez(root / 'replay3d.npz', cam_to_world=cams)
            # A new original avoids retaining already-generated visuals in this check.
            (root / 'model.xml').write_text(mjcf('test', links))
            ambiguous = complete_geometry(root, {'category': 'drawer', 'drawer_count': 3}, log=lambda _: None)
            self.assertEqual(ambiguous['inferred_visuals'], [])

    def test_laptop_hinge_connector_is_visual_only_in_both_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'meshes').mkdir()
            rect = np.array([[-.15, .003], [.15, .003], [.15, .20], [-.15, .20]])
            v, f = prism(rect, np.zeros(3), *np.eye(3), -.022, -.012)
            write_stl(root / 'meshes/base.stl', v, f)
            rect = np.array([[-.15, 0], [.15, 0], [.15, .20], [-.15, .20]])
            v, f = prism(rect, np.zeros(3), np.eye(3)[0], np.eye(3)[2], -np.eye(3)[1], -.004, .004)
            write_stl(root / 'meshes/lid.stl', v, f)
            links = [Link3D('base', [Visual('meshes/base.stl', (.5, .5, .5))]),
                     Link3D('lid', [Visual('meshes/lid.stl', (.5, .5, .5))], parent='base',
                            joint_type='revolute', axis=(1, 0, 0), limits=(-1.8, .2))]
            (root / 'model.xml').write_text(mjcf('test', links))
            (root / 'model.urdf').write_text(urdf('test', links))
            before = mujoco.MjModel.from_xml_path(str(root / 'model.xml'))
            connector = complete_laptop_hinge(root)
            self.assertIsNotNone(connector)
            after = mujoco.MjModel.from_xml_path(str(root / 'model.xml'))
            np.testing.assert_allclose(before.body_mass, after.body_mass)
            np.testing.assert_allclose(before.body_inertia, after.body_inertia)
            geom = after.geom('inferred_laptop_hinge').id
            self.assertEqual(after.geom_contype[geom], 0)
            self.assertIsNotNone(ET.parse(root / 'model.urdf').getroot().find("./link[@name='base']/visual[@name='inferred_laptop_hinge']/geometry/cylinder"))
            complete_laptop_hinge(root)
            self.assertEqual(mujoco.MjModel.from_xml_path(str(root / 'model.xml')).ngeom, after.ngeom)

    def test_base_comparison_requires_open_indicator_when_provided(self):
        d = Detection(2, 0, (1, 1, 3, 3), np.ones((3, 3), bool))
        tracks = {"keyboard": {2: d}}
        self.assertFalse(open_frame_available(tracks, "keyboard", 1, 10, 10))
        self.assertTrue(open_frame_available(tracks, "keyboard", 2, 10, 10))
        self.assertTrue(open_frame_available(tracks, None, 1, 10, 10))

    def test_drawer_base_coverage_combines_simultaneous_tracked_pieces(self):
        a = Detection(0, 0, (1, 1, 3, 3), np.ones((3, 3), bool))
        b = Detection(0, 1, (6, 6, 8, 8), np.ones((3, 3), bool))
        tracks = {"side": {0: a}, "front": {0: b}}
        self.assertEqual(observed_mask(tracks, ["side", "front"], 0, 10, 10, combine=True).sum(), 18)
        self.assertEqual(observed_mask(tracks, ["side", "front"], 0, 10, 10).sum(), 9)

    def test_shape_category_must_match_fitted_joint_type(self):
        slider = ET.fromstring('<mujoco><worldbody><body><joint type="slide"/></body></worldbody></mujoco>')
        hinge = ET.fromstring('<mujoco><worldbody><body><joint type="hinge"/></body></worldbody></mujoco>')
        self.assertIsNotNone(category_joint_mismatch("laptop", slider))
        self.assertIsNone(category_joint_mismatch("drawer", slider))
        self.assertIsNotNone(category_joint_mismatch("drawer", hinge))
        self.assertIsNone(category_joint_mismatch("laptop", hinge))
        self.assertIsNone(category_joint_mismatch("other", hinge))
        self.assertIsNone(category_joint_mismatch("box", hinge))
        self.assertIsNotNone(category_joint_mismatch("box", slider))

    def test_laptop_visual_completes_missing_corners_within_bounds(self):
        outline = np.array([[0, 0], [.30, 0], [.30, .12], [.18, .20], [0, .20]])
        vertices, faces = prism(outline, np.zeros(3), *np.eye(3), 0, .015)
        completed, _ = laptop_base_visual(vertices[faces])
        np.testing.assert_allclose(np.ptp(completed, axis=0), [.30, .20, .015], atol=.002)
        self.assertGreater(completed[(completed[:, 0] > .29) & (completed[:, 1] > .19)].shape[0], 0)

    def test_only_laptop_visual_regularizes_implausible_base_thickness(self):
        outline = np.array([[0, 0], [.3, 0], [.3, .2], [0, .2]])
        vertices, faces = prism(outline, np.zeros(3), *np.eye(3), 0, .08)
        unmodified, _ = laptop_base_visual(vertices[faces])
        laptop, _ = laptop_base_visual(vertices[faces], regularize_thickness=True)
        self.assertAlmostEqual(np.ptp(unmodified[:, 2]), .08, places=4)
        self.assertLess(np.ptp(laptop[:, 2]), .018)
        self.assertAlmostEqual(laptop[:, 2].max(), .08, places=4)

    def test_laptop_screen_fills_irregular_corners(self):
        outline = np.array([[0, 0], [.30, 0], [.30, .12], [.18, .20], [0, .20]])
        vertices, faces = prism(outline, np.zeros(3), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2], 0, .001)
        screen, _ = laptop_screen_visual(vertices[faces])
        np.testing.assert_allclose(np.ptp(screen, axis=0), [.30, .20, .001], atol=1e-6)
        self.assertTrue(np.any((screen[:, 0] > .29) & (screen[:, 1] > .19)))
        inset, _ = laptop_screen_visual(vertices[faces], inset=.04)
        np.testing.assert_allclose(np.ptp(inset, axis=0), [.276, .184, .001], atol=1e-6)
        self.assertTrue(screen_needs_bezel(vertices[faces], vertices[faces]))
        smaller, _ = laptop_screen_visual(vertices[faces], inset=.1)
        self.assertFalse(screen_needs_bezel(smaller, vertices[faces]))

    def test_laptop_width_hint_requires_aligned_modest_extension(self):
        def box(x0, x1):
            v, f = prism(np.array([[x0, 0], [x1, 0], [x1, .2], [x0, .2]]),
                         np.zeros(3), *np.eye(3), 0, .01)
            return v[f]
        base = box(0, .20)
        self.assertEqual(laptop_base_width_hint(base, box(0, .22)), (0., .22))
        self.assertIsNone(laptop_base_width_hint(base, box(-.02, .22)))
        self.assertIsNone(laptop_base_width_hint(base, box(0, .30)))

    def test_tall_laptop_lid_hint_is_bounded(self):
        def lid(height):
            v, f = prism(np.array([[0, 0], [.2, 0], [.2, height], [0, height]]),
                         np.zeros(3), *np.eye(3), 0, .006)
            return v[f]
        self.assertIsNone(laptop_lid_height_hint(lid(.19)))
        _, factor = laptop_lid_height_hint(lid(.3))
        self.assertEqual(factor, .85)
        detached = lid(.3).copy()
        detached[..., 1] += .2
        self.assertIsNone(laptop_lid_height_hint(detached))

    def test_schematic_keyboard_preserves_physics_and_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meshes").mkdir()
            outline = np.array([[0, 0], [.3, 0], [.3, .2], [0, .2]])
            vertices, faces = prism(outline, np.zeros(3), *np.eye(3), 0, .01)
            write_stl(root / "meshes/base.stl", vertices, faces)
            write_stl(root / "meshes/lid.stl", vertices, faces)
            links = [Link3D("base", [Visual("meshes/base.stl", (.5, .5, .5))]),
                     Link3D("lid", [Visual("meshes/lid.stl", (.5, .5, .5))], parent="base",
                            joint_type="revolute", limits=(-1, 1))]
            (root / "model.xml").write_text(mjcf("test", links))
            (root / "model.urdf").write_text(urdf("test", links))
            before = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
            self.assertEqual(len(complete_laptop_details(root)), 2)
            complete_laptop_details(root)
            after = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
            self.assertEqual(after.ngeom, before.ngeom + 2)
            np.testing.assert_allclose(after.body_mass, before.body_mass)
            np.testing.assert_allclose(after.body_inertia, before.body_inertia)
            self.assertTrue(all(after.geom_contype[i] == 0 for i in range(after.ngeom)
                                if after.geom(i).name in ("inferred_keyboard", "inferred_trackpad")))
            self.assertEqual(len(ET.parse(root / "model.urdf").getroot().findall(".//visual[@name='inferred_keyboard']")), 1)
            remove_laptop_details(root)
            self.assertEqual(mujoco.MjModel.from_xml_path(str(root / "model.xml")).ngeom, before.ngeom)
            self.assertEqual(len(ET.parse(root / "model.urdf").getroot().findall(".//visual[@name='inferred_keyboard']")), 0)

    def test_texture_masks_background_even_for_a_narrow_face(self):
        cell = np.full((10, 10, 3), 200, np.uint8)
        inside = np.zeros((10, 10), bool)
        inside[0, :5] = True
        cell[inside] = [10, 20, 30]
        clean = mask_rectified_cell(cell, inside)
        self.assertTrue(np.all(clean == [10, 20, 30]))
        inside[0, 4] = False
        self.assertIsNone(mask_rectified_cell(cell, inside))

    def test_retry_frames_are_diverse_and_require_usable_pose(self):
        from types import SimpleNamespace
        tracks = {i: SimpleNamespace(area=100 if i != 9 else 10) for i in range(10)}
        capture = SimpleNamespace(by={"part": tracks}, n=10, has=lambda part, frame: True)
        joint = ("hinge", np.array([1., 0, 0]), np.array([0, 0, 0, 0, np.nan, 0, 0, 0, 0, 0]))
        selected = retry_frames(capture, "part", joint, first=5, count=2)
        self.assertEqual(selected, [2, 8])
        capture.has = lambda part, frame: part != "indicator" or frame >= 3
        selected = retry_frames(capture, "part", joint, first=5, count=2, open_indicator="indicator")
        self.assertEqual(selected, [3, 8])

    def test_sam3d_requires_known_joint_position(self):
        class Detection:
            def __init__(self, area):
                self.area = area
        class Capture:
            n, ref = 3, 0
            by = {"lid": {0: Detection(100), 1: Detection(40), 2: Detection(80)}}
            def has(self, part, t):
                return t in self.by[part]
        joint = ("hinge", np.array([1., 0., 0.]), np.array([np.nan, .1, .2]))
        self.assertEqual(select_frame(Capture(), "lid", joint), 2)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            select_frame(Capture(), "lid", joint, 0)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            model_to_link(joint, 0)
        self.assertEqual(select_frame(Capture(), "lid", None), 0)

    def test_sam3d_undoes_measured_hinge_and_slide(self):
        axis = np.array([1., 0., 0.])
        hinge = ("hinge", axis, np.array([np.pi / 2]))
        link_point = np.array([0., 1., 0., 1.])
        observed = np.array([0., 0., 1., 1.])
        np.testing.assert_allclose(model_to_link(hinge, 0) @ observed, link_point, atol=1e-8)
        slide = ("slide", axis, np.array([.12]))
        np.testing.assert_allclose(model_to_link(slide, 0) @ np.array([.15, .2, 0., 1.]),
                                   np.array([.03, .2, 0., 1.]), atol=1e-8)

    def test_visual_guard_rejects_degradation_and_insufficient_evidence(self):
        comparison = {"links": {"lid": {"frames": 21, "baseline": .90, "inferred": .789},
                                "base": {"frames": 3, "baseline": .95, "inferred": .95},
                                "other": {"frames": 10, "baseline": .91, "inferred": .90}}}
        self.assertEqual(rejected_links(comparison), ["lid", "base"])

    def test_visual_guard_rejects_mask_covering_but_oversized_shape(self):
        comparison = {"links": {"base": {"frames": 20, "baseline": .9, "inferred": .95,
                                         "baseline_iou": .8, "inferred_iou": .5}}}
        self.assertEqual(rejected_links(comparison), ["base"])
        comparison["links"]["base"]["iou_informative"] = False
        self.assertEqual(rejected_links(comparison), [])

    def test_book_cover_uses_visible_face_and_stays_behind_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meshes").mkdir()
            rect = np.array([[.05, .02], [.25, .02], [.25, .18], [.05, .18]])
            e1, e2, e3 = np.eye(3)[0], np.eye(3)[2], -np.eye(3)[1]
            v, f = prism(rect, np.zeros(3), e1, e2, e3, .002, .003)
            write_stl(root / "meshes/screen.stl", v, f)
            v, f = prism(rect, np.zeros(3), e1, e2, e3, -.003, .002)
            write_stl(root / "meshes/lid.stl", v, f)
            (root / "model3d.json").write_text(json.dumps({"lid_thickness_m": .005}))
            cover, _ = book_cover_mesh(root)
            self.assertLessEqual(cover[:, 0].max() - cover[:, 0].min(), .201)
            self.assertLessEqual((cover @ e3).max(), .00201)
            larger = np.array([[0., .02], [.30, .02], [.30, .18], [0., .18]])
            v, f = prism(larger, np.zeros(3), e1, e2, e3, -.003, .002)
            write_stl(root / "meshes/lid.stl", v, f)
            blended, _ = book_cover_mesh(root, outer_blend=.5)
            self.assertGreater(np.ptp(blended[:, 0]), np.ptp(cover[:, 0]))
            self.assertLess(np.ptp(blended[:, 0]), .30)

    def test_depth_match_cannot_hide_wrong_overall_size(self):
        fitted = np.array([248., 327., 122.])
        self.assertFalse(quality_passes(2.1, 4.5, np.array([52., 468., 207.]), fitted))
        self.assertFalse(quality_passes(8., 12., fitted, fitted))
        self.assertFalse(quality_passes(float("nan"), 1., fitted, fitted))
        self.assertTrue(quality_passes(2., 6., fitted, fitted))
        self.assertFalse(quality_passes(2., 6., fitted, fitted, part_override=True))

    def test_drawer_completion_adds_only_noncolliding_visuals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meshes").mkdir()
            rect = np.array([[0, 0], [.3, 0], [.3, .3], [0, .3]])
            v, f = prism(rect, np.zeros(3), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2], 0, .6)
            write_stl(root / "meshes/base.stl", v, f)
            v, f = prism(rect, np.zeros(3), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2], .2, .4)
            write_stl(root / "meshes/drawer.stl", v, f)
            links = [Link3D("base", [Visual("meshes/base.stl", (.5, .5, .5))]),
                     Link3D("drawer1", [Visual("meshes/drawer.stl", (.5, .5, .5))], parent="base",
                            joint_type="prismatic", limits=(0, .1))]
            (root / "model.xml").write_text(mjcf("test", links))
            (root / "model.urdf").write_text(urdf("test", links))
            before = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
            appearance = {"category": "drawer", "object": "cabinet", "drawer_count": 3,
                          "drawer_has_handles": True}
            report = complete_geometry(root, appearance, log=lambda _: None)
            after = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
            self.assertEqual(len(report["inferred_visuals"]), 6)
            moving_handle = next(v for v in report["inferred_visuals"] if v["name"] == "inferred_drawer_handle_drawer1")
            self.assertEqual(moving_handle["name"], "inferred_drawer_handle_drawer1")
            self.assertIsNotNone(ET.parse(root / "model.urdf").getroot().find("./link[@name='drawer1']/visual[@name='inferred_drawer_handle_drawer1']"))
            data = mujoco.MjData(after)
            panel = after.geom("inferred_drawer_front_drawer1").id
            mujoco.mj_forward(after, data)
            before_move = data.geom_xpos[panel].copy()
            data.qpos[0] = .05
            mujoco.mj_forward(after, data)
            self.assertAlmostEqual(np.linalg.norm(data.geom_xpos[panel] - before_move), .05)
            np.testing.assert_allclose(before.body_mass, after.body_mass)
            np.testing.assert_allclose(before.body_inertia, after.body_inertia)
            self.assertTrue(all(after.geom_contype[i] == 0 for i in range(after.ngeom)
                                if after.geom(i).name.startswith("inferred_")))
            complete_geometry(root, appearance, log=lambda _: None)
            self.assertEqual(mujoco.MjModel.from_xml_path(str(root / "model.xml")).ngeom, after.ngeom)
            unknown = dict(appearance, drawer_count=0)
            guessed = complete_geometry(root, unknown, log=lambda _: None)
            self.assertEqual(guessed.get("drawer_count_assumed"), 3)
            self.assertTrue(any(v.get("count_assumed") for v in guessed["inferred_visuals"]))
            self.assertEqual(mujoco.MjModel.from_xml_path(str(root / "model.xml")).ngeom, after.ngeom)
            single = complete_geometry(root, dict(appearance, drawer_count=1), log=lambda _: None)
            self.assertEqual({v["name"] for v in single["inferred_visuals"]},
                             {"inferred_drawer_front_drawer1", "inferred_drawer_handle_drawer1"})
            single_model = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
            single_data = mujoco.MjData(single_model)
            handle = single_model.geom("inferred_drawer_handle_drawer1").id
            mujoco.mj_forward(single_model, single_data)
            handle_before = single_data.geom_xpos[handle].copy()
            single_data.qpos[0] = .05
            mujoco.mj_forward(single_model, single_data)
            self.assertAlmostEqual(np.linalg.norm(single_data.geom_xpos[handle] - handle_before), .05)

    def test_hinged_box_gets_rectangular_visual_without_changing_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meshes").mkdir()
            outline = np.array([[0, 0], [.2, 0], [.2, .1], [0, .1]])
            vertices, faces = prism(outline, np.zeros(3), *np.eye(3), 0, .08)
            write_stl(root / "meshes/base.stl", vertices, faces)
            vertices, faces = prism(outline, np.zeros(3), *np.eye(3), 0, .004)
            write_stl(root / "meshes/lid.stl", vertices, faces)
            write_stl(root / "meshes/screen.stl", vertices, faces)
            links = [Link3D("base", [Visual("meshes/base.stl", (.5, .5, .5))]),
                     Link3D("lid", [Visual("meshes/lid.stl", (.5, .5, .5))], parent="base",
                            joint_type="revolute", limits=(-1, 1))]
            (root / "model.xml").write_text(mjcf("test", links))
            (root / "model.urdf").write_text(urdf("test", links))
            (root / "model3d.json").write_text(json.dumps({"base_thickness_mm": 80, "lid_thickness_m": .004}))
            before = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
            report = complete_geometry(root, {"category": "box", "object": "box"}, log=lambda _: None)
            after = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
            self.assertEqual({v["link"] for v in report["inferred_visuals"]}, {"base", "lid"})
            np.testing.assert_allclose(before.body_mass, after.body_mass)
            self.assertEqual(sum(after.geom_contype != 0), sum(before.geom_contype != 0))

    def test_generated_obj_has_mujoco_texture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            verts = np.array([[0., 0., 0.], [.1, 0, 0], [0, .1, 0], [0, 0, .1]])
            faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
            colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]])
            write_obj(root / "mesh.obj", verts, faces, colors)
            xml = ('<mujoco><asset><mesh name="x" file="mesh.obj"/>'
                   '<texture name="t" type="2d" file="mesh.png"/><material name="m" texture="t"/>'
                   '</asset><worldbody><geom type="mesh" mesh="x" material="m"/></worldbody></mujoco>')
            (root / "test.xml").write_text(xml)
            model = mujoco.MjModel.from_xml_path(str(root / "test.xml"))
            self.assertEqual(model.ntex, 1)
            self.assertIn("map_Kd mesh.png", (root / "mesh.mtl").read_text())

    def test_generated_mesh_does_not_replace_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meshes").mkdir()
            rect = np.array([[0, 0], [.2, 0], [.2, .1], [0, .1]])
            vertices, faces = prism(rect, np.zeros(3), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2], 0, .02)
            write_stl(root / "meshes/base.stl", vertices, faces)
            links = [Link3D("base", [Visual("meshes/base.stl", (.5, .5, .5))])]
            (root / "model.xml").write_text(mjcf("test", links))
            (root / "model.urdf").write_text(urdf("test", links))
            write_obj(root / "meshes/base_sam3d.obj", vertices, faces, np.full((len(vertices), 3), 150))
            before = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
            complete_geometry(root, {"object": "laptop", "category": "laptop"}, log=lambda _: None)
            for _ in range(2):
                add_visuals(root, {"base": ("meshes/base_sam3d.obj", [.5, .5, .5])})
            after = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
            np.testing.assert_allclose(before.body_mass, after.body_mass)
            self.assertEqual(after.ngeom, 2)
            self.assertEqual(sorted(after.geom_contype.tolist()), [0, 1])
            urdf_root = ET.parse(root / "model.urdf").getroot()
            self.assertEqual(urdf_root.find(".//collision/geometry/mesh").get("filename"), "meshes/base.stl")
            self.assertEqual(urdf_root.find(".//visual/material/texture").get("filename"), "meshes/base_sam3d.png")

    def test_depth_registration_is_bounded_and_does_not_rescale(self):
        rect = np.array([[0, 0], [.3, 0], [.3, .17], [0, .17]])
        verts, faces = prism(rect, np.zeros(3), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2], 0, .05)
        observed = surface_samples(verts, faces, n=5000, rng=np.random.default_rng(1)) + np.array([.008, -.005, .003])
        aligned, report = align_to_depth(verts, faces, observed)
        self.assertLess(report["after_median_mm"], report["before_median_mm"])
        self.assertLessEqual(report["center_displacement_mm"], 30.001)
        self.assertLessEqual(report["rotation_deg"], 20.001)
        np.testing.assert_allclose(np.linalg.norm(aligned[1] - aligned[0]), np.linalg.norm(verts[1] - verts[0]))

    def test_bevel_stays_inside_measured_box(self):
        rect = np.array([[0, 0], [0.3, 0], [0.3, 0.2], [0, 0.2]])
        verts, faces = prism(rect, np.zeros(3), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2], 0, 0.01)
        new, triangles = bevel_mesh(verts[faces])
        self.assertTrue(np.all(new >= verts.min(axis=0) - 1e-8))
        self.assertTrue(np.all(new <= verts.max(axis=0) + 1e-8))
        self.assertGreater(len(triangles), 0)

    def test_inferred_visual_preserves_collision_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meshes").mkdir()
            rect = np.array([[0, 0], [.3, 0], [.3, .2], [0, .2]])
            verts, faces = prism(rect, np.zeros(3), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2], 0, .01)
            write_stl(root / "meshes/base.stl", verts, faces)
            links = [Link3D("base", [Visual("meshes/base.stl", (.5, .5, .5))])]
            (root / "model.xml").write_text(mjcf("test", links))
            (root / "model.urdf").write_text(urdf("test", links))
            before = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
            complete_geometry(root, {"object": "laptop", "category": "laptop"}, log=lambda _: None)
            after = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
            np.testing.assert_allclose(before.body_mass, after.body_mass)
            np.testing.assert_allclose(before.body_inertia, after.body_inertia)
            self.assertEqual(list(after.geom_contype), [1, 0])
            urdf_root = ET.parse(root / "model.urdf").getroot()
            self.assertEqual(urdf_root.find(".//collision/geometry/mesh").get("filename"), "meshes/base.stl")
            self.assertEqual(urdf_root.find(".//visual/geometry/mesh").get("filename"), "meshes/base_infer.stl")
            complete_geometry(root, {"object": "laptop", "category": "laptop"}, log=lambda _: None)
            self.assertEqual(mujoco.MjModel.from_xml_path(str(root / "model.xml")).ngeom, 2)
            self.assertEqual(len(json.loads((root / "shape_inference.json").read_text())["inferred_visuals"]), 1)


if __name__ == "__main__":
    unittest.main()
