import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video2urdf.assessment import assess
from video2urdf.track_checks import overclosed_hinge_warning
from scripts.infer_appearance import infer_appearance, refresh_fit_warnings


class AssessmentTest(unittest.TestCase):
    def test_flat_static_drawer_warns_about_upright_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model3d.json').write_text(json.dumps({'static_envelope': True,
                                                          'generated_size_mm': [2500, 2400, 150]}))
            (root / 'shape_inference.json').write_text(json.dumps({'category': 'drawer'}))
            self.assertTrue(any('very flat' in w for w in assess(root)['warnings']))
            (root / 'model3d.json').write_text(json.dumps({'static_envelope': True,
                                                          'generated_size_mm': [400, 250, 500]}))
            self.assertFalse(any('very flat' in w for w in assess(root)['warnings']))

    def test_experimental_alignment_cannot_inherit_original_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model3d.json').write_text('{}')
            (root / 'experimental_alignment.json').write_text(json.dumps({'warning': 'Experimental, not physics-ready.'}))
            report = assess(root)
            self.assertEqual(report['status'], 'review')
            self.assertIn('Experimental, not physics-ready.', report['warnings'])
            self.assertTrue(any('not this experimental alignment' in w for w in report['warnings']))

    def test_hinge_overclose_detects_both_axis_signs(self):
        import math
        for sign in (1, -1):
            q = [math.radians(sign * x) for x in [0, 10, 20, 30, 40, 50, 60, 70, 85, 110]]
            self.assertIn('beyond its estimated closed position',
                          overclosed_hinge_warning({'q': q, 'opening_on_reference_deg': 85}))
            self.assertIsNone(overclosed_hinge_warning({'q': q, 'opening_on_reference_deg': 110}))
        self.assertIsNone(overclosed_hinge_warning({'q': [None], 'opening_on_reference_deg': 85}))

    def test_static_appearance_records_sampled_visual_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'static.json').write_text(json.dumps({'part': 'object', 'shape': 'box'}))
            (root / 'appearance.json').write_text(json.dumps({'category': 'box'}))
            (root / 'model.xml').write_text('<mujoco><worldbody><body name="base"/></worldbody></mujoco>')
            check = {'links': {'base': {'frames': 24, 'inferred_iou': .74}}}
            with patch('scripts.infer_appearance.link_masks', return_value={'base': ['object']}), \
                 patch('scripts.infer_appearance.compare_visuals', return_value=check), \
                 patch('scripts.infer_appearance.refresh_fit_warnings'), \
                 patch('scripts.infer_appearance.assess', return_value={}):
                infer_appearance(root, root / 'video.mp4', reuse=True, texture=False)
            result = json.loads((root / 'shape_inference.json').read_text())
            self.assertEqual(result['visual_coverage_check']['links']['base']['inferred_iou'], .74)

    def test_static_envelope_and_unproven_physics_need_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model3d.json').write_text(json.dumps({'static_envelope': True,
                'mask_coverage': {'base': .95}, 'mask_coverage_frames': {'base': 24}, 'video_frames': 24}))
            (root / 'physics.json').write_text('{}')
            warnings = assess(root)['warnings']
            self.assertTrue(any('hidden surfaces' in w for w in warnings))
            self.assertTrue(any('estimated from video' in w for w in warnings))

    def test_legacy_fit_reports_low_sampled_coverage_of_retained_visual(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text('{}')
            (root / "shape_inference.json").write_text(json.dumps({"active": True,
                "visual_coverage_check": {"links": {"drawer1": {"frames": 60,
                    "baseline": .55, "inferred": .55, "iou_informative": False}}}}))
            warnings = assess(root)["warnings"]
            self.assertTrue(any("55%" in w and "60 sampled frames" in w for w in warnings))
            (root / "shape_inference.json").write_text(json.dumps({"active": False,
                "visual_coverage_check": {"links": {"drawer1": {"frames": 60,
                    "baseline": .9, "inferred": .2}}}}))
            self.assertFalse(any("retained visual covers" in w for w in assess(root)["warnings"]))

    def test_large_hinge_range_needs_video_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text(json.dumps({"mask_coverage": {"lid": .9},
                "mask_coverage_frames": {"lid": 100}, "video_frames": 100,
                "hinge_range_deg": [-166.5, 94.3]}))
            report = assess(root)
            self.assertEqual(report["status"], "review")
            self.assertTrue(any("Hinge range spans 261°" in w for w in report["warnings"]))

    def test_mesh_with_no_observed_texture_requires_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text('{}')
            (root / "texture.json").write_text(json.dumps({"meshes": {"lid_shell": {"faces": 6, "faces_seen": 0}}}))
            self.assertTrue(any("lid_shell" in w and "no face" in w for w in assess(root)["warnings"]))

    def test_manual_repair_cannot_be_reported_as_autonomous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text('{}')
            (root / "repair.json").write_text(json.dumps({"method": "manual"}))
            self.assertTrue(any("manually repaired" in w for w in assess(root)["warnings"]))

    def test_manual_segmentation_selection_is_not_an_autonomous_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model3d.json').write_text('{}')
            (root / 'manual_selection.json').write_text('{}')
            self.assertTrue(any('manually selected' in w for w in assess(root)['warnings']))

    def test_low_visual_iou_requires_review_even_if_coverage_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text('{}')
            (root / "shape_inference.json").write_text(json.dumps({"active": True,
                "visual_coverage_check": {"links": {"base": {"inferred_iou": .41, "iou_informative": True}}}}))
            self.assertTrue(any("visual-mask IoU" in w for w in assess(root)["warnings"]))

    def test_rejected_candidate_iou_does_not_replace_retained_visual_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text('{}')
            shape = {"active": False, "visual_coverage_check": {"links": {"base": {
                "baseline_iou": .8, "inferred_iou": .3, "iou_informative": True}}}}
            (root / "shape_inference.json").write_text(json.dumps(shape))
            self.assertFalse(any("visual-mask IoU" in w for w in assess(root)["warnings"]))
            shape["visual_coverage_check"]["links"]["base"].update(baseline_iou=.4, inferred_iou=.8)
            (root / "shape_inference.json").write_text(json.dumps(shape))
            self.assertTrue(any("IoU is 0.40" in w for w in assess(root)["warnings"]))

    def test_missing_reference_angle_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text('{}')
            (root / "roles.json").write_text('{}')
            (root / "hinge.json").write_text(json.dumps({"reference_frame": 1, "q": [0., None]}))
            self.assertTrue(any("Reference frame 1" in w for w in assess(root)["warnings"]))

    def test_implausibly_thick_laptop_base_requires_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text(json.dumps({"base_thickness_mm": 78,
                                                           "base_thickness_measured": True,
                                                           "plate_size_mm": [223, 107]}))
            (root / "appearance.json").write_text(json.dumps({"category": "laptop"}))
            warnings = assess(root)["warnings"]
            self.assertTrue(any("base thickness exceeds" in w for w in warnings))

    def test_unknown_shape_completion_requires_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text(json.dumps({"mask_coverage": {"lid": .9},
                                                           "mask_coverage_frames": {"lid": 90},
                                                           "video_frames": 100}))
            (root / "shape_inference.json").write_text(json.dumps({"category": "other", "inferred_visuals": []}))
            report = assess(root)
            self.assertEqual(report["status"], "review")
            self.assertTrue(any("No object-specific shape" in warning for warning in report["warnings"]))

    def test_reuse_preserves_selected_shape_and_its_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shape = {"category": "book", "active": True,
                     "inferred_visuals": [{"link": "lid", "silhouette_blend": 0.5}],
                     "visual_coverage_attempts": [{"blend": 0}, {"blend": 0.5}]}
            (root / "appearance.json").write_text(json.dumps({"category": "book"}))
            (root / "shape_inference.json").write_text(json.dumps(shape))
            (root / "model.xml").write_text('<mujoco><worldbody><body><geom mesh="lid_infer"/></body></worldbody></mujoco>')
            with patch("scripts.infer_appearance.link_masks", return_value={"lid": []}), \
                 patch("scripts.infer_appearance.complete_geometry") as complete, \
                 patch("scripts.infer_appearance.texture_model"), \
                 patch("scripts.infer_appearance.refresh_fit_warnings"), \
                 patch("scripts.infer_appearance.assess", return_value={}):
                infer_appearance(root, root / "video.mp4", reuse=True)
            complete.assert_not_called()
            self.assertEqual(json.loads((root / "shape_inference.json").read_text()), shape)

    def test_refresh_keeps_agent_warnings_and_adds_raw_track_findings(self):
        class Detection:
            def usable(self, w, h):
                return True
            def full_mask(self, h, w):
                import numpy as np
                return np.zeros((h, w), bool)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "roles.json").write_text(json.dumps({"floor": None, "base": "base", "plate": "plate"}))
            (root / "model3d.json").write_text(json.dumps({"fit_warnings": ["WARNING: existing"]}))
            tracks = {"base": {0: Detection()}, "plate": {0: Detection()}}
            with patch("scripts.infer_appearance.video_info", return_value=((20, 20), 1, 30)), \
                 patch("scripts.infer_appearance.load_parts", return_value=(["base", "plate"], tracks)), \
                 patch("scripts.infer_appearance.Agent.mask_checks", return_value=["WARNING: fragment"]):
                refresh_fit_warnings(root, root / "video.mp4")
            warnings = json.loads((root / "model3d.json").read_text())["fit_warnings"]
            self.assertEqual(warnings, ["WARNING: existing", "WARNING: fragment"])

    def test_missing_coverage_frame_count_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text(json.dumps({"mask_coverage": {"lid": .99}}))
            report = assess(root)
            self.assertEqual(report["status"], "review")
            self.assertTrue(any("not recorded" in w for w in report["warnings"]))

    def test_nearly_closed_reference_with_open_indicator_needs_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text(json.dumps({"mask_coverage": {"lid": .9},
                                                           "mask_coverage_frames": {"lid": 90},
                                                           "video_frames": 100}))
            (root / "roles.json").write_text(json.dumps({"open_indicator": "keyboard"}))
            (root / "hinge.json").write_text(json.dumps({"opening_on_reference_deg": 1.0}))
            result = assess(root)
            self.assertEqual(result["status"], "review")
            self.assertTrue(any("reference opening" in w for w in result["warnings"]))

    def test_suspicious_plate_thickness_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text(json.dumps({"mask_coverage": {"base": .86, "lid": .94},
                                                           "mask_coverage_frames": {"base": 80, "lid": 80},
                                                           "video_frames": 100,
                                                           "plate_size_mm": [305, 209],
                                                           "plate_thickness_mm": 24.1,
                                                           "plate_thickness_clipped": False}))
            report = assess(root)
            self.assertEqual(report["status"], "review")
            self.assertTrue(any("thickness exceeds" in x for x in report["warnings"]))

    def test_coverage_from_only_a_few_frames_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text(json.dumps({"mask_coverage": {"lid": .99},
                "mask_coverage_frames": {"lid": 10}, "video_frames": 100}))
            report = assess(root)
            self.assertTrue(any("10/100" in x for x in report["warnings"]))

    def test_adjusted_visual_warns_about_collision_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text(json.dumps({"mask_coverage": {"base": .9},
                "mask_coverage_frames": {"base": 100}, "video_frames": 100}))
            (root / "shape_inference.json").write_text(json.dumps({"category": "laptop", "active": True,
                "visual_thickness_assumed": "adjusted"}))
            report = assess(root)
            self.assertTrue(any("differs from collision geometry" in w for w in report["warnings"]))

    def test_invalid_base_thickness_is_not_treated_as_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text(json.dumps({"mask_coverage": {"base": .9},
                "mask_coverage_frames": {"base": 100}, "video_frames": 100, "base_thickness_mm": -83.8}))
            report = assess(root)
            self.assertEqual(report["status"], "review")
            self.assertTrue(any("Base thickness is invalid" in w for w in report["warnings"]))

    def test_accepted_sam3d_still_needs_visual_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text(json.dumps({"mask_coverage": {"lid": .99},
                "mask_coverage_frames": {"lid": 100}, "video_frames": 100}))
            (root / "sam3d.json").write_text(json.dumps({"lid": {"accepted": True}}))
            report = assess(root)
            self.assertEqual(report["status"], "review")
            self.assertTrue(any("texture and hidden surfaces" in x for x in report["warnings"]))

    def test_flags_bad_coverage_and_assumed_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model3d.json").write_text(json.dumps({"mask_coverage": {"base": .16, "lid": .93},
                                                           "mask_coverage_frames": {"base": 80, "lid": 80},
                                                           "video_frames": 100,
                                                           "base_thickness_measured": False,
                                                           "fit_warnings": ["WARNING: plate tracking drifted"]}))
            (root / "sam3d.json").write_text(json.dumps({"lid": {"accepted": False}}))
            (root / "sam3d_status.json").write_text(json.dumps({"state": "failed"}))
            report = assess(root)
            self.assertEqual(report["status"], "review")
            self.assertEqual(len(report["warnings"]), 5)
            self.assertTrue((root / "quality.json").exists())


if __name__ == "__main__":
    unittest.main()
