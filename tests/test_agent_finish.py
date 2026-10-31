import json
import numpy as np
import tempfile
import unittest
from pathlib import Path

from video2urdf.agent import Agent


def agent(track_dir: Path, model=True, fit=True, warnings=()) -> Agent:
    a = Agent.__new__(Agent)  # no video or clients: only the finish logic
    a.track_dir, a.out = track_dir, track_dir / "agent"
    a.out.mkdir(parents=True, exist_ok=True)
    if model:
        (track_dir / "model.xml").write_text("<mujoco/>")
    a.last_fit = object() if fit else None
    a.open_warnings, a.finish_refused, a.turns_left, a.report = list(warnings), False, 10, None
    return a


class FinishTest(unittest.TestCase):
    def test_clean_fit_finishes(self):
        with tempfile.TemporaryDirectory() as d:
            a = agent(Path(d))
            self.assertEqual(a.tool_finish("ok"), "done")
            self.assertTrue(a.report.startswith("Automatic checks: checks passed"))
            self.assertEqual(json.loads((a.out / "status.json").read_text())["status"], "checks_passed")

    def test_warning_refused_once_then_needs_review(self):
        with tempfile.TemporaryDirectory() as d:
            a = agent(Path(d), warnings=["WARNING: the base mask holds the plate"])
            self.assertIn("Not finished yet", a.tool_finish("first"))
            self.assertIsNone(a.report)
            self.assertEqual(a.tool_finish("NEEDS REVIEW: base"), "done")
            status = json.loads((a.out / "status.json").read_text())
            self.assertEqual(status["status"], "needs_review")
            self.assertTrue(status["finish_refused_once"])

    def test_no_refusal_near_the_end(self):
        with tempfile.TemporaryDirectory() as d:
            a = agent(Path(d), warnings=["WARNING: x"])
            a.turns_left = 1
            self.assertEqual(a.tool_finish("r"), "done")

    def test_no_model_is_failed(self):
        with tempfile.TemporaryDirectory() as d:
            a = agent(Path(d), model=False)
            a.turns_left = 0
            a.tool_finish("r")
            self.assertEqual(json.loads((a.out / "status.json").read_text())["status"], "failed")

    def test_retracked_after_fit_needs_review(self):
        with tempfile.TemporaryDirectory() as d:
            a = agent(Path(d), fit=False)
            a.turns_left = 0
            a.tool_finish("r")
            self.assertEqual(json.loads((a.out / "status.json").read_text())["status"], "needs_review")


if __name__ == "__main__":
    unittest.main()


class PlanarFitWarningTest(unittest.TestCase):
    """A planar fit on a depth capture with an assumed scale and a link that misses its mask must
    not pass the finish checks (a phone-stand arm replayed at IoU 0.04 once did)."""

    def run_fit(self, ious, capture=True, scale_note="ASSUMED (3200 px/m)"):
        from unittest import mock
        from video2urdf import agent as A

        class FakeFit:
            def __init__(self):
                self.ious = {k: np.array(v, float) for k, v in ious.items()}

            def summary(self):
                return "fit summary"

        with tempfile.TemporaryDirectory() as d:
            a = agent(Path(d), fit=False)
            a.tracked, a.capture, a.px_per_m, a.scale_note = True, (Path(d) if capture else None), 3200.0, scale_note
            a.video, a.size = Path(d) / "v.mp4", (1920, 1440)
            a.replay_sheet = lambda result: np.zeros((2, 2, 3), np.uint8)
            a.show = lambda *args, **kw: None
            with mock.patch.object(A, "fit", lambda *args, **kw: FakeFit()):
                out = a.tool_fit(parents={"arm": "pole", "forearm": "arm"})
            status = a.check_status()
        return out, a.open_warnings, status

    def test_bad_planar_fit_on_depth_capture_needs_review(self):
        out, warnings, (status, _) = self.run_fit({"pole": [0.8, 0.7], "arm": [0.04, 0.05]})
        self.assertEqual(status, "needs_review")
        self.assertTrue(any("planar approximation" in w for w in warnings))
        self.assertTrue(any("arm overlaps" in w for w in warnings))
        self.assertIn("WARNING", out)

    def test_good_planar_fit_with_measured_scale_passes(self):
        _, warnings, (status, _) = self.run_fit({"lid": [0.6, 0.7]}, capture=False, scale_note="card: 5094 px/m")
        self.assertEqual(warnings, [])
        self.assertEqual(status, "checks_passed")
