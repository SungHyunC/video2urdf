import json
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
