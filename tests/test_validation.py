"""validation.json (video2urdf/validation.py): the artifact verdict from the export checks, how it
combines with the agent's, and staleness when the model changes after it was checked."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from video2urdf import validation as V
from video2urdf.summary import write_summary

ROOT = Path(__file__).resolve().parent.parent


def row(check, status, detail=""):
    return {"check": check, "status": status, "detail": detail or f"{check} detail"}


GOOD = [row("files", "pass"), row("inertia", "pass"), row("joints", "pass"), row("collision", "pass"),
        row("mjcf:mujoco", "pass"), row("urdf:mujoco", "pass"), row("urdf:pybullet", "not_tested"),
        row("urdf:yourdfpy", "not_tested"), row("single-file URDF", "warn")]


def with_status(rows, check, status, detail=""):
    return [row(check, status, detail) if r["check"] == check else r for r in rows]


class ArtifactStatusTest(unittest.TestCase):
    def test_warnings_and_untested_loaders_still_pass(self):
        self.assertEqual(V.artifact_status(GOOD), ("checks_passed", []))

    def test_a_failing_format_lowers_to_review_and_is_named(self):
        rows = with_status(GOOD, "urdf:mujoco", "fail", "the overlap throws a part; mujoco 3.17.0")
        status, reasons = V.artifact_status(rows)
        self.assertEqual(status, "needs_review")
        self.assertEqual(len(reasons), 1)
        self.assertTrue(reasons[0].startswith("urdf:mujoco failed: the overlap throws a part"))

    def test_a_failing_record_check_lowers_to_review(self):
        self.assertEqual(V.artifact_status(with_status(GOOD, "joints", "fail"))[0], "needs_review")

    def test_missing_files_or_no_format_loading_fails(self):
        self.assertEqual(V.artifact_status(with_status(GOOD, "files", "fail"))[0], "failed")
        rows = with_status(with_status(GOOD, "mjcf:mujoco", "fail"), "urdf:mujoco", "fail")
        status, reasons = V.artifact_status(rows)
        self.assertEqual(status, "failed")
        self.assertIn("no format loads", reasons)

    def test_nothing_installed_is_not_tested_never_pass(self):
        rows = with_status(with_status(GOOD, "mjcf:mujoco", "not_tested"), "urdf:mujoco", "not_tested")
        self.assertEqual(V.artifact_status(rows)[0], "not_tested")
        self.assertEqual(V.artifact_status([])[0], "not_tested")

    def test_checker_skip_maps_to_not_tested(self):
        self.assertEqual(V.ROW_STATUS["skip"], "not_tested")


class CombineTest(unittest.TestCase):
    def test_worse_of_the_two(self):
        self.assertEqual(V.combine("checks_passed", "checks_passed"), "checks_passed")
        self.assertEqual(V.combine("checks_passed", "needs_review"), "needs_review")
        self.assertEqual(V.combine("needs_review", "checks_passed"), "needs_review")
        self.assertEqual(V.combine("checks_passed", "failed"), "failed")
        self.assertEqual(V.combine("failed", "checks_passed"), "failed")

    def test_untested_or_unknown_is_review(self):
        self.assertEqual(V.combine("checks_passed", "not_tested"), "needs_review")
        self.assertEqual(V.combine("checks_passed", None), "needs_review")
        self.assertEqual(V.combine(None, "checks_passed"), "needs_review")
        self.assertEqual(V.combine("failed", "not_tested"), "failed")


def make_run(tmp: Path) -> Path:
    """A run folder with a model and a viewer download (contents don't matter: the checker is faked)."""
    run = Path(tmp) / "run"
    m = run / "model"
    (m / "meshes").mkdir(parents=True)
    (m / "model.urdf").write_text("<robot name='x'/>")
    (m / "model.xml").write_text("<mujoco/>")
    (m / "meshes" / "base.stl").write_bytes(b"solid x\nendsolid x\n")
    v = run / "viewer" / "models" / "x"
    v.mkdir(parents=True)
    for f in ("model.urdf", "model.xml"):
        (v / f).write_text((m / f).read_text())
    with zipfile.ZipFile(v / "model.zip", "w") as z:
        z.write(m / "model.urdf", "model.urdf")
    return run


class FreshnessTest(unittest.TestCase):
    def validate(self, run, rows=GOOD):
        with mock.patch.object(V, "run_checker", return_value=([dict(r) for r in rows], None)) as rc:
            v = V.validate_run(run, "x")
        return v, rc

    def test_checks_the_viewer_download_and_records_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(tmp)
            v, rc = self.validate(run)
            self.assertEqual(rc.call_args[0][0], run / "viewer" / "models" / "x")
            self.assertEqual(rc.call_args[0][1], run / "model")
            self.assertEqual(v["target"], "viewer/models/x")
            self.assertEqual(set(v["files"]), {"model/model.urdf", "model/model.xml", "model/meshes",
                                               "viewer/models/x/model.zip", "viewer/models/x/model.urdf",
                                               "viewer/models/x/model.xml"})
            self.assertEqual(json.loads((run / "validation.json").read_text())["artifact_status"], "checks_passed")
            self.assertEqual(V.freshness(run), {"state": "current", "changed": []})
            self.assertEqual(V.current_artifact_status(run)[0], "checks_passed")

    def test_without_a_viewer_the_model_folder_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(tmp)
            (run / "viewer" / "models" / "x" / "model.zip").unlink()
            v, rc = self.validate(run)
            self.assertEqual(rc.call_args[0][0], run / "model")
            self.assertEqual(v["target"], "model")

    def test_a_later_change_makes_it_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(tmp)
            self.validate(run)
            (run / "model" / "model.xml").write_text("<mujoco model='changed'/>")
            (run / "model" / "meshes" / "lid.stl").write_bytes(b"solid y\nendsolid y\n")
            fresh = V.freshness(run)
            self.assertEqual(fresh["state"], "stale")
            self.assertEqual(fresh["changed"], ["model/model.xml (changed)", "model/meshes (changed)"])
            status, reasons, _ = V.current_artifact_status(run)
            self.assertEqual(status, "not_tested")
            self.assertIn("stale", reasons[0])
            (run / "model" / "model.urdf").unlink()
            self.assertIn("model/model.urdf (gone)", V.freshness(run)["changed"])

    def test_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(V.freshness(Path(tmp))["state"], "missing")
            self.assertEqual(V.current_artifact_status(Path(tmp))[0], "not_tested")

    def test_rezipping_the_same_files_is_not_a_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.zip", Path(tmp) / "b.zip"
            for p, when in ((a, (2020, 1, 1, 0, 0, 0)), (b, (2026, 11, 2, 12, 0, 0))):
                with zipfile.ZipFile(p, "w") as z:
                    z.writestr(zipfile.ZipInfo("model.urdf", when), "<robot/>")
            self.assertNotEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(V.digest(a), V.digest(b))

    def test_checker_that_did_not_finish_is_not_tested(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(tmp)
            with mock.patch.object(V, "run_checker", return_value=([], "check_exports.py took longer than 90 s")):
                v = V.validate_run(run, "x")
            self.assertEqual(v["artifact_status"], "not_tested")
            self.assertIn("took longer", v["reasons"][0])


class SummaryTest(unittest.TestCase):
    def test_both_verdicts_and_the_failing_format_in_summary_md(self):
        rows = with_status(GOOD, "urdf:mujoco", "fail", "the overlap throws a part; lid/world 13 mm")
        v = {"rows": rows}
        s = {"name": "x", "source": "x.mov", "status": "needs_review", "agent_status": "checks_passed",
             "artifact_status": "needs_review", "steps": [], "seconds": 1,
             "validation": {"state": "current", "changed": [], "target": "viewer/models/x",
                            "counts": {"pass": 6, "warn": 1, "fail": 1, "not_tested": 2},
                            "problems": V.problem_rows(v)}}
        with tempfile.TemporaryDirectory() as tmp:
            write_summary(Path(tmp), s)
            md = (Path(tmp) / "summary.md").read_text()
            js = json.loads((Path(tmp) / "summary.json").read_text())
        self.assertIn("**Status: needs review**", md)
        self.assertIn("- Agent: checks passed", md)
        self.assertIn("- Downloads: needs review", md)
        self.assertIn("  - FAIL urdf:mujoco: the overlap throws a part; lid/world 13 mm", md)
        self.assertIn("  - WARN single-file URDF:", md)
        self.assertLess(md.index("FAIL urdf:mujoco"), md.index("WARN single-file URDF"))
        self.assertEqual((js["agent_status"], js["artifact_status"]), ("checks_passed", "needs_review"))


def _has_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_mujoco() and (ROOT / "docs" / "models" / "notebook" / "model.zip").exists(),
                     "needs MuJoCo and the published models")
class PublishedModelsTest(unittest.TestCase):
    """The real checker on two published downloads (validation.json goes to a temp folder)."""

    def check(self, name):
        target = ROOT / "docs" / "models" / name
        with tempfile.TemporaryDirectory() as tmp:
            v = V.validate(target, Path(tmp), [target / f for f in ("model.zip", "model.urdf", "model.xml")])
            self.assertEqual(V.freshness(Path(tmp))["state"], "current")
        self.assertFalse((target / "validation.json").exists())
        return v, {r["check"]: r["status"] for r in v["rows"]}

    def test_notebook_urdf_loads_in_mujoco_after_the_root_trim(self):
        # The side-view laptop's URDF used to fail MuJoCo's URDF import (the lid started 13 mm inside
        # the base); video2urdf/joint_clearance.py now trims the root's URDF collision. A failing
        # format lowering the status is covered with a fake checker in test_run_pipeline.py.
        v, st = self.check("notebook")
        self.assertEqual(st["urdf:mujoco"], "pass")
        self.assertEqual(st["mjcf:mujoco"], "pass")
        self.assertNotIn("fail", st.values())
        self.assertEqual(v["artifact_status"], "checks_passed")

    def test_laptop_passes_or_warns_only(self):
        v, st = self.check("laptop_lidar")
        self.assertNotIn("fail", st.values())
        self.assertEqual(v["artifact_status"], "checks_passed")


if __name__ == "__main__":
    unittest.main()
