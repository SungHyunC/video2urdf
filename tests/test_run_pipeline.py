"""The one-command pipeline's glue: run folders, fallback part choice, summaries, and a local
check-only run on a generated clip (ffmpeg and OpenCV only, no model calls)."""
import json
import tempfile
import unittest
from pathlib import Path

from video2urdf.fallback import choose_floor, object_parts
from video2urdf.run import Options, run, slug, unique_dir
from video2urdf.summary import measured_vs_estimated, mjcf_joints, write_summary

try:  # discover -s tests puts tests/ on the path; python -m unittest tests.x doesn't
    from test_ingest import make_stray, make_video
except ImportError:
    from tests.test_ingest import make_stray, make_video

MJCF = """<mujoco model="m"><compiler angle="radian"/><worldbody><body name="base">
<geom type="box" size="0.1 0.1 0.01"/><body name="lid" pos="0 0.1 0.01">
<joint name="lid_joint" type="hinge" axis="1 0 0" range="-1.5708 0.5" frictionloss="0.2"/>
<geom type="box" size="0.1 0.005 0.1"/></body>
<body name="drawer"><joint name="drawer_joint" type="slide" axis="0 1 0" range="0 0.25"/>
<geom type="box" size="0.05 0.05 0.05"/></body></body></worldbody></mujoco>"""


class FolderTest(unittest.TestCase):
    def test_runs_never_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "laptop"
            self.assertEqual(unique_dir(base), base)
            base.mkdir()
            self.assertEqual(unique_dir(base), base)  # empty: reuse
            (base / "summary.json").write_text("{}")
            self.assertEqual(unique_dir(base).name, "laptop_2")
            (Path(tmp) / "laptop_2").mkdir()
            self.assertEqual(unique_dir(base).name, "laptop_3")

    def test_slug(self):
        self.assertEqual(slug("IMG_0042 (1).MOV"), "img_0042_1_mov")
        self.assertEqual(slug("my scan!"), "my_scan")


class FallbackNamesTest(unittest.TestCase):
    def test_floor_by_role_then_name_and_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            names = ["desk_lamp_base", "pole", "white_table", "hand"]
            self.assertEqual(choose_floor(names, d, lambda n: n == "white_table"), "white_table")
            self.assertIsNone(choose_floor(["desk_lamp_base", "pole"], d, lambda n: False))
            (d / "roles.json").write_text(json.dumps({"floor": "pole"}))
            self.assertEqual(choose_floor(names, d), "pole")  # the agent's own choice wins

    def test_object_parts_skip_hands_cards_and_the_floor(self):
        names = ["floor", "cardboard_box_lid", "box_front_wall", "loyalty_card", "left_hand", "whole_object"]
        self.assertEqual(object_parts(names, "floor"), ["cardboard_box_lid", "box_front_wall"])


class SummaryTest(unittest.TestCase):
    def test_joints_from_mjcf(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "model.xml"
            xml.write_text(MJCF)
            j = {x["name"]: x for x in mjcf_joints(xml)}
            self.assertEqual(j["lid_joint"]["type"], "revolute")
            self.assertEqual(j["lid_joint"]["range"], [-90.0, 28.6])
            self.assertEqual(j["drawer_joint"]["range"], [0.0, 250.0])
            self.assertEqual(j["drawer_joint"]["unit"], "mm")

    def test_measured_vs_estimated_says_what_was_assumed(self):
        s = {"input": {"kind": "video"}, "path": {"chosen": "planar"},
             "model": {"measurement_source": {"metric": "assumed, not measured"},
                       "physics": {"joints": {"lid_joint": {"measured": False}}}}}
        text = " ".join(measured_vs_estimated(s))
        self.assertIn("ASSUMED scale", text)
        self.assertIn("estimated (no hand-free", text)
        s = {"input": {"kind": "stray"}, "path": {"chosen": "3d"}, "fallback": {"used": True}, "model": {}}
        text = " ".join(measured_vs_estimated(s))
        self.assertIn("LiDAR", text)
        self.assertIn("rigid fallback", text)

    def test_failed_run_still_writes_a_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = {"name": "x", "source": "x.txt", "status": "failed", "issues": ["unusable input: not a video"],
                 "input": {"kind": "unknown", "usable": False}, "steps": [], "seconds": 0.1}
            write_summary(Path(tmp), s)
            self.assertIn("unusable input", (Path(tmp) / "summary.md").read_text())
            self.assertEqual(json.loads((Path(tmp) / "summary.json").read_text())["status"], "failed")


class StepsTest(unittest.TestCase):
    """The glue after the agent, with the agent and the repo scripts replaced by fakes."""

    def run_with(self, agent_writes_model: bool, agent_error: str | None = None):
        from unittest import mock

        from video2urdf import run as R

        def fake_agent(self, video, capture):
            self.model_dir.mkdir(parents=True, exist_ok=True)
            if agent_writes_model:
                (self.model_dir / "model.xml").write_text(MJCF)
                (self.model_dir / "model.urdf").write_text("<robot name='m'/>")
                (self.model_dir / "agent").mkdir(exist_ok=True)
                (self.model_dir / "agent" / "status.json").write_text(json.dumps(
                    {"status": "checks_passed", "issues": []}))
            return {"report": "ok", "error": agent_error, "last_fit": agent_writes_model, "scale_note": "x",
                    "px_per_m": None, "status": {"status": "checks_passed", "issues": []} if agent_writes_model else None}

        def fake_script(self, name, args):
            if name == "physics":
                raise RuntimeError("add_physics.py exited 1")
            return {"_note": "ok"}

        with tempfile.TemporaryDirectory() as tmp:
            src = make_video(Path(tmp) / "clip.mp4", frames=60, fps=30, size=(320, 240))
            with mock.patch.object(R.Run, "_agent", fake_agent), mock.patch.object(R.Run, "script", fake_script), \
                    mock.patch.object(R.Run, "_viewer", lambda self, video: {"_note": "skipped in test"}), \
                    mock.patch.object(R.Run, "_fallback", lambda self, *a: {"ok": False, "error": "no tracks"}):
                return run(src, Options(out=Path(tmp) / "runs", look=False, design=False, path="planar"))

    def test_a_failed_later_step_keeps_the_model(self):
        s = self.run_with(agent_writes_model=True)
        self.assertEqual(s["status"], "checks_passed")
        self.assertTrue(any("physics" in i for i in s["issues"]))
        steps = {x["name"]: x["status"] for x in s["steps"]}
        self.assertEqual((steps["agent"], steps["physics"], steps["fallback"]), ("ok", "failed", "skipped"))
        self.assertEqual([j["name"] for j in s["model"]["joints"]], ["lid_joint", "drawer_joint"])

    def test_no_model_and_no_fallback_fails(self):
        s = self.run_with(agent_writes_model=False)
        self.assertEqual(s["status"], "failed")
        self.assertTrue(s["fallback"]["used"])
        self.assertIn("no model could be built", s["issues"])

    def test_agent_crash_after_a_model_keeps_it_for_review(self):
        s = self.run_with(agent_writes_model=True, agent_error="APIConnectionError: gone")
        self.assertFalse(s["fallback"]["used"])
        self.assertTrue(any("stopped with an error after building a model" in i for i in s["issues"]))


class CheckOnlyRunTest(unittest.TestCase):
    def test_plain_video_check_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = make_video(Path(tmp) / "IMG_0001.MOV", frames=90, fps=60, size=(320, 240), rotation=90)
            s = run(src, Options(out=Path(tmp) / "runs", look=False, check_only=True))
            self.assertEqual(s["status"], "checked")
            self.assertEqual(s["input"]["kind"], "video")
            self.assertEqual(s["input"]["size"], [240, 320])
            self.assertEqual(s["input"]["frames"], 45)
            self.assertIn(s["path"]["chosen"], ("planar", "3d"))
            self.assertTrue((Path(s["run_dir"]) / "summary.md").exists())

    def test_capture_zip_check_only_routes_3d(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            make_stray(Path(tmp) / "src" / "abc", frames=40, size=(320, 240))  # 0.7 s at 60 fps: long enough to check
            z = Path(tmp) / "scan.zip"
            with zipfile.ZipFile(z, "w") as zf:
                for f in (Path(tmp) / "src").rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(Path(tmp) / "src"))
            s = run(z, Options(out=Path(tmp) / "runs", look=False, check_only=True, path="auto"))
            self.assertEqual(s["input"]["kind"], "stray")
            self.assertEqual(s["path"]["chosen"], "3d")

    def test_unusable_input_fails_with_a_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "notes.txt").write_text("hello")
            s = run(Path(tmp) / "notes.txt", Options(out=Path(tmp) / "runs", look=False))
            self.assertEqual(s["status"], "failed")
            self.assertTrue(any("unusable input" in i for i in s["issues"]))


if __name__ == "__main__":
    unittest.main()
