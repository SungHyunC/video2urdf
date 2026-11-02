import hashlib
import io
import json
import os
import socket
import tempfile
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from video2urdf import reproduce as R


def fake_capture_zip(path: Path, video: bytes = b"video") -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("cap/rgb.mp4", video)
        z.writestr("cap/odometry.csv", "timestamp, frame, x, y, z, qx, qy, qz, qw\n")
        z.writestr("cap/camera_matrix.csv", "1,0,0\n0,1,0\n0,0,1\n")
        for i in range(3):
            z.writestr(f"cap/depth/{i:06d}.png", b"png")
            z.writestr(f"cap/confidence/{i:06d}.png", b"png")
    return path


def fake_run(folder: Path) -> Path:
    folder.mkdir()
    (folder / "tracks.sam.txt").write_text("<0f>1<|box;x1=0;y1=0;x2=1;y2=1;w=2;h=2;c=0.9|>\n")
    (folder / "parts.json").write_text(json.dumps({"1": {"name": "lid", "prompt": "lid"}}))
    (folder / "roles.json").write_text(json.dumps({"floor": "floor", "base": "base", "plate": "lid"}))
    (folder / "physics_estimate.json").write_text(json.dumps({"parts": [], "released": []}))
    return folder


class BundleTest(unittest.TestCase):
    def test_same_files_make_the_same_zip_and_it_checks_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run, capture = fake_run(tmp / "run"), fake_capture_zip(tmp / "capture.zip")
            a = R.build_bundle(run, capture, tmp / "a.zip", "data/x/run")
            R.build_bundle(run, capture, tmp / "b.zip", "data/x/run")
            self.assertEqual((tmp / "a.zip").read_bytes(), (tmp / "b.zip").read_bytes())
            self.assertEqual(a["capture"]["depth_frames"], 3)
            self.assertEqual(a["capture"]["files"]["rgb.mp4"], hashlib.sha256(b"video").hexdigest())
            root, manifest = R.open_bundle(tmp / "a.zip", tmp / "work")
            self.assertEqual(manifest, a)
            self.assertEqual((root / "roles.json").read_text(), (run / "roles.json").read_text())
            self.assertEqual(R.open_capture(capture, tmp / "cache", manifest).name, "cap")

    def test_a_damaged_or_incomplete_bundle_stops_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run = fake_run(tmp / "run")
            R.build_bundle(run, fake_capture_zip(tmp / "capture.zip"), tmp / "a.zip", "data/x/run")
            folder = tmp / "unpacked"
            zipfile.ZipFile(tmp / "a.zip").extractall(folder)
            root = folder / R.BUNDLE_NAME
            (root / "tracks.sam.txt").write_text("edited")
            with self.assertRaisesRegex(R.ReproduceError, "tracks.sam.txt doesn't match its manifest"):
                R.open_bundle(folder, tmp / "work")
            manifest = json.loads((root / "MANIFEST.json").read_text())
            del manifest["files"]["physics_estimate.json"]
            (root / "MANIFEST.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(R.ReproduceError, "won't ask a model instead"):
                R.open_bundle(folder, tmp / "work")

    def test_another_capture_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest = R.build_bundle(fake_run(tmp / "run"), fake_capture_zip(tmp / "capture.zip"), tmp / "a.zip", "x")
            other = fake_capture_zip(tmp / "other.zip", video=b"another video")
            with self.assertRaisesRegex(R.ReproduceError, "isn't the capture these tracks were made on"):
                R.open_capture(other, tmp / "cache", manifest)
            zipfile.ZipFile(other).extractall(tmp / "other")  # as a folder: its video is checked instead
            with self.assertRaisesRegex(R.ReproduceError, "rgb.mp4 .* isn't the one these tracks were made on"):
                R.open_capture(tmp / "other", tmp / "cache", manifest)


class DownloadTest(unittest.TestCase):
    def test_unpublished_bundle_names_the_release_and_the_local_option(self):
        from scripts.reproduce import fetch_inputs
        missing = urllib.error.HTTPError(R.BUNDLE_URL, 404, "Not Found", {}, io.BytesIO())
        with tempfile.TemporaryDirectory() as tmp, patch("urllib.request.urlopen", side_effect=missing):
            args = SimpleNamespace(bundle=R.BUNDLE_URL, capture=R.CAPTURE_URL, cache=Path(tmp))
            with self.assertRaises(R.ReproduceError) as caught:
                fetch_inputs(args, log=lambda _: None)
            self.assertIn("sample-data-v2", str(caught.exception))
            self.assertIn("--bundle PATH", str(caught.exception))
            self.assertEqual(list(Path(tmp).iterdir()), [])  # no partial file left

    def test_a_missing_local_file_is_named(self):
        with self.assertRaisesRegex(R.ReproduceError, "doesn't exist"):
            R.fetch("/nonexistent/laptop_lidar_tracks.zip", Path("/tmp"), "saved tracks")


class OfflineTest(unittest.TestCase):
    def test_network_and_keys_are_off_inside_and_back_after(self):
        before = (socket.socket.connect, socket.getaddrinfo, socket.create_connection)
        with patch.dict(os.environ, {"MODEL_API_KEY": "secret"}):
            with R.offline() as set_aside:
                self.assertEqual(set_aside, {"MODEL_API_KEY": "secret"})
                self.assertNotIn("MODEL_API_KEY", os.environ)
                with self.assertRaises(R.NetworkRefused):
                    socket.create_connection(("api.meta.ai", 443))
                with self.assertRaises(R.NetworkRefused):
                    socket.getaddrinfo("inference.baseten.co", 443)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, self.assertRaises(R.NetworkRefused):
                    s.connect(("127.0.0.1", 9))
                with self.assertRaises(R.NetworkRefused):  # not wrapped into a URLError that a client might retry
                    urllib.request.urlopen("http://127.0.0.1:9/", timeout=1)
                a, b = socket.socketpair()  # local pipes (process pools) still work
                a.sendall(b"ok")
                self.assertEqual(b.recv(2), b"ok")
                a.close(), b.close()
            self.assertEqual(os.environ["MODEL_API_KEY"], "secret")
        self.assertEqual((socket.socket.connect, socket.getaddrinfo, socket.create_connection), before)

    def test_model_clients_refuse_use(self):
        spark = R.NoModel("Muse Spark")
        self.assertTrue(spark)  # Agent keeps it instead of making a real client
        with self.assertRaisesRegex(RuntimeError, "makes no model calls.*Muse Spark"):
            spark.chat.completions.create(model="x", messages=[])


def published_model(q, reference=850, coverage="replay 95% / 94%"):
    return {"measured": {"reference_frame": reference, "plate_size_mm": [308.9133, 205.1832],
                         "plate_thickness_mm": 3.8403, "base_thickness_mm": 13.6185,
                         "hinge_range_deg": [-106.64, 9.39], "opening_range_deg": [0.48, 117.35],
                         "opening_on_reference_deg": 107.62, "frames_with_plate": 863, "frames": 923,
                         "residual_m": 0.00465, "plate_span_along_hinge_mm": 303.93,
                         "from_closed_part": 415, "from_open_object": 360,
                         "axis": [0.0, 1.0, 0.0], "point": [0.0, 0.0, 0.5]},
            "joints": {"lid_joint": q},
            "physics": {"links": {"base": {"mass_kg": 0.9}}, "joints": {"lid_joint": {
                "friction": 0.2835, "drift_with_deg": [0.04, -0.076], "drift_without_deg": [11.5, 2.6]}}},
            "provenance": f"... the improved fit_lidar gave the numbers above (309 x 205 mm, {coverage}). ..."}


class CompareTest(unittest.TestCase):
    def test_coverage_from_the_provenance_note(self):
        self.assertEqual(R.coverage_from_text("(309 x 205 mm, replay 95% / 94%)"), {"lid": 95.0, "base": 94.0})
        self.assertIsNone(R.coverage_from_text("no numbers here"))
        self.assertIsNone(R.coverage_from_text(None))

    def test_same_reference_rows(self):
        q = [0.1, None, 0.0, -1.8]
        pub = R.published_values(published_model(q))
        rep = dict(pub, lid_mm=(308.9180, 205.1844), coverage={"lid": 94.6, "base": 92.4},
                   point=np.array([0.0001, 0.0, 0.5]), from_closed=(415, 395))
        rows = {r.quantity: r for r in R.compare(pub, rep)}
        self.assertEqual(rows["reference frame (the joint's zero)"].difference, "same")
        self.assertEqual(rows["lid size, mm"].reproduced, "308.9 x 205.2")
        self.assertEqual(rows["lid size, mm"].difference, "+0.005 / +0.001")
        self.assertEqual(rows["base thickness, mm"].difference, "0")
        self.assertEqual(rows["  of them from plate_closed's mask: shut / raised"].difference, "0 / +35")
        self.assertEqual(rows["replay covers the lid's SAM masks, %"].difference, "same to the whole %")
        self.assertEqual(rows["replay covers the base's SAM masks, %"].difference, "-2 pts")
        self.assertEqual(rows["hinge line vs the published one"].difference, "0.000 deg, 0.10 mm apart")
        self.assertEqual(rows["opening on each frame both measured (3), deg"].difference, "median 0.00, max 0.00")
        self.assertEqual(rows["released without it, worst drift, deg"].published, "11.5")
        self.assertIn("lid size, mm", R.text_table(list(rows.values())))
        self.assertTrue(R.markdown_table(list(rows.values())).startswith("| quantity | published |"))

    def test_other_reference_compares_openings_not_joint_values(self):
        q = np.array([0.1, 0.0, -1.0, -1.8])
        pub = R.published_values(published_model(list(q)))
        # the same motion measured from another frame: joint values shifted, openings unchanged
        rep = R.values(dict(published_model(list(q))["measured"], reference_frame=1,
                            opening_on_reference_deg=107.62 + np.rad2deg(0.1)), list(q - 0.1), None, None)
        rows = {r.quantity: r for r in R.compare(pub, rep)}
        self.assertEqual(rows["reference frame (the joint's zero)"].difference, "other frame")
        self.assertNotIn("hinge line vs the published one", rows)  # axes are in different cameras' coordinates
        self.assertIn("hinge range (joint limits), deg (zero at each fit's reference frame)", rows)
        self.assertEqual(rows["opening on each frame both measured (4), deg"].difference, "median 0.00, max 0.00")
        self.assertEqual(rows["hinge friction, N m"].reproduced, "-")


if __name__ == "__main__":
    unittest.main()
