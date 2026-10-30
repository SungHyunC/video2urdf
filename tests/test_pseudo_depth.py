import unittest
import tempfile
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from video2urdf.pseudo_depth import plan_chunks, rescale_capture
from video2urdf.stray import load


class ChunkPlanningTest(unittest.TestCase):
    def test_short_videos_keep_every_frame_once(self):
        for n in (1, 2, 6, 7):
            anchors, chunks = plan_chunks(n, chunk=2, anchors=6)
            self.assertEqual(len(anchors), len(set(anchors)))
            self.assertTrue(chunks)
            self.assertTrue(all(c[:len(anchors)] == anchors for c in chunks))
            self.assertEqual(set().union(*(set(c) for c in chunks)), set(range(n)))
            self.assertTrue(all(len(c) == len(set(c)) for c in chunks))

    def test_invalid_plan_is_rejected(self):
        for args in ((0, 2, 2), (2, 0, 2), (2, 2, 0)):
            with self.assertRaises(ValueError):
                plan_chunks(*args)

    def test_rescale_rejects_overflow_before_changing_any_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "depth").mkdir()
            (root / "odometry.csv").write_text("time,frame,x,y,z\n0,0,1,2,3\n")
            for i, value in enumerate((1000, 40000)):
                cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), np.full((2, 2), value, np.uint16))
            with self.assertRaisesRegex(ValueError, "16-bit"):
                rescale_capture(root, 2)
            self.assertEqual(int(cv2.imread(str(root / "depth/000000.png"), -1)[0, 0]), 1000)

    def test_rescale_rejects_bad_odometry_without_changing_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "depth").mkdir()
            path = root / "depth/000000.png"
            cv2.imwrite(str(path), np.full((2, 2), 1000, np.uint16))
            (root / "odometry.csv").write_text("time,frame,x,y,z\n0,0,bad,2,3\n")
            with self.assertRaisesRegex(ValueError, "odometry"):
                rescale_capture(root, 2)
            self.assertEqual(int(cv2.imread(str(path), -1)[0, 0]), 1000)

    def test_rescale_updates_depth_and_camera_translation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "depth").mkdir()
            path = root / "depth/000000.png"
            cv2.imwrite(str(path), np.array([[0, 1000]], np.uint16))
            (root / "odometry.csv").write_text("time,frame,x,y,z\n0,0,1,2,3\n")
            (root / "depth.npz").touch()
            rescale_capture(root, 0.5)
            self.assertEqual(cv2.imread(str(path), -1).tolist(), [[0, 500]])
            self.assertEqual((root / "odometry.csv").read_text().splitlines()[1],
                             "0,0,0.500000,1.000000,1.500000")
            self.assertFalse((root / "depth.npz").exists())

    def test_depth_cache_does_not_override_requested_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "depth").mkdir()
            (root / "confidence").mkdir()
            (root / "rgb.mp4").touch()
            np.savetxt(root / "camera_matrix.csv", np.eye(3), delimiter=",")
            cv2.imwrite(str(root / "depth/000000.png"), np.full((2, 2), 1000, np.uint16))
            cv2.imwrite(str(root / "confidence/000000.png"), np.array([[2, 1], [0, 2]], np.uint8))
            self.assertEqual(np.count_nonzero(load(root).depth), 2)
            self.assertEqual(np.count_nonzero(load(root, min_confidence=1).depth), 3)
            self.assertEqual(np.count_nonzero(load(root).depth), 2)

    def test_conflicting_output_is_rejected_before_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "input.mp4"
            video.touch()
            capture = root / "capture"
            existing = root / "results" / capture.name
            existing.mkdir(parents=True)
            (existing / "model.xml").write_text("existing")
            script = Path(__file__).resolve().parents[1] / "scripts/video_to_model.py"
            result = subprocess.run([sys.executable, str(script), str(video), "--capture-out", str(capture),
                                     "--out", str(root / "results")], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already has results", result.stderr)
            self.assertFalse(capture.exists())
            self.assertEqual((existing / "model.xml").read_text(), "existing")
