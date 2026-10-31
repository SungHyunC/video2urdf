"""Input detection and normalisation (local ffmpeg only, no model calls)."""
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from video2urdf import ingest as I


def make_video(path: Path, frames: int = 10, fps: int = 30, size=(64, 48), rotation: int | None = None) -> Path:
    """A tiny H.264 test clip (optionally tagged with rotation metadata, as phones do)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plain = path.with_name(path.stem + ".plain.mp4") if rotation else path
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc=size={size[0]}x{size[1]}:rate={fps}", "-frames:v", str(frames), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(plain)], check=True)
    if rotation:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", str(rotation), "-i", str(plain),
                        "-c", "copy", str(path)], check=True)
        plain.unlink()
    return path


def make_stray(root: Path, frames: int = 10, video: bool = True, size=(64, 48)) -> Path:
    """A minimal Stray Scanner capture: rgb.mp4, depth/ and confidence/ PNGs, odometry, intrinsics."""
    (root / "depth").mkdir(parents=True, exist_ok=True)
    (root / "confidence").mkdir(exist_ok=True)
    if video:
        make_video(root / "rgb.mp4", frames, fps=60, size=size)
    else:
        (root / "rgb.mp4").write_bytes(b"")
    for t in range(frames):
        cv2.imwrite(str(root / "depth" / f"{t:06d}.png"), np.full((12, 16), 500 + t, np.uint16))
        cv2.imwrite(str(root / "confidence" / f"{t:06d}.png"), np.full((12, 16), 2, np.uint8))
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy"]
    rows += [f"{t / 60:.6f}, {t:06d}, {t * 0.01:.4f}, 0, 0, 0, 0, 0, 1, 50, 50, 32, 24" for t in range(frames)]
    (root / "odometry.csv").write_text("\n".join(rows) + "\n")
    (root / "camera_matrix.csv").write_text("50, 0, 32\n0, 50, 24\n0, 0, 1\n")
    return root


class DetectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_stray_folder_flat_and_nested(self):
        make_stray(self.dir / "a" / "b" / "9a84cd46da", video=False)
        for root in (self.dir, self.dir / "a" / "b" / "9a84cd46da"):
            found = I.detect(root)
            self.assertEqual(found.kind, "stray")
            self.assertEqual(found.path, self.dir / "a" / "b" / "9a84cd46da")

    def test_mapanything_capture(self):
        make_stray(self.dir / "cap", video=False)
        (self.dir / "cap" / "mapanything.json").write_text("{}")
        self.assertEqual(I.detect(self.dir).kind, "mapanything")

    def test_capture_wins_over_a_loose_video(self):
        make_video(self.dir / "preview.mp4")
        make_stray(self.dir / "capture" / "x", video=False)
        self.assertEqual(I.detect(self.dir).kind, "stray")

    def test_rgbd_folder(self):
        d = self.dir / "laptop02"
        make_video(d / "laptop02.mp4")
        np.savez(d / "depth.npz", depth=np.ones((10, 12, 16), np.float32))
        (d / "intrinsics.json").write_text(json.dumps({"f": 50, "cx": 32, "cy": 24}))
        self.assertEqual(I.detect(self.dir).kind, "rgbd")

    def test_video_by_content_not_name(self):
        v = make_video(self.dir / "clip.mp4")
        renamed = v.rename(self.dir / "clip.dat")
        self.assertEqual(I.detect(renamed).kind, "video")
        self.assertEqual(I.detect(self.dir).kind, "video")

    def test_single_photo_and_text_are_unusable(self):
        cv2.imwrite(str(self.dir / "p.jpg"), np.zeros((20, 20, 3), np.uint8))
        self.assertEqual(I.detect(self.dir / "p.jpg").kind, "image")
        (self.dir / "notes.txt").write_text("hello")
        self.assertEqual(I.detect(self.dir / "notes.txt").kind, "unknown")
        self.assertEqual(I.detect(self.dir / "missing.mov").kind, "unknown")

    def test_photo_folder_needs_enough_photos(self):
        for k in range(I.MIN_IMAGES - 1):
            cv2.imwrite(str(self.dir / f"IMG_{k}.JPG"), np.full((20, 30, 3), k * 10, np.uint8))
        self.assertEqual(I.detect(self.dir).kind, "unknown")
        cv2.imwrite(str(self.dir / "IMG_10.JPG"), np.zeros((20, 30, 3), np.uint8))
        found = I.detect(self.dir)
        self.assertEqual(found.kind, "images")
        self.assertEqual([p.name for p in found.images][-2:], ["IMG_6.JPG", "IMG_10.JPG"])  # natural order

    def test_zip_holding_a_capture(self):
        make_stray(self.dir / "src" / "9a84cd46da", video=False)
        z = self.dir / "scan.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for f in (self.dir / "src").rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(self.dir / "src"))
            zf.writestr("__MACOSX/9a84cd46da/._rgb.mp4", b"junk")
        self.assertEqual(I.detect(z).kind, "zip")
        self.assertTrue(I.archive_has_capture(z))
        holder = self.dir / "holder"
        holder.mkdir()
        (holder / "scan.zip").write_bytes(z.read_bytes())
        self.assertEqual(I.detect(holder).kind, "zip")  # a folder holding the zip of a capture

    def test_unpack_refuses_paths_out_of_the_folder(self):
        z = self.dir / "evil.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("../evil.txt", b"x")
            zf.writestr("/abs.txt", b"x")
            zf.writestr("ok/fine.txt", b"x")
        out = self.dir / "out"
        I.unpack(z, out, log=lambda s: None)
        self.assertTrue((out / "ok" / "fine.txt").exists())
        self.assertFalse((self.dir / "evil.txt").exists())
        self.assertEqual(sorted(p.name for p in out.rglob("*") if p.is_file()), ["fine.txt"])


class IngestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rotated_60fps_video_becomes_upright_30fps_h264(self):
        src = make_video(self.dir / "IMG_0001.MOV", frames=60, fps=60, rotation=90)
        ing = I.ingest(src, self.dir / "run" / "input", "clip", log=lambda s: None)
        self.assertTrue(ing.usable)
        self.assertEqual(ing.kind, "video")
        self.assertEqual(ing.size, (48, 64))  # turned upright: portrait
        self.assertAlmostEqual(ing.fps, 30.0)
        self.assertEqual(ing.frames, 30)
        out = I.video_info(I.ffprobe(Path(ing.video)))
        self.assertEqual((out.codec, out.rotation, out.has_audio), ("h264", 0, False))

    def test_zip_of_a_capture_keeps_frames_with_depth(self):
        make_stray(self.dir / "src" / "abc123", frames=10)
        z = self.dir / "abc123.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for f in (self.dir / "src").rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(self.dir / "src"))
        ing = I.ingest(z, self.dir / "run" / "input", "abc123", log=lambda s: None)
        self.assertEqual(ing.kind, "stray")
        self.assertEqual(ing.frames, 10)
        cap = Path(ing.capture)
        self.assertEqual(I.capture_counts(cap), {"depth": 10, "confidence": 10, "odometry": 10})
        self.assertEqual(I.count_frames(Path(ing.video)), 10)
        self.assertEqual(ing.warnings, [])

    def test_long_capture_is_decimated_with_its_depth_and_poses(self):
        src = make_stray(self.dir / "cap", frames=12)
        out = self.dir / "copy"
        counts = I.copy_stray(src, out, keep_every=3, log=lambda s: None)
        self.assertEqual(counts, {"depth": 4, "confidence": 4, "odometry": 4})
        self.assertEqual(I.count_frames(out / "rgb.mp4"), 4)
        # frame 2 of the copy is frame 6 of the original: its depth value says so (500 + t)
        self.assertEqual(int(cv2.imread(str(out / "depth" / "000002.png"), cv2.IMREAD_UNCHANGED)[0, 0]), 506)
        rows = (out / "odometry.csv").read_text().splitlines()[1:]
        self.assertTrue(rows[2].split(",")[2].strip().startswith("0.06"))
        self.assertEqual(I.plan_capture(7200, 3600), 2)
        self.assertEqual(I.plan_capture(3600, 3600), 1)

    def test_older_layout_npy_depth_without_confidence(self):
        import shutil
        src = make_stray(self.dir / "cap", frames=40, size=(320, 240))
        for f in sorted((src / "depth").glob("*.png")):
            d = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            np.save(f.with_suffix(".npy"), d.astype(np.float32) / 1000)  # metres, as float
            f.unlink()
        shutil.rmtree(src / "confidence")
        ing = I.ingest(src, self.dir / "run" / "input", "old", log=lambda s: None)
        self.assertTrue(ing.usable)
        cap = Path(ing.capture)
        self.assertEqual(I.capture_counts(cap), {"depth": 40, "confidence": 40, "odometry": 40})
        d = cv2.imread(str(cap / "depth" / "000003.png"), cv2.IMREAD_UNCHANGED)
        self.assertEqual((d.dtype, int(d[0, 0])), (np.uint16, 503))  # back in millimetres
        conf = cv2.imread(str(cap / "confidence" / "000003.png"), cv2.IMREAD_UNCHANGED)
        self.assertEqual(int(conf[0, 0]), 2)

    def test_animated_gif_is_a_video(self):
        gif = self.dir / "a.gif"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10",
                        "-frames:v", "12", str(gif)], check=True)
        self.assertEqual(I.sniff(gif), "video")

    def test_capture_whose_parts_disagree_is_flagged(self):
        src = make_stray(self.dir / "cap", frames=10, video=False)
        for f in sorted((src / "depth").glob("*.png"))[5:]:
            f.unlink()
        warnings = I.check_capture(src, video_frames=10, rotation=0)
        self.assertTrue(any("depth/ has 5" in w for w in warnings))

    def test_photo_folder_becomes_a_video(self):
        for k in range(10):
            cv2.imwrite(str(self.dir / f"frame{k}.png"), np.full((40, 60, 3), 20 * k, np.uint8))
        cv2.imwrite(str(self.dir / "frame_odd.png"), np.zeros((60, 40, 3), np.uint8))  # turned: skipped
        ing = I.ingest(self.dir, self.dir / "run" / "input", "photos", image_fps=5, log=lambda s: None)
        self.assertEqual(ing.kind, "images")
        self.assertEqual(ing.frames, 10)
        self.assertEqual(ing.details["skipped"], ["frame_odd.png"])

    def test_unusable_inputs_say_why(self):
        (self.dir / "a.txt").write_text("not a video")
        ing = I.ingest(self.dir / "a.txt", self.dir / "run", "a", log=lambda s: None)
        self.assertFalse(ing.usable)
        self.assertIn("not a video", ing.reason)


class PlanTest(unittest.TestCase):
    def info(self, **kw):
        base = dict(codec="h264", width=1080, height=1920, rotation=0, fps=30.0, nominal_fps=30.0, vfr=False,
                    frames=300, duration=10.0, pix_fmt="yuv420p", bit_depth=8, transfer="bt709",
                    primaries="bt709", hdr=None, dolby_vision=None, has_audio=False, container="mov",
                    make=None, model=None, has_location=False)
        base.update(kw)
        return I.VideoInfo(**base)

    def test_frame_rate_divides_the_source_rate(self):
        for src, want in [(30, 30), (24, 24), (31, 31), (60, 30), (50, 25), (120, 30), (240, 30),
                          (59.94, 29.97), (90, 30)]:
            plan = I.plan_video(self.info(fps=src, nominal_fps=src, frames=int(src * 10)))
            self.assertAlmostEqual(plan.fps_value, want, places=2, msg=f"{src} fps")

    def test_variable_rate_resamples_at_its_average(self):
        plan = I.plan_video(self.info(fps=11.3, nominal_fps=30, vfr=True))
        self.assertEqual(plan.fps, "11")
        self.assertTrue(any("variable frame rate" in n for n in plan.notes))

    def test_long_video_lowers_the_rate_then_trims(self):
        plan = I.plan_video(self.info(duration=100.0, frames=3000))
        self.assertAlmostEqual(plan.fps_value, 18.0, places=1)
        self.assertIsNone(plan.trim_s)
        plan = I.plan_video(self.info(duration=300.0, frames=9000))
        self.assertAlmostEqual(plan.fps_value, I.MIN_FPS)
        self.assertAlmostEqual(plan.trim_s, I.MAX_FRAMES / I.MIN_FPS)
        self.assertTrue(plan.warnings)
        self.assertIn("-t", I.ffmpeg_command(Path("a"), Path("b"), plan))

    def test_4k_hdr_iphone_clip(self):
        plan = I.plan_video(self.info(width=2160, height=3840, codec="hevc", hdr="hlg", bit_depth=10,
                                      has_location=True, has_audio=True))
        self.assertEqual((plan.width, plan.height), (1080, 1920))
        cmd = " ".join(I.ffmpeg_command(Path("in.mov"), Path("out.mp4"), plan))
        self.assertIn("tonemap", cmd)
        self.assertIn("arib-std-b67", cmd)
        self.assertIn("-map_metadata -1", cmd)
        self.assertIn("-an", cmd)
        self.assertTrue(any("GPS" in n for n in plan.notes))

    def test_dolby_vision_profile_5_warns(self):
        plan = I.plan_video(self.info(hdr="pq", dolby_vision=5, bit_depth=10))
        self.assertTrue(any("Dolby Vision" in w for w in plan.warnings))

    def test_upright_turn_swaps_the_size_and_adds_a_filter(self):
        plan = I.plan_video(self.info(width=1920, height=1080), turn=270)
        self.assertEqual((plan.width, plan.height), (1080, 1920))
        self.assertIn("transpose=2", " ".join(I.ffmpeg_command(Path("a"), Path("b"), plan)))
        self.assertIn("hflip,vflip", " ".join(I.ffmpeg_command(Path("a"), Path("b"), I.plan_video(self.info(), turn=180))))

    def test_odd_sizes_become_even(self):
        plan = I.plan_video(self.info(width=1079, height=1919))
        self.assertEqual((plan.width % 2, plan.height % 2), (0, 0))
        self.assertTrue(plan.scale)

    def test_probe_parsing(self):
        probe = {"streams": [
            {"codec_type": "video", "codec_name": "hevc", "width": 1920, "height": 1080, "pix_fmt": "yuv420p10le",
             "r_frame_rate": "30/1", "avg_frame_rate": "2400/91", "nb_frames": "240", "duration": "9.1",
             "color_transfer": "arib-std-b67", "color_primaries": "bt2020",
             "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90},
                                {"side_data_type": "DOVI configuration record", "dv_profile": 8}]},
            {"codec_type": "audio"}],
            "format": {"format_name": "mov,mp4", "duration": "9.1",
                       "tags": {"com.apple.quicktime.location.ISO6709": "+36.6+127.2/",
                                "com.apple.quicktime.model": "iPhone"}}}
        info = I.video_info(probe)
        self.assertEqual((info.width, info.height, info.rotation), (1080, 1920, 90))
        self.assertTrue(info.vfr)
        self.assertEqual((info.hdr, info.dolby_vision, info.bit_depth), ("hlg", 8, 10))
        self.assertTrue(info.has_audio and info.has_location)

    def test_cover_art_is_not_a_video(self):
        probe = {"streams": [{"codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}},
                             {"codec_type": "audio"}], "format": {"duration": "200"}}
        self.assertIsNone(I.video_stream(probe))


if __name__ == "__main__":
    unittest.main()
