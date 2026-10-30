"""Cut the demo video from the project's own outputs: captions, no voice.

Each scene has narration text; its length sets the scene's duration (about 2.6 words a
second, the pace of a calm voice-over), so a voice recorded from docs/demo_script.md fits.

Usage:
  .venv/bin/python scripts/make_demo.py [--out docs/demo.mp4] [--voice narration.m4a]

--voice muxes a recorded narration in. Inputs: the LiDAR laptop run (data/tracks_agent/
laptop_lidar), its phone video, the RBO laptop04 result, the URDF Doctor report and two
viewer screenshots in data/demo_assets/.
"""

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

W, H, FPS = 1920, 1080, 30
WORDS_PER_S = 2.6
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
BG = (18, 18, 20)
ACCENT = (110, 160, 255)
RUN = ROOT / "data" / "tracks_agent" / "laptop_lidar"
PHONE = ROOT / "data" / "real" / "laptop_lidar" / "laptop_lidar.mp4"
ASSETS = ROOT / "data" / "demo_assets"
DRAWER_RUN = ROOT / "data" / "tracks_agent" / "drawer2_lidar"
DRAWER_PHONE = ROOT / "data" / "real" / "drawer2_lidar" / "drawer2_lidar.mp4"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


class Clip:
    """Frames of a video read forward only (scenes play them in order)."""

    def __init__(self, path: Path, first: int = 0, last: int | None = None):
        self.cap = cv2.VideoCapture(str(path))
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.first, self.last = first, (n - 1 if last is None else last)
        self.pos, self.img = -1, None

    def at(self, p: float) -> np.ndarray:
        want = self.first + int(round(p * (self.last - self.first)))
        while self.pos < want:
            ok, img = self.cap.read()
            if not ok:
                break
            self.pos, self.img = self.pos + 1, img
            if self.pos < self.first:
                continue
        return self.img


def fit(img: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    x, y, bw, bh = box
    s = min(bw / img.shape[1], bh / img.shape[0])
    out = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return out, (x + (bw - out.shape[1]) // 2, y + (bh - out.shape[0]) // 2)


def place(canvas: np.ndarray, img: np.ndarray, box) -> None:
    im, (x, y) = fit(img, box)
    canvas[y: y + im.shape[0], x: x + im.shape[1]] = im


def text_layer(canvas_bgr: np.ndarray, items: list[tuple]) -> np.ndarray:
    """items: (xy, text, size, bold, rgb, anchor)."""
    im = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    for xy, text, size, bold, rgb, anchor in items:
        d.multiline_text(xy, text, font=font(size, bold), fill=rgb, anchor=anchor, spacing=int(size * 0.35))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def caption(canvas: np.ndarray, section: str, line: str) -> np.ndarray:
    """A section label top left and the caption on a dark band at the bottom."""
    band = canvas[H - 150:].astype(np.float32)
    canvas[H - 150:] = (band * 0.25).astype(np.uint8)
    return text_layer(canvas, [((60, 40), section.upper(), 30, True, ACCENT, "la"),
                               ((W // 2, H - 75), line, 40, False, (240, 240, 240), "mm")])


@dataclass
class Scene:
    section: str
    narration: str
    captions: list[str]  # shown in turn over the scene
    draw: Callable[[float], np.ndarray]  # progress 0..1 -> 1920x1080 BGR frame without captions
    min_s: float = 4.0

    @property
    def seconds(self) -> float:
        return max(self.min_s, len(self.narration.split()) / WORDS_PER_S + 1.0)


def blank() -> np.ndarray:
    return np.full((H, W, 3), BG, np.uint8)


def card(lines: list[tuple[str, int, bool, tuple]], top: int = 260) -> Callable[[float], np.ndarray]:
    def draw(p):
        y, items = top, []
        for text, size, bold, rgb in lines:
            items.append(((W // 2, y), text, size, bold, rgb, "ma"))
            y += int(size * (1.5 + text.count("\n") * 1.35))
        return text_layer(blank(), items)
    return draw


def video_scene(path: Path, first: int = 0, last: int | None = None, box=(60, 90, W - 120, H - 260)):
    clip = Clip(path, first, last)

    def draw(p):
        c = blank()
        place(c, clip.at(p), box)
        return c
    return draw


def stills(paths: list[Path], labels: list[str] | None = None):
    imgs = [cv2.imread(str(p)) for p in paths]

    def draw(p):
        i = min(int(p * len(imgs)), len(imgs) - 1)
        c = blank()
        place(c, imgs[i], (60, 90, W - 120, H - 260))
        if labels:
            c = text_layer(c, [((W - 60, 40), labels[i], 30, False, (200, 200, 200), "ra")])
        return c
    return draw


def pair(left: Path, right: Path, labels: tuple[str, str], left_range=(0, None), right_range=(0, None)):
    """Two videos side by side, each played over the whole scene."""
    a, b = Clip(left, *left_range), Clip(right, *right_range)

    def draw(p):
        c = blank()
        place(c, a.at(p), (40, 110, W // 2 - 60, H - 290))
        place(c, b.at(p), (W // 2 + 20, 110, W // 2 - 60, H - 290))
        return text_layer(c, [((W // 4, H - 200), labels[0], 30, False, (200, 200, 200), "ma"),
                              ((3 * W // 4, H - 200), labels[1], 30, False, (200, 200, 200), "ma")])
    return draw


def side_by_side(left: Path, right: Path, first: int, last: int):
    a, b = Clip(left, first, last), Clip(right, first, last)

    def draw(p):
        c = blank()
        place(c, a.at(p), (40, 110, W // 2 - 60, H - 290))
        place(c, b.at(p), (W // 2 + 20, 110, W // 2 - 60, H - 290))
        return text_layer(c, [((W // 4, H - 200), "phone video", 30, False, (200, 200, 200), "ma"),
                              ((3 * W // 4, H - 200), "MuJoCo replay of the measured model", 30, False,
                               (200, 200, 200), "ma")])
    return draw


def scenes() -> list[Scene]:
    agent = RUN / "agent"
    white, grey, blue = (240, 240, 240), (170, 170, 175), ACCENT
    green, red = (120, 220, 140), (255, 140, 120)
    return [
        Scene("Video2URDF",
              "Robots need simulation models of the objects they handle, and people build them by hand. "
              "Video2URDF measures them from one phone video. This laptop was filmed once; the robot is closing "
              "the model measured from that video.",
              ["One phone video in. A measured, simulation-ready URDF out.",
               "a Franka Panda in MuJoCo closes the laptop model measured from one video"],
              video_scene(RUN / "robot_demo.mp4")),
        Scene("1  Capture",
              "I film my laptop with an iPhone while I open and close it. No markers and no calibration card: "
              "the LiDAR gives metric depth.",
              ["iPhone LiDAR (Stray Scanner), handheld: no markers, no card"],
              video_scene(PHONE, 0, 922)),
        Scene("2  Agent",
              "Muse Spark runs the pipeline as an agent. It watches the whole video, chooses short noun prompts "
              "for SAM 3.1, and rephrases when a prompt finds nothing.",
              ["Muse Spark watches the video and picks noun prompts for SAM 3.1",
               "'silver laptop base': no match -> it rephrases: 'silver laptop body'",
               "the closed lid gets its own anchors"],
              stills([agent / "step001.jpg", agent / "step013.jpg", agent / "step044.jpg", agent / "step034.jpg"],
                     ["look: frames across the video", "segment 'laptop screen'", "segment 'silver laptop body'",
                      "segment 'gray laptop lid' (closed)"])),
        Scene("3  Measure and verify",
              "SAM 3.1 tracks every part through all nine hundred frames. MuJoCo then replays the measured model "
              "from the phone's own camera, on every frame: it covers ninety five percent of the lid. The lid "
              "measures three oh nine by two oh five millimetres; the spec says three oh four by two twelve.",
              ["SAM 3.1 tracks the parts; hinge = where the lid and base planes meet",
               "MuJoCo replays the model from the phone's camera pose, every frame",
               "covers 95% of the lid, 94% of the base",
               "lid 309 x 205 mm  (spec: 304 x 212 mm)"],
              side_by_side(PHONE, RUN / "replay3d.mp4", 0, 922)),
        Scene("4  It checks itself",
              "Spark checks its own work. In this run its first model had the base as a thin sliver. Spark "
              "watched the replay, segmented the base again and re-tracked, and the base came back. The fit tool "
              "also names such mistakes in plain words.",
              ["first fit: the base came out as a sliver (blue bar)",
               "Spark watched the replay, re-segmented the base, re-tracked",
               "after: the base is covered"],
              stills([ASSETS / "agent_first_fit.jpg", ASSETS / "agent_refit.jpg"],
                     ["Spark's first fit", "after Spark's own fix"])),
        Scene("5  It looks real",
              "The measured shapes are painted with the video itself. Every face takes the frame where it was seen "
              "best, so the screen shows its wallpaper, the base its keyboard, and the closed lid its logo.",
              ["each face painted from the frame where it was seen best",
               "shapes unchanged: collisions and inertia stay measured"],
              pair(PHONE, RUN / "turntable.mp4", ("phone video", "the measured model, painted with that video"))),
        Scene("6  Physics",
              "A robot needs mass and friction too. Spark judges the materials and masses. Where the lid stands "
              "with no hand on it, only hinge friction holds it, so the friction is measured there. Let go in "
              "MuJoCo, the lid stays; without that friction it falls.",
              ["masses: Spark judges the materials (0.9 + 0.5 kg; spec 1.4 kg)",
               "hinge friction measured where the lid stands with no hand on it: 0.28 N m",
               "let go in MuJoCo: stays (+0.1 deg) vs. falls (+11.5 deg) without it"],
              video_scene(RUN / "physics_demo.mp4")),
        Scene("7  Drawers too",
              "Drawers work too. The agent finds which drawer moves and measures two hundred and forty eight "
              "millimetres of travel against the cabinet.",
              ["the agent finds which drawer moves: pulls out 248 mm"],
              pair(DRAWER_PHONE, DRAWER_RUN / "turntable.mp4", ("phone video", "the measured cabinet and drawer"),
                   (150, 700))),
        Scene("8  Benchmark",
              "On the RBO dataset, real RGB-D videos with motion capture, the agent runs every sequence from "
              "scratch. Over thirteen sequences the median hinge axis error is two and a half degrees. Box lids "
              "still fail.",
              ["RBO dataset (TU Berlin, CC BY 4.0): Kinect RGB-D + motion capture",
               "13 sequences, median: axis 2.5 deg, angle 3.3 deg",
               "laptops 2-3 deg; box lids still fail"],
              stills([ROOT / "data" / "rbo" / "results" / "laptop04.jpg"],
                     ["laptop04: red = measured, green = motion capture"])),
        Scene("9  URDF Doctor",
              "URDF Doctor checks the models you already have against the video. Three injected bugs and a sign "
              "flip in a joint log: all found.",
              ["URDF Doctor: an existing URDF vs. the motion in the video"],
              card([("URDF Doctor", 64, True, white),
                    ("laptop_spec.urdf (published dimensions)", 40, True, grey),
                    ("consistent: axis 2.3 deg / 8 mm from the measured hinge", 38, False, green),
                    ("laptop_buggy.urdf (hinge +25 mm, axis tilted 6 deg, limit 90 deg)", 40, True, grey),
                    ("axis 7.6 deg off   |   hinge 18 mm off   |   video needs 127 deg, limit is 90", 38, False, red),
                    ("joint log with the sign reversed:  log = -1.00 x expected", 38, False, red)], top=170)),
        Scene("10  Open weights",
              "Every model it calls is a Meta model, and it runs end to end on open weights: Muse Glimmer on our "
              "own GPU, with SAM 3.1 weights, through to the physics.",
              ["--open-weights --glimmer: Muse Glimmer 30B + SAM 3.1 weights, on our own GPUs"],
              card([("Open weights, end to end", 64, True, white),
                    ("run_agent.py --open-weights --glimmer", 44, False, blue),
                    ("Muse Glimmer 30B (our Modal endpoint)  +  SAM 3.1 / SAM 3 weights (Modal GPU)", 40, False, grey),
                    ("same laptop: lid 307 x 201 mm, 0..108 deg, hinge friction measured", 40, False, grey),
                    ("also: Muse Spark Open (Baseten): lid 305 x 209 mm", 36, False, grey)], top=310)),
        Scene("Try it",
              "Everything is in the browser: the video, the replay, the robot and the painted model, with URDF "
              "downloads. One video in. A model that moves like the real thing out.",
              ["sunghyunc.github.io/video2urdf   |   github.com/SungHyunC/video2urdf"],
              stills([ASSETS / "viewer_textured.jpg"])),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "demo.mp4")
    ap.add_argument("--voice", type=Path, help="recorded narration to mux in")
    args = ap.parse_args()
    sc = scenes()
    total = sum(s.seconds for s in sc)
    print(f"{len(sc)} scenes, {total:.0f} s")

    script, t = ["# Demo narration", "", f"Total {total:.0f} s. Read at a calm pace; each line starts at its time.", ""], 0.0
    for s in sc:
        script.append(f"**{int(t // 60)}:{t % 60:04.1f}  {s.section}**  \n{s.narration}\n")
        t += s.seconds
    (args.out.parent / "demo_script.md").write_text("\n".join(script))

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for s in sc:
            n = int(round(s.seconds * FPS))
            for i in range(n):
                p = i / max(n - 1, 1)
                frame = s.draw(p)
                line = s.captions[min(int(p * len(s.captions)), len(s.captions) - 1)]
                frame = caption(frame.copy(), s.section, line)
                fade = min(1.0, i / 6, (n - 1 - i) / 6)  # short fades between scenes
                writer.write((frame * max(fade, 0)).astype(np.uint8))
        writer.release()
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw)]
        if args.voice:
            cmd += ["-i", str(args.voice), "-c:a", "aac", "-b:a", "128k", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(args.out)]
        subprocess.run(cmd, check=True)
    print(f"{args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
