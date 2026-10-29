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
    return [
        Scene("Video2URDF",
              "Robots need simulation models of the objects they handle. Today, people build them by hand. "
              "Video2URDF measures them from one phone video.",
              ["One phone video in. A measured, simulation-ready URDF out."],
              card([("Video2URDF", 110, True, white),
                    ("One phone video of a real object in.\nA measured joint model a robot can use out.", 48, False, grey),
                    ("Muse Spark  +  SAM 3.1  +  MuJoCo", 40, True, blue)], top=300)),
        Scene("1  Capture",
              "I film my laptop with an iPhone while I open and close it. No markers and no calibration card: "
              "the LiDAR gives metric depth.",
              ["iPhone LiDAR (Stray Scanner), handheld: no markers, no card"],
              video_scene(PHONE, 0, 922)),
        Scene("2  Agent",
              "Muse Spark runs the pipeline as an agent. It looks at frames, and chooses short noun prompts for "
              "SAM 3.1. When a prompt finds nothing, it rephrases. Parts that disappear, like the screen when the "
              "lid closes, get anchors on several frames.",
              ["Muse Spark looks at frames and picks noun prompts for SAM 3.1",
               "'silver laptop base': no match -> it rephrases: 'silver laptop body'",
               "the closed lid gets its own anchors"],
              stills([agent / "step001.jpg", agent / "step013.jpg", agent / "step044.jpg", agent / "step034.jpg"],
                     ["look: frames across the video", "segment 'laptop screen'", "segment 'silver laptop body'",
                      "segment 'gray laptop lid' (closed)"])),
        Scene("3  Measure",
              "SAM 3.1 tracks every part through all nine hundred frames. From the depth of the lid in each frame, "
              "the hinge is fit by least squares, where the lid and the base planes meet, and the opening is read "
              "frame by frame.",
              ["SAM 3.1 tracks the parts through all 923 frames",
               "hinge = where the lid and base planes meet (least squares); opening per frame"],
              stills([RUN / "hinge.jpg"])),
        Scene("4  Verify",
              "Then the check. MuJoCo replays the measured model from the phone's own camera pose, on every frame, "
              "on top of the video. It covers ninety five percent of the lid and ninety four percent of the base. "
              "The lid measures three hundred and nine by two hundred and five millimetres. "
              "The spec sheet says three oh four by two twelve.",
              ["MuJoCo replays the model from the phone's camera pose, every frame",
               "covers 95% of the lid and 94% of the base",
               "lid 309 x 205 mm  (spec: 304 x 212 mm)"],
              side_by_side(PHONE, RUN / "replay3d.mp4", 0, 922)),
        Scene("4b  Drawers too",
              "Drawers work too. The agent finds which drawer moves, measures it against the cabinet's front in "
              "every frame, and builds the cabinet and the drawer. It pulls out two hundred and forty eight "
              "millimetres, and the replay stays on the drawer, shut and open.",
              ["a drawer: the agent finds which one moves and measures it against the cabinet's front",
               "pulls out 248 mm; the replay stays on the drawer, shut and open"],
              side_by_side(DRAWER_PHONE, DRAWER_RUN / "replay3d.mp4", 150, 700)),
        Scene("5  Use",
              "The URDF is ready to use. A Franka arm in MuJoCo closes the measured laptop by pushing along the "
              "measured hinge.",
              ["a Franka Panda closes the measured laptop in MuJoCo"],
              video_scene(RUN / "robot_demo.mp4")),
        Scene("6  Benchmark",
              "Does it generalise? On the RBO dataset, real RGB-D videos with motion capture, the agent runs every "
              "sequence from scratch. Across thirteen sequences the median hinge axis error is two and a half "
              "degrees, and the angle error three point three degrees, even when people push the whole object "
              "around. Box lids are where it still fails.",
              ["RBO dataset (TU Berlin, CC BY 4.0): Kinect RGB-D + motion capture",
               "13 sequences, median: axis 2.5 deg, position 47 mm, angle 3.3 deg",
               "laptops 2-3 deg; box lids still fail (SAM loses them at 640 x 480)"],
              stills([ROOT / "data" / "rbo" / "results" / "laptop04.jpg"],
                     ["laptop04: red = measured, green = motion capture"])),
        Scene("7  URDF Doctor",
              "It also checks the models you already have. URDF Doctor compares a URDF with the video. One written "
              "from the published specs passes. One with three injected bugs: a moved hinge, a tilted axis, and a "
              "joint limit that is too tight. All three are found, and so is a sign flip in a joint log.",
              ["URDF Doctor: an existing URDF vs. the motion in the video",
               "spec URDF: consistent.  Injected bugs: all 3 found",
               "flipped joint log: found"],
              card([("URDF Doctor", 64, True, white),
                    ("laptop_spec.urdf (published dimensions)", 40, True, grey),
                    ("consistent: axis 2.3 deg / 8 mm from the measured hinge", 38, False, (120, 220, 140)),
                    ("laptop_buggy.urdf (hinge +25 mm, axis tilted 6 deg, limit 90 deg)", 40, True, grey),
                    ("axis 7.6 deg off   |   hinge 18 mm off   |   video needs 127 deg, limit is 90",
                     38, False, (255, 140, 120)),
                    ("joint log with the sign reversed:  log = -1.00 x expected", 38, False, (255, 140, 120))],
                   top=170)),
        Scene("8  Open weights",
              "Every model it calls is a Meta model, and one flag runs the whole pipeline on open weights: Muse "
              "Spark Open, and SAM weights on our own GPU.",
              ["--open-weights: Muse Spark Open + SAM 3.1 weights on a Modal GPU"],
              card([("Open weights, one flag", 64, True, white),
                    ("run_agent.py --open-weights", 44, False, blue),
                    ("Muse Spark Open (Baseten)  +  SAM 3.1 / SAM 3 weights (Modal GPU)", 40, False, grey),
                    ("same laptop, open weights only: lid 305 x 209 mm, 3..122 deg", 40, False, grey)], top=330)),
        Scene("Try it",
              "Everything is in the browser: the video, the replay, the robot and the measured model, with URDF "
              "downloads. One video in. A model that moves like the real thing out.",
              ["sunghyunc.github.io/video2urdf   |   github.com/SungHyunC/video2urdf"],
              stills([ASSETS / "viewer_replay.jpg", ASSETS / "viewer_robot.jpg"])),
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
