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
DESIGN = ROOT / "data" / "tracks_design"  # runs with Spark's shape design applied
LAMP = ROOT / "data" / "tracks_agent2" / "stand_lidar"  # the LiDAR desk lamp (rods on hinges)
MOUSE = ROOT / "data" / "tracks_agent2" / "mouse_lidar"  # the black mouse (visual hull + SAM 3D)


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


def sequence(paths: list[Path], labels: list[str]):
    """Videos one after another, each over an equal share of the scene."""
    clips = [Clip(q) for q in paths]

    def draw(p):
        i = min(int(p * len(clips)), len(clips) - 1)
        c = blank()
        place(c, clips[i].at(min(1.0, p * len(clips) - i)), (60, 90, W - 120, H - 260))
        return text_layer(c, [((W - 60, 40), labels[i], 30, False, (200, 200, 200), "ra")])
    return draw


def scenes() -> list[Scene]:
    lamp_agent = LAMP / "agent"
    white, grey, blue = (240, 240, 240), (170, 170, 175), ACCENT
    return [
        Scene("Video2URDF",
              "Robots need simulation models of the objects they handle, and people build them by hand. "
              "Video2URDF measures them from one phone video, and a robot can use them right away.",
              ["One phone video in. A model a robot can use out.",
               "a Franka Panda in MuJoCo works four objects, each measured from one video"],
              video_scene(ASSETS / "robot_grid.mp4")),
        Scene("1  Capture",
              "I film the object with an iPhone while I move it: no markers, no card. The LiDAR gives metric depth.",
              ["iPhone LiDAR (Stray Scanner), handheld: no markers, no card"],
              video_scene(PHONE, 0, 922)),
        Scene("2  Agent",
              "Muse Spark runs the pipeline as an agent. It watches the video, chooses noun prompts for SAM 3.1, "
              "picks the right fitter, and judges what it sees: for this desk lamp it decided the pole never moves.",
              ["Muse Spark watches the video and looks at frames",
               "it prompts SAM 3.1: 'silver metal rod'",
               "it picks fit_chain: two rods on hinges, fitted in 3D from their depth"],
              stills([lamp_agent / "step009.jpg", lamp_agent / "step016.jpg", lamp_agent / "step032.jpg"],
                     ["look", "segment 'silver metal rod'", "fit_chain: the model over the video"])),
        Scene("3  Measure and verify",
              "MuJoCo replays the measured model from the phone's own camera; it covers ninety five percent of the lid, which measures three oh nine by two oh five millimetres against a spec of three oh four by two twelve.",
              ["MuJoCo replays the model from the phone's camera pose, every frame",
               "covers 95% of the lid, 94% of the base",
               "lid 309 x 205 mm  (spec: 304 x 212 mm)"],
              side_by_side(PHONE, RUN / "replay3d.mp4", 0, 922)),
        Scene("4  It checks itself",
              "Spark can't finish over its own warnings. Its first finish here was refused: the screen's track had slid onto the lid's back. It re-tracked the lid and reported what it couldn't verify. Five fresh runs land within four millimetres of the spec.",
              ["finish refuses once while the fit's own warnings are unresolved",
               "five fresh runs: lid 304-306 x 208-212 mm  (spec 304 x 212)"],
              card([("Spark:  finish(\"lid 306 x 210 mm, hinge on the rear edge ...\")", 36, False, white),
                    ("Not finished yet. The automatic checks have unresolved issues:\n"
                     "the plate tracker switched to the closed surface on 806/923 frames", 34, False, (255, 140, 120)),
                    ("Spark:  \"I'll re-segment the lid to fix that.\"", 36, False, white),
                    ("segment  ->  retrack_part  ->  fit_lidar  ->  watch the replay", 34, False, blue),
                    ("Spark:  finish(\"NEEDS REVIEW: ... what stays unverified ...\")", 36, False, white),
                    ("status: needs_review   (checks_passed / needs_review / failed)", 34, False, grey)],
                   top=230)),
        Scene("5  It knows what it is",
              "The measured meshes have the right sizes but look like blobs, so Spark names the object and rebuilds it from clean parts. Each round three Meta models propose a design, and the video picks the best.",
              ["Spark names the object and rebuilds each part from the measured sizes",
               "three Meta models propose; the video picks the design",
               "collisions, joints and physics stay measured"],
              sequence([DESIGN / "drawer2_lidar" / "compare.mp4", LAMP / "compare.mp4"],
                       ["office pedestal cabinet with three drawers", "LED desk lamp"])),
        Scene("6  It looks real",
              "For a curved object, SAM 3D Objects builds a realistic mesh, placed by our LiDAR depth: the depth the "
              "phone measured lies within half a millimetre of the mouse's mesh. Circling it with the phone gives "
              "its size from every side.",
              ["SAM 3D Objects: the measured depth lies 0.5 mm from its mesh (median)",
               "masks from every side: 121 x 75 x 55 mm  (spec 125 x 84 x 51)"],
              video_scene(MOUSE / "compare.mp4")),
        Scene("7  The robot uses it",
              "In MuJoCo physics the Panda grips the handle Spark drew and pulls the drawer out of its measured "
              "cabinet, then carries the mouse and turns it. Only the grip moves them.",
              ["drawer out 199 mm by its handle, the cabinet's measured collision on, and back in",
               "mouse carried 18 cm, then brought back turned 90 deg"],
              pair(DESIGN / "drawer2_lidar" / "robot_demo.mp4", MOUSE / "robot_demo.mp4",
                   ("the measured drawer", "the measured mouse"))),
        Scene("8  Physics",
              "Spark judges materials and masses. Where a part stands with no hand on it, the joint friction is "
              "measured, and checked by letting go in MuJoCo.",
              ["hinge friction measured where the lid stands on its own: 0.28 N m",
               "let go in MuJoCo: stays (+0.1 deg) vs. falls (+11.5 deg) without it"],
              video_scene(RUN / "physics_demo.mp4")),
        Scene("9  Benchmark",
              "On the RBO motion-capture benchmark, laptops and books come out with a median hinge axis error of two degrees, under three on open weights alone; box lids still fail.",
              ["laptops and books: hinge axis 2.2 deg (Meta API), 2.8 deg (open weights only), median",
               "box lids still fail: SAM loses the lid at 640 x 480"],
              stills([ROOT / "data" / "rbo" / "results" / "laptop04.jpg"],
                     ["laptop04: red = measured, green = motion capture"])),
        Scene("10  URDF Doctor",
              "URDF Doctor checks the models you already have against a video. Three injected bugs and a sign flip "
              "in a joint log: all found.",
              ["URDF Doctor: an existing URDF vs. the motion in the video"],
              card([("URDF Doctor", 64, True, white),
                    ("laptop_buggy.urdf (hinge +25 mm, axis tilted 6 deg, limit 90 deg)", 40, True, grey),
                    ("axis 7.6 deg off   |   hinge 18 mm off   |   video needs 127 deg, limit is 90", 38, False,
                     (255, 140, 120)),
                    ("joint log with the sign reversed:  log = -1.00 x expected", 38, False, (255, 140, 120))],
                   top=300)),
        Scene("11  Open weights",
              "Every model it calls is a Meta model, and it runs end to end on open weights, Muse Spark Open or Muse Glimmer on our own GPU, reproduced from a fresh clone.",
              ["--open-weights: Muse Spark Open + SAM 3.1 weights   |   --glimmer: Muse Glimmer 30B on our GPU"],
              card([("Open weights, end to end", 64, True, white),
                    ("run_agent.py --open-weights [--glimmer]", 44, False, blue),
                    ("Muse Spark Open (Baseten) or Muse Glimmer 30B (our Modal endpoint)", 40, False, grey),
                    ("+ SAM 3.1 / SAM 3 weights on a Modal GPU", 40, False, grey),
                    ("same laptop: lid 305 x 209 mm (Spark Open), 307 x 201 mm (Glimmer)", 36, False, grey),
                    ("fresh clone, README commands: lid 306 x 213 mm, replay 96% / 93%", 36, False, grey)],
                   top=310)),
        Scene("Try it",
              "Everything is in the browser, with URDF downloads and every step the agent took. One video in, "
              "a model a robot can use out.",
              ["sunghyunc.github.io/video2urdf   |   github.com/SungHyunC/video2urdf"],
              stills([ASSETS / "viewer_trace.jpg"])),
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
