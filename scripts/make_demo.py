"""Cut the demo video from the project's own outputs: captions burned in, and a voice if given.

Each scene has a narration line. With --voice-dir, every scene lasts its voiced line plus
LEAD + TAIL seconds (paced to the voice), and the lines are laid on one track at their scene's
start + LEAD, loudness-normalised and muxed. Without a voice a scene lasts about 2.6 words a
second (a calm voice-over). Either way the script with each line's start time is written next
to the video (docs/demo_script.md for docs/demo.mp4), so a voice recorded from it fits the cut.

Usage:
  .venv/bin/python scripts/make_demo.py --lines lines.json     # write the lines to voice, then stop
  # voice them: one 16-bit PCM WAV per line, NN.wav in scene order, + that lines.json, in one folder
  .venv/bin/python scripts/make_demo.py --voice-dir data/demo_assets/voice_v3_af_heart \
      [--out docs/demo.mp4] [--narration-out data/demo_assets/narration_v3_af_heart.m4a]
  .venv/bin/python scripts/make_demo.py --voice recorded.m4a  # a recorded track, muxed as it is

--baseline adds "Why not just ask Spark for a URDF?" (data/spark_alone/report.md).
The robot scene shows the Panda opening and shutting the completed microwave's door at twice
its speed (data/demo_assets/robot_door.mp4); --door-clip PATH picks another clip, --no-door
leaves it (and its words) out.

Inputs: the LiDAR runs in data/tracks_agent*, data/tracks_design and data/viewer_new, their
phone videos, docs/models/microwave_lidar/phone.mp4, and the stills and clips in
data/demo_assets/ (microwave_completion.mp4 from scripts/render_completion_demo.py).
"""

import argparse
import json
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

W, H, FPS = 1920, 1080, 30
WORDS_PER_S = 2.6  # without a voice: a calm voice-over
LEAD, TAIL = 0.3, 0.3  # s of silence before and after each voiced line (fades take 0.2 s)
MAX_SECONDS = 178.0  # the rules cut the demo at 3:00
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
BG = (18, 18, 20)
ACCENT = (110, 160, 255)
RUN = ROOT / "data" / "tracks_agent" / "laptop_lidar"
PHONE = ROOT / "data" / "real" / "laptop_lidar" / "laptop_lidar.mp4"
ASSETS = ROOT / "data" / "demo_assets"
DESIGN = ROOT / "data" / "tracks_design"  # runs with Spark's shape design applied
LAMP = ROOT / "data" / "tracks_agent2" / "stand_lidar"  # the LiDAR desk lamp (rods on hinges)
MOUSE = ROOT / "data" / "tracks_agent2" / "mouse_lidar"  # the black mouse (visual hull + SAM 3D)
MICROWAVE_PHONE = ROOT / "docs" / "models" / "microwave_lidar" / "phone.mp4"  # the viewer tab's (reviewed) frames
SPARK_ALONE = ROOT / "data" / "spark_alone"
DOOR_ROBOT = ASSETS / "robot_door.mp4"  # the Panda opening and shutting the completed microwave's door
NOW = {"scene_s": 0.0, "part_s": 0.0}  # seconds of the scene (and part) being drawn, for clips at a set speed


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


class Clip:
    """Frames of a video read forward only (scenes play them in order); opened on first use.
    By default the clip is stretched over its scene (or part); with ``speed`` it plays at that
    rate instead, ending with its scene (``align="end"``, still on its first frame until then)
    or starting with it (still on its last frame after)."""

    def __init__(self, path: Path, first: int = 0, last: int | None = None, speed: float | None = None,
                 align: str = "end"):
        self.path, self.first, self.last_arg, self.speed, self.align = Path(path), first, last, speed, align
        self.cap = None
        self.pos, self.img = -1, None

    def at(self, p: float) -> np.ndarray:
        if self.cap is None:
            if not self.path.exists():
                raise SystemExit(f"missing demo input: {self.path}")
            self.cap = cv2.VideoCapture(str(self.path))
            n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or FPS
            self.last = n - 1 if self.last_arg is None else self.last_arg
        if self.speed:
            span = NOW["part_s"] or NOW["scene_s"]
            plays = (self.last - self.first) / self.fps / self.speed  # seconds on screen
            t = p * span - (max(0.0, span - plays) if self.align == "end" else 0.0)
            want = self.first + int(round(min(max(t, 0.0) * self.speed * self.fps, self.last - self.first)))
        else:
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


def place(canvas: np.ndarray, img: np.ndarray, box) -> tuple[int, int, int, int]:
    """Fit img into box (centred); returns where it landed (x, y, w, h)."""
    im, (x, y) = fit(img, box)
    canvas[y: y + im.shape[0], x: x + im.shape[1]] = im
    return x, y, im.shape[1], im.shape[0]


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
    title: str
    narration: str
    captions: list[str]  # shown in turn over the scene (at caption_starts, if given)
    draw: Callable[[float], np.ndarray]  # progress 0..1 -> 1920x1080 BGR frame without captions
    min_s: float = 4.0
    hold: float = 0.0  # s kept on screen after the line (an end card)
    caption_starts: list[float] | None = None
    numbered: bool = True
    section: str = ""  # the label shown, e.g. "3  Measure and verify" (set by number())
    speech: float | None = None  # s of the voiced line, when paced to a voice

    @property
    def seconds(self) -> float:
        if self.speech is not None:
            return LEAD + self.speech + TAIL + self.hold
        return max(self.min_s, len(self.narration.split()) / WORDS_PER_S + 1.0) + self.hold

    def caption_at(self, p: float) -> str:
        if self.caption_starts:
            return self.captions[max(i for i, s in enumerate(self.caption_starts) if s <= p + 1e-9)]
        return self.captions[min(int(p * len(self.captions)), len(self.captions) - 1)]


@dataclass
class Part:
    """A piece of a scene's narration with its own captions and, optionally, its own visual."""
    narration: str
    draw: Callable[[float], np.ndarray] | None = None
    captions: list[str] = field(default_factory=list)


def in_parts(title: str, parts: list[Part], draw: Callable[[float], np.ndarray] | None = None, **kw) -> Scene:
    """A scene whose parts play in turn, each over the share of the scene its words take (the
    voice reads them in that order at an even pace). ``draw`` runs over the whole scene
    instead of the parts' own visuals."""
    words = np.array([len(p.narration.split()) for p in parts], float)
    edges = np.concatenate([[0.0], np.cumsum(words) / words.sum()])

    def by_part(p):
        i = min(int(np.searchsorted(edges, p, side="right")) - 1, len(parts) - 1)
        NOW["part_s"] = NOW["scene_s"] * (edges[i + 1] - edges[i])
        return parts[i].draw(float(np.clip((p - edges[i]) / (edges[i + 1] - edges[i]), 0, 1)))

    caps, starts = [], []
    for i, part in enumerate(parts):
        for j, c in enumerate(part.captions):
            caps.append(c)
            starts.append(edges[i] + (edges[i + 1] - edges[i]) * j / len(part.captions))
    return Scene(title, " ".join(p.narration for p in parts), caps, draw or by_part, caption_starts=starts, **kw)


def number(sc: list[Scene]) -> list[Scene]:
    k = 0
    for s in sc:
        if s.numbered:
            k += 1
            s.section = f"{k}  {s.title}"
        else:
            s.section = s.title
    return sc


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


def stills(paths: list[Path], labels: list[str] | None = None, shares: list[float] | None = None):
    """Images in turn (equal shares of the scene unless given)."""
    imgs = []
    edges = np.cumsum([0.0] + list(shares or [1.0] * len(paths)))
    edges /= edges[-1]

    def draw(p):
        if not imgs:
            for q in paths:
                img = cv2.imread(str(q))
                if img is None:
                    raise SystemExit(f"missing demo input: {q}")
                imgs.append(img)
        i = min(int(np.searchsorted(edges, p, side="right")) - 1, len(imgs) - 1)
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


def sequence(paths: list[Path], labels: list[str]):
    """Videos one after another, each over an equal share of the scene."""
    clips = [Clip(q) for q in paths]

    def draw(p):
        i = min(int(p * len(clips)), len(clips) - 1)
        c = blank()
        place(c, clips[i].at(min(1.0, p * len(clips) - i)), (60, 90, W - 120, H - 260))
        return text_layer(c, [((W - 60, 40), labels[i], 30, False, (200, 200, 200), "ra")])
    return draw


def before_after(top: Path, bottom: Path, labels: tuple[str, str], active: int):
    """Two wide stills stacked, the one not being talked about dimmed (a first fit above, the
    fit after the agent's correction below)."""
    imgs = []

    def draw(p):
        if not imgs:
            for q in (top, bottom):
                img = cv2.imread(str(q))
                if img is None:
                    raise SystemExit(f"missing demo input: {q}")
                imgs.append(img)
        c = blank()
        items = []
        for i, (img, label, y) in enumerate(zip(imgs, labels, (100, 520))):
            shown = img if i == active else (img * 0.35).astype(np.uint8)
            place(c, shown, (60, y + 44, W - 120, 340))
            items.append(((70, y), label, 30, i == active, (240, 240, 240) if i == active else (120, 120, 125), "la"))
        return text_layer(c, items)
    return draw


def clips_grid(clips: list[tuple]):
    """Robot clips played together: two side by side; three as one large and two stacked.
    clips: (path, label) or (path, label, speed) -- see Clip."""
    cs = [Clip(c[0], speed=c[2] if len(c) > 2 else None) for c in clips]
    grey = (200, 200, 200)
    if len(cs) <= 2:
        boxes = [(40, 110, W // 2 - 60, H - 290), (W // 2 + 20, 110, W // 2 - 60, H - 290)][: len(cs)]
    else:
        big_w = 1180
        boxes = [(40, 100, big_w, 700), (40 + big_w + 30, 110, W - big_w - 110, 300),
                 (40 + big_w + 30, 500, W - big_w - 110, 300)]

    def draw(p):
        c = blank()
        items = []
        for clip, (_, label, *_), box in zip(cs, clips, boxes):
            x, y, w, h = place(c, clip.at(p), box)
            items.append(((x + w // 2, y + h + 10), label, 28 if len(cs) <= 2 else 26, False, grey, "ma"))
        return text_layer(c, items)
    return draw


def microwave_scene():
    """The phone video (its back is never seen) next to the completed model in MuJoCo
    (scripts/render_completion_demo.py): measured front, estimated body, the probe."""
    phone = Clip(MICROWAVE_PHONE, 150, 700)  # shut, opened (the cavity seen through the door), held open
    model = Clip(ASSETS / "microwave_completion.mp4")
    grey = (200, 200, 200)

    def draw(p):
        c = blank()
        x, y, w, h = place(c, phone.at(p), (40, 140, 780, 585))
        x2, y2, w2, h2 = place(c, model.at(p), (850, 95, W - 850 - 40, 790))
        return text_layer(c, [((x + w // 2, y + h + 14), "phone video: its back is never seen", 28, False, grey, "ma"),
                              ((x2 + w2 // 2, y2 + h2 + 8), "the completed model in MuJoCo", 28, False, grey, "ma")])
    return draw


def scenes(baseline: bool = False, door_clip: Path | None = DOOR_ROBOT) -> list[Scene]:
    lamp_agent = LAMP / "agent"
    white, grey, blue, red = (240, 240, 240), (170, 170, 175), ACCENT, (255, 140, 120)
    robots = [(DESIGN / "drawer2_lidar" / "robot_policy.mp4", "a learned policy opens the measured drawer")]
    robot_line = ("In MuJoCo, a reinforcement learning policy trained on the measured drawer pinches its handle and "
                  "pulls it open, nineteen times in twenty on the full arm. The Panda also carries the mouse.")
    robot_caps = ["PPO policy trained on Modal on the measured drawer: 99/100 held-out, 19/20 on the arm",
                  "98% of the drawer's travel made while pinching the handle; simulation only"]
    if door_clip:  # at twice its speed, ending with the line that names it
        robots.append((door_clip, "the completed microwave's door, opened and shut (2x)", 2.0))
        robot_line = robot_line.replace("The Panda also carries the mouse.",
                                        "The Panda also opens and shuts the microwave door and carries the mouse.")
        robot_caps.append("the Panda pulls the measured door open to 46 deg and pushes it shut, body collision on")
    robots.append((MOUSE / "robot_demo.mp4", "the measured mouse, carried and turned"))
    fits = (ASSETS / "mw_first_fit.jpg", ASSETS / "mw_refit.jpg",
            ("Spark's first fit, drawn over the video", "after Spark re-tracked and fitted again"))
    sc = [
        Scene("Video2URDF",
              "Robots need simulation models of the objects they handle, and people build them by hand. "
              "Video2URDF measures them from one phone video, and a robot can use them right away.",
              ["One phone video in. A model a robot can use out.",
               "a Franka Panda in MuJoCo works four objects, each measured from one video"],
              video_scene(ASSETS / "robot_grid.mp4"), numbered=False),
        Scene("Capture",
              "I film the object with an iPhone while I move it: no markers, no card. The LiDAR gives metric depth.",
              ["iPhone LiDAR (Stray Scanner), handheld: no markers, no card"],
              video_scene(PHONE, 0, 922)),
        Scene("Agent",
              "Muse Spark runs the pipeline as an agent. It watches the video, prompts SAM 3.1 with nouns, picks the "
              "fitter, and judges what it sees: for this desk lamp, it decided the pole never moves.",
              ["Muse Spark watches the video and looks at frames",
               "it prompts SAM 3.1: 'silver metal rod'",
               "it picks fit_chain: two rods on hinges, fitted in 3D from their depth"],
              stills([lamp_agent / "step009.jpg", lamp_agent / "step016.jpg", lamp_agent / "step032.jpg"],
                     ["look", "segment 'silver metal rod'", "fit_chain: the model over the video"])),
        Scene("Measure and verify",
              "MuJoCo replays the model from the phone's camera and covers ninety five percent of the lid: three "
              "oh nine by two oh five millimetres, against a spec of three oh four by two twelve.",
              ["MuJoCo replays the model from the phone's camera pose, every frame",
               "covers 95% of the lid, 94% of the base",
               "lid 309 x 205 mm  (spec: 304 x 212 mm)"],
              side_by_side(PHONE, RUN / "replay3d.mp4", 0, 922)),
    ]
    if baseline:
        laptop = SPARK_ALONE / "laptop_lidar" / "run2" / "overlay_max_open.jpg"
        drawer = SPARK_ALONE / "drawer2_lidar" / "travel_still.jpg"
        sc.append(in_parts("Why not just ask Spark?", [
            Part("Why not just ask Spark for a URDF? Asked plainly, it recalls this MacBook's spec",
                 stills([laptop], ["one plain request, no tools: Muse Spark's URDF on the measured laptop"]),
                 ["asked plainly, no tools: the same Muse Spark recalls the MacBook's spec, 304 x 212 mm: right"]),
            Part("but gives every drawer three sixty to four hundred millimetres of travel where the video shows two "
                 "forty eight;",
                 stills([drawer], ["the measured drawer (left) vs. Spark alone's travel (right)"]),
                 ["drawer travel 360-400 mm in all 5 answers; the video shows 248 mm"]),
            Part("nothing says which numbers are right. With tools, it measures them.",
                 stills([laptop], ["its lid stops at its own 110 deg limit; the video opens it 117 deg"]),
                 ["joint range shorter than the video's opening in 8 of 19 hinge answers",
                  "nothing in an answer says which numbers are right; with tools it measures them"]),
        ]))
    sc += [
        in_parts("It checks itself", [
            Part("Spark checks its own work. On a microwave it had never seen, its first model was wrong: a door "
                 "twice too big and no body.",
                 before_after(*fits, 0), ["first fit, never-seen microwave: door 795 x 506 mm, the body not covered"]),
            Part("It watched the replay, re-tracked the floor and the body, and fitted again: now the door swings on "
                 "a vertical hinge, where the real one is.",
                 before_after(*fits, 1),
                 ["after Spark re-tracked floor and body: hinge 1.3 deg off vertical, door 361 x 255 mm"]),
        ]),
        in_parts("It builds what it never saw", [
            Part("This microwave sits in a shelf; its back was never filmed.",
                 captions=["a microwave in a shelf: its back was never filmed"]),
            Part("The front is measured; the video bounds the depth to three thirty four to three eighty seven "
                 "millimetres.",
                 captions=["measured: the front, 477 x 269 mm; through the open door, the cavity's back 329 mm behind it",
                           "the video bounds the depth: 334-387 mm"]),
            Part("Muse Spark recognises a Samsung twenty three litre and puts it at three sixty five, labelled an "
                 "estimate.",
                 captions=["Muse Spark recognises a Samsung 23 L: depth 365 mm, labelled as its estimate"]),
            Part("A probe pushed at the back stops on it.",
                 captions=["a probe pushed at the back stops on it (the measured-only model let it through)"]),
        ], draw=microwave_scene()),
        Scene("It knows what it is",
              "Measured meshes have the right sizes but look like blobs, so Spark names the object and rebuilds it "
              "from clean parts: three Meta models propose, and the video picks.",
              ["Spark names the object and rebuilds each part from the measured sizes",
               "three Meta models propose; the video picks the design",
               "collisions, joints and physics stay measured"],
              sequence([DESIGN / "drawer2_lidar" / "compare.mp4", LAMP / "compare.mp4"],
                       ["office pedestal cabinet with three drawers", "LED desk lamp"])),
        Scene("It looks real",
              "For a curved object, SAM 3D Objects builds a realistic mesh, placed by our LiDAR depth: the phone's "
              "depth points lie half a millimetre from it, at the median.",
              ["SAM 3D Objects: the measured depth lies 0.5 mm from its mesh (median)",
               "masks from every side: 121 x 75 x 55 mm  (spec 125 x 84 x 51)"],
              video_scene(MOUSE / "compare.mp4")),
        in_parts("The robot uses it", [
            Part(robot_line, clips_grid(robots), robot_caps),
            Part("Spark judges masses; joint friction is measured where a part stands alone.",
                 video_scene(RUN / "physics_demo.mp4"),
                 ["friction measured where the lid stands alone: 0.28 N m; let go, it moves 0.1 deg (11.5 without)"]),
        ]),
        Scene("Benchmark",
              "On the RBO motion-capture benchmark, laptops and books get a median hinge axis error under two "
              "degrees, under three on open weights; box lids still fail.",
              ["laptops and books, median hinge axis: 1.7 deg (Meta API, 6 of 7), 2.8 deg (open weights, 7 of 7)",
               "box lids still fail: SAM loses the lid at 640 x 480"],
              stills([ASSETS / "rbo_laptop04.jpg"],
                     ["laptop04: red = measured, green = motion capture"])),
        Scene("URDF Doctor",
              "URDF Doctor checks the models you already have against a video. Three injected bugs and a sign flip "
              "in a joint log: all found.",
              ["URDF Doctor: an existing URDF vs. the motion in the video"],
              card([("URDF Doctor", 64, True, white),
                    ("laptop_buggy.urdf (hinge +25 mm, axis tilted 6 deg, limit 90 deg)", 40, True, grey),
                    ("axis 7.6 deg off   |   hinge 18 mm off   |   video needs 127 deg, limit is 90", 38, False, red),
                    ("joint log with the sign reversed:  log = -1.00 x expected", 38, False, red)],
                   top=300)),
        Scene("Open weights",
              "Every model it calls is a Meta model, and it runs end to end on open weights, Muse Spark Open or "
              "Muse Glimmer on our own GPU, reproduced from a fresh clone.",
              ["--open-weights: Muse Spark Open + SAM 3.1 / SAM 3 weights  |  --glimmer: Muse Glimmer 30B"],
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
              stills([ASSETS / "viewer_trace.jpg"]), numbered=False, hold=0.5),
    ]
    return number(sc)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: 16-bit PCM WAV expected")
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        if w.getnchannels() > 1:
            a = a.reshape(-1, w.getnchannels()).mean(axis=1)
        return a, w.getframerate()


def pace(sc: list[Scene], voice_dir: Path) -> tuple[list[np.ndarray], int]:
    """Each scene's length from its voiced line (NN.wav), after checking the folder voiced
    exactly these lines."""
    voiced = json.loads((voice_dir / "lines.json").read_text())
    if [v["narration"] for v in voiced] != [s.narration for s in sc]:
        raise SystemExit(f"{voice_dir} voices other lines than these scenes: write them with --lines and voice them again")
    clips, rates = [], set()
    for i, s in enumerate(sc):
        a, sr = read_wav(voice_dir / f"{i:02d}.wav")
        s.speech = len(a) / sr
        clips.append(a)
        rates.add(sr)
    if len(rates) != 1:
        raise SystemExit(f"{voice_dir}: the clips have different sample rates {rates}")
    return clips, rates.pop()


def timeline(sc: list[Scene]) -> list[tuple[int, int]]:
    """(first frame, frame count) of each scene."""
    out, f = [], 0
    for s in sc:
        n = int(round(s.seconds * FPS))
        out.append((f, n))
        f += n
    return out


def narration_track(sc, clips, sr, frames, total_s: float, wav_out: Path) -> list[dict]:
    """The voiced lines on one track, each at its scene's start + LEAD."""
    track = np.zeros(int(round(total_s * sr)), np.float32)
    rows = []
    for s, a, (f0, n) in zip(sc, clips, frames):
        start, end = f0 / FPS, (f0 + n) / FPS
        i0 = int(round((start + LEAD) * sr))
        track[i0: i0 + len(a)] += a[: max(0, len(track) - i0)]
        rows.append({"section": s.section, "start": round(start, 2), "seconds": round(end - start, 2),
                     "speech_start": round(start + LEAD, 2), "speech_end": round(start + LEAD + len(a) / sr, 2),
                     "next_start": round(end, 2)})
    pcm = (np.clip(track, -1, 1) * 32767).astype("<i2")
    with wave.open(str(wav_out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "demo.mp4")
    ap.add_argument("--script-out", type=Path,
                    help="the narration with start times (default: docs/demo_script.md for docs/demo.mp4, "
                         "else <out>_script.md)")
    ap.add_argument("--lines", type=Path, help="write the scenes' lines (JSON) for a voice to read, then stop")
    ap.add_argument("--voice-dir", type=Path, help="NN.wav per line + lines.json: pace the cut to them and mux them")
    ap.add_argument("--narration-out", type=Path, help="also keep the narration track (.m4a)")
    ap.add_argument("--voice", type=Path, help="a recorded narration of the whole cut to mux in as it is")
    ap.add_argument("--baseline", action="store_true", help="add 'Why not just ask Spark for a URDF?'")
    ap.add_argument("--door-clip", type=Path, default=DOOR_ROBOT,
                    help="the robot clip opening and shutting the microwave door (robot scene)")
    ap.add_argument("--no-door", action="store_true", help="leave the door clip out of the robot scene")
    ap.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    args = ap.parse_args()
    sc = scenes(baseline=args.baseline, door_clip=None if args.no_door else args.door_clip)
    if args.lines:
        args.lines.write_text(json.dumps([{"section": s.section, "narration": s.narration} for s in sc], indent=1))
        print(f"{len(sc)} lines -> {args.lines}")
        return
    clips, sr = pace(sc, args.voice_dir) if args.voice_dir else (None, None)
    frames = timeline(sc)
    total = sum(n for _, n in frames) / FPS
    for s, (f0, n) in zip(sc, frames):
        speech = f"speech {s.speech:5.1f} s" if s.speech is not None else ""
        print(f"{f0 / FPS:6.1f} s  {s.section[:34]:34s} {n / FPS:5.1f} s  {speech}")
    print(f"{len(sc)} scenes, {total:.1f} s")
    if total > args.max_seconds:
        raise SystemExit(f"{total:.1f} s is over {args.max_seconds:.0f} s: tighten the lines")

    script_out = args.script_out or (args.out.parent / "demo_script.md" if args.out.parent == ROOT / "docs"
                                     else args.out.with_name(args.out.stem + "_script.md"))
    pace_note = (f"Paced to a synthesized voice: each line starts at its time, after about {LEAD:.1f} s of silence."
                 if args.voice_dir else "Read at a calm pace; each line starts at its time.")
    script = ["# Demo narration", "", f"Total {total:.0f} s. {pace_note}", ""]
    for s, (f0, _) in zip(sc, frames):
        t = f0 / FPS
        script.append(f"**{int(t // 60)}:{t % 60:04.1f}  {s.section}**  \n{s.narration}\n")
    script_out.write_text("\n".join(script))

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for s, (_, n) in zip(sc, frames):
            NOW["scene_s"] = NOW["part_s"] = n / FPS
            for i in range(n):
                p = i / max(n - 1, 1)
                frame = caption(s.draw(p).copy(), s.section, s.caption_at(p))
                fade = min(1.0, i / 6, (n - 1 - i) / 6)  # short fades between scenes
                writer.write((frame * max(fade, 0)).astype(np.uint8))
        writer.release()
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw)]
        if args.voice_dir:
            wav = Path(tmp) / "narration.wav"
            rows = narration_track(sc, clips, sr, frames, total, wav)
            for r in rows:
                ok = r["speech_end"] <= r["next_start"] - 0.2
                print(f"{r['start']:6.1f} s  {r['section'][:34]:34s} speech {r['speech_start']:6.1f}-{r['speech_end']:6.1f} s, "
                      f"next {r['next_start']:6.1f} s  {'OK' if ok else 'OVER'}")
            m4a = args.narration_out or Path(tmp) / "narration.m4a"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-af",
                            "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-c:a", "aac", "-b:a", "160k", str(m4a)],
                           check=True)
            cmd += ["-i", str(m4a), "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
        elif args.voice:
            cmd += ["-i", str(args.voice), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "160k", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(args.out)]
        subprocess.run(cmd, check=True)
    print(f"{args.out} ({args.out.stat().st_size / 1e6:.1f} MB), script {script_out}")


if __name__ == "__main__":
    main()
