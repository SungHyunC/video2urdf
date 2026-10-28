"""Muse Spark drives the pipeline: it looks at the video, names the parts, finds them with
SAM 3.1, sets the metric scale from a card, tracks, fits, checks the replay and re-fits.

Every step lands in <track_dir>/agent/log.md with the images Spark saw.
"""

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI

from .pipeline import CARD_LONG_SIDE, Fit, fit, long_side_px, slug, video_info
from .sam import Detection, SamClient, parse

MODEL = os.environ.get("AGENT_MODEL", "muse-spark-1.3-contributor")
BASE_URL = os.environ.get("AGENT_BASE_URL", "https://api.meta.ai/v1")
API_KEY_ENV = os.environ.get("AGENT_API_KEY_ENV", "MODEL_API_KEY")
MAX_STEPS = 20  # Spark turns; the last one must call finish
ASSUMED_VIEW_M = 0.6  # frame's long side in metres when no card sets the scale
VIEW_WIDTH = 540  # px; frames shown to Spark
COLORS = [(0, 200, 255), (255, 80, 80), (80, 220, 80), (255, 0, 255), (255, 255, 0), (0, 128, 255)]

SYSTEM = """You are the reasoning core of Video2URDF. From a phone video of an articulated object
(desk lamp, laptop, drawer, scissors...) you build a kinematic model: rigid links joined by
revolute or prismatic joints, written as URDF/MJCF, then proven by replaying it in MuJoCo over
the video. The camera roughly faces the plane of motion, so the model is planar.

Tools:
- look(frames): see video frames, to understand the object and how it moves.
- segment(prompt, frame): SAM 3.1 finds every match of a short concept prompt on one frame.
  Returns candidate ids and an annotated image. One concept per call. SAM is literal: short
  visual noun phrases work ("white base", "metal pole", "laptop lid", "laptop keyboard",
  "drawer front"); functional names often find nothing. If a prompt fails or grabs the wrong
  region, rephrase by colour, material or shape, or try another frame.
- measure_scale(candidate): metric scale from a segmented ID-1 card (85.6 mm long side), if one
  is visible. Without it fit assumes a rough scale, so look for a card first.
- track(parts): track the chosen candidates through the whole video (both directions). The
  FIRST part is the root: the part that stays fixed (base, keyboard, cabinet). One candidate per
  rigid part; skip the hand, the card and the background. This is the expensive call.
- fit(parents, fixed): fit joints between tracked parts and replay the model in MuJoCo. By
  default the tree is chosen automatically; pass parents {child: parent} to override it and
  fixed [child...] to make a joint rigid. Returns joints, a fit residual and per-part
  silhouette IoU, plus an image of the worst replay frames.
- finish(report): end with a short report for a robotics developer.

How to judge a fit: the tree must make physical sense (each joint where two parts touch).
Replay IoU above ~0.7 is good; thin parts score lower. A joint whose range is a few degrees
probably did not move in this video: say so, and make it fixed only if the object really has
no joint there. If a part is mis-segmented (wrong region, swallowed by a neighbour) or a
part's motion is inconsistent, re-segment and re-track rather than accept a bad fit.
Budget: about 20 turns and at most 3 track calls; each turn tells you how many remain.
If the video can't give a good planar model (the part moves toward the camera, a part is
hidden most of the time, the camera faces the motion plane at a steep angle), stop early:
finish with the best model you have, say plainly what is wrong, and say how to reshoot.
Be decisive. Write every message in English."""

TOOLS = [
    {"name": "look", "description": "Show video frames (up to 4).",
     "parameters": {"type": "object", "properties": {"frames": {"type": "array", "items": {"type": "integer"}}},
                    "required": ["frames"]}},
    {"name": "segment", "description": "Run SAM 3.1 with one concept prompt on one frame.",
     "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "frame": {"type": "integer"}},
                    "required": ["prompt", "frame"]}},
    {"name": "measure_scale", "description": "Set metric scale from a segmented ID-1 card candidate.",
     "parameters": {"type": "object", "properties": {"candidate": {"type": "string"}}, "required": ["candidate"]}},
    {"name": "track", "description": "Track parts through the video. First part is the fixed root.",
     "parameters": {"type": "object", "properties": {"parts": {"type": "array", "items": {
         "type": "object", "properties": {"name": {"type": "string"}, "candidate": {"type": "string"}},
         "required": ["name", "candidate"]}}}, "required": ["parts"]}},
    {"name": "fit", "description": "Fit joints and replay in MuJoCo.",
     "parameters": {"type": "object", "properties": {
         "parents": {"type": "object", "additionalProperties": {"type": "string"},
                     "description": "Optional {child: parent} tree override."},
         "fixed": {"type": "array", "items": {"type": "string"}, "description": "Children whose joint is rigid."}}}},
    {"name": "finish", "description": "End with a report.",
     "parameters": {"type": "object", "properties": {"report": {"type": "string"}}, "required": ["report"]}},
]


def jpeg_url(img: np.ndarray, width: int = VIEW_WIDTH) -> str:
    small = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()


def read_frame(video: Path, index: int) -> np.ndarray:
    """Decode frames in order up to ``index``: seeking is unreliable on variable-frame-rate phone video."""
    cap = cv2.VideoCapture(str(video))
    for _ in range(index):
        cap.grab()
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"no frame {index}")
    return img


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    scale = img.shape[1] / 540
    cv2.putText(img, text, (int(10 * scale), int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (0, 0, 0), int(6 * scale))
    cv2.putText(img, text, (int(10 * scale), int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (255, 255, 255), int(2 * scale))
    return img


@dataclass
class Agent:
    video: Path
    track_dir: Path
    client: OpenAI = None
    sam: SamClient = None
    candidates: dict[str, tuple[str, Detection, int]] = field(default_factory=dict)  # id -> prompt, det, frame
    px_per_m: float | None = None
    scale_note: str = "no scale set"
    tracked: bool = False
    last_fit: Fit | None = None
    report: str | None = None
    images: list[str] = field(default_factory=list)  # data URLs queued for the next user turn
    n_segment: int = 0
    log_lines: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.client = self.client or OpenAI(base_url=BASE_URL, api_key=os.environ[API_KEY_ENV], timeout=600)
        self.sam = self.sam or SamClient()
        self.size, self.n, self.fps = video_info(self.video)
        self.out = self.track_dir / "agent"
        self.out.mkdir(parents=True, exist_ok=True)

    # --- logging -------------------------------------------------------------------------
    def log(self, text: str, image: np.ndarray | None = None):
        if image is not None:
            name = f"step{len(self.log_lines):03d}.jpg"
            k = min(1.0, 1600 / image.shape[1])
            cv2.imwrite(str(self.out / name), cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA))
            text += f"\n\n![]({name})"
        self.log_lines.append(text)
        (self.out / "log.md").write_text("\n\n".join(self.log_lines) + "\n")
        print(text.split("\n\n![]")[0], flush=True)

    def show(self, img: np.ndarray, caption: str, width: int = VIEW_WIDTH):
        """Queue an image for Spark's next turn (``width`` px wide) and log it."""
        self.images.append(jpeg_url(img, width))
        self.log(caption, img)

    # --- tools ---------------------------------------------------------------------------
    def tool_look(self, frames: list[int]) -> str:
        frames = [max(0, min(self.n - 1, int(f))) for f in frames[:4]]
        imgs = [label(read_frame(self.video, f), f"frame {f}") for f in frames]
        strip = cv2.hconcat([cv2.resize(i, (VIEW_WIDTH, round(i.shape[0] * VIEW_WIDTH / i.shape[1]))) for i in imgs])
        self.show(strip, f"look {frames}", width=strip.shape[1])
        return f"Showing frames {frames} (left to right)."

    def tool_segment(self, prompt: str, frame: int) -> str:
        frame = max(0, min(self.n - 1, int(frame)))
        img = read_frame(self.video, frame)
        dets = parse(self.sam.segment_image(img, prompt))
        self.n_segment += 1
        if not dets:
            return f"segment {prompt!r} on frame {frame}: no match. Rephrase or try another frame."
        h, w = img.shape[:2]
        view = img.copy()
        lines = []
        for i, d in enumerate(dets):
            cid = f"s{self.n_segment}.{i}"
            self.candidates[cid] = (prompt, d, frame)
            color = np.array(COLORS[i % len(COLORS)])
            m = d.full_mask(h, w)
            view[m] = (0.45 * view[m] + 0.55 * color).astype(np.uint8)
            cv2.rectangle(view, d.box[:2], d.box[2:], color.tolist(), 4)
            cv2.putText(view, cid, (d.box[0] + 8, d.box[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8)
            cv2.putText(view, cid, (d.box[0] + 8, d.box[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color.tolist(), 3)
            lines.append(f"{cid}: box {d.box}, confidence {d.confidence:.2f}, area {100 * m.mean():.1f}% of frame")
        self.show(label(view, f"segment '{prompt}' frame {frame}"), f"segment {prompt!r} frame {frame}")
        return f"segment {prompt!r} on frame {frame}:\n" + "\n".join(lines) + "\nCheck the annotated image."

    def tool_measure_scale(self, candidate: str) -> str:
        if candidate not in self.candidates:
            return f"unknown candidate {candidate}"
        _, d, _ = self.candidates[candidate]
        w, h = self.size
        px = long_side_px(d.full_mask(h, w))
        self.px_per_m = px / CARD_LONG_SIDE
        self.scale_note = f"from card {candidate}: long side {px:.0f} px = 85.6 mm"
        return (f"Scale set to {self.px_per_m:.0f} px/m ({self.scale_note}). Note: a card lying flat in front of "
                "the object is nearer the camera than the motion plane, so sizes come out a little small.")

    def tool_track(self, parts: list[dict]) -> str:
        chosen = []
        for p in parts:
            if p.get("candidate") not in self.candidates:
                return f"unknown candidate {p.get('candidate')}; segment first"
            chosen.append((slug(p["name"]), *self.candidates[p["candidate"]]))
        if len({c[0] for c in chosen}) != len(chosen):
            return "part names must be unique"
        boxes = {oid: (frame, d.box) for oid, (_, _, d, frame) in enumerate(chosen)}
        raw = self.sam.track_boxes(self.video, self.size, boxes)
        (self.track_dir / "tracks.sam.npz").unlink(missing_ok=True)
        (self.track_dir / "tracks.sam.txt").write_text(raw)
        (self.track_dir / "parts.json").write_text(json.dumps({
            oid: {"name": name, "prompt": prompt, "frame": frame, "box": list(d.box), "confidence": d.confidence}
            for oid, (name, prompt, d, frame) in enumerate(chosen)}, indent=2))
        counts = {}
        for det in parse(raw):
            counts[det.obj_id] = counts.get(det.obj_id, 0) + 1
        self.tracked = True
        self.last_fit = None
        return "Tracked: " + ", ".join(f"{name} {counts.get(oid, 0)}/{self.n} frames"
                                        for oid, (name, *_) in enumerate(chosen))

    def tool_fit(self, parents: dict | None = None, fixed: list | None = None) -> str:
        if not self.tracked:
            return "track parts first"
        if self.px_per_m is None:
            w, h = self.size
            self.px_per_m = max(w, h) / ASSUMED_VIEW_M
            self.scale_note = (f"ASSUMED ({self.px_per_m:.0f} px/m, as if the frame spanned {ASSUMED_VIEW_M} m): "
                               "no card was measured, so sizes are rough; angles are unaffected")
        parents = {slug(k): slug(v) for k, v in (parents or {}).items()} or None
        result = fit(self.video, self.track_dir, self.px_per_m, parents=parents,
                     fixed={slug(f) for f in (fixed or [])}, log=lambda s: None)
        self.last_fit = result
        sheet = self.replay_sheet(result)
        self.show(sheet, "fit: worst replay frames", width=sheet.shape[1])
        return (result.summary() + f"\nScale: {self.scale_note}.\n"
                "The image shows the replay (colored model over the video) at the worst-IoU frames.")

    def replay_sheet(self, result: Fit) -> np.ndarray:
        worst = sorted({int(np.nanargmin(v)) for v in result.ious.values()})
        frames = (worst + [int(f) for f in np.linspace(0, self.n - 1, 4)])[:4]
        cap = cv2.VideoCapture(str(self.track_dir / "replay.mp4"))
        imgs, t, want = [], 0, sorted(set(frames))
        while want:
            ok, img = cap.read()
            if not ok:
                break
            if t == want[0]:
                imgs.append(label(cv2.resize(img, (VIEW_WIDTH, round(img.shape[0] * VIEW_WIDTH / img.shape[1]))),
                                  f"frame {t}"))
                want.pop(0)
            t += 1
        return cv2.hconcat(imgs)

    def tool_finish(self, report: str) -> str:
        self.report = report
        return "done"

    # --- loop ----------------------------------------------------------------------------
    def run(self) -> str:
        w, h = self.size
        intro = (f"Video {self.video.name}: {self.n} frames at {self.fps:.0f} fps, {w}x{h}. "
                 "Build its kinematic model. Frames below are spread over the video.")
        self.log(f"# Video2URDF agent: {self.video.name}\n\nmodel {MODEL}")
        self.tool_look([0, self.n // 3, 2 * self.n // 3, self.n - 1])
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [{"type": "text", "text": intro}] + self._take_images()}]
        tools = [{"type": "function", "function": t} for t in TOOLS]
        for step in range(MAX_STEPS):
            t0 = time.time()
            r = self.client.chat.completions.create(model=MODEL, messages=messages, tools=tools, max_tokens=6000)
            msg = r.choices[0].message
            calls = msg.tool_calls or []
            self.log(f"## Spark ({time.time() - t0:.0f}s)\n\n{(msg.content or '').strip()}")
            messages.append({"role": "assistant", "content": msg.content or "",
                             **({"tool_calls": [c.model_dump() for c in calls]} if calls else {})})
            if not calls:
                messages.append({"role": "user", "content": "Call a tool, or finish with a report."})
                continue
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                    self.log(f"**{c.function.name}** `{json.dumps(args)}`")
                    result = getattr(self, f"tool_{c.function.name}")(**args)
                except Exception as e:  # report tool errors back to Spark instead of crashing the run
                    result = f"error: {type(e).__name__}: {e}"
                self.log(f"> {result}".replace("\n", "\n> "))
                messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
            if self.report is not None:
                break
            left = MAX_STEPS - step - 1
            note = f"{left} turns left." + (" Call finish now with your report." if left <= 1 else "")
            messages.append({"role": "user", "content": [{"type": "text", "text": note}] + self._take_images()})
        if self.report is None:  # out of turns; the API can't force a finish call, so ask for text
            messages.append({"role": "user", "content": "Out of turns. Write your final report now as plain text."})
            r = self.client.chat.completions.create(model=MODEL, messages=messages, max_tokens=6000)
            self.report = (r.choices[0].message.content or "").strip() or "(no report)"
            self.log(f"## Spark (final report, out of turns)\n\n{self.report}")
        report = self.report
        (self.out / "report.md").write_text(report + "\n")
        return report

    def _take_images(self) -> list[dict]:
        out = [{"type": "image_url", "image_url": {"url": u}} for u in self.images]
        self.images = []
        return out
