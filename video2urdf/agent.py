"""Muse Spark drives the pipeline: it looks at the video, names the parts, finds them with
SAM 3.1, sets the metric scale from a card, tracks, fits, checks the replay and re-fits.

Every step lands in <track_dir>/agent/log.md with the images Spark saw.
"""

import base64
import json
import os
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI

from . import lidar, physics
from .pipeline import CARD_LONG_SIDE, Fit, fit, load_parts, long_side_px, read_frame, slug, video_info
from .sam import Detection, SamClient, parse, replace_tracked_part, sam_client
from .track_checks import (alternate_plate_hint, base_fragment_warning, named_base_warning,
                           plate_covers_base_warning, suggested_open_anchors)

MODEL = os.environ.get("AGENT_MODEL", "muse-spark-1.3-contributor")
BASE_URL = os.environ.get("AGENT_BASE_URL", "https://api.meta.ai/v1")
API_KEY_ENV = os.environ.get("AGENT_API_KEY_ENV", "MODEL_API_KEY")
MAX_IMAGES = int(os.environ.get("AGENT_MAX_IMAGES", "0"))  # per request; 0 = no cap (Baseten allows 8)
MAX_VIDEOS = 2  # videos kept in the conversation; older ones become a note
VIDEO = os.environ.get("AGENT_VIDEO", "1") != "0"  # send Spark whole videos (Muse Spark 1.3 and Spark Open take mp4)
VIDEO_WIDTH, VIDEO_FPS = 480, 10  # videos shown to Spark are downscaled; timestamps are kept
MAX_STEPS = 22  # model turns; the last one must call finish
WHO = "Spark" if "spark" in MODEL.lower() else MODEL  # log headings
ASSUMED_VIEW_M = 0.6  # frame's long side in metres when no card sets the scale
VIEW_WIDTH = 540  # px; frames shown to Spark
COLORS = [(0, 200, 255), (255, 80, 80), (80, 220, 80), (255, 0, 255), (255, 255, 0), (0, 128, 255)]

SYSTEM = """You are the reasoning core of Video2URDF. From a phone video of an articulated object
(desk lamp, laptop, drawer, scissors...) you build a kinematic model: rigid links joined by
revolute or prismatic joints, written as URDF/MJCF, then proven by replaying it in MuJoCo over
the video. The camera roughly faces the plane of motion, so the model is planar.

Tools:
{video_tools}- look(frames): see video frames at full resolution, with their frame numbers.
- segment(prompt, frame): SAM 3.1 finds every match of a short concept prompt on one frame.
  Returns candidate ids and an annotated image. One concept per call. SAM is literal: short
  visual noun phrases work ("white base", "metal pole", "laptop lid", "laptop keyboard",
  "drawer front"); functional names often find nothing. If a prompt fails or grabs the wrong
  region, rephrase by colour, material or shape, or try another frame.
- measure_scale(candidate): metric scale from a segmented ID-1 card (85.6 mm long side), if one
  is visible. Without it fit assumes a rough scale, so look for a card first.
- track(parts): track the chosen candidates through the whole video (both directions). The
  FIRST part is the root: the part that stays fixed (base or cabinet). One candidate per
  rigid part; skip the hand, the card and the background. This is the expensive call.
- combine_tracks(name, parts, gate): combine separately tracked regions of the same rigid
  part (such as keyboard and touchpad) without another tracking call. Only use frames where
  gate is visible; the union covers observed regions and does not prove unseen edges.
- retrack_part(part, candidates): redo only one tracked part using new segment candidates on
  distinct frames. Other tracked parts are kept. Afterward refit and inspect the replay.
- fit(parents, fixed): fit joints between tracked parts and replay the model in MuJoCo. By
  default the tree is chosen automatically; pass parents {child: parent} to override it and
  fixed [child...] to make a joint rigid, and widths {part: metres} for each part's depth
  across the motion plane (the video can't see it; estimate it from what the object is), and
  thickness {part: metres} when an oblique view makes a part's silhouette too thick.
  Returns joints, a fit residual and per-part
  silhouette IoU, plus an image of the worst replay frames.
- set_physics(parts, released): after the final fit, give each link its material, mass (kg)
  and contact friction, judged from what the object is and its measured size, and the time
  spans (seconds) when a moving part stood still with NO hand touching it. A hinge that holds
  its part against gravity there gets its joint friction measured from those spans; the tool
  checks it by releasing the model in MuJoCo. Masses are your estimate: the tool returns the
  density they imply over the measured solid, so check that it is plausible for the material.
- finish(report): end with a short report for a robotics developer.

How to judge a fit: the tree must make physical sense (each joint where two parts touch).
Replay IoU above ~0.7 is good; thin parts score lower. A joint whose range is a few degrees
probably did not move in this video: say so, and make it fixed only if the object really has
no joint there. If a part is mis-segmented (wrong region, swallowed by a neighbour) or a
part's motion is inconsistent, re-segment and re-track rather than accept a bad fit.
Before finishing: {check_replay}then set_physics, then finish.
Budget: about 22 turns and at most 3 track calls; each turn tells you how many remain.
If the video can't give a good planar model (the part moves toward the camera, a part is
hidden most of the time, the camera faces the motion plane at a steep angle), stop early:
finish with the best model you have, say plainly what is wrong, and say how to reshoot.
Be decisive. Write every message in English."""

VIDEO_TOOLS = """- The whole video is attached to the first message: watch it to see what moves, how, and
  when. Timestamps in it are seconds; frame = seconds x fps.
- watch(what, start_s, end_s): watch a video again: "video" (the source) or "replay" (the
  latest replay: the fitted model drawn over the video). Watching the replay is how you check
  a fit over time, not just at a few frames: say where and when the model leaves the object.
"""


def system_prompt(lidar_capture: bool) -> str:
    """SYSTEM for this model: without video input, no watch tool (times come from frames)."""
    text = (SYSTEM.replace("{video_tools}", VIDEO_TOOLS if VIDEO else "")
            .replace("{check_replay}", "watch the replay once, " if VIDEO else "look at the replay images, "))
    if not VIDEO:
        text = text.replace("time\n  spans (seconds)", "time\n  spans (seconds = frame / fps)")
    return text + (LIDAR if lidar_capture else "")


TOOLS = [
    {"name": "look", "description": "Show video frames (up to 4).",
     "parameters": {"type": "object", "properties": {"frames": {"type": "array", "items": {"type": "integer"}}},
                    "required": ["frames"]}},
    {"name": "watch", "description": "Watch the source video or the latest replay (model over video), whole or a span.",
     "parameters": {"type": "object", "properties": {
         "what": {"type": "string", "enum": ["video", "replay"]},
         "start_s": {"type": "number"}, "end_s": {"type": "number"}}, "required": ["what"]}},
    {"name": "segment", "description": "Run SAM 3.1 with one concept prompt on one frame.",
     "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "frame": {"type": "integer"}},
                    "required": ["prompt", "frame"]}},
    {"name": "measure_scale", "description": "Set metric scale from a segmented ID-1 card candidate.",
     "parameters": {"type": "object", "properties": {"candidate": {"type": "string"}}, "required": ["candidate"]}},
    {"name": "track", "description": "Track parts through the video. First part is the fixed root.",
     "parameters": {"type": "object", "properties": {"parts": {"type": "array", "items": {
         "type": "object", "properties": {
             "name": {"type": "string"},
             "candidate": {"type": "string"},
             "candidates": {"type": "array", "items": {"type": "string"},
                            "description": "Several candidates of the same part on different frames, so "
                                           "tracking re-acquires it after it disappears (one per frame)."}},
         "required": ["name"]}}}, "required": ["parts"]}},
    {"name": "combine_tracks", "description": "Combine tracked regions belonging to one rigid part.",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string", "description": "new combined part name"},
         "parts": {"type": "array", "items": {"type": "string"}, "description": "existing tracked part names"},
         "gate": {"type": "string", "description": "member whose usable visibility is required"}},
         "required": ["name", "parts", "gate"]}},
    {"name": "retrack_part", "description": "Re-track one part from new candidate anchors; keep other tracks.",
     "parameters": {"type": "object", "properties": {
         "part": {"type": "string", "description": "existing original tracked part name"},
         "candidates": {"type": "array", "items": {"type": "string"},
                        "description": "segment candidate ids at distinct frames"}},
         "required": ["part", "candidates"]}},
    {"name": "fit", "description": "Fit joints and replay in MuJoCo.",
     "parameters": {"type": "object", "properties": {
         "parents": {"type": "object", "additionalProperties": {"type": "string"},
                     "description": "Optional {child: parent} tree override."},
         "fixed": {"type": "array", "items": {"type": "string"}, "description": "Children whose joint is rigid."},
         "widths": {"type": "object", "additionalProperties": {"type": "number"},
                    "description": "Optional {part: metres}: each part's depth across the motion plane "
                                   "(e.g. a laptop is ~0.30 wide), which the video can't measure."},
         "thickness": {"type": "object", "additionalProperties": {"type": "number"},
                       "description": "Optional {part: metres}: a part's true thickness when the view is "
                                      "oblique and its silhouette shows faces (a laptop lid is ~0.006)."}}}},
    {"name": "fit_lidar", "description": "RGB-D-format captures, including estimated RGB depth: fit one hinged plate in 3D and replay it.",
     "parameters": {"type": "object", "properties": {
         "floor": {"type": "string", "description": "tracked part: the static surface the object stands on"},
         "base": {"type": "string", "description": "tracked part the plate hinges on, as seen on frame 0"},
         "plate": {"type": "string", "description": "tracked part: the plate's inner face (e.g. a screen)"},
         "plate_closed": {"type": "string", "description": "tracked part covering the object when closed"},
         "open_indicator": {"type": "string", "description": "tracked part seen only while open"},
         "joint": {"type": "string", "enum": ["hinge", "slide"],
                   "description": "hinge (a lid, door, flap; default) or slide (a drawer: plate = the drawer)"}},
         "required": ["floor", "base", "plate"]}},
    {"name": "fit_drawers", "description": "LiDAR captures only: drawers sliding out of a cabinet. Fits each "
     "drawer's travel (how far its front stands out of the cabinet's front, in every frame), writes the model and "
     "replays it.",
     "parameters": {"type": "object", "properties": {
         "floor": {"type": "string", "description": "tracked part: the static surface the cabinet stands on"},
         "base": {"type": "array", "items": {"type": "string"},
                  "description": "tracked parts of the cabinet that never move (its side or top panel, fronts of "
                                 "drawers that stay shut); they are joined into one base"},
         "drawers": {"type": "array", "items": {"type": "object", "properties": {
             "part": {"type": "string", "description": "tracked part: this drawer's front panel"},
             "first_frame": {"type": "integer"}, "last_frame": {"type": "integer",
                                                               "description": "frames in which this drawer is operated"}},
             "required": ["part", "first_frame", "last_frame"]}}},
         "required": ["base", "drawers"]}},
    {"name": "fit_static", "description": "RGB-D only: approximate a stationary rigid object as a box or upright cylinder; do not use for moving objects.",
     "parameters": {"type": "object", "properties": {
         "part": {"type": "string", "description": "tracked whole object"},
         "floor": {"type": "string", "description": "tracked supporting surface"},
         "shape": {"type": "string", "enum": ["box", "cylinder"], "description": "Visible overall shape; cylinder only for upright cylindrical objects."},
         "upright_hint": {"type": "string", "enum": ["drawer_handles"], "description": "For an RGB-video drawer cabinet with multiple visible drawer handles, estimate upright from their arrangement and the supporting floor."}},
         "required": ["part", "floor"]}},
    {"name": "set_physics", "description": "Masses, inertia and friction for the fitted model. Joint friction is "
     "measured where a part stood still on its own.",
     "parameters": {"type": "object", "properties": {
         "parts": {"type": "array", "items": {"type": "object", "properties": {
             "link": {"type": "string", "description": "a link of the fitted model"},
             "material": {"type": "string"},
             "mass_kg": {"type": "number"},
             "friction": {"type": "number", "description": "sliding friction coefficient of its surface"}},
             "required": ["link", "material", "mass_kg", "friction"]}},
         "released": {"type": "array", "items": {"type": "object", "properties": {
             "joint": {"type": "string"}, "start_s": {"type": "number"}, "end_s": {"type": "number"}},
             "required": ["joint", "start_s", "end_s"]},
             "description": "spans where a moving part stands still and no hand touches it"}},
         "required": ["parts", "released"]}},
    {"name": "finish", "description": "End with a report.",
     "parameters": {"type": "object", "properties": {"report": {"type": "string"}}, "required": ["report"]}},
]


LIDAR = """

This is an RGB-D-format capture: every frame includes depth in metres, but depth may be
estimated from ordinary video rather than measured by a depth camera. Follow the source
note in the user message: estimated depth and scale can be inaccurate, including after
global calibration. The view may be oblique; skip the planar measure_scale tool. For a hinged
plate (laptop lid, door, flap) use fit_lidar, not fit; for drawers use fit_drawers (see the
end). If the video shows a rigid stationary roughly box-shaped or upright cylindrical object
with no moving parts, track the complete object and supporting floor, then call fit_static
with its visible shape. It makes only a simple approximation: visible depth cannot establish
hidden dimensions. For an RGB-video drawer cabinet with multiple visible handles, pass
upright_hint="drawer_handles" to test a visual upright direction; inspect the replay because
the handles and floor can be misidentified. Do not use it for moving objects or other silhouettes.
For an articulated object with two or more moving joints seen approximately from the side,
the 3D hinged-plate fitter cannot represent the whole chain. Track each rigid part and use
fit with an explicit parent for each moving part; this is a planar visual approximation
whose cross-view depth and metric scale are assumed, even when an RGB-D capture is present.
For a single hinged object track, in one track call:
- the floor or table the object stands on (static scene),
- the entire base surface the plate hinges on, on an open frame. A keyboard or touchpad
  alone is only a fragment of a laptop base and cannot determine its full dimensions;
  a mask containing the screen as well is too broad. Inspect the annotated candidate and
  try other visual descriptions or frames until the base surface is covered without the lid,
- the plate's inner face (e.g. "laptop screen"). The plate is one rigid piece that turns
  as a whole: for a book, the front cover (not the pages, which bend; not the spine),
- what covers the object when closed (segment it on a closed frame; for a laptop the
  whole closed laptop, which may be the same tracked part as the base; for a book, the
  closed book),
- a part seen only while open (e.g. "laptop keyboard"), so closed frames can be told apart.
Look at frames across the video first to see when it is open and closed. A part that
disappears and comes back (a screen that closes and reopens) needs candidates on several
frames, at least one before and one after it disappears. Then call fit_lidar with the
tracked part names in their roles. Judge the result: the hinge line should sit on the real
hinge, the opening over time should follow the video, and the replayed model should stay
on the object. Report sizes in mm; say which numbers are clipped or unmeasured.
An open-indicator mask alone does not prove the object is open: if its reported reference
opening is nearly zero, inspect that frame and the indicator's tracking before accepting the fit.

Drawers (fit_drawers): look at frames across the video first to see WHICH drawer moves and
WHEN. When a drawer is out, the slot it came from is gone from the row of fronts: count the
fronts on a shut frame and on an open frame. Track, in one track call:
- the floor,
- each moving drawer's FRONT PANEL (segment it while it is out; only add a shut-frame
  candidate if the video clearly shows it is the same front, for example by following its
  handle through the motion). Similar neighbouring fronts are easy to confuse: an uncertain
  shut-frame anchor can force the track onto the wrong drawer, so omit it if unsure,
- cabinet parts that never move (its side or top panel, fronts of drawers that stay shut),
  each its own part; pass them together as base.
Give each drawer the frame range in which it is operated. Look-alike fronts can swap
identities while tracking: in the replay, the drawer's box must sit on that drawer both shut
and open, and a part in base must not follow the moving drawer (if one does, drop it from
base and fit again)."""


def cap_images(messages: list[dict]) -> None:
    """Keep only the newest MAX_IMAGES images and MAX_VIDEOS videos in the conversation; older
    ones become a note."""
    seen = {"image_url": 0, "video_url": 0}
    cap = {"image_url": MAX_IMAGES or 10 ** 6, "video_url": MAX_VIDEOS}
    for m in reversed(messages):
        if not isinstance(m["content"], list):
            continue
        for i in range(len(m["content"]) - 1, -1, -1):
            kind = m["content"][i]["type"]
            if kind in seen:
                seen[kind] += 1
                if seen[kind] > cap[kind]:
                    m["content"][i] = {"type": "text", "text": f"(older {kind.split('_')[0]} dropped)"}


def capture_depth_note(capture: Path | None) -> str:
    if capture is not None and (capture / "mapanything.json").exists():
        return ("Depth and camera poses are estimated by MapAnything from ordinary RGB video, not measured by LiDAR. "
                "Metric scale and local dimensions may be wrong even after global calibration; "
                "label all mm sizes as estimates and do not claim they were measured.")
    if capture is not None:
        return "Depth comes from the RGB-D capture and may contain noise, drift, and missing surfaces."
    return "No depth capture is available; scale must be established separately."


def mp4_url(path: Path, start_s: float | None = None, end_s: float | None = None) -> str:
    """A small H.264 copy of a video (or a span of it) as a data URL, for Spark."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "v.mp4"
        span = (["-ss", f"{start_s:.2f}"] if start_s else []) + (["-to", f"{end_s:.2f}"] if end_s else [])
        subprocess.run(["ffmpeg", "-v", "error", "-y", *span, "-i", str(path), "-vf",
                        f"scale={VIDEO_WIDTH}:-2,fps={VIDEO_FPS}", "-an", "-c:v", "libx264", "-crf", "30",
                        "-pix_fmt", "yuv420p", str(out)], check=True)
        return "data:video/mp4;base64," + base64.b64encode(out.read_bytes()).decode()


def jpeg_url(img: np.ndarray, width: int = VIEW_WIDTH) -> str:
    small = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()


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
    capture: Path | None = None  # iPhone LiDAR capture (Stray Scanner): depth and camera poses
    client: OpenAI = None
    sam: SamClient = None
    candidates: dict[str, tuple[str, Detection, int]] = field(default_factory=dict)  # id -> prompt, det, frame
    px_per_m: float | None = None
    scale_note: str = "no scale set"
    tracked: bool = False
    last_fit: Fit | None = None
    report: str | None = None
    images: list[str] = field(default_factory=list)  # data URLs queued for the next user turn
    videos: list[str] = field(default_factory=list)
    n_segment: int = 0
    log_lines: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.client = self.client or OpenAI(base_url=BASE_URL, api_key=os.environ[API_KEY_ENV], timeout=600)
        self.sam = self.sam or sam_client()
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
        if not frames:
            return "No frames requested; provide at least one frame number."
        requested = len(frames)
        if requested > 4:
            # Preserve the requested time span instead of silently dropping all
            # later frames, which may include the object in its open state.
            frames = [frames[i] for i in np.linspace(0, requested - 1, 4).astype(int)]
        frames = [max(0, min(self.n - 1, int(f))) for f in frames]
        imgs = [label(read_frame(self.video, f), f"frame {f}") for f in frames]
        strip = cv2.hconcat([cv2.resize(i, (VIEW_WIDTH, round(i.shape[0] * VIEW_WIDTH / i.shape[1]))) for i in imgs])
        self.show(strip, f"look {frames}", width=strip.shape[1])
        note = f" Sampled 4 of {requested} requested frames across the list; call look again for other frames." if requested > 4 else ""
        return f"Showing frames {frames} (left to right).{note}"

    def tool_watch(self, what: str = "video", start_s: float | None = None, end_s: float | None = None) -> str:
        if not VIDEO:
            return "video input is off for this model; use look"
        path = self.video
        if what == "replay":
            replays = [p for p in (self.track_dir / "replay3d.mp4", self.track_dir / "replay.mp4") if p.exists()]
            if self.last_fit is None or not replays:
                return "no replay yet: fit first"
            path = max(replays, key=lambda p: p.stat().st_mtime)
        self.videos.append(mp4_url(path, start_s, end_s))
        span = f" {start_s or 0:.1f}-{end_s or self.n / self.fps:.1f} s" if start_s or end_s else ""
        self.log(f"watch {what}{span} ({path.name})")
        return (f"The {what}{span} is attached to the next message"
                + (" (fitted model in colour over the video)." if what == "replay" else "."))

    def tool_set_physics(self, parts: list[dict], released: list[dict]) -> str:
        if self.last_fit is None or not (self.track_dir / "model.xml").exists():
            return "fit a model first"
        parts = json.loads(parts) if isinstance(parts, str) else parts  # models sometimes send JSON strings
        released = json.loads(released) if isinstance(released, str) else released
        released = [released] if isinstance(released, dict) else released
        aliases = {}
        static = self.track_dir / "static.json"
        if static.exists():
            part = slug(json.loads(static.read_text()).get("part", ""))
            if part:
                aliases.setdefault(part, []).append("base")
        slides = self.track_dir / "slides.json"
        if slides.exists():
            for index, drawer in enumerate(json.loads(slides.read_text()).get("drawers", []), 1):
                part = slug(drawer.get("part", ""))
                if part:
                    aliases.setdefault(part, []).append(f"drawer{index}")
        props, mapped = {}, []
        for p in parts:
            name = slug(p["link"])
            if name in aliases and len(aliases[name]) == 1:
                target = aliases[name][0]
                if target != name:
                    mapped.append(f"Mapped tracked part {name} to model link {target}.")
                name = target
            if name in props:
                return f"Multiple physics inputs refer to {name}; provide one set of properties per model link."
            props[name] = {"material": p["material"], "mass_kg": float(p["mass_kg"]),
                           "friction": float(p["friction"])}
        windows: dict[str, list] = {}
        for r in released:
            name = slug(r["joint"])
            name = name if name.endswith("_joint") else f"{name}_joint"
            windows.setdefault(name, []).append((int(r["start_s"] * self.fps), int(r["end_s"] * self.fps)))
        lines = mapped
        report = physics.apply(self.track_dir, props, windows, self.fps, log=lines.append)
        for link, r in report["links"].items():
            rho = r["implied_density_kg_m3"]
            if rho and not 150 <= rho <= 3000:  # beyond wood foam .. aluminium-and-glass
                lines.append(f"WARNING: {link}'s estimated mass over its fitted volume implies {rho} kg/m^3. "
                             "Check the mass and fitted volume; a hollow object can also have a low average density. "
                             "Do not treat the estimate as a measured material density.")
        return "\n".join(lines)

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

    def tool_retrack_part(self, part: str, candidates: list[str]) -> str:
        if not self.tracked:
            return "track parts first"
        if not candidates or any(cid not in self.candidates for cid in candidates):
            return "provide existing segment candidate ids on distinct frames"
        part = slug(part)
        metadata_path = self.track_dir / "parts.json"
        metadata = json.loads(metadata_path.read_text())
        ids = [int(k) for k, v in metadata.items() if slug(v.get("name") or v.get("prompt", "")) == part]
        if len(ids) != 1:
            return f"{part} is not a unique original tracked part"
        selections = [self.candidates[cid] for cid in candidates]
        anchors = sorted((int(frame), det.box) for _, det, frame in selections)
        if len({t for t, _ in anchors}) != len(anchors):
            return "use at most one candidate per frame"
        raw_path = self.track_dir / "tracks.sam.txt"
        current = raw_path.read_text()
        # A retry may return to an earlier anchor set. Reuse that exact track
        # instead of running the remote video tracker again.
        cache = getattr(self, '_retrack_cache', None)
        if cache is None:
            cache = self._retrack_cache = {}
        def key(items):
            return (ids[0], tuple(sorted((int(t), tuple(map(int, box))) for t, box in items)))
        previous = metadata[str(ids[0])].get('anchors', [])
        if previous:
            try:
                previous_track = replace_tracked_part('', current, 0, source_id=ids[0])
            except ValueError:
                previous_track = None  # A missing track must remain recoverable.
            if previous_track is not None:
                cache.setdefault(key(previous), previous_track)
        requested = key(anchors)
        reused = requested in cache
        raw = cache[requested] if reused else self.sam.track_boxes(self.video, self.size, {0: anchors})
        merged = replace_tracked_part(current, raw, ids[0])
        cache[requested] = raw
        prompt, det, frame = selections[0]
        metadata[str(ids[0])].update({"prompt": prompt, "frame": frame, "box": list(det.box),
                                      "confidence": det.confidence,
                                      "anchors": [[t, list(b)] for t, b in anchors]})
        tmp = raw_path.with_suffix(".retrack.tmp")
        tmp.write_text(merged)
        self.last_fit = None
        tmp.replace(raw_path)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        raw_path.with_suffix(".npz").unlink(missing_ok=True)
        count = sum(d.obj_id == 0 for d in parse(raw))
        action = 'Restored cached track for' if reused else 'Re-tracked'
        return f"{action} {part} on {count} frames; other tracks retained. Refit and inspect the replay."

    def tool_combine_tracks(self, name: str, parts: list[str], gate: str) -> str:
        if not self.tracked:
            return "track the component parts first"
        name, parts, gate = slug(name), [slug(p) for p in parts], slug(gate)
        raw_parts = json.loads((self.track_dir / "parts.json").read_text())
        raw_names = {slug(v.get("name") or v["prompt"]) for v in raw_parts.values()}
        if not name or name in raw_names or len(parts) < 2 or len(set(parts)) != len(parts):
            return "use a new name and at least two distinct tracked parts"
        if any(p not in raw_names for p in parts) or gate not in parts:
            return "all parts must be original tracked parts and gate must be one of them"
        path = self.track_dir / "track_unions.json"
        previous = path.read_text() if path.exists() else None
        unions = json.loads(previous) if previous is not None else {}
        if name in unions:
            return f"combined part {name} already exists; choose another name"
        unions[name] = {"parts": parts, "gate": gate, "frame_size": list(self.size)}
        path.write_text(json.dumps(unions, indent=2) + "\n")
        try:
            _, tracks = load_parts(self.track_dir)
        except Exception:
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(previous)
            raise
        count = len(tracks[name])
        if not count:
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(previous)
            return "no usable frames for the gate; inspect the tracks"
        return (f"Combined {', '.join(parts)} as {name} on {count} frames where {gate} is usable. "
                "Use that name as the base in fit_lidar; unseen margins are still unmeasured.")

    def tool_track(self, parts: list[dict]) -> str:
        if not parts:
            return "Choose at least one segmented part to track."
        chosen = []  # (name, prompt, best detection, its frame, all anchors)
        for p in parts:
            ids = p.get("candidates") or p.get("candidate")
            ids = ids if isinstance(ids, list) else [ids]  # Spark sometimes puts a list in "candidate"
            missing = [i for i in ids if i not in self.candidates]
            if missing:
                return f"unknown candidate(s) {missing}; segment first"
            anchors = sorted({self.candidates[i][2]: self.candidates[i][1].box for i in ids}.items())
            if len(anchors) < len(ids):
                return f"{p['name']}: give at most one candidate per frame"
            prompt, det, frame = self.candidates[ids[0]]
            chosen.append((slug(p["name"]), prompt, det, frame, anchors))
        if len({c[0] for c in chosen}) != len(chosen):
            return "part names must be unique"
        raw = self.sam.track_boxes(self.video, self.size, {oid: c[4] for oid, c in enumerate(chosen)})
        counts, tiny = {}, {}
        for det in parse(raw):
            counts[det.obj_id] = counts.get(det.obj_id, 0) + 1
            if not det.usable(*self.size):
                tiny[det.obj_id] = tiny.get(det.obj_id, 0) + 1
        missing = [name for oid, (name, *_) in enumerate(chosen)
                   if counts.get(oid, 0) <= tiny.get(oid, 0)]
        if missing:
            return (f"Tracking returned no masks of usable size for {missing}; previous tracks and fit were retained. "
                    "Inspect the candidates and retry.")
        (self.track_dir / "tracks.sam.npz").unlink(missing_ok=True)
        (self.track_dir / "track_unions.json").unlink(missing_ok=True)
        (self.track_dir / "tracks.sam.txt").write_text(raw)
        (self.track_dir / "parts.json").write_text(json.dumps({
            oid: {"name": name, "prompt": prompt, "frame": frame, "box": list(d.box), "confidence": d.confidence,
                  "anchors": [[f, list(b)] for f, b in anchors]}
            for oid, (name, prompt, d, frame, anchors) in enumerate(chosen)}, indent=2))
        self.tracked = True
        self.last_fit = None

        def line(oid, name):
            n = counts.get(oid, 0)
            if not tiny.get(oid):
                return f"{name} {n}/{self.n} frames"
            return (f"{name} {n - tiny[oid]}/{self.n} frames (plus {tiny[oid]} where the mask shrank to a few "
                    f"pixels: lost there; anchor it on frames where it looks different)")
        return "Tracked: " + ", ".join(line(oid, name) for oid, (name, *_) in enumerate(chosen))

    def tool_fit(self, parents: dict | None = None, fixed: list | None = None, widths: dict | None = None,
                 thickness: dict | None = None) -> str:
        if not self.tracked:
            return "track parts first"
        # A failed replacement fit must not leave an earlier fit marked current.
        self.last_fit = None
        if self.capture is not None and len(parents or {}) < 2:
            return ("This is a depth capture: use fit_lidar for a single hinged plate, fit_drawers for drawers, "
                    "or fit_static for a stationary object. Planar fit is only for a side-view chain with "
                    "at least two explicit moving-part parent links; its depth and scale are assumed.")
        if self.capture is not None and (self.track_dir / 'model3d.json').exists():
            return 'A depth fit already exists in this output; use a fresh run for a planar model to avoid mixing results.'
        if self.px_per_m is None:
            w, h = self.size
            self.px_per_m = max(w, h) / ASSUMED_VIEW_M
            self.scale_note = (f"ASSUMED ({self.px_per_m:.0f} px/m, as if the frame spanned {ASSUMED_VIEW_M} m): "
                               "no card was measured, so sizes are rough; angles are unaffected")
        parents = {slug(k): slug(v) for k, v in (parents or {}).items()} or None
        result = fit(self.video, self.track_dir, self.px_per_m, parents=parents,
                     fixed={slug(f) for f in (fixed or [])},
                     widths={slug(k): float(v) for k, v in (widths or {}).items()},
                     thickness={slug(k): float(v) for k, v in (thickness or {}).items()}, log=lambda s: None)
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

    def tool_fit_lidar(self, floor: str, base: str, plate: str, plate_closed: str | None = None,
                       open_indicator: str | None = None, joint: str = "hinge") -> str:
        if self.capture is None:
            return "fit_lidar needs a LiDAR capture; use fit"
        if not self.tracked:
            return "track parts first"
        self.last_fit = None
        roles = lidar.Roles(slug(floor), slug(base), slug(plate), plate_closed and slug(plate_closed),
                            open_indicator and slug(open_indicator))
        c = lidar.open_capture(self.video, self.capture, self.track_dir)
        missing = [r for r in vars(roles).values() if r and r not in c.names]
        if missing:
            return f"not tracked: {missing}; tracked parts are {c.names}"
        lines = []
        if joint == "slide":  # a drawer: the plate role is the part that slides out
            ref = c.ref_with(roles.base, roles.plate)
            if ref != c.ref:
                c.set_ref(ref)
            if c.moving_camera:
                c.level_with(roles.floor)
            (self.track_dir / "roles.json").write_text(json.dumps(vars(roles)))
            summary = lidar.fit_and_build_slide(c, roles, log=lines.append)
            (self.track_dir / "static.json").unlink(missing_ok=True)
            frames = sorted({int(f) for f in np.linspace(0, self.n - 1, 4)})
            sheet = self.video_sheet(self.track_dir / "replay3d.mp4", frames)
            self.show(sheet, "fit_lidar (slide): 3D replay from the camera", width=sheet.shape[1])
            self.last_fit = summary
            return "\n".join(lines) + "\nImage: the replayed model (colored) over the video."
        slid = c.drop_where_same(roles.plate, roles.plate_closed)
        if slid:
            lines.append(f"{slid} frames where the plate's mask was plate_closed's (the track slid onto the shut lid's "
                         f"back) are measured from plate_closed instead")
            if slid > 0.3 * c.n:
                lines.append(f"WARNING: the plate tracker switched to the closed surface on {slid}/{c.n} frames; "
                             "those frames use a fallback mask only when appropriate, so inspect the fitted shape and joint motion.")
        overlapping = c.drop_open_base_overlap(roles)
        if overlapping:
            lines.append(f"WARNING: removed {overlapping} open frames where the plate mask substantially "
                         "overlapped the base; these frames have no reliable plate observation.")
        raw_world, raw_ref = c.world.copy(), c.ref
        ref = c.ref_with(roles.base, roles.plate, roles.plate_closed, roles.open_indicator)
        if ref != c.ref:
            c.set_ref(ref)
            lines.append(f"reference frame {c.ref}: where the plate is seen largest")
        if c.moving_camera:
            fix = c.level_with(roles.floor)
            lines.append(f"floor levelling adjusted the estimated camera pose by up to {fix['shift_mm_max']:.0f} mm")
        else:  # a fixed camera: the object itself may be pushed around, so measure from its base
            moved = c.follow(roles.base, exclude=(roles.plate,))
            if moved["frames"]:
                lines.append(f"base followed on {moved['frames']} frames: moved up to {moved['moved_mm_max']:.0f} mm, "
                             f"turned {moved['turned_deg_max']:.1f} deg")
        def prepare_reference(capture, frame):
            # Restart camera transforms from the unlevelled capture; rebasing an
            # already levelled/followed frame can change the measured opening.
            capture.world = raw_world.copy()
            capture.ref = raw_ref
            capture.set_ref(frame)
            if capture.moving_camera:
                capture.level_with(roles.floor)
            else:
                capture.follow(roles.base, exclude=(roles.plate,))

        try:
            _, references = lidar.fit_hinge_with_reference_retries(
                c, roles, log=lines.append, prepare_reference=prepare_reference)
            if len(references) > 1:
                lines.append(f"Reference attempts: {references}; using frame {c.ref}.")
        except ValueError as error:
            if "angle on reference frame" not in str(error):
                raise
            frames = suggested_open_anchors(c, roles)
            if frames:
                return (f"{error}\nSuggested open-indicator frames to inspect and segment: {frames}. "
                        f"Use distinct good {roles.plate} candidates with retrack_part, then fit again. "
                        "Check each candidate; an indicator or candidate can itself be wrong.")
            raise
        (self.track_dir / "roles.json").write_text(json.dumps(vars(roles)))
        summary = lidar.build_model(c, roles, log=lines.append)
        (self.track_dir / "static.json").unlink(missing_ok=True)
        metadata_path = self.track_dir / "model3d.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["reference_attempts"] = references
        metadata_path.write_text(json.dumps(metadata, indent=2))
        lines += self.mask_checks(c, roles, summary)
        from .track_checks import overclosed_hinge_warning
        warning = overclosed_hinge_warning(json.loads((self.track_dir / 'hinge.json').read_text()))
        if warning:
            lines.append(warning)
        hint = alternate_plate_hint(c, roles)
        if hint:
            lines.append(hint)
        self.record_fit_warnings(lines)
        hinge_img = cv2.imread(str(self.track_dir / "hinge.jpg"))
        self.show(hinge_img, "fit_lidar: measured hinge and opening over time", width=VIEW_WIDTH * 2)
        frames = sorted(set(summary["worst_frames"].values()) | {int(f) for f in np.linspace(0, self.n - 1, 3)})[:4]
        sheet = self.video_sheet(self.track_dir / "replay3d.mp4", frames)
        self.show(sheet, "fit_lidar: 3D replay from the phone's camera", width=sheet.shape[1])
        self.last_fit = summary
        return ("\n".join(lines) + "\nImages: the hinge drawn on the reference frame with the opening over time, "
                "then the replayed model (colored) over the video at its worst frames and a few others.")

    def mask_checks(self, c, roles, summary: dict) -> list[str]:
        """Plain warnings about the part choice that the numbers show: a base candidate that is
        the whole object (lid included), and parts whose mask the replayed model barely covers."""
        out = []
        both_set = set(c.by[roles.base]) & set(c.by[roles.plate])
        if roles.open_indicator:
            both_set &= set(c.by.get(roles.open_indicator, {}))
        both = sorted(both_set)
        if both:
            ratios = []
            for t in both[:: max(1, len(both) // 40)]:
                plate = c.by[roles.plate][t].full_mask(c.h, c.w)
                base_mask = c.by[roles.base][t].full_mask(c.h, c.w)
                if plate.any():
                    ratios.append((plate & base_mask).sum() / plate.sum())
            inside = float(np.median(ratios)) if ratios else 0.0
            if inside > 0.5:
                out.append(f"WARNING: on frames showing both, {100 * inside:.0f}% of the plate's mask lies inside the "
                           f"base's mask: the base candidate is the whole object, plate included. Segment the base's own "
                           f"surface instead (its top face or keyboard, on an open frame), track again and refit.")
        out += base_fragment_warning(c, roles) + named_base_warning(c, roles) + plate_covers_base_warning(c, roles)
        if roles.plate_closed and roles.plate_closed != roles.plate:
            shared = sorted(set(c.by[roles.plate]) & set(c.by[roles.plate_closed]))
            if roles.open_indicator:
                shared = sorted(set(shared) & set(c.by.get(roles.open_indicator, {})))
            same = []
            for t in shared[:: max(1, len(shared) // 60)]:
                a, b = c.by[roles.plate][t].full_mask(c.h, c.w), c.by[roles.plate_closed][t].full_mask(c.h, c.w)
                same.append((a & b).sum() / max(1, (a | b).sum()) > 0.8)
            if same and np.mean(same) > 0.3 and len(shared) > 0.3 * c.n:
                out.append(f"WARNING: the plate's mask is the same as plate_closed's on {100 * np.mean(same):.0f}% of the "
                           f"checked frames: the plate track may have drifted. Inspect the masks and re-anchor the "
                           f"moving face on open frames.")
        for part, cover in summary.get("mask_coverage", {}).items():
            if cover < 0.5:
                out.append(f"WARNING: the replayed {part} covers only {100 * cover:.0f}% of its SAM mask: the mask is "
                           f"much bigger than the fitted {part} (wrong candidate) or the fit is off. Check that mask.")
        return out

    def tool_fit_static(self, part: str, floor: str, shape: str = "box", upright_hint: str | None = None) -> str:
        if self.capture is None or not self.tracked:
            return "fit_static needs an RGB-D capture and tracked object and floor"
        self.last_fit = None
        part, floor = slug(part), slug(floor)
        c = lidar.open_capture(self.video, self.capture, self.track_dir)
        if part == floor or part not in c.names or floor not in c.names:
            return f"Choose distinct tracked object and floor parts; tracked parts are {c.names}."
        from .static_shape import build_static_model, drawer_handle_up
        lines = []
        up = None
        if upright_hint:
            if upright_hint != 'drawer_handles' or shape != 'box':
                return 'The drawer_handles upright hint is only for stationary box-shaped drawer cabinets.'
            if not (self.capture / 'mapanything.json').exists():
                lines.append('Upright hint skipped: capture already has a sensor gravity direction.')
            else:
                frames = [t for t in c.by.get(part, {}) if 0 <= t < c.n and c.has(floor, t)]
                if frames:
                    frame = max(frames, key=lambda t: c.by[part][t].area)
                    try:
                        candidates = parse(self.sam.segment_image(read_frame(self.video, frame), 'drawer handle'))
                        up = drawer_handle_up(c, part, floor, frame, candidates)
                    except Exception as error:
                        lines.append(f'WARNING: handle-based upright estimation failed: {type(error).__name__}.')
                if up is None:
                    lines.append('WARNING: handle-based upright is ambiguous or unavailable; using the original estimated up direction.')
                else:
                    lines.append(f'Upright direction hypothesized from aligned drawer handles at frame {frame}; inspect the replay.')
        summary = build_static_model(c, part, floor, shape=shape, log=lines.append, up_override=up)
        self.last_fit = summary
        self.record_fit_warnings(lines + summary.get("fit_warnings", []))
        frames = sorted({int(f) for f in np.linspace(0, self.n - 1, 4)})
        sheet = self.video_sheet(self.track_dir / "replay3d.mp4", frames)
        self.show(sheet, f"fit_static: {shape} model replay", width=sheet.shape[1])
        return "\n".join(lines + summary.get("fit_warnings", [])) + "\nInspect whether this envelope follows the object."

    def tool_fit_drawers(self, base: list, drawers: list, floor: str | None = None) -> str:
        if self.capture is None:
            return "fit_drawers needs a LiDAR capture"
        if not self.tracked:
            return "track parts first"
        self.last_fit = None
        base = [slug(b) for b in (base if isinstance(base, list) else [base])]
        spec = [(slug(d["part"]), (int(d["first_frame"]), int(d["last_frame"]))) for d in drawers]
        c = lidar.open_capture(self.video, self.capture, self.track_dir)
        missing = [p for p in base + [d for d, _ in spec] + ([slug(floor)] if floor else []) if p not in c.names]
        if missing:
            return f"not tracked: {missing}; tracked parts are {c.names}"
        from .track_checks import conflicting_drawer_tracks, drawer_prompt_warnings
        conflicts = conflicting_drawer_tracks(c, base, spec)
        if conflicts:
            return "Potentially conflicting drawer tracks; inspect them before fitting:\n" + "\n".join(conflicts)
        lines = drawer_prompt_warnings(self.track_dir, base, [part for part, _ in spec])
        if c.moving_camera and floor:
            fix = c.level_with(slug(floor))
            lines.append(f"floor levelling adjusted the estimated camera pose by up to {fix['shift_mm_max']:.0f} mm")
        b = c.merge("cabinet", base) if len(base) > 1 else base[0]
        (self.track_dir / "drawers.json").write_text(json.dumps({"floor": floor and slug(floor), "base": base,
                                                                  "drawers": [[d, list(w)] for d, w in spec]}))
        summary = lidar.fit_and_build_slides(c, b, slug(floor) if floor else None, spec, log=lines.append)
        (self.track_dir / "static.json").unlink(missing_ok=True)
        # Look-alike fronts swap identities while tracking, and a drawer that is out hides the
        # front next to it: check that nothing in base stands out of the cabinet like a drawer.
        axis = np.array(json.loads((self.track_dir / "slides.json").read_text())["drawers"][0]["axis"])
        for p in base:
            rest = [o for o in base if o != p]
            if not rest:
                continue
            ref_base = c.merge(f"_without_{p}", rest) if len(rest) > 1 else rest[0]
            try:
                m = lidar.front_travel(c, ref_base, p, (0, c.n - 1), axis)
            except ValueError as error:
                lines.append(f"WARNING: could not check whether base part {p!r} moves: {error}")
                continue
            out = float(np.nanpercentile(m["q"], 95))
            if out > 0.05:
                lines.append(f"WARNING: base part {p!r} stands {1000 * out:.0f} mm out of the rest of the cabinet on some "
                             f"frames: it moves like a drawer, so it is probably the drawer that moves (or its mask "
                             f"swapped onto it). Check the masks and fit again.")
        for link, coverage in summary.get("mask_coverage", {}).items():
            if coverage is not None and np.isfinite(coverage) and coverage < .7:
                lines.append(f"WARNING: {link} replay covers only {100 * coverage:.0f}% of its tracked mask. "
                             "The track may have drifted or the fitted shape may be inaccurate; inspect masks "
                             "across the motion and re-segment/retrack before trusting the dimensions.")
        for j in summary["joints"]:
            if j["travel_mm"] < 30:
                lines.append(f"WARNING: {j['part']!r} barely moves ({j['travel_mm']:.0f} mm): it may be a front that "
                             f"stays shut, not the drawer that moves.")
        self.record_fit_warnings(lines)
        frames = []
        fitted = json.loads((self.track_dir / "slides.json").read_text())["drawers"]
        for (d, (t0, t1)), f in zip(spec, fitted):  # each drawer shut and at its most open
            q = np.array([np.nan if v is None else v for v in f["q"]])
            frames += [int(np.nanargmin(q)), int(np.nanargmax(q))]
        sheet = self.video_sheet(self.track_dir / "replay3d.mp4", sorted(set(frames))[:4])
        self.show(sheet, "fit_drawers: replay (drawers shut and at their most open)", width=sheet.shape[1])
        self.last_fit = summary
        return "\n".join(lines) + "\nImage: the replayed model (colored) over the video, each drawer shut and out."

    def record_fit_warnings(self, lines: list[str]):
        path = self.track_dir / "model3d.json"
        info = json.loads(path.read_text())
        info["fit_warnings"] = [line for line in lines if line.startswith("WARNING:")]
        path.write_text(json.dumps(info, indent=2))

    def video_sheet(self, path: Path, frames: list[int]) -> np.ndarray:
        cap = cv2.VideoCapture(str(path))
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
        intro = (f"Video {self.video.name}: {self.n} frames at {self.fps:.2f} fps ({self.n / self.fps:.1f} s), "
                 f"{w}x{h}. Build its kinematic model. "
                 + capture_depth_note(self.capture) + " "
                 + ("The whole video is attached, then frames spread over it." if VIDEO else
                    "Frames below are spread over the video."))
        self.log(f"# Video2URDF agent: {self.video.name}\n\nmodel {MODEL}")
        if VIDEO:
            self.tool_watch("video")
        self.tool_look([0, self.n // 3, 2 * self.n // 3, self.n - 1])
        messages = [{"role": "system", "content": system_prompt(self.capture is not None)},
                    {"role": "user", "content": [{"type": "text", "text": intro}] + self._take_images()}]
        tools = [{"type": "function", "function": t} for t in TOOLS if VIDEO or t["name"] != "watch"]
        for step in range(MAX_STEPS):
            t0 = time.time()
            cap_images(messages)
            r = self._chat(messages=messages, tools=tools)
            msg = r.choices[0].message
            calls = msg.tool_calls or []
            self.log(f"## {WHO} ({time.time() - t0:.0f}s)\n\n{(msg.content or '').strip()}")
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
                    self.log("```\n" + traceback.format_exc() + "```")
                    where = traceback.extract_tb(e.__traceback__)[-1]
                    result = f"error: {type(e).__name__}: {e} (in {where.name}, {Path(where.filename).name}:{where.lineno})"
                self.log(f"> {result}".replace("\n", "\n> "))
                messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
            if self.report is not None:
                break
            left = MAX_STEPS - step - 1
            note = f"{left} turns left." + (" Call finish now with your report." if left <= 1 else "")
            if not self.tracked and left <= MAX_STEPS // 2:
                note += (" Nothing is tracked yet: call track now with the candidates you have (you can "
                         "segment and track again later if the result is wrong).")
            messages.append({"role": "user", "content": [{"type": "text", "text": note}] + self._take_images()})
        if self.report is None:  # out of turns; the API can't force a finish call, so ask for text
            messages.append({"role": "user", "content": "Out of turns. Write your final report now as plain text."})
            cap_images(messages)
            r = self._chat(messages=messages)
            self.report = (r.choices[0].message.content or "").strip() or "(no report)"
            self.log(f"## {WHO} (final report, out of turns)\n\n{self.report}")
        report = self.report
        (self.out / "report.md").write_text(report + "\n")
        return report

    def _chat(self, **kw):
        """One completion, waiting out a scaled-to-zero endpoint (503 while a GPU starts) and
        dropped connections."""
        from openai import APIConnectionError, APIStatusError
        for attempt in range(8):
            try:
                return self.client.chat.completions.create(model=MODEL, max_tokens=6000, **kw)
            except (APIConnectionError, APIStatusError) as e:
                status = getattr(e, "status_code", None)
                if status is not None and status < 500 and status != 429 or attempt == 7:
                    raise
                wait = min(120, 15 * 2 ** attempt)
                self.log(f"(model call failed: {type(e).__name__} {status or ''}; retrying in {wait}s)")
                time.sleep(wait)

    def _take_images(self) -> list[dict]:
        out = ([{"type": "video_url", "video_url": {"url": u}} for u in self.videos]
               + [{"type": "image_url", "image_url": {"url": u}} for u in self.images])
        self.images, self.videos = [], []
        return out
