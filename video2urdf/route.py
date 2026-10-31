"""Which path a capture takes, and why.

- A depth capture (iPhone LiDAR, RGB-D, or one already reconstructed) takes the 3D path.
- A plain video takes the planar path when the camera holds still and faces the plane the part
  moves in (a laptop filmed from its side, a desk lamp from the side): the joint is measured
  from the parts' 2D motion. Otherwise it takes the 3D path: MapAnything on Modal estimates
  depth and camera motion, and the LiDAR fits run on that.

The camera's motion is measured locally (capture_check.camera_motion). Whether the view faces
the motion plane is a judgement, so one cheap Muse Spark call looks at a few frames
(``spark_look``). It also says which image edge is the world's up, so a sideways video can be
turned upright before anything else sees it, and names the object (the fallback's prompt).
A flag always overrides the choice.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

LOOK_FRAMES = 6
LOOK_WIDTH = 448
MAX_TOKENS = 16000  # Spark reasons before it answers; the answer itself is ~150 tokens
UPS = ("top", "right", "bottom", "left")
TURN_FOR_UP = {"top": 0, "right": 270, "bottom": 180, "left": 90}  # clockwise turn that brings up to the top

LOOK_PROMPT = """These {k} frames are spread over a {seconds:.1f} s video (frame numbers on each) that a
user filmed so that a robot simulator can get a model of an object with moving parts (URDF).
Look at them and answer with ONE JSON object, nothing else:
{{
 "object": "the main object, in 1-4 plain words (e.g. laptop, desk lamp, microwave, drawer cabinet)",
 "moving_parts": ["parts that move relative to the object's body during the video, e.g. lid, door, drawer, arm"],
 "anything_moves": true or false (does any part move relative to the rest of the object?),
 "up": "top", "left", "right" or "bottom": the image edge nearest the world's up (the ceiling or sky, away
        from the floor or table). A normally held video is "top"; a sideways video has the floor on a side,
 "up_confidence": "high", "medium" or "low",
 "camera": "fixed" (tripod or held still), "shaky" (held by hand, about the same view) or "moving" (walks or
           circles around the object, the view direction changes),
 "view": "side" if the moving part swings or slides across the picture, i.e. its hinge axis or slide is
         roughly toward the camera and the motion happens in a plane facing the camera (a laptop seen from
         its side, its lid sweeping across the frame); "oblique" if the motion is seen at a slant; "top" if
         the camera looks down on it; "toward" if the part mostly moves toward or away from the camera
         (a drawer pulled straight at the camera),
 "whole_object_in_frame": true if the whole object stays inside the frame in every frame,
 "lighting": "good", "dark", "backlit" or "overexposed",
 "card_visible": true if a card-sized flat card (like a loyalty card) is visible next to the object,
 "problems": ["short reshoot tips, only for real problems you see in these frames"]
}}"""


def spark_look(video: Path, frames: int, fps: float, k: int = LOOK_FRAMES, log=print) -> dict | None:
    """One Muse Spark call on ``k`` frames (the agent's backend). None when no model can answer."""
    import os

    import cv2
    import numpy as np
    from openai import OpenAI

    from .agent import API_KEY_ENV, BASE_URL, MODEL, jpeg_url, label
    from .capture_check import sample

    if not os.environ.get(API_KEY_ENV):
        log(f"  no {API_KEY_ENV}: skipping Spark's look")
        return None
    idx, imgs = sample(video, frames, k=k, width=960)
    content = [{"type": "text", "text": LOOK_PROMPT.format(k=len(idx), seconds=frames / fps if fps else 0)}]
    for t, img in zip(idx, imgs):
        content.append({"type": "image_url", "image_url": {"url": jpeg_url(label(img, f"frame {t}"), LOOK_WIDTH)}})
    client = OpenAI(base_url=BASE_URL, api_key=os.environ[API_KEY_ENV], timeout=300, max_retries=2)
    text = ""
    for attempt in range(2):  # Spark reasons first (~3.5k tokens on 6 frames): an empty answer ran out
        try:
            r = client.chat.completions.create(model=MODEL, max_tokens=MAX_TOKENS,
                                               response_format={"type": "json_object"},
                                               messages=[{"role": "user", "content": content}])
            text = r.choices[0].message.content or ""
        except Exception as error:  # the look is a convenience: the pipeline goes on without it
            log(f"  Spark's look failed ({type(error).__name__}: {str(error)[:200]})")
            return None
        if text.strip():
            break
        log(f"  Spark's look came back empty ({r.choices[0].finish_reason}); asking again")
    look = parse_look(text)
    if look is None:
        log(f"  Spark's look gave no usable JSON: {text[:200]!r}")
        return None
    look["model"] = MODEL
    look["frames"] = idx
    return look


def parse_look(text: str) -> dict | None:
    """The JSON object in a model's answer, with fields checked and normalised."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None

    def pick(key, allowed, default=None):
        v = str(raw.get(key, "")).strip().lower()
        return v if v in allowed else default

    def flag(key):
        v = raw.get(key)
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            return v.strip().lower() == "true"
        return None

    parts = raw.get("moving_parts") or []
    problems = raw.get("problems") or []
    return {"object": str(raw.get("object") or "").strip()[:60] or None,
            "moving_parts": [str(p)[:40] for p in parts if p] if isinstance(parts, list) else [],
            "anything_moves": flag("anything_moves"),
            "up": pick("up", UPS, "top"), "up_confidence": pick("up_confidence", ("high", "medium", "low"), "low"),
            "camera": pick("camera", ("fixed", "shaky", "moving")),
            "view": pick("view", ("side", "oblique", "top", "toward")),
            "whole_object_in_frame": flag("whole_object_in_frame"),
            "lighting": pick("lighting", ("good", "dark", "backlit", "overexposed")),
            "card_visible": flag("card_visible"),
            "problems": [str(p)[:200] for p in problems if p] if isinstance(problems, list) else []}


def upright_turn(look: dict | None) -> int:
    """Clockwise degrees that turn the video upright, when Spark is sure it isn't."""
    if not look or look.get("up_confidence") != "high":
        return 0
    return TURN_FOR_UP.get(look.get("up") or "top", 0)


def choose_path(kind: str, camera: dict | None, look: dict | None, forced: str | None = None) -> tuple[str, list[str]]:
    """'3d' or 'planar', with the reasons in plain words."""
    if forced in ("3d", "planar"):
        return forced, [f"--path {forced} given"]
    if kind in ("stray", "mapanything", "rgbd"):
        return "3d", [f"the input has depth ({kind})"]
    cam = (camera or {}).get("camera", "unknown")
    seen = (camera or {}).get("why", "")
    view = (look or {}).get("view")
    look_cam = (look or {}).get("camera")
    if look and look.get("anything_moves") is False:
        return "3d", ["Spark sees no moving part: a rigid object is measured best in 3D (MapAnything depth)"]
    if cam == "moving":
        return "3d", [f"the camera moves ({seen}); a planar model needs a still camera, so MapAnything "
                      "estimates depth and the camera's motion"]
    if view in ("oblique", "top", "toward"):
        return "3d", [f"Spark sees the motion {'at a slant' if view == 'oblique' else 'from above' if view == 'top' else 'toward the camera'}"
                      f" ({view} view): it doesn't happen in a plane facing the camera, so the 3D path"]
    if cam in ("unsteady", "unknown") and look_cam == "moving":
        return "3d", [f"camera: {cam} by the numbers ({seen}) and Spark sees it moving around the object"]
    if view == "side":
        return "planar", [f"camera {cam} ({seen}) and Spark sees a side view: the parts move in a plane facing "
                          "the camera, so their 2D motion measures the joints"]
    if cam in ("fixed", "steady", "unsteady"):
        return "planar", [f"camera {cam} ({seen}); no view judgement from Spark, so the filming notes' default "
                          "for a still camera: a side view (use --path 3d if the motion isn't in a plane facing "
                          "the camera)"]
    return "3d", [f"camera motion unknown ({seen}) and no view judgement: the general 3D path"]
