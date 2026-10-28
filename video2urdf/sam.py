"""SAM 3.1 through the Meta Model API: part detection and video tracking.

Wire format (see dev.meta.ai/docs/sam/reading-segmentation): one line per frame,
`<{frame}f>` then records `{obj_id}<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|><|mask;x=0;y=0;data=H,W,{payload}|>`.
The mask raster is H x W, anchored at the box's (x1, y1). The raw text is the cache format.
"""

import base64
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from meta_sam_parser import FramePrompt, PromptBox, PromptObject, build_object_prompt
from meta_sam_parser import _mask_codec  # private decoder: skip the public API's re-encode check
from openai import OpenAI

from .fast_mask import decode_one_bit

BASE_URL = "https://api.meta.ai/v1"
MODEL = "sam-3.1"

_FRAME = re.compile(r"<(\d+)f>")
_RECORD = re.compile(r"(\d+)<\|box;([^|]*)\|><\|mask;([^|]*)\|>")


@dataclass(frozen=True)
class Detection:
    frame: int
    obj_id: int
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in source pixels
    crop: np.ndarray  # bool, anchored at (x1, y1)
    confidence: float | None = None

    def full_mask(self, height: int, width: int) -> np.ndarray:
        m = np.zeros((height, width), bool)
        x1, y1 = self.box[:2]
        h, w = self.crop.shape
        m[y1:y1 + h, x1:x1 + w] = self.crop[: height - y1, : width - x1]
        return m


def _fields(s: str) -> dict[str, str]:
    return dict(kv.split("=", 1) for kv in s.split(";"))


def _decode_mask(token: str) -> np.ndarray:
    """Decode a mask token. one_bit (box-prompt tracks) uses the compiled decoder; lossless
    (concept prompts) still goes through the slow pure-Python reference."""
    h, w, payload = token.split("data=", 1)[1].split(",", 2)
    h, w = int(h), int(w)
    if payload[0] == "!":
        return decode_one_bit(payload, w, h)
    elif payload[0] == "~":
        raw = np.frombuffer(bytes(_mask_codec._decode_lossless_raster(payload, w, h)), np.uint8) >= 129
    else:
        raise ValueError(f"unknown mask encoding {payload[0]!r}")
    return raw.reshape(h, w)


def _records(text: str):
    for line in text.splitlines():
        m = _FRAME.match(line)
        if not m:
            continue
        frame = int(m.group(1))
        for rec in _RECORD.finditer(line, m.end()):
            box = _fields(rec.group(2))
            c = box.get("c")
            yield (frame, int(rec.group(1)), tuple(int(box[k]) for k in ("x1", "y1", "x2", "y2")),
                   float(c) if c is not None else None, rec.group(3))


def parse(text: str, workers: int = 1) -> list[Detection]:
    """Parse raw SAM output text into detections, in source order."""
    recs = list(_records(text))
    tokens = [r[4] for r in recs]
    if workers > 1 and len(tokens) > 8:
        with ProcessPoolExecutor(workers) as pool:
            crops = list(pool.map(_decode_mask, tokens, chunksize=4))
    else:
        crops = [_decode_mask(t) for t in tokens]
    return [Detection(frame=f, obj_id=o, box=b, crop=crop, confidence=c)
            for (f, o, b, c, _), crop in zip(recs, crops)]


def load_tracks(path: Path) -> dict[int, dict[int, Detection]]:
    """tracks(parse(...)) for a raw output file, cached next to it as .npz."""
    cache = path.with_suffix(".npz")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        with np.load(cache) as f:
            z = {k: f[k] for k in f.files}
        dets, offsets = [], np.concatenate([[0], np.cumsum(z["nbytes"])])
        for i in range(len(z["frame"])):
            h, w = z["shape"][i]
            bits = np.unpackbits(z["bits"][offsets[i]:offsets[i + 1]], count=h * w).reshape(h, w).astype(bool)
            c = z["conf"][i]
            dets.append(Detection(int(z["frame"][i]), int(z["obj"][i]), tuple(int(v) for v in z["box"][i]), bits,
                                  None if np.isnan(c) else float(c)))
        return tracks(dets)
    dets = parse(path.read_text(), workers=os.cpu_count() or 1)
    packed = [np.packbits(d.crop) for d in dets]
    np.savez_compressed(
        cache,
        frame=np.array([d.frame for d in dets]), obj=np.array([d.obj_id for d in dets]),
        box=np.array([d.box for d in dets]).reshape(-1, 4),
        conf=np.array([np.nan if d.confidence is None else d.confidence for d in dets]),
        shape=np.array([d.crop.shape for d in dets]).reshape(-1, 2),
        nbytes=np.array([len(b) for b in packed]),
        bits=np.concatenate(packed) if packed else np.zeros(0, np.uint8),
    )
    return tracks(dets)


def tracks(dets: list[Detection]) -> dict[int, dict[int, Detection]]:
    """Group detections as {obj_id: {frame: Detection}}."""
    out: dict[int, dict[int, Detection]] = {}
    for d in dets:
        out.setdefault(d.obj_id, {})[d.frame] = d
    return out


class SamClient:
    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(base_url=BASE_URL, api_key=api_key or os.environ["MODEL_API_KEY"], timeout=600)

    def upload(self, video: Path) -> str:
        with open(video, "rb") as f:
            return self.client.files.create(file=f, purpose="user_data").id

    def _stream(self, content: list[dict], metadata: dict[str, str]) -> str:
        chunks = []
        with self.client.responses.stream(
            model=MODEL,
            input=[{"type": "message", "role": "user", "content": content}],
            metadata={"include_confidence": "true", **metadata},
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    chunks.append(event.delta)
                elif event.type in ("error", "response.failed"):
                    # Without this a failed backward pass looks like a short, successful track.
                    raise RuntimeError(f"SAM {event.type}: {getattr(event, 'message', None) or event}")
        return "".join(chunks)

    def segment_image(self, image_bgr: np.ndarray, prompt: str) -> str:
        """One concept prompt on one image (frame 0). Returns raw output text."""
        png = base64.b64encode(cv2.imencode(".png", image_bgr)[1]).decode()
        return self._stream([
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:image/png;base64,{png}"},
        ], {})

    def segment_video(self, file_id: str, prompt: str, mask_encoding: str = "lossless") -> str:
        """Run one concept prompt over an uploaded video and return the raw output text.

        One concept per request: "a, b" or "a and b" don't work. Tracking only starts at
        the first frame where the concept is found and runs forward, so parts that are
        unclear early get missed. Prefer track_boxes.
        """
        return self._stream([
            {"type": "input_text", "text": prompt},
            {"type": "input_video", "file_id": file_id},
        ], {"mask_encoding": mask_encoding})

    def _track_forward(self, video: Path, size: tuple[int, int],
                       boxes: dict[int, tuple[int, tuple[int, int, int, int]]]) -> str:
        prompt = build_object_prompt(size=size, objects=[
            PromptObject(id=oid, prompts=[FramePrompt(frame=frame, box=PromptBox(*box))])
            for oid, (frame, box) in boxes.items()
        ])
        return self._stream([
            {"type": "input_text", "text": prompt},
            {"type": "input_video", "file_id": self.upload(video)},
        ], {"propagation_direction": "forward"})

    def track_boxes(self, video: Path, size: tuple[int, int],
                    boxes: dict[int, tuple[int, tuple[int, int, int, int]]]) -> str:
        """Track up to 16 objects through the whole video from {obj_id: (frame, (x1, y1, x2, y2))}.

        ``size`` is the video's displayed (width, height). Masks come back one_bit. The API's
        backward propagation fails with a server error on some phone videos, so frames before
        a prompt are tracked forward through a reversed copy of the start of the video.
        """
        text = self._track_forward(video, size, boxes)
        last = max(frame for frame, _ in boxes.values())
        if last == 0:
            return text
        with tempfile.TemporaryDirectory() as tmp:
            clip = reversed_clip(video, last, Path(tmp) / "reversed.mp4")
            back = self._track_forward(clip, size, {oid: (last - f, box) for oid, (f, box) in boxes.items()})
        # Forward output can start before an object's own prompt frame (SAM tracks every
        # object from the earliest prompt in the request), so keep one record per
        # (frame, object), preferring the forward pass.
        records: dict[int, dict[int, str]] = {}
        for line in text.splitlines():
            m = _FRAME.match(line)
            if m:
                for r in _RECORD.finditer(line, m.end()):
                    records.setdefault(int(m.group(1)), {})[int(r.group(1))] = r.group(0)
        for line in back.splitlines():
            m = _FRAME.match(line)
            if not m:
                continue
            frame = last - int(m.group(1))
            for r in _RECORD.finditer(line, m.end()):
                records.setdefault(frame, {}).setdefault(int(r.group(1)), r.group(0))
        return "\n".join(f"<{f}f>" + ",".join(records[f][o] for o in sorted(records[f]))
                         for f in sorted(records)) + "\n"


def reversed_clip(video: Path, last: int, out: Path) -> Path:
    """Frames 0..last of ``video`` in reverse order, rotation applied, one output frame per input frame."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video),
                    "-vf", f"trim=end_frame={last + 1},reverse", "-fps_mode", "passthrough", "-an",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out)], check=True)
    cap, n = cv2.VideoCapture(str(out)), 0
    while cap.grab():
        n += 1
    cap.release()
    if n != last + 1:
        raise RuntimeError(f"reversed clip has {n} frames, expected {last + 1}")
    return out
