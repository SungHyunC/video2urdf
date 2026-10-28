"""Track parts in a video with SAM 3.1 and write an overlay for checking.

Each prompt names one part. For each part we look for it with a concept prompt on a few
key frames (cheap image calls), then track every found part through the whole video in
one box-prompt request (plus a reversed pass for frames before the key frame).

Usage:
  .venv/bin/python scripts/track_video.py VIDEO "red rod" "blue rod" [--out data/tracks]

Writes <out>/<video stem>/: tracks.sam.txt (raw API output, the cache), parts.json
(object id -> prompt, key frame, box) and overlay.mp4.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from video2urdf.pipeline import video_info  # noqa: E402
from video2urdf.sam import SamClient, load_tracks, parse, sam_client  # noqa: E402

COLORS = [(0, 200, 255), (255, 80, 80), (80, 255, 80), (255, 0, 255), (255, 255, 0), (0, 128, 255)]
KEY_FRAMES = tuple(i / 8 for i in range(8))  # fractions of the video searched for each part
MIN_ANCHOR_CONFIDENCE = 0.5


def read_frame(video: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"can't read frame {index} of {video}")
    return img


def box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def find_parts(client: SamClient, video: Path, n_frames: int, prompts: list[str]) -> dict:
    """{obj_id: {prompt, frame, box, confidence, anchors}}: the best key-frame detection per
    prompt, plus every key frame where the part is found ("anchors", [[frame, box], ...]) so
    tracking re-acquires a part that disappears and comes back.

    On each key frame, a detection whose box overlaps an earlier part's detection on that
    frame (IoU > 0.5) is skipped, so list unambiguous parts first: "gray panel" after
    "base plate" picks the panel, not the base. Comparing within a frame keeps this
    working when the camera or the part moves between key frames.
    """
    frames = sorted({int(f * n_frames) for f in KEY_FRAMES})
    images = {f: read_frame(video, f) for f in frames}
    taken = {f: [] for f in frames}  # boxes claimed by earlier parts, per key frame
    parts = {}
    for oid, prompt in enumerate(prompts):
        best = None
        best_per_frame = {}
        for f, img in images.items():
            for det in parse(client.segment_image(img, prompt)):
                if any(box_iou(det.box, b) > 0.5 for b in taken[f]):
                    continue
                if f not in best_per_frame or (det.confidence or 0) > best_per_frame[f].confidence:
                    best_per_frame[f] = det
                if best is None or (det.confidence or 0) > best[1].confidence:
                    best = (f, det)
        for f, det in best_per_frame.items():
            taken[f].append(det.box)
        if best is None:
            print(f"  {prompt!r}: not found on key frames {frames}, skipped")
            continue
        f, det = best
        anchors = [[t, list(d.box)] for t, d in sorted(best_per_frame.items())
                   if (d.confidence or 0) >= MIN_ANCHOR_CONFIDENCE]
        parts[oid] = {"prompt": prompt, "frame": f, "box": list(det.box), "confidence": det.confidence,
                      "anchors": anchors or [[f, list(det.box)]]}
        print(f"  {prompt!r}: id {oid}, best frame {f}, c={det.confidence:.2f}, anchors on frames "
              f"{[a[0] for a in parts[oid]['anchors']]}")
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("prompts", nargs="+")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "tracks")
    args = ap.parse_args()

    out_dir = args.out / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cache, parts_file = out_dir / "tracks.sam.txt", out_dir / "parts.json"

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = video_info(args.video)[1]

    cached = parts_file.exists() and [p["prompt"] for p in json.loads(parts_file.read_text()).values()] == args.prompts
    if not (cached and cache.exists()):
        client = sam_client()
        parts = find_parts(client, args.video, n_frames, args.prompts)
        if not parts:
            sys.exit("no parts found")
        cache.write_text(client.track_boxes(args.video, (w, h), {
            oid: [(f, tuple(box)) for f, box in p.get("anchors", [[p["frame"], p["box"]]])] for oid, p in parts.items()}))
        parts_file.write_text(json.dumps(parts, indent=2))
    parts = {int(k): v for k, v in json.loads(parts_file.read_text()).items()}
    objs = load_tracks(cache)
    for oid, p in parts.items():
        print(f"{p['prompt']!r} (id {oid}): {len(objs.get(oid, {}))}/{n_frames} frames tracked")

    writer = cv2.VideoWriter(str(out_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    frame = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        for oid, frames in objs.items():
            det = frames.get(frame)
            if det is None:
                continue
            color = COLORS[oid % len(COLORS)]
            m = det.full_mask(h, w)
            img[m] = (0.5 * img[m] + 0.5 * np.array(color)).astype(np.uint8)
            cv2.rectangle(img, det.box[:2], det.box[2:], color, 1)
            label = parts.get(oid, {}).get("prompt", "?")
            cv2.putText(img, f"{label}#{oid}", (det.box[0], max(det.box[1] - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        writer.write(img)
        frame += 1
    writer.release()
    print(f"overlay -> {out_dir / 'overlay.mp4'} ({frame} frames)")


if __name__ == "__main__":
    main()
