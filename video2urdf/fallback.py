"""Never end empty-handed: when the agent builds no model, a rigid model of the whole object.

With depth (LiDAR, RGB-D or MapAnything): static_shape.build_static_model on the union of the
parts the agent tracked (the whole object), else on its single parts, largest first; an
articulated object whose union changes shape as it moves falls through to its body. Without
depth: the planar fit with every joint rigid. When nothing was tracked at all, the object is
segmented by name (Spark's look) and tracked with SAM 3.1 first. The result is labelled as a
fallback: no joint was measured.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

FLOOR_WORDS = {"floor", "ground", "table", "tabletop", "desk", "counter", "countertop", "worktop", "shelf",
               "carpet", "rug"}
BACKGROUND_WORDS = {"hand", "hands", "finger", "fingers", "person", "background", "card", "shadow", "ceiling", "sky"}
UNION = "whole_object"


def _tokens(name: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", name.lower()))


def choose_floor(names: list[str], track_dir: Path, reaches_bottom=None) -> str | None:
    """The supporting surface: the agent's own choice if it made one, else a part named like a
    floor or table that reaches the bottom of the frame (``reaches_bottom(name) -> bool``)."""
    for role_file in ("roles.json", "drawers.json", "chain.json", "static.json"):
        p = track_dir / role_file
        if p.exists():
            floor = json.loads(p.read_text()).get("floor")
            if floor in names:
                return floor
    for n in names:
        if _tokens(n) & FLOOR_WORDS and (reaches_bottom is None or reaches_bottom(n)):
            return n
    return None


def object_parts(names: list[str], floor: str | None) -> list[str]:
    return [n for n in names if n not in (floor, UNION) and not _tokens(n) & BACKGROUND_WORDS]


def _frames_at(video: Path, idx: list[int]) -> dict[int, np.ndarray]:
    import cv2
    want, out, t = sorted(set(idx)), {}, 0
    cap = cv2.VideoCapture(str(video))
    while want and cap.grab():
        if t == want[0]:
            ok, img = cap.retrieve()
            if ok:
                out[t] = img
            want.pop(0)
        t += 1
    cap.release()
    return out


def track_whole_object(video: Path, track_dir: Path, prompt: str, with_floor: bool, log=print) -> list[str]:
    """Segment the object (and the surface under it) by name on a few frames and track them."""
    from .pipeline import slug, video_info
    from .sam import parse, sam_client
    size, n, _ = video_info(video)
    w, h = size
    sam = sam_client()
    frames = _frames_at(video, [n // 2, n // 6, (5 * n) // 6])
    anchors, first = [], None
    for p in [prompt, "object"] if prompt and prompt != "object" else ["object"]:
        for t, img in frames.items():
            dets = [d for d in parse(sam.segment_image(img, p)) if d.usable(w, h)]
            if dets:
                best = max(dets, key=lambda d: d.confidence * np.sqrt(d.area))
                anchors.append((t, best.box))
                first = first or (p, best, t)
        if anchors:
            prompt = p
            break
    if not anchors:
        raise ValueError(f"SAM found no {prompt!r} on frames {sorted(frames)}")
    objects = {0: sorted(anchors)}
    parts = {0: {"name": slug(prompt) or "object", "prompt": prompt, "frame": first[2], "box": list(first[1].box),
                 "confidence": first[1].confidence, "anchors": [[t, list(b)] for t, b in sorted(anchors)]}}
    if with_floor:
        t, img = n // 2, frames[n // 2]
        best = None
        for p in ("floor", "table"):
            for d in parse(sam.segment_image(img, p)):
                if d.box[3] >= 0.85 * h and (best is None or d.area > best[1].area):  # reaches the frame's bottom
                    best = (p, d)
        if best:
            objects[1] = [(t, best[1].box)]
            parts[1] = {"name": "floor", "prompt": best[0], "frame": t, "box": list(best[1].box),
                        "confidence": best[1].confidence, "anchors": [[t, list(best[1].box)]]}
    raw = sam.track_boxes(video, size, objects)
    (track_dir / "tracks.sam.npz").unlink(missing_ok=True)
    (track_dir / "track_unions.json").unlink(missing_ok=True)
    (track_dir / "tracks.sam.txt").write_text(raw)
    (track_dir / "parts.json").write_text(json.dumps(parts, indent=2))
    log(f"  tracked {', '.join(v['name'] + ' (' + repr(v['prompt']) + ')' for v in parts.values())}")
    return [v["name"] for v in parts.values()]


def _add_union(track_dir: Path, parts: list[str], gate: str, size) -> None:
    path = track_dir / "track_unions.json"
    unions = json.loads(path.read_text()) if path.exists() else {}
    unions[UNION] = {"parts": parts, "gate": gate, "frame_size": list(size)}
    path.write_text(json.dumps(unions, indent=2) + "\n")


def static_rigid(video: Path, track_dir: Path, capture: Path, log=print) -> dict:
    from . import lidar
    from .static_shape import build_static_model
    for stale in ("chain.json", "links.json"):  # a failed chain fit's link map would mislead texturing
        (track_dir / stale).unlink(missing_ok=True)
    c = lidar.open_capture(video, capture, track_dir)

    def reaches_bottom(part):
        dets = list(c.by.get(part, {}).values())
        return bool(dets) and np.mean([d.box[3] >= 0.9 * c.h for d in dets]) >= 0.5

    floor = choose_floor(c.names, track_dir, reaches_bottom)
    objects = [p for p in object_parts(c.names, floor) if c.by.get(p)]
    if not objects:
        raise ValueError(f"no tracked object part (tracked: {c.names})")
    by_area = sorted(objects, key=lambda p: -float(np.median([d.area for d in c.by[p].values()])))
    candidates = []
    if len(objects) > 1:
        _add_union(track_dir, objects, by_area[0], (c.w, c.h))
        c = lidar.open_capture(video, capture, track_dir)
        candidates.append(UNION)
    candidates += by_area
    errors = []
    # The whole object first, as a snapshot when its parts move (that's why it is articulated);
    # then single parts that hold still; then any single part as a snapshot.
    attempts = [(p, p == UNION) for p in candidates] + [(p, True) for p in by_area]
    for part, snapshot in attempts:
        try:
            summary = build_static_model(c, part, floor, shape="box", log=log, allow_motion=snapshot)
        except ValueError as error:
            errors.append(f"{part}: {error}")
            log(f"  rigid model of {part} failed: {error}")
            continue
        what = f"the union of {', '.join(objects)}" if part == UNION else part
        moved = any("snapshot of frame" in w for w in summary.get("fit_warnings", []))
        return {"ok": True, "how": "rigid box from depth" + (", a snapshot of one frame (its parts move)"
                                                             if moved else " (static_shape)"),
                "part": part, "described": what, "floor": floor, "snapshot": moved,
                "size_mm": [round(x, 1) for x in summary.get("generated_size_mm", [])], "tried": errors}
    raise ValueError("; ".join(errors))


def planar_rigid(video: Path, track_dir: Path, px_per_m: float | None, log=print) -> dict:
    from .pipeline import fit, load_parts, video_info
    names, by = load_parts(track_dir)
    (w, h), n, _ = video_info(video)
    px_per_m = px_per_m or max(w, h) / 0.6  # the agent's assumed view: the frame spans 0.6 m
    try:
        fit(video, track_dir, px_per_m, fixed=set(names[1:]), log=lambda s: None)
        return {"ok": True, "how": "planar fit with every joint rigid", "part": names[0], "described": ", ".join(names)}
    except StopIteration:  # no frame shows every part: keep the part seen longest
        keep = max(names, key=lambda p: len(by[p]))
        parts_path = track_dir / "parts.json"
        meta = json.loads(parts_path.read_text())
        (track_dir / "parts.agent.json").write_text(json.dumps(meta, indent=2))
        from .pipeline import slug
        meta = {k: v for k, v in meta.items() if slug(v.get("name") or v["prompt"]) == keep}
        parts_path.write_text(json.dumps(meta, indent=2))
        (track_dir / "track_unions.json").unlink(missing_ok=True)
        fit(video, track_dir, px_per_m, log=lambda s: None)
        return {"ok": True, "how": "planar fit of the part seen longest (no frame shows every part)", "part": keep,
                "described": keep}


def rigid_model(video: Path, track_dir: Path, capture: Path | None, prompt: str | None,
                px_per_m: float | None = None, log=print) -> dict:
    """A rigid model in track_dir, or {'ok': False, 'error': ...}."""
    track_dir.mkdir(parents=True, exist_ok=True)
    tracked = (track_dir / "parts.json").exists() and (track_dir / "tracks.sam.txt").exists()
    try:
        if not tracked:
            log(f"  nothing tracked yet: segmenting {prompt or 'object'!r} with SAM 3.1")
            track_whole_object(video, track_dir, re.sub(r"[^\w\s-]", "", prompt or "object").strip() or "object",
                               with_floor=capture is not None, log=log)
        if capture is not None:
            return static_rigid(video, track_dir, capture, log=log)
        return planar_rigid(video, track_dir, px_per_m, log=log)
    except Exception as error:  # report, never raise: the run must still write its summary
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
