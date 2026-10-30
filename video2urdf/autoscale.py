"""Metric scale for a capture whose depth was estimated from ordinary video (MapAnything).

Its scale can be several times off. Two ways to fix it without a person picking pixels:
1. An ID-1 card (85.6 x 54 mm) in view: SAM 3.1 finds "card"s on a few frames; each mask is
   lifted with the estimated depth, a plane is fitted, and the card's outline in that plane is
   measured; only its edge across the view counts (estimated depth stretches or squeezes lengths
   along the line of sight). The scale is the card's edge length over the measured one.
2. Otherwise Muse Spark names the object and the usual size of its largest dimension; SAM 3.1
   segments the object, and on frames where it is wholly in view with its long axis across the view, its depth points
   give the measured length; the scale is the usual size over that. This is an estimate from the
   object's category, recorded as such; with no frame showing the long axis across, no scale.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np

from . import stray
from .lift3d import fit_plane, mask_cloud
from .pipeline import read_frame, video_info
from .sam import parse

CARD_MM = (85.6, 53.98)
CARD_ASPECT = CARD_MM[0] / CARD_MM[1]


def _frames(n: int, k: int) -> list[int]:
    return sorted({int(t) for t in np.linspace(0.05 * n, 0.95 * n - 1, k)})


def card_edge_mm(points: np.ndarray, rng) -> dict | None:
    """The card's outline in its fitted plane: each side's length (mm, in the estimated depth's
    units) and how far it runs along the line of sight (|cos|). Estimated depth stretches or
    squeezes a side that runs away from the camera, so only the side across the view is used."""
    if len(points) < 30:
        return None
    z = max(1.0, float(np.median(points[:, 2])))
    n, d, frac = fit_plane(points, rng, tol=0.004 * z)
    if frac < 0.6:
        return None
    on = points[np.abs(points @ n - d) < 0.01 * z]
    e1 = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    uv = np.column_stack([on @ e1, on @ e2]).astype(np.float32)
    (_, _), (a, b), angle = cv2.minAreaRect(uv)
    th = np.deg2rad(angle)
    da, db = np.cos(th) * e1 + np.sin(th) * e2, -np.sin(th) * e1 + np.cos(th) * e2
    ray = np.median(on, axis=0)
    ray /= np.linalg.norm(ray)
    sides = sorted([(abs(float(da @ ray)), a * 1000), (abs(float(db @ ray)), b * 1000)])
    (cos_across, across), (cos_along, along) = sides
    if min(across, along) <= 0 or cos_across > 0.5:
        return None
    return {"across_mm": across, "along_mm": along, "cos_across": cos_across, "cos_along": cos_along,
            "plane_inliers": frac, "depth_m": float(np.median(on[:, 2]))}


def card_scale(capture: Path, video: Path, sam, frames: int = 8, log=print) -> dict | None:
    """Scale from ID-1 cards found on a few frames, or None."""
    cap = stray.load(capture)
    (w, h), n, _ = video_info(video)
    rng = np.random.default_rng(0)
    found = []
    for t in _frames(min(n, len(cap.depth)), frames):
        img = read_frame(video, t)
        for det in parse(sam.segment_image(img, "card")):
            if (det.confidence or 1.0) < 0.5:
                continue
            m = det.full_mask(h, w)
            area = m.sum()
            if area < 0.0005 * w * h or area > 0.2 * w * h:
                continue
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.minAreaRect(max(cs, key=cv2.contourArea))
            if area / max(1.0, rect[1][0] * rect[1][1]) < 0.75:  # a card's image is a (projected) rectangle
                continue
            inner = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            edge = card_edge_mm(mask_cloud(inner, cap.depth[t], cap.intrinsics), rng)
            if edge is None:
                continue
            # The side across the view is the long edge when it measures longer than the other.
            long_edge = edge["across_mm"] >= edge["along_mm"]
            px = 5 * edge["depth_m"] / cap.intrinsics.f * 1000  # the 5 px erosion, back at this depth
            known = CARD_MM[0] if long_edge else CARD_MM[1]
            found.append({"frame": t, "across_mm": edge["across_mm"] + px, "along_mm": edge["along_mm"] + px,
                          "long_edge_across": long_edge, "scale": known / (edge["across_mm"] + px),
                          "depth_m": edge["depth_m"]})
            log(f"card on frame {t}: {edge['across_mm'] + px:.1f} mm across the view "
                f"({'long' if long_edge else 'short'} edge), {edge['along_mm'] + px:.1f} along it")
    if not found:
        return None
    scales = np.array([f["scale"] for f in found])
    med = float(np.median(scales))
    spread = float(np.percentile(scales, 90) / max(np.percentile(scales, 10), 1e-9)) if len(scales) > 1 else 1.0
    return {"method": "card", "scale_factor": med, "frames": found, "spread_p90_p10": spread,
            "depth_m": float(np.median([f["depth_m"] for f in found])),
            "note": "ID-1 card (85.6 x 54 mm): its edge across the view measured in the estimated depth; "
                    "one global scale."}


def prior_scale(capture: Path, video: Path, sam, frames: int = 6, log=print) -> dict | None:
    """Scale from the object's usual size, as Muse Spark judges it, or None."""
    from openai import OpenAI
    from .agent import API_KEY_ENV, BASE_URL, MODEL, jpeg_url

    (w, h), n, _ = video_info(video)
    mid = read_frame(video, n // 2)
    client = OpenAI(base_url=BASE_URL, api_key=os.environ[API_KEY_ENV], timeout=600)
    resp = client.chat.completions.create(model=MODEL, max_tokens=4000, response_format={"type": "json_object"}, messages=[
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": jpeg_url(mid)}},
            {"type": "text", "text": "Name the main object in this video frame (the one being handled or filmed), a short "
             "noun phrase to segment it (e.g. 'laptop', 'desk lamp'), and the usual range of its largest dimension in "
             "millimetres for this kind of object, as narrow as the visible model allows. Answer JSON: "
             '{"object": str, "prompt": str, "largest_dimension_mm": [lo, hi], "reason": str}'}]}])
    try:
        ans = json.loads(resp.choices[0].message.content)
        lo, hi = map(float, ans["largest_dimension_mm"])
        prompt = str(ans["prompt"])
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if not (0 < lo <= hi):
        return None
    cap = stray.load(capture)
    extents, depths = [], []
    for t in _frames(min(n, len(cap.depth)), frames):
        dets = [d for d in parse(sam.segment_image(read_frame(video, t), prompt)) if (d.confidence or 1.0) >= 0.5]
        if not dets:
            continue
        d = max(dets, key=lambda d: d.area)
        x1, y1, x2, y2 = d.box
        if x1 <= 2 or y1 <= 2 or x2 >= w - 3 or y2 >= h - 3:  # cut off by the frame: its length isn't all there
            log(f"  frame {t}: the object touches the frame's edge; not used")
            continue
        inner = cv2.erode(d.full_mask(h, w).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        p = mask_cloud(inner, cap.depth[t], cap.intrinsics)
        if len(p) < 100:
            continue
        c = p - np.median(p, axis=0)
        axes = np.linalg.svd(c[:: max(1, len(c) // 3000)], full_matrices=False)[2]
        ray = np.median(p, axis=0) / np.linalg.norm(np.median(p, axis=0))
        # Estimated depth stretches or squeezes lengths along the line of sight: only a largest
        # axis running across the view (as the card's edge) is a fair length.
        if abs(axes[0] @ ray) > 0.4:
            log(f"  frame {t}: the object's long axis runs into the view ({abs(axes[0] @ ray):.2f}); not used")
            continue
        extents.append(float(np.ptp(np.percentile(c @ axes[0], [2, 98]))) * 1000)
        depths.append(float(np.median(p[:, 2])))
    if len(extents) < 2:
        log("the object's long axis is never across the view: no size-based scale")
        return None
    measured = float(np.percentile(extents, 75))  # partial views shrink it; the larger views see more
    typical = (lo * hi) ** 0.5
    log(f"Spark: {ans['object']}, usually {lo:.0f}-{hi:.0f} mm; measured {measured:.0f} mm in the estimated depth")
    return {"method": "object_prior", "scale_factor": typical / measured, "object": ans["object"], "prompt": prompt,
            "typical_mm": [lo, hi], "measured_largest_before_mm": measured, "views": len(extents),
            "depth_m": float(np.median(depths)),
            "reason": ans.get("reason"),
            "note": "Scale from the object's usual size (Muse Spark), not a measurement: verify a dimension."}


def auto_scale(capture: Path, video: Path, sam=None, log=print) -> dict | None:
    """The card's scale when it lies about as far from the camera as the object; otherwise the
    object's usual size, measured at the object's own distance. Estimated depth is not one
    global scale: a card near the lens and an object further back can disagree by half."""
    if sam is None:
        from .sam import sam_client
        sam = sam_client()
    card = card_scale(capture, video, sam, log=log)
    if card is not None and card["spread_p90_p10"] >= 1.25:
        log(f"card measurements disagree (p90/p10 {card['spread_p90_p10']:.2f}); not used")
        card = None
    prior = prior_scale(capture, video, sam, log=log)
    if card is None or prior is None:
        return card or prior
    near = abs(card["depth_m"] / prior["depth_m"] - 1) < 0.3
    chosen, other = (card, prior) if near else (prior, card)
    chosen = dict(chosen)
    chosen["check"] = {"method": other["method"], "scale_factor": other["scale_factor"],
                       "ratio": other["scale_factor"] / chosen["scale_factor"]}
    log(f"card at {card['depth_m']:.2f} m, object at {prior['depth_m']:.2f} (estimated depth): using the "
        f"{chosen['method']} (x{chosen['scale_factor']:.3f}); the {other['method']} gives x{other['scale_factor']:.3f}")
    if not near:
        chosen["note"] += " A card was in view but at another distance than the object; its scale is kept as a check."
    return chosen
