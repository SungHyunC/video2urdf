"""A cheap check of a capture before any model is paid for: length, frame rate, size, exposure,
sharpness, whether the camera holds still and whether anything moves. The numbers come from a
few frames decoded on this machine; nothing is sent anywhere. Problems become reshoot tips
(the filming notes in CLAUDE.md). Only input that nothing can be measured from is refused.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SAMPLES = 24  # frames spread over the video
WIDTH = 480  # px, analysis width
MIN_FRAMES = 8
MIN_SECONDS = 0.5
MIN_SIDE = 120
SHORT_S, LONG_S = 3.0, 90.0
MIN_INLIERS = 20  # background matches for a frame pair to count as matched
FIXED_SHIFT, FIXED_TURN = 0.06, 3.0  # fraction of the diagonal, degrees: a tripod
STEADY_SHIFT, STEADY_TURN = 0.25, 10.0  # a hand holding the same view
MOVING_AREA = 0.005  # fraction of the frame that changes when a part moves


def sample(video: Path, n: int, k: int = SAMPLES, width: int = WIDTH) -> tuple[list[int], list[np.ndarray]]:
    """Frames spread over the video (decoded in order: seeking is unreliable on phone video)."""
    want = sorted({int(round(x)) for x in np.linspace(0, max(0, n - 1), min(k, max(1, n)))})
    cap = cv2.VideoCapture(str(video))
    out, idx, t, j = [], [], 0, 0
    while j < len(want):
        if not cap.grab():
            break
        if t == want[j]:
            ok, img = cap.retrieve()
            if ok:
                h, w = img.shape[:2]
                out.append(cv2.resize(img, (width, max(2, round(h * width / w))), interpolation=cv2.INTER_AREA))
                idx.append(t)
            j += 1
        t += 1
    cap.release()
    return idx, out


def _orb(gray: np.ndarray):
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=10)
    return orb.detectAndCompute(gray, None)


def _affine(a, b) -> tuple[np.ndarray | None, int]:
    """Similarity transform taking frame b's points onto frame a's, and its inlier count."""
    (ka, da), (kb, db) = a, b
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(db, da)
    if len(matches) < 8:
        return None, 0
    pb = np.float32([kb[m.queryIdx].pt for m in matches])
    pa = np.float32([ka[m.trainIdx].pt for m in matches])
    m, inl = cv2.estimateAffinePartial2D(pb, pa, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if m is None:
        return None, 0
    return m, int(inl.sum())


def camera_motion(grays: list[np.ndarray]) -> dict:
    """How the background moves: each sampled frame registered to the first by a similarity
    transform (RANSAC over ORB matches; a moving part or hand is a minority of the matches).
    'fixed' (tripod), 'steady' (a hand holding about the same view), 'moving' (walking or
    circling: the view changes) or 'unknown' (too little texture to tell)."""
    if len(grays) < 2:
        return {"camera": "unknown", "why": "too few frames"}
    h, w = grays[0].shape
    diag = float(np.hypot(w, h))
    feats = [_orb(g) for g in grays]
    centre = np.array([w / 2, h / 2, 1.0])
    to_first, shifts, turns, scales, matched = [], [], [], [], 0
    for f in feats[1:]:
        m, inl = _affine(feats[0], f)
        to_first.append(m if inl >= MIN_INLIERS else None)
        if inl >= MIN_INLIERS:
            matched += 1
            shifts.append(float(np.linalg.norm(m @ centre - centre[:2])) / diag)
            turns.append(abs(float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))))
            scales.append(float(np.sqrt(abs(np.linalg.det(m[:, :2])))))
    steps = [_affine(a, b)[1] >= MIN_INLIERS for a, b in zip(feats, feats[1:])]
    pairs = len(grays) - 1
    zooms = [abs(float(np.log(s))) for s in scales]
    # A plain wall gives few background matches, and then a registration can lock onto the moving
    # part instead: judge by the typical (75th percentile) frame, not the worst one.
    p75 = (lambda v: float(np.percentile(v, 75)) if v else None)
    out = {"pairs": pairs, "matched_to_first": matched, "consecutive_matched": int(sum(steps)),
           "max_shift": round(max(shifts), 4) if shifts else None,
           "max_turn_deg": round(max(turns), 2) if turns else None,
           "p75_shift": round(p75(shifts), 4) if shifts else None,
           "p75_turn_deg": round(p75(turns), 2) if turns else None,
           "scale_range": [round(min(scales), 3), round(max(scales), 3)] if scales else None,
           "transforms": to_first}
    share = matched / pairs
    if share == 1 and max(shifts) <= FIXED_SHIFT and max(turns) <= FIXED_TURN and max(zooms) <= 0.05:
        out.update(camera="fixed", why=f"the background stays put, moving at most {100 * max(shifts):.1f}% of the frame")
    elif share >= 0.85 and p75(shifts) <= STEADY_SHIFT and p75(turns) <= STEADY_TURN and p75(zooms) <= 0.15:
        out.update(camera="steady", why=f"the camera keeps about the same view: the background shifts "
                                        f"{100 * p75(shifts):.0f}% of the frame on most frames")
    elif sum(steps) < 0.7 * len(steps):
        out.update(camera="unknown", why=f"too little texture to register frames: {sum(steps)}/{len(steps)} "
                                         "neighbouring samples matched")
    elif share < 0.6 or (shifts and (p75(shifts) > 0.35 or p75(turns) > 20 or p75(zooms) > 0.25)):
        out.update(camera="moving", why=f"the view changes: {matched}/{pairs} sampled frames still match the first"
                                        + (f", the background shifts {100 * p75(shifts):.0f}% of the frame"
                                           if shifts else ""))
    else:  # a hand-held camera drifting, or a plain background: the numbers alone can't tell
        out.update(camera="unsteady", why=f"{matched}/{pairs} sampled frames match the first, the background "
                                          f"shifts {100 * p75(shifts):.0f}% of the frame on most of them")
    return out


def moving_area(grays: list[np.ndarray], to_first: list) -> dict:
    """With a fixed or steady camera: the part of the frame that changes once the background
    is aligned (a moving part, and the hand moving it)."""
    h, w = grays[0].shape
    first = cv2.GaussianBlur(grays[0], (5, 5), 0).astype(np.int16)
    count = np.zeros((h, w), np.int16)
    used = 0
    for g, m in zip(grays[1:], to_first):
        if m is None:
            continue
        warped = cv2.warpAffine(cv2.GaussianBlur(g, (5, 5), 0), m, (w, h), borderMode=cv2.BORDER_REFLECT)
        count += (np.abs(warped.astype(np.int16) - first) > 30).astype(np.int16)
        used += 1
    if not used:
        return {"fraction": None}
    b = max(2, int(0.03 * min(h, w)))
    changed = count[b:-b, b:-b] >= 2
    frac = float(changed.mean())
    ys, xs = np.nonzero(changed)
    box = [round(float(xs.min()) / w, 3), round(float(ys.min()) / h, 3), round(float(xs.max()) / w, 3),
           round(float(ys.max()) / h, 3)] if len(xs) else None
    return {"fraction": round(frac, 4), "box": box, "frames": used}


def exposure(frames: list[np.ndarray]) -> dict:
    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    return {"mean": round(float(grays.mean()), 1), "dark_fraction": round(float((grays < 20).mean()), 3),
            "clipped_fraction": round(float((grays > 250).mean()), 3),
            "contrast": round(float(np.median([g.std() for g in grays])), 1)}


def sharpness(grays: list[np.ndarray]) -> float:
    """Median variance of the Laplacian at the analysis width (sharp phone video: ~100-1000)."""
    return round(float(np.median([cv2.Laplacian(g, cv2.CV_64F).var() for g in grays])), 1)


def depth_coverage(capture: Path, idx: list[int]) -> float | None:
    """Share of pixels with confident depth on the sampled frames (Stray layout only)."""
    depth = sorted((capture / "depth").glob("*.png")) if (capture / "depth").is_dir() else []
    if not depth:
        return None
    vals = []
    for t in idx:
        if t >= len(depth):
            continue
        d = cv2.imread(str(depth[t]), cv2.IMREAD_UNCHANGED)
        conf_path = capture / "confidence" / depth[t].name
        conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED) if conf_path.exists() else None
        if d is None:
            continue
        ok = d > 0
        if conf is not None and conf.shape == d.shape:
            ok &= conf >= 2
        vals.append(ok.mean())
    return round(float(np.mean(vals)), 3) if vals else None


def check(video: Path, frames: int, fps: float, size: tuple[int, int], has_depth: bool,
          capture: Path | None = None) -> dict:
    """Measure the capture. Writes nothing; ``assess`` turns the numbers into a verdict and tips."""
    idx, bgr = sample(video, frames)
    metrics = {"frames": frames, "fps": round(fps, 3), "seconds": round(frames / fps, 2) if fps else None,
               "width": size[0], "height": size[1], "sampled": len(idx)}
    if not bgr:
        return {**metrics, "decodable": False}
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in bgr]
    motion = camera_motion(grays)
    transforms = motion.pop("transforms", [])
    metrics.update(decodable=True, exposure=exposure(bgr), sharpness=sharpness(grays), camera=motion)
    if motion["camera"] in ("fixed", "steady", "unsteady"):
        metrics["moving_area"] = moving_area(grays, transforms)
    if has_depth and capture is not None:
        metrics["depth_coverage"] = depth_coverage(capture, idx)
    metrics["sample_frames"] = idx
    return metrics


def assess(m: dict, has_depth: bool, look: dict | None = None, assumed_fps: bool = False) -> dict:
    """Usable or not, warnings and reshoot tips for the measured capture (and Spark's look, if any).
    ``assumed_fps``: the frame rate was given (a photo folder), so it isn't judged."""
    warnings, tips = [], []
    if not m.get("decodable", False):
        return {"usable": False, "reason": "no frame could be decoded", "warnings": [], "tips": []}
    if m["frames"] < MIN_FRAMES or (m["seconds"] or 0) < MIN_SECONDS:
        return {"usable": False, "reason": f"only {m['frames']} frames ({m['seconds']} s): a joint can't be "
                                           "seen moving; film 5-20 s", "warnings": [], "tips": []}
    if min(m["width"], m["height"]) < MIN_SIDE:
        return {"usable": False, "reason": f"{m['width']}x{m['height']} px is too small to segment parts; "
                                           "film at 720p or more", "warnings": [], "tips": []}
    ex = m["exposure"]
    if ex["mean"] < 8 or ex["contrast"] < 2:
        return {"usable": False, "reason": "the frames are black or blank", "warnings": [], "tips": []}
    if m["seconds"] < SHORT_S:
        warnings.append(f"short video ({m['seconds']} s)")
        tips.append("Film 5-20 s and move the part through its whole range at least once, slowly.")
    if m["seconds"] > LONG_S:
        warnings.append(f"long video ({m['seconds']:.0f} s): tracking costs grow with length")
        tips.append("Trim the video to the part where the object moves.")
    if m["fps"] < 15 and not assumed_fps:
        warnings.append(f"low frame rate ({m['fps']:.1f} fps)")
        tips.append("Film at 30 fps: at low rates fast motion jumps between frames.")
    if min(m["width"], m["height"]) < 480:
        warnings.append(f"low resolution ({m['width']}x{m['height']})")
        tips.append("Film at 1080p: small parts get lost at low resolution.")
    if ex["mean"] < 55 or ex["dark_fraction"] > 0.4:
        warnings.append(f"dark (mean brightness {ex['mean']:.0f}/255)")
        tips.append("Add light: dark surfaces give poor masks" + (" and no LiDAR depth." if has_depth else "."))
    if ex["clipped_fraction"] > 0.15:
        warnings.append(f"{100 * ex['clipped_fraction']:.0f}% of pixels clipped white")
        tips.append("Avoid direct sun or a bright window behind the object; lower the exposure.")
    if m["sharpness"] < 20:
        warnings.append(f"blurry frames (sharpness {m['sharpness']:.0f})")
        tips.append("Hold the camera still (a tripod helps), move the part slowly, and tap to focus on it.")
    cam = m.get("camera", {})
    area = (m.get("moving_area") or {}).get("fraction")
    if area is not None and area < MOVING_AREA and not (look or {}).get("anything_moves", False):
        warnings.append(f"nothing seems to move ({100 * area:.2f}% of the frame changes)")
        tips.append("Open and close (or pull out) the part through its whole range while filming; with no "
                    "motion only a rigid model can be built.")
    if has_depth:
        cover = m.get("depth_coverage")
        if cover is not None and cover < 0.3:
            warnings.append(f"LiDAR depth on only {100 * cover:.0f}% of the frame")
            tips.append("Film from 0.3-2 m: LiDAR misses far, very close, black, glossy and glass surfaces.")
    elif cam.get("camera") == "moving" and (look or {}).get("view") == "side":
        tips.append("For a side view, keep the camera still (a tripod) facing the plane the part moves in; a "
                    "moving camera needs the 3D path, which estimates depth and is less accurate.")
    if look:
        if look.get("whole_object_in_frame") is False:
            warnings.append("the object leaves the frame (Spark's look)")
            tips.append("Keep the whole object in frame the whole time: step back or zoom out.")
        if look.get("lighting") in ("dark", "backlit", "overexposed") and not any("dark" in w or "clipped" in w
                                                                                   for w in warnings):
            warnings.append(f"lighting: {look['lighting']} (Spark's look)")
            tips.append("Light the object evenly from the camera's side; avoid a bright window behind it.")
        if look.get("anything_moves") is False and not any("nothing seems to move" in w for w in warnings):
            warnings.append("nothing seems to move (Spark's look)")
            tips.append("Move the part through its whole range while filming; with no motion only a rigid "
                        "model can be built.")
        if not has_depth and look.get("card_visible") is False:
            tips.append("For real sizes, lay a card-sized reference (85.6 x 54 mm; a loyalty card, never a "
                        "payment card: frames go to a cloud API) flat next to the object, in view.")
        for p in look.get("problems") or []:
            if not isinstance(p, str) or not p.strip() or p in tips:
                continue
            if has_depth and any(w in p.lower() for w in ("card", "scale", "reference")):
                continue  # depth captures are metric: Spark can't tell from frames that no card is needed
            tips.append(p.strip())
    return {"usable": True, "reason": "", "warnings": warnings, "tips": tips}
