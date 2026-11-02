"""How far a part's top stands above the floor, from parallax between video frames: for captures
whose depth is estimated from RGB (MapAnything), which places an object's surfaces centimetres off
the floor next to it (the LiDAR laptop's own video: its 12 mm base read 50-75 mm thick).

Geometry, not the depth network: points tracked on the floor around the object and on the part's
top, from the reference frame to frames where the camera has moved. The floor's homography
between two views gives the camera's turn and its move over its distance to the floor (the floor
is the dominant, textured plane, which makes the five-point essential matrix degenerate but the
homography well posed); each top point's parallax off that homography is its height in units of
that distance. The capture's depth is used only for that one distance (metres), and its poses only
to pick frames and the homography's solution.
"""

import cv2
import numpy as np

MIN_BASELINE = 0.12  # camera move / distance to the floor, for a partner frame
SPAN = 150  # frames each way from the reference searched for partners
PARTNERS = 8  # partner frames used
MIN_GOOD = 3  # partners that must agree for a measurement
WORK_SCALE = 0.5  # images tracked at half size
MIN_PLANE_POINTS, MIN_TOP_POINTS = 60, 40


def plane_heights(x1: np.ndarray, x2: np.ndarray, on_plane: np.ndarray, k: np.ndarray, turn_hint: np.ndarray,
                  thresh_px: float = 3.0):
    """Heights above a plane, in units of view 1's distance to it, of the points tracked x1 -> x2
    (pixels, N x 2), from the plane's homography (fitted on the points flagged ``on_plane``)
    decomposed with the intrinsics ``k``; of its solutions the one whose turn is nearest
    ``turn_hint`` (view 1 -> view 2 rotation) with every plane point in front of both views.
    Positive toward view 1. Returns (heights for all points, inlier mask of the plane fit) or None."""
    if on_plane.sum() < MIN_PLANE_POINTS:
        return None
    h, inl = cv2.findHomography(x1[on_plane], x2[on_plane], cv2.RANSAC, thresh_px)
    if h is None:
        return None
    n_sol, rs, ts, ns = cv2.decomposeHomographyMat(h, k)
    kinv = np.linalg.inv(k)
    r1 = np.column_stack([x1, np.ones(len(x1))]) @ kinv.T  # rays of view 1
    best = None
    for r, t, n in zip(rs, ts, ns):
        n = n[:, 0]
        t = t[:, 0]
        # plane points: n.X = 1 (in units of the distance); depth along each ray z = 1 / (n . ray)
        z = 1.0 / (r1[on_plane] @ n)
        if np.median(z) <= 0:
            continue
        x_plane = r1[on_plane] * z[:, None]
        in_front = ((x_plane @ r.T + t)[:, 2] > 0).mean()
        angle = np.arccos(np.clip((np.trace(r @ turn_hint.T) - 1) / 2, -1, 1))
        score = (in_front > 0.9, -angle)
        if best is None or score > best[0]:
            best = (score, r, t, n)
    if best is None or not best[0][0]:
        return None
    _, r, t, n = best
    p1 = k @ np.hstack([np.eye(3), np.zeros((3, 1))])
    p2 = k @ np.hstack([r, t[:, None]])
    xh = cv2.triangulatePoints(p1, p2, x1.T.astype(np.float64), x2.T.astype(np.float64))
    pts = (xh[:3] / xh[3]).T
    return 1.0 - pts @ n, inl[:, 0].astype(bool)


def track_points(grays: dict, ref: int, p0: np.ndarray, frames: list[int]) -> dict:
    """KLT from the reference frame to each of ``frames``, frame by frame through the ones between,
    forward-backward checked: frame -> (points, alive)."""
    out = {}
    for step in (1, -1):
        targets = sorted((t for t in frames if (t - ref) * step > 0), key=lambda t: abs(t - ref))
        if not targets:
            continue
        pts, alive, prev = p0.astype(np.float32).copy(), np.ones(len(p0), bool), ref
        for t in range(ref + step, targets[-1] + step, step):
            if t not in grays or prev not in grays:
                break
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(grays[prev], grays[t], pts, None, winSize=(21, 21), maxLevel=3)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(grays[t], grays[prev], nxt, None, winSize=(21, 21), maxLevel=3)
            alive &= (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(back - pts, axis=1) < 0.7)
            pts, prev = nxt, t
            if t in targets:
                out[t] = (pts.copy(), alive.copy())
    return out


def top_height(c, top: str, around: list[str], floor: str, exclude: list[str], log=print) -> dict | None:
    """The height (m) of part ``top``'s upper surface above ``floor`` next to the object, by
    parallax (module docstring), around the capture's reference frame. ``around``: the parts
    making the object (the floor ring is taken around them), ``exclude``: parts that move (the
    lid), kept out of the tracked points. Needs the camera's focal length to be known. Returns
    {"height_m", "partners", "heights_m", ...} or None when the frames don't allow it."""
    from .lift3d import fit_plane, mask_cloud
    ref, w, h = c.ref, c.w, c.h
    if not (c.has(top, ref) and c.has(floor, ref)):
        return None
    k = np.array([[c.cap.intrinsics.f, 0, c.cap.intrinsics.cx], [0, c.cap.intrinsics.f, c.cap.intrinsics.cy],
                  [0, 0, 1.0]])
    obj = np.zeros((h, w), bool)
    for p in around:
        if c.has(p, ref):
            obj |= c.by[p][ref].full_mask(h, w)
    moving = np.zeros((h, w), bool)
    for p in exclude:
        if c.has(p, ref):
            moving |= c.by[p][ref].full_mask(h, w)
    pad = int(0.04 * max(w, h))
    ring = (cv2.dilate(obj.astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) > 0)
    ring &= c.by[floor][ref].full_mask(h, w) & ~obj & ~moving
    edge = max(3, int(0.004 * max(w, h)))
    inner = cv2.erode(c.by[top][ref].full_mask(h, w).astype(np.uint8), np.ones((2 * edge + 1,) * 2, np.uint8)) > 0
    inner &= ~(cv2.dilate(moving.astype(np.uint8), np.ones((2 * edge + 1,) * 2, np.uint8)) > 0)
    floor_cloud = mask_cloud(ring, c.cap.depth[ref], c.cap.intrinsics)
    if len(floor_cloud) < 50 or inner.sum() < 200:
        return None
    nf, df, _ = fit_plane(floor_cloud, np.random.default_rng(0))
    distance = abs(df)  # the reference camera's distance to the floor near the object, capture units
    # Partner frames: the camera moved at least MIN_BASELINE of that distance (the capture's poses).
    lo, hi = max(0, ref - SPAN), min(c.n - 1, ref + SPAN)
    moved = {t: float(np.linalg.norm(np.linalg.inv(c.world[t])[:3, 3])) for t in range(lo, hi + 1)}
    far = [t for t in moved if moved[t] >= MIN_BASELINE * distance]
    if not far:
        return None
    far = sorted(far, key=lambda t: -moved[t])
    partners = sorted({far[int(i)] for i in np.linspace(0, len(far) - 1, min(PARTNERS, len(far)))})
    need = range(min(partners + [ref]), max(partners + [ref]) + 1)
    grays, cap = {}, cv2.VideoCapture(str(c.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, need.start)
    for t in need:
        ok, img = cap.read()
        if not ok:
            break
        grays[t] = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=WORK_SCALE, fy=WORK_SCALE,
                              interpolation=cv2.INTER_AREA)
    cap.release()
    small = cv2.resize((inner | ring).astype(np.uint8), None, fx=WORK_SCALE, fy=WORK_SCALE,
                       interpolation=cv2.INTER_NEAREST)
    p0 = cv2.goodFeaturesToTrack(grays[ref], 4000, 0.005, 5, mask=small)
    if p0 is None:
        return None
    p0 = p0[:, 0, :]
    full = p0 / WORK_SCALE
    is_top = inner[np.clip(full[:, 1].astype(int), 0, h - 1), np.clip(full[:, 0].astype(int), 0, w - 1)]
    tracked = track_points(grays, ref, p0, partners)
    results = []
    for t, (pts, alive) in sorted(tracked.items()):
        x1, x2, top_pts = full[alive], pts[alive] / WORK_SCALE, is_top[alive]
        if top_pts.sum() < MIN_TOP_POINTS:
            continue
        turn = c.world[t][:3, :3]  # reference camera -> frame t camera, from the capture's poses
        found = plane_heights(x1, x2, ~top_pts, k, turn)
        if found is None:
            continue
        heights, inliers = found
        hm, floor_off = heights[top_pts] * distance, np.abs(heights[~top_pts]) * distance
        # The floor points must sit on the floor (a stray point on a chair leg or the object's
        # edge is fine, most must not be), well below the top's own height.
        results.append({"frame": int(t), "height": float(np.median(hm)),
                        "spread": float(np.subtract(*np.percentile(hm, [75, 25]))), "points": int(top_pts.sum()),
                        "floor_off": float(np.median(floor_off)), "plane_share": float(inliers.mean())})
    good = [r for r in results if r["height"] > 0 and r["floor_off"] < max(0.25 * r["height"], 0.001)]
    if len(good) < MIN_GOOD:
        log(f"parallax: {len(good)} usable partner frames of {len(partners)} ({len(results)} solved): no measurement")
        return None
    heights = np.array([r["height"] for r in good])
    med = float(np.median(heights))
    agree = np.abs(heights - med) <= max(0.3 * med, 0.003)
    if agree.sum() < MIN_GOOD:
        log(f"parallax: partner frames disagree ({', '.join(f'{1000 * x:.0f}' for x in heights)} mm): no measurement")
        return None
    height = float(np.median(heights[agree]))
    return {"height_m": height, "partners": [r["frame"] for r, a in zip(good, agree) if a],
            "heights_m": [r["height"] for r in good], "floor_distance": distance, "solved": len(results),
            "tried": len(partners)}
