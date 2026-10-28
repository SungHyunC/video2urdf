"""Per-part 2D rigid motion from SAM masks, and camera motion from the background.

Each part's mask on the reference frame is registered to its mask on every other frame
with ECC (Euclidean: rotation + translation) on blurred, downscaled masks, warm-started
from the neighbouring frame. Poses are 3x3 image-space transforms that map points of the
part on the reference frame to the same points on frame t.

Occlusion (a hand on the base, a lid closing over a keyboard) bends a mask and so the
mask pose. Those frames take their motion from points tracked on the part's surface
(surface_poses) instead, re-anchored to the nearest unoccluded frame (fuse_poses). The
background can't stand in for the root: a handheld phone moves the far wall and the
object by different amounts (parallax).
"""

from pathlib import Path

import cv2
import numpy as np

from .sam import Detection

WORK_SIZE = 384  # long side of the downscaled masks used for registration
_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-5)
TRACK_SIZE = 960  # long side of the frames used for surface point tracking
MIN_POINTS = 20  # fewer RANSAC inliers than this means no usable surface pose
MIN_MASK_FIT = 0.9  # IoU of the registered reference mask below this means occluded


def _mask(det: Detection, h: int, w: int, k: float) -> np.ndarray:
    m = det.full_mask(h, w).astype(np.float32)
    small = cv2.resize(m, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (0, 0), 1.5)


def part_poses(frames: dict[int, Detection], size: tuple[int, int], n_frames: int,
               ref: int) -> tuple[np.ndarray, np.ndarray]:
    """Mask-registration poses (n_frames, 3, 3) relative to frame ``ref`` (NaN where untracked
    or lost), and per frame the IoU of the warped reference mask with that frame's mask,
    which drops when the part is occluded."""
    w, h = size
    k = WORK_SIZE / max(w, h)
    scale, unscale = np.diag([k, k, 1.0]), np.diag([1 / k, 1 / k, 1.0])
    template = _mask(frames[ref], h, w, k)
    small = template.shape[::-1]
    poses = np.full((n_frames, 3, 3), np.nan)
    fit = np.zeros(n_frames)
    poses[ref], fit[ref] = np.eye(3), 1.0
    for order in (range(ref + 1, n_frames), range(ref - 1, -1, -1)):
        warp = np.eye(2, 3, dtype=np.float32)
        for t in order:
            if t not in frames:
                continue
            target = _mask(frames[t], h, w, k)
            try:
                _, new = cv2.findTransformECC(template, target, warp.copy(), cv2.MOTION_EUCLIDEAN, _CRITERIA, None, 5)
            except cv2.error:
                continue  # didn't converge: leave NaN, keep the last warm start
            warp = new
            poses[t] = unscale @ np.vstack([warp, [0, 0, 1]]) @ scale
            moved = cv2.warpAffine(template, warp, small) > 0.5
            target = target > 0.5
            fit[t] = (moved & target).sum() / max((moved | target).sum(), 1)
    return poses, fit


def fuse_poses(mask_poses: np.ndarray, mask_fit: np.ndarray, surf_poses: np.ndarray,
               min_fit: float = MIN_MASK_FIT) -> np.ndarray:
    """Mask poses where the mask fits (no drift: each is registered to the reference), and
    surface-point motion elsewhere, re-anchored to the nearest frame where both exist."""
    n = len(mask_poses)
    good_mask = (mask_fit >= min_fit) & ~np.isnan(mask_poses).any(axis=(1, 2))
    has_surf = ~np.isnan(surf_poses).any(axis=(1, 2))
    anchors = np.flatnonzero(good_mask & has_surf)
    out = np.full_like(mask_poses, np.nan)
    out[good_mask] = mask_poses[good_mask]
    for t in np.flatnonzero(~good_mask & has_surf):
        if len(anchors) == 0:
            break
        a = anchors[np.argmin(np.abs(anchors - t))]
        out[t] = mask_poses[a] @ np.linalg.inv(surf_poses[a]) @ surf_poses[t]
    return out


def _rigid(a: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Drop the scale from a similarity, keeping the point centroids aligned."""
    r = a[:, :2] / np.sqrt(abs(np.linalg.det(a[:, :2])))
    return np.hstack([r, (dst.mean(axis=0) - r @ src.mean(axis=0))[:, None]])


def interpolate_poses(poses: np.ndarray) -> np.ndarray:
    """Fill NaN poses by interpolating angle and translation between valid neighbours."""
    valid = np.flatnonzero(~np.isnan(poses).any(axis=(1, 2)))
    out = poses.copy()
    if len(valid) == 0:
        return out
    t = np.arange(len(poses))
    ang = np.unwrap(np.arctan2(poses[valid, 1, 0], poses[valid, 0, 0]))
    a = np.interp(t, valid, ang)
    x = np.interp(t, valid, poses[valid, 0, 2])
    y = np.interp(t, valid, poses[valid, 1, 2])
    c, s = np.cos(a), np.sin(a)
    out[:] = np.stack([np.stack([c, -s, x], 1), np.stack([s, c, y], 1), np.tile([0.0, 0.0, 1.0], (len(t), 1))], 1)
    return out


def load_gray(video: Path) -> tuple[list[np.ndarray], float]:
    """Grayscale frames downscaled to TRACK_SIZE on the long side, and the scale factor."""
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    k = TRACK_SIZE / max(w, h)
    grays = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        grays.append(cv2.cvtColor(cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA),
                                  cv2.COLOR_BGR2GRAY))
    cap.release()
    return grays, k


def surface_poses(grays: list[np.ndarray], k: float, own: dict[int, Detection],
                  others: dict[int, list[Detection]], ref: int, size: tuple[int, int]) -> np.ndarray:
    """(n_frames, 3, 3) image-space poses of one part from points on its surface; NaN where too few.

    Corners inside the part's SAM mask (eroded, and away from other parts) are tracked frame
    to frame with KLT, outward from ``ref`` in both directions, and a RANSAC similarity from
    their reference positions gives the pose. The mask leaves out whatever covers the part,
    such as a hand, and points that fail a forward-backward check or leave the mask are
    dropped and replenished. Unlike the background, the surface has no parallax.
    """
    w, h = size
    small = grays[0].shape[::-1]
    erode, dilate = np.ones((7, 7), np.uint8), np.ones((15, 15), np.uint8)

    def region(t):
        if t not in own:
            return None
        m = cv2.resize(own[t].full_mask(h, w).astype(np.uint8), small, interpolation=cv2.INTER_NEAREST)
        m = cv2.erode(m, erode)
        for det in others.get(t, []):
            o = cv2.resize(det.full_mask(h, w).astype(np.uint8), small, interpolation=cv2.INTER_NEAREST)
            m[cv2.dilate(o, dilate) > 0] = 0
        return m * 255

    def detect(t, m):
        pts = cv2.goodFeaturesToTrack(grays[t], 800, 0.001, 5, mask=m)
        return np.empty((0, 2), np.float32) if pts is None else pts.reshape(-1, 2)

    poses = np.full((len(grays), 3, 3), np.nan)
    poses[ref] = np.eye(3)
    scale, unscale = np.diag([k, k, 1.0]), np.diag([1 / k, 1 / k, 1.0])
    lk = dict(winSize=(21, 21), maxLevel=3)
    for order in (range(ref + 1, len(grays)), range(ref - 1, -1, -1)):
        pts_ref = detect(ref, region(ref))
        pts = pts_ref.copy()
        prev, a = ref, np.eye(2, 3)
        for t in order:
            m = region(t)
            if m is None or len(pts) == 0:
                continue
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(grays[prev], grays[t], pts, None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(grays[t], grays[prev], nxt, None, **lk)
            keep = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(back - pts, axis=1) < 0.5)
            xy = np.round(nxt).astype(int)
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < small[0]) & (xy[:, 1] >= 0) & (xy[:, 1] < small[1])
            keep &= inside
            keep[inside] &= m[xy[inside, 1], xy[inside, 0]] > 0
            pts_ref, pts = pts_ref[keep], nxt[keep]
            prev = t
            if len(pts) >= MIN_POINTS:
                est, inl = cv2.estimateAffinePartial2D(pts_ref, pts, method=cv2.RANSAC, ransacReprojThreshold=1.0)
                if est is not None and inl.sum() >= MIN_POINTS:
                    inl = inl[:, 0] == 1
                    pts_ref, pts = pts_ref[inl], pts[inl]
                    a = _rigid(est, pts_ref, pts)
                    poses[t] = unscale @ np.vstack([a, [0, 0, 1]]) @ scale
            # Replenish only from a well-supported pose: new points inherit its error forever.
            if len(pts) < 200 and len(pts) >= 3 * MIN_POINTS and not np.isnan(poses[t]).any():
                # replenish: new corners on frame t, mapped back to the reference
                new = detect(t, m)
                inv = cv2.invertAffineTransform(a)
                pts = np.vstack([pts, new]).astype(np.float32)
                pts_ref = np.vstack([pts_ref, new @ inv[:, :2].T + inv[:, 2]]).astype(np.float32)
    return poses
