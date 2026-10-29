"""3D joints from video plus per-frame depth: no planar-view requirement.

Points on each part's surface are tracked frame to frame (KLT, inside its SAM mask), lifted
to 3D with that frame's depth, and a RANSAC Kabsch fit gives the part's rigid motion in the
camera frame. Camera poses are not needed: for a parent p and child c with camera-frame
poses P_t = C_t^-1 W_t C_ref, the relative motion P_p^-1 P_c = C_ref^-1 (W_p^-1 W_c) C_ref
is the true relative motion seen from the reference camera, so a moving phone cancels out.
Only the depth has to be metric (or at least consistently scaled) across frames.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree

from .sam import Detection

MIN_POINTS = 12
INLIER_M = 0.004  # RANSAC inlier distance
MIN_ROTATION = np.deg2rad(5)
REFINE_POINTS = 150  # surface points used by the hinge refit


@dataclass
class Intrinsics:
    f: float
    cx: float
    cy: float

    def lift(self, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) and depth along the view axis (N,) -> camera-frame points (N, 3), x right, y down, z forward."""
        return np.stack([(uv[:, 0] - self.cx) / self.f * z, (uv[:, 1] - self.cy) / self.f * z, z], axis=1)


def depth_edges(depth: np.ndarray, jump: float = 0.01) -> np.ndarray:
    """Pixels next to a depth discontinuity (> ``jump`` m within 3x3): at a mask's edge these
    read the background, metres away, and wreck any 3D fit."""
    d = depth.astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    return (cv2.dilate(d, k) - cv2.erode(d, k)) > jump


def mask_cloud(mask: np.ndarray, depth: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Camera-frame points of a full-size mask, sampled on the depth grid, away from depth edges."""
    h, w = mask.shape
    dh, dw = depth.shape
    m = cv2.resize(mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool)
    m &= (depth > 0) & ~depth_edges(depth)
    vv, uu = np.nonzero(m)
    uv = np.stack([(uu + 0.5) * w / dw, (vv + 0.5) * h / dh], axis=1)
    return intr.lift(uv, depth[vv, uu].astype(np.float64))


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """4x4 rigid transform T with T a ~= b."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, cb - r @ ca
    return t


def ransac_rigid(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, iters: int = 200):
    best = None
    for _ in range(iters):
        idx = rng.choice(len(a), 3, replace=False)
        t = kabsch(a[idx], b[idx])
        err = np.linalg.norm(a @ t[:3, :3].T + t[:3, 3] - b, axis=1)
        inl = err < INLIER_M
        if best is None or inl.sum() > best.sum():
            best = inl
    if best.sum() < MIN_POINTS:
        return None, best
    return kabsch(a[best], b[best]), best


def part_poses_3d(grays: list[np.ndarray], k: float, depth: np.ndarray, intr: Intrinsics,
                  own: dict[int, Detection], others: dict[int, list[Detection]], ref: int,
                  size: tuple[int, int]) -> np.ndarray:
    """(n, 4, 4) camera-frame poses of one part relative to frame ``ref``; NaN where too few points.

    ``grays`` are downscaled by ``k`` (see motion.load_gray); ``depth`` is (n, H, W) at full size.
    """
    w, h = size
    small = grays[0].shape[::-1]
    erode, dilate = np.ones((5, 5), np.uint8), np.ones((9, 9), np.uint8)
    rng = np.random.default_rng(0)

    def region(t):
        if t not in own:
            return None
        m = cv2.erode(cv2.resize(own[t].full_mask(h, w).astype(np.uint8), small, interpolation=cv2.INTER_NEAREST), erode)
        for det in others.get(t, []):
            o = cv2.resize(det.full_mask(h, w).astype(np.uint8), small, interpolation=cv2.INTER_NEAREST)
            m[cv2.dilate(o, dilate) > 0] = 0
        return m * 255

    dh, dw = depth.shape[1:]  # depth may be coarser than the video (iPhone LiDAR: 256x192)

    def lift(t, pts_small):
        uv = pts_small / k
        ij = np.clip(np.round(uv * [dw / w, dh / h]).astype(int), 0, [dw - 1, dh - 1])
        z = depth[t][ij[:, 1], ij[:, 0]].astype(np.float64)
        ok = (z > 0) & np.isfinite(z) & ~depth_edges(depth[t])[ij[:, 1], ij[:, 0]]
        return intr.lift(uv, z), ok

    def detect(t, m):
        pts = cv2.goodFeaturesToTrack(grays[t], 600, 0.001, 4, mask=m)
        return np.empty((0, 2), np.float32) if pts is None else pts.reshape(-1, 2)

    poses = np.full((len(grays), 4, 4), np.nan)
    poses[ref] = np.eye(4)
    lk = dict(winSize=(15, 15), maxLevel=3)
    for order in (range(ref + 1, len(grays)), range(ref - 1, -1, -1)):
        pts = detect(ref, region(ref))
        x_ref, ok = lift(ref, pts)  # each track's 3D point on the part, in the reference camera frame
        pts, x_ref = pts[ok], x_ref[ok]
        prev, pose = ref, np.eye(4)
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
            pts, x_ref = nxt[keep], x_ref[keep]
            prev = t
            if len(pts) < MIN_POINTS:
                continue
            x_t, ok = lift(t, pts)
            if ok.sum() < MIN_POINTS:
                continue
            est, inl = ransac_rigid(x_ref[ok], x_t[ok], rng)
            if est is None:
                continue
            pose = est
            poses[t] = pose
            keep = np.zeros(len(pts), bool)
            keep[np.flatnonzero(ok)[inl]] = True
            pts, x_ref = pts[keep], x_ref[keep]
            if len(pts) < 150 and len(pts) >= 3 * MIN_POINTS:
                new = detect(t, m)
                x_new, ok = lift(t, new)
                inv = np.linalg.inv(pose)
                pts = np.vstack([pts, new[ok]]).astype(np.float32)
                x_ref = np.vstack([x_ref, x_new[ok] @ inv[:3, :3].T + inv[:3, 3]])
    return poses


@dataclass
class Joint3D:
    type: str  # "revolute" or "prismatic"
    axis: np.ndarray  # unit direction, reference camera frame
    point: np.ndarray  # a point on the axis (revolute) or the child's centroid (prismatic)
    q: np.ndarray  # rad or m per frame, 0 on the reference frame
    residual: float  # m


def rotvec(r: np.ndarray) -> np.ndarray:
    v, _ = cv2.Rodrigues(r)
    return v[:, 0]


def axis_rotation(axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(n, 3, 3) rotations by ``theta`` about unit ``axis`` (Rodrigues)."""
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    s, c = np.sin(theta)[:, None, None], np.cos(theta)[:, None, None]
    return np.eye(3) + s * kx + (1 - c) * (kx @ kx)


def refine_revolute(rel: np.ndarray, points: np.ndarray, axis: np.ndarray, c: np.ndarray, theta: np.ndarray):
    """Refit a hinge to how the child's surface points move, not to its rotation matrices.

    A thin part's spin about its own long axis is barely observable and noisy, and it leaks
    into a fit on rotations; on points it only moves them by about the part's radius.
    ``rel`` (n, 4, 4) are child-in-parent motions, ``points`` (m, 3) the child on the
    reference frame. Returns axis, point on axis, per-frame angle, RMS point error (m).
    """
    if len(points) > REFINE_POINTS:
        points = points[np.random.default_rng(0).choice(len(points), REFINE_POINTS, replace=False)]
    target = points @ rel[:, :3, :3].transpose(0, 2, 1) + rel[:, None, :3, 3]  # (n, m, 3)
    n, m = len(rel), len(points)
    # Each residual depends on the 6 shared axis parameters and its own frame's angle only.
    sparsity = lil_matrix((n * m * 3 + 1, 6 + n), dtype=int)
    sparsity[:, :6] = 1
    rows = np.arange(n * m * 3)
    sparsity[rows, 6 + rows // (m * 3)] = 1

    def unpack(x):
        a = x[:3] / np.linalg.norm(x[:3])
        return a, x[3:6], x[6:]

    def residual(x):
        a, cc, th = unpack(x)
        pred = (points - cc) @ axis_rotation(a, th).transpose(0, 2, 1) + cc
        return np.concatenate([(pred - target).ravel(), [10 * (a @ (cc - c))]])  # pin c along the axis

    sol = least_squares(residual, np.concatenate([axis, c, theta]), method="trf", x_scale="jac",
                        loss="soft_l1", f_scale=0.003,  # frames with a bad part pose are outliers
                        jac_sparsity=sparsity)
    a, cc, th = unpack(sol.x)
    err = np.linalg.norm(sol.fun[:-1].reshape(n, -1, 3), axis=2)
    rms = float(np.sqrt(np.mean(np.minimum(err, 0.02) ** 2)))  # clipped so a few bad frames don't dominate
    if th.sum() < 0:
        a, th = -a, -th
    return a, cc, th, rms


def fit_joint_3d(parent: np.ndarray, child: np.ndarray, child_centroid: np.ndarray,
                 child_points: np.ndarray | None = None) -> Joint3D:
    """Fit one joint to the child's motion relative to the parent. ``child_points`` (the child's
    surface on the reference frame, camera coordinates) enables the point-based hinge refit."""
    valid = ~(np.isnan(parent).any(axis=(1, 2)) | np.isnan(child).any(axis=(1, 2)))
    rel = np.linalg.inv(parent[valid]) @ child[valid]
    r, tr = rel[:, :3, :3], rel[:, :3, 3]
    w = np.array([rotvec(x) for x in r])
    q = np.full(len(parent), np.nan)
    if np.linalg.norm(w, axis=1).max() >= MIN_ROTATION:
        axis = np.linalg.svd(w, full_matrices=False)[2][0]
        if (w @ axis).sum() < 0:
            axis = -axis
        theta = w @ axis
        # (I - R) c = t has a free direction along the axis: pin it with axis . c = axis . centroid.
        a = np.vstack([(np.eye(3) - r).reshape(-1, 3), axis[None]])
        b = np.concatenate([tr.reshape(-1), [axis @ child_centroid]])
        inliers = np.ones(len(r), bool)
        for _ in range(3):
            sel = np.concatenate([np.repeat(inliers, 3), [True]])
            c = np.linalg.lstsq(a[sel], b[sel], rcond=None)[0]
            err = np.linalg.norm((np.eye(3) - r) @ c - tr, axis=1)
            inliers = err <= max(3 * np.median(err), 1e-3)
        res = float(np.sqrt(np.mean(err[inliers] ** 2)))
        if child_points is not None and len(child_points) >= 3:
            axis, c, theta, res = refine_revolute(rel, child_points, axis, c, theta)
        q[valid] = theta
        return Joint3D("revolute", axis, c, q, res)
    centered = tr - tr.mean(axis=0)
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    d = tr @ axis
    if d[np.argmax(np.abs(d))] < 0:
        axis, d = -axis, -d
    q[valid] = d
    perp = tr - np.outer(d, axis)
    return Joint3D("prismatic", axis, child_centroid, q, float(np.sqrt(np.mean(np.sum(perp ** 2, axis=1)))))


def joint_poses(parent: np.ndarray, joint: Joint3D) -> np.ndarray:
    """The child's poses predicted by ``joint`` on top of ``parent``'s poses: a clean stand-in
    for the child's measured poses when it becomes a parent further down the chain. A thin
    link's measured pose carries spin noise that would otherwise leak into the next joint."""
    q = np.nan_to_num(joint.q)
    t = np.tile(np.eye(4), (len(q), 1, 1))
    if joint.type == "revolute":
        r = axis_rotation(joint.axis, q)
        t[:, :3, :3] = r
        t[:, :3, 3] = joint.point - r @ joint.point
    else:
        t[:, :3, 3] = q[:, None] * joint.axis
    out = parent @ t
    out[np.isnan(joint.q)] = np.nan
    return out


def track_angle(joint: Joint3D, parent: np.ndarray, model: np.ndarray, observed: dict[int, np.ndarray],
                n: int, step: float = np.deg2rad(1.0), window: float = np.deg2rad(25)) -> np.ndarray:
    """Per-frame hinge angle from depth alone, given the fitted axis.

    Where surface points are lost (a lid closing shows its back, not the screen that was
    tracked), the joint is still a single number per frame: rotate the child's reference
    cloud ``model`` about the axis and keep the angle whose points best match that frame's
    observed depth cloud (camera frame), searched near the previous frame's angle.
    ``parent`` (n, 4, 4) maps the observed cloud into the reference frame.
    """
    q = joint.q.copy()
    known = np.flatnonzero(~np.isnan(q))
    if len(known) == 0:
        return q
    order = [range(known[0] + 1, n), range(known[0] - 1, -1, -1)]
    for frames in order:
        prev = q[known[0]]
        for t in frames:
            if t not in observed or np.isnan(parent[t]).any():
                continue
            if not np.isnan(joint.q[t]):
                prev = q[t]
                continue
            inv = np.linalg.inv(parent[t])
            obs = observed[t] @ inv[:3, :3].T + inv[:3, 3]
            tree = cKDTree(obs)
            cands = prev + np.arange(-window, window + step / 2, step)
            r = axis_rotation(joint.axis, cands)
            moved = (model - joint.point) @ r.transpose(0, 2, 1) + joint.point  # (k, m, 3)
            cost = [np.median(tree.query(m_)[0]) for m_ in moved]
            q[t] = prev = cands[int(np.argmin(cost))]
    return q


def fit_plane(points: np.ndarray, rng: np.random.Generator, tol: float = 0.004, iters: int = 150):
    """RANSAC plane: unit normal n and offset d with n . x = d, and the inlier fraction."""
    if len(points) < 3:
        return np.array([0.0, 0.0, 1.0]), 0.0, 0.0
    best = None
    for _ in range(iters):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((points - a) @ n) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 3:  # degenerate (collinear) points: no plane
        return np.array([0.0, 0.0, 1.0]), 0.0, 0.0
    p = points[best]
    centroid = p.mean(axis=0)
    n = np.linalg.svd(p - centroid, full_matrices=False)[2][2]
    return n, float(n @ centroid), float(best.mean())


@dataclass
class PlanarHinge:
    axis: np.ndarray  # unit direction, reference frame
    point: np.ndarray  # a point on the axis
    q: np.ndarray  # rotation of the plate about the axis per frame (rad), 0 on the reference frame
    residual: float  # m, RMS distance of the axis from the fitted planes
    centroids: dict  # frame -> plate centroid


def fit_planar_hinge(clouds: dict[int, np.ndarray], n_frames: int, ref: int, pin: np.ndarray,
                     min_inliers: float = 0.6, base_normal: np.ndarray | None = None) -> PlanarHinge:
    """Hinge of a flat plate (a laptop lid) from depth alone, no texture needed.

    Fit the plate's plane in every frame (``clouds``, in the reference frame). A plate
    turning about a hinge keeps its normal perpendicular to the axis, and every plane passes
    (nearly) through the axis. A plane's normal has no sign, and the face seen can switch
    (screen, then the lid's back), so nothing here depends on it: the axis direction is the
    one most perpendicular to all normals, its position the line closest to all planes, and
    the angle is how far the direction from the axis to the plate's centroid has turned.
    ``pin`` fixes the free position along the axis.
    """
    rng = np.random.default_rng(0)
    planes, centroids = {}, {}
    for t, c in clouds.items():
        if len(c) >= 50:
            n, d, frac = fit_plane(c, rng)
            if frac >= min_inliers:
                planes[t] = (n, d)
                centroids[t] = c[np.abs(c @ n - d) < 0.004].mean(axis=0)
    frames = sorted(planes)
    normals = np.array([planes[t][0] for t in frames])
    offsets = np.array([planes[t][1] for t in frames])
    axis = np.linalg.eigh(normals.T @ normals)[1][:, 0]
    if base_normal is not None:
        # The hinge is where the plate meets the base: the line where their planes cross.
        # From the frames where the plate stands well clear of the base (30-150 deg) that
        # direction is sharp, and it doesn't depend on every frame's plane being right.
        crossings = []
        for n in normals:
            v = np.cross(n, base_normal)
            if np.linalg.norm(v) > np.sin(np.deg2rad(30)):
                v /= np.linalg.norm(v)
                crossings.append(v if v @ axis >= 0 else -v)
        if len(crossings) >= 5:
            axis = np.median(crossings, axis=0)
            axis /= np.linalg.norm(axis)
    a = np.vstack([normals, axis[None]])
    b = np.append(offsets, axis @ pin)
    point = np.linalg.lstsq(a, b, rcond=None)[0]
    res = normals @ point - offsets

    def radial(t):
        v = centroids[t] - point
        v -= (v @ axis) * axis
        return v / np.linalg.norm(v)

    start = min(frames, key=lambda t: abs(t - ref))
    u0 = radial(start)
    theta = np.array([np.arctan2(np.cross(u0, radial(t)) @ axis, u0 @ radial(t)) for t in frames])
    q = np.full(n_frames, np.nan)
    q[frames] = theta
    return PlanarHinge(axis, point, q, float(np.sqrt(np.mean(res ** 2))), centroids)


def clean_angles(q: np.ndarray, window: int = 15, max_dev: float = np.deg2rad(25)) -> np.ndarray:
    """Drop frames far from the running median of their neighbours: a frame where the plane
    fit caught the wrong surface (a hand, the base) jumps away and back. Works for any joint
    value (``max_dev`` in its units)."""
    out = q.copy()
    valid = np.flatnonzero(~np.isnan(q))
    for i, t in enumerate(valid):
        near = q[valid[max(0, i - window):i + window + 1]]
        if abs(q[t] - np.median(near)) > max_dev:
            out[t] = np.nan
    return out


@dataclass
class Slide:
    axis: np.ndarray  # unit direction of travel, reference frame
    point: np.ndarray  # the part's centroid on the reference frame
    q: np.ndarray  # travel per frame (m), 0 on the reference frame
    residual: float  # m, RMS distance of the travel from a straight line


def fit_slide(clouds: dict[int, np.ndarray], n_frames: int, ref: int, sample: int = 2000) -> Slide:
    """Prismatic joint from depth alone: register each frame's cloud of the part to the
    reference cloud by translation only (trimmed ICP, warm-started from the neighbouring
    frame), then take the travel's principal direction as the axis. Every face helps, and
    newly exposed surface (a drawer coming out) just finds no partner and is trimmed."""
    rng = np.random.default_rng(0)

    def thin(c):
        return c[rng.choice(len(c), sample, replace=False)] if len(c) > sample else c

    frames = sorted(t for t, c in clouds.items() if len(c) >= 50)
    start = min(frames, key=lambda t: abs(t - ref))
    target = thin(clouds[start])
    tree = cKDTree(target)
    shift = {start: np.zeros(3)}
    for seq in (frames[frames.index(start) + 1:], frames[frames.index(start) - 1::-1]):
        x = np.zeros(3)
        for t in seq:
            c = thin(clouds[t])
            for _ in range(30):
                d, idx = tree.query(c - x)
                keep = d < max(3 * np.median(d), 0.003)
                step = (c[keep] - x - target[idx[keep]]).mean(axis=0)
                x = x + step
                if np.linalg.norm(step) < 1e-5:
                    break
            shift[t] = x.copy()
    ts = np.array(sorted(shift))
    xs = np.array([shift[t] for t in ts])
    axis = np.linalg.svd(xs - xs.mean(axis=0), full_matrices=False)[2][0]
    q = xs @ axis
    if q.max() < -q.min():  # make the larger travel positive
        axis, q = -axis, -q
    out = np.full(n_frames, np.nan)
    out[ts] = q
    off = xs - np.outer(q, axis)
    return Slide(axis, clouds[start].mean(axis=0), out, float(np.sqrt(np.mean(np.sum((off - off.mean(axis=0)) ** 2, axis=1)))))


def track_rigid(clouds: dict[int, np.ndarray], n_frames: int, ref: int, sample: int = 2000) -> np.ndarray:
    """(n, 4, 4) motion of a rigid part from depth alone: each frame's cloud of the part
    (in that frame's camera coordinates) registered to the reference cloud by trimmed
    point-to-point ICP, warm-started from the neighbouring frame. Maps reference coordinates
    to frame-t coordinates; NaN where the part has too few points."""
    rng = np.random.default_rng(0)

    def thin(c):
        return c[rng.choice(len(c), sample, replace=False)] if len(c) > sample else c

    frames = sorted(t for t, c in clouds.items() if len(c) >= 50)
    start = min(frames, key=lambda t: abs(t - ref))
    target = thin(clouds[start])
    tree = cKDTree(target)
    poses = np.full((n_frames, 4, 4), np.nan)
    poses[start] = np.eye(4)
    for seq in (frames[frames.index(start) + 1:], frames[frames.index(start) - 1::-1]):
        inv = np.eye(4)  # frame-t coordinates -> reference coordinates
        for t in seq:
            c = thin(clouds[t])
            for _ in range(30):
                moved = c @ inv[:3, :3].T + inv[:3, 3]
                d, idx = tree.query(moved)
                keep = d < max(3 * np.median(d), 0.004)
                step = kabsch(moved[keep], target[idx[keep]])
                inv = step @ inv
                if np.linalg.norm(step[:3, 3]) < 1e-5 and np.allclose(step[:3, :3], np.eye(3), atol=1e-5):
                    break
            poses[t] = np.linalg.inv(inv)
    return poses
