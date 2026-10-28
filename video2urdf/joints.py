"""Joint fitting and kinematic tree from per-part 2D poses.

World frame: the motion plane is world x-z, seen by a camera on -y looking along +y, so
image u maps to +x and image v (down) to -z. The origin is the root part's centroid on
the reference frame, and ``px_per_m`` sets metric scale. Joint values are 0 on the
reference frame. Revolute axes are world -y, so positive angles turn counter-clockwise
on screen.
"""

from dataclasses import dataclass

import cv2
import numpy as np

MIN_ROTATION = np.deg2rad(5)  # less rotation range than this means prismatic
ROT_LENGTH = 0.1  # m; lever arm that turns rotation into a residual for prismatic fits


@dataclass
class Joint:
    type: str  # "revolute", "prismatic" or "fixed"
    anchor: np.ndarray  # (x, z) world point on the reference frame
    axis: np.ndarray  # (x, z) unit direction for prismatic; unused for revolute
    q: np.ndarray  # (n_frames,) joint value, NaN where untracked
    residual: float  # m, RMS misfit


@dataclass
class Link:
    name: str
    parent: str | None
    joint: Joint | None
    box: tuple[np.ndarray, np.ndarray, float]  # center (x, z), half sizes, angle (rad, CCW)
    mesh: str | None = None  # path relative to the model files; replaces the box when set
    color: tuple[float, float, float] = (0.7, 0.7, 0.7)  # RGB 0..1


def image_to_world(px_per_m: float, origin_px: tuple[float, float]) -> np.ndarray:
    u0, v0 = origin_px
    return np.array([[1 / px_per_m, 0, -u0 / px_per_m], [0, -1 / px_per_m, v0 / px_per_m], [0, 0, 1]])


def to_world(poses_img: np.ndarray, m: np.ndarray) -> np.ndarray:
    return m @ poses_img @ np.linalg.inv(m)


def fit_joint(parent: np.ndarray, child: np.ndarray, child_centroid: np.ndarray) -> Joint:
    """Fit the child's motion relative to the parent (world poses, (n, 3, 3)) as one joint."""
    valid = ~(np.isnan(parent).any(axis=(1, 2)) | np.isnan(child).any(axis=(1, 2)))
    rel = np.linalg.inv(parent[valid]) @ child[valid]
    r, tr = rel[:, :2, :2], rel[:, :2, 2]
    theta = np.unwrap(np.arctan2(r[:, 1, 0], r[:, 0, 0]))
    q = np.full(len(parent), np.nan)

    if np.ptp(theta) >= MIN_ROTATION:
        # p' = R (p - c) + c  =>  (I - R) c = t. Frames whose poses are off (occlusion
        # bends the mask) are dropped as outliers and the center is refit.
        a = np.eye(2) - r
        inliers = np.ones(len(r), bool)
        for _ in range(3):
            c = np.linalg.lstsq(a[inliers].reshape(-1, 2), tr[inliers].reshape(-1), rcond=None)[0]
            err = np.linalg.norm(a @ c - tr, axis=1)
            inliers = err <= max(3 * np.median(err), 1e-3)
        res = np.sqrt(np.mean(err[inliers] ** 2))
        q[valid] = theta
        return Joint("revolute", c, np.zeros(2), q, float(res))

    centered = tr - tr.mean(axis=0)
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    if axis @ (tr[-1] - tr[0]) < 0:
        axis = -axis
    d = tr @ axis
    perp = tr - np.outer(d, axis)
    res = np.sqrt(np.mean(np.sum(perp ** 2, axis=1)) + np.mean(theta ** 2) * ROT_LENGTH ** 2)
    q[valid] = d
    return Joint("prismatic", child_centroid, axis, q, float(res))


def mask_box(mask: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Minimum-area rectangle of a full-frame mask, in world units."""
    pts = np.argwhere(mask)[:, ::-1].astype(np.float64)  # (u, v)
    pts_w = (m[:2, :2] @ pts.T).T + m[:2, 2]
    (cx, cz), (bw, bh), deg = cv2.minAreaRect(pts_w.astype(np.float32))
    return np.array([cx, cz]), np.array([bw / 2, bh / 2]), np.deg2rad(deg)


def box_distance(point: np.ndarray, box: tuple[np.ndarray, np.ndarray, float]) -> float:
    """Distance from a world point to a link's box (0 inside)."""
    center, half, angle = box
    c, s = np.cos(angle), np.sin(angle)
    local = np.array([[c, s], [-s, c]]) @ (point - center)
    return float(np.linalg.norm(np.maximum(np.abs(local) - half, 0)))


def joint_cost(joint: Joint, parent_box, child_box) -> float:
    """Fit residual plus how far a hinge sits from the two parts it joins. A pair that
    barely moves relative to each other fits any parent equally well, and the hinge has to
    touch both parts to be physical."""
    if joint.type != "revolute":
        return joint.residual
    return joint.residual + box_distance(joint.anchor, parent_box) + box_distance(joint.anchor, child_box)


def build_tree(names: list[str], poses_w: dict[str, np.ndarray], centroids: dict[str, np.ndarray],
               boxes: dict) -> list[Link]:
    """Prim's tree from the first part (the root), joining at each step the pair with the
    lowest joint_cost. Parts not joined directly move as a chain of joints, which one joint
    fits badly."""
    links = [Link(names[0], None, None, boxes[names[0]])]
    done = {names[0]}
    costs, fits = {}, {}
    while len(done) < len(names):
        for p in done:
            for c in names:
                if c not in done and (p, c) not in fits:
                    fits[p, c] = fit_joint(poses_w[p], poses_w[c], centroids[c])
                    costs[p, c] = joint_cost(fits[p, c], boxes[p], boxes[c])
        p, c = min(((p, c) for (p, c) in costs if p in done and c not in done), key=costs.get)
        links.append(Link(c, p, fits[p, c], boxes[c]))
        done.add(c)
    return links
