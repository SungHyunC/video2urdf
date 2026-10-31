"""Amodal completion: the hidden body of a partly seen object, measured where seen, bounded by
the video, estimated otherwise.

A door model (lidar.build_door_model) keeps only the body parts SAM tracked. For a countertop
microwave filmed from the front that is a strip of its control panel: the model shows a door
and a floating panel, and a robot reaches straight through where the oven is. This step
completes the body:

1. Measured where seen: the front plane from the shut door's and the body's depth points; the
   whole object's front (width, top edge, down to the floor) from depth points inside a SAM 3.1
   mask of the whole object on shut frames; depth points on its top or side faces, if any, give
   the least depth the body has.
2. Bounded by the video: a body d deep behind the measured front, projected into frames where
   the object is in view, must stay inside the (dilated) whole-object mask. The largest such d
   bounds the hidden depth from above; the smallest d whose silhouette reaches the far edges of
   the (eroded) mask bounds it from below. Loose where the camera never saw past the front.
3. Estimated otherwise: Muse Spark (the agent's backend, so Muse Spark Open in an open-weights
   run) watches the video and gives the hidden sizes typical for the object (or its exact model
   if it recognises it): depth, the cavity behind the door, wall thickness, mass; each with a
   range and a reason. A size it doesn't give falls back to a category default. Every estimate
   is clamped to the measurements and the video's bound, and labelled as an estimate.
4. Built: a hollow carcass of boxes (each convex, so it collides as built): a pocket the shut
   door sits in, the cavity behind it, the column beside the door flush with the front. The
   door, its hinge and the measured parts are unchanged; the door is checked to sweep its whole
   joint range without entering the carcass.

completion.json records what was measured, bounded and estimated (and by which model), and the
lines a viewer's "Not measured" note shows.
"""

import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SHUT_TOL_DEG = 4.0  # a door this close to its shut angle (and no open indicator) is shut
PLANE_TOL_M = 0.008  # RANSAC inlier distance for the front plane (black glass is noisy)
FRONT_TOL_M = 0.015  # depth points this close to the front plane are on the front
BEHIND_M = 0.02  # points this far behind the front plane, inside the object's mask, are its top/sides
DEPTH_STEP_M = 0.01  # body depths tried for the silhouette bound
MAX_DEPTH_FACTOR = 2.5  # ...up to this times the front's larger side
EDGE_PX_FRAC = 0.004  # mask dilation/erosion, as a fraction of the image diagonal (~10 px at 1920x1440)
MIN_STEP_PX = 150  # a depth step must add this many pixels to the silhouette to be judged
MIN_SEEN_PX = 3000  # ...and the whole range this many, or the frame says nothing about depth
MARGIN_FRAC = 0.01  # an object mask this close to the image border is cut by it
N_SHUT_FRAMES, N_OPEN_FRAMES = 6, 3  # SAM whole-object masks: shut frames bound, open ones only score
DOOR_GAP_M = 0.003  # clearance between the door and the carcass
MIN_WALL_M = 0.005
MIN_CAVITY_M = 0.05
FRAME_MARGIN_M = 0.01  # the cavity's opening stays this far inside the door's outline
STOPWORDS = {"open", "opened", "closed", "shut", "the", "a", "an", "of", "and", "with", "white", "black", "silver",
             "grey", "gray", "metal", "plastic", "glass", "wooden", "wood", "front", "side", "top", "back", "left",
             "right", "inner", "outer", "interior", "surface", "part", "panel", "door", "lid", "cavity", "control",
             "handle", "edge", "frame", "base", "table", "tabletop", "shelf", "floor", "desk", "counter"}

# Typical hidden sizes per category, used only when Muse Spark gives no usable value. Ratios are to
# the measured front's width; walls in metres. Rough, from common product sizes: estimates.
CATEGORY_DEFAULTS = {
    "microwave": {"depth_ratio": 0.78, "wall": 0.025, "back_wall": 0.04, "frame": 0.025, "density": 230,
                  "reason": "countertop microwaves are usually about 0.75-0.85 times as deep as wide, with a "
                            "metal cavity inside a case 2-4 cm larger"},
    "oven": {"depth_ratio": 0.95, "wall": 0.04, "back_wall": 0.05, "frame": 0.03, "density": 250,
             "reason": "ovens are about as deep as wide with insulated walls"},
    "refrigerator": {"depth_ratio": 1.1, "wall": 0.05, "back_wall": 0.06, "frame": 0.03, "density": 120,
                     "reason": "fridges are slightly deeper than wide with ~5 cm insulated walls"},
    "dishwasher": {"depth_ratio": 0.95, "wall": 0.03, "back_wall": 0.04, "frame": 0.02, "density": 150,
                   "reason": "dishwashers are about 57 cm deep and 60 cm wide"},
    "washing machine": {"depth_ratio": 0.9, "wall": 0.03, "back_wall": 0.05, "frame": 0.03, "density": 250,
                        "reason": "front-loaders are about 55-60 cm deep and 60 cm wide"},
    "safe": {"depth_ratio": 0.9, "wall": 0.04, "back_wall": 0.04, "frame": 0.02, "density": 1500,
             "reason": "small safes are about as deep as wide with thick steel walls"},
    "cabinet": {"depth_ratio": 0.8, "wall": 0.018, "back_wall": 0.008, "frame": 0.0, "density": 150,
                "reason": "cupboards are built from ~18 mm boards with a thin back panel"},
    "box": {"depth_ratio": 0.8, "wall": 0.01, "back_wall": 0.01, "frame": 0.0, "density": 200,
            "reason": "boxes are thin-walled"},
}
GENERIC_DEFAULT = {"depth_ratio": 0.8, "wall": 0.02, "back_wall": 0.02, "frame": 0.01, "density": 300,
                   "reason": "an object with a door is usually about as deep as it is wide"}
CATEGORY_WORDS = {"microwave": ("microwave",), "oven": ("oven", "toaster"), "refrigerator": ("fridge", "refrigerator", "freezer"),
                  "dishwasher": ("dishwasher",), "washing machine": ("washing", "washer", "dryer"), "safe": ("safe",),
                  "cabinet": ("cabinet", "cupboard", "wardrobe", "closet", "locker", "nightstand", "sideboard"),
                  "box": ("box", "chest", "case", "crate")}


def category_of(name: str | None) -> tuple[str | None, dict]:
    """The defaults row for an object name ("microwave oven" -> microwave), else the generic one."""
    words = (name or "").lower()
    for cat, keys in CATEGORY_WORDS.items():  # dict order: microwave before oven
        if any(k in words for k in keys):
            return cat, CATEGORY_DEFAULTS[cat]
    return None, GENERIC_DEFAULT


# ----------------------------------------------------------------------------------------------
# The run: tracks, the replay's cameras (camera -> MuJoCo world) and the door's joint values


@dataclass
class Run:
    dir: Path
    video: Path
    names: list
    by: dict  # part -> frame -> sam.Detection (usable masks only)
    roles: dict
    cam_to_world: np.ndarray  # (n, 4, 4): OpenCV camera axes -> the model's world (floor at z = 0)
    intr: object  # lift3d.Intrinsics
    q: np.ndarray  # the door's joint value per frame in the model's sense, NaN where unmeasured
    root_pos: np.ndarray  # the root body's position in the world: model frame = world - root_pos
    size: tuple  # (w, h)
    joint: str = "lid_joint"
    base: str = "base"
    door: str = "lid"

    @property
    def n(self) -> int:
        return len(self.cam_to_world)

    def has(self, role: str, t: int) -> bool:
        part = self.roles.get(role)
        return bool(part) and part in self.by and t in self.by[part]

    def mask(self, role: str, t: int) -> np.ndarray:
        w, h = self.size
        return self.by[self.roles[role]][t].full_mask(h, w)

    def object_mask(self, t: int) -> np.ndarray:
        """The union of the tracked object parts on frame t (not the floor)."""
        w, h = self.size
        m = np.zeros((h, w), bool)
        for role in ("base", "plate", "plate_closed", "open_indicator"):
            if self.has(role, t):
                m |= self.mask(role, t)
        return m

    def world_points(self, t: int, mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Depth points inside a mask on frame t, in the model frame."""
        from .lift3d import mask_cloud
        c = mask_cloud(mask, depth[t], self.intr)
        return c @ self.cam_to_world[t][:3, :3].T + self.cam_to_world[t][:3, 3] - self.root_pos

    def project(self, t: int, points_model: np.ndarray) -> np.ndarray:
        """Model-frame points -> pixels (u, v) and camera depth on frame t, as (N, 3)."""
        w2c = np.linalg.inv(self.cam_to_world[t])
        c = (points_model + self.root_pos) @ w2c[:3, :3].T + w2c[:3, 3]
        z = c[:, 2]
        safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
        return np.stack([self.intr.f * c[:, 0] / safe + self.intr.cx, self.intr.f * c[:, 1] / safe + self.intr.cy, z], 1)


def load_run(run: Path, video: Path) -> Run:
    from .lidar import read_hinge
    from .lift3d import Intrinsics
    from .pipeline import load_parts
    names, by = load_parts(run)
    cap = cv2.VideoCapture(str(video))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    by = {k: {t: d for t, d in v.items() if d.usable(*size)} for k, v in by.items()}
    replay = np.load(run / "replay3d.npz")
    m3 = json.loads((run / "model3d.json").read_text())
    q = np.array([np.nan if v is None else v for v in read_hinge(run)["q"]], float)
    n = len(replay["cam_to_world"])
    q = np.concatenate([q, np.full(max(0, n - len(q)), np.nan)])[:n]
    return Run(run, video, names, by, json.loads((run / "roles.json").read_text()), replay["cam_to_world"],
               Intrinsics(float(replay["f"]), float(replay["cx"]), float(replay["cy"])), q,
               np.array(m3["camera_to_model"]["root_pos"], float), size)


# ----------------------------------------------------------------------------------------------
# When to complete


def needs_completion(run: Path) -> tuple[bool, str]:
    """Whether a run's body is known to be partial: a door model (vertical hinge) whose base is a
    fragment, i.e. its measured solid is far shallower behind the front than the front is wide,
    or a completion.json / complete_body flag already says so. Reads files only."""
    m3_path = run / "model3d.json"
    if (run / "complete_body.flag").exists():
        return True, "flagged by complete_body.flag"
    if not m3_path.exists():
        return False, "no 3D model (model3d.json)"
    m3 = json.loads(m3_path.read_text())
    if not m3.get("hinge_vertical"):
        return False, "not a door model (no vertical hinge): its body is measured as fitted"
    if (run / "completion.json").exists():
        return True, "already completed (completion.json): redone from the measured-only model"
    from .texture import read_stl
    base_mesh, door_mesh = run / "meshes" / "base.stl", run / "meshes" / "lid.stl"
    if not base_mesh.exists() or not door_mesh.exists():
        return False, "door model without meshes/base.stl and meshes/lid.stl"
    door = read_stl(door_mesh).reshape(-1, 3)
    base = read_stl(base_mesh).reshape(-1, 3)
    # the door's plane at the reference pose: its thin direction; the body's depth along it
    c = door - door.mean(axis=0)
    normal = np.linalg.svd(c, full_matrices=False)[2][2]
    normal[2] = 0
    normal /= max(np.linalg.norm(normal), 1e-9)
    width = float(np.ptp(door[:, :2] @ np.array([-normal[1], normal[0]])))
    depth = float(np.ptp(base @ normal))
    if depth < 0.25 * width:
        return True, (f"door model whose body is a fragment: it reaches {1000 * depth:.0f} mm behind the front, "
                      f"the door is {1000 * width:.0f} mm wide")
    return False, f"door model with a body {1000 * depth:.0f} mm deep (door {1000 * width:.0f} mm wide): not partial"


# ----------------------------------------------------------------------------------------------
# 1. Measure what is seen


def shut_state(q: np.ndarray, open_frames, tol_deg: float = SHUT_TOL_DEG) -> tuple[float, np.ndarray]:
    """The door's shut joint value and the frames it is shut on: no open indicator, and within
    tol_deg of the shut value. The shut value is the end of the indicator-free frames' range
    away from the open frames (or, without open frames, their median)."""
    q = np.asarray(q, float)
    open_frames = set(int(t) for t in open_frames)
    closed = np.array([t for t in range(len(q)) if t not in open_frames and np.isfinite(q[t])], int)
    if not len(closed):
        raise ValueError("no frame without the open indicator has a measured door angle")
    opened = np.array([t for t in open_frames if t < len(q) and np.isfinite(q[t])], int)
    if len(opened):
        sign = 1.0 if np.median(q[closed]) > np.median(q[opened]) else -1.0
        q_shut = float(sign * np.percentile(sign * q[closed], 95))
    else:
        q_shut = float(np.median(q[closed]))
    shut = closed[np.abs(q[closed] - q_shut) <= np.deg2rad(tol_deg)]
    return q_shut, shut


def fit_vertical_plane(points: np.ndarray, rng, tol: float = PLANE_TOL_M, iters: int = 400):
    """RANSAC for a vertical plane (normal perpendicular to the model's z): a line in the points'
    horizontal projection. Returns the unit normal, the offset (n . p = d) and the inliers."""
    xy = points[:, :2]
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(xy), 2, replace=False)
        t = xy[j] - xy[i]
        if np.linalg.norm(t) < 0.02:
            continue
        nrm = np.array([-t[1], t[0]]) / np.linalg.norm(t)
        inl = np.abs((xy - xy[i]) @ nrm) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 10:
        raise ValueError("no vertical plane in the front's depth points")
    p = xy[best]
    c = p.mean(axis=0)
    t = np.linalg.svd(p - c, full_matrices=False)[2][0]
    nrm = np.array([-t[1], t[0], 0.0])
    inl = np.abs(points @ nrm - c @ nrm[:2]) < tol
    return nrm, float(c @ nrm[:2]), inl


@dataclass
class Front:
    """The object's front in the model frame: ``n`` points out of the body (toward the cameras),
    ``u`` = z x n runs along it, ``z`` is up; coordinates (u, v, z) with v the depth into the
    body (v = 0 on the front plane); the origin is the hinge's foot on the plane."""
    origin: np.ndarray
    n: np.ndarray
    z: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    @property
    def u(self) -> np.ndarray:
        return np.cross(self.z, self.n)

    @property
    def yaw(self) -> float:
        """Rotation about z taking the model's x axis to u (the front frame's axes are u, -n, z)."""
        return float(np.arctan2(self.u[1], self.u[0]))

    def uvz(self, p: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(p) - self.origin
        return np.stack([d @ self.u, -(d @ self.n), d @ self.z], axis=1)

    def xyz(self, uvz: np.ndarray) -> np.ndarray:
        a = np.atleast_2d(uvz)
        return self.origin + a[:, :1] * self.u - a[:, 1:2] * self.n + a[:, 2:3] * self.z


def box_corners(front: Front, lo, hi) -> np.ndarray:
    """The 8 corners (model frame) of a box given in front coordinates (u, v, z) by its low and
    high corners."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[a, b, d] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for d in (lo[2], hi[2])])
    return front.xyz(c)


# ----------------------------------------------------------------------------------------------
# 2. Bound the hidden depth with the video's silhouettes


def silhouette(points_px: np.ndarray, size) -> np.ndarray | None:
    """Filled convex hull of projected points (pixels, camera depth); None if any is behind the camera."""
    if (points_px[:, 2] < 0.02).any():
        return None
    w, h = size
    m = np.zeros((h, w), np.uint8)
    hull = cv2.convexHull(np.round(points_px[:, :2]).astype(np.int32))
    cv2.fillConvexPoly(m, hull, 1)
    return m.astype(bool)


def frame_depth_bounds(corners_at, project, mask: np.ndarray, depths: np.ndarray, edge_px: int) -> dict:
    """How deep a body behind the front can be on one frame. ``corners_at(d)`` gives the body's
    corners (model frame) d deep, ``project`` maps them to pixels and camera depth, ``mask`` is
    the whole object's mask. Each depth step adds pixels to the body's silhouette: while the body
    is shallower than the object those pixels land on the object's top/sides (inside the mask),
    once deeper they land outside it. Upper bound: the last depth whose added pixels are mostly
    inside the dilated mask; lower bound: the last depth whose added pixels are still mostly
    inside the eroded mask (from the start). None where steps add too few pixels to tell."""
    h, w = mask.shape
    k = np.ones((2 * edge_px + 1, 2 * edge_px + 1), np.uint8)
    dil = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    ero = cv2.erode(mask.astype(np.uint8), k).astype(bool)
    prev, rows = None, []
    for d in depths:
        s = silhouette(project(corners_at(d)), (w, h))
        if s is None:
            break
        if prev is not None:
            new = s & ~prev
            px = int(new.sum())
            rows.append((float(d), px, int((new & ~dil).sum()), int((new & ero).sum())))
        prev = s
    out = {"steps": rows, "upper_m": None, "lower_m": None, "seen_px": int(sum(r[1] for r in rows))}
    if out["seen_px"] < MIN_SEEN_PX:
        return out
    judged = [r for r in rows if r[1] >= MIN_STEP_PX]
    upper = None
    for i, (d, px, outside, _) in enumerate(judged):
        if outside > 0.5 * px:
            upper = judged[i - 1][0] if i else float(depths[0])
            break
    out["upper_m"] = upper  # None: never left the mask within the depths tried
    lower = float(depths[0])
    for d, px, _, inside in judged:
        if inside < 0.5 * px:
            break
        lower = d
    out["lower_m"] = lower
    return out


def combine_bounds(per_frame: dict, seen_depth: float | None = None) -> dict:
    """The depth bound over frames: the largest depth every usable frame allows (the smallest
    upper bound), and a lower bound from the frames that agree: the most frames whose brackets
    share a depth, if they are at least two thirds of them, give the largest of their lower
    bounds (frames outside that consensus are listed). A frame whose upper bound is below the
    depth already seen on the object (depth points behind its front) or below its own lower bound
    is inconsistent (its mask misses the object's top or sides) and is not used. Without a
    consensus, or if it contradicts the upper bound, the silhouettes give no lower bound (a top
    seen at a grazing angle: millimetres of height error per frame move its far edge by many
    centimetres); the depth points' least depth stays the lower bound."""
    used, dropped = {}, {}
    for t, b in per_frame.items():
        if b.get("upper_m") is None and b.get("lower_m") is None:
            dropped[t] = "too few pixels change with depth (the camera sees no top or side face)"
        elif b.get("upper_m") is not None and seen_depth is not None and b["upper_m"] < seen_depth:
            dropped[t] = f"its upper bound {1000 * b['upper_m']:.0f} mm is below the depth seen on the object"
        elif b.get("upper_m") is not None and b.get("lower_m") is not None and b["upper_m"] < b["lower_m"]:
            dropped[t] = "upper bound below its own lower bound"
        elif b.get("upper_m") is not None and b["upper_m"] < 2 * DEPTH_STEP_M:
            dropped[t] = "its mask shows no top or side face (the body leaves it at once)"
        else:
            used[t] = b
    uppers = {t: b["upper_m"] for t, b in used.items() if b.get("upper_m") is not None}
    lowers = {t: b["lower_m"] for t, b in used.items() if b.get("lower_m") is not None}
    out = {"upper_m": None, "upper_frames": [], "lower_m": None, "lower_frames": [], "used_frames": sorted(used),
           "dropped_frames": {str(k): v for k, v in sorted(dropped.items())},
           "per_frame": {str(t): {"upper_m": b.get("upper_m"), "lower_m": b.get("lower_m"), "seen_px": b.get("seen_px")}
                         for t, b in sorted(per_frame.items())}}
    if uppers:
        out["upper_m"] = min(uppers.values())
        out["upper_frames"] = sorted(t for t, v in uppers.items() if v <= out["upper_m"] + 1e-9)
    if lowers:
        hi = {t: uppers.get(t, np.inf) for t in lowers}
        best, agree = None, []
        for d in sorted(set(lowers.values()) | {v for v in hi.values() if np.isfinite(v)}):
            inside = [t for t in lowers if lowers[t] - 1e-9 <= d <= hi[t] + 1e-9]
            if len(inside) > len(agree):
                best, agree = d, inside
        out["consensus_frames"] = sorted(agree)
        out["outside_consensus"] = sorted(set(lowers) - set(agree))
        lower = max(lowers[t] for t in agree) if agree else None
        out["frames_agree"] = (lower is not None and len(agree) >= max(2, int(np.ceil(2 * len(lowers) / 3)))
                               and (out["upper_m"] is None or lower <= out["upper_m"] + 1e-9))
        out["silhouette_lower_m"] = lower
        if out["frames_agree"]:
            out["lower_m"] = lower
            out["lower_frames"] = sorted(t for t in agree if lowers[t] >= lower - 1e-9)
    if seen_depth is not None and (out["lower_m"] is None or seen_depth > out["lower_m"]):
        out["lower_m"], out["lower_frames"], out["lower_from_depth_points"] = seen_depth, [], True
    return out


# ----------------------------------------------------------------------------------------------
# SAM 3.1: the whole object's mask on a few frames (the tracked parts may be fragments of it)


def object_prompt(run_dir: Path) -> str | None:
    """A noun for the whole object: the word the tracked parts' prompts share ("microwave" from
    "microwave control panel" and "open microwave door"), else the object design.json names."""
    parts = json.loads((run_dir / "parts.json").read_text())
    roles = json.loads((run_dir / "roles.json").read_text()) if (run_dir / "roles.json").exists() else {}
    floor = roles.get("floor")
    counts: dict[str, int] = {}
    for p in parts.values():
        if p.get("name") == floor:
            continue
        for word in set(str(p.get("prompt", "")).lower().replace("-", " ").split()):
            if word.isalpha() and word not in STOPWORDS:
                counts[word] = counts.get(word, 0) + 1
    shared = [w for w, c in sorted(counts.items(), key=lambda kv: -kv[1]) if c >= 2]
    if shared:
        return shared[0]
    if (run_dir / "design.json").exists():
        name = json.loads((run_dir / "design.json").read_text()).get("object")
        if name:
            return str(name).split(",")[0].split(" with ")[0].strip()
    return None


def spread_frames(frames, positions: np.ndarray, k: int, first: int | None = None) -> list[int]:
    """Up to k frames whose cameras are far apart (farthest-point sampling), starting from ``first``."""
    frames = list(frames)
    if not frames:
        return []
    chosen = [first if first in frames else frames[len(frames) // 2]]
    while len(chosen) < min(k, len(frames)):
        far = max((t for t in frames if t not in chosen),
                  key=lambda t: min(np.linalg.norm(positions[t] - positions[c]) + 1e-4 * abs(t - c) for c in chosen))
        chosen.append(far)
    return sorted(chosen)


def mask_frames(run: Run, shut: np.ndarray) -> tuple[list[int], list[int]]:
    """Shut frames to bound the depth on (the tracked parts well inside the image; the first is
    the one with the highest camera, which sees most of a top face), and open frames to score on."""
    w, h = run.size
    margin = MARGIN_FRAC * max(w, h)

    def inside(t):
        ys, xs = np.nonzero(run.object_mask(t))
        return len(xs) and xs.min() > margin and ys.min() > margin and xs.max() < w - margin and ys.max() < h - margin

    step = max(1, len(shut) // 120)
    candidates = [int(t) for t in shut[::step] if inside(int(t))] or [int(t) for t in shut[::step]]
    pos = run.cam_to_world[:, :3, 3]
    highest = max(candidates, key=lambda t: pos[t][2]) if candidates else None
    shut_pick = spread_frames(candidates, pos, N_SHUT_FRAMES, highest)
    opened = [t for t in sorted(run.by.get(run.roles.get("open_indicator") or "", {})) if np.isfinite(run.q[t])]
    open_pick = []
    if opened:
        wide = np.abs(run.q[opened] - np.nanmedian(run.q))
        far = [t for t, a in zip(opened, wide) if a >= np.percentile(wide, 50)]
        open_pick = spread_frames(far[:: max(1, len(far) // 120)], pos, N_OPEN_FRAMES)
    return shut_pick, open_pick


def whole_object_masks(run: Run, frames: list[int], prompt: str, cache: Path, client=None, log=print) -> dict:
    """SAM 3.1's mask of the whole object on each frame (one concept-prompt image call per frame,
    cached in ``cache``): of its detections, the one that holds the tracked parts' union best,
    if it holds most of it and isn't far larger (not the shelf or the room)."""
    from .pipeline import read_frame
    from .sam import parse
    w, h = run.size
    saved = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["prompt"]) == prompt:
            for i, t in enumerate(z["frames"]):
                saved[int(t)] = np.unpackbits(z["bits"][i])[: h * w].reshape(h, w).astype(bool)
    out, info = {}, {}
    for t in frames:
        if t in saved:
            out[t], info[t] = saved[t], "cached"
            continue
        if client is None:
            from .sam import sam_client
            client = sam_client()
        dets = parse(client.segment_image(read_frame(run.video, t), prompt))
        union = run.object_mask(t)
        best, best_hold = None, 0.0
        for d in dets:
            m = d.full_mask(h, w)
            hold = (m & union).sum() / max(1, union.sum())
            if hold > best_hold and m.sum() <= 4 * max(1, union.sum()):
                best, best_hold = m, hold
        if best is None or best_hold < 0.8:
            info[t] = f"rejected: {len(dets)} detections, the best holds {100 * best_hold:.0f}% of the tracked parts"
        else:
            out[t] = best
            info[t] = f"holds {100 * best_hold:.0f}% of the tracked parts, {best.sum() / max(1, union.sum()):.2f}x their area"
        log(f"SAM '{prompt}' frame {t}: {info[t]}")
    keep = sorted(out)
    if keep and set(keep) != set(saved):
        np.savez_compressed(cache, prompt=np.array(prompt), frames=np.array(keep),
                            bits=np.stack([np.packbits(out[t].ravel()) for t in keep]))
    return {"masks": out, "info": {str(k): v for k, v in info.items()}}


# ----------------------------------------------------------------------------------------------
# Per-frame levelling. The replay's camera poses come from ARKit, levelled against the floor track
# where it is seen; elsewhere they can tilt several degrees (the microwave's last 1000 frames: 4-9
# deg), which moves a projected body by tens of pixels. Measuring and bounding use each frame's own
# support surface (horizontal, at the floor's height) and the object's own front (the pooled plane).


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector a onto b."""
    v, c = np.cross(a, b), float(np.clip(a @ b, -1, 1))
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    ang = np.arctan2(s, c)
    return np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * k @ k


def _about(r: np.ndarray, pivot: np.ndarray, shift=(0.0, 0.0, 0.0)) -> np.ndarray:
    """4x4: rotate by r about pivot, then shift."""
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = pivot - r @ pivot + np.asarray(shift, float)
    return m


def _apply(m: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ m[:3, :3].T + m[:3, 3]


def support_correction(points: np.ndarray, z_floor: float, rng, max_tilt_deg: float = 15.0) -> tuple[np.ndarray, dict]:
    """A frame's support surface (points near the floor's height, in front of the object) made
    horizontal at the floor's height: the 4x4 correction (model frame) and what it did."""
    from .lift3d import fit_plane
    if len(points) < 300:
        return np.eye(4), {"support_points": int(len(points)), "levelled": False}
    n, _, frac = fit_plane(points, rng)
    if n[2] < 0:
        n = -n
    tilt = float(np.rad2deg(np.arccos(np.clip(n[2], -1, 1))))
    if tilt > max_tilt_deg or frac < 0.3:
        return np.eye(4), {"support_points": int(len(points)), "levelled": False, "tilt_deg": tilt}
    on = points[np.abs((points - points.mean(axis=0)) @ n) < 0.006]
    pivot = on.mean(axis=0)
    r = _rotation_between(n, np.array([0.0, 0.0, 1.0]))
    m = _about(r, pivot)
    height = float(np.median(_apply(m, on)[:, 2]))
    m[2, 3] += z_floor - height
    return m, {"support_points": int(len(points)), "levelled": True, "tilt_deg": tilt,
               "height_shift_mm": 1000 * (z_floor - height)}


def front_correction(points: np.ndarray, front: "Front", rng, yaw_only: bool = False) -> tuple[np.ndarray, dict]:
    """A frame's own front points turned onto the pooled (vertical) front plane and shifted along
    its normal onto it. The front decides the frame's pitch: a large, near, well-measured face,
    where the support in front is seen at a grazing angle (``yaw_only``: the support decided the
    pitch, the front only turns the frame about the vertical)."""
    from .lift3d import fit_plane
    if len(points) < 200:
        return np.eye(4), {"front_points": int(len(points)), "aligned": False}
    n, _, _ = fit_plane(points, rng)
    if n @ front.n < 0:
        n = -n
    lean = float(np.rad2deg(np.arcsin(np.clip(n[2], -1, 1))))
    if yaw_only:
        n = n.copy()
        n[2] = 0
        n /= max(np.linalg.norm(n), 1e-9)
    turn = float(np.rad2deg(np.arccos(np.clip(n @ front.n, -1, 1))))
    if turn > 15:
        return np.eye(4), {"front_points": int(len(points)), "aligned": False, "turn_deg": turn}
    m = _about(_rotation_between(n, front.n), points.mean(axis=0))
    v = float(np.median(front.uvz(_apply(m, points))[:, 1]))
    m[:3, 3] += v * front.n  # the front's median depth onto v = 0
    return m, {"front_points": int(len(points)), "aligned": True, "turn_deg": turn, "lean_deg": lean, "shift_mm": 1000 * v}


def support_roll(points: np.ndarray, front: "Front", z_floor: float, rng) -> tuple[np.ndarray, dict]:
    """With the front fixed, the support surface levelled side to side (a turn about the front's
    normal, which keeps the front where it is) and set at the floor's height."""
    from .lift3d import fit_plane
    if len(points) < 300:
        return np.eye(4), {"roll_points": int(len(points))}
    n, _, frac = fit_plane(points, rng)
    if n[2] < 0:
        n = -n
    if frac < 0.3 or n[2] < np.cos(np.deg2rad(15)):
        return np.eye(4), {"roll_points": int(len(points))}
    roll = float(np.arctan2(n @ front.u, n @ front.z))
    on = points[np.abs((points - points.mean(axis=0)) @ n) < 0.006]
    a = front.n  # turning about +n by +roll takes the normal's u-component back to z
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    r = np.eye(3) + np.sin(roll) * k + (1 - np.cos(roll)) * k @ k
    m = _about(r, on.mean(axis=0))
    if abs((r @ n) @ front.u) > abs(n @ front.u):  # wrong way round: turn the other way
        r = r.T
        m = _about(r, on.mean(axis=0))
    height = float(np.median(_apply(m, on)[:, 2]))
    m[2, 3] += z_floor - height
    return m, {"roll_points": int(len(points)), "roll_deg": float(np.rad2deg(roll)),
               "pitch_left_deg": float(np.rad2deg(np.arcsin(np.clip(n @ front.n, -1, 1)))),
               "floor_shift_mm": 1000 * (z_floor - height)}


@dataclass
class FrameView:
    """One frame's camera (OpenCV axes -> model frame), levelled for this frame, and its points."""
    t: int
    cam_to_model: np.ndarray
    info: dict

    def project(self, intr, p: np.ndarray) -> np.ndarray:
        w2c = np.linalg.inv(self.cam_to_model)
        c = p @ w2c[:3, :3].T + w2c[:3, 3]
        z = np.where(np.abs(c[:, 2]) < 1e-9, 1e-9, c[:, 2])
        return np.stack([intr.f * c[:, 0] / z + intr.cx, intr.f * c[:, 1] / z + intr.cy, c[:, 2]], 1)

    def points(self, intr, mask: np.ndarray, depth_t: np.ndarray) -> np.ndarray:
        from .lift3d import mask_cloud
        c = mask_cloud(mask, depth_t, intr)
        return c @ self.cam_to_model[:3, :3].T + self.cam_to_model[:3, 3]


# ----------------------------------------------------------------------------------------------
# 1. (continued) the front, its extent and what is seen behind it


def _front_mask(run: Run, t: int) -> np.ndarray:
    m = np.zeros(run.size[::-1], bool)
    for role in ("plate_closed", "plate", "base"):
        if run.has(role, t):
            m |= run.mask(role, t)
    return m


def _pooled_front(run: Run, views: dict, depth: np.ndarray, rng) -> tuple["Front", float]:
    pts = np.vstack([v.points(run.intr, _front_mask(run, t), depth[t]) for t, v in views.items()])
    if len(pts) > 60000:
        pts = pts[rng.choice(len(pts), 60000, replace=False)]
    n, off, inl = fit_vertical_plane(pts, rng)
    cams = np.array([v.cam_to_model[:3, 3] for v in views.values()])
    if (cams.mean(axis=0) @ n - off) < 0:  # out of the body, toward the phone
        n, off = -n, -off
    origin = off * n  # the hinge (the model's origin) projected onto the plane, along n
    origin[2] = 0.0
    return Front(origin, n), float(inl.mean())


def level_views(run: Run, frames, depth: np.ndarray, z_floor: float, rng, object_masks: dict | None = None,
                log=print, pitch_from: str = "front") -> tuple[dict, "Front", float]:
    """Each frame's camera levelled: first roughly on its own support surface, then its own front
    turned and shifted onto the pooled (vertical) front plane, then the support levelled side to
    side and set at the floor's height. ``pitch_from="support"`` keeps the support's pitch and
    turns the front about the vertical only (the two disagree by a few degrees on the microwave,
    which is how uncertain the levelling is). Returns the views, the front and the share of front
    points on its plane."""
    base = {}
    for t in frames:
        c2m = run.cam_to_world[t].copy()
        c2m[:3, 3] -= run.root_pos
        base[t] = FrameView(t, c2m, {})
    front, _ = _pooled_front(run, base, depth, rng)  # rough, only to find the support in front
    everything = np.ones(run.size[::-1], bool)
    lo, hi = np.inf, -np.inf
    for t, v in base.items():
        p = front.uvz(v.points(run.intr, _front_mask(run, t), depth[t]))
        if len(p):
            lo, hi = min(lo, np.percentile(p[:, 0], 2)), max(hi, np.percentile(p[:, 0], 98))

    def support(v, t, front):
        mask = (object_masks or {}).get(t, run.object_mask(t))
        p = v.points(run.intr, everything & ~cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool),
                     depth[t])
        f = front.uvz(p)
        return p[(f[:, 1] < -0.01) & (f[:, 1] > -0.35) & (f[:, 0] > lo - 0.15) & (f[:, 0] < hi + 0.15)
                 & (np.abs(f[:, 2] - z_floor) < 0.06)]

    views = {}
    for t, v in base.items():
        m, info = support_correction(support(v, t, front), z_floor, rng)
        views[t] = FrameView(t, m @ v.cam_to_model, info)
    front, _ = _pooled_front(run, views, depth, rng)
    for t, v in views.items():
        p = v.points(run.intr, _front_mask(run, t), depth[t])
        p = p[np.abs(front.uvz(p)[:, 1]) < 0.04]
        m, info = front_correction(p, front, rng, yaw_only=pitch_from == "support")
        v = FrameView(t, m @ v.cam_to_model, {**v.info, **info})
        if pitch_from == "front":
            m, info = support_roll(support(v, t, front), front, z_floor, rng)
            v = FrameView(t, m @ v.cam_to_model, {**v.info, **info})
        views[t] = v
    front, share = _pooled_front(run, views, depth, rng)
    tilts = [v.info.get("tilt_deg") for v in views.values() if v.info.get("levelled")]
    if tilts:
        log(f"levelled {len(tilts)} of {len(views)} frames on their own support surface (tilt was "
            f"{np.median(tilts):.1f} deg median, up to {max(tilts):.1f})")
    return views, front, share


def top_edges(run: Run, views: dict, front: "Front", depth: np.ndarray, masks: dict) -> dict:
    """Each frame's own top edge of the front: a high percentile of its front points' height over
    the middle of the front (its corners are rounded)."""
    out = {}
    for t, m in masks.items():
        p = front.uvz(views[t].points(run.intr, m, depth[t]))
        near = p[np.abs(p[:, 1]) < FRONT_TOL_M]
        if len(near) < 500:
            continue
        lo, hi = np.percentile(near[:, 0], [1, 99])
        mid = near[(near[:, 0] > lo + 0.1 * (hi - lo)) & (near[:, 0] < hi - 0.1 * (hi - lo))]
        if len(mid) >= 300:
            out[t] = float(np.percentile(mid[:, 2], 99.5))
    return out


def align_tops(views: dict, tops: dict, z_top: float) -> dict:
    """Shift each frame's camera up or down so its own top edge sits at the pooled one. The top
    face is seen at a grazing angle from just above it, so a few millimetres of height error per
    frame (levelling residue) move its far edge by many centimetres of depth."""
    out = {}
    for t, v in views.items():
        if t in tops:
            m = v.cam_to_model.copy()
            m[2, 3] += z_top - tops[t]
            v = FrameView(t, m, {**v.info, "top_shift_mm": 1000 * (z_top - tops[t])})
        out[t] = v
    return out


def measure_front(run: Run, views: dict, front: "Front", depth: np.ndarray, whole: dict, log=print) -> dict:
    """The front's extent from the depth points near the front plane inside the whole object's
    masks (the tracked parts where SAM gave none), on levelled frames: its width, its top edge
    (the median of each frame's own), down to the floor (z = 0 in the world); the deepest depth
    points behind the front inside those masks (its top or sides) as the least depth seen; the
    nearest surface seen behind the object just above its top (a wall it stands against), as a
    hard limit if the object can't pass through it. Returns the views with their tops aligned."""
    z_floor = -float(run.root_pos[2])
    src = {t: m for t, m in whole.items() if t in views} or {t: run.object_mask(t) for t in views}
    tops = top_edges(run, views, front, depth, src)
    z_top = float(np.median(list(tops.values())))
    views = align_tops(views, tops, z_top)
    near, behind = [], []
    for t, m in src.items():
        p = front.uvz(views[t].points(run.intr, m, depth[t]))
        near.append(p[np.abs(p[:, 1]) < FRONT_TOL_M])
        behind.append(p[p[:, 1] > BEHIND_M])
    near = np.vstack(near)
    u0, u1 = np.percentile(near[:, 0], [0.5, 99.5])
    behind = np.vstack(behind)
    behind = behind[(behind[:, 0] > u0) & (behind[:, 0] < u1) & (behind[:, 2] > z_floor) & (behind[:, 2] < z_top + 0.02)]
    seen = float(np.percentile(behind[:, 1], 98)) if len(behind) >= 100 else None
    wall = background_wall(run, views, front, depth, src, (u0, u1), z_top)
    out = {"front": front, "views": views, "u": [float(u0), float(u1)], "z": [z_floor, z_top],
           "width_m": float(u1 - u0), "height_m": z_top - z_floor, "front_points": int(len(near)),
           "top_edge_per_frame_mm": {str(t): round(1000 * (v - z_floor), 1) for t, v in sorted(tops.items())},
           "seen_behind_points": int(len(behind)), "seen_depth_m": seen, "background": wall,
           "extent_from": "SAM whole-object masks" if whole else "tracked parts (no whole-object mask)"}
    log(f"front: {1000 * out['width_m']:.0f} mm wide, {1000 * out['height_m']:.0f} mm tall from the floor "
        f"({len(near)} depth points on it; each frame's top edge {1000 * (min(tops.values()) - z_floor):.0f}-"
        f"{1000 * (max(tops.values()) - z_floor):.0f} mm); {len(behind)} points on faces behind it" +
        (f", the deepest {1000 * seen:.0f} mm" if seen is not None else "") +
        (f"; a wall {1000 * wall['depth_m']:.0f} mm behind the front stands right above its top"
         if wall.get("depth_m") is not None else ""))
    return out


def background_wall(run: Run, views: dict, front: "Front", depth: np.ndarray, masks: dict, u_range, z_top: float) -> dict:
    """The surface seen right above the object's top, behind its front (in a ring around its
    masks, over its width, up to 15 cm above its top). If it is one near-vertical plane facing the
    front (a wall the object stands against), the body can't reach behind it, assuming the wall
    continues down behind the object: its distance is a hard limit on the depth."""
    from .lift3d import fit_plane
    u0, u1 = u_range
    pts = []
    for t, m in masks.items():
        m8 = m.astype(np.uint8)
        ring = cv2.dilate(m8, np.ones((61, 61), np.uint8)).astype(bool) & ~cv2.dilate(m8, np.ones((9, 9), np.uint8)).astype(bool)
        p = front.uvz(views[t].points(run.intr, ring, depth[t]))
        pts.append(p[(p[:, 1] > BEHIND_M) & (p[:, 0] > u0) & (p[:, 0] < u1) & (p[:, 2] > z_top) & (p[:, 2] < z_top + 0.15)])
    pts = np.vstack(pts) if pts else np.zeros((0, 3))
    out = {"points": int(len(pts)), "depth_m": None}
    if len(pts) < 200:
        out["why_not"] = "too few depth points behind the object above its top"
        return out
    lo, hi = np.percentile(pts[:, 1], [5, 95])
    span = float(np.ptp(np.percentile(pts[:, 0], [2, 98])))
    out.update({"v_spread_m": float(hi - lo), "span_m": span})
    if hi - lo > 0.06 or span < 0.5 * (u1 - u0):
        out["why_not"] = "the surface behind the object isn't one wall facing its front across its width"
        return out
    out["depth_m"] = float(lo)
    return out


# ----------------------------------------------------------------------------------------------
# The door in the model: its surface over its whole joint range, in front coordinates


@dataclass
class DoorModel:
    joint: str
    body: str
    base: str
    axis: np.ndarray
    limits: tuple
    mass: float | None


def door_of(model_xml: Path) -> DoorModel:
    """The root body and its child on a hinge (a door model's base and door)."""
    root = ET.parse(model_xml).getroot()
    base = root.find("worldbody").find("body")
    for child in base.findall("body"):
        j = child.find("joint")
        if j is not None and j.get("type", "hinge") == "hinge":
            inertial = child.find("inertial")
            return DoorModel(j.get("name"), child.get("name"), base.get("name"),
                             np.array([float(v) for v in j.get("axis").split()]),
                             tuple(float(v) for v in j.get("range").split()),
                             float(inertial.get("mass")) if inertial is not None else None)
    raise ValueError(f"{model_xml} has no body on a hinge under its root")


def door_sweep(model_xml: Path, door: DoorModel, root_pos: np.ndarray, steps: int = 61, n: int = 3000, seed: int = 0):
    """Points over the door's collidable surface at joint values across its range, model frame."""
    import mujoco
    from .physics import _surface_points
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    jid, bid = model.joint(door.joint).id, model.body(door.body).id
    out = []
    for q in np.linspace(*door.limits, steps):
        data.qpos[:] = 0
        data.qpos[model.jnt_qposadr[jid]] = q
        mujoco.mj_forward(model, data)
        # the same points on the door at every angle
        out.append(_surface_points(model, data, bid, np.random.default_rng(seed), n) - root_pos)
    return np.vstack(out)


def door_pocket(door_uvz: np.ndarray, gap: float = DOOR_GAP_M) -> dict | None:
    """Where the door reaches behind the front plane (v > -gap) over its range: the carcass leaves
    that box free (the door sits in it when shut), plus the gap. None if it never does."""
    behind = door_uvz[door_uvz[:, 1] > -gap]
    if not len(behind):
        return None
    return {"u": [float(behind[:, 0].min() - gap), float(behind[:, 0].max() + gap)],
            "z": [float(behind[:, 2].min() - gap), float(behind[:, 2].max() + gap)],
            "v": float(max(0.0, behind[:, 1].max()) + gap)}


# ----------------------------------------------------------------------------------------------
# 3. Estimate the rest with Muse Spark (category defaults where it gives nothing usable)

ESTIMATE_DIMS = ("outer_depth", "cavity_width", "cavity_height", "cavity_depth", "wall_thickness")

PROMPT = """This video shows a {object}. An iPhone LiDAR capture measured its front, but the camera stayed in
front of it: the body behind the front was never seen. We complete a simulation model (URDF/MJCF) that a
robot can touch, so we need the hidden sizes.

Measured from the LiDAR depth (millimetres):
- the front: {width:.0f} wide x {height:.0f} tall (from the surface it stands on to its top edge)
- the door, on a vertical hinge at one side of the front: {door_w:.0f} wide x {door_h:.0f} tall; the part of the
  front beside the door is {side_w:.0f} wide
{seen}
Bounded by the video: {bound}

Estimate the hidden dimensions typical for this kind of object, or for this exact model if you recognise it
(say which). The cavity is the space behind the door that the door closes over (for a microwave or an oven,
the cooking chamber; for a cupboard, its inside); wall_thickness is the casing between the cavity and the
outside on the top, bottom and sides. mass_kg is the whole object without its door. For each size give
"typical": the usual size for this kind of object (or this model's specification) on its own, and "value":
your estimate for this object, consistent with the measurements and the bound above. Keep each reason to one
line. Answer with one JSON object only:
{{"object": str, "recognised_model": str or null, "has_cavity": bool,
 "outer_depth_mm": {{"value": number, "typical": number, "range": [low, high], "reason": str}},
 "cavity_width_mm": {{"value": number, "typical": number, "range": [low, high], "reason": str}},
 "cavity_height_mm": {{"value": number, "typical": number, "range": [low, high], "reason": str}},
 "cavity_depth_mm": {{"value": number, "typical": number, "range": [low, high], "reason": str}},
 "wall_thickness_mm": {{"value": number, "typical": number, "range": [low, high], "reason": str}},
 "mass_kg": {{"value": number, "typical": number, "range": [low, high], "reason": str}}}}"""


def parse_json(text: str) -> dict:
    import re
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in the answer")
    return json.loads(text[start:end + 1])


def ask_spark(video: Path, facts: dict, images: list[np.ndarray], log=print) -> dict:
    """One request to the agent's reasoning backend (Muse Spark on the Meta API; Muse Spark Open
    in an open-weights run): the video, a few annotated frames and the measured facts. Returns
    the parsed answer with the model's name, or an "error"."""
    import os
    from openai import OpenAI
    from .agent import API_KEY_ENV, BASE_URL, MAX_IMAGES, MODEL, VIDEO, jpeg_url, mp4_url
    content = []
    if VIDEO:
        content.append({"type": "video_url", "video_url": {"url": mp4_url(video)}})
    room = MAX_IMAGES - len(content) if MAX_IMAGES else len(images)
    content += [{"type": "image_url", "image_url": {"url": jpeg_url(im, width=1080)}} for im in images[:max(0, room)]]
    content.append({"type": "text", "text": PROMPT.format(**facts)})
    client = OpenAI(base_url=BASE_URL, api_key=os.environ[API_KEY_ENV], timeout=900, max_retries=2)
    messages = [{"role": "user", "content": content}]
    failures = []
    for attempt in range(2):
        text = ""
        try:
            resp = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=6000,
                                                  response_format={"type": "json_object"})
            text = resp.choices[0].message.content or ""
            return {"model": MODEL, "base_url": BASE_URL, "answer": parse_json(text), "raw": text}
        except (ValueError, json.JSONDecodeError) as e:
            failures.append(f"attempt {attempt + 1}: {e}: {text[:200]!r}")
            messages = messages + [{"role": "assistant", "content": text or "(empty)"},
                                   {"role": "user", "content": "That was not one valid JSON object. Answer with the JSON only."}]
        except Exception as e:  # the backend being down: fall back to category defaults
            failures.append(f"{type(e).__name__}: {str(e)[:300]}")
            break
    log("Muse Spark gave no usable answer: " + " | ".join(failures))
    return {"model": MODEL, "base_url": BASE_URL, "answer": None, "error": failures}


def _number(entry, key: str = "value") -> float | None:
    """A positive number from an answer entry ({"value": x} or x), else None."""
    v = entry.get(key) if isinstance(entry, dict) else entry
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def resolve_dims(geometry: dict, bounds: dict, answer: dict | None, model: str | None, category: str | None,
                 defaults: dict) -> dict:
    """The hidden sizes, each with where it came from: Muse Spark's estimate if usable, else the
    category default; then clamped to what the measurements and the video allow. ``geometry``
    holds the measured front (width_m, height_m), the pocket the door needs (pocket_u, pocket_z,
    pocket_v) and ``bounds`` the depth limits (lower_m, upper_m, with what set them)."""
    answer = answer or {}
    width, height = geometry["width_m"], geometry["height_m"]
    pu, pz, pv = geometry["pocket_u"], geometry["pocket_z"], geometry["pocket_v"]
    open_w, open_h = (pu[1] - pu[0]) - 2 * FRAME_MARGIN_M, (pz[1] - pz[0]) - 2 * FRAME_MARGIN_M
    default_name = f"{category} default" if category else "generic default"
    defaults_m = {
        "outer_depth": defaults["depth_ratio"] * width,
        "wall_thickness": defaults["wall"],
        "cavity_width": (pu[1] - pu[0]) - 2 * max(defaults["frame"], FRAME_MARGIN_M),
        "cavity_height": (pz[1] - pz[0]) - 2 * max(defaults["frame"], FRAME_MARGIN_M),
    }
    out = {}

    def pick(dim: str, fallback: float | None):
        entry = answer.get(f"{dim}_mm")
        v = _number(entry)
        if v is not None:
            rng = entry.get("range") if isinstance(entry, dict) else None
            rng = [float(x) / 1000 for x in rng] if isinstance(rng, list) and len(rng) == 2 and all(
                isinstance(x, (int, float)) for x in rng) else None
            typical = _number(entry, "typical") if isinstance(entry, dict) else None
            return {"value_m": v / 1000, "source": "Muse Spark", "model": model, "range_m": rng,
                    "typical_m": None if typical is None else typical / 1000,
                    "reason": str(entry.get("reason", "")) if isinstance(entry, dict) else ""}
        return {"value_m": fallback, "source": default_name, "model": None, "range_m": None,
                "reason": defaults["reason"]}

    def clamp(entry: dict, lo: float, hi: float, lo_why: str, hi_why: str):
        v = entry["value_m"]
        if hi < lo:  # the limits contradict each other: keep the stronger (lower, measured) one
            hi, hi_why = lo, lo_why
        c = float(np.clip(v, lo, hi))
        if abs(c - v) > 1e-6:
            entry["clamped"] = {"from_m": v, "to_m": c, "by": lo_why if c == lo else hi_why}
        entry["value_m"] = c
        return entry

    depth_lo = max(bounds.get("lower_m") or 0.0, pv + 2 * MIN_WALL_M + MIN_CAVITY_M)
    lo_why = bounds.get("lower_why") or "room for the door's pocket, a cavity and a back wall"
    if (bounds.get("lower_m") or 0.0) < pv + 2 * MIN_WALL_M + MIN_CAVITY_M:
        lo_why = "room for the door's pocket, a cavity and a back wall"
    depth_hi = bounds.get("upper_m") if bounds.get("upper_m") is not None else np.inf
    out["outer_depth"] = clamp(pick("outer_depth", defaults_m["outer_depth"]), depth_lo, depth_hi, lo_why,
                               bounds.get("upper_why") or "the video's bound")
    d = out["outer_depth"]["value_m"]
    out["wall_thickness"] = clamp(pick("wall_thickness", defaults_m["wall_thickness"]), MIN_WALL_M,
                                  max(MIN_WALL_M, 0.5 * (min(open_w, open_h) - MIN_CAVITY_M)),
                                  "a wall needs some thickness", "the door's outline leaves no room for thicker walls")
    wall = out["wall_thickness"]["value_m"]
    has_cavity = answer.get("has_cavity", True) is not False
    if has_cavity:
        cw = pick("cavity_width", None)
        if cw["value_m"] is None:
            cw.update(value_m=max(MIN_CAVITY_M, (pu[1] - pu[0]) - 2 * wall))
        ch = pick("cavity_height", None)
        if ch["value_m"] is None:
            ch.update(value_m=max(MIN_CAVITY_M, (pz[1] - pz[0]) - 2 * wall))
        cd = pick("cavity_depth", None)
        if cd["value_m"] is None:
            cd.update(value_m=d - pv - defaults["back_wall"])
        why_w = "the opening must stay behind the door (its measured outline)"
        out["cavity_width"] = clamp(cw, MIN_CAVITY_M, max(MIN_CAVITY_M, open_w), "a usable cavity", why_w)
        out["cavity_height"] = clamp(ch, MIN_CAVITY_M, max(MIN_CAVITY_M, open_h), "a usable cavity", why_w)
        out["cavity_depth"] = clamp(cd, MIN_CAVITY_M, max(MIN_CAVITY_M, d - pv - MIN_WALL_M), "a usable cavity",
                                    "it must fit inside the body's depth with a back wall")
    mass = _number(answer.get("mass_kg"))
    volume = width * height * d
    if mass is not None:
        entry = answer.get("mass_kg")
        out["mass"] = {"value_kg": float(np.clip(mass, 0.1, 2000)), "source": "Muse Spark", "model": model,
                       "range_kg": entry.get("range") if isinstance(entry, dict) else None,
                       "reason": str(entry.get("reason", "")) if isinstance(entry, dict) else ""}
    else:
        out["mass"] = {"value_kg": float(defaults["density"] * volume), "source": default_name, "model": None,
                       "reason": f"{defaults['density']} kg per cubic metre of the body's outer volume"}
    out["has_cavity"] = has_cavity
    return out


# ----------------------------------------------------------------------------------------------
# 4. Build: the carcass as boxes in front coordinates


def carcass_boxes(outer_u, outer_z, depth: float, pocket: dict | None, cavity: dict | None,
                  min_size: float = 0.001) -> list[dict]:
    """The completed body as non-overlapping boxes (lo, hi in front coordinates u, v, z): the
    outer box [u0, u1] x [0, depth] x [z0, z1] minus the pocket the door sits in ([pu] x [0, pv]
    x [pz]) and minus the cavity behind it ([cu] x [pv, pv + cd] x [cz], open toward the door).
    Each box is convex, so a simulator collides with it as built (one hollow mesh would collide
    as its convex hull: solid)."""
    u0, u1 = outer_u
    z0, z1 = outer_z
    boxes = []

    def add(name, lo, hi):
        if all(h - l >= min_size for l, h in zip(lo, hi)):
            boxes.append({"name": name, "lo": [float(x) for x in lo], "hi": [float(x) for x in hi]})

    if pocket is None:  # the door never reaches behind the front: a degenerate pocket
        pocket = {"u": [u0, u0], "z": [z0, z0], "v": 0.0}
    pu = [float(np.clip(pocket["u"][0], u0, u1)), float(np.clip(pocket["u"][1], u0, u1))]
    pz = [float(np.clip(pocket["z"][0], z0, z1)), float(np.clip(pocket["z"][1], z0, z1))]
    pv = float(np.clip(pocket["v"], 0.0, depth))
    # around the pocket: flush with the front, full depth
    add("top", (u0, 0, pz[1]), (u1, depth, z1))
    add("bottom", (u0, 0, z0), (u1, depth, pz[0]))
    add("side_low_u", (u0, 0, pz[0]), (pu[0], depth, pz[1]))
    add("side_high_u", (pu[1], 0, pz[0]), (u1, depth, pz[1]))
    if pu[1] - pu[0] < min_size or pz[1] - pz[0] < min_size:
        return boxes
    # behind the pocket: the walls around the cavity and its back, or a solid block
    if cavity is None:
        add("behind_door", (pu[0], pv, pz[0]), (pu[1], depth, pz[1]))
        return boxes
    cu = [max(pu[0], cavity["u"][0]), min(pu[1], cavity["u"][1])]
    cz = [max(pz[0], cavity["z"][0]), min(pz[1], cavity["z"][1])]
    cv = min(depth, pv + cavity["depth"])
    add("cavity_top_wall", (pu[0], pv, cz[1]), (pu[1], depth, pz[1]))
    add("cavity_bottom_wall", (pu[0], pv, pz[0]), (pu[1], depth, cz[0]))
    add("cavity_wall_low_u", (pu[0], pv, cz[0]), (cu[0], depth, cz[1]))
    add("cavity_wall_high_u", (cu[1], pv, cz[0]), (pu[1], depth, cz[1]))
    add("cavity_back_wall", (cu[0], cv, cz[0]), (cu[1], depth, cz[1]))
    return boxes


def liner_boxes(cavity_lo, cavity_hi, t: float = 0.0015) -> list[dict]:
    """Thin visual-only panels on the cavity's inner faces, in the colour seen inside it."""
    (u0, v0, z0), (u1, v1, z1) = cavity_lo, cavity_hi
    return [{"name": "liner_top", "lo": [u0, v0, z1 - t], "hi": [u1, v1, z1]},
            {"name": "liner_bottom", "lo": [u0, v0, z0], "hi": [u1, v1, z0 + t]},
            {"name": "liner_low_u", "lo": [u0, v0, z0], "hi": [u0 + t, v1, z1]},
            {"name": "liner_high_u", "lo": [u1 - t, v0, z0], "hi": [u1, v1, z1]},
            {"name": "liner_back", "lo": [u0, v1 - t, z0], "hi": [u1, v1, z1]}]


def overlaps(boxes: list[dict], tol: float = 1e-6) -> list[tuple[str, str]]:
    """Pairs of boxes whose interiors overlap."""
    out = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if all(min(a["hi"][k], b["hi"][k]) - max(a["lo"][k], b["lo"][k]) > tol for k in range(3)):
                out.append((a["name"], b["name"]))
    return out


MODEL_FILES = ("model", "model.nophysics", "model.predesign", "model.predesign.nophysics")
BACKUP_FILES = tuple(f"{s}.{e}" for s in MODEL_FILES for e in ("xml", "urdf")) + (
    "physics.json", "model3d.json", "unmeasured.txt", "replay3d.mp4", "replay3d.npz")
PREFIX = "completed_"


def _fmt(x) -> str:
    return " ".join(f"{float(v):.6g}" for v in x)


def box_pose(front: Front, b: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """A front-frame box as a model-frame centre, half sizes along (u, -n, z) and its yaw."""
    lo, hi = np.array(b["lo"]), np.array(b["hi"])
    return front.xyz((lo + hi) / 2)[0], (hi - lo) / 2, front.yaw


def add_to_mjcf(path: Path, base: str, front: Front, boxes: list[dict], rgba, visual: list[dict], visual_rgba) -> None:
    tree = ET.parse(path)
    body = next(b for b in tree.getroot().iter("body") if b.get("name") == base)
    for g in [g for g in body.findall("geom") if g.get("name", "").startswith(PREFIX)]:
        body.remove(g)
    friction = next((g.get("friction") for g in body.findall("geom") if g.get("friction")), None)
    at = max([i for i, c in enumerate(body) if c.tag in ("geom", "inertial")] or [-1]) + 1
    for b, colour, collide in [(b, rgba, True) for b in boxes] + [(b, visual_rgba, False) for b in visual]:
        c, half, yaw = box_pose(front, b)
        el = ET.Element("geom", name=PREFIX + b["name"], type="box", pos=_fmt(c), size=_fmt(half),
                        quat=_fmt([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), rgba=_fmt(colour))
        if collide and friction:
            el.set("friction", friction)
        if not collide:
            el.set("contype", "0")
            el.set("conaffinity", "0")
            el.set("mass", "0")
        body.insert(at, el)
        at += 1
    ET.indent(tree, "  ")
    tree.write(path, xml_declaration=False)


def add_to_urdf(path: Path, base: str, front: Front, boxes: list[dict], rgba, visual: list[dict], visual_rgba) -> None:
    tree = ET.parse(path)
    link = next(l for l in tree.getroot().iter("link") if l.get("name") == base)
    for el in [e for e in link if e.get("name", "").startswith(PREFIX)]:
        link.remove(el)
    for b, colour, collide in [(b, rgba, True) for b in boxes] + [(b, visual_rgba, False) for b in visual]:
        c, half, yaw = box_pose(front, b)
        for tag in ("visual", "collision") if collide else ("visual",):
            el = ET.SubElement(link, tag, name=PREFIX + b["name"])
            ET.SubElement(el, "origin", xyz=_fmt(c), rpy=_fmt([0, 0, yaw]))
            ET.SubElement(ET.SubElement(el, "geometry"), "box", size=_fmt(2 * half))
            if tag == "visual":
                ET.SubElement(ET.SubElement(el, "material", name=PREFIX + b["name"] + "_color"), "color", rgba=_fmt(colour))
    ET.indent(tree, "  ")
    tree.write(path, xml_declaration=True, encoding="unicode")


def set_base_inertial(run_dir: Path, stem: str, base: str, mass: float) -> dict | None:
    """The base's inertial over its completed (collidable) geometry at ``mass``, in the MJCF and
    URDF that carry inertials (physics applied); the other links keep theirs."""
    from .physics import _quat_to_rpy, inertials
    xml, urdf = run_dir / f"{stem}.xml", run_dir / f"{stem}.urdf"
    root = ET.parse(xml).getroot()
    masses = {}
    for b in root.iter("body"):
        i = b.find("inertial")
        if i is None:
            return None  # no physics in this file: MuJoCo derives inertia from the geoms
        masses[b.get("name")] = float(i.get("mass"))
    masses[base] = mass
    inert = inertials(xml, masses)[base]
    tree = ET.parse(xml)
    body = next(b for b in tree.getroot().iter("body") if b.get("name") == base)
    el = body.find("inertial")
    el.set("pos", _fmt(inert["pos"]))
    el.set("quat", _fmt(inert["quat"]))
    el.set("mass", f"{mass:.6g}")
    el.set("diaginertia", _fmt(inert["diaginertia"]))
    ET.indent(tree, "  ")
    tree.write(xml, xml_declaration=False)
    if urdf.exists():
        tree = ET.parse(urdf)
        link = next(l for l in tree.getroot().iter("link") if l.get("name") == base)
        el = link.find("inertial")
        if el is not None:
            el.find("origin").set("xyz", _fmt(inert["pos"]))
            el.find("origin").set("rpy", _fmt(_quat_to_rpy(inert["quat"])))
            el.find("mass").set("value", f"{mass:.6g}")
            d = inert["diaginertia"]
            i = el.find("inertia")
            for k, v in zip(("ixx", "iyy", "izz"), d):
                i.set(k, f"{v:.6g}")
            ET.indent(tree, "  ")
            tree.write(urdf, xml_declaration=True, encoding="unicode")
    return inert


def backup(run_dir: Path) -> Path:
    """The measured-only files, kept once in completion/measured/ (and model.measured.{xml,urdf}
    next to the model, so it loads with its meshes); later runs restore from them first."""
    keep = run_dir / "completion" / "measured"
    if keep.exists():
        restore(run_dir)
        return keep
    keep.mkdir(parents=True)
    for f in BACKUP_FILES:
        if (run_dir / f).exists():
            shutil.copy2(run_dir / f, keep / f)
    for e in ("xml", "urdf"):
        shutil.copy2(run_dir / f"model.{e}", run_dir / f"model.measured.{e}")
    return keep


def restore(run_dir: Path) -> None:
    """Back to the measured-only model (the files completion changed, as they were)."""
    keep = run_dir / "completion" / "measured"
    if not keep.exists():
        raise FileNotFoundError(f"no measured-only backup in {keep}")
    for f in BACKUP_FILES:
        if (keep / f).exists():
            shutil.copy2(keep / f, run_dir / f)
        elif (run_dir / f).exists() and f == "unmeasured.txt":
            (run_dir / f).unlink()


# ----------------------------------------------------------------------------------------------
# Checks: coverage of the whole object's masks, a probe pushed at the body, renders


def render_bodies(model_xml: Path, cams: dict, intr, q: dict, joint: str, size) -> dict:
    """Per frame, the pixels any of the model's geoms cover, seen from the given camera
    (OpenCV axes -> world) with the door at q[t]."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    cam = model.camera("video").id
    w, h = size
    model.cam_fovy[cam] = np.rad2deg(2 * np.arctan(h / 2 / intr.f))
    renderer = mujoco.Renderer(model, h, w)
    renderer.enable_segmentation_rendering()
    flip = np.diag([1.0, -1.0, -1.0])
    out = {}
    for t, c2w in cams.items():
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (c2w[:3, :3] @ flip).flatten())
        model.cam_pos[cam], model.cam_quat[cam] = c2w[:3, 3], quat
        data.qpos[:] = 0
        data.qpos[model.jnt_qposadr[model.joint(joint).id]] = q[t]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="video")
        seg = renderer.render()
        out[t] = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
    renderer.close()
    return out


def mask_coverage(rendered: dict, masks: dict, edge_px: int) -> dict:
    """How much of each whole-object mask the model covers, and how much of the model lies on
    the object (inside the mask dilated by edge_px), per frame and in the mean."""
    k = np.ones((2 * edge_px + 1, 2 * edge_px + 1), np.uint8)
    per = {}
    for t, m in masks.items():
        r = rendered[t]
        dil = cv2.dilate(m.astype(np.uint8), k).astype(bool)
        per[str(t)] = {"covers": float((r & m).sum() / max(1, m.sum())),
                       "on_object": float((r & dil).sum() / max(1, r.sum()))}
    return {"per_frame": per, "covers": float(np.mean([v["covers"] for v in per.values()])) if per else None,
            "on_object": float(np.mean([v["on_object"] for v in per.values()])) if per else None}


def probe(model_xml: Path, joint: str, q: float, start, end, radius: float = 0.012, force: float = 30.0,
          seconds: float = 3.0) -> dict:
    """A sphere pushed from ``start`` toward ``end`` (world frame) with at most ``force`` N per
    axis, the door held at ``q``: where it stops. Returns the distance travelled along the line
    and how far short of ``end`` it stayed."""
    import mujoco
    spec = mujoco.MjSpec.from_file(str(model_xml))
    spec.option.gravity = [0, 0, 0]
    body = spec.worldbody.add_body(name="probe", pos=list(start))
    for i, ax in enumerate(np.eye(3)):
        body.add_joint(name=f"probe_{i}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=list(ax), damping=20)
    body.add_geom(name="probe", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius, 0, 0], mass=0.3,
                  rgba=[1, 0, 0, 1], solref=[0.002, 1])
    target = np.asarray(end, float) - np.asarray(start, float)
    for i in range(3):
        a = spec.add_actuator(name=f"push_{i}", target=f"probe_{i}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=2000)
        a.forcelimited = True
        a.forcerange = [-force, force]
    model = spec.compile()
    data = mujoco.MjData(model)
    door = model.joint(joint)
    data.qpos[door.qposadr[0]] = q
    model.jnt_range[door.id] = [q - 1e-4, q + 1e-4]  # held there
    for i in range(3):
        data.ctrl[i] = target[i]
    steps = int(seconds / model.opt.timestep)
    path = target / max(np.linalg.norm(target), 1e-9)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    pos = np.array(start) + np.array([data.qpos[model.joint(f"probe_{i}").qposadr[0]] for i in range(3)])
    travelled = float((pos - np.asarray(start)) @ path)
    return {"start": [float(v) for v in start], "end": [float(v) for v in end], "stopped_at": pos.tolist(),
            "travelled_m": travelled, "short_of_end_m": float(np.linalg.norm(target) - travelled)}


def render_views(model_xml: Path, joint: str, q: float, front: Front, root_pos, out: Path, title: str,
                 size=(960, 720)) -> None:
    """The model from the front-left (like the phone), from behind and above, and from above."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.6
    model.vis.headlight.specular[:] = 0.1
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint(joint).id]] = q
    mujoco.mj_forward(model, data)
    corners = []
    for g in range(model.ngeom):  # every geom's box, in the world
        c, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        local = c + half * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        corners.append(local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    corners = np.vstack(corners)
    renderer = mujoco.Renderer(model, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (corners.min(axis=0) + corners.max(axis=0)) / 2
    cam.distance = 1.5 * float(np.linalg.norm(np.ptp(corners, axis=0)))
    out_dir = Path(out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    n = front.n
    facing = np.rad2deg(np.arctan2(-n[1], -n[0]))  # azimuth looking at the front
    tiles = []
    for name, az, el in (("front", facing + 30, -20), ("behind, above", facing + 180 - 35, -35), ("above", facing, -80)):
        cam.azimuth, cam.elevation = az, el
        renderer.update_scene(data, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        img[seg[..., 0] < 0] = (235, 235, 235)  # a light background: the body is black
        cv2.putText(img, f"{title}: {name}", (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4)
        cv2.putText(img, f"{title}: {name}", (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        tiles.append(img)
    renderer.close()
    cv2.imwrite(str(out), cv2.hconcat(tiles))


def penetration(points_uvz: np.ndarray, boxes: list[dict]) -> tuple[float, str | None]:
    """How deep the deepest point lies inside any box (0 if none does), and which box."""
    worst, name = 0.0, None
    for b in boxes:
        lo, hi = np.array(b["lo"]), np.array(b["hi"])
        inside = np.all((points_uvz > lo) & (points_uvz < hi), axis=1)
        if inside.any():
            p = points_uvz[inside]
            depth = np.minimum(p - lo, hi - p).min(axis=1).max()
            if depth > worst:
                worst, name = float(depth), b["name"]
    return worst, name


# ----------------------------------------------------------------------------------------------
# The whole step


def _colour(images: dict, masks: dict, fallback=(0.3, 0.3, 0.3)) -> list[float]:
    """Median colour (RGB 0..1) of the masked pixels over several frames."""
    px = [images[t][m] for t, m in masks.items() if t in images and m.any()]
    if not px:
        return list(fallback)
    b, g, r = np.median(np.vstack(px), axis=0)
    return [float(r / 255), float(g / 255), float(b / 255)]


def _frames(video: Path, wanted) -> dict:
    wanted, out, t = set(int(x) for x in wanted), {}, 0
    cap = cv2.VideoCapture(str(video))
    while wanted - set(out):
        ok, img = cap.read()
        if not ok:
            break
        if t in wanted:
            out[t] = img
        t += 1
    cap.release()
    return out


def _mm(x) -> str:
    return f"{1000 * x:.0f}"


def not_measured_lines(name: str, dims: dict, front: dict, bounds: dict) -> list[str]:
    """Plain lines for the viewer's "Not measured" note: what was estimated, by what, within what."""
    d = dims["outer_depth"]
    lines = [f"the {name}'s body behind its front was never seen (only the front is measured: "
             f"{_mm(front['width_m'])} x {_mm(front['height_m'])} mm), so it is completed from estimates."]
    clamp = d.get("clamped")
    src = "Muse Spark's estimate" if d["source"] == "Muse Spark" else f"a typical size ({d['source']})"
    line = f"Its depth, {_mm(d['value_m'])} mm, is {src}"
    if clamp:
        line += f" ({_mm(clamp['from_m'])} mm, clamped: {clamp['by']})"
    if d.get("typical_m") and abs(d["typical_m"] - d["value_m"]) > 0.01:
        line += f"; it puts this kind of object at {_mm(d['typical_m'])} mm"
    if bounds.get("upper_m") is not None:
        line += f"; the video allows at most {_mm(bounds['upper_m'])} mm ({bounds['upper_why']})"
    if bounds.get("lower_m"):
        line += f" and at least {_mm(bounds['lower_m'])} mm ({bounds['lower_why']})"
    lines.append(line + ".")
    if dims.get("has_cavity") and "cavity_width" in dims:
        srcs = {dims[k]["source"] for k in ("cavity_width", "cavity_height", "cavity_depth", "wall_thickness")}
        who = "Muse Spark's estimates" if srcs == {"Muse Spark"} else "estimates (" + ", ".join(sorted(srcs)) + ")"
        lines.append(f"The cavity behind the door ({_mm(dims['cavity_width']['value_m'])} x "
                     f"{_mm(dims['cavity_height']['value_m'])} x {_mm(dims['cavity_depth']['value_m'])} mm) and its walls "
                     f"are {who}, kept behind the door's measured outline.")
    m = dims["mass"]
    lines.append(f"The body's mass ({m['value_kg']:.1f} kg) is " +
                 ("Muse Spark's estimate." if m["source"] == "Muse Spark" else f"a typical value ({m['source']})."))
    return lines


def complete_body(run_dir: Path, video: Path, capture: Path, log=print, spark: bool = True, prompt: str | None = None,
                  note: str | None = None, sam=None) -> dict:
    """Complete a partial body in place (the measured-only model is kept): see the module's doc."""
    from . import stray
    from .replay import replay3d
    from .lidar import read_hinge
    run_dir = Path(run_dir)
    out_dir = run_dir / "completion"
    out_dir.mkdir(exist_ok=True)
    _, why = needs_completion(run_dir)
    backup(run_dir)  # restores the measured-only files first when a backup exists
    run = load_run(run_dir, video)
    door = door_of(run_dir / "model.xml")
    run.joint, run.base, run.door = door.joint, door.base, door.body
    open_frames = run.by.get(run.roles.get("open_indicator") or "", {})
    q_shut, shut = shut_state(run.q, open_frames)
    depth = stray.load(capture).depth
    rng = np.random.default_rng(0)
    w, h = run.size
    edge = int(round(EDGE_PX_FRAC * np.hypot(w, h)))
    design = json.loads((run_dir / "design.json").read_text()) if (run_dir / "design.json").exists() else {}
    name = design.get("object") or prompt or object_prompt(run_dir) or "object"
    prompt = prompt or object_prompt(run_dir) or name

    # 1. measure: the whole object's masks, levelled frames, the front
    fs, fo = mask_frames(run, shut)
    wm = whole_object_masks(run, fs + fo, prompt, out_dir / "whole_masks.npz", client=sam, log=log)
    whole = wm["masks"]
    whole_shut = {t: m for t, m in whole.items() if t in fs}
    z_floor = -float(run.root_pos[2])
    frames = sorted(set(int(t) for t in shut[:: max(1, len(shut) // 30)]) | set(whole_shut))

    # 2. bound the hidden depth: silhouettes, and the wall behind if there is one. The top is seen
    # from just above it, so the silhouettes hang on how each frame is levelled: they are bounded
    # with the frames levelled on the object's front and, again, on the support in front of it,
    # and only what both allow is kept (the larger upper bound, the smaller lower bound).
    sil_by, meas_by = {}, {}
    for mode in ("front", "support"):
        quiet = log if mode == "front" else (lambda *a, **k: None)
        v_, f_, _ = level_views(run, frames, depth, z_floor, rng, object_masks=whole_shut, log=quiet, pitch_from=mode)
        m_ = measure_front(run, v_, f_, depth, whole_shut, log=quiet)
        m_["front"] = f_
        meas_by[mode] = m_
        a0, a1 = m_["u"]
        b0, b1 = m_["z"]
        grid = np.arange(0, MAX_DEPTH_FACTOR * max(m_["width_m"], m_["height_m"]) + DEPTH_STEP_M, DEPTH_STEP_M)
        per = {t: frame_depth_bounds(lambda d, f_=f_, a0=a0, a1=a1, b0=b0, b1=b1: box_corners(f_, [a0, 0, b0], [a1, d, b1]),
                                     lambda p, t=t, vs=m_["views"]: vs[t].project(run.intr, p), m, grid, edge)
               for t, m in whole_shut.items()}
        sil_by[mode] = combine_bounds(per, m_["seen_depth_m"])
        log(f"silhouette bound, frames levelled on the {mode}: " + ", ".join(
            f"{t}: {_mm(b['lower_m'] or 0)}-{_mm(b['upper_m']) if b['upper_m'] is not None else 'open'} mm"
            for t, b in sorted(per.items())))
    meas = meas_by["front"]  # the carcass is a box on the object's front: build on the front-levelled frames
    front, views = meas["front"], meas["views"]
    u0, u1 = meas["u"]
    zf, zt = meas["z"]
    uppers = {k: s["upper_m"] for k, s in sil_by.items()}
    lowers = {k: s.get("lower_m") for k, s in sil_by.items()}
    upper_mode = None if any(v is None for v in uppers.values()) else max(uppers, key=uppers.get)
    lower_mode = None if any(v is None for v in lowers.values()) else min(lowers, key=lowers.get)
    sil = {"by_levelling": sil_by, "upper_m": uppers[upper_mode] if upper_mode else None,
           "lower_m": lowers[lower_mode] if lower_mode else None, "upper_levelling": upper_mode,
           "lower_levelling": lower_mode}
    if lower_mode:
        s = sil_by[lower_mode]
        lower_why = ("depth points on its faces behind the front" if s.get("lower_from_depth_points") else
                     f"the silhouettes of {len(s.get('consensus_frames', []))} of {len(s.get('used_frames', []))} frames "
                     f"need at least that (frame {', '.join(map(str, s.get('lower_frames', [])))}, levelled on the {lower_mode})")
    else:
        lower_why = None
    bounds = {"silhouette": sil, "wall": meas["background"], "lower_m": sil["lower_m"], "lower_why": lower_why,
              "upper_m": None, "upper_why": None}
    choices = []
    if sil["upper_m"] is not None:
        s = sil_by[upper_mode]
        choices.append((sil["upper_m"], f"its silhouette on frame {', '.join(map(str, s['upper_frames']))} leaves the "
                                        f"object's mask beyond that, with the frames levelled on the {upper_mode} "
                                        f"(levelled on the {'support' if upper_mode == 'front' else 'front'}: "
                                        f"{_mm(uppers['support' if upper_mode == 'front' else 'front'])} mm)"))
    if meas["background"].get("depth_m") is not None:
        choices.append((meas["background"]["depth_m"], "the wall seen right behind its top, assumed to continue "
                                                       "down behind it"))
    if choices:
        bounds["upper_m"], bounds["upper_why"] = min(choices)
    log("depth bound: " + (f"at most {_mm(bounds['upper_m'])} mm ({bounds['upper_why']})" if bounds["upper_m"] else
                           "none from the video") + (f", at least {_mm(bounds['lower_m'])} mm ({bounds['lower_why']})"
                                                     if bounds["lower_m"] else ""))

    # the door: where it needs room behind the front over its whole range
    sweep = front.uvz(door_sweep(run_dir / "model.xml", door, run.root_pos))
    pocket = door_pocket(sweep)
    shut_door = front.uvz(door_sweep(run_dir / "model.xml", DoorModel(door.joint, door.body, door.base, door.axis,
                                                                      (q_shut, q_shut), door.mass),
                                     run.root_pos, steps=1))
    door_u = np.percentile(shut_door[:, 0], [0, 100])
    door_z = np.percentile(shut_door[:, 2], [0, 100])
    if pocket is None:  # the door stays in front of the front: the cavity still sits behind its outline
        pocket_geom = {"u": [float(door_u[0]), float(door_u[1])], "z": [float(door_z[0]), float(door_z[1])], "v": 0.0}
    else:
        pocket_geom = dict(pocket)
    pocket_geom["u"] = [float(np.clip(x, u0, u1)) for x in pocket_geom["u"]]  # within the measured front
    pocket_geom["z"] = [float(np.clip(x, zf, zt)) for x in pocket_geom["z"]]
    shut_normal = np.linalg.svd(shut_door - shut_door.mean(axis=0), full_matrices=False)[2][2]
    door_fit = {"yaw_vs_front_deg": float(np.rad2deg(np.arctan2(shut_normal[0], abs(shut_normal[1])))),
                "lean_deg": float(np.rad2deg(np.arcsin(min(1.0, abs(shut_normal[2]))))),
                "behind_front_mm": [1000 * float(shut_door[:, 1].min()), 1000 * float(shut_door[:, 1].max())]}
    if pocket is not None:
        log(f"the shut door (as fitted) is turned {door_fit['yaw_vs_front_deg']:.1f} deg from the measured front and "
            f"leans {door_fit['lean_deg']:.1f} deg; over its range it reaches {1000 * pocket['v']:.0f} mm behind the front: "
            "the carcass leaves it a pocket that deep")
    side_w = max(u1 - pocket_geom["u"][1], pocket_geom["u"][0] - u0, 0.0)

    # 3. estimate the rest
    images = _frames(video, list(whole) + ([max(open_frames, key=lambda t: open_frames[t].area)] if open_frames else []))
    shown = []
    t_show = max(fs, key=lambda t: run.cam_to_world[t][2, 3]) if fs else None
    if t_show is not None and t_show in images:
        img = images[t_show].copy()
        for d, col, label in ((0.0, (255, 0, 0), f"measured front {_mm(meas['width_m'])} x {_mm(meas['height_m'])} mm"),
                              (bounds["upper_m"], (0, 0, 255), f"deepest the video allows: {_mm(bounds['upper_m'] or 0)} mm")):
            if d is None:
                continue
            s = silhouette(views[t_show].project(run.intr, box_corners(front, [u0, 0, zf], [u1, d, zt])), (w, h))
            if s is not None:
                cs, _ = cv2.findContours(s.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, col, 4)
                y = 60 if d == 0.0 else 120
                cv2.putText(img, label, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 8)
                cv2.putText(img, label, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, col, 3)
        shown.append(img)
    if open_frames:
        t_open = max(open_frames, key=lambda t: open_frames[t].area)
        if t_open in images:
            shown.append(images[t_open])
    seen = (f"- seen behind the front: {_mm(meas['seen_depth_m'])} mm (depth points on its top or sides)"
            if meas["seen_depth_m"] else "- nothing behind the front was measured (no depth points on its top or sides)")
    bound_text = (f"the body is at most {_mm(bounds['upper_m'])} mm deep ({bounds['upper_why']})"
                  if bounds["upper_m"] else "no limit (the camera never saw past the front)")
    if bounds["lower_m"]:
        bound_text += f", and at least {_mm(bounds['lower_m'])} mm ({bounds['lower_why']})"
    m3 = json.loads((run_dir / "model3d.json").read_text())
    door_hw = m3.get("plate_size_mm") or [1000 * (door_z[1] - door_z[0]), 1000 * (door_u[1] - door_u[0])]
    facts = {"object": name, "width": 1000 * meas["width_m"], "height": 1000 * meas["height_m"],
             "door_w": door_hw[1], "door_h": door_hw[0], "side_w": 1000 * side_w, "seen": seen, "bound": bound_text}
    for i, img in enumerate(shown):
        cv2.imwrite(str(out_dir / f"spark_input_{i}.jpg"), img)
    est = ask_spark(video, facts, shown, log=log) if spark else {"model": None, "answer": None, "error": ["not asked"]}
    answer = est.get("answer")
    if answer:
        log(f"{est['model']}: {json.dumps(answer, ensure_ascii=False)[:600]}")
    category, defaults = category_of(str((answer or {}).get("object") or name))
    if category is None:
        category, defaults = category_of(name)
    geometry = {"width_m": meas["width_m"], "height_m": meas["height_m"], "pocket_u": pocket_geom["u"],
                "pocket_z": pocket_geom["z"], "pocket_v": pocket_geom["v"]}
    dims = resolve_dims(geometry, bounds, answer, est.get("model"), category, defaults)

    # 4. build
    d = dims["outer_depth"]["value_m"]
    cavity = None
    if dims["has_cavity"]:
        cu = float(np.mean(pocket_geom["u"]))
        cz = float(np.mean(pocket_geom["z"]))
        cw, ch, cd = (dims[k]["value_m"] for k in ("cavity_width", "cavity_height", "cavity_depth"))
        cavity = {"u": [cu - cw / 2, cu + cw / 2], "z": [cz - ch / 2, cz + ch / 2], "depth": cd}
    boxes = carcass_boxes((u0, u1), (zf, zt), d, pocket_geom, cavity)
    clash = overlaps(boxes)
    pen, pen_box = penetration(sweep, boxes)
    if pen > 0.001:  # the door would enter the carcass somewhere in its range: deepen the pocket
        raise ValueError(f"the door enters the completed body by {1000 * pen:.1f} mm ({pen_box})")
    liners = []
    if cavity is not None:
        pv = pocket_geom["v"]
        lo = [max(cavity["u"][0], pocket_geom["u"][0]), pv, max(cavity["z"][0], pocket_geom["z"][0])]
        hi = [min(cavity["u"][1], pocket_geom["u"][1]), min(d, pv + cavity["depth"]), min(cavity["z"][1], pocket_geom["z"][1])]
        liners = liner_boxes(lo, hi)
    seen_body = {t: whole[t] & ~cv2.dilate(run.object_mask(t).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
                 for t in whole_shut}
    body_rgb = _colour(images, seen_body)
    inside = {t: open_frames[t].full_mask(h, w) for t in images if t in open_frames}
    cavity_rgb = _colour(images, inside, fallback=[min(1, c + 0.3) for c in body_rgb])
    for stem in MODEL_FILES:
        if (run_dir / f"{stem}.xml").exists():
            add_to_mjcf(run_dir / f"{stem}.xml", door.base, front, boxes, body_rgb + [1], liners, cavity_rgb + [1])
        if (run_dir / f"{stem}.urdf").exists():
            add_to_urdf(run_dir / f"{stem}.urdf", door.base, front, boxes, body_rgb + [1], liners, cavity_rgb + [1])
    mass = dims["mass"]["value_kg"]
    inertia = None
    for stem in ("model", "model.predesign"):
        if (run_dir / f"{stem}.xml").exists():
            inertia = set_base_inertial(run_dir, stem, door.base, mass) or inertia
    if inertia and (run_dir / "physics.json").exists():
        phys = json.loads((run_dir / "physics.json").read_text())
        link = phys["links"].get(door.base, {})
        link.update({"material": f"{name} body (completed: estimated)", "mass_kg": mass, "inertial": {**inertia, "mass": mass},
                     "mass_source": dims["mass"]["source"] + (f" ({dims['mass']['model']})" if dims["mass"].get("model") else ""),
                     "measured_part_mass_kg": link.get("mass_kg")})
        phys["links"][door.base] = link
        (run_dir / "physics.json").write_text(json.dumps(phys, indent=2))

    # checks: the replay over the video, coverage of the whole object's masks, a probe, renders
    hinge = read_hinge(run_dir)
    q = np.array([np.nan if v is None else v for v in hinge["q"]], float)
    replay = np.load(out_dir / "measured" / "replay3d.npz")
    cam_to_world = replay["cam_to_world"]
    oi = run.roles.get("open_indicator")
    cover = replay3d(run_dir / "model.xml", cam_to_world, run.intr, {door.joint: q},
                     {door.body: run.by[run.roles["plate"]], door.base: run.by[run.roles["base"]]}, video,
                     run_dir / "replay3d.mp4",
                     only_when={door.base: (lambda t: t in run.by.get(oi, {})) if oi else (lambda t: True)})
    m3 = json.loads((run_dir / "model3d.json").read_text())
    m3["mask_coverage"] = {k: float(np.nanmean(v)) if np.isfinite(v).any() else None for k, v in cover.items()}
    m3["mask_coverage_frames"] = {k: int(np.isfinite(v).sum()) for k, v in cover.items()}
    m3["worst_frames"] = {k: int(np.nanargmin(v)) for k, v in cover.items() if np.isfinite(v).any()}
    (run_dir / "model3d.json").write_text(json.dumps(m3, indent=2))
    q_replay = {t: float(replay[f"q_{door.joint}"][t]) for t in whole}
    cams = {t: cam_to_world[t] for t in whole}
    coverage = {}
    levelled = {}
    for t in whole_shut:  # the levelled cameras, model frame -> the MuJoCo world
        levelled[t] = views[t].cam_to_model.copy()
        levelled[t][:3, 3] += run.root_pos
    for label, xml in (("measured_only", run_dir / "model.measured.xml"), ("completed", run_dir / "model.xml")):
        coverage[label] = mask_coverage(render_bodies(xml, cams, run.intr, q_replay, door.joint, run.size), whole, edge)
        coverage[label + "_levelled_shut"] = mask_coverage(
            render_bodies(xml, levelled, run.intr, q_replay, door.joint, run.size), whole_shut, edge)
    log("whole-object masks covered: " + ", ".join(f"{k} {100 * v['covers']:.0f}% (on the object {100 * v['on_object']:.0f}%)"
                                                   for k, v in coverage.items()))
    def world(uvz):
        return front.xyz(np.atleast_2d(uvz))[0] + run.root_pos

    cu = float(np.mean(cavity["u"])) if cavity else float(np.mean([u0, u1]))
    cz = float(np.mean(cavity["z"])) if cavity else float((zf + zt) / 2)
    q_open = door.limits[0] if abs(door.limits[0] - q_shut) > abs(door.limits[1] - q_shut) else door.limits[1]
    tests = {
        "from_behind_door_shut": (q_shut, world([cu, d + 0.12, cz]), world([cu, -0.10, cz])),
        "from_above": (q_shut, world([cu, d / 2, zt + 0.12]), world([cu, d / 2, zf + 0.01])),
        "into_the_opening_door_open": (q_open, world([cu, -0.08, cz]), world([cu, d + 0.10, cz])),
        "at_the_shut_door": (q_shut, world([float(np.mean(door_u)), -0.15, cz]), world([float(np.mean(door_u)), d + 0.10, cz])),
    }
    probes = {}
    for label, xml in (("measured_only", run_dir / "model.measured.xml"), ("completed", run_dir / "model.xml")):
        probes[label] = {}
        for test, (qd, a, b) in tests.items():
            r = probe(xml, door.joint, qd, a, b)
            r["stopped_at_front_coords"] = [float(x) for x in front.uvz(np.array(r["stopped_at"]) - run.root_pos)[0]]
            r["blocked"] = bool(r["short_of_end_m"] > 0.02)  # stopped well before the end of its path
            probes[label][test] = r
    for label in probes:
        log(f"probe ({label}): " + "; ".join(f"{k} stopped at v {1000 * r['stopped_at_front_coords'][1]:.0f} mm, "
                                            f"z {1000 * (r['stopped_at_front_coords'][2] - zf):.0f} mm"
                                            for k, r in probes[label].items()))
    for label, xml in (("measured_only", run_dir / "model.measured.xml"), ("completed", run_dir / "model.xml")):
        for state, qd in (("shut", q_shut), ("open", q_open)):
            render_views(xml, door.joint, qd, front, run.root_pos, out_dir / f"views_{label}_{state}.png",
                         f"{label.replace('_', ' ')}, door {state}")

    # 5. provenance
    lines = not_measured_lines(name, dims, meas, bounds)
    if note:
        lines.append(note)
    result = {
        "object": name, "category": category, "trigger": why, "sam_prompt": prompt,
        "measured": {
            "front_width_mm": 1000 * meas["width_m"], "front_height_mm": 1000 * meas["height_m"],
            "front_u_mm": [1000 * u0, 1000 * u1], "top_edge_per_frame_mm": meas["top_edge_per_frame_mm"],
            "front_normal_model": front.n.tolist(), "front_origin_model": front.origin.tolist(),
            "front_extent_from": meas["extent_from"], "front_depth_points": meas["front_points"],
            "seen_depth_mm": None if meas["seen_depth_m"] is None else 1000 * meas["seen_depth_m"],
            "seen_behind_points": meas["seen_behind_points"],
            "door_shut_q_deg": float(np.rad2deg(q_shut)), "door_outline_mm_wide_tall": [float(door_hw[1]), float(door_hw[0])],
            "door_extent_mm_along_front_up": [1000 * float(np.ptp(door_u)), 1000 * float(np.ptp(door_z))],
            "side_beside_door_mm": 1000 * side_w, "frames": frames,
            "levelling": {str(t): v.info for t, v in views.items() if t in whole_shut},
            "whole_object_masks": {"prompt": prompt, "shut_frames": sorted(whole_shut), "open_frames": sorted(set(whole) - set(whole_shut)),
                                   "info": wm["info"], "segmenter": "SAM 3.1 (" + __import__("os").environ.get("SAM_BACKEND", "api") + ")"},
        },
        "bounded": {"upper_mm": None if bounds["upper_m"] is None else 1000 * bounds["upper_m"], "upper_why": bounds["upper_why"],
                    "lower_mm": None if bounds["lower_m"] is None else 1000 * bounds["lower_m"], "lower_why": bounds["lower_why"],
                    "silhouette": sil, "wall": meas["background"], "dilate_px": edge},
        "estimated": {"model": est.get("model"), "base_url": est.get("base_url"), "answer": answer,
                      "error": est.get("error"), "dims": dims},
        "built": {"boxes": boxes, "liners": liners, "pocket": pocket, "cavity": cavity, "outer_depth_mm": 1000 * d,
                  "size_mm": [1000 * (u1 - u0), 1000 * d, 1000 * (zt - zf)], "box_overlaps": clash,
                  "door_penetration_mm": 1000 * pen, "door_fit": door_fit, "body_rgb": body_rgb, "cavity_rgb": cavity_rgb, "body_mass_kg": mass},
        "checks": {"replay_mask_coverage": m3["mask_coverage"], "whole_object_coverage": coverage, "probes": probes},
        "not_measured": lines,
        "measured_only_model": ["model.measured.xml", "model.measured.urdf"],
    }
    (run_dir / "completion.json").write_text(json.dumps(result, indent=2, default=float))
    (run_dir / "unmeasured.txt").write_text(" ".join(lines) + "\n")
    log("not measured: " + " ".join(lines))
    return result
