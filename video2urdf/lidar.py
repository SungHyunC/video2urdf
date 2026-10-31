"""Hinged plate (laptop lid, door, flap) from an iPhone LiDAR capture: hinge, model, replay.

The camera's own motion comes from ARKit (Stray Scanner odometry), so the static scene is
the reference; the plate is tracked by fitting its plane in each frame's depth, which works
when it shows a different face at different angles (screen, then aluminium back). Shapes are
measured from the depth points: the base's footprint and thickness (its top surface to the
floor), the plate's outline (from its back when closed, turned back about the hinge) and
thickness (how high that back sits), colours, and the hinge with its observed range. The
model is then replayed over the video from the iPhone's own camera poses.

Part roles, by tracked part name:
  floor          the static surface the object stands on
  base           the part the plate hinges on, on the reference frame (e.g. "silver laptop")
  plate          the plate's inner face (e.g. "laptop screen")
  plate_closed   what covers the object when closed: the plate's back (often the same as base)
  open_indicator a part seen only while open (e.g. "laptop keyboard"); frames without it are closed
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

from . import emit3d, stray
from .lift3d import (clean_angles, drop_overclosed, fit_plane, fit_planar_hinge, fit_slide, is_door, mask_cloud,
                     part_poses_3d)
from .motion import load_gray
from .pipeline import load_parts, read_frame, video_info
from .sam import Detection
from .replay import replay3d
from .shape import part_color, prism, write_stl

SCREEN_DEPTH = 0.0008  # m, the dark display layer drawn on the plate's inner face
BASE_FALLBACK_M = 0.012  # base thickness when the capture can't measure it
DOOR_THICKNESS_M = 0.02  # a door's thickness when it isn't measured (only seen edge-on)
VERTICAL_HINGE_DEG = 30  # a hinge this close to the floor's normal is a door's
OFF_BASE_M = 0.015  # points this far off the base's top plane belong to a half-open plate
MAX_LEVEL_TILT_DEG = 4.0  # a floor fit tilted more than this from the reference is a bad fit
MAX_LEVEL_SHIFT_M = 0.1  # ...or shifted this far off it: another surface (ARKit drifts 1-3 cm)
MIN_FLOOR_POINTS = 50  # depth points a floor mask needs for its plane to be fitted
LEVEL_WINDOW = 31  # frames; ARKit drift is slow


class ReferenceAngleError(ValueError):
    """The reference frame's plate angle is missing; ``valid`` holds the frames whose angle the same
    fit measured and kept, so a retry can start from one of them."""

    def __init__(self, message: str, valid=()):
        super().__init__(message)
        self.valid = list(valid)


def require_reference_angle(centroids, angles, ref: int) -> None:
    if ref not in centroids or ref >= len(angles) or not np.isfinite(angles[ref]):
        valid = [t for t in centroids if t < len(angles) and np.isfinite(angles[t])]
        raise ReferenceAngleError(
            f"The plate's angle on reference frame {ref} was unavailable or rejected as an outlier. "
            "A nearby mask may have switched to a different surface; inspect the plate track "
            "around the reference, re-anchor it on open frames and fit again.", valid)


def fit_hinge_with_reference_retries(c, roles, log=print, max_retries: int = 2, prepare_reference=None):
    """Try a few well-observed open references if the chosen reference angle is unusable.

    Re-basing updates all camera transforms together. Failed angles are never replaced
    with a neighbour's value; each candidate runs the complete hinge fit again. The next
    candidate is the open frame with the largest plate mask among those whose angle the failed
    fit did measure (a glossy plate's plane fit fails on many frames, often on the largest
    view); without that, frames near 20% and 80% of the video.
    """
    tracks = c.by.get(roles.plate, {})
    usable = [t for t in tracks if c.has(roles.base, t) and c.has(roles.plate, t)
              and (not roles.open_indicator or c.has(roles.open_indicator, t))]
    original = c.ref
    spread = []
    for target in (.2 * (c.n - 1), .8 * (c.n - 1)):
        choices = [t for t in usable if t != original and t not in spread]
        if choices:
            spread.append(min(choices, key=lambda t: abs(t - target)))
    attempts = [original]
    for index in range(max(0, max_retries) + 1):
        frame = attempts[-1]
        if index:
            if prepare_reference is None:
                c.set_ref(frame)
            else:
                prepare_reference(c, frame)
            log(f"Retrying hinge fit at open reference frame {frame} after a rejected reference.")
        try:
            result = fit_hinge(c, roles, log=log)
            return result, attempts
        except ValueError as error:
            if "angle on reference frame" not in str(error) or index == max(0, max_retries):
                raise
            log(f"WARNING: {error}")
            measured = set(getattr(error, "valid", ())) & set(usable)
            measured -= set(attempts)
            if measured:
                attempts.append(max(measured, key=lambda t: tracks[t].area))
            else:
                rest = [t for t in spread if t not in attempts]
                if not rest:
                    raise
                attempts.append(rest[0])
    raise AssertionError("at least one reference attempt is required")


def can_use_closed_fallback(c, roles, t: int) -> bool:
    """A visible open base cannot simultaneously stand in for a missing lid."""
    if not roles.open_indicator or not c.has(roles.open_indicator, t):
        return True
    if roles.plate_closed == roles.base:
        return False
    if c.has(roles.plate_closed, t) and c.has(roles.base, t):
        back = c.by[roles.plate_closed][t].full_mask(c.h, c.w)
        base = c.by[roles.base][t].full_mask(c.h, c.w)
        overlap = (back & base).sum()
        if back.any() and base.any() and overlap > .5 * back.sum() and overlap > .8 * base.sum():
            return False
    return True


@dataclass
class Roles:
    floor: str | None
    base: str
    plate: str
    plate_closed: str | None = None
    open_indicator: str | None = None


def _hull(points2d: np.ndarray) -> np.ndarray:
    return cv2.convexHull(points2d.astype(np.float32))[:, 0, :].astype(np.float64)


def _project(intr, x):
    return np.stack([intr.f * x[:, 0] / x[:, 2] + intr.cx, intr.f * x[:, 1] / x[:, 2] + intr.cy], axis=1)


class Capture:
    """A tracked RGB-D video: tracks, depth, intrinsics and the static scene's motion
    (``world``: reference-camera coordinates -> frame-t camera coordinates)."""

    def __init__(self, video: Path, track_dir: Path, cap: stray.Capture, world: np.ndarray, ref: int = 0):
        self.video, self.track_dir, self.ref, self.cap = video, track_dir, ref, cap
        self.names, by = load_parts(track_dir)
        (self.w, self.h), n, _ = video_info(video)
        self.by = {nm: {t: d for t, d in dets.items() if d.usable(self.w, self.h)} for nm, dets in by.items()}
        self.n = min(n, len(cap.depth), len(world))
        self.world = world[: self.n]

    @classmethod
    def from_stray(cls, video: Path, capture: Path, track_dir: Path, ref: int = 0) -> "Capture":
        """An iPhone capture: the scene's motion comes from ARKit odometry."""
        cap = stray.load(capture)
        c = cls(video, track_dir, cap, stray.ref_to_frame_poses(capture, len(cap.depth), ref), ref)
        c.source = capture
        return c

    @classmethod
    def fixed_camera(cls, video: Path, track_dir: Path, depth: np.ndarray, intrinsics, ref: int = 0) -> "Capture":
        """A camera that doesn't move (a tripod, the RBO dataset): the scene's motion is identity."""
        c = cls(video, track_dir, stray.Capture(video, depth, intrinsics), np.tile(np.eye(4), (len(depth), 1, 1)), ref)
        c.moving_camera = False
        return c

    moving_camera = True  # ARKit odometry drifts and gets levelled against the floor
    source: Path | None = None  # the Stray Scanner capture, when there is one

    @property
    def up(self) -> np.ndarray | None:
        """Gravity's up in reference-camera coordinates, from ARKit (None without odometry)."""
        return stray.gravity_up(self.source, self.ref) if self.source is not None else None

    def set_ref(self, t: int):
        """Use frame t as the reference: the scene's motion is re-expressed from it."""
        self.world = self.world @ np.linalg.inv(self.world[t])
        self.ref = t

    def ref_with(self, base: str, plate: str, closed: str | None = None,
                 open_indicator: str | None = None) -> int:
        """A reference frame showing both parts: the one where the plate's mask is largest.
        Angles are measured from it, so a frame with only a sliver of plate (barely open)
        would offset every angle. Frames where the plate's mask is the ``closed`` part's (a
        plate track that slid onto the shut lid's back, largest when shut) don't count, nor
        frames where the base is much smaller than its well-visible views or its mask
        swallows most of the plate."""
        both = set(self.by[base]) & set(self.by[plate]) or set(self.by[plate]) or set(self.by[base])
        if open_indicator:
            # The hinge solver uses presence of this part as evidence that the
            # object is open. Do not silently fit a closed frame if tracking
            # supplies no frame where both measured parts and the indicator exist.
            both = set(self.by[base]) & set(self.by[plate]) & set(self.by.get(open_indicator, {}))
            if not both:
                raise ValueError("No frame has a usable base, plate and open indicator together; "
                                 "inspect the tracked parts before fitting the hinge.")
        if both <= set(self.by[base]):  # the base well in view: a shut lid hides the base's top
            factor = 0.75 if self.moving_camera else 0.5
            well_seen = factor * np.median([self.by[base][t].area for t in both])
            both = {t for t in both if self.by[base][t].area >= well_seen} or both
        if base != plate:
            def unsuitable(t):
                if t not in self.by[plate]:
                    return False
                a = self.by[plate][t].full_mask(self.h, self.w)
                if closed and closed != plate and t in self.by.get(closed, {}):
                    b = self.by[closed][t].full_mask(self.h, self.w)
                    if (a & b).sum() > 0.8 * (a | b).sum():
                        return True
                if self.moving_camera and t in self.by[base] and a.any():
                    base_mask = self.by[base][t].full_mask(self.h, self.w)
                    if (a & base_mask).sum() > 0.5 * a.sum():
                        return True
                    if base_mask.any() and (a & base_mask).sum() > 0.5 * base_mask.sum():
                        return True  # the plate track has swallowed the base
                return False
            # largest first; skip shut ones (checking masks is slow, so stop at the first open one)
            part = self.by[plate] if set(self.by[plate]) & both else self.by[base]
            for t in sorted(both, key=lambda t: -part[t].area):
                if not unsuitable(t):
                    return t
            if open_indicator:
                raise ValueError("No open-indicator frame has a suitable plate and base mask; "
                                 "inspect the tracked parts before fitting the hinge.")
        if not both:
            raise ValueError(f"neither {base} nor {plate} is tracked with a usable mask")
        part = self.by[plate] if set(self.by[plate]) & both else self.by[base]
        return max(both, key=lambda t: part[t].area)

    def level_with(self, floor: str, sample: int = 1500) -> dict:
        """Remove ARKit drift off the floor plane: the floor is static, so on every frame turn
        and shift the scene until that frame's floor plane matches the reference frame's.
        (Drift along the floor stays; a plane can't show it.) Returns the correction sizes."""
        rng = np.random.default_rng(0)

        def plane(t):
            """The floor's plane on frame t, or None where its depth is too sparse to fit one
            (the random draws stay those of the fit either way)."""
            c = self.cloud(floor, t)
            enough = len(c) >= MIN_FLOOR_POINTS
            if len(c) > sample:
                c = c[rng.choice(len(c), sample, replace=False)]
            n, d, _ = fit_plane(c, rng)
            return (n, d) if enough else None

        none = {"frames_used": 0, "tilt_deg_max": 0.0, "shift_mm_max": 0.0, "shift_mm_median": 0.0}
        seen = sorted(self.by.get(floor, {}))
        if not seen:
            return none
        # The floor's plane on the reference frame, or the nearest frame it is seen with depth on
        # (a hand or the opened lid can hide it there; a mask with no depth would fit a made-up
        # plane, and every correction would follow it).
        reference = None
        for t in sorted(seen, key=lambda t: abs(t - self.ref)):
            reference = plane(t)
            if reference is not None:
                break
        if reference is None:
            return {**none, "no_floor_depth": True}
        n0, d0 = reference
        # Per-frame correction as a small rotation vector (turning that frame's floor normal onto
        # the reference one) and a height shift. Drift is slow and plane fits are noisy (a hand
        # or a chair leg in the floor mask), so implausible frames (tilted, or shifted further
        # than drift goes: another surface) are dropped and the rest smoothed over time.
        rotvec = np.full((self.n, 3), np.nan)
        height = np.full(self.n, np.nan)
        for t in range(self.n):
            if not self.has(floor, t):
                continue
            fitted = plane(t)
            if fitted is None:
                continue
            n, d = fitted
            if n @ n0 < 0:
                n, d = -n, -d
            ang = np.arccos(np.clip(n @ n0, -1, 1))
            if ang > np.deg2rad(MAX_LEVEL_TILT_DEG):
                continue
            axis = np.cross(n, n0)
            turn = axis / max(np.linalg.norm(axis), 1e-12) * ang
            shift = d0 - n0 @ (Rotation.from_rotvec(turn).as_matrix() @ (n * d))
            if abs(shift) > MAX_LEVEL_SHIFT_M:
                continue
            rotvec[t], height[t] = turn, shift
        ts = np.arange(self.n)
        ok = ~np.isnan(height)
        for series in (*rotvec.T, height):
            series[:] = np.interp(ts, ts[ok], median_filter(series[ok], size=LEVEL_WINDOW, mode="nearest"))
        for t in range(self.n):
            if t == self.ref:
                continue
            fix = np.eye(4)
            fix[:3, :3] = Rotation.from_rotvec(rotvec[t]).as_matrix()
            fix[:3, 3] = height[t] * n0
            self.world[t] = self.world[t] @ np.linalg.inv(fix)
        tilt = np.rad2deg(np.linalg.norm(rotvec, axis=1))
        return {"frames_used": int(ok.sum()), "tilt_deg_max": float(tilt.max()),
                "shift_mm_max": float(1000 * np.abs(height).max()),
                "shift_mm_median": float(1000 * np.median(np.abs(height)))}

    def follow(self, part: str, exclude: tuple[str, ...] = ()) -> dict:
        """Measure everything relative to ``part`` (the base) instead of the static scene, for
        when the whole object gets moved (a hand shoves the laptop while opening it). The
        part's motion comes from KLT points on its surface lifted with depth (depth-only ICP
        slides along a flat base); ``exclude`` parts (the plate) are masked out of it. Fixed
        camera only: the motion is in camera coordinates. Returns how far the part moved."""
        assert not self.moving_camera, "follow() needs a fixed camera"
        if self.ref not in self.by.get(part, {}):
            return {"frames": 0}  # its points are picked on the reference frame
        grays, k = load_gray(self.video)
        others: dict[int, list] = {}
        for o in exclude:
            for t, det in self.by.get(o, {}).items():
                others.setdefault(t, []).append(det)
        motion = part_poses_3d(grays[: self.n], k, self.cap.depth, self.cap.intrinsics, self.by[part], others,
                               self.ref, (self.w, self.h))
        gap = np.isnan(motion).any(axis=(1, 2))
        if gap.all():
            return {"frames": 0}
        idx = np.flatnonzero(~gap)
        nearest = idx[np.abs(idx[None] - np.arange(self.n)[:, None]).argmin(axis=1)]
        motion = motion[nearest]  # hold the nearest measured pose where the part is hidden
        self.world = self.world @ motion
        moved = np.linalg.norm(motion[:, :3, 3], axis=1)
        turned = np.rad2deg(np.arccos(np.clip((np.trace(motion[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
        return {"frames": int((~gap).sum()), "moved_mm_max": float(1000 * moved.max()), "turned_deg_max": float(turned.max())}

    def merge(self, name: str, parts: list[str]) -> str:
        """A part made of several tracked ones (a cabinet seen as its side panel and its fixed
        drawer fronts): on each frame, the union of their masks. Returns ``name``."""
        merged = {}
        for t in sorted(set().union(*(self.by[p] for p in parts))):
            m = np.zeros((self.h, self.w), bool)
            for p in parts:
                if t in self.by[p]:
                    m |= self.by[p][t].full_mask(self.h, self.w)
            ys, xs = np.nonzero(m)
            if len(xs):
                box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                merged[t] = Detection(t, -1, box, m[box[1]: box[3] + 1, box[0]: box[2] + 1])
        self.by[name] = merged
        self.names.append(name)
        return name

    def drop_where_same(self, part: str, other: str | None, iou: float = 0.8) -> int:
        """Forget ``part`` on frames where its mask is ``other``'s. SAM keeps a screen's identity
        as the lid shuts, so a plate (inner face) track slides onto the closed lid's back, which
        is plate_closed; those frames then go through the plate_closed path, built for them.
        Returns how many frames were dropped; the frames are kept in ``slid_frames``."""
        self.slid_frames = []
        if not other or other == part or part not in self.by or other not in self.by:
            return 0
        drop = []
        for t in sorted(set(self.by[part]) & set(self.by[other])):
            a, b = self.by[part][t].full_mask(self.h, self.w), self.by[other][t].full_mask(self.h, self.w)
            if (a & b).sum() > iou * (a | b).sum():
                drop.append(t)
        for t in drop:
            del self.by[part][t]
        self.slid_frames = drop
        return len(drop)

    def drop_open_base_overlap(self, roles: Roles) -> int:
        """Discard plate masks that substantially coincide with the base on open frames."""
        if not roles.open_indicator or roles.base == roles.plate:
            return 0
        available = set(self.by.get(roles.plate, {})) & set(self.by.get(roles.base, {}))
        available &= set(self.by.get(roles.open_indicator, {}))
        dropped = []
        for t in sorted(available):
            a = self.by[roles.plate][t].full_mask(self.h, self.w)
            b = self.by[roles.base][t].full_mask(self.h, self.w)
            overlap = (a & b).sum()
            if a.any() and b.any() and overlap > .5 * a.sum() and overlap > .8 * b.sum():
                dropped.append(t)
        for t in dropped:
            del self.by[roles.plate][t]
        return len(dropped)

    def has(self, part: str | None, t: int) -> bool:
        return part is not None and part in self.by and t in self.by[part]

    def cloud(self, part: str, t: int) -> np.ndarray:
        """The part's depth points on frame t, in reference-camera coordinates."""
        c = mask_cloud(self.by[part][t].full_mask(self.h, self.w), self.cap.depth[t], self.cap.intrinsics)
        inv = np.linalg.inv(self.world[t])
        return c @ inv[:3, :3].T + inv[:3, 3]


def open_capture(video: Path, source: Path, track_dir: Path) -> Capture:
    """An iPhone capture (Stray Scanner folder) or a fixed-camera RGB-D folder holding
    depth.npz (metres) and intrinsics.json, as scripts/rbo_eval.py prepares them."""
    if (source / "depth.npz").exists() and (source / "intrinsics.json").exists():
        from .lift3d import Intrinsics
        k = json.loads((source / "intrinsics.json").read_text())
        depth = np.load(source / "depth.npz")["depth"].astype(np.float32)
        return Capture.fixed_camera(video, track_dir, depth, Intrinsics(k["f"], k["cx"], k["cy"]))
    return Capture.from_stray(video, source, track_dir)


def base_top(base: np.ndarray, floor: np.ndarray | None) -> tuple[np.ndarray, float]:
    """The base's face the plate closes onto. Its largest plane, unless a floor is known: a
    thick base (a book's page block) can show a side face bigger than its top, so then take,
    of its few largest planes, the one most parallel to the floor. If none is within 30 deg
    of it (the top is barely in view), the object rests on the floor: its top is parallel."""
    rng = np.random.default_rng(0)
    n, d, _ = fit_plane(base, rng)
    if floor is None or len(floor) < 50:
        return n, d
    nf = fit_plane(floor, rng)[0]
    best, rest = (n, d), base
    for _ in range(3):
        if len(rest) < max(200, 0.1 * len(base)):
            break
        n, d, _ = fit_plane(rest, rng)
        on = np.abs(rest @ n - d) < 0.004
        if on.sum() >= 0.1 * len(base) and abs(n @ nf) > abs(best[0] @ nf):
            best = (n, d)
        rest = rest[~on]
    if abs(best[0] @ nf) < np.cos(np.deg2rad(30)):
        if nf @ (-floor.mean(axis=0)) < 0:  # toward the camera, which looks down on the floor
            nf = -nf
        return nf, float(np.percentile(base @ nf, 95))
    return best


DOOR_CHECK_FRAMES = 60  # plate frames sampled to tell a door before the hinge fit


def looks_like_door(c: Capture, roles: Roles, base_normal: np.ndarray) -> bool:
    """A door on a vertical hinge, told from the plate track's own planes on frames spread over
    the video (lift3d.is_door), before the closed-part fallback, which is built for a lid that
    shuts onto the base's top. Uses its own random draws: the hinge fit's stay as they were."""
    frames = sorted(c.by.get(roles.plate, {}))
    rng = np.random.default_rng(1)
    normals = []
    for t in frames[:: max(1, len(frames) // DOOR_CHECK_FRAMES)]:
        cloud = c.cloud(roles.plate, t)
        if len(cloud) >= 50:
            n, _, frac = fit_plane(cloud, rng)
            if frac >= 0.6:
                normals.append(n)
    return len(normals) >= 5 and is_door(np.array(normals), base_normal)


def fit_hinge(c: Capture, roles: Roles, log=print) -> dict:
    """Fit the plate's hinge; writes hinge.json and hinge.jpg. Returns a summary dict."""
    if not c.has(roles.plate, c.ref):
        raise ValueError(f"The plate's angle on reference frame {c.ref} cannot be used: its tracked face "
                         "is unavailable there. Inspect and re-anchor the plate on an open frame before fitting.")
    seen = sorted(c.by[roles.base])
    if not seen:
        raise ValueError(f"{roles.base} has no usable mask on any frame")
    base = c.cloud(roles.base, min(seen, key=lambda t: abs(t - c.ref)))  # the base is static here
    nb, db = base_top(base, c.cloud(roles.floor, c.ref) if c.has(roles.floor, c.ref) else None)
    if nb @ (-base.mean(axis=0)) < 0:  # point the base's normal up, toward the camera
        nb, db = -nb, -db
    up = floor_up(c, roles)  # the base's top may be out of view (a door's base is a side part)
    door = looks_like_door(c, roles, nb if up is None else up)
    plate, used_closed, used_open = {}, 0, 0
    raised = []  # frames measured from plate_closed's points off the base: its track holds the raised plate
    for t in range(c.n):
        if c.has(roles.plate, t):
            plate[t] = c.cloud(roles.plate, t)
        elif c.has(roles.plate_closed, t):
            if not can_use_closed_fallback(c, roles, t):
                # The base is visible while open. If the plate track has been
                # discarded, its base mask is not evidence of the lid's back.
                # Leave a gap instead of turning the fixed base into a plate.
                continue
            whole = c.cloud(roles.plate_closed, t)
            if door:  # a door shuts beside its base, not onto its top: the closed part is its face
                plate[t] = whole
                used_closed += 1
                continue
            # The plate's face isn't tracked: half open with the face turned away, or closed.
            # Decide by geometry, not by whether the open-indicator is seen (a lid half way
            # down already hides the keyboard from a camera in front): points well above the
            # base's top plane are the plate (below it are the base's own sides); if there are
            # hardly any, the object is closed and its top face is the plate's back.
            off = whole[whole @ nb - db > OFF_BASE_M]
            if len(off) > max(200, 0.2 * len(whole)):
                plate[t] = off
                used_open += 1
                raised.append(t)
            else:  # closed: its top face is the plate's back (not a box's walls below it)
                top = whole[whole @ nb - db > -OFF_BASE_M]
                plate[t] = top if len(top) >= 50 else whole
                used_closed += 1
    # Frames known shut (no open indicator) place a door's hinge on its closed face.
    shut = {t for t in plate if not c.has(roles.open_indicator, t)} if roles.open_indicator else None
    hinge = fit_planar_hinge(plate, c.n, c.ref, pin=base.mean(axis=0), base_normal=nb, closed_frames=shut,
                             door=True if door else None)
    raw = int((~np.isnan(hinge.q)).sum())
    hinge.q = clean_angles(hinge.q)
    if hinge.placed_by != "planes" and shut:  # a door: nothing swings through its cabinet
        hinge.q = drop_overclosed(hinge.q, shut)
    require_reference_angle(hinge.centroids, hinge.q, c.ref)
    valid = ~np.isnan(hinge.q)

    def radial(x):
        v = x - hinge.point
        v -= (v @ hinge.axis) * hinge.axis
        return v / np.linalg.norm(v)

    # Opening per frame, measured directly between the plate and the base about the hinge
    # (0 = closed): independent of the sign the axis happened to get.
    u_base = radial(base.mean(axis=0))
    open_deg = {t: float(np.rad2deg(np.arccos(np.clip(radial(m) @ u_base, -1, 1)))) for t, m in hinge.centroids.items()
                if not np.isnan(hinge.q[t])}
    opening = open_deg[c.ref]
    q = np.rad2deg(hinge.q[valid])
    along = (plate[c.ref] - hinge.point) @ hinge.axis if c.ref in plate else np.zeros(2)
    summary = {
        "frames_with_plate": int(valid.sum()), "frames": c.n, "from_closed_part": used_closed,
        "outliers_dropped": raw - int(valid.sum()),
        "hinge_to_base_plane_deg": float(np.rad2deg(np.arcsin(abs(hinge.axis @ nb)))),
        "plane_residual_mm": 1000 * hinge.residual,
        "opening_on_reference_deg": opening,
        "opening_range_deg": [min(open_deg.values()), max(open_deg.values())],
        "plate_span_along_hinge_mm": float(1000 * np.ptp(along)),
    }
    summary["from_open_object"] = used_open
    log(f"hinge: plate planes on {summary['frames_with_plate']}/{c.n} frames ({used_closed} closed and {used_open} "
        f"half-open from {roles.plate_closed}), "
        f"{summary['outliers_dropped']} outliers dropped; axis {summary['hinge_to_base_plane_deg']:.1f} deg off the "
        f"base plane; opening {opening:.0f} deg on the reference frame, "
        f"{summary['opening_range_deg'][0]:.0f}..{summary['opening_range_deg'][1]:.0f} deg over the video; "
        f"plate spans {summary['plate_span_along_hinge_mm']:.0f} mm along the hinge")
    if hinge.placed_by != "planes":
        summary["hinge_placed_by"] = hinge.placed_by
        log(f"door: hinge placed by its {hinge.placed_by}" + (
            " (on the shut door's plane, where the open frames' door points stay equidistant)"
            if hinge.placed_by == "door closed face" else
            " (a circle through the door's centroids: no shut frames to place it on the closed face)"))
    from .track_checks import reference_opening_warning
    warning = reference_opening_warning(opening, roles.open_indicator)
    if warning:
        log(warning)

    image = read_frame(c.video, c.ref)
    ends = hinge.point + np.outer([along.min(), along.max()], hinge.axis)
    (u0, v0), (u1, v1) = _project(c.cap.intrinsics, ends).astype(int)
    cv2.line(image, (u0, v0), (u1, v1), (0, 0, 255), 10)
    cv2.putText(image, "measured hinge", (u0, v0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
    strip = np.full((300, c.w, 3), 255, np.uint8)
    ts = np.flatnonzero(valid)
    lo, hi = q.min(), q.max()
    pts = np.stack([ts * (c.w - 1) / max(c.n - 1, 1), 280 - (q - lo) / max(hi - lo, 1e-6) * 260], axis=1).astype(int)
    for run in np.split(np.arange(len(ts)), np.flatnonzero(np.diff(ts) > 3) + 1):  # gaps stay gaps
        cv2.polylines(strip, [pts[run]], False, (200, 60, 0), 4)
    cv2.putText(strip, f"opening over time: {summary['opening_range_deg'][0]:.0f}..{summary['opening_range_deg'][1]:.0f} deg", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.imwrite(str(c.track_dir / "hinge.jpg"), cv2.vconcat([image, strip]))
    (c.track_dir / "hinge.json").write_text(json.dumps({
        "reference_frame": c.ref, "axis": hinge.axis.tolist(), "point": hinge.point.tolist(),
        "residual_m": hinge.residual, "q": [None if np.isnan(v) else float(v) for v in hinge.q], **summary}))
    return {**summary, "raised_from_closed_frames": raised}


def build_model(c: Capture, roles: Roles, log=print) -> dict:
    """URDF/MJCF from hinge.json plus shapes measured from depth, and a replay over the video
    (replay3d.mp4). Returns a summary dict, including how much of each SAM mask the replayed
    model covers per frame."""
    hinge = json.loads((c.track_dir / "hinge.json").read_text())
    ref, n, h, w = c.ref, c.n, c.h, c.w
    rng = np.random.default_rng(0)
    axis, point = np.array(hinge["axis"]), np.array(hinge["point"])
    q = np.array([np.nan if v is None else v for v in hinge["q"]])

    up = floor_up(c, roles)
    if up is not None and abs(axis @ up) > np.cos(np.deg2rad(VERTICAL_HINGE_DEG)):
        return build_door_model(c, roles, hinge, up, log=log)

    # Model frame: z up off the base, x along the hinge, origin on the hinge.
    seen = sorted(c.by[roles.base])
    base_c = c.cloud(roles.base, min(seen, key=lambda t: abs(t - ref)))
    floor_c = c.cloud(roles.floor, ref) if c.has(roles.floor, ref) else None
    if floor_c is not None and len(floor_c) < 50:
        floor_c = None
    nb, db, _ = fit_plane(base_c, rng)
    top = base_c[np.abs(base_c @ nb - db) < 0.005]
    if nb @ (-top.mean(axis=0)) < 0:  # the camera (origin) is above the base
        nb = -nb
    x = axis - (axis @ nb) * nb
    x /= np.linalg.norm(x)
    rot = np.stack([x, np.cross(nb, x), nb])  # rows: model axes in camera coordinates

    def to_model(p):
        return (p - point) @ rot.T

    top_m = to_model(top)
    z_top = float(np.median(top_m[:, 2]))
    z_floor = float(np.median(to_model(floor_c)[:, 2])) if floor_c is not None else z_top - BASE_FALLBACK_M
    # When the floor is not visible at the reference frame, z_floor is a fallback;
    # a plausible fallback must never be reported as a depth measurement.
    base_measured = floor_c is not None and 0.002 < z_top - z_floor < 0.2
    if not base_measured:  # a fragmented base mask: keep the model valid and say so
        log(f"base thickness reads {1000 * (z_top - z_floor):.0f} mm: not measured, using "
            f"{1000 * BASE_FALLBACK_M:.0f} mm")
        z_floor = z_top - BASE_FALLBACK_M

    # Plate on the reference frame: along the hinge (e1), away from it in its plane (e2), across (e3).
    plate_m = to_model(c.cloud(roles.plate, ref))
    centre = plate_m.mean(axis=0)
    e1 = np.array([1.0, 0, 0])
    e2 = np.array([0, centre[1], centre[2]]) / np.linalg.norm(centre[1:])
    e3 = np.cross(e1, e2)
    if e3 @ (top_m.mean(axis=0) - centre) < 0:  # e3 points from the plate's back to its inner face
        e3 = -e3
    s, r = plate_m @ e1, plate_m @ e2
    w_face = float(np.median(plate_m @ e3))

    # Closed frames: the plate's back shows its full outline and how thick it is.
    # q's sign follows the fitted axis, so the closed end is the end of its range that sits
    # the reference frame's opening away from the reference (q = 0), not simply the low end.
    shut = np.deg2rad(hinge["opening_on_reference_deg"])
    lo, hi = np.nanmin(q), np.nanmax(q)
    toward = 1.0 if abs(hi - shut) < abs(lo + shut) else -1.0
    near_shut = toward * q > np.nanpercentile(toward * q, 90)
    closed = [t for t in range(n) if not np.isnan(q[t]) and near_shut[t]
              and c.has(roles.plate_closed, t) and not c.has(roles.open_indicator, t)]
    heights, back = [], []
    for t in closed[:: max(1, len(closed) // 40)]:
        pm = to_model(c.cloud(roles.plate_closed, t))
        if len(pm) > 100:
            heights.append(np.median(pm[:, 2]) - z_top)
            cq, sq = np.cos(-q[t]), np.sin(-q[t])  # turn it back to the reference pose about the hinge
            back.append(pm @ np.array([[1, 0, 0], [0, cq, -sq], [0, sq, cq]]).T)
    thick = float(np.clip(np.median(heights), 0.001, 0.03)) if heights else 0.006

    mesh_dir = c.track_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    image = read_frame(c.video, ref)
    base_rgb = part_color(image, c.by[roles.base][ref].full_mask(h, w))
    screen_rgb = part_color(image, c.by[roles.plate][ref].full_mask(h, w))
    plate_rgb = base_rgb
    if closed:
        t = closed[len(closed) // 2]
        plate_rgb = part_color(read_frame(c.video, t), c.by[roles.plate_closed][t].full_mask(h, w))

    v, f = prism(_hull(top_m[:, :2]), np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                 np.array([0, 0, 1.0]), z_floor, z_top)
    write_stl(mesh_dir / "base.stl", v, f)
    sr = np.column_stack([s, r])
    outline = [sr, [[s.min(), 0.0], [s.max(), 0.0]]]  # the plate reaches the hinge
    plate_size = (np.ptp(s), r.max())
    if back:
        b_ = np.vstack(back)
        b_ = b_[np.abs(b_ @ e3 - np.median(b_ @ e3)) < 0.01]  # on the plate, not the base's edge
        lo, hi = np.percentile(np.column_stack([b_ @ e1, b_ @ e2]), [1, 99], axis=0)
        outline.append(np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]))
        plate_size = (hi[0] - lo[0], hi[1] - lo[1])
    v, f = prism(_hull(np.vstack(outline)), np.zeros(3), e1, e2, e3, w_face - thick, w_face)
    write_stl(mesh_dir / "lid.stl", v, f)
    v, f = prism(_hull(sr), np.zeros(3), e1, e2, e3, w_face, w_face + SCREEN_DEPTH)
    write_stl(mesh_dir / "screen.stl", v, f)

    valid = q[~np.isnan(q)]
    limits = (float(np.percentile(valid, 1)), float(np.percentile(valid, 99)))
    links = [
        emit3d.Link3D("base", [emit3d.Visual("meshes/base.stl", base_rgb)]),
        emit3d.Link3D("lid", [emit3d.Visual("meshes/lid.stl", plate_rgb),
                              emit3d.Visual("meshes/screen.stl", screen_rgb, collide=False)],
                      parent="base", joint_type="revolute", axis=np.array([1.0, 0, 0]), limits=limits),
    ]
    root_pos = np.array([0, 0, -z_floor])  # floor at z = 0
    (c.track_dir / "model.urdf").write_text(emit3d.urdf(c.video.stem, links))
    (c.track_dir / "model.xml").write_text(emit3d.mjcf(c.video.stem, links, root_pos, (w, h)))
    summary = {
        "base_thickness_mm": 1000 * (z_top - z_floor), "base_thickness_measured": base_measured,
        "plate_thickness_mm": 1000 * thick,
        "plate_thickness_clipped": not heights or not 0.001 < float(np.median(heights)) < 0.03,
        "plate_size_mm": [float(1000 * plate_size[0]), float(1000 * plate_size[1])],
        "hinge_range_deg": [float(np.rad2deg(limits[0])), float(np.rad2deg(limits[1]))],
        "closed_frames_used": len(heights),
    }
    (c.track_dir / "model3d.json").write_text(json.dumps({
        "reference_frame": ref, "camera_to_model": {"rotation": rot.tolist(), "hinge_point": point.tolist(),
                                                    "root_pos": root_pos.tolist()},
        "base_thickness_m": z_top - z_floor, "lid_thickness_m": thick, "hinge_limits_rad": limits, **summary}))
    return replay_hinged(c, roles, rot, point, root_pos, q, summary, log)


def floor_up(c: Capture, roles: Roles) -> np.ndarray | None:
    """Up, in reference-camera coordinates: the floor's normal toward the camera, else gravity."""
    if roles.floor and c.has(roles.floor, c.ref):
        f = c.cloud(roles.floor, c.ref)
        if len(f) >= 50:
            up, _, _ = fit_plane(f, np.random.default_rng(0))
            return up if up @ (-f.mean(axis=0)) > 0 else -up
    return None if c.up is None else np.asarray(c.up, float)


def _turn(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) * np.cos(angle) + np.sin(angle) * k + (1 - np.cos(angle)) * np.outer(axis, axis)


def build_door_model(c: Capture, roles: Roles, hinge: dict, up: np.ndarray, log=print) -> dict:
    """A plate on a vertical hinge beside its base (a microwave's or cupboard's door). Model frame:
    z up (the floor's normal), x horizontal from the hinge toward the door on the reference frame,
    origin on the hinge; the joint turns about the measured axis. The door's outline gathers its
    points from open frames turned back to the reference pose; its thickness isn't seen face-on
    and is set (DOOR_THICKNESS_M). The body is a box from the floor to the base's top over the
    base's footprint as seen: its hidden depth is unmeasured."""
    ref, n, h, w = c.ref, c.n, c.h, c.w
    axis, point = np.array(hinge["axis"]), np.array(hinge["point"])
    q = np.array([np.nan if v is None else v for v in hinge["q"]])
    if axis @ up < 0:  # turn about +up: the joint value keeps its sense about the axis
        axis, q = -axis, -q
    plate_ref = c.cloud(roles.plate, ref)
    out = plate_ref.mean(axis=0) - point
    out -= (out @ up) * up
    out /= np.linalg.norm(out)
    rot = np.stack([out, np.cross(up, out), up])

    def to_model(p):
        return (p - point) @ rot.T

    seen = sorted(c.by[roles.base])
    base_m = to_model(c.cloud(roles.base, min(seen, key=lambda t: abs(t - ref))))
    floor_c = c.cloud(roles.floor, ref) if roles.floor and c.has(roles.floor, ref) else None
    measured_floor = floor_c is not None and len(floor_c) >= 50
    z_floor = float(np.median(to_model(floor_c)[:, 2])) if measured_floor else float(np.percentile(base_m[:, 2], 2))
    z_top = float(np.percentile(base_m[:, 2], 98))

    a_m = rot @ axis  # the hinge in the model frame, nearly +z
    pm = to_model(plate_ref)
    centre = pm.mean(axis=0)
    e2 = centre - (centre @ a_m) * a_m
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(a_m, e2)
    w_face = float(np.median(pm @ e3))
    # Open frames, each turned back to the reference pose about the hinge, fill in the door.
    frames = [t for t in range(n) if c.has(roles.plate, t) and not np.isnan(q[t])]
    pooled = [pm]
    for t in frames[:: max(1, len(frames) // 40)]:
        p_t = to_model(c.cloud(roles.plate, t))
        if len(p_t) > 100:
            pooled.append(p_t @ _turn(a_m, -q[t]).T)
    pts = np.vstack(pooled)
    pts = pts[np.abs(pts @ e3 - w_face) < 0.03]  # on the door's plane, not the cabinet behind it
    s_, r_ = pts @ a_m, pts @ e2
    lo, hi = np.percentile(np.column_stack([s_, r_]), [1, 99], axis=0)
    thick = DOOR_THICKNESS_M

    mesh_dir = c.track_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    image = read_frame(c.video, ref)
    base_rgb = part_color(image, c.by[roles.base][ref].full_mask(h, w)) if c.has(roles.base, ref) else (0.5, 0.5, 0.5)
    door_rgb = part_color(image, c.by[roles.plate][ref].full_mask(h, w))
    v, f = prism(_hull(base_m[:, :2]), np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                 np.array([0, 0, 1.0]), z_floor, z_top)
    write_stl(mesh_dir / "base.stl", v, f)
    door = np.array([[lo[0], 0.0], [hi[0], 0.0], [hi[0], hi[1]], [lo[0], hi[1]]])  # the door reaches its hinge
    v, f = prism(door, np.zeros(3), a_m, e2, e3, w_face - thick / 2, w_face + thick / 2)
    write_stl(mesh_dir / "lid.stl", v, f)

    valid = q[~np.isnan(q)]
    limits = (float(np.percentile(valid, 1)), float(np.percentile(valid, 99)))
    links = [emit3d.Link3D("base", [emit3d.Visual("meshes/base.stl", base_rgb)]),
             emit3d.Link3D("lid", [emit3d.Visual("meshes/lid.stl", door_rgb)], parent="base",
                           joint_type="revolute", axis=a_m, limits=limits)]
    root_pos = np.array([0, 0, -z_floor])
    (c.track_dir / "model.urdf").write_text(emit3d.urdf(c.video.stem, links))
    (c.track_dir / "model.xml").write_text(emit3d.mjcf(c.video.stem, links, root_pos, (w, h)))
    summary = {
        "hinge_vertical": True,
        "base_thickness_mm": 1000 * (z_top - z_floor), "base_thickness_measured": measured_floor,
        "plate_thickness_mm": 1000 * thick, "plate_thickness_clipped": True,
        "plate_size_mm": [float(1000 * (hi[0] - lo[0])), float(1000 * hi[1])],
        "hinge_range_deg": [float(np.rad2deg(limits[0])), float(np.rad2deg(limits[1]))],
        "closed_frames_used": 0, "door_frames_pooled": len(pooled),
    }
    log(f"door: the hinge stands {np.rad2deg(np.arccos(min(1.0, abs(axis @ up)))):.1f} deg off vertical; door "
        f"{summary['plate_size_mm'][1]:.0f} mm wide x {summary['plate_size_mm'][0]:.0f} mm tall from {len(pooled)} "
        f"views turned back to the reference; its thickness is not seen and is set to {1000 * thick:.0f} mm; the "
        f"body is {summary['base_thickness_mm']:.0f} mm tall over its seen footprint (depth behind it unmeasured)")
    (c.track_dir / "model3d.json").write_text(json.dumps({
        "reference_frame": ref, "camera_to_model": {"rotation": rot.tolist(), "hinge_point": point.tolist(),
                                                    "root_pos": root_pos.tolist()},
        "base_thickness_m": z_top - z_floor, "lid_thickness_m": thick, "hinge_limits_rad": limits, **summary}))
    return replay_hinged(c, roles, rot, point, root_pos, q, summary, log)


def replay_hinged(c: Capture, roles: Roles, rot, point, root_pos, q, summary: dict, log=print) -> dict:
    """Replay a written base + lid model from the phone's camera over the video, score it against
    the masks, and add the coverage to model3d.json and the summary."""
    n = c.n
    a = np.eye(4)
    a[:3, :3], a[:3, 3] = rot, -rot @ point + root_pos
    cam_to_world = np.array([a @ np.linalg.inv(c.world[t]) for t in range(n)])
    cover = replay3d(c.track_dir / "model.xml", cam_to_world, c.cap.intrinsics, {"lid_joint": q},
                     {"lid": c.by[roles.plate], "base": c.by[roles.base]}, c.video, c.track_dir / "replay3d.mp4",
                     only_when={"base": lambda t: c.has(roles.open_indicator, t) if roles.open_indicator else True})
    summary["mask_coverage"] = {k: float(np.nanmean(v)) for k, v in cover.items()}
    summary["mask_coverage_frames"] = {k: int(np.isfinite(v).sum()) for k, v in cover.items()}
    summary["worst_frames"] = {k: int(np.nanargmin(v)) for k, v in cover.items() if not np.isnan(v).all()}
    metadata = json.loads((c.track_dir / "model3d.json").read_text())
    metadata["mask_coverage"] = {k: v if np.isfinite(v) else None for k, v in summary["mask_coverage"].items()}
    metadata["worst_frames"] = summary["worst_frames"]
    metadata["mask_coverage_frames"] = summary["mask_coverage_frames"]
    metadata["video_frames"] = c.n
    (c.track_dir / "model3d.json").write_text(json.dumps(metadata, indent=2))
    log(f"model: base {summary['base_thickness_mm']:.0f} mm thick; plate {summary['plate_size_mm'][0]:.0f} x "
        f"{summary['plate_size_mm'][1]:.0f} mm, {summary['plate_thickness_mm']:.1f} mm thick"
        f"{' (clipped: not really measured)' if summary['plate_thickness_clipped'] else ''}; hinge "
        f"{summary['hinge_range_deg'][0]:.0f}..{summary['hinge_range_deg'][1]:.0f} deg; replay covers "
        + ", ".join(f"{k} {100 * v:.0f}%" for k, v in summary["mask_coverage"].items()) + " of the SAM masks")
    return summary


def fit_and_build_slide(c: Capture, roles: Roles, log=print) -> dict:
    """A part that slides out of a base (a drawer): fit the travel from depth (translation-only
    ICP of the part's cloud, see lift3d.fit_slide), then write a model with box links measured
    from the reference frame and replay it. ``roles.plate`` is the sliding part; plate_closed
    and open_indicator are unused. Writes slide.json, model.urdf, model.xml, meshes/,
    model3d.json and replay3d.mp4."""
    clouds = {t: c.cloud(roles.plate, t) for t in range(c.n) if c.has(roles.plate, t)}
    slide = fit_slide(clouds, c.n, c.ref)
    q = clean_angles(slide.q, max_dev=0.02)  # 2 cm off its neighbours is a bad frame
    valid = ~np.isnan(q)
    lo, hi = float(np.percentile(q[valid], 1)), float(np.percentile(q[valid], 99))

    # Model frame: z up off the floor (or the camera's up), x along the travel, origin at the
    # sliding part's centroid on the reference frame.
    if roles.floor:
        f = c.cloud(roles.floor, c.ref)
        up, _, _ = fit_plane(f, np.random.default_rng(0))
        if up @ (-f.mean(axis=0)) < 0:  # the camera is above the floor
            up = -up
    else:
        up = np.array([0.0, -1.0, 0.0])  # the reference camera's up
    up = up - (up @ slide.axis) * slide.axis
    up /= np.linalg.norm(up)
    rot = np.stack([slide.axis, np.cross(up, slide.axis), up])
    point = slide.point

    def to_model(p):
        return (p - point) @ rot.T

    def box_of(p):  # robust axis-aligned extent in the model frame
        return np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)

    mesh_dir = c.track_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    image = read_frame(c.video, c.ref)
    links, sizes = [], {}
    for name, part, back in (("base", roles.base, 0.0), ("drawer", roles.plate, hi - lo)):
        lo_b, hi_b = box_of(to_model(c.cloud(part, c.ref)))
        if name == "drawer":
            lo_b[0] -= back  # the part hidden inside the base: at least as deep as it travels
        rect = np.array([[lo_b[0], lo_b[1]], [hi_b[0], lo_b[1]], [hi_b[0], hi_b[1]], [lo_b[0], hi_b[1]]])
        v, fc = prism(rect, np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]),
                      lo_b[2], hi_b[2])
        write_stl(mesh_dir / f"{name}.stl", v, fc)
        sizes[name] = (1000 * (hi_b - lo_b)).round().tolist()
        rgb = part_color(image, c.by[part][c.ref].full_mask(c.h, c.w))
        links.append(emit3d.Link3D(name, [emit3d.Visual(f"meshes/{name}.stl", rgb)],
                                   parent=None if name == "base" else "base", joint_type="prismatic",
                                   axis=np.array([1.0, 0, 0]), limits=(lo, hi)))
    links[0].joint_type = "fixed"
    floor_z = float(np.percentile(to_model(c.cloud(roles.floor, c.ref))[:, 2], 50)) if roles.floor else \
        float(min(box_of(to_model(c.cloud(p, c.ref)))[0][2] for p in (roles.base, roles.plate)))
    root_pos = np.array([0, 0, -floor_z])
    (c.track_dir / "model.urdf").write_text(emit3d.urdf(c.video.stem, links))
    (c.track_dir / "model.xml").write_text(emit3d.mjcf(c.video.stem, links, root_pos, (c.w, c.h)))
    summary = {"joint": "prismatic", "travel_range_mm": [1000 * lo, 1000 * hi], "frames_measured": int(valid.sum()),
               "frames": c.n, "line_residual_mm": 1000 * slide.residual, "box_sizes_mm": sizes}
    (c.track_dir / "slide.json").write_text(json.dumps({
        "reference_frame": c.ref, "axis": slide.axis.tolist(), "point": point.tolist(),
        "q": [None if np.isnan(v) else float(v) for v in q], **summary}))
    (c.track_dir / "model3d.json").write_text(json.dumps({
        "reference_frame": c.ref, "camera_to_model": {"rotation": rot.tolist(), "hinge_point": point.tolist(),
                                                      "root_pos": root_pos.tolist()}, **summary}))
    a = np.eye(4)
    a[:3, :3], a[:3, 3] = rot, -rot @ point + root_pos
    cam_to_world = np.array([a @ np.linalg.inv(c.world[t]) for t in range(c.n)])
    cover = replay3d(c.track_dir / "model.xml", cam_to_world, c.cap.intrinsics, {"drawer_joint": q},
                     {"drawer": c.by[roles.plate], "base": c.by[roles.base]}, c.video, c.track_dir / "replay3d.mp4")
    summary["mask_coverage"] = {k: float(np.nanmean(v)) for k, v in cover.items()}
    summary["mask_coverage_frames"] = {k: int(np.isfinite(v).sum()) for k, v in cover.items()}
    metadata = json.loads((c.track_dir / "model3d.json").read_text())
    metadata["mask_coverage"] = {k: v if np.isfinite(v) else None for k, v in summary["mask_coverage"].items()}
    metadata["mask_coverage_frames"] = summary["mask_coverage_frames"]
    metadata["video_frames"] = c.n
    (c.track_dir / "model3d.json").write_text(json.dumps(metadata, indent=2))
    log(f"slide: travel {1000 * lo:.0f}..{1000 * hi:.0f} mm measured on {summary['frames_measured']}/{c.n} frames, "
        f"{1000 * slide.residual:.1f} mm off a straight line; boxes (mm) base {sizes['base']}, drawer {sizes['drawer']} "
        f"(its hidden depth is assumed equal to the travel); replay covers "
        + ", ".join(f"{k} {100 * v:.0f}%" for k, v in summary["mask_coverage"].items()) + " of the SAM masks")
    return summary


def _planes(points: np.ndarray, rng, k: int = 4, min_frac: float = 0.08):
    """Up to k largest planes (normal, offset) by sequential RANSAC."""
    out, rest = [], points
    for _ in range(k):
        if len(rest) < max(100, min_frac * len(points)):
            break
        n, d, _ = fit_plane(rest, rng, tol=0.006)
        on = np.abs(rest @ n - d) < 0.006
        if on.sum() < min_frac * len(points):
            break
        out.append((n, d))
        rest = rest[~on]
    return out


def front_normal(c: Capture, base: str, part: str, window: tuple[int, int], up: np.ndarray):
    """Find an upright base face near a visible drawer, preferring a large drawer mask."""
    rng = np.random.default_rng(0)
    seen = [t for t in range(window[0], min(c.n, window[1] + 1)) if c.has(part, t) and c.has(base, t)]
    if not seen:
        return None
    largest = max(seen, key=lambda t: c.by[part][t].area)
    # The largest mask can coincide with an occluded or fragmented cabinet.
    # Check a few other times rather than declaring the entire video unusable.
    candidates = [largest] + [seen[i] for i in np.unique(np.linspace(0, len(seen) - 1, min(12, len(seen))).astype(int))
                              if seen[i] != largest]
    for ref in candidates:
        drawer = c.cloud(part, ref)
        if len(drawer) < 50:
            continue
        upright = [(n, d) for n, d in _planes(c.cloud(base, ref), rng) if abs(n @ up) < 0.35]
        if upright:
            gm = drawer.mean(axis=0)
            break
    else:
        return None
    # Which face: the one whose normal follows how the drawer moved (seen smallest = shut, to
    # seen largest = out); a side panel is upright too, and the drawer may stand far from it.
    with_depth = [t for t in seen if len(c.cloud(part, t)) >= 50] or [ref]
    shut = min(with_depth, key=lambda t: c.by[part][t].area)
    moved = gm - c.cloud(part, shut).mean(axis=0)
    moved -= (moved @ up) * up
    if np.linalg.norm(moved) > 0.03:
        n, d = max(upright, key=lambda nd: abs(nd[0] @ moved))
        return (n if n @ moved > 0 else -n), ref
    n, d = max(upright, key=lambda nd: abs(gm @ nd[0] - nd[1]))
    return (n if gm @ n - d > 0 else -n), ref


def front_travel(c: Capture, base: str, part: str, window: tuple[int, int], axis: np.ndarray) -> dict:
    """How far a drawer stands out of the base along ``axis`` (the front's outward normal), in
    each frame: the drawer's outermost points against the base's (its front face, or its top's
    front edge when only the top is seen). Both come from the same frame's depth, so camera
    drift cancels. Missing observations remain unknown: occlusion or tracking
    failure does not imply that the drawer is shut."""
    t0, t1 = window
    q = np.full(c.n, np.nan)
    for t in range(t0, min(c.n, t1 + 1)):
        if not c.has(base, t):
            continue
        if not c.has(part, t):
            continue
        dr, cab = c.cloud(part, t), c.cloud(base, t)
        if len(dr) < 50 or len(cab) < 200:
            continue
        q[t] = max(0.0, float(np.percentile(dr @ axis, 98) - np.percentile(cab @ axis, 97)))
    seen = [t for t in range(t0, min(c.n, t1 + 1)) if c.has(part, t) and c.has(base, t)]
    if not seen or not np.isfinite(q).any():
        raise ValueError(f"No usable joint observations for {part!r}; inspect and retrack the drawer and base.")
    return {"ref": max(seen, key=lambda t: c.by[part][t].area), "q": q}


PANEL_M = 0.012  # drawer and carcass panels, where a front view can't measure them
FRONT_M = 0.018  # drawer front thickness
GAP_M = 0.003  # between a drawer and its slot
INSET_M = 0.008  # a drawer's tray inside its front's outline


def _subtract(rect: tuple, holes: list[tuple]) -> list[tuple]:
    """Rectangles (y0, z0, y1, z1) covering ``rect`` minus ``holes``."""
    out = [rect]
    for hy0, hz0, hy1, hz1 in holes:
        nxt = []
        for y0, z0, y1, z1 in out:
            if hy0 >= y1 or hy1 <= y0 or hz0 >= z1 or hz1 <= z0:
                nxt.append((y0, z0, y1, z1))
                continue
            if hy0 > y0:
                nxt.append((y0, z0, hy0, z1))
            if hy1 < y1:
                nxt.append((hy1, z0, y1, z1))
            if hz0 > z0:
                nxt.append((max(y0, hy0), z0, min(y1, hy1), hz0))
            if hz1 < z1:
                nxt.append((max(y0, hy0), hz1, min(y1, hy1), z1))
        out = nxt
    return [r for r in out if r[2] - r[0] > 1e-3 and r[3] - r[1] > 1e-3]


def _box_visual(mesh_dir: Path, name: str, lo, hi, rgb) -> "emit3d.Visual":
    rect = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    v, fc = prism(rect, np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), lo[2], hi[2])
    write_stl(mesh_dir / f"{name}.stl", v, fc)
    return emit3d.Visual(f"meshes/{name}.stl", rgb)


def fit_and_build_slides(c: Capture, base: str, floor: str | None, drawers: list[tuple[str, tuple[int, int]]],
                         log=print) -> dict:
    """Several parts sliding out of one base (a chest of drawers), each operated in its own
    stretch of the video: ``drawers`` is [(tracked part, (first frame, last frame))]. Each
    drawer's travel is how far it stands out of the base's front in each frame (front_travel),
    so all drawers slide along the front's normal; outside its window a drawer is shut.
    q = 0 is shut, positive pulls it out. Writes slides.json, model.urdf, model.xml, meshes/,
    model3d.json and replay3d.mp4."""
    rng = np.random.default_rng(0)
    if c.up is not None:  # gravity from ARKit: right whatever was picked as the floor
        up = c.up
    elif floor:
        f = c.cloud(floor, c.ref)
        up, _, _ = fit_plane(f, rng)
        if up @ (-f.mean(axis=0)) < 0:
            up = -up
    else:
        up = np.array([0.0, -1.0, 0.0])
    # One outward direction for all: drawers of one base slide parallel. Take it where a drawer
    # comes out of an upright face that is in view.
    found = [fn for part, w in drawers if (fn := front_normal(c, base, part, w, up)) is not None]
    if not found:
        raise ValueError(f"no upright face of {base} seen next to a drawer")
    axis = found[0][0] - (found[0][0] @ up) * up
    axis /= np.linalg.norm(axis)
    # Which way is out: drawers only ever stand out of the front, so out is the sign along
    # which they stand out farther (how a drawer's centroid moves is no guide: opening it also
    # reveals its inside, behind its front).
    trials = {sg: [front_travel(c, base, part, w, sg * axis) for part, w in drawers] for sg in (1.0, -1.0)}
    sign = max(trials, key=lambda sg: sum(np.nanpercentile(m["q"], 95) for m in trials[sg]))
    axis = sign * axis
    fits = []
    for (part, (t0, t1)), m in zip(drawers, trials[sign]):
        m["axis"] = axis
        q = clean_angles(m["q"], max_dev=0.03)  # 3 cm off its neighbours is a bad frame
        hi = float(np.nanpercentile(q, 99))
        full = np.zeros(c.n)  # shut outside its window
        full[t0: t1 + 1] = q[t0: t1 + 1]
        fits.append({"part": part, "window": (t0, t1), "ref": m["ref"], "axis": m["axis"], "q": full, "hi": hi,
                     "at_ref": float(q[m["ref"]]) if not np.isnan(q[m["ref"]]) else hi,
                     "frames_measured": int((~np.isnan(q[t0: t1 + 1])).sum())})

    # Model frame: z up off the floor, x along the travel (out of the base).
    x = axis - (axis @ up) * up
    x /= np.linalg.norm(x)
    rot = np.stack([x, np.cross(up, x), up])
    origin = c.cloud(base, min(c.by[base], key=lambda t: abs(t - c.ref))).mean(axis=0)

    def to_model(p):
        return (p - origin) @ rot.T

    def box_of(p):
        return np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)

    # The base's shape on a frame with every drawer closed (else its mask holds an open drawer).
    closed = [t for t in c.by[base] if all(not (f["window"][0] <= t <= f["window"][1]) for f in fits)]
    t_base = max(closed or list(c.by[base]), key=lambda t: c.by[base][t].area)
    mesh_dir = c.track_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    floor_pts = None
    if floor:  # the floor's height, from the frame nearest the reference where its depth is seen
        for t in sorted(c.by.get(floor, {}), key=lambda t: abs(t - c.ref)):
            if len(cl := c.cloud(floor, t)) >= 50:
                floor_pts = cl
                break
    floor_z = float(np.percentile(to_model(floor_pts)[:, 2], 50)) if floor_pts is not None else \
        float(box_of(to_model(c.cloud(base, t_base)))[0][2])
    # Each drawer, shut (q = 0): its front face's cross-section, as deep as it travels at least.
    boxes = []
    for f in fits:
        pts = to_model(c.cloud(f["part"], f["ref"]))
        lo_b, hi_b = box_of(pts)
        front = pts[pts[:, 0] > hi_b[0] - 0.02]  # an open drawer's mask also takes in the opening
        if len(front) >= 50:
            lo_b[1:], hi_b[1:] = np.percentile(front[:, 1:], 2, axis=0), np.percentile(front[:, 1:], 98, axis=0)
        lo_b[0] = min(lo_b[0], hi_b[0] - f["hi"])
        lo_b[0], hi_b[0] = lo_b[0] - f["at_ref"], hi_b[0] - f["at_ref"]
        boxes.append((lo_b, hi_b))
    # The cabinet: one carcass from the floor up, around what was seen of it and its drawers,
    # with a slot for each drawer (a front view shows neither its depth nor its lower part, so
    # it is as deep as its drawers and stands on the floor).
    b_lo, b_hi = box_of(to_model(c.cloud(base, t_base)))
    out_lo = np.min([b_lo] + [lo for lo, _ in boxes], axis=0)
    out_hi = np.max([b_hi] + [hi for _, hi in boxes], axis=0)
    out_lo[0] -= PANEL_M  # a back panel behind the deepest drawer
    if floor_pts is not None:
        out_lo[2] = floor_z
    slots = [(lo[1] - GAP_M, lo[2] - GAP_M, hi[1] + GAP_M, hi[2] + GAP_M) for lo, hi in boxes]
    slot_back = min(lo[0] for lo, _ in boxes) - GAP_M
    pieces = [((out_lo[0], out_lo[1], out_lo[2]), (slot_back, out_hi[1], out_hi[2]))]  # behind the slots
    for y0, z0, y1, z1 in _subtract((out_lo[1], out_lo[2], out_hi[1], out_hi[2]), slots):  # around them
        pieces.append(((slot_back, y0, z0), (out_hi[0], y1, z1)))
    rgb = part_color(read_frame(c.video, t_base), c.by[base][t_base].full_mask(c.h, c.w))
    links = [emit3d.Link3D("base", [_box_visual(mesh_dir, f"base_{k}", lo, hi, rgb) for k, (lo, hi) in enumerate(pieces)])]
    sizes = {"base": (1000 * (out_hi - out_lo)).round().tolist()}
    masks, joint_q = {"base": c.by[base]}, {}
    # Each drawer: an open tray behind its front panel.
    for k, (f, (lo, hi)) in enumerate(zip(fits, boxes)):
        name = f"drawer{k + 1}"
        rgb = part_color(read_frame(c.video, f["ref"]), c.by[f["part"]][f["ref"]].full_mask(c.h, c.w))
        inner = tuple(0.7 * np.array(rgb))
        xf = hi[0] - FRONT_M
        y0, y1, z0, z1 = lo[1] + INSET_M, hi[1] - INSET_M, lo[2] + INSET_M, hi[2] - 2 * INSET_M
        parts = [((xf, lo[1], lo[2]), (hi[0], hi[1], hi[2]), rgb),  # front panel
                 ((lo[0], y0, z0), (xf, y1, z0 + PANEL_M), inner),  # bottom
                 ((lo[0], y0, z0), (xf, y0 + PANEL_M, z1), inner),  # sides
                 ((lo[0], y1 - PANEL_M, z0), (xf, y1, z1), inner),
                 ((lo[0], y0, z0), (lo[0] + PANEL_M, y1, z1), inner)]  # back
        link = emit3d.Link3D(name, [_box_visual(mesh_dir, f"{name}_{i}", a, b, col) for i, (a, b, col) in enumerate(parts)],
                             parent="base", joint_type="prismatic", axis=rot @ f["axis"], limits=(0.0, f["hi"]))
        links.append(link)
        sizes[name] = (1000 * (hi - lo)).round().tolist()
        masks[name], joint_q[f"{name}_joint"] = c.by[f["part"]], f["q"]
    root_pos = np.array([0, 0, -floor_z])
    (c.track_dir / "model.urdf").write_text(emit3d.urdf(c.video.stem, links))
    (c.track_dir / "model.xml").write_text(emit3d.mjcf(c.video.stem, links, root_pos, (c.w, c.h)))

    summary = {"joints": [{"name": f"drawer{i + 1}_joint", "part": f["part"], "window": list(f["window"]),
                           "travel_mm": 1000 * f["hi"], "frames_measured": f["frames_measured"],
                           "axis_vs_first_deg": float(np.rad2deg(np.arccos(min(1.0, abs(f["axis"] @ axis)))))}
                          for i, f in enumerate(fits)], "box_sizes_mm": sizes}
    (c.track_dir / "slides.json").write_text(json.dumps({
        "drawers": [{"part": f["part"], "reference_frame": f["ref"], "axis": f["axis"].tolist(),
                     "q": [None if np.isnan(v) else float(v) for v in f["q"]]} for f in fits], **summary}))
    (c.track_dir / "model3d.json").write_text(json.dumps({
        "reference_frame": c.ref, "camera_to_model": {"rotation": rot.tolist(), "hinge_point": origin.tolist(),
                                                      "root_pos": root_pos.tolist()}, **summary}))
    a = np.eye(4)
    a[:3, :3], a[:3, 3] = rot, -rot @ origin + root_pos
    cam_to_world = np.array([a @ np.linalg.inv(c.world[t]) for t in range(c.n)])
    cover = replay3d(c.track_dir / "model.xml", cam_to_world, c.cap.intrinsics, joint_q, masks, c.video,
                     c.track_dir / "replay3d.mp4")
    summary["mask_coverage"] = {k: float(np.nanmean(v)) for k, v in cover.items()}
    summary["mask_coverage_frames"] = {k: int(np.isfinite(v).sum()) for k, v in cover.items()}
    metadata = json.loads((c.track_dir / "model3d.json").read_text())
    metadata["mask_coverage"] = {k: v if np.isfinite(v) else None for k, v in summary["mask_coverage"].items()}
    metadata["mask_coverage_frames"] = summary["mask_coverage_frames"]
    metadata["video_frames"] = c.n
    (c.track_dir / "model3d.json").write_text(json.dumps(metadata, indent=2))
    for j in summary["joints"]:
        log(f"{j['name']} ({j['part']}, frames {j['window'][0]}..{j['window'][1]}): pulled out up to "
            f"{j['travel_mm']:.0f} mm, measured on {j['frames_measured']} frames; axis {j['axis_vs_first_deg']:.1f} deg "
            f"from drawer 1's")
    log("boxes (mm, along travel x across x up): " + ", ".join(f"{k} {v}" for k, v in sizes.items())
        + "; replay covers " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in summary["mask_coverage"].items())
        + " of the SAM masks")
    return summary
