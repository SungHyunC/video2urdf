"""A chain of rods on hinges (a desk lamp's pole and arm) from an iPhone LiDAR capture.

Thin rods defeat the plate and point-tracking fits: at the 256x192 depth resolution a rod
is 2-3 depth pixels wide, ARKit marks them low confidence and every one of them sits on a
depth edge. Their depth is still there, so each frame each rod's mask is lifted with the
raw depth and a 3D line is fitted by RANSAC. A hinge then shows as the rod's direction
turning in a plane: the axis is that plane's normal, the pivot the point all the rod's lines
pass closest to, the angle the turn from the reference frame. A child rod is first moved
back by its parent's fitted rotation, so each joint is measured relative to its parent.
The base stays put in the static scene (ARKit odometry cancels the moving phone).

The base is placed from its depth points, on any frame that shows it (it doesn't move). When
no frame gives it enough (mostly out of view, or a thin pole), or the base given is the floor
itself, the chain is anchored to the static scene instead: the rods' lines are already in one
world frame (ARKit poses, levelled on the floor when its frames agree), so the first hinge is
measured there just the same; the root link is then the floor (its plane measured, a disc under
the chain's foot) and the base is left out of the model.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import emit3d, stray
from .design import primitive_mesh
from .lift3d import clean_angles, fit_plane, mask_cloud
from .pipeline import read_frame
from .replay import replay3d
from .shape import part_color, prism, write_stl

LINE_INLIER_M = 0.01
FOREGROUND_M = 0.02
MIN_ROD_POINTS = 15
MIN_TURN = np.deg2rad(5)  # a rod turning less than this over the video is fixed to its parent
MIN_BASE_POINTS = 30  # high-confidence depth points the base needs on one frame to be placed
BELOW_M = 1000.0  # anchored chains: the first rod's parent end is the one nearer a point this far down
UPRIGHT_DEG = 30.0  # an anchored first rod this close to vertical stands on the floor: its foot goes down to it
FOOT_GUESS_M = 0.05  # ...and extending it further than this down to the floor is mostly a guess
FLOOR_DISC_M = 0.1  # the anchored root: a floor disc this wide (radius) under the chain's foot
FLOOR_DISC_THICK_M = 0.003
# A hinged rod's directions spread in its turning plane; when they spread about as much across it
# (singular values s3/s2 near 1), the rod turned about more than one axis (a swivel) or its lines are
# unreliable, and the fitted axis is arbitrary. Seen: phone stand upper arm 0.97 (likely also swivelling
# about the pole), its forearm 0.37, the LED lamp's arm 0.54.
PLANE_RATIO_MAX = 0.75


@dataclass
class RodLine:
    center: np.ndarray  # reference-camera coordinates
    direction: np.ndarray  # unit, from the parent end to the free end
    lo: float  # extent along the direction from the center (m)
    hi: float


@dataclass
class ChainJoint:
    parent: str
    child: str
    axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0]))
    point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fixed: bool = False
    residual_m: float = 0.0
    plane_ratio: float = 0.0  # spread of the rod's directions across the hinge's plane over that within it


def mask_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """The rod's centre line in the image (a point and unit direction, pixels) and its width."""
    ys, xs = np.nonzero(mask[::2, ::2])
    if len(xs) < 20:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64) * 2
    c = pts.mean(axis=0)
    _, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    across = (pts - c) @ np.array([-d[1], d[0]])
    return c, d, float(np.percentile(np.abs(across), 90) * 2)


def rod_points(c, raw: np.ndarray, part: str, t: int) -> tuple[np.ndarray, np.ndarray | None]:
    """The rod's depth points on frame t in frame-t camera coordinates, and the normal of the plane
    through the camera and the rod's image centre line (the rod lies in it). Any depth pixel the
    mask covers counts, whatever its confidence; pixels off the centre line (a hand at an
    occlusion) or behind the nearest surface (the wall seen past the rod's edge) are dropped."""
    mask = c.by[part][t].full_mask(c.h, c.w)
    axis = mask_axis(mask)
    dh, dw = raw.shape[1:]
    m = cv2.resize(mask.astype(np.float32), (dw, dh), interpolation=cv2.INTER_AREA) > 0.05
    d = raw[t]
    near = cv2.erode(np.where(d > 0, d, 1e3).astype(np.float32), np.ones((5, 5), np.uint8))
    m &= (d > 0) & (d <= near + FOREGROUND_M)
    vv, uu = np.nonzero(m)
    uv = np.stack([(uu + 0.5) * c.w / dw, (vv + 0.5) * c.h / dh], axis=1)
    k = c.cap.intrinsics
    normal = None
    if axis is not None:
        c2, d2, width = axis
        off = np.abs((uv - c2) @ np.array([-d2[1], d2[0]]))
        keep = off <= max(width, 1.5 * c.w / dw)
        uv, vv, uu = uv[keep], vv[keep], uu[keep]
        rays = [np.array([(q[0] - k.cx) / k.f, (q[1] - k.cy) / k.f, 1.0]) for q in (c2, c2 + 100 * d2)]
        normal = np.cross(*rays)
        normal /= np.linalg.norm(normal)
    return k.lift(uv, d[vv, uu].astype(np.float64)), normal


def fit_line(points: np.ndarray, rng: np.random.Generator, iters: int = 200) -> tuple[RodLine, float] | None:
    if len(points) < MIN_ROD_POINTS:
        return None
    best = None
    for _ in range(iters):
        a, b = points[rng.choice(len(points), 2, replace=False)]
        d = b - a
        if np.linalg.norm(d) < 0.02:
            continue
        d /= np.linalg.norm(d)
        r = points - a
        inl = np.linalg.norm(r - np.outer(r @ d, d), axis=1) < LINE_INLIER_M
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < MIN_ROD_POINTS:
        return None
    p = points[best]
    center = p.mean(axis=0)
    d = np.linalg.svd(p - center, full_matrices=False)[2][0]
    s = (p - center) @ d
    lo, hi = np.percentile(s, [2, 98])
    if hi - lo < 0.05:  # a blob, not a rod
        return None
    return RodLine(center, d, float(lo), float(hi)), float(best.mean())


def rod_lines(c, raw, part: str, near: np.ndarray, log=print) -> dict[int, RodLine]:
    """A line per frame, directed away from ``near`` (the parent's side)."""
    rng = np.random.default_rng(0)
    out = {}
    for t in sorted(c.by.get(part, {})):
        if t >= c.n:
            continue
        pts, normal = rod_points(c, raw, part, t)
        fit = fit_line(pts, rng)
        if fit is None:
            continue
        line, _ = fit
        if normal is not None:  # into the plane the image shows the rod in: depth noise can't tilt it out
            d = line.direction - (line.direction @ normal) * normal
            if np.linalg.norm(d) < 0.5:
                continue
            line = RodLine(line.center - (line.center @ normal) * normal, d / np.linalg.norm(d), line.lo, line.hi)
        inv = np.linalg.inv(c.world[t])  # frame-t camera -> reference coordinates
        line = RodLine(inv[:3, :3] @ line.center + inv[:3, 3], inv[:3, :3] @ line.direction, line.lo, line.hi)
        out[t] = toward_free_end(line, near)
    log(f"{part}: a 3D line on {len(out)}/{len(c.by.get(part, {}))} tracked frames")
    return out


def toward_free_end(line: RodLine, near: np.ndarray) -> RodLine:
    """The line directed from its end nearer ``near`` (its parent's side) to its free end."""
    ends = [line.center + line.lo * line.direction, line.center + line.hi * line.direction]
    if np.linalg.norm(ends[1] - near) < np.linalg.norm(ends[0] - near):
        return RodLine(line.center, -line.direction, -line.hi, -line.lo)
    return line


def rod_cloud(c, raw, part: str, frames: int = 40) -> np.ndarray:
    """A rod's depth points over the video in reference coordinates, from up to ``frames`` of its
    tracked frames: where it is, even when no single frame gives it a line."""
    ts = [t for t in sorted(c.by.get(part, {})) if t < c.n]
    out = []
    for t in ts[:: max(1, len(ts) // frames)]:
        pts, _ = rod_points(c, raw, part, t)
        if len(pts):
            inv = np.linalg.inv(c.world[t])
            out.append(pts @ inv[:3, :3].T + inv[:3, 3])
    return np.vstack(out) if out else np.zeros((0, 3))


def base_points(c, base: str, ref: int, candidates: int = 60) -> tuple[np.ndarray, int, int]:
    """The base's depth points (reference coordinates), the frame they come from, and how many the
    tracked frame nearest the reference has. That frame places the base when it has
    MIN_BASE_POINTS; the base is static, so otherwise the frame with the most of the
    ``candidates`` where its mask is largest does (a base cut by the picture's edge reads
    low-confidence depth there, and the reference frame is chosen for the rods, not the base)."""
    seen = sorted(c.by.get(base, {}))
    if not seen:
        return np.zeros((0, 3)), -1, 0
    t0 = min(seen, key=lambda t: abs(t - ref))
    best, t_best = c.cloud(base, t0), t0
    n_near = len(best)
    if n_near < MIN_BASE_POINTS:
        for t in sorted(seen, key=lambda t: -c.by[base][t].area)[:candidates]:
            pts = c.cloud(base, t)
            if len(pts) > len(best):
                best, t_best = pts, t
    return best, t_best, n_near


def floor_plane(c, floor: str | None, ref: int, up: np.ndarray, rng, search: int = 300):
    """The floor's plane as (unit normal along ``up``, offset d with n.x = d, frame): fitted on the
    frame nearest the reference that shows it with depth. None when it is never seen with depth
    there or is more than 15 deg off level (another surface)."""
    if not floor:
        return None
    for t in sorted(c.by.get(floor, {}), key=lambda t: abs(t - ref))[:search]:
        pts = c.cloud(floor, t)
        if len(pts) <= 100:
            continue
        nf, df, _ = fit_plane(pts, rng)
        if nf @ up < 0:
            nf, df = -nf, -df
        if np.rad2deg(np.arccos(np.clip(nf @ up, -1, 1))) >= 15:
            return None
        return nf, float(df), t
    return None


def stand_on_floor(point: np.ndarray, direction: np.ndarray, normal: np.ndarray, offset: float,
                   max_tilt_deg: float = UPRIGHT_DEG) -> np.ndarray | None:
    """Where a rod standing up from the floor meets it: its line (through its lowest seen
    ``point``, ``direction`` pointing away from the floor) followed down to the plane n.x = d.
    None when the rod leans further than ``max_tilt_deg`` from the floor's normal."""
    cos = float(direction @ normal)
    if cos < np.cos(np.deg2rad(max_tilt_deg)):
        return None
    return point - (point @ normal - offset) / cos * direction


def drop_to_floor(point: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """The point straight below ``point`` on the plane n.x = d."""
    return point - (point @ normal - offset) * normal


def model_rotation(up: np.ndarray, axis: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Rows x, y, z of the model frame: z up, x along ``axis``'s horizontal part (``fallback``'s
    when the axis is about vertical, any horizontal when both are)."""
    z = up / np.linalg.norm(up)
    x = axis - (axis @ z) * z
    if np.linalg.norm(x) < 0.3:
        x = fallback - (fallback @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [1.0, 0, 0]) if abs(z[0]) < 0.9 else np.cross(z, [0, 1.0, 0])
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def fit_rod_joint(lines: dict[int, RodLine], ref: int, n: int, near: np.ndarray) -> ChainJoint:
    """Axis, pivot and angle per frame of a rod turning about a hinge (lines in its parent's frame)."""
    ts = sorted(lines)
    dirs = np.array([lines[t].direction for t in ts])
    _, sv, vt = np.linalg.svd(dirs, full_matrices=False)
    axis = vt[2]  # the directions lie in the plane normal to the axis
    d_ref = lines[min(ts, key=lambda t: abs(t - ref))].direction
    q = np.full(n, np.nan)
    for t in ts:
        d = lines[t].direction
        q[t] = np.arctan2(axis @ np.cross(d_ref, d), d_ref @ d)
    q = clean_angles(clean_angles(q), window=8, max_dev=np.deg2rad(10))  # spikes, then hand-occlusion jitter
    joint = ChainJoint("", "", axis=axis, q=q, plane_ratio=float(sv[2] / max(sv[1], 1e-12)))
    if np.nanpercentile(q, 95) - np.nanpercentile(q, 5) < MIN_TURN:
        joint.fixed = True
        first = lines[min(ts, key=lambda t: abs(t - ref))]
        joint.point = first.center + first.lo * d_ref
        return joint
    # Pivot: rods in a chain hinge at their ends, so the hinge is where the rod starts (its near
    # end, in the parent's frame), the median over frames. The point all the lines pass closest to
    # would need the rod to turn a lot, and depth noise in each line's tilt drags it along the rod.
    good = [t for t in ts if not np.isnan(q[t])]
    starts = np.array([lines[t].center + lines[t].lo * lines[t].direction for t in good])
    point = np.median(starts, axis=0)
    r = [np.linalg.norm((point - lines[t].center) - ((point - lines[t].center) @ lines[t].direction) * lines[t].direction)
         for t in good]
    joint.point, joint.residual_m = point, float(np.median(r))
    return joint


def consensus_line(lines: list[RodLine]) -> RodLine:
    """One line standing for many of the same rod in the same pose: the median direction, the
    median position across it and the median of its ends along it."""
    d = np.median([l.direction for l in lines], axis=0)
    d /= np.linalg.norm(d)
    c = np.array([l.center for l in lines])
    s = c @ d
    p = np.median(c - np.outer(s, d), axis=0) + np.median(s) * d
    lo = float(np.median([(l.center + l.lo * l.direction - p) @ d for l in lines]))
    hi = float(np.median([(l.center + l.hi * l.direction - p) @ d for l in lines]))
    return RodLine(p, d, lo, hi)


def line_offset(a: RodLine, b: RodLine) -> tuple[float, float]:
    """How far line a's centre sits from line b, across it (m), and the angle between them (deg)."""
    v = a.center - b.center
    return (float(np.linalg.norm(v - (v @ b.direction) * b.direction)),
            float(np.rad2deg(np.arccos(np.clip(abs(a.direction @ b.direction), -1, 1)))))


def static_reference_line(lines: dict[int, RodLine], ref: int, max_off: float = 0.02,
                          max_deg: float = 5.0) -> tuple[RodLine, tuple[float, float] | None]:
    """The line that places a rod that doesn't move: the reference frame's own line, unless it sits
    more than ``max_off`` m or ``max_deg`` off the consensus of all its frames' lines (one frame's
    thin-rod depth can be centimetres off). Returns the line and, when the consensus replaced it,
    how far off the reference frame's line was."""
    own = lines[min(lines, key=lambda t: abs(t - ref))]
    if len(lines) < 10:
        return own, None
    mid = consensus_line(list(lines.values()))
    off, deg = line_offset(own, mid)
    return (mid, (off, deg)) if off > max_off or deg > max_deg else (own, None)


def _unturn(line: RodLine, joint: ChainJoint, t: int) -> RodLine | None:
    """A child's line moved back by its parent's rotation on frame t (into the parent's reference pose)."""
    if joint.fixed:
        return line
    if t >= len(joint.q) or np.isnan(joint.q[t]):
        return None
    r = rotation(joint.axis, -joint.q[t])
    return RodLine(r @ (line.center - joint.point) + joint.point, r @ line.direction, line.lo, line.hi)


def _rod_radius(c, part: str, lines: dict[int, RodLine]) -> float:
    """Half the rod's width: mask area over its length in pixels, at its depth."""
    radii = []
    for t in list(lines)[:: max(1, len(lines) // 20)]:
        m = c.by[part][t].full_mask(c.h, c.w)
        line = lines[t]
        cam = c.world[t]
        ends = np.array([line.center + line.lo * line.direction, line.center + line.hi * line.direction])
        ends = ends @ cam[:3, :3].T + cam[:3, 3]
        if (ends[:, 2] <= 0).any():
            continue
        f = c.cap.intrinsics.f
        px = np.linalg.norm((ends[0, :2] / ends[0, 2] - ends[1, :2] / ends[1, 2]) * f)
        if px > 20:
            radii.append(m.sum() / px / 2 * ends[:, 2].mean() / f)
    return float(np.clip(np.median(radii), 0.003, 0.03)) if radii else 0.006


def _box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / max(1, area(a) + area(b) - inter)


def chain_reference(c, base: str | None, chain: list[str]) -> int | None:
    """A frame showing the base (None: an anchored chain, no base needed) and every rod, the rods
    large and apart (folded rods overlap and their masks mix, which would make every angle
    relative to a wrong pose)."""
    best, score = None, -1.0
    shown = ([base] if base else []) + list(chain)
    for t in range(c.n):
        if not all(c.has(p, t) for p in shown):
            continue
        boxes = [c.by[p][t].box for p in chain]
        overlap = max([_box_iou(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:]], default=0.0)
        s = min(c.by[p][t].area for p in chain) * (1 - overlap) ** 2
        if s > score:
            best, score = t, s
    return best


def _smooth(q: np.ndarray, window: int = 9) -> np.ndarray:
    """A running median over measured frames (NaN stays NaN): depth jitter on a thin rod."""
    from scipy.ndimage import median_filter
    ok = np.isfinite(q)
    if ok.sum() < window:
        return q
    out = q.copy()
    out[ok] = median_filter(q[ok], size=window, mode="nearest")
    return out


def fit_chain(c, base: str, chain: list[str], floor: str | None = None, log=print,
              fixed: tuple[str, ...] = ()) -> dict:
    """Fit base -> chain[0] -> chain[1] ..., write model.xml/urdf, replay3d and model3d.json.

    ``base`` == ``floor``, or a base with too few depth points on every frame (given a floor),
    anchors the chain to the static scene (see the module docstring); a fixed first rod with too
    little depth for a line is then placed upright from the floor to the hinge it carries. Log
    lines starting with "WARNING:" name what was placed rather than measured."""
    track_dir = c.track_dir
    raw = stray.load(c.source, min_confidence=0).depth if c.source is not None else c.cap.depth
    ref = c.ref
    rng = np.random.default_rng(0)
    # Up: gravity, or the floor's plane when it is level with it.
    up = c.up if c.up is not None else np.array([0, -1.0, 0])
    plane = floor_plane(c, floor, ref, up, rng)
    if plane is not None:
        up = plane[0]
    anchor, base_cloud, t_base = None, np.zeros((0, 3)), None
    if base == floor:
        anchor = {"reason": f"the floor ({floor}) was given as the base"}
    else:
        base_cloud, t_base, n_near = base_points(c, base, ref)
        if len(base_cloud) < MIN_BASE_POINTS:
            why = (f"{base} has too few depth points to place it (at most {len(base_cloud)} high-confidence points on any "
                   f"tracked frame, {MIN_BASE_POINTS} needed: mostly out of view, or too thin)")
            if not floor:
                raise ValueError(why + "; give the floor it stands on as floor (or as base) to anchor the chain to it")
            anchor = {"reason": why, "base_points_max": len(base_cloud)}
        elif n_near < MIN_BASE_POINTS:
            log(f"{base}: placed from frame {t_base} ({len(base_cloud)} depth points; the tracked frame nearest the "
                f"reference has {n_near}): it doesn't move, so any frame places it")
    root = base if anchor is None else floor
    if anchor is not None:
        log(f"Anchored to the static scene: {anchor['reason']}. The chain stands on the floor ({floor}), placed by the "
            "phone's tracked poses: the hinges are measured in that frame all the same; the base itself is left out of "
            "the model.")
        if plane is None:
            log(f"WARNING: {floor} has no level depth plane near the reference frame: the model's floor is put at the "
                "chain's lowest seen point, so heights above the floor are not measured.")
    # The first rod starts at its end nearer the base; anchored, at its lower end.
    near = np.median(base_cloud, axis=0) if anchor is None else -BELOW_M * up / np.linalg.norm(up)
    joints, lines_by, estimated, placed = [], {}, {}, set()  # estimated: {part: why}; placed: no line at all
    ref_lines = {}  # the line placing each rod in its reference pose
    parent_joint = None
    for k, part in enumerate(chain):
        lines = rod_lines(c, raw, part, near, log)
        parent = root if k == 0 else chain[k - 1]
        if len(lines) < 10:
            if not (anchor is not None and k == 0 and part in fixed and len(chain) > 1 and plane is not None):
                raise ValueError(f"{part}: only {len(lines)} frames with a 3D line; is it a rod? (tracked {len(c.by.get(part, {}))})")
            pts = rod_cloud(c, raw, part)
            if len(pts) < MIN_ROD_POINTS:
                raise ValueError(f"{part}: only {len(lines)} frames with a 3D line and {len(pts)} depth points; it can't be placed")
            # A fixed pole too thin for depth: placed once the hinge it carries is measured.
            estimated[part] = (f"{part} has a 3D line on only {len(lines)} frames: the model stands it upright on the floor "
                               "under the hinge it carries (its position and length are estimated, not measured)")
            log(f"WARNING: {estimated[part]}.")
            placed.add(part)
            j = ChainJoint(parent, part, fixed=True, q=np.zeros(c.n))
            joints.append(j)
            lines_by[part] = {}
            near = np.median(pts, axis=0)  # the next rod starts at its end nearer this one
            parent_joint = j
            continue
        if parent_joint is not None:  # into the parent's reference pose
            moved = {t: _unturn(l, parent_joint, t) for t, l in lines.items()}
            lines = {t: l for t, l in moved.items() if l is not None}
        j = fit_rod_joint(lines, ref, c.n, near)
        j.parent, j.child = parent, part
        if part in fixed:  # seen not to turn (the video shows it; its depth jitter says otherwise)
            j.fixed = True
            j.q = np.zeros(c.n)
            first = lines[min(lines, key=lambda t: abs(t - ref))]
            j.point = first.center + first.lo * first.direction
        else:
            j.q = _smooth(j.q)
        ref_line, outlier = lines[min(lines, key=lambda t: abs(t - ref))], None
        if anchor is not None and j.fixed:  # a fixed rod of an anchored chain (its pole): placed by all its frames
            ref_line, outlier = static_reference_line(lines, ref)
        if outlier is not None:
            log(f"{part}: its line on the reference frame sits {1000 * outlier[0]:.0f} mm / {outlier[1]:.1f} deg off the "
                f"consensus of its {len(lines)} lines (that frame's depth on the rod is off): the consensus places it")
            j.point = ref_line.center + ref_line.lo * ref_line.direction
        ref_lines[part] = ref_line
        joints.append(j)
        lines_by[part] = lines
        near = ref_line.center + ref_line.hi * ref_line.direction  # the next rod starts at this one's free end
        parent_joint = j
        span = np.rad2deg([np.nanmin(j.q), np.nanmax(j.q)])
        log(f"{j.parent} -> {part}: {'fixed' if j.fixed else f'hinge, {span[0]:.0f}..{span[1]:.0f} deg'}, "
            f"pivot residual {1000 * j.residual_m:.1f} mm, angle on {int(np.isfinite(j.q).sum())}/{c.n} frames")
        if not j.fixed and j.plane_ratio > PLANE_RATIO_MAX:
            log(f"WARNING: {part} doesn't turn in one plane: its directions spread {j.plane_ratio:.2f}x as much across "
                "the fitted hinge's plane as within it, so it turned about more than one axis (a swivel as well as the "
                "hinge?) or its 3D lines are unreliable; this hinge's axis and angles are not measured, and one hinge "
                "can't follow it.")

    foot_extended = None
    if anchor is not None:  # where the chain stands on the floor
        first = joints[0]
        if first.child in placed:  # upright from the floor to the hinge it carries
            top = joints[1].point
            first.point = drop_to_floor(top, plane[0], plane[1])
            length = float(np.linalg.norm(top - first.point))
            d = (top - first.point) / max(length, 1e-6)
            line = RodLine((first.point + top) / 2, d, -length / 2, length / 2)
            lines_by[first.child] = {t: line for t in c.by[first.child] if t < c.n}
            ref_lines[first.child] = line
        elif first.fixed:  # a pole: from its lowest seen point (any frame: it doesn't move) down to the floor
            lines = lines_by[first.child]
            r = ref_lines[first.child]
            low = r.center + np.percentile([(l.center + l.lo * l.direction - r.center) @ r.direction
                                            for l in lines.values()], 5) * r.direction
            foot = stand_on_floor(low, r.direction, plane[0], plane[1]) if plane is not None else None
            first.point = low if foot is None else foot
            if foot is not None:
                foot_extended = float((low - foot) @ r.direction)
                log(f"{first.child}: its foot is followed {1000 * foot_extended:.0f} mm down its line from the lowest point "
                    "seen on it to the floor")
                if foot_extended > FOOT_GUESS_M:
                    estimated[first.child] = (f"{first.child}'s lowest {1000 * foot_extended:.0f} mm were never seen with "
                                              "depth: the model extends it straight down its line to the floor")
                    log(f"WARNING: {estimated[first.child]} (estimated).")

    # Model frame: z up (gravity, or the floor's plane), x along the first hinge's horizontal part.
    if anchor is None:
        rot = model_rotation(up, joints[0].axis,
                             np.linalg.svd(base_cloud - base_cloud.mean(axis=0), full_matrices=False)[2][0])
        base_m = base_cloud @ rot.T
        z_floor = (float(((plane[0] * plane[1]) @ rot.T)[2]) if plane is not None
                   else float(np.percentile(base_m[:, 2], 1)) - 0.01)
        origin = np.array([*np.median(base_m[:, :2], axis=0), z_floor])
    else:  # x along the first moving hinge; the origin on the floor under the chain's foot
        moving = [j for j in joints if not j.fixed] or joints
        rot = model_rotation(up, moving[0].axis, ref_lines[moving[0].child].direction)
        foot_m = joints[0].point @ rot.T
        origin = np.array([foot_m[0], foot_m[1],
                           float(((plane[0] * plane[1]) @ rot.T)[2]) if plane is not None else foot_m[2]])

    def to_model(p):
        return p @ rot.T - origin

    mesh_dir = track_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    image = read_frame(c.video, ref)
    if anchor is None:
        top = to_model(base_cloud)
        z_top = float(np.percentile(top[:, 2], 90))
        v, f = prism(_hull2d(top[:, :2]), np.zeros(3), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2], 0.0, max(z_top, 0.005))
        write_stl(mesh_dir / "base.stl", v, f)
        links = [emit3d.Link3D(base, [emit3d.Visual("meshes/base.stl", part_color(image, c.by[base][ref].full_mask(c.h, c.w))
                                                    if ref in c.by[base] else (0.8, 0.8, 0.8))])]
    else:  # the floor the chain stands on: a thin disc under its foot (the plane is measured, the disc's size isn't)
        v, f = primitive_mesh({"type": "cylinder", "from": [0, 0, -1000 * FLOOR_DISC_THICK_M], "to": [0, 0, 0],
                               "radius": 1000 * FLOOR_DISC_M})
        write_stl(mesh_dir / "base.stl", v / 1000, f)
        floor_seen = sorted(c.by.get(floor, {}), key=lambda t: abs(t - ref))
        t_floor = plane[2] if plane is not None else (floor_seen[0] if floor_seen else None)
        rgb = (part_color(read_frame(c.video, t_floor), c.by[floor][t_floor].full_mask(c.h, c.w))
               if t_floor is not None else (0.6, 0.6, 0.6))
        links = [emit3d.Link3D(root, [emit3d.Visual("meshes/base.stl", rgb)])]
        anchor.update({"root": root, "base": None if base == floor else base, "floor": floor,
                       "floor_frame": None if plane is None else plane[2], "disc_radius_mm": 1000 * FLOOR_DISC_M,
                       "foot_extended_mm": None if foot_extended is None else 1000 * foot_extended})
    pivots = {root: np.zeros(3)}
    summary_joints = []
    for j in joints:
        lines = lines_by[j.child]
        ref_line = ref_lines[j.child]
        pivot = to_model(j.point)
        # The rod runs from its hinge to where its far end is seen on the reference frame (the
        # hinge is a median over frames, so the reference line's own direction could miss the end).
        tip = to_model(ref_line.center + ref_line.hi * ref_line.direction)
        d = (tip - pivot) / max(float(np.linalg.norm(tip - pivot)), 1e-6)
        # Its length from every frame: the far end can leave the picture on the reference frame.
        length = float(np.percentile([l.hi - (j.point - l.center) @ l.direction for l in lines.values()], 90))
        radius = _rod_radius(c, j.child, lines)
        v, f = primitive_mesh({"type": "cylinder", "from": [0, 0, 0], "to": (d * length * 1000).tolist(), "radius": radius * 1000})
        write_stl(mesh_dir / f"{j.child}.stl", v / 1000, f)
        t_col = min(c.by[j.child], key=lambda t: abs(t - ref))
        rgb = part_color(read_frame(c.video, t_col) if t_col != ref else image, c.by[j.child][t_col].full_mask(c.h, c.w))
        valid = j.q[np.isfinite(j.q)]
        limits = (float(np.percentile(valid, 1)), float(np.percentile(valid, 99))) if len(valid) else (0.0, 0.0)
        links.append(emit3d.Link3D(j.child, [emit3d.Visual(f"meshes/{j.child}.stl", rgb)], parent=j.parent,
                                   origin=pivot - pivots[j.parent], joint_type="fixed" if j.fixed else "revolute",
                                   axis=j.axis @ rot.T, limits=limits))
        pivots[j.child] = pivot
        summary_joints.append({"parent": j.parent, "child": j.child, "type": "fixed" if j.fixed else "revolute",
                               "axis_model": (j.axis @ rot.T).tolist(), "pivot_model_mm": (1000 * pivot).tolist(),
                               "pivot_residual_mm": 1000 * j.residual_m, "rod_length_mm": 1000 * length,
                               "rod_radius_mm": 1000 * radius, "range_deg": np.rad2deg(limits).tolist(),
                               "frames_measured": 0 if j.child in placed else int(np.isfinite(j.q).sum()),
                               "turn_plane_ratio": None if j.fixed else j.plane_ratio,
                               "estimated": estimated.get(j.child)})
    (track_dir / "model.urdf").write_text(emit3d.urdf(c.video.stem, links))
    (track_dir / "model.xml").write_text(emit3d.mjcf(c.video.stem, links, np.zeros(3), (c.w, c.h)))
    (track_dir / "links.json").write_text(json.dumps({root: [root], **{p: [p] for p in chain}}))
    a = np.eye(4)
    a[:3, :3], a[:3, 3] = rot, -origin
    cam_to_world = np.array([a @ np.linalg.inv(c.world[t]) for t in range(c.n)])
    qs = {f"{j.child}_joint": j.q for j in joints if not j.fixed}
    scored = ([base] if anchor is None else []) + list(chain)  # an anchored chain's floor disc isn't a part of it
    cover = replay3d(track_dir / "model.xml", cam_to_world, c.cap.intrinsics, qs,
                     {p: c.by[p] for p in scored}, c.video, track_dir / "replay3d.mp4")
    summary = {"reference_frame": ref, "base_height_mm": 1000 * z_top if anchor is None else None,
               "base_frame": t_base, "anchor": anchor, "joints": summary_joints,
               "camera_to_model": {"rotation": rot.tolist(), "origin": origin.tolist()},
               "floor_used": plane is not None,
               "mask_coverage": {k: float(np.nanmean(v)) if np.isfinite(v).any() else None for k, v in cover.items()},
               "video_frames": c.n}
    (track_dir / "model3d.json").write_text(json.dumps(summary, indent=2))
    (track_dir / "chain.json").write_text(json.dumps({"base": base, "root": root, "anchored": anchor is not None,
                                                      "chain": chain, "floor": floor, "fixed": list(fixed), "joints": [
        {"child": j.child, "type": "fixed" if j.fixed else "revolute", "estimated": j.child in estimated,
         "q": [None if not np.isfinite(v) else float(v) for v in j.q]} for j in joints]}))
    log("model: " + (f"anchored on {root}; " if anchor is not None else "")
        + "; ".join(f"{s['child']} {s['type']} rod {s['rod_length_mm']:.0f} mm x r{s['rod_radius_mm']:.0f}"
                    + (f", {s['range_deg'][0]:.0f}..{s['range_deg'][1]:.0f} deg" if s['type'] == 'revolute' else '')
                    + (" (estimated)" if s["estimated"] else "")
                    for s in summary_joints)
        + "; replay covers " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in summary["mask_coverage"].items() if v is not None)
        + " of the SAM masks")
    return summary


def _hull2d(points: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(points.astype(np.float32))
    return hull[:, 0, :].astype(np.float64)
