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
from .lift3d import clean_angles, fit_plane, fit_planar_hinge, mask_cloud
from .pipeline import load_parts, read_frame, video_info
from .replay import replay3d
from .shape import part_color, prism, write_stl

SCREEN_DEPTH = 0.0008  # m, the dark display layer drawn on the plate's inner face
OFF_BASE_M = 0.015  # points this far off the base's top plane belong to a half-open plate
MAX_LEVEL_TILT_DEG = 4.0  # a floor fit tilted more than this from the reference is a bad fit
LEVEL_WINDOW = 31  # frames; ARKit drift is slow


@dataclass
class Roles:
    floor: str
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
        self.names, self.by = load_parts(track_dir)
        (self.w, self.h), n, _ = video_info(video)
        self.n = min(n, len(cap.depth), len(world))
        self.world = world[: self.n]

    @classmethod
    def from_stray(cls, video: Path, capture: Path, track_dir: Path, ref: int = 0) -> "Capture":
        """An iPhone capture: the scene's motion comes from ARKit odometry."""
        cap = stray.load(capture)
        return cls(video, track_dir, cap, stray.ref_to_frame_poses(capture, len(cap.depth), ref), ref)

    @classmethod
    def fixed_camera(cls, video: Path, track_dir: Path, depth: np.ndarray, intrinsics, ref: int = 0) -> "Capture":
        """A camera that doesn't move (a tripod, the RBO dataset): the scene's motion is identity."""
        return cls(video, track_dir, stray.Capture(video, depth, intrinsics), np.tile(np.eye(4), (len(depth), 1, 1)), ref)

    def level_with(self, floor: str, sample: int = 1500) -> dict:
        """Remove ARKit drift off the floor plane: the floor is static, so on every frame turn
        and shift the scene until that frame's floor plane matches the reference frame's.
        (Drift along the floor stays; a plane can't show it.) Returns the correction sizes."""
        rng = np.random.default_rng(0)

        def plane(t):
            c = self.cloud(floor, t)
            if len(c) > sample:
                c = c[rng.choice(len(c), sample, replace=False)]
            n, d, _ = fit_plane(c, rng)
            return n, d

        n0, d0 = plane(self.ref)
        # Per-frame correction as a small rotation vector (turning that frame's floor normal onto
        # the reference one) and a height shift. Drift is slow and plane fits are noisy (a hand
        # or a chair leg in the floor mask), so implausible frames are dropped and the rest
        # smoothed over time before applying.
        rotvec = np.full((self.n, 3), np.nan)
        height = np.full(self.n, np.nan)
        for t in range(self.n):
            if not self.has(floor, t):
                continue
            n, d = plane(t)
            if n @ n0 < 0:
                n, d = -n, -d
            ang = np.arccos(np.clip(n @ n0, -1, 1))
            if ang > np.deg2rad(MAX_LEVEL_TILT_DEG):
                continue
            axis = np.cross(n, n0)
            rotvec[t] = axis / max(np.linalg.norm(axis), 1e-12) * ang
            r = Rotation.from_rotvec(rotvec[t]).as_matrix()
            height[t] = d0 - n0 @ (r @ (n * d))
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

    def has(self, part: str | None, t: int) -> bool:
        return part is not None and part in self.by and t in self.by[part]

    def cloud(self, part: str, t: int) -> np.ndarray:
        """The part's depth points on frame t, in reference-camera coordinates."""
        c = mask_cloud(self.by[part][t].full_mask(self.h, self.w), self.cap.depth[t], self.cap.intrinsics)
        inv = np.linalg.inv(self.world[t])
        return c @ inv[:3, :3].T + inv[:3, 3]


def fit_hinge(c: Capture, roles: Roles, log=print) -> dict:
    """Fit the plate's hinge; writes hinge.json and hinge.jpg. Returns a summary dict."""
    base = c.cloud(roles.base, c.ref)
    nb, db, _ = fit_plane(base, np.random.default_rng(0))
    if nb @ (-base.mean(axis=0)) < 0:  # point the base's normal up, toward the camera
        nb, db = -nb, -db
    plate, used_closed, used_open = {}, 0, 0
    for t in range(c.n):
        if c.has(roles.plate, t):
            plate[t] = c.cloud(roles.plate, t)
        elif c.has(roles.plate_closed, t):
            # The plate's face isn't tracked: half open with the face turned away, or closed.
            # Decide by geometry, not by whether the open-indicator is seen (a lid half way
            # down already hides the keyboard from a camera in front): points well above the
            # base's top plane are the plate (below it are the base's own sides); if there are
            # hardly any, the object is closed and its top face is the plate's back.
            whole = c.cloud(roles.plate_closed, t)
            off = whole[whole @ nb - db > OFF_BASE_M]
            if len(off) > max(200, 0.2 * len(whole)):
                plate[t] = off
                used_open += 1
            else:
                plate[t] = whole
                used_closed += 1
    hinge = fit_planar_hinge(plate, c.n, c.ref, pin=base.mean(axis=0))
    raw = int((~np.isnan(hinge.q)).sum())
    hinge.q = clean_angles(hinge.q)
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
    t_ref = min(open_deg, key=lambda t: abs(t - c.ref))
    opening = open_deg[t_ref]
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
    return summary


def build_model(c: Capture, roles: Roles, log=print) -> dict:
    """URDF/MJCF from hinge.json plus shapes measured from depth, and a replay over the video
    (replay3d.mp4). Returns a summary dict, including how much of each SAM mask the replayed
    model covers per frame."""
    hinge = json.loads((c.track_dir / "hinge.json").read_text())
    ref, n, h, w = c.ref, c.n, c.h, c.w
    rng = np.random.default_rng(0)
    axis, point = np.array(hinge["axis"]), np.array(hinge["point"])
    q = np.array([np.nan if v is None else v for v in hinge["q"]])

    # Model frame: z up off the base, x along the hinge, origin on the hinge.
    base_c = c.cloud(roles.base, ref)
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
    z_floor = float(np.median(to_model(c.cloud(roles.floor, ref))[:, 2]))

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
    closed = [t for t in range(n) if not np.isnan(q[t]) and q[t] < np.nanpercentile(q, 10)
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
        "base_thickness_mm": 1000 * (z_top - z_floor), "plate_thickness_mm": 1000 * thick,
        "plate_thickness_clipped": not heights or not 0.001 < float(np.median(heights)) < 0.03,
        "plate_size_mm": [float(1000 * plate_size[0]), float(1000 * plate_size[1])],
        "hinge_range_deg": [float(np.rad2deg(limits[0])), float(np.rad2deg(limits[1]))],
        "closed_frames_used": len(heights),
    }
    (c.track_dir / "model3d.json").write_text(json.dumps({
        "reference_frame": ref, "camera_to_model": {"rotation": rot.tolist(), "hinge_point": point.tolist(),
                                                    "root_pos": root_pos.tolist()},
        "base_thickness_m": z_top - z_floor, "lid_thickness_m": thick, "hinge_limits_rad": limits, **summary}))

    a = np.eye(4)
    a[:3, :3], a[:3, 3] = rot, -rot @ point + root_pos
    cam_to_world = np.array([a @ np.linalg.inv(c.world[t]) for t in range(n)])
    cover = replay3d(c.track_dir / "model.xml", cam_to_world, c.cap.intrinsics, {"lid_joint": q},
                     {"lid": c.by[roles.plate], "base": c.by[roles.base]}, c.video, c.track_dir / "replay3d.mp4",
                     only_when={"base": lambda t: c.has(roles.open_indicator, t) if roles.open_indicator else True})
    summary["mask_coverage"] = {k: float(np.nanmean(v)) for k, v in cover.items()}
    summary["worst_frames"] = {k: int(np.nanargmin(v)) for k, v in cover.items() if not np.isnan(v).all()}
    log(f"model: base {summary['base_thickness_mm']:.0f} mm thick; plate {summary['plate_size_mm'][0]:.0f} x "
        f"{summary['plate_size_mm'][1]:.0f} mm, {summary['plate_thickness_mm']:.1f} mm thick"
        f"{' (clipped: not really measured)' if summary['plate_thickness_clipped'] else ''}; hinge "
        f"{summary['hinge_range_deg'][0]:.0f}..{summary['hinge_range_deg'][1]:.0f} deg; replay covers "
        + ", ".join(f"{k} {100 * v:.0f}%" for k, v in summary["mask_coverage"].items()) + " of the SAM masks")
    return summary
