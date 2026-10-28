"""URDF/MJCF for a hinged plate measured with iPhone LiDAR, shapes from the depth points.

Run scripts/fit_lidar_hinge.py first (hinge.json). Everything below is measured from the
capture: the base's footprint and thickness (its top surface to the floor), the plate's
outline and thickness (how far its back sits above the base when closed), colours, and the
hinge with its observed range. Then the model is replayed over the video from the iPhone's
own camera poses.

Usage:
  .venv/bin/python scripts/build_lidar_model.py data/real/laptop_lidar/laptop_lidar.mp4 \\
      --stray data/real/laptop_lidar --floor wooden_floor --base silver_laptop \\
      --plate laptop_screen --plate-closed silver_laptop --open-indicator laptop_keyboard
Writes model.urdf, model.xml, meshes/, model3d.json and replay3d.mp4 into the track folder.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf import emit3d, stray  # noqa: E402
from video2urdf.lift3d import fit_plane, mask_cloud  # noqa: E402
from video2urdf.pipeline import load_parts, read_frame, video_info  # noqa: E402
from video2urdf.replay import replay3d  # noqa: E402
from video2urdf.shape import part_color, prism, write_stl  # noqa: E402

SCREEN_DEPTH = 0.0008  # m, the dark display layer drawn on the plate's inner face


def hull(points2d: np.ndarray) -> np.ndarray:
    return cv2.convexHull(points2d.astype(np.float32))[:, 0, :].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--stray", type=Path, required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--base", required=True, help="part that is the base on the reference frame")
    ap.add_argument("--plate", required=True, help="part that is the plate's inner face (e.g. the screen)")
    ap.add_argument("--plate-closed", help="part that is the plate's back when closed")
    ap.add_argument("--open-indicator", help="part seen only while open (frames without it count as closed)")
    ap.add_argument("--tracks", type=Path, default=ROOT / "data" / "tracks")
    args = ap.parse_args()

    track_dir = args.tracks / args.video.stem
    names, by = load_parts(track_dir)
    hinge = json.loads((track_dir / "hinge.json").read_text())
    ref = hinge["reference_frame"]
    cap = stray.load(args.stray)
    (w, h), n, _ = video_info(args.video)
    n = min(n, len(cap.depth))
    world = stray.ref_to_frame_poses(args.stray, n, ref)
    rng = np.random.default_rng(0)

    def cloud(part, t):
        c = mask_cloud(by[part][t].full_mask(h, w), cap.depth[t], cap.intrinsics)
        inv = np.linalg.inv(world[t])
        return c @ inv[:3, :3].T + inv[:3, 3]

    axis, point = np.array(hinge["axis"]), np.array(hinge["point"])
    q = np.array([np.nan if v is None else v for v in hinge["q"]])

    # Model frame: z up off the base, x along the hinge, origin on the hinge.
    base_c = cloud(args.base, ref)
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
    floor_m = to_model(cloud(args.floor, ref))
    z_top, z_floor = np.median(top_m[:, 2]), np.median(floor_m[:, 2])
    print(f"base: top {1000 * z_top:.1f} mm, floor {1000 * z_floor:.1f} mm from the hinge -> "
          f"thickness {1000 * (z_top - z_floor):.1f} mm")

    # Plate (lid) on the reference frame, in coordinates along the hinge (s), away from it in
    # its plane (r) and across it (w).
    plate_m = to_model(cloud(args.plate, ref))
    c = plate_m.mean(axis=0)
    e1 = np.array([1.0, 0, 0])
    e2 = np.array([0, c[1], c[2]]) / np.linalg.norm(c[1:])
    e3 = np.cross(e1, e2)
    if e3 @ (top_m.mean(axis=0) - c) < 0:  # e3 points from the plate's back to its inner face
        e3 = -e3
    s, r, wv = plate_m @ e1, plate_m @ e2, plate_m @ e3
    w_face = float(np.median(wv))

    # Plate thickness: when closed, its back sits this far above the base's top.
    closed = [t for t in range(n) if not np.isnan(q[t]) and q[t] < np.nanpercentile(q, 10)
              and args.plate_closed and t in by[args.plate_closed]
              and not (args.open_indicator and t in by[args.open_indicator])]
    heights, back = [], []
    for t in closed[:: max(1, len(closed) // 40)]:
        pm = to_model(cloud(args.plate_closed, t))
        if len(pm) > 100:
            heights.append(np.median(pm[:, 2]) - z_top)
            # The closed back shows the whole plate (bezel included): turn it back to the
            # reference pose about the hinge (model x) to get the plate's full outline.
            c_, s_ = np.cos(-q[t]), np.sin(-q[t])
            back.append(pm @ np.array([[1, 0, 0], [0, c_, -s_], [0, s_, c_]]).T)
    thick = float(np.clip(np.median(heights), 0.003, 0.03)) if heights else 0.006
    print(f"plate: {1000 * np.ptp(s):.0f} mm along the hinge, {1000 * r.max():.0f} mm out from it, "
          f"thickness {1000 * thick:.1f} mm (from {len(heights)} closed frames)")

    # Shapes.
    mesh_dir = track_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    image = read_frame(args.video, ref)
    base_rgb = part_color(image, by[args.base][ref].full_mask(h, w))
    screen_rgb = part_color(image, by[args.plate][ref].full_mask(h, w))
    plate_rgb = base_rgb
    if closed:
        t = closed[len(closed) // 2]
        plate_rgb = part_color(read_frame(args.video, t), by[args.plate_closed][t].full_mask(h, w))

    base_poly = hull(top_m[:, :2])
    v, f = prism(base_poly, np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]),
                 z_floor, z_top)
    write_stl(mesh_dir / "base.stl", v, f)
    sr = np.column_stack([s, r])
    outline = [sr, [[s.min(), 0.0], [s.max(), 0.0]]]  # the plate reaches the hinge
    if back:
        b_ = np.vstack(back)
        b_ = b_[np.abs(b_ @ e3 - np.median(b_ @ e3)) < 0.01]  # on the plate, not the base's edge
        outline.append(np.percentile(np.column_stack([b_ @ e1, b_ @ e2]), [1, 99], axis=0))  # robust extent
        lo, hi = outline[-1]
        outline[-1] = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        print(f"plate outline from its closed back: {1000 * (hi[0] - lo[0]):.0f} x {1000 * (hi[1] - lo[1]):.0f} mm")
    lid_poly = hull(np.vstack(outline))
    v, f = prism(lid_poly, np.zeros(3), e1, e2, e3, w_face - thick, w_face)
    write_stl(mesh_dir / "lid.stl", v, f)
    v, f = prism(hull(sr), np.zeros(3), e1, e2, e3, w_face, w_face + SCREEN_DEPTH)
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
    (track_dir / "model.urdf").write_text(emit3d.urdf(args.video.stem, links))
    (track_dir / "model.xml").write_text(emit3d.mjcf(args.video.stem, links, root_pos, (w, h)))
    (track_dir / "model3d.json").write_text(json.dumps({
        "reference_frame": ref, "camera_to_model": {"rotation": rot.tolist(), "hinge_point": point.tolist(),
                                                    "root_pos": root_pos.tolist()},
        "base_thickness_m": float(z_top - z_floor), "lid_thickness_m": thick,
        "hinge_limits_rad": limits}))
    print(f"hinge range {np.rad2deg(limits[0]):.0f}..{np.rad2deg(limits[1]):.0f} deg; writing replay...")

    # Replay: the model, moved by the measured hinge, seen from the iPhone's camera each frame.
    cam_to_world = []
    a = np.eye(4)
    a[:3, :3], a[:3, 3] = rot, -rot @ point + root_pos
    for t in range(n):
        cam_to_world.append(a @ np.linalg.inv(world[t]))
    cover = replay3d(track_dir / "model.xml", np.array(cam_to_world), cap.intrinsics, {"lid_joint": q},
                     {"lid": by[args.plate], "base": by[args.base]}, args.video, track_dir / "replay3d.mp4",
                     only_when={"base": lambda t: not args.open_indicator or t in by[args.open_indicator]})
    for k, v in cover.items():
        print(f"  {k}: {100 * np.nanmean(v):.0f}% of its SAM mask covered by the model (mean over frames)")
    print(f"-> {track_dir}/model.urdf, model.xml, replay3d.mp4")


if __name__ == "__main__":
    main()
