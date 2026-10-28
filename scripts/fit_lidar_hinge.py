"""Hinge of a flat plate (laptop lid, door, flap) from an iPhone LiDAR capture.

The camera's own motion comes from ARKit (Stray Scanner odometry), so the static scene is
the reference; the plate is tracked by fitting its plane in each frame's depth, which works
when the plate shows a different face at different angles (screen, then aluminium back).

Usage:
  .venv/bin/python scripts/fit_lidar_hinge.py data/real/laptop_lidar/laptop_lidar.mp4 \\
      --stray data/real/laptop_lidar --plate laptop_screen --plate-fallback silver_laptop --base laptop_keyboard

Run scripts/track_video.py on the video first. Writes hinge.json and hinge.jpg (axis drawn
on the reference frame, angle over time) into the track folder.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf import stray  # noqa: E402
from video2urdf.lift3d import clean_angles, fit_plane, fit_planar_hinge, mask_cloud  # noqa: E402
from video2urdf.pipeline import load_parts, read_frame, video_info  # noqa: E402


def project(intr, x):
    return np.stack([intr.f * x[:, 0] / x[:, 2] + intr.cx, intr.f * x[:, 1] / x[:, 2] + intr.cy], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--stray", type=Path, required=True)
    ap.add_argument("--plate", required=True, help="tracked part that is the moving plate")
    ap.add_argument("--plate-fallback", help="part to use for the plate in frames where --plate is not tracked")
    ap.add_argument("--base", required=True, help="tracked part the plate hinges on (seen only while open)")
    ap.add_argument("--base-plane", help="part whose reference-frame cloud gives the base plane (default --base)")
    ap.add_argument("--ref", type=int, default=0)
    ap.add_argument("--tracks", type=Path, default=ROOT / "data" / "tracks")
    args = ap.parse_args()

    track_dir = args.tracks / args.video.stem
    names, by = load_parts(track_dir)
    cap = stray.load(args.stray)
    (w, h), n, _ = video_info(args.video)
    n = min(n, len(cap.depth))
    world = stray.ref_to_frame_poses(args.stray, n, args.ref)

    def cloud(part, t):
        c = mask_cloud(by[part][t].full_mask(h, w), cap.depth[t], cap.intrinsics)
        inv = np.linalg.inv(world[t])
        return c @ inv[:3, :3].T + inv[:3, 3]  # into the reference frame

    plate, used_fallback = {}, 0
    for t in range(n):
        if t in by[args.plate]:
            plate[t] = cloud(args.plate, t)
        elif args.plate_fallback and t in by[args.plate_fallback] and t not in by[args.base]:
            plate[t] = cloud(args.plate_fallback, t)  # closed: the fallback's top face is the plate
            used_fallback += 1
    base = cloud(args.base_plane or args.base, args.ref)
    nb, db, _ = fit_plane(base, np.random.default_rng(0))
    hinge = fit_planar_hinge(plate, n, args.ref, pin=base.mean(axis=0))
    raw = int((~np.isnan(hinge.q)).sum())
    hinge.q = clean_angles(hinge.q)
    valid = ~np.isnan(hinge.q)
    print(f"dropped {raw - int(valid.sum())} outlier frames")
    # Opening angle on the reference frame: between the base's direction away from the hinge
    # and the plate's, both perpendicular to the axis (0 = closed, 180 = flat open).
    def radial(x):
        v = x - hinge.point
        v -= (v @ hinge.axis) * hinge.axis
        return v / np.linalg.norm(v)
    t_ref = min(hinge.centroids, key=lambda t: abs(t - args.ref))
    opening = np.rad2deg(np.arccos(np.clip(radial(hinge.centroids[t_ref]) @ radial(base.mean(axis=0)), -1, 1)))
    print(f"plate planes on {int(valid.sum())}/{n} frames ({used_fallback} from {args.plate_fallback})")
    print(f"hinge axis {np.round(hinge.axis, 3)}, angle to base plane {np.rad2deg(np.arcsin(abs(hinge.axis @ nb))):.1f} deg "
          f"(0 = hinge lies in the base plane), plate offset residual {1000 * hinge.residual:.1f} mm")
    q = np.rad2deg(hinge.q[valid])
    print(f"plate rotation range {q.min():.1f}..{q.max():.1f} deg (span {np.ptp(q):.0f}); "
          f"opening on the reference frame {opening:.0f} deg, so {opening + q.min():.0f}..{opening + q.max():.0f} deg "
          f"over the video (0 = closed)")
    along = (plate[args.ref] - hinge.point) @ hinge.axis if args.ref in plate else np.zeros(2)
    print(f"plate spans {1000 * np.ptp(along):.0f} mm along the hinge")

    image = read_frame(args.video, args.ref)
    ends = hinge.point + np.outer([along.min(), along.max()], hinge.axis)
    (u0, v0), (u1, v1) = project(cap.intrinsics, ends).astype(int)
    cv2.line(image, (u0, v0), (u1, v1), (0, 0, 255), 10)
    cv2.putText(image, "measured hinge", (u0, v0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
    # Angle over time, as a strip under the frame.
    strip = np.full((300, w, 3), 255, np.uint8)
    ts = np.flatnonzero(valid)
    lo, hi = q.min(), q.max()
    pts = np.stack([ts * (w - 1) / max(n - 1, 1), 280 - (q - lo) / max(hi - lo, 1e-6) * 260], axis=1).astype(int)
    for run in np.split(np.arange(len(ts)), np.flatnonzero(np.diff(ts) > 3) + 1):  # gaps stay gaps
        cv2.polylines(strip, [pts[run]], False, (200, 60, 0), 4)
    cv2.putText(strip, f"plate angle over time: {lo:.0f}..{hi:.0f} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.imwrite(str(track_dir / "hinge.jpg"), cv2.vconcat([image, strip]))
    (track_dir / "hinge.json").write_text(json.dumps({
        "reference_frame": args.ref, "axis": hinge.axis.tolist(), "point": hinge.point.tolist(),
        "residual_m": hinge.residual, "q": [None if np.isnan(v) else float(v) for v in hinge.q]}))
    print(f"-> {track_dir}/hinge.jpg, hinge.json")


if __name__ == "__main__":
    main()
