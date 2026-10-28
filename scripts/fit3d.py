"""3D joint fit from tracks plus per-frame depth.

Synthetic (scored against ground truth):
  .venv/bin/python scripts/fit3d.py data/synth3d/lamp_az40_el25.mp4 \\
      --depth data/synth3d/lamp_az40_el25.depth.npz --gt data/synth3d/lamp_az40_el25.gt3d.json \\
      --map red_rod=arm,blue_object=forearm
iPhone LiDAR (Stray Scanner export; the video is its rgb.mp4, tracked under its own name):
  .venv/bin/python scripts/fit3d.py data/real/laptop_lidar/laptop_lidar.mp4 --stray data/real/laptop_lidar \\
      --pairs wooden_floor:laptop_lid,wooden_floor:laptop_keyboard

Run scripts/track_video.py on the video first. Without --pairs, parts chain in prompt order.
Writes axes.jpg (each fitted axis drawn on the reference frame) and joints3d.json into the
track folder.
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
from video2urdf.lift3d import (Intrinsics, fit_joint_3d, joint_poses, mask_cloud, part_poses_3d,  # noqa: E402
                                track_angle)
from video2urdf.motion import load_gray  # noqa: E402
from video2urdf.pipeline import load_parts, read_frame, video_info  # noqa: E402

MUJOCO_TO_CV = np.diag([1.0, -1.0, -1.0])  # MuJoCo camera (y up, z back) -> x right, y down, z forward


def line_distance(p: np.ndarray, point: np.ndarray, axis: np.ndarray) -> float:
    d = p - point
    return float(np.linalg.norm(d - (d @ axis) * axis))


def project(intr: Intrinsics, x: np.ndarray) -> np.ndarray:
    return np.stack([intr.f * x[:, 0] / x[:, 2] + intr.cx, intr.f * x[:, 1] / x[:, 2] + intr.cy], axis=1)


def evaluate(gt: dict, j, child_body: str, ref: int):
    gj = next(x for x in gt["joints"] if x["child"] == child_body)
    rot = np.array(gt["camera"]["rot"][ref])
    pos = np.array(gt["camera"]["pos"][ref])
    axis_gt = MUJOCO_TO_CV @ rot.T @ np.array(gj["axis_world"][ref])
    anchor_gt = MUJOCO_TO_CV @ rot.T @ (np.array(gj["anchor_world"][ref]) - pos)
    ang = np.rad2deg(np.arccos(min(1.0, abs(axis_gt @ j.axis))))
    sign = np.sign(axis_gt @ j.axis)
    q_gt = np.array(gj["q"]) - gj["q"][ref]
    valid = ~np.isnan(j.q)
    err = np.abs(j.q[valid] - sign * q_gt[valid])
    if j.type == "revolute":
        print(f"   vs ground truth: axis direction error {ang:.2f} deg, axis position error "
              f"{1000 * line_distance(anchor_gt, j.point, j.axis):.1f} mm, angle error mean "
              f"{np.rad2deg(err.mean()):.2f} deg, max {np.rad2deg(err.max()):.2f} deg")
    else:
        print(f"   vs ground truth: axis direction error {ang:.2f} deg, travel error mean "
              f"{1000 * err.mean():.1f} mm, max {1000 * err.max():.1f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--depth", type=Path, help="npz with depth (n, H, W) in metres (synthetic)")
    ap.add_argument("--gt", type=Path, help="synthetic gt3d.json: intrinsics, and errors against it")
    ap.add_argument("--stray", type=Path, help="Stray Scanner capture folder (depth and intrinsics)")
    ap.add_argument("--pairs", default="", help="parent:child,... (default: chain in prompt order)")
    ap.add_argument("--map", default="", help="part=gt_body,... for evaluation")
    ap.add_argument("--tracks", type=Path, default=ROOT / "data" / "tracks")
    args = ap.parse_args()

    track_dir = args.tracks / args.video.stem
    names, by_name = load_parts(track_dir)
    size, n, _ = video_info(args.video)
    w, h = size
    gt = json.loads(args.gt.read_text()) if args.gt else None
    if args.stray:
        cap = stray.load(args.stray)
        depth, intr = cap.depth, cap.intrinsics
    else:
        depth, intr = np.load(args.depth)["depth"].astype(np.float32), Intrinsics(**gt["intrinsics"])
    n = min(n, len(depth))
    grays, k = load_gray(args.video)
    grays = grays[:n]
    ref = next(t for t in range(n) if all(t in by_name[nm] for nm in names))
    print(f"reference frame {ref}, {n} frames, depth {depth.shape[2]}x{depth.shape[1]}")

    poses, clouds = {}, {}
    for nm in names:
        others = {}
        for o in names:
            if o != nm:
                for t, det in by_name[o].items():
                    others.setdefault(t, []).append(det)
        poses[nm] = part_poses_3d(grays, k, depth, intr, {t: d for t, d in by_name[nm].items() if t < n},
                                  others, ref, size)
        cloud = mask_cloud(by_name[nm][ref].full_mask(h, w), depth[ref], intr)
        clouds[nm] = cloud[np.random.default_rng(0).choice(len(cloud), min(400, len(cloud)), replace=False)]
        print(f"  {nm}: 3D pose on {int((~np.isnan(poses[nm]).any(axis=(1, 2))).sum())}/{n} frames, "
              f"{len(cloud)} depth points")

    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p] or list(zip(names, names[1:]))
    body_of = dict(kv.split("=") for kv in args.map.split(",") if kv)
    model_pose = {names[0]: poses[names[0]]}  # down a chain, parents move as their fitted joints say
    image = read_frame(args.video, ref)
    results = []
    for parent, child in pairs:
        base = model_pose.get(parent, poses[parent])
        j = fit_joint_3d(base, poses[child], clouds[child].mean(axis=0), clouds[child])
        if j.type == "revolute" and np.isnan(j.q).any():
            # Fill frames where surface tracking was lost by matching depth to the rotated part.
            observed = {}
            for t, det in by_name[child].items():
                if t < n and np.isnan(j.q[t]):
                    c = mask_cloud(det.full_mask(h, w), depth[t], intr)
                    if len(c) >= 30:
                        observed[t] = c[np.random.default_rng(t).choice(len(c), min(300, len(c)), replace=False)]
            filled = track_angle(j, base, clouds[child], observed, n)
            print(f"   angle from depth matching on {int((np.isnan(j.q) & ~np.isnan(filled)).sum())} more frames")
            j.q = filled
        model_pose[child] = joint_poses(base, j)
        lo, hi = np.nanmin(j.q), np.nanmax(j.q)
        span = (f"{np.rad2deg(lo):.1f}..{np.rad2deg(hi):.1f} deg" if j.type == "revolute"
                else f"{1000 * lo:.0f}..{1000 * hi:.0f} mm")
        along = (clouds[child] - j.point) @ j.axis  # the part's extent along the axis
        print(f"{parent} -> {child}: {j.type}, range {span}, residual {1000 * j.residual:.1f} mm, "
              f"{child} spans {1000 * np.ptp(along):.0f} mm along the axis, "
              f"{int((~np.isnan(j.q)).sum())}/{n} frames")
        results.append({"parent": parent, "child": child, "type": j.type, "axis": j.axis.tolist(),
                        "point": j.point.tolist(), "residual_m": j.residual,
                        "q": [None if np.isnan(v) else float(v) for v in j.q]})
        if gt and child in body_of:
            evaluate(gt, j, body_of[child], ref)
        ends = j.point + np.outer([along.min(), along.max()], j.axis)
        (u0, v0), (u1, v1) = project(intr, ends).astype(int)
        color = (0, 0, 255) if j.type == "revolute" else (255, 0, 0)
        cv2.line(image, (u0, v0), (u1, v1), color, 8)
        cv2.putText(image, f"{parent}->{child}", (u0, v0 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    cv2.imwrite(str(track_dir / "axes.jpg"), image)
    (track_dir / "joints3d.json").write_text(json.dumps({"reference_frame": ref, "joints": results}))
    print(f"-> {track_dir}/axes.jpg, joints3d.json")


if __name__ == "__main__":
    main()
