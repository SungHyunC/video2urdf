"""3D joint fit from tracks plus per-frame depth (experiment).

Usage:
  .venv/bin/python scripts/fit3d.py data/synth3d/lamp_az40_el25.mp4 \\
      --depth data/synth3d/lamp_az40_el25.depth.npz --gt data/synth3d/lamp_az40_el25.gt3d.json \\
      --map red_rod=arm,blue_object=forearm

Run scripts/track_video.py on the video first (root part first). With --gt, intrinsics come
from the ground truth and errors are reported against it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf.lift3d import Intrinsics, depth_edges, fit_joint_3d, joint_poses, part_poses_3d  # noqa: E402
from video2urdf.motion import load_gray  # noqa: E402
from video2urdf.pipeline import load_parts, video_info  # noqa: E402

MUJOCO_TO_CV = np.diag([1.0, -1.0, -1.0])  # MuJoCo camera (y up, z back) -> x right, y down, z forward


def line_distance(p: np.ndarray, point: np.ndarray, axis: np.ndarray) -> float:
    d = p - point
    return float(np.linalg.norm(d - (d @ axis) * axis))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--depth", type=Path, required=True, help="npz with depth (n, H, W) in metres")
    ap.add_argument("--gt", type=Path)
    ap.add_argument("--map", default="", help="part=gt_body,... for evaluation")
    ap.add_argument("--tracks", type=Path, default=ROOT / "data" / "tracks")
    args = ap.parse_args()

    names, by_name = load_parts(args.tracks / args.video.stem)
    size, n, _ = video_info(args.video)
    w, h = size
    depth = np.load(args.depth)["depth"].astype(np.float32)
    gt = json.loads(args.gt.read_text()) if args.gt else None
    intr = Intrinsics(**gt["intrinsics"])
    grays, k = load_gray(args.video)
    ref = next(t for t in range(n) if all(t in by_name[nm] for nm in names))

    poses, centroids, clouds = {}, {}, {}
    for nm in names:
        others = {}
        for o in names:
            if o != nm:
                for t, det in by_name[o].items():
                    others.setdefault(t, []).append(det)
        poses[nm] = part_poses_3d(grays, k, depth, intr, by_name[nm], others, ref, size)
        m = by_name[nm][ref].full_mask(h, w) & ~depth_edges(depth[ref])
        vv, uu = np.nonzero(m)
        z = depth[ref][vv, uu]
        ok = z > 0
        cloud = intr.lift(np.stack([uu[ok], vv[ok]], 1).astype(float), z[ok])
        centroids[nm] = cloud.mean(axis=0)
        clouds[nm] = cloud[np.random.default_rng(0).choice(len(cloud), min(300, len(cloud)), replace=False)]
        print(f"  {nm}: 3D pose on {int((~np.isnan(poses[nm]).any(axis=(1, 2))).sum())}/{n} frames")

    # Chain in prompt order for now: each part hangs off the previous one.
    body_of = dict(kv.split("=") for kv in args.map.split(",") if kv)
    model_pose = {names[0]: poses[names[0]]}  # down the chain, parents move as their fitted joints say
    for parent, child in zip(names, names[1:]):
        j = fit_joint_3d(model_pose[parent], poses[child], centroids[child], clouds[child])
        model_pose[child] = joint_poses(model_pose[parent], j)
        lo, hi = np.nanmin(j.q), np.nanmax(j.q)
        rng = f"{np.rad2deg(lo):.1f}..{np.rad2deg(hi):.1f} deg" if j.type == "revolute" else f"{1000 * lo:.0f}..{1000 * hi:.0f} mm"
        print(f"{parent} -> {child}: {j.type}, axis {np.round(j.axis, 3)}, range {rng}, residual {1000 * j.residual:.1f} mm")
        if not gt or child not in body_of:
            continue
        gj = next(x for x in gt["joints"] if x["child"] == body_of[child])
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


if __name__ == "__main__":
    main()
