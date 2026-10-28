"""Tracks -> joints -> URDF/MJCF -> MuJoCo replay over the video.

Run scripts/track_video.py first. The first tracked prompt is the root (the fixed part).

Usage:
  .venv/bin/python scripts/fit_model.py data/real/notebook.mov --ref "laptop lid=0.212"
  .venv/bin/python scripts/fit_model.py data/synth/lamp.mp4 --gt data/synth/lamp.gt.json

Scale: --px-per-m, or --ref "PROMPT=METERS" (the long side of that part's mask on the
reference frame), or --gt (synthetic camera). Writes into the track folder: model.urdf,
model.xml, joints.json and replay.mp4.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf.pipeline import fit, load_parts, long_side_px, slug, video_info  # noqa: E402


def gt_px_per_m(gt: dict) -> float:
    w, h = gt["size"]
    f = 0.5 * h / np.tan(np.deg2rad(gt["camera"]["fovy"]) / 2)
    return f / -gt["camera"]["pos"][1]  # parts move in the y = 0 plane


def evaluate(gt: dict, links, ref: int, px_per_m: float, origin_px) -> None:
    body_of = {slug(p): b for b, p in gt["prompts"].items()}
    u0, v0 = origin_px
    for l in links:
        if not l.joint:
            continue
        gj = next((j for j in gt["joints"] if j["child"] == body_of.get(l.name)), None)
        if gj is None:
            print(f"  {l.name}: no ground-truth joint")
            continue
        q_gt = np.array(gj["q"]) - gj["q"][ref]
        gt_axis = np.array(gj["axis"])
        if l.joint.type == "revolute":
            sign = gt_axis @ np.array([0, -1, 0])
        else:
            sign = gt_axis @ np.array([l.joint.axis[0], 0, l.joint.axis[1]])
        valid = ~np.isnan(l.joint.q)
        err = np.abs(l.joint.q[valid] - sign * q_gt[valid])
        expected = "revolute" if gj["type"] == "hinge" else "prismatic"
        line = f"  {l.name}: type {l.joint.type} (gt {expected}{', OK' if l.joint.type == expected else ', WRONG'})"
        if l.joint.type == "revolute":
            a = l.joint.anchor
            px = np.array([u0 + px_per_m * a[0], v0 - px_per_m * a[1]])
            d = np.linalg.norm(px - np.array(gj["anchor_px"][ref]))
            line += (f", axis position error {d:.1f} px = {1000 * d / px_per_m:.1f} mm, "
                     f"angle error mean {np.rad2deg(err.mean()):.2f} deg, max {np.rad2deg(err.max()):.2f} deg")
        else:
            gt_dir = np.array([gt_axis[0], gt_axis[2]])
            ang = np.rad2deg(np.arccos(min(1.0, abs(l.joint.axis @ gt_dir) / np.linalg.norm(gt_dir))))
            line += f", axis direction error {ang:.2f} deg, position error mean {1000 * err.mean():.1f} mm"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--tracks", type=Path, default=ROOT / "data" / "tracks")
    ap.add_argument("--px-per-m", type=float)
    ap.add_argument("--ref", help='"PART=METERS": real length of the long side of a part')
    ap.add_argument("--gt", type=Path, help="synthetic gt.json: scale from its camera, and report errors")
    args = ap.parse_args()

    track_dir = args.tracks / args.video.stem
    gt = json.loads(args.gt.read_text()) if args.gt else None
    if args.px_per_m:
        px_per_m = args.px_per_m
    elif args.ref:
        part, meters = args.ref.rsplit("=", 1)
        names, by_name = load_parts(track_dir)
        (w, h), n, _ = video_info(args.video)
        ref = next(t for t in range(n) if all(t in by_name[nm] for nm in names))
        px_per_m = long_side_px(by_name[slug(part)][ref].full_mask(h, w)) / float(meters)
    elif gt:
        px_per_m = gt_px_per_m(gt)
    else:
        sys.exit("need a scale: --px-per-m, --ref or --gt")

    print("part poses (mask registration; surface points where the mask is occluded)...")
    result = fit(args.video, track_dir, px_per_m)
    print(result.summary())
    if gt:
        print("ground truth:")
        evaluate(gt, result.links, result.ref, result.px_per_m, result.origin)
    print(f"-> {track_dir}/model.urdf, model.xml, replay.mp4")


if __name__ == "__main__":
    main()
