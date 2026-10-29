"""Score a MapAnything capture (scripts/video_to_capture.py) against the iPhone LiDAR capture of
the same video: depth error and scale, and camera motion against ARKit's.

Usage: .venv/bin/python scripts/eval_pseudo_depth.py data/real/laptop_mapanything data/real/laptop_lidar
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pseudo", type=Path)
    ap.add_argument("lidar", type=Path)
    args = ap.parse_args()
    p, l = stray.load(args.pseudo), stray.load(args.lidar)
    n = min(len(p.depth), len(l.depth))
    dh, dw = l.depth.shape[1:]
    err, ratio, rel = [], [], []
    for t in range(n):
        d = cv2.resize(p.depth[t], (dw, dh), interpolation=cv2.INTER_NEAREST)
        ok = (d > 0) & (l.depth[t] > 0)
        if ok.sum() < 100:
            continue
        err.append(np.median(np.abs(d[ok] - l.depth[t][ok])))
        ratio.append(np.median(l.depth[t][ok] / d[ok]))
        rel.append(np.median(np.abs(d[ok] * ratio[-1] - l.depth[t][ok]) / l.depth[t][ok]))
    tp = stray.ref_to_frame_poses(args.pseudo, n, 0)
    tl = stray.ref_to_frame_poses(args.lidar, n, 0)
    rot = [np.rad2deg(np.arccos(np.clip((np.trace(a[:3, :3] @ b[:3, :3].T) - 1) / 2, -1, 1))) for a, b in zip(tp, tl)]
    cp = np.array([np.linalg.inv(a)[:3, 3] for a in tp])  # camera centres in frame 0's coordinates
    cl = np.array([np.linalg.inv(b)[:3, 3] for b in tl])
    trans = np.linalg.norm(cp - cl, axis=1)
    res = {
        "frames": n,
        "depth_abs_err_mm_median": 1000 * float(np.median(err)),
        "depth_scale_lidar_over_pseudo_median": float(np.median(ratio)),
        "depth_scale_spread_p10_p90": [float(np.percentile(ratio, 10)), float(np.percentile(ratio, 90))],
        "depth_rel_err_after_scale_median": float(np.median(rel)),
        "lidar_depth_median_m": float(np.median(l.depth[l.depth > 0])),
        "camera_rot_err_deg_median_max": [float(np.median(rot)), float(np.max(rot))],
        "camera_pos_err_mm_median_max": [1000 * float(np.median(trans)), 1000 * float(np.max(trans))],
        "camera_path_mm_lidar": 1000 * float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum()),
        "camera_path_mm_pseudo": 1000 * float(np.linalg.norm(np.diff(cp, axis=0), axis=1).sum()),
        "camera_max_offset_mm_lidar": 1000 * float(np.linalg.norm(cl, axis=1).max()),
    }
    (args.pseudo / "eval_vs_lidar.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
