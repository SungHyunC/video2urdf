"""Compare two hinge fits of the same video (hinge.json from lidar.fit_hinge, same reference
frame): axis direction, axis position and the opening angle frame by frame.

Usage: .venv/bin/python scripts/compare_hinges.py data/tracks_manual/laptop_mapanything data/tracks_agent/laptop_lidar
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", type=Path)
    ap.add_argument("truth", type=Path)
    args = ap.parse_args()
    a, b = (json.loads((p / "hinge.json").read_text()) for p in (args.test, args.truth))
    if a["reference_frame"] != b["reference_frame"]:
        raise SystemExit(f"reference frames differ: {a['reference_frame']} vs {b['reference_frame']}")
    ua, ub = np.array(a["axis"]), np.array(b["axis"])
    sign = 1.0 if ua @ ub >= 0 else -1.0
    angle = np.rad2deg(np.arccos(np.clip(abs(ua @ ub), -1, 1)))
    d = np.array(a["point"]) - np.array(b["point"])
    offset = np.linalg.norm(d - (d @ ub) * ub)  # test's axis point from the true axis line
    qa = np.array([np.nan if v is None else v for v in a["q"]]) * sign
    qb = np.array([np.nan if v is None else v for v in b["q"]])
    n = min(len(qa), len(qb))
    diff = np.rad2deg(np.abs(qa[:n] - qb[:n]))
    ok = ~np.isnan(diff)
    res = {"axis_angle_deg": float(angle), "axis_offset_mm": 1000 * float(offset),
           "opening_err_deg_median": float(np.median(diff[ok])), "opening_err_deg_p90": float(np.percentile(diff[ok], 90)),
           "frames_compared": int(ok.sum()),
           "range_deg_test": [float(np.rad2deg(np.nanmin(qa))), float(np.rad2deg(np.nanmax(qa)))],
           "range_deg_truth": [float(np.rad2deg(np.nanmin(qb))), float(np.rad2deg(np.nanmax(qb)))]}
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
