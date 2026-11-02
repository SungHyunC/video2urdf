"""The object-prior scale of a MapAnything capture, measured on saved tracks: the same length
measurement as autoscale.prior_scale (video2urdf/autoscale.largest_extent, on the frames it samples),
with the object's mask taken as the union of the tracked parts (the floor and any --exclude parts
left out) and its usual size given (Muse Spark's recorded answer), so the scale step is
repeatable for the plain-video benchmark (scripts/eval_plain_video.py). Prints the factor; with
--apply, rescales the capture by it (as video_to_capture.py --scale does).

Usage: .venv/bin/python scripts/scale_by_usual_size.py CAPTURE TRACK_DIR LO_MM HI_MM [--apply]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from video2urdf import pseudo_depth, stray  # noqa: E402
from video2urdf.autoscale import _frames, largest_extent  # noqa: E402
from video2urdf.pipeline import load_parts, video_info  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path)
    ap.add_argument("tracks", type=Path, help="a run folder with tracks.sam.* (e.g. the LiDAR run of the same video)")
    ap.add_argument("lo", type=float, help="usual largest dimension, low end (mm)")
    ap.add_argument("hi", type=float, help="usual largest dimension, high end (mm)")
    ap.add_argument("--exclude", nargs="*", default=["floor", "table"], help="tracked parts that aren't the object")
    ap.add_argument("--apply", action="store_true", help="rescale the capture by the factor")
    args = ap.parse_args()
    cap = stray.load(args.capture)
    video = stray.find_capture(args.capture) / "rgb.mp4"
    (w, h), n, _ = video_info(video)
    names, by = load_parts(args.tracks)
    masks = {}
    for t in _frames(min(n, len(cap.depth)), 6):  # prior_scale's frames
        m = np.zeros((h, w), bool)
        for name in names:
            if name not in args.exclude and t in by[name]:
                m |= by[name][t].full_mask(h, w)
        if m.any():
            masks[t] = m
    found = largest_extent(cap, masks)
    if found is None:
        raise SystemExit("no frame shows the whole object with its long axis across the view")
    measured, views, depth = found
    typical = (args.lo * args.hi) ** 0.5
    factor = typical / measured
    print(f"measured {measured:.1f} mm on {views} views at {depth:.2f} m; usual {typical:.1f} mm: x{factor:.4f}")
    if args.apply:
        root = stray.find_capture(args.capture)
        pseudo_depth.rescale_capture(root, factor)
        meta_path = root / "mapanything.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["scale_applied"] = meta.get("scale_applied", 1.0) * factor
            meta["auto_scale"] = {"method": "object_prior", "scale_factor": factor, "typical_mm": [args.lo, args.hi],
                                  "measured_largest_before_mm": measured, "views": views,
                                  "note": "usual size given (benchmark), object mask from saved tracks"}
            meta_path.write_text(json.dumps(meta, indent=2))
        print(f"rescaled {root} x{factor:.4f}")


if __name__ == "__main__":
    main()
