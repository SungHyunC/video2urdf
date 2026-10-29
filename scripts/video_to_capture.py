"""A plain phone video -> an RGB-D capture in the Stray Scanner layout, with depth, intrinsics
and camera motion from MapAnything (video2urdf/pseudo_depth.py), so the LiDAR path runs on it.

Usage:
  modal deploy modal_apps/mapanything_open.py        (once)
  .venv/bin/python scripts/video_to_capture.py data/real/notebook_2.MOV --focal-35mm 26
  .venv/bin/python scripts/run_agent.py data/real/notebook_2_mapanything

MapAnything's own metric scale can be far off on close-up tabletop scenes (3x on the LiDAR
laptop), so sizes need one known length: measure the model once, then rerun with
--scale KNOWN/MEASURED (on an existing capture this only rescales it, no new inference). Give the camera's
focal length when known (--focal px, or --focal-35mm; an iPhone's 1x camera is ~26 mm):
the model's own estimate was 35% long on that capture, which distorts angles.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf import pseudo_depth  # noqa: E402
from video2urdf.pipeline import video_info  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, help="capture folder (default: next to the video, <stem>_mapanything)")
    ap.add_argument("--focal", type=float, help="focal length in pixels at the video's size")
    ap.add_argument("--focal-35mm", type=float, help="35 mm-equivalent focal length (iPhone 1x: ~26)")
    ap.add_argument("--scale", type=float, help="multiply the model's metric scale (known length / measured)")
    ap.add_argument("--chunk", type=int, default=pseudo_depth.CHUNK)
    ap.add_argument("--anchors", type=int, default=pseudo_depth.ANCHORS)
    args = ap.parse_args()
    out = args.out or args.video.parent / f"{args.video.stem}_mapanything"
    meta = out / "mapanything.json"
    if args.scale and meta.exists():  # an existing capture: rescale it, no new inference
        info = json.loads(meta.read_text())
        pseudo_depth.rescale_capture(out, args.scale)
        info["scale_applied"] = info.get("scale_applied", 1.0) * args.scale
        meta.write_text(json.dumps(info, indent=2))
        print(f"rescaled x{args.scale} (total x{info['scale_applied']:.4f}) -> {out}")
        return
    focal = args.focal
    if args.focal_35mm:
        focal = pseudo_depth.focal_from_35mm(args.focal_35mm, video_info(args.video)[0])
    rec = pseudo_depth.reconstruct(args.video, args.chunk, args.anchors, focal, args.scale)
    pseudo_depth.write_capture(rec, args.video, out)
    meta.write_text(json.dumps({
        "source": str(args.video), "frames": len(rec.depth), "depth_size": list(rec.depth.shape[2:0:-1]),
        "focal_px": float(rec.k[0, 0]), "focal_given": focal is not None, "scale_applied": args.scale or 1.0,
        "chunk_scales": rec.chunk_scales}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
