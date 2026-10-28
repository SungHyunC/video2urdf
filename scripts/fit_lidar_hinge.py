"""Hinge of a flat plate from an iPhone LiDAR capture (see video2urdf/lidar.py).

Usage:
  .venv/bin/python scripts/fit_lidar_hinge.py data/real/laptop_lidar/laptop_lidar.mp4 \\
      --stray data/real/laptop_lidar --floor wooden_floor --base silver_laptop \\
      --plate laptop_screen --plate-closed silver_laptop --open-indicator laptop_keyboard

Run scripts/track_video.py on the video first. Writes hinge.json and hinge.jpg (axis drawn on
the reference frame, opening over time) into the track folder.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf import lidar  # noqa: E402


def roles_args(ap: argparse.ArgumentParser):
    ap.add_argument("video", type=Path)
    ap.add_argument("--stray", type=Path, required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--base", required=True, help="part the plate hinges on, on the reference frame")
    ap.add_argument("--plate", required=True, help="the plate's inner face (e.g. the screen)")
    ap.add_argument("--plate-closed", help="what covers the object when closed (the plate's back)")
    ap.add_argument("--open-indicator", help="part seen only while open; frames without it count as closed")
    ap.add_argument("--ref", type=int, default=0)
    ap.add_argument("--tracks", type=Path, default=ROOT / "data" / "tracks")


def load(args) -> tuple[lidar.Capture, lidar.Roles]:
    c = lidar.Capture.from_stray(args.video, args.stray, args.tracks / args.video.stem, args.ref)
    fix = c.level_with(args.floor)
    print(f"floor levelling: ARKit drift up to {fix['tilt_deg_max']:.1f} deg tilt, {fix['shift_mm_max']:.0f} mm "
          f"height (median {fix['shift_mm_median']:.0f} mm), removed")
    return c, lidar.Roles(args.floor, args.base, args.plate, args.plate_closed, args.open_indicator)


def main():
    ap = argparse.ArgumentParser()
    roles_args(ap)
    c, roles = load(ap.parse_args())
    lidar.fit_hinge(c, roles)
    print(f"-> {c.track_dir}/hinge.jpg, hinge.json")


if __name__ == "__main__":
    main()
