"""Whatever you filmed -> a URDF/MJCF a simulated robot can use, in one command.

Usage:
  .venv/bin/python scripts/video2urdf.py INPUT [--out DIR] [--robot] [--viewer-out DIR] [--open-weights]

INPUT can be an iPhone LiDAR capture from the Stray Scanner app (its zip, or the folder, at any
depth), a phone video (iPhone .MOV: HEVC, HDR, rotated, 60/120 fps, variable frame rate;
Android .mp4; any format ffmpeg reads), a fixed-camera RGB-D folder (depth.npz +
intrinsics.json + video), a folder of photos taken while the part moves, or a zip of any of
these. What it is comes from the content, not the name.

The run: ingest and normalise -> a cheap capture check with reshoot tips -> Muse Spark looks at a
few frames -> the 3D path (depth: LiDAR, or MapAnything on Modal for a moving or oblique camera)
or the planar path (a still camera facing the motion) -> the agent builds and checks the model
-> a rigid fallback if it couldn't -> physics, video texture, shape design, completion (when
available), optional robot tasks -> a viewer folder and summary.json / summary.md.
--check-only stops before anything expensive (only Spark's look is called; --no-look skips it).

Examples:
  .venv/bin/python scripts/video2urdf.py ~/Downloads/9a84cd46da.zip --robot
  .venv/bin/python scripts/video2urdf.py data/real/notebook_2.MOV --out data/runs
  .venv/bin/python scripts/video2urdf.py my_clip.mp4 --check-only
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from video2urdf.ingest import IMAGE_FPS, MAX_FPS, MAX_FRAMES, MAX_SIDE  # noqa: E402
from video2urdf.run import Options, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="video, photo folder, Stray Scanner capture/zip or RGB-D folder")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "runs", help="runs go to OUT/NAME (default data/runs)")
    ap.add_argument("--name", help="run name (default: the input's name)")
    ap.add_argument("--path", choices=["auto", "3d", "planar"], default="auto",
                    help="force the 3D path (depth) or the planar side-view path")
    ap.add_argument("--rotate", choices=["auto", "0", "90", "180", "270"], default="auto",
                    help="turn a plain video clockwise by this much (auto: when Spark sees it sideways)")
    ap.add_argument("--robot", action="store_true", help="also let a simulated Franka Panda use the model")
    ap.add_argument("--viewer-out", type=Path, help="write the standalone viewer here (default RUN/viewer)")
    ap.add_argument("--open-weights", action="store_true", help="Muse Spark Open + SAM 3.1 open weights only")
    ap.add_argument("--glimmer", action="store_true", help="reasoning on Muse Glimmer 30B (our Modal endpoint)")
    ap.add_argument("--sam", choices=["api", "open"], help="SAM 3.1 on the Meta API or open weights on Modal")
    ap.add_argument("--no-design", action="store_true", help="skip Spark's shape design")
    ap.add_argument("--no-texture", action="store_true", help="skip projecting the video onto the meshes")
    ap.add_argument("--no-look", action="store_true", help="skip Spark's look at a few frames before routing")
    ap.add_argument("--check-only", action="store_true", help="ingest, check and route only (no agent)")
    ap.add_argument("--focal-35mm", type=float, help="plain video, 3D path: the camera's 35 mm-equivalent focal")
    ap.add_argument("--scale", type=float, help="plain video, 3D path: multiply MapAnything's scale by this")
    ap.add_argument("--no-auto-scale", action="store_true", help="plain video, 3D path: keep MapAnything's scale")
    ap.add_argument("--max-fps", type=float, default=MAX_FPS, help=f"plain video frame-rate cap (default {MAX_FPS:g})")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                    help=f"plain video frame cap after normalising (default {MAX_FRAMES})")
    ap.add_argument("--max-side", type=int, default=MAX_SIDE, help=f"plain video long-side cap, px (default {MAX_SIDE})")
    ap.add_argument("--image-fps", type=float, default=IMAGE_FPS, help="frame rate given to a photo folder")
    args = ap.parse_args()
    if args.scale is not None and args.scale <= 0:
        ap.error("--scale must be positive")
    if args.viewer_out and args.viewer_out.exists() and any(args.viewer_out.iterdir()):
        ap.error(f"{args.viewer_out} is not empty; choose a new --viewer-out")
    if args.max_fps <= 0 or args.max_frames < 8 or args.max_side < 160:
        ap.error("--max-fps must be > 0, --max-frames >= 8, --max-side >= 160")
    opt = Options(out=args.out, name=args.name, path=args.path, rotate=args.rotate, look=not args.no_look,
                  open_weights=args.open_weights, glimmer=args.glimmer, sam=args.sam, design=not args.no_design,
                  texture=not args.no_texture, robot=args.robot, viewer_out=args.viewer_out,
                  check_only=args.check_only, focal_35mm=args.focal_35mm, scale=args.scale,
                  auto_scale=not args.no_auto_scale, max_fps=args.max_fps, max_frames=args.max_frames,
                  max_side=args.max_side, image_fps=args.image_fps)
    summary = run(args.input, opt)
    sys.exit(0 if summary["status"] in ("checks_passed", "needs_review", "checked") else 1)


if __name__ == "__main__":
    main()
