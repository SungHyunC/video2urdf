"""Spark names the object and rebuilds each link of a fitted model as clean primitives.

Usage: python scripts/design_shape.py TRACK_DIR VIDEO [--rounds 3] [--open-weights | --glimmer]
       python scripts/design_shape.py TRACK_DIR --remove     (back to the measured visuals)

Writes design.json, design/log.md (every overlay Spark saw) and, if the best design scores
close to the measured meshes on the SAM masks, puts it into model.xml / model.urdf as the
visual (the measured meshes stay as collision). Run after fit and physics, before export.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track_dir", type=Path)
    ap.add_argument("video", type=Path, nargs="?")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--no-apply", action="store_true", help="only write design.json and the logs")
    ap.add_argument("--remove", action="store_true", help="restore the model files from before the design")
    ap.add_argument("--open-weights", action="store_true", help="Muse Spark Open on Baseten")
    ap.add_argument("--glimmer", action="store_true", help="Muse Glimmer 30B on our Modal endpoint")
    args = ap.parse_args()
    if args.remove:
        for src in args.track_dir.glob("model.predesign.*"):
            shutil.copy(src, args.track_dir / src.name.replace("model.predesign.", "model.", 1))
        print("restored the measured visuals")
        return
    if args.video is None:
        ap.error("VIDEO is required")
    if args.open_weights or args.glimmer:
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_agent import OPEN_WEIGHTS
        if args.open_weights:
            os.environ.update(OPEN_WEIGHTS)
        if args.glimmer:
            os.environ["GLIMMER_API_KEY"] = f"{os.environ['MODAL_PROXY_TOKEN_ID']}.{os.environ['MODAL_PROXY_TOKEN_SECRET']}"
            os.environ.update({"AGENT_MODEL": "glimmer", "AGENT_BASE_URL": os.environ["GLIMMER_URL"].rstrip("/") + "/v1",
                               "AGENT_API_KEY_ENV": "GLIMMER_API_KEY", "AGENT_MAX_IMAGES": "8", "AGENT_VIDEO": "0"})
    from video2urdf.designer import design_shape  # reads the backend settings on import
    result = design_shape(args.track_dir, args.video, rounds=args.rounds, apply=not args.no_apply,
                          log=lambda s: print(s, flush=True))
    print(f"{result.get('object')}: {'applied' if result['applied'] else 'not applied'}; "
          f"see {args.track_dir / 'design' / 'log.md'}")


if __name__ == "__main__":
    main()
