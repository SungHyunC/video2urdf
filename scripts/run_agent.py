"""Let Muse Spark build the model for one video.

Usage: .venv/bin/python scripts/run_agent.py data/real/stand_2.MOV [--out data/tracks]
       .venv/bin/python scripts/run_agent.py data/real/laptop_lidar   (a Stray Scanner LiDAR capture)
       .venv/bin/python scripts/run_agent.py data/rbo/prep/laptop02   (a fixed-camera RGB-D folder)

Writes the model files plus agent/log.md (every step, with the images Spark saw) and
agent/report.md into <out>/<video stem>/. AGENT_MODEL, AGENT_BASE_URL and AGENT_API_KEY_ENV
switch the reasoning model and SAM_BACKEND the segmenter. --open-weights runs everything on
open weights: Muse Spark Open on Baseten and SAM 3.1 (facebook/sam3.1) on Modal
(modal deploy modal_apps/sam3_open.py first).
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

from video2urdf.stray import find_capture  # noqa: E402

OPEN_WEIGHTS = {"AGENT_MODEL": "meta-models/muse-spark-open", "AGENT_BASE_URL": "https://inference.baseten.co/v1",
                "AGENT_API_KEY_ENV": "BASETEN_API_KEY", "AGENT_MAX_IMAGES": "8", "SAM_BACKEND": "open"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "tracks")
    ap.add_argument("--open-weights", action="store_true", help="Muse Spark Open + SAM 3.1 weights only")
    ap.add_argument("--sam", choices=["api", "open"], help="override the segmenter alone")
    args = ap.parse_args()
    if args.open_weights:
        os.environ.update(OPEN_WEIGHTS)
    if args.sam:
        os.environ["SAM_BACKEND"] = args.sam
    from video2urdf.agent import Agent  # reads the backend settings on import
    video, capture = args.video, None
    if args.video.is_dir() and (args.video / "intrinsics.json").exists():  # fixed-camera RGB-D (rbo_eval.py --prepare-only)
        capture, video = args.video, next(args.video.glob("*.mp4"))
    elif args.video.is_dir():  # an iPhone LiDAR capture (Stray Scanner export, possibly nested)
        capture = args.video
        rgb = find_capture(capture) / "rgb.mp4"
        video = capture / f"{capture.name}.mp4"  # tracks are filed under the video's name
        if not video.exists():
            shutil.copy(rgb, video)
    report = Agent(video, args.out / video.stem, capture=capture).run()
    print("\n=== report ===\n" + report)


if __name__ == "__main__":
    main()
