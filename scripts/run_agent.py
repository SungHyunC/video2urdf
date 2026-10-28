"""Let Muse Spark build the model for one video.

Usage: .venv/bin/python scripts/run_agent.py data/real/stand_2.MOV [--out data/tracks]
       .venv/bin/python scripts/run_agent.py data/real/laptop_lidar   (a Stray Scanner LiDAR capture)

Writes the model files plus agent/log.md (every step, with the images Spark saw) and
agent/report.md into <out>/<video stem>/. AGENT_MODEL, AGENT_BASE_URL and AGENT_API_KEY_ENV
switch the reasoning model, e.g. to Muse Spark Open on Baseten.
"""

import argparse
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from video2urdf.agent import Agent  # noqa: E402
from video2urdf.stray import find_capture  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "tracks")
    args = ap.parse_args()
    video, capture = args.video, None
    if args.video.is_dir():  # an iPhone LiDAR capture (Stray Scanner export, possibly nested)
        capture = args.video
        rgb = find_capture(capture) / "rgb.mp4"
        video = capture / f"{capture.name}.mp4"  # tracks are filed under the video's name
        if not video.exists():
            shutil.copy(rgb, video)
    report = Agent(video, args.out / video.stem, capture=capture).run()
    print("\n=== report ===\n" + report)


if __name__ == "__main__":
    main()
