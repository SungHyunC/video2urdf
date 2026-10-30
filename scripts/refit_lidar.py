"""Refit saved LiDAR tracks with the agent's own fit_lidar tool, on a copy, for regression checks.

Reuses a run's tracks and the roles the agent chose (roles.json): no new segmentation or model
calls. The source folder is left unchanged; the refit goes to --out.

Usage: .venv/bin/python scripts/refit_lidar.py data/repeat/run1/laptop_lidar data/real/laptop_lidar \
           --out data/refit/run1
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KEEP = ("tracks.sam.txt", "tracks.sam.npz", "parts.json", "roles.json", "measurement_source.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="track folder with tracks.sam.* and roles.json")
    ap.add_argument("capture", type=Path, help="the Stray Scanner capture folder")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from video2urdf.agent import Agent
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; pick a new folder")
    args.out.mkdir(parents=True)
    for name in KEEP:
        if (args.run / name).exists():
            shutil.copy(args.run / name, args.out / name)
    roles = json.loads((args.run / "roles.json").read_text())
    video = next(args.capture.glob("*.mp4"))
    agent = Agent(video, args.out, capture=args.capture, client=object(), sam=object())
    agent.tracked = True
    print(agent.tool_fit_lidar(**{k: v for k, v in roles.items() if v}))
    meta = json.loads((args.out / "model3d.json").read_text())
    keys = ("reference_frame", "reference_attempts", "reference_choice", "plate_size_mm", "base_thickness_mm",
            "plate_thickness_mm", "opening_range_deg", "mask_coverage")
    print(json.dumps({k: meta.get(k) for k in keys}))


if __name__ == "__main__":
    main()
