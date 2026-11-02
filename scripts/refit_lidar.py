"""Refit saved LiDAR tracks with the agent's own fit tool, on a copy, for regression checks.

Reuses a run's tracks and the parts the agent chose: roles.json (fit_lidar, a hinged plate),
drawers.json (fit_drawers) or static.json (fit_static); rod chains have scripts/refit_chain.py.
No new segmentation or model calls. The source folder is left unchanged; the refit goes to --out.
The capture can also be a MapAnything capture of the same video (scripts/video_to_capture.py), to
score the plain-video path against the LiDAR fit (scripts/eval_plain_video.py).

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
    ap.add_argument("run", type=Path, help="track folder with tracks.sam.* and roles.json / drawers.json / static.json")
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
    video = next(args.capture.glob("*.mp4"))
    agent = Agent(video, args.out, capture=args.capture, client=object(), sam=object())
    agent.tracked = True
    if (args.run / "drawers.json").exists():
        spec = json.loads((args.run / "drawers.json").read_text())
        drawers = [{"part": d, "first_frame": w[0], "last_frame": w[1]} for d, w in spec["drawers"]]
        print(agent.tool_fit_drawers(base=spec["base"], drawers=drawers, floor=spec.get("floor")))
        keys = ("reference_frame", "joints", "box_sizes_mm", "mask_coverage")
    elif (args.run / "static.json").exists():
        spec = json.loads((args.run / "static.json").read_text())
        print(agent.tool_fit_static(part=spec["part"], floor=spec["floor"], shape=spec.get("shape", "box")))
        keys = ("reference_frame", "observed_box_size_mm", "mask_coverage")
    else:
        roles = json.loads((args.run / "roles.json").read_text())
        print(agent.tool_fit_lidar(**{k: v for k, v in roles.items() if v}))
        keys = ("reference_frame", "reference_attempts", "reference_choice", "plate_size_mm", "base_thickness_mm",
                "plate_thickness_mm", "opening_range_deg", "mask_coverage")
    meta = json.loads((args.out / "model3d.json").read_text())
    print(json.dumps({k: meta.get(k) for k in keys}))


if __name__ == "__main__":
    main()
