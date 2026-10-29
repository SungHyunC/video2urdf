"""URDF Doctor: check an existing URDF's revolute joint against the motion measured from video.

Usage:
  .venv/bin/python scripts/urdf_doctor.py examples/doctor/laptop_buggy.urdf data/tracks_agent/laptop_lidar
  .venv/bin/python scripts/urdf_doctor.py my.urdf RUN --joint-log log.csv   (frame,q rows recorded for the video)

RUN is a Video2URDF run folder (model.xml, hinge.json; from run_agent.py or fit_lidar). Writes
<RUN>/doctor_<urdf name>.md and .json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf import doctor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf", type=Path)
    ap.add_argument("run", type=Path)
    ap.add_argument("--joint-log", type=Path, help="CSV of frame,q (radians) for the video's frames")
    args = ap.parse_args()
    log = np.loadtxt(args.joint_log, delimiter=",", ndmin=2) if args.joint_log else None
    d = doctor.diagnose(args.urdf, args.run, joint_log=log)
    text = doctor.report(d, args.urdf)
    stem = f"doctor_{args.urdf.stem}"
    (args.run / f"{stem}.md").write_text(text)
    (args.run / f"{stem}.json").write_text(json.dumps({"findings": d.findings, **d.numbers}, indent=2))
    print(text)


if __name__ == "__main__":
    main()
