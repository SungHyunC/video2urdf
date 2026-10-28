"""URDF/MJCF for a hinged plate measured with iPhone LiDAR, and its replay (video2urdf/lidar.py).

Usage: same arguments as scripts/fit_lidar_hinge.py, run after it. Writes model.urdf,
model.xml, meshes/, model3d.json and replay3d.mp4 into the track folder.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fit_lidar_hinge import load, roles_args  # noqa: E402

from video2urdf import lidar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    roles_args(ap)
    c, roles = load(ap.parse_args())
    lidar.build_model(c, roles)
    print(f"-> {c.track_dir}/model.urdf, model.xml, replay3d.mp4")


if __name__ == "__main__":
    main()
