"""Paint a fitted 3D model with its own video (see video2urdf/texture.py).

Usage: .venv/bin/python scripts/texture_model.py data/tracks_agent/laptop_lidar data/real/laptop_lidar/laptop_lidar.mp4
Needs replay3d.npz, which fit_lidar / fit_drawers write with the replay.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf.texture import texture_model  # noqa: E402

if __name__ == "__main__":
    texture_model(Path(sys.argv[1]), Path(sys.argv[2]))
