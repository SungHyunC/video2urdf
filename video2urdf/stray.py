"""Load an iPhone LiDAR capture exported by the Stray Scanner app.

Layout: rgb.mp4, depth/NNNNNN.png (uint16 mm, 256x192), confidence/NNNNNN.png (0 low,
1 medium, 2 high), camera_matrix.csv (3x3 intrinsics at the RGB size), odometry.csv
(per-frame ARKit pose and intrinsics). Depth is metric, so no card is needed.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .lift3d import Intrinsics


@dataclass
class Capture:
    video: Path
    depth: np.ndarray  # (n, h, w) float32 metres, 0 where invalid or low confidence
    intrinsics: Intrinsics  # at the RGB video's resolution


def find_capture(path: Path) -> Path:
    """The folder holding rgb.mp4, given it or any parent of it."""
    hits = sorted(path.rglob("rgb.mp4")) if path.is_dir() else []
    if not hits:
        raise FileNotFoundError(f"no Stray Scanner capture (rgb.mp4) under {path}")
    return hits[0].parent


def load(path: Path, min_confidence: int = 2) -> Capture:
    """Depth is cached as depth.npz next to the PNGs on first load."""
    root = find_capture(path)
    cache = root / "depth.npz"
    if cache.exists():
        depth = np.load(cache)["depth"]
    else:
        frames = sorted((root / "depth").glob("*.png"))
        depth = np.stack([cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in frames]).astype(np.float32) / 1000
        conf = np.stack([cv2.imread(str(root / "confidence" / f.name), cv2.IMREAD_UNCHANGED) for f in frames])
        depth[conf < min_confidence] = 0
        np.savez_compressed(cache, depth=depth)
    k = np.loadtxt(root / "camera_matrix.csv", delimiter=",")
    return Capture(root / "rgb.mp4", depth, Intrinsics(f=float(k[0, 0]), cx=float(k[0, 2]), cy=float(k[1, 2])))
