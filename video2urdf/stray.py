"""Load an iPhone LiDAR capture exported by the Stray Scanner app.

Layout: rgb.mp4, depth/NNNNNN.png (uint16 mm, 256x192), confidence/NNNNNN.png (0 low,
1 medium, 2 high), camera_matrix.csv (3x3 intrinsics at the RGB size), odometry.csv
(per-frame ARKit pose and intrinsics). Depth is metric, so no card is needed.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

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


def ref_to_frame_poses(path: Path, n: int, ref: int) -> np.ndarray:
    """(n, 4, 4) mapping reference-camera coordinates to frame-t camera coordinates for the
    static scene, from ARKit odometry. Stray Scanner's poses are camera-to-world in a frame
    matching the depth's camera axes (checked by aligning depth clouds across frames)."""
    od = np.genfromtxt(find_capture(path) / "odometry.csv", delimiter=",", skip_header=1)[:n]
    c = np.tile(np.eye(4), (len(od), 1, 1))
    c[:, :3, :3] = Rotation.from_quat(od[:, 5:9]).as_matrix()
    c[:, :3, 3] = od[:, 2:5]
    return np.linalg.inv(c) @ c[ref]


def gravity_up(path: Path, ref: int) -> np.ndarray:
    """Up (against gravity) in the reference camera's coordinates: ARKit's world frame has +y
    up, so this needs no floor in view."""
    od = np.genfromtxt(find_capture(path) / "odometry.csv", delimiter=",", skip_header=1)
    r = Rotation.from_quat(od[ref, 5:9]).as_matrix()  # camera-to-world
    return r.T @ np.array([0.0, 1.0, 0.0])
