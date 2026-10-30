"""Calibrate estimated RGB-video depth using a known distance on a frame."""
import cv2
import numpy as np
from pathlib import Path

from .pipeline import video_info


def reference_scale(capture: Path, frame: int, p1, p2, known_mm: float) -> dict:
    """Return known / estimated length; pixels are in the original RGB frame coordinates.

    Both points must be on visible, rigid, depth-measured surfaces. This corrects one global
    scale only, and cannot fix focal error, local depth error, or motion-estimation drift.
    """
    if not np.isfinite(known_mm) or known_mm <= 0:
        raise ValueError("known length must be positive")
    (w, h), n, _ = video_info(capture / "rgb.mp4")
    if frame < 0 or frame >= n:
        raise ValueError(f"frame must be between 0 and {n - 1}")
    depth = cv2.imread(str(capture / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise ValueError(f"depth is missing for frame {frame}")
    k = np.loadtxt(capture / "camera_matrix.csv", delimiter=",")
    if k.shape != (3, 3) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("invalid capture camera intrinsics")
    dh, dw = depth.shape

    def lift(p):
        x, y = map(float, p)
        if not np.isfinite([x, y]).all() or not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"reference pixel ({x}, {y}) is outside the {w}x{h} frame")
        ix, iy = min(int(x * dw / w), dw - 1), min(int(y * dh / h), dh - 1)
        patch = depth[max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2]
        valid = patch[patch > 0]
        if len(valid) < 3:
            raise ValueError(f"no reliable depth near reference pixel ({x}, {y})")
        z = float(np.median(valid)) / 1000
        if (float(np.percentile(valid, 90)) - float(np.percentile(valid, 10))) / 1000 > max(.03, .08 * z):
            raise ValueError(f"reference pixel ({x}, {y}) is near a depth edge; choose a point inside the surface")
        return np.array([(x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z])

    distance = float(np.linalg.norm(lift(p1) - lift(p2)))
    if not np.isfinite(distance) or distance < .001:
        raise ValueError("estimated reference distance is too small")
    return {"frame": frame, "pixels": [list(map(float, p1)), list(map(float, p2))],
            "known_length_mm": float(known_mm), "estimated_length_before_mm": distance * 1000,
            "scale_factor": float(known_mm / (distance * 1000)),
            "note": "One global scale correction; focal and local depth errors remain."}


def check_references(capture: Path, references, scale: float) -> list[dict]:
    """Evaluate independent known distances without using them to fit global scale."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    checks = []
    for frame, x1, y1, x2, y2, length in references:
        result = reference_scale(capture, int(frame), (x1, y1), (x2, y2), length)
        predicted = result["estimated_length_before_mm"] * scale
        result["predicted_after_calibration_mm"] = predicted
        result["relative_error"] = (predicted - length) / length
        checks.append(result)
    return checks
