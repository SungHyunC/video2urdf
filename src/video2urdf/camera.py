"""Pinhole camera in OpenCV convention (x right, y down, z forward; pixel centers at integer coordinates).

MuJoCo renders with OpenGL pixel centers at i + 0.5, so a centered principal point is (W-1)/2, (H-1)/2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])


@dataclass
class PinholeCamera:
    width: int
    height: int
    fovy_deg: float
    R: np.ndarray  # world -> camera rotation (3x3)
    t: np.ndarray  # world -> camera translation (3,)

    @property
    def K(self) -> np.ndarray:
        f = (self.height / 2) / np.tan(np.deg2rad(self.fovy_deg) / 2)
        return np.array([[f, 0, (self.width - 1) / 2], [0, f, (self.height - 1) / 2], [0, 0, 1]])

    @property
    def position(self) -> np.ndarray:
        return -self.R.T @ self.t

    @classmethod
    def look_at(cls, position, target, width: int, height: int, fovy_deg: float, up=WORLD_UP) -> PinholeCamera:
        position = np.asarray(position, dtype=float)
        forward = np.asarray(target, dtype=float) - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.stack([right, down, forward])
        return cls(width, height, fovy_deg, R, -R @ position)

    def mujoco_xyaxes(self) -> np.ndarray:
        """MuJoCo camera `xyaxes`: image right and image up in world coordinates."""
        return np.concatenate([self.R[0], -self.R[1]])

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float) @ self.R.T + self.t

    def project(self, points: np.ndarray) -> np.ndarray:
        """World points (..., 3) -> pixel coordinates (..., 2)."""
        pc = self.to_camera(points)
        uv = pc @ self.K.T
        return uv[..., :2] / uv[..., 2:3]

    def plane_homography(self, origin, u_axis, v_axis) -> np.ndarray:
        """3x3 H mapping plane coordinates (s, t) in meters, point = origin + s*u + t*v, to pixels."""
        cols = np.stack(
            [self.R @ np.asarray(u_axis), self.R @ np.asarray(v_axis), self.R @ np.asarray(origin) + self.t]
        )
        return self.K @ cols.T

    def to_dict(self) -> dict:
        return {
            "convention": "opencv",
            "width": self.width,
            "height": self.height,
            "fovy_deg": self.fovy_deg,
            "K": self.K.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "position": self.position.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PinholeCamera:
        return cls(d["width"], d["height"], d["fovy_deg"], np.array(d["R"]), np.array(d["t"]))
