"""Articulated model shared by the synthetic generator (ground truth) and the estimator (prediction).

Conventions (v1 is planar):
- World frame: x right, z up, y pointing away from the camera. Motion happens in the x-z plane.
- Revolute joints rotate about -y, so a positive angle is counter-clockwise as seen from the camera.
- Joints follow URDF semantics: `origin_xyz`/`origin_rpy` place the joint (= child link) frame in the
  parent link frame, `axis` is expressed in that joint frame, rpy is fixed-axis X-Y-Z (R = Rz @ Ry @ Rx).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

REVOLUTE = "revolute"
PRISMATIC = "prismatic"
JOINT_TYPES = (REVOLUTE, PRISMATIC)

# Revolute axis that makes positive angles counter-clockwise in the camera image.
PLANAR_AXIS = (0.0, -1.0, 0.0)


@dataclass
class Box:
    """Visual box in its link frame. `size` holds full extents (not MuJoCo half-sizes)."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


@dataclass
class Link:
    name: str
    boxes: list[Box] = field(default_factory=list)


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = PLANAR_AXIS
    lower: float = 0.0
    upper: float = 0.0


@dataclass
class ArticulatedModel:
    name: str
    links: list[Link]
    joints: list[Joint]
    root_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    root_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def root(self) -> str:
        children = {j.child for j in self.joints}
        return next(link.name for link in self.links if link.name not in children)

    def link(self, name: str) -> Link:
        return next(link for link in self.links if link.name == name)

    def child_joints(self, link_name: str) -> list[Joint]:
        return [j for j in self.joints if j.parent == link_name]

    def validate(self) -> None:
        names = [link.name for link in self.links]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate link names in {names}")
        joint_names = [j.name for j in self.joints]
        if len(set(joint_names)) != len(joint_names):
            raise ValueError(f"duplicate joint names in {joint_names}")
        children = [j.child for j in self.joints]
        if len(set(children)) != len(children):
            raise ValueError("a link has more than one parent joint")
        for j in self.joints:
            if j.type not in JOINT_TYPES:
                raise ValueError(f"joint {j.name}: unknown type {j.type!r}")
            if j.parent not in names or j.child not in names:
                raise ValueError(f"joint {j.name} references a missing link")
            if not j.lower <= j.upper:
                raise ValueError(f"joint {j.name}: lower > upper")
        roots = [n for n in names if n not in children]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root link, got {roots}")
        # Every link must be reachable from the root, which also rules out cycles.
        seen, stack = set(), [roots[0]]
        while stack:
            name = stack.pop()
            seen.add(name)
            stack.extend(j.child for j in self.joints if j.parent == name)
        if seen != set(names):
            raise ValueError(f"links not connected to the root: {sorted(set(names) - seen)}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ArticulatedModel:
        links = [
            Link(link["name"], [Box(tuple(b["center"]), tuple(b["size"]), tuple(b["rgba"])) for b in link["boxes"]])
            for link in d["links"]
        ]
        joints = [
            Joint(
                name=j["name"],
                type=j["type"],
                parent=j["parent"],
                child=j["child"],
                origin_xyz=tuple(j["origin_xyz"]),
                origin_rpy=tuple(j["origin_rpy"]),
                axis=tuple(j["axis"]),
                lower=j["lower"],
                upper=j["upper"],
            )
            for j in d["joints"]
        ]
        return cls(d["name"], links, joints, tuple(d["root_xyz"]), tuple(d["root_rpy"]))


def planar_rpy(theta: float) -> tuple[float, float, float]:
    """rpy for a counter-clockwise (as seen from the camera) in-plane rotation by `theta`."""
    return (0.0, -theta, 0.0)


def rpy_to_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix to a unit quaternion (w, x, y, z), the MuJoCo order (Shepperd's method)."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2 * np.sqrt(1 + tr)
        q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2 * np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4]
    q = np.array(q)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def axis_angle_matrix(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def transform(xyz, rot: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = xyz
    return t


def forward_kinematics(model: ArticulatedModel, q: dict[str, float], root: np.ndarray | None = None) -> dict:
    """World 4x4 pose of every link for joint values `q` (missing joints default to 0)."""
    poses = {model.root: transform(model.root_xyz, rpy_to_matrix(model.root_rpy)) if root is None else root}
    stack = [model.root]
    while stack:
        parent = stack.pop()
        for j in model.child_joints(parent):
            value = q.get(j.name, 0.0)
            if j.type == REVOLUTE:
                motion = transform((0.0, 0.0, 0.0), axis_angle_matrix(j.axis, value))
            else:
                axis = np.asarray(j.axis, dtype=float)
                motion = transform(value * axis / np.linalg.norm(axis), np.eye(3))
            poses[j.child] = poses[parent] @ transform(j.origin_xyz, rpy_to_matrix(j.origin_rpy)) @ motion
            stack.append(j.child)
    return poses


def planar_pose(pose: np.ndarray) -> tuple[float, float, float]:
    """(x, z, theta) of a 4x4 pose whose rotation is about the y axis; theta is CCW from the camera."""
    x_axis = pose[:3, 0]
    return float(pose[0, 3]), float(pose[2, 3]), float(np.arctan2(x_axis[2], x_axis[0]))
