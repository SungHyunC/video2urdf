"""Write an ArticulatedModel plus camera, credit card and backdrop as a MuJoCo MJCF scene."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from .camera import PinholeCamera
from .model import REVOLUTE, ArticulatedModel, matrix_to_quat, rpy_to_matrix

# ISO/IEC 7810 ID-1 card: 85.60 x 53.98 x 0.76 mm.
CARD_SIZE = (0.0856, 0.05398)
CARD_THICKNESS = 0.00076

# Planar free motion of the root link, for objects held in the hand.
BASE_DOFS = (("base_x", "slide", (1, 0, 0)), ("base_z", "slide", (0, 0, 1)), ("base_rot", "hinge", (0, -1, 0)))


@dataclass
class Card:
    """Credit card standing in the motion plane: its front face lies at y = 0."""

    center_xz: tuple[float, float]
    rotation: float = 0.0  # in-plane, CCW

    def corners_world(self) -> np.ndarray:
        """Front-face corners (4, 3): bottom-left, bottom-right, top-right, top-left."""
        w, h = CARD_SIZE
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        xz = local @ np.array([[c, s], [-s, c]]) + np.asarray(self.center_xz)
        return np.stack([xz[:, 0], np.zeros(4), xz[:, 1]], axis=1)


@dataclass
class Scene:
    camera: PinholeCamera
    card: Card | None = None
    floor_z: float | None = 0.0
    wall_y: float = 0.08
    base_dofs: bool = False
    light_pos: tuple[float, float, float] | None = None
    light_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    colors: dict = field(
        default_factory=lambda: {
            "wall": (0.56, 0.58, 0.62, 1),
            "floor": (0.62, 0.47, 0.33, 1),
            "card": (0.95, 0.95, 0.93, 1),
            "chip": (0.85, 0.7, 0.3, 1),
        }
    )


def _fmt(values) -> str:
    return " ".join(f"{float(v):.10g}" for v in values)


def _quat_attr(rot: np.ndarray) -> str:
    return _fmt(matrix_to_quat(rot))


def scene_xml(model: ArticulatedModel, scene: Scene) -> str:
    cam = scene.camera
    root = ET.Element("mujoco", model=model.name)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", gravity="0 0 0")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(cam.width), offheight=str(cam.height))
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="8")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.45 0.45 0.45", specular="0.05 0.05 0.05")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", contype="0", conaffinity="0")

    world = ET.SubElement(root, "worldbody")
    x_right, y_up = cam.mujoco_xyaxes()[:3], cam.mujoco_xyaxes()[3:]
    ET.SubElement(
        world, "camera", name="cam", pos=_fmt(cam.position), xyaxes=_fmt([*x_right, *y_up]), fovy=f"{cam.fovy_deg:.6g}"
    )
    if scene.light_pos is not None:
        direction = np.asarray(scene.light_target) - np.asarray(scene.light_pos)
        # Directional: a spotlight this close to the wall shows shadow-map acne.
        ET.SubElement(
            world,
            "light",
            name="key",
            pos=_fmt(scene.light_pos),
            dir=_fmt(direction / np.linalg.norm(direction)),
            directional="true",
            castshadow="true",
            diffuse="0.55 0.55 0.55",
            specular="0.1 0.1 0.1",
        )

    # Wall behind the motion plane facing the camera (plane normal +z rotated to -y), optional floor.
    ET.SubElement(
        world,
        "geom",
        name="wall",
        type="plane",
        size="5 5 0.1",
        pos=_fmt((0, scene.wall_y, 0)),
        quat=_fmt((np.sqrt(0.5), np.sqrt(0.5), 0, 0)),
        rgba=_fmt(scene.colors["wall"]),
    )
    if scene.floor_z is not None:
        ET.SubElement(
            world,
            "geom",
            name="floor",
            type="plane",
            size="5 5 0.1",
            pos=_fmt((0, 0, scene.floor_z)),
            rgba=_fmt(scene.colors["floor"]),
        )

    if scene.card is not None:
        card = scene.card
        rot = rpy_to_matrix((0.0, -card.rotation, 0.0))
        cx, cz = card.center_xz
        ET.SubElement(
            world,
            "geom",
            name="card",
            type="box",
            pos=_fmt((cx, CARD_THICKNESS / 2, cz)),
            quat=_quat_attr(rot),
            size=_fmt((CARD_SIZE[0] / 2, CARD_THICKNESS / 2, CARD_SIZE[1] / 2)),
            rgba=_fmt(scene.colors["card"]),
        )
        # EMV chip on the left half of the card face.
        chip_local = np.array([-0.025, -0.0002, 0.006])
        ET.SubElement(
            world,
            "geom",
            name="card_chip",
            type="box",
            pos=_fmt(np.array([cx, 0, cz]) + rot @ chip_local),
            quat=_quat_attr(rot),
            size=_fmt((0.0062, 0.0002, 0.0048)),
            rgba=_fmt(scene.colors["chip"]),
        )

    def add_link(parent_el: ET.Element, link_name: str, pos, rot, joint=None) -> None:
        body = ET.SubElement(parent_el, "body", name=link_name, pos=_fmt(pos), quat=_quat_attr(rot))
        if joint is None and scene.base_dofs:
            for name, jtype, axis in BASE_DOFS:
                ET.SubElement(body, "joint", name=name, type=jtype, axis=_fmt(axis))
        if joint is not None:
            jtype = "hinge" if joint.type == REVOLUTE else "slide"
            ET.SubElement(
                body,
                "joint",
                name=joint.name,
                type=jtype,
                axis=_fmt(joint.axis),
                range=_fmt((joint.lower, joint.upper)),
            )
        for i, box in enumerate(model.link(link_name).boxes):
            ET.SubElement(
                body,
                "geom",
                name=f"{link_name}_{i}",
                type="box",
                pos=_fmt(box.center),
                size=_fmt(np.asarray(box.size) / 2),
                rgba=_fmt(box.rgba),
            )
        for child in model.child_joints(link_name):
            add_link(body, child.child, child.origin_xyz, rpy_to_matrix(child.origin_rpy), child)

    add_link(world, model.root, model.root_xyz, rpy_to_matrix(model.root_rpy))
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")
