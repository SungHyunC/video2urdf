"""Planar mechanisms with known joints, used as ground truth for synthetic videos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model import PLANAR_AXIS, PRISMATIC, REVOLUTE, ArticulatedModel, Box, Joint, Link, planar_rpy

PLATE = 0.004  # part thickness along the viewing axis
LAYER = 0.005  # each child sits this much closer to the camera than its parent

GRAY = (0.32, 0.33, 0.36, 1)
PALETTE = [
    (0.86, 0.18, 0.18, 1),  # red
    (0.18, 0.4, 0.86, 1),  # blue
    (0.16, 0.68, 0.3, 1),  # green
    (0.95, 0.58, 0.08, 1),  # orange
    (0.58, 0.28, 0.74, 1),  # purple
]


@dataclass
class Preset:
    model: ArticulatedModel
    floor_z: float | None  # None for objects hanging or held in the air
    base_motion: bool  # the root drifts like a hand-held object
    motion: str  # "sequential" (one joint at a time) or "simultaneous"


def bar(x0: float, x1: float, width: float, rgba, z: float = 0.0) -> Box:
    """Bar along the link's x axis from x0 to x1, `width` tall in the motion plane."""
    return Box(((x0 + x1) / 2, 0.0, z), (x1 - x0, PLATE, width), rgba)


def block(cx: float, cz: float, sx: float, sz: float, rgba) -> Box:
    return Box((cx, 0.0, cz), (sx, PLATE, sz), rgba)


def _deg(d: float) -> float:
    return float(np.deg2rad(d))


def pendulum(rng: np.random.Generator) -> Preset:
    links = [
        Link("mount", [block(0, 0, 0.08, 0.05, GRAY)]),
        Link("arm", [bar(-0.015, 0.28, 0.024, PALETTE[0]), block(0.29, 0, 0.06, 0.06, PALETTE[0])]),
    ]
    joints = [
        Joint("hinge", REVOLUTE, "mount", "arm", (0, -LAYER, 0), planar_rpy(_deg(-90)), lower=_deg(-50), upper=_deg(50))
    ]
    return Preset(ArticulatedModel("pendulum", links, joints, root_xyz=(0, 0, 0.42)), 0.0, False, "simultaneous")


def drawer(rng: np.random.Generator) -> Preset:
    links = [
        Link("cabinet", [block(0.2, 0.08, 0.4, 0.16, GRAY)]),
        Link("drawer", [block(0.18, 0, 0.36, 0.1, PALETTE[1]), block(-0.012, 0, 0.024, 0.03, PALETTE[1])]),
    ]
    joints = [
        Joint("slide", PRISMATIC, "cabinet", "drawer", (0.02, -LAYER, 0.08), axis=(-1, 0, 0), lower=0.0, upper=0.22)
    ]
    return Preset(ArticulatedModel("drawer", links, joints), 0.0, False, "simultaneous")


def laptop(rng: np.random.Generator) -> Preset:
    links = [Link("base", [bar(0, 0.32, 0.024, GRAY)]), Link("lid", [bar(0, 0.32, 0.01, PALETTE[1], z=0.005)])]
    joints = [Joint("hinge", REVOLUTE, "base", "lid", (0, -LAYER, 0.012), lower=_deg(5), upper=_deg(125))]
    return Preset(ArticulatedModel("laptop", links, joints, root_xyz=(0, 0, 0.012)), 0.0, False, "simultaneous")


def scissors(rng: np.random.Generator) -> Preset:
    links = [
        Link("blade_a", [bar(-0.09, 0.11, 0.016, PALETTE[0])]),
        Link("blade_b", [bar(-0.09, 0.11, 0.016, PALETTE[1])]),
    ]
    joints = [Joint("pivot", REVOLUTE, "blade_a", "blade_b", (0, -LAYER, 0), lower=_deg(-38), upper=_deg(-3))]
    model = ArticulatedModel("scissors", links, joints, root_xyz=(0, 0, 0.3), root_rpy=planar_rpy(_deg(15)))
    return Preset(model, None, True, "simultaneous")


def lamp(rng: np.random.Generator) -> Preset:
    links = [
        Link("base", [block(0, 0, 0.16, 0.03, GRAY)]),
        Link("lower_arm", [bar(-0.01, 0.24, 0.018, PALETTE[0])]),
        Link("upper_arm", [bar(-0.01, 0.22, 0.016, PALETTE[1])]),
        Link("head", [bar(-0.01, 0.07, 0.04, PALETTE[2]), block(0.09, 0, 0.04, 0.08, PALETTE[2])]),
    ]
    joints = [
        Joint(
            "shoulder",
            REVOLUTE,
            "base",
            "lower_arm",
            (0, -LAYER, 0.015),
            planar_rpy(_deg(70)),
            lower=_deg(-20),
            upper=_deg(20),
        ),
        Joint(
            "elbow",
            REVOLUTE,
            "lower_arm",
            "upper_arm",
            (0.24, -LAYER, 0),
            planar_rpy(_deg(-60)),
            lower=_deg(-25),
            upper=_deg(30),
        ),
        Joint(
            "wrist",
            REVOLUTE,
            "upper_arm",
            "head",
            (0.22, -LAYER, 0),
            planar_rpy(_deg(-80)),
            lower=_deg(-35),
            upper=_deg(35),
        ),
    ]
    return Preset(ArticulatedModel("lamp", links, joints, root_xyz=(0, 0, 0.015)), 0.0, False, "sequential")


def cart_pendulum(rng: np.random.Generator) -> Preset:
    links = [
        Link("rail", [bar(-0.26, 0.26, 0.02, GRAY)]),
        Link("cart", [block(0, 0, 0.08, 0.05, PALETTE[3])]),
        Link("pole", [bar(-0.01, 0.22, 0.015, PALETTE[4])]),
    ]
    joints = [
        Joint("slide", PRISMATIC, "rail", "cart", (0, -LAYER, 0), axis=(1, 0, 0), lower=-0.15, upper=0.15),
        Joint(
            "swing", REVOLUTE, "cart", "pole", (0, -LAYER, 0), planar_rpy(_deg(-90)), lower=_deg(-40), upper=_deg(40)
        ),
    ]
    return Preset(ArticulatedModel("cart_pendulum", links, joints, root_xyz=(0, 0, 0.35)), None, False, "simultaneous")


def random_chain(rng: np.random.Generator) -> Preset:
    """Serial chain of 1-3 joints with random types, lengths, attachment points and limits."""
    n_joints = int(rng.integers(1, 4))
    root_len = float(rng.uniform(0.12, 0.25))
    links = [Link("link0", [bar(0, root_len, float(rng.uniform(0.025, 0.05)), GRAY)])]
    joints = []
    # Bars span x in [0, length] and accept revolute or prismatic children; slider blocks are centered on
    # their joint and only carry revolute children near their middle.
    parent, parent_len, parent_is_bar = "link0", root_len, True
    for i in range(1, n_joints + 1):
        rgba = PALETTE[(i - 1) % len(PALETTE)]
        name = f"link{i}"
        if not parent_is_bar or rng.random() < 0.7:
            length = float(rng.uniform(0.1, 0.25))
            links.append(Link(name, [bar(-0.01, length, float(rng.uniform(0.014, 0.03)), rgba)]))
            span = float(rng.uniform(_deg(40), _deg(130)))
            center = float(rng.uniform(-0.3, 0.3))
            attach = float(rng.uniform(0.6, 0.95) if parent_is_bar else rng.uniform(-0.2, 0.2)) * parent_len
            joints.append(
                Joint(
                    f"joint{i}",
                    REVOLUTE,
                    parent,
                    name,
                    (attach, -LAYER, 0),
                    planar_rpy(float(rng.uniform(-np.pi * 0.7, np.pi * 0.7))),
                    PLANAR_AXIS,
                    center - span / 2,
                    center + span / 2,
                )
            )
            is_bar = True
        else:
            length = float(rng.uniform(0.05, 0.08))
            links.append(Link(name, [block(0, 0, length, float(rng.uniform(0.03, 0.05)), rgba)]))
            start = float(rng.uniform(0.1, 0.3)) * parent_len
            span = float(rng.uniform(0.35, 0.6)) * parent_len
            joints.append(
                Joint(f"joint{i}", PRISMATIC, parent, name, (start, -LAYER, 0), axis=(1, 0, 0), lower=0.0, upper=span)
            )
            is_bar = False
        parent, parent_len, parent_is_bar = name, length, is_bar
    root_rpy = planar_rpy(float(rng.uniform(-np.pi, np.pi)))
    model = ArticulatedModel(f"random{n_joints}", links, joints, root_xyz=(0, 0, 0.4), root_rpy=root_rpy)
    return Preset(model, None, bool(rng.random() < 0.3), str(rng.choice(["sequential", "simultaneous"])))


PRESETS = {
    "pendulum": pendulum,
    "drawer": drawer,
    "laptop": laptop,
    "scissors": scissors,
    "lamp": lamp,
    "cart_pendulum": cart_pendulum,
    "random": random_chain,
}


def build(name: str, rng: np.random.Generator) -> Preset:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name](rng)
