"""URDF and MJCF for a 3D model: links with meshes, joints with arbitrary axes.

The model frame is z-up. Each link's frame sits at its joint (the root's at the model
origin), axis-aligned with the model, so the zero pose is the reference frame's pose.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Visual:
    mesh: str  # path relative to the model files
    rgb: tuple[float, float, float]
    collide: bool = True


@dataclass
class Link3D:
    name: str
    visuals: list[Visual]
    parent: str | None = None
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))  # joint position in the parent's frame
    joint_type: str = "fixed"  # "revolute", "prismatic" or "fixed"
    axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0]))
    limits: tuple[float, float] = (0.0, 0.0)


def _v(x) -> str:
    return " ".join(f"{float(v):.5f}" for v in x)


def urdf(name: str, links: list[Link3D]) -> str:
    out = [f'<?xml version="1.0"?>\n<robot name="{name}">']
    for l in links:
        out.append(f'  <link name="{l.name}">')
        for i, v in enumerate(l.visuals):
            rgba = _v(v.rgb) + " 1"
            out.append(f'    <visual name="{l.name}_{i}"><geometry><mesh filename="{v.mesh}"/></geometry>'
                       f'<material name="{l.name}_{i}_color"><color rgba="{rgba}"/></material></visual>')
            if v.collide:
                out.append(f'    <collision><geometry><mesh filename="{v.mesh}"/></geometry></collision>')
        out.append("  </link>")
    for l in links:
        if l.parent is None:
            continue
        out.append(f'  <joint name="{l.name}_joint" type="{l.joint_type}">\n'
                   f'    <parent link="{l.parent}"/>\n    <child link="{l.name}"/>\n'
                   f'    <origin xyz="{_v(l.origin)}" rpy="0 0 0"/>')
        if l.joint_type != "fixed":
            out.append(f'    <axis xyz="{_v(l.axis)}"/>\n'
                       f'    <limit lower="{l.limits[0]:.4f}" upper="{l.limits[1]:.4f}" effort="10" velocity="2"/>')
        out.append("  </joint>")
    out.append("</robot>\n")
    return "\n".join(out)


def mjcf(name: str, links: list[Link3D], root_pos=(0, 0, 0), size: tuple[int, int] | None = None) -> str:
    """``root_pos`` places the root link in the world (e.g. so the floor is z = 0). A camera
    named "video" is included for replays; its pose is set per frame."""
    children: dict[str | None, list[Link3D]] = {}
    for l in links:
        children.setdefault(l.parent, []).append(l)

    def body(l: Link3D, depth: int, pos) -> str:
        pad = "  " * depth
        lines = [f'{pad}<body name="{l.name}" pos="{_v(pos)}">']
        if l.parent is not None and l.joint_type != "fixed":
            jt = "hinge" if l.joint_type == "revolute" else "slide"
            lines.append(f'{pad}  <joint name="{l.name}_joint" type="{jt}" axis="{_v(l.axis)}" '
                         f'range="{l.limits[0]:.4f} {l.limits[1]:.4f}" limited="true"/>')
        for i, v in enumerate(l.visuals):
            extra = "" if v.collide else ' contype="0" conaffinity="0" mass="0.001"'
            lines.append(f'{pad}  <geom type="mesh" mesh="{l.name}_{i}" rgba="{_v(v.rgb)} 1"{extra}/>')
        lines += [body(c, depth + 1, c.origin) for c in children.get(l.name, [])]
        lines.append(f"{pad}</body>")
        return "\n".join(lines)

    meshes = "".join(f'<mesh name="{l.name}_{i}" file="{v.mesh}" inertia="exact"/>'
                     for l in links for i, v in enumerate(l.visuals))
    visual = f'  <visual><global offwidth="{size[0]}" offheight="{size[1]}"/></visual>\n' if size else ""
    roots = "\n".join(body(l, 2, root_pos) for l in children[None])
    return (f'<mujoco model="{name}">\n  <compiler angle="radian"/>\n{visual}  <asset>{meshes}</asset>\n'
            f'  <worldbody>\n    <camera name="video" pos="0 0 1"/>\n{roots}\n  </worldbody>\n'
            f'{contact_excludes(links)}</mujoco>\n')


def contact_excludes(links) -> str:
    """Links overlap at their joints. MuJoCo skips parent-child contacts, except when the parent
    is welded to the world, as a fixed root is: without these the overlap throws the child off."""
    pairs = "".join(f'<exclude body1="{l.parent}" body2="{l.name}"/>' for l in links if l.parent)
    return f"  <contact>{pairs}</contact>\n" if pairs else ""
