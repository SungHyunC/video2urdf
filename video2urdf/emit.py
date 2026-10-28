"""Write the fitted links as MJCF and URDF.

Each body's frame sits at its joint anchor on the reference frame, axis-aligned with the
world, so the zero pose reproduces the reference frame. Geometry is one box per link from
the part's mask; THICKNESS sets its depth along y.
"""

import numpy as np

from .joints import Link

THICKNESS = 0.02  # m


def _v3(xz) -> str:
    return f"{xz[0]:.5f} 0 {xz[1]:.5f}"


def _origin(link: Link) -> np.ndarray:
    return link.joint.anchor if link.joint else np.zeros(2)


def _limits(link: Link) -> tuple[float, float]:
    q = link.joint.q
    return float(np.nanmin(q)), float(np.nanmax(q))


def _axis(link: Link) -> str:
    return "0 -1 0" if link.joint.type == "revolute" else _v3(link.joint.axis)


def mjcf(name: str, links: list[Link], camera: dict | None = None, size: tuple[int, int] | None = None) -> str:
    """MJCF string. ``camera`` (pos, fovy) adds a fixed camera "video" matching the source view."""
    by_name = {l.name: l for l in links}
    children: dict[str | None, list[Link]] = {}
    for l in links:
        children.setdefault(l.parent, []).append(l)

    def body(link: Link, depth: int) -> str:
        pad = "  " * depth
        parent_origin = _origin(by_name[link.parent]) if link.parent else np.zeros(2)
        center, half, angle = link.box
        lines = [f'{pad}<body name="{link.name}" pos="{_v3(_origin(link) - parent_origin)}">']
        if link.joint and link.joint.type != "fixed":
            lo, hi = _limits(link)
            jtype = "hinge" if link.joint.type == "revolute" else "slide"
            lines.append(f'{pad}  <joint name="{link.name}_joint" type="{jtype}" axis="{_axis(link)}" '
                         f'range="{lo:.4f} {hi:.4f}" limited="true"/>')
        lines.append(f'{pad}  <geom type="box" pos="{_v3(center - _origin(link))}" '
                     f'size="{half[0]:.5f} {THICKNESS / 2} {half[1]:.5f}" axisangle="0 -1 0 {angle:.5f}"/>')
        lines += [body(c, depth + 1) for c in children.get(link.name, [])]
        lines.append(f"{pad}</body>")
        return "\n".join(lines)

    visual = ""
    if size:
        visual = f'  <visual><global offwidth="{size[0]}" offheight="{size[1]}"/></visual>\n'
    cam = ""
    if camera:
        cam = (f'    <camera name="video" pos="{camera["pos"][0]:.5f} {camera["pos"][1]:.5f} {camera["pos"][2]:.5f}" '
               f'xyaxes="1 0 0 0 0 1" fovy="{camera["fovy"]:.5f}"/>\n')
    roots = "\n".join(body(l, 2) for l in children[None])
    return (f'<mujoco model="{name}">\n  <compiler angle="radian"/>\n{visual}'
            f"  <worldbody>\n{cam}{roots}\n  </worldbody>\n</mujoco>\n")


def urdf(name: str, links: list[Link]) -> str:
    by_name = {l.name: l for l in links}
    out = [f'<?xml version="1.0"?>\n<robot name="{name}">']
    for l in links:
        center, half, angle = l.box
        out.append(f'  <link name="{l.name}">\n    <visual>\n'
                   f'      <origin xyz="{_v3(center - _origin(l))}" rpy="0 {-angle:.5f} 0"/>\n'
                   f'      <geometry><box size="{2 * half[0]:.5f} {THICKNESS} {2 * half[1]:.5f}"/></geometry>\n'
                   f'    </visual>\n  </link>')
    for l in links:
        if not l.joint:
            continue
        if l.joint.type == "fixed":
            out.append(f'  <joint name="{l.name}_joint" type="fixed">\n'
                       f'    <parent link="{l.parent}"/>\n    <child link="{l.name}"/>\n'
                       f'    <origin xyz="{_v3(_origin(l) - _origin(by_name[l.parent]))}" rpy="0 0 0"/>\n  </joint>')
            continue
        lo, hi = _limits(l)
        out.append(f'  <joint name="{l.name}_joint" type="{l.joint.type}">\n'
                   f'    <parent link="{l.parent}"/>\n    <child link="{l.name}"/>\n'
                   f'    <origin xyz="{_v3(_origin(l) - _origin(by_name[l.parent]))}" rpy="0 0 0"/>\n'
                   f'    <axis xyz="{_axis(l)}"/>\n'
                   f'    <limit lower="{lo:.4f}" upper="{hi:.4f}" effort="10" velocity="2"/>\n  </joint>')
    out.append("</robot>\n")
    return "\n".join(out)
