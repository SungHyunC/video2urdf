"""Checks on an exported model bundle (model.urdf + model.xml + meshes/) that need no
simulator: what scripts/check_exports.py runs before any loader. Standard library + numpy.

  parse_urdf        links (inertial, visuals, collisions), joints, the root, a <mujoco> block
  link_frames       each link's pose in the root frame at q = 0 (axes, anchors)
  tree_problems     names, parents, one root, joint limits, finite numbers
  inertia_problems  mass > 0, symmetric, positive definite, A + B >= C (MuJoCo's test)
  asset_problems    every file the URDF / MJCF names: relative, inside the bundle, present, in a
                    format each loader reads; OBJ material libraries and their images
  measured_joints   the joints the fit itself recorded (model3d.json, joints.json, ...)
  compare_joints    URDF joints against expected ones (names, types, axes, limits)
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np

# Mesh formats each loader reads (by extension). PyBullet reads OBJ, STL and COLLADA, not PLY;
# Drake reads OBJ, VTK and glTF (its URDF parser refuses STL collision meshes); yourdfpy reads
# what trimesh does; MuJoCo reads STL, OBJ and its own MSH.
MESH_FORMATS = {
    "mujoco": {".stl", ".obj", ".msh"},
    "pybullet": {".stl", ".obj", ".dae"},
    "yourdfpy": {".stl", ".obj", ".ply", ".dae", ".glb", ".gltf", ".off"},
    "drake": {".obj", ".vtk", ".gltf", ".glb"},
}
IMAGE_FORMATS = {".png", ".jpg", ".jpeg"}
URI_SCHEMES = ("package://", "file://", "model://", "http://", "https://")
LIMIT_TOL = 6e-5  # limits are written with 4 decimals


def rpy_matrix(rpy) -> np.ndarray:
    """URDF's fixed-axis roll, pitch, yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _floats(text: str | None, n: int, default) -> np.ndarray:
    if text is None or not text.strip():
        return np.array(default, float)
    v = np.array([float(x) for x in text.split()], float)
    if len(v) != n:
        raise ValueError(f"expected {n} numbers, got {text!r}")
    return v


def _origin(el: ET.Element | None) -> np.ndarray:
    """4x4 pose of an element's <origin> (identity when absent)."""
    o = None if el is None else el.find("origin")
    T = np.eye(4)
    if o is not None:
        T[:3, :3] = rpy_matrix(_floats(o.get("rpy"), 3, [0, 0, 0]))
        T[:3, 3] = _floats(o.get("xyz"), 3, [0, 0, 0])
    return T


@dataclass
class Geometry:
    kind: str  # mesh, box, cylinder, sphere (or "missing")
    origin: np.ndarray  # 4x4 in the link frame
    name: str | None = None
    filename: str | None = None
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    size: tuple = ()  # box: x y z; cylinder: radius, length; sphere: radius
    texture: str | None = None  # <material><texture filename> of a visual


@dataclass
class Link:
    name: str
    mass: float | None = None
    com: np.ndarray | None = None  # in the link frame
    inertia: np.ndarray | None = None  # 3x3 about the COM, link axes
    visuals: list[Geometry] = field(default_factory=list)
    collisions: list[Geometry] = field(default_factory=list)


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None
    damping: float | None = None
    friction: float | None = None

    @property
    def movable(self) -> bool:
        return self.type in ("revolute", "prismatic", "continuous")


@dataclass
class Robot:
    name: str
    links: dict[str, Link]
    joints: list[Joint]
    root: str | None
    problems: list[str]  # what the parser itself found wrong
    mujoco_compiler: dict[str, str]  # attributes of <mujoco><compiler>, if any
    textures: list[str] = field(default_factory=list)  # every <texture filename>

    def joint(self, name: str) -> Joint | None:
        return next((j for j in self.joints if j.name == name), None)


def _geometry(el: ET.Element) -> Geometry:
    T, g = _origin(el), el.find("geometry")
    shape = None if g is None else next(iter(g), None)
    name = el.get("name")
    tex = el.find("material/texture")
    texture = tex.get("filename") if tex is not None else None
    if shape is None:
        return Geometry("missing", T, name)
    if shape.tag == "mesh":
        return Geometry("mesh", T, name, shape.get("filename"), _floats(shape.get("scale"), 3, [1, 1, 1]),
                        texture=texture)
    if shape.tag == "box":
        return Geometry("box", T, name, size=tuple(_floats(shape.get("size"), 3, [0, 0, 0])), texture=texture)
    if shape.tag == "cylinder":
        return Geometry("cylinder", T, name, size=(float(shape.get("radius", 0)), float(shape.get("length", 0))),
                        texture=texture)
    if shape.tag == "sphere":
        return Geometry("sphere", T, name, size=(float(shape.get("radius", 0)),), texture=texture)
    return Geometry(shape.tag, T, name, texture=texture)


def parse_urdf(source: Path | str) -> Robot:
    """A URDF file (or its text). Raises on malformed XML; collects other problems."""
    text = Path(source).read_text() if isinstance(source, Path) else source
    root = ET.fromstring(text)
    problems: list[str] = []
    if root.tag != "robot":
        raise ValueError(f"root element is <{root.tag}>, not <robot>")
    links: dict[str, Link] = {}
    for el in root.findall("link"):
        name = el.get("name")
        if not name:
            problems.append("a <link> has no name")
            continue
        if name in links:
            problems.append(f"link {name!r} defined twice")
        link = Link(name)
        inertial = el.find("inertial")
        if inertial is not None:
            m = inertial.find("mass")
            link.mass = float(m.get("value")) if m is not None and m.get("value") is not None else None
            T = _origin(inertial)
            link.com = T[:3, 3]
            i = inertial.find("inertia")
            if i is not None:
                v = {k: float(i.get(k, 0)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
                local = np.array([[v["ixx"], v["ixy"], v["ixz"]], [v["ixy"], v["iyy"], v["iyz"]],
                                  [v["ixz"], v["iyz"], v["izz"]]])
                link.inertia = T[:3, :3] @ local @ T[:3, :3].T
        link.visuals = [_geometry(v) for v in el.findall("visual")]
        link.collisions = [_geometry(c) for c in el.findall("collision")]
        links[name] = link
    joints = []
    for el in root.findall("joint"):
        name, jtype = el.get("name"), el.get("type")
        parent, child = el.find("parent"), el.find("child")
        if parent is None or child is None or not name or not jtype:
            problems.append(f"joint {name!r} lacks a name, type, parent or child")
            continue
        lim, dyn = el.find("limit"), el.find("dynamics")
        axis_el = el.find("axis")
        j = Joint(name, jtype, parent.get("link"), child.get("link"), _origin(el),
                  _floats(axis_el.get("xyz") if axis_el is not None else None, 3, [1, 0, 0]))

        def num(e, k):
            return float(e.get(k)) if e is not None and e.get(k) is not None else None
        j.lower, j.upper, j.effort, j.velocity = (num(lim, k) for k in ("lower", "upper", "effort", "velocity"))
        j.damping, j.friction = num(dyn, "damping"), num(dyn, "friction")
        joints.append(j)
    children = {j.child for j in joints}
    roots = [n for n in links if n not in children]
    if len(roots) != 1:
        problems.append(f"expected one root link, found {roots}")
    compiler = root.find("mujoco/compiler")
    textures = [t.get("filename") for t in root.iter("texture") if t.get("filename")]
    return Robot(root.get("name", ""), links, joints, roots[0] if len(roots) == 1 else None, problems,
                 dict(compiler.attrib) if compiler is not None else {}, textures)


def link_frames(robot: Robot) -> dict[str, np.ndarray]:
    """Each link's pose (4x4) in the root link's frame at q = 0."""
    frames = {robot.root: np.eye(4)} if robot.root else {}
    pending = list(robot.joints)
    while pending:
        progressed = False
        for j in list(pending):
            if j.parent in frames:
                frames[j.child] = frames[j.parent] @ j.origin
                pending.remove(j)
                progressed = True
        if not progressed:
            break
    return frames


def joint_frames(robot: Robot) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Joint name -> (anchor, unit axis) in the root frame at q = 0."""
    frames = link_frames(robot)
    out = {}
    for j in robot.joints:
        if j.child in frames:
            T = frames[j.child]
            a = T[:3, :3] @ j.axis
            n = np.linalg.norm(a)
            out[j.name] = (T[:3, 3].copy(), a / n if n > 0 else a)
    return out


def tree_problems(robot: Robot) -> list[str]:
    out = list(robot.problems)
    seen = set()
    for j in robot.joints:
        if j.name in seen:
            out.append(f"joint {j.name!r} defined twice")
        seen.add(j.name)
        for end in (j.parent, j.child):
            if end not in robot.links:
                out.append(f"joint {j.name!r} names a missing link {end!r}")
        if j.type not in ("revolute", "continuous", "prismatic", "fixed", "floating", "planar"):
            out.append(f"joint {j.name!r} has unknown type {j.type!r}")
        if not np.all(np.isfinite(j.origin)):
            out.append(f"joint {j.name!r}: origin is not finite")
        if j.type in ("revolute", "prismatic"):
            if j.lower is None or j.upper is None or j.effort is None or j.velocity is None:
                out.append(f"joint {j.name!r}: <limit> needs lower, upper, effort and velocity")
            elif not (math.isfinite(j.lower) and math.isfinite(j.upper)) or j.lower > j.upper:
                out.append(f"joint {j.name!r}: limits {j.lower}..{j.upper}")
        if j.type in ("revolute", "prismatic", "continuous"):
            n = float(np.linalg.norm(j.axis))
            if not math.isfinite(n) or n < 1e-9:
                out.append(f"joint {j.name!r}: axis {j.axis.tolist()} has no direction")
            elif abs(n - 1) > 1e-3:
                out.append(f"joint {j.name!r}: axis {j.axis.tolist()} is not a unit vector")
    children = [j.child for j in robot.joints]
    for c in {c for c in children if children.count(c) > 1}:
        out.append(f"link {c!r} has more than one parent joint")
    if len(link_frames(robot)) != len(robot.links):
        out.append("not every link is reachable from the root (a cycle or a detached link)")
    return out


def inertia_problems(mass: float | None, inertia: np.ndarray | None, rtol: float = 1e-6) -> list[str]:
    """What a simulator would refuse: no or non-positive mass; an inertia that is not symmetric,
    not positive definite, or whose principal moments break A + B >= C."""
    if mass is None:
        return ["no <inertial> mass"]
    if not math.isfinite(mass) or mass <= 0:
        return [f"mass {mass} is not positive"]
    if inertia is None:
        return ["no <inertia>"]
    I = np.asarray(inertia, float)
    if not np.all(np.isfinite(I)):
        return ["inertia is not finite"]
    if np.abs(I - I.T).max() > rtol * max(np.abs(I).max(), 1e-12):
        return ["inertia is not symmetric"]
    ev = np.sort(np.linalg.eigvalsh(I))
    if ev[0] <= 0:
        return [f"inertia is not positive definite (principal moments {ev.tolist()})"]
    if ev[0] + ev[1] < ev[2] * (1 - rtol):
        return [f"principal moments {ev.tolist()} break A + B >= C"]
    return []


def _inside(bundle: Path, rel: str) -> Path | None:
    """The file a relative path names inside the bundle, or None when it escapes it."""
    p = (bundle / rel).resolve()
    try:
        p.relative_to(bundle.resolve())
    except ValueError:
        return None
    return p


def obj_dependencies(obj: Path) -> list[tuple[str, Path]]:
    """(name as written, path) of each material library an OBJ uses and each image those name."""
    out = []
    try:
        lines = obj.read_text(errors="replace").splitlines()
    except OSError:
        return out
    for line in lines:
        if line.startswith("mtllib "):
            name = line.split(maxsplit=1)[1].strip()
            mtl = obj.parent / name
            out.append((name, mtl))
            if mtl.is_file():
                for m in mtl.read_text(errors="replace").splitlines():
                    parts = m.split()
                    if parts and parts[0] in ("map_Kd", "map_Ka", "map_Ks", "map_d", "map_Bump", "bump", "norm"):
                        out.append((parts[-1], mtl.parent / parts[-1]))
    return out


def mjcf_assets(mjcf: Path) -> list[tuple[str, str, str]]:
    """(kind, file as written, path relative to the MJCF's folder) of every file an MJCF names."""
    root = ET.parse(mjcf).getroot()
    comp = root.find("compiler")
    attr = comp.attrib if comp is not None else {}
    meshdir = attr.get("meshdir", attr.get("assetdir", ""))
    texdir = attr.get("texturedir", attr.get("assetdir", ""))
    out = []
    for kind, tag, base in (("mesh", "mesh", meshdir), ("texture", "texture", texdir), ("hfield", "hfield", meshdir),
                            ("skin", "skin", meshdir), ("include", "include", "")):
        for el in root.iter(tag):
            f = el.get("file")
            if f:
                out.append((kind, f, str(PurePosixPath(base) / f) if base else f))
    return out


@dataclass
class AssetReport:
    problems: list[str]  # missing files, absolute paths, files outside the bundle
    unsupported: dict[str, list[str]]  # loader -> URDF meshes it can't read
    files: list[str]  # every file the URDF needs (relative to the bundle)
    uris: list[str]  # package:// and other URIs, which only some tools resolve


def asset_problems(bundle: Path, urdf: Path, mjcf: Path | None = None) -> AssetReport:
    """Every file model.urdf (and model.xml) names, checked against the bundle folder."""
    robot = parse_urdf(urdf)
    problems, files, uris = [], [], []
    unsupported = {k: [] for k in MESH_FORMATS}
    urdf_meshes = [g.filename for l in robot.links.values() for g in l.visuals + l.collisions if g.kind == "mesh"]
    for f in [m for m in urdf_meshes if m] + robot.textures:
        if f.startswith(URI_SCHEMES):
            uris.append(f)
            continue
        if Path(f).is_absolute():
            problems.append(f"{f}: absolute path (breaks on any other machine)")
            continue
        p = _inside(urdf.parent, f)
        if p is None:
            problems.append(f"{f}: outside the bundle")
            continue
        files.append(f)
        if not p.is_file():
            problems.append(f"{f}: missing")
            continue
        if f in urdf_meshes:
            ext = p.suffix.lower()
            for loader, ok in MESH_FORMATS.items():
                if ext not in ok:
                    unsupported[loader].append(f)
            if ext == ".obj":
                for name, dep in obj_dependencies(p):
                    rel = str(PurePosixPath(f).parent / name)
                    files.append(rel)
                    if not dep.is_file():
                        problems.append(f"{rel} (named by {f}): missing")
        elif p.suffix.lower() not in IMAGE_FORMATS:
            problems.append(f"{f}: texture is not PNG/JPEG")
    if mjcf is not None and mjcf.exists():
        for kind, written, rel in mjcf_assets(mjcf):
            if Path(rel).is_absolute():
                problems.append(f"model.xml {kind} {written}: absolute path")
                continue
            p = _inside(mjcf.parent, rel)
            if p is None:
                problems.append(f"model.xml {kind} {written}: outside the bundle")
            elif not p.is_file():
                problems.append(f"model.xml {kind} {written}: missing")
            elif kind == "mesh" and p.suffix.lower() == ".obj":
                for name, dep in obj_dependencies(p):
                    if not dep.is_file():
                        problems.append(f"model.xml {written} names {name}: missing")
    return AssetReport(problems, {k: v for k, v in unsupported.items() if v}, sorted(set(files)), uris)


# ------------------------------------------------------------------ what the fit recorded

def _nanrange(q) -> tuple[float, float] | None:
    v = np.array([np.nan if x is None else x for x in q], float)
    v = v[np.isfinite(v)]
    return (float(v.min()), float(v.max())) if len(v) else None


def measured_joints(read) -> dict[str, dict] | None:
    """Joint name -> {"type", "lower", "upper", "axis" (unit, or None when not recorded)} as the
    fit recorded them. ``read(name)`` returns a JSON file's content or None. None when the
    records don't say (no fit files)."""
    if read("static.json") is not None:
        return {}
    m3 = read("model3d.json") or {}
    joints3 = m3.get("joints") or []
    if joints3 and all("range_deg" in j and "child" in j for j in joints3):  # a rod chain (chain3d)
        out = {}
        for j in joints3:
            name = f"{j['child']}_joint"
            if j["type"] == "fixed":
                out[name] = {"type": "fixed", "lower": None, "upper": None, "axis": None}
            else:
                lo, hi = (math.radians(v) for v in j["range_deg"])
                out[name] = {"type": "revolute", "lower": lo, "upper": hi, "axis": _unit(j.get("axis_model"))}
        return out
    slides = read("slides.json") or ({"joints": joints3} if joints3 and all("travel_mm" in j for j in joints3) else None)
    if slides is not None:  # drawers: shut at 0, out by the travel
        return {j["name"]: {"type": "prismatic", "lower": 0.0, "upper": j["travel_mm"] / 1000, "axis": None}
                for j in slides["joints"]}
    if "hinge_limits_rad" in m3:
        lo, hi = m3["hinge_limits_rad"]
        axis = m3.get("joint_axis_model")
        hinge = read("hinge.json")
        if axis is None and hinge is not None and "camera_to_model" in m3:  # the fitted axis, model frame
            axis = (np.array(m3["camera_to_model"]["rotation"]) @ np.array(hinge["axis"])).tolist()
            return {"lid_joint": {"type": "revolute", "lower": lo, "upper": hi, "axis": _unit(axis), "axis_sign_free": True}}
        return {"lid_joint": {"type": "revolute", "lower": lo, "upper": hi, "axis": _unit(axis) if axis else None}}
    joints = read("joints.json")
    if joints is not None and "joints" in joints:
        out = {}
        for j in joints["joints"]:
            name = f"{j['child']}_joint"
            if j["type"] == "fixed":
                out[name] = {"type": "fixed", "lower": None, "upper": None, "axis": None}
                continue
            r = _nanrange(j["q"])
            axis = [0.0, -1.0, 0.0] if j["type"] == "revolute" else [j["axis"][0], 0.0, j["axis"][1]]
            out[name] = {"type": j["type"], "lower": r[0] if r else None, "upper": r[1] if r else None,
                         "axis": _unit(axis)}
        return out
    return None


def _unit(v):
    if v is None:
        return None
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 0 else None


def compare_joints(robot: Robot, expected: dict[str, dict], axis_tol_deg: float = 0.05,
                   limit_tol: float = LIMIT_TOL) -> list[str]:
    """Differences between the URDF's joints and ``expected`` (as measured_joints gives)."""
    out = []
    movable = {j.name for j in robot.joints if j.type != "fixed"}
    want_movable = {k for k, v in expected.items() if v["type"] != "fixed"}
    for name in sorted(want_movable - movable):
        out.append(f"{name}: measured, not a moving joint in the URDF")
    for name in sorted(movable - want_movable):
        out.append(f"{name}: in the URDF, not measured")
    for name, e in expected.items():
        j = robot.joint(name)
        if j is None:
            continue
        if e["type"] is not None and j.type != e["type"]:
            out.append(f"{name}: type {j.type}, measured {e['type']}")
            continue
        if e["type"] == "fixed":
            continue
        if e.get("lower") is not None and (j.lower is None or abs(j.lower - e["lower"]) > limit_tol):
            out.append(f"{name}: lower limit {j.lower}, measured {e['lower']:.6g}")
        if e.get("upper") is not None and (j.upper is None or abs(j.upper - e["upper"]) > limit_tol):
            out.append(f"{name}: upper limit {j.upper}, measured {e['upper']:.6g}")
        if e.get("axis") is not None:
            a = j.axis / np.linalg.norm(j.axis)
            c = float(np.dot(a, e["axis"]))
            if e.get("axis_sign_free"):
                c = abs(c)
            angle = math.degrees(math.acos(max(-1.0, min(1.0, c))))
            if angle > axis_tol_deg:
                out.append(f"{name}: axis {np.round(a, 5).tolist()} is {angle:.2f} deg from the measured "
                           f"{np.round(e['axis'], 5).tolist()}")
    return out


def read_json_from(*folders: Path):
    """A ``read(name)`` for measured_joints: the first folder holding that JSON file wins."""
    def read(name):
        for f in folders:
            p = f / name
            if p.is_file():
                return json.loads(p.read_text())
        return None
    return read
