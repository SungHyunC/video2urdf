"""Model files that load outside our own MuJoCo scripts (checked by scripts/check_exports.py).

What a downloaded URDF needs that the pipeline's own tools don't:
- MuJoCo's URDF compiler strips mesh paths down to the file name (``strippath`` defaults to
  true for URDF up to at least MuJoCo 3.5), so ``meshes/base.stl`` isn't found. Every URDF
  carries ``<mujoco><compiler strippath="false"/></mujoco>``, which other parsers ignore
  (the emitters write it for new runs; ``portable_urdf`` adds it to older ones).
- Drake reads OBJ, VTK and glTF meshes, not STL. An exported URDF names an OBJ twin of each
  STL: the same triangles, written out vertex for vertex (model.xml keeps the STL).
- A header comment says what the file is, where its meshes are, and the loader settings that
  matter: PyBullet's inertia flag and joint friction, MuJoCo's parent-child contacts, ROS paths.

Nothing here changes a measured value: sizes, axes, limits and masses are copied as written.
"""

from __future__ import annotations

import math
import re
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

URDF_MUJOCO = {"strippath": "false"}  # what every URDF we write tells MuJoCo's URDF compiler
MUJOCO_BLOCK = '<mujoco><compiler strippath="false"/></mujoco>'  # the same, for string emitters
OBJ_MARK = "# OBJ twin of {name}, written by Video2URDF's export: the same triangles, metres"
HEADER_MARK = "Video2URDF model"

_COMMENT = re.compile(r"<!--.*?-->", re.S)


def _first_tag(text: str, tag: str, start: int = 0) -> re.Match | None:
    """The first <tag ...> (or <tag .../>) at or after ``start`` that isn't inside a comment."""
    pat = re.compile(r"<!--.*?-->|<" + tag + r"\b([^>]*?)(/?)>", re.S)
    for m in pat.finditer(text, start):
        if not m.group(0).startswith("<!--"):
            return m
    return None


def _attrs(s: str) -> dict[str, str]:
    return {k: v1 if v1 is not None and q == '"' else v2
            for k, q, v1, v2 in ((m.group(1), m.group(2), m.group(3), m.group(4))
                                 for m in re.finditer(r"""([\w:.-]+)\s*=\s*(?:(")([^"]*)"|'([^']*)')""", s))}


def _tag(name: str, attrs: dict[str, str], empty: bool) -> str:
    inner = "".join(f' {k}="{v}"' for k, v in attrs.items())
    return f"<{name}{inner}{'/' if empty else ''}>"


def mujoco_compiler(text: str) -> dict[str, str]:
    """Attributes of a URDF's <mujoco><compiler> (empty when there is none)."""
    m = _first_tag(text, "mujoco")
    if m is None or m.group(2) == "/":
        return {}
    end = text.find("</mujoco>", m.end())
    c = _first_tag(text[:end], "compiler", m.end()) if end >= 0 else None
    return _attrs(c.group(1)) if c is not None else {}


def with_mujoco_compiler(text: str, **attrs: str) -> str:
    """The URDF text with ``attrs`` set on <mujoco><compiler> (other attributes kept), adding the
    block as the first child of <robot> when there is none. Text-level, so comments, namespaces
    and formatting elsewhere survive."""
    m = _first_tag(text, "mujoco")
    if m is None:
        r = _first_tag(text, "robot")
        if r is None:
            raise ValueError("no <robot> element")
        return text[:r.end()] + "\n  <mujoco>" + _tag("compiler", attrs, True) + "</mujoco>" + text[r.end():]
    if m.group(2) == "/":  # <mujoco/>
        return text[:m.start()] + "<mujoco" + m.group(1) + ">" + _tag("compiler", attrs, True) + "</mujoco>" + text[m.end():]
    end = text.find("</mujoco>", m.end())
    if end < 0:
        raise ValueError("<mujoco> is not closed")
    c = _first_tag(text[:end], "compiler", m.end())
    if c is None:
        return text[:m.end()] + _tag("compiler", attrs, True) + text[m.end():]
    merged = {**_attrs(c.group(1)), **attrs}
    return text[:c.start()] + _tag("compiler", merged, c.group(2) == "/") + text[c.end():]


# ------------------------------------------------------------------ rotations

def quat_matrix(q) -> np.ndarray:
    """Rotation matrix of a MuJoCo quaternion (w, x, y, z), normalised first."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_rpy(R) -> tuple[float, float, float]:
    """URDF roll, pitch, yaw with R = Rz(yaw) Ry(pitch) Rx(roll), also at pitch = +-90 deg, where
    only roll - yaw (or roll + yaw) is defined: there yaw = 0 and roll takes all of it. (Taking
    roll and yaw separately from the near-zero entries there gives noise: a principal frame
    turned exactly 90 deg about y came out up to 5 deg wrong.)"""
    R = np.asarray(R, float)
    cp = math.hypot(R[2, 1], R[2, 2])
    if cp > 1e-6:
        return math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], cp), math.atan2(R[1, 0], R[0, 0])
    s = -1.0 if R[2, 0] > 0 else 1.0  # sin(pitch)
    return math.atan2(s * R[0, 1], R[1, 1]), s * math.pi / 2, 0.0


def quat_rpy(q) -> tuple[float, float, float]:
    return matrix_rpy(quat_matrix(q))


def fix_inertial_frames(urdf: ET.Element, mjcf: ET.Element, tol: float = 1e-6) -> list[str]:
    """Where a URDF link's <inertial> has the MJCF body's mass, COM and principal moments but its
    origin rpy turns the principal axes elsewhere (an Euler conversion that failed at gimbal
    lock), rewrite that rpy from the MJCF's quaternion. Returns the links fixed."""
    from .urdf_checks import rpy_matrix
    bodies = {b.get("name"): b.find("inertial") for b in mjcf.iter("body")}
    fixed = []
    for link in urdf.findall("link"):
        mi, ui = bodies.get(link.get("name")), link.find("inertial")
        if mi is None or ui is None or mi.get("quat") is None or mi.get("diaginertia") is None:
            continue
        o, m, i = ui.find("origin"), ui.find("mass"), ui.find("inertia")
        if o is None or m is None or i is None:
            continue
        diag = np.array([float(i.get(k, 0)) for k in ("ixx", "iyy", "izz")])
        off = np.array([float(i.get(k, 0)) for k in ("ixy", "ixz", "iyz")])
        mdiag = np.array([float(v) for v in mi.get("diaginertia").split()])
        mpos = np.array([float(v) for v in mi.get("pos", "0 0 0").split()])
        upos = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()])
        if np.abs(off).max() > 0 or np.abs(diag - mdiag).max() > tol * max(mdiag.max(), 1e-12) \
                or np.abs(upos - mpos).max() > 1e-9 or abs(float(m.get("value")) - float(mi.get("mass", "nan"))) > 1e-9:
            continue  # not the same inertial: leave it
        Rm = quat_matrix([float(v) for v in mi.get("quat").split()])
        Ru = rpy_matrix([float(v) for v in (o.get("rpy") or "0 0 0").split()])
        Im, Iu = Rm @ np.diag(mdiag) @ Rm.T, Ru @ np.diag(diag) @ Ru.T
        if np.abs(Im - Iu).max() <= 1e-4 * mdiag.max():
            continue
        o.set("rpy", " ".join(f"{v:.9g}" for v in quat_rpy([float(v) for v in mi.get("quat").split()])))
        fixed.append(link.get("name"))
    return fixed


# ------------------------------------------------------------------ meshes

def read_stl(path: Path) -> np.ndarray:
    """Triangles (T, 3, 3) of a binary or ASCII STL."""
    data = path.read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data[84:], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
            return rec["v"].reshape(n, 3, 3).astype(np.float64)
    text = data.decode("ascii", errors="replace")
    if not text.lstrip().lower().startswith("solid"):
        raise ValueError(f"{path}: not a binary or ASCII STL")
    v = [[float(x) for x in line.split()[1:4]] for line in text.splitlines() if line.strip().startswith("vertex")]
    if len(v) % 3:
        raise ValueError(f"{path}: ASCII STL with {len(v)} vertices")
    return np.array(v, float).reshape(-1, 3, 3)


def stl_to_obj(stl: Path, obj: Path) -> int:
    """Write the STL's triangles as an OBJ (each triangle its own three vertices, so every loader
    shades it flat, as an STL). Values keep float32 precision. Returns the triangle count."""
    tris = read_stl(stl)
    lines = [OBJ_MARK.format(name=stl.name)]
    lines += [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in tris.reshape(-1, 3)]
    lines += [f"f {3 * i + 1} {3 * i + 2} {3 * i + 3}" for i in range(len(tris))]
    obj.write_text("\n".join(lines) + "\n")
    return len(tris)


def twin_name(stl_rel: str, taken) -> str:
    """Where an STL's OBJ twin goes: next to it, same stem; ``taken(name)`` says a different file
    already holds that name (then _twin is added)."""
    p = PurePosixPath(stl_rel)
    name = str(p.with_suffix(".obj"))
    return str(p.with_name(p.stem + "_twin.obj")) if taken(name) else name


def is_twin(path: Path) -> bool:
    return twin_source(path) is not None


def twin_source(path: Path) -> str | None:
    """The STL file name an OBJ twin was written from (None for any other file)."""
    try:
        with open(path) as f:
            m = re.match(r"# OBJ twin of (.+?), written by", f.readline())
    except (OSError, UnicodeDecodeError):
        return None
    return m.group(1) if m else None


# ------------------------------------------------------------------ the downloadable files

def _fmt(x: float) -> str:
    return f"{x:.4g}"


def header(name: str, robot: ET.Element, root_height: float | None, kind: str) -> str:
    """The comment at the top of an exported model.urdf (kind "urdf") or model.xml ("mjcf")."""
    lines = [f"{HEADER_MARK} \"{name}\", measured from a video (https://github.com/SungHyunC/video2urdf).",
             "Mesh and texture paths are relative to this file (meshes/...): this file alone won't load;",
             "unzip model.zip (the viewer's \"Download model + meshes\") and load it from there."]
    if kind == "urdf":
        children = {j.find("child").get("link") for j in robot.findall("joint") if j.find("child") is not None}
        root = next((l.get("name") for l in robot.findall("link") if l.get("name") not in children), None)
        moving = [j for j in robot.findall("joint") if j.get("type") not in ("fixed", None)]
        where = f", its origin {_fmt(root_height)} m above the floor" if root_height else ""
        lines += [f"Units: metres, kilograms, radians. Fix the root link \"{root}\"{where}.",
                  "Collisions are the measured geometry; visuals may be the design, video texture or SAM 3D mesh.",
                  "PyBullet: loadURDF(path, useFixedBase=True, flags=URDF_USE_INERTIA_FROM_FILE)."]
        if moving:
            lines += ["  PyBullet doesn't apply <dynamics friction>: set VELOCITY_CONTROL, targetVelocity=0,",
                      "  force=friction on each joint. MuJoCo 3.2.5+: simulate model.xml, which excludes contacts",
                      "  between parts joined by a joint (they overlap there); MuJoCo's URDF import can't."]
        lines += ["The <mujoco> element only keeps mesh paths for MuJoCo's URDF import; other parsers ignore it.",
                  "ROS: mesh paths must become package://<your_package>/meshes/... for RViz and Gazebo."]
    else:
        lines += ["Needs MuJoCo 3.2.5 or later (mesh inertia attribute). Measured geoms are group 3 (they",
                  "collide), the visible design group 0, superseded visuals group 4."]
    body = "\n     ".join(lines)
    return f"<!-- {body} -->\n"


def _strip_header(text: str) -> str:
    """The text without a header written by an earlier export (and without the XML declaration)."""
    text = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", text)
    return re.sub(r"^\s*<!--\s*" + re.escape(HEADER_MARK) + r".*?-->\s*", "", text, flags=re.S)


def portable_urdf(text: str, name: str, mesh_map: dict[str, str], mjcf_text: str | None = None) -> tuple[str, list[str]]:
    """The URDF as we hand it out: the header comment, the <mujoco> compiler block, each mesh
    filename in ``mesh_map`` (an STL) replaced by its OBJ twin, and inertial frames that a
    failed Euler conversion turned (fix_inertial_frames, against ``mjcf_text``) put right.
    Masses, principal moments, sizes, axes and limits are kept as written. Returns the text and
    the links whose inertial frame was fixed; applying it again changes nothing."""
    root = ET.fromstring(with_mujoco_compiler(_strip_header(text), **URDF_MUJOCO))
    for el in root.iter("mesh"):
        if el.get("filename") in mesh_map:
            el.set("filename", mesh_map[el.get("filename")])
    fixed = fix_inertial_frames(root, ET.fromstring(_strip_header(mjcf_text))) if mjcf_text else []
    ET.indent(root, "  ")
    body = ET.tostring(root, encoding="unicode")
    height = root_height(mjcf_text) if mjcf_text else None
    return "<?xml version='1.0' encoding='utf-8'?>\n" + header(name, root, height, "urdf") + body + "\n", fixed


def portable_mjcf(text: str, name: str) -> str:
    """model.xml with the header comment (nothing else changes)."""
    body = _strip_header(text)
    return header(name, ET.fromstring(body), None, "mjcf") + body.strip() + "\n"


def root_height(mjcf_text: str) -> float | None:
    """Height of the root body's origin above z = 0 (the floor) in model.xml, when it is raised."""
    world = ET.fromstring(_strip_header(mjcf_text)).find("worldbody")
    body = world.find("body") if world is not None else None
    if body is None or not body.get("pos"):
        return None
    z = float(body.get("pos").split()[2])
    return z if z > 1e-6 else None


def urdf_stl_meshes(text: str) -> list[str]:
    """Relative STL filenames a URDF names, in order, once each."""
    out = []
    for el in ET.fromstring(_strip_header(text)).iter("mesh"):
        f = el.get("filename") or ""
        if f.lower().endswith(".stl") and not f.startswith(("package://", "file://", "model://")) and f not in out:
            out.append(f)
    return out


def make_portable(folder: Path, name: str) -> dict:
    """Rewrite folder/model.urdf and folder/model.xml for download (writing OBJ twins next to the
    STLs the URDF names). Returns {"twins": {stl: obj}, "files": [new files], "inertial_fixed": [links]}."""
    urdf, mjcf = folder / "model.urdf", folder / "model.xml"
    text = urdf.read_text()
    twins = {}
    for stl in urdf_stl_meshes(text):
        rel = twin_name(stl, lambda n: (folder / n).exists() and not is_twin(folder / n))
        stl_to_obj(folder / stl, folder / rel)
        twins[stl] = rel
    rewritten = []
    for el in ET.fromstring(_strip_header(text)).iter("mesh"):  # twins of an earlier pass: from their STL again
        f = el.get("filename") or ""
        src = twin_source(folder / f) if f.lower().endswith(".obj") else None
        if src and (folder / f).parent.joinpath(src).is_file() and f not in rewritten:
            stl_to_obj((folder / f).parent / src, folder / f)
            rewritten.append(f)
    mjcf_text = mjcf.read_text() if mjcf.exists() else None
    new, fixed = portable_urdf(text, name, twins, mjcf_text)
    urdf.write_text(new)
    if mjcf_text is not None:
        mjcf.write_text(portable_mjcf(mjcf_text, name))
    return {"twins": twins, "files": sorted(set(twins.values()) | set(rewritten)), "inertial_fixed": fixed}


def repack(folder: Path, name: str | None = None) -> dict:
    """Make an already exported viewer folder (docs/models/<name>/) portable in place: model.urdf,
    model.xml, the OBJ twins in meshes/, and model.zip rebuilt from its old entries with those
    replaced or added. Nothing else in the folder (model.json, videos, traces) is touched."""
    name = name or folder.name
    done = make_portable(folder, name)
    z = folder / "model.zip"
    if z.exists():
        with zipfile.ZipFile(z) as old:
            entries = [(i, old.read(i.filename)) for i in old.infolist() if not i.is_dir()]
        replace = {"model.urdf", "model.xml", *done["files"]}
        tmp = z.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as new:
            names = [i.filename for i, _ in entries]
            for info, data in entries:
                if info.filename in replace:
                    new.write(folder / info.filename, info.filename)
                else:
                    new.writestr(info, data)
            for f in ["model.urdf", "model.xml", *done["files"]]:
                if f not in names and (folder / f).exists():
                    new.write(folder / f, f)
        tmp.replace(z)
    return done
