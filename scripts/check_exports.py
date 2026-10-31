#!/usr/bin/env python3
"""Load exported models the way a user would, and report what works: one row per model per
check (pass / warn / fail / skip, and why).

    uv run --no-project --python 3.12 --with pybullet --with yourdfpy --with mujoco \\
        python scripts/check_exports.py docs/models

Add ``--with drake`` for the Drake row (a large download), ``--with mujoco==3.3.7`` to try an
older MuJoCo. On Apple Silicon pybullet has no wheel: uv builds it from source the first time
(a few minutes, needs the Xcode command line tools) and reuses that build afterwards. A loader
that isn't installed is reported as "skip". No model calls, nothing written outside a temp dir.

PATH is a model.zip (unpacked into a temporary folder), a viewer folder docs/models/<name>/
(its model.zip is checked as a user downloads it, plus the single-file downloads next to it),
a run folder holding model.urdf and model.xml, or a folder of any of these (docs/models).
``--run NAME=RUN_DIR`` also compares model NAME with the fit's own records in RUN_DIR (e.g.
--run laptop_lidar=data/tracks_design/laptop_lidar); without it the records in the bundle
(model3d.json, physics.json) and the viewer's joint tracks (model.json) are used.

Checks:
  files          every mesh and texture the URDF and MJCF name: relative, inside the bundle,
                 present, in a format each loader reads; OBJ material libraries and images
  inertia        each link: mass > 0, inertia positive definite with A + B >= C; URDF == MJCF;
                 masses == physics.json; inertia near that of its collision geometry
  joints         names, types, axes, anchors and limits: URDF == MJCF == the fit's records
                 (model3d.json, chain.json, slides.json, joints.json, the viewer's joint tracks)
  collision      each link's URDF <collision> is its measured geometry (the MJCF's colliding
                 geoms, bounds within 1 mm); no design primitive collides; visuals where
                 model.xml's visible geoms are
  mjcf:mujoco    model.xml compiles; 2 s under gravity: finite, inside the joint ranges, nothing
                 flies off; how far each joint drifts (measured friction should hold it); each
                 joint swept through its range: where the model's own parts interpenetrate
  urdf:mujoco    MuJoCo's URDF compiler on model.urdf; the same 2 s; parts overlapping their
                 parent at the start (MuJoCo's URDF import can't exclude those contacts)
  urdf:pybullet  loadURDF(useFixedBase=True, URDF_USE_INERTIA_FROM_FILE): links, shapes, masses;
                 2 s under gravity with the default motors off, with the URDF's joint friction
                 as velocity motors, and with self-collision on
  urdf:yourdfpy  URDF.load with visual and collision meshes, validate(), forward kinematics,
                 the loaded geometry where the URDF puts it
  urdf:drake     Parser + Finalize with the root welded, 2 s (only if pydrake is installed)
  single-file    model.urdf / model.xml downloaded alone: the files they still need

Exit status 1 when any row fails.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf import urdf_checks as uc  # noqa: E402
from video2urdf.portable import with_mujoco_compiler  # noqa: E402

SECONDS = 2.0
FLY_M = 0.5  # a part this far from where it started has been thrown
FAST_M_S = 1.0  # still this fast after 2 s at rest under gravity: it blew up
LIMIT_SLACK = (math.radians(2.0), 0.002)  # how far past a (soft) limit a joint may sit: rad, m
OVERLAP_M = 0.001  # parent-child penetration that counts as an overlap
THROWN = math.radians(10)  # a joint an overlap moves this far (rad, or m for a slide) was thrown
BOUNDS_M = 0.001  # URDF collision bounds vs the MJCF's colliding geoms
INERTIA_RATIO = 3.0  # stated principal moments vs the collision geometry's at that mass


@dataclass
class Row:
    model: str
    check: str
    status: str  # pass, warn, fail, skip
    detail: str = ""


@dataclass
class Target:
    name: str
    bundle: Path  # folder with model.urdf, model.xml, meshes/
    source: str  # what was given
    viewer: Path | None = None  # docs/models/<name>/: single-file downloads, model.json
    run: Path | None = None  # the fit's records


# ------------------------------------------------------------------ inputs

def unpack(zip_path: Path, dest: Path) -> Path:
    """Unzip into dest, refusing entries that would land outside it."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            n = PurePosixPath(info.filename)
            if n.is_absolute() or ".." in n.parts or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"{zip_path}: unsafe entry {info.filename!r}")
        z.extractall(dest)
    return dest


def discover(paths: list[Path], work: Path, runs: dict[str, Path]) -> list[Target]:
    out: list[Target] = []

    def add(p: Path):
        if p.is_file() and p.suffix.lower() == ".zip":
            name = p.parent.name if p.name == "model.zip" else p.stem
            viewer = p.parent if p.name == "model.zip" and (p.parent / "model.urdf").exists() else None
            out.append(Target(name, unpack(p, work / f"{len(out):02d}_{name}"), str(p), viewer))
        elif p.is_dir() and (p / "model.zip").exists():
            out.append(Target(p.name, unpack(p / "model.zip", work / f"{len(out):02d}_{p.name}"), str(p / "model.zip"), p))
        elif p.is_dir() and (p / "model.urdf").exists():
            out.append(Target(p.name, p, str(p), None, p))
        elif p.is_dir():
            kids = sorted(c for c in p.iterdir() if (c / "model.urdf").exists() or (c / "model.zip").exists()
                          or (c.is_file() and c.suffix.lower() == ".zip"))
            if not kids:
                raise SystemExit(f"{p}: no model.urdf / model.zip in it or its subfolders")
            for c in kids:
                add(c)
        else:
            raise SystemExit(f"{p}: not a model.zip or a folder")

    for p in paths:
        add(p)
    for t in out:
        if t.name in runs:
            t.run = runs[t.name]
    return out


@contextlib.contextmanager
def captured():
    """Collect what C libraries print to stdout/stderr (PyBullet, MuJoCo, Drake) and Python
    warnings logged meanwhile; yields a list that holds the lines afterwards."""
    lines: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda r: lines.append(f"{r.levelname.lower()}: {r.getMessage()}")
    logging.getLogger().addHandler(handler)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        try:
            yield lines
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
            logging.getLogger().removeHandler(handler)
            tmp.seek(0)
            lines[:0] = [l for l in tmp.read().decode(errors="replace").splitlines() if l.strip()]


def _short(lines: list[str], n: int = 3) -> str:
    keep = [l.strip() for l in lines if not l.startswith("pybullet build time")]
    return " | ".join(keep[:n]) + (f" | (+{len(keep) - n} more)" if len(keep) > n else "")


def _try_import(name: str):
    try:
        with captured():
            return __import__(name)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ static checks

def check_files(t: Target, robot: uc.Robot) -> Row:
    urdf, mjcf = t.bundle / "model.urdf", t.bundle / "model.xml"
    problems = uc.tree_problems(robot)
    rep = uc.asset_problems(t.bundle, urdf, mjcf if mjcf.exists() else None)
    problems += rep.problems
    if not mjcf.exists():
        problems.append("no model.xml")
    if t.viewer is not None:
        for f in ("model.urdf", "model.xml"):
            if (t.viewer / f).exists() and (t.viewer / f).read_bytes() != (t.bundle / f).read_bytes():
                problems.append(f"the single-file {f} differs from the one in model.zip")
    if problems:
        return Row(t.name, "files", "fail", "; ".join(problems[:6]) + (f" (+{len(problems) - 6})" if len(problems) > 6 else ""))
    meshes = sum(1 for l in robot.links.values() for g in l.visuals + l.collisions if g.kind == "mesh")
    notes = [f"{len(rep.files)} files resolve ({meshes} mesh references)"]
    status = "pass"
    if rep.unsupported:
        status = "warn"
        notes += [f"{loader} can't read {', '.join(sorted({Path(f).suffix for f in fs}))} "
                  f"({len(fs)} meshes)" for loader, fs in sorted(rep.unsupported.items())]
    if rep.uris:
        notes.append(f"{len(rep.uris)} URIs ({rep.uris[0]}...) need a ROS package")
    return Row(t.name, "files", status, "; ".join(notes))


def mjcf_model(t: Target):
    """model.xml compiled by MuJoCo, or None."""
    mujoco = _try_import("mujoco")
    if mujoco is None or not (t.bundle / "model.xml").exists():
        return None
    try:
        with captured():
            return mujoco.MjModel.from_xml_path(str(t.bundle / "model.xml"))
    except Exception:  # noqa: BLE001
        return None


def _body_frames(model, data, root: str):
    """Rotation and position of the body named ``root`` (world when it was fused away)."""
    import mujoco
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
    if b < 0:
        return np.eye(3), np.zeros(3)
    return data.xmat[b].reshape(3, 3).copy(), data.xpos[b].copy()


def mj_joints(model, root: str) -> dict[str, dict]:
    """Joint name -> type, range, anchor and axis in the root body's frame, at qpos0."""
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    R, p = _body_frames(model, data, root)
    names = {int(mujoco.mjtJoint.mjJNT_HINGE): "revolute", int(mujoco.mjtJoint.mjJNT_SLIDE): "prismatic",
             int(mujoco.mjtJoint.mjJNT_FREE): "free", int(mujoco.mjtJoint.mjJNT_BALL): "ball"}
    out = {}
    for j in range(model.njnt):
        out[model.joint(j).name] = {
            "type": names[int(model.jnt_type[j])], "limited": bool(model.jnt_limited[j]),
            "range": model.jnt_range[j].tolist(), "anchor": R.T @ (data.xanchor[j] - p),
            "axis": R.T @ data.xaxis[j], "body": model.body(model.jnt_bodyid[j]).name,
            "frictionloss": float(model.dof_frictionloss[model.jnt_dofadr[j]]),
            "damping": float(model.dof_damping[model.jnt_dofadr[j]])}
    return out


def mj_inertials(model) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """Body name -> (mass, COM, 3x3 inertia about the COM) in the body's frame."""
    import mujoco
    out = {}
    for b in range(1, model.nbody):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.body_iquat[b])
        R = R.reshape(3, 3)
        out[model.body(b).name] = (float(model.body_mass[b]), model.body_ipos[b].copy(),
                                   R @ np.diag(model.body_inertia[b]) @ R.T)
    return out


def geometry_inertia(t: Target, robot: uc.Robot) -> dict[str, np.ndarray]:
    """Principal moments of each link's collision geometry at the URDF's mass (uniform density),
    from MuJoCo's URDF compiler with inertia taken from the geoms."""
    import mujoco
    text = with_mujoco_compiler((t.bundle / "model.urdf").read_text(), meshdir=str(t.bundle) + "/", strippath="false",
                                fusestatic="false", discardvisual="true", inertiafromgeom="true")
    with captured():
        model = mujoco.MjModel.from_xml_string(text)
    out = {}
    for name, link in robot.links.items():
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0 or not link.mass or model.body_mass[b] <= 0:
            continue
        out[name] = np.sort(model.body_inertia[b] * link.mass / model.body_mass[b])
    return out


def check_inertia(t: Target, robot: uc.Robot, mj) -> Row:
    problems, warns = [], []
    for name, link in robot.links.items():
        for p in uc.inertia_problems(link.mass, link.inertia):
            problems.append(f"{name}: {p}")
    if problems:
        return Row(t.name, "inertia", "fail", "; ".join(problems))
    if mj is not None:
        mji = mj_inertials(mj)
        for name, link in robot.links.items():
            if name not in mji:
                problems.append(f"{name}: no body in model.xml")
                continue
            m, com, I = mji[name]
            if abs(m - link.mass) > 1e-6 * max(1.0, m):
                problems.append(f"{name}: mass {link.mass} in the URDF, {m:.6g} in model.xml")
            if np.abs(com - link.com).max() > 1e-5:
                problems.append(f"{name}: COM differs from model.xml by {1000 * np.abs(com - link.com).max():.2f} mm")
            if np.abs(I - link.inertia).max() > 1e-4 * max(np.abs(I).max(), 1e-9):
                problems.append(f"{name}: inertia tensor differs from model.xml's")
    phys = _read(t, "physics.json")
    if phys:
        for name, link in robot.links.items():
            want = phys.get("links", {}).get(name, {}).get("mass_kg")
            if want is not None and abs(want - link.mass) > 1e-6 * max(1.0, want):
                warns.append(f"{name}: mass {link.mass} kg, physics.json says {want}")
    try:
        geo = geometry_inertia(t, robot)
    except Exception as e:  # noqa: BLE001
        geo = {}
        warns.append(f"collision-geometry inertia not computed ({str(e).splitlines()[0][:80]})")
    for name, g in geo.items():
        stated = np.sort(np.linalg.eigvalsh(robot.links[name].inertia))
        ratio = stated / np.maximum(g, 1e-12)
        if ratio.min() < 1 / INERTIA_RATIO or ratio.max() > INERTIA_RATIO:
            warns.append(f"{name}: principal moments {ratio.min():.2g}-{ratio.max():.2g}x those of its collision "
                         f"geometry at {robot.links[name].mass:g} kg")
    if problems:
        return Row(t.name, "inertia", "fail", "; ".join(problems[:5]))
    masses = ", ".join(f"{n} {l.mass:g} kg" for n, l in robot.links.items())
    return Row(t.name, "inertia", "warn" if warns else "pass",
               "; ".join(warns) if warns else f"valid; == model.xml{' and physics.json' if phys else ''} ({masses})")


def _read(t: Target, name: str):
    """A JSON record: from the run folder, else the bundle, else the viewer folder."""
    return uc.read_json_from(*[f for f in (t.run, t.bundle, t.viewer) if f is not None])(name)


def viewer_tracks(t: Target) -> dict[str, dict] | None:
    """The viewer's joint tracks (model.json) as measured joints: a 2D fit's limits are their range."""
    spec = _read(t, "model.json")
    if not spec or spec.get("three_d") or not spec.get("joints"):
        return None
    out = {}
    for name, q in spec["joints"].items():
        v = np.array([np.nan if x is None else x for x in q], float)
        v = v[np.isfinite(v)]
        if len(v):
            out[name] = {"type": None, "lower": float(v.min()), "upper": float(v.max()), "axis": None}
    return out


def check_joints(t: Target, robot: uc.Robot, mj) -> Row:
    problems, notes = [], []
    frames = uc.joint_frames(robot)
    if mj is not None:
        mjj = mj_joints(mj, robot.root)
        for j in robot.joints:
            if j.type == "fixed":
                continue
            m = mjj.get(j.name)
            if m is None:
                problems.append(f"{j.name}: not in model.xml")
                continue
            if m["type"] != ("revolute" if j.type == "continuous" else j.type):
                problems.append(f"{j.name}: {j.type} in the URDF, {m['type']} in model.xml")
            anchor, axis = frames[j.name]
            if np.linalg.norm(anchor - m["anchor"]) > 1e-4:
                problems.append(f"{j.name}: anchor {1000 * np.linalg.norm(anchor - m['anchor']):.2f} mm from model.xml's")
            if math.degrees(math.acos(min(1.0, abs(float(axis @ m["axis"]))))) > 0.05 or float(axis @ m["axis"]) < 0:
                problems.append(f"{j.name}: axis differs from model.xml's")
            if j.type != "continuous" and (abs(m["range"][0] - j.lower) > uc.LIMIT_TOL or abs(m["range"][1] - j.upper) > uc.LIMIT_TOL):
                problems.append(f"{j.name}: limits {j.lower}..{j.upper}, model.xml {m['range']}")
            if j.friction is not None and abs(m["frictionloss"] - j.friction) > 1e-6 * max(1, j.friction):
                problems.append(f"{j.name}: friction {j.friction} in the URDF, {m['frictionloss']:.6g} in model.xml")
        extra = set(k for k, v in mjj.items() if v["type"] in ("revolute", "prismatic")) - {j.name for j in robot.joints}
        if extra:
            problems.append(f"model.xml has joints the URDF lacks: {sorted(extra)}")
        notes.append("== model.xml")
    read = uc.read_json_from(*[f for f in (t.run, t.bundle) if f is not None])
    measured = uc.measured_joints(read)
    source = None
    if measured is not None:
        source = "the fit's records" + (f" in {t.run}" if t.run else " (bundle)")
    else:
        measured = viewer_tracks(t)
        source = "the viewer's joint tracks" if measured is not None else None
    if measured is not None:
        for p in uc.compare_joints(robot, measured):
            problems.append(p + f" ({source})")
        notes.append(f"== {source}")
    phys = _read(t, "physics.json")
    if phys:
        for name, info in phys.get("joints", {}).items():
            j = robot.joint(name)
            if j is not None and info.get("friction") is not None and j.friction is not None \
                    and abs(info["friction"] - j.friction) > 1e-5 * max(1, info["friction"]):
                problems.append(f"{name}: friction {j.friction}, physics.json {info['friction']:.6g}")
    moving = [j for j in robot.joints if j.type != "fixed"]
    desc = ", ".join(f"{j.name} {j.type} {_lim(j)}" for j in moving) or "no moving joints"
    if problems:
        return Row(t.name, "joints", "fail", "; ".join(problems[:5]))
    if measured is None:
        notes.append("no fit records to compare (pass --run NAME=RUN_DIR)")
    return Row(t.name, "joints", "pass" if measured is not None else "warn", f"{desc}; " + "; ".join(notes))


def _lim(j: uc.Joint) -> str:
    if j.lower is None:
        return ""
    if j.type == "revolute":
        return f"{math.degrees(j.lower):.1f}..{math.degrees(j.upper):.1f} deg"
    return f"{1000 * j.lower:.0f}..{1000 * j.upper:.0f} mm"


def _geom_points(t: Target, g: uc.Geometry) -> np.ndarray | None:
    """Points spanning a URDF geometry, in its link's frame."""
    if g.kind == "mesh":
        p = t.bundle / g.filename
        if p.suffix.lower() == ".stl":
            from video2urdf.portable import read_stl
            v = read_stl(p).reshape(-1, 3)
        elif p.suffix.lower() == ".obj":
            v = np.array([[float(x) for x in line.split()[1:4]] for line in p.read_text().splitlines()
                          if line.startswith("v ")])
        else:
            return None
        v = v * g.scale
    elif g.kind == "box":
        v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.array(g.size) / 2
    elif g.kind == "cylinder":
        r, h = g.size
        a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ring = np.column_stack([r * np.cos(a), r * np.sin(a)])
        v = np.vstack([np.column_stack([ring, np.full(64, z)]) for z in (-h / 2, h / 2)])
    elif g.kind == "sphere":
        r = g.size[0]
        v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * r
    else:
        return None
    return v @ g.origin[:3, :3].T + g.origin[:3, 3]


def mj_geom_points(model, select) -> dict[str, np.ndarray]:
    """Body name -> points spanning its geoms for which ``select(geom id)`` holds, body frame."""
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out: dict[str, list] = {}
    for g in range(model.ngeom):
        if not select(g):
            continue
        b = model.geom_bodyid[g]
        Rb, pb = data.xmat[b].reshape(3, 3), data.xpos[b]
        Rg, pg = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            m = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[m]: model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        else:
            s = model.geom_size[g]
            gt = int(model.geom_type[g])
            if gt == int(mujoco.mjtGeom.mjGEOM_BOX):
                v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s
            elif gt == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
                ring = np.column_stack([s[0] * np.cos(a), s[0] * np.sin(a)])
                v = np.vstack([np.column_stack([ring, np.full(64, z)]) for z in (-s[1], s[1])])
            else:
                v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * model.geom_rbound[g]
        world = v @ Rg.T + pg
        out.setdefault(model.body(b).name, []).append((world - pb) @ Rb)
    return {k: np.vstack(v) for k, v in out.items()}


DESIGN_PREFIXES = ("design_", "inferred_")
GENERATED = ("_sam3d", "_infer")


def _bounds_gap(u: np.ndarray, m: np.ndarray) -> float:
    return float(max(np.abs(u.min(0) - m.min(0)).max(), np.abs(u.max(0) - m.max(0)).max()))


def declared_points(t: Target, robot: uc.Robot, which: str, frame: str = "link") -> dict[str, np.ndarray]:
    """Link name -> points spanning its URDF visuals or collisions, in the link frame (or the root
    frame at q = 0 with frame="root")."""
    frames = uc.link_frames(robot)
    out = {}
    for name, link in robot.links.items():
        pts = [p for p in (_geom_points(t, g) for g in getattr(link, which)) if p is not None]
        if pts:
            p = np.vstack(pts)
            if frame == "root":
                T = frames[name]
                p = p @ T[:3, :3].T + T[:3, 3]
            out[name] = p
    return out


def check_collision(t: Target, robot: uc.Robot, mj) -> Row:
    problems, warns, notes = [], [], []
    for name, link in robot.links.items():
        for g in link.collisions:
            stem = Path(g.filename).stem if g.filename else ""
            if (g.name or "").startswith(DESIGN_PREFIXES) or stem.startswith(DESIGN_PREFIXES) or stem.endswith(GENERATED):
                problems.append(f"{name}: a design/generated shape collides ({g.name or g.filename})")
    if mj is not None:
        import mujoco
        col = declared_points(t, robot, "collisions")
        mjc = mj_geom_points(mj, lambda g: mj.geom_contype[g] != 0 or mj.geom_conaffinity[g] != 0)
        for name in robot.links:
            if name not in col and name not in mjc:
                continue
            if name not in col:
                problems.append(f"{name}: collides in model.xml but has no URDF <collision>")
            elif name not in mjc:
                problems.append(f"{name}: URDF <collision> but nothing collides in model.xml")
            elif _bounds_gap(col[name], mjc[name]) > BOUNDS_M:
                problems.append(f"{name}: collision bounds {1000 * _bounds_gap(col[name], mjc[name]):.1f} mm "
                                "from model.xml's colliding geoms")
        notes.append("URDF <collision> == model.xml's colliding geoms (bounds within 1 mm)")
        # The visuals a URDF user sees vs what MuJoCo's viewer shows by default (groups 0-2).
        vis = declared_points(t, robot, "visuals")
        mjv = mj_geom_points(mj, lambda g: mj.geom_group[g] <= 2 and int(mj.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE))
        off = [f"{n} {1000 * _bounds_gap(vis[n], mjv[n]):.0f} mm" for n in vis if n in mjv and _bounds_gap(vis[n], mjv[n]) > 0.002]
        if off:
            warns.append("URDF visuals sit off model.xml's visible geoms: " + ", ".join(off))
        else:
            notes.append("visuals == model.xml's visible geoms")
    n_vis = sum(len(l.visuals) for l in robot.links.values())
    n_col = sum(len(l.collisions) for l in robot.links.values())
    if problems:
        return Row(t.name, "collision", "fail", "; ".join(problems[:5]))
    return Row(t.name, "collision", "warn" if warns else "pass",
               f"{n_col} collision / {n_vis} visual elements; " + "; ".join(warns + notes))


# ------------------------------------------------------------------ MuJoCo

def mj_settle(model, seconds: float = SECONDS) -> dict:
    """Start every limited joint inside its range (q = 0 clipped), let go under gravity."""
    import mujoco
    data = mujoco.MjData(model)
    for j in range(model.njnt):
        if int(model.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)) \
                and model.jnt_limited[j]:
            a = model.jnt_qposadr[j]
            data.qpos[a] = np.clip(data.qpos[a], *model.jnt_range[j])
    mujoco.mj_forward(model, data)
    q0, x0 = data.qpos.copy(), data.xpos.copy()
    out = {"q0": q0}
    for i in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            out["nan_at"] = i * model.opt.timestep
            return out
    prev = data.xpos.copy()
    mujoco.mj_step(model, data)
    out.update(q=data.qpos.copy(), moved=float(np.linalg.norm(data.xpos - x0, axis=1).max()),
               speed=float(np.linalg.norm(data.xpos - prev, axis=1).max() / model.opt.timestep))
    return out


def _judge(model, res: dict) -> tuple[list[str], list[str]]:
    """Failures and notes from a settle run."""
    import mujoco
    if "nan_at" in res:
        return [f"NaN after {res['nan_at']:.3f} s"], []
    fails, notes = [], []
    if res["moved"] > FLY_M:
        fails.append(f"a part flew {res['moved']:.2f} m")
    if res["speed"] > FAST_M_S:
        fails.append(f"still moving {res['speed']:.1f} m/s after {SECONDS:g} s")
    for j in range(model.njnt):
        jt = int(model.jnt_type[j])
        if jt not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        a = model.jnt_qposadr[j]
        q, q0 = res["q"][a], res["q0"][a]
        hinge = jt == int(mujoco.mjtJoint.mjJNT_HINGE)
        slack = LIMIT_SLACK[0] if hinge else LIMIT_SLACK[1]
        lo, hi = model.jnt_range[j]
        if model.jnt_limited[j] and not (lo - slack <= q <= hi + slack):
            fails.append(f"{model.joint(j).name} left its range: {_val(q, hinge)} outside {_val(lo, hinge)}..{_val(hi, hinge)}")
        notes.append(f"{model.joint(j).name} {'+' if q >= q0 else ''}{_val(q - q0, hinge)}")
    return fails, notes


def _val(v: float, hinge: bool) -> str:
    return f"{math.degrees(v):.1f} deg" if hinge else f"{1000 * v:.1f} mm"


def blocked_ranges(model, depth: float = 0.005, steps: int = 31) -> list[str]:
    """Each joint swept through its range (the others at q = 0, clipped): where the model's own
    parts interpenetrate by more than ``depth`` with the model's contact filters, i.e. where a
    simulator that collides them can't follow the measured motion."""
    import mujoco
    data = mujoco.MjData(model)
    hinge, slide = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
    base = np.zeros(model.nq)
    for j in range(model.njnt):
        if int(model.jnt_type[j]) in (hinge, slide) and model.jnt_limited[j]:
            base[model.jnt_qposadr[j]] = np.clip(0.0, *model.jnt_range[j])
    out = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) not in (hinge, slide) or not model.jnt_limited[j]:
            continue
        lo, hi = model.jnt_range[j]
        hit = []
        for q in np.linspace(lo, hi, steps):
            data.qpos[:] = base
            data.qpos[model.jnt_qposadr[j]] = q
            mujoco.mj_forward(model, data)
            pairs = {tuple(sorted((model.body(model.geom_bodyid[c.geom1]).name or "world",
                                   model.body(model.geom_bodyid[c.geom2]).name or "world")))
                     for c in data.contact[: data.ncon] if -c.dist > depth}
            if pairs:
                hit.append((q, pairs))
        if hit:
            h = int(model.jnt_type[j]) == hinge
            pairs = sorted({"/".join(p) for _, ps in hit for p in ps})
            out.append(f"{model.joint(j).name} {_val(hit[0][0], h)}..{_val(hit[-1][0], h)} of {_val(lo, h)}..{_val(hi, h)}: "
                       f"{', '.join(pairs)} interpenetrate")
    return out


def check_mjcf_mujoco(t: Target) -> Row:
    mujoco = _try_import("mujoco")
    if mujoco is None:
        return Row(t.name, "mjcf:mujoco", "skip", "mujoco not installed")
    try:
        with captured() as log:
            model = mujoco.MjModel.from_xml_path(str(t.bundle / "model.xml"))
    except Exception as e:  # noqa: BLE001
        return Row(t.name, "mjcf:mujoco", "fail", str(e).strip().splitlines()[0])
    res = mj_settle(model)
    fails, notes = _judge(model, res)
    detail = f"mujoco {mujoco.__version__}; {model.nbody - 1} bodies, {model.ngeom} geoms; drift in {SECONDS:g} s: " + \
             (", ".join(notes) or "no joints")
    if fails:
        return Row(t.name, "mjcf:mujoco", "fail", "; ".join(fails) + "; " + detail)
    blocked = blocked_ranges(model)
    if blocked:
        detail += "; inside the measured range " + "; ".join(blocked)
    warn = _worrying(log) or blocked
    return Row(t.name, "mjcf:mujoco", "warn" if warn else "pass", detail + (f"; log: {_short(log)}" if _worrying(log) else ""))


def _worrying(log: list[str]) -> bool:
    return any(w in l.lower() for l in log for w in ("error", "warning", "cannot", "failed", "not found"))


def parent_overlaps(model) -> list[str]:
    """Contacts at the start between a body and its parent deeper than OVERLAP_M."""
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    seen = {}
    for c in data.contact[: data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        w1, w2 = model.body_weldid[b1], model.body_weldid[b2]
        related = model.body_parentid[w1] == w2 or model.body_parentid[w2] == w1 or \
            model.body_weldid[model.body_parentid[w1]] == w2 or model.body_weldid[model.body_parentid[w2]] == w1
        if related and -c.dist > OVERLAP_M:
            key = tuple(sorted((model.body(b1).name or "world", model.body(b2).name or "world")))
            seen[key] = max(seen.get(key, 0), -c.dist)
    return [f"{a}/{b} {1000 * d:.0f} mm" for (a, b), d in seen.items()]


def check_urdf_mujoco(t: Target, robot: uc.Robot) -> Row:
    mujoco = _try_import("mujoco")
    if mujoco is None:
        return Row(t.name, "urdf:mujoco", "skip", "mujoco not installed")
    try:
        with captured() as log:
            model = mujoco.MjModel.from_xml_path(str(t.bundle / "model.urdf"))
    except Exception as e:  # noqa: BLE001
        msg = str(e).strip().splitlines()[0]
        if "Error opening file" in msg and not robot.mujoco_compiler.get("strippath") == "false":
            msg += " (MuJoCo strips URDF mesh paths to the file name: add <mujoco><compiler strippath=\"false\"/></mujoco>)"
        return Row(t.name, "urdf:mujoco", "fail", msg)
    problems = []
    frames = uc.joint_frames(robot)
    mjj = mj_joints(model, robot.root)
    for j in robot.joints:
        if j.type == "fixed":
            continue
        m = mjj.get(j.name)
        if m is None:
            problems.append(f"{j.name} missing")
            continue
        anchor, axis = frames[j.name]
        if np.linalg.norm(anchor - m["anchor"]) > 1e-4 or float(axis @ m["axis"]) < 0.99999:
            problems.append(f"{j.name}: MuJoCo places it elsewhere than the URDF says")
        if j.type != "continuous" and np.abs(np.array(m["range"]) - [j.lower, j.upper]).max() > 1e-6:
            problems.append(f"{j.name}: range {m['range']}")
    res = mj_settle(model)
    fails, notes = _judge(model, res)
    fails = problems + fails
    overlaps = parent_overlaps(model)
    detail = f"mujoco {mujoco.__version__}; drift in {SECONDS:g} s: " + (", ".join(notes) or "no joints")
    if overlaps:
        detail += "; parent-child overlap at the start: " + ", ".join(overlaps) + \
                  " (MuJoCo's URDF import can't exclude these contacts: simulate model.xml)"
        if "q" in res and np.abs(res["q"] - res["q0"]).max() > THROWN:
            fails.append("the overlap throws a part")
    if fails:
        return Row(t.name, "urdf:mujoco", "fail", "; ".join(fails) + "; " + detail)
    warn = overlaps or _worrying(log)
    return Row(t.name, "urdf:mujoco", "warn" if warn else "pass", detail + (f"; log: {_short(log)}" if _worrying(log) else ""))


# ------------------------------------------------------------------ PyBullet

def pb_settle(p, urdf: Path, flags: int, seconds: float = SECONDS, friction: bool = False) -> dict:
    """Load with the root fixed, start each joint at q = 0 (clipped into its limits), switch the
    default motors off (or, with ``friction``, make them the URDF's joint friction: a velocity
    motor holding 0 with at most that force, as PyBullet users emulate it), 2 s under gravity."""
    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        b = p.loadURDF(str(urdf), useFixedBase=True, flags=flags, physicsClientId=cid)
        n = p.getNumJoints(b, physicsClientId=cid)
        info = [p.getJointInfo(b, j, physicsClientId=cid) for j in range(n)]
        movable = [j for j in range(n) if info[j][2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]
        for j in movable:
            lo, hi = info[j][8], info[j][9]
            p.resetJointState(b, j, float(np.clip(0.0, lo, hi)) if lo <= hi else 0.0, physicsClientId=cid)
            p.setJointMotorControl2(b, j, p.VELOCITY_CONTROL, targetVelocity=0,
                                    force=info[j][7] if friction else 0, physicsClientId=cid)
        start = [np.array(p.getLinkState(b, l, physicsClientId=cid)[0]) for l in range(n)]
        q0 = [p.getJointState(b, j, physicsClientId=cid)[0] for j in movable]
        self_contacts = [(c[3], c[4], c[8]) for c in p.getContactPoints(b, b, physicsClientId=cid)]
        for i in range(int(seconds * 240)):
            p.stepSimulation(physicsClientId=cid)
        q = [p.getJointState(b, j, physicsClientId=cid)[0] for j in movable]
        states = [p.getLinkState(b, l, computeLinkVelocity=1, physicsClientId=cid) for l in range(n)]
        moved = max([float(np.linalg.norm(np.array(s[0]) - s0)) for s, s0 in zip(states, start)], default=0.0)
        speed = max([float(np.linalg.norm(s[6])) for s in states], default=0.0)
        return {"body": b, "info": info, "movable": movable, "q0": q0, "q": q, "moved": moved, "speed": speed,
                "self_contacts": self_contacts,
                "dyn": [p.getDynamicsInfo(b, l, physicsClientId=cid) for l in range(-1, n)],
                "coll": [len(p.getCollisionShapeData(b, l, physicsClientId=cid)) for l in range(-1, n)],
                "vis": p.getVisualShapeData(b, physicsClientId=cid)}
    finally:
        p.disconnect(cid)


def _pb_judge(res: dict, tag: str) -> list[str]:
    fails = []
    if not all(map(math.isfinite, res["q"])) or not math.isfinite(res["moved"]):
        return [f"{tag}: NaN"]
    if res["moved"] > FLY_M:
        fails.append(f"{tag}: a part flew {res['moved']:.2f} m")
    if res["speed"] > FAST_M_S:
        fails.append(f"{tag}: still moving {res['speed']:.1f} m/s after {SECONDS:g} s")
    for k, j in enumerate(res["movable"]):
        lo, hi, kind = res["info"][j][8], res["info"][j][9], res["info"][j][2]
        slack = LIMIT_SLACK[0] if kind == 0 else LIMIT_SLACK[1]
        if lo <= hi and not (lo - slack <= res["q"][k] <= hi + slack):
            fails.append(f"{tag}: {res['info'][j][1].decode()} at {res['q'][k]:.3f} outside {lo:.3f}..{hi:.3f}")
    return fails


def check_pybullet(t: Target, robot: uc.Robot) -> Row:
    p = _try_import("pybullet")
    if p is None:
        return Row(t.name, "urdf:pybullet", "skip", "pybullet not installed")
    urdf = t.bundle / "model.urdf"
    flags = p.URDF_USE_INERTIA_FROM_FILE
    try:
        with captured() as log:
            res = pb_settle(p, urdf, flags)
    except Exception as e:  # noqa: BLE001
        return Row(t.name, "urdf:pybullet", "fail", f"{e}: {_short(log, 4)}")
    fails = []
    bad = [l for l in log if any(w in l.lower() for w in ("cannot find", "failed", "error", "unsupported"))]
    if bad:
        fails.append("load: " + _short(bad, 3))
    names = [i[12].decode() for i in res["info"]]  # child link names, in PyBullet's link order
    links = [robot.root] + names
    if sorted(links) != sorted(robot.links):
        fails.append(f"links {links} vs URDF {sorted(robot.links)}")
    for k, name in enumerate(links):
        if name not in robot.links:
            continue
        want = len(robot.links[name].collisions)
        if want and res["coll"][k] < want:
            fails.append(f"{name}: {res['coll'][k]} collision shapes of {want}")
        mass = res["dyn"][k][0]
        if k > 0 and abs(mass - robot.links[name].mass) > 1e-6 * max(1, mass):
            fails.append(f"{name}: mass {mass} vs URDF {robot.links[name].mass}")
        if k > 0:
            want_I = np.sort(np.linalg.eigvalsh(robot.links[name].inertia))
            got_I = np.sort(res["dyn"][k][2])
            if np.abs(want_I - got_I).max() > 1e-4 * want_I.max():
                fails.append(f"{name}: inertia {got_I.tolist()} vs URDF {want_I.tolist()}")
    vis_links = {v[1] for v in res["vis"]}
    for k, name in enumerate(links):
        if name in robot.links and robot.links[name].visuals and (k - 1) not in vis_links:
            fails.append(f"{name}: no visual shape loaded")
    fails += _pb_judge(res, "free")
    notes = [f"{len(links)} links, {sum(res['coll'])} collision / {len(res['vis'])} visual shapes; free joints in "
             f"{SECONDS:g} s: " + (", ".join(f"{res['info'][j][1].decode()} {res['q0'][k]:+.3f}->{res['q'][k]:+.3f}"
                                             for k, j in enumerate(res["movable"])) or "none")]
    if any(res["info"][j][7] > 0 for j in res["movable"]):  # the URDF's joint friction, as motors
        with captured():
            fr = pb_settle(p, urdf, flags, friction=True)
        fails += _pb_judge(fr, "friction")
        notes.append("with <dynamics friction> as motors: " + ", ".join(
            f"{fr['info'][j][1].decode()} {_val(fr['q'][k] - fr['q0'][k], fr['info'][j][2] == p.JOINT_REVOLUTE)}"
            for k, j in enumerate(fr["movable"])))
    # Self-collision as a robot user turns it on (parents excluded): do non-adjacent parts collide?
    if len(robot.links) > 2:
        with captured():
            sc = pb_settle(p, urdf, flags | p.URDF_USE_SELF_COLLISION)
        fails += _pb_judge(sc, "self-collision")
        deep = [(a, b, d) for a, b, d in sc["self_contacts"] if d < -OVERLAP_M]
        warn = []
        if deep:
            pairs = sorted({f"{links[a + 1]}/{links[b + 1]}" for a, b, _ in deep})
            warn.append(f"with self-collision, overlapping at the start: {', '.join(pairs)}")
        if any(abs(a - b) > 1e-3 for a, b in zip(sc["q"], res["q"])):
            warn.append("with self-collision on, the parts stop each other: " + ", ".join(
                f"{res['info'][j][1].decode()} at {sc['q'][k]:+.3f} (free {res['q'][k]:+.3f})" for k, j in enumerate(sc["movable"])))
        notes += warn
    else:
        warn = []
    if fails:
        return Row(t.name, "urdf:pybullet", "fail", "; ".join(fails[:5]))
    return Row(t.name, "urdf:pybullet", "warn" if warn else "pass", "; ".join(notes))


# ------------------------------------------------------------------ yourdfpy

def check_yourdfpy(t: Target, robot: uc.Robot) -> Row:
    yd = _try_import("yourdfpy")
    if yd is None:
        return Row(t.name, "urdf:yourdfpy", "skip", "yourdfpy not installed")
    try:
        with captured() as log:
            u = yd.URDF.load(str(t.bundle / "model.urdf"), build_scene_graph=True, load_meshes=True,
                             build_collision_scene_graph=True, load_collision_meshes=True)
            valid = u.validate()
    except Exception as e:  # noqa: BLE001
        return Row(t.name, "urdf:yourdfpy", "fail", f"{type(e).__name__}: {e}")
    fails = []
    missing = [l for l in log if "can't find" in l.lower() or "unable to resolve" in l.lower()]
    if missing:
        fails.append("meshes not found: " + _short(missing, 2))
    if not valid:
        fails.append("validate() failed: " + _short(log, 3))
    n_vis = sum(len(l.visuals) for l in robot.links.values())
    got_vis = len(u.scene.geometry)
    if got_vis < n_vis:
        fails.append(f"{got_vis} of {n_vis} visuals loaded")
    # What it loaded spans what the URDF declares (its collision scene merges each link's shapes).
    for which, scene in (("visuals", u.scene), ("collisions", u.collision_scene)):
        want = declared_points(t, robot, which, frame="root")
        if not want:
            continue
        if scene is None or not len(scene.geometry):
            fails.append(f"no {which[:-1]} geometry loaded")
            continue
        gap = _bounds_gap(np.vstack(list(want.values())), np.asarray(scene.bounds))
        if gap > 0.002:
            fails.append(f"{which} loaded {1000 * gap:.0f} mm off the declared bounds")
    got_col = len(u.collision_scene.geometry) if u.collision_scene else 0
    frames = uc.link_frames(robot)
    worst = 0.0
    for name, T in frames.items():
        worst = max(worst, float(np.abs(u.get_transform(name, robot.root) - T).max()))
    if worst > 1e-6:
        fails.append(f"forward kinematics differ by {worst:.2g}")
    for j in robot.joints:
        yj = u.joint_map.get(j.name)
        if yj is None:
            fails.append(f"{j.name} missing")
        elif yj.type != j.type or (j.lower is not None and (abs(yj.limit.lower - j.lower) > 1e-9 or abs(yj.limit.upper - j.upper) > 1e-9)):
            fails.append(f"{j.name}: read as {yj.type} {yj.limit.lower if yj.limit else ''}..{yj.limit.upper if yj.limit else ''}")
    if fails:
        return Row(t.name, "urdf:yourdfpy", "fail", "; ".join(fails[:4]))
    return Row(t.name, "urdf:yourdfpy", "pass", f"valid; {got_vis} visual / {got_col} collision geometries loaded "
                                                "where the URDF puts them; FK == URDF")


# ------------------------------------------------------------------ Drake

def check_drake(t: Target, robot: uc.Robot) -> Row:
    if _try_import("pydrake") is None:
        return Row(t.name, "urdf:drake", "skip", "pydrake not installed (uv run --with drake ...)")
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.systems.analysis import Simulator
    from pydrake.systems.framework import DiagramBuilder
    try:
        with captured() as log:
            builder = DiagramBuilder()
            plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
            model = Parser(plant).AddModels(str(t.bundle / "model.urdf"))[0]
            plant.WeldFrames(plant.world_frame(), plant.GetBodyByName(robot.root, model).body_frame())
            plant.Finalize()
            diagram = builder.Build()
            ctx = diagram.CreateDefaultContext()
            pctx = plant.GetMyMutableContextFromRoot(ctx)
            q0 = plant.GetPositions(pctx).copy()
            Simulator(diagram, ctx).AdvanceTo(SECONDS)
            q = plant.GetPositions(pctx).copy()
    except Exception as e:  # noqa: BLE001
        return Row(t.name, "urdf:drake", "fail", str(e).strip().splitlines()[0][:300])
    fails = []
    if not np.all(np.isfinite(q)):
        fails.append("NaN")
    for j in robot.joints:
        if j.type not in ("revolute", "prismatic"):
            continue
        dj = plant.GetJointByName(j.name, model)
        v = q[dj.position_start()]
        slack = LIMIT_SLACK[0] if j.type == "revolute" else LIMIT_SLACK[1]
        if not (j.lower - slack <= v <= j.upper + slack):
            fails.append(f"{j.name} at {v:.3f} outside {j.lower}..{j.upper}")
    warns = [l for l in log if "warning" in l.lower()]
    friction = any("friction" in w for w in warns)
    other = [w for w in warns if "friction" not in w]
    detail = f"{plant.num_bodies() - 1} bodies; {SECONDS:g} s: q {np.round(q0, 3).tolist()} -> {np.round(q, 3).tolist()}"
    if friction:
        detail += "; Drake ignores joint friction"
    if other:
        detail += "; " + _short(other, 2)
    if fails:
        return Row(t.name, "urdf:drake", "fail", "; ".join(fails) + "; " + detail)
    return Row(t.name, "urdf:drake", "warn" if other else "pass", detail)


# ------------------------------------------------------------------ single-file downloads

def check_single(t: Target, work: Path) -> list[Row]:
    if t.viewer is None:
        return []
    rows = []
    for f, kind in (("model.urdf", "URDF"), ("model.xml", "MJCF")):
        src = t.viewer / f
        if not src.exists():
            continue
        alone = work / f"single_{t.name}_{f}"
        alone.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, alone / f)
        if f == "model.urdf":
            rep = uc.asset_problems(alone, alone / f)
            missing = [p for p in rep.problems if p.endswith("missing")]
        else:
            missing = [w for _, w, rel in uc.mjcf_assets(alone / f) if not (alone / rel).exists()]
        if missing:
            rows.append(Row(t.name, f"single-file {kind}", "warn",
                            f"{f} alone needs {len(missing)} files from model.zip (e.g. {missing[0].split(':')[0]})"))
        else:
            rows.append(Row(t.name, f"single-file {kind}", "pass", f"{f} loads alone"))
    return rows


# ------------------------------------------------------------------ main

def check(t: Target, work: Path, drake: bool) -> list[Row]:
    try:
        robot = uc.parse_urdf(t.bundle / "model.urdf")
    except Exception as e:  # noqa: BLE001
        return [Row(t.name, "files", "fail", f"model.urdf doesn't parse: {e}")]
    mj = mjcf_model(t)
    rows = [check_files(t, robot), check_inertia(t, robot, mj), check_joints(t, robot, mj),
            check_collision(t, robot, mj), check_mjcf_mujoco(t), check_urdf_mujoco(t, robot),
            check_pybullet(t, robot), check_yourdfpy(t, robot)]
    if drake:
        rows.append(check_drake(t, robot))
    return rows + check_single(t, work)


def print_table(rows: list[Row]):
    w = max(len(r.model) for r in rows)
    c = max(len(r.check) for r in rows)
    last = None
    for r in rows:
        if last is not None and r.model != last:
            print()
        last = r.model
        print(f"{r.model:<{w}}  {r.check:<{c}}  {r.status.upper():<4}  {r.detail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="model.zip, model folder, or a folder of them")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=RUN_DIR",
                    help="compare model NAME with the fit's records in RUN_DIR (repeatable)")
    ap.add_argument("--no-drake", action="store_true", help="skip Drake even when it is installed")
    ap.add_argument("--json", type=Path, help="also write the rows here")
    ap.add_argument("--keep", type=Path, help="unpack into this folder (kept) instead of a temporary one")
    args = ap.parse_args()
    runs = dict(r.split("=", 1) for r in args.run)
    runs = {k: Path(v) for k, v in runs.items()}
    with tempfile.TemporaryDirectory(prefix="check_exports_") as tmp:
        work = args.keep or Path(tmp)
        if args.keep and work.exists():
            shutil.rmtree(work)
        rows = []
        for t in discover(args.paths, work, runs):
            rows += check(t, work, drake=not args.no_drake and _try_import("pydrake") is not None)
    print_table(rows)
    fails = [r for r in rows if r.status == "fail"]
    print(f"\n{len(rows)} rows: {sum(r.status == 'pass' for r in rows)} pass, "
          f"{sum(r.status == 'warn' for r in rows)} warn, {len(fails)} fail, {sum(r.status == 'skip' for r in rows)} skip")
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in rows], indent=1))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
