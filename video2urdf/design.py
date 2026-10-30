"""A clean, object-like visual for a fitted model, designed by Spark from primitives.

The fitted meshes are evidence: silhouettes voted over the video or depth points, so they
read as blobs. Spark names the object, then rebuilds each link from a few primitives
(boxes, rounded boxes, cylinders, spheres, bars) in the link's own frame, sized from the
measured extents and, where a side is never seen, from what such an object usually looks
like. Each design is rendered over the video from the recorded camera and joint values and
scored against the SAM masks; Spark sees the overlay and the scores and revises.

The measured meshes stay as collision geometry (MJCF group 3, URDF <collision>); the design
replaces only the visuals. Joints, limits and physics are untouched.
"""

import json
import math
import shutil
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .pipeline import load_parts, video_info
from .texture import link_masks

PRIMITIVES = ("box", "rounded_box", "cylinder", "sphere", "bar")
GENERATED = ("_sam3d", "_infer")  # earlier generated visuals: hidden under a design
PREFIX = "design_"
SAMPLE_SIDE = 480  # px; long side of the frames the design is scored on
TOLERANCE = 0.01  # of the long side: how far a rendered pixel may be from the mask and still count
LINK_COLORS = [(0, 200, 255), (255, 80, 80), (80, 220, 80), (255, 0, 255), (255, 255, 0), (0, 128, 255)]


# ----------------------------------------------------------------- primitives -> meshes

def _rotation(p: dict) -> np.ndarray:
    return Rotation.from_euler("xyz", p.get("euler_deg") or [0, 0, 0], degrees=True).as_matrix()


def _bar_frame(p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center, rotation (columns: length, width, thickness) and full size of a bar, in mm."""
    a, b = np.asarray(p["from"], float), np.asarray(p["to"], float)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        raise ValueError(f"bar {p.get('name')} has zero length")
    d /= length
    axes = np.eye(3)
    if p.get("width_axis") in ("x", "y", "z"):
        w = axes["xyz".index(p["width_axis"])]
    else:  # the link axis most across the bar; y first (across the plane of a planar model)
        w = min((axes[1], axes[0], axes[2]), key=lambda v: abs(v @ d) - 1e-3 * (v[1] != 0))
    w = w - (w @ d) * d
    if np.linalg.norm(w) < 1e-6:
        w = np.cross(d, [0, 0, 1.0]) if abs(d[2]) < 0.9 else np.cross(d, [1.0, 0, 0])
    w /= np.linalg.norm(w)
    return (a + b) / 2, np.column_stack([d, w, np.cross(d, w)]), np.array([length, p["width"], p["thickness"]], float)


def _cylinder_frame(p: dict) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Center, rotation (z along the axis), radius and length of a cylinder, in mm."""
    a, b = np.asarray(p["from"], float), np.asarray(p["to"], float)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        raise ValueError(f"cylinder {p.get('name')} has zero length")
    z = d / length
    x = np.cross([0, 1.0, 0], z) if abs(z[1]) < 0.9 else np.cross([1.0, 0, 0], z)
    x /= np.linalg.norm(x)
    return (a + b) / 2, np.column_stack([x, np.cross(z, x), z]), float(p["radius"]), length


def _box_points(size) -> np.ndarray:
    return np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)]) * np.asarray(size)


def _sphere_points(n: int = 96) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi, theta = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


def _hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    center = points.mean(0)
    for f in faces:  # outward winding
        a, b, c = points[f]
        if np.cross(b - a, c - a) @ (a - center) < 0:
            f[[1, 2]] = f[[2, 1]]
    return points, faces


def primitive_mesh(p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (mm, link frame) and faces of one primitive."""
    t = p["type"]
    if t in ("box", "rounded_box"):
        size = np.asarray(p["size"], float)
        if t == "rounded_box":
            r = float(np.clip(p.get("corner_radius", 0) or 0, 0, 0.49 * size.min()))
            if r > 0:
                corners = _box_points(size - 2 * r)
                pts = (corners[:, None, :] + r * _sphere_points(48)[None]).reshape(-1, 3)
                v, f = _hull(pts)
                return v @ _rotation(p).T + np.asarray(p["center"], float), f
        v, f = _hull(_box_points(size))
        return v @ _rotation(p).T + np.asarray(p["center"], float), f
    if t == "bar":
        c, R, size = _bar_frame(p)
        v, f = _hull(_box_points(size))
        return v @ R.T + c, f
    if t == "cylinder":
        c, R, r, length = _cylinder_frame(p)
        a = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        ring = np.column_stack([r * np.cos(a), r * np.sin(a)])
        pts = np.vstack([np.column_stack([ring, np.full(32, -length / 2)]), np.column_stack([ring, np.full(32, length / 2)])])
        v, f = _hull(pts)
        return v @ R.T + c, f
    if t == "sphere":
        v, f = _hull(float(p["radius"]) * _sphere_points())
        return v + np.asarray(p["center"], float), f
    raise ValueError(f"unknown primitive type {t!r}")


def validate_design(design: dict, links: list[str]) -> list[str]:
    """Problems that stop a design from being built (an empty list when it can be)."""
    problems = []
    seen = {d.get("link") for d in design.get("links", [])}
    for missing in sorted(set(links) - seen):
        problems.append(f"link {missing!r} has no primitives")
    for d in design.get("links", []):
        if d.get("link") not in links:
            problems.append(f"unknown link {d.get('link')!r}; links are {links}")
            continue
        if not d.get("primitives"):
            problems.append(f"link {d['link']!r} has no primitives")
        for p in d.get("primitives", []):
            try:
                if p.get("type") not in PRIMITIVES:
                    raise ValueError(f"type must be one of {PRIMITIVES}")
                need = {"box": ("center", "size"), "rounded_box": ("center", "size"), "sphere": ("center", "radius"),
                        "cylinder": ("from", "to", "radius"), "bar": ("from", "to", "width", "thickness")}[p["type"]]
                for key in need:
                    if key not in p:
                        raise ValueError(f"needs {key}")
                for key in ("size", "radius", "width", "thickness"):
                    if key in need and np.any(np.asarray(p[key], float) <= 0):
                        raise ValueError(f"{key} must be positive")
                v, _ = primitive_mesh(p)
                if not np.isfinite(v).all() or np.abs(v).max() > 1e5:
                    raise ValueError("coordinates out of range")
            except (ValueError, TypeError, KeyError, IndexError) as e:
                problems.append(f"{d['link']}/{p.get('name', '?')}: {e}")
    return problems


# ----------------------------------------------------------------- model files

def _is_design(el: ET.Element) -> bool:
    return (el.get("name") or "").startswith(PREFIX) or (el.get("mesh") or "").startswith(PREFIX)


def _is_generated(geom: ET.Element) -> bool:
    return (geom.get("mesh") or "").endswith(GENERATED) or (geom.get("name") or "").startswith("inferred_")


def has_design(track_dir: Path) -> bool:
    root = ET.parse(track_dir / "model.xml").getroot()
    return any(_is_design(g) for g in root.iter("geom"))


def originals(track_dir: Path) -> dict[str, Path]:
    """model files as they were before any design: saved once, refreshed when the model was refitted."""
    names = [n for n in ("model.xml", "model.urdf", "model.nophysics.xml", "model.nophysics.urdf") if (track_dir / n).exists()]
    if not has_design(track_dir):
        for n in names:
            shutil.copy(track_dir / n, track_dir / n.replace("model.", "model.predesign.", 1))
    return {n: track_dir / n.replace("model.", "model.predesign.", 1) for n in names}


def measured_xml(track_dir: Path) -> str:
    """The MJCF with only measured geoms visible: they move to group 3, generated visuals to 4."""
    root = ET.parse(originals(track_dir)["model.xml"]).getroot()
    for g in root.iter("geom"):
        g.set("group", "4" if _is_generated(g) else "3")
    return ET.tostring(root, encoding="unicode")


def _quat(R: np.ndarray) -> str:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return f"{w:.6f} {x:.6f} {y:.6f} {z:.6f}"


def _v(x) -> str:
    return " ".join(f"{float(v):.5f}" for v in x)


def _mjcf_geom(p: dict, name: str, rgba: str) -> tuple[dict, str | None]:
    """MJCF geom attributes (metres) and, for a rounded box, the mesh asset name it needs."""
    common = {"name": name, "rgba": rgba, "contype": "0", "conaffinity": "0", "mass": "0"}
    t = p["type"]
    if t == "box":
        return {**common, "type": "box", "pos": _v(np.asarray(p["center"]) / 1000),
                "quat": _quat(_rotation(p)), "size": _v(np.asarray(p["size"]) / 2000)}, None
    if t == "bar":
        c, R, size = _bar_frame(p)
        return {**common, "type": "box", "pos": _v(c / 1000), "quat": _quat(R), "size": _v(size / 2000)}, None
    if t == "cylinder":
        c, R, r, length = _cylinder_frame(p)
        return {**common, "type": "cylinder", "pos": _v(c / 1000), "quat": _quat(R), "size": _v([r / 1000, length / 2000])}, None
    if t == "sphere":
        return {**common, "type": "sphere", "pos": _v(np.asarray(p["center"]) / 1000), "size": f"{p['radius'] / 1000:.5f}"}, None
    return {**common, "type": "mesh", "mesh": name}, name  # rounded_box: vertices already in the link frame


def _urdf_visual(p: dict, name: str, rgb, mesh_file: str | None) -> ET.Element:
    vis = ET.Element("visual", name=name)
    t = p["type"]
    if t == "rounded_box":
        origin, geometry = (np.zeros(3), np.eye(3)), ("mesh", {"filename": mesh_file})
    elif t == "box":
        origin, geometry = (np.asarray(p["center"]) / 1000, _rotation(p)), ("box", {"size": _v(np.asarray(p["size"]) / 1000)})
    elif t == "bar":
        c, R, size = _bar_frame(p)
        origin, geometry = (c / 1000, R), ("box", {"size": _v(size / 1000)})
    elif t == "cylinder":
        c, R, r, length = _cylinder_frame(p)
        origin, geometry = (c / 1000, R), ("cylinder", {"radius": f"{r / 1000:.5f}", "length": f"{length / 1000:.5f}"})
    else:
        origin, geometry = (np.asarray(p["center"]) / 1000, np.eye(3)), ("sphere", {"radius": f"{p['radius'] / 1000:.5f}"})
    with warnings.catch_warnings():  # gimbal lock is harmless: any of the equivalent angles will do
        warnings.simplefilter("ignore")
        rpy = Rotation.from_matrix(origin[1]).as_euler("xyz")
    ET.SubElement(vis, "origin", xyz=_v(origin[0]), rpy=_v(rpy))
    ET.SubElement(ET.SubElement(vis, "geometry"), geometry[0], **geometry[1])
    ET.SubElement(ET.SubElement(vis, "material", name=name + "_color"), "color", rgba=_v(rgb) + " 1")
    return vis


def _rgb(p: dict) -> list[float]:
    rgb = p.get("rgb") or [0.7, 0.7, 0.7]
    return [float(np.clip(c, 0, 1)) for c in rgb]


def design_mjcf(base_xml: str, design: dict, mesh_dir: str = "meshes") -> tuple[str, dict[str, tuple]]:
    """The MJCF with the design's visuals added to each body, and the rounded-box meshes to write."""
    root = ET.fromstring(base_xml)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    meshes = {}
    bodies = {b.get("name"): b for b in root.iter("body")}
    for d in design["links"]:
        body = bodies[d["link"]]
        for i, p in enumerate(d["primitives"]):
            name = f"{PREFIX}{d['link']}_{i}"
            attrs, mesh = _mjcf_geom(p, name, _v(_rgb(p)) + " 1")
            if mesh:
                v, f = primitive_mesh(p)
                meshes[name] = (v / 1000, f)
                ET.SubElement(asset, "mesh", name=name, file=f"{mesh_dir}/{name}.stl")
            # after the measured geoms, before child bodies
            index = max([k + 1 for k, c in enumerate(body) if c.tag in ("geom", "joint", "inertial")], default=0)
            body.insert(index, ET.Element("geom", **attrs))
    return ET.tostring(root, encoding="unicode"), meshes


def apply_design(track_dir: Path, design: dict) -> None:
    """Write the design into model.xml/model.urdf (and the nophysics copies)."""
    from .shape import write_stl
    sources = originals(track_dir)
    (track_dir / "meshes").mkdir(exist_ok=True)
    for old in (track_dir / "meshes").glob(PREFIX + "*.stl"):
        old.unlink()
    for name, src in sources.items():
        out = track_dir / name
        if name.endswith(".xml"):
            root = ET.parse(src).getroot()
            for g in root.iter("geom"):
                if _is_generated(g):  # an older generated visual: hide it, it no longer collides
                    g.set("group", "4")
                    g.set("contype", "0")
                    g.set("conaffinity", "0")
                else:
                    g.set("group", "3")
            xml, meshes = design_mjcf(ET.tostring(root, encoding="unicode"), design)
            for mesh_name, (v, f) in meshes.items():
                write_stl(track_dir / "meshes" / f"{mesh_name}.stl", v, f)
            tree = ET.ElementTree(ET.fromstring(xml))
            ET.indent(tree, "  ")
            tree.write(out)
        else:
            tree = ET.parse(src)
            links = {l.get("name"): l for l in tree.getroot().findall("link")}
            for d in design["links"]:
                link = links.get(d["link"])
                if link is None:
                    continue
                for vis in link.findall("visual"):
                    link.remove(vis)
                for i, p in enumerate(d["primitives"]):
                    name = f"{PREFIX}{d['link']}_{i}"
                    mesh = f"meshes/{name}.stl" if p["type"] == "rounded_box" else None
                    link.insert(i, _urdf_visual(p, name, _rgb(p), mesh))
            ET.indent(tree, "  ")
            tree.write(out, xml_declaration=True, encoding="unicode")


# ----------------------------------------------------------------- evidence

def _load(track_dir: Path, xml: str) -> mujoco.MjModel:
    tmp = track_dir / ".design-check.xml"  # next to the model, so mesh paths resolve
    tmp.write_text(xml)
    try:
        return mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)


def _geom_triangles(model: mujoco.MjModel, data: mujoco.MjData, g: int) -> np.ndarray:
    """A geom's triangles in the world frame, (n, 3, 3) metres."""
    t = int(model.geom_type[g])
    if t == int(mujoco.mjtGeom.mjGEOM_MESH):
        m = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[m]: model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        f = model.mesh_face[model.mesh_faceadr[m]: model.mesh_faceadr[m] + model.mesh_facenum[m]]
        tri = v[f]
    else:
        size = model.geom_size[g]
        p = ({"type": "box", "center": [0, 0, 0], "size": 2 * size * 1000} if t == int(mujoco.mjtGeom.mjGEOM_BOX) else
             {"type": "sphere", "center": [0, 0, 0], "radius": size[0] * 1000} if t == int(mujoco.mjtGeom.mjGEOM_SPHERE) else
             {"type": "cylinder", "from": [0, 0, -size[1] * 1000], "to": [0, 0, size[1] * 1000], "radius": size[0] * 1000}
             if t in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)) else None)
        if p is None:
            return np.zeros((0, 3, 3))
        v, f = primitive_mesh(p)
        tri = (v / 1000)[f]
    R, x = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
    return tri @ R.T + x


def link_geometry(model: mujoco.MjModel, data: mujoco.MjData, group: int) -> dict[str, np.ndarray]:
    """Triangles per body (world frame, metres) of the geoms in one group."""
    out = {}
    for g in range(model.ngeom):
        if model.geom_group[g] != group:
            continue
        name = model.body(model.geom_bodyid[g]).name
        out.setdefault(name, []).append(_geom_triangles(model, data, g))
    return {k: np.concatenate(v) for k, v in out.items() if v}


def _round(x, nd=1):
    return [round(float(v), nd) for v in x]


def evidence(track_dir: Path) -> dict:
    """What was measured, per link, in each link's own frame (mm)."""
    model = _load(track_dir, measured_xml(track_dir))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tris = link_geometry(model, data, 3)
    planar = (track_dir / "joints.json").exists() and not (track_dir / "model3d.json").exists()
    static = (track_dir / "static.json").exists()
    links = []
    for b in range(1, model.nbody):
        name = model.body(b).name
        R, x = data.xmat[b].reshape(3, 3), data.xpos[b]
        item = {"link": name, "parent": model.body(model.body_parentid[b]).name if model.body_parentid[b] else None,
                "origin_in_model_mm": _round(x * 1000)}
        jnt = [j for j in range(model.njnt) if model.jnt_bodyid[j] == b]
        if jnt:
            j = jnt[0]
            kind = {int(mujoco.mjtJoint.mjJNT_HINGE): "revolute", int(mujoco.mjtJoint.mjJNT_SLIDE): "prismatic"}.get(int(model.jnt_type[j]), "other")
            lo, hi = model.jnt_range[j]
            item["joint"] = {"type": kind, "axis_in_link": _round(model.jnt_axis[j], 3), "at_link_origin": True,
                             "range": _round(np.rad2deg([lo, hi]) if kind == "revolute" else np.array([lo, hi]) * 1000)
                             , "range_unit": "deg" if kind == "revolute" else "mm"}
        kids = [c for c in range(1, model.nbody) if model.body_parentid[c] == b]
        item["child_joints_in_link_mm"] = {model.body(c).name: _round(R.T @ (data.xpos[c] - x) * 1000) for c in kids}
        if name in tris:
            pts = (tris[name].reshape(-1, 3) - x) @ R * 1000
            lo, hi = np.percentile(pts, 1, 0), np.percentile(pts, 99, 0)
            centered = pts - pts.mean(0)
            _, s, vt = np.linalg.svd(centered[:: max(1, len(centered) // 4000)], full_matrices=False)
            proj = centered @ vt[0]
            a, z = pts.mean(0) + vt[0] * np.percentile(proj, 1), pts.mean(0) + vt[0] * np.percentile(proj, 99)
            cross = [float(np.ptp(np.percentile(centered @ vt[k], [2, 98]))) for k in (1, 2)]
            item["measured_bbox_mm"] = {"min": _round(lo), "max": _round(hi), "size": _round(hi - lo)}
            item["measured_long_axis_mm"] = {"from": _round(a), "to": _round(z), "cross_section": _round(cross)}
        rgba = [model.geom_rgba[g] for g in range(model.ngeom) if model.geom_bodyid[g] == b and model.geom_group[g] == 3]
        if rgba and not np.allclose(rgba[0][:3], 1):  # textured meshes are white; their colour is in the video
            item["measured_colour_rgb"] = _round(rgba[0][:3], 2)
        links.append(item)
    if planar:
        frame = ("Planar side-view model: x = right in the video, z = up in the video, y = into the scene. "
                 "x and z extents are measured from the silhouettes; the y extent (depth across the view) was "
                 "NOT measured and is a placeholder.")
        unmeasured = ["y"]
    else:
        frame = ("3D model from depth: z is up (away from the floor/base); each link frame is parallel to the model "
                 "frame at the reference pose. Surfaces the camera never saw (backs, undersides, the depth of a "
                 "cabinet seen from the front) are missing or thin in the measured extents.")
        unmeasured = []
    for item in links:  # an unmeasured extent is a placeholder: hide it so it isn't copied
        for ax in unmeasured:
            i = "xyz".index(ax)
            if "measured_bbox_mm" in item:
                for key in ("min", "max", "size"):
                    item["measured_bbox_mm"][key][i] = None
            if "measured_long_axis_mm" in item:
                item["measured_long_axis_mm"].pop("cross_section", None)
                for key in ("from", "to"):
                    item["measured_long_axis_mm"][key][i] = None
    source = track_dir / "measurement_source.json"
    return {"frame_convention": frame, "unmeasured_axes": unmeasured, "static_model": static,
            "scale_source": json.loads(source.read_text()) if source.exists() else None, "links": links}


# ----------------------------------------------------------------- scoring over the video

class Scene:
    """Sampled frames, masks and camera/joint values to render any variant of the model over the video."""

    def __init__(self, track_dir: Path, video: Path, samples: int = 16):
        self.track_dir, self.video = track_dir, video
        (self.w, self.h), n, _ = video_info(video)
        s = min(1.0, SAMPLE_SIDE / max(self.w, self.h))
        self.s, self.sw, self.sh = s, max(1, round(self.w * s)), max(1, round(self.h * s))
        _, tracks = load_parts(track_dir)
        shows = {k: [p for p in v if p in tracks] for k, v in link_masks(track_dir).items()}
        self.planar = (track_dir / "joints.json").exists() and not (track_dir / "replay3d.npz").exists()
        if self.planar:
            j = json.loads((track_dir / "joints.json").read_text())
            self.q = {f"{x['child']}_joint": _hold(np.array(x["q"], float)) for x in j["joints"] if x["type"] != "fixed"}
            rp = track_dir / "planar_replay.npz"
            self.root = _hold_poses(np.load(rp)["root_poses"]) if rp.exists() else None
            n = min([n] + [len(q) for q in self.q.values()] + ([len(self.root)] if self.root is not None else []))
        else:
            rec = np.load(track_dir / "replay3d.npz")
            self.cams, self.f = rec["cam_to_world"], float(rec["f"])
            self.q = {k[2:]: rec[k] for k in rec.files if k.startswith("q_")}
            n = min(n, len(self.cams))
        usable = {t: {link: [p for p in parts if t in tracks[p] and tracks[p][t].usable(self.w, self.h)]
                      for link, parts in shows.items()} for t in range(n)}
        good = [t for t in range(n) if sum(bool(v) for v in usable[t].values()) >= max(1, len(shows) // 2)]
        pick = sorted({good[int(i)] for i in np.linspace(0, len(good) - 1, min(samples, len(good)))}) if good else []
        self.frames = pick
        self.masks = {}
        for t in pick:
            for link, parts in usable[t].items():
                if parts:
                    m = np.zeros((self.h, self.w), bool)
                    for p in parts:
                        m |= tracks[p][t].full_mask(self.h, self.w)
                    self.masks[(t, link)] = cv2.resize(m.astype(np.uint8), (self.sw, self.sh), interpolation=cv2.INTER_NEAREST) > 0
        self.links = list(shows)
        self.images = _read_frames(video, pick, (self.sw, self.sh))

    def render(self, xml: str, group: int) -> dict[int, np.ndarray]:
        """Body id per pixel for each sampled frame, showing only one geom group; and body names."""
        model = _load(self.track_dir, xml)
        data = mujoco.MjData(model)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[group] = 1
        renderer = mujoco.Renderer(model, self.sh, self.sw)
        renderer.enable_segmentation_rendering()
        cam = model.camera("video").id
        if not self.planar:
            model.cam_fovy[cam] = np.rad2deg(2 * np.arctan(self.h / (2 * self.f)))
        out = {}
        try:
            for t in self.frames:
                for name, q in self.q.items():
                    if t < len(q) and np.isfinite(q[t]):
                        data.joint(name).qpos[0] = q[t]
                if not self.planar:
                    quat = np.zeros(4)
                    mujoco.mju_mat2Quat(quat, (self.cams[t][:3, :3] @ np.diag([1., -1., -1.])).flatten())
                    model.cam_pos[cam], model.cam_quat[cam] = self.cams[t][:3, 3], quat
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam, scene_option=opt)
                seg = renderer.render()
                geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
                body = np.where(geom >= 0, model.geom_bodyid[np.maximum(geom, 0)], -1).astype(np.int16)
                if self.planar and self.root is not None:
                    A = self.root[t][:2].copy()
                    A[:, 2] *= self.s
                    body = cv2.warpAffine(body, A, (self.sw, self.sh), flags=cv2.INTER_NEAREST, borderValue=-1)
                out[t] = body
        finally:
            renderer.close()
        self.body_names = {model.body(b).name: b for b in range(model.nbody)}
        return out

    def score(self, xml: str, group: int) -> dict:
        labels = self.render(xml, group)
        res = {}
        r = max(1, round(TOLERANCE * max(self.sw, self.sh)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        for link in self.links:
            ious, cover, fs = [], [], []
            b = self.body_names.get(link)
            for t in self.frames:
                m = self.masks.get((t, link))
                if m is None or b is None:
                    continue
                pred = labels[t] == b
                inter = (pred & m).sum()
                ious.append(inter / max(1, (pred | m).sum()))
                cover.append(inter / max(1, m.sum()))
                # within a few pixels: thin parts lose most of their IoU to a one-pixel shift
                near_m = cv2.dilate(m.astype(np.uint8), kernel) > 0
                near_p = cv2.dilate(pred.astype(np.uint8), kernel) > 0
                precision = (pred & near_m).sum() / max(1, pred.sum())
                recall = (m & near_p).sum() / max(1, m.sum())
                fs.append(2 * precision * recall / max(1e-9, precision + recall))
            res[link] = {"iou": round(float(np.mean(ious)), 3) if ious else None,
                         "mask_covered": round(float(np.mean(cover)), 3) if cover else None,
                         "f_near": round(float(np.mean(fs)), 3) if fs else None, "frames": len(ious)}
        self._last = labels
        return res

    def overlay(self, labels: dict[int, np.ndarray], title: str, count: int = 4) -> np.ndarray:
        """Frames with each link's render filled in its colour and the SAM mask outlined in white."""
        ts = [self.frames[int(i)] for i in np.linspace(0, len(self.frames) - 1, min(count, len(self.frames)))]
        tiles = []
        for t in ts:
            img = self.images[t].copy()
            for k, link in enumerate(self.links):
                color = np.array(LINK_COLORS[k % len(LINK_COLORS)])
                pred = labels[t] == self.body_names.get(link, -2)
                img[pred] = (0.45 * img[pred] + 0.55 * color).astype(np.uint8)
                m = self.masks.get((t, link))
                if m is not None:
                    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, cs, -1, (255, 255, 255), 2)
            cv2.putText(img, f"frame {t}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(img, f"frame {t}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            tiles.append(img)
        strip = np.concatenate(tiles, 1)
        key = np.full((34, strip.shape[1], 3), 30, np.uint8)
        x = 8
        for k, link in enumerate(self.links):
            cv2.rectangle(key, (x, 9), (x + 16, 25), LINK_COLORS[k % len(LINK_COLORS)], -1)
            cv2.putText(key, link, (x + 22, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            x += 40 + 11 * len(link)
        cv2.putText(key, title + "  (white outline = SAM mask)", (x + 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return np.concatenate([key, strip], 0)


def _hold(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    valid = np.flatnonzero(np.isfinite(q))
    if not len(valid):
        return np.zeros_like(q)
    q[: valid[0]] = q[valid[0]]
    for i in range(1, len(q)):
        if not np.isfinite(q[i]):
            q[i] = q[i - 1]
    return q


def _hold_poses(poses: np.ndarray) -> np.ndarray:
    poses = poses.copy()
    bad = np.isnan(poses).any(axis=(1, 2))
    valid = np.flatnonzero(~bad)
    if not len(valid):
        return np.tile(np.eye(3), (len(poses), 1, 1))
    poses[: valid[0]] = poses[valid[0]]
    for i in range(1, len(poses)):
        if bad[i]:
            poses[i] = poses[i - 1]
    return poses


def _read_frames(video: Path, frames: list[int], size: tuple[int, int]) -> dict[int, np.ndarray]:
    """Decode in order once (seeking is unreliable on variable-frame-rate phone video)."""
    want, out = set(frames), {}
    cap = cv2.VideoCapture(str(video))
    t = 0
    while want - out.keys():
        ok = cap.grab()
        if not ok:
            break
        if t in want:
            _, img = cap.retrieve()
            out[t] = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        t += 1
    cap.release()
    return out


# ----------------------------------------------------------------- three views

def design_triangles(track_dir: Path, design: dict) -> dict[str, np.ndarray]:
    """The design's triangles per link in the model frame at the reference pose (metres)."""
    model = _load(track_dir, measured_xml(track_dir))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = {}
    for d in design["links"]:
        b = model.body(d["link"]).id
        R, x = data.xmat[b].reshape(3, 3), data.xpos[b]
        tris = []
        for p in d["primitives"]:
            v, f = primitive_mesh(p)
            tris.append(((v / 1000) @ R.T + x)[f])
        out[d["link"]] = np.concatenate(tris)
    return out


def three_views(track_dir: Path, design: dict | None, links: list[str], panel: int = 400) -> np.ndarray:
    """Front / side / top orthographic views of the model at the reference pose, with a mm grid.

    Measured meshes are filled in pale link colours; the design, if any, is outlined in strong ones.
    """
    model = _load(track_dir, measured_xml(track_dir))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    measured = link_geometry(model, data, 3)
    designed = design_triangles(track_dir, design) if design else {}
    allpts = np.concatenate([t.reshape(-1, 3) for t in list(measured.values()) + list(designed.values())])
    lo, hi = np.percentile(allpts, 0.5, 0), np.percentile(allpts, 99.5, 0)
    span = float((hi - lo).max()) * 1.2 + 1e-3
    mid = (lo + hi) / 2
    px = (panel - 40) / span
    step = next(s for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0) if span / s <= 10)
    views = [("FRONT  x right, z up", 0, 2), ("SIDE  y right, z up", 1, 2), ("TOP  x right, y up", 0, 1)]
    panels = []
    for title, a, b in views:
        img = np.full((panel, panel, 3), 245, np.uint8)

        def to_px(p):
            return np.stack([(p[..., a] - mid[a]) * px + panel / 2, panel / 2 - (p[..., b] - mid[b]) * px], -1)

        for k in range(-20, 21):
            for axis in (a, b):
                v = (math.floor(mid[axis] / step) + k) * step
                q = (v - mid[axis]) * px
                if abs(q) > panel / 2:
                    continue
                if axis == a:
                    x = int(panel / 2 + q)
                    cv2.line(img, (x, 0), (x, panel), (215, 215, 215), 1)
                    cv2.putText(img, f"{v * 1000:.0f}", (x + 2, panel - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1)
                else:
                    y = int(panel / 2 - q)
                    cv2.line(img, (0, y), (panel, y), (215, 215, 215), 1)
                    cv2.putText(img, f"{v * 1000:.0f}", (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1)
        for k, link in enumerate(links):
            color = np.array(LINK_COLORS[k % len(LINK_COLORS)], float)
            if link in measured:
                mask = np.zeros((panel, panel), np.uint8)
                for tri in to_px(measured[link]).astype(np.int32):
                    cv2.fillConvexPoly(mask, tri, 1)
                img[mask > 0] = (0.55 * img[mask > 0] + 0.45 * (0.5 * color + 127)).astype(np.uint8)
        for k, link in enumerate(links):
            color = LINK_COLORS[k % len(LINK_COLORS)]
            if link in designed:
                mask = np.zeros((panel, panel), np.uint8)
                for tri in to_px(designed[link]).astype(np.int32):
                    cv2.fillConvexPoly(mask, tri, 1)
                cs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, tuple(int(c * 0.8) for c in color), 2)
        for k, link in enumerate(links):
            bid = model.body(link).id
            if model.body_parentid[bid]:
                p = to_px(data.xpos[bid]).astype(int)
                cv2.circle(img, tuple(p), 5, (0, 0, 0), 2)
                cv2.putText(img, f"{link} joint", (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        cv2.putText(img, title + "  (mm)", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        panels.append(img)
    out = np.concatenate(panels, 1)
    note = "pale fill = measured mesh" + ("; outline = design" if design else "")
    cv2.putText(out, note, (8, panel - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
    return out


def bbox_check(track_dir: Path, design: dict, ev: dict) -> dict:
    """Per link and axis: the design's extent minus the measured one (mm), in the link frame."""
    out = {}
    measured = {l["link"]: l.get("measured_bbox_mm") for l in ev["links"]}
    for d in design["links"]:
        m = measured.get(d["link"])
        if not m:
            continue
        v = np.concatenate([primitive_mesh(p)[0] for p in d["primitives"]])
        size = v.max(0) - v.min(0)
        out[d["link"]] = {ax: {"design": round(float(size[i]), 1), "measured": m["size"][i],  # None if unmeasured
                               "unmeasured": ax in ev["unmeasured_axes"]} for i, ax in enumerate("xyz")}
    return out
