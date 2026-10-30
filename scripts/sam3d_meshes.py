"""Realistic link meshes from SAM 3D Objects, placed and checked against the LiDAR measurement.

For each link of a fitted LiDAR model: one frame, the link's SAM 3.1 mask and that frame's
LiDAR depth as the pointmap go to SAM 3D (modal_apps/sam3d.py), which returns a
coloured mesh in the camera frame, in metres. It is moved into the link's frame (the model's
camera_to_model, with the joint's value on that frame undone) and scored against the part's
own depth points: how far each measured point lies from the generated surface. The mesh is
added to the model as the link's visual; the measured geometry stays as its collision.

Usage: .venv/bin/python scripts/sam3d_meshes.py data/tracks_agent/laptop_lidar
           [--video data/real/laptop_lidar/laptop_lidar.mp4] [--capture data/real/laptop_lidar]
           [--links base lid] [--frame lid=850] [--no-pointmap]
Needs: modal deploy modal_apps/sam3d.py, and its weights (see that file).
"""

import argparse
import io
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf import lidar  # noqa: E402
from video2urdf.lift3d import depth_edges  # noqa: E402
from video2urdf.pipeline import read_frame  # noqa: E402
from video2urdf.texture import read_stl  # noqa: E402

FACES = 60000  # per link, after decimation
GROUP_MEASURED = 3  # MJCF geom group for the measured shapes once a realistic visual exists (hidden by default)


def quality_passes(median_mm: float, p90_mm: float, extent_mm: np.ndarray, fitted_mm: np.ndarray,
                   max_median_mm: float = 5, max_p90_mm: float = 15,
                   part_override: bool = False) -> bool:
    if part_override:
        return False  # alternate masks may cover multiple rigid links
    values = np.r_[median_mm, p90_mm, extent_mm, fitted_mm]
    if not np.isfinite(values).all() or np.any(fitted_mm <= 0):
        return False
    ratio = extent_mm / fitted_mm
    return bool(median_mm <= max_median_mm and p90_mm <= max_p90_mm
                and np.all((ratio >= .8) & (ratio <= 1.2)))


def fitted_extent(track_dir: Path, link: str) -> np.ndarray:
    """Axis-aligned extent of collision meshes in the link frame, in millimetres."""
    root = ET.parse(track_dir / "model.xml").getroot()
    assets = {m.get("name"): m.get("file", "").replace("_tex.obj", ".stl") for m in root.findall("./asset/mesh")}
    body = next((b for b in root.iter("body") if b.get("name") == link), None)
    if body is None:
        raise ValueError(f"no body {link}")
    points = []
    for geom in body.findall("geom"):
        if geom.get("contype") == "0":
            continue
        filename = assets.get(geom.get("mesh"), "")
        if filename.endswith(".stl") and (track_dir / filename).exists():
            vertices = read_stl(track_dir / filename).reshape(-1, 3)
            pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            points.append(vertices + pos)
    if not points:
        raise ValueError(f"no fitted collision meshes for {link}")
    return np.ptp(np.concatenate(points), axis=0) * 1000


def links_of(track_dir: Path) -> dict[str, dict]:
    """link -> {"part": tracked part(s) to show SAM 3D, "joint": (type, axis in model, q per frame) or None}."""
    if not (track_dir / "roles.json").exists() and not (track_dir / "slides.json").exists():
        from video2urdf.texture import link_masks  # a static object or a rod chain: poses come from the replay
        return {link: {"part": parts, "joint": None} for link, parts in link_masks(track_dir).items()}
    if (track_dir / "slides.json").exists():  # drawers: cabinet = its fixed parts, each drawer = its front
        spec = json.loads((track_dir / "drawers.json").read_text())
        slides = json.loads((track_dir / "slides.json").read_text())
        out = {"base": {"part": spec["base"], "joint": None}}
        for i, (j, d) in enumerate(zip(slides["joints"], slides["drawers"])):
            q = np.array([np.nan if v is None else v for v in d["q"]])
            out[f"drawer{i + 1}"] = {"part": [d["part"]], "joint": ("slide", np.array([1.0, 0, 0]), q)}
        return out
    roles = json.loads((track_dir / "roles.json").read_text())
    if (track_dir / "slide.json").exists():
        slide = json.loads((track_dir / "slide.json").read_text())
        q = np.array([np.nan if v is None else v for v in slide["q"]])
        return {"base": {"part": [roles["base"]], "joint": None},
                "drawer": {"part": [roles["plate"]], "joint": ("slide", np.array([1., 0, 0]), q)}}
    hinge = json.loads((track_dir / "hinge.json").read_text())
    q = np.array([np.nan if v is None else v for v in hinge["q"]])
    return {"base": {"part": [roles["base"]], "joint": None},
            "lid": {"part": [roles["plate"]], "joint": ("hinge", np.array([1.0, 0, 0]), q)}}


class ReplayPoses:
    """Frame-t camera coordinates -> a link's frame, from the replay's camera poses and joint values
    (any fitted model: a static object, a rod chain)."""

    def __init__(self, track_dir: Path):
        import mujoco
        self.mujoco = mujoco
        self.rec = np.load(track_dir / "replay3d.npz")
        self.model = mujoco.MjModel.from_xml_path(str(track_dir / "model.xml"))
        self.data = mujoco.MjData(self.model)

    def camera_to_link(self, link: str, t: int) -> np.ndarray:
        for key in self.rec.files:
            if key.startswith("q_"):
                self.data.joint(key[2:]).qpos[0] = self.rec[key][t]
        self.mujoco.mj_forward(self.model, self.data)
        b = self.model.body(link).id
        world_to_link = np.eye(4)
        r = self.data.xmat[b].reshape(3, 3)
        world_to_link[:3, :3], world_to_link[:3, 3] = r.T, -r.T @ self.data.xpos[b]
        return world_to_link @ self.rec["cam_to_world"][t]


def video_score(run: Path, video: Path, link: str, mesh_file: str, max_drop: float = 0.10) -> dict:
    """Mask F-score over sampled frames of the measured link and of the SAM 3D mesh in its place."""
    from video2urdf import design as D
    scene = D.Scene(run, video)
    measured_xml = D.measured_xml(run)
    root = ET.fromstring(measured_xml)
    ET.SubElement(root.find("asset"), "mesh", name="sam3d_check", file=mesh_file)
    body = next(b for b in root.iter("body") if b.get("name") == link)
    body.append(ET.Element("geom", type="mesh", mesh="sam3d_check", contype="0", conaffinity="0", group="0"))
    before = scene.score(measured_xml, 3)[link]["f_near"]
    after = scene.score(ET.tostring(root, encoding="unicode"), 0)[link]["f_near"]
    return {"measured": before, "sam3d": after, "max_drop": max_drop,
            "passed": bool(before is not None and after is not None and after >= before - max_drop)}


def model_to_link(joint, t: int) -> np.ndarray:
    """4x4 taking model-frame points of a link seen on frame t to its own frame (joint at 0)."""
    m = np.eye(4)
    if joint is None:
        return m
    kind, axis, q = joint
    if t < 0 or t >= len(q) or not np.isfinite(q[t]):
        raise ValueError(f"joint position is unavailable at frame {t}; choose a frame with measured motion")
    v = float(q[t])
    if kind == "slide":
        m[:3, 3] = -v * axis
    else:  # undo a turn of v about the axis through the origin (the hinge)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        m[:3, :3] = (np.eye(3) + np.sin(-v) * k + (1 - np.cos(-v)) * k @ k)
    return m


def select_frame(c, part: str, joint, override: int | None = None) -> int:
    """Select a visible frame whose link pose is known, or reject an invalid override."""
    def usable(t):
        return (0 <= t < c.n and c.has(part, t) and
                (joint is None or (t < len(joint[2]) and np.isfinite(joint[2][t]))))
    if override is not None:
        if not usable(override):
            raise ValueError(f"{part} or its joint position is unavailable on frame {override}; "
                             "choose a frame where both are measured")
        return override
    if usable(c.ref):
        return c.ref
    candidates = [t for t in c.by[part] if usable(t)]
    if not candidates:
        raise ValueError(f"{part} has no frame with both a visible mask and a measured joint position")
    return max(candidates, key=lambda t: c.by[part][t].area)


def retry_frames(c, part, joint, first, count, open_indicator=None):
    """Select usable, temporally diverse frames; require an open indicator if available."""
    candidates = [t for t, det in c.by[part].items() if 0 <= t < c.n and c.has(part, t)
                  and (open_indicator is None or c.has(open_indicator, t))
                  and (joint is None or (t < len(joint[2]) and np.isfinite(joint[2][t])))]
    if not candidates or count <= 0:
        return []
    largest = max(c.by[part][t].area for t in candidates)
    candidates = [t for t in candidates if c.by[part][t].area >= .4 * largest and t != first]
    selected = []
    while candidates and len(selected) < count:
        anchors = [first] + selected
        # Avoid defaulting to the very first/last frames, where recording and
        # manipulation often begin or end, while still sampling a different period.
        if not selected:
            target = .2 * (c.n - 1) if first >= c.n / 2 else .8 * (c.n - 1)
            t = min(candidates, key=lambda x: (abs(x - target), -c.by[part][x].area))
        else:
            t = max(candidates, key=lambda x: (min(abs(x - a) for a in anchors), c.by[part][x].area, -x))
        selected.append(t)
        candidates.remove(t)
    return selected


def pointmap(c: lidar.Capture, t: int) -> np.ndarray:
    """Frame t's depth as an HxWx3 point map (depth resolution, metres, camera frame), NaN where unknown."""
    d = c.cap.depth[t].astype(np.float64)
    dh, dw = d.shape
    vv, uu = np.mgrid[0:dh, 0:dw]
    uv = np.stack([(uu.ravel() + 0.5) * c.w / dw, (vv.ravel() + 0.5) * c.h / dh], axis=1)
    pts = c.cap.intrinsics.lift(uv, d.ravel()).reshape(dh, dw, 3)
    pts[(d <= 0) | depth_edges(c.cap.depth[t])] = np.nan
    return pts.astype(np.float32)


def surface_samples(v: np.ndarray, f: np.ndarray, n: int = 300000, rng=np.random.default_rng(0)) -> np.ndarray:
    a, b, cc = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, cc - a), axis=1)
    i = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return (1 - s)[:, None] * a[i] + (s * (1 - r2))[:, None] * b[i] + (s * r2)[:, None] * cc[i]


def align_to_depth(vertices: np.ndarray, faces: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, dict]:
    """Conservatively register a generated visual to depth, without changing its scale.

    A single visible plane cannot constrain a whole 3D shape, so this correction is bounded
    and followed by independent depth and size checks. It does not validate hidden surfaces.
    """
    rng = np.random.default_rng(7)
    surface = surface_samples(vertices, faces, n=60000, rng=rng)
    target = observed[rng.choice(len(observed), min(5000, len(observed)), replace=False)]
    rot_total, trans_total = np.eye(3), np.zeros(3)
    before = cKDTree(surface).query(target)[0]
    best = (float(np.median(before)), surface.copy(), rot_total.copy(), trans_total.copy())
    for _ in range(25):
        distances, idx = cKDTree(surface).query(target)
        keep = distances <= min(float(np.percentile(distances, 85)), 0.10)
        if keep.sum() < 100:
            break
        moving, fixed = surface[idx[keep]], target[keep]
        cm, cf = moving.mean(axis=0), fixed.mean(axis=0)
        u, _, vt = np.linalg.svd((moving - cm).T @ (fixed - cf))
        rot = vt.T @ u.T
        if np.linalg.det(rot) < 0:
            vt[-1] *= -1
            rot = vt.T @ u.T
        trans = cf - rot @ cm
        new_rot, new_trans = rot @ rot_total, rot @ trans_total + trans
        angle = np.degrees(np.arccos(np.clip((np.trace(new_rot) - 1) / 2, -1, 1)))
        # Bound the motion around the mesh centre, rather than the arbitrary link origin.
        center = vertices.mean(axis=0)
        displacement = np.linalg.norm(new_rot @ center + new_trans - center)
        if angle > 20 or displacement > .03:
            break
        surface = surface @ rot.T + trans
        rot_total, trans_total = new_rot, new_trans
        after = cKDTree(surface).query(target)[0]
        score = float(np.median(after))
        if score < best[0] and np.percentile(after, 90) <= 1.1 * np.percentile(before, 90):
            best = (score, surface.copy(), rot_total.copy(), trans_total.copy())
    _, _, rot, trans = best
    angle = np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
    center = vertices.mean(axis=0)
    info = {"method": "bounded rigid ICP to observed depth; no scale change", "rotation_deg": float(angle),
            "center_displacement_mm": float(1000 * np.linalg.norm(rot @ center + trans - center)),
            "rotation_matrix": rot.tolist(), "translation_m": trans.tolist(),
            "before_median_mm": float(1000 * np.median(before)), "after_median_mm": float(1000 * best[0])}
    return vertices @ rot.T + trans, info


def write_obj(path: Path, v, f, rgb):
    """Bake vertex colours into a tiny triangle atlas usable by MuJoCo and OBJ renderers."""
    tile, columns = 6, 256
    rows = max(1, int(np.ceil(len(f) / columns)))
    width, height = columns * tile, rows * tile
    atlas = np.zeros((height, width, 3), np.uint8)
    corners = np.array([[1.0, 1.0], [tile - 2.0, 1.0], [1.0, tile - 2.0]])
    yy, xx = np.mgrid[0:tile, 0:tile]
    u = np.maximum((xx - corners[0, 0]) / (tile - 3), 0)
    w = np.maximum((yy - corners[0, 1]) / (tile - 3), 0)
    weights = np.stack([np.maximum(1 - u - w, 0), u, w], axis=-1)
    weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)
    uv = np.empty((len(f), 3, 2), float)
    for start in range(0, len(f), 2048):
        ids = np.arange(start, min(start + 2048, len(f)))
        ox, oy = (ids % columns) * tile, (ids // columns) * tile
        colors = np.asarray(rgb)[f[ids], :3]
        patches = np.einsum("hwk,nkc->nhwc", weights, colors).clip(0, 255).astype(np.uint8)
        atlas[oy[:, None, None] + yy, ox[:, None, None] + xx] = patches[..., ::-1]
        uv[ids, :, 0] = (ox[:, None] + corners[:, 0] + .5) / width
        uv[ids, :, 1] = 1 - (oy[:, None] + corners[:, 1] + .5) / height
    png, mtl = path.with_suffix(".png"), path.with_suffix(".mtl")
    cv2.imwrite(str(png), atlas)
    mtl.write_text(f"newmtl generated\nKd 1 1 1\nmap_Kd {png.name}\n")
    with open(path, "w") as fh:
        fh.write(f"# SAM 3D Objects mesh, link frame, metres\nmtllib {mtl.name}\nusemtl generated\n")
        np.savetxt(fh, v, fmt="v %.6f %.6f %.6f")
        np.savetxt(fh, uv.reshape(-1, 2), fmt="vt %.6f %.6f")
        vt = np.arange(len(f) * 3).reshape(-1, 3) + 1
        indices = np.stack([f + 1, vt], axis=-1).reshape(-1, 6)
        np.savetxt(fh, indices, fmt="f %d/%d %d/%d %d/%d")


def write_ply(path: Path, v, f, rgb):
    """Binary PLY with vertex colours (three.js PLYLoader reads it)."""
    vert = np.zeros(len(v), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vert["x"], vert["y"], vert["z"] = v.T
    vert["red"], vert["green"], vert["blue"] = rgb.T
    face = np.zeros(len(f), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    face["n"], face["i"] = 3, f
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\n"
            f"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n")
    path.write_bytes(head.encode() + vert.tobytes() + face.tobytes())


def add_visuals(track_dir: Path, meshes: dict[str, tuple[str, list[float]]]):
    """The realistic mesh becomes each link's visual: in MJCF an extra visual-only geom (the measured
    geoms move to group GROUP_MEASURED, still colliding), in URDF the <visual> (collision stays
    measured). Applied to model.nophysics.* too, which physics.apply restores from."""
    for name in ("model.xml", "model.nophysics.xml"):
        path = track_dir / name
        if not path.exists():
            continue
        tree = ET.parse(path)
        root = tree.getroot()
        asset = root.find("asset")
        for el in list(asset):  # a re-run replaces the meshes of the links it made
            if el.get("name") in {f"{link}_sam3d" for link in meshes}:
                asset.remove(el)
        for link, (mesh, _) in meshes.items():
            ET.SubElement(asset, "mesh", name=f"{link}_sam3d", file=mesh, inertia="convex")
            texture_name = f"{link}_sam3d_texture"
            for el in list(asset):
                if el.get("name") in (texture_name, f"{link}_sam3d_material"):
                    asset.remove(el)
            png = str(Path(mesh).with_suffix(".png"))
            if (track_dir / png).exists():
                ET.SubElement(asset, "texture", name=texture_name, type="2d", file=png)
                ET.SubElement(asset, "material", name=f"{link}_sam3d_material", texture=texture_name)
        for body in root.iter("body"):
            if body.get("name") not in meshes:
                continue
            for g in body.findall("geom"):
                if (g.get("mesh") == f"{body.get('name')}_sam3d" or g.get("mesh", "").endswith("_infer")
                        or g.get("name", "").startswith("inferred_")):
                    body.remove(g)
            for g in body.findall("geom"):  # Spark's primitive design gives way to the realistic mesh
                g.set("group", "4" if g.get("name", "").startswith("design_") else str(GROUP_MEASURED))
            rgb = meshes[body.get("name")][1]
            geom = ET.Element("geom", type="mesh", mesh=f"{body.get('name')}_sam3d", contype="0", conaffinity="0",
                              mass="0", rgba=" ".join(f"{x:.3f}" for x in rgb) + " 1")
            if (track_dir / Path(meshes[body.get("name")][0]).with_suffix(".png")).exists():
                geom.set("material", f"{body.get('name')}_sam3d_material")
                geom.set("rgba", "1 1 1 1")
            body.insert(len([e for e in body if e.tag in ("joint", "inertial")]), geom)
        ET.indent(tree, "  ")
        tree.write(path)
    for name in ("model.urdf", "model.nophysics.urdf"):
        path = track_dir / name
        if not path.exists():
            continue
        tree = ET.parse(path)
        for link in tree.getroot().iter("link"):
            if link.get("name") not in meshes:
                continue
            for v in link.findall("visual"):
                link.remove(v)
            vis = ET.Element("visual", name=f"{link.get('name')}_sam3d")
            ET.SubElement(ET.SubElement(vis, "geometry"), "mesh", filename=meshes[link.get("name")][0])
            mat = ET.SubElement(vis, "material", name=f"{link.get('name')}_sam3d_color")
            png = str(Path(meshes[link.get("name")][0]).with_suffix(".png"))
            if (track_dir / png).exists():
                ET.SubElement(mat, "texture", filename=png)
                ET.SubElement(mat, "color", rgba="1 1 1 1")
            else:
                ET.SubElement(mat, "color", rgba=" ".join(f"{x:.3f}" for x in meshes[link.get("name")][1]) + " 1")
            link.insert(len(link.findall("inertial")), vis)
        ET.indent(tree, "  ")
        tree.write(path, xml_declaration=True, encoding="unicode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track_dir", type=Path)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--capture", type=Path)
    ap.add_argument("--links", nargs="*")
    ap.add_argument("--frame", nargs="*", default=[], help="link=frame overrides (default: the reference frame)")
    ap.add_argument("--part", nargs="*", default=[], help="link=tracked-part overrides for alternative views")
    ap.add_argument("--no-pointmap", action="store_true", help="use SAM 3D's non-Meta MoGe depth model; "
                    "not suitable where all runtime models must be Meta models")
    ap.add_argument("--video-texture", action="store_true", help="experimentally project recorded video textures onto accepted visuals")
    ap.add_argument("--retries", type=int, default=0, help="try up to this many additional frames when a candidate fails")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-median-mm", type=float, default=5,
                    help="do not activate a mesh if measured points are farther away (default: 5 mm)")
    ap.add_argument("--max-p90-mm", type=float, default=15,
                    help="do not activate a mesh if the 90th percentile exceeds this (default: 15 mm)")
    args = ap.parse_args()
    if args.retries < 0 or args.retries > 4:
        ap.error("--retries must be between 0 and 4")
    if (not np.isfinite([args.max_median_mm, args.max_p90_mm]).all()
            or args.max_median_mm <= 0 or args.max_p90_mm <= 0):
        ap.error("quality thresholds must be positive finite numbers")
    run = args.track_dir
    name = run.name
    video = args.video or ROOT / "data" / "real" / name / f"{name}.mp4"
    capture = args.capture or ROOT / "data" / "real" / name
    m3 = json.loads((run / "model3d.json").read_text())
    ctm = m3.get("camera_to_model", {})
    replay = None if "hinge_point" in ctm else ReplayPoses(run)
    if replay is None:
        rot, point = np.array(ctm["rotation"]), np.array(ctm["hinge_point"])

    c = lidar.open_capture(video, capture, run)
    if c.ref != m3["reference_frame"]:
        c.set_ref(m3["reference_frame"])
    floor = (json.loads((run / "roles.json").read_text()).get("floor") if (run / "roles.json").exists()
             else json.loads((run / "drawers.json").read_text()).get("floor") if (run / "drawers.json").exists()
             else None)  # replay poses already include the fit's levelling
    if c.moving_camera and floor:
        c.level_with(floor)  # as the fit did, so frame poses match the model's
    elif not c.moving_camera and (run / "roles.json").exists():
        # Fixed-camera RGB-D captures can still show the object moving. Recreate the
        # same base-following transform used when fit_lidar built the model.
        roles = json.loads((run / "roles.json").read_text())
        c.follow(roles["base"], exclude=(roles["plate"],))
    links = links_of(run)
    frames = {k: int(v) for k, v in (s.split("=") for s in args.frame)}
    part_overrides = set()
    for item in args.part:
        link, part = item.split("=", 1)
        if link not in links or part not in c.by:
            raise ValueError(f"unknown link or tracked part in --part {item}")
        links[link]["part"] = [part]
        part_overrides.add(link)

    import modal
    sam3d = modal.Cls.from_name("video2urdf-sam3d-objects", "Sam3D")()

    report = json.loads((run / "sam3d.json").read_text()) if (run / "sam3d.json").exists() else {}
    for link in args.links or list(links):
        spec = links[link]
        parts = spec["part"]
        part = parts[0] if len(parts) == 1 else c.merge(f"_{link}", parts)
        t = select_frame(c, part, spec["joint"], frames.get(link))
        rgb = cv2.cvtColor(read_frame(video, t), cv2.COLOR_BGR2RGB)
        mask = c.by[part][t].full_mask(c.h, c.w)
        png = lambda a: cv2.imencode(".png", a)[1].tobytes()  # noqa: E731
        pm = None
        if not args.no_pointmap:
            buf = io.BytesIO()
            np.save(buf, pointmap(c, t))
            pm = buf.getvalue()
        print(f"{link}: SAM 3D on frame {t} ({part}, mask {100 * mask.mean():.1f}% of the frame)", flush=True)
        out = sam3d.reconstruct.remote(png(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), png(mask.astype(np.uint8) * 255),
                                       pm, args.seed, FACES)
        v_cam = out["vertices_camera"].astype(np.float64)
        if replay is not None:
            from video2urdf.lift3d import mask_cloud
            c2l = replay.camera_to_link(link, t)
            v_link = v_cam @ c2l[:3, :3].T + c2l[:3, 3]
            meas = mask_cloud(mask, c.cap.depth[t], c.cap.intrinsics) @ c2l[:3, :3].T + c2l[:3, 3]
        else:
            inv = np.linalg.inv(c.world[t])  # frame-t camera -> reference camera
            v_ref = v_cam @ inv[:3, :3].T + inv[:3, 3]
            v_model = (v_ref - point) @ rot.T
            m2l = model_to_link(spec["joint"], t)
            v_link = v_model @ m2l[:3, :3].T + m2l[:3, 3]

            # How far the part's measured depth points lie from the generated surface (and back).
            meas = c.cloud(part, t)
            meas = ((meas - point) @ rot.T) @ m2l[:3, :3].T + m2l[:3, 3]
        if len(meas) < 100:
            raise ValueError(f"{link} frame {t} has only {len(meas)} depth points; choose another frame")
        if not len(out["faces"]) or not np.isfinite(v_link).all():
            raise ValueError(f"{link}: SAM 3D returned an empty or non-finite mesh")
        v_link, alignment = align_to_depth(v_link, out["faces"], meas)
        surf = surface_samples(v_link, out["faces"])
        d_meas, _ = cKDTree(surf).query(meas)
        d_mesh, _ = cKDTree(meas).query(surf)
        ext = np.ptp(v_link, axis=0) * 1000
        report[link] = {
            "frame": int(t), "part": part, "faces": int(len(out["faces"])), "pointmap": pm is not None,
            "layout_iou": out["layout_iou"], "extent_mm": [round(float(e), 1) for e in ext],
            "measured_to_mesh_mm": {"median": float(np.median(d_meas) * 1000), "p90": float(np.percentile(d_meas, 90) * 1000)},
            "mesh_to_measured_mm_median": float(np.median(d_mesh) * 1000), "measured_points": int(len(meas))}
        report[link]["alignment"] = alignment
        report[link]["sam3d_layout"] = {key: out[key] for key in ("scale", "rotation", "translation")}
        fitted = fitted_extent(run, link)
        extent_ratio = ext / np.maximum(fitted, 1e-6)
        good = quality_passes(report[link]["measured_to_mesh_mm"]["median"],
                              report[link]["measured_to_mesh_mm"]["p90"], ext, fitted,
                              args.max_median_mm, args.max_p90_mm, part_override=link in part_overrides)
        # An alternate tracked mask can include neighbouring rigid links (e.g. the closed
        # laptop). It is useful for experimentation but cannot validate this link alone.
        if link in part_overrides:
            report[link]["part_override_warning"] = "Alternate mask may include other links; candidate is not auto-activated."
        if replay is not None:
            # A static object or a rod chain: its fitted extent is itself an envelope (a visual hull
            # or a rod), not a size to match. Judge the mesh as the shape design is judged: drawn over
            # the video from the recorded cameras and joint values, against the SAM masks.
            write_obj(mesh_dir_candidate := run / "meshes" / f"{link}_sam3d_candidate.obj", v_link, out["faces"], out["colors"])
            video_check = video_score(run, video, link, mesh_dir_candidate.relative_to(run).as_posix())
            report[link]["video_check"] = video_check
            depth_ok = (report[link]["measured_to_mesh_mm"]["median"] <= args.max_median_mm
                        and report[link]["measured_to_mesh_mm"]["p90"] <= args.max_p90_mm)
            good = bool(depth_ok and video_check["passed"] and link not in part_overrides)
            print(f"  over the video: near-F {video_check['sam3d']} vs measured {video_check['measured']}", flush=True)
        report[link]["accepted"] = bool(good)
        report[link]["acceptance_thresholds_mm"] = {"median": args.max_median_mm, "p90": args.max_p90_mm}
        report[link]["fitted_extent_mm"] = fitted.round(1).tolist()
        report[link]["extent_ratio"] = extent_ratio.round(3).tolist()
        report[link]["extent_ratio_allowed"] = [0.8, 1.2]
        print(f"  measured points to the mesh: median {report[link]['measured_to_mesh_mm']['median']:.1f} mm, "
              f"p90 {report[link]['measured_to_mesh_mm']['p90']:.1f} mm; extent {ext.round(0).tolist()} mm", flush=True)

        mesh_dir = run / "meshes"
        candidate_obj = mesh_dir / f"{link}_sam3d_candidate.obj"
        candidate_ply = mesh_dir / f"{link}_sam3d_candidate.ply"
        write_obj(candidate_obj, v_link, out["faces"], out["colors"])
        write_ply(candidate_ply, v_link, out["faces"], out["colors"])
        report[link]["candidate"] = f"meshes/{link}_sam3d_candidate.ply"
        if good:
            write_obj(mesh_dir / f"{link}_sam3d.obj", v_link, out["faces"], out["colors"])
            shutil.copyfile(candidate_ply, mesh_dir / f"{link}_sam3d.ply")
            add_visuals(run, {link: (f"meshes/{link}_sam3d.obj", (np.median(out["colors"], axis=0) / 255).tolist())})
            report[link]["active_visual"] = "new SAM 3D mesh"
            if args.video_texture:
                try:
                    from transfer_sam3d_texture import observed_views, project_views
                    views = observed_views(run, link)
                    colors, history = project_views(run, video, link, v_link, out["colors"], views)
                    write_obj(mesh_dir / f"{link}_sam3d.obj", v_link, out["faces"], colors)
                    write_ply(mesh_dir / f"{link}_sam3d.ply", v_link, out["faces"], colors)
                    report[link]["texture_experiment"] = {"manual": False, "views": history,
                        "projection_limit_mm": 25,
                        "note": "Visual interpolation, not a measurement; inspect all sides."}
                except (ValueError, OSError, KeyError, IndexError) as error:
                    report[link]["texture_experiment_error"] = str(error)
                    print(f"  Video texture transfer unavailable: {error}; SAM 3D texture retained.", flush=True)
        else:
            model = ET.parse(run / "model.xml").getroot()
            active = any(g.get("mesh") == f"{link}_sam3d" for g in model.iter("geom"))
            report[link]["active_visual"] = "previous SAM 3D mesh" if active else "fitted visual"
            print(f"  REJECTED: depth/extent checks failed; {report[link]['active_visual']} retained.", flush=True)
        (run / "sam3d.json").write_text(json.dumps(report, indent=2))
        from video2urdf.assessment import assess
        assess(run)

    # Retry only failed links; each subprocess persists its own report and retains
    # the previous accepted visual if a later candidate fails. Explicit frame/part
    # overrides are experimental and are not automatically expanded.
    attempts = {}
    if args.retries:
        for link in args.links or list(links):
            current = json.loads((run / "sam3d.json").read_text())[link]
            if current.get("accepted") or link in frames or link in part_overrides:
                continue
            spec = links[link]
            part = spec["part"][0] if len(spec["part"]) == 1 else f"_{link}"
            roles = json.loads((run / "roles.json").read_text()) if (run / "roles.json").exists() else {}
            indicator = roles.get("open_indicator") if link == "lid" else None
            chosen = retry_frames(c, part, spec["joint"], current["frame"], args.retries, indicator)
            attempts[link] = [{"frame": current["frame"], "accepted": False,
                               "metrics": current.get("measured_to_mesh_mm"),
                               "extent_mm": current.get("extent_mm"), "extent_ratio": current.get("extent_ratio")}]
            for index, frame in enumerate(chosen):
                command = [sys.executable, str(Path(__file__).resolve()), str(run), "--video", str(video),
                           "--capture", str(capture), "--links", link, "--frame", f"{link}={frame}",
                           "--seed", str(args.seed + index + 1), "--max-median-mm", str(args.max_median_mm),
                           "--max-p90-mm", str(args.max_p90_mm)]
                if args.no_pointmap:
                    command.append("--no-pointmap")
                if args.video_texture:
                    command.append("--video-texture")
                print(f"Retrying {link} on frame {frame}", flush=True)
                result = subprocess.run(command, check=False)
                current = json.loads((run / "sam3d.json").read_text())[link]
                attempts[link].append({"frame": frame, "exit_code": result.returncode,
                                       "accepted": bool(result.returncode == 0 and current.get("accepted")),
                                       "metrics": current.get("measured_to_mesh_mm") if result.returncode == 0 else None,
                                       "extent_mm": current.get("extent_mm") if result.returncode == 0 else None,
                                       "extent_ratio": current.get("extent_ratio") if result.returncode == 0 else None})
                (run / "sam3d_attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
                if attempts[link][-1]["accepted"]:
                    break
        if attempts:
            (run / "sam3d_attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")


if __name__ == "__main__":
    main()
