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

FACES = 60000  # per link, after decimation
GROUP_MEASURED = 3  # MJCF geom group for the measured shapes once a realistic visual exists (hidden by default)


def links_of(track_dir: Path) -> dict[str, dict]:
    """link -> {"part": tracked part(s) to show SAM 3D, "joint": (type, axis in model, q per frame) or None}."""
    if (track_dir / "slides.json").exists():  # drawers: cabinet = its fixed parts, each drawer = its front
        spec = json.loads((track_dir / "drawers.json").read_text())
        slides = json.loads((track_dir / "slides.json").read_text())
        out = {"base": {"part": spec["base"], "joint": None}}
        for i, (j, d) in enumerate(zip(slides["joints"], slides["drawers"])):
            q = np.array([np.nan if v is None else v for v in d["q"]])
            out[f"drawer{i + 1}"] = {"part": [d["part"]], "joint": ("slide", np.array([1.0, 0, 0]), q)}
        return out
    roles = json.loads((track_dir / "roles.json").read_text())
    hinge = json.loads((track_dir / "hinge.json").read_text())
    q = np.array([np.nan if v is None else v for v in hinge["q"]])
    return {"base": {"part": [roles["base"]], "joint": None},
            "lid": {"part": [roles["plate"]], "joint": ("hinge", np.array([1.0, 0, 0]), q)}}


def model_to_link(joint, t: int) -> np.ndarray:
    """4x4 taking model-frame points of a link seen on frame t to its own frame (joint at 0)."""
    m = np.eye(4)
    if joint is None:
        return m
    kind, axis, q = joint
    v = 0.0 if np.isnan(q[t]) else float(q[t])
    if kind == "slide":
        m[:3, 3] = -v * axis
    else:  # undo a turn of v about the axis through the origin (the hinge)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        m[:3, :3] = (np.eye(3) + np.sin(-v) * k + (1 - np.cos(-v)) * k @ k)
    return m


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


def write_obj(path: Path, v, f, rgb):
    with open(path, "w") as fh:
        fh.write("# SAM 3D Objects mesh, link frame, metres\n")
        np.savetxt(fh, np.column_stack([v, rgb / 255.0]), fmt="v %.6f %.6f %.6f %.4f %.4f %.4f")
        np.savetxt(fh, f + 1, fmt="f %d %d %d")


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
        for body in root.iter("body"):
            if body.get("name") not in meshes:
                continue
            for g in body.findall("geom"):
                if g.get("mesh") == f"{body.get('name')}_sam3d":
                    body.remove(g)
            for g in body.findall("geom"):
                g.set("group", str(GROUP_MEASURED))
            rgb = meshes[body.get("name")][1]
            geom = ET.Element("geom", type="mesh", mesh=f"{body.get('name')}_sam3d", contype="0", conaffinity="0",
                              mass="0", rgba=" ".join(f"{x:.3f}" for x in rgb) + " 1")
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
    ap.add_argument("--no-pointmap", action="store_true", help="let SAM 3D guess depth (MoGe) instead of the LiDAR's")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run = args.track_dir
    name = run.name
    video = args.video or ROOT / "data" / "real" / name / f"{name}.mp4"
    capture = args.capture or ROOT / "data" / "real" / name
    m3 = json.loads((run / "model3d.json").read_text())
    ctm = m3["camera_to_model"]
    rot, point = np.array(ctm["rotation"]), np.array(ctm["hinge_point"])

    c = lidar.open_capture(video, capture, run)
    if c.ref != m3["reference_frame"]:
        c.set_ref(m3["reference_frame"])
    floor = (json.loads((run / "roles.json").read_text()).get("floor") if (run / "roles.json").exists()
             else json.loads((run / "drawers.json").read_text()).get("floor"))
    if c.moving_camera and floor:
        c.level_with(floor)  # as the fit did, so frame poses match the model's
    links = links_of(run)
    frames = {k: int(v) for k, v in (s.split("=") for s in args.frame)}

    import modal
    sam3d = modal.Cls.from_name("video2urdf-sam3d-objects", "Sam3D")()

    report, meshes = {}, {}
    for link in args.links or list(links):
        spec = links[link]
        parts = spec["part"]
        part = parts[0] if len(parts) == 1 else c.merge(f"_{link}", parts)
        t = frames.get(link, c.ref if c.has(part, c.ref) else max(c.by[part], key=lambda k: c.by[part][k].area))
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
        inv = np.linalg.inv(c.world[t])  # frame-t camera -> reference camera
        v_ref = v_cam @ inv[:3, :3].T + inv[:3, 3]
        v_model = (v_ref - point) @ rot.T
        m2l = model_to_link(spec["joint"], t)
        v_link = v_model @ m2l[:3, :3].T + m2l[:3, 3]

        # How far the part's measured depth points lie from the generated surface (and back).
        meas = c.cloud(part, t)
        meas = ((meas - point) @ rot.T) @ m2l[:3, :3].T + m2l[:3, 3]
        surf = surface_samples(v_link, out["faces"])
        d_meas, _ = cKDTree(surf).query(meas)
        d_mesh, _ = cKDTree(meas).query(surf)
        ext = np.ptp(v_link, axis=0) * 1000
        report[link] = {
            "frame": int(t), "part": part, "faces": int(len(out["faces"])), "pointmap": pm is not None,
            "layout_iou": out["layout_iou"], "extent_mm": [round(float(e), 1) for e in ext],
            "measured_to_mesh_mm": {"median": float(np.median(d_meas) * 1000), "p90": float(np.percentile(d_meas, 90) * 1000)},
            "mesh_to_measured_mm_median": float(np.median(d_mesh) * 1000), "measured_points": int(len(meas))}
        print(f"  measured points to the mesh: median {report[link]['measured_to_mesh_mm']['median']:.1f} mm, "
              f"p90 {report[link]['measured_to_mesh_mm']['p90']:.1f} mm; extent {ext.round(0).tolist()} mm", flush=True)

        mesh_dir = run / "meshes"
        write_obj(mesh_dir / f"{link}_sam3d.obj", v_link, out["faces"], out["colors"])
        write_ply(mesh_dir / f"{link}_sam3d.ply", v_link, out["faces"], out["colors"])
        meshes[link] = (f"meshes/{link}_sam3d.obj", (np.median(out["colors"], axis=0) / 255).tolist())

    add_visuals(run, meshes)
    old = json.loads((run / "sam3d.json").read_text()) if (run / "sam3d.json").exists() else {}
    (run / "sam3d.json").write_text(json.dumps({**old, **report}, indent=2))


if __name__ == "__main__":
    main()
