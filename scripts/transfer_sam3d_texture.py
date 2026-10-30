"""Experimentally project fitted video textures onto an accepted SAM 3D visual.

Creates a new result directory. Each --view specifies a textured OBJ in the source
result and a frame where its relevant face is visible. Inputs must share link coordinates.
This is a visual interpolation, not a measurement or an automatic quality guarantee.
"""
import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from video2urdf.pipeline import video_info  # noqa: E402
from video2urdf.texture_transfer import transfer_texture, visible_from_depth  # noqa: E402
from video2urdf.assessment import assess  # noqa: E402
from video2urdf.texture import planar_faces, read_stl  # noqa: E402
from sam3d_meshes import write_obj, write_ply  # noqa: E402


def read_colored_ply(path):
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii").splitlines()
    # Only our own binary writer is supported: avoid silently misreading arbitrary PLY.
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("expected binary little-endian SAM 3D PLY")
    n = int(next(x.split()[-1] for x in header if x.startswith("element vertex ")))
    m = int(next(x.split()[-1] for x in header if x.startswith("element face ")))
    vertex_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    if len(raw) != end + n * vertex_dtype.itemsize + m * face_dtype.itemsize:
        raise ValueError("unexpected PLY layout")
    vs = np.frombuffer(raw, vertex_dtype, n, end)
    fs = np.frombuffer(raw, face_dtype, m, end + n * vertex_dtype.itemsize)
    if not np.all(fs["count"] == 3):
        raise ValueError("PLY must contain triangles")
    return (np.stack([vs[k] for k in ("x", "y", "z")], axis=1),
            fs["indices"].copy(), np.stack([vs[k] for k in ("red", "green", "blue")], axis=1))


def read_video_obj(path):
    vertices, uvs, faces, uv_faces = [], [], [], []
    for line in path.read_text().splitlines():
        items = line.split()
        if not items:
            continue
        if items[0] == "v":
            vertices.append([float(x) for x in items[1:4]])
        elif items[0] == "vt":
            uvs.append([float(x) for x in items[1:3]])
        elif items[0] == "f":
            if len(items) != 4:
                raise ValueError("source OBJ must be triangulated")
            pairs = [entry.split("/") for entry in items[1:]]
            faces.append([int(x[0]) - 1 for x in pairs])
            uv_faces.append([int(x[1]) - 1 for x in pairs])
    atlas = cv2.imread(str(path.parent / "video_tex.png"))
    if atlas is None:
        raise ValueError(f"missing video_tex.png next to {path}")
    return vertices, uvs, faces, uv_faces, atlas


def observed_views(run, link):
    """Use only faces marked observed by the video-texturing stage, at their recorded frames."""
    textures = json.loads((run / "texture.json").read_text()).get("meshes", {})
    root = ET.parse(run / "model.xml").getroot()
    assets = {m.get("name"): m.get("file", "") for m in root.findall("./asset/mesh")}
    result = []
    for name, info in textures.items():
        if not name.startswith(link + "_") or not info.get("by_face"):
            continue
        source = assets.get(name, "")
        if not source.endswith("_tex.obj"):
            continue
        stl = run / source.replace("_tex.obj", ".stl")
        if not stl.exists():
            raise ValueError(f"source mesh missing for {name}: {stl}")
        groups = planar_faces(read_stl(stl))
        vertices, uv, faces, uv_faces, atlas = read_video_obj(run / source)
        if sum(map(len, groups)) != len(faces):
            raise ValueError(f"texture face layout does not match source mesh {name}")
        offsets = np.cumsum([0] + [len(group) for group in groups])
        by_frame = {}
        for face, detail in info["by_face"].items():
            index, frame = int(face), int(detail["frame"])
            if index < 0 or index >= len(groups) or detail["triangles"] != len(groups[index]):
                raise ValueError(f"invalid observed face record for {name}: {face}")
            by_frame.setdefault(frame, []).extend(range(offsets[index], offsets[index + 1]))
        for frame, indices in sorted(by_frame.items()):
            result.append((source, frame, (vertices, uv, [faces[j] for j in indices],
                                           [uv_faces[j] for j in indices], atlas)))
    if not result:
        raise ValueError(f"no video-observed textured faces for {link}")
    return result


def visibility(model, data, replay, vertices, link, frame, size):
    if not 0 <= frame < len(replay["cam_to_world"]):
        raise ValueError(f"frame {frame} is outside replay")
    w, h = size
    cam_pose = replay["cam_to_world"][frame]
    if not np.isfinite(cam_pose).all():
        raise ValueError(f"camera pose missing on frame {frame}")
    for name in replay.files:
        if name.startswith("q_"):
            q = replay[name][frame]
            if not np.isfinite(q):
                raise ValueError(f"joint position {name} missing on frame {frame}")
            data.joint(name[2:]).qpos[0] = q
    camera = model.camera("video").id
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (cam_pose[:3, :3] @ np.diag([1., -1., -1.])).flatten())
    model.cam_pos[camera], model.cam_quat[camera] = cam_pose[:3, 3], quat
    f, cx, cy = (float(replay[x]) for x in ("f", "cx", "cy"))
    model.cam_fovy[camera] = np.degrees(2 * np.arctan(h / (2 * f)))
    mujoco.mj_forward(model, data)
    body = model.body(link).id
    world = data.xpos[body] + vertices @ data.xmat[body].reshape(3, 3).T
    inverse = np.linalg.inv(cam_pose)
    camera_points = world @ inverse[:3, :3].T + inverse[:3, 3]
    renderer = mujoco.Renderer(model, int(h * .5), int(w * .5))
    try:
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="video")
        depth = renderer.render().copy()
    finally:
        renderer.close()
    return visible_from_depth(camera_points, depth, f, cx, cy, scale=.5)


def project_views(run, video, link, vertices, colors, views, distance_mm=25):
    """Compute video colours for an active SAM 3D mesh without modifying the result."""
    model = mujoco.MjModel.from_xml_path(str(run / "model.xml"))
    data = mujoco.MjData(model)
    size, _, _ = video_info(video)
    history = []
    with np.load(run / "replay3d.npz") as replay:
        for source, frame, (v, uv, faces, uv_faces, atlas) in views:
            seen = visibility(model, data, replay, vertices, link, frame, size)
            colors, hit = transfer_texture(vertices, colors, seen, v, uv, faces, uv_faces, atlas, distance_mm / 1000)
            history.append({"source": source, "frame": frame, "visible_vertices": int(seen.sum()),
                            "transferred_vertices": int(hit.sum())})
    return colors, history


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", required=True, type=Path, help="new result directory")
    ap.add_argument("--link", required=True)
    ap.add_argument("--view", nargs=2, action="append", metavar=("TEXTURED_OBJ", "FRAME"),
                    help="source OBJ relative to result and replay frame; repeat for other faces")
    ap.add_argument("--auto-views", action="store_true", help="use only recorded video-observed faces and their frames")
    ap.add_argument("--distance-mm", type=float, default=25)
    args = ap.parse_args()
    if args.out.exists() or args.out.resolve().is_relative_to(args.run.resolve()):
        ap.error("--out must be a new directory outside the source result")
    if not np.isfinite(args.distance_mm) or args.distance_mm <= 0 or args.distance_mm > 25:
        ap.error("--distance-mm must be in (0, 25]")
    report = json.loads((args.run / "sam3d.json").read_text())
    if not report.get(args.link, {}).get("accepted"):
        ap.error("the link must have an accepted SAM 3D candidate")
    if bool(args.auto_views) == bool(args.view):
        ap.error("choose either --auto-views or one or more --view arguments")
    views = observed_views(args.run, args.link) if args.auto_views else []
    for source, frame in args.view or []:
        path = (args.run / source).resolve()
        if not path.is_relative_to(args.run.resolve()) or not path.is_file():
            ap.error(f"texture source must be an existing file inside the result: {source}")
        views.append((source, int(frame), read_video_obj(path)))
    vertices, faces, colors = read_colored_ply(args.run / "meshes" / f"{args.link}_sam3d.ply")
    colors, history = project_views(args.run, args.video, args.link, vertices, colors, views, args.distance_mm)
    # Compute and validate everything before copying or changing a model.
    shutil.copytree(args.run, args.out)
    mesh = args.out / "meshes" / f"{args.link}_sam3d"
    write_obj(mesh.with_suffix(".obj"), vertices, faces, colors)
    write_ply(mesh.with_suffix(".ply"), vertices, faces, colors)
    report[args.link]["texture_experiment"] = {"manual": not args.auto_views, "views": history,
        "projection_limit_mm": args.distance_mm,
        "note": "Visual interpolation, not a measurement; inspect all sides."}
    (args.out / "sam3d.json").write_text(json.dumps(report, indent=2) + "\n")
    quality = assess(args.out)
    (args.out / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    print(json.dumps(history, indent=2))
    print(f"Experimental result: {args.out}")


if __name__ == "__main__":
    main()
