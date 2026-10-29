"""Paint the measured model with the video itself.

Every flat face of every link mesh (a lid's inner face, its back, the base's top, each side)
gets the frame where it was seen largest without being hidden: hidden behind another part of
the model (MuJoCo renders which geom is in front, from the phone's own camera pose on that
frame, saved by replay3d) or behind a hand (the point must lie in that link's SAM mask). The
face's pixels from that frame go into a texture atlas, so the lid shows the screen that was
on it and the closed lid's back, the base its keyboard. Shapes stay exactly as measured.

Writes meshes/<mesh>_tex.{obj,mtl,png} and switches model.xml / model.urdf (and their
model.nophysics.* copies) to them.
"""

import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .pipeline import load_parts, read_frame, video_info

CANDIDATES = 60  # frames scored per face
SEG_SCALE = 0.5  # visibility is rendered at this fraction of the video's size
MAX_CELL = 1024  # px, a face's longest side in the atlas
PX_PER_M = 4000  # texture resolution (0.25 mm per px) unless the face is too big for MAX_CELL
ATLAS_W = 4096
MASK_DILATE = 12  # px: SAM masks hug the outline; a face's edge may sit just outside
MIN_SEEN = 0.6  # of a face's sample points visible on a frame (a big face is often partly hidden)
MIN_FACE_M = 0.01  # faces thinner than this (a lid's edge) take the link's colour: a mm off shows the floor


def read_stl(path: Path) -> np.ndarray:
    """Triangles (T, 3, 3) of a binary STL."""
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84: 84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(n, 3, 3).astype(np.float64)


def planar_faces(tris: np.ndarray) -> list[np.ndarray]:
    """Group triangles into flat faces: same normal, same plane (within 1 mm)."""
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    off = np.einsum("ij,ij->i", nrm, tris[:, 0])
    groups, used = [], np.zeros(len(tris), bool)
    for i in range(len(tris)):
        if used[i]:
            continue
        same = (~used) & (nrm @ nrm[i] > 0.999) & (np.abs(off - off[i]) < 0.001)
        used |= same
        groups.append(np.flatnonzero(same))
    return groups


def link_masks(track_dir: Path) -> dict[str, list[str]]:
    """Which tracked parts show each link: from roles.json (a hinged plate) or drawers.json."""
    if (track_dir / "drawers.json").exists():
        d = json.loads((track_dir / "drawers.json").read_text())
        out = {"base": list(d["base"])}
        for i, (part, _) in enumerate(d["drawers"]):
            out[f"drawer{i + 1}"] = [part]
        return out
    r = json.loads((track_dir / "roles.json").read_text())
    # the open indicator (a keyboard) is part of the base, seen only while the lid is up
    return {"base": [p for p in (r["base"], r.get("open_indicator")) if p],
            "lid": [p for p in (r["plate"], r.get("plate_closed")) if p and p != r["base"]]}


def texture_model(track_dir: Path, video: Path, log=print) -> dict:
    cams = np.load(track_dir / "replay3d.npz")
    cam_to_world = cams["cam_to_world"]
    f, cx, cy = float(cams["f"]), float(cams["cx"]), float(cams["cy"])
    qs = {k[2:]: cams[k] for k in cams.files if k.startswith("q_")}
    xml = track_dir / "model.xml"
    src = track_dir / "model.nophysics.xml" if (track_dir / "model.nophysics.xml").exists() else xml
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    (w, h), n, _ = video_info(video)
    n = min(n, len(cam_to_world))
    names, tracks = load_parts(track_dir)
    shows = link_masks(track_dir)

    # Mesh geoms by asset file; the file's coordinates are the body's frame (emit/emit3d put no pose on geoms).
    root = ET.parse(src).getroot()
    files = {m.get("name"): m.get("file").replace("_tex.obj", ".stl") for m in root.iter("mesh")}  # again: from the STL
    geoms = []  # (geom id, body id, body name, mesh name, triangles in body frame, faces)
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.mesh(model.geom_dataid[g]).name
        if not files.get(mesh, "").endswith(".stl"):
            continue
        tris = read_stl(track_dir / files[mesh])
        body = int(model.geom_bodyid[g])
        geoms.append((g, body, model.body(body).name, mesh, tris, planar_faces(tris)))

    cam = model.camera("video").id
    sw, sh = int(w * SEG_SCALE), int(h * SEG_SCALE)
    model.cam_fovy[cam] = np.rad2deg(2 * np.arctan(h / 2 / f))
    renderer = mujoco.Renderer(model, sh, sw)
    renderer.enable_segmentation_rendering()
    flip = np.diag([1.0, -1.0, -1.0])

    def pose(t):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (cam_to_world[t][:3, :3] @ flip).flatten())
        model.cam_pos[cam], model.cam_quat[cam] = cam_to_world[t][:3, 3], quat
        for j, q in qs.items():
            data.joint(j).qpos[0] = q[t]
        mujoco.mj_forward(model, data)

    def project(t, body, pts):
        """Body-frame points -> pixels and camera depth on frame t."""
        world = data.xpos[body] + pts @ data.xmat[body].reshape(3, 3).T
        inv = np.linalg.inv(cam_to_world[t])
        c = world @ inv[:3, :3].T + inv[:3, 3]
        z = np.maximum(c[:, 2], 1e-6)
        return np.stack([f * c[:, 0] / z + cx, f * c[:, 1] / z + cy], axis=1), c[:, 2]

    kernel = np.ones((2 * MASK_DILATE + 1,) * 2, np.uint8)
    thin = set()
    for gi, (_, _, _, _, tris, faces) in enumerate(geoms):
        for fi, face in enumerate(faces):
            p = tris[face].reshape(-1, 3)
            _, sv, _ = np.linalg.svd(p - p.mean(axis=0), full_matrices=False)
            extent = np.ptp((p - p.mean(axis=0)) @ np.linalg.svd(p - p.mean(axis=0))[2][1])  # second axis
            if extent < MIN_FACE_M:
                thin.add((gi, fi))
    frames = np.unique(np.linspace(0, n - 1, CANDIDATES).astype(int))
    best = {}  # (geom index, face index) -> (score, frame)
    # 7 points per triangle, pulled 15% toward its centre: corners sit on the face's outline,
    # where the render shows the neighbouring face as often as this one
    bary = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3],
                     [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    bary = 0.85 * bary + 0.15 / 3
    for t in frames:
        pose(t)
        renderer.update_scene(data, camera="video")
        seg = renderer.render()
        gid = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        masks = {}
        for gi, (g, body, bname, mesh, tris, faces) in enumerate(geoms):
            if bname not in masks:
                m = np.zeros((h, w), np.uint8)
                for part in shows.get(bname, []):
                    if t in tracks.get(part, {}):
                        m |= tracks[part][t].full_mask(h, w).astype(np.uint8)
                masks[bname] = cv2.dilate(m, kernel) > 0
            mask = masks[bname]
            if not mask.any():
                continue
            for fi, face in enumerate(faces):
                if (gi, fi) in thin:
                    continue
                pts = np.einsum("kj,tjd->tkd", bary, tris[face]).reshape(-1, 3)  # 7 points per triangle
                uv, z = project(t, body, pts)
                inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
                if inside.mean() < 0.9:
                    continue
                ui, vi = np.clip(uv[:, 0], 0, w - 1).astype(int), np.clip(uv[:, 1], 0, h - 1).astype(int)
                si, sj = np.minimum((vi * SEG_SCALE).astype(int), sh - 1), np.minimum((ui * SEG_SCALE).astype(int), sw - 1)
                seen = (gid[si, sj] == g) & mask[vi, ui] & inside
                if seen.mean() < MIN_SEEN:
                    continue
                tri_uv = uv.reshape(len(face), 7, 2)[:, :3]
                d1, d2 = tri_uv[:, 1] - tri_uv[:, 0], tri_uv[:, 2] - tri_uv[:, 0]
                area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]).sum()
                # prefer a front view (an oblique one smears the texture): weight by cos of the view angle
                p3 = data.xpos[body] + pts[:3] @ data.xmat[body].reshape(3, 3).T
                fn = np.cross(p3[1] - p3[0], p3[2] - p3[0])
                view = cam_to_world[t][:3, 3] - p3.mean(axis=0)
                area *= abs(fn @ view) / (np.linalg.norm(fn) * np.linalg.norm(view) + 1e-12) * seen.mean() ** 2
                if area > best.get((gi, fi), (0, -1))[0]:
                    best[(gi, fi)] = (area, int(t))
    renderer.close()

    # Atlas: each seen face rectified to a front view (a homography from its best frame), at
    # PX_PER_M but at most MAX_CELL px, so oblique views aren't skewed.
    cells = []  # (gi, fi, frame, in-plane coords of its corners (3T, 2), lo, k, cw, ch, basis)
    for (gi, fi), (_, t) in sorted(best.items(), key=lambda kv: kv[1][1]):
        g, body, bname, mesh, tris, faces = geoms[gi]
        pts = tris[faces[fi]].reshape(-1, 3)
        nrm = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        nrm /= np.linalg.norm(nrm)
        e1 = np.cross(nrm, [0, 0, 1.0]) if abs(nrm[2]) < 0.9 else np.cross(nrm, [1.0, 0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nrm, e1)
        ab = np.stack([pts @ e1, pts @ e2], axis=1)
        lo, hi = ab.min(axis=0), ab.max(axis=0)
        k = min(PX_PER_M, MAX_CELL / max(hi - lo))
        cw, ch = (np.maximum(4, np.ceil((hi - lo) * k)).astype(int) + 2)
        cells.append((gi, fi, t, ab, lo, k, cw, ch, (e1, e2, float(pts[0] @ nrm), nrm)))
    place, x, y, row = {}, 0, 0, 0
    for c in sorted(cells, key=lambda c: -c[7]):  # shelf packing, tallest first
        if x + c[6] > ATLAS_W:
            x, y, row = 0, y + row, 0
        place[c[:2]] = (x, y)
        x, row = x + c[6], max(row, c[7])
    atlas_h = y + row + 8
    atlas = np.zeros((atlas_h, ATLAS_W, 3), np.uint8)
    # one 8x8 patch per mesh along the bottom row, for its faces never seen (filled below)
    cache = {}
    face_uv = {}  # (gi, fi) -> per-triangle-corner uv in atlas pixels
    for gi, fi, t, ab, lo, k, cw, ch, (e1, e2, d, nrm) in cells:
        g, body, bname, mesh, tris, faces = geoms[gi]
        if t not in cache:
            cache.clear()
            cache[t] = read_frame(video, t)
        pose(t)
        corners_ab = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], float) / k + lo
        corners = corners_ab[:, :1] * e1 + corners_ab[:, 1:] * e2 + d * nrm  # back in the body frame
        img_pts, _ = project(t, body, corners)
        hmg = cv2.getPerspectiveTransform(img_pts.astype(np.float32),
                                          np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], np.float32))
        ax, ay = place[(gi, fi)]
        cell = cv2.warpPerspective(cache[t], hmg, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # Pixels outside the link's own mask on that frame (floor at a slightly-off edge, a
        # hand) take the face's median colour.
        own = np.zeros((h, w), np.uint8)
        for part in shows.get(bname, []):
            if t in tracks.get(part, {}):
                own |= tracks[part][t].full_mask(h, w).astype(np.uint8)
        inside = cv2.warpPerspective(own, hmg, (cw, ch), flags=cv2.INTER_NEAREST) > 0
        if inside.mean() > 0.2:
            cell[~inside] = np.median(cell[inside], axis=0)
        atlas[ay: ay + ch, ax: ax + cw] = cell
        face_uv[(gi, fi)] = (ab - lo) * k + np.array([ax, ay])

    # One textured OBJ per mesh asset, all on one atlas; faces never seen take the link's colour.
    done, out = {}, {"meshes": {}}
    mesh_dir = track_dir / Path(files[geoms[0][3]]).parent if geoms else track_dir
    for gi, (g, body, bname, mesh, tris, faces) in enumerate(geoms):
        if mesh in done:
            continue
        fill = (ATLAS_W - 8 * (gi + 1), atlas_h - 8)
        seen = [(gi, fi) for fi in range(len(faces)) if (gi, fi) in face_uv]
        if seen:  # the link's colour: the median of its largest painted face
            k = max(seen, key=lambda k: np.ptp(face_uv[k], axis=0).prod())
            (x0, y0), (x1, y1) = face_uv[k].min(axis=0).astype(int), face_uv[k].max(axis=0).astype(int)
            mean = np.median(atlas[y0: y1 + 1, x0: x1 + 1].reshape(-1, 3), axis=0)
        else:
            rgba = model.geom_rgba[g]
            mean = np.array([rgba[2], rgba[1], rgba[0]]) * 255
        stem = Path(files[mesh]).stem + "_tex"
        lines = ["mtllib video_tex.mtl", "usemtl video_tex"]
        vt = []
        for fi, face in enumerate(faces):
            if (gi, fi) in face_uv:
                vt.append(face_uv[(gi, fi)])
            else:
                vt.append(np.tile(np.array(fill) + 4, (3 * len(face), 1)).astype(float))
        vt = np.concatenate(vt)
        order = np.concatenate(faces)
        for v in tris[order].reshape(-1, 3):
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        for u, v in vt:
            lines.append(f"vt {u / ATLAS_W:.6f} {1 - v / atlas_h:.6f}")
        for i in range(len(order)):
            a = 3 * i + 1
            lines.append(f"f {a}/{a} {a + 1}/{a + 1} {a + 2}/{a + 2}")
        atlas[fill[1]:, fill[0]: fill[0] + 8] = mean
        (mesh_dir / f"{stem}.obj").write_text("\n".join(lines) + "\n")
        done[mesh] = str(Path(files[mesh]).parent / f"{stem}.obj")
        out["meshes"][mesh] = {"faces_seen": len(seen), "faces": len(faces),
                               "frames": sorted({best[k][1] for k in seen}),
                               "by_face": {int(fi): {"frame": best[(gi, fi)][1], "triangles": len(faces[fi])}
                                           for _, fi in seen}}
        log(f"{mesh}: {len(seen)}/{len(faces)} faces painted from frames {out['meshes'][mesh]['frames']}")

    cv2.imwrite(str(mesh_dir / "video_tex.png"), atlas)
    (mesh_dir / "video_tex.mtl").write_text("newmtl video_tex\nKd 1 1 1\nmap_Kd video_tex.png\n")
    for old in mesh_dir.glob("*_tex.png"):  # per-mesh atlases from before
        if old.name != "video_tex.png":
            old.unlink()
    for old in mesh_dir.glob("*_tex.mtl"):
        if old.name != "video_tex.mtl":
            old.unlink()
    for path in [p for p in (xml, track_dir / "model.nophysics.xml") if p.exists()]:
        _switch_mjcf(path, done)
    for path in [p for p in (track_dir / "model.urdf", track_dir / "model.nophysics.urdf") if p.exists()]:
        _switch_urdf(path, {files[m]: f for m, f in done.items()})
    (track_dir / "texture.json").write_text(json.dumps(out, indent=2))
    return out


def _switch_mjcf(path: Path, done: dict[str, str]):
    tree = ET.parse(path)
    root = tree.getroot()
    asset = root.find("asset")
    for m in asset.findall("mesh"):
        if m.get("name") in done:
            m.set("file", done[m.get("name")])
    for old in asset.findall("texture") + asset.findall("material"):
        if old.get("name", "").endswith("_tex"):
            asset.remove(old)
    if done:
        png = str(Path(next(iter(done.values()))).parent / "video_tex.png")
        ET.SubElement(asset, "texture", name="video_tex", type="2d", file=png)
        ET.SubElement(asset, "material", name="video_tex", texture="video_tex")
    for g in root.iter("geom"):
        if g.get("mesh") in done:
            g.set("material", "video_tex")
            g.set("rgba", "1 1 1 1")
    ET.indent(tree, "  ")
    tree.write(path)


def _switch_urdf(path: Path, done: dict[str, str]):
    """Visuals use the textured OBJ (with its .mtl); collisions keep the measured STL."""
    tree = ET.parse(path)
    for vis in tree.getroot().iter("visual"):
        for m in vis.iter("mesh"):
            if m.get("filename") in done:
                m.set("filename", done[m.get("filename")])
        for mat in vis.findall("material"):
            vis.remove(mat)
    ET.indent(tree, "  ")
    tree.write(path, xml_declaration=True, encoding="unicode")
