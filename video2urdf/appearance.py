"""Object-aware visual completion on top of an already fitted RGB-D model.

The fitted meshes and joint data are evidence; generated visuals are hypotheses. This
module never changes the measured meshes, collision shapes, joints or inertial values.
"""
import copy
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from .lidar import write_stl
from .shape import prism
from .texture import read_stl


def category_of(appearance: dict) -> str:
    category = appearance.get("category", "")
    if category in ("laptop", "drawer", "book", "box", "other"):
        return category
    name = appearance.get("object", "").lower()
    if any(word in name for word in ("laptop", "notebook", "macbook")):
        return "laptop"
    if any(word in name for word in ("drawer", "cabinet", "filing")):
        return "drawer"
    if "book" in name:
        return "book"
    if any(word in name for word in ("box", "chest", "case")):
        return "box"
    return "other"


def book_cover_mesh(track_dir: Path, outer_blend: float = 0.) -> tuple[np.ndarray, np.ndarray]:
    """Hypothesize a rectangular cover using the visible inner face and hinge axis.

    The shape is constrained by the fitted face; it does not measure unseen corners.
    """
    path = track_dir / "meshes/screen.stl"
    if not path.exists():
        raise ValueError("no fitted cover face available")
    pts = read_stl(path).reshape(-1, 3)
    yz = pts[:, 1:3]
    _, _, axes = np.linalg.svd(yz - yz.mean(axis=0), full_matrices=False)
    e2 = np.r_[0., axes[0]]
    if e2 @ pts.mean(axis=0) < 0:
        e2 = -e2
    e1 = np.array([1., 0., 0.])
    e3 = np.cross(e1, e2)
    old_lid = track_dir / "meshes/lid.stl"
    old_pts = None
    if old_lid.exists():
        old_pts = read_stl(old_lid).reshape(-1, 3)
        old_center = old_pts.mean(axis=0)
        if (pts.mean(axis=0) - old_center) @ e3 < 0:
            e3 = -e3
    x0, x1 = np.percentile(pts @ e1, [1, 99])
    far = float(np.percentile(pts @ e2, 99))
    if old_pts is not None and outer_blend:
        # Blend toward the measured silhouette when a face-only rectangle
        # would cut off observed pixels. This remains a visual hypothesis.
        old_x0, old_x1 = np.percentile(old_pts @ e1, [1, 99])
        old_far = float(np.percentile(old_pts @ e2, 99))
        x0 += outer_blend * (min(x0, old_x0) - x0)
        x1 += outer_blend * (max(x1, old_x1) - x1)
        far += outer_blend * (max(far, old_far) - far)
    if x1 - x0 < .02 or far < .02:
        raise ValueError("cover face is too small to infer a shape")
    # The visible face must remain outside the generated cover, otherwise it
    # occludes its own texture in the renderer.
    center = float(np.min(pts @ e3))
    model = json.loads((track_dir / "model3d.json").read_text())
    thickness = float(np.clip(model.get("lid_thickness_m", .004), .001, .015))
    outline = np.array([[x0, 0], [x1, 0], [x1, far], [x0, far]])
    return prism(outline, np.zeros(3), e1, e2, e3, center - thickness, center)


def bevel_mesh(triangles: np.ndarray, radius: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """Bevel an existing convex measured prism by at most radius, in metres.

    Every new point lies on an original hull edge. This intentionally small shape
    hypothesis cannot enlarge the measured object or move its joint.
    """
    vertices = np.unique(np.round(triangles.reshape(-1, 3), 7), axis=0)
    hull = ConvexHull(vertices)
    edges = set()
    for tri in hull.simplices:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges.add(tuple(sorted((int(a), int(b)))))
    points = []
    for a, b in edges:
        edge = vertices[b] - vertices[a]
        length = np.linalg.norm(edge)
        if length < 1e-7:
            continue
        amount = min(radius, length * 0.2)
        points.extend((vertices[a] + edge * amount / length, vertices[b] - edge * amount / length))
    points = np.unique(np.round(points, 7), axis=0)
    result = ConvexHull(points)
    # Qhull facets are not consistently oriented; orient outwards for STL and textures.
    faces = result.simplices.copy()
    normal = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    wrong = np.einsum("ij,ij->i", normal, result.equations[:, :3]) < 0
    faces[wrong] = faces[wrong][:, [0, 2, 1]]
    return points, faces


def laptop_screen_visual(triangles: np.ndarray, inset: float = 0.) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular visual screen from the fitted face, in hinge-link coordinates."""
    pts = np.asarray(triangles, dtype=float).reshape(-1, 3)
    if len(pts) < 3 or not np.isfinite(pts).all():
        raise ValueError("invalid fitted screen")
    _, _, axes = np.linalg.svd(pts[:, 1:3] - pts[:, 1:3].mean(axis=0), full_matrices=False)
    along = np.r_[0., axes[0]]
    hinge = np.array([1., 0., 0.])
    normal = np.cross(hinge, along)
    x0, x1 = np.min(pts @ hinge), np.max(pts @ hinge)
    y0, y1 = np.min(pts @ along), np.max(pts @ along)
    z0, z1 = np.min(pts @ normal), np.max(pts @ normal)
    if x1 - x0 < .02 or y1 - y0 < .02:
        raise ValueError("fitted screen is too small for a rectangular visual")
    if not 0 <= inset < .25:
        raise ValueError("invalid screen inset")
    dx, dy = inset * (x1 - x0), inset * (y1 - y0)
    x0, x1, y0, y1 = x0 + dx, x1 - dx, y0 + dy, y1 - dy
    if z1 - z0 < .0002:
        z0 -= .0001
        z1 += .0001
    outline = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return prism(outline, np.zeros(3), hinge, along, normal, z0, z1)


def screen_needs_bezel(screen: np.ndarray, lid: np.ndarray) -> bool:
    """Detect a face that fills its entire lid, where no bezel is visible."""
    screen, lid = np.asarray(screen).reshape(-1, 3), np.asarray(lid).reshape(-1, 3)
    if len(screen) < 3 or len(lid) < 3 or not np.isfinite(screen).all() or not np.isfinite(lid).all():
        return False
    _, _, axes = np.linalg.svd(screen[:, 1:] - screen[:, 1:].mean(axis=0), full_matrices=False)
    along = np.r_[0., axes[0]]
    screen_size = np.array([np.ptp(screen[:, 0]), np.ptp(screen @ along)])
    lid_size = np.array([np.ptp(lid[:, 0]), np.ptp(lid @ along)])
    return bool(np.all(lid_size > .02) and np.all(screen_size / lid_size >= .9))


def laptop_base_visual(triangles: np.ndarray, regularize_thickness: bool = False,
                       target_x_bounds: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Complete a laptop base's missing corners inside its fitted axis-aligned bounds.

    Only the appearance is inferred: filling the bounds neither measures the
    true outline nor changes collision geometry or physical dimensions.
    """
    pts = triangles.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    if not np.isfinite(pts).all() or np.any(hi - lo < 1e-4):
        raise ValueError("fitted laptop base has invalid bounds")
    if target_x_bounds is not None:
        lo[0], hi[0] = target_x_bounds
    outline = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    # A noisy RGB depth fit can make a laptop base implausibly thick. This
    # affects only the visual; the fitted collision and recorded values remain.
    limit = .08 * min(hi[0] - lo[0], hi[1] - lo[1])
    visual_lo = (max(lo[2], hi[2] - limit)
                 if regularize_thickness and hi[2] - lo[2] > .15 * min(hi[:2] - lo[:2]) else lo[2])
    vertices, faces = prism(outline, np.zeros(3), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2], visual_lo, hi[2])
    return bevel_mesh(vertices[faces])


def laptop_base_width_hint(base: np.ndarray, lid: np.ndarray) -> tuple[float, float] | None:
    """Use a slightly wider, aligned lid as a visual hint for a cropped base."""
    base, lid = np.asarray(base).reshape(-1, 3), np.asarray(lid).reshape(-1, 3)
    if len(base) < 3 or len(lid) < 3 or not np.isfinite(base).all() or not np.isfinite(lid).all():
        return None
    a, b = float(base[:, 0].min()), float(base[:, 0].max())
    c, d = float(lid[:, 0].min()), float(lid[:, 0].max())
    width = b - a
    if (width <= .02 or not 1.03 <= (d - c) / width <= 1.25
            or c > a + .02 * width or d < b - .02 * width
            or min(abs(a - c), abs(b - d)) > .05 * width):
        return None
    return min(a, c), max(b, d)


def laptop_lid_height_hint(lid: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Limit a strongly portrait-proportioned lid visual near its fitted hinge."""
    pts = np.asarray(lid, dtype=float).reshape(-1, 3)
    if len(pts) < 3 or not np.isfinite(pts).all():
        return None
    _, values, axes = np.linalg.svd(pts[:, 1:] - pts[:, 1:].mean(axis=0), full_matrices=False)
    direction = np.r_[0., axes[0]]
    projected = pts @ direction
    low, high = float(projected.min()), float(projected.max())
    width, height = np.ptp(pts[:, 0]), high - low
    hinge_gap = max(low, -high, 0.)
    if (width < .02 or height <= 1.08 * width or values[0] < 2 * values[1]
            or hinge_gap > .12 * height):
        return None
    return direction, max(.85, width / height)


def _stl_file(filename: str) -> str:
    return filename.replace("_tex.obj", ".stl")


def _mesh_bounds(track_dir: Path, body: ET.Element, assets: dict) -> tuple[np.ndarray, np.ndarray] | None:
    points = []
    for geom in body.findall("geom"):
        if geom.get("contype") == "0" or geom.get("name", "").startswith("inferred_"):
            continue
        file = _stl_file(assets.get(geom.get("mesh"), ""))
        if file.endswith(".stl") and (track_dir / file).exists():
            pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            points.append(read_stl(track_dir / file).reshape(-1, 3) + pos)
    if not points:
        return None
    points = np.concatenate(points)
    return points.min(axis=0), points.max(axis=0)


def complete_laptop_hinge(track_dir: Path) -> dict | None:
    """Add a small visual-only hinge when the fitted axis is close to the base."""
    root = ET.parse(track_dir / 'model.xml').getroot()
    assets = {m.get('name'): m.get('file', '') for m in root.findall('./asset/mesh')}
    base = next((b for b in root.iter('body') if b.get('name') == 'base'), None)
    lid = next((b for b in base.findall('body') if b.get('name') == 'lid'), None) if base is not None else None
    joint = lid.find('joint') if lid is not None else None
    bounds = _mesh_bounds(track_dir, base, assets) if base is not None else None
    lid_bounds = _mesh_bounds(track_dir, lid, assets) if lid is not None else None
    if joint is None or bounds is None or lid_bounds is None or joint.get('type', 'hinge') != 'hinge':
        return None
    axis = np.fromstring(joint.get('axis', '0 0 1'), sep=' ')
    pos = np.fromstring(lid.get('pos', '0 0 0'), sep=' ')
    anchor = np.fromstring(joint.get('pos', '0 0 0'), sep=' ')
    if axis.size != 3 or pos.size != 3 or anchor.size != 3 or not np.allclose(axis, [1, 0, 0], atol=1e-4):
        return None
    if lid.get('quat') or lid.get('euler') or lid.get('axisangle'):
        return None
    pivot = pos + anchor
    lo, hi = bounds
    llo, lhi = lid_bounds
    left, right = max(lo[0], llo[0] + pos[0]), min(hi[0], lhi[0] + pos[0])
    nearest = np.clip(pivot[1:], lo[1:], hi[1:])
    gap = float(np.linalg.norm(pivot[1:] - nearest))
    if not np.isfinite([*pivot, left, right, gap]).all() or right - left < .05 or not .003 <= gap <= .018:
        return None
    center = np.array([(left + right) / 2, *(0.5 * (pivot[1:] + nearest))])
    radius = min(.01, gap / 2 + .002)
    length = .8 * (right - left)
    vec = lambda v: ' '.join(f'{x:.6f}' for x in v)  # noqa: E731
    for name in ('model.xml', 'model.nophysics.xml'):
        path = track_dir / name
        if not path.exists():
            continue
        tree = ET.parse(path)
        body = next(b for b in tree.getroot().iter('body') if b.get('name') == 'base')
        for old in list(body.findall('geom')):
            if old.get('name') == 'inferred_laptop_hinge':
                body.remove(old)
        ET.SubElement(body, 'geom', name='inferred_laptop_hinge', type='cylinder', pos=vec(center),
                      size=vec([radius, length / 2]), quat='.70710678 0 .70710678 0',
                      rgba='.22 .22 .22 1', contype='0', conaffinity='0', mass='0')
        ET.indent(tree, '  '); tree.write(path)
    for name in ('model.urdf', 'model.nophysics.urdf'):
        path = track_dir / name
        if not path.exists():
            continue
        tree = ET.parse(path)
        link = next(b for b in tree.getroot().findall('link') if b.get('name') == 'base')
        for old in list(link.findall('visual')):
            if old.get('name') == 'inferred_laptop_hinge':
                link.remove(old)
        visual = ET.SubElement(link, 'visual', name='inferred_laptop_hinge')
        ET.SubElement(visual, 'origin', xyz=vec(center), rpy='0 1.57079632679 0')
        ET.SubElement(ET.SubElement(visual, 'geometry'), 'cylinder', radius=f'{radius:.6f}', length=f'{length:.6f}')
        ET.SubElement(ET.SubElement(visual, 'material', name='inferred_laptop_hinge_material'), 'color', rgba='.22 .22 .22 1')
        ET.indent(tree, '  '); tree.write(path, xml_declaration=True, encoding='unicode')
    return {'name': 'inferred_laptop_hinge', 'link': 'base', 'radius_m': radius, 'length_m': length,
            'basis': 'visual connector estimated from fitted hinge axis and base bounds'}


def _drawer_front_color(track_dir: Path, root: ET.Element, assets: dict) -> list[float]:
    """Use the existing video texture's median for the drawer front when available."""
    for body in root.iter("body"):
        if not body.get("name", "").startswith("drawer"):
            continue
        for geom in body.findall("geom"):
            file = assets.get(geom.get("mesh"), "")
            obj = track_dir / file
            atlas = cv2.imread(str(obj.parent / "video_tex.png")) if obj.suffix == ".obj" else None
            if atlas is None or not obj.exists():
                continue
            uv = [line.split()[1:3] for line in obj.read_text().splitlines() if line.startswith("vt ")]
            if not uv:
                continue
            uv = np.asarray(uv, dtype=float)
            x = np.clip((uv[:, 0] * atlas.shape[1]).astype(int), 0, atlas.shape[1] - 1)
            y = np.clip(((1 - uv[:, 1]) * atlas.shape[0]).astype(int), 0, atlas.shape[0] - 1)
            return (np.median(atlas[y, x], axis=0)[::-1] / 255).tolist()
    return [0.56, 0.43, 0.29]


def remove_laptop_details(track_dir: Path) -> None:
    """Remove previously added schematic details before projecting video textures."""
    for name in ("model.xml", "model.nophysics.xml", "model.urdf", "model.nophysics.urdf"):
        path = track_dir / name
        if not path.exists():
            continue
        tree = ET.parse(path)
        parent_tag, child_tag = ("link", "visual") if name.endswith(".urdf") else ("body", "geom")
        for body in tree.getroot().iter(parent_tag):
            for child in list(body.findall(child_tag)):
                if child.get("name") in ("inferred_keyboard", "inferred_trackpad"):
                    body.remove(child)
        ET.indent(tree, "  ")
        tree.write(path, xml_declaration=name.endswith(".urdf"), encoding="unicode")


def complete_laptop_details(track_dir: Path) -> list[dict]:
    """Add schematic, visual-only keyboard and trackpad within the current visual base bounds."""
    root = ET.parse(track_dir / "model.xml").getroot()
    assets = {m.get("name"): m.get("file", "") for m in root.findall("./asset/mesh")}
    base = next((b for b in root.iter("body") if b.get("name") == "base"), None)
    if base is None:
        return []
    bounds = _mesh_bounds(track_dir, base, assets)
    for geom in base.findall("geom"):
        name = geom.get("mesh", "")
        file = _stl_file(assets.get(name, ""))
        if name.endswith("_infer") and file.endswith(".stl") and (track_dir / file).exists():
            points = read_stl(track_dir / file).reshape(-1, 3)
            pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            bounds = ((points + pos).min(axis=0), (points + pos).max(axis=0))
            break
    lid = next((b for b in base.findall("body") if b.get("name") == "lid"), None)
    if bounds is None or lid is None:
        return []
    lo, hi = bounds
    width, depth = hi[0] - lo[0], hi[1] - lo[1]
    if not np.isfinite([*lo, *hi]).all() or width < .05 or depth < .05:
        return []
    hinge_y = np.fromstring(lid.get("pos", "0 0 0"), sep=" ")[1]
    from_low = abs(hinge_y - lo[1]) <= abs(hinge_y - hi[1])
    def y_at(fraction):
        return lo[1] + depth * fraction if from_low else hi[1] - depth * fraction
    specs = [
        {"name": "inferred_keyboard", "center_m": [(lo[0] + hi[0]) / 2, y_at(.32), hi[2] + .0006],
         "size_m": [width * .78, depth * .43, .001], "rgb": [.10, .11, .12],
         "basis": "schematic visual within visual base bounds; exact keys are not measured"},
        {"name": "inferred_trackpad", "center_m": [(lo[0] + hi[0]) / 2, y_at(.78), hi[2] + .0006],
         "size_m": [width * .32, depth * .22, .001], "rgb": [.38, .40, .41],
         "basis": "schematic visual within visual base bounds; position is inferred"}]
    vec = lambda values: " ".join(f"{v:.6f}" for v in values)  # noqa: E731
    for name in ("model.xml", "model.nophysics.xml"):
        path = track_dir / name
        if not path.exists():
            continue
        tree = ET.parse(path)
        body = next(b for b in tree.getroot().iter("body") if b.get("name") == "base")
        for g in list(body.findall("geom")):
            if g.get("name") in {x["name"] for x in specs}:
                body.remove(g)
        for item in specs:
            ET.SubElement(body, "geom", name=item["name"], type="box", pos=vec(item["center_m"]),
                          size=vec(np.asarray(item["size_m"]) / 2), rgba=vec(item["rgb"] + [1]),
                          contype="0", conaffinity="0", mass="0")
        ET.indent(tree, "  ")
        tree.write(path)
    for name in ("model.urdf", "model.nophysics.urdf"):
        path = track_dir / name
        if not path.exists():
            continue
        tree = ET.parse(path)
        base_link = next(b for b in tree.getroot().findall("link") if b.get("name") == "base")
        for visual in list(base_link.findall("visual")):
            if visual.get("name") in {x["name"] for x in specs}:
                base_link.remove(visual)
        for item in specs:
            visual = ET.SubElement(base_link, "visual", name=item["name"])
            ET.SubElement(visual, "origin", xyz=vec(item["center_m"]), rpy="0 0 0")
            ET.SubElement(ET.SubElement(visual, "geometry"), "box", size=vec(item["size_m"]))
            material = ET.SubElement(visual, "material", name=item["name"] + "_material")
            ET.SubElement(material, "color", rgba=vec(item["rgb"] + [1]))
        ET.indent(tree, "  ")
        tree.write(path, xml_declaration=True, encoding="unicode")
    return specs


def complete_drawer_fronts(track_dir: Path, appearance: dict) -> list[dict]:
    """Complete visible fronts; if their count is unknown, hypothesize three for appearance."""
    count = appearance.get("drawer_count", 0)
    if not count:
        match = re.search(r"\b(one|two|three|four|five|\d+)[ -]drawer", appearance.get("object", "").lower())
        if match:
            word = match.group(1)
            count = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}.get(word, int(word) if word.isdigit() else 0)
    assumed_count = count == 0
    if assumed_count:
        count = 3
    if not isinstance(count, int) or not 1 <= count <= 8:
        return []
    root = ET.parse(track_dir / "model.xml").getroot()
    assets = {m.get("name"): m.get("file", "") for m in root.findall("./asset/mesh")}
    base = next((b for b in root.iter("body") if b.get("name") == "base"), None)
    drawers = [b for b in root.iter("body") if b.get("name", "").startswith("drawer")]
    bounds = _mesh_bounds(track_dir, base, assets) if base is not None else None
    if bounds is None or not drawers:
        return []
    lo, hi = bounds
    moving = []
    for body in drawers:
        box = _mesh_bounds(track_dir, body, assets)
        if box:
            pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
            moving.append((box[0] + pos, box[1] + pos))
    if not moving:
        return []
    color = _drawer_front_color(track_dir, root, assets)
    has_handles = appearance.get("drawer_has_handles", any("handle" in v.get("reason", "").lower()
                                                           for v in appearance.get("links", {}).values()))
    z_edges = np.linspace(lo[2], hi[2], count + 1)
    specs = []
    for i in range(count):
        z0, z1 = z_edges[i:i + 2]
        # Any overlap with the measured moving front belongs to that link, not the cabinet.
        if any(max(0., min(z1, b[2]) - max(z0, a[2])) > .05 * (z1 - z0) for a, b in moving):
            continue
        y0 = max(lo[1], min(a[1] for a, _ in moving)) + .003
        y1 = min(hi[1], max(b[1] for _, b in moving)) - .003
        z0, z1 = z0 + .003, z1 - .003
        if y1 - y0 < .02 or z1 - z0 < .02:
            continue
        specs.append({"name": f"inferred_drawer_front_{i + 1}",
                      "center_m": [float(hi[0] + .0005), float((y0 + y1) / 2), float((z0 + z1) / 2)],
                      "size_m": [.001, float(y1 - y0), float(z1 - z0)], "rgb": color,
                      "basis": ("uniformly spaced within cabinet bounds; three drawers assumed because count was uncertain"
                                if assumed_count else "uniformly spaced from the cabinet bounds and the video-estimated drawer count"),
                      "count_assumed": assumed_count})
        if has_handles:
            specs.append({"name": f"inferred_drawer_handle_{i + 1}",
                          "center_m": [float(hi[0] + .006), float((y0 + y1) / 2), float(z1 - .025)],
                          "size_m": [.010, float(min(.09, .3 * (y1 - y0))), .008],
                          "rgb": [.23, .23, .22],
                          "basis": "a handle was visible elsewhere; this handle's shape and location are guessed"})
    for body in drawers:
        box = _mesh_bounds(track_dir, body, assets)
        if not box:
            continue
        a, b = box
        pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        aa, bb = a + pos, b + pos
        overlaps = [max(0., min(z_edges[i + 1], bb[2]) - max(z_edges[i], aa[2])) for i in range(count)]
        slot = int(np.argmax(overlaps))
        y0 = max(lo[1], min(x[1] for x, _ in moving)) + .003
        y1 = min(hi[1], max(x[1] for _, x in moving)) - .003
        z0, z1 = z_edges[slot] + .003, z_edges[slot + 1] - .003
        if overlaps[slot] > 0 and y1 - y0 >= .02 and z1 - z0 >= .02:
            specs.append({"name": f"inferred_drawer_front_{body.get('name')}", "link": body.get("name"),
                          "center_m": [float(b[0] + .0005), float((y0 + y1) / 2 - pos[1]),
                                       float((z0 + z1) / 2 - pos[2])],
                          "size_m": [.001, float(y1 - y0), float(z1 - z0)], "rgb": color,
                          "basis": "moving front aligned to the overlapping cabinet slot; edges are estimated"})
            if has_handles:
                specs.append({"name": f"inferred_drawer_handle_{body.get('name')}",
                              "link": body.get("name"),
                              "center_m": [float(b[0] + .006), float((y0 + y1) / 2 - pos[1]),
                                           float(z1 - .025 - pos[2])],
                              "size_m": [.010, float(min(.09, .3 * (y1 - y0))), .008],
                              "rgb": [.23, .23, .22],
                              "basis": "a handle was visible; this moving drawer handle is an estimated visual"})
    for filename in ("model.xml", "model.nophysics.xml"):
        path = track_dir / filename
        if not path.exists():
            continue
        tree = ET.parse(path)
        bodies = {b.get("name"): b for b in tree.getroot().iter("body")}
        for body in bodies.values():
            for geom in body.findall("geom"):
                if geom.get("name", "").startswith(("inferred_drawer_front_", "inferred_drawer_handle_")):
                    body.remove(geom)
        for s in specs:
            body = bodies[s.get("link", "base")]
            vec = lambda vals: " ".join(f"{v:.6f}" for v in vals)  # noqa: E731
            ET.SubElement(body, "geom", name=s["name"], type="box", pos=vec(s["center_m"]),
                          size=vec(np.array(s["size_m"]) / 2), rgba=vec(s["rgb"] + [1]),
                          contype="0", conaffinity="0", mass="0")
        ET.indent(tree, "  ")
        tree.write(path)
    for filename in ("model.urdf", "model.nophysics.urdf"):
        path = track_dir / filename
        if not path.exists():
            continue
        tree = ET.parse(path)
        bodies = {l.get("name"): l for l in tree.getroot().findall("link")}
        for body in bodies.values():
            for visual in body.findall("visual"):
                if visual.get("name", "").startswith(("inferred_drawer_front_", "inferred_drawer_handle_")):
                    body.remove(visual)
        for s in specs:
            body = bodies[s.get("link", "base")]
            vec = lambda vals: " ".join(f"{v:.6f}" for v in vals)  # noqa: E731
            visual = ET.SubElement(body, "visual", name=s["name"])
            ET.SubElement(visual, "origin", xyz=vec(s["center_m"]), rpy="0 0 0")
            ET.SubElement(ET.SubElement(visual, "geometry"), "box", size=vec(s["size_m"]))
            material = ET.SubElement(visual, "material", name=s["name"] + "_material")
            ET.SubElement(material, "color", rgba=vec(s["rgb"] + [1]))
        ET.indent(tree, "  ")
        tree.write(path, xml_declaration=True, encoding="unicode")
    return specs


def complete_static_drawer_envelope(track_dir: Path) -> dict | None:
    """Hypothesize cabinet depth when the fitted box is only a thin observed face."""
    root = ET.parse(track_dir / 'model.xml').getroot()
    body = next((b for b in root.iter('body') if b.get('name') == 'base'), None)
    assets = {m.get('name'): m.get('file', '') for m in root.findall('./asset/mesh')}
    bounds = _mesh_bounds(track_dir, body, assets) if body is not None else None
    if bounds is None or not (track_dir / 'replay3d.npz').exists():
        return None
    lo, hi = (a.copy() for a in bounds)
    size = hi - lo
    thin = int(np.argmin(size[:2])); wide = 1 - thin
    if size[thin] >= .12 * size[wide] or size[wide] < .05 or size[2] < .05:
        return None
    pos = np.fromstring(body.get('pos', '0 0 0'), sep=' ')
    if pos.size != 3:
        return None
    with np.load(track_dir / 'replay3d.npz') as data:
        camera = data['cam_to_world'][:, :3, 3]
    delta = camera[:, :2] - (pos + (lo + hi) / 2)[:2]
    delta = delta[np.isfinite(delta).all(axis=1)]
    if len(delta) < 5:
        return None
    axis = np.argmax(np.abs(delta), axis=1)
    signs = np.sign(delta[:, thin])
    side = 1 if np.sum((axis == thin) & (signs > 0)) >= np.sum((axis == thin) & (signs < 0)) else -1
    if np.mean((axis == thin) & (signs == side)) < .7:
        return None
    inferred_depth = .65 * size[wide]
    if side < 0:
        hi[thin] = lo[thin] + inferred_depth
    else:
        lo[thin] = hi[thin] - inferred_depth
    outline = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    vertices, faces = prism(outline, np.zeros(3), *np.eye(3), lo[2], hi[2])
    file = 'meshes/static_drawer_infer.stl'
    write_stl(track_dir / file, vertices, faces)
    for name in ('model.xml', 'model.nophysics.xml'):
        path = track_dir / name
        if not path.exists():
            continue
        tree = ET.parse(path); model = tree.getroot()
        base = next(b for b in model.iter('body') if b.get('name') == 'base')
        original = next((g for g in base.findall('geom') if g.get('mesh') == 'base_0'), None)
        if original is None:
            raise ValueError(f'missing static base mesh in {name}')
        ET.SubElement(model.find('asset'), 'mesh', name='base_0_infer', file=file, inertia='exact')
        visual = copy.deepcopy(original); visual.set('mesh', 'base_0_infer')
        visual.set('contype', '0'); visual.set('conaffinity', '0'); visual.set('mass', '0')
        original.set('group', '3'); base.append(visual)
        ET.indent(tree, '  '); tree.write(path)
    for name in ('model.urdf', 'model.nophysics.urdf'):
        path = track_dir / name
        if not path.exists():
            continue
        tree = ET.parse(path)
        visual = tree.getroot().find("./link[@name='base']/visual/geometry/mesh")
        if visual is None:
            raise ValueError(f'missing static base visual in {name}')
        visual.set('filename', file)
        ET.indent(tree, '  '); tree.write(path, xml_declaration=True, encoding='unicode')
    return {'link': 'base', 'source': assets.get('base_0'), 'generated': file,
            'basis': 'unobserved cabinet depth inferred as 65% of the fitted visible width',
            'inferred_depth_mm': float(inferred_depth * 1000)}


def complete_static_drawer(track_dir: Path, appearance: dict) -> list[dict]:
    """Guess stationary drawer fronts on the consistently most viewed vertical face."""
    count = appearance.get('drawer_count', 0)
    assumed = count == 0
    if assumed:
        count = 3
    if type(count) is not int or not 1 <= count <= 8:
        return []
    root = ET.parse(track_dir / 'model.xml').getroot()
    assets = {m.get('name'): m.get('file', '') for m in root.findall('./asset/mesh')}
    base = next((b for b in root.iter('body') if b.get('name') == 'base'), None)
    bounds = _mesh_bounds(track_dir, base, assets) if base is not None else None
    inferred = track_dir / assets.get('base_0_infer', '')
    if assets.get('base_0_infer') and inferred.is_file():
        points = read_stl(inferred).reshape(-1, 3)
        bounds = points.min(axis=0), points.max(axis=0)
    if bounds is None or not (track_dir / 'replay3d.npz').exists():
        return []
    lo, hi = bounds
    pos = np.fromstring(base.get('pos', '0 0 0'), sep=' ')
    if pos.size != 3 or not np.isfinite([*lo, *hi, *pos]).all() or min(hi - lo) < .02:
        return []
    with np.load(track_dir / 'replay3d.npz') as replay:
        cameras = replay['cam_to_world'][:, :3, 3]
    direction = cameras[:, :2] - (pos + (lo + hi) / 2)[:2]
    norm = np.linalg.norm(direction, axis=1)
    direction = direction[np.isfinite(direction).all(axis=1) & (norm > .01)]
    if len(direction) < 5:
        return []
    choices = np.argmax(np.abs(direction), axis=1)
    signs = np.sign(direction[np.arange(len(direction)), choices]).astype(int)
    face, side = max(((i, s) for i in (0, 1) for s in (-1, 1)),
                     key=lambda pair: np.count_nonzero((choices == pair[0]) & (signs == pair[1])))
    if np.mean((choices == face) & (signs == side)) < .7:
        return []  # no consistently viewed side; do not choose a random face
    cross = 1 - face
    width, height = hi[cross] - lo[cross], hi[2] - lo[2]
    # Keep panel gaps and handles recognizable even when RGB-only metric
    # scale is wrong; these are visual proportions, not measured hardware.
    margin = min(width * .008, height / count * .03)
    color_geom = next((g for g in base.findall('geom') if g.get('mesh')), None)
    rgba = np.fromstring(color_geom.get('rgba', '.5 .5 .5 1'), sep=' ') if color_geom is not None else np.array([.5, .5, .5, 1])
    color = np.clip(rgba[:3] + .04, 0, 1).tolist() if rgba.size >= 3 else [.55, .55, .55]
    specs = []
    for i in range(count):
        z0, z1 = lo[2] + i * height / count + margin, lo[2] + (i + 1) * height / count - margin
        center = (lo + hi) / 2
        thickness = min(width, height / count) * .004
        center[face] = (hi[face] if side > 0 else lo[face]) + side * thickness / 2
        center[2] = (z0 + z1) / 2
        size = np.zeros(3); size[face] = thickness; size[cross] = width - 2 * margin; size[2] = z1 - z0
        specs.append({'name': f'inferred_drawer_front_{i + 1}', 'center_m': center.tolist(),
                      'size_m': size.tolist(), 'rgb': color, 'count_assumed': assumed,
                      'front_axis': int(face), 'front_sign': int(side),
                      'basis': 'uniformly spaced on the most viewed side; front direction and panel edges are inferred'})
        if appearance.get('drawer_has_handles', False):
            handle_depth = min(width, z1 - z0) * .04
            hc = center.copy(); hc[face] += side * (thickness + handle_depth) / 2; hc[2] = z1 - .16 * (z1 - z0)
            hs = np.zeros(3); hs[face] = handle_depth; hs[cross] = width * .25; hs[2] = (z1 - z0) * .04
            specs.append({'name': f'inferred_drawer_handle_{i + 1}', 'center_m': hc.tolist(),
                          'size_m': hs.tolist(), 'rgb': [.23, .23, .22],
                          'basis': 'handle location and shape are inferred'})
    for name in ('model.xml', 'model.nophysics.xml'):
        path = track_dir / name
        if not path.exists():
            continue
        tree = ET.parse(path); body = next(b for b in tree.getroot().iter('body') if b.get('name') == 'base')
        for old in list(body.findall('geom')):
            if old.get('name', '').startswith(('inferred_drawer_front_', 'inferred_drawer_handle_')):
                body.remove(old)
        for item in specs:
            vec = lambda v: ' '.join(f'{x:.6f}' for x in v)  # noqa: E731
            ET.SubElement(body, 'geom', name=item['name'], type='box', pos=vec(item['center_m']),
                          size=vec(np.asarray(item['size_m']) / 2), rgba=vec(item['rgb'] + [1]),
                          contype='0', conaffinity='0', mass='0')
        ET.indent(tree, '  '); tree.write(path)
    for name in ('model.urdf', 'model.nophysics.urdf'):
        path = track_dir / name
        if not path.exists():
            continue
        tree = ET.parse(path); body = next(b for b in tree.getroot().findall('link') if b.get('name') == 'base')
        for old in list(body.findall('visual')):
            if old.get('name', '').startswith(('inferred_drawer_front_', 'inferred_drawer_handle_')):
                body.remove(old)
        for item in specs:
            vec = lambda v: ' '.join(f'{x:.6f}' for x in v)  # noqa: E731
            visual = ET.SubElement(body, 'visual', name=item['name'])
            ET.SubElement(visual, 'origin', xyz=vec(item['center_m']), rpy='0 0 0')
            ET.SubElement(ET.SubElement(visual, 'geometry'), 'box', size=vec(item['size_m']))
            ET.SubElement(ET.SubElement(visual, 'material', name=item['name'] + '_material'), 'color', rgba=vec(item['rgb'] + [1]))
        ET.indent(tree, '  '); tree.write(path, xml_declaration=True, encoding='unicode')
    return specs


def category_joint_mismatch(category: str, root: ET.Element) -> str | None:
    """Avoid applying a hinged template to sliders or a drawer template to hinges."""
    joints = [j.get("type", "hinge") for body in root.iter("body") for j in body.findall("joint")]
    if not joints:
        return None  # A static fitted object may still have a recognizable visual.
    if category in ("laptop", "book", "box") and "hinge" not in joints:
        return f"{category} classification has no fitted hinge; the fitted visual was retained"
    if category == "drawer" and "slide" not in joints:
        return "drawer classification has no fitted sliding joint; the fitted visual was retained"
    return None


def complete_geometry(track_dir: Path, appearance: dict, log=print) -> dict:
    """Add category-specific visual meshes, retaining the original collision meshes.

    Laptops receive a conservative bevel on the measured base and lid. Drawer models
    already include a hypothesized carcass and open tray from lidar.fit_and_build_slides.
    Other categories retain their fitted geometry; they are not silently called reconstructed.
    """
    category = category_of(appearance)
    report = {"category": category, "inferred_visuals": [], "source": "object category from video",
              "measured_geometry_preserved": True}
    root = ET.parse(track_dir / "model.xml").getroot()
    if (track_dir / "static.json").exists():
        static = json.loads((track_dir / "static.json").read_text())
        report["shape"] = ("The stationary model is a simple approximation of observed depth; "
                           "its hidden surfaces and exact object shape are not measured.")
        if category in ("laptop", "drawer", "book"):
            report["classification_mismatch"] = ("The object is represented as stationary; "
                                                 "record its moving parts and fit them separately if needed.")
            if category == 'drawer' and static.get('shape') == 'box':
                envelope = complete_static_drawer_envelope(track_dir)
                report['inferred_visuals'] = ([envelope] if envelope else []) + complete_static_drawer(track_dir, appearance)
                if report['inferred_visuals']:
                    report['shape'] += ' Drawer fronts are inferred on the most viewed side; that side may not be the front.'
                    report['static_drawer_side_assumed'] = True
                    if envelope:
                        report['static_drawer_depth_assumed_mm'] = envelope['inferred_depth_mm']
                        report['shape'] += ' Cabinet depth is an assumed visual because the fitted box was very thin.'
                    if any(item.get('count_assumed') for item in report['inferred_visuals']):
                        report['drawer_count_assumed'] = 3
        elif category == "box" and static.get("shape") == "cylinder":
            report["classification_mismatch"] = ("A box was classified but the static fit is cylindrical; "
                                                 "inspect the video and select a matching shape.")
        (track_dir / "shape_inference.json").write_text(json.dumps(report, indent=2) + "\n")
        log(report["shape"])
        return report
    mismatch = category_joint_mismatch(category, root)
    model_path = track_dir / "model3d.json"
    invalid_geometry = False
    if category == "box" and model_path.exists():
        thickness = json.loads(model_path.read_text()).get("base_thickness_mm")
        if isinstance(thickness, (int, float)) and (not math.isfinite(thickness) or thickness <= 0):
            mismatch = "box base thickness is invalid; refit before constructing a box visual"
            invalid_geometry = True
    if mismatch:
        report["shape"] = mismatch
        report["invalid_geometry" if invalid_geometry else "classification_mismatch"] = mismatch
        (track_dir / "shape_inference.json").write_text(json.dumps(report, indent=2) + "\n")
        log(mismatch)
        return report
    if category == "drawer":
        report["shape"] = ("The fitted drawer builder completes a carcass and tray behind the observed front; "
                           "panel thickness and hidden depth are assumptions, not measured surfaces.")
        report["inferred_visuals"] = complete_drawer_fronts(track_dir, appearance)
        if report["inferred_visuals"]:
            count = sum(s["name"].startswith("inferred_drawer_front_") and not s.get("link") for s in report["inferred_visuals"])
            assumed = any(s.get("count_assumed") for s in report["inferred_visuals"])
            report["shape"] += (f" {count} extra closed fronts and any handles are inferred; "
                                "their positions and surfaces are hypothetical.")
            if assumed:
                report["shape"] += " The drawer count was uncertain; three were assumed for the visual."
                report["drawer_count_assumed"] = 3
    elif category in ("laptop", "book", "box"):
        report["shape"] = ("The laptop base and lid are completed as straight-edged panels from the fitted "
                           "bounds and visible face; measured collisions are preserved."
                           if category == "laptop" else
                           "A box-like base and rectangular lid are inferred from fitted bounds and the hinge; "
                           "unseen surfaces are hypothetical and fitted collision geometry is preserved."
                           if category == "box" else
                           "A rectangular book cover is inferred from its visible inner face and hinge; "
                           "unseen corners are hypothetical and fitted collision geometry is preserved.")
        src = track_dir / "model.xml"
        root = ET.parse(src).getroot()
        assets = {m.get("name"): m.get("file") for m in root.findall("./asset/mesh")}
        if category == "laptop":
            base_body = next((b for b in root.iter("body") if b.get("name") == "base"), None)
            bounds = _mesh_bounds(track_dir, base_body, assets) if base_body is not None else None
            if bounds is not None:
                base_lo, base_hi = bounds
                base_width = base_hi[0] - base_lo[0]
                base_depth = base_hi[1] - base_lo[1]
                if min(base_width, base_depth) > 0 and base_hi[2] - base_lo[2] > .15 * min(base_width, base_depth):
                    report["visual_thickness_assumed"] = "laptop base visual thickness reduced to 8% of its smaller footprint"
        candidates = []
        height_hint = None
        if category == 'laptop' and (track_dir / 'meshes/lid.stl').exists() and (track_dir / 'meshes/screen.stl').exists():
            lid_body = next((b for b in root.iter('body') if b.get('name') == 'lid'), None)
            joint = lid_body.find('joint') if lid_body is not None else None
            if joint is not None:
                axis = np.fromstring(joint.get('axis', '0 0 1'), sep=' ')
                anchor = np.fromstring(joint.get('pos', '0 0 0'), sep=' ')
                if axis.size == 3 and anchor.size == 3 and np.allclose(axis, [1, 0, 0]) and np.allclose(anchor, 0):
                    height_hint = laptop_lid_height_hint(read_stl(track_dir / 'meshes/lid.stl'))
        for body in root.iter("body"):
            if body.get("name") not in ("base", "lid"):
                continue
            if category == "book" and body.get("name") != "lid":
                continue
            for geom in body.findall("geom"):
                name = geom.get("mesh", "")
                if name.endswith("_infer") or geom.get("group") == "3":
                    continue
                old = assets.get(name, "")
                stl = _stl_file(old)
                is_screen = "screen" in Path(stl).stem.lower()
                if not stl.endswith(".stl") or not (track_dir / stl).exists() or (is_screen and category != "laptop"):
                    continue
                new = str(Path(stl).with_name(Path(stl).stem + "_infer.stl"))
                try:
                    screen_pts = read_stl(track_dir / stl) if is_screen and category == "laptop" else None
                    lid_path = track_dir / 'meshes/lid.stl'
                    inset = .04 if screen_pts is not None and lid_path.exists() and screen_needs_bezel(screen_pts, read_stl(lid_path)) else 0.
                    width_hint = None
                    if category == 'laptop' and body.get('name') == 'base' and lid_path.exists():
                        lid_body = next((b for b in body.findall('body') if b.get('name') == 'lid'), None)
                        joint = lid_body.find('joint') if lid_body is not None else None
                        if (joint is not None and np.allclose(np.fromstring(joint.get('axis', '0 0 1'), sep=' '), [1, 0, 0])
                                and np.allclose(np.fromstring(lid_body.get('pos', '0 0 0'), sep=' '), [0, 0, 0])
                                and np.allclose(np.fromstring(joint.get('pos', '0 0 0'), sep=' '), [0, 0, 0])
                                and not any(lid_body.get(k) for k in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis'))):
                            width_hint = laptop_base_width_hint(read_stl(track_dir / stl), read_stl(lid_path))
                    vertices, faces = (book_cover_mesh(track_dir) if category == "book" else
                                       laptop_screen_visual(screen_pts, inset=inset) if is_screen else
                                       laptop_base_visual(read_stl(track_dir / stl), regularize_thickness=category == "laptop", target_x_bounds=width_hint) if body.get("name") == "base" else
                                       book_cover_mesh(track_dir, outer_blend=1.) if (track_dir / "meshes/screen.stl").exists() else
                                       bevel_mesh(read_stl(track_dir / stl)))
                    if height_hint and body.get('name') == 'lid':
                        direction, factor = height_hint
                        vertices = vertices + np.outer(vertices @ direction, direction) * (factor - 1.)
                        report['visual_lid_height_assumed'] = f'lid visual height scaled by {factor:.3f}'
                except ValueError as error:
                    log(f"Could not infer {body.get('name')} visual: {error}")
                    continue
                write_stl(track_dir / new, vertices, faces)
                if inset:
                    report['screen_bezel_assumed'] = True
                if width_hint:
                    report['visual_alignment_assumed'] = 'base visual width extended to aligned fitted lid width'
                candidates.append((body.get("name"), name, stl, new))
        # On repeat, keep the existing inferred assets instead of adding another layer.
        if not candidates:
            candidates = [(b.get("name"), g.get("mesh")[:-6],
                           _stl_file(assets.get(g.get("mesh")[:-6], "")),
                           _stl_file(assets.get(g.get("mesh"), "")))
                          for b in root.iter("body") for g in b.findall("geom")
                          if g.get("mesh", "").endswith("_infer")]
        else:
            for filename in ("model.xml", "model.nophysics.xml"):
                path = track_dir / filename
                if not path.exists():
                    continue
                tree = ET.parse(path)
                asset = tree.getroot().find("asset")
                for body_name, name, old, new in candidates:
                    body = next((b for b in tree.getroot().iter("body") if b.get("name") == body_name), None)
                    if body is None:
                        continue
                    geom = next((g for g in body.findall("geom") if g.get("mesh") == name), None)
                    original = next((m for m in asset.findall("mesh") if m.get("name") == name), None)
                    if geom is None or original is None:
                        raise ValueError(f"missing {body_name}/{name} in {filename}")
                    original.set("file", old)
                    ET.SubElement(asset, "mesh", name=name + "_infer", file=new, inertia="exact")
                    visual = copy.deepcopy(geom)
                    visual.set("mesh", name + "_infer")
                    visual.set("contype", "0")
                    visual.set("conaffinity", "0")
                    visual.set("mass", "0")
                    visual.attrib.pop("group", None)
                    visual.attrib.pop("material", None)
                    geom.set("group", "3")
                    geom.attrib.pop("material", None)
                    geom.set("rgba", "0.65 0.65 0.65 1")
                    body.append(visual)
                ET.indent(tree, "  ")
                tree.write(path)
            for filename in ("model.urdf", "model.nophysics.urdf"):
                path = track_dir / filename
                if not path.exists():
                    continue
                tree = ET.parse(path)
                for body_name, name, old, new in candidates:
                    link = next((l for l in tree.getroot().findall("link") if l.get("name") == body_name), None)
                    if link is None:
                        continue
                    for visual in link.findall("visual"):
                        mesh = visual.find("./geometry/mesh")
                        if mesh is not None and _stl_file(mesh.get("filename", "")) == old:
                            mesh.set("filename", new)
                ET.indent(tree, "  ")
                tree.write(path, xml_declaration=True, encoding="unicode")
        report["inferred_visuals"] = [{"link": b, "source": old, "generated": new,
                                       **({"completion": "rectangular panel inferred from fitted bounds and visible face"} if category in ("laptop", "box")
                                          else {"basis": "visible cover face and hinge"})}
                                      for b, _, old, new in candidates]
        if category == 'laptop' and candidates:
            if report.get('screen_bezel_assumed'):
                report['shape'] += ' A screen bezel is inferred for appearance.'
            if report.get('visual_lid_height_assumed'):
                report['shape'] += ' An unusually tall lid visual is shortened heuristically.'
            hinge = complete_laptop_hinge(track_dir)
            if hinge:
                report['inferred_visuals'].append(hinge)
                report['shape'] += ' A small visual hinge connector is estimated from the fitted axis and base bounds.'
        if not candidates:
            report["shape"] = f"No compatible fitted {category} meshes found; fitted geometry retained."
    else:
        report["shape"] = "No object-specific shape template available; fitted geometry retained."
    (track_dir / "shape_inference.json").write_text(json.dumps(report, indent=2) + "\n")
    log(report["shape"])
    return report
