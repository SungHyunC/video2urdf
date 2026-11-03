"""URDF collision shapes that clear each other at a planar joint.

model.xml excludes contacts between a body and its parent (emit3d.contact_excludes): a fitted
part's outline overlaps its parent's at the hinge (the masks meet there) and, at the end of its
measured range, often by a few mm where it closes onto it. A URDF can't say that: MuJoCo's URDF
import fuses the root link into the world and never filters contacts with the world, so a part
jointed to the root collides with it (the 2D laptop's lid started 13 mm inside its base and was
thrown 65 deg). Deeper in the tree MuJoCo filters parent-child contacts itself, as PyBullet and
Drake do everywhere, so only joints whose parent is the root (or welded to it) are looked at.

For a planar model (each link's collision a profile extruded along y, its joint turning about
y or sliding in the x-z plane) this trims one of the two collision shapes, in the URDF only, so
that over the joint's whole measured range they stay CLEARANCE apart:

- A simulator collides each mesh as its convex hull, so each link is taken as the hulls of its
  collision meshes, and the trimmed shape is written as convex pieces (one mesh each), whose
  hulls are the pieces themselves.
- The other link swept through the joint's range is rasterised (RES), grown by CLEARANCE and
  cut out of the trimmed link; the result is outlined, triangulated, clipped exactly to the
  original hull (it never grows) and merged back into as few convex pieces as stay convex.
- The link trimmed is the one that loses the smaller share of its area; nothing is done when
  the two don't overlap by more than TRIGGER anywhere in the range, or when it would remove more
  than MAX_REMOVED of the link.

Visuals, model.xml, joints and inertials are untouched (each <inertial> is written out, so no
loader recomputes it from the trimmed shape unless told to ignore the file's inertia).
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mapbox_earcut
import numpy as np

from .shape import extrude, write_stl
from .urdf_checks import Joint, _geometry

CLEARANCE = 0.0005  # m kept between the trimmed shape and the other link anywhere in the range
TRIGGER = 0.0005  # m: overlaps thinner than this are left alone
RES = 5e-5  # m per raster pixel
EPS_PX = 3  # outline simplification (px); the free region is eroded by as much first
STEP_RAD = 0.004  # sweep step of a hinge (the swept hull bulges r * step^2 / 8 between steps)
STEP_M = 0.001  # sweep step of a slide
MAX_REMOVED = 0.35  # never trim away more than this share of a link's area
MIN_PIECE_M2 = 5e-8  # convex pieces smaller than 0.05 mm^2 are dropped
Y_TOL = 1e-5  # m: a vertex is on an extrusion face when this close to it
SUFFIX = "_clear_"  # collision names (and mesh stems) of trimmed pieces: <link>_clear_<i>


# ------------------------------------------------------------------ 2D geometry

def hull(points: np.ndarray) -> np.ndarray:
    """Convex hull (counter-clockwise, no repeated end point) of (N, 2) points, float64."""
    p = np.unique(np.asarray(points, float), axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        out = []
        for q in pts:
            while len(out) >= 2 and _cross(out[-2], out[-1], q) <= 0:
                out.pop()
            out.append(q)
        return out
    lower, upper = half(p), half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def area(poly: np.ndarray) -> float:
    """Signed area (positive counter-clockwise)."""
    p = np.asarray(poly, float)
    if len(p) < 3:
        return 0.0
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: ``subject`` (convex) inside the convex, counter-clockwise ``clip``."""
    out = np.asarray(subject, float)
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        if len(out) == 0:
            break
        inp, out = out, []
        side = [_cross(a, b, p) for p in inp]
        for k in range(len(inp)):
            p, q = inp[k], inp[(k + 1) % len(inp)]
            sp, sq = side[k], side[(k + 1) % len(inp)]
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append(p + t * (q - p))
        out = np.array(out) if out else np.zeros((0, 2))
    return out


def merge_convex(pieces: list[np.ndarray], tol: float = 1e-11) -> list[np.ndarray]:
    """Greedily merge convex pieces whose union is convex (the hull adds no area)."""
    pieces = [hull(p) for p in pieces]
    merged = True
    while merged:
        merged = False
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                h = hull(np.vstack([pieces[i], pieces[j]]))
                if area(h) <= area(pieces[i]) + area(pieces[j]) + tol:
                    pieces[i] = h
                    del pieces[j]
                    merged = True
                    break
            if merged:
                break
    return pieces


# ------------------------------------------------------------------ planar model

@dataclass
class PlanarLink:
    name: str
    pieces: list[np.ndarray]  # convex (x, z) polygons in the link frame (the hulls loaders collide)
    y: tuple[float, float]  # extrusion range
    names: list[str] = field(default_factory=list)  # the collision elements' names


def _rot(theta: float) -> np.ndarray:
    """Rotation about +y by theta, acting on (x, z)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, s], [-s, c]])


def read_mesh(path: Path) -> np.ndarray:
    """Vertices (N, 3) of an STL or OBJ mesh."""
    if path.suffix.lower() == ".stl":
        from .portable import read_stl
        return read_stl(path).reshape(-1, 3)
    if path.suffix.lower() == ".obj":
        return np.array([[float(x) for x in line.split()[1:4]] for line in path.read_text().splitlines()
                         if line.startswith("v ")])
    raise ValueError(f"{path}: not an STL or OBJ")


def _geometry_vertices(folder: Path, el: ET.Element) -> np.ndarray | None:
    """A <collision>'s vertices in its link frame (None for shapes this module doesn't handle)."""
    g = _geometry(el)
    if g.kind == "mesh" and g.filename and not g.filename.startswith(("package://", "file://", "model://")):
        p = folder / g.filename
        if not p.is_file():
            return None
        v = read_mesh(p) * g.scale
    elif g.kind == "box":
        v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.array(g.size) / 2
    else:
        return None
    return v @ g.origin[:3, :3].T + g.origin[:3, 3]


def planar_link(folder: Path, link: ET.Element) -> PlanarLink | None:
    """The link's collisions as convex (x, z) profiles, when every one is an extrusion along y
    over the same range; else None."""
    cols = link.findall("collision")
    if not cols:
        return None
    pieces, ys = [], []
    for c in cols:
        v = _geometry_vertices(folder, c)
        if v is None or len(v) < 6:
            return None
        lo, hi = float(v[:, 1].min()), float(v[:, 1].max())
        if hi - lo < 10 * Y_TOL or not np.all((np.abs(v[:, 1] - lo) < Y_TOL) | (np.abs(v[:, 1] - hi) < Y_TOL)):
            return None
        h = hull(v[:, [0, 2]])
        if len(h) < 3 or area(h) <= 0:
            return None
        pieces.append(h)
        ys.append((lo, hi))
    if max(abs(a - ys[0][0]) + abs(b - ys[0][1]) for a, b in ys) > 2 * Y_TOL:
        return None
    return PlanarLink(link.get("name"), pieces, ys[0], [c.get("name") or "" for c in cols])


def planar_joint(j: Joint):
    """(A, b, motion) mapping child (x, z) to parent (x, z) at q: A @ M(q) p + b, when the joint
    keeps y (origin rotation about y, y offset 0, hinge about +-y or slide in x-z); else None."""
    T = j.origin
    if abs(T[1, 1] - 1) > 1e-9 or abs(T[1, 3]) > 1e-9 or j.type not in ("revolute", "continuous", "prismatic"):
        return None
    A = np.array([[T[0, 0], T[0, 2]], [T[2, 0], T[2, 2]]])
    b = np.array([T[0, 3], T[2, 3]])
    a = j.axis / np.linalg.norm(j.axis)
    if j.type == "prismatic":
        if abs(a[1]) > 1e-9:
            return None
        return A, b, ("slide", np.array([a[0], a[2]]))
    if abs(abs(a[1]) - 1) > 1e-9:
        return None
    return A, b, ("hinge", float(np.sign(a[1])))


def joint_values(j: Joint, motion) -> np.ndarray:
    lo, hi = (j.lower, j.upper) if j.type != "continuous" else (-math.pi, math.pi)
    step = STEP_RAD if motion[0] == "hinge" else STEP_M
    return np.linspace(lo, hi, max(2, int(math.ceil((hi - lo) / step)) + 1))


def child_to_parent(A, b, motion, q: float):
    """2x2 matrix and offset taking child (x, z) to parent (x, z) at joint value q."""
    if motion[0] == "hinge":
        return A @ _rot(motion[1] * q), b
    return A, A @ (q * motion[1]) + b


# ------------------------------------------------------------------ raster

class Grid:
    def __init__(self, polys: list[np.ndarray], margin: float):
        pts = np.vstack(polys)
        self.org = pts.min(0) - margin
        size = np.ceil((pts.max(0) + margin - self.org) / RES).astype(int) + 1
        self.shape = (int(size[1]), int(size[0]))  # rows = z, cols = x

    def empty(self) -> np.ndarray:
        return np.zeros(self.shape, np.uint8)

    def fill(self, img: np.ndarray, poly: np.ndarray):
        px = np.round((np.asarray(poly) - self.org) / RES * 16).astype(np.int32)
        cv2.fillPoly(img, [px], 1, lineType=cv2.LINE_8, shift=4)

    def to_metric(self, px: np.ndarray) -> np.ndarray:
        return np.asarray(px, float) * RES + self.org


def swept(grid: Grid, movers: list[np.ndarray], maps) -> np.ndarray:
    """Raster of ``movers`` placed by each of ``maps`` (a list of (R, t)), with the hull of each
    pair of consecutive placements filled so nothing between steps is missed."""
    img = grid.empty()
    placed = [[p @ R.T + t for p in movers] for R, t in maps]
    for k in range(len(placed)):
        for i, p in enumerate(placed[k]):
            pts = p if k + 1 == len(placed) else np.vstack([p, placed[k + 1][i]])
            grid.fill(img, hull(pts))
    return img.astype(bool)


def _thickness(mask: np.ndarray) -> float:
    """Twice the deepest point of a mask (m): about how thick the overlap is."""
    if not mask.any():
        return 0.0
    d = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)
    return 2 * float(d.max()) * RES


def _free_polygons(free: np.ndarray, grid: Grid) -> list[np.ndarray]:
    """Triangles covering a free-space raster (outer outlines with their holes), metric."""
    contours, hier = cv2.findContours(free.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    tris = []
    for i, c in enumerate(contours):
        if hier[i][3] != -1:
            continue  # a hole: handled with its outline
        rings = [c]
        k = hier[i][2]
        while k != -1:
            rings.append(contours[k])
            k = hier[k][0]
        polys = []
        for r in rings:
            a = cv2.approxPolyDP(r, EPS_PX, True)[:, 0, :]
            if len(a) >= 3:
                polys.append(grid.to_metric(a))
        if not polys or abs(area(polys[0])) <= 0:
            continue
        verts = np.vstack(polys)
        ends = np.cumsum([len(p) for p in polys]).astype(np.uint32)
        idx = mapbox_earcut.triangulate_float64(verts, ends).reshape(-1, 3)
        tris += [verts[t] for t in idx]
    return tris


@dataclass
class Trim:
    link: str  # the link trimmed
    other: str  # the link swept through it
    joint: str
    pieces: list[np.ndarray]  # convex (x, z) pieces in the link frame
    y: tuple[float, float]
    area_before: float  # m^2 (its collision hulls)
    area_after: float
    depth: float  # m: deepest the cut goes below the original outline
    overlap: float  # m: how thick the overlap was before
    removed_x: tuple[float, float]  # extent of the removed area, link frame (m)
    removed_z: tuple[float, float]
    files: list[str] = field(default_factory=list)

    @property
    def removed_share(self) -> float:
        return 1 - self.area_after / self.area_before

    def summary(self) -> str:
        return (f"{self.link}: trimmed where {self.other} swings through it ({self.joint}): "
                f"{1e6 * (self.area_before - self.area_after):.0f} of {1e6 * self.area_before:.0f} mm^2 "
                f"({100 * self.removed_share:.1f}%), at most {1000 * self.depth:.1f} mm deep, "
                f"x {1000 * self.removed_x[0]:.1f}..{1000 * self.removed_x[1]:.1f} mm, "
                f"z {1000 * self.removed_z[0]:.1f}..{1000 * self.removed_z[1]:.1f} mm in its frame; "
                f"overlap before {1000 * self.overlap:.1f} mm, {len(self.pieces)} convex pieces")


def overlap(target: list[np.ndarray], movers: list[np.ndarray], maps) -> float:
    """How thick the overlap of ``target`` with ``movers`` swept by ``maps`` is (m)."""
    grid = Grid(target, 4 * RES)
    t = grid.empty()
    for p in target:
        grid.fill(t, p)
    return _thickness(t.astype(bool) & swept(grid, movers, maps))


def trim(target: PlanarLink, mover: PlanarLink, maps, joint: str, clearance: float = CLEARANCE) -> Trim | None:
    """``target`` with ``mover`` (placed by each of ``maps``, mover -> target frame) cut out,
    ``clearance`` around it; None if nothing would be left."""
    if min(target.y[1], mover.y[1]) <= max(target.y[0], mover.y[0]):
        return None  # side by side along y: they never touch
    margin = clearance + (EPS_PX + 4) * RES
    grid = Grid(target.pieces, margin)
    tmask = grid.empty()
    for p in target.pieces:
        grid.fill(tmask, p)
    tmask = tmask.astype(bool)
    hit = swept(grid, mover.pieces, maps)
    before = _thickness(tmask & hit)
    r = int(math.ceil(clearance / RES))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    keep_out = cv2.dilate(hit.astype(np.uint8), disk) > 0
    free = cv2.erode((~keep_out).astype(np.uint8), np.ones((2 * EPS_PX + 1, 2 * EPS_PX + 1), np.uint8))
    free[[0, -1], :] = 0
    free[:, [0, -1]] = 0
    tris = _free_polygons(free, grid)
    pieces = []
    for orig in target.pieces:
        for t in tris:
            c = clip_convex(t if area(t) > 0 else t[::-1], orig)
            if len(c) >= 3 and area(c) > 1e-14:
                pieces.append(c)
    pieces = [p for p in merge_convex(pieces) if area(p) >= MIN_PIECE_M2]
    if not pieces:
        return None
    after = grid.empty()
    for p in pieces:
        grid.fill(after, p)
    removed = tmask & ~after.astype(bool)
    # Seams: pixels on a shared edge of two pieces that neither fill. Not removed area.
    removed = cv2.morphologyEx(removed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    depth = 0.0
    rx = rz = (0.0, 0.0)
    if removed.any():
        d = cv2.distanceTransform(np.pad(tmask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
        depth = float(d[removed].max()) * RES
        zs, xs = np.nonzero(removed)
        rx = (float(xs.min() * RES + grid.org[0]), float(xs.max() * RES + grid.org[0]))
        rz = (float(zs.min() * RES + grid.org[1]), float(zs.max() * RES + grid.org[1]))
    return Trim(target.name, mover.name, joint, pieces, target.y, sum(area(p) for p in target.pieces),
                sum(area(p) for p in pieces), depth, before, rx, rz)


def _inverse(maps):
    out = []
    for R, t in maps:
        Ri = np.linalg.inv(R)
        out.append((Ri, -Ri @ t))
    return out


def plan(folder: Path, root: ET.Element, clearance: float = CLEARANCE) -> list[Trim]:
    """The trims a URDF (parsed, its meshes under ``folder``) needs, one per overlapping planar
    joint, in joint order; each later joint sees the earlier trims."""
    links = {l.get("name"): l for l in root.findall("link")}
    shapes = {n: planar_link(folder, l) for n, l in links.items()}
    welded = welded_to_root(root)
    out = []
    for el in root.findall("joint"):
        j = _joint(el)
        if j is None or j.parent not in shapes or j.child not in shapes or j.parent not in welded:
            continue
        parent, child = shapes[j.parent], shapes[j.child]
        geo = planar_joint(j)
        if parent is None or child is None or geo is None:
            continue
        if any(SUFFIX in n for n in parent.names + child.names):
            continue  # trimmed by an earlier export
        A, b, motion = geo
        maps = [child_to_parent(A, b, motion, q) for q in joint_values(j, motion)]
        if overlap(parent.pieces, child.pieces, maps) <= TRIGGER:
            continue
        options = [t for t in (trim(parent, child, maps, j.name, clearance),
                               trim(child, parent, _inverse(maps), j.name, clearance)) if t is not None]
        options = [t for t in options if t.removed_share <= MAX_REMOVED]
        if not options:
            continue
        best = min(options, key=lambda t: t.removed_share)
        shapes[best.link] = PlanarLink(best.link, best.pieces, best.y, [f"{best.link}{SUFFIX}{i}" for i in range(len(best.pieces))])
        out.append(best)
    return out


def welded_to_root(root: ET.Element) -> set[str]:
    """The root link and the links joined to it by fixed joints only: what MuJoCo's URDF import
    fuses into the world body, whose contacts it never filters."""
    joints = [(j.find("parent"), j.find("child"), j.get("type")) for j in root.findall("joint")]
    joints = [(p.get("link"), c.get("link"), t) for p, c, t in joints if p is not None and c is not None]
    children = {c for _, c, _ in joints}
    out = {l.get("name") for l in root.findall("link") if l.get("name") not in children}
    grown = True
    while grown:
        grown = False
        for p, c, t in joints:
            if t == "fixed" and p in out and c not in out:
                out.add(c)
                grown = True
    return out


def _joint(el: ET.Element) -> Joint | None:
    from .urdf_checks import _floats, _origin
    parent, child, lim = el.find("parent"), el.find("child"), el.find("limit")
    if parent is None or child is None:
        return None
    axis = el.find("axis")
    j = Joint(el.get("name"), el.get("type"), parent.get("link"), child.get("link"), _origin(el),
              _floats(axis.get("xyz") if axis is not None else None, 3, [1, 0, 0]))
    if lim is not None and lim.get("lower") is not None and lim.get("upper") is not None:
        j.lower, j.upper = float(lim.get("lower")), float(lim.get("upper"))
    elif j.type != "continuous":
        return None
    return j


def apply(folder: Path, root: ET.Element, trims: list[Trim], mesh_dir: str = "meshes") -> list[str]:
    """Write each trim's pieces as STL meshes (``<mesh_dir>/<link>_clear_<i>.stl``) and make them
    the link's <collision> elements. Returns the files written (relative to ``folder``)."""
    files = []
    links = {l.get("name"): l for l in root.findall("link")}
    for t in trims:
        link = links[t.link]
        old = link.findall("collision")
        at = list(link).index(old[0])
        for c in old:
            link.remove(c)
        y0, y1 = t.y
        for i, p in enumerate(t.pieces):
            rel = f"{mesh_dir}/{t.link}{SUFFIX}{i}.stl"
            v, f = extrude(p if area(p) > 0 else p[::-1], y1 - y0)
            v[:, 1] += (y0 + y1) / 2
            (folder / rel).parent.mkdir(parents=True, exist_ok=True)
            write_stl(folder / rel, v, f)
            c = ET.Element("collision", {"name": f"{t.link}{SUFFIX}{i}"})
            ET.SubElement(c, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            ET.SubElement(ET.SubElement(c, "geometry"), "mesh", {"filename": rel})
            link.insert(at + i, c)
            files.append(rel)
        t.files = files[-len(t.pieces):]
    return files


def trimmed_links(root: ET.Element) -> list[str]:
    """Links whose collisions are trimmed pieces."""
    return [l.get("name") for l in root.findall("link")
            if any(re.search(re.escape(SUFFIX) + r"\d+$", c.get("name") or "") for c in l.findall("collision"))]


def clear_urdf(folder: Path, text: str, clearance: float = CLEARANCE) -> tuple[str, list[Trim], list[str]]:
    """The URDF text (no header comment) with planar joints' overlapping collisions trimmed, the
    trims, and the mesh files written. Unchanged text when nothing overlaps."""
    root = ET.fromstring(text)
    trims = plan(folder, root, clearance)
    if not trims:
        return text, [], []
    files = apply(folder, root, trims)
    return ET.tostring(root, encoding="unicode"), trims, files
