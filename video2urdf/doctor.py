"""URDF Doctor: check an existing URDF's hinge against one measured from video.

The measured model (a Video2URDF run: model.xml and hinge.json) knows how the real object
moves. The URDF under test was written some other way (by hand, from CAD, by a vendor) in its
own frame. The doctor

1. registers the URDF's base to the measured base by geometry alone (principal axes, four
   proper orientations, then ICP from the measured points onto the URDF's surface), so the
   joint under test plays no part in the alignment;
2. for openings seen in the video, finds the URDF joint value whose child best covers the
   measured child (a 1-D search over the whole circle, ignoring the URDF's limits); of the
   four orientations, the one whose child fits best wins;
3. reports what disagrees: the axis direction and position, which sign of q opens it, the
   URDF's zero, whether its limits cover the motion in the video, and how well the shapes
   agree. With a joint log (values a robot or sim recorded for the video's frames) it also
   finds a sign flip or an offset in the log.

One revolute joint between a fixed base and one child, as fit_lidar measures.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from .lift3d import kabsch

SURFACE_STEP_M = 0.01  # spacing of points sampled on box faces
AXIS_TOL_DEG = 3.0
OFFSET_TOL_M = 0.01
LIMIT_TOL_DEG = 3.0
SHAPE_TOL_M = 0.01
REFINE_ROUNDS = 8


@dataclass
class Joint:
    name: str
    parent: int  # body ids
    child: int
    jid: int
    lo: float
    hi: float
    limited: bool


def load(path: Path) -> mujoco.MjModel:
    """A URDF or MJCF. URDF mesh paths are taken relative to the file."""
    if path.suffix.lower() == ".urdf":
        text = path.read_text()
        if "<mujoco" not in text:  # point MuJoCo's URDF reader at the meshes and keep visuals
            i = text.index(">", text.index("<robot")) + 1
            text = text[:i] + f'<mujoco><compiler meshdir="{path.parent}/" strippath="false" ' \
                              f'discardvisual="false" fusestatic="false" inertiafromgeom="auto"/></mujoco>' + text[i:]
        return mujoco.MjModel.from_xml_string(text)
    return mujoco.MjModel.from_xml_path(str(path))


def hinge(model: mujoco.MjModel) -> Joint:
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    if len(hinges) != 1:
        raise ValueError(f"expected one revolute joint, found {len(hinges)}")
    j = hinges[0]
    child = int(model.jnt_bodyid[j])
    return Joint(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}",
                 int(model.body_parentid[child]), child, j, float(model.jnt_range[j][0]),
                 float(model.jnt_range[j][1]), bool(model.jnt_limited[j]))


def _box_surface(half: np.ndarray) -> np.ndarray:
    pts = []
    for ax in range(3):
        u, v = [a for a in range(3) if a != ax]
        gu = np.arange(-half[u], half[u] + 1e-9, SURFACE_STEP_M) if half[u] > 0 else np.array([0.0])
        gv = np.arange(-half[v], half[v] + 1e-9, SURFACE_STEP_M) if half[v] > 0 else np.array([0.0])
        uu, vv = np.meshgrid(gu, gv)
        for sgn in (-1, 1):
            p = np.zeros((uu.size, 3))
            p[:, u], p[:, v], p[:, ax] = uu.ravel(), vv.ravel(), sgn * half[ax]
            pts.append(p)
    return np.vstack(pts)


def _mesh_surface(model: mujoco.MjModel, m: int) -> np.ndarray:
    """Points spread over a mesh's triangles about SURFACE_STEP_M apart (its vertices alone
    only outline a flat part)."""
    v = model.mesh_vert[model.mesh_vertadr[m]: model.mesh_vertadr[m] + model.mesh_vertnum[m]]
    f = model.mesh_face[model.mesh_faceadr[m]: model.mesh_faceadr[m] + model.mesh_facenum[m]]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.maximum(1, np.round(area / SURFACE_STEP_M ** 2)).astype(int)
    rng = np.random.default_rng(0)
    idx = np.repeat(np.arange(len(f)), n)
    r1, r2 = rng.random(len(idx)), rng.random(len(idx))
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * a[idx] + (s * (1 - r2))[:, None] * b[idx] + (s * r2)[:, None] * c[idx]
    return np.vstack([v, pts])


def body_points(model: mujoco.MjModel, data: mujoco.MjData, bodies: list[int]) -> np.ndarray:
    """Surface points of the bodies' own geoms in world coordinates."""
    keep = set(bodies)
    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in keep:
            continue
        t = model.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            local = _mesh_surface(model, model.geom_dataid[g])
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            local = _box_surface(model.geom_size[g])
        else:  # cylinders, capsules, spheres: their bounding box is close enough here
            local = _box_surface(model.geom_rbound[g] * np.ones(3) * 0.7)
        out.append(local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    return np.vstack(out)


def at(model, data, joint: Joint, q: float):
    data.qpos[:] = 0
    data.qpos[model.jnt_qposadr[joint.jid]] = q
    mujoco.mj_kinematics(model, data)


def _pca(p: np.ndarray):
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    return c, vt  # rows: largest to smallest spread


def register(user_pts: np.ndarray, meas_pts: np.ndarray, iters: int = 30) -> list[np.ndarray]:
    """Four 4x4 transforms taking the URDF's base onto the measured base: its principal axes
    turned onto the measured ones in every proper way, each refined by ICP from the measured
    points (which may only show part of the base) onto the URDF's surface."""
    cu, vu = _pca(user_pts)
    cm, vm = _pca(meas_pts)
    tree = cKDTree(user_pts)
    out = []
    for s0, s1 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        su = np.array([s0 * vu[0], s1 * vu[1], np.cross(s0 * vu[0], s1 * vu[1])])
        sm = np.array([vm[0], vm[1], np.cross(vm[0], vm[1])])
        r = sm.T @ su  # user -> measured
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = r, cm - r @ cu
        for _ in range(iters):
            inv = np.linalg.inv(t)
            m_in_user = meas_pts @ inv[:3, :3].T + inv[:3, 3]
            d, i = tree.query(m_in_user)
            ok = d < max(np.percentile(d, 80), 0.002)
            step = kabsch(user_pts[i[ok]] @ t[:3, :3].T + t[:3, 3], meas_pts[ok])
            t = step @ t
        out.append(t)
    return out


def chamfer(a: np.ndarray, tree_b: cKDTree) -> float:
    return float(np.mean(tree_b.query(a)[0]))


@dataclass
class Diagnosis:
    joint: str
    findings: list[str] = field(default_factory=list)
    numbers: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.findings


def diagnose(urdf: Path, run: Path, joint_log: np.ndarray | None = None, openings_deg=None) -> Diagnosis:
    """Check ``urdf`` against the measured run folder (model.xml, hinge.json)."""
    um, mm = load(urdf), mujoco.MjModel.from_xml_path(str(run / "model.xml"))
    ud, md = mujoco.MjData(um), mujoco.MjData(mm)
    uj, mj = hinge(um), hinge(mm)
    h = json.loads((run / "hinge.json").read_text())

    # The measured joint's opening (0 = closed) from its q, whose sign follows the fitted axis.
    q_meas = np.array([np.nan if v is None else v for v in h["q"]])
    shut = np.deg2rad(h["opening_on_reference_deg"])
    toward = 1.0 if abs(np.nanmax(q_meas) - shut) < abs(np.nanmin(q_meas) + shut) else -1.0
    opening = shut - toward * q_meas  # per video frame
    seen = np.nanpercentile(opening, [1, 99])
    o_samples = np.deg2rad(openings_deg) if openings_deg is not None else np.linspace(seen[0], seen[1], 7)

    at(mm, md, mj, 0.0)
    meas_base = body_points(mm, md, [mj.parent])
    at(um, ud, uj, 0.0)
    user_base = body_points(um, ud, [uj.parent])
    cands = register(user_base, meas_base)

    grid = np.deg2rad(np.arange(-180, 180, 2.0))

    def best_q(t_reg, meas_child, q0=None):
        inv = np.linalg.inv(t_reg)
        target = meas_child @ inv[:3, :3].T + inv[:3, 3]  # measured child in the URDF's frame
        tt = cKDTree(target)

        def cost(q):
            at(um, ud, uj, q)
            return chamfer(body_points(um, ud, [uj.child]), tt)

        qs = grid if q0 is None else q0 + np.deg2rad(np.arange(-20, 21, 2.0))
        costs = [cost(q) for q in qs]
        q = qs[int(np.argmin(costs))]
        for step in np.deg2rad([1.0, 0.5, 0.25, 0.1]):  # refine
            q = min((q - step, q, q + step), key=cost)
        return float(q), cost(q)

    meas_children = []
    for o in o_samples:
        at(mm, md, mj, toward * (shut - o))
        meas_children.append(body_points(mm, md, [mj.child]))

    results = []
    for t_reg in cands:
        fits = [best_q(t_reg, c) for c in meas_children]
        results.append((sum(f[1] for f in fits), t_reg, fits))
    _, t_reg, fits = min(results, key=lambda r: r[0])

    # Refine the alignment on the base and the child at every sampled opening together: a flat
    # base pins its in-plane turn only loosely, and one rigid move can't make a wrong axis
    # match at all openings, so the joint's errors stay visible instead of being absorbed.
    base_tree = cKDTree(user_base)
    for _ in range(REFINE_ROUNDS):
        src, dst = [], []
        inv = np.linalg.inv(t_reg)
        d, i = base_tree.query(meas_base @ inv[:3, :3].T + inv[:3, 3])
        src.append(user_base[i]), dst.append(meas_base)
        for (q, _), mc in zip(fits, meas_children):
            at(um, ud, uj, q)
            uc = body_points(um, ud, [uj.child])
            d, i = cKDTree(uc).query(mc @ inv[:3, :3].T + inv[:3, 3])
            src.append(uc[i]), dst.append(mc)
        src, dst = np.vstack(src), np.vstack(dst)
        moved = src @ t_reg[:3, :3].T + t_reg[:3, 3]
        err = np.linalg.norm(moved - dst, axis=1)
        ok = err < np.percentile(err, 85)
        t_reg = kabsch(moved[ok], dst[ok]) @ t_reg
        fits = [best_q(t_reg, c, q0=q) for (q, _), c in zip(fits, meas_children)]
    q_user = np.unwrap([f[0] for f in fits])
    shape_err = float(np.median([f[1] for f in fits]))

    d = Diagnosis(uj.name)
    slope, q_at_closed = np.polyfit(o_samples, q_user, 1)
    opens_positive = slope > 0
    fit_res = float(np.sqrt(np.mean((np.polyval([slope, q_at_closed], o_samples) - q_user) ** 2)))

    # Axis: the URDF's axis and anchor, carried into the measured frame, against the measured.
    at(um, ud, uj, 0.0)
    at(mm, md, mj, 0.0)
    ua = t_reg[:3, :3] @ ud.xaxis[uj.jid]
    up = t_reg[:3, :3] @ ud.xanchor[uj.jid] + t_reg[:3, 3]
    ma, mp = md.xaxis[mj.jid].copy(), md.xanchor[mj.jid].copy()
    axis_err = float(np.rad2deg(np.arccos(min(1.0, abs(ua @ ma)))))
    child_mid = body_points(mm, md, [mj.child]).mean(axis=0)
    mid = mp + ((child_mid - mp) @ ma) * ma  # measure the offset at the middle of the child
    gap = (mid - up) - ((mid - up) @ ua) * ua
    offset = float(np.linalg.norm(gap))

    need = np.polyval([slope, q_at_closed], seen)
    lo_need, hi_need = float(min(need)), float(max(need))
    d.numbers = {
        "axis_direction_error_deg": axis_err, "axis_offset_mm": 1000 * offset,
        "opens_with_positive_q": bool(opens_positive), "q_per_opening": float(slope),
        "q_when_closed_deg": float(np.rad2deg(q_at_closed)),
        "opening_seen_deg": [float(np.rad2deg(s)) for s in seen],
        "q_needed_deg": [float(np.rad2deg(lo_need)), float(np.rad2deg(hi_need))],
        "urdf_limits_deg": [float(np.rad2deg(uj.lo)), float(np.rad2deg(uj.hi))] if uj.limited else None,
        "shape_mismatch_mm": 1000 * shape_err, "q_fit_residual_deg": float(np.rad2deg(fit_res)),
        "samples": [{"opening_deg": float(np.rad2deg(o)), "urdf_q_deg": float(np.rad2deg(q)), "mismatch_mm": 1000 * c}
                    for o, q, (_, c) in zip(o_samples, q_user, fits)],
    }
    if axis_err > AXIS_TOL_DEG:
        d.findings.append(f"axis direction is {axis_err:.1f} deg off the measured hinge")
    if offset > OFFSET_TOL_M:
        d.findings.append(f"hinge axis sits {1000 * offset:.0f} mm from the measured hinge "
                          f"(joint origin misplaced)")
    if abs(abs(slope) - 1) > 0.1:
        d.findings.append(f"q changes {abs(slope):.2f} rad per rad of real opening (expected 1): the "
                          f"axis or the child's geometry is wrong")
    if uj.limited:
        short = []
        if lo_need < uj.lo - np.deg2rad(LIMIT_TOL_DEG):
            short.append(f"needs {np.rad2deg(lo_need):.0f} deg, lower limit is {np.rad2deg(uj.lo):.0f}")
        if hi_need > uj.hi + np.deg2rad(LIMIT_TOL_DEG):
            short.append(f"needs {np.rad2deg(hi_need):.0f} deg, upper limit is {np.rad2deg(uj.hi):.0f}")
        if short:
            d.findings.append("limits don't cover the motion in the video (" + "; ".join(short) + ")")
    if shape_err > SHAPE_TOL_M:
        d.findings.append(f"the moving part's geometry differs from the measured one by {1000 * shape_err:.0f} mm "
                          f"on average")

    if joint_log is not None:  # (frame, q) pairs recorded for the video
        frames, logged = joint_log[:, 0].astype(int), joint_log[:, 1]
        ok = (frames >= 0) & (frames < len(opening)) & ~np.isnan(opening[np.clip(frames, 0, len(opening) - 1)])
        expected = np.polyval([slope, q_at_closed], opening[frames[ok]])
        a, b = np.polyfit(expected, logged[ok], 1)
        d.numbers["log_scale"], d.numbers["log_offset_deg"] = float(a), float(np.rad2deg(b))
        if a < 0:
            d.findings.append(f"joint log has the opposite sign to the URDF (log = {a:.2f} x expected "
                              f"{'+' if b >= 0 else '-'} {abs(np.rad2deg(b)):.0f} deg)")
        elif abs(np.rad2deg(b)) > LIMIT_TOL_DEG:
            d.findings.append(f"joint log is offset by {np.rad2deg(b):.1f} deg from where the video puts the joint")
    return d


def report(d: Diagnosis, urdf: Path) -> str:
    n = d.numbers
    lines = [f"# URDF Doctor: {urdf.name}, joint `{d.joint}`", ""]
    lines.append("**Result: " + ("consistent with the video.**" if d.ok else f"{len(d.findings)} problem(s).**"))
    lines += [f"- {f}" for f in d.findings]
    lines += ["", "| check | value |", "|---|---|",
              f"| axis direction vs measured | {n['axis_direction_error_deg']:.1f} deg |",
              f"| axis position vs measured | {n['axis_offset_mm']:.0f} mm |",
              f"| positive q | {'opens' if n['opens_with_positive_q'] else 'closes'} |",
              f"| URDF q when closed | {n['q_when_closed_deg']:.0f} deg |",
              f"| opening seen in the video | {n['opening_seen_deg'][0]:.0f}..{n['opening_seen_deg'][1]:.0f} deg |",
              f"| URDF q that motion needs | {n['q_needed_deg'][0]:.0f}..{n['q_needed_deg'][1]:.0f} deg |",
              f"| URDF limits | " + (f"{n['urdf_limits_deg'][0]:.0f}..{n['urdf_limits_deg'][1]:.0f} deg" if n['urdf_limits_deg'] else "none") + " |",
              f"| moving-part shape mismatch | {n['shape_mismatch_mm']:.1f} mm |"]
    if "log_scale" in n:
        lines.append(f"| joint log vs video | {n['log_scale']:.2f} x {'+' if n['log_offset_deg'] >= 0 else '-'} "
                     f"{abs(n['log_offset_deg']):.1f} deg |")
    return "\n".join(lines) + "\n"
