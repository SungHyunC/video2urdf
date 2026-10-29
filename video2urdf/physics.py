"""Physical properties for a measured model: each link's mass and inertia, contact friction,
and joint friction, measured from the video where it can be.

Spark judges each part's material and mass from the video (it knows what a MacBook lid or a
particle-board drawer weighs); inertia follows from the link's measured mesh at that mass.
A hinge that holds its angle with no hand on it (a laptop lid left standing) measures its
own friction: at least the gravity torque at every angle where it was left. That bound is
checked by releasing the model in MuJoCo at those angles, with and without the friction.

Works on any model the pipeline writes (2D, LiDAR hinge, drawers): it edits model.urdf and
model.xml in place and records what it did in physics.json.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HOLD_MARGIN = 1.25  # friction = margin x the largest gravity torque the joint was seen to hold
STILL_DEG_S = 3.0  # a hinge moving slower than this (smoothed) counts as still
STILL_M_S = 0.005
LYING_M = 0.01  # a part within this of its parent's bounding box lies on it (a closed lid on its base)
LYING_MARGIN = (np.deg2rad(5.0), 0.005)  # closed frames read a few degrees (mm) off: widen the lying region
# MuJoCo's dry friction is a soft constraint: at the default softness a loaded hinge creeps
# (a lid sank 7 deg in 2 s at 1.25x its holding torque). These keep it under 0.3 deg.
SOLREF_FRICTION, SOLIMP_FRICTION = "0.005 1", "0.99 0.999 0.001"
RELEASE_S = 2.0  # how long the released model is simulated
DEFAULT_DAMPING = {"revolute": 0.02, "prismatic": 1.0}  # N m s / N s: not measured, keeps sims calm


def joint_trajectories(track_dir: Path) -> dict[str, np.ndarray]:
    """Measured joint value per frame (NaN where unmeasured), by MJCF joint name."""
    def arr(q):
        return np.array([np.nan if v is None else v for v in q], float)

    if (track_dir / "slides.json").exists():
        s = json.loads((track_dir / "slides.json").read_text())
        return {j["name"]: arr(d["q"]) for j, d in zip(s["joints"], s["drawers"])}
    if (track_dir / "hinge.json").exists():
        return {"lid_joint": arr(json.loads((track_dir / "hinge.json").read_text())["q"])}
    if (track_dir / "slide.json").exists():
        return {"drawer_joint": arr(json.loads((track_dir / "slide.json").read_text())["q"])}
    joints = json.loads((track_dir / "joints.json").read_text())["joints"]
    return {f"{j['child']}_joint": arr(j["q"]) for j in joints if j["type"] != "fixed"}


def _bodies(model: mujoco.MjModel) -> list[str]:
    return [model.body(i).name for i in range(1, model.nbody)]


def link_sizes(model_xml: Path) -> dict[str, dict]:
    """Bounding box (mm) and solid volume (litres) of each link's geometry, at density 1."""
    spec = mujoco.MjSpec.from_file(str(model_xml))
    for g in spec.geoms:
        visual_only = g.contype == 0 and g.conaffinity == 0
        g.density, g.mass = (0.0, 0.0) if visual_only else (1.0, float("nan"))  # mass from volume
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = {}
    for name in _bodies(model):
        b = model.body(name).id
        pts = []
        for g in np.flatnonzero((model.geom_bodyid == b) & (model.geom_contype != 0)):
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                m = model.geom_dataid[g]
                v = model.mesh_vert[model.mesh_vertadr[m]: model.mesh_vertadr[m] + model.mesh_vertnum[m]]
                pts.append(v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
            else:
                pts.append(data.geom_xpos[g][None] + np.array([-1, 1])[:, None] * model.geom_rbound[g])
        pts = np.vstack(pts)
        out[name] = {"size_mm": [round(float(x), 1) for x in 1000 * np.ptp(pts, axis=0)],
                     "solid_volume_l": round(float(model.body_mass[b]) * 1000, 4)}  # kg at 1 kg/m^3 -> m^3 -> l
    return out


def _quat_to_rpy(q) -> tuple[float, float, float]:
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(q, float))
    r = r.reshape(3, 3)
    pitch = float(np.arcsin(-np.clip(r[2, 0], -1, 1)))
    return float(np.arctan2(r[2, 1], r[2, 2])), pitch, float(np.arctan2(r[1, 0], r[0, 0]))


def inertials(model_xml: Path, masses: dict[str, float]) -> dict[str, dict]:
    """Each link's inertial at its given mass, from its geometry at uniform density (visual-only
    geoms, which have contype 0, don't count)."""
    spec = mujoco.MjSpec.from_file(str(model_xml))
    for g in spec.geoms:
        if g.contype == 0 and g.conaffinity == 0:
            g.density, g.mass = 0.0, 0.0
        else:
            g.density, g.mass = 1000.0, float("nan")
    model = spec.compile()
    out = {}
    for name in _bodies(model):
        b = model.body(name).id
        k = masses[name] / model.body_mass[b]
        out[name] = {"mass": float(masses[name]), "pos": model.body_ipos[b].tolist(),
                     "quat": model.body_iquat[b].tolist(), "diaginertia": (k * model.body_inertia[b]).tolist()}
    return out


def gravity_torque(model_xml: Path, joint: str, q: np.ndarray) -> np.ndarray:
    """Generalized gravity force on ``joint`` at each value in ``q`` (other joints at 0)."""
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    j = model.joint(joint)
    out = []
    for v in q:
        data.qpos[:] = 0
        data.qpos[j.qposadr[0]] = v
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        out.append(data.qfrc_bias[j.dofadr[0]])
    return np.array(out)


def release_drift(model_xml: Path, joint: str, q0: list[float], frictionloss: float | None = None) -> list[float]:
    """Let go of the model at each q0 and return how far the joint moves in RELEASE_S seconds.
    ``frictionloss`` overrides the model's joint friction (0 = frictionless hinge)."""
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    j = model.joint(joint)
    if frictionloss is not None:
        model.dof_frictionloss[j.dofadr[0]] = frictionloss
    data = mujoco.MjData(model)
    out = []
    for v in q0:
        mujoco.mj_resetData(model, data)
        data.qpos[j.qposadr[0]] = v
        mujoco.mj_forward(model, data)
        for _ in range(int(RELEASE_S / model.opt.timestep)):
            mujoco.mj_step(model, data)
        out.append(float(data.qpos[j.qposadr[0]] - v))
    return out


def _surface_points(model: mujoco.MjModel, data: mujoco.MjData, body: int, rng, n: int = 1500) -> np.ndarray:
    """Points spread over a body's collidable mesh surfaces (area-weighted), in world."""
    tris = []
    for g in np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0)):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        m = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[m]: model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        f = model.mesh_face[model.mesh_faceadr[m]: model.mesh_faceadr[m] + model.mesh_facenum[m]]
        v = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        tris.append(v[f])
    if not tris:
        return np.zeros((0, 3))
    tris = np.concatenate(tris)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    pick = rng.choice(len(tris), n, p=area / area.sum())
    a, b = rng.random((2, n, 1))
    flip = a + b > 1
    a, b = np.where(flip, 1 - a, a), np.where(flip, 1 - b, b)
    t = tris[pick]
    return t[:, 0] + a * (t[:, 1] - t[:, 0]) + b * (t[:, 2] - t[:, 0])


def lying_on_parent(model_xml: Path, joint: str, qs: np.ndarray) -> np.ndarray:
    """For each joint value: does the moving part lie on (or in) its parent, like a closed lid
    on its base? Then the parent holds it up, not joint friction. True when most of the part's
    surface is within LYING_M of the parent's bounding box."""
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    j = model.joint(joint)
    child, parent = j.bodyid[0], model.body_parentid[j.bodyid[0]]
    rng = np.random.default_rng(0)
    out = []
    for v in qs:
        data.qpos[:] = 0
        data.qpos[j.qposadr[0]] = v
        mujoco.mj_forward(model, data)
        pp, cp = _surface_points(model, data, parent, rng), _surface_points(model, data, child, rng)
        if not len(pp) or not len(cp):
            out.append(False)
            continue
        lo, hi = pp.min(axis=0) - LYING_M, pp.max(axis=0) + LYING_M
        out.append(bool(np.mean(np.all((cp >= lo) & (cp <= hi), axis=1)) > 0.5))
    return np.array(out, bool)


def lying_region(model_xml: Path, joint: str, limits: tuple[float, float], revolute: bool, steps: int = 61):
    """Joint values where the part lies on its parent, as a test function. The part lies on it
    from the innermost lying value on a half of the range out to that end: past a closed lid,
    a measured range can run on into the base (the closed end reads noisily), and there it
    pokes out of the parent's box instead of lying in it."""
    grid = np.linspace(limits[0], limits[1], steps)
    lying = lying_on_parent(model_xml, joint, grid)
    mid = (limits[0] + limits[1]) / 2
    low = grid[lying & (grid <= mid)]
    high = grid[lying & (grid > mid)]
    margin = LYING_MARGIN[0] if revolute else LYING_MARGIN[1]
    lo_edge = low.max() + margin if len(low) else -np.inf
    hi_edge = high.min() - margin if len(high) else np.inf
    return lambda q: (np.asarray(q) <= lo_edge) | (np.asarray(q) >= hi_edge)


def still_frames(q: np.ndarray, fps: float, windows: list[tuple[int, int]], revolute: bool) -> np.ndarray:
    """Frames inside the given windows where the measured joint is still."""
    t = np.arange(len(q))
    valid = ~np.isnan(q)
    if valid.sum() < 3:
        return np.array([], int)
    qi = np.interp(t, t[valid], q[valid])
    k = max(1, int(round(fps / 4)))  # smooth over a quarter second
    qs = np.convolve(qi, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    speed = np.abs(np.gradient(qs)) * fps
    limit = np.deg2rad(STILL_DEG_S) if revolute else STILL_M_S
    inside = np.zeros(len(q), bool)
    for a, b in windows:
        inside[max(0, a): min(len(q), b + 1)] = True
    return np.flatnonzero(inside & valid & (speed < limit))


def _joints(model_xml: Path) -> dict[str, tuple[str, tuple[float, float]]]:
    """Type ("revolute"/"prismatic") and range of each joint."""
    root = ET.parse(model_xml).getroot()
    return {j.get("name"): ("revolute" if j.get("type", "hinge") == "hinge" else "prismatic",
                            tuple(float(v) for v in j.get("range").split())) for j in root.iter("joint")}


def apply(track_dir: Path, parts: dict[str, dict], released: dict[str, list[tuple[int, int]]], fps: float,
          log=print) -> dict:
    """Write masses, inertia, friction into model.urdf and model.xml.

    ``parts``: {link: {"material": str, "mass_kg": float, "friction": float (contact, sliding)}}.
    ``released``: {joint: [(first_frame, last_frame), ...]} where nothing but the joint holds
    the moving part (no hand on it). Joints with released stills that hold against gravity get
    their friction from it; the rest keep a small default and are reported as not measured."""
    xml, urdf = track_dir / "model.xml", track_dir / "model.urdf"
    original = track_dir / "model.nophysics.xml"
    fresh = ET.parse(xml).getroot().find(".//inertial") is None  # just written by a fit: no physics yet
    if fresh or not original.exists():  # keep the geometry-only model, so applying twice starts from it
        original.write_text(xml.read_text())
        (track_dir / "model.nophysics.urdf").write_text(urdf.read_text())
    else:
        xml.write_text(original.read_text())
        urdf.write_text((track_dir / "model.nophysics.urdf").read_text())

    links = _bodies(mujoco.MjModel.from_xml_path(str(xml)))
    missing = [l for l in links if l not in parts]
    if missing:
        raise ValueError(f"no mass for links {missing}; links are {links}")
    sizes = link_sizes(xml)
    inert = inertials(xml, {l: float(parts[l]["mass_kg"]) for l in links})
    _write_mjcf(xml, inert, parts, {})  # masses first: gravity torques depend on them
    types = _joints(xml)
    q_all = joint_trajectories(track_dir)

    joints, report = {}, {"links": {}, "joints": {}}
    for l in links:
        vol = sizes[l]["solid_volume_l"]
        report["links"][l] = {**parts[l], **sizes[l], "implied_density_kg_m3": round(parts[l]["mass_kg"] / (vol / 1000))
                              if vol > 0 else None, "inertial": inert[l]}
    for name, (jtype, limits) in types.items():
        q = q_all.get(name)
        info = {"type": jtype, "damping": DEFAULT_DAMPING[jtype], "damping_measured": False}
        still = still_frames(q, fps, released.get(name, []), jtype == "revolute") if q is not None else np.array([], int)
        tau = np.abs(gravity_torque(xml, name, q[still])) if len(still) else np.array([])
        loaded = len(tau) and tau.max() > 1e-4
        if loaded:  # where the part lies on its parent (a closed lid), the parent holds it, not friction
            free = ~lying_region(xml, name, limits, jtype == "revolute")(q[still])
            info["lying_frames"] = int((~free).sum())
            still, tau = still[free], tau[free]
        if len(tau) and tau.max() > 1e-4:
            f = HOLD_MARGIN * float(tau.max())
            probe = sorted({float(q[t]) for t in still[np.linspace(0, len(still) - 1, 5).astype(int)]})
            info.update(friction=f, friction_measured=True, held_frames=int(len(still)),
                        held_range=[float(np.nanmin(q[still])), float(np.nanmax(q[still]))],
                        gravity_torque_max=float(tau.max()), probe_q=probe)
        else:
            # Nothing to measure: no hand-free still frames, or gravity doesn't load the joint
            # (a horizontal drawer). Friction from the contact estimate of the moving part.
            child = next((l for l in links if f"{l}_joint" == name), None)
            mu = float(parts.get(child, {}).get("friction", 0.3))
            load = 9.81 * float(parts.get(child, {}).get("mass_kg", 0.1))
            f = 0.1 * mu * load if jtype == "prismatic" else 0.02
            why = ("gravity does not load it" if not loaded and len(still) else
                   "no hand-free still frames where it isn't lying on its parent")
            info.update(friction=f, friction_measured=False, held_frames=int(len(still)), why_not=why)
        joints[name] = info
    _write_mjcf(xml, inert, parts, joints)
    _write_urdf(urdf, inert, joints)

    for name, info in joints.items():
        if info.get("friction_measured"):
            info["drift_with_deg"] = [float(np.rad2deg(d)) for d in release_drift(xml, name, info["probe_q"])]
            info["drift_without_deg"] = [float(np.rad2deg(d)) for d in release_drift(xml, name, info["probe_q"], 0.0)]
        report["joints"][name] = info
    (track_dir / "physics.json").write_text(json.dumps(report, indent=2))
    for l, r in report["links"].items():
        log(f"{l}: {r['material']}, {r['mass_kg']:.3f} kg over {r['solid_volume_l']:.3f} l of measured solid "
            f"({r['implied_density_kg_m3']} kg/m^3)")
    for name, info in report["joints"].items():
        if info.get("friction_measured"):
            unit = "N m" if info["type"] == "revolute" else "N"
            log(f"{name}: held by itself on {info['held_frames']} frames between "
                f"{np.rad2deg(info['held_range'][0]):.0f} and {np.rad2deg(info['held_range'][1]):.0f} deg; gravity torque "
                f"there up to {info['gravity_torque_max']:.3f} {unit} -> friction {info['friction']:.3f} {unit}. "
                f"Released in MuJoCo for {RELEASE_S:.0f} s at {len(info['probe_q'])} held angles it moves "
                f"{max(abs(d) for d in info['drift_with_deg']):.1f} deg (without that friction: "
                f"{max(abs(d) for d in info['drift_without_deg']):.0f} deg)")
        else:
            log(f"{name}: friction not measured ({info['why_not']}); set to {info['friction']:.3f} from the estimate")
    return report


def _fmt(x) -> str:
    return " ".join(f"{float(v):.6g}" for v in x)


def _write_mjcf(path: Path, inert: dict, parts: dict, joints: dict):
    tree = ET.parse(path)
    root = tree.getroot()
    if root.find("contact") is None:  # models written before emit added these (see emit3d.contact_excludes)
        contact = ET.SubElement(root, "contact")
        for parent in root.iter("body"):
            for child in parent.findall("body"):
                ET.SubElement(contact, "exclude", body1=parent.get("name"), body2=child.get("name"))
    for body in root.iter("body"):
        name = body.get("name")
        for old in body.findall("inertial"):
            body.remove(old)
        i = inert[name]
        el = ET.Element("inertial", pos=_fmt(i["pos"]), quat=_fmt(i["quat"]), mass=f"{i['mass']:.6g}",
                        diaginertia=_fmt(i["diaginertia"]))
        body.insert(1 if body.find("joint") is not None else 0, el)
        for g in body.findall("geom"):
            if g.get("contype") != "0":
                g.set("friction", f"{float(parts[name].get('friction', 1.0)):.3g} 0.005 0.0001")
        for j in body.findall("joint"):
            if j.get("name") in joints:
                info = joints[j.get("name")]
                j.set("frictionloss", f"{info['friction']:.6g}")
                j.set("solreffriction", SOLREF_FRICTION)
                j.set("solimpfriction", SOLIMP_FRICTION)
                j.set("damping", f"{info['damping']:.6g}")
    ET.indent(tree, "  ")
    tree.write(path, xml_declaration=False)


def _write_urdf(path: Path, inert: dict, joints: dict):
    tree = ET.parse(path)
    root = tree.getroot()
    for link in root.iter("link"):
        for old in link.findall("inertial"):
            link.remove(old)
        i = inert[link.get("name")]
        el = ET.Element("inertial")
        ET.SubElement(el, "origin", xyz=_fmt(i["pos"]), rpy=_fmt(_quat_to_rpy(i["quat"])))
        ET.SubElement(el, "mass", value=f"{i['mass']:.6g}")
        d = i["diaginertia"]
        ET.SubElement(el, "inertia", ixx=f"{d[0]:.6g}", ixy="0", ixz="0", iyy=f"{d[1]:.6g}", iyz="0", izz=f"{d[2]:.6g}")
        link.insert(0, el)
    for j in root.iter("joint"):
        info = joints.get(j.get("name"))
        if info is None:
            continue
        for old in j.findall("dynamics"):
            j.remove(old)
        ET.SubElement(j, "dynamics", damping=f"{info['damping']:.6g}", friction=f"{info['friction']:.6g}")
    ET.indent(tree, "  ")
    tree.write(path, xml_declaration=True, encoding="unicode")
