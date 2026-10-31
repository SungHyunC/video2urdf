"""Robot tasks on an object measured from video: a Franka Panda grasps it and works it, in MuJoCo physics.

What the robot does depends on what was measured:
- a hinged plate (a laptop lid): pinch its free edge, close it, open it again;
- a door (a plate on a vertical hinge: a microwave's): the robot stands in front of it; pinch the free
  edge at mid-height (fingers closing across the door), pull it open along its arc to most of its
  measured range, let go, then push it shut with the fingertips on its outer face. A model without a
  handle starts ajar, just far enough that the open fingers fit behind the free edge;
- a hinged rod (a lamp's arm): pinch it near its tip, fold it, raise it;
- a drawer: grip the handle Shape design drew on its front (its collision is switched on here), pull
  the drawer out, push it back in;
- an object with no joints (a mouse): pick it up from above, carry it aside, put it down.
The fingers close with the Panda's own gripper actuator and the object moves only because it is
held: friction between the finger pads and the measured collision geometry does the work. The
video titles each task and shows the joint value the object actually reaches.

Needs the Panda from MuJoCo Menagerie (Apache-2.0): run scripts/fetch_assets.sh once.

Usage: .venv/bin/python scripts/robot_tasks.py data/tracks_agent2/stand_lidar [--out robot_tasks.mp4]
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from robot_demo import ARM_JOINTS, HOME, PANDA_DIR, geom_points, smooth  # noqa: E402

W, H, FPS = 1920, 1080, 30
GRASP = np.array([0, 0, 0.1035])  # between the finger pads, in the hand frame
FINGER_TIP = 0.1123  # how far the fingertips reach along the hand's z (Menagerie's finger collision)
TIP_CLEARANCE = 0.002  # fingertips this far off a drawer's front while they hold its handle
OPEN, CLOSED = 255, 0  # gripper actuator
REACH_X, GRASP_Z = 0.52, 0.32  # where the grasp point is placed in front of the robot
MAX_JOINT_STEP = 0.05
DOOR_AXIS_DEG = 30  # a plate hinged this close to vertical is a door
DOOR_OPEN_SHARE = 0.85  # a door is pulled open to this share of its measured range, counted from shut
DOOR_START_MAX_DEG = 35  # a handle-less door starts ajar, at most this far, so a finger fits behind its edge
DOOR_START_STEP_DEG = 2.5
DOOR_PUSH_INSET = 0.07  # m: the fingertips push the door's outer face this far in from its free edge
DOOR_PUSH_DEPTH = 0.003  # m: the push target sits this far inside the face, so the face is pressed
DOOR_PUSH_TILT_DEG = 30  # the pushing hand points this far down from square on to the door's face
DOOR_SHUT_MARGIN_DEG = 1.0  # the push target stops this far short of the shut end; the press does the rest
PRESS_BIAS = 0.005  # m: the servo-error correction allowed while the hand presses on the object
DOOR_BACK = 0.10  # m: the hand backs off this far from the door between letting go and pushing
DOOR_REACH_X = 0.58  # where the middle of the door's working arc is placed in front of the robot
DESIGN_NAMES: dict[str, str] = {}
DEBUG = False  # filled from the run's design.json


@dataclass
class Grasp:
    body: str  # object body (prefixed) the fingers hold
    pos: np.ndarray  # grasp point, body frame
    approach: np.ndarray  # hand z (towards the object), body frame
    fingers: np.ndarray  # hand y (the fingers close along it), body frame


def design_names(run: Path) -> dict[str, str]:
    """Shape-design geom name (design_<link>_<i>) -> what Spark called that primitive ("drawer_handle")."""
    import json
    path = run / "design.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text()).get("design") or {}
    return {f"design_{l['link']}_{i}": p.get("name", "") for l in d.get("links", []) for i, p in enumerate(l["primitives"])}


def object_xml(model_xml: Path, pos, yaw: float, free: bool, handles: bool, ghost: str | None = None,
               solid: str | None = None, hold: str | None = None) -> tuple[str, str, str]:
    """(assets, body, contact excludes) of the measured object, prefixed with obj_, at pos/yaw.
    ``free``: the object is loose (a freejoint), not welded to the world. ``handles``: the shape
    design's handle geoms collide, so a gripper can hold them."""
    tree = ET.parse(model_xml)
    assets = tree.getroot().find("asset")
    mesh_xml = ""
    if assets is not None:
        for el in assets:
            el.set("name", "obj_" + el.get("name"))
            if el.get("file"):
                el.set("file", str((model_xml.parent / el.get("file")).resolve()))
            if el.get("texture"):
                el.set("texture", "obj_" + el.get("texture"))
            mesh_xml += ET.tostring(el, encoding="unicode")
    root = tree.getroot().find("worldbody").find("body")
    measured = root.find("inertial") is not None
    for el in root.iter():
        if "name" in el.attrib:
            el.set("name", "obj_" + el.get("name"))
        if el.tag == "geom":
            for key in ("mesh", "material"):
                if el.get(key):
                    el.set(key, "obj_" + el.get(key))
            if handles and "handle" in DESIGN_NAMES.get(el.get("name", "")[4:], "").lower():
                el.set("contype", "1")
                el.set("conaffinity", "1")
                el.set("friction", "1.2 0.01 0.001")
                el.set("solref", "0.002 1")  # stiff, so a squeezing gripper doesn't sink into it
                el.set("solimp", "0.99 0.999 0.0005")
            if el.get("contype") != "0":  # stiff contact: a squeezing gripper mustn't sink into the part
                el.set("solref", "0.002 1")
                el.set("solimp", "0.99 0.999 0.0005")
            if solid and _body_of(root, el) == "obj_" + solid:
                # a side-view rod: its measured box is a 6 mm silhouette; the fingers hold the rod Spark drew
                designed = el.get("name", "").startswith("obj_design_")
                el.set("contype", "1" if designed else "0")
                el.set("conaffinity", "1" if designed else "0")
                if designed:
                    el.set("solref", "0.002 1")
                    el.set("solimp", "0.99 0.999 0.0005")
            is_handle = handles and "handle" in DESIGN_NAMES.get(el.get("name", "")[4:], "").lower()
            if ghost and not is_handle and el.get("contype") != "0" and _body_of(root, el) == "obj_" + ghost:
                # the drawer's measured box: it slides in its slot (parent contacts are excluded) and
                # is moved by its handle; leave the fingers only the handle to touch
                el.set("contype", "0")
                el.set("conaffinity", "0")
            if el.get("group") == "4":  # a superseded visual
                el.set("rgba", "0 0 0 0")
            if not measured and el.get("contype") != "0":
                el.set("density", "300")
        if el.tag == "joint":
            el.set("armature", "0.01")
            if hold and el.get("name") != "obj_" + hold:
                el.set("frictionloss", "20")  # a joint not being worked stays where it is
            if not measured:
                el.set("damping", "0.3")
                el.set("frictionloss", "0.3")
    if free:
        root.insert(0, ET.Element("freejoint", name="obj_free"))
    excludes = "".join(f'<exclude body1="{p.get("name")}" body2="{c.get("name")}"/>'
                       for p in root.iter("body") for c in p.findall("body"))
    body = (f'<body name="object" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" euler="0 0 {yaw:.5f}">'
            f"{ET.tostring(root, encoding='unicode')}</body>") if not free else \
        ET.tostring(_placed(root, pos, yaw), encoding="unicode")
    return mesh_xml, body, excludes


def _body_of(root: ET.Element, el: ET.Element) -> str | None:
    for b in root.iter("body"):
        if el in list(b):
            return b.get("name")
    return root.get("name") if el in list(root) else None


def _placed(root: ET.Element, pos, yaw: float) -> ET.Element:
    """A free body can't sit inside a static wrapper: place it directly."""
    root.set("pos", " ".join(f"{float(v):.4f}" for v in np.asarray(pos) + np.fromstring(root.get("pos", "0 0 0"), sep=" ")))
    root.set("euler", f"0 0 {yaw:.5f}")
    return root


def gripper_excludes(root: str) -> str:
    """The measured cabinet is a coarse box around its drawer slots: let the fingers past it."""
    return "".join(f'<exclude body1="{b}" body2="{root}"/>' for b in ("hand", "left_finger", "right_finger"))


def scene_xml(meshes: str, obj: str, excludes: str, pedestal: tuple[float, float, float, float, float]) -> str:
    """pedestal: centre x, y, half-size x, y, top height (0: none)."""
    px, py, hx, hy, top = pedestal
    ped = (f'<geom name="pedestal" type="box" pos="{px:.3f} {py:.3f} {top / 2:.4f}" size="{hx:.3f} {hy:.3f} {top / 2:.4f}" '
           f'material="wood"/>') if top > 0.005 else ""
    return f"""<mujoco model="video2urdf robot tasks">
  <include file="panda.xml"/>
  <option cone="elliptic" impratio="10"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global offwidth="{W}" offheight="{H}"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.92 0.94 0.97" rgb2="0.75 0.8 0.86" width="512" height="3072"/>
    <texture type="2d" name="floor" builtin="checker" mark="edge" rgb1="0.85 0.86 0.88" rgb2="0.8 0.81 0.83"
      markrgb="0.7 0.7 0.7" width="300" height="300"/>
    <material name="floor" texture="floor" texuniform="true" texrepeat="5 5" reflectance="0.05"/>
    <material name="wood" rgba="0.62 0.47 0.33 1"/>
    {meshes}
  </asset>
  <worldbody>
    <light pos="0.4 -0.6 1.8" dir="0 0.3 -1" directional="true" castshadow="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="floor"/>
    {ped}
    {obj}
  </worldbody>
  <contact>{excludes}</contact>
</mujoco>"""


class Ik:
    """Damped least-squares IK for the grasp point's position and the hand's full orientation."""

    def __init__(self, model: mujoco.MjModel):
        self.m, self.d = model, mujoco.MjData(model)
        self.hand = model.body("hand").id
        self.qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.lo, self.hi = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS]].T
        self.error = 0.0

    SEEDS = [HOME, np.array([0, 0.4, 0, -1.9, 0, 2.3, 0.785]), np.array([0, 0.2, 0, -2.2, 0, 1.0, 0.785]),
             np.array([0, -0.3, 0, -2.4, 0, 2.1, -0.785]), np.array([0, 0.6, 0, -1.4, 0, 0.6, 0.785])]

    def solve(self, q: np.ndarray, pos: np.ndarray, rot: np.ndarray, iters: int = 80) -> np.ndarray:
        """From the current configuration; if that gets stuck short of the target (a local
        minimum), from a few typical Panda postures too, keeping the best."""
        best = self._solve(q, pos, rot, iters)
        best_err = self.error
        if best_err > 0.03:  # stuck: try other postures, preferring one near the current one
            score = best_err
            for seed in self.SEEDS:
                cand = self._solve(seed, pos, rot, 150)
                s = self.error + 0.01 * np.linalg.norm(cand - q)
                if s < score - 0.005:
                    best, best_err, score = cand, self.error, s
        self.error = best_err
        return best

    def _solve(self, q: np.ndarray, pos: np.ndarray, rot: np.ndarray, iters: int = 80) -> np.ndarray:
        q = q.copy()
        for _ in range(iters):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            r = self.d.xmat[self.hand].reshape(3, 3)
            tip = self.d.xpos[self.hand] + r @ GRASP
            jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
            mujoco.mj_jac(self.m, self.d, jp, jr, tip, self.hand)
            err_p = pos - tip
            err_r = 0.5 * sum(np.cross(r[:, k], rot[:, k]) for k in range(3))
            j = np.vstack([jp[:, self.dadr], 0.4 * jr[:, self.dadr]])
            e = np.concatenate([err_p, 0.4 * err_r])
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(6), e)
            dq += 0.03 * (np.eye(7) - np.linalg.pinv(j) @ j) @ (HOME - q)
            q = np.clip(q + np.clip(dq, -0.1, 0.1), self.lo, self.hi)
            if np.linalg.norm(err_p) < 5e-4 and np.linalg.norm(err_r) < 5e-3:
                break
        self.error = float(np.linalg.norm(err_p))
        return q


def hand_rot(approach: np.ndarray, fingers: np.ndarray) -> np.ndarray:
    z = approach / np.linalg.norm(approach)
    y = fingers - (fingers @ z) * z
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


def collision_points(model, data, body: int) -> np.ndarray:
    gs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_contype[g] != 0]
    return np.vstack([geom_points(model, data, g) for g in gs]) if gs else np.zeros((0, 3))


def to_body(data, body: int, p_world, v_world=None):
    r, x = data.xmat[body].reshape(3, 3), data.xpos[body]
    return (r.T @ (p_world - x)) if v_world is None else r.T @ v_world


def plan_grasp(model, data, kind: str, joint: str | None) -> Grasp:
    """Where and how to hold the object, worked out on the object as measured (world frame of the probe)."""
    if kind == "static":
        body = next(b for b in range(model.nbody) if model.body(b).name.startswith("obj_"))
        pts = collision_points(model, data, body)
        c = pts.mean(axis=0)
        flat = pts[:, :2] - c[:2]
        axes = np.linalg.svd(flat, full_matrices=False)[2]
        narrow = np.array([*axes[1], 0.0])  # the fingers close across the object's narrow side
        top, bottom = pts[:, 2].max(), pts[:, 2].min()
        p = np.array([c[0], c[1], bottom + 0.55 * (top - bottom)])
        return Grasp(model.body(body).name, to_body(data, body, p), to_body(data, body, None, np.array([0, 0, -1.0])),
                     to_body(data, body, None, narrow))
    jid = model.joint(joint).id
    body = model.jnt_bodyid[jid]
    axis = data.xaxis[jid]
    anchor = data.xanchor[jid]
    if kind == "drawer":
        handles = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_contype[g] != 0
                   and "handle" in DESIGN_NAMES.get(model.geom(g).name[4:], "").lower()]
        out = axis  # the slide's positive direction pulls the drawer out
        if handles:
            hp = geom_points(model, data, handles[0])
            c = hp.mean(axis=0)
            along = np.linalg.svd(hp - c, full_matrices=False)[2][0]  # the handle bar's length
            fingers = np.cross(along, out)  # close across the bar, perpendicular to the pull
            # The pads hold the bar's outer part, the fingertips just off the drawer's front: the
            # drawer and the cabinet keep their measured collision.
            rest = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_contype[g] != 0
                    and g not in handles]
            p = c
            if rest:
                front = float(np.max((np.vstack([geom_points(model, data, g) for g in rest]) - c) @ out))
                p = c + (front + TIP_CLEARANCE + FINGER_TIP - GRASP[2]) * out
            return Grasp(model.body(body).name, to_body(data, body, p), to_body(data, body, None, -out),
                         to_body(data, body, None, fingers))
        pts = collision_points(model, data, body)
        front = pts[(pts - anchor) @ out > np.percentile((pts - anchor) @ out, 95)]
        p = front.mean(axis=0)
        p[2] = pts[:, 2].max() - 0.01
        return Grasp(model.body(body).name, to_body(data, body, p), to_body(data, body, None, np.array([0, 0, -1.0])),
                     to_body(data, body, None, out))
    pts = collision_points(model, data, body)
    rel = pts - anchor
    radial = rel - np.outer(rel @ axis, axis)
    dist = np.linalg.norm(radial, axis=1)
    c = pts.mean(axis=0)
    cov_axes = np.linalg.svd(pts - c, full_matrices=False)[2]
    size = [np.ptp((pts - c) @ a) for a in cov_axes]
    if kind in ("plate", "door"):  # a door: its free edge at mid-height, the fingers across the door
        edge = pts[dist > 0.97 * dist.max()]
        p_edge = edge.mean(axis=0)
        r_dir = (p_edge - anchor) - ((p_edge - anchor) @ axis) * axis
        r_dir /= np.linalg.norm(r_dir)
        normal = cov_axes[2]  # the plate's thin direction
        p = p_edge - r_dir * 0.015 + axis * 0.0
        return Grasp(model.body(body).name, to_body(data, body, p), to_body(data, body, None, -r_dir),
                     to_body(data, body, None, normal))
    # a rod: pinch it near its tip from the side, the fingers closing in its plane of motion
    d = cov_axes[0] if (cov_axes[0] @ (c - anchor)) > 0 else -cov_axes[0]
    s = (pts - anchor) @ d
    p = anchor + d * (0.85 * s.max())
    p = p + (c - anchor - ((c - anchor) @ d) * d)  # onto the rod's centre line
    side = axis if axis[2] <= 0 else -axis  # come in from a side; prefer from above
    return Grasp(model.body(body).name, to_body(data, body, p), to_body(data, body, None, side),
                 to_body(data, body, None, np.cross(axis, d)))


def folded_onto_parent(model, data, jid: int) -> bool:
    """Whether the rod lies along its parent away from the joint (within 2 cm)."""
    from scipy.spatial import cKDTree
    child = model.jnt_bodyid[jid]
    parent = model.body_parentid[child]
    a, b = collision_points(model, data, child), collision_points(model, data, parent)
    if not len(a) or not len(b):
        return False
    a = a[np.linalg.norm(a - data.xanchor[jid], axis=1) > 0.06]
    return len(a) > 0 and float(np.median(cKDTree(b).query(a)[0])) < 0.02


def is_door_axis(axis) -> bool:
    """A hinge this close to vertical carries a door: it swings beside its body, not down onto it."""
    axis = np.asarray(axis, float)
    return bool(abs(axis[2]) > np.cos(np.deg2rad(DOOR_AXIS_DEG)) * np.linalg.norm(axis))


def door_yaw(normal_xy) -> float:
    """The yaw that turns a shut door's outward normal towards the robot (-x): the robot stands in front of
    the door, which swings towards it as it opens."""
    return float(np.angle(np.exp(1j * (np.pi - np.arctan2(normal_xy[1], normal_xy[0])))))


def opening_deg(q: float, shut: float, open_end: float) -> float:
    """How far a door stands open from shut (deg), counted towards the open end of its range."""
    return float(np.rad2deg((q - shut) * np.sign(open_end - shut)))


def door_open_target(shut: float, open_end: float) -> float:
    """The joint value a door is pulled open to: DOOR_OPEN_SHARE of its measured range from shut."""
    return float(shut + DOOR_OPEN_SHARE * (open_end - shut))


def door_start_candidates(ref_deg: float, open_deg: float, max_deg: float = DOOR_START_MAX_DEG,
                          step_deg: float = DOOR_START_STEP_DEG, keep_deg: float = 10.0) -> list[float]:
    """Openings (deg from shut) to try starting a door at, in order: the reference pose's own (the video's
    reference frame), then wider ones in steps up to ``max_deg``, always leaving ``keep_deg`` to pull."""
    wider = np.arange(step_deg, max_deg + 1e-6, step_deg)
    return [float(o) for o in [ref_deg, *wider[wider > ref_deg + 0.5 * step_deg]] if 0 <= o <= open_deg - keep_deg]


def free_edge(model, data, jid: int) -> np.ndarray:
    """The collision points of a hinged body's free edge: those farthest from its hinge."""
    pts = collision_points(model, data, model.jnt_bodyid[jid])
    rel = pts - data.xanchor[jid]
    dist = np.linalg.norm(rel - np.outer(rel @ data.xaxis[jid], data.xaxis[jid]), axis=1)
    return pts[dist > 0.97 * dist.max()]


def door_ends(model, joint: str) -> tuple[float, float, np.ndarray]:
    """A door's shut end of its range (where its free edge comes nearest the body it hangs on), its open
    end, and its outward face normal when shut (door_normal), world frame."""
    from scipy.spatial import cKDTree
    data = mujoco.MjData(model)
    jid = model.joint(joint).id
    door = model.jnt_bodyid[jid]
    lo, hi = (float(v) for v in model.jnt_range[jid])
    mujoco.mj_kinematics(model, data)
    body = [collision_points(model, data, b) for b in range(model.nbody)
            if model.body(b).name.startswith("obj_") and b != door]
    body = [p for p in body if len(p)]
    if not body:
        raise ValueError("the door's body has no collision geometry to shut against")
    tree = cKDTree(np.vstack(body))

    def edge(q):
        data.joint(joint).qpos[0] = q
        mujoco.mj_kinematics(model, data)
        return free_edge(model, data, jid)

    near = {q: float(tree.query(edge(q))[0].min()) for q in (lo, hi)}
    shut = lo if near[lo] < near[hi] else hi
    open_end = hi if shut == lo else lo
    return shut, open_end, door_normal(model, data, joint, shut, open_end)


def door_normal(model, data, joint: str, shut: float, open_end: float) -> np.ndarray:
    """A shut door's outward face normal (world frame): its thin direction, signed the way its free edge
    heads as it opens. Leaves the door shut in ``data``."""
    jid = model.joint(joint).id
    data.joint(joint).qpos[0] = shut + 0.05 * np.sign(open_end - shut)
    mujoco.mj_kinematics(model, data)
    ahead = free_edge(model, data, jid).mean(axis=0)
    data.joint(joint).qpos[0] = shut
    mujoco.mj_kinematics(model, data)
    pts = collision_points(model, data, model.jnt_bodyid[jid])
    n = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)[2][2]
    return n if n @ (ahead - free_edge(model, data, jid).mean(axis=0)) > 0 else -n


def door_push_point(model, data, jid: int, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where the fingertips push a door shut, in the door's body frame: on its outer face (``out``, world),
    mid-height, DOOR_PUSH_INSET in from its free edge; and that face's outward normal (body frame)."""
    body = model.jnt_bodyid[jid]
    pts = collision_points(model, data, body)
    c = free_edge(model, data, jid).mean(axis=0)  # the free edge's middle
    axis = data.xaxis[jid]
    along = (c - data.xanchor[jid]) - ((c - data.xanchor[jid]) @ axis) * axis
    along -= (along @ out) * out  # in the door's plane, away from the hinge
    along /= np.linalg.norm(along)
    p = c - DOOR_PUSH_INSET * along + float(np.max((pts - c) @ out)) * out
    return to_body(data, body, p), to_body(data, body, None, out)


def slerp_rot(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    """Rotation matrix a turned towards b by the share u (0..1) of the turn between them."""
    from scipy.spatial.transform import Rotation, Slerp
    return Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([a, b])))(float(np.clip(u, 0, 1))).as_matrix()


def kind_of(model) -> tuple[str, str | None]:
    """What the object is to the robot: plate / rod hinge, door (a plate on a vertical hinge), drawer, or
    static; and the joint to work."""
    joints = [j for j in range(model.njnt) if model.joint(j).name.startswith("obj_")
              and model.jnt_type[j] in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
    if not joints:
        return "static", None
    j = joints[-1]  # the outermost moving link
    if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE):
        return "drawer", model.joint(j).name
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    pts = collision_points(model, data, model.jnt_bodyid[j])
    sv = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)[1]
    if sv[1] < 0.25 * sv[0]:
        return "rod", model.joint(j).name
    return ("door" if is_door_axis(data.xaxis[j]) else "plate"), model.joint(j).name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="fitted run folder with model.xml")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if not (PANDA_DIR / "panda.xml").exists():
        sys.exit("Panda model missing: run scripts/fetch_assets.sh")
    import os
    scene = PANDA_DIR / f"video2urdf_tasks_scene_{os.getpid()}.xml"  # one per run: runs may go in parallel
    DESIGN_NAMES.update(design_names(args.run))

    def build(pos, yaw, free, pedestal=(0.6, 0, 0.2, 0.2, 0.0), pass_root=False, ghost=None, solid=None, hold=None):
        meshes, obj, excludes = object_xml(args.run / "model.xml", pos, yaw, free, handles=True, ghost=ghost,
                                           solid=solid, hold=hold)
        if pass_root:
            excludes += gripper_excludes("obj_" + ET.parse(args.run / "model.xml").getroot().find("worldbody").find("body").get("name"))
        scene.write_text(scene_xml(meshes, obj, excludes, pedestal))
        return mujoco.MjModel.from_xml_path(str(scene))

    # 1. Probe the object as measured: what it is, where to hold it.
    probe_model = build((0, 0, 0), 0.0, free=False)
    kind, joint = kind_of(probe_model)
    free = kind == "static"
    model = build((0, 0, 0), 0.0, free)
    data = mujoco.MjData(model)
    lo = hi = q_start = None
    if joint:
        jid = model.joint(joint).id
        lo, hi = model.jnt_range[jid]
        q_start = float(np.clip(0.0, lo, hi))
        data.joint(joint).qpos[0] = q_start
        mujoco.mj_kinematics(model, data)
        if kind == "rod" and folded_onto_parent(model, data, jid):
            q_start = float((lo + hi) / 2)  # start unfolded: a folded rod lies on its parent, which the fingers would take too
            data.joint(joint).qpos[0] = q_start
    mujoco.mj_kinematics(model, data)
    grasp = plan_grasp(model, data, kind, joint)

    # 2. Turn and place it: the hand comes in heading away from the robot, the grasp point at
    #    REACH_X in front of it, GRASP_Z high (on a pedestal if the object is low).
    b = model.body(grasp.body).id
    r = data.xmat[b].reshape(3, 3)
    a_w = r @ grasp.approach
    f_w = r @ grasp.fingers
    head = a_w[:2] if np.linalg.norm(a_w[:2]) > 0.3 else np.cross([0, 0, 1.0], f_w)[:2]
    yaw = -np.arctan2(head[1], head[0])
    shut = open_end = q_open = None
    if kind == "door":  # the robot stands in front of the door: its outer face, shut, looks at the robot
        shut, open_end, out = door_ends(model, joint)
        q_open = door_open_target(shut, open_end)
        yaw = door_yaw(out[:2])
    model = build((0, 0, 0), yaw, free)
    data = mujoco.MjData(model)
    if joint:
        data.joint(joint).qpos[0] = q_start
    mujoco.mj_kinematics(model, data)
    b = model.body(grasp.body).id
    p_w = data.xmat[b].reshape(3, 3) @ grasp.pos + data.xpos[b]
    pts = np.vstack([geom_points(model, data, g) for g in range(model.ngeom)
                     if model.body(model.geom_bodyid[g]).name.startswith("obj_") and model.geom_contype[g] != 0])
    lift = max(0.0, (0.40 if kind == "drawer" else GRASP_Z) - (p_w[2] - pts[:, 2].min()))
    reach_x = REACH_X + (0.8 * (hi - q_start) if kind == "drawer" else 0.0)  # room to pull it towards the robot
    pos = np.array([reach_x - p_w[0], -p_w[1], lift - pts[:, 2].min() + 0.001])
    if kind == "door":
        # The middle of the free edge's arc, shut to pulled open, at DOOR_REACH_X ahead; the door's lowest
        # point over its whole swing (a hinge off vertical dips its free corner) clears the pedestal.
        arc, low = [], pts[:, 2].min()
        for q in np.linspace(*model.jnt_range[model.joint(joint).id], 25):
            data.joint(joint).qpos[0] = q
            mujoco.mj_kinematics(model, data)
            low = min(low, collision_points(model, data, b)[:, 2].min())
            if (q - shut) * (q - q_open) <= 0:
                arc.append(data.xmat[b].reshape(3, 3) @ grasp.pos + data.xpos[b])
        data.joint(joint).qpos[0] = q_start
        mujoco.mj_kinematics(model, data)
        mid = (np.min(arc, axis=0) + np.max(arc, axis=0)) / 2
        lift = max(0.0, GRASP_Z - (p_w[2] - low))
        pos = np.array([DOOR_REACH_X - mid[0], -mid[1], lift - low + 0.002])
    foot = pts[:, :2] + pos[:2]
    pedestal = ((foot[:, 0].min() + foot[:, 0].max()) / 2, (foot[:, 1].min() + foot[:, 1].max()) / 2,
                np.ptp(foot[:, 0]) / 2 + 0.05, np.ptp(foot[:, 1]) / 2 + (0.25 if free else 0.05), lift)
    worked = model.body(model.jnt_bodyid[model.joint(joint).id]).name[4:] if joint else None
    thin = False
    if kind == "rod":  # a rod measured thinner than 1 cm across (a side-view silhouette): hold the designed one
        rp = collision_points(model, data, model.jnt_bodyid[model.joint(joint).id])
        thin = np.linalg.svd(rp - rp.mean(axis=0), full_matrices=False)[1][2] / np.sqrt(len(rp)) * 4 < 0.01
        thin = thin and any(n.startswith(f"design_{worked}_") for n in DESIGN_NAMES)
    model = build(pos, yaw, free, pedestal, solid=worked if thin else None,
                  hold=joint[4:] if joint else None)
    # Menagerie's gripper servo squeezes in proportion to how far open it is: about 1 N on an 18 mm
    # handle. The real Panda Hand holds up to 70 N; a stiffer servo squeezes ~30 N.
    grip = model.actuator("actuator8").id
    model.actuator_biasprm[grip, 1:3] = [-3000.0, -30.0]
    model.actuator_gainprm[grip, 0] = 3000.0 * 0.04 / 255
    data = mujoco.MjData(model)
    plan = mujoco.MjData(model)
    ik = Ik(model)
    if joint:
        data.joint(joint).qpos[0] = q_start
    arm_q = HOME.copy()
    data.qpos[ik.qadr] = arm_q
    data.ctrl[:7], data.ctrl[7] = arm_q, OPEN
    mujoco.mj_forward(model, data)
    b = model.body(grasp.body).id

    def grasp_pose(q=None, shift=np.zeros(3)):
        """World grasp point and hand rotation with the object at joint value q (plan copy)."""
        plan.qpos[:] = data.qpos
        if joint and q is not None:
            plan.joint(joint).qpos[0] = q
        mujoco.mj_kinematics(model, plan)
        rr, xx = plan.xmat[b].reshape(3, 3), plan.xpos[b]
        return rr @ grasp.pos + xx + shift, hand_rot(rr @ grasp.approach, rr @ grasp.fingers)

    robot_bodies = {i for i in range(model.nbody) if model.body(i).name not in ("world", "object")
                    and not model.body(i).name.startswith("obj_")}
    start_note = ""
    if kind == "door":
        jid = model.joint(joint).id
        plan.qpos[:] = data.qpos
        push_point, push_normal = door_push_point(model, plan, jid, door_normal(model, plan, joint, shut, open_end))

        def push_pose(q, depth=DOOR_PUSH_DEPTH):
            """Grasp-point target and hand rotation that put the closed fingertips ``depth`` inside the door's
            outer face at joint value q: the hand comes at the face from the front and DOOR_PUSH_TILT_DEG
            above (a Panda reaches a low, far face poorly square on), the fingers side by side along it."""
            plan.qpos[:] = data.qpos
            plan.joint(joint).qpos[0] = q
            mujoco.mj_kinematics(model, plan)
            rr, xx = plan.xmat[b].reshape(3, 3), plan.xpos[b]
            n = rr @ push_normal
            tilt = np.deg2rad(DOOR_PUSH_TILT_DEG)
            a = -n * np.cos(tilt) - np.array([0, 0, 1.0]) * np.sin(tilt)
            a /= np.linalg.norm(a)
            tip = rr @ push_point + xx - n * depth
            return tip - a * (FINGER_TIP - GRASP[2]), hand_rot(a, np.cross([0, 0, 1.0], n))

        def clearance(q) -> tuple[float, float]:
            """The open hand at the free edge with the door at q: its deepest contact with the object's
            other parts and the IK error."""
            p, rot = grasp_pose(q)
            chk = mujoco.MjData(model)
            chk.qpos[:] = data.qpos
            chk.joint(joint).qpos[0] = q
            chk.qpos[ik.qadr] = ik.solve(HOME.copy(), p, rot)
            for f in ("finger_joint1", "finger_joint2"):
                chk.joint(f).qpos[0] = 0.04
            mujoco.mj_forward(model, chk)
            deep = 0.0
            for c in chk.contact[:chk.ncon]:
                b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
                other = b2 if b1 in robot_bodies else b1 if b2 in robot_bodies else None
                if other is not None and other != b and model.body(other).name.startswith("obj_"):
                    deep = max(deep, -c.dist)
            return deep, ik.error

        # Without a handle the fingers go behind the free edge: start the door ajar, as little as lets them in.
        sense = np.sign(open_end - shut)
        tried = []
        for o in door_start_candidates(opening_deg(q_start, shut, open_end), opening_deg(q_open, shut, open_end)):
            q = shut + sense * np.deg2rad(o)
            deep, err = clearance(q)
            tried.append((o, deep, err))
            if deep < 0.001 and err < 0.005:
                q_start = float(q)
                break
        start_note = (f"door: shut at {np.rad2deg(shut):+.1f} deg, open end {np.rad2deg(open_end):+.1f} deg "
                      f"(measured range {opening_deg(open_end, shut, open_end):.0f} deg), pulled open to "
                      f"{opening_deg(q_open, shut, open_end):.0f} deg; starts {opening_deg(q_start, shut, open_end):.1f} "
                      "deg open (open hand at the free edge, deepest contact with the rest of the object: "
                      + ", ".join(f"{o:.1f} deg -> {1000 * d:.1f} mm" for o, d, _ in tried) + ")")
        data.joint(joint).qpos[0] = q_start
        mujoco.mj_forward(model, data)

    pts_now = np.vstack([geom_points(model, data, g) for g in range(model.ngeom)
                         if model.body(model.geom_bodyid[g]).name.startswith("obj_") and model.geom_contype[g] != 0])
    # 3. The tasks: segments of (title, seconds, target function of u in 0..1, gripper).
    p0, rot0 = grasp_pose(q_start)
    # pre-grasp: back off along the approach (a door's open hand, 20 cm across it, stays clear of its edge)
    back = -rot0[:, 2] * (0.22 if kind == "door" else 0.12)
    mujoco.mj_kinematics(model, data)
    hand0 = data.xpos[ik.hand] + data.xmat[ik.hand].reshape(3, 3) @ GRASP
    rot_home = data.xmat[ik.hand].reshape(3, 3).copy()
    segments = []
    pressing = set()  # segments that press on the object: their servo-error correction is kept small
    reach = [(2.0, lambda u: (hand0 + smooth(u) * (p0 + back - hand0), rot0), OPEN),
             (1.0, lambda u: (p0 + back * (1 - smooth(u)), rot0), OPEN),
             (0.8, lambda u: (p0, rot0), CLOSED)]
    if kind == "static":
        up, aside = np.array([0, 0, 0.12]), np.array([0, 0.18, 0])
        t1, t2 = "Pick it up and put it down elsewhere", "Bring it back, turned a quarter"

        def turned(u):  # the hand turned about the vertical by up to 90 deg
            a = np.pi / 2 * smooth(u)
            rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            return rz @ rot0
        segments = [(t1, *s) for s in reach] + [
            (t1, 1.5, lambda u: (p0 + smooth(u) * up, rot0), CLOSED),
            (t1, 2.0, lambda u: (p0 + up + smooth(u) * aside, rot0), CLOSED),
            (t1, 1.5, lambda u: (p0 + up + aside - smooth(u) * up, rot0), CLOSED),
            (t1, 0.8, lambda u: (p0 + aside, rot0), OPEN),
            (t1, 1.0, lambda u: (p0 + aside + smooth(u) * up * 0.6, rot0), OPEN),
            (t2, 1.0, lambda u: (p0 + aside + up * 0.6 * (1 - smooth(u)), rot0), OPEN),
            (t2, 0.8, lambda u: (p0 + aside, rot0), CLOSED),
            (t2, 1.5, lambda u: (p0 + aside + smooth(u) * up, rot0), CLOSED),
            (t2, 2.5, lambda u: (p0 + up + aside * (1 - smooth(u)), turned(u)), CLOSED),
            (t2, 1.5, lambda u: (p0 + up * (1 - smooth(u)), turned(1)), CLOSED),
            (t2, 0.8, lambda u: (p0, turned(1)), OPEN),
            (t2, 1.5, lambda u: (p0 + smooth(u) * (back + up), turned(1)), OPEN)]
    elif kind == "door":
        t1, t2 = "Pull the door open", "Push it shut"
        p_o, rot_o = grasp_pose(q_open)
        away = -rot_o[:, 2] * DOOR_BACK  # straight off the free edge, away from the hinge
        t_o, rot_p = push_pose(q_open)
        stand = -rot_p[:, 2] * DOOR_BACK  # in front of the outer face
        # The push ends just short of the shut end: the press closes the rest, and pressing a door into
        # its (soft) joint limit makes it spring back open when the hand lets go.
        q_close = shut + np.deg2rad(DOOR_SHUT_MARGIN_DEG) * np.sign(open_end - shut)
        t_s, rot_s = push_pose(q_close)
        segments = [(t1, *s) for s in reach] + [
            (t1, 4.0, lambda u: grasp_pose(q_start + smooth(u) * (q_open - q_start)), CLOSED),
            (t1, 0.8, lambda u: (p_o, rot_o), OPEN),  # let go
            (t2, 1.2, lambda u: (p_o + smooth(u) * away, rot_o), OPEN),
            (t2, 2.0, lambda u: (p_o + away + smooth(u) * (t_o + stand - p_o - away), slerp_rot(rot_o, rot_p, smooth(u))),
             CLOSED),
            (t2, 1.0, lambda u: (t_o + stand * (1 - smooth(u)), rot_p), CLOSED),
            (t2, 4.0, lambda u: push_pose(q_open + smooth(u) * (q_close - q_open)), CLOSED),
            (t2, 1.5, lambda u: (t_s + smooth(u) * -rot_s[:, 2] * DOOR_BACK, rot_s), CLOSED)]
        pressing = {len(segments) - 2}
    else:
        if kind == "drawer":
            ends = [("Pull the drawer out", q_start + 0.8 * (hi - q_start)), ("Push it back in", q_start + 0.02 * (hi - lo))]
        elif kind == "plate":
            # which end closes it: where the plate sits lowest
            low = {}
            for q in (lo, hi):
                pp, _ = grasp_pose(q)
                low[q] = pp[2]
            shut = lo if low[lo] < low[hi] else hi
            other = hi if shut == lo else lo
            ends = [("Close the lid", shut + 0.12 * np.sign(q_start - shut)),
                    ("Open it again", q_start)]
        else:
            floor_z = min(pts_now[:, 2])

            def clear(q_to):  # stop short where the hand would come down onto the object's own base
                for q in np.linspace(q_start, q_to, 40):
                    if grasp_pose(q)[0][2] < floor_z + 0.12:
                        return q - 0.05 * np.sign(q_to - q_start)
                return q_to
            far = hi - 0.1 * (hi - lo) if hi - q_start > q_start - lo else lo + 0.1 * (hi - lo)
            far = clear(far)
            up = grasp_pose(far)[0][2] > p0[2]
            ends = [("Raise the arm" if up else "Lower the arm", far),
                    ("Lower it back" if up else "Raise it back", q_start)]
        title0 = ends[0][0]
        segments = [(title0, *s) for s in reach]
        q_from = q_start
        for title, q_to in ends:
            segments.append((title, 4.0, (lambda a, z: lambda u: grasp_pose(a + smooth(u) * (z - a)))(q_from, q_to), CLOSED))
            q_from = q_to
        segments += [(ends[-1][0], 0.8, lambda u: grasp_pose(q_from), OPEN),
                     (ends[-1][0], 1.5, (lambda qq: lambda u: (grasp_pose(qq)[0] + smooth(u) * -grasp_pose(qq)[1][:, 2] * 0.15,
                                                             grasp_pose(qq)[1]))(q_from), OPEN)]

    # 4. Run it.
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    extent = float(np.ptp(pts_now, axis=0).max())
    cam.lookat[:] = 0.65 * p0 + 0.35 * np.array([0.2, 0, 0.45])
    cam.distance = float(np.clip(0.75 + 1.3 * extent, 0.95, 1.7))  # small objects close up
    cam.azimuth, cam.elevation = 115, -16  # from the robot's right: robot left, object right
    if kind == "door":  # from the front, on the hinge side, above: the door's face and its swing, the robot right
        hinge_at = data.xanchor[model.joint(joint).id]
        cam.lookat[:] = [0.4 * (DOOR_REACH_X + hinge_at[0]), 0.5 * hinge_at[1], GRASP_Z + 0.06]
        cam.azimuth, cam.elevation, cam.distance = -60, -28, 1.3
    out = args.out or (args.run / "robot_tasks.mp4")
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
    steps = round(1 / FPS / model.opt.timestep)
    titles = list(dict.fromkeys(s[0] for s in segments))
    log, ik_err = [], []
    obj_start = data.xpos[b].copy()
    bias = np.zeros(3)
    hand_bodies = {model.body(n).id for n in ("hand", "left_finger", "right_finger")}
    held_geoms = {g for g in range(model.ngeom) if "handle" in DESIGN_NAMES.get(model.geom(g).name[4:], "").lower()}
    deepest = {}  # object body -> the hand's deepest contact with it (a handle being held aside)
    for k_seg, (title, dur, target, grip) in enumerate(segments):
        seg0, seg_pairs = len(ik_err), {}
        limit = PRESS_BIAS if k_seg in pressing else 0.03
        for f in range(int(dur * FPS)):
            u = (f + 1) / (dur * FPS)
            pos_t, rot_t = target(u)
            # the servos leave the hand a few millimetres off its target: correct by integrating the error
            # (pressing on something, the error is the contact and correcting it would press harder)
            mujoco.mj_kinematics(model, data)
            actual = data.xpos[ik.hand] + data.xmat[ik.hand].reshape(3, 3) @ GRASP
            bias = np.clip(bias + 0.3 * (pos_t - actual), -limit, limit)
            arm_q = arm_q + np.clip(ik.solve(arm_q, pos_t + bias, rot_t) - arm_q, -MAX_JOINT_STEP, MAX_JOINT_STEP)
            ik_err.append(ik.error)
            data.ctrl[:7] = arm_q
            data.ctrl[7] = grip
            for _ in range(steps):
                data.qfrc_applied[ik.dadr] = data.qfrc_bias[ik.dadr]  # gravity compensation, as the real Panda has
                mujoco.mj_step(model, data)
            for c in data.contact[:data.ncon]:
                b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
                if (b1 in hand_bodies) != (b2 in hand_bodies) and not {c.geom1, c.geom2} & held_geoms \
                        and model.body(b2 if b1 in hand_bodies else b1).name.startswith("obj_"):
                    other = model.body(b2 if b1 in hand_bodies else b1).name[4:]
                    deepest[other] = max(deepest.get(other, 0.0), -c.dist)
                if DEBUG and (b1 in robot_bodies) != (b2 in robot_bodies):
                    pair = (model.body(b1).name, model.body(b2).name)
                    if any(n.startswith("obj_") for n in pair):
                        seg_pairs[pair] = max(seg_pairs.get(pair, 0.0), -c.dist)
            renderer.update_scene(data, camera=cam)
            img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            k = titles.index(title) + 1
            cv2.putText(img, f"{k}/{len(titles)}  {title}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 4)
            if kind == "door":
                val = data.joint(joint).qpos[0]
                txt = (f"door open {round(opening_deg(val, shut, open_end))} deg (0 = shut; measured range "
                       f"0-{opening_deg(open_end, shut, open_end):.0f})")
            elif joint:
                val = data.joint(joint).qpos[0]
                txt = (f"{joint[4:]}: {np.rad2deg(val - q_start):+.0f} deg" if int(model.jnt_type[model.joint(joint).id])
                       == int(mujoco.mjtJoint.mjJNT_HINGE) else f"{joint[4:]}: {1000 * (val - q_start):+.0f} mm")
            else:
                txt = f"object moved {100 * np.linalg.norm(data.xpos[b] - obj_start):.0f} cm"
            cv2.putText(img, txt + "  (measured joint, moved by the grasp)", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (60, 60, 60), 2)
            writer.write(img)
            if DEBUG and title == segments[0][0] and abs(u - 1) < 1e-6 and dur == 0.8:
                c2 = mujoco.MjvCamera()
                c2.lookat[:] = pos_t
                c2.distance, c2.azimuth, c2.elevation = 0.35, 70, -10
                renderer.update_scene(data, camera=c2)
                cv2.imwrite(str(out.with_name("grasp_closeup.jpg")), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        log.append((title, data.joint(joint).qpos[0] if joint else data.xpos[b].copy()))
        if DEBUG:
            pairs = set()
            for c in data.contact[:data.ncon]:
                n1, n2 = model.geom(c.geom1).name or model.body(model.geom_bodyid[c.geom1]).name, model.geom(c.geom2).name or model.body(model.geom_bodyid[c.geom2]).name
                if "obj_" in n1 + n2:
                    pairs.add((n1, n2))
            rr = data.xmat[ik.hand].reshape(3, 3)
            print("   contacts:", sorted(pairs)[:12], "finger q", round(float(data.joint("finger_joint1").qpos[0]), 4),
                  "target", np.round(pos_t, 3), "hand", np.round(data.xpos[ik.hand] + rr @ GRASP, 3), "ik", round(ik.error, 3),
                  "segment ik max", round(max(ik_err[seg0:]), 3), "robot-object contacts (mm)",
                  {f"{p[0]}/{p[1]}": round(1000 * v, 1) for p, v in seg_pairs.items()},
                  "joint", round(float(np.rad2deg(data.joint(joint).qpos[0])), 1) if joint else "")
    writer.release()
    renderer.close()
    scene.unlink(missing_ok=True)
    print(f"{kind} ({joint or 'free body'}); IK error max {1000 * max(ik_err):.0f} mm; video -> {out}")
    if start_note:
        print("  " + start_note)
    print("  deepest hand contact per part (a handle being held aside; the held part includes the squeeze): "
          + (", ".join(f"{k} {1000 * v:.1f} mm" for k, v in sorted(deepest.items())) or "none"))
    for title, v in log:
        hinge = joint and int(model.jnt_type[model.joint(joint).id]) == int(mujoco.mjtJoint.mjJNT_HINGE)
        print(f"  after '{title}': " + (f"{np.rad2deg(v - q_start):+.1f} deg" if hinge else
                                        f"{1000 * (v - q_start):+.0f} mm" if joint else
                                        f"object moved {100 * np.linalg.norm(v - obj_start):.1f} cm")
              + (f" (door open {opening_deg(v, shut, open_end):.1f} deg)" if kind == "door" else ""))


if __name__ == "__main__":
    main()
