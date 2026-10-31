"""A drawer-opening task for reinforcement learning, built on a drawer measured from video.

The cabinet and drawer are the fitted model with their measured collision on, as in
scripts/robot_tasks.py: the slotted carcass and the drawer's front and tray collide with the
hand, and the handle Shape design drew collides too (it is what the fingers hold). The robot is
a Panda hand (MuJoCo Menagerie) on a Cartesian mount: three slide joints driven by force-limited
position servos, gravity compensated. The policy moves the servos' setpoint in x, y, z and opens
or closes the fingers. The hand comes in level, fingers closing vertically, as the full arm does
in robot_tasks.py, so a trained policy can drive the arm through IK.

Physics a policy could otherwise lean on, removed here:
- a hand welded to a mocap target (the earlier env) pushes with whatever force the weld needs:
  pressing the drawer's front it pushed with ~2 kN. The mount's servos push with at most
  MOUNT_KP x one step's lead (~36 N), never more than MOUNT_FMAX per axis;
- MuJoCo's default joint limits are soft: the shut drawer was pushed 14 mm past its closed stop
  and closed fingers were forced 6 mm past their own limit. End stops are stiff here;
- MuJoCo averages two geoms' contact softness, so Menagerie's soft hand geoms halved the stiff
  contact robot_tasks.py gives the object. The hand's geoms are stiff too.

Only drawer travel made while the fingers pinch the handle counts, in the reward and for
success: both fingers press on the handle from opposite sides along their closing direction,
each finger open to within PINCH_SLACK of the handle's half height (the pads on its faces, not
round its edge). Pushing, hooking or wedging the drawer out earns nothing.

Each episode moves and turns the cabinet a little and scales the drawer's friction, so the
policy can't memorise one placement or one friction value.
"""

import re
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import robot_tasks as rt  # noqa: E402

PANDA_DIR = rt.PANDA_DIR
GRASP = rt.GRASP
HANDLE_AT = np.array([0.72, 0.0, 0.40])  # where the handle sits: as robot_tasks.py places a drawer, in the Panda arm's reach while pulled
STEP_M = 0.012  # largest setpoint move per action (0.3 m/s)
SUBSTEPS = 20  # 0.04 s per action
EPISODE = 200  # actions (8 s)
SUCCESS_M = 0.15  # drawer pulled this far while held
PINCH_SLACK = 0.002  # on a hold each finger is open at least the handle's half height less this: an 18 mm
# handle squarely between the pads holds each finger ~9 mm open; pads slipping round its edge close further
PINCH_COS = 0.7  # a pinching contact pushes along the fingers' closing direction (within ~45 deg)
GRIP_FORCE = 3000.0  # gripper servo stiffness (N/m on the finger tendon): ~30 N squeeze, as robot_tasks.py
MOUNT = ("mount_x", "mount_y", "mount_z")
MOUNT_KP, MOUNT_KV, MOUNT_FMAX = 3000.0, 100.0, 60.0  # the mount's servos: N/m, N s/m, N per axis
MOUNT_AT = HANDLE_AT + np.array([-0.2, 0, 0.05])  # the mount's zero: the grasp point 20 cm in front of the handle
STIFF_REF, STIFF_IMP = [0.002, 1.0], [0.99, 0.999, 0.0005, 0.5, 2.0]  # robot_tasks.py's stiff contact


def hand_rot() -> np.ndarray:
    """The hand level, approaching along +x, fingers closing vertically."""
    return rt.hand_rot(np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))


def hand_quat() -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, hand_rot().flatten())
    return q


def stiff_gripper(m: mujoco.MjModel) -> None:
    """Menagerie's gripper servo squeezes about 1 N on an 18 mm handle; stiffened to ~30 N."""
    grip = m.actuator("actuator8").id
    m.actuator_biasprm[grip, 1:3] = [-GRIP_FORCE, -30.0]
    m.actuator_gainprm[grip, 0] = GRIP_FORCE * 0.04 / 255


def stiff_stops(m: mujoco.MjModel, joints) -> None:
    """End stops as stiff as the contacts (a real drawer's stop and a gripper's fingers don't give)."""
    for j in joints:
        jid = m.joint(j).id
        m.jnt_solref[jid] = STIFF_REF
        m.jnt_solimp[jid] = STIFF_IMP


def stiff_hand(m: mujoco.MjModel) -> None:
    """The hand's collision geoms as stiff as the object's (MuJoCo averages the pair's softness)."""
    hand = {m.body(n).id for n in ("hand", "left_finger", "right_finger")}
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in hand and m.geom_contype[g]:
            m.geom_solref[g] = STIFF_REF
            m.geom_solimp[g] = STIFF_IMP


def harden(m: mujoco.MjModel, joint: str) -> None:
    """The physics both the env and the arm run with: a firm gripper, rigid end stops, stiff contacts."""
    stiff_gripper(m)
    stiff_stops(m, [joint, "finger_joint1", "finger_joint2"])
    stiff_hand(m)


def _probe(run: Path):
    """The object alone at the origin (no robot), as robot_tasks.py probes it."""
    meshes, obj, excludes = rt.object_xml(run / "model.xml", (0, 0, 0), 0.0, False, handles=True)
    xml = rt.scene_xml(meshes, obj, excludes, (0.6, 0, 0.2, 0.2, 0.0)).replace('<include file="panda.xml"/>', "")
    probe = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(probe)
    mujoco.mj_kinematics(probe, data)
    return probe, data


def placement(run: Path) -> dict:
    """Where the env puts the measured cabinet (its handle at HANDLE_AT, the drawer pulling out
    towards the robot along -x), and the grasp robot_tasks.py plans on it: the pads on the
    handle's outer part, the fingertips 2 mm off the drawer's front."""
    rt.DESIGN_NAMES.clear()
    rt.DESIGN_NAMES.update(rt.design_names(run))
    probe, data = _probe(run)
    kind, joint = rt.kind_of(probe)
    if kind != "drawer":
        raise ValueError(f"{run} is a {kind}, not a drawer")
    g = rt.plan_grasp(probe, data, kind, joint)
    b = probe.body(g.body).id
    r = data.xmat[b].reshape(3, 3)
    approach = r @ g.approach
    yaw = -np.arctan2(approach[1], approach[0])  # the hand comes in along +x
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    grasp_w = rz @ (r @ g.pos + data.xpos[b])
    pts = np.vstack([rt.geom_points(probe, data, i) for i in range(probe.ngeom)
                     if probe.body(probe.geom_bodyid[i]).name.startswith("obj_") and probe.geom_contype[i] != 0])
    pos = HANDLE_AT - grasp_w
    pos[2] = max(pos[2], -(pts @ rz.T)[:, 2].min())  # never below the floor
    jid = probe.joint(joint).id
    root = next(probe.body(i).name for i in range(probe.nbody) if probe.body(i).name.startswith("obj_"))
    handle = next(i for i in range(probe.ngeom) if probe.geom_bodyid[i] == b
                  and "handle" in rt.DESIGN_NAMES.get(probe.geom(i).name[4:], "").lower())
    fingers = r @ g.fingers  # the fingers close along this: the handle's height between them
    half = float(np.ptp(rt.geom_points(probe, data, handle) @ fingers)) / 2
    return {"joint": joint, "root": root, "drawer": probe.body(probe.jnt_bodyid[jid]).name, "grasp": g,
            "pos": pos, "yaw": yaw, "range": probe.jnt_range[jid].copy(),
            "foot": (pts @ rz.T + pos)[:, :2], "bottom": float((pts @ rz.T)[:, 2].min() + pos[2]),
            "handle_half_m": half, "pinch_m": half - PINCH_SLACK}


def object_parts(run: Path, info: dict) -> tuple[str, str, str]:
    """(assets, body, contact excludes) of the cabinet as placed, all of its measured collision on.
    The only excluded pair is the cabinet and its own drawer (the drawer slides in its slot)."""
    return rt.object_xml(run / "model.xml", info["pos"], info["yaw"], False, handles=True)


def pedestal_geom(info: dict) -> str:
    """A stand under the cabinet, so it doesn't float in the video (the cabinet is welded to the
    world either way; the stand doesn't collide)."""
    foot = info["foot"]
    cx, cy = (foot[:, 0].min() + foot[:, 0].max()) / 2, (foot[:, 1].min() + foot[:, 1].max()) / 2
    hx, hy = np.ptp(foot[:, 0]) / 2 + 0.03, np.ptp(foot[:, 1]) / 2 + 0.03
    top = info["bottom"]
    if top < 0.005:
        return ""
    return (f'<geom name="pedestal" type="box" pos="{cx:.3f} {cy:.3f} {top / 2:.4f}" '
            f'size="{hx:.3f} {hy:.3f} {top / 2:.4f}" rgba="0.62 0.47 0.33 1" contype="0" conaffinity="0"/>')


def scene(run: Path) -> tuple[str, dict]:
    """The hand-on-a-mount scene XML (loaded from a string: Menagerie's mesh folder is given as an
    absolute path, so nothing is written next to it) and the placement facts."""
    info = placement(run)
    meshes, obj, excludes = object_parts(run, info)
    hand = (PANDA_DIR / "hand.xml").read_text()
    hand = hand.replace('meshdir="assets"', f'meshdir="{(PANDA_DIR / "assets").resolve()}"')
    q = hand_quat()
    origin = MOUNT_AT - hand_rot() @ GRASP  # the hand body's position at the mount's zero
    slides = "".join(f'<joint name="{n}" type="slide" axis="{ax}" damping="0" armature="0"/>'
                     for n, ax in zip(MOUNT, ("1 0 0", "0 1 0", "0 0 1")))
    hand = hand.replace('<body name="hand" childclass="panda" quat="0 0 0 1">',
                        f'<body name="mount" pos="{origin[0]:.4f} {origin[1]:.4f} {origin[2]:.4f}">{slides}'
                        f'<inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>'
                        f'<body name="hand" childclass="panda" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}">')
    hand = hand.replace("</worldbody>", "</body></worldbody>")  # the hand is the last body: close the mount after it
    hand = hand.replace("<worldbody>", f"""<worldbody>
    <light pos="0.4 -0.6 1.8" dir="0 0.3 -1" directional="true" castshadow="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" rgba="0.85 0.86 0.88 1"/>
    {pedestal_geom(info)}
    {obj}""", 1)
    hand = re.sub(r"<asset>", "<asset>" + meshes, hand, count=1)
    servos = "".join(f'<position name="{n}" joint="{n}" kp="{MOUNT_KP}" kv="{MOUNT_KV}" forcelimited="true" '
                     f'forcerange="{-MOUNT_FMAX} {MOUNT_FMAX}"/>' for n in MOUNT)
    hand = hand.replace("<actuator>", "<actuator>" + servos, 1)
    hand = hand.replace("</mujoco>", f"<contact>{excludes}</contact>\n"
                        f'<option timestep="0.002" cone="elliptic" impratio="10"/>\n'
                        f'<visual><global offwidth="1280" offheight="720"/></visual>\n</mujoco>')
    info.update(meshes=meshes, obj=obj, excludes=excludes, origin=origin)
    return hand, info


def arm_scene(run: Path) -> tuple[str, dict]:
    """The same cabinet, placed the same way, with the full Panda arm (Menagerie's panda.xml,
    loaded from a string like scene()). Its keyframe is dropped: the drawer adds a joint."""
    info = placement(run)
    meshes, obj, excludes = object_parts(run, info)
    xml = (PANDA_DIR / "panda.xml").read_text()
    xml = xml.replace('meshdir="assets"', f'meshdir="{(PANDA_DIR / "assets").resolve()}"')
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
    xml = xml.replace("<worldbody>", f"""<worldbody>
    <light pos="0.4 -0.6 1.8" dir="0 0.3 -1" directional="true" castshadow="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="floor"/>
    {pedestal_geom(info)}
    {obj}""", 1)
    xml = xml.replace("<asset>", """<asset>
    <texture type="skybox" builtin="gradient" rgb1="0.92 0.94 0.97" rgb2="0.75 0.8 0.86" width="512" height="3072"/>
    <texture type="2d" name="floor" builtin="checker" mark="edge" rgb1="0.85 0.86 0.88" rgb2="0.8 0.81 0.83"
      markrgb="0.7 0.7 0.7" width="300" height="300"/>
    <material name="floor" texture="floor" texuniform="true" texrepeat="5 5" reflectance="0.05"/>""" + meshes, 1)
    xml = xml.replace("</mujoco>", f"""<contact>{excludes}</contact>
  <option timestep="0.002" cone="elliptic" impratio="10"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
  </visual>
</mujoco>""")
    info.update(meshes=meshes, obj=obj, excludes=excludes)
    return xml, info


class Contacts:
    """What the hand touches: the strict pinch test and the honesty bookkeeping, shared by the
    env and the full arm (scripts/run_drawer_policy.py). ``robot``: more bodies that count as the
    robot (the arm's links) when looking for contacts with anything but the handle."""

    def __init__(self, m: mujoco.MjModel, joint: str, pinch_m: float, robot=()):
        self.m, self.pinch_m = m, pinch_m
        jid = m.joint(joint).id
        drawer = m.jnt_bodyid[jid]
        self.handle = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == drawer
                           and "handle" in rt.DESIGN_NAMES.get(m.geom(g).name[4:], "").lower())
        self.left, self.right = m.body("left_finger").id, m.body("right_finger").id
        self.hand_bodies = {m.body("hand").id, self.left, self.right, *robot}
        self.finger_qadr = m.joint("finger_joint1").qposadr[0]
        self.finger2_qadr = m.joint("finger_joint2").qposadr[0]
        self.hand = m.body("hand").id
        self._f = np.zeros(6)

    def finger_open(self, d) -> float:
        """The smaller of the two fingers' openings (m)."""
        return float(min(d.qpos[self.finger_qadr], d.qpos[self.finger2_qadr]))

    def pinched(self, d) -> bool:
        """Both fingers press on the handle from opposite sides along the closing direction, each
        finger at least pinch_m open. Contact alone isn't enough: a policy learned to crush the
        fingers past their joint limit into a gap behind the handle and hook it with one finger;
        fingertips pushed against the handle's face, or pads slipping round its edge, don't count."""
        if self.finger_open(d) < self.pinch_m:
            return False
        close = d.xmat[self.hand].reshape(3, 3)[:, 1]  # hand y: the left finger opens along +y
        sides = set()
        for c in d.contact[:d.ncon]:
            if self.handle not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.handle else c.geom1
            b = self.m.geom_bodyid[other]
            if b not in (self.left, self.right):
                continue
            n = np.asarray(c.frame[:3])  # contact normal, from geom1 to geom2
            n_out = n if c.geom1 == self.handle else -n  # from the handle towards the finger
            along = float(n_out @ close)
            if b == self.left and along > PINCH_COS:
                sides.add("left")
            elif b == self.right and along < -PINCH_COS:
                sides.add("right")
        return len(sides) == 2

    def other_contacts(self, d) -> tuple[float, str, float]:
        """The hand's contacts with anything but the handle: the deepest (m), what it touched, and
        the total normal force (N)."""
        deep, what, force = 0.0, "", 0.0
        for i, c in enumerate(d.contact[:d.ncon]):
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if (b1 in self.hand_bodies) == (b2 in self.hand_bodies) or self.handle in (c.geom1, c.geom2):
                continue
            if c.efc_address >= 0:
                mujoco.mj_contactForce(self.m, d, i, self._f)
                force += float(self._f[0])
            if -c.dist > deep:
                other = c.geom2 if b1 in self.hand_bodies else c.geom1
                deep = -float(c.dist)
                what = self.m.geom(other).name or self.m.body(self.m.geom_bodyid[other]).name
        return deep, what, force


def new_stats() -> dict:
    return {"travel_m": 0.0, "held_travel_m": 0.0, "held_steps": 0, "finger_open_held_min_m": None,
            "finger_open_min_m": 1.0, "deepest_other_m": 0.0, "deepest_other_what": "", "push_force_max_n": 0.0,
            "q_min_m": 0.0, "q_peak_m": 0.0}


def track_substep(stats: dict, contacts: Contacts, d, q: float) -> None:
    """Honesty bookkeeping after every physics step (the same on the mount and on the arm)."""
    deep, what, force = contacts.other_contacts(d)
    if deep > stats["deepest_other_m"]:
        stats["deepest_other_m"], stats["deepest_other_what"] = deep, what
    stats["push_force_max_n"] = max(stats["push_force_max_n"], force)
    stats["finger_open_min_m"] = min(stats["finger_open_min_m"], contacts.finger_open(d))
    stats["q_min_m"] = min(stats["q_min_m"], q)
    stats["q_peak_m"] = max(stats["q_peak_m"], q)


def track_step(stats: dict, contacts: Contacts, d, dq: float, held: bool) -> None:
    """Honesty bookkeeping after every action: drawer travel, and how much of it was held."""
    if dq > 0:
        stats["travel_m"] += dq
        if held:
            stats["held_travel_m"] += dq
    if held:
        stats["held_steps"] += 1
        fo = contacts.finger_open(d)
        prev = stats["finger_open_held_min_m"]
        stats["finger_open_held_min_m"] = fo if prev is None else min(prev, fo)


class DrawerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, run: str | Path, seed: int = 0, randomize: bool = True):
        self.run, self.randomize = Path(run), randomize
        xml, self.info = scene(self.run)
        self.model = m = mujoco.MjModel.from_xml_string(xml)
        harden(m, self.info["joint"])
        self.data = mujoco.MjData(m)
        self.jid = m.joint(self.info["joint"]).id
        self.qadr, self.dadr = m.jnt_qposadr[self.jid], m.jnt_dofadr[self.jid]
        self.root = m.body(self.info["root"]).id
        self.drawer = m.body(self.info["drawer"]).id
        self.root_pos0, self.root_quat0 = m.body_pos[self.root].copy(), m.body_quat[self.root].copy()
        self.friction0 = float(m.dof_frictionloss[self.dadr])
        self.hand = m.body("hand").id
        self.mount_q = np.array([m.joint(n).qposadr[0] for n in MOUNT])
        self.mount_v = np.array([m.joint(n).dofadr[0] for n in MOUNT])
        self.mount_u = np.array([m.actuator(n).id for n in MOUNT])
        mount = m.body("mount").id
        carried = [b for b in range(m.nbody) if b == mount or m.body_parentid[b] == mount
                   or m.body_parentid[m.body_parentid[b]] == mount]
        self.gravity_comp = float(m.body_mass[carried].sum() * -m.opt.gravity[2])
        self.grip = m.actuator("actuator8").id
        self.contacts = Contacts(m, self.info["joint"], self.info["pinch_m"])
        self.handle = self.contacts.handle
        self.rot = hand_rot()
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (15,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.q_max = float(self.info["range"][1])

    def _grasp_point(self) -> np.ndarray:
        return self.data.xpos[self.hand] + self.data.xmat[self.hand].reshape(3, 3) @ GRASP

    def _grasp_target(self) -> np.ndarray:
        """The planned grasp point on the handle, in world (it moves with the drawer)."""
        g = self.info["grasp"]
        return self.data.xmat[self.drawer].reshape(3, 3) @ g.pos + self.data.xpos[self.drawer]

    def _obs(self) -> np.ndarray:
        return observation(self.model, self.data, self.hand, self.contacts, self.qadr, self.dadr, self.q_max,
                           self._grasp_target(), self.jid, self._held)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        if self.randomize:  # where the cabinet stands, how it's turned, how stiff the drawer runs
            m.body_pos[self.root] = self.root_pos0 + np.array([*self.rng.uniform(-0.03, 0.03, 2), 0])
            a = self.rng.uniform(-np.deg2rad(10), np.deg2rad(10))
            qz = np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, qz, self.root_quat0)
            m.body_quat[self.root] = q
            m.dof_frictionloss[self.dadr] = self.friction0 * self.rng.uniform(0.5, 1.5)
        mujoco.mj_resetData(m, d)
        start = HANDLE_AT + np.array([-0.22, 0, 0.0]) + (self.rng.uniform([-0.05, -0.1, -0.05], [0.05, 0.1, 0.1])
                                                         if self.randomize else 0)
        d.qpos[self.mount_q] = d.ctrl[self.mount_u] = start - MOUNT_AT
        d.qfrc_applied[self.mount_v[2]] = self.gravity_comp  # as the real arm's controller holds its own weight
        d.ctrl[self.grip] = 255
        d.qpos[self.contacts.finger_qadr] = d.qpos[self.contacts.finger2_qadr] = 0.04
        mujoco.mj_forward(m, d)
        self.t, self.best, self.q_prev, self.held_travel = 0, 0.0, 0.0, 0.0
        self._held = False
        self.stats = new_stats()
        return self._obs(), {}

    def step(self, action):
        m, d = self.model, self.data
        a = np.clip(action, -1, 1)
        # The new setpoint is one action's move from where the hand is (as the arm runs it): a
        # setpoint that kept going would drag the hand through whatever stops it. It slides there
        # over the step, as a controller interpolates: one jump per step snapped the hand (and
        # whatever it held) forward with a jerk that tore the pinch.
        start = d.ctrl[self.mount_u].copy()
        goal = d.qpos[self.mount_q] + a[:3] * STEP_M
        d.ctrl[self.grip] = 0 if a[3] > 0 else 255
        for k in range(SUBSTEPS):
            d.ctrl[self.mount_u] = start + (goal - start) * (k + 1) / SUBSTEPS
            mujoco.mj_step(m, d)
            track_substep(self.stats, self.contacts, d, float(d.qpos[self.qadr]))
        self.t += 1
        p = self._grasp_point()
        dist = float(np.linalg.norm(self._grasp_target() - p))
        q = float(d.qpos[self.qadr])
        held = self._held = self.contacts.pinched(d)
        # Only travel made while pinching the handle counts (in the reward and for success):
        # pushing or hooking the drawer out isn't the task. The episode runs its full length:
        # ending it on success would make staying near the handle (the reach reward) worth more
        # than opening it.
        dq = q - self.q_prev
        if dq > 0 and held:
            self.held_travel += dq
        track_step(self.stats, self.contacts, d, dq, held)
        self.q_prev = q
        # hooked out first and grabbed at the end: not counted; pulled past the measured range: not counted
        self.best = max(self.best, min(self.held_travel, q, self.q_max))
        reward = (1 - np.tanh(8 * dist)) + 1.0 * held + 10.0 * self.best / self.q_max
        truncated = self.t >= EPISODE
        success = self.best >= SUCCESS_M
        info = {"success": success, "drawer_m": self.best, "drawer_now_m": q, "held": held}
        if truncated:
            info["stats"] = dict(self.stats)
        return self._obs(), float(reward), False, truncated, info

    def render(self, renderer=None, camera=None):
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()


def observation(m, d, hand: int, contacts: Contacts, qadr: int, dadr: int, q_max: float, target: np.ndarray,
                jid: int, held: bool) -> np.ndarray:
    """What the policy sees, the same on the mount and on the arm: the grasp point, its velocity,
    the finger opening, the planned grasp point relative to it, the drawer's opening and speed,
    the slide's direction (from the measured model), and whether the handle is pinched."""
    r = d.xmat[hand].reshape(3, 3)
    p = d.xpos[hand] + r @ GRASP
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, hand, vel, 0)  # [angular, linear] at the body origin, world frame
    axis = d.xaxis[jid]
    return np.concatenate([p, vel[3:], [contacts.finger_open(d) * 25], target - p,
                           [d.qpos[qadr] / q_max, d.qvel[dadr]], axis[:2], [float(held)]]).astype(np.float32)
