"""A drawer-opening task for reinforcement learning, built on a drawer measured from video.

The cabinet and drawer are the fitted model (its measured collision, joint range and friction;
the handle is the one Shape design drew, colliding here as in scripts/robot_tasks.py). The
robot is a Panda hand (MuJoCo Menagerie) held by a weld to a mocap target: the policy moves the
target in x, y, z and opens or closes the fingers. The hand comes in level, fingers vertical,
as the full arm does in robot_tasks.py, so a trained policy can drive the arm through IK.

Each episode moves and turns the cabinet a little and scales the drawer's friction, so the
policy can't memorise one placement or one friction value.
"""

import os
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
STEP_M = 0.012  # largest target move per action (0.3 m/s)
SUBSTEPS = 20  # 0.04 s per action
EPISODE = 200  # actions (8 s)
SUCCESS_M = 0.15  # drawer pulled this far while held
PINCH_M = 0.004  # each finger at least this far open on a hold: the handle is 18 mm tall between them


def hand_quat() -> np.ndarray:
    """The hand level, approaching along +x, fingers closing vertically."""
    rot = rt.hand_rot(np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot.flatten())
    return q


def scene(run: Path) -> tuple[str, dict]:
    """The scene XML (written next to Menagerie's hand.xml so its meshes resolve) and placement facts."""
    rt.DESIGN_NAMES.clear()
    rt.DESIGN_NAMES.update(rt.design_names(run))
    tmp = PANDA_DIR / f"video2urdf_rl_probe_{os.getpid()}.xml"
    meshes, obj, excludes = rt.object_xml(run / "model.xml", (0, 0, 0), 0.0, False, handles=True)
    tmp.write_text(rt.scene_xml(meshes, obj, excludes, (0.6, 0, 0.2, 0.2, 0.0)).replace(
        '<include file="panda.xml"/>', ""))
    try:
        probe = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    data = mujoco.MjData(probe)
    mujoco.mj_kinematics(probe, data)
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
    handle = rz @ (r @ g.pos + data.xpos[b])
    pts = np.vstack([rt.geom_points(probe, data, i) for i in range(probe.ngeom)
                     if probe.body(probe.geom_bodyid[i]).name.startswith("obj_") and probe.geom_contype[i] != 0])
    floor_z = (pts @ rz.T)[:, 2].min()
    pos = HANDLE_AT - handle
    pos[2] = max(pos[2], -floor_z)  # never below the floor
    root = probe.body(1).name if probe.body(1).name.startswith("obj_") else next(
        probe.body(i).name for i in range(probe.nbody) if probe.body(i).name.startswith("obj_"))
    jid = probe.joint(joint).id
    meshes, obj, excludes = rt.object_xml(run / "model.xml", pos, yaw, False, handles=True,
                                          ghost=probe.body(probe.jnt_bodyid[jid]).name[4:])
    excludes += rt.gripper_excludes(root)
    obj = add_wall(obj, root, probe, data, g, jid)
    hand = (PANDA_DIR / "hand.xml").read_text()
    q = hand_quat()
    start = HANDLE_AT + np.array([-0.2, 0, 0.05]) - rt.hand_rot(np.array([1.0, 0, 0]), np.array([0, 0, 1.0])) @ GRASP
    hand = hand.replace('<body name="hand" childclass="panda" quat="0 0 0 1">',
                        f'<body name="hand" childclass="panda" pos="{start[0]:.4f} {start[1]:.4f} {start[2]:.4f}" '
                        f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"><freejoint name="hand_free"/>')
    hand = hand.replace("<worldbody>", f"""<worldbody>
    <light pos="0.4 -0.6 1.8" dir="0 0.3 -1" directional="true" castshadow="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" rgba="0.85 0.86 0.88 1"/>
    <body name="target" mocap="true" pos="{start[0]:.4f} {start[1]:.4f} {start[2]:.4f}" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"/>
    {obj}""", 1)
    hand = re.sub(r"<asset>", "<asset>" + meshes, hand, count=1)
    hand = hand.replace("<equality>", '<equality><weld body1="target" body2="hand" solref="0.01 1"/>', 1)
    hand = hand.replace("</mujoco>", f"<contact>{excludes}</contact>\n"
                        f'<option timestep="0.002" cone="elliptic" impratio="10"/>\n'
                        f'<visual><global offwidth="1280" offheight="720"/></visual>\n</mujoco>')
    return hand, {"joint": joint, "root": root, "pos": pos, "yaw": yaw, "range": probe.jnt_range[jid].copy(),
                  "meshes": meshes, "obj": obj, "excludes": excludes}  # the object as placed, for the arm


def add_wall(obj: str, root: str, probe, data, grasp, jid: int) -> str:
    """An invisible wall just inside the cabinet's front, on the cabinet (it moves with it). The
    fingers may pass the measured cabinet box, which is coarse around the drawer slots, but not
    into the cabinet: a policy trained without it learned to ram the handle from inside, a
    contact the real arm can't make."""
    rb = probe.body(root).id
    hb = probe.jnt_bodyid[jid]
    handles = [i for i in range(probe.ngeom) if probe.geom_bodyid[i] == hb
               and "handle" in rt.DESIGN_NAMES.get(probe.geom(i).name[4:], "").lower()]
    pts = rt.geom_points(probe, data, handles[0])
    r_root, x_root = data.xmat[rb].reshape(3, 3), data.xpos[rb]
    out = -(data.xmat[hb].reshape(3, 3) @ grasp.approach)  # the way the drawer comes out
    back = float(np.min(pts @ out))  # the handle's back, against the drawer front
    centre = pts.mean(axis=0)
    centre = centre + out * (back - centre @ out - 0.008 - 0.05)  # 8 mm in behind the front, 10 cm thick
    local = r_root.T @ (centre - x_root)
    o = r_root.T @ out
    yaw = np.arctan2(o[1], o[0])
    wall = (f'<body name="wall" pos="{local[0]:.4f} {local[1]:.4f} {local[2]:.4f}" euler="0 0 {yaw:.5f}">'
            f'<geom name="wall" type="box" size="0.05 0.5 0.7" rgba="0 0 0 0" group="4"/></body>')
    i = obj.index(f'<body name="{root}"')
    j = obj.index(">", i) + 1
    return obj[:j] + wall + obj[j:]


def pinched(m, d, handle: int, fingers: set, finger_qadr: int) -> bool:
    """Both fingers touch the handle with the handle between them. Contact alone isn't enough: a
    policy learned to crush the fingers past their joint limit into the gap behind the handle and
    hook it with one finger."""
    if d.qpos[finger_qadr] < PINCH_M:
        return False
    touching = set()
    for c in d.contact[:d.ncon]:
        pair = {c.geom1, c.geom2}
        if handle in pair:
            b = m.geom_bodyid[(pair - {handle}).pop()]
            if b in fingers:
                touching.add(b)
    return len(touching) == 2


class DrawerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, run: str | Path, seed: int = 0, randomize: bool = True):
        self.run, self.randomize = Path(run), randomize
        xml, self.info = scene(self.run)
        path = PANDA_DIR / f"video2urdf_rl_{os.getpid()}_{id(self)}.xml"
        path.write_text(xml)
        try:
            self.model = mujoco.MjModel.from_xml_path(str(path))
        finally:
            path.unlink(missing_ok=True)
        m = self.model
        grip = m.actuator("actuator8").id  # a firm squeeze, as in robot_tasks.py
        m.actuator_biasprm[grip, 1:3] = [-3000.0, -30.0]
        m.actuator_gainprm[grip, 0] = 3000.0 * 0.04 / 255
        self.data = mujoco.MjData(m)
        self.jid = m.joint(self.info["joint"]).id
        self.qadr, self.dadr = m.jnt_qposadr[self.jid], m.jnt_dofadr[self.jid]
        self.root = m.body(self.info["root"]).id
        self.root_pos0, self.root_quat0 = m.body_pos[self.root].copy(), m.body_quat[self.root].copy()
        self.friction0 = float(m.dof_frictionloss[self.dadr])
        self.hand = m.body("hand").id
        self.target = m.body("target").mocapid[0]
        handles = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("obj_")
                   and "handle" in rt.DESIGN_NAMES.get(m.geom(g).name[4:], "").lower()
                   and m.geom_bodyid[g] == m.jnt_bodyid[self.jid]]
        self.handle = handles[0]
        self.fingers = {m.body("left_finger").id, m.body("right_finger").id}
        self.finger_qadr = m.joint("finger_joint1").qposadr[0]
        self.rot = rt.hand_rot(np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (13,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.q_max = float(self.info["range"][1])

    def _grasp_point(self) -> np.ndarray:
        return self.data.xpos[self.hand] + self.data.xmat[self.hand].reshape(3, 3) @ GRASP

    def _obs(self) -> np.ndarray:
        d = self.data
        p = self._grasp_point()
        return np.concatenate([p, d.cvel[self.hand][3:], [d.qpos[self.model.joint("finger_joint1").qposadr[0]] * 25],
                               d.geom_xpos[self.handle] - p, [d.qpos[self.qadr] / self.q_max, d.qvel[self.dadr]],
                               [float(self._held())]]).astype(np.float32)

    def _held(self) -> bool:
        return pinched(self.model, self.data, self.handle, self.fingers, self.finger_qadr)

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
        hand_pos = start - self.rot @ GRASP
        fadr = m.joint("hand_free").qposadr[0]
        d.qpos[fadr:fadr + 3] = hand_pos
        d.qpos[fadr + 3:fadr + 7] = hand_quat()
        d.mocap_pos[self.target] = hand_pos
        d.mocap_quat[self.target] = hand_quat()
        d.ctrl[m.actuator("actuator8").id] = 255
        mujoco.mj_forward(m, d)
        self.t, self.best, self.q_prev, self.held_travel = 0, 0.0, 0.0, 0.0
        return self._obs(), {}

    def step(self, action):
        m, d = self.model, self.data
        a = np.clip(action, -1, 1)
        # from where the hand is (as the arm runs it): a target that keeps going would drag the hand
        # through whatever stops it
        d.mocap_pos[self.target] = d.xpos[self.hand] + a[:3] * STEP_M
        d.ctrl[m.actuator("actuator8").id] = 0 if a[3] > 0 else 255
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
        self.t += 1
        p = self._grasp_point()
        dist = float(np.linalg.norm(d.geom_xpos[self.handle] - p))
        q = float(d.qpos[self.qadr])
        held = self._held()
        # Only travel made while holding the handle counts (in the reward and for success): pushing
        # or hooking the drawer out isn't the task. The episode runs its full length: ending it on
        # success would make staying near the handle (the reach reward) worth more than opening it.
        if held:
            self.held_travel += max(q - self.q_prev, 0.0)
        self.q_prev = q
        self.best = max(self.best, min(self.held_travel, q))  # hooked out first and grabbed at the end: not counted
        reward = (1 - np.tanh(8 * dist)) + 1.0 * held + 10.0 * self.best / self.q_max
        truncated = self.t >= EPISODE
        success = self.best >= SUCCESS_M
        return self._obs(), float(reward), False, truncated, {"success": success, "drawer_m": self.best, "drawer_now_m": q, "held": held}

    def render(self, renderer=None, camera=None):
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()
