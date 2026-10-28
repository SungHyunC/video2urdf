"""Robot demo: a Franka Panda closes an object measured from video, in MuJoCo physics.

The object comes from a fitted run (model.xml); the robot pushes the first revolute link
along its measured hinge until the joint reaches its measured upper limit. Contact does the
work: the robot only follows a path, the lid moves because it is pushed.

Needs the Panda from MuJoCo Menagerie (Apache-2.0): run scripts/fetch_assets.sh once.

Usage: .venv/bin/python scripts/robot_demo.py data/tracks/notebook_2 [--out demo.mp4]
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PANDA_DIR = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda"
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
FINGERTIP = np.array([0, 0, 0.11])  # in the hand frame
PEDESTAL_TOP = 0.22  # low enough that an open lid clears the hand at home
HINGE_X = 0.50  # where the pushed joint sits in front of the robot
W, H, FPS = 1920, 1080, 30
LIFT = np.array([-0.05, 0, 0.15])
MAX_JOINT_STEP = 0.06  # rad per frame (~1.8 rad/s)
START_OPENING_DEG = 80


def object_xml(model_xml: Path, pos, yaw: float) -> tuple[str, str, str, str]:
    """Measured bodies as an MJCF snippet, prefixed, at ``pos`` turned by ``yaw`` about z.
    Returns (assets, body, contact excludes, pushed joint)."""
    tree = ET.parse(model_xml)
    assets = tree.getroot().find("asset")
    mesh_xml = ""
    if assets is not None:
        for mesh in assets.iter("mesh"):
            mesh.set("name", "obj_" + mesh.get("name"))
            mesh.set("file", str((model_xml.parent / mesh.get("file")).resolve()))  # Panda's meshdir is elsewhere
            mesh_xml += ET.tostring(mesh, encoding="unicode")
    root = tree.getroot().find("worldbody").find("body")
    for el in root.iter():
        if "name" in el.attrib:
            el.set("name", "obj_" + el.get("name"))
        if el.tag == "geom":
            if el.get("mesh"):
                el.set("mesh", "obj_" + el.get("mesh"))
            el.set("density", "300")
        if el.tag == "joint":
            el.set("damping", "0.3")
            el.set("frictionloss", "0.5")  # laptop hinges are stiff: the lid holds its angle
            el.set("armature", "0.01")
    joint = next(el.get("name") for el in root.iter("joint") if el.get("type") == "hinge")
    # Links overlap at their joints. MuJoCo skips parent-child contacts, except when the
    # parent is welded to the world, as the fixed root is here, so exclude them explicitly.
    excludes = "".join(f'<exclude body1="{parent.get("name")}" body2="{child.get("name")}"/>'
                       for parent in root.iter("body") for child in parent.findall("body"))
    body = (f'<body name="object" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" euler="0 0 {yaw:.5f}">'
            f"{ET.tostring(root, encoding='unicode')}</body>")
    return mesh_xml, body, excludes, joint


def geom_points(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> np.ndarray:
    """A geom's surface points in world: mesh vertices, or a box's corners."""
    r, c = data.geom_xmat[gid].reshape(3, 3), data.geom_xpos[gid]
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[gid]
        local = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
    else:
        local = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * model.geom_size[gid]
    return local @ r.T + c


def scene_xml(meshes: str, obj: str, excludes: str, pedestal_xy=(0.6, 0.0)) -> str:
    return f"""<mujoco model="video2urdf robot demo">
  <include file="panda.xml"/>
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
    <geom name="pedestal" type="box" pos="{pedestal_xy[0]:.3f} {pedestal_xy[1]:.3f} {PEDESTAL_TOP / 2}" size="0.22 0.25 {PEDESTAL_TOP / 2}" material="wood"/>
    {obj}
  </worldbody>
  <contact>{excludes}</contact>
</mujoco>"""


class Ik:
    """Damped least-squares IK on a kinematics-only copy of the model."""

    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.d = mujoco.MjData(model)
        self.hand = model.body("hand").id
        self.qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        lo, hi = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS]].T
        self.lo, self.hi = lo, hi

    def solve(self, q: np.ndarray, pos: np.ndarray, z_axis: np.ndarray, iters: int = 60) -> np.ndarray:
        q = q.copy()
        for _ in range(iters):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            r = self.d.xmat[self.hand].reshape(3, 3)
            tip = self.d.xpos[self.hand] + r @ FINGERTIP
            jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
            mujoco.mj_jac(self.m, self.d, jp, jr, tip, self.hand)
            err_p = pos - tip
            err_r = np.cross(r[:, 2], z_axis)  # turn the approach axis onto z_axis
            j = np.vstack([jp[:, self.dadr], 0.3 * jr[:, self.dadr]])
            e = np.concatenate([err_p, 0.3 * err_r])
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(6), e)
            dq += 0.05 * (np.eye(7) - np.linalg.pinv(j) @ j) @ (HOME - q)  # stay near home
            q = np.clip(q + np.clip(dq, -0.1, 0.1), self.lo, self.hi)  # small steps: stay on one solution branch
            if np.linalg.norm(err_p) < 5e-4:
                break
        self.error = float(np.linalg.norm(err_p))
        return q


def push_target(model, plan: mujoco.MjData, joint: str, q: float, standoff: float, robot: np.ndarray):
    """Fingertip target on the back of the pushed link at joint angle q, and push direction."""
    plan.joint(joint).qpos[0] = q
    mujoco.mj_kinematics(model, plan)
    body = model.joint(joint).bodyid[0]
    pts = np.vstack([geom_points(model, plan, g) for g in range(model.ngeom) if model.geom_bodyid[g] == body])
    c = pts.mean(axis=0)
    axes = np.linalg.svd(pts - c, full_matrices=False)[2]
    width = np.argmax(np.abs(axes[:, 1]))  # across the motion plane (world y)
    rest = [i for i in range(3) if i != width]
    half = {i: np.ptp((pts - c) @ axes[i]) / 2 for i in rest}
    long_i, thin_i = sorted(rest, key=lambda i: -half[i])
    up = axes[long_i] if axes[long_i][2] > 0 else -axes[long_i]
    n = axes[thin_i] if axes[thin_i] @ (robot - c) > 0 else -axes[thin_i]
    point = c + up * 0.75 * half[long_i] + n * (half[thin_i] + standoff)
    return point, -n


def smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="fitted run folder with model.xml")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if not (PANDA_DIR / "panda.xml").exists():
        sys.exit("Panda model missing: run scripts/fetch_assets.sh")

    scene = PANDA_DIR / "video2urdf_demo_scene.xml"  # next to panda.xml so its meshdir resolves
    def build(pos, yaw, pedestal_xy=(0.6, 0.0)):
        meshes, obj, excludes, joint = object_xml(args.run / "model.xml", pos, yaw)
        scene.write_text(scene_xml(meshes, obj, excludes, pedestal_xy))
        return mujoco.MjModel.from_xml_path(str(scene)), joint  # already prefixed

    def body_points(model, data, body):
        return np.vstack([geom_points(model, data, g) for g in range(model.ngeom) if model.geom_bodyid[g] == body])

    # Probe the object where it was measured: which joint limit closes it (the pushed link
    # sits lowest), and which way it moves while closing.
    model, joint_name = build((0, 0, 0), 0.0)
    probe = mujoco.MjData(model)
    jid = model.joint(joint_name).id
    pushed = model.jnt_bodyid[jid]
    lo, hi = model.jnt_range[jid]
    centroid = {}
    for q in (0.0, lo, hi):
        probe.joint(joint_name).qpos[0] = q
        mujoco.mj_kinematics(model, probe)
        centroid[q] = body_points(model, probe, pushed).mean(axis=0)
    closed_end = lo if centroid[lo][2] < centroid[hi][2] else hi
    move = centroid[closed_end] - centroid[0.0]
    yaw = -np.arctan2(move[1], move[0])  # closing pushes along +x, away from the robot

    # Turned that way, rest it on the pedestal with the hinge HINGE_X in front of the robot.
    model, joint_name = build((0, 0, 0), yaw)
    probe = mujoco.MjData(model)
    mujoco.mj_kinematics(model, probe)
    obj_body = model.body("object").id
    pts = np.vstack([geom_points(model, probe, g) for g in range(model.ngeom)
                     if model.body_rootid[model.geom_bodyid[g]] == obj_body])
    anchor = probe.xanchor[model.joint(joint_name).id]
    pos = np.array([HINGE_X - anchor[0], -anchor[1], PEDESTAL_TOP - pts[:, 2].min() + 0.001])
    model, joint_name = build(pos, yaw, (pts[:, 0].mean() + pos[0], pts[:, 1].mean() + pos[1]))
    data = mujoco.MjData(model)
    plan = mujoco.MjData(model)
    ik = Ik(model)

    # Start the lid at most START_OPENING_DEG open, inside its measured range: opened further,
    # a lid leans back over the robot and its back can only be reached from underneath.
    q_ref = data.joint(joint_name).qpos[0]
    toward_open = np.sign(q_ref - closed_end)
    data.joint(joint_name).qpos[0] = closed_end + toward_open * min(abs(q_ref - closed_end),
                                                                   np.deg2rad(START_OPENING_DEG))
    arm_q = HOME.copy()
    data.qpos[ik.qadr] = arm_q
    data.ctrl[:7] = arm_q
    data.ctrl[7] = 0  # gripper closed: push with the fingertips
    mujoco.mj_forward(model, data)
    q_open = data.joint(joint_name).qpos[0]
    q_closed = closed_end - 0.05 * np.sign(closed_end - q_open)  # stop just short of the limit
    robot = np.array([0.0, 0.0, 0.5])

    # Phases (seconds): hold, lift clear of the lid, reach behind it, drop, touch, sweep along
    # the hinge, retract, hold. Home puts the fingers just above the lid, so lift before turning.
    phases = [("hold", 0.8), ("lift", 0.7), ("reach", 1.2), ("descend", 0.8), ("touch", 0.7), ("sweep", 3.2),
              ("retract", 1.2), ("hold", 1.0)]
    total = sum(d for _, d in phases)
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.42, 0, 0.38]
    cam.distance, cam.azimuth, cam.elevation = 1.75, 115, -18
    out = args.out or (args.run / "robot_demo.mp4")
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
    steps_per_frame = round(1 / FPS / model.opt.timestep)

    start_tip, start_axis = None, None
    lid_q, ik_errors = [], []
    for f in range(int(total * FPS)):
        t = f / FPS
        acc, name, u = 0.0, phases[-1][0], 1.0
        for nm, dur in phases:
            if t < acc + dur:
                name, u = nm, (t - acc) / dur
                break
            acc += dur
        pre, pre_n = push_target(model, plan, joint_name, q_open, 0.08, robot)
        touch, _ = push_target(model, plan, joint_name, q_open, 0.012, robot)
        if name == "hold" and t < 1.0:
            target = None
        elif name == "lift":
            if start_tip is None:
                mujoco.mj_kinematics(model, data)
                r = data.xmat[ik.hand].reshape(3, 3)
                start_tip, start_axis = data.xpos[ik.hand] + r @ FINGERTIP, r[:, 2]
            target = (start_tip + smooth(u) * LIFT, start_axis)
        elif name == "reach":
            # Come in along the pushed face's normal, from well off the face: a lid leaning back
            # over the robot has to be approached from underneath, not dropped onto from above.
            high = pre + np.array([0, 0, 0.12])  # come in above the lid, then drop behind it
            target = (start_tip + LIFT + smooth(u) * (high - start_tip - LIFT), pre_n)
        elif name == "descend":
            high = pre + np.array([0, 0, 0.12])
            target = (high + smooth(u) * (pre - high), pre_n)
        elif name == "touch":
            target = (pre + smooth(u) * (touch - pre), pre_n)
        elif name == "sweep":
            q = q_open + smooth(u) * (q_closed - q_open)
            target = push_target(model, plan, joint_name, q, 0.012, robot)
            end_tip, end_n = target
        elif name == "retract":
            target = (end_tip + smooth(u) * np.array([-0.15, 0, 0.12]), end_n)
        else:
            target = None
        if target is not None:
            # Rate-limit the joint targets so a jump in the IK solution can't whip the arm.
            arm_q = arm_q + np.clip(ik.solve(arm_q, target[0], target[1]) - arm_q, -MAX_JOINT_STEP, MAX_JOINT_STEP)
            ik_errors.append(ik.error)
        data.ctrl[:7] = arm_q
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        lid_q.append(data.joint(joint_name).qpos[0])
        renderer.update_scene(data, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        deg = abs(np.rad2deg(lid_q[-1] - q_open))
        cv2.putText(img, f"measured hinge '{joint_name[4:]}': {deg:5.1f} deg closed of "
                         f"{abs(np.rad2deg(closed_end - q_open)):.0f} measured",
                    (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (40, 40, 40), 3)
        writer.write(img)
    writer.release()
    renderer.close()
    print(f"lid {joint_name}: {np.rad2deg(lid_q[0] - q_open):.1f} -> {np.rad2deg(lid_q[-1] - q_open):.1f} deg "
          f"(target {np.rad2deg(q_closed - q_open):.1f}); IK error max {1000 * max(ik_errors):.0f} mm; video -> {out}")


if __name__ == "__main__":
    main()
