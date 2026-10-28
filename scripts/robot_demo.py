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
WIDTH_HALF = 0.15  # m; depth across the motion plane is not measured from a side view
PEDESTAL_TOP = 0.30
OBJECT_X = 0.62
W, H, FPS = 1920, 1080, 30
LIFT = np.array([-0.05, 0, 0.15])


def object_xml(model_xml: Path) -> tuple[str, str, str]:
    """Measured bodies as an MJCF snippet, prefixed, sitting on the pedestal and turned so the
    pushed link's back faces the robot. Returns (xml, contact excludes, pushed joint)."""
    tree = ET.parse(model_xml)
    world = tree.getroot().find("worldbody")
    root = world.find("body")
    for el in root.iter():
        if "name" in el.attrib:
            el.set("name", "obj_" + el.get("name"))
        if el.tag == "geom":
            size = [float(v) for v in el.get("size").split()]
            size[1] = WIDTH_HALF
            el.set("size", " ".join(f"{v:.5f}" for v in size))
            el.set("density", "300")
            el.set("rgba", "0.75 0.76 0.78 1")
        if el.tag == "joint":
            el.set("damping", "0.3")
            el.set("frictionloss", "0.5")  # laptop hinges are stiff: the lid holds its angle
            el.set("armature", "0.01")
    joint = next(el.get("name") for el in root.iter("joint") if el.get("type") == "hinge")
    # Link boxes overlap at their joints. MuJoCo skips parent-child contacts, except when the
    # parent is welded to the world, as the fixed root is here, so exclude them explicitly.
    excludes = "".join(f'<exclude body1="{parent.get("name")}" body2="{child.get("name")}"/>'
                       for parent in root.iter("body") for child in parent.findall("body"))
    # Lift so the root link's box rests on the pedestal.
    g = root.find("geom")
    pz = float(g.get("pos").split()[2])
    hx, _, hz = (float(v) for v in g.get("size").split())
    a = float(g.get("axisangle").split()[3])
    bottom = pz - (abs(hx * np.sin(a)) + abs(hz * np.cos(a)))
    snippet = ET.tostring(root, encoding="unicode")
    wrapper = (f'<body name="object" pos="{OBJECT_X} 0 {PEDESTAL_TOP - bottom + 0.001:.4f}" euler="0 0 3.14159">'
               f"{snippet}</body>")
    return wrapper, excludes, joint


def scene_xml(obj: str, excludes: str) -> str:
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
  </asset>
  <worldbody>
    <light pos="0.4 -0.6 1.8" dir="0 0.3 -1" directional="true" castshadow="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="floor"/>
    <geom name="pedestal" type="box" pos="{OBJECT_X} 0 {PEDESTAL_TOP / 2}" size="0.2 0.25 {PEDESTAL_TOP / 2}" material="wood"/>
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
            q = np.clip(q + dq, self.lo, self.hi)
            if np.linalg.norm(err_p) < 5e-4:
                break
        return q


def push_target(model, plan: mujoco.MjData, joint: str, q: float, standoff: float, robot: np.ndarray):
    """Fingertip target on the back of the pushed link at joint angle q, and push direction."""
    plan.joint(joint).qpos[0] = q
    mujoco.mj_kinematics(model, plan)
    body = model.joint(joint).bodyid[0]
    gid = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == body)
    c, r, size = plan.geom_xpos[gid], plan.geom_xmat[gid].reshape(3, 3), model.geom_size[gid]
    long_i = 0 if size[0] > size[2] else 2
    thin_i = 2 - long_i
    up = r[:, long_i] if r[2, long_i] > 0 else -r[:, long_i]
    n = r[:, thin_i] if r[:, thin_i] @ (robot - c) > 0 else -r[:, thin_i]
    point = c + up * 0.75 * size[long_i] + n * (size[thin_i] + standoff)
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

    obj, excludes, joint = object_xml(args.run / "model.xml")
    joint_name = "obj_" + joint if not joint.startswith("obj_") else joint
    scene = PANDA_DIR / "video2urdf_demo_scene.xml"  # next to panda.xml so its meshdir resolves
    scene.write_text(scene_xml(obj, excludes))
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    plan = mujoco.MjData(model)
    ik = Ik(model)

    arm_q = HOME.copy()
    data.qpos[ik.qadr] = arm_q
    data.ctrl[:7] = arm_q
    data.ctrl[7] = 0  # gripper closed: push with the fingertips
    mujoco.mj_forward(model, data)
    lo, hi = model.jnt_range[model.joint(joint_name).id]
    q_open, q_closed = data.joint(joint_name).qpos[0], hi - 0.05
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
    lid_q = []
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
            arm_q = ik.solve(arm_q, target[0], target[1])
        data.ctrl[:7] = arm_q
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        lid_q.append(data.joint(joint_name).qpos[0])
        renderer.update_scene(data, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        deg = np.rad2deg(lid_q[-1] - q_open)
        cv2.putText(img, f"measured hinge '{joint}': {deg:5.1f} deg  (measured range 0..{np.rad2deg(hi - q_open):.0f})",
                    (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (40, 40, 40), 3)
        writer.write(img)
    writer.release()
    renderer.close()
    print(f"lid {joint}: {np.rad2deg(lid_q[0] - q_open):.1f} -> {np.rad2deg(lid_q[-1] - q_open):.1f} deg "
          f"(target {np.rad2deg(q_closed - q_open):.1f}); video -> {out}")


if __name__ == "__main__":
    main()
