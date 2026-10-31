"""Run a trained drawer policy on the full Franka Panda arm, in MuJoCo physics.

The policy was trained with a free-floating Panda hand (video2urdf/rl_drawer.py). Here the
same observations come from the arm's hand, and each action moves the hand's target, which the
arm follows through IK (as in scripts/robot_tasks.py). Writes a video and the drawer travel.

Usage: .venv/bin/python scripts/run_drawer_policy.py data/tracks_design/drawer2_lidar
           [--policy data/policies/drawer_ppo/policy.zip] [--out robot_policy.mp4]
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import robot_tasks as rt  # noqa: E402
from video2urdf import rl_drawer as rl  # noqa: E402

W, H, FPS = 1920, 1080, 25  # one frame per policy step (0.04 s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--policy", type=Path, default=ROOT / "data" / "policies" / "drawer_ppo" / "policy.zip")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    import pickle
    from stable_baselines3 import PPO
    policy = PPO.load(args.policy, device="cpu")
    stats = args.policy.parent / "vecnormalize.pkl"
    norm = pickle.loads(stats.read_bytes()) if stats.exists() else None  # the observation scaling it was trained with

    _, info = rl.scene(args.run)  # where the env put the cabinet: the handle at rl.HANDLE_AT
    # the same object as the env (handle collision, the cabinet's inner wall), with the arm
    tmp = rt.PANDA_DIR / f"video2urdf_policy_{os.getpid()}.xml"
    tmp.write_text(rt.scene_xml(info["meshes"], info["obj"], info["excludes"], (0.6, 0, 0.2, 0.2, 0.0)))
    try:
        m = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    grip = m.actuator("actuator8").id
    m.actuator_biasprm[grip, 1:3] = [-3000.0, -30.0]
    m.actuator_gainprm[grip, 0] = 3000.0 * 0.04 / 255
    d = mujoco.MjData(m)
    ik = rt.Ik(m)
    jid = m.joint(info["joint"]).id
    qadr, dadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
    q_max = float(m.jnt_range[jid][1])
    handle = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.jnt_bodyid[jid]
                  and "handle" in rt.DESIGN_NAMES.get(m.geom(g).name[4:], "").lower())
    fingers = {m.body("left_finger").id, m.body("right_finger").id}
    hand = ik.hand
    rot = rt.hand_rot(np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))
    arm_q = rt.HOME.copy()
    d.qpos[ik.qadr] = arm_q
    d.ctrl[:7], d.ctrl[7] = arm_q, 255
    mujoco.mj_forward(m, d)

    def grasp_point():
        return d.xpos[hand] + d.xmat[hand].reshape(3, 3) @ rt.GRASP

    def held():
        touching = set()
        for c in d.contact[:d.ncon]:
            pair = {c.geom1, c.geom2}
            if handle in pair:
                b = m.geom_bodyid[(pair - {handle}).pop()]
                if b in fingers:
                    touching.add(b)
        return len(touching) == 2

    def obs():
        p = grasp_point()
        return np.concatenate([p, d.cvel[hand][3:], [d.qpos[m.joint("finger_joint1").qposadr[0]] * 25],
                               d.geom_xpos[handle] - p, [d.qpos[qadr] / q_max, d.qvel[dadr]], [float(held())]]
                              ).astype(np.float32)

    renderer = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.45, 0, 0.42]
    cam.distance, cam.azimuth, cam.elevation = 1.6, 115, -16
    out = args.out or (args.run / "robot_policy.mp4")
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
    bias = np.zeros(3)

    def step_to(target, grip_ctrl, label):
        nonlocal arm_q, bias
        mujoco.mj_kinematics(m, d)
        bias = np.clip(bias + 0.3 * (target - grasp_point()), -0.03, 0.03)
        arm_q = arm_q + np.clip(ik.solve(arm_q, target + bias, rot) - arm_q, -rt.MAX_JOINT_STEP, rt.MAX_JOINT_STEP)
        d.ctrl[:7], d.ctrl[7] = arm_q, grip_ctrl
        for _ in range(rl.SUBSTEPS):
            d.qfrc_applied[ik.dadr] = d.qfrc_bias[ik.dadr]
            mujoco.mj_step(m, d)
        renderer.update_scene(d, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(img, label, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 30), 4)
        cv2.putText(img, f"drawer: {1000 * d.qpos[qadr]:.0f} mm out (measured slide, physics)", (40, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2)
        writer.write(img)

    rng = np.random.default_rng(args.seed)  # a start the policy hasn't seen
    start = rl.HANDLE_AT + np.array([-0.22, 0, 0.0]) + rng.uniform([-0.05, -0.1, -0.05], [0.05, 0.1, 0.1])
    p0 = grasp_point()
    for k in range(50):  # bring the hand to a start pose, as an episode starts
        u = rt.smooth((k + 1) / 50)
        step_to(p0 + u * (start - p0), 255, "moving to a start pose")
    target = grasp_point().copy()
    success_at = None
    for t in range(rl.EPISODE):
        o = obs()
        a, _ = policy.predict(norm.normalize_obs(o) if norm is not None else o, deterministic=True)
        a = np.clip(a, -1, 1)
        # from where the hand is, not an accumulated target: the arm lags the floating hand it was
        # trained with, and an accumulating target would run away from it
        target = grasp_point() + a[:3] * rl.STEP_M
        step_to(target, 0 if a[3] > 0 else 255, "PPO policy trained in sim on the drawer measured from video")
        if success_at is None and d.qpos[qadr] >= rl.SUCCESS_M:
            success_at = t
        if success_at is not None and t > success_at + 25:
            break
    writer.release()
    renderer.close()
    print(f"drawer out {1000 * d.qpos[qadr]:.0f} mm; success at step {success_at}; IK error {1000 * ik.error:.0f} mm; -> {out}")


if __name__ == "__main__":
    main()
