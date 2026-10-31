"""Run a trained drawer policy on the full Franka Panda arm, in MuJoCo physics.

The policy was trained with a Panda hand on a force-limited mount (video2urdf/rl_drawer.py).
Here the same observations come from the arm's hand, and each action moves the hand's target,
which the arm follows through IK (as in scripts/robot_tasks.py): the joint setpoints slide to
the new IK solution over the step, the arm's own servos (force-limited) do the work, and the
arm's weight is compensated as the real Panda's controller does. The cabinet is the same as in
training: placed the same way, its measured collision on, the same stiff stops and contacts.

Writes a video (the arm, with a close-up of the hand on the handle) and the arm's result, with
the same honesty numbers as the training test: drawer travel made while pinched, finger
opening on the hold, the robot's deepest contact and largest push on anything but the handle.

Usage: .venv/bin/python scripts/run_drawer_policy.py data/tracks_design/drawer2_lidar
           [--policy data/policies/drawer_ppo_honest/policy.zip] [--out robot_policy.mp4] [--seed 12345]
"""

import argparse
import json
import pickle
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
INSET = (640, 360)  # close-up of the hand on the handle, bottom right


class LsqIk:
    """IK for the grasp point's position and the hand's orientation, joint limits as bounds,
    solved by least squares from the current posture (a small pull towards it keeps the arm on
    one branch). The level hand pointing away from the robot at the handle's height (0.40 m) is
    at the edge of the Panda's joint limits (joint 5 at its stop; the best pose misses by ~1 mm
    and ~2 deg), where robot_tasks.py's damped least squares, which reports only the position
    error, drifted into other branches; here both errors are reported."""

    W_ROT, W_REG = 0.05, 0.002  # 1 deg of tilt ~ 1 mm; a small pull towards the current posture

    def __init__(self, model: mujoco.MjModel):
        self.m, self.d = model, mujoco.MjData(model)
        self.hand = model.body("hand").id
        self.qadr = np.array([model.joint(j).qposadr[0] for j in rt.ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in rt.ARM_JOINTS])
        lo, hi = model.jnt_range[[model.joint(j).id for j in rt.ARM_JOINTS]].T
        self.lo, self.hi = lo + 1e-3, hi - 1e-3
        self.error, self.rot_error_deg = 0.0, 0.0

    def pose(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        r = self.d.xmat[self.hand].reshape(3, 3)
        return self.d.xpos[self.hand] + r @ rt.GRASP, r

    def solve(self, q0, pos, rot, reg: float | None = None):
        from scipy.optimize import least_squares
        reg = self.W_REG if reg is None else reg
        q0 = np.clip(q0, self.lo, self.hi)

        def res(q):
            p, r = self.pose(q)
            return np.concatenate([p - pos, self.W_ROT * (r - rot).ravel(), reg * (q - q0)])
        q = least_squares(res, q0, bounds=(self.lo, self.hi), max_nfev=200, xtol=1e-10, ftol=1e-12).x
        p, r = self.pose(q)
        self.error = float(np.linalg.norm(p - pos))
        self.rot_error_deg = float(np.degrees(np.arccos(np.clip((np.trace(r.T @ rot) - 1) / 2, -1, 1))))
        return q


class CartesianImpedance:
    """Torque control of the arm as the mount it was trained on: at the grasp point a spring
    (MOUNT_KP) and damper (MOUNT_KP x MOUNT_LAG) towards the setpoint, the force clipped to
    MOUNT_FMAX per axis; the hand held level by a stiff rotational spring; a weak pull towards
    the start posture in the null space; gravity compensated; joint torques within the Panda's
    limits. (Joint position servos, as robot_tasks.py uses, pressed the handle with ~300 N
    averaged over a step, ten times what the policy ever did on the mount.) Real Pandas are run
    this way (Franka's Cartesian impedance controller)."""

    K_ROT, D_ROT = 300.0, 12.0  # N m / rad, N m s / rad: the hand level within a few degrees under ~60 N
    K_NULL, D_NULL = 20.0, 3.0  # posture in the null space
    TAU_MAX = np.array([87, 87, 87, 87, 12, 12, 12.0])  # the Panda's joint torque limits (N m)

    def __init__(self, m, d, ik: LsqIk, rot, rest, lag: float = rl.MOUNT_LAG):
        self.m, self.d, self.ik, self.rot, self.rest = m, d, ik, rot, np.asarray(rest, float)
        self.kp, self.kd = rl.MOUNT_KP, rl.MOUNT_KP * lag
        self.setpoint = None
        for i in range(7):  # the joint position servos off: torques come from here
            m.actuator_gainprm[i, 0] = 0.0
            m.actuator_biasprm[i, :3] = 0.0
        self.jp, self.jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def grasp_point(self):
        d = self.d
        return d.xpos[self.ik.hand] + d.xmat[self.ik.hand].reshape(3, 3) @ rt.GRASP

    def torques(self, x_d):
        m, d, ik = self.m, self.d, self.ik
        r = d.xmat[ik.hand].reshape(3, 3)
        x = d.xpos[ik.hand] + r @ rt.GRASP
        mujoco.mj_jac(m, d, self.jp, self.jr, x, ik.hand)
        jp, jr = self.jp[:, ik.dadr], self.jr[:, ik.dadr]
        qd = d.qvel[ik.dadr]
        f = np.clip(self.kp * (x_d - x) - self.kd * (jp @ qd), -rl.MOUNT_FMAX, rl.MOUNT_FMAX)
        e_r = 0.5 * sum(np.cross(r[:, k], self.rot[:, k]) for k in range(3))
        torque = self.K_ROT * e_r - self.D_ROT * (jr @ qd)
        j = np.vstack([jp, jr])
        null = np.eye(7) - np.linalg.pinv(j) @ j
        tau = jp.T @ f + jr.T @ torque + null @ (self.K_NULL * (self.rest - d.qpos[ik.qadr]) - self.D_NULL * qd)
        return np.clip(tau, -self.TAU_MAX, self.TAU_MAX)


def ready_branch(ik: LsqIk, rot, path, seeds: int = 40) -> tuple[np.ndarray, dict]:
    """The arm posture to start from: of many postures that put the level hand on the path's
    first point, the one that follows the whole path (small steps, each solved from the last)
    with the smallest errors and no jump between branches. Returns the posture for the first
    point and the worst errors along the path."""
    rng = np.random.default_rng(0)
    starts = [rng.uniform(ik.lo, ik.hi) for _ in range(seeds)] + [np.clip(s, ik.lo, ik.hi) for s in rt.Ik.SEEDS]
    best, best_score, best_info = None, np.inf, {}
    for s in starts:
        q0 = ik.solve(s, path[0], rot, reg=0.0)
        if ik.error > 0.005 or ik.rot_error_deg > 5:
            continue
        q, worst_p, worst_r, jump = q0, ik.error, ik.rot_error_deg, 0.0
        for p in path[1:]:
            qn = ik.solve(q, p, rot)
            worst_p, worst_r = max(worst_p, ik.error), max(worst_r, ik.rot_error_deg)
            jump = max(jump, float(np.abs(qn - q).max()))
            q = qn
        score = 1000 * worst_p + worst_r + (100 if jump > 0.3 else 0)  # mm + deg; a branch flip disqualifies
        if score < best_score:
            best, best_score = q0, score
            best_info = {"worst_pos_err_mm": round(1000 * worst_p, 2), "worst_rot_err_deg": round(worst_r, 2),
                         "largest_joint_step_rad": round(jump, 3)}
    return best, best_info


class Scripted:
    """A hand-written pinch-and-pull acting on the policy's observation (a check that the arm can
    do the task under the env's rules): in front of the handle, in, close, pull along the slide."""

    def __init__(self, speed: float = 0.5):
        self.phase, self.n, self.speed = 0, 0, speed

    def predict(self, obs, deterministic=True):
        rel, axis = obs[7:10], np.array([obs[12], obs[13], 0.0])
        self.n += 1
        if self.phase == 0:  # 6 cm in front of the handle
            goal = rel - np.array([0.06, 0, 0])
            a, grip = goal / rl.STEP_M, -1.0
            if np.linalg.norm(goal) < 0.004 or self.n > 120:
                self.phase, self.n = 1, 0
        elif self.phase == 1:  # in, fingers open
            a, grip = rel / rl.STEP_M, -1.0
            if np.linalg.norm(rel) < 0.0015 or self.n > 60:
                self.phase, self.n = 2, 0
        elif self.phase == 2:  # close on the handle
            a, grip = rel / rl.STEP_M, 1.0
            if self.n >= 12:
                self.phase, self.n = 3, 0
        else:  # pull along the slide
            a, grip = axis * self.speed, 1.0
        return np.array([*np.clip(a, -1, 1), grip]), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--policy", type=Path, default=ROOT / "data" / "policies" / "drawer_ppo_honest" / "policy.zip")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed", type=int, default=12345, help="the hand's start pose (one the policy hasn't seen)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--scripted", action="store_true", help="a hand-written pinch-and-pull instead of the policy")
    ap.add_argument("--result", type=Path, help="also write the result JSON here")
    ap.add_argument("--control", choices=("cartesian", "joint"), default="cartesian",
                    help="cartesian: torque control as the training mount (default); joint: IK and joint position servos")
    ap.add_argument("--arm-gain", type=float, default=1.0,
                    help="joint control only: scale the joint servos' stiffness and damping (same lag)")
    args = ap.parse_args()
    if args.scripted:
        policy, norm = Scripted(), None
    else:
        from stable_baselines3 import PPO
        policy = PPO.load(args.policy, device="cpu")
        stats = args.policy.parent / "vecnormalize.pkl"
        norm = pickle.loads(stats.read_bytes()) if stats.exists() else None  # the observation scaling it was trained with

    xml, info = rl.arm_scene(args.run)
    m = mujoco.MjModel.from_xml_string(xml)
    rl.harden(m, info["joint"])
    for i in range(7):  # the arm's joint servos: kp and kv scaled together, so the lag (kv / kp) stays ~0.1 s
        m.actuator_gainprm[i, 0] *= args.arm_gain
        m.actuator_biasprm[i, 1:3] *= args.arm_gain
    d = mujoco.MjData(m)
    rot = rl.hand_rot()
    rng = np.random.default_rng(args.seed)  # a start the policy hasn't seen, drawn like the env's
    start = rl.HANDLE_AT + np.array([-0.22, 0, 0.0]) + rng.uniform([-0.05, -0.1, -0.05], [0.05, 0.1, 0.1])
    # the path the hand will take: from the start to the shut handle, along the whole stroke and back
    stroke = info["range"][1]
    path = [start] + [start + u * (rl.HANDLE_AT - start) for u in np.linspace(0.1, 1, 10)] + \
           [rl.HANDLE_AT - [u * (stroke + 0.01), 0, 0] for u in np.linspace(0.05, 1, 20)]
    ik = LsqIk(m)
    ready, ready_info = ready_branch(ik, rot, path)
    print(f"ready posture {np.round(ready, 3)}: along the path {ready_info}")
    cart = CartesianImpedance(m, d, ik, rot, ready) if args.control == "cartesian" else None
    jid = m.joint(info["joint"]).id
    qadr, dadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
    q_max = float(m.jnt_range[jid][1])
    arm = [b for b in range(m.nbody) if m.body(b).name.startswith("link")]
    contacts = rl.Contacts(m, info["joint"], info["pinch_m"], robot=arm)
    drawer = m.body(info["drawer"]).id
    grip = m.actuator("actuator8").id
    hand = ik.hand
    arm_q = ready.copy()  # the arm starts at the start pose, as the env starts its hand there
    d.qpos[ik.qadr] = arm_q
    d.ctrl[:7], d.ctrl[grip] = arm_q, 255
    for j in ("finger_joint1", "finger_joint2"):
        d.qpos[m.joint(j).qposadr[0]] = 0.04
    mujoco.mj_forward(m, d)

    def grasp_point():
        return d.xpos[hand] + d.xmat[hand].reshape(3, 3) @ rt.GRASP

    def grasp_target():
        return d.xmat[drawer].reshape(3, 3) @ info["grasp"].pos + d.xpos[drawer]

    renderer = None if args.no_video else mujoco.Renderer(m, H, W)
    close = None if args.no_video else mujoco.Renderer(m, INSET[1], INSET[0])
    cam = mujoco.MjvCamera()
    p0 = grasp_target()
    cam.lookat[:] = 0.65 * p0 + 0.35 * np.array([0.2, 0, 0.45])
    cam.distance, cam.azimuth, cam.elevation = 1.45, 115, -16  # from the robot's right: robot left, cabinet right
    cam2 = mujoco.MjvCamera()
    cam2.distance, cam2.azimuth, cam2.elevation = 0.30, 70, -12  # beside the handle, on the robot's side of the front
    out = args.out or (args.run / "robot_policy.mp4")
    writer = None if args.no_video else cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
    st = rl.new_stats()
    held = False
    ik_err, rot_err = [], []

    def step_to(target, grip_ctrl, title, joints=None):
        """One 0.04 s step: IK for the target (or the given joint posture), the joint setpoints
        slide there over the step."""
        nonlocal arm_q
        mujoco.mj_kinematics(m, d)
        if cart is not None:  # the setpoint slides from the last one to the target, as on the mount
            d.ctrl[grip] = grip_ctrl
            x_from = cart.setpoint if cart.setpoint is not None else grasp_point().copy()
            for k in range(rl.SUBSTEPS):
                d.qfrc_applied[ik.dadr] = d.qfrc_bias[ik.dadr] + cart.torques(x_from + (target - x_from) * (k + 1) / rl.SUBSTEPS)
                mujoco.mj_step(m, d)
                rl.track_substep(st, contacts, d, float(d.qpos[qadr]))
            cart.setpoint = np.array(target, float)
            r = d.xmat[hand].reshape(3, 3)
            ik_err.append(0.0)
            rot_err.append(float(np.degrees(np.arccos(np.clip((np.trace(r.T @ rot) - 1) / 2, -1, 1)))))
            if writer is not None:
                frame(title)
            return
        if joints is None:
            goal = arm_q + np.clip(ik.solve(arm_q, target, rot) - arm_q, -rt.MAX_JOINT_STEP, rt.MAX_JOINT_STEP)
            ik_err.append(ik.error)
            rot_err.append(ik.rot_error_deg)
        else:
            goal = joints
        q_from = arm_q.copy()
        d.ctrl[grip] = grip_ctrl
        for k in range(rl.SUBSTEPS):
            d.ctrl[:7] = q_from + (goal - q_from) * (k + 1) / rl.SUBSTEPS
            d.qfrc_applied[ik.dadr] = d.qfrc_bias[ik.dadr]  # gravity compensation, as the real Panda has
            mujoco.mj_step(m, d)
            rl.track_substep(st, contacts, d, float(d.qpos[qadr]))
        arm_q = goal
        if writer is not None:
            frame(title)

    def frame(title):
        renderer.update_scene(d, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cam2.lookat[:] = grasp_target() + np.array([0.0, 0.0, 0.0])
        close.update_scene(d, camera=cam2)
        inset = cv2.cvtColor(close.render(), cv2.COLOR_RGB2BGR)
        x0, y0 = W - INSET[0] - 30, H - INSET[1] - 30
        img[y0:y0 + INSET[1], x0:x0 + INSET[0]] = inset
        cv2.rectangle(img, (x0, y0), (x0 + INSET[0], y0 + INSET[1]), (60, 60, 60), 2)
        cv2.putText(img, "close-up: fingers on the handle", (x0 + 12, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (40, 40, 40), 2)
        cv2.putText(img, title, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (30, 30, 30), 4)
        cv2.putText(img, f"drawer: {1000 * d.qpos[qadr]:.0f} mm out (measured slide, physics)", (40, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2)
        cv2.putText(img, "handle pinched" if held else "not holding", (40, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (40, 120, 40) if held else (60, 60, 160), 2)
        writer.write(img)

    for _ in range(10):
        step_to(start, 255, "Start pose (fingers open)")
    approach = dict(st)  # settling at the start, before the policy acts
    ik_err.clear()
    rot_err.clear()
    st = rl.new_stats()  # the policy's episode starts here
    q_prev, held_travel, best, success_at = float(d.qpos[qadr]), 0.0, 0.0, None
    for t in range(rl.EPISODE):
        o = rl.observation(m, d, hand, contacts, qadr, dadr, q_max, grasp_target(), jid, held)
        a, _ = policy.predict(norm.normalize_obs(o) if norm is not None else o, deterministic=True)
        a = np.clip(a, -1, 1)
        # from where the hand is, as in training: a target that kept accumulating would run away
        # from the lagging hand
        step_to(grasp_point() + a[:3] * rl.STEP_M, 0 if a[3] > 0 else 255,
                "Hand-written pinch and pull (check)" if args.scripted else "PPO policy, trained in sim on the drawer measured from video")
        held = contacts.pinched(d)
        q = float(d.qpos[qadr])
        dq = q - q_prev
        if dq > 0 and held:
            held_travel += dq
        rl.track_step(st, contacts, d, dq, held)
        q_prev = q
        best = max(best, min(held_travel, q, q_max))
        if success_at is None and best >= rl.SUCCESS_M:
            success_at = t + 1
    if writer is not None:
        writer.release()
        renderer.close()
        close.close()
    result = {"policy": "scripted" if args.scripted else str(args.policy), "seed": args.seed, "control": args.control,
              "arm_gain": args.arm_gain if args.control == "joint" else None,
              "success": best >= rl.SUCCESS_M,
              "drawer_mm_held": round(1000 * best, 1), "drawer_end_mm": round(1000 * float(d.qpos[qadr]), 1),
              "success_at_step": success_at, "ready_branch": ready_info,
              "ik_error_max_mm": round(1000 * max(ik_err), 2) if cart is None else None,
              "hand_tilt_max_deg": round(max(rot_err), 2), "hand_tilt_median_deg": round(float(np.median(rot_err)), 2),
              "approach_deepest_contact_mm": round(1000 * approach["deepest_other_m"], 2),
              "approach_deepest_contact_what": approach["deepest_other_what"],
              "held_share_of_travel": round(st["held_travel_m"] / st["travel_m"], 4) if st["travel_m"] > 0 else None,
              "travel_mm": round(1000 * st["travel_m"], 1), "held_steps": st["held_steps"],
              "finger_open_on_hold_min_mm": None if st["finger_open_held_min_m"] is None
              else round(1000 * st["finger_open_held_min_m"], 2),
              "finger_open_min_any_mm": round(1000 * st["finger_open_min_m"], 2),
              "deepest_contact_not_handle_mm": round(1000 * st["deepest_other_m"], 2),
              "deepest_contact_not_handle_what": st["deepest_other_what"],
              "net_force_on_drawer_max_n": round(st["net_force_drawer_max_n"], 1),
              "net_force_on_cabinet_max_n": round(st["net_force_cabinet_max_n"], 1),
              "net_force_on_drawer_step_avg_max_n": round(st["net_force_drawer_step_max_n"], 1),
              "steps_with_step_avg_force_over_60n": st["net_force_drawer_steps_over_60n"],
              "contact_normal_sum_not_handle_max_n": round(st["normal_sum_not_handle_max_n"], 1),
              "drawer_min_mm": round(1000 * st["q_min_m"], 2), "drawer_peak_mm": round(1000 * st["q_peak_m"], 2),
              "video": None if args.no_video else str(out)}
    print(json.dumps(result, indent=1))
    if args.result:
        args.result.write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
