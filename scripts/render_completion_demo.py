"""A clip of a completed body for the demo: what was measured, what was estimated, and a probe.

The completed model (scripts/complete_body.py) is shown on a floor: its measured parts in the
video's texture, the boxes completion.py added behind the front in orange. The camera starts
in front, like the phone, and circles to behind and above (the side the phone never saw); then
a probe is pushed at the back with the force completion.py's own check uses (30 N per axis,
door shut) and stops on it. The probe's path is the simulation's, played back slowed down.

Usage:
  .venv/bin/python scripts/render_completion_demo.py data/viewer_new/microwave_lidar_completed \
      --out data/demo_assets/microwave_completion.mp4 [--seconds 18]
"""

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 30
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
ESTIMATED = [0.96, 0.55, 0.16, 1.0]  # the boxes completion.py added (collision)
LINER = [0.72, 0.38, 0.12, 1.0]  # their visual-only cavity liners
FRONT = [0.09, 0.09, 0.1, 1.0]  # the measured front outline where no measured mesh covers it
PREFIX = "completed_"
RECESS = 0.009  # m: a box flush with the front is drawn from this far behind it


def quat_mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def show_front(spec: mujoco.MjSpec, measured: np.ndarray) -> None:
    """A completed box whose face lies on the measured front plane would hide the thin measured
    mesh just behind that plane, and colouring it would paint the measured front as estimated.
    Draw such a box from RECESS behind the front instead, with its front face as a dark,
    visual-only skin set just behind the measured mesh's face (which then shows): the outline
    is measured, the depth is not. ``measured``: the measured meshes' corner points (base frame)."""
    boxes = [g for g in spec.geoms if g.name.startswith(PREFIX) and "liner" not in g.name]
    # local y of a completed box points back from the front (completion.box_pose: u, -n, z)
    fronts = {g.name: (np.array(g.pos, float) - quat_mat(g.quat) @ [0, g.size[1], 0]) for g in boxes}
    normal = quat_mat(boxes[0].quat) @ [0, -1, 0]  # out of the front
    front_v = max(float(f @ normal) for f in fronts.values())
    for g in boxes:
        if float(fronts[g.name] @ normal) < front_v - 0.003:
            continue
        r = quat_mat(g.quat)
        pos, half = np.array(g.pos, float), np.array(g.size[:3], float)  # copies: g.pos is a view
        local = (measured - pos) @ r  # the measured points in the box's frame
        inside = (np.abs(local[:, 0]) < half[0]) & (np.abs(local[:, 2]) < half[2])
        behind = local[inside, 1] + half[1]  # how far behind the box's front face
        behind = behind[(behind > -0.002) & (behind < RECESS)]
        skin_from = float(behind.min()) + 0.0008 if len(behind) else 0.0
        g.pos = pos + r @ [0, RECESS / 2, 0]
        g.size = [half[0], half[1] - RECESS / 2, half[2]]
        depth = RECESS - max(skin_from, 0.0)
        g.parent.add_geom(name="front_skin_" + g.name, type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=pos + r @ [0, -half[1] + max(skin_from, 0.0) + depth / 2, 0],
                          quat=g.quat, size=[half[0], depth / 2, half[2]],
                          rgba=FRONT, contype=0, conaffinity=0, mass=0)


def staged(model_xml: Path, w: int, h: int, start, end, radius: float, force: float,
           front: bool = True) -> mujoco.MjModel:
    """The model on a floor under a light, the completed boxes recoloured, a probe at ``start``
    (``front``: draw the measured front as such, see show_front; off for the probe's simulation)."""
    spec = mujoco.MjSpec.from_file(str(model_xml))
    model0 = spec.compile()
    data0 = mujoco.MjData(model0)
    mujoco.mj_forward(model0, data0)
    measured = []  # the base's measured meshes, corner points of their boxes, in the base's frame
    for g in range(model0.ngeom):
        if model0.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model0.body(model0.geom_bodyid[g]).name == "base":
            c, half = model0.geom_aabb[g][:3], model0.geom_aabb[g][3:]
            local = c + half * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            world = local @ data0.geom_xmat[g].reshape(3, 3).T + data0.geom_xpos[g]
            body = model0.geom_bodyid[g]
            measured.append((world - data0.xpos[body]) @ data0.xmat[body].reshape(3, 3))
    if front and measured:
        show_front(spec, np.vstack(measured))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = w, h
    spec.visual.quality.shadowsize = 4096
    spec.option.gravity = [0, 0, 0]  # as completion.probe: the probe is pushed, not dropped
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.92, 0.94, 0.97], rgb2=[0.75, 0.8, 0.86], width=512, height=3072)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.86, 0.87, 0.89], rgb2=[0.81, 0.82, 0.84], width=300, height=300)
    spec.add_material(name="floor", textures=["", "grid"], texrepeat=[10, 10], texuniform=True, reflectance=0.03)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="floor",
                            contype=0, conaffinity=0)
    light = spec.worldbody.add_light(pos=[0.4, -0.6, 1.6], dir=[-0.2, 0.35, -1], castshadow=True)
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    for g in spec.geoms:
        if g.name.startswith(PREFIX):
            g.rgba = LINER if "liner" in g.name else ESTIMATED
    body = spec.worldbody.add_body(name="probe", pos=list(start))
    for i, ax in enumerate(np.eye(3)):
        body.add_joint(name=f"probe_{i}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=list(ax), damping=20)
    body.add_geom(name="probe", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius, 0, 0], mass=0.3,
                  rgba=[0.85, 0.1, 0.1, 1], solref=[0.002, 1])
    back = -(np.asarray(end, float) - np.asarray(start, float))
    back = 0.25 * back / np.linalg.norm(back)  # the probe's rod trails behind it, along the push
    body.add_geom(name="probe_rod", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[radius * 0.35, 0, 0],
                  fromto=[0, 0, 0, *back], rgba=[0.5, 0.5, 0.52, 1], contype=0, conaffinity=0, mass=0.01)
    for i in range(3):
        a = spec.add_actuator(name=f"push_{i}", target=f"probe_{i}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=2000)
        a.forcelimited = True
        a.forcerange = [-force, force]
    model = spec.compile()
    model.vis.headlight.ambient[:] = 0.42
    model.vis.headlight.diffuse[:] = 0.45
    model.vis.headlight.specular[:] = 0.0
    return model


def probe_path(model: mujoco.MjModel, joint: str, q: float, start, end, seconds: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """The probe's centre at every simulation step (pushed from start toward end, door held at q)."""
    data = mujoco.MjData(model)
    door = model.joint(joint)
    data.qpos[door.qposadr[0]] = q
    model.jnt_range[door.id] = [q - 1e-4, q + 1e-4]
    target = np.asarray(end, float) - np.asarray(start, float)
    data.ctrl[:3] = target
    times, pos = [], []
    adr = [model.joint(f"probe_{i}").qposadr[0] for i in range(3)]
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        times.append(data.time)
        pos.append(np.asarray(start) + data.qpos[adr])
    return np.array(times), np.array(pos)


def label(img: np.ndarray, items: list[tuple]) -> np.ndarray:
    """items: (xy, text, size, rgb, anchor, box) -- text with an optional dark box behind it."""
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    for xy, text, size, rgb, anchor, box in items:
        f = ImageFont.truetype(FONT, size, index=0)
        if box:
            l, t, r, b = d.multiline_textbbox(xy, text, font=f, anchor=anchor, spacing=6)
            d.rounded_rectangle((l - 12, t - 8, r + 12, b + 8), radius=8, fill=(20, 20, 24, 200))
        d.multiline_text(xy, text, font=f, fill=rgb, anchor=anchor, spacing=6)
    return np.array(im)


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="a run completed by scripts/complete_body.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=18.0)
    ap.add_argument("--size", type=int, nargs=2, default=(1100, 800))
    ap.add_argument("--orbit", type=float, nargs=2, default=(0.12, 0.40),
                    help="the camera circles from the front to behind between these fractions of the clip")
    ap.add_argument("--facts-at", type=float, default=0.36, help="fraction where the depth's numbers replace the front's")
    ap.add_argument("--probe", type=float, nargs=2, default=(0.84, 0.10),
                    help="fraction where the probe starts, and the fraction its push takes")
    args = ap.parse_args()
    comp = json.loads((args.run / "completion.json").read_text())
    m3 = json.loads((args.run / "model3d.json").read_text())
    joint = "lid_joint"
    q_shut = np.deg2rad(comp["measured"]["door_shut_q_deg"])
    test = comp["checks"]["probes"]["completed"]["from_behind_door_shut"]
    start, end = np.array(test["start"]), np.array(test["end"])
    w, h = args.size
    sim = staged(args.run / "model.xml", w, h, start, end, radius=0.012, force=30.0, front=False)
    times, path = probe_path(sim, joint, q_shut, start, end)  # the model as built, as completion.probe
    model = staged(args.run / "model.xml", w, h, start, end, radius=0.012, force=30.0)
    data = mujoco.MjData(model)
    data.qpos[model.joint(joint).qposadr[0]] = q_shut
    mujoco.mj_forward(model, data)

    # Where the probe stopped (front coordinates from completion.json) and when it got there.
    stop = path[-1]
    moving = np.linalg.norm(path - stop, axis=1) > 0.001
    t_stop = times[np.flatnonzero(moving)[-1]] if moving.any() else times[0]
    stopped_mm = 1000 * test["stopped_at_front_coords"][1]
    gap = float(np.linalg.norm(stop - np.array(test["stopped_at"])))
    if gap > 0.002:
        raise SystemExit(f"the probe stopped {1000 * gap:.1f} mm from where completion.json says it did")

    # The object's box (completed geoms + measured meshes, not the floor or the probe).
    corners = []
    for g in range(model.ngeom):
        name = model.geom(g).name
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or name.startswith("probe"):
            continue
        c, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        local = c + half * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        corners.append(local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    corners = np.vstack(corners)
    centre = (corners.min(axis=0) + corners.max(axis=0)) / 2
    n = np.array(comp["measured"]["front_normal_model"])
    facing = float(np.rad2deg(np.arctan2(-n[1], -n[0])))  # azimuth that looks at the front

    probe_geoms = [g for g in range(model.ngeom) if model.geom(g).name.startswith("probe")]
    renderer = mujoco.Renderer(model, h, w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = centre
    diag = float(np.linalg.norm(np.ptp(corners, axis=0)))
    size_mm = comp["built"]["size_mm"]
    front_mm = (comp["measured"]["front_width_mm"], comp["measured"]["front_height_mm"])
    bound = comp["bounded"]
    est = comp["estimated"]["dims"]["outer_depth"]
    white, orange = (245, 245, 245), (255, 170, 80)

    frames = int(round(args.seconds * FPS))
    (orbit_start, orbit_end), (probe_start, probe_len) = args.orbit, args.probe  # fractions of the clip
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"avc1"), FPS, (w, h))
    for i in range(frames):
        p = i / max(frames - 1, 1)
        o = ease((p - orbit_start) / (orbit_end - orbit_start))
        side = ease((p - (probe_start - 0.07)) / 0.06)  # then a little to the side, so the push runs across the view
        cam.azimuth = facing + 25 + o * (180 - 25 - 40) - side * 40
        cam.elevation = -14 - o * 24 + side * 10
        cam.distance = (1.55 - 0.15 * o) * diag
        # the probe: still at its start until probe_start, then the simulated path, slowed down
        s = (p - probe_start) / probe_len
        k = 0 if s <= 0 else int(np.searchsorted(times, min(s, 1.0) * t_stop * 1.15))
        k = min(k, len(path) - 1)
        for j in range(3):
            data.qpos[model.joint(f"probe_{j}").qposadr[0]] = (path[k] - start)[j]
        for g in probe_geoms:  # the probe appears just before it is pushed
            model.geom_rgba[g, 3] = 1.0 if p >= probe_start - 0.04 else 0.0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        items = [((24, 22), "orange: estimated\nvideo texture: measured", 26, white, "la", True)]
        if p < args.facts_at:
            items.append(((24, h - 24), f"measured front: {front_mm[0]:.0f} x {front_mm[1]:.0f} mm", 30, white, "ld", True))
        else:
            items.append(((24, h - 24),
                          f"depth {est['value_m'] * 1000:.0f} mm: Muse Spark's estimate\n"
                          f"the video allows {bound['lower_mm']:.0f}-{bound['upper_mm']:.0f} mm",
                          30, orange, "ld", True))
        if p >= probe_start:
            text = ("probe pushed at the back, 30 N (slowed down)" if s < 1.0 else
                    f"the probe stops on the back,\n{stopped_mm:.0f} mm behind the front")
            items.append(((w - 24, 22), text, 28, white, "ra", True))
        writer.write(cv2.cvtColor(label(img, items), cv2.COLOR_RGB2BGR))
    writer.release()
    renderer.close()
    print(f"{args.out}: {frames} frames, body {size_mm[0]:.0f} x {size_mm[1]:.0f} x {size_mm[2]:.0f} mm, "
          f"probe stopped {stopped_mm:.0f} mm behind the front after {t_stop:.3f} s (sim); "
          f"reference frame {m3.get('reference_frame')}")


if __name__ == "__main__":
    main()
