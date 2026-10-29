"""Physics check, as a video: the measured model let go at an angle where the video shows the
part standing on its own, twice side by side: with the joint friction measured from the video
(it stays) and without it (it falls to its stop). Uses physics.json (physics.apply).

Usage: .venv/bin/python scripts/physics_demo.py data/tracks_agent/laptop_lidar [--out physics_demo.mp4]
"""

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

W, H, FPS, SECONDS = 1920, 1080, 30, 3.0
AZIMUTH = 200  # deg off the hinge axis: the swing in profile, from the screen's side


def staged(model_xml: Path) -> mujoco.MjModel:
    """The model on a floor, under a sky and a shadow-casting light."""
    spec = mujoco.MjSpec.from_file(str(model_xml))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    spec.visual.quality.shadowsize = 4096
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.92, 0.94, 0.97], rgb2=[0.75, 0.8, 0.86], width=512, height=3072)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.85, 0.86, 0.88], rgb2=[0.8, 0.81, 0.83], width=300, height=300)
    spec.add_material(name="floor", textures=["", "grid"], texrepeat=[8, 8], texuniform=True, reflectance=0.05)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="floor",
                            contype=0, conaffinity=0)
    light = spec.worldbody.add_light(pos=[0.3, -0.5, 1.5], dir=[-0.2, 0.3, -1], castshadow=True)
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    model = spec.compile()
    model.vis.headlight.ambient[:] = 0.4
    model.vis.headlight.diffuse[:] = 0.4
    model.vis.headlight.specular[:] = 0.0
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    phys = json.loads((args.run / "physics.json").read_text())
    name, info = next((k, v) for k, v in phys["joints"].items() if v.get("friction_measured"))
    q0 = max(info["probe_q"], key=lambda q: abs(q - np.mean(info["probe_q"])))  # a held angle, the most loaded one

    sims = []
    for friction in (info["friction"], 0.0):
        model = staged(args.run / "model.xml")
        model.dof_frictionloss[model.joint(name).dofadr[0]] = friction
        data = mujoco.MjData(model)
        data.joint(name).qpos[0] = q0
        mujoco.mj_forward(model, data)
        sims.append((model, data, mujoco.Renderer(model, H, W // 2)))

    # Look along the hinge axis so the part's swing shows in profile.
    model, data, _ = sims[0]
    j = model.joint(name)
    axis = data.xaxis[j.id]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.subtree_com[0]
    cam.distance = 0.9
    cam.azimuth = float(np.rad2deg(np.arctan2(axis[1], axis[0]))) + AZIMUTH
    cam.elevation = -10

    out = args.out or args.run / "physics_demo.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
    unit = "N m" if info["type"] == "revolute" else "N"
    titles = [f"joint friction measured from the video: {info['friction']:.3f} {unit}", "no joint friction"]
    steps = int(round(1 / FPS / model.opt.timestep))
    for frame in range(int(SECONDS * FPS)):
        halves = []
        for (m, d, r), title in zip(sims, titles):
            if frame:
                for _ in range(steps):
                    mujoco.mj_step(m, d)
            r.update_scene(d, camera=cam)
            img = r.render()[..., ::-1].copy()
            moved = np.rad2deg(d.joint(name).qpos[0] - q0)
            for text, y, s in ((title, 70, 1.3), (f"let go {frame / FPS:.1f} s ago: moved {moved:+.1f} deg", 125, 1.1)):
                cv2.putText(img, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 8)  # on a light sky
                cv2.putText(img, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, s, (40, 30, 20), 3)
            halves.append(img)
        frame_img = cv2.hconcat(halves)
        cv2.line(frame_img, (W // 2, 0), (W // 2, H), (255, 255, 255), 3)
        writer.write(frame_img)
    writer.release()
    print(f"{name}: let go at {np.rad2deg(q0):.0f} deg (a held angle): "
          + ", ".join(f"{t.split(':')[0]} -> {np.rad2deg(d.joint(name).qpos[0] - q0):+.1f} deg"
                      for (_, d, _), t in zip(sims, titles)) + f"; video -> {out}")


if __name__ == "__main__":
    main()
