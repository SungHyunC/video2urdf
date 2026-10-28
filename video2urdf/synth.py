"""Synthetic ground-truth videos: planar mechanisms rendered in MuJoCo with known joints.

The mechanism moves in the world x-z plane. A fixed camera on -y looks along +y, so it
faces the motion plane like the tripod shot in the real pipeline. Revolute axes are
world y; prismatic axes lie in x-z.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import mujoco
import numpy as np

WIDTH, HEIGHT, FPS = 640, 480, 30
CAMERA = dict(pos=(0.05, -0.8, 0.2), fovy=45.0)


@dataclass
class Joint:
    name: str
    type: str  # "hinge" or "slide"
    child: str  # body the joint moves
    amplitude: float  # rad or m
    period: float  # s
    phase: float = 0.0
    offset: float = 0.0
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def q(self, t: np.ndarray) -> np.ndarray:
        return self.offset + self.amplitude * np.sin(2 * np.pi * t / self.period + self.phase)


@dataclass
class Preset:
    name: str
    bodies: str  # MJCF worldbody content
    joints: list[Joint]
    prompts: dict[str, str] = field(default_factory=dict)  # body -> SAM concept prompt


PRESETS = {
    # Desk lamp: fixed base, two revolute links, head rigid on the forearm.
    "lamp": Preset(
        name="lamp",
        bodies="""
        <geom name="base" type="box" size="0.08 0.05 0.02" pos="0 0 0.02" rgba="0.3 0.3 0.3 1"/>
        <body name="arm" pos="0 0 0.04">
          <joint name="shoulder" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.012" rgba="0.85 0.1 0.1 1"/>
          <body name="forearm" pos="0 0 0.25">
            <joint name="elbow" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.012" rgba="0.1 0.2 0.9 1"/>
            <geom type="cylinder" fromto="0.2 0 0 0.2 0 -0.05" size="0.035" rgba="0.1 0.2 0.9 1"/>
          </body>
        </body>""",
        joints=[
            Joint("shoulder", "hinge", "arm", amplitude=0.5, period=4.0, offset=0.2),
            Joint("elbow", "hinge", "forearm", amplitude=0.6, period=3.0, phase=1.0, offset=0.3),
        ],
        prompts={"arm": "red rod", "forearm": "blue rod"},
    ),
    # Drawer: a box sliding out of a fixed cabinet along x.
    "drawer": Preset(
        name="drawer",
        bodies="""
        <geom name="cabinet" type="box" size="0.15 0.1 0.1" pos="0 0.02 0.1" rgba="0.3 0.3 0.3 1"/>
        <body name="drawer" pos="0.0 -0.01 0.1">
          <joint name="slide" type="slide" axis="1 0 0"/>
          <geom type="box" size="0.14 0.09 0.06" rgba="0.1 0.6 0.2 1"/>
        </body>""",
        joints=[Joint("slide", "slide", "drawer", amplitude=0.07, period=4.0, offset=0.07, axis=(1.0, 0.0, 0.0))],
        prompts={"drawer": "green box"},
    ),
}


def mjcf(preset: Preset) -> str:
    x, y, z = CAMERA["pos"]
    return f"""<mujoco model="{preset.name}">
  <visual><headlight ambient="0.5 0.5 0.5" diffuse="0.4 0.4 0.4"/><global offwidth="{WIDTH}" offheight="{HEIGHT}"/></visual>
  <asset><texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="32" height="32"/></asset>
  <worldbody>
    <light pos="0 -1 2" dir="0 0.5 -1"/>
    <camera name="front" pos="{x} {y} {z}" xyaxes="1 0 0 0 0 1" fovy="{CAMERA['fovy']}"/>
    {preset.bodies}
  </worldbody>
</mujoco>"""


def project(points: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """World points (N, 3) -> pixel coords (N, 2) for camera "front"."""
    cam = model.camera("front").id
    pos, rot = data.cam_xpos[cam], data.cam_xmat[cam].reshape(3, 3)
    p = (points - pos) @ rot  # camera frame: x right, y up, -z forward
    f = 0.5 * HEIGHT / np.tan(np.deg2rad(model.cam_fovy[cam]) / 2)
    return np.stack([WIDTH / 2 + f * p[:, 0] / -p[:, 2], HEIGHT / 2 - f * p[:, 1] / -p[:, 2]], axis=1)


def render(preset_name: str, out_dir: Path, seconds: float = 4.0) -> Path:
    preset = PRESETS[preset_name]
    model = mujoco.MjModel.from_xml_string(mjcf(preset))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)

    t = np.arange(int(seconds * FPS)) / FPS
    qs = {j.name: j.q(t) for j in preset.joints}
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"{preset.name}.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"avc1"), FPS, (WIDTH, HEIGHT))
    anchors_px = {j.name: [] for j in preset.joints}
    for i in range(len(t)):
        for j in preset.joints:
            data.joint(j.name).qpos[0] = qs[j.name][i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="front")
        writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        for j in preset.joints:
            anchors_px[j.name].append(project(data.joint(j.name).xanchor[None], model, data)[0].tolist())
    writer.release()
    renderer.close()

    gt = {
        "preset": preset.name,
        "fps": FPS,
        "size": [WIDTH, HEIGHT],
        "camera": {**CAMERA, "xyaxes": [1, 0, 0, 0, 0, 1]},
        "prompts": preset.prompts,
        "joints": [{**asdict(j), "q": qs[j.name].tolist(), "anchor_px": anchors_px[j.name]} for j in preset.joints],
    }
    (out_dir / f"{preset.name}.gt.json").write_text(json.dumps(gt))
    (out_dir / f"{preset.name}.xml").write_text(mjcf(preset))
    return video


def gt_masks(gt_json: Path) -> dict[str, np.ndarray]:
    """Per-body silhouettes {body: (T, H, W) bool} for a rendered preset, occlusion included."""
    gt = json.loads(gt_json.read_text())
    model = mujoco.MjModel.from_xml_path(str(gt_json.with_name(f"{gt['preset']}.xml")))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    renderer.enable_segmentation_rendering()
    bodies = list(gt["prompts"])
    body_ids = {b: model.body(b).id for b in bodies}
    n = len(gt["joints"][0]["q"])
    out = {b: np.zeros((n, HEIGHT, WIDTH), bool) for b in bodies}
    for i in range(n):
        for j in gt["joints"]:
            data.joint(j["name"]).qpos[0] = j["q"][i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="front")
        seg = renderer.render()
        geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        geom_body = np.where(geom >= 0, model.geom_bodyid[np.maximum(geom, 0)], -1)
        for b, bid in body_ids.items():
            out[b][i] = geom_body == bid
    renderer.close()
    return out


# --- 3D: any viewpoint, textured parts, depth and camera ground truth -------------------

def textured(bodies: str) -> tuple[str, str]:
    """Swap each rgba for a speckled material of that colour, so parts have trackable texture.
    Returns (bodies, asset xml)."""
    colors = sorted(set(re.findall(r'rgba="([^"]+)"', bodies)))
    assets = []
    for i, c in enumerate(colors):
        r, g, b, _ = (float(v) for v in c.split())
        assets.append(f'<texture name="t{i}" type="2d" builtin="checker" rgb1="{r} {g} {b}" '
                      f'rgb2="{0.6 * r:.3f} {0.6 * g:.3f} {0.6 * b:.3f}" mark="random" random="0.25" '
                      f'markrgb="1 1 1" width="128" height="128"/>'
                      f'<material name="m{i}" texture="t{i}" texrepeat="6 6" texuniform="true"/>')
        bodies = bodies.replace(f'rgba="{c}"', f'material="m{i}"')
    return bodies, "".join(assets)


def look_at(lookat, distance: float, azimuth: float, elevation: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera position and 3x3 rotation (columns: right, up, back) looking at ``lookat``.
    Azimuth 90 deg puts the camera on -y looking along +y (the planar view)."""
    az, el = np.deg2rad(azimuth), np.deg2rad(elevation)
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)])
    pos = np.asarray(lookat) - distance * forward
    x = np.cross(forward, [0, 0, 1])
    x /= np.linalg.norm(x)
    z = -forward
    return pos, np.stack([x, np.cross(z, x), z], axis=1)


def render3d(preset_name: str, out_dir: Path, azimuth: float, elevation: float, distance: float = 0.8,
             lookat=(0.05, 0.0, 0.18), seconds: float = 4.0, shake: float = 0.0, tag: str = "") -> Path:
    """Render ``preset`` from a chosen viewpoint. ``shake`` (m) adds a smooth handheld wobble.

    Writes <name>.mp4, <name>.depth.npz (metres along the view axis, float16) and
    <name>.gt3d.json (intrinsics, camera pose per frame, joint axes and anchors in world).
    """
    preset = PRESETS[preset_name]
    bodies, assets = textured(preset.bodies)
    xml = f"""<mujoco model="{preset.name}">
  <visual><headlight ambient="0.5 0.5 0.5" diffuse="0.4 0.4 0.4"/><global offwidth="{WIDTH}" offheight="{HEIGHT}"/></visual>
  <asset><texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="32" height="32"/>{assets}</asset>
  <worldbody>
    <light pos="0 -1 2" dir="0 0.5 -1"/>
    <camera name="view" pos="0 0 1" fovy="{CAMERA['fovy']}"/>
    {bodies}
  </worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    cam = model.camera("view").id
    rgb = mujoco.Renderer(model, HEIGHT, WIDTH)
    depth = mujoco.Renderer(model, HEIGHT, WIDTH)
    depth.enable_depth_rendering()

    name = f"{preset.name}_{tag or f'az{azimuth:.0f}_el{elevation:.0f}'}"
    t = np.arange(int(seconds * FPS)) / FPS
    qs = {j.name: j.q(t) for j in preset.joints}
    rng = np.random.default_rng(0)
    wobble = shake * np.stack([np.sin(2 * np.pi * (t / p) + ph) for p, ph in zip(rng.uniform(1.5, 3.0, 3),
                                                                                  rng.uniform(0, 6, 3))], axis=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"avc1"), FPS, (WIDTH, HEIGHT))
    depths, cam_pos, cam_rot, joints = [], [], [], {j.name: {"axis": [], "anchor": []} for j in preset.joints}
    for i in range(len(t)):
        pos, rot = look_at(np.asarray(lookat) + wobble[i] * 0.3, distance, azimuth, elevation)
        pos = pos + wobble[i]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.flatten())
        model.cam_pos[cam], model.cam_quat[cam] = pos, quat
        for j in preset.joints:
            data.joint(j.name).qpos[0] = qs[j.name][i]
        mujoco.mj_forward(model, data)
        rgb.update_scene(data, camera="view")
        depth.update_scene(data, camera="view")
        writer.write(cv2.cvtColor(rgb.render(), cv2.COLOR_RGB2BGR))
        depths.append(depth.render().astype(np.float16))
        cam_pos.append(data.cam_xpos[cam].tolist())
        cam_rot.append(data.cam_xmat[cam].reshape(3, 3).tolist())
        for j in preset.joints:
            jid = model.joint(j.name).id
            joints[j.name]["axis"].append(data.xaxis[jid].tolist())
            joints[j.name]["anchor"].append(data.xanchor[jid].tolist())
    writer.release()
    rgb.close()
    depth.close()

    f = 0.5 * HEIGHT / np.tan(np.deg2rad(CAMERA["fovy"]) / 2)
    np.savez_compressed(out_dir / f"{name}.depth.npz", depth=np.stack(depths))
    gt = {
        "preset": preset.name, "fps": FPS, "size": [WIDTH, HEIGHT],
        "intrinsics": {"f": f, "cx": WIDTH / 2, "cy": HEIGHT / 2},
        "camera": {"pos": cam_pos, "rot": cam_rot},  # rot columns: right, up, back (MuJoCo camera)
        "prompts": preset.prompts,
        "joints": [{**asdict(j), "q": qs[j.name].tolist(), "axis_world": joints[j.name]["axis"],
                    "anchor_world": joints[j.name]["anchor"]} for j in preset.joints],
    }
    (out_dir / f"{name}.gt3d.json").write_text(json.dumps(gt))
    return out_dir / f"{name}.mp4"
