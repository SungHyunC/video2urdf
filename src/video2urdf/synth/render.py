"""Render a synthetic video of a known mechanism with per-frame masks and ground truth.

Output folder:
    video.mp4    H.264 video (what the pipeline sees)
    masks.npz    `labels` (T, H, W) uint8: 0 background, i = i-th link, 255 = credit card
    gt.json      camera, plane homography, card corners, model, per-frame joint values and anchors
                 (anchor = joint frame origin at q=0, fixed to the parent: the pivot of a revolute joint,
                 the slide origin of a prismatic one)
    scene.xml    MJCF that reproduces the render
    overlay.png  first frame with GT pivots, prismatic axes and card corners drawn on top
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np

from ..camera import PinholeCamera
from ..mjcf import BASE_DOFS, CARD_SIZE, Card, Scene, scene_xml
from ..model import REVOLUTE, ArticulatedModel, planar_pose, rpy_to_matrix
from .motion import base_drift, joint_trajectories
from .presets import build

CARD_LABEL = 255
FORMAT = "video2urdf-synth/1"
# Plain int: comparing an array against the pybind enum falls back to per-element Python calls (~0.4 s/frame).
OBJ_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)


@dataclass
class SynthConfig:
    preset: str = "lamp"
    seed: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    seconds: float = 5.0
    fovy_deg: float = 45.0
    tilt_deg: float = 0.0  # camera tilt away from facing the motion plane, in a random direction
    motion: str | None = None  # sequential | simultaneous; None = preset default
    base_motion: bool | None = None  # hand-held drift of the root; None = preset default
    noise_std: float = 0.0  # Gaussian pixel noise (0-255 scale)

    @property
    def num_frames(self) -> int:
        return int(round(self.seconds * self.fps))


def _qpos_trajectory(m: mujoco.MjModel, tracks: dict, num_frames: int) -> np.ndarray:
    qpos = np.zeros((num_frames, m.nq))
    for name, values in tracks.items():
        qpos[:, m.jnt_qposadr[m.joint(name).id]] = values
    return qpos


def _link_geom_corners(m: mujoco.MjModel, d: mujoco.MjData, geom_ids: list[int]) -> np.ndarray:
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    out = []
    for g in geom_ids:
        local = signs * m.geom_size[g]
        out.append(d.geom_xpos[g] + local @ d.geom_xmat[g].reshape(3, 3).T)
    return np.concatenate(out)


def _swept_corners(model: ArticulatedModel, tracks: dict, num_frames: int, base_dofs: bool) -> np.ndarray:
    """Every link-box corner over the whole trajectory, using a throwaway scene."""
    placeholder = PinholeCamera.look_at((0, -1, 0), (0, 0, 0), 64, 64, 45.0)
    m = mujoco.MjModel.from_xml_string(scene_xml(model, Scene(placeholder, floor_z=None, base_dofs=base_dofs)))
    d = mujoco.MjData(m)
    link_bodies = {m.body(link.name).id for link in model.links}
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in link_bodies]
    corners = []
    for q in _qpos_trajectory(m, tracks, num_frames):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        corners.append(_link_geom_corners(m, d, geoms))
    return np.concatenate(corners)


def _place_card(corners: np.ndarray, floor_z: float | None) -> Card:
    """Card standing in the motion plane just right of everything the mechanism sweeps."""
    w, h = CARD_SIZE
    x = corners[:, 0].max() + 0.03 + w / 2
    z = (floor_z + h / 2 + 0.003) if floor_z is not None else corners[:, 2].min() + h / 2
    return Card((float(x), float(z)))


def _frame_camera(points: np.ndarray, cfg: SynthConfig, rng: np.random.Generator) -> tuple[PinholeCamera, np.ndarray]:
    """Camera roughly facing the motion plane that keeps `points` inside the image with a margin.

    Returns the camera and the point on the motion plane it looks at.
    """
    lo, hi = points.min(axis=0), points.max(axis=0)
    target = np.array([(lo[0] + hi[0]) / 2, 0.0, (lo[2] + hi[2]) / 2])
    tan_y = np.tan(np.deg2rad(cfg.fovy_deg) / 2)
    tan_x = tan_y * cfg.width / cfg.height
    dist = max((hi[2] - lo[2]) / 2 / tan_y, (hi[0] - lo[0]) / 2 / tan_x) * 1.1
    yaw, pitch = 0.0, 0.0
    if cfg.tilt_deg > 0:
        phi = rng.uniform(0, 2 * np.pi)
        yaw, pitch = np.deg2rad(cfg.tilt_deg) * np.cos(phi), np.deg2rad(cfg.tilt_deg) * np.sin(phi)
    # Direction from the target to the camera: -y rotated by yaw (about z) and pitch (about x).
    direction = np.array([np.sin(yaw) * np.cos(pitch), -np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
    margin = 0.06 * cfg.height
    for _ in range(60):
        cam = PinholeCamera.look_at(target + dist * direction, target, cfg.width, cfg.height, cfg.fovy_deg)
        uv = cam.project(points)
        inside = (
            (cam.to_camera(points)[:, 2] > 0).all()
            and uv[:, 0].min() >= margin
            and uv[:, 0].max() <= cfg.width - 1 - margin
            and uv[:, 1].min() >= margin
            and uv[:, 1].max() <= cfg.height - 1 - margin
        )
        if inside:
            return cam, target
        dist *= 1.05
    raise RuntimeError("could not frame the mechanism")


def _r(a, nd: int = 6):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def _draw_overlay(rgb: np.ndarray, gt: dict) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    corners = np.array(gt["card"]["corners_px"])
    cv2.polylines(img, [np.round(corners).astype(np.int32)], True, (0, 220, 255), 2, cv2.LINE_AA)
    cam = PinholeCamera.from_dict(gt["camera"])
    frames = gt["frames"]
    for j, joint in enumerate(gt["joints"]):
        p = np.array(frames["anchor_px"][0][j])
        if joint["type"] == REVOLUTE:
            cv2.circle(img, tuple(np.round(p).astype(int)), 7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(img, tuple(np.round(p).astype(int)), (0, 0, 0), cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
        else:
            anchor = np.array(frames["anchor_world"][0][j])
            tip = cam.project(anchor + 0.06 * np.array(frames["axis_world"][0][j]))
            cv2.arrowedLine(
                img,
                tuple(np.round(p).astype(int)),
                tuple(np.round(tip).astype(int)),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
                tipLength=0.25,
            )
        cv2.putText(
            img,
            joint["name"],
            tuple(np.round(p + (10, -10)).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return img


def generate(cfg: SynthConfig, out_dir: str | Path) -> dict:
    if cfg.width % 2 or cfg.height % 2:
        raise ValueError("width and height must be even for H.264 yuv420p")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    preset = build(cfg.preset, rng)
    model = preset.model
    mode = cfg.motion or preset.motion
    base_motion = preset.base_motion if cfg.base_motion is None else cfg.base_motion
    T = cfg.num_frames

    tracks = joint_trajectories(model, T, mode, rng)
    if base_motion:
        tracks.update(base_drift(T, rng))

    # Pass 1: sweep the motion to place the card and frame the camera.
    corners = _swept_corners(model, tracks, T, base_motion)
    card = _place_card(corners, preset.floor_z)
    cam, target = _frame_camera(np.concatenate([corners, card.corners_world()]), cfg, rng)
    dist = float(np.linalg.norm(cam.position - target))
    scene = Scene(
        cam,
        card,
        preset.floor_z,
        base_dofs=base_motion,
        light_pos=tuple(target + np.array([-0.5, -0.9, 0.8]) * dist),
        light_target=tuple(target),
    )

    # Pass 2: render the final scene.
    xml = scene_xml(model, scene)
    (out_dir / "scene.xml").write_text(xml, encoding="utf-8")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    qpos = _qpos_trajectory(m, tracks, T)

    link_names = [link.name for link in model.links]
    link_bodies = [m.body(name).id for name in link_names]
    lut = np.zeros(m.ngeom + 1, dtype=np.uint8)
    for g in range(m.ngeom):
        if m.geom(g).name.startswith("card"):
            lut[g + 1] = CARD_LABEL
        elif m.geom_bodyid[g] in link_bodies:
            lut[g + 1] = link_bodies.index(m.geom_bodyid[g]) + 1
    joint_ids = [m.joint(j.name).id for j in model.joints]

    labels = np.zeros((T, cfg.height, cfg.width), dtype=np.uint8)
    anchor_world = np.zeros((T, len(joint_ids), 3))
    axis_world = np.zeros((T, len(joint_ids), 3))
    link_pose = np.zeros((T, len(link_bodies), 3))
    noise_rng = np.random.default_rng([cfg.seed, 1])
    first_rgb = None

    renderer = mujoco.Renderer(m, cfg.height, cfg.width)
    # Segmentation gets its own context without MSAA: averaged ID colors at edges decode to wrong geoms.
    m.vis.quality.offsamples = 0
    seg_renderer = mujoco.Renderer(m, cfg.height, cfg.width)
    seg_renderer.enable_segmentation_rendering()
    # macro_block_size=1: never let imageio resize frames, which would break pixel alignment with the GT.
    writer = imageio.get_writer(
        out_dir / "video.mp4",
        fps=cfg.fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=None,
        macro_block_size=1,
        ffmpeg_params=["-crf", "17", "-preset", "medium", "-movflags", "+faststart"],
    )
    try:
        for t in range(T):
            d.qpos[:] = qpos[t]
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera="cam")
            rgb = renderer.render()
            if cfg.noise_std > 0:
                rgb = np.clip(rgb + noise_rng.normal(0, cfg.noise_std, rgb.shape), 0, 255).astype(np.uint8)
            writer.append_data(rgb)
            if first_rgb is None:
                first_rgb = rgb.copy()

            seg_renderer.update_scene(d, camera="cam")
            seg = seg_renderer.render()
            ids = np.where(seg[..., 1] == OBJ_GEOM, seg[..., 0], -1)
            labels[t] = lut[ids + 1]

            anchor_world[t] = d.xanchor[joint_ids]
            axis_world[t] = d.xaxis[joint_ids]
            for k, b in enumerate(link_bodies):
                pose = np.eye(4)
                pose[:3, :3] = d.xmat[b].reshape(3, 3)
                pose[:3, 3] = d.xpos[b]
                link_pose[t, k] = planar_pose(pose)
    finally:
        writer.close()
        renderer.close()
        seg_renderer.close()

    np.savez_compressed(out_dir / "masks.npz", labels=labels)

    plane_h = cam.plane_homography((0, 0, 0), (1, 0, 0), (0, 0, 1))
    joints_gt = []
    for j in model.joints:
        axis_parent = rpy_to_matrix(j.origin_rpy) @ np.asarray(j.axis, dtype=float)
        joints_gt.append(
            {
                "name": j.name,
                "type": j.type,
                "parent": j.parent,
                "child": j.child,
                "lower": j.lower,
                "upper": j.upper,
                # Pivot (revolute) or slide origin (prismatic) in the parent link's x-z plane, meters.
                "anchor_parent_xz": _r([j.origin_xyz[0], j.origin_xyz[2]]),
                "axis_parent": _r(axis_parent),
            }
        )
    labels_map = {"0": "background", **{str(i + 1): n for i, n in enumerate(link_names)}, str(CARD_LABEL): "card"}
    gt = {
        "format": FORMAT,
        "config": asdict(cfg),
        "motion": {"mode": mode, "base_motion": base_motion},
        "video": {"file": "video.mp4", "fps": cfg.fps, "num_frames": T, "width": cfg.width, "height": cfg.height},
        "masks": {"file": "masks.npz", "key": "labels", "labels": labels_map},
        "camera": {k: (_r(v) if k in ("K", "R", "t", "position") else v) for k, v in cam.to_dict().items()},
        "plane": {
            "origin": [0.0, 0.0, 0.0],
            "u_axis": [1.0, 0.0, 0.0],
            "v_axis": [0.0, 0.0, 1.0],
            "H_plane_to_image": _r(plane_h / plane_h[2, 2], 9),
        },
        "card": {
            "size_m": list(CARD_SIZE),
            "center_xz": _r(card.center_xz),
            "corners_world": _r(card.corners_world()),
            "corners_px": _r(cam.project(card.corners_world()), 3),
        },
        "model": model.to_dict(),
        "joints": joints_gt,
        "frames": {
            "q": _r(np.stack([tracks[j.name] for j in model.joints], axis=1)),
            "anchor_world": _r(anchor_world),
            "anchor_px": _r(cam.project(anchor_world), 3),
            "axis_world": _r(axis_world),
            "link_pose": _r(link_pose),
            **({"base_q": _r(np.stack([tracks[n] for n, _, _ in BASE_DOFS], axis=1))} if base_motion else {}),
        },
    }
    (out_dir / "gt.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
    cv2.imwrite(str(out_dir / "overlay.png"), _draw_overlay(first_rgb, gt))
    return gt
