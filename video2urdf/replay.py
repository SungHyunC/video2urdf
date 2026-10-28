"""Replay a fitted model in MuJoCo over the source video and score it per frame.

The model is rendered from a camera that maps the world x-z plane onto the reference
frame's pixels (a distant camera, so close to orthographic). The render is then moved by
the root part's image pose, which absorbs any camera shake. Score: silhouette IoU of each
link's render against its SAM mask.
"""

from pathlib import Path

import cv2
import mujoco
import numpy as np

from .joints import Link
from .sam import Detection

CAMERA_DISTANCE = 5.0  # m
COLORS = [(0, 200, 255), (255, 80, 80), (80, 220, 80), (255, 0, 255), (255, 255, 0), (0, 128, 255)]


def video_camera(size: tuple[int, int], px_per_m: float, origin_px: tuple[float, float]) -> dict:
    w, h = size
    u0, v0 = origin_px
    f = px_per_m * CAMERA_DISTANCE
    return {"pos": ((w / 2 - u0) / px_per_m, -CAMERA_DISTANCE, (v0 - h / 2) / px_per_m),
            "fovy": float(np.rad2deg(2 * np.arctan(h / 2 / f)))}


def _hold(q: np.ndarray) -> np.ndarray:
    """Fill NaNs with the nearest earlier value (or the first valid one)."""
    q = q.copy()
    valid = np.flatnonzero(~np.isnan(q))
    if len(valid) == 0:
        return np.zeros_like(q)
    q[: valid[0]] = q[valid[0]]
    for i in range(1, len(q)):
        if np.isnan(q[i]):
            q[i] = q[i - 1]
    return q


def replay(model_xml: Path, links: list[Link], root_poses_img: np.ndarray, tracks: dict[str, dict[int, Detection]],
           video: Path, out_video: Path) -> dict[str, np.ndarray]:
    """Write the overlay video and return per-link IoU per frame (NaN where a mask is missing)."""
    model = mujoco.MjModel.from_xml_path(str(model_xml))  # a path, so mesh files resolve
    data = mujoco.MjData(model)
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = len(root_poses_img)
    renderer = mujoco.Renderer(model, h, w)
    renderer.enable_segmentation_rendering()
    body_ids = {l.name: model.body(l.name).id for l in links}
    qs = {l.name: _hold(l.joint.q) for l in links if l.joint and l.joint.type != "fixed"}
    root = _hold_poses(root_poses_img)
    ious = {l.name: np.full(n, np.nan) for l in links}
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"avc1"), cap.get(cv2.CAP_PROP_FPS) or 30, (w, h))

    for t in range(n):
        ok, img = cap.read()
        if not ok:
            break
        for name, q in qs.items():
            data.joint(f"{name}_joint").qpos[0] = q[t]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="video")
        seg = renderer.render()
        geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        body = np.where(geom >= 0, model.geom_bodyid[np.maximum(geom, 0)], -1).astype(np.int16)
        body = cv2.warpAffine(body, root[t][:2], (w, h), flags=cv2.INTER_NEAREST, borderValue=-1)
        for i, l in enumerate(links):
            pred = body == body_ids[l.name]
            det = tracks[l.name].get(t)
            if det is not None:
                gt = det.full_mask(h, w)
                union = (pred | gt).sum()
                ious[l.name][t] = (pred & gt).sum() / union if union else 1.0
            color = np.array(COLORS[i % len(COLORS)])
            img[pred] = (0.45 * img[pred] + 0.55 * color).astype(np.uint8)
            contours, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color.tolist(), 2)
        text = "  ".join(f"{l.name} {ious[l.name][t]:.2f}" for l in links if not np.isnan(ious[l.name][t]))
        cv2.putText(img, f"MuJoCo replay  IoU: {text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(img, f"MuJoCo replay  IoU: {text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(img)
    writer.release()
    renderer.close()
    return ious


def _hold_poses(poses: np.ndarray) -> np.ndarray:
    poses = poses.copy()
    valid = np.flatnonzero(~np.isnan(poses).any(axis=(1, 2)))
    poses[: valid[0]] = poses[valid[0]]
    for i in range(1, len(poses)):
        if np.isnan(poses[i]).any():
            poses[i] = poses[i - 1]
    return poses
