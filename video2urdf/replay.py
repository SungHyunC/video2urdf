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


def replay3d(model_xml: Path, cam_to_world: np.ndarray, intr, joint_q: dict[str, np.ndarray],
             masks: dict[str, dict[int, Detection]], video: Path, out_video: Path,
             only_when: dict | None = None) -> dict[str, np.ndarray]:
    """Render a 3D model from the phone's own camera pose on every frame, over the video.

    ``cam_to_world`` (n, 4, 4) maps camera coordinates (x right, y down, z forward) to the
    model's world; ``joint_q`` gives each joint's value per frame. Returns, per body in
    ``masks``, the fraction of its SAM mask the rendered body covers (NaN where not scored;
    ``only_when[body](t)`` can skip frames).
    """
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    cam = model.camera("video").id
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model.cam_fovy[cam] = np.rad2deg(2 * np.arctan(h / 2 / intr.f))
    renderer = mujoco.Renderer(model, h, w)
    renderer.enable_segmentation_rendering()
    # Frames without a measurement (the lid edge-on, say) get the value interpolated between
    # their measured neighbours, and the label says so.
    gaps = np.zeros(len(cam_to_world), bool)
    qs = {}
    for j, q in joint_q.items():
        known = np.flatnonzero(~np.isnan(q))
        gaps[: len(q)] |= np.isnan(q)
        qs[j] = np.interp(np.arange(len(q)), known, q[known]) if len(known) else np.zeros_like(q)
    body_ids = {b: model.body(b).id for b in masks}
    n = len(cam_to_world)
    np.savez(out_video.with_suffix(".npz"), cam_to_world=cam_to_world, f=intr.f, cx=intr.cx, cy=intr.cy,
             **{f"q_{j}": q for j, q in qs.items()})
    cover = {b: np.full(n, np.nan) for b in masks}
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"avc1"), cap.get(cv2.CAP_PROP_FPS) or 30, (w, h))
    flip = np.diag([1.0, -1.0, -1.0])  # camera axes: OpenCV -> MuJoCo (y up, looking along -z)
    plot = _plot(model, joint_q, n, w, h)
    for t in range(n):
        ok, img = cap.read()
        if not ok:
            break
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (cam_to_world[t][:3, :3] @ flip).flatten())
        model.cam_pos[cam], model.cam_quat[cam] = cam_to_world[t][:3, 3], quat
        for j, q in qs.items():
            data.joint(j).qpos[0] = q[t]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="video")
        seg = renderer.render()
        geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        body = np.where(geom >= 0, model.geom_bodyid[np.maximum(geom, 0)], -1)
        for i, (b, bid) in enumerate(body_ids.items()):
            pred = body == bid
            det = masks[b].get(t)
            if det is not None and (not only_when or b not in only_when or only_when[b](t)):
                m = det.full_mask(h, w)
                if m.any():
                    cover[b][t] = (pred & m).sum() / m.sum()
            color = np.array(COLORS[i % len(COLORS)])
            img[pred] = (0.5 * img[pred] + 0.5 * color).astype(np.uint8)
            contours, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color.tolist(), 3)
        img = _inset(img, plot, t, n, [f"{b} covers {100 * cover[b][t]:.0f}% of its mask" for b in body_ids
                                       if not np.isnan(cover[b][t])])
        label = "MuJoCo replay of the depth-measured model  " + "  ".join(
            f"{j} {np.rad2deg(q[t]):.0f} deg" if model.jnt_type[model.joint(j).id] == mujoco.mjtJoint.mjJNT_HINGE
            else f"{j} {1000 * q[t]:.0f} mm" for j, q in qs.items()) + ("  (interpolated)" if gaps[t] else "")
        k = w / 1920  # text scales with the frame
        cv2.putText(img, label, (int(30 * k), int(60 * k)), cv2.FONT_HERSHEY_SIMPLEX, 1.4 * k, (0, 0, 0), max(1, int(6 * k)))
        cv2.putText(img, label, (int(30 * k), int(60 * k)), cv2.FONT_HERSHEY_SIMPLEX, 1.4 * k, (255, 255, 255), max(1, int(2 * k)))
        writer.write(img)
    writer.release()
    renderer.close()
    return cover


def _plot(model, joint_q: dict[str, np.ndarray], n: int, w: int, h: int) -> dict:
    """The measured joint values over the whole video, drawn once (deg or mm)."""
    pw, ph = int(0.36 * w), int(0.2 * h)
    pad = int(0.012 * w)
    img = np.full((ph, pw, 3), 255, np.uint8)
    curves = []
    for i, (j, q) in enumerate(joint_q.items()):
        hinge = model.jnt_type[model.joint(j).id] == mujoco.mjtJoint.mjJNT_HINGE
        v = np.rad2deg(q) if hinge else 1000 * q
        curves.append((j, v, "deg" if hinge else "mm", COLORS[(i + 1) % len(COLORS)]))
    allv = np.concatenate([v[~np.isnan(v)] for _, v, _, _ in curves]) if curves else np.zeros(1)
    lo, hi = float(allv.min()), float(allv.max())
    hi = hi if hi > lo else lo + 1
    xs = lambda t: pad + (pw - 2 * pad) * t / max(1, n - 1)
    ys = lambda v: ph - pad - (ph - 3 * pad) * (v - lo) / (hi - lo)
    for _, v, _, color in curves:
        for t in np.flatnonzero(~np.isnan(v)):
            cv2.circle(img, (int(xs(t)), int(ys(v[t]))), max(1, w // 960), color, -1)
    k = w / 1920
    cv2.putText(img, f"measured from the video: {lo:.0f}..{hi:.0f} {curves[0][2] if curves else ''}",
                (pad, int(2.2 * pad)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k, (60, 60, 60), max(1, int(2 * k)))
    return {"img": img, "xs": xs, "ys": ys, "curves": curves}


def _inset(img: np.ndarray, plot: dict, t: int, n: int, lines: list[str]) -> np.ndarray:
    """Paste the joint plot bottom-left with a cursor at frame t, and the per-frame coverage."""
    h, w = img.shape[:2]
    p = plot["img"].copy()
    ph, pw = p.shape[:2]
    k = w / 1920
    x = int(plot["xs"](t))
    cv2.line(p, (x, 0), (x, ph), (40, 40, 40), max(1, int(2 * k)))
    for _, v, _, color in plot["curves"]:
        if t < len(v) and not np.isnan(v[t]):
            cv2.circle(p, (x, int(plot["ys"](v[t]))), int(9 * k), color, -1)
            cv2.circle(p, (x, int(plot["ys"](v[t]))), int(9 * k), (40, 40, 40), max(1, int(2 * k)))
    y0, x0 = h - ph - int(30 * k), int(30 * k)
    roi = img[y0: y0 + ph, x0: x0 + pw]
    img[y0: y0 + ph, x0: x0 + pw] = (0.15 * roi + 0.85 * p).astype(np.uint8)
    for i, text in enumerate(lines):
        y = y0 - int((24 + 46 * (len(lines) - 1 - i)) * k)
        cv2.putText(img, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * k, (0, 0, 0), max(1, int(6 * k)))
        cv2.putText(img, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * k, (255, 255, 255), max(1, int(2 * k)))
    return img
