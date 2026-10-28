"""The RBO dataset (TU Berlin, CC BY 4.0) as a benchmark: RGB-D video with motion-capture truth.

R. Martín-Martín, C. Eppner, O. Brock, "The RBO Dataset of Articulated Objects and
Interactions", IJRR 2019. https://zenodo.org/records/1036660

A sequence folder (<name>_o/) holds camera_rgb/*.png (640x480), camera_depth_registered/*.txt
(depth in metres, registered to the RGB camera, "nan" where missing), camera info CSVs,
<object>_joint_states.csv (joint values from motion capture), rb_poses_array.csv (poses of
the marker bodies in /map) and tf.csv (where the camera sits in /map).

prepare() turns one into what the pipeline reads: an mp4, depth.npz and intrinsics; truth()
gives, per RGB frame, the ground-truth joint values and the marker bodies' poses in the RGB
camera's optical frame (x right, y down, z forward), the frame our 3D fits use.
"""

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

FPS = 30  # the Asus Xtion's nominal rate; frames are written in their recorded order


def _rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def _stamp(name: str) -> float:
    return float(re.match(r"\d+-([\d.]+)\.", name).group(1))


def _pose(t, q) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(q).as_matrix()
    m[:3, 3] = t
    return m


def sequence_dir(root: Path) -> Path:
    """The <name>_o folder inside an extracted sequence."""
    return next(p for p in [root, *root.iterdir()] if (p / "camera_rgb").is_dir())


def rgb_frames(seq: Path) -> tuple[list[Path], np.ndarray]:
    files = sorted((seq / "camera_rgb").glob("*.png"), key=lambda p: int(p.name.split("-")[0]))
    return files, np.array([_stamp(p.name) for p in files])


def intrinsics(seq: Path) -> dict:
    h, rows = _rows(seq / "camera_rgb_camera_info.csv")
    r = rows[0]
    k = [float(r[h.index(f"field.K{i}")]) for i in range(9)]
    return {"f": (k[0] + k[4]) / 2, "cx": k[2], "cy": k[5], "fx": k[0], "fy": k[4],
            "width": int(r[h.index("field.width")]), "height": int(r[h.index("field.height")])}


def prepare(root: Path, out: Path) -> Path:
    """Write <out>/<name>.mp4, depth.npz (metres, 0 where missing) and intrinsics.json."""
    seq = sequence_dir(root)
    name = seq.name.removesuffix("_o")
    out.mkdir(parents=True, exist_ok=True)
    video = out / f"{name}.mp4"
    files, times = rgb_frames(seq)
    depth_files = sorted((seq / "camera_depth_registered").glob("*.txt"))
    depth_times = np.array([_stamp(p.name) for p in depth_files])
    if not video.exists():
        first = cv2.imread(str(files[0]))
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"avc1"), FPS, first.shape[1::-1])
        for f in files:
            writer.write(cv2.imread(str(f)))
        writer.release()
    if not (out / "depth.npz").exists():
        depth = []
        for t in times:  # the depth frame nearest in time to each RGB frame
            f = depth_files[int(np.argmin(np.abs(depth_times - t)))]
            d = np.fromstring(f.read_text(), sep=" ", dtype=np.float32).reshape(480, 640)
            depth.append(np.nan_to_num(d, nan=0.0).astype(np.float16))
        np.savez_compressed(out / "depth.npz", depth=np.stack(depth))
    (out / "intrinsics.json").write_text(json.dumps(intrinsics(seq)))
    return video


def _tf_bodies(seq: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Motion-captured rigid bodies in /map from tf.csv: {name: (times, (n, 4, 4) poses)}.
    These are proper body frames; rb_poses_array.csv's orientations don't compose into
    consistent relative motion, its positions do."""
    h, rows = _rows(seq / "tf.csv")
    out: dict[str, list] = {}
    for r in rows:
        for i in range(8):
            try:
                fi = h.index(f"field.transforms{i}.header.frame_id")
            except ValueError:
                break
            if fi >= len(r) or r[fi].lstrip("/") != "map":
                continue
            p = f"field.transforms{i}.transform."
            pose = _pose([float(r[h.index(p + f"translation.{a}")]) for a in "xyz"],
                         [float(r[h.index(p + f"rotation.{a}")]) for a in "xyzw"])
            stamp = float(r[h.index(f"field.transforms{i}.header.stamp")]) * 1e-9
            out.setdefault(r[h.index(f"field.transforms{i}.child_frame_id")], []).append((stamp, pose))
    return {k: (np.array([t for t, _ in v]), np.array([m for _, m in v])) for k, v in out.items()}


def _camera_in_map(seq: Path) -> tuple[np.ndarray, np.ndarray]:
    """Times and (n, 4, 4) poses of the RGB optical frame in /map: the motion-captured camera
    body, then the camera's fixed internal chain down to the optical frame."""
    h, rows = _rows(seq / "tf.csv")
    static, body_t, body = {}, [], []
    for r in rows:
        for i in range(8):
            try:
                fi = h.index(f"field.transforms{i}.header.frame_id")
            except ValueError:
                break
            if fi >= len(r) or not r[fi]:
                continue
            p = f"field.transforms{i}.transform."
            t = [float(r[h.index(p + f"translation.{a}")]) for a in "xyz"]
            q = [float(r[h.index(p + f"rotation.{a}")]) for a in "xyzw"]
            parent, child = r[fi].lstrip("/"), r[h.index(f"field.transforms{i}.child_frame_id")].lstrip("/")
            if parent == "map" and child == "AsusXtionCameraFrame":
                body_t.append(float(r[h.index(f"field.transforms{i}.header.stamp")]) * 1e-9)
                body.append(_pose(t, q))
            elif parent != "map":
                static[child] = (parent, _pose(t, q))
    chain, frame = np.eye(4), "camera_rgb_optical_frame"
    while frame != "AsusXtionCameraFrame":
        parent, m = static[frame]
        chain, frame = m @ chain, parent
    return np.array(body_t), np.array(body) @ chain


def _interp_poses(times: np.ndarray, at: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Nearest-in-time poses (mocap runs at ~100 Hz, so nearest is within 5 ms)."""
    idx = np.clip(np.searchsorted(times, at), 1, len(times) - 1)
    idx -= (at - times[idx - 1]) < (times[idx] - at)
    return poses[idx]


def truth(root: Path) -> dict:
    """Per RGB frame, in the RGB optical frame: joint values {joint: (n,)}, marker-body poses
    {id: (n, 4, 4)} from rb_poses_array (positions reliable) and rigid-body frames
    {name: (n, 4, 4)} from tf (full poses). The camera's own body is left out."""
    seq = sequence_dir(root)
    _, times = rgb_frames(seq)
    cam_t, cam = _camera_in_map(seq)
    cam = _interp_poses(cam_t, times, cam)

    joints = {}
    js = next(seq.glob("*_joint_states.csv"))
    h, rows = _rows(js)
    t = np.array([float(r[h.index("field.header.stamp")]) * 1e-9 for r in rows])
    for k in range(8):
        try:
            ni, pi = h.index(f"field.name{k}"), h.index(f"field.position{k}")
        except ValueError:
            break
        joints[rows[0][ni]] = np.interp(times, t, [float(r[pi]) for r in rows])

    h, rows = _rows(seq / "rb_poses_array.csv")
    t = np.array([float(r[h.index("field.markers0.header.stamp")]) * 1e-9 for r in rows])
    bodies = {}
    for m in range(16):
        p = f"field.markers{m}."
        if p + "id" not in h:
            break
        cols = [h.index(p + f"pose.position.{a}") for a in "xyz"] + [h.index(p + f"pose.orientation.{a}") for a in "xyzw"]
        full = [i for i, r in enumerate(rows) if len(r) > max(cols)]  # rows can omit a marker body
        if len(full) < len(rows) // 2:
            continue
        vals = np.array([[float(rows[i][c]) for c in cols] for i in full])
        in_map = np.array([_pose(v[:3], v[3:]) for v in vals])
        in_cam = np.linalg.inv(cam) @ _interp_poses(t[full], times, in_map)
        if in_cam[:, 2, 3].max() < 0.2:
            continue  # the camera's own marker body, at (not in front of) the camera
        bodies[rows[0][h.index(p + "id")]] = in_cam
    frames = {name: np.linalg.inv(cam) @ _interp_poses(t, times, poses)
              for name, (t, poses) in _tf_bodies(seq).items() if name != "AsusXtionCameraFrame"}
    return {"times": times, "joints": joints, "bodies": bodies, "frames": frames}
