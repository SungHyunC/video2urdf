"""A plain RGB video as an RGB-D capture: metric depth, intrinsics and camera motion from
MapAnything (facebook/map-anything-apache, Meta, Apache-2.0; modal_apps/mapanything_open.py),
written in the Stray Scanner layout (see stray.py) so the LiDAR path (lidar.py, the agent's
fit_lidar / fit_drawers) runs on any phone video unchanged.

MapAnything reconstructs a set of views jointly, and a whole video is too many views for one
pass, so the video goes in chunks of consecutive frames. Every chunk also carries the same few
anchor frames spread over the video, the first anchor as its view 0: each chunk's world is
then that anchor's camera, and a similarity transform fitted on the anchors' own depth points
(the same pixels in every chunk) brings each chunk onto the first one's scale and frame.
The metric scale is the geometric mean of the chunks' own estimates.

Up is unknown without an IMU: it is taken as the normal of the plane, among the reference
frame's largest few, that points most toward the image's top (a floor or table in view).
"""

import csv
import io
import shutil
import subprocess
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import cv2
import numpy as np

from .lift3d import fit_plane

APP = "video2urdf-mapanything"
CLS = "MapAnything"
PATCH = 14
LONG_SIDE = 518  # px, the model's training resolution
CHUNK = 40  # consecutive frames per chunk
ANCHORS = 6  # frames shared by every chunk
CONF_PERCENTILE = 10  # per frame, the least confident pixels are dropped like LiDAR's low confidence


def model_size(w: int, h: int) -> tuple[int, int]:
    """The largest size with the video's exact aspect ratio, sides multiples of 14 and the long
    side at most 518 px (1920x1440 -> 504x378, 1080x1920 -> 252x448), so the depth grid maps
    uniformly onto the frame. Odd aspect ratios round each side instead."""
    g = gcd(w, h)
    a, b = w // g, h // g
    m = LONG_SIDE // (PATCH * max(a, b))
    if m >= 1 and PATCH * max(a, b) * m >= 300:
        return PATCH * a * m, PATCH * b * m
    k = LONG_SIDE / max(w, h)
    return max(PATCH, round(w * k / PATCH) * PATCH), max(PATCH, round(h * k / PATCH) * PATCH)


def read_frames(video: Path, size: tuple[int, int]) -> tuple[list[np.ndarray], tuple[int, int], float]:
    """Every frame as decoded by OpenCV (rotation applied, as the rest of the pipeline sees it),
    resized to ``size``; also the displayed frame size and fps."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, full = [], None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        full = img.shape[1], img.shape[0]
        frames.append(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
    cap.release()
    return frames, full, fps


def plan_chunks(n: int, chunk: int = CHUNK, anchors: int = ANCHORS) -> tuple[list[int], list[list[int]]]:
    """Anchor frames, and each chunk's frames: the anchors first, then its consecutive frames."""
    anc = [int(round(x)) for x in np.linspace(0, n - 1, anchors)]
    rest = [t for t in range(n) if t not in anc]
    return anc, [anc + rest[i:i + chunk] for i in range(0, len(rest), chunk)]


def umeyama(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) with s R a + t ~= b."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    a0, b0 = a - ca, b - cb
    u, sig, vt = np.linalg.svd(b0.T @ a0 / len(a))
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    s = float(np.trace(np.diag(sig) @ d) / (a0 ** 2).sum(axis=1).mean())
    return s, r, cb - s * r @ ca


def robust_umeyama(a: np.ndarray, b: np.ndarray, rounds: int = 4) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama, refitted on the residuals' best 80% a few times (the moving part, edges)."""
    keep = np.ones(len(a), bool)
    for _ in range(rounds):
        s, r, t = umeyama(a[keep], b[keep])
        res = np.linalg.norm(s * a @ r.T + t - b, axis=1)
        keep = res <= np.percentile(res, 80)
    return s, r, t


def world_points(depth: np.ndarray, k: np.ndarray, pose: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vv, uu = np.nonzero(valid)
    z = depth[vv, uu]
    cam = np.stack([(uu + 0.5 - k[0, 2]) / k[0, 0] * z, (vv + 0.5 - k[1, 2]) / k[1, 1] * z, z], axis=1)
    return cam @ pose[:3, :3].T + pose[:3, 3]


@dataclass
class Reconstruction:
    depth: np.ndarray  # (n, H, W) metres along z, 0 where invalid
    poses: np.ndarray  # (n, 4, 4) camera-to-world, OpenCV axes, world y up
    k: np.ndarray  # 3x3 at the video's full size
    size: tuple[int, int]  # full frame size
    fps: float
    chunk_scales: list[float]  # each chunk's scale relative to the first, before averaging


def focal_from_35mm(f35: float, size: tuple[int, int]) -> float:
    """Focal length in pixels from a 35 mm-equivalent focal length. Phone video crops a 4:3
    sensor to 16:9 keeping its long side, so the equivalent refers to the 4:3 diagonal."""
    long = max(size)
    return f35 * long * 1.25 / 43.27  # 4:3 diagonal = 1.25 x long side; 35 mm diagonal = 43.27 mm


def infer_chunks(video: Path, chunk: int = CHUNK, anchors: int = ANCHORS, focal: float | None = None,
                 log=print) -> dict:
    """MapAnything on every chunk (on Modal). ``focal`` is in pixels at the video's size."""
    import modal

    w0, h0 = cv2.VideoCapture(str(video)).read()[1].shape[1::-1]
    size = model_size(w0, h0)
    frames, full, fps = read_frames(video, size)
    n = len(frames)
    anc, chunks = plan_chunks(n, chunk, anchors)
    log(f"{n} frames at {size[0]}x{size[1]} in {len(chunks)} chunks of <= {chunk} + {len(anc)} anchors")
    jpegs = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes() for f in frames]
    remote = modal.Cls.from_name(APP, CLS)()
    model_focal = focal * size[0] / full[0] if focal else None
    results = []
    for i, out in enumerate(remote.infer.map([[jpegs[t] for t in c] for c in chunks],
                                             kwargs={"focal": model_focal})):
        results.append(dict(np.load(io.BytesIO(out))))
        log(f"chunk {i + 1}/{len(chunks)}")
    return {"results": results, "anchors": anc, "chunks": chunks, "size": size, "full": full, "fps": fps}


def reconstruct(video: Path, chunk: int = CHUNK, anchors: int = ANCHORS, focal: float | None = None,
                scale: float | None = None, log=print) -> Reconstruction:
    """Stitched depth and poses for the whole video. ``focal`` (px at the video's size) is a
    known camera focal length, given to the model and used for the capture; ``scale`` multiplies
    the model's metric scale (e.g. known length / measured length)."""
    data = infer_chunks(video, chunk, anchors, focal, log)
    results, anc, chunks = data["results"], data["anchors"], data["chunks"]
    size, full, fps = data["size"], data["full"], data["fps"]
    n = sum(len(c) - len(anc) for c in chunks) + len(anc)

    # Bring every chunk onto chunk 0 with the anchors' points (same pixels in every chunk).
    def valid(r, v):
        d = r["depth"][v].astype(np.float32)
        c = r["conf"][v].astype(np.float32)
        return r["mask"][v] & (d > 0) & (c >= np.percentile(c[d > 0], CONF_PERCENTILE))

    ref = results[0]
    sims, rng = [], np.random.default_rng(0)
    for r in results:
        a_pts, b_pts = [], []
        for v in range(len(anc)):
            ok = valid(r, v) & valid(ref, v)
            pa = world_points(r["depth"][v].astype(np.float32), r["intrinsics"][v], r["poses"][v], ok)
            pb = world_points(ref["depth"][v].astype(np.float32), ref["intrinsics"][v], ref["poses"][v], ok)
            pick = rng.choice(len(pa), min(len(pa), 3000), replace=False)
            a_pts.append(pa[pick])
            b_pts.append(pb[pick])
        sims.append(robust_umeyama(np.concatenate(a_pts), np.concatenate(b_pts)))
    scales = [s for s, _, _ in sims]
    g = float(np.exp(np.mean(-np.log(scales))))  # each chunk's own metric belief, averaged
    log("chunk scales vs chunk 0: " + " ".join(f"{s:.3f}" for s in scales) + f"; metric factor {g:.3f}")
    if scale:
        g *= scale
        log(f"scale x{scale:.4f} (given)")

    h, w = size[1], size[0]
    depth = np.zeros((n, h, w), np.float32)
    poses = np.tile(np.eye(4), (n, 1, 1))
    ks = []
    owner = {}
    for ci, c in enumerate(chunks):
        for v, t in enumerate(c):
            owner.setdefault(t, (ci, v))  # anchors come from chunk 0
    for t in range(n):
        ci, v = owner[t]
        r, (s, rot, tr) = results[ci], sims[ci]
        d = r["depth"][v].astype(np.float32) * s * g
        d[~valid(r, v)] = 0
        depth[t] = d
        p = r["poses"][v]
        poses[t, :3, :3] = rot @ p[:3, :3]
        poses[t, :3, 3] = g * (s * rot @ p[:3, 3] + tr)
        ks.append(r["intrinsics"][v])
    ks = np.array(ks)
    sx, sy = full[0] / w, full[1] / h
    f_model = float(np.median((ks[:, 0, 0] * sx + ks[:, 1, 1] * sy) / 2))
    if focal:  # a known camera: its focal, principal point at the centre
        f = float(focal)
        k = np.array([[f, 0, full[0] / 2], [0, f, full[1] / 2], [0, 0, 1]])
    else:
        f = f_model
        k = np.array([[f, 0, np.median(ks[:, 0, 2]) * sx], [0, f, np.median(ks[:, 1, 2]) * sy], [0, 0, 1]])
    # Depth (z) is kept per frame and lifted with this one shared camera.
    log(f"focal {f:.0f} px at {full[0]}x{full[1]} (model's own estimate {f_model:.0f}, "
        f"spread {np.std(ks[:, 0, 0] * sx) / f_model * 100:.1f}%)")

    up = estimate_up(depth[0], k, size, full, rng)
    poses = level_world(poses, up)
    return Reconstruction(depth, poses, k, full, fps, scales)


def estimate_up(depth: np.ndarray, k: np.ndarray, size, full, rng) -> np.ndarray:
    """Up in frame 0's camera coordinates: of the few largest planes, the normal most aligned
    with the image's up (-y)."""
    h, w = depth.shape
    vv, uu = np.nonzero(depth > 0)
    z = depth[vv, uu]
    u = (uu + 0.5) * full[0] / w
    v = (vv + 0.5) * full[1] / h
    pts = np.stack([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z], axis=1)
    pts = pts[rng.choice(len(pts), min(len(pts), 20000), replace=False)]
    best, rest = None, pts
    for _ in range(3):
        if len(rest) < 500:
            break
        nrm, d, _ = fit_plane(rest, rng, tol=0.01)
        on = np.abs(rest @ nrm - d) < 0.01
        if nrm[1] > 0:
            nrm = -nrm
        if on.sum() > 0.05 * len(pts) and (best is None or -nrm[1] > -best[1]):
            best = nrm
        rest = rest[~on]
    return best if best is not None else np.array([0.0, -1.0, 0.0])


def level_world(poses: np.ndarray, up_cam0: np.ndarray) -> np.ndarray:
    """Re-express camera-to-world poses in a world whose +y is ``up`` (given in frame 0's camera
    coordinates), as ARKit's world is (stray.gravity_up reads it)."""
    up = poses[0, :3, :3] @ up_cam0
    up /= np.linalg.norm(up)
    x = np.array([1.0, 0, 0]) - up[0] * up
    x /= np.linalg.norm(x)
    r = np.stack([x, up, np.cross(x, up)])
    out = poses.copy()
    out[:, :3, :3] = r @ poses[:, :3, :3]
    out[:, :3, 3] = poses[:, :3, 3] @ r.T
    return out


def write_capture(rec: Reconstruction, video: Path, out: Path) -> Path:
    """The Stray Scanner layout: rgb.mp4, depth/ and confidence/ PNGs, camera_matrix.csv,
    odometry.csv. Returns the folder."""
    from scipy.spatial.transform import Rotation

    out.mkdir(parents=True, exist_ok=True)
    for sub in ("depth", "confidence"):
        shutil.rmtree(out / sub, ignore_errors=True)
        (out / sub).mkdir()
    (out / "depth.npz").unlink(missing_ok=True)  # stray.load's cache
    for t, d in enumerate(rec.depth):
        cv2.imwrite(str(out / "depth" / f"{t:06d}.png"), np.clip(d * 1000, 0, 65535).astype(np.uint16))
        cv2.imwrite(str(out / "confidence" / f"{t:06d}.png"), np.where(d > 0, 2, 0).astype(np.uint8))
    np.savetxt(out / "camera_matrix.csv", rec.k, delimiter=", ", fmt="%.4f")
    q = Rotation.from_matrix(rec.poses[:, :3, :3]).as_quat()
    with open(out / "odometry.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["timestamp", " frame", " x", " y", " z", " qx", " qy", " qz", " qw", " fx", " fy", " cx", " cy"])
        for t in range(len(rec.poses)):
            wr.writerow([f"{t / rec.fps:.6f}", f"{t:06d}", *(f"{x:.6f}" for x in rec.poses[t, :3, 3]),
                         *(f"{x:.8f}" for x in q[t]), f"{rec.k[0, 0]:.4f}", f"{rec.k[1, 1]:.4f}",
                         f"{rec.k[0, 2]:.4f}", f"{rec.k[1, 2]:.4f}"])
    rgb = out / "rgb.mp4"
    if video.suffix.lower() == ".mp4" and video.resolve() != rgb.resolve():
        shutil.copy(video, rgb)
    elif video.resolve() != rgb.resolve():  # H.264 mp4, frames 1:1 with the decode above, rotation applied
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-an", "-fps_mode", "passthrough",
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(rgb)], check=True)
    return out


def rescale_capture(out: Path, factor: float) -> None:
    """Multiply a written capture's metric scale (depth and camera translations) by ``factor``:
    a known length over the length the model measured."""
    for f in sorted((out / "depth").glob("*.png")):
        d = cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float64) * factor
        cv2.imwrite(str(f), np.clip(np.round(d), 0, 65535).astype(np.uint16))
    (out / "depth.npz").unlink(missing_ok=True)
    rows = list(csv.reader(open(out / "odometry.csv")))
    for r in rows[1:]:
        r[2:5] = [f"{float(x) * factor:.6f}" for x in r[2:5]]
    with open(out / "odometry.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
