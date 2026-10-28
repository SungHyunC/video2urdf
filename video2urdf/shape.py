"""Link geometry from video: each part's side profile, extruded into a mesh, in its own colour.

The profile is a vote over the whole video, not one frame: every frame's mask is moved back
into the reference frame by the part's pose, and a pixel stays if the part covers it in most
frames. Things seen only some of the time drop out: a hand in front, or the face of a lid
that the camera sees at some angles (which made the old one-box lid 4.5 cm thick).
"""

import struct
from pathlib import Path

import cv2
import mapbox_earcut
import numpy as np

from .sam import Detection

VOTE = 0.7  # fraction of visible frames a pixel must be covered in
WORK_SIZE = 640  # long side of the vote raster
SIMPLIFY_M = 0.0015  # polygon tolerance


def part_profile(frames: dict[int, Detection], poses_img: np.ndarray, size: tuple[int, int], ref: int,
                 vote: float = VOTE) -> np.ndarray:
    """Full-size bool mask of the part's profile, in reference-frame pixels."""
    w, h = size
    k = WORK_SIZE / max(w, h)
    small = (round(w * k), round(h * k))
    s, s_inv = np.diag([k, k, 1.0]), np.diag([1 / k, 1 / k, 1.0])
    acc = np.zeros(small[::-1], np.float32)
    n = 0
    for t, det in frames.items():
        pose = poses_img[t]
        if np.isnan(pose).any():
            continue
        m = cv2.resize(det.full_mask(h, w).astype(np.float32), small, interpolation=cv2.INTER_AREA)
        a = (s @ pose @ s_inv)[:2]  # reference -> frame t, in work pixels
        acc += cv2.warpAffine(m, a, small, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        n += 1
    profile = acc >= vote * max(n, 1)
    if profile.sum() < 0.3 * (frames[ref].full_mask(h, w).sum() * k * k):
        # Poses too poor to agree (or the part barely visible): keep the reference mask.
        return frames[ref].full_mask(h, w)
    profile = cv2.morphologyEx(profile.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.resize(profile, (w, h), interpolation=cv2.INTER_NEAREST) > 0


def profile_polygon(mask: np.ndarray, image_to_world: np.ndarray, origin: np.ndarray, px_per_m: float) -> np.ndarray:
    """Outer outline of the largest region as an (N, 2) counter-clockwise (x, z) polygon,
    in metres relative to ``origin`` (the link frame)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, SIMPLIFY_M * px_per_m, True)[:, 0, :].astype(np.float64)
    xz = c @ image_to_world[:2, :2].T + image_to_world[:2, 2] - origin
    area = 0.5 * np.sum(xz[:, 0] * np.roll(xz[:, 1], -1) - np.roll(xz[:, 0], -1) * xz[:, 1])
    return xz if area > 0 else xz[::-1]


def extrude(poly: np.ndarray, depth: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed, outward-facing triangle mesh of ``poly`` (x, z) extruded along y by ``depth``."""
    n = len(poly)
    lo = np.column_stack([poly[:, 0], np.full(n, -depth / 2), poly[:, 1]])
    hi = lo.copy()
    hi[:, 1] = depth / 2
    verts = np.vstack([lo, hi])
    # A triangle counter-clockwise in (x, z) has its normal along -y: right for the y = -d/2 cap.
    cap = mapbox_earcut.triangulate_float64(poly, np.array([n], np.uint32)).reshape(-1, 3)
    p = poly[cap]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    ccw = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] > 0
    cap[~ccw] = cap[~ccw, ::-1]
    i = np.arange(n)
    j = (i + 1) % n
    # Sides of a counter-clockwise outline face out with this winding.
    faces = np.vstack([cap, cap[:, ::-1] + n, np.column_stack([i, j + n, j]), np.column_stack([i, i + n, j + n])])
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    assert np.einsum("ij,ij->i", a, np.cross(b, c)).sum() > 0, "mesh faces point inward"
    return verts, faces


def write_stl(path: Path, verts: np.ndarray, faces: np.ndarray):
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    with open(path, "wb") as f:
        f.write(b"video2urdf".ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        for nrm, t in zip(normals, tri):
            f.write(struct.pack("<12fH", *nrm, *t.ravel(), 0))


def part_color(image_bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """Median colour of the part on the reference frame, as RGB in 0..1."""
    b, g, r = np.median(image_bgr[mask], axis=0)
    return (r / 255, g / 255, b / 255)


def squash(poly: np.ndarray, pivot: np.ndarray, thickness: float) -> np.ndarray:
    """Scale ``poly`` along its thin axis to ``thickness``, about ``pivot`` (its joint).

    In an oblique view a part's silhouette includes faces seen at an angle (a keyboard deck,
    a screen), so its thin dimension comes out too thick. Length, orientation and the joint
    are measured well; this rescales only the thickness, keeping the joint where it is.
    """
    c = poly - poly.mean(axis=0)
    axes = np.linalg.svd(c, full_matrices=False)[2]
    thin = axes[1]
    s = (poly - pivot) @ thin
    current = np.ptp(s)
    if current <= thickness:
        return poly
    return poly + np.outer(s * (thickness / current - 1), thin)


def prism(poly: np.ndarray, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray, e3: np.ndarray,
          w0: float, w1: float) -> tuple[np.ndarray, np.ndarray]:
    """``poly`` (u, v) laid out at origin + u e1 + v e2 and extruded along e3 from w0 to w1.
    Polygon orientation is handled; faces point out."""
    xz = poly if np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]) > 0 else poly[::-1]
    verts, faces = extrude(xz, abs(w1 - w0))
    basis = np.column_stack([e1, e3, e2])  # extrude's (x, y, z) -> (e1, e3, e2)
    out = origin + verts @ basis.T + ((w0 + w1) / 2) * e3
    if np.linalg.det(basis) < 0:
        faces = faces[:, ::-1]
    return out, faces
