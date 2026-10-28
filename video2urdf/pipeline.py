"""Tracks -> part poses -> joints -> URDF/MJCF -> MuJoCo replay, as one call.

Used by scripts/fit_model.py and by the agent. A track folder holds tracks.sam.txt and
parts.json ({obj_id: {prompt, name?, frame, box}}); the first part is the root.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import emit
from .joints import Joint, Link, build_tree, fit_joint, image_to_world, mask_box, to_world
from .motion import MIN_MASK_FIT, fuse_poses, interpolate_poses, load_gray, part_poses, surface_poses
from .replay import replay, video_camera
from .sam import Detection, load_tracks

CARD_LONG_SIDE = 0.0856  # m, ID-1 card


def slug(text: str) -> str:
    return re.sub(r"\W+", "_", text.strip().lower()).strip("_")


def video_info(video: Path) -> tuple[tuple[int, int], int, float]:
    """Displayed (width, height), frame count and fps. Frames are counted by decoding: the
    header count is off by one on variable-frame-rate phone video."""
    cap = cv2.VideoCapture(str(video))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return size, n, fps


def load_parts(track_dir: Path) -> tuple[list[str], dict[str, dict[int, Detection]]]:
    """Part names in root-first order, and their tracks."""
    parts = {int(k): v for k, v in json.loads((track_dir / "parts.json").read_text()).items()}
    objs = load_tracks(track_dir / "tracks.sam.txt")
    names = [slug(parts[oid].get("name") or parts[oid]["prompt"]) for oid in sorted(parts)]
    return names, {nm: objs.get(oid, {}) for nm, oid in zip(names, sorted(parts))}


def long_side_px(mask: np.ndarray) -> float:
    pts = np.argwhere(mask)[:, ::-1].astype(np.float32)
    return float(max(cv2.minAreaRect(pts)[1]))


@dataclass
class Fit:
    links: list[Link]
    names: list[str]
    ref: int
    px_per_m: float
    origin: tuple[float, float]
    ious: dict[str, np.ndarray]
    pose_sources: dict[str, tuple[int, int]]  # frames from mask, from surface points

    def summary(self) -> str:
        lines = [f"reference frame {self.ref}, scale {self.px_per_m:.0f} px/m"]
        for l in self.links:
            iou = self.ious[l.name]
            fit = f"replay IoU mean {np.nanmean(iou):.2f}, worst {np.nanmin(iou):.2f} (frame {int(np.nanargmin(iou))})"
            if not l.joint:
                lines.append(f"- {l.name}: root, {fit}")
                continue
            j = l.joint
            if j.type == "fixed":
                lines.append(f"- {l.parent} -> {l.name}: fixed, {fit}")
                continue
            lo, hi = np.nanmin(j.q), np.nanmax(j.q)
            if j.type == "revolute":
                what = (f"revolute at ({1000 * j.anchor[0]:.0f}, {1000 * j.anchor[1]:.0f}) mm, "
                        f"range {np.rad2deg(lo):.1f}..{np.rad2deg(hi):.1f} deg")
            else:
                what = f"prismatic along ({j.axis[0]:.2f}, {j.axis[1]:.2f}), range {1000 * lo:.0f}..{1000 * hi:.0f} mm"
            lines.append(f"- {l.parent} -> {l.name}: {what}, fit residual {1000 * j.residual:.1f} mm, {fit}")
        return "\n".join(lines)


def fit(video: Path, track_dir: Path, px_per_m: float, parents: dict[str, str] | None = None,
        fixed: set[str] = frozenset(), log=print) -> Fit:
    """Fit and write model.urdf, model.xml, joints.json and replay.mp4 into ``track_dir``.

    ``parents`` ({child: parent}) overrides the automatic tree; ``fixed`` names children
    whose joint is rigid.
    """
    names, by_name = load_parts(track_dir)
    size, n, _ = video_info(video)
    w, h = size
    ref = next(t for t in range(n) if all(t in by_name[nm] for nm in names))
    ref_masks = {nm: by_name[nm][ref].full_mask(h, w) for nm in names}
    centroids_px = {nm: np.argwhere(m).mean(axis=0)[::-1] for nm, m in ref_masks.items()}
    origin = tuple(centroids_px[names[0]])

    grays, k = load_gray(video)
    poses_img, sources = {}, {}
    for nm in names:
        others = {}
        for other in names:
            if other != nm:
                for t, det in by_name[other].items():
                    others.setdefault(t, []).append(det)
        surf = surface_poses(grays, k, by_name[nm], others, ref, size)
        mask, mask_fit = part_poses(by_name[nm], size, n, ref)
        fused = fuse_poses(mask, mask_fit, surf)
        n_mask = int((mask_fit >= MIN_MASK_FIT).sum())
        sources[nm] = (n_mask, int((~np.isnan(fused).any(axis=(1, 2))).sum()) - n_mask)
        if nm == names[0]:
            fused = interpolate_poses(fused)  # the root mostly holds still: bridge the gaps
        else:  # textureless and occluded: a bent mask pose beats none
            gap = np.isnan(fused).any(axis=(1, 2))
            fused[gap] = mask[gap]
        poses_img[nm] = fused
        log(f"  {nm}: {sources[nm][0]} frames from its mask, {sources[nm][1]} from surface points")

    m = image_to_world(px_per_m, origin)
    poses_w = {nm: to_world(p, m) for nm, p in poses_img.items()}
    centroids_w = {nm: (m @ np.append(c, 1))[:2] for nm, c in centroids_px.items()}
    boxes = {nm: mask_box(ref_masks[nm], m) for nm in names}
    if parents:
        links = [Link(names[0], None, None, boxes[names[0]])]
        for nm in names[1:]:
            p = parents[nm]
            j = fit_joint(poses_w[p], poses_w[nm], centroids_w[nm])
            links.append(Link(nm, p, j, boxes[nm]))
    else:
        links = build_tree(names, poses_w, centroids_w, boxes)
    for l in links:
        if l.name in fixed and l.joint:
            l.joint = Joint("fixed", centroids_w[l.name], np.zeros(2), np.zeros(n), l.joint.residual)

    camera = video_camera(size, px_per_m, origin)
    (track_dir / "model.urdf").write_text(emit.urdf(video.stem, links))
    xml = emit.mjcf(video.stem, links, camera, size)
    (track_dir / "model.xml").write_text(xml)
    (track_dir / "joints.json").write_text(json.dumps({
        "reference_frame": ref, "px_per_m": px_per_m,
        "joints": [{"parent": l.parent, "child": l.name, "type": l.joint.type,
                    "anchor_m": l.joint.anchor.tolist(), "axis": l.joint.axis.tolist(),
                    "residual_m": l.joint.residual, "q": [None if np.isnan(v) else v for v in l.joint.q]}
                   for l in links if l.joint]}))
    ious = replay(xml, links, poses_img[names[0]], by_name, video, track_dir / "replay.mp4")
    return Fit(links, names, ref, px_per_m, origin, ious, sources)
