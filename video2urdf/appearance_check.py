"""Compare generated visuals against the tracked video before activating them."""
import json
import tempfile
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .pipeline import load_parts, video_info
from .texture import link_masks


def observed_mask(tracks, candidates, t: int, w: int, h: int, combine: bool = False):
    """The visible tracked pixels, combining simultaneous rigid pieces when requested."""
    full = np.zeros((h, w), np.uint8)
    for part in candidates if combine else candidates[:1]:
        d = tracks[part].get(t)
        if d is not None and d.usable(w, h):
            full |= d.full_mask(h, w).astype(np.uint8)
    return full


def open_frame_available(tracks, indicator: str | None, t: int, w: int, h: int) -> bool:
    if not indicator:
        return True
    detection = tracks.get(indicator, {}).get(t)
    return detection is not None and detection.usable(w, h)


def rejected_links(comparison: dict, *, minimum_frames: int = 5, max_drop: float = .05) -> list[str]:
    """Reject only against recorded masks; insufficient evidence also needs review."""
    rejected = []
    for link, scores in comparison["links"].items():
        old, new = scores.get("baseline"), scores.get("inferred")
        old_iou, new_iou = scores.get("baseline_iou"), scores.get("inferred_iou")
        if (scores.get("frames", 0) < minimum_frames or old is None or new is None
                or not np.isfinite([old, new]).all() or old - new > max_drop
                or (scores.get("iou_informative", True) and old_iou is not None and new_iou is not None and
                    (not np.isfinite([old_iou, new_iou]).all() or old_iou - new_iou > max_drop))):
            rejected.append(link)
    return rejected


def compare_visuals(track_dir: Path, video: Path, baseline_xml: str,
                    links: list[str], samples: int = 60, *, baseline_replay: Path | None = None) -> dict:
    """Coverage of the same usable masks for the old and generated model.

    Samples use the actual recorded camera pose and joint trajectory. This is a
    guard against visibly worse shape, not independent ground-truth validation.
    """
    cams = np.load(track_dir / "replay3d.npz")
    old_cams = np.load(baseline_replay) if baseline_replay else cams
    (w, h), n, _ = video_info(video)
    n = min(n, len(cams["cam_to_world"]), len(old_cams["cam_to_world"]))
    scale = min(1., 480 / max(w, h))
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    _, tracks = load_parts(track_dir)
    shows = link_masks(track_dir)
    parts = {link: [p for p in shows.get(link, []) if p in tracks] for link in links}
    # Multiple cabinet parts are simultaneous pieces of the fixed body. Hinged
    # lid tracks, by contrast, can be alternate open/closed observations.
    multi_base = (track_dir / "drawers.json").exists()
    roles_path = track_dir / "roles.json"
    open_indicator = None
    if roles_path.exists() and not (track_dir / "slide.json").exists():
        open_indicator = json.loads(roles_path.read_text()).get("open_indicator")
    unions_path = track_dir / "track_unions.json"
    unions = json.loads(unions_path.read_text()) if unions_path.exists() else {}
    part_info = json.loads((track_dir / "parts.json").read_text())
    partial = set(unions)
    for spec in part_info.values():
        prompt = spec.get("prompt", "").lower()
        if "laptop" in prompt and any(word in prompt for word in ("keyboard", "touchpad", "trackpad")):
            from .pipeline import slug
            partial.add(slug(spec.get("name") or spec.get("prompt", "")))
    # MuJoCo resolves mesh paths relative to the XML, so put the temporary file
    # next to the model and unlink it even when loading or rendering fails.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", prefix=".appearance-check-",
                                     dir=track_dir, delete=False) as tmp:
        tmp.write(baseline_xml)
        baseline = Path(tmp.name)
    renderers = []
    try:
        models = [mujoco.MjModel.from_xml_path(str(p)) for p in (baseline, track_dir / "model.xml")]
        datas = [mujoco.MjData(m) for m in models]
        for model in models:
            model.cam_fovy[model.camera("video").id] = np.rad2deg(2 * np.arctan(h / (2 * float(cams["f"]))))
            renderer = mujoco.Renderer(model, sh, sw)
            renderer.enable_segmentation_rendering()
            renderers.append(renderer)
        counts = {k: [[], []] for k in links}
        ious = {k: [[], []] for k in links}
        flip = np.diag([1., -1., -1.])
        for t in np.unique(np.linspace(0, n - 1, min(n, samples)).astype(int)):
            masks = {}
            for link, candidates in parts.items():
                if link == "base" and open_indicator:
                    if not open_frame_available(tracks, open_indicator, int(t), w, h):
                        continue  # closed-lid masks can include a different rigid body
                full = observed_mask(tracks, candidates, int(t), w, h, combine=multi_base and link == "base")
                if full.any():
                    masks[link] = cv2.resize(full, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
            if not masks:
                continue
            for i, (model, data, renderer) in enumerate(zip(models, datas, renderers)):
                rec = old_cams if i == 0 else cams
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, (rec["cam_to_world"][t][:3, :3] @ flip).flatten())
                camera = model.camera("video").id
                model.cam_pos[camera], model.cam_quat[camera] = rec["cam_to_world"][t][:3, 3], quat
                for key in rec.files:
                    if key.startswith("q_"):
                        data.joint(key[2:]).qpos[0] = rec[key][t]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="video")
                seg = renderer.render()
                geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
                body = np.where(geom >= 0, model.geom_bodyid[np.maximum(geom, 0)], -1)
                for link, mask in masks.items():
                    if mask.any():
                        rendered = body == model.body(link).id
                        intersection = (rendered & mask).sum()
                        counts[link][i].append(float(intersection / mask.sum()))
                        ious[link][i].append(float(intersection / max(1, (rendered | mask).sum())))
        result = {}
        for link, (old, new) in counts.items():
            result[link] = {"frames": len(old), "baseline": float(np.mean(old)) if old else None,
                            "inferred": float(np.mean(new)) if new else None,
                            "baseline_iou": float(np.mean(ious[link][0])) if old else None,
                            "inferred_iou": float(np.mean(ious[link][1])) if new else None,
                            # Drawer tracks are front/side panels, not complete 3D link silhouettes.
                            "iou_informative": not (multi_base or any(p in partial for p in parts[link][:1]))}
        return {"resolution": [sw, sh], "sample_limit": samples, "links": result}
    finally:
        for renderer in renderers:
            renderer.close()
        baseline.unlink(missing_ok=True)
