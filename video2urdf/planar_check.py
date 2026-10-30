"""Compare planar visual candidates to the same tracked video masks."""
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .pipeline import load_parts, video_info
from .replay import _hold, _hold_poses


def compare_planar(baseline: Path, candidate: Path, video: Path, links: list[str], samples: int = 60) -> dict:
    """Compare side-view IoU with recorded root motion; masks are not ground truth."""
    (w, h), n, _ = video_info(video)
    names, tracks = load_parts(baseline)
    if any(name not in names for name in links):
        raise ValueError('comparison link is not a tracked part')
    with np.load(baseline / 'planar_replay.npz') as data:
        root = _hold_poses(data['root_poses'])
    joints = json.loads((baseline / 'joints.json').read_text())['joints']
    q = {j['child']: _hold(np.array([np.nan if x is None else x for x in j['q']], float))
         for j in joints if j['type'] != 'fixed'}
    n = min([n, len(root), *(len(x) for x in q.values())])
    if n < 1 or samples < 1:
        raise ValueError('no frames to compare')
    scale = min(1., 480 / max(w, h))
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    resize = np.diag([sw / w, sh / h, 1.])
    renderers = []
    scores = {name: [[], []] for name in links}
    try:
        models = [mujoco.MjModel.from_xml_path(str(p / 'model.xml')) for p in (baseline, candidate)]
        datas = [mujoco.MjData(model) for model in models]
        for model in models:
            renderer = mujoco.Renderer(model, sh, sw)
            renderer.enable_segmentation_rendering()
            renderers.append(renderer)
        for t in np.unique(np.linspace(0, n - 1, min(n, samples)).astype(int)):
            masks = {name: cv2.resize(tracks[name][int(t)].full_mask(h, w).astype(np.uint8),
                                      (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
                     for name in links if int(t) in tracks[name] and tracks[name][int(t)].usable(w, h)}
            if not masks:
                continue
            transform = resize @ root[t] @ np.linalg.inv(resize)
            for i, (model, data, renderer) in enumerate(zip(models, datas, renderers)):
                for name, values in q.items():
                    data.joint(name + '_joint').qpos[0] = values[t]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera='video')
                seg = renderer.render()
                geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
                body = np.where(geom >= 0, model.geom_bodyid[np.maximum(geom, 0)], -1).astype(np.int16)
                body = cv2.warpAffine(body, transform[:2], (sw, sh), flags=cv2.INTER_NEAREST, borderValue=-1)
                for name, mask in masks.items():
                    pred = body == model.body(name).id
                    union = np.count_nonzero(pred | mask)
                    scores[name][i].append(float(np.count_nonzero(pred & mask) / union) if union else 1.)
        return {'resolution': [sw, sh], 'sample_limit': samples,
                'links': {name: {'frames': len(old), 'baseline_iou': float(np.mean(old)) if old else None,
                                 'inferred_iou': float(np.mean(new)) if new else None}
                          for name, (old, new) in scores.items()},
                'note': 'Same sampled tracked masks and root motion; not independent ground truth.'}
    finally:
        for renderer in renderers:
            renderer.close()
