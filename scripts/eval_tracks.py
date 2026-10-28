"""Score SAM tracks of a synthetic video against MuJoCo ground-truth silhouettes.

Usage: .venv/bin/python scripts/eval_tracks.py lamp [--synth data/synth] [--tracks data/tracks]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf.sam import load_tracks  # noqa: E402
from video2urdf.synth import gt_masks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset")
    ap.add_argument("--synth", type=Path, default=ROOT / "data" / "synth")
    ap.add_argument("--tracks", type=Path, default=ROOT / "data" / "tracks")
    args = ap.parse_args()

    gt_json = args.synth / f"{args.preset}.gt.json"
    body_of_prompt = {p: b for b, p in json.loads(gt_json.read_text())["prompts"].items()}
    masks = gt_masks(gt_json)
    track_dir = args.tracks / args.preset
    parts = json.loads((track_dir / "parts.json").read_text())
    objs = load_tracks(track_dir / "tracks.sam.txt")
    for oid, part in parts.items():
        body = body_of_prompt.get(part["prompt"])
        if body is None:
            print(f"{part['prompt']!r}: no ground-truth body for this prompt")
            continue
        gt = masks[body]
        n, h, w = gt.shape
        frames = objs.get(int(oid), {})
        ious = []
        for i in range(n):
            pred = frames[i].full_mask(h, w) if i in frames else np.zeros((h, w), bool)
            union = (pred | gt[i]).sum()
            ious.append((pred & gt[i]).sum() / union if union else 1.0)
        ious = np.array(ious)
        print(f"{part['prompt']!r} -> {body}: tracked {len(frames)}/{n}, IoU mean {ious.mean():.3f}, "
              f"min {ious.min():.3f} (frame {ious.argmin()}), frames < 0.8: {(ious < 0.8).sum()}")


if __name__ == "__main__":
    main()
