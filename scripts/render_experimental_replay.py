"""Render a review-only overlay for an experimental laptop alignment."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from video2urdf.pipeline import load_parts  # noqa: E402
from video2urdf.replay import replay3d  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('video', type=Path)
    args = parser.parse_args()
    if not (args.run / 'experimental_alignment.json').exists():
        parser.error('run must be an experimental alignment copy')
    out = args.run / 'experimental_replay.mp4'
    if out.exists() or out.with_suffix('.npz').exists():
        parser.error('experimental replay already exists; preserve it and use a new copy')
    roles = json.loads((args.run / 'roles.json').read_text())
    _, tracks = load_parts(args.run)
    with np.load(args.run / 'replay3d.npz') as rec:
        camera = rec['cam_to_world'].copy()
        intr = SimpleNamespace(f=float(rec['f']), cx=float(rec['cx']), cy=float(rec['cy']))
        q = rec['q_lid_joint'].copy()
    masks = {'base': tracks[roles['base']], 'lid': tracks[roles['plate']]}
    indicator = roles.get('open_indicator')
    only_when = {'base': lambda t: t in tracks.get(indicator, {})} if indicator else None
    coverage = replay3d(args.run / 'model.xml', camera, intr, {'lid_joint': q}, masks,
                        args.video, out, only_when=only_when, experimental=True)
    summary = {key: {'frames': int(np.isfinite(values).sum()),
                     'mean': float(np.nanmean(values)) if np.isfinite(values).any() else None}
               for key, values in coverage.items()}
    (args.run / 'experimental_replay_coverage.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
