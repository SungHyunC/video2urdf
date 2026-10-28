"""Render synthetic ground-truth videos.

Usage: .venv/bin/python scripts/make_synth.py [lamp drawer ...] [--out data/synth] [--seconds 4]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf.synth import PRESETS, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("presets", nargs="*", default=list(PRESETS))
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synth")
    ap.add_argument("--seconds", type=float, default=4.0)
    args = ap.parse_args()
    for name in args.presets:
        print(render(name, args.out, args.seconds))


if __name__ == "__main__":
    main()
