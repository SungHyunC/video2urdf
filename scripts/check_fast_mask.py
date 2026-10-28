"""Check video2urdf.fast_mask against meta-sam-parser's reference decoder on cached tracks.

Usage: .venv/bin/python scripts/check_fast_mask.py data/tracks/stand_2/tracks.sam.txt [--n 30]
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
from meta_sam_parser import _mask_codec

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf.fast_mask import decode_one_bit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", type=Path)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    tokens = re.findall(r"data=(\d+),(\d+),(![^|]*)", args.tracks.read_text())
    idx = np.linspace(0, len(tokens) - 1, min(args.n, len(tokens))).astype(int)
    decode_one_bit(tokens[0][2], int(tokens[0][1]), int(tokens[0][0]))  # compile
    t_fast = t_ref = 0.0
    for i in idx:
        h, w, payload = int(tokens[i][0]), int(tokens[i][1]), tokens[i][2]
        t0 = time.time()
        fast = decode_one_bit(payload, w, h)
        t1 = time.time()
        ref = np.frombuffer(_mask_codec._decode_raster(payload, w, h), np.uint8).reshape(h, w) > 0
        t2 = time.time()
        t_fast, t_ref = t_fast + t1 - t0, t_ref + t2 - t1
        if not np.array_equal(fast, ref):
            sys.exit(f"MISMATCH on mask {i} ({h}x{w}): {int((fast != ref).sum())} pixels differ")
    print(f"{len(idx)} masks identical; fast {1000 * t_fast / len(idx):.1f} ms/mask, "
          f"reference {1000 * t_ref / len(idx):.0f} ms/mask")


if __name__ == "__main__":
    main()
