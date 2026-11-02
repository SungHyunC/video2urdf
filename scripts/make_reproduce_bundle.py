"""Build the bundle scripts/reproduce.py reruns: the saved model decisions for one published
capture (SAM 3.1's tracks, the parts Muse Spark tracked, the roles it gave them in fit_lidar,
its physics estimate) with a manifest of their hashes and of the capture they belong to.
Entries have fixed times and order, so the same files give the same zip (and sha256).

Usage: .venv/bin/python scripts/make_reproduce_bundle.py data/tracks_design/laptop_lidar \
           data/real/laptop_lidar/9a84cd46da.zip --out data/reproduce_bundle/laptop_lidar_tracks.zip
(the run folder of the published model, and the capture zip published as sample-data-v1)
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf import reproduce as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the published model's run folder (tracks.sam.txt, parts.json, ...)")
    ap.add_argument("capture_zip", type=Path, help="the capture zip, byte for byte the published asset")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", help="how the manifest names the run folder (default data/<parent>/<name>)")
    args = ap.parse_args()
    if R.sha256(args.capture_zip) != R.CAPTURE_SHA256:
        raise SystemExit(f"{args.capture_zip} isn't the published capture ({R.CAPTURE_URL})")
    manifest = R.build_bundle(args.run, args.capture_zip, args.out,
                              args.source or f"data/{args.run.parent.name}/{args.run.name}")
    print(json.dumps(manifest, indent=2))
    print(f"\n{args.out}: {args.out.stat().st_size / 2 ** 20:.2f} MB, sha256 {R.sha256(args.out)}")


if __name__ == "__main__":
    main()
