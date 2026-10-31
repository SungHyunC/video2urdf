"""Refit saved rod-chain tracks with the agent's own fit_chain tool, on a copy, for regression checks.

Reuses a run's tracks: no new segmentation or model calls. The parts come from the run's
chain.json (written by an earlier fit_chain) unless --base/--chain/--floor/--fixed are given.
The source folder is left unchanged; the refit goes to --out.

Usage: .venv/bin/python scripts/refit_chain.py data/tracks_agent2/stand_lidar data/real/stand_lidar \
           --out data/refit/stand_lidar
       .venv/bin/python scripts/refit_chain.py data/tracks_phonestand/phoneStand_lidar data/real/phoneStand_lidar \
           --out data/refit/phonestand --base floor --chain pole arm_prox arm_dist --floor floor --fixed pole
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KEEP = ("tracks.sam.txt", "tracks.sam.npz", "parts.json", "track_unions.json", "measurement_source.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="track folder with tracks.sam.* and parts.json")
    ap.add_argument("capture", type=Path, help="the Stray Scanner capture folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base")
    ap.add_argument("--chain", nargs="+")
    ap.add_argument("--floor")
    ap.add_argument("--fixed", nargs="*")
    args = ap.parse_args()
    from video2urdf.agent import Agent
    from video2urdf.stray import find_capture
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; pick a new folder")
    spec = json.loads((args.run / "chain.json").read_text()) if (args.run / "chain.json").exists() else {}
    base, chain = args.base or spec.get("base"), args.chain or spec.get("chain")
    floor = args.floor if args.floor is not None else spec.get("floor")
    fixed = args.fixed if args.fixed is not None else spec.get("fixed", [])
    if not base or not chain:
        raise SystemExit("no chain.json in the run: give --base and --chain")
    args.out.mkdir(parents=True)
    for name in KEEP:  # the text before its .npz cache, so the cache stays newer
        if (args.run / name).exists():
            shutil.copy(args.run / name, args.out / name)
    video = next((p for p in sorted(args.capture.glob("*.mp4"))), find_capture(args.capture) / "rgb.mp4")
    agent = Agent(video, args.out, capture=args.capture, client=object(), sam=object())
    agent.tracked = True
    print(agent.tool_fit_chain(base=base, chain=chain, floor=floor, fixed=fixed))
    meta = args.out / "model3d.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        print(json.dumps({"reference_frame": m.get("reference_frame"), "anchor": m.get("anchor"),
                          "joints": [{k: j.get(k) for k in ("parent", "child", "type", "rod_length_mm", "range_deg",
                                                            "pivot_model_mm", "estimated")} for j in m["joints"]],
                          "mask_coverage": m.get("mask_coverage"), "fit_warnings": m.get("fit_warnings")}, indent=1))
    print("status:", agent.check_status())


if __name__ == "__main__":
    main()
