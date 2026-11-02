"""Reproduce the published iPhone LiDAR laptop numbers offline: no API keys, no GPU, a few minutes.

The model-driven steps of that run (Muse Spark's decisions, SAM 3.1's tracks) are saved in a
2 MB bundle (release sample-data-v2); the capture is the 41 MB sample (release sample-data-v1).
This reruns everything after the models on the CPU, the agent's own fit_lidar tool (hinge, link
geometry from depth, URDF and MJCF, the MuJoCo replay from the phone's camera poses and how much
of each SAM mask it covers) and the physics step on Spark's saved estimate (joint friction, the
release check), then prints the published numbers (docs/models/laptop_lidar/model.json) next to
the reproduced ones. No model or API is called; once the downloads are done the network is
refused. Details: docs/REPRODUCE.md.

Usage:
  .venv/bin/python scripts/reproduce.py                        # downloads both files
  .venv/bin/python scripts/reproduce.py --bundle laptop_lidar_tracks.zip --capture laptop_lidar_stray.zip
  .venv/bin/python scripts/reproduce.py --reference auto      # today's code picks the reference frame

Writes --out (default data/reproduce/laptop_lidar; never overwritten, a second run gets _2):
model.urdf, model.xml, meshes/, replay3d.mp4, hinge.json, hinge.jpg, model3d.json, physics.json,
fit_lidar.txt (what the fit tool said), reproduce.md (the table) and reproduce.json.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf import reproduce as R  # noqa: E402


def fresh_folder(path: Path) -> Path:
    candidate, i = path, 1
    while candidate.exists():
        i += 1
        candidate = path.with_name(f"{path.name}_{i}")
    return candidate


def fetch_inputs(args, log=print) -> tuple[Path, Path]:
    try:
        bundle = R.fetch(args.bundle, args.cache, "saved tracks", R.BUNDLE_SHA256, log=log)
    except R.DownloadError as error:
        if error.status == 404 and args.bundle == R.BUNDLE_URL:
            raise R.ReproduceError(
                f"the saved tracks aren't published yet: {R.BUNDLE_URL} answered HTTP 404 (the sample-data-v2 "
                f"release doesn't exist yet). If you have {R.BUNDLE_NAME}.zip, pass it with --bundle PATH.") from error
        raise R.ReproduceError(f"couldn't download the saved tracks ({error}). Download {args.bundle} yourself "
                               "and pass it with --bundle PATH.") from error
    if bundle.is_file() and R.BUNDLE_SHA256 and R.sha256(bundle) != R.BUNDLE_SHA256:
        log(f"note: {bundle.name} isn't the published sample-data-v2 asset (its sha256 differs); "
            "its own manifest is checked instead")
    try:
        capture = R.fetch(args.capture, args.cache, "capture", R.CAPTURE_SHA256, log=log)
    except R.DownloadError as error:
        raise R.ReproduceError(f"couldn't download the capture ({error}). Download {args.capture} yourself "
                               "and pass it with --capture PATH.") from error
    return bundle, capture


def versions() -> dict:
    import cv2
    import mujoco
    import numpy
    import scipy
    v = {"code": str(Path(R.__file__).resolve().parent), "python": platform.python_version(),
         "platform": platform.platform(), "numpy": numpy.__version__, "scipy": scipy.__version__,
         "mujoco": mujoco.__version__, "opencv": cv2.__version__}
    try:
        v["commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                     text=True, timeout=10).stdout.strip() or None
    except OSError:
        v["commit"] = None
    return v


def run(args) -> int:
    t0 = time.time()
    published = json.loads(args.published.read_text())
    published_ref = published["measured"]["reference_frame"]
    if args.reference == "published":
        reference = published_ref
    elif args.reference == "auto":
        reference = None
    elif args.reference.isdigit():
        reference = int(args.reference)
    else:
        raise R.ReproduceError(f"--reference takes 'published', 'auto' or a frame number, not {args.reference!r}")
    out = fresh_folder(args.out)
    print(f"Reproducing {args.published} into {out}.\nNo API key or model call is used; the network is refused "
          "once the downloads are done.\n")

    R.preflight(args.cache)
    bundle, capture = fetch_inputs(args)
    bundle_root, manifest = R.open_bundle(bundle, args.cache)
    capture_root = R.open_capture(capture, args.cache, manifest)
    timings = {"downloads, unpacking and checks": time.time() - t0}
    print(f"saved tracks: {bundle_root} (made from {manifest['made_from']})\ncapture: {capture_root}\n")

    t1 = time.time()
    with R.offline() as set_aside:
        if set_aside:
            print(f"(set aside for this run: {', '.join(sorted(set_aside))} from the environment)")
        said = R.rerun(bundle_root, capture_root, out, reference)
    timings["fit, replay and physics"] = time.time() - t1
    print("\nfit_lidar said:\n" + said["fit"] + "\n\nphysics said:\n" + "\n".join(said["physics"]))

    pub, rep = R.published_values(published), R.rerun_values(out)
    rows = R.compare(pub, rep)
    timings["total"] = time.time() - t0
    how = {"published": f"pinned to the published fit's (model.json: {published_ref})",
           "auto": "chosen by today's code (--reference auto)"}.get(args.reference, f"given: {reference}")
    notes = [
        f"Reference frame {how}. " + ("--reference auto lets today's code choose it; docs/REPRODUCE.md says why it "
                                      "takes another frame on this capture." if args.reference != "auto" else
                                      "The published fit used frame %d; docs/REPRODUCE.md says why today's code "
                                      "takes another." % published_ref),
        "Published coverage is the provenance note's (whole percents); the masses are inputs from Spark's saved "
        "estimate: " + ", ".join(f"{k} {v['mass_kg']} kg" for k, v in published.get("physics", {}).get("links", {}).items())
        + ".",
        "Wrote " + ", ".join(n for n in ("model.urdf", "model.xml", "meshes/", "replay3d.mp4", "hinge.jpg")
                             if (out / n.rstrip("/")).exists())
        + f", reproduce.md and reproduce.json in {out} (replay3d.mp4: the model drawn over the video from the "
          "phone's own camera poses).",
        "Took " + ", ".join(f"{k} {v:.0f} s" for k, v in timings.items())
        + " (load average %.1f at the end)." % os.getloadavg()[0],
    ]
    (out / "reproduce.md").write_text(f"# Reproduced: {R.CASE}\n\n" + R.markdown_table(rows) + "\n\n"
                                      + "\n\n".join(notes) + "\n")
    (out / "reproduce.json").write_text(json.dumps({
        "case": R.CASE, "reference": args.reference, "reference_frame": rep["reference_frame"],
        "rows": [vars(r) for r in rows], "timings_s": timings, "load_average": os.getloadavg(),
        "versions": versions(), "bundle": {"source": args.bundle, "manifest": manifest},
        "capture": {"source": args.capture, "folder": str(capture_root)},
    }, indent=2) + "\n")
    print("\n" + R.text_table(rows) + "\n\n" + "\n".join(notes))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=R.BUNDLE_URL,
                    help="the saved tracks: the zip, its unpacked folder or a URL (default: the sample-data-v2 asset)")
    ap.add_argument("--capture", default=R.CAPTURE_URL,
                    help="the capture: the zip, its unpacked folder or a URL (default: the sample-data-v1 asset)")
    ap.add_argument("--reference", default="published",
                    help="the reference frame (the joint's zero; the lid's face and the base's top are measured on "
                         "it): 'published' (the published fit's, from model.json), 'auto' (today's code chooses) or "
                         "a frame number")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "reproduce" / R.CASE,
                    help="output folder (an existing one is kept: the run goes to <out>_2, ...)")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "reproduce" / "downloads",
                    help="downloads and the unpacked capture, reused by later runs")
    ap.add_argument("--published", type=Path, default=ROOT / "docs" / "models" / R.CASE / "model.json",
                    help="the published model.json to compare with")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress and the final stop message in order in a log
    try:
        sys.exit(run(args))
    except R.ReproduceError as error:
        print(f"\nStopped: {error}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
