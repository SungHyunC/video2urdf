"""Build an object-aware model from an RGB-D capture or an ordinary video.

Ordinary video first goes through MapAnything on Modal. Its scale is not guaranteed:
use --scale when a known length is available and validate dimensions independently.
An RGB-D capture with metric depth is preferred for dimensional measurements.
"""
import argparse
import json
import math
import shutil
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def laptop_review_candidate(target: Path, video: Path) -> Path | None:
    """Produce a separate visual hypothesis when the fitted hinge over-closes."""
    appearance_path, hinge_path = target / 'appearance.json', target / 'hinge.json'
    if not appearance_path.exists() or not hinge_path.exists():
        return None
    appearance = json.loads(appearance_path.read_text())
    hinge = json.loads(hinge_path.read_text())
    from video2urdf.track_checks import overclosed_hinge_warning
    if appearance.get('category') != 'laptop' or not overclosed_hinge_warning(hinge):
        return None
    candidate = target.with_name(target.name + '_laptop_review')
    if candidate.exists():
        print(f'Laptop review candidate already exists; leaving it untouched: {candidate}', flush=True)
        return None
    from scripts.experimental_laptop_align import align_copy
    comparison_warning = None
    try:
        align_copy(target, candidate)
    except Exception as error:
        print(f'Laptop review candidate unavailable ({type(error).__name__}): {error}', flush=True)
        return None
    # Compare both trajectories on the same sampled masks, using each model's
    # original mesh assets; candidate meshes deliberately have the same names.
    try:
        from video2urdf.appearance_check import compare_visuals, rejected_links
        baseline = ET.parse(target / 'model.xml').getroot()
        for asset in baseline.findall('./asset/*'):
            file = asset.get('file')
            if file:
                asset.set('file', str((target / file).resolve()))
        comparison = compare_visuals(candidate, video, ET.tostring(baseline, encoding='unicode'),
                                     ['base', 'lid'], baseline_replay=target / 'replay3d.npz')
        comparison['rejected_links'] = rejected_links(comparison)
        comparison['note'] = 'Same sampled tracked masks; tracking may be wrong and this is not ground truth.'
        (candidate / 'experimental_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
        if comparison['rejected_links']:
            comparison_warning = f"Laptop review candidate failed sampled mask checks for {', '.join(comparison['rejected_links'])}; inspect before use."
            print(f"Laptop review candidate did not pass the sampled mask comparison: {comparison['rejected_links']}", flush=True)
    except Exception as error:
        comparison_warning = 'Laptop review candidate could not be compared with the original on sampled masks.'
        print(f'Laptop review comparison unavailable ({type(error).__name__}): {error}', flush=True)
    if comparison_warning:
        quality_path = candidate / 'quality.json'
        quality = json.loads(quality_path.read_text())
        quality['warnings'].append(comparison_warning)
        quality_path.write_text(json.dumps(quality, indent=2) + '\n')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/render_experimental_replay.py'),
                             str(candidate), str(video)], check=False)
    if result.returncode:
        # The renderer may leave an incomplete file; do not export it as a replay.
        (candidate / 'experimental_replay.mp4').unlink(missing_ok=True)
        (candidate / 'experimental_replay.npz').unlink(missing_ok=True)
        print(f'Laptop review overlay failed; inspect the candidate separately: {candidate}', flush=True)
    print(f'Laptop review candidate (visual and angles assumed; physics not validated): {candidate}', flush=True)
    return candidate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="Stray Scanner/RGB-D capture directory or phone video")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "tracks")
    ap.add_argument("--capture-out", type=Path, help="MapAnything capture folder for a video input")
    ap.add_argument("--focal-35mm", type=float, help="known 35 mm equivalent camera focal length")
    ap.add_argument("--scale", type=float, help="MapAnything scale factor: known length / predicted length")
    ap.add_argument("--reference", nargs=6, type=float, metavar=("FRAME", "X1", "Y1", "X2", "Y2", "LENGTH_MM"),
                    help="calibrate new RGB-video capture with two image pixels and their known distance in mm")
    ap.add_argument("--check-reference", nargs=6, type=float, action="append", default=[],
                    metavar=("FRAME", "X1", "Y1", "X2", "Y2", "LENGTH_MM"),
                    help="independent known distance to check the calibrated depth; repeatable")
    ap.add_argument("--no-auto-scale", action="store_true",
                    help="keep MapAnything's own scale when no --scale/--reference is given (default: find an ID-1 "
                         "card with SAM 3.1, else use the object's usual size from Muse Spark)")
    ap.add_argument("--open-weights", action="store_true")
    ap.add_argument("--planar", action="store_true", help="use an approximate side-view model for articulated objects")
    ap.add_argument("--glimmer", action="store_true")
    ap.add_argument("--sam", choices=["api", "open"])
    ap.add_argument("--video-texture", action="store_true", help="project video colours onto the inferred shape")
    ap.add_argument("--sam3d", action="store_true", help="also generate depth-checked SAM 3D visual meshes on Modal")
    ap.add_argument("--sam3d-retries", type=int, default=0, help="retry failed SAM 3D links on other frames (0–4)")
    ap.add_argument("--sam3d-video-texture", action="store_true", help="experimentally transfer video textures to accepted SAM 3D visuals")
    ap.add_argument("--viewer-out", type=Path, help="write a standalone local viewer to a new empty directory")
    args = ap.parse_args()
    if args.sam3d_retries < 0 or args.sam3d_retries > 4 or (args.sam3d_retries and not args.sam3d):
        ap.error("--sam3d-retries requires --sam3d and must be 0–4")
    if args.sam3d_video_texture and not args.sam3d:
        ap.error("--sam3d-video-texture requires --sam3d")
    if args.scale is not None and (not math.isfinite(args.scale) or args.scale <= 0):
        ap.error("--scale must be positive")
    if args.focal_35mm is not None and (not math.isfinite(args.focal_35mm) or args.focal_35mm <= 0):
        ap.error("--focal-35mm must be positive and finite")
    if args.scale is not None and args.reference:
        ap.error("use either --scale or --reference, not both")
    if args.check_reference and not args.reference:
        ap.error("--check-reference requires --reference")
    for reference in ([args.reference] if args.reference else []) + args.check_reference:
        if (not all(map(math.isfinite, reference)) or not reference[0].is_integer()
                or reference[0] < 0 or reference[5] <= 0):
            ap.error("reference FRAME must be a nonnegative integer and LENGTH_MM must be positive")
    if args.reference:
        frame, x1, y1, x2, y2, _ = args.reference
        for check in args.check_reference:
            if check[0] == frame and {(check[1], check[2]), (check[3], check[4])} == {(x1, y1), (x2, y2)}:
                ap.error("--check-reference must use different points from the calibration reference")
    if args.viewer_out and args.viewer_out.exists() and any(args.viewer_out.iterdir()):
        ap.error(f"{args.viewer_out} is not empty; choose a new --viewer-out directory")
    if args.planar:
        if not args.input.is_file():
            ap.error('--planar requires an ordinary video file')
        if (args.capture_out or args.scale is not None or args.reference or args.check_reference or
                args.focal_35mm is not None or args.sam3d or args.video_texture or args.sam3d_video_texture):
            ap.error('--planar does not use depth calibration, SAM 3D or video texture options')
        target = args.out / args.input.stem
        if target.exists() and any(target.iterdir()):
            ap.error(f'{target} already has results; choose another --out to preserve them')
        command = [sys.executable, str(ROOT / 'scripts/run_agent.py'), str(args.input), '--out', str(args.out), '--appearance']
        if args.open_weights:
            command.append('--open-weights')
        if args.glimmer:
            command.append('--glimmer')
        if args.sam:
            command += ['--sam', args.sam]
        subprocess.run(command, check=True)
        if args.viewer_out:
            args.viewer_out.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / 'docs/index.html', args.viewer_out / 'index.html')
            subprocess.run([sys.executable, str(ROOT / 'scripts/export_viewer.py'),
                            f'{args.input.stem}:{target}:{args.input}', '--out', str(args.viewer_out)], check=True)
            print(f'Local viewer: python -m http.server --bind 127.0.0.1 --directory {shlex.quote(str(args.viewer_out))}')
        return
    if args.input.is_file():
        if args.reference:
            from video2urdf.pipeline import video_info
            (width, height), frames, _ = video_info(args.input)
            for reference in [args.reference] + args.check_reference:
                frame, x1, y1, x2, y2, _ = reference
                if frame >= frames or not all(0 <= x < width and 0 <= y < height for x, y in ((x1, y1), (x2, y2))):
                    ap.error(f"reference must use a frame in 0..{frames - 1} and pixels inside {width}x{height}")
        capture = args.capture_out or args.input.parent / (args.input.stem + "_mapanything")
        # MapAnything is potentially expensive; reject a conflicting final
        # result before starting reconstruction or creating a capture.
        target = args.out / capture.name
        if target.exists() and any(target.iterdir()):
            ap.error(f"{target} already has results; choose another --out to preserve them")
        if (capture / "mapanything.json").exists():
            meta = json.loads((capture / "mapanything.json").read_text())
            if Path(meta["source"]).resolve() != args.input.resolve():
                ap.error(f"{capture} contains a capture of a different video")
            if args.scale is not None or args.reference:
                ap.error("capture already exists: choose a new --capture-out rather than modifying it")
            print(f"Reusing MapAnything capture: {capture}", flush=True)
        else:
            if capture.exists() and any(capture.iterdir()):
                ap.error(f"{capture} already contains files but is not a MapAnything capture; choose --capture-out")
            command = [sys.executable, str(ROOT / "scripts/video_to_capture.py"), str(args.input), "--out", str(capture)]
            if args.focal_35mm is not None:
                command += ["--focal-35mm", str(args.focal_35mm)]
            if args.scale is not None:
                command += ["--scale", str(args.scale)]
            print("Reconstructing video depth with MapAnything on Modal. Metric scale is not guaranteed.", flush=True)
            subprocess.run(command, check=True)
            if args.reference:
                from video2urdf.calibration import reference_scale, check_references
                from video2urdf.pseudo_depth import rescale_capture
                frame, x1, y1, x2, y2, length = args.reference
                ref = reference_scale(capture, int(frame), (x1, y1), (x2, y2), length)
                checks = check_references(capture, args.check_reference, ref["scale_factor"])
                rescale_capture(capture, ref["scale_factor"])
                meta_path = capture / "mapanything.json"
                info = json.loads(meta_path.read_text())
                info["scale_applied"] = info.get("scale_applied", 1.0) * ref["scale_factor"]
                info["reference_calibration"] = ref
                info["reference_checks"] = checks
                meta_path.write_text(json.dumps(info, indent=2) + "\n")
                print(f"Reference calibration: estimated {ref['estimated_length_before_mm']:.1f} mm, "
                      f"known {ref['known_length_mm']:.1f} mm, scale x{ref['scale_factor']:.5f}", flush=True)
                for check in checks:
                    print(f"Independent check: {check['predicted_after_calibration_mm']:.1f} mm vs "
                          f"{check['known_length_mm']:.1f} mm ({check['relative_error']:+.1%})", flush=True)
            elif args.scale is None and not args.no_auto_scale:
                from video2urdf.autoscale import auto_scale
                from video2urdf.pseudo_depth import rescale_capture
                print("No reference given: looking for an ID-1 card, else the object's usual size.", flush=True)
                found = auto_scale(capture, capture / "rgb.mp4", log=lambda s: print("  " + s, flush=True))
                if found is None:
                    print("Automatic scale not found; MapAnything's scale is kept (it can be several times off).", flush=True)
                else:
                    rescale_capture(capture, found["scale_factor"])
                    meta_path = capture / "mapanything.json"
                    info = json.loads(meta_path.read_text())
                    info["scale_applied"] = info.get("scale_applied", 1.0) * found["scale_factor"]
                    info["auto_scale"] = found
                    meta_path.write_text(json.dumps(info, indent=2) + "\n")
                    print(f"Automatic scale ({found['method']}): x{found['scale_factor']:.3f}", flush=True)
    elif args.input.is_dir():
        if args.capture_out or args.scale or args.focal_35mm or args.reference:
            ap.error("--capture-out, --scale, --reference and --focal-35mm apply only to ordinary video inputs")
        capture = args.input
    else:
        ap.error(f"input does not exist: {args.input}")
    if (capture / "intrinsics.json").exists():
        videos = sorted(capture.glob("*.mp4"))
        if not videos:
            ap.error(f"no mp4 in RGB-D capture {capture}")
        video_stem = videos[0].stem
    else:
        video_stem = capture.name
    target = args.out / video_stem
    if target.exists() and any(target.iterdir()):
        ap.error(f"{target} already has results; choose another --out to preserve them")
    command = [sys.executable, str(ROOT / "scripts/run_agent.py"), str(capture), "--out", str(args.out), "--appearance"]
    if args.open_weights:
        command.append("--open-weights")
    if args.glimmer:
        command.append("--glimmer")
    if args.sam:
        command += ["--sam", args.sam]
    if args.video_texture:
        command.append("--video-texture")
    if args.sam3d:
        command.append("--sam3d")
    if args.sam3d_video_texture:
        command.append("--sam3d-video-texture")
    if args.sam3d_retries:
        command += ["--sam3d-retries", str(args.sam3d_retries)]
    subprocess.run(command, check=True)
    video = videos[0] if (capture / "intrinsics.json").exists() else capture / (capture.name + ".mp4")
    candidate = laptop_review_candidate(target, video)
    if args.viewer_out:
        args.viewer_out.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "docs/index.html", args.viewer_out / "index.html")
        runs = [f"{video_stem}:{target}:{video}"]
        if candidate:
            runs.append(f"{video_stem}_laptop_review:{candidate}:{video}")
        subprocess.run([sys.executable, str(ROOT / "scripts/export_viewer.py"),
                        *runs, "--out", str(args.viewer_out)], check=True)
        print(f"Local viewer: python -m http.server --bind 127.0.0.1 --directory {shlex.quote(str(args.viewer_out))}")


if __name__ == "__main__":
    main()
