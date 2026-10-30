"""Let Muse Spark build the model for one video.

Usage: .venv/bin/python scripts/run_agent.py data/real/stand_2.MOV [--out data/tracks]
       .venv/bin/python scripts/run_agent.py data/real/laptop_lidar   (a Stray Scanner LiDAR capture)
       .venv/bin/python scripts/run_agent.py data/rbo/prep/laptop02   (a fixed-camera RGB-D folder)

Writes the model files plus agent/log.md (every step, with the images Spark saw) and
agent/report.md into <out>/<video stem>/. AGENT_MODEL, AGENT_BASE_URL and AGENT_API_KEY_ENV
switch the reasoning model and SAM_BACKEND the segmenter. --open-weights runs everything on
open weights: Muse Spark Open on Baseten and SAM 3.1 (facebook/sam3.1) on Modal
(modal deploy modal_apps/sam3_open.py first). --glimmer runs the agent on Muse Glimmer 30B
(open weights) on our own Modal Dedicated Endpoint (GLIMMER_URL; README has the commands).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from video2urdf.stray import find_capture  # noqa: E402

OPEN_WEIGHTS = {"AGENT_MODEL": "meta-models/muse-spark-open", "AGENT_BASE_URL": "https://inference.baseten.co/v1",
                "AGENT_API_KEY_ENV": "BASETEN_API_KEY", "AGENT_MAX_IMAGES": "8", "SAM_BACKEND": "open"}


def write_measurement_source(track_dir: Path, capture: Path | None) -> None:
    """Record capture provenance before inference can fail or be interrupted."""
    if capture is None:
        return
    map_meta = capture / 'mapanything.json'
    if map_meta.exists():
        info = json.loads(map_meta.read_text())
        ref = info.get('reference_calibration')
        if ref:
            note = (f"Global scale calibrated from a {ref['known_length_mm']:.1f} mm reference; "
                    'local depth and focal errors remain.')
        elif info.get('scale_applied', 1.0) != 1.0:
            note = 'A user-supplied global scale was applied; local depth and focal errors remain.'
        else:
            note = 'A known reference length is needed to calibrate scale; depth can still be inaccurate.'
        source = {'depth': 'MapAnything estimated from RGB video', 'metric': 'estimated, not guaranteed',
                  'scale_applied': info.get('scale_applied', 1.0), 'reference_calibration': ref,
                  'reference_checks': info.get('reference_checks', []), 'note': note}
    else:
        source = {'depth': 'RGB-D capture', 'metric': 'capture depth in metres',
                  'note': 'Depth is measured by the capture device and can contain noise and drift; '
                          'hidden dimensions may be assumed by the shape builder.'}
    track_dir.mkdir(parents=True, exist_ok=True)
    (track_dir / 'measurement_source.json').write_text(json.dumps(source, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "tracks")
    ap.add_argument("--open-weights", action="store_true", help="Muse Spark Open + SAM 3.1 weights only")
    ap.add_argument("--glimmer", action="store_true",
                    help="reasoning on Muse Glimmer 30B on our Modal endpoint (with --open-weights: SAM too)")
    ap.add_argument("--sam", choices=["api", "open"], help="override the segmenter alone")
    ap.add_argument("--appearance", action="store_true", help="after a 3D fit, infer object-specific visuals from the video")
    ap.add_argument("--video-texture", action="store_true", help="project video colours onto the inferred shape")
    ap.add_argument("--sam3d", action="store_true", help="generate depth-checked SAM 3D visuals after appearance completion")
    ap.add_argument("--sam3d-retries", type=int, default=0, help="retry failed SAM 3D links on other frames (0–4)")
    ap.add_argument("--sam3d-video-texture", action="store_true", help="experimentally transfer observed textures to accepted SAM 3D visuals")
    args = ap.parse_args()
    if args.sam3d_retries < 0 or args.sam3d_retries > 4 or (args.sam3d_retries and not args.sam3d):
        ap.error("--sam3d-retries requires --sam3d and must be 0–4")
    if args.sam3d_video_texture and not args.sam3d:
        ap.error("--sam3d-video-texture requires --sam3d")
    if args.open_weights:
        os.environ.update(OPEN_WEIGHTS)
    if args.glimmer:  # a Modal Dedicated Endpoint: proxy token as "wk-....ws-..." bearer
        os.environ["GLIMMER_API_KEY"] = f"{os.environ['MODAL_PROXY_TOKEN_ID']}.{os.environ['MODAL_PROXY_TOKEN_SECRET']}"
        os.environ.update({"AGENT_MODEL": "glimmer", "AGENT_BASE_URL": os.environ["GLIMMER_URL"].rstrip("/") + "/v1",
                           "AGENT_API_KEY_ENV": "GLIMMER_API_KEY", "AGENT_MAX_IMAGES": "8",
                           "AGENT_VIDEO": "0"})  # the endpoint drops video input; Glimmer sees frames
    if args.sam:
        os.environ["SAM_BACKEND"] = args.sam
    from video2urdf.agent import Agent  # reads the backend settings on import
    video, capture = args.video, None
    if args.video.is_dir() and (args.video / "intrinsics.json").exists():  # fixed-camera RGB-D (rbo_eval.py --prepare-only)
        capture, video = args.video, next(args.video.glob("*.mp4"))
    elif args.video.is_dir():  # an iPhone LiDAR capture (Stray Scanner export, possibly nested)
        capture = args.video
        rgb = find_capture(capture) / "rgb.mp4"
        video = capture / f"{capture.name}.mp4"  # tracks are filed under the video's name
        if not video.exists():
            shutil.copy(rgb, video)
    track_dir = args.out / video.stem
    agent = Agent(video, track_dir, capture=capture)
    write_measurement_source(track_dir, capture)
    report = agent.run()
    print("\n=== report ===\n" + report)
    if args.appearance or args.sam3d:
        from video2urdf.pipeline import Fit
        depth_fit = (track_dir / 'replay3d.npz').exists() and (track_dir / 'model3d.json').exists()
        planar_fit = isinstance(agent.last_fit, Fit) and (track_dir / 'joints.json').exists() and (track_dir / 'replay.mp4').exists()
        if agent.last_fit is None or not (depth_fit or planar_fit):
            raise SystemExit('Appearance completion requires a successful fitted model and replay. Review agent/report.md.')
        if planar_fit and (args.sam3d or args.video_texture or args.sam3d_video_texture):
            raise SystemExit('Planar appearance does not support SAM 3D or video texture projection.')
        if planar_fit:
            source = {'depth': 'Planar side-view video', 'metric': 'estimated, not guaranteed',
                      'note': 'The planar fit uses image silhouettes; cross-view depth and possibly scale are assumed.'}
            (track_dir / 'measurement_source.json').write_text(json.dumps(source, indent=2) + '\n')
        from infer_appearance import infer_appearance
        infer_appearance(track_dir, video, texture=args.video_texture or args.sam3d_video_texture)
        if args.sam3d:
            command = [sys.executable, str(ROOT / "scripts/sam3d_meshes.py"), str(track_dir),
                       "--video", str(video), "--capture", str(capture)]
            if args.sam3d_video_texture:
                command.append("--video-texture")
            if args.sam3d_retries:
                command += ["--retries", str(args.sam3d_retries)]
            result = subprocess.run(command, check=False)
            state = {"exit_code": result.returncode, "state": "completed" if result.returncode == 0 else "failed"}
            (track_dir / "sam3d_status.json").write_text(json.dumps(state, indent=2) + "\n")
            if result.returncode:
                print("SAM 3D failed; keeping the fitted appearance. See the error above and quality.json.", flush=True)
        from video2urdf.assessment import assess
        assess(track_dir)

if __name__ == "__main__":
    main()
