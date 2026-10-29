"""Give an already fitted model its physics: Muse Spark watches the video and judges each
link's material, mass and friction and when a moving part stood still with no hand on it;
physics.apply measures joint friction from those spans and writes it all into the model.

Usage: .venv/bin/python scripts/add_physics.py data/tracks_agent/laptop_lidar [--video path] [--reuse]
(--reuse applies the estimate saved in physics_estimate.json again, without asking Spark.)

The agent does the same through its set_physics tool; this is for runs made before it had one.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from openai import OpenAI  # noqa: E402

from video2urdf import physics  # noqa: E402
from video2urdf.agent import API_KEY_ENV, BASE_URL, MODEL, mp4_url  # noqa: E402
from video2urdf.pipeline import video_info  # noqa: E402

SCHEMA = {"type": "object", "properties": {
    "parts": {"type": "array", "items": {"type": "object", "properties": {
        "link": {"type": "string"}, "material": {"type": "string"}, "mass_kg": {"type": "number"},
        "friction": {"type": "number"}, "why": {"type": "string"}},
        "required": ["link", "material", "mass_kg", "friction", "why"]}},
    "released": {"type": "array", "items": {"type": "object", "properties": {
        "joint": {"type": "string"}, "start_s": {"type": "number"}, "end_s": {"type": "number"}},
        "required": ["joint", "start_s", "end_s"]}}},
    "required": ["parts", "released"]}

PROMPT = """This video shows an articulated object. Its kinematic model has been measured from the
video: links (with their measured bounding box in mm and the volume of their measured solid
geometry) and joints, listed below. Give each link its material, its real mass in kg (judge
from what the object is: e.g. a 13-inch laptop weighs about 1.4 kg in all), and the sliding
friction coefficient of its surface, with a short reason. Then list every time span (seconds)
in which a moving part stands still and NO hand touches it, per joint.

Links: {links}
Joints: {joints}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track_dir", type=Path)
    ap.add_argument("--video", type=Path, help="default: <name>.mp4 or <name>.* in data/real")
    ap.add_argument("--reuse", action="store_true", help="apply the saved physics_estimate.json again")
    args = ap.parse_args()
    video = args.video or find_video(args.track_dir)
    _, n, fps = video_info(video)
    saved = args.track_dir / "physics_estimate.json"
    if args.reuse:
        est = json.loads(saved.read_text())
    else:
        xml = args.track_dir / ("model.nophysics.xml" if (args.track_dir / "model.nophysics.xml").exists() else "model.xml")
        sizes = physics.link_sizes(xml)
        joints = {k: t for k, (t, _) in physics._joints(xml).items()}
        client = OpenAI(base_url=BASE_URL, api_key=os.environ[API_KEY_ENV], timeout=600)
        r = client.chat.completions.create(
            model=MODEL, max_tokens=6000,
            response_format={"type": "json_schema", "json_schema": {"name": "physics", "schema": SCHEMA}},
            messages=[{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": mp4_url(video)}},
                {"type": "text", "text": PROMPT.format(links=json.dumps(sizes), joints=json.dumps(joints))}]}])
        est = {"model": MODEL, **json.loads(r.choices[0].message.content)}
        saved.write_text(json.dumps(est, indent=2))
        print(json.dumps(est, indent=2))
    parts = {p["link"]: {k: p[k] for k in ("material", "mass_kg", "friction", "why")} for p in est["parts"]}
    windows: dict[str, list] = {}
    for s in est["released"]:
        name = s["joint"] if s["joint"].endswith("_joint") else f"{s['joint']}_joint"
        windows.setdefault(name, []).append((int(s["start_s"] * fps), int(s["end_s"] * fps)))
    physics.apply(args.track_dir, parts, windows, fps)


def find_video(track_dir: Path) -> Path:
    for p in [ROOT / "data" / "real" / track_dir.name / f"{track_dir.name}.mp4",
              *sorted((ROOT / "data" / "real").glob(f"{track_dir.name}.*"))]:
        if p.is_file():
            return p
    raise SystemExit(f"no video found for {track_dir.name}; pass --video")


if __name__ == "__main__":
    main()
