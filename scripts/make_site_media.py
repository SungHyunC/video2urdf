"""Light copies of the published viewer videos for the landing page (docs/assets/).

Usage: .venv/bin/python scripts/make_site_media.py [--docs docs]

Writes, from docs/models/<name>/{phone,replay}.mp4 only (nothing new is filmed or shown):
  docs/assets/compare/<name>_{phone,replay}.{mp4,jpg}  the hero's before/after pair: every other
      frame of both videos, picked by frame index so the two stay frame-aligned, 640 px wide,
      1 s keyframes (cheap loop seeks), no audio, no metadata; the .jpg is the first frame
  docs/assets/cards/<name>.{mp4,jpg}  the showcase card loop: 7 s of the model over the video,
      the window where the joints move most (from model.json's joint tracks), 480 px, 24 fps
  docs/assets/cards/{robot_microwave,policy_drawer}.jpg  posters for the results' robot clips
"""
import argparse
import json
import subprocess
from pathlib import Path

COMPARE = ["laptop_lidar", "drawer", "microwave_lidar", "mouse_lidar"]
CARDS = ["laptop_lidar", "drawer", "microwave_lidar", "lamp_lidar", "mouse_lidar", "airpods_lidar",
         "notebook", "stand"]
CARD_SECONDS = 7.0
POSTERS = [("microwave_lidar/robot.mp4", 0.4, "robot_microwave.jpg"), ("drawer/policy.mp4", 0.6, "policy_drawer.jpg")]


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,nb_read_packets", "-show_entries",
                          "format=duration", "-of", "json", str(path)], capture_output=True, text=True,
                         check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    return {"w": s["width"], "h": s["height"], "frames": int(s["nb_read_packets"]),
            "duration": float(d["format"]["duration"])}


def encode(args: list, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, "-an", "-map_metadata", "-1", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart", str(dst)], check=True)


def poster(video: Path, dst: Path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-frames:v", "1", "-q:v", "4",
                    "-map_metadata", "-1", str(dst)], check=True)


def compare_pair(docs: Path, name: str):
    src = docs / "models" / name
    info = {k: probe(src / f"{k}.mp4") for k in ("phone", "replay")}
    if info["phone"]["frames"] != info["replay"]["frames"]:
        raise SystemExit(f"{name}: phone and replay differ in frame count {info}")
    fps = info["replay"]["frames"] / info["replay"]["duration"] / 2  # every other frame, real time
    for k in ("phone", "replay"):
        dst = docs / "assets" / "compare" / f"{name}_{k}.mp4"
        encode(["-i", str(src / f"{k}.mp4"), "-vf",
                f"select='not(mod(n\\,2))',setpts=N/({fps:.4f}*TB),scale=640:-2:flags=lanczos",
                "-r", f"{fps:.4f}", "-preset", "slow", "-crf", "30", "-g", str(round(fps)),
                "-keyint_min", str(round(fps)), "-sc_threshold", "0"], dst)
        poster(dst, dst.with_suffix(".jpg"))
        print(dst, f"{dst.stat().st_size / 1e3:.0f} kB")


def busiest_window(spec: dict, seconds: float) -> float:
    """Start time of the window where the joints move most (0 for rigid objects)."""
    fps, frames = spec["fps"], spec["frames"]
    tracks = [t for t in spec.get("joints", {}).values() if t]
    if not tracks:
        return 0.0
    motion = [0.0] * frames
    for t in tracks:
        last = None
        for i, v in enumerate(t[:frames]):
            if v is not None and last is not None:
                motion[i] += abs(v - last)
            if v is not None:
                last = v
    n = max(1, int(seconds * fps))
    best, best_i, run = -1.0, 0, sum(motion[:n])
    for i in range(0, max(1, frames - n)):
        if run > best:
            best, best_i = run, i
        run += motion[i + n] - motion[i] if i + n < frames else -motion[i]
    return best_i / fps


def card(docs: Path, name: str):
    src = docs / "models" / name
    spec = json.loads((src / "model.json").read_text())
    info = probe(src / "replay.mp4")
    start = min(busiest_window(spec, CARD_SECONDS), max(0.0, info["duration"] - CARD_SECONDS))
    scale = "scale=480:-2:flags=lanczos" if info["w"] >= info["h"] else "scale=-2:480:flags=lanczos"
    dst = docs / "assets" / "cards" / f"{name}.mp4"
    encode(["-ss", f"{start:.2f}", "-t", f"{CARD_SECONDS}", "-i", str(src / "replay.mp4"), "-vf", f"fps=24,{scale}",
            "-preset", "slow", "-crf", "29", "-g", "48"], dst)
    poster(dst, dst.with_suffix(".jpg"))
    print(dst, f"from {start:.1f} s, {dst.stat().st_size / 1e3:.0f} kB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=Path, default=Path(__file__).resolve().parents[1] / "docs")
    args = ap.parse_args()
    for name in COMPARE:
        compare_pair(args.docs, name)
    for name in CARDS:
        card(args.docs, name)
    for src, frac, name in POSTERS:
        video = args.docs / "models" / src
        dst = args.docs / "assets" / "cards" / name
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{probe(video)['duration'] * frac:.2f}", "-i", str(video),
                        "-frames:v", "1", "-vf", "scale=640:-2:flags=lanczos", "-q:v", "4", "-map_metadata", "-1", str(dst)],
                       check=True)
        print(dst, f"{dst.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
