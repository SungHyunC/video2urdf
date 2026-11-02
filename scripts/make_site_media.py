"""The site's hero clip and its step-card pictures, all from one published run (docs/models/laptop_lidar).

    python scripts/make_site_media.py [--docs docs]

Writes docs/assets/hero.mp4 + hero_poster.jpg (docs/media/hero.gif as a small looping H.264 clip) and
docs/assets/steps/1-film.jpg ... 7-robot.jpg (640 x 480): the phone video, SAM 3.1's mask on the agent's
segment call, the measured hinge (side view), the textured model, the replay over the video, the friction check
and the robot. Needs ffmpeg/ffprobe.
"""
import argparse
import json
import subprocess
from pathlib import Path

CARD = "scale=640:480:force_original_aspect_ratio=increase,crop=640:480"  # fill the card, crop the edges
FIT = "scale=640:-2,pad=640:480:0:(oh-ih)/2:color=0xeef1f4"            # keep a wide frame whole (its captions)


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def duration(p):
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(p)]))
    return float(j["format"]["duration"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=Path("docs"))
    docs = ap.parse_args().docs
    run = docs / "models" / "laptop_lidar"
    out = docs / "assets" / "steps"
    out.mkdir(parents=True, exist_ok=True)

    def frame(name, frac, dst, pre="", fit=False):
        src = run / f"{name}.mp4"
        vf = (pre + "," if pre else "") + (FIT if fit else CARD)
        ff("-ss", f"{duration(src) * frac:.2f}", "-i", str(src), "-frames:v", "1", "-vf", vf, "-q:v", "5", str(out / dst))

    def image(src, dst, pre=""):
        ff("-i", str(src), "-vf", (pre + "," if pre else "") + CARD, "-q:v", "5", str(out / dst))

    frame("phone", 0.45, "1-film.jpg")
    image(run / "agent" / "step008.jpg", "2-track.jpg")                                      # segment 'laptop screen'
    image(run / "design" / "measured_views.jpg", "3-joint.jpg", pre="crop=402:312:397:0")    # the SIDE panel
    frame("physics", 0.15, "4-model.jpg", pre="crop=iw/2:ih:0:0")                            # left half: with friction
    frame("replay", 0.45, "5-check.jpg")
    frame("physics", 0.75, "6-physics.jpg", fit=True)
    frame("robot", 0.45, "7-robot.jpg", fit=True)

    gif = docs / "media" / "hero.gif"
    ff("-i", str(gif), "-movflags", "+faststart", "-pix_fmt", "yuv420p", "-vf", "scale=960:540:flags=lanczos,fps=20",
       "-c:v", "libx264", "-crf", "24", "-preset", "slow", "-an", str(docs / "assets" / "hero.mp4"))
    ff("-i", str(gif), "-frames:v", "1", "-vf", "scale=960:540", "-q:v", "4", str(docs / "assets" / "hero_poster.jpg"))
    for p in sorted(out.iterdir()) + [docs / "assets" / "hero.mp4", docs / "assets" / "hero_poster.jpg"]:
        print(f"{p}: {p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
