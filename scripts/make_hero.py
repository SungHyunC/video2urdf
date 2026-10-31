"""The README's hero GIF from one published viewer model: the phone video and the measured model
replayed over it side by side, then the simulated robot using the model.

Usage: .venv/bin/python scripts/make_hero.py [--model docs/models/laptop_lidar] [--out docs/media/hero.gif]

Reads only the viewer's published videos (phone.mp4, replay.mp4, robot.mp4; phone and replay share
frame numbers). Shot A plays the lid closing in the phone video next to its replay; shot B plays the
Franka Panda closing the measured laptop. 960 x 540, 10 fps, a 128-colour palette (about 4.7 MB).
The captions' numbers are the laptop's (docs/models/laptop_lidar/model.json): change them with the
model.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
W, H = 960, 540
BG = (13, 17, 23)
WHITE, GRAY, DIM = (240, 243, 246), (150, 158, 168), (88, 96, 108)
ACCENT = (56, 139, 253)
STEPS = ["Phone video", "Measured model, replayed over it", "Robot uses the model"]
LEFT = 'iPhone LiDAR video of a 13" laptop'
RIGHT = "Measured URDF, drawn from the phone's pose"
FOOT = ["Lid measured at 309 × 205 mm (spec 304 × 212), opening 0–117°",
        "A Muse Spark agent prompts SAM 3.1, fits the hinge, and checks the replay over the video",
        "Every frame: yellow = lid, purple = base, both from the URDF"]
ROBOT = ["Franka Panda in MuJoCo physics", "closes the measured laptop by its lid's edge"]


def font(size: int, index: int):
    """Helvetica Neue on macOS (index 0 regular, 1 bold, 10 medium); PIL's own font elsewhere."""
    try:
        return ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", size, index=index)
    except OSError:
        return ImageFont.load_default(size)


F_HEAD, F_NUM, F_LABEL, F_SMALL = font(19, 10), font(15, 1), font(18, 10), font(15, 0)


def header(d: ImageDraw.ImageDraw, active: set[int]):
    """The three steps in one row, the current ones highlighted."""
    d.rectangle([0, 0, W, 46], fill=BG)
    chip, gap, arrow = 22, 8, 34
    widths = [d.textlength(s, font=F_HEAD) for s in STEPS]
    x = (W - sum(chip + gap + w for w in widths) - arrow * (len(STEPS) - 1)) / 2
    for i, (s, w) in enumerate(zip(STEPS, widths)):
        on = i in active
        d.rounded_rectangle([x, 12, x + chip, 12 + chip], radius=5, fill=ACCENT if on else DIM)
        d.text((x + chip / 2, 23), str(i + 1), font=F_NUM, fill=WHITE, anchor="mm")
        x += chip + gap
        d.text((x, 23), s, font=F_HEAD, fill=WHITE if on else GRAY, anchor="lm")
        x += w
        if i < len(STEPS) - 1:
            cx = x + arrow / 2
            d.line([cx - 8, 23, cx + 6, 23], fill=GRAY, width=2)
            d.polygon([(cx + 9, 23), (cx + 2, 18), (cx + 2, 28)], fill=GRAY)
            x += arrow


def rate(path: Path) -> tuple[float, int]:
    """Average frame rate (frames / duration: the viewer's videos can carry a nominal rate) and count."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True).stdout)
    return n / dur, n


def pick(path: Path, t0: float, t1: float, speed: float, fps: float) -> list[np.ndarray]:
    """Frames from t0 to t1 s of the source, played ``speed`` x faster at ``fps``."""
    src_fps, n = rate(path)
    want = [min(n - 1, int(round(t0 * src_fps + k * src_fps * speed / fps)))
            for k in range(int(round((t1 - t0) / speed * fps)))]
    cap, i, got = cv2.VideoCapture(str(path)), 0, {}
    while i <= max(want):
        ok, img = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = img
        i += 1
    cap.release()
    return [got[i] for i in want if i in got]


def pil(img: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT / "docs/models/laptop_lidar")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/media/hero.gif")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--a", type=float, nargs=2, default=[3.2, 6.6], help="shot A source window, s (the lid closing)")
    ap.add_argument("--a-speed", type=float, default=1.5)
    ap.add_argument("--a-hold", type=float, nargs=2, default=[0.5, 0.4], help="hold shot A's first / last frame, s")
    ap.add_argument("--b", type=float, nargs=2, default=[1.0, 8.0], help="shot B source window, s (the robot closing it)")
    ap.add_argument("--b-speed", type=float, default=2.0)
    ap.add_argument("--b-hold", type=float, default=0.9, help="hold the last frame, s")
    ap.add_argument("--colors", type=int, default=128)
    args = ap.parse_args()

    frames: list[Image.Image] = []
    phone = pick(args.model / "phone.mp4", *args.a, args.a_speed, args.fps)
    replay = pick(args.model / "replay.mp4", *args.a, args.a_speed, args.fps)
    pw, ph = 476, 357
    for a, b in zip(phone, replay):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        header(d, {0, 1})
        im.paste(pil(a, (pw, ph)), (2, 84))
        im.paste(pil(b, (pw, ph)), (484, 84))
        d.text((2 + pw / 2, 66), LEFT, font=F_LABEL, fill=WHITE, anchor="mm")
        d.text((484 + pw / 2, 66), RIGHT, font=F_LABEL, fill=WHITE, anchor="mm")
        d.text((W / 2, 470), FOOT[0], font=F_LABEL, fill=WHITE, anchor="mm")
        d.text((W / 2, 497), FOOT[1], font=F_SMALL, fill=GRAY, anchor="mm")
        d.text((W / 2, 520), FOOT[2], font=F_SMALL, fill=GRAY, anchor="mm")
        frames.append(im)
    frames = [frames[0]] * round(args.a_hold[0] * args.fps) + frames + [frames[-1]] * round(args.a_hold[1] * args.fps)

    for a in pick(args.model / "robot.mp4", *args.b, args.b_speed, args.fps):
        im = pil(a, (W, H))
        d = ImageDraw.Draw(im)
        header(d, {2})
        box = (W - 400, 56, W - 12, 108)  # top right: the floor there is empty all shot
        d.rounded_rectangle(box, radius=8, fill=BG)
        d.text(((box[0] + box[2]) / 2, box[1] + 16), ROBOT[0], font=F_LABEL, fill=WHITE, anchor="mm")
        d.text(((box[0] + box[2]) / 2, box[1] + 37), ROBOT[1], font=F_SMALL, fill=GRAY, anchor="mm")
        frames.append(im)
    frames += [frames[-1]] * round(args.b_hold * args.fps)

    with tempfile.TemporaryDirectory() as tmp:
        for k, im in enumerate(frames):
            im.save(Path(tmp) / f"{k:04d}.png")
        src = ["-framerate", str(args.fps), "-i", str(Path(tmp) / "%04d.png")]
        pal = Path(tmp) / "palette.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-vf",
                        f"palettegen=max_colors={args.colors}:stats_mode=full", str(pal)], check=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-i", str(pal), "-lavfi",
                        "paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle", "-loop", "0", str(args.out)],
                       check=True)
    print(f"{args.out}: {len(frames)} frames, {len(frames) / args.fps:.1f} s, {args.out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
