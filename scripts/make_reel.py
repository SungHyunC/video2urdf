"""Demo reel for one run: the phone video next to its MuJoCo replay, then the robot demo.

Usage: .venv/bin/python scripts/make_reel.py data/real/notebook_2.MOV data/tracks/notebook_2 [--out reel.mp4]
Run scripts/robot_demo.py on the run first.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1920, 1080, 30
BG = (250, 250, 250)
INK = (30, 30, 30)


def frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, img = cap.read()
        if not ok:
            break
        yield img
    cap.release()


def fit_height(img: np.ndarray, h: int) -> np.ndarray:
    return cv2.resize(img, (round(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)


def caption(canvas: np.ndarray, text: str, x: int, y: int, scale: float = 1.0):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, INK, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or (args.run / "reel.mp4")
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))

    joints = json.loads((args.run / "joints.json").read_text())["joints"]
    summary = "; ".join(
        f"{j['parent']} -> {j['child']}: {j['type']} "
        + (f"{np.rad2deg(min(v for v in j['q'] if v is not None)):.0f}..{np.rad2deg(max(v for v in j['q'] if v is not None)):.0f} deg"
           if j["type"] == "revolute" else "")
        for j in joints)

    # Part 1: phone video | measured model replayed over it.
    panel_h = H - 170
    for src, rep in zip(frames(args.video), frames(args.run / "replay.mp4")):
        canvas = np.full((H, W, 3), BG, np.uint8)
        a, b = fit_height(src, panel_h), fit_height(rep, panel_h)
        gap = 60
        x0 = (W - a.shape[1] - b.shape[1] - gap) // 2
        canvas[110:110 + panel_h, x0:x0 + a.shape[1]] = a
        canvas[110:110 + panel_h, x0 + a.shape[1] + gap:x0 + a.shape[1] + gap + b.shape[1]] = b
        caption(canvas, "1. Phone video", x0, 80)
        caption(canvas, "2. Measured joint model, replayed in MuJoCo", x0 + a.shape[1] + gap, 80)
        caption(canvas, f"Muse Spark + SAM 3.1 measured: {summary}", x0, H - 25, 0.8)
        writer.write(canvas)

    # Part 2: a robot uses the measured model.
    for img in frames(args.run / "robot_demo.mp4"):
        canvas = cv2.resize(img, (W, H))
        cv2.rectangle(canvas, (0, H - 70), (W, H), BG, -1)
        caption(canvas, "3. A Franka Panda closes it in MuJoCo physics, using the measured hinge", 40, H - 25, 1.0)
        writer.write(canvas)
    writer.release()
    print(out)


if __name__ == "__main__":
    main()
