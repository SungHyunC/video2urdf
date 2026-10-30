"""A turntable of the measured model: the camera circles it while its joints replay the values
measured from the video (replay3d.npz), on a floor under a shadow-casting light.

Usage: .venv/bin/python scripts/turntable.py data/tracks_agent/laptop_lidar [--seconds 8] [--out turntable.mp4]
"""

import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from physics_demo import H, W, staged  # noqa: E402

FPS = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", action="store_true",
                    help="side by side: the measured meshes | the visual the model shows (Spark's design or SAM 3D)")
    args = ap.parse_args()
    model = staged(args.run / "model.xml")
    data = mujoco.MjData(model)
    rec = np.load(args.run / "replay3d.npz")
    qs = {k[2:]: rec[k] for k in rec.files if k.startswith("q_")}
    n = len(next(iter(qs.values()))) if qs else 1
    mujoco.mj_forward(model, data)
    meshes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
    pts = np.vstack([data.geom_xpos[g] + np.array([[-1], [1]]) * model.geom_rbound[g] for g in meshes])
    size = float(np.max(np.ptp(pts, axis=0)))  # the object's extent (not the floor's)
    renderer = mujoco.Renderer(model, H, W // 2 if args.compare else W)
    looks = [None]
    if args.compare:
        measured, shown = mujoco.MjvOption(), mujoco.MjvOption()
        floor = next(g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE)
        model.geom_group[floor] = 2  # the floor in both views
        measured.geomgroup[:] = [0, 0, 1, 1, 0, 0]
        shown.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        looks = [measured, shown]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = pts.mean(axis=0)
    cam.distance = 1.15 * size
    cam.elevation = -22
    out = args.out or args.run / "turntable.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
    frames = int(args.seconds * FPS)
    for i in range(frames):
        t = int(i / max(1, frames - 1) * (n - 1))
        for j, q in qs.items():
            data.joint(j).qpos[0] = q[t]
        mujoco.mj_forward(model, data)
        cam.azimuth = 200 + 120 * np.sin(np.pi * i / frames)  # sweep around the front and back
        tiles = []
        for look in looks:
            renderer.update_scene(data, camera=cam, scene_option=look) if look else renderer.update_scene(data, camera=cam)
            tiles.append(renderer.render()[..., ::-1].copy())
        if args.compare:
            for img, text in zip(tiles, ("measured from the video", "what it is: Spark's design / SAM 3D")):
                cv2.putText(img, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (40, 40, 40), 3)
        writer.write(np.hstack(tiles))
    writer.release()
    print(f"turntable -> {out} ({frames} frames)")


if __name__ == "__main__":
    main()
