"""Score the plate-hinge fit on an RBO sequence against motion capture.

RBO dataset: TU Berlin, CC BY 4.0 (see video2urdf/rbo.py). Run scripts/rbo_download.py,
then track the sequence's video (prepared here on first run), then score:

  .venv/bin/python scripts/rbo_eval.py laptop01 --prepare-only
  .venv/bin/python scripts/track_video.py data/rbo/prep/laptop01/laptop01.mp4 \\
      "table" "laptop screen" "laptop keyboard" "laptop" --out data/rbo/tracks
  .venv/bin/python scripts/rbo_eval.py laptop01 --floor table --base laptop_keyboard \\
      --plate laptop_screen --plate-closed laptop --open-indicator laptop_keyboard
Or let the agent track and pick the roles, then score what it chose:
  .venv/bin/python scripts/run_agent.py data/rbo/prep/laptop02 --out data/rbo/agent
  .venv/bin/python scripts/rbo_eval.py laptop02 --tracks data/rbo/agent

Ground truth: the plate's marker body (the one that moves most) traces a circle about the
true hinge, and how far it has swept about it is the true angle. The dataset's own joint
values (from a fitted model) are scored as well. Writes data/rbo/results/<name>.json and <name>.jpg (our hinge red, truth green).
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf import lidar, rbo  # noqa: E402
from video2urdf.lift3d import Intrinsics  # noqa: E402
from video2urdf.pipeline import read_frame  # noqa: E402


def line_gap(p: np.ndarray, a: np.ndarray, q: np.ndarray) -> float:
    """Distance from point q to the line through p along unit a."""
    d = q - p
    return float(np.linalg.norm(d - (d @ a) * a))


def circle_axis(track: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Hinge from one marker's positions (n, 3): they lie on a circle about it. The circle's
    plane normal is the axis and its centre a point on it. Uses positions only, so it doesn't
    depend on the dataset's orientation conventions. Returns axis, point, RMS residual (m)."""
    c = track.mean(axis=0)
    u, s, vt = np.linalg.svd(track - c, full_matrices=False)
    e1, e2, normal = vt
    xy = np.column_stack([(track - c) @ e1, (track - c) @ e2])
    # Algebraic circle fit: x^2 + y^2 + D x + E y + F = 0.
    a = np.column_stack([xy, np.ones(len(xy))])
    d, e, f = np.linalg.lstsq(a, -(xy ** 2).sum(axis=1), rcond=None)[0]
    centre2 = np.array([-d / 2, -e / 2])
    radius = np.sqrt(centre2 @ centre2 - f)
    res = np.sqrt(np.mean((np.linalg.norm(xy - centre2, axis=1) - radius) ** 2 + ((track - c) @ normal) ** 2))
    return normal, c + centre2[0] * e1 + centre2[1] * e2, float(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--floor")
    ap.add_argument("--base")
    ap.add_argument("--plate")
    ap.add_argument("--plate-closed")
    ap.add_argument("--open-indicator")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "rbo")
    ap.add_argument("--tracks", type=Path, help="track folder root (default <data>/tracks)")
    args = ap.parse_args()

    prep = args.data / "prep" / args.name
    video = rbo.prepare(args.data / "x" / args.name, prep)
    if args.prepare_only:
        print(video)
        return
    k = json.loads((prep / "intrinsics.json").read_text())
    intr = Intrinsics(k["f"], k["cx"], k["cy"])
    depth = np.load(prep / "depth.npz")["depth"].astype(np.float32)
    c = lidar.Capture.fixed_camera(video, (args.tracks or args.data / "tracks") / args.name, depth, intr)
    if args.floor:
        roles = lidar.Roles(args.floor, args.base, args.plate, args.plate_closed, args.open_indicator)
    else:  # the roles the agent chose
        roles = lidar.Roles(**json.loads((c.track_dir / "roles.json").read_text()))
    c.set_ref(c.ref_with(roles.base, roles.plate))
    ours = lidar.fit_hinge(c, roles)
    hinge = json.loads((c.track_dir / "hinge.json").read_text())
    axis, point = np.array(hinge["axis"]), np.array(hinge["point"])
    q = np.array([np.nan if v is None else v for v in hinge["q"]])

    # Truth: hinge from the marker bodies, joint values from the dataset.
    gt = rbo.truth(args.data / "x" / args.name)
    bodies = sorted(gt["bodies"].values(), key=lambda p: np.ptp(p[:, :3, 3], axis=0).sum())
    plate_gt = bodies[-1]
    true_axis, true_point, true_res = circle_axis(plate_gt[:, :3, 3])
    # True angle: how far the marker has swept about the true hinge (the dataset's own joint
    # values come from a fitted model and read ~4% short of this on laptop01; scored too).
    v = plate_gt[: c.n, :3, 3] - true_point
    v -= np.outer(v @ true_axis, true_axis)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q_true = np.unwrap(np.arctan2(np.cross(v[c.ref], v) @ true_axis, v @ v[c.ref]))
    joint_name, q_dataset = next(iter(gt["joints"].items()))
    q_dataset = q_dataset[: c.n] - q_dataset[c.ref]

    axis_err = float(np.rad2deg(np.arccos(min(1.0, abs(axis @ true_axis)))))
    # Where along the hinge to measure the offset: the middle of the plate.
    mid = point + ((plate_gt[c.ref][:3, 3] - point) @ axis) * axis
    pos_err = line_gap(true_point, true_axis, mid)
    valid = ~np.isnan(q[: c.n])
    # The circle's normal has no inherent sign, so match the rotation sense to the data.
    def err_to(target):
        errs = [np.rad2deg(np.abs(sg * q[: c.n][valid] - target[valid])) for sg in (1, -1)]
        return min(errs, key=np.mean)
    err, err_dataset = err_to(q_true), err_to(q_dataset)
    result = {
        "sequence": args.name, "joint": joint_name, "frames": c.n, "frames_measured": int(valid.sum()),
        "axis_direction_error_deg": axis_err, "axis_position_error_mm": 1000 * pos_err,
        "angle_error_deg_mean": float(err.mean()), "angle_error_deg_median": float(np.median(err)),
        "angle_error_deg_p90": float(np.percentile(err, 90)),
        "angle_error_vs_dataset_joint_deg_median": float(np.median(err_dataset)),
        "true_range_deg": float(np.rad2deg(np.ptp(q_true))), "measured_range_deg": float(np.rad2deg(np.nanmax(q) - np.nanmin(q))),
        "true_hinge_fit_residual_mm": 1000 * true_res, "ours": ours,
    }
    out = args.data / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(result, indent=2))

    image = read_frame(video, c.ref)
    for a, p, color, label in ((axis, point, (0, 0, 255), "measured"), (true_axis, true_point, (0, 200, 0), "mocap")):
        p = p + ((mid - p) @ a) * a  # centre the drawn segment on the plate
        ends = p + np.outer([-0.15, 0.15], a)
        uv = np.stack([intr.f * ends[:, 0] / ends[:, 2] + intr.cx, intr.f * ends[:, 1] / ends[:, 2] + intr.cy], 1).astype(int)
        cv2.line(image, tuple(uv[0]), tuple(uv[1]), color, 3)
        cv2.putText(image, label, tuple(uv[1] + [5, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(out / f"{args.name}.jpg"), image)
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in result.items() if k != "ours"}, indent=1))


if __name__ == "__main__":
    main()
