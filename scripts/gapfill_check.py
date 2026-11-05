"""Check silhouette gap filling (video2urdf/gapfill.py, replay3d's ``gap_fill``) on a published
3D run before using it: hide measured frames like the gaps (in blocks inside the fast motion,
plus random ones), fill them by silhouette and by interpolation, compare both with the measured
value; on the real gap frames, compare the silhouette IoU of both. Writes OUT/check.json.

Result (11/2, laptop_lidar): on the real gaps the silhouette fill matches the video far better
(object IoU 0.70 -> 0.90, lid IoU 0.25 -> 0.68), but on held-out measured frames it is further
from the measured value than interpolation (median 12 vs 0.7 deg): at open angles the
silhouette's best angle sits ~14 deg below the depth-measured one even on still frames, and
some measured frames in the fast closing are themselves off the video. So replay3d keeps
interpolating by default.

Usage:
  .venv/bin/python scripts/gapfill_check.py check laptop_lidar --out data/agentR_gapfill/check/laptop_lidar
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf import gapfill  # noqa: E402
from video2urdf.lift3d import Intrinsics  # noqa: E402
from video2urdf.pipeline import load_parts  # noqa: E402

# Each published 3D tab with unmeasured joint frames: its run, video, the moving joint and body,
# the tracks that show the moving part (its silhouette), and the replay's scored masks.
MODELS = {
    "laptop_lidar": dict(run="data/tracks_design/laptop_lidar", video="data/real/laptop_lidar/laptop_lidar.mp4",
                         joint="lid_joint", body="lid", parts=["plate", "plate_closed"], kind="hinge"),
    "microwave_lidar": dict(run="data/agentH/microwave", video="data/real/microwave_lidar/microwave_lidar.mp4",
                            joint="lid_joint", body="lid", parts=["plate", "plate_closed"], kind="hinge"),
    "lamp_lidar": dict(run="data/tracks_agent2/stand_lidar", video="data/real/stand_lidar/stand_lidar.mp4",
                       joint="rod2_joint", body="rod2", parts=["rod2"], kind="chain"),
    "drawer": dict(run="data/tracks_design/drawer2_lidar", video="data/real/drawer2_lidar/drawer2_lidar.mp4",
                   joint="drawer1_joint", body="drawer1", parts=["drawer_front"], kind="slides"),
}


def measured_q(run: Path, joint: str) -> np.ndarray:
    """The joint's measured values (NaN where unmeasured), as export_viewer publishes them."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from export_viewer import joint_tracks
    return np.array([np.nan if v is None else v for v in joint_tracks(run)[joint]], float)


def render_model(run: Path) -> Path:
    """The measured model the replay draws (before shape design swapped the visuals)."""
    pre = run / "model.predesign.xml"
    return pre if pre.exists() else run / "model.xml"


class Setup:
    def __init__(self, name: str, run: Path | None = None, scale: int = 4):
        cfg = MODELS[name]
        self.name, self.cfg = name, cfg
        self.run = Path(run or ROOT / cfg["run"])
        self.video = ROOT / cfg["video"]
        z = np.load(self.run / "replay3d.npz")
        self.cam_to_world = z["cam_to_world"]
        self.intr = Intrinsics(float(z["f"]), float(z["cx"]), float(z["cy"]))
        cap = cv2.VideoCapture(str(self.video))
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        _, by = load_parts(self.run)
        w, h = self.size
        self.by = {k: {t: d for t, d in v.items() if d.usable(w, h)} for k, v in by.items()}
        self.q = measured_q(self.run, cfg["joint"])
        n = len(self.cam_to_world)
        self.q = np.concatenate([self.q, np.full(max(0, n - len(self.q)), np.nan)])[:n]
        self.r = gapfill.SilhouetteRenderer(render_model(self.run), self.cam_to_world, self.intr, self.size, scale)
        jid = self.r.model.joint(cfg["joint"]).id
        self.lo, self.hi = (float(v) for v in self.r.model.jnt_range[jid])
        self.bodies = self.r.subtree(cfg["body"])
        self.scorer = gapfill.joint_scorer(self.r, cfg["joint"], self.bodies, self.target)

    def target(self, t: int):
        w, h = self.size
        m = None
        for p in self.cfg["parts"]:
            if t in self.by.get(p, {}):
                d = self.by[p][t].full_mask(h, w)
                m = d if m is None else m | d
        return m

    def object_target(self, t: int):
        """All tracked parts but the floor/table (for the whole-object IoU)."""
        w, h = self.size
        m = np.zeros((h, w), bool)
        for p, dets in self.by.items():
            if p in ("floor", "table") or t not in dets:
                continue
            m |= dets[t].full_mask(h, w)
        return m

    def object_iou(self, t: int, v: float) -> tuple[float, float]:
        """(object IoU, moving-part IoU) at value v on frame t."""
        img = self.r.bodies(t, {self.cfg["joint"]: v})
        obj = self.r.shrink(self.object_target(t))
        part_m = self.target(t)
        part = gapfill.iou(np.isin(img, self.bodies), self.r.shrink(part_m)) if part_m is not None else np.nan
        return gapfill.iou(img >= 0, obj), part


def holdout_blocks(q: np.ndarray, rng, lengths=(1, 5, 10, 20, 40), n_random: int = 40) -> list[list[int]]:
    """Blocks of measured frames to hide: inside the fast motion (where the joint moves most
    per frame, like the real gaps) at several lengths, and random ones anywhere."""
    known = np.flatnonzero(np.isfinite(q))
    speed = np.zeros(len(q))
    qi = gapfill.interpolate(q)
    speed[1:-1] = np.abs(qi[2:] - qi[:-2]) / 2
    fast = known[speed[known] >= np.percentile(speed[known], 85)]
    blocks = []
    for L in lengths:
        for c in rng.choice(fast, size=min(len(fast), 12), replace=False):
            blocks.append([t for t in range(int(c) - L // 2, int(c) - L // 2 + L) if 0 < t < len(q) - 1 and np.isfinite(q[t])])
    for c in rng.choice(known[1:-1], size=n_random, replace=False):
        L = int(rng.choice(lengths))
        blocks.append([t for t in range(int(c), int(c) + L) if t < len(q) - 1 and np.isfinite(q[t])])
    return [b for b in blocks if b]


def degrees(name, v):
    return np.rad2deg(v) if MODELS[name]["kind"] != "slides" else 1000 * v  # mm for slides


def check(name: str, out: Path, seed: int = 0):
    s = Setup(name)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    unit = "mm" if s.cfg["kind"] == "slides" else "deg"
    t0 = time.time()
    # 1. hold-out
    rows = []
    for block in holdout_blocks(s.q, rng):
        q = s.q.copy()
        q[block] = np.nan
        prior = gapfill.interpolate(q)
        for first, last in gapfill.gap_runs(np.isnan(q)):
            if not (first <= block[0] <= last):
                continue
            half = gapfill.window(q, first, last, s.lo, s.hi, gapfill.Settings())
            for t in block:
                sc = s.scorer(t)
                f = (gapfill.search(sc, float(prior[t]), half, s.lo, s.hi, gapfill.Settings()) if sc is not None
                     else gapfill.FrameFill(float(prior[t]), gapfill.INTERPOLATED, "too few silhouette pixels"))
                rows.append({"frame": t, "block_len": len(block), "first": block[0], "truth": float(s.q[t]),
                             "interp": float(prior[t]), "fill": f.value, "source": f.source})
    truth = np.array([r["truth"] for r in rows])
    e_i = np.abs(degrees(name, np.array([r["interp"] for r in rows]) - truth))
    e_f = np.abs(degrees(name, np.array([r["fill"] for r in rows]) - truth))
    used = np.array([r["source"] == gapfill.SILHOUETTE for r in rows])

    def stats(e):
        return {"n": int(len(e)), "median": float(np.median(e)) if len(e) else None,
                "p90": float(np.percentile(e, 90)) if len(e) else None, "mean": float(np.mean(e)) if len(e) else None}

    by_len = {}
    for L in sorted({r["block_len"] for r in rows}):
        k = np.array([r["block_len"] == L for r in rows])
        by_len[str(L)] = {"interp": stats(e_i[k]), "fill": stats(e_f[k]), "silhouette_share": float(used[k].mean())}
    hold = {"unit": unit, "frames": int(len(rows)), "unique_frames": int(len({r["frame"] for r in rows})),
            "silhouette_share": float(used.mean()), "interp": stats(e_i), "fill": stats(e_f),
            "fill_on_silhouette_frames": stats(e_f[used]), "interp_on_silhouette_frames": stats(e_i[used]),
            "by_block_length": by_len}
    print(json.dumps(hold, indent=1))
    # 2. the real gaps
    fill = gapfill.fill(s.q, s.scorer, s.lo, s.hi)
    prior = gapfill.interpolate(s.q)
    gaps = []
    for t, f in sorted(fill.frames.items()):
        oi, pi = s.object_iou(t, prior[t])
        of, pf = s.object_iou(t, f.value)
        gaps.append({"frame": t, "interp": float(degrees(name, prior[t])), "fill": float(degrees(name, f.value)),
                     "source": f.source, "reason": f.reason, "object_iou_interp": oi, "object_iou_fill": of,
                     "part_iou_interp": pi, "part_iou_fill": pf})
    g = lambda k: np.array([x[k] for x in gaps], float)
    real = {"frames": len(gaps), "silhouette": int(sum(x["source"] == gapfill.SILHOUETTE for x in gaps)),
            "object_iou_interp_mean": float(np.nanmean(g("object_iou_interp"))),
            "object_iou_fill_mean": float(np.nanmean(g("object_iou_fill"))),
            "part_iou_interp_mean": float(np.nanmean(g("part_iou_interp"))) if np.isfinite(g("part_iou_interp")).any() else None,
            "part_iou_fill_mean": float(np.nanmean(g("part_iou_fill"))) if np.isfinite(g("part_iou_fill")).any() else None,
            "object_iou_worse_frames": int((g("object_iou_fill") < g("object_iou_interp") - 0.01).sum()),
            "max_change": float(np.max(np.abs(g("fill") - g("interp")))) if gaps else 0.0}
    print(json.dumps(real, indent=1))
    (out / "check.json").write_text(json.dumps({"model": name, "holdout": hold, "gaps": real,
                                                "holdout_rows": rows, "gap_rows": gaps,
                                                "seconds": time.time() - t0}, indent=1))
    return s, fill


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check"])
    ap.add_argument("name", choices=sorted(MODELS))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    check(a.name, a.out or ROOT / "data" / "agentR_gapfill" / "check" / a.name)
