"""Score a model fitted on estimated depth (MapAnything from the plain RGB video) against the
LiDAR fit of the same capture: link sizes, base thickness, hinge axis direction and position
and the opening frame by frame (hinged plates); box size (static objects); travel and box sizes
(drawers). Both runs must be fitted on the same video frames, e.g. the LiDAR run's own SAM
tracks refitted on a MapAnything capture of that capture's rgb.mp4 (scripts/refit_lidar.py), so
tracking is not a variable.

Usage:
  .venv/bin/python scripts/eval_plain_video.py REF_RUN TEST_RUN [--lidar CAPTURE]
  .venv/bin/python scripts/eval_plain_video.py --cases CASES.json [--md OUT.md] [--json OUT.json]

CASES.json: [{"object": "laptop", "variant": "rgb only", "ref": RUN, "test": RUN, "lidar": CAPTURE}, ...]
(--lidar, the LiDAR capture, brings the test's hinge axis into the reference's camera frame with
ARKit's poses when the two fits picked different reference frames.)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def kind_of(run: Path) -> str:
    if (run / "slides.json").exists():
        return "slide"
    if (run / "static.json").exists():
        return "static"
    if (run / "hinge.json").exists():
        return "hinge"
    raise ValueError(f"{run}: no hinge.json, static.json or slides.json")


def absolute_opening(hinge: dict) -> np.ndarray:
    """Opening per frame in degrees (0 = shut), from the opening on the reference frame and the
    signed turn q from it; the sign is the one that reproduces the fit's own opening range."""
    q = np.rad2deg(np.array([np.nan if v is None else v for v in hinge["q"]], float))
    ref, (lo, hi) = hinge["opening_on_reference_deg"], hinge["opening_range_deg"]
    best = None
    for s in (1.0, -1.0):
        o = ref + s * q
        err = abs(np.nanmin(o) - lo) + abs(np.nanmax(o) - hi)
        if best is None or err < best[0]:
            best = (err, o)
    return best[1]


def axis_in_ref_camera(test: dict, ref_frame: int, lidar: Path | None):
    """The test hinge's axis and point in the reference fit's reference camera."""
    axis, point = np.array(test["axis"]), np.array(test["point"])
    if test["reference_frame"] == ref_frame:
        return axis, point
    if lidar is None:
        return None, None
    from video2urdf import stray
    n = max(test["reference_frame"], ref_frame) + 1
    world = stray.ref_to_frame_poses(lidar, n, ref_frame)  # ref camera -> frame t camera (ARKit)
    inv = np.linalg.inv(world[test["reference_frame"]])
    return inv[:3, :3] @ axis, inv[:3, :3] @ point + inv[:3, 3]


def score_hinge(ref: Path, test: Path, lidar: Path | None) -> dict:
    hr, ht = (json.loads((p / "hinge.json").read_text()) for p in (ref, test))
    mr, mt = (json.loads((p / "model3d.json").read_text()) for p in (ref, test))
    out = {"kind": "hinge", "reference_frame": [hr["reference_frame"], ht["reference_frame"]]}
    pr, pt = mr["plate_size_mm"], mt["plate_size_mm"]
    out["plate_mm"] = [round(x, 1) for x in pt]
    out["plate_ref_mm"] = [round(x, 1) for x in pr]
    out["plate_err_mm"] = [round(a - b, 1) for a, b in zip(pt, pr)]
    out["plate_err_pct"] = [round(100 * (a - b) / b, 1) for a, b in zip(pt, pr)]
    out["base_thickness_mm"] = round(mt["base_thickness_mm"], 1)
    out["base_thickness_ref_mm"] = round(mr["base_thickness_mm"], 1)
    out["base_thickness_measured"] = mt.get("base_thickness_measured")
    out["plate_thickness_mm"] = round(mt["plate_thickness_mm"], 1)
    out["plate_thickness_ref_mm"] = round(mr["plate_thickness_mm"], 1)
    axis, point = axis_in_ref_camera(ht, hr["reference_frame"], lidar)
    if axis is not None:
        ua, ub = axis / np.linalg.norm(axis), np.array(hr["axis"])
        out["axis_err_deg"] = round(float(np.rad2deg(np.arccos(np.clip(abs(ua @ ub), 0, 1)))), 2)
        d = point - np.array(hr["point"])
        out["axis_offset_mm"] = round(1000 * float(np.linalg.norm(d - (d @ ub) * ub)), 1)
    oa, ob = absolute_opening(ht), absolute_opening(hr)
    k = min(len(oa), len(ob))
    diff = np.abs(oa[:k] - ob[:k])
    ok = np.isfinite(diff)
    out["opening_range_deg"] = [round(float(np.nanmin(oa)), 1), round(float(np.nanmax(oa)), 1)]
    out["opening_range_ref_deg"] = [round(float(np.nanmin(ob)), 1), round(float(np.nanmax(ob)), 1)]
    out["opening_err_deg_median"] = round(float(np.median(diff[ok])), 1) if ok.any() else None
    out["opening_err_deg_p90"] = round(float(np.percentile(diff[ok], 90)), 1) if ok.any() else None
    out["frames_compared"] = int(ok.sum())
    out["frames_with_plate"] = [hr["frames_with_plate"], ht["frames_with_plate"]]
    out["hinge_range_deg"] = [round(x, 1) for x in mt["hinge_range_deg"]]
    out["hinge_range_ref_deg"] = [round(x, 1) for x in mr["hinge_range_deg"]]
    out["mask_coverage"] = {k: round(v, 2) for k, v in (mt.get("mask_coverage") or {}).items()}
    return out


def score_static(ref: Path, test: Path) -> dict:
    mr, mt = (json.loads((p / "model3d.json").read_text()) for p in (ref, test))
    br, bt = mr["observed_box_size_mm"], mt["observed_box_size_mm"]
    return {"kind": "static", "box_mm": [round(x, 1) for x in bt], "box_ref_mm": [round(x, 1) for x in br],
            "box_err_mm": [round(a - b, 1) for a, b in zip(bt, br)],
            "box_err_pct": [round(100 * (a - b) / b, 1) for a, b in zip(bt, br)],
            "mask_coverage": {k: round(v, 2) for k, v in (mt.get("mask_coverage") or {}).items()}}


def score_slide(ref: Path, test: Path) -> dict:
    sr, st = (json.loads((p / "slides.json").read_text()) for p in (ref, test))
    out = {"kind": "slide", "joints": []}
    for a, b in zip(st["joints"], sr["joints"]):
        out["joints"].append({"part": a["part"], "travel_mm": round(a["travel_mm"], 1),
                              "travel_ref_mm": round(b["travel_mm"], 1),
                              "travel_err_mm": round(a["travel_mm"] - b["travel_mm"], 1)})
    for a, b in zip(st["drawers"], sr["drawers"]):
        if a["reference_frame"] == b["reference_frame"]:
            ua, ub = np.array(a["axis"]), np.array(b["axis"])
            out.setdefault("axis_err_deg", []).append(
                round(float(np.rad2deg(np.arccos(np.clip(abs(ua @ ub), 0, 1)))), 2))
    out["box_mm"] = st["box_sizes_mm"]
    out["box_ref_mm"] = sr["box_sizes_mm"]
    mt = json.loads((test / "model3d.json").read_text())
    out["mask_coverage"] = {k: round(v, 2) for k, v in (mt.get("mask_coverage") or {}).items()}
    return out


def score(ref: Path, test: Path, lidar: Path | None = None) -> dict:
    kind = kind_of(ref)
    if kind_of(test) != kind:
        raise ValueError(f"{test} is a {kind_of(test)} model, {ref} a {kind}")
    if kind == "hinge":
        return score_hinge(ref, test, lidar)
    return score_static(ref, test) if kind == "static" else score_slide(ref, test)


def row(case: dict, s: dict) -> str:
    name = f"{case['object']} | {case['variant']}"
    if s["kind"] == "hinge":
        cov = " / ".join(f"{100 * v:.0f}%" for v in s["mask_coverage"].values())
        return (f"| {name} | {s['plate_mm'][0]:.0f} x {s['plate_mm'][1]:.0f} ({s['plate_err_pct'][0]:+.0f}% / "
                f"{s['plate_err_pct'][1]:+.0f}%) | {s['base_thickness_mm']:.0f} | {s.get('axis_err_deg', 'n/a')} | "
                f"{s.get('axis_offset_mm', 'n/a')} | {s['opening_range_deg'][0]:.0f}..{s['opening_range_deg'][1]:.0f} | "
                f"{s['opening_err_deg_median']} / {s['opening_err_deg_p90']} | {cov} |")
    if s["kind"] == "static":
        b = " x ".join(f"{x:.0f}" for x in s["box_mm"])
        e = " / ".join(f"{x:+.0f}%" for x in s["box_err_pct"])
        cov = " / ".join(f"{100 * v:.0f}%" for v in s["mask_coverage"].values())
        return f"| {name} | box {b} ({e}) | | | | | | {cov} |"
    j = s["joints"][0]
    cov = " / ".join(f"{100 * v:.0f}%" for v in s["mask_coverage"].values() if v is not None)
    return (f"| {name} | travel {j['travel_mm']:.0f} ({j['travel_err_mm']:+.0f} mm) | | "
            f"{(s.get('axis_err_deg') or ['n/a'])[0]} | | | | {cov} |")


HEADER = ("| object | variant | lid / box / travel (vs LiDAR) | base thickness mm | axis err deg | axis offset mm | "
          "opening range deg | opening err median / p90 deg | replay coverage |\n|---|---|---|---|---|---|---|---|---|")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", type=Path, nargs="?")
    ap.add_argument("test", type=Path, nargs="?")
    ap.add_argument("--lidar", type=Path)
    ap.add_argument("--cases", type=Path)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if args.cases:
        cases = json.loads(args.cases.read_text())
    elif args.ref and args.test:
        cases = [{"object": args.ref.name, "variant": args.test.name, "ref": str(args.ref), "test": str(args.test),
                  "lidar": str(args.lidar) if args.lidar else None}]
    else:
        ap.error("give REF_RUN TEST_RUN or --cases")
    results, lines = [], [HEADER]
    for case in cases:
        try:
            s = score(Path(case["ref"]), Path(case["test"]), Path(case["lidar"]) if case.get("lidar") else None)
        except (OSError, ValueError, KeyError) as error:
            s = {"kind": "error", "error": f"{type(error).__name__}: {error}"}
            lines.append(f"| {case['object']} | {case['variant']} | failed: {s['error'][:120]} | | | | | | |")
        else:
            lines.append(row(case, s))
        results.append({**case, **s})
    table = "\n".join(lines)
    print(table)
    print(json.dumps(results, indent=1))
    if args.md:
        args.md.write_text(table + "\n")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
