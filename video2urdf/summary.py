"""What a run produced, in numbers (summary.json) and in words (summary.md): the model's links and
joints, how well the replay covers the video, and which numbers were measured and which estimated."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

KEEP_3D = ("plate_size_mm", "plate_thickness_mm", "base_thickness_mm", "hinge_range_deg", "box_sizes_mm",
           "observed_box_size_mm", "generated_size_mm", "base_height_mm", "visual_hull", "door")


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _vertical_hinge(xml: Path) -> bool:
    """The first joint's axis is vertical in the model's world (a door), as the viewer tells doors."""
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model.njnt > 0 and abs(float(data.xaxis[0][2])) > 0.9
    except Exception:
        return False


def kind_of(d: Path) -> str:
    if (d / "static.json").exists():
        return "rigid (no joints)"
    if (d / "chain.json").exists():
        return "rod chain (3D)"
    if (d / "slides.json").exists() or (d / "slide.json").exists():
        return "drawer (3D)"
    if (d / "hinge.json").exists() and (d / "model3d.json").exists():
        return "door on a vertical hinge (3D)" if _vertical_hinge(d / "model.xml") else "hinged plate (3D)"
    if (d / "joints.json").exists():
        return "planar chain (side view)"
    return "unknown"


def mjcf_joints(xml: Path) -> list[dict]:
    out = []
    for j in ET.parse(xml).getroot().iter("joint"):
        t = j.get("type", "hinge")
        rng = [float(x) for x in (j.get("range") or "0 0").split()]
        rev = t == "hinge"
        out.append({"name": j.get("name"), "type": "revolute" if rev else "prismatic" if t == "slide" else t,
                    "range": [round(math.degrees(x) if rev else 1000 * x, 1) for x in rng],
                    "unit": "deg" if rev else "mm",
                    "axis": [round(float(x), 3) for x in (j.get("axis") or "0 0 1").split()],
                    "friction": float(j.get("frictionloss")) if j.get("frictionloss") else None})
    return out


def model_facts(d: Path) -> dict:
    from .physics import link_sizes
    xml = d / "model.xml"
    facts = {"kind": kind_of(d), "links": {}, "joints": mjcf_joints(xml)}
    try:
        facts["links"] = {k: {"size_mm": v["size_mm"]} for k, v in link_sizes(xml).items()}
    except Exception as error:  # an odd model must not stop the summary
        facts["links_error"] = f"{type(error).__name__}: {error}"
    m3 = _read(d / "model3d.json")
    if m3:
        facts["coverage"] = {k: round(v, 3) for k, v in (m3.get("mask_coverage") or {}).items() if v is not None}
        facts["fit_warnings"] = m3.get("fit_warnings") or []
        facts["measured"] = {k: m3[k] for k in KEEP_3D if k in m3}
        if m3.get("joints"):
            facts["measured"]["joints"] = [{k: v for k, v in j.items() if k in ("name", "part", "child", "type",
                                                                                 "travel_mm", "length_mm")}
                                           for j in m3["joints"] if isinstance(j, dict)]
    elif (d / "joints.json").exists():
        j = _read(d / "joints.json") or {}
        facts["measured"] = {"px_per_m": j.get("px_per_m"),
                             "joints": [{"child": x["child"], "parent": x["parent"], "type": x["type"],
                                         "residual_mm": round(1000 * x["residual_m"], 1)}
                                        for x in j.get("joints", [])]}
        replay = d / "planar_replay.npz"
        if replay.exists():
            with np.load(replay) as z:
                facts["coverage"] = {k[4:]: round(float(np.nanmean(z[k])), 3) for k in z.files
                                     if k.startswith("iou_") and np.isfinite(z[k]).any()}
            facts["coverage_metric"] = "mean replay IoU against the SAM masks"
    if m3:
        facts["coverage_metric"] = "share of each part's SAM mask the replayed model covers"
    phys = _read(d / "physics.json")
    if phys:
        facts["physics"] = {
            "links": {k: {"material": v.get("material"), "mass_kg": v.get("mass_kg")} for k, v in phys["links"].items()},
            "joints": {k: {"friction": round(v.get("friction", 0), 4), "measured": bool(v.get("friction_measured"))}
                       for k, v in phys.get("joints", {}).items()}}
    design = _read(d / "design.json")
    if design:
        facts["design"] = {"object": design.get("object"), "applied": design.get("applied"),
                           "score": (design.get("rounds") or [{}])[max(0, (design.get("best_round") or 1) - 1)].get(
                               "mean_score") if design.get("rounds") else None,
                           "measured_score": design.get("measured_mean_score"), "warnings": design.get("warnings")}
    facts["textured"] = (d / "texture.json").exists()
    for name in ("completion.json", "completion_status.json"):
        c = _read(d / name)
        if c is not None:
            facts["completion"] = c
    facts["measurement_source"] = _read(d / "measurement_source.json")
    return facts


def measured_vs_estimated(s: dict) -> list[str]:
    """Plain lines: what came from the capture and what was judged or assumed."""
    out = []
    kind = (s.get("input") or {}).get("kind")
    path = (s.get("path") or {}).get("chosen")
    m = s.get("model") or {}
    src = m.get("measurement_source") or {}
    fallback = (s.get("fallback") or {}).get("used")
    what = "Geometry" if fallback or not m.get("joints") else "Geometry and joint motion"
    if kind == "stray":
        out.append(f"{what}: measured from iPhone LiDAR depth and ARKit camera poses.")
    elif kind == "rgbd":
        out.append(f"{what}: measured from the RGB-D capture's depth (fixed camera).")
    elif path == "3d":
        scale = ((src.get("auto_scale") or {}).get("method") if isinstance(src, dict) else None)
        how = {"card": "set from an ID-1 card in view", "object_prior": "estimated from the object's usual size "
               "(Muse Spark): an estimate, not a measurement"}.get(scale, "MapAnything's own (can be several times off)")
        out.append(f"{what}: from depth that MapAnything estimated from RGB video; metric scale {how}. "
                   "Sizes are estimates.")
    elif path == "planar":
        metric = src.get("metric", "") if isinstance(src, dict) else ""
        first = ("Geometry: the parts' silhouettes in a side view." if fallback else
                 "Joint motion: measured from the parts' 2D motion in a side view.")
        out.append(first + " Sizes in the motion plane: "
                   + ("scaled by an ID-1 card in view." if "card" in metric else "ASSUMED scale (no card measured): "
                      "rough.") + " Depth across the plane: assumed.")
    if fallback:
        out.append("No joint was measured: the model is a rigid fallback of the whole object.")
    phys = m.get("physics")
    if phys:
        measured = [k for k, v in phys["joints"].items() if v["measured"]]
        friction = ("" if not phys["joints"] else " Joint friction: " + (
            f"measured on {', '.join(measured)} (still spans without a hand)." if measured else
            "estimated (no hand-free still spans that load the joint)."))
        out.append("Masses and materials: estimated by Muse Spark from what the object is." + friction)
    else:
        out.append("Physics: not set.")
    d = m.get("design")
    if d and d.get("applied"):
        out.append(f"Visual shape: Muse Spark's design of a {d.get('object')} from the measured extents (an "
                   "estimate); collision uses the measured geometry.")
    elif d:
        out.append("Visual shape: the measured meshes (Spark's design scored worse on the video and wasn't applied).")
    if m.get("textured"):
        out.append("Textures: projected from the video onto the measured meshes; unseen faces take the part's colour.")
    if m.get("completion"):
        out.append("Unseen back: completed with typical sizes (estimates, labelled as such).")
    return out


def _fmt_size(v) -> str:
    return " x ".join(f"{x:.0f}" for x in v) + " mm" if isinstance(v, (list, tuple)) else str(v)


def _fmt(k: str, v) -> str:
    """A measured value for reading: sizes as A x B mm, other numbers rounded."""
    if k.endswith("_mm") and isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
        return _fmt_size(v)
    if isinstance(v, float):
        return f"{v:.1f}"
    if isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
        return "[" + ", ".join(f"{x:.1f}" for x in v) + "]"
    if isinstance(v, dict):
        return ", ".join(f"{a}: {_fmt(k, b)}" for a, b in v.items())
    return str(v)


def write_summary(run_dir: Path, s: dict) -> None:
    s["measured_vs_estimated"] = measured_vs_estimated(s) if s.get("model") else []
    (run_dir / "summary.json").write_text(json.dumps(s, indent=2, default=str) + "\n")
    inp = s.get("input") or {}
    lines = [f"# Video2URDF run: {s['name']}", "",
             f"**Status: {s['status'].replace('_', ' ')}**" + (f" ({len(s['issues'])} issues)" if s.get("issues") else ""),
             "", f"- Input: `{s['source']}`" + (f" -> {inp.get('kind')} (`{inp.get('found')}`)" if inp else "")]
    if inp.get("usable"):
        lines.append(f"- Normalised: {inp.get('frames')} frames at {inp.get('fps', 0):.2f} fps, "
                     f"{inp['size'][0]}x{inp['size'][1]}" + (f"; {'; '.join(inp['notes'])}" if inp.get("notes") else ""))
    p = s.get("path")
    if p:
        lines.append(f"- Path: **{p['chosen']}**" + (" (forced)" if p.get("forced") else "") + f": {' '.join(p['reasons'])}")
    out = s.get("outputs") or {}
    if out.get("urdf"):
        lines.append(f"- Model: `{out['urdf']}` and `{out['mjcf']}`")
    if out.get("viewer"):
        lines.append(f"- Viewer: `python -m http.server 8931 --directory {Path(out['viewer']).parent}` then "
                     "http://localhost:8931")
    if s.get("issues"):
        lines += ["", "## Issues", ""] + [f"- {i}" for i in s["issues"]]
    chk = s.get("check") or {}
    met = chk.get("metrics") or {}
    if met:
        cam = met.get("camera") or {}
        lines += ["", "## Capture check", "",
                  f"- {met.get('seconds')} s, {met.get('frames')} frames at {met.get('fps')} fps, "
                  f"{met.get('width')}x{met.get('height')}; brightness {(met.get('exposure') or {}).get('mean')}, "
                  f"sharpness {met.get('sharpness')}; camera {cam.get('camera')} ({cam.get('why', '')})"]
        lines += [f"- WARNING: {w}" for w in chk.get("warnings", [])]
        lines += [f"- Tip: {t}" for t in chk.get("tips", [])]
    look = s.get("look")
    if look:
        lines.append(f"- Spark's look: {look.get('object')}; moving parts {look.get('moving_parts')}; view "
                     f"{look.get('view')}; camera {look.get('camera')}; up at the {look.get('up')}")
    m = s.get("model") or {}
    if m:
        lines += ["", "## Model", "", f"Kind: {m.get('kind')}", ""]
        if m.get("links"):
            lines += ["| link | size (mm, collision geometry) |", "|---|---|"]
            lines += [f"| {k} | {_fmt_size(v['size_mm'])} |" for k, v in m["links"].items()]
        if m.get("joints"):
            lines += ["", "| joint | type | range | friction |", "|---|---|---|---|"]
            lines += [f"| {j['name']} | {j['type']} | {j['range'][0]} .. {j['range'][1]} {j['unit']} | "
                      f"{j['friction'] if j['friction'] is not None else '-'} |" for j in m["joints"]]
        if m.get("coverage"):
            lines += ["", f"Coverage ({m.get('coverage_metric')}): " +
                      ", ".join(f"{k} {100 * v:.0f}%" for k, v in m["coverage"].items())]
        for k, v in (m.get("measured") or {}).items():
            if k == "joints":
                lines += [f"- joint: {', '.join(f'{a} {_fmt(a, b)}' for a, b in j.items())}" for j in v]
            else:
                lines.append(f"- {k}: {_fmt(k, v)}")
        if m.get("design"):
            d = m["design"]
            lines.append(f"- Shape design: {d.get('object')}, {'applied' if d.get('applied') else 'not applied'} "
                         f"(near-F {d.get('score')} vs measured {d.get('measured_score')})")
    if s.get("measured_vs_estimated"):
        lines += ["", "## Measured vs estimated", ""] + [f"- {x}" for x in s["measured_vs_estimated"]]
    if s.get("steps"):
        lines += ["", "## Steps", "", "| step | status | seconds | note |", "|---|---|---|---|"]
        lines += [f"| {x['name']} | {x['status']} | {x['seconds']} | {str(x['note']).replace('|', '/')[:160]} |"
                  for x in s["steps"]]
        lines.append(f"\nTotal: {s.get('seconds', 0) / 60:.1f} min")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
