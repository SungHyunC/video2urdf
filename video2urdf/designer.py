"""Spark designs a clean visual for a fitted model, checks it over the video and revises it.

Round 1: Spark watches the video, sees the measured model (three views and an overlay on
frames) and the measured extents per link, names the object and its parts, and answers with
primitives per link. Each later round: the design is rendered from the recorded camera at the
recorded joint values, scored against the SAM masks (IoU), and Spark sees that overlay, the
three views with its design outlined over the measured meshes, and per-axis size differences.
The best-scoring design is applied if it is not clearly worse than the measured meshes.
"""

import base64
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI

from . import design as D
from .agent import API_KEY_ENV, BASE_URL, MAX_IMAGES, MODEL, VIDEO, jpeg_url, mp4_url

ROUNDS = 3
SCORE = "f_near"  # mask F-score within a few pixels (design.TOLERANCE): fair to thin parts
MAX_DROP = 0.10  # how much lower than the measured meshes' mean score a design may score

SYSTEM = """You are the shape designer of Video2URDF. From a phone video we have already measured an
articulated object's joints and a rough mesh of each link (silhouettes voted over the video, or depth points).
Those meshes are accurate in size but look blobby. Your job: name the object, then rebuild each link as a
clean, recognisable CAD-style model from a few primitives, so that anyone who sees the model says "that's
the object in the video".

Rules
- Identify the object and every link's real part (e.g. "desk lamp: weighted base, aluminium pole, arm with
  LED bar"). Model what the part really looks like: a laptop lid is a thin slab with a screen inset, a lamp
  pole is a thin cylinder or bar, a cabinet has a carcass and drawer fronts with handles, a base plate has
  rounded corners. Add small recognisable details (screen, keyboard area, handle, knob, hinge barrel, LED
  strip) as separate thin primitives with their own colours when you can see them.
- Coordinates are millimetres in each link's OWN frame. Each link's joint is at its origin (0,0,0). The
  joints of its children are at the given child_joints_in_link_mm points: the geometry of a link must reach
  those points so that parts stay connected through their joints.
- Sizes: on measured axes keep each link's overall extent within about 10% of measured_bbox_mm and in the
  same place (min/max), unless the overlay shows the measured mesh is clearly wrong. On unmeasured axes, or
  for sides the camera never saw, use realistic proportions for this kind of object, derived from the
  measured dimensions, and list each such choice in "assumptions".
- Colours (rgb 0..1) from the video, per primitive.
- Primitives (all in mm):
  box: center [x,y,z], size [sx,sy,sz], optional euler_deg [rx,ry,rz] (extrinsic xyz)
  rounded_box: as box plus corner_radius (all edges rounded)
  cylinder: from [x,y,z], to [x,y,z], radius
  sphere: center, radius
  bar: from, to, width, thickness: a rectangular beam between two points; width_axis "x"/"y"/"z"
       (optional) is the link axis the width lies along.
- Typically 1-8 primitives per link. Don't model the floor, table, hands or other objects. Only the links
  listed; every link needs at least one primitive.
- typical_size_mm: the usual range of the object's largest dimension. If the measured model is far outside
  it, say so in "scale_warning" (the scale may be wrong); still keep the measured sizes.
- measurement_consistent: false if the measured model cannot be this object as seen in the video: its
  proportions or orientation contradict it (e.g. a cabinet measured as a thin slab lying flat, or the
  measured mesh not covering the object in the overlay), so that fitting a design inside it would give a
  wrong-looking object. Explain in "measurement_problem". Don't force the real object into a wrong box.
Answer with one JSON object only:
{"object": str, "parts": {link: "what the real part is"}, "typical_size_mm": [lo, hi], "scale_warning": str,
 "measurement_consistent": bool, "measurement_problem": str,
 "links": [{"link": str, "primitives": [{"name": str, "type": ..., ..., "rgb": [r,g,b]}]}],
 "assumptions": [str], "notes": str, "done": bool}
Set done=true only when you would not change the design any more."""


def _client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=os.environ[API_KEY_ENV], timeout=900, max_retries=3)


def _png_url(img: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1]).decode()


def _parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in the answer")
    return json.loads(text[start:end + 1])


def _cap_images(messages: list[dict]) -> None:
    """Keep only the newest MAX_IMAGES images (Baseten allows 8 per request)."""
    if not MAX_IMAGES:
        return
    seen = 0
    for m in reversed(messages):
        if not isinstance(m.get("content"), list):
            continue
        for k in range(len(m["content"]) - 1, -1, -1):
            part = m["content"][k]
            if part.get("type") in ("image_url", "video_url"):
                seen += 1
                if seen > MAX_IMAGES:
                    m["content"][k] = {"type": "text", "text": "[older image dropped]"}


def _mean(scores: dict, key: str = SCORE) -> float:
    vals = [v[key] for v in scores.values() if v.get(key) is not None]
    return float(np.mean(vals)) if vals else 0.0


def _feedback(scores: dict, measured: dict, sizes: dict, problems: list[str]) -> str:
    lines = ["Your design rendered over the video at the recorded camera poses and joint values "
             "(first image) and in three views over the measured meshes (second image).",
             "Scores per link against the SAM masks, design vs the measured mesh on the same frames. "
             "near-F: F-score counting pixels within a few px of the mask (the score that decides); IoU: plain overlap:"]
    for link, s in scores.items():
        m = measured.get(link, {})
        lines.append(f"- {link}: design near-F {s['f_near']}, IoU {s['iou']}; "
                     f"measured mesh near-F {m.get('f_near')}, IoU {m.get('iou')} ({s['frames']} frames)")
    lines.append("Overall extent per axis, design vs measured (mm):")
    for link, axes in sizes.items():
        lines.append(f"- {link}: " + ", ".join(
            f"{ax} {v['design']} vs {v['measured']}{' (unmeasured)' if v['unmeasured'] else ''}" for ax, v in axes.items()))
    if problems:
        lines.append("Problems: " + "; ".join(problems))
    lines.append("The masks come from SAM and can be wrong; judge the overlay too. Fix anything misplaced, "
                 "detached, mis-sized or unlike the real object, and answer with the full revised JSON "
                 "(done=true if you would not change it).")
    return "\n".join(lines)


def design_shape(track_dir: Path, video: Path, rounds: int = ROUNDS, log=print, apply: bool = True) -> dict:
    """Run the design loop, write design.json and the logs, and apply the best design if it passes."""
    out = track_dir / "design"
    out.mkdir(exist_ok=True)
    ev = D.evidence(track_dir)
    links = [l["link"] for l in ev["links"]]
    scene = D.Scene(track_dir, video)
    if not scene.frames:
        raise ValueError("no frames with usable masks to check a design on")
    measured_xml = D.measured_xml(track_dir)
    measured = scene.score(measured_xml, 3)
    cv2.imwrite(str(out / "measured_overlay.jpg"), scene.overlay(scene._last, "measured mesh"))
    cv2.imwrite(str(out / "measured_views.png"), D.three_views(track_dir, None, links))
    log(f"measured mesh IoU: {json.dumps(measured)}")
    known = {}
    for name in ("appearance.json",):
        if (track_dir / name).exists():
            a = json.loads((track_dir / name).read_text())
            known = {"earlier_identification": a.get("object"), "materials": {k: v.get("material") for k, v in a.get("links", {}).items()}}
    if VIDEO:
        media = [{"type": "video_url", "video_url": {"url": mp4_url(video)}}]
    else:
        media = [{"type": "image_url", "image_url": {"url": jpeg_url(scene.images[t])}} for t in scene.frames[:: max(1, len(scene.frames) // 4)][:4]]
    user = media + [
        {"type": "text", "text": "Measured model at the reference pose, three views (pale fill per link, joints circled):"},
        {"type": "image_url", "image_url": {"url": _png_url(D.three_views(track_dir, None, links))}},
        {"type": "text", "text": "The measured meshes rendered over video frames (link colours; white = SAM mask of that link):"},
        {"type": "image_url", "image_url": {"url": jpeg_url(cv2.imread(str(out / "measured_overlay.jpg")), width=1080)}},
        {"type": "text", "text": "Measurements:\n" + json.dumps({**ev, **known}, indent=1) +
         f"\nLink colours in the images: " + ", ".join(f"{l}={c}" for l, c in zip(links, ["orange", "blue", "green", "magenta", "cyan", "light blue"])) +
         "\nDesign the clean model now."}]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    client = _client()
    history, best = [], None
    md = [f"# Shape design ({MODEL})\n", f"Measured mesh IoU: `{json.dumps(measured)}`\n",
          "![measured](measured_overlay.jpg)\n", "![views](measured_views.png)\n"]
    for k in range(1, rounds + 1):
        _cap_images(messages)
        text, failures = "", []
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=16000,
                                                      response_format={"type": "json_object"})
                text = resp.choices[0].message.content or ""
                finish = resp.choices[0].finish_reason
                design = _parse(text)
                problems = D.validate_design(design, links)
                break
            except (ValueError, json.JSONDecodeError) as e:
                design, problems = None, [f"unreadable answer: {e}"]
                failures.append(f"attempt {attempt + 1}: {e} (finish_reason {finish}, {len(text)} chars: {text[:200]!r})")
                messages += [{"role": "assistant", "content": text or "(empty)"},
                             {"role": "user", "content": f"That was not a valid JSON object ({e}). Answer with one short, "
                              "complete JSON object only: at most 6 primitives per link, numbers rounded to 1 mm."}]
        if design is None:
            log(f"round {k}: no usable answer: " + " | ".join(failures))
            md.append(f"\n## Round {k}: no usable answer\n\n" + "\n".join(f"- {f}" for f in failures) + "\n")
            if history:
                break
            continue
        messages.append({"role": "assistant", "content": json.dumps(design)})
        if problems:
            log(f"round {k}: problems {problems}")
            md.append(f"\n## Round {k}: rejected\n\n" + "\n".join(f"- {p}" for p in problems) + "\n")
            messages.append({"role": "user", "content": "The design can't be built: " + "; ".join(problems) +
                             ". Answer with the full corrected JSON."})
            continue
        xml, meshes = D.design_mjcf(measured_xml, design, mesh_dir=f"design/r{k}")
        for name, (v, f) in meshes.items():
            from .shape import write_stl
            (out / f"r{k}").mkdir(exist_ok=True)
            write_stl(out / f"r{k}" / f"{name}.stl", v, f)
        scores = scene.score(xml, 0)
        overlay = scene.overlay(scene._last, f"design round {k}")
        views = D.three_views(track_dir, design, links)
        cv2.imwrite(str(out / f"round{k}_overlay.jpg"), overlay)
        cv2.imwrite(str(out / f"round{k}_views.png"), views)
        sizes = D.bbox_check(track_dir, design, ev)
        entry = {"round": k, "design": design, "scores": scores, "mean_score": round(_mean(scores), 3), "sizes": sizes}
        history.append(entry)
        log(f"round {k}: {design.get('object')!r}, mean near-F {entry['mean_score']} "
            f"(measured {_mean(measured):.3f}) {json.dumps(scores)}")
        md += [f"\n## Round {k}: {design.get('object')}\n",
               f"Parts: `{json.dumps(design.get('parts', {}), ensure_ascii=False)}`\n",
               f"Mean near-F {entry['mean_score']} (measured mesh {_mean(measured):.3f}); per link `{json.dumps(scores)}`\n",
               f"Assumptions: {design.get('assumptions')}\n", f"Notes: {design.get('notes')}\n",
               f"![overlay](round{k}_overlay.jpg)\n", f"![views](round{k}_views.png)\n"]
        if best is None or entry["mean_score"] > best["mean_score"] + 0.01:
            best = entry
        if design.get("done") and k > 1:
            break
        if k < rounds:
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": jpeg_url(overlay, width=1080)}},
                {"type": "image_url", "image_url": {"url": _png_url(views)}},
                {"type": "text", "text": _feedback(scores, measured, sizes, [])}]})
    result = {"model": MODEL, "score": SCORE, "measured_scores": measured, "measured_mean_score": round(_mean(measured), 3),
              "rounds": history, "applied": False}
    if best:
        m_mean = _mean(measured)
        worst = [l for l, s in best["scores"].items() if s[SCORE] is not None and measured.get(l, {}).get(SCORE)
                 and s[SCORE] < 0.75 * measured[l][SCORE]]
        last = history[-1]["design"]  # Spark's latest judgement of the measurement
        consistent = last.get("measurement_consistent", True) is not False
        passed = best["mean_score"] >= m_mean - MAX_DROP and not worst and consistent
        warnings = ([f"Measured model doesn't match the object: {last.get('measurement_problem')}"] if not consistent else []) + \
                   ([f"Scale: {last['scale_warning']}"] if last.get("scale_warning") else [])
        result["warnings"] = warnings
        for w in warnings:
            log("WARNING: " + w)
        result.update({"object": best["design"].get("object"), "best_round": best["round"], "design": best["design"],
                       "passed": passed, "links_much_worse": worst,
                       "note": "Visual geometry designed by the model from the measured extents; "
                               "collision, joints and physics are the measured ones."})
        if apply and passed:
            D.apply_design(track_dir, best["design"])
            result["applied"] = True
        md.append(f"\n## Result\n\nBest round {best['round']} (mean near-F {best['mean_score']} vs measured {m_mean:.3f}): "
                  f"{'applied' if result['applied'] else 'NOT applied'}" + (f"; much worse on {worst}" if worst else "") + "\n"
                  + "".join(f"\n**Warning:** {w}\n" for w in warnings))
        log(f"best round {best['round']}: {'applied' if result['applied'] else 'not applied'}"
            + (f" (much worse on {worst})" if worst else "") + ("" if consistent else " (measurement inconsistent)"))
    (track_dir / "design.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (out / "log.md").write_text("\n".join(md))
    return result
