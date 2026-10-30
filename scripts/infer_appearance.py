"""Infer conservative materials from the video and fill unobserved faces.

Usage: python scripts/infer_appearance.py TRACK_DIR VIDEO [--reuse]
This changes visuals only. Inspect appearance.json: neither the category nor the
inferred surfaces are measurements. Run after fit, physics and before export.
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from openai import OpenAI  # noqa: E402
from video2urdf.agent import API_KEY_ENV, BASE_URL, MODEL, VIDEO, Agent, jpeg_url, mp4_url  # noqa: E402
from video2urdf.appearance import book_cover_mesh, category_of, complete_geometry, complete_laptop_details, remove_laptop_details  # noqa: E402
from video2urdf.appearance_check import compare_visuals, rejected_links  # noqa: E402
from video2urdf.assessment import assess  # noqa: E402
from video2urdf.lidar import Roles  # noqa: E402
from video2urdf.shape import write_stl  # noqa: E402
from video2urdf.track_checks import base_fragment_warning, named_base_warning, plate_covers_base_warning  # noqa: E402
from video2urdf.pipeline import load_parts, read_frame, video_info  # noqa: E402
from video2urdf.texture import link_masks, texture_model  # noqa: E402


def validate_classification(value):
    """Validate provider output even when the advertised JSON schema is not enforced."""
    categories = {"laptop", "drawer", "book", "box", "other"}
    materials = {"metal", "wood", "plastic", "glass", "paper", "fabric", "unknown"}
    if (not isinstance(value, dict) or not isinstance(value.get("object"), str)
            or not value["object"].strip() or len(value["object"].strip()) > 100
            or any(c in value["object"] for c in '{}\n') or value.get("category") not in categories
            or type(value.get("drawer_count")) is not int or value["drawer_count"] < 0
            or type(value.get("drawer_has_handles")) is not bool or not isinstance(value.get("links"), list)):
        raise ValueError("invalid object classification fields")
    for link in value["links"]:
        if (not isinstance(link, dict) or not isinstance(link.get("link"), str)
                or link.get("material") not in materials or not isinstance(link.get("reason"), str)):
            raise ValueError("invalid link material entry")
    return value


def object_crops(track_dir: Path, video: Path) -> list[dict]:
    """Crop the union of tracked object parts, excluding support surfaces."""
    _, tracks = load_parts(track_dir)
    (w, h), n, _ = video_info(video)
    names = {part for parts in link_masks(track_dir).values() for part in parts}
    frames = sorted({t for name in names for t, d in tracks.get(name, {}).items()
                     if 0 <= t < n and d.usable(w, h)})
    if not frames:
        return []
    selected = sorted({frames[0], frames[len(frames) // 2], frames[-1]})
    crops = []
    for t in selected:
        boxes = [tracks[name][t].box for name in names if t in tracks.get(name, {})
                 and tracks[name][t].usable(w, h)]
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        pad = max(4, int(.1 * max(x1 - x0, y1 - y0)))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad + 1), min(h, y1 + pad + 1)
        if x1 > x0 and y1 > y0:
            frame = read_frame(video, t)
            crops.append({"type": "image_url", "image_url": {"url": jpeg_url(frame[y0:y1, x0:x1])}})
    return crops


def refresh_fit_warnings(track_dir: Path, video: Path) -> None:
    """Check the original tracks, including frames a fit may have filtered out."""
    roles_path, model_path = track_dir / "roles.json", track_dir / "model3d.json"
    if not roles_path.exists() or not model_path.exists():
        return
    roles = Roles(**json.loads(roles_path.read_text()))
    (w, h), n, _ = video_info(video)
    _, tracks = load_parts(track_dir)
    usable = {name: {t: d for t, d in frames.items() if d.usable(w, h)} for name, frames in tracks.items()}
    if roles.base not in usable or roles.plate not in usable:
        model = json.loads(model_path.read_text())
        model.setdefault("fit_warnings", []).append("WARNING: required base or plate track is missing; raw-mask cross-check was unavailable.")
        model_path.write_text(json.dumps(model, indent=2) + "\n")
        return
    missing = []
    for field in ("plate_closed", "open_indicator"):
        name = getattr(roles, field)
        if name and name not in usable:
            missing.append(f"WARNING: tracked {field} {name!r} is missing; that cross-check was unavailable.")
            setattr(roles, field, None)
    reference = json.loads(model_path.read_text()).get("reference_frame")
    c = SimpleNamespace(track_dir=track_dir, by=usable, w=w, h=h, n=n, ref=reference)
    # The parent agent may have been running since before this checker was updated.
    # Re-check the raw anchor masks explicitly, rather than relying on its cached code.
    extra = (missing + Agent.mask_checks(object.__new__(Agent), c, roles, {})
             + base_fragment_warning(c, roles) + named_base_warning(c, roles) + plate_covers_base_warning(c, roles))
    model = json.loads(model_path.read_text())
    model["fit_warnings"] = list(dict.fromkeys(model.get("fit_warnings", []) + extra))
    model_path.write_text(json.dumps(model, indent=2) + "\n")


def infer_appearance(track_dir: Path, video: Path, reuse: bool = False, texture: bool = True):
    path = track_dir / "appearance.json"
    planar = (track_dir / 'joints.json').exists() and not (track_dir / 'model3d.json').exists()
    if planar and texture:
        raise ValueError('Planar models do not have depth and camera poses for video texture projection.')
    links = list(link_masks(track_dir))
    if reuse:
        appearance = json.loads(path.read_text())
    else:
        schema = {"type": "object", "properties": {
            "object": {"type": "string", "description": "The physical object name, such as a three-drawer cabinet, not its material."},
            "category": {"type": "string", "enum": ["laptop", "drawer", "book", "box", "other"]},
            "drawer_count": {"type": "integer", "description": "Number of visible drawer fronts, or 0 if uncertain or not a drawer cabinet."},
            "drawer_has_handles": {"type": "boolean", "description": "Whether handles are visibly present on drawer fronts."},
            "links": {"type": "array", "items": {"type": "object", "properties": {
                "link": {"type": "string"},
                "material": {"type": "string", "enum": ["metal", "wood", "plastic", "glass", "paper", "fabric", "unknown"]},
                "reason": {"type": "string"}}, "required": ["link", "material", "reason"]}}},
            "required": ["object", "category", "drawer_count", "drawer_has_handles", "links"]}
        client = OpenAI(base_url=BASE_URL, api_key=os.environ[API_KEY_ENV], timeout=600)
        if VIDEO:
            media = [{"type": "video_url", "video_url": {"url": mp4_url(video)}}]
        else:
            _, n, _ = video_info(video)
            frames = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
            media = [{"type": "image_url", "image_url": {"url": jpeg_url(read_frame(video, t))}} for t in frames]
        try:
            crops = object_crops(track_dir, video)
        except (ValueError, OSError, KeyError) as error:
            print(f"Object crops unavailable; classifying from the original video: {error}", flush=True)
            crops = []
        if crops:
            media += [{"type": "text", "text": "The following images are close-ups of the tracked target object. "
                       "Identify that object, not the supporting table, floor, or background. "
                       "Crops are based on tracked parts and may contain background or tracking errors."}] + crops
        content = media + [
                {"type": "text", "text": "Identify the object and the likely dominant surface material for "
                 f"each model link {json.dumps(links)}. Choose unknown when uncertain. "
                 "Choose category laptop, drawer (a drawer cabinet), book, box (a hinged box or chest), or other. Count the distinct visible "
                 "drawer fronts on closed frames; use drawer_count 0 when uncertain or not applicable. "
                 "Set drawer_has_handles only if drawer handles are visibly present. "
                 "In the object field name the physical item (for example 'three-drawer cabinet'), not its material; "
                 "put materials in the links entries. "
                 "If the function is not visually clear, use a broad shape name such as 'articulated stand' "
                 "rather than asserting a specific use. Call an object a lamp only when a light-emitting "
                 "component is visibly identifiable; an arm and a flat panel alone do not establish that. "
                 "Describe visual evidence briefly; do not invent measured dimensions or hidden features. "
                 "Return one concise JSON object only, with one entry per link."}]
        for attempt in range(2):
            response = client.chat.completions.create(
                model=MODEL, max_tokens=3000,
                response_format={"type": "json_schema", "json_schema": {"name": "appearance", "schema": schema}},
                messages=[{"role": "user", "content": content + ([{"type": "text", "text":
                    "Your previous response was invalid or incomplete JSON. Return only a short, complete JSON object."}]
                    if attempt else [])}])
            try:
                result = validate_classification(json.loads(response.choices[0].message.content))
                break
            except (TypeError, ValueError) as error:
                if attempt:
                    raise ValueError("Object classification returned invalid JSON twice; the fitted model is preserved. "
                                     "Retry appearance inference before exporting.") from error
                print("Object classification returned invalid JSON; retrying once.", flush=True)
        materials = {p["link"]: {"material": p["material"], "reason": p["reason"]}
                     for p in result["links"] if p["link"] in links}
        object_name = result["object"].strip()
        # Some providers return the requested link name (e.g. "base") in the
        # object field. Show the inferred category instead of presenting that
        # internal name as a physical object; preserve the raw answer below.
        material_names = {"cardboard", "wood", "metal", "plastic", "glass", "paper", "fabric", "unknown"}
        if object_name.lower() in {link.lower() for link in links} | material_names:
            object_name = result["category"] if result["category"] != "other" else "unidentified object"
        appearance = {"object": object_name, "category": result.get("category", "other"),
                      "drawer_count": result.get("drawer_count", 0),
                      "drawer_has_handles": result.get("drawer_has_handles", False), "model": MODEL,
                      "links": {link: materials.get(link, {"material": "unknown", "reason": "not identified"})
                                for link in links},
                      "note": "Object and materials are model estimates. Unseen surfaces are synthetic; "
                              "their texture is not observed. Inferred geometry is recorded separately from measurements."}
        if object_name != result["object"].strip():
            appearance["raw_object_response"] = result["object"]
        path.write_text(json.dumps(appearance, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(appearance, indent=2, ensure_ascii=False), flush=True)
    if planar:
        from video2urdf.planar_shape import complete_stand_base
        existing_path = track_dir / 'shape_inference.json'
        existing_shape = json.loads(existing_path.read_text()) if existing_path.exists() else {}
        if existing_shape.get('active') and existing_shape.get('inferred_visuals'):
            if existing_shape.get('object') != appearance.get('object'):
                raise ValueError('Planar visual already completed for another classification; start from the original fit.')
            shape = existing_shape
        else:
            try:
                shape = complete_stand_base(track_dir, video, appearance)
            except Exception as error:
                shape = {'active': False, 'inferred_visuals': [],
                         'shape': f'Planar visual completion unavailable ({type(error).__name__}); fitted visual retained.'}
        shape.update(category=category_of(appearance), object=appearance.get('object'), source='object category from video; planar fit')
        (track_dir / 'shape_inference.json').write_text(json.dumps(shape, indent=2) + '\n')
        print(json.dumps(assess(track_dir), indent=2), flush=True)
        return
    paths = [track_dir / name for name in ("model.xml", "model.urdf", "model.nophysics.xml", "model.nophysics.urdf")]
    originals = {p: p.read_text() for p in paths if p.exists()}
    shape_path = track_dir / "shape_inference.json"
    previous_shape = json.loads(shape_path.read_text()) if shape_path.exists() else {}
    existing = ET.fromstring(originals[track_dir / "model.xml"])
    already_inferred = any((g.get("mesh") or "").endswith("_infer") or
                           g.get("name", "").startswith("inferred_") for g in existing.iter("geom"))
    # Preserve the comparison and chosen blend for an already active visual.
    # Rebuilding its report would discard the evidence used to accept it and
    # could replace a selected book blend with the unverified first candidate.
    if already_inferred and previous_shape.get("active") is True:
        if previous_shape.get("category") != category_of(appearance):
            raise ValueError("The model already contains an inferred shape for a different category; "
                             "start from the original fitted model to change its appearance category.")
        shape = previous_shape
    else:
        shape = complete_geometry(track_dir, appearance)
    if shape["inferred_visuals"] and not already_inferred:
        links = sorted({v.get("link", "base") for v in shape["inferred_visuals"]})
        try:
            comparison = compare_visuals(track_dir, video, originals[track_dir / "model.xml"], links)
            failures = rejected_links(comparison)
            attempts = [{"blend": 0., "comparison": comparison}]
            if failures and shape["category"] == "book":
                # Preserve more of the measured silhouette when the idealized
                # rectangle cuts into visible pixels; choose the least expansion
                # that passes the same guard, never alter the collision mesh.
                current_mesh = shape["inferred_visuals"][0]["generated"]
                for blend in (.25, .5, .75, 1.):
                    # Unique paths prevent a renderer mesh cache from comparing
                    # a previous candidate when a file is rewritten rapidly.
                    candidate = str(Path(current_mesh).with_name(f"lid_infer_b{int(blend * 100):03}.stl"))
                    vertices, faces = book_cover_mesh(track_dir, outer_blend=blend)
                    write_stl(track_dir / candidate, vertices, faces)
                    for path in originals:
                        path.write_text(path.read_text().replace(current_mesh, candidate))
                    current_mesh = candidate
                    shape["inferred_visuals"][0]["generated"] = candidate
                    comparison = compare_visuals(track_dir, video, originals[track_dir / "model.xml"], links)
                    failures = rejected_links(comparison)
                    attempts.append({"blend": blend, "comparison": comparison})
                    if not failures:
                        shape["shape"] += (f" It was blended {blend:.0%} toward the fitted silhouette "
                                           "to preserve agreement with visible masks.")
                        shape["inferred_visuals"][0]["silhouette_blend"] = blend
                        break
            shape["visual_coverage_attempts"] = attempts
        except Exception as error:
            comparison = {"error": f"{type(error).__name__}: {error}"}
            failures = links
        shape["visual_coverage_check"] = comparison
        shape["active"] = not failures
        if failures:
            for path, content in originals.items():
                path.write_text(content)
            shape["shape"] += " Generated shape was not activated: observed-mask coverage could not be verified or worsened."
            shape["rejected_links"] = failures
            print("Inferred shape rejected for", ", ".join(failures), "; fitted visual retained.", flush=True)
        (track_dir / "shape_inference.json").write_text(json.dumps(shape, indent=2) + "\n")
    elif shape["inferred_visuals"] and shape is not previous_shape:
        shape["visual_coverage_check"] = previous_shape.get(
            "visual_coverage_check", {"error": "Model already had inferred visuals; no original baseline was available."})
        if "active" in previous_shape:
            shape["active"] = previous_shape["active"]
        (track_dir / "shape_inference.json").write_text(json.dumps(shape, indent=2) + "\n")
    if (track_dir / "static.json").exists():
        # Static fits already contain the generated envelope, so compare it
        # with tracked pixels without pretending there is an older baseline.
        try:
            check = compare_visuals(track_dir, video, (track_dir / "model.xml").read_text(), ["base"])
            check["note"] = "Same model rendered twice; IoU and coverage are sampled mask comparisons, not ground truth."
        except Exception as error:
            check = {"error": f"{type(error).__name__}: {error}"}
        shape["visual_coverage_check"] = check
        shape_path.write_text(json.dumps(shape, indent=2) + "\n")
    if texture:
        remove_laptop_details(track_dir)
        if shape.pop("schematic_visuals", None) is not None:
            shape_path.write_text(json.dumps(shape, indent=2) + "\n")
        texture_model(track_dir, video, appearance=appearance)
    else:
        if shape.get("category") == "laptop" and not shape.get("classification_mismatch"):
            shape["schematic_visuals"] = complete_laptop_details(track_dir)
            shape_path.write_text(json.dumps(shape, indent=2) + "\n")
        print("Simple appearance: retaining model colours without video texture projection.", flush=True)
    refresh_fit_warnings(track_dir, video)
    print(json.dumps(assess(track_dir), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track_dir", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--reuse", action="store_true", help="reuse appearance.json without a model call")
    parser.add_argument("--simple", action="store_true", help="use simple colours without projecting video textures")
    args = parser.parse_args()
    infer_appearance(args.track_dir, args.video, args.reuse, texture=not args.simple)


if __name__ == "__main__":
    main()
