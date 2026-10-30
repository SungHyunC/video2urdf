"""Summarize known limitations of a fitted and visually completed model."""
import json
import math
from pathlib import Path
import numpy as np
from .track_checks import reference_opening_warning, drawer_prompt_warnings, overclosed_hinge_warning


def assess(track_dir: Path) -> dict:
    if not (track_dir / 'model3d.json').exists() and (track_dir / 'joints.json').exists():
        result = {'status': 'review', 'warnings': [
            'Planar model uses a side-view silhouette; hidden depth and thickness are assumed.',
            'Metric scale may be assumed; verify dimensions against a known reference.',
            'Inspect the motion and shape from the video; no depth-based 3D validation was performed.'],
            'note': 'The planar result is an approximate visual model, not measured 3D geometry.'}
        shape_path = track_dir / 'shape_inference.json'
        if shape_path.exists() and json.loads(shape_path.read_text()).get('active'):
            result['warnings'].append('The simplified planar visual differs from the fitted collision geometry.')
        if (track_dir / 'physics.json').exists():
            result['warnings'].append('Physics properties use estimated mass, shape and release intervals; they are not independently measured.')
        elif (track_dir / 'agent' / 'report.md').exists():
            result['warnings'].append('Physics properties were not assigned; do not assume mass or friction estimates were applied.')
        replay_path = track_dir / 'planar_replay.npz'
        if replay_path.exists():
            with np.load(replay_path) as replay_data:
                scores = [key for key in replay_data.files if key.startswith('iou_')]
                for key in scores:
                    values = replay_data[key]
                    valid = values[np.isfinite(values)]
                    if not len(valid):
                        result['warnings'].append(f'{key[4:]}: no usable planar replay mask scores were recorded.')
                    elif float(valid.mean()) < .7:
                        result['warnings'].append(f'{key[4:]}: mean planar replay mask IoU is {valid.mean():.2f} over '
                                                  f'{len(valid)}/{len(values)} frames; inspect the fit and tracking.')
                if not scores:
                    result['warnings'].append('Per-part planar replay mask scores were not recorded.')
        else:
            result['warnings'].append('Planar replay motion and per-part mask scores were not recorded.')
        (track_dir / 'quality.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    model = json.loads((track_dir / "model3d.json").read_text())
    warnings = []
    experiment = track_dir / 'experimental_alignment.json'
    if experiment.exists():
        info = json.loads(experiment.read_text())
        warnings.append(info.get('warning', 'Experimental laptop alignment: visual and replay changes are not physics-ready.'))
        warnings.append('Stored fit coverage and shape comparisons refer to the original model, not this experimental alignment.')
        comparison_path = track_dir / 'experimental_comparison.json'
        if comparison_path.exists():
            failed = json.loads(comparison_path.read_text()).get('rejected_links', [])
            if failed:
                warnings.append(f"Laptop review candidate failed sampled mask checks for {', '.join(failed)}; inspect before use.")
        else:
            warnings.append('Laptop review candidate could not be compared with the original on sampled masks.')
    drawer_collision = track_dir / 'experimental_drawer_collision.json'
    if drawer_collision.exists():
        info = json.loads(drawer_collision.read_text())
        warnings.append(info.get('warning', 'Drawer collision depth is an unvalidated visual assumption.'))
    if model.get("static_envelope"):
        warnings.append("The static model is an approximate shape based on observed depth; hidden surfaces and complete dimensions are not measured.")
    coverage = model.get("mask_coverage")
    if not coverage:
        warnings.append("Full-replay mask coverage was not recorded; sampled visual checks, if present, do not replace it.")
    else:
        for link, value in coverage.items():
            if value is None or not math.isfinite(value):
                warnings.append(f"{link}: replay mask coverage unavailable.")
            elif value < .7:
                warnings.append(f"{link}: replay covers {100 * value:.0f}% of the tracked mask (review threshold 70%).")
            frames = model.get("mask_coverage_frames", {}).get(link)
            total = model.get("video_frames")
            if frames is None or not total:
                warnings.append(f"{link}: number of frames included in replay coverage was not recorded.")
            elif frames < .5 * total:
                warnings.append(f"{link}: coverage was evaluated on only {frames}/{total} frames; unseen or filtered frames are unscored.")
    if model.get("base_thickness_measured") is False:
        warnings.append("Base thickness is a fallback, not a measurement.")
    base_thickness = model.get("base_thickness_mm")
    if isinstance(base_thickness, (int, float)) and (not math.isfinite(base_thickness) or base_thickness <= 0):
        warnings.append("Base thickness is invalid; the fitted geometry must be reviewed or refitted.")
    appearance_file = track_dir / "appearance.json"
    if appearance_file.exists():
        appearance = json.loads(appearance_file.read_text())
        thickness = model.get("base_thickness_mm")
        face = model.get("plate_size_mm")
        if (appearance.get("category") == "laptop" and thickness is not None and face and
                min(face) > 0 and thickness > .15 * min(face)):
            warnings.append("Laptop base thickness exceeds 15% of the smaller fitted lid dimension; "
                            "check the depth and floor fit before trusting the dimensions.")
    if model.get("plate_thickness_clipped") is True:
        warnings.append("Plate thickness was clipped or assumed, not reliably measured.")
    elif model.get("plate_thickness_mm") and model.get("plate_size_mm"):
        width = min(model["plate_size_mm"])
        if width > 0 and model["plate_thickness_mm"] / width > .05:
            warnings.append("Plate thickness exceeds 5% of its smaller face dimension; check whether the depth fit is plausible.")
    limits = model.get("hinge_range_deg")
    if (isinstance(limits, list) and len(limits) == 2
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in limits)
            and limits[1] - limits[0] > 200):
        warnings.append(f"Hinge range spans {limits[1] - limits[0]:.0f}°; check for tracking drift or angle wrapping. "
                        "Some hinges can rotate this far, so compare against the video.")
    warnings.extend(model.get("fit_warnings", []))
    drawers_path = track_dir / "drawers.json"
    if drawers_path.exists():
        drawers = json.loads(drawers_path.read_text())
        warnings.extend(drawer_prompt_warnings(track_dir, drawers.get("base", []),
                                               [item[0] for item in drawers.get("drawers", [])]))
    hinge_path, roles_path = track_dir / "hinge.json", track_dir / "roles.json"
    if hinge_path.exists() and roles_path.exists():
        hinge = json.loads(hinge_path.read_text())
        roles = json.loads(roles_path.read_text())
        ref, angles = hinge.get("reference_frame"), hinge.get("q", [])
        if isinstance(ref, int) and 0 <= ref < len(angles):
            q_ref = angles[ref]
            if q_ref is None or not math.isfinite(q_ref):
                warnings.append(f"Reference frame {ref}: hinge angle was filtered or unavailable; "
                                "the reported opening may come from a different frame. Re-track and refit.")
        warning = reference_opening_warning(hinge.get("opening_on_reference_deg"), roles.get("open_indicator"))
        if warning and warning not in warnings:
            warnings.append(warning)
        warning = overclosed_hinge_warning(hinge)
        if warning:
            warnings.append(warning)
    unions_path = track_dir / "track_unions.json"
    if roles_path.exists() and unions_path.exists():
        roles = json.loads(roles_path.read_text())
        unions = json.loads(unions_path.read_text())
        if roles.get("base") in unions:
            warnings.append("Base geometry combines tracked regions; unobserved gaps and outer margins are assumptions.")
    shape_path = track_dir / "shape_inference.json"
    if shape_path.exists():
        shape = json.loads(shape_path.read_text())
        if shape.get("active") is not False and (shape.get("visual_thickness_assumed") or shape.get("visual_alignment_assumed") or shape.get("visual_lid_height_assumed")):
            warnings.append("Laptop visual proportions were adjusted heuristically; fitted geometry and dimensions remain unchanged.")
            if shape.get("visual_thickness_assumed"):
                warnings.append("Visual base thickness differs from collision geometry; physics placement may not match the appearance.")
        if shape.get("invalid_geometry"):
            warnings.append(f"Shape completion skipped: {shape['invalid_geometry']}.")
        if shape.get("classification_mismatch"):
            warnings.append(f"Object classification and fitted motion disagree: {shape['classification_mismatch']}.")
        if shape.get("drawer_count_assumed"):
            warnings.append(f"Drawer count was uncertain; {shape['drawer_count_assumed']} fronts were assumed for appearance.")
        if shape.get('static_drawer_side_assumed') and shape.get('active') is not False:
            warnings.append('Drawer fronts were placed on the most viewed side; that side may not be the actual front.')
        if shape.get('static_drawer_depth_assumed_mm') and shape.get('active') is not False and not drawer_collision.exists():
            warnings.append(f"Cabinet visual depth of {shape['static_drawer_depth_assumed_mm']:.0f} mm is assumed "
                            'from width; the fitted collision depth remains unchanged.')
        dims = model.get('generated_size_mm')
        if (shape.get('category') == 'drawer' and model.get('static_envelope')
                and isinstance(dims, list) and len(dims) == 3
                and all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in dims)
                and dims[2] < .15 * min(dims[:2])):
            warnings.append('The static drawer model is very flat relative to its horizontal footprint; '
                            'the estimated upright direction or depth may be wrong. Inspect the shape before use.')
        if shape.get("active") is False:
            warnings.append("Inferred shape failed the observed-mask comparison; fitted visual was retained.")
        elif shape.get("visual_coverage_check", {}).get("error"):
            warnings.append("Inferred shape could not be compared against an original visual baseline.")
        if shape.get("category") == "other" and not shape.get("inferred_visuals") and not model.get("static_envelope"):
            warnings.append("No object-specific shape completion is available for this category; "
                            "the fitted visual was retained unless a SAM 3D candidate was accepted.")
        for link, scores in shape.get("visual_coverage_check", {}).get("links", {}).items():
            # Older fitted runs may lack full-replay coverage even though the
            # appearance comparison contains a useful, explicitly sampled check.
            # A rejected candidate is not the visual left in the model.
            field = "baseline" if shape.get("active") is False else "inferred"
            sampled = scores.get(field)
            if (isinstance(sampled, (int, float)) and math.isfinite(sampled)
                    and sampled < .7 and (not coverage or link not in coverage)):
                frames = scores.get("frames")
                extent = f" on {frames} sampled frames" if isinstance(frames, int) else " on sampled frames"
                warnings.append(f"{link}: retained visual covers {100 * sampled:.0f}% of the tracked mask{extent}; "
                                "inspect the fit and tracking before trusting the shape.")
            if scores.get("iou_informative") is False:
                warnings.append(f"{link}: the tracked mask covers only selected regions; visual excess outside it "
                                "cannot be reliably checked by mask IoU.")
            elif (isinstance(scores.get(field + "_iou"), (int, float))
                  and math.isfinite(scores[field + "_iou"]) and scores[field + "_iou"] < .5):
                warnings.append(f"{link}: sampled visual-mask IoU is {scores[field + '_iou']:.2f}; "
                                "inspect the fit and the tracked mask before using the geometry.")
    sam3d = track_dir / "sam3d.json"
    if sam3d.exists():
        for link, result in json.loads(sam3d.read_text()).items():
            if not result.get("accepted", False):
                warnings.append(f"{link}: the latest SAM 3D candidate failed the activation checks.")
            else:
                warnings.append(f"{link}: SAM 3D passed geometric checks, but texture and hidden surfaces "
                                "were not validated; inspect the visual from multiple views.")
            if result.get("texture_experiment_error"):
                warnings.append(f"{link}: video texture transfer failed; inspect the retained SAM 3D visual.")
            if result.get("texture_experiment"):
                warnings.append(f"{link}: experimental texture transfer; inspect all sides and texture sources.")
    sam3d_status = track_dir / "sam3d_status.json"
    if sam3d_status.exists() and json.loads(sam3d_status.read_text()).get("state") == "failed":
        warnings.append("SAM 3D execution failed; fitted appearance was retained for unfinished links.")
    texture_path = track_dir / "texture.json"
    if texture_path.exists():
        textures = json.loads(texture_path.read_text()).get("meshes", {})
        for mesh, info in textures.items():
            if info.get("faces", 0) and info.get("faces_seen") == 0:
                warnings.append(f"{mesh}: no face was textured from observed video; its appearance is inferred or a fallback.")
    source_file = track_dir / "measurement_source.json"
    if source_file.exists():
        source = json.loads(source_file.read_text())
        if source.get("depth") == "MapAnything estimated from RGB video":
            warnings.append("RGB-only depth and metric scale are estimates; verify dimensions against a reference.")
            for check in source.get("reference_checks", []):
                error = check.get("relative_error")
                if isinstance(error, (int, float)) and abs(error) > .05:
                    warnings.append(f"Independent reference at frame {check.get('frame')} differs by {error:+.1%} "
                                    "after global calibration; local dimensions may be inaccurate.")
    repair = track_dir / "repair.json"
    if repair.exists() and json.loads(repair.read_text()).get("method") == "manual":
        warnings.append("Part tracking was manually repaired; this result is not an independent autonomous run.")
    if (track_dir / 'manual_selection.json').exists():
        warnings.append('Segmentation candidates were manually selected; this is not an uninterrupted autonomous run.')
    resumed = track_dir / 'resume.json'
    if resumed.exists():
        info = json.loads(resumed.read_text())
        warnings.append('Appearance was resumed from an existing fit; this is not an uninterrupted autonomous run.')
        if info.get('classification') == 'reused':
            warnings.append('Object classification was reused from an earlier result; it was not rerun for this fit.')
    physics_source = track_dir / "physics_input_provenance.json"
    if physics_source.exists():
        warnings.append("Physics uses estimated masses and release intervals; inspect physics_input_provenance.json.")
    elif (track_dir / "physics.json").exists():
        warnings.append("Physics properties were estimated from video, not independently measured.")
    elif (track_dir / "agent" / "report.md").exists() and not (track_dir / "physics.json").exists():
        warnings.append("Physics properties were not recorded for this agent result; do not assume estimated masses or friction were applied.")
    warnings = list(dict.fromkeys(warnings))
    result = {"status": "review" if warnings else "basic checks passed", "warnings": warnings,
              "note": "Coverage uses SAM masks, not independent ground truth. Passing basic checks does not prove accuracy."}
    (track_dir / "quality.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
