"""Checks on raw tracked masks that do not depend on a generated model."""
import json
import math

import numpy as np

from .pipeline import slug


def drawer_prompt_warnings(track_dir, base_parts, drawer_parts):
    """Highlight ambiguous seeds without assuming identical prompts mean identical masks."""
    path = track_dir / "parts.json"
    if not path.exists():
        return []
    parts = json.loads(path.read_text())
    prompts = {slug(item.get("name") or item.get("prompt", "")): item.get("prompt", "").strip().casefold()
               for item in parts.values()}
    warnings = []
    for moving in drawer_parts:
        for base in base_parts:
            if base != moving and prompts.get(base) and prompts.get(base) == prompts.get(moving):
                warnings.append(f"WARNING: base {base!r} and moving drawer {moving!r} were seeded with the same "
                                "description. Their masks may still be different; inspect both across the motion "
                                "to confirm the moving drawer was selected.")
    return warnings


def conflicting_drawer_tracks(c, base_parts, drawers, samples=40):
    """Find base and moving tracks repeatedly identifying the same visible pixels."""
    warnings = []
    for part, (start, end) in drawers:
        for base in base_parts:
            shared = sorted(t for t in set(c.by.get(part, {})) & set(c.by.get(base, {}))
                            if start <= t <= end)
            if not shared:
                continue
            indices = np.unique(np.linspace(0, len(shared) - 1, min(samples, len(shared))).astype(int))
            same = []
            for i in indices:
                t = shared[i]
                a = c.by[part][t].full_mask(c.h, c.w)[::4, ::4]
                b = c.by[base][t].full_mask(c.h, c.w)[::4, ::4]
                union = (a | b).sum()
                if union and (a & b).sum() / union > .8:
                    same.append(t)
            if len(same) >= 3 and len(same) >= .2 * len(indices):
                warnings.append(f"{base!r} and moving part {part!r} identify essentially the same surface "
                                f"on {len(same)}/{len(indices)} sampled frames (e.g. {same[:4]}). "
                                "Inspect both tracks; remove the moving front from base or re-track it.")
    return warnings


def suggested_open_anchors(c, roles, limit: int = 7) -> list[int]:
    """Spread candidate anchor frames over the tracked open intervals."""
    if not roles.open_indicator or limit < 1:
        return []
    frames = sorted(c.by.get(roles.open_indicator, {}))
    if not frames:
        return []
    indexes = np.unique(np.linspace(0, len(frames) - 1, min(limit, len(frames))).astype(int))
    return [frames[i] for i in indexes]


def reference_opening_warning(opening, open_indicator) -> str | None:
    if open_indicator and opening is not None and np.isfinite(opening) and opening < 5:
        return (f"WARNING: reference opening is {opening:.1f}° despite the open indicator; "
                "the indicator or hinge fit may be wrong, or the object may only be slightly open. "
                "Inspect the reference frame and measured motion before trusting dimensions.")
    return None


def overclosed_hinge_warning(hinge) -> str | None:
    """Compare the fitted angular range to the estimated closed direction."""
    opening = hinge.get('opening_on_reference_deg')
    angles = [v for v in hinge.get('q', []) if isinstance(v, (int, float)) and math.isfinite(v)]
    if not isinstance(opening, (int, float)) or not math.isfinite(opening) or opening < 0 or len(angles) < 10:
        return None
    q = np.rad2deg(angles)
    lo, hi = np.percentile(q, [1, 99])
    toward = 1 if abs(hi - opening) < abs(lo + opening) else -1
    excess = (hi if toward > 0 else -lo) - opening
    if excess > 10:
        return (f'WARNING: the fitted hinge extends approximately {excess:.0f}° beyond its estimated closed '
                'position. Inspect the closed pose and tracking; the lid may pass through the base.')
    return None


def named_base_warning(c, roles) -> list[str]:
    """Flag a laptop base seeded using a small component, even when roles overlap."""
    path = getattr(c, "track_dir", None)
    if path is None or not (path / "parts.json").exists():
        return []
    parts = json.loads((path / "parts.json").read_text())
    info = next((v for v in parts.values() if slug(v.get("name") or v.get("prompt", "")) == roles.base), None)
    prompt = (info or {}).get("prompt", "").lower()
    if "laptop" in prompt and any(word in prompt for word in ("keyboard", "touchpad", "trackpad")):
        return ["WARNING: the laptop base was seeded as a keyboard or touchpad. Inspect whether the "
                "mask includes the entire base surface; a fragment cannot determine the full base size."]
    return []


def plate_covers_base_warning(c, roles) -> list[str]:
    if roles.base == roles.plate or roles.base not in c.by or roles.plate not in c.by:
        return []
    shared = sorted(set(c.by[roles.base]) & set(c.by[roles.plate]))
    if roles.open_indicator:
        shared = sorted(set(shared) & set(c.by.get(roles.open_indicator, {})))
    ratios = []
    ref = getattr(c, "ref", None)
    sampled = sorted(set(shared[::max(1, len(shared) // 40)]) | ({ref} if ref in shared else set()))
    ref_ratio = None
    for t in sampled:
        base = c.by[roles.base][t].full_mask(c.h, c.w)
        plate = c.by[roles.plate][t].full_mask(c.h, c.w)
        if base.any():
            ratio = (plate & base).sum() / base.sum()
            ratios.append(ratio)
            if t == ref:
                ref_ratio = ratio
    if ref_ratio is not None and ref_ratio > .5:
        return [f"WARNING: on reference frame {ref}, {100 * ref_ratio:.0f}% of the base mask lies inside the "
                "plate mask. The plate track may have drifted onto the base; inspect and re-segment the "
                "moving face before trusting its size or hinge."]
    if ratios and np.median(ratios) > .8:
        return [f"WARNING: on frames showing both, {100 * np.median(ratios):.0f}% of the base mask lies inside "
                "the plate mask. The plate track may have drifted onto the base; inspect the open-frame "
                "masks and re-segment the moving face before trusting its size or hinge."]
    return []


def alternate_plate_hint(c, roles) -> str | None:
    """Suggest a test when the closed-track mask is distinct and the plate covers the base."""
    ref = getattr(c, "ref", None)
    names = (roles.base, roles.plate, roles.plate_closed)
    if ref is None or len(set(names)) != 3 or any(not name or ref not in c.by.get(name, {}) for name in names):
        return None
    base, plate, alternate = (c.by[name][ref].full_mask(c.h, c.w) for name in names)
    if not base.any() or not alternate.any():
        return None
    base_overlap = (base & plate).sum() / base.sum()
    alt_overlap = (base & alternate).sum() / base.sum()
    if base_overlap > .3 and alt_overlap < .2:
        return (f"WARNING: at reference frame {ref}, the plate mask covers the base while "
                f"{roles.plate_closed} does not. Inspect both masks; if {roles.plate_closed} is the moving face, "
                "try swapping plate and plate_closed in fit_lidar, then verify the replay and dimensions.")
    return None


def base_fragment_warning(c, roles) -> list[str]:
    if not roles.open_indicator or roles.open_indicator == roles.base:
        return []
    if roles.base not in c.by or roles.open_indicator not in c.by:
        return []
    shared = sorted(set(c.by[roles.base]) & set(c.by[roles.open_indicator]))
    anchored = False
    path = getattr(c, "track_dir", None)
    if path is not None and (path / "parts.json").exists():
        parts = json.loads((path / "parts.json").read_text())
        info = next((v for v in parts.values() if slug(v.get("name") or v.get("prompt", "")) == roles.base), None)
        if info:
            anchors = {int(x[0]) for x in info.get("anchors", [])}
            if anchors:
                shared = [t for t in shared if t in anchors]
                anchored = True
    overlaps = []
    for t in shared[::max(1, len(shared) // 40)]:
        indicator = c.by[roles.open_indicator][t].full_mask(c.h, c.w)
        if indicator.any():
            overlaps.append((indicator & c.by[roles.base][t].full_mask(c.h, c.w)).sum() / indicator.sum())
    if len(overlaps) >= (1 if anchored else 5) and np.median(overlaps) < .5:
        return ["WARNING: the base mask misses most of the open indicator on checked frames. If the "
                "indicator is part of the base (such as a keyboard), the base candidate may be only "
                "a fragment; inspect it and choose a full base surface before trusting its size."]
    return []
