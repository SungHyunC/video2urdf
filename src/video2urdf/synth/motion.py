"""Smooth joint trajectories. Every joint visits both of its limits, so the observed range equals the GT limits."""

from __future__ import annotations

import numpy as np

from ..model import ArticulatedModel


def ease_through(times, values, num_frames: int) -> np.ndarray:
    """Piecewise smoothstep through (time, value) keyframes, holding the end values."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    frames = np.arange(num_frames, dtype=float)
    seg = np.clip(np.searchsorted(times, frames, side="right") - 1, 0, len(times) - 2)
    t0, t1 = times[seg], times[seg + 1]
    s = np.clip((frames - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)
    s = s * s * (3 - 2 * s)
    return values[seg] + (values[seg + 1] - values[seg]) * s


def _knot_times(rng: np.random.Generator, start: float, end: float, n: int) -> np.ndarray:
    """n increasing knot times on whole frames, so every keyframe value is actually rendered."""
    weights = rng.uniform(0.7, 1.3, size=n - 1)
    times = np.round(start + np.concatenate([[0.0], np.cumsum(weights) / weights.sum()]) * (end - start))
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{n} keyframes do not fit in frames {start:.0f}-{end:.0f}; render more frames")
    return times


def _sweep(rng: np.random.Generator, lower: float, upper: float, start: float, end: float, n_mid: int, num_frames: int):
    """Track that holds a random value, then passes through both limits in random order within [start, end]."""
    extremes = [lower, upper] if rng.random() < 0.5 else [upper, lower]
    mids = list(rng.uniform(lower, upper, size=n_mid))
    values = [float(rng.uniform(lower, upper)), *extremes, *mids]
    # Shuffle the middle so the extremes are not always first, but keep a random start value.
    middle = values[1:]
    rng.shuffle(middle)
    values = [values[0], *middle, float(rng.uniform(lower, upper))]
    return ease_through(_knot_times(rng, start, end, len(values)), values, num_frames)


def joint_trajectories(model: ArticulatedModel, num_frames: int, mode: str, rng: np.random.Generator) -> dict:
    """{joint name: (num_frames,) values}. `sequential` moves one joint at a time, like adjusting a lamp."""
    tracks = {}
    last = num_frames - 1
    n = len(model.joints)
    for i, joint in enumerate(model.joints):
        if mode == "sequential":
            start, end = i * last / n, (i + 1) * last / n
            tracks[joint.name] = _sweep(rng, joint.lower, joint.upper, start, end, 0, num_frames)
        elif mode == "simultaneous":
            tracks[joint.name] = _sweep(rng, joint.lower, joint.upper, 0, last, int(rng.integers(1, 3)), num_frames)
        else:
            raise ValueError(f"unknown motion mode {mode!r}")
    return tracks


def base_drift(num_frames: int, rng: np.random.Generator, shift: float = 0.02, turn: float = 0.08) -> dict:
    """Small smooth planar drift of the root, like an object held in the hand."""
    out = {}
    for name, amp in (("base_x", shift), ("base_z", shift), ("base_rot", turn)):
        values = [0.0, *rng.uniform(-amp, amp, size=3)]
        out[name] = ease_through(_knot_times(rng, 0, num_frames - 1, len(values)), values, num_frames)
    return out
