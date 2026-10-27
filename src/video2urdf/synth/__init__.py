"""Synthetic MuJoCo videos of planar mechanisms with known joints (ground truth for the estimator)."""

from .presets import PRESETS
from .render import SynthConfig, generate

__all__ = ["PRESETS", "SynthConfig", "generate"]
