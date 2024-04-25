"""Shared configuration for the built-in sound-event detectors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseDetector:
    """Common post-processing knobs shared by concrete detectors.

    Concrete detectors add their own ``detect(...)`` entry point; this base only
    carries the thresholds and smoothing parameters they have in common so the
    behaviour stays consistent across implementations.
    """

    threshold: float = 0.5
    min_duration_on: float = 0.0
    min_duration_off: float = 0.0
    median_filter_frames: int = 1
