"""Evaluation metrics for sound-event detection."""

from __future__ import annotations

#: Default segment length (seconds) for segment-based metrics.
DEFAULT_TIME_RESOLUTION = 1.0

#: Default onset collar (seconds) for event-based matching.
DEFAULT_T_COLLAR = 0.2

#: Default offset tolerance as a fraction of the reference event length.
DEFAULT_PERCENTAGE_OF_LENGTH = 0.5
