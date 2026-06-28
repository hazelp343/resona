"""Threshold-based detector over precomputed activations."""

from __future__ import annotations

from .._typing import FloatArray
from .base import BaseDetector
from .events import Event
from .postprocess import activations_to_events


class ThresholdDetector(BaseDetector):
    """Turn frame-level class activations into events.

    This is the decision layer you place on top of any model that emits
    per-frame, per-class scores (probabilities, logits passed through a sigmoid,
    energies, ...). It thresholds, optionally smooths, and enforces minimum
    event/gap durations using the shared post-processing pipeline.
    """

    def detect(
        self,
        activations: FloatArray,
        *,
        labels: list[str],
        sr: int,
        hop_length: int,
    ) -> list[Event]:
        """Detect events from an ``(n_frames, n_labels)`` activation matrix."""
        return activations_to_events(
            activations,
            labels,
            sr=sr,
            hop_length=hop_length,
            threshold=self.threshold,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
            median_filter_frames=self.median_filter_frames,
        )
