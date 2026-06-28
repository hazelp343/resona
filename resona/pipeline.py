"""High-level helpers that tie embedding, detection, and evaluation together.

These wrap the lower-level building blocks behind three verbs -- *embed*,
*detect*, *evaluate* -- selectable by name, so a typical script (and the CLI)
never has to import concrete classes.
"""

from __future__ import annotations

from typing import Any

from ._typing import FloatArray
from .detection.energy import EnergyDetector
from .detection.events import Event
from .embeddings import Embedding, create_embedder
from .exceptions import InvalidParameterError
from .metrics import event_based_metrics, segment_based_metrics


def extract_embedding(
    signal: FloatArray, sr: int, *, embedder: str = "logmel", **kwargs: Any
) -> Embedding:
    """Embed ``signal`` with the named embedder (see ``available_embedders``)."""
    return create_embedder(embedder, **kwargs).embed(signal, sr)


def detect_events(
    signal: FloatArray, sr: int, *, detector: str = "energy", **kwargs: Any
) -> list[Event]:
    """Detect events in raw audio with the named detector."""
    if detector == "energy":
        return EnergyDetector(**kwargs).detect(signal, sr)
    raise InvalidParameterError(
        f"unknown detector {detector!r}; only 'energy' operates on raw audio "
        "(use ThresholdDetector directly for model activations)"
    )


def evaluate(
    reference: list[Event],
    estimated: list[Event],
    *,
    mode: str = "segment",
    **kwargs: Any,
) -> dict[str, float]:
    """Score detections against a reference using segment- or event-based metrics."""
    if mode == "segment":
        return segment_based_metrics(reference, estimated, **kwargs)
    if mode == "event":
        return event_based_metrics(reference, estimated, **kwargs)
    raise InvalidParameterError(f"unknown mode {mode!r}; use 'segment' or 'event'")
