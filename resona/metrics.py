"""Evaluation metrics for sound-event detection.

Two complementary views, following the conventions popularised by ``sed_eval``:

* **Segment-based** metrics rasterise reference and estimate onto a fixed time
  grid and compare activity segment by segment -- forgiving of small temporal
  errors.
* **Event-based** metrics match whole events under onset/offset tolerances --
  stricter, and closer to how a listener would judge the output.

Both report an F-measure (precision/recall) and an error rate decomposed into
substitutions, deletions, and insertions.
"""

from __future__ import annotations

import numpy as np

from ._typing import BoolArray
from .detection.events import Event

#: Default segment length (seconds) for segment-based metrics.
DEFAULT_TIME_RESOLUTION = 1.0

#: Default onset collar (seconds) for event-based matching.
DEFAULT_T_COLLAR = 0.2

#: Default offset tolerance as a fraction of the reference event length.
DEFAULT_PERCENTAGE_OF_LENGTH = 0.5


def _prf(n_ref: int, n_sys: int, n_correct: int) -> tuple[float, float, float]:
    """Precision, recall, and F1 from intermediate counts (0 when undefined)."""
    precision = n_correct / n_sys if n_sys else 0.0
    recall = n_correct / n_ref if n_ref else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    f_measure = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f_measure


def _resolve_labels(
    reference: list[Event], estimated: list[Event], labels: list[str] | None
) -> list[str]:
    if labels is not None:
        return list(labels)
    return sorted({event.label for event in (*reference, *estimated)})


def _segment_roll(
    events: list[Event], labels: list[str], time_resolution: float, n_segments: int
) -> BoolArray:
    index = {label: i for i, label in enumerate(labels)}
    roll = np.zeros((n_segments, len(labels)), dtype=bool)
    for event in events:
        if event.label not in index:
            continue
        start = max(0, int(np.floor(event.onset / time_resolution)))
        end = min(n_segments, int(np.ceil(event.offset / time_resolution)))
        if end > start:
            roll[start:end, index[event.label]] = True
    return roll


def segment_based_metrics(
    reference: list[Event],
    estimated: list[Event],
    *,
    labels: list[str] | None = None,
    time_resolution: float = DEFAULT_TIME_RESOLUTION,
    duration: float | None = None,
) -> dict[str, float]:
    """Segment-based precision/recall/F and error rate (micro-averaged)."""
    reference = list(reference)
    estimated = list(estimated)
    labels = _resolve_labels(reference, estimated, labels)

    if duration is None:
        duration = max((e.offset for e in (*reference, *estimated)), default=0.0)
    n_segments = int(np.ceil(duration / time_resolution)) if duration > 0 else 0

    ref_roll = _segment_roll(reference, labels, time_resolution, n_segments)
    sys_roll = _segment_roll(estimated, labels, time_resolution, n_segments)

    ref_counts = ref_roll.sum(axis=1)
    sys_counts = sys_roll.sum(axis=1)
    tp_counts = (ref_roll & sys_roll).sum(axis=1)

    n_ref = int(ref_counts.sum())
    n_sys = int(sys_counts.sum())
    n_correct = int(tp_counts.sum())

    substitutions = int(np.minimum(ref_counts, sys_counts).sum() - n_correct)
    deletions = int(np.maximum(0, ref_counts - sys_counts).sum())
    insertions = int(np.maximum(0, sys_counts - ref_counts).sum())

    precision, recall, f_measure = _prf(n_ref, n_sys, n_correct)
    denom = float(n_ref) if n_ref else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
        "error_rate": (substitutions + deletions + insertions) / denom,
        "substitution_rate": substitutions / denom,
        "deletion_rate": deletions / denom,
        "insertion_rate": insertions / denom,
        "n_ref": float(n_ref),
        "n_sys": float(n_sys),
        "n_correct": float(n_correct),
    }
