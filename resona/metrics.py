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

from typing import Any

import numpy as np

from ._typing import BoolArray
from .detection.events import Event
from .exceptions import InvalidParameterError

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


def _events_match(ref: Event, est: Event, t_collar: float, percentage_of_length: float) -> bool:
    """Whether ``est`` matches ``ref`` under the onset collar and offset tolerance."""
    onset_ok = abs(ref.onset - est.onset) <= t_collar
    offset_tolerance = max(t_collar, percentage_of_length * ref.duration)
    offset_ok = abs(ref.offset - est.offset) <= offset_tolerance
    return onset_ok and offset_ok


def _max_bipartite_matching(adjacency: list[list[int]], n_right: int) -> int:
    """Size of the maximum matching via Kuhn's augmenting-path algorithm.

    ``adjacency[i]`` lists the right-hand nodes that left node ``i`` may match.
    Optimal matching (rather than greedy) ensures the true-positive count does
    not depend on the order events happen to be listed in.
    """
    match_right = [-1] * n_right

    def augment(left: int, visited: list[bool]) -> bool:
        for right in adjacency[left]:
            if not visited[right]:
                visited[right] = True
                if match_right[right] == -1 or augment(match_right[right], visited):
                    match_right[right] = left
                    return True
        return False

    matched = 0
    for left in range(len(adjacency)):
        if augment(left, [False] * n_right):
            matched += 1
    return matched


def event_based_metrics(
    reference: list[Event],
    estimated: list[Event],
    *,
    labels: list[str] | None = None,
    t_collar: float = DEFAULT_T_COLLAR,
    percentage_of_length: float = DEFAULT_PERCENTAGE_OF_LENGTH,
) -> dict[str, float]:
    """Event-based precision/recall/F and error rate.

    Matching is label-constrained: an estimate may only match a reference event
    with the same label whose onset falls within ``t_collar`` seconds and whose
    offset falls within ``max(t_collar, percentage_of_length * length)`` seconds.
    Mislabelled-but-well-timed events therefore surface as a deletion/insertion
    pair rather than as a substitution.
    """
    reference = list(reference)
    estimated = list(estimated)
    labels = _resolve_labels(reference, estimated, labels)

    n_ref = 0
    n_sys = 0
    n_correct = 0
    for label in labels:
        ref_events = [e for e in reference if e.label == label]
        sys_events = [e for e in estimated if e.label == label]
        n_ref += len(ref_events)
        n_sys += len(sys_events)
        adjacency = [
            [
                j
                for j, s in enumerate(sys_events)
                if _events_match(r, s, t_collar, percentage_of_length)
            ]
            for r in ref_events
        ]
        n_correct += _max_bipartite_matching(adjacency, len(sys_events))

    deletions = n_ref - n_correct
    insertions = n_sys - n_correct
    precision, recall, f_measure = _prf(n_ref, n_sys, n_correct)
    denom = float(n_ref) if n_ref else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
        "error_rate": (deletions + insertions) / denom,
        "substitution_rate": 0.0,
        "deletion_rate": deletions / denom,
        "insertion_rate": insertions / denom,
        "n_ref": float(n_ref),
        "n_sys": float(n_sys),
        "n_correct": float(n_correct),
    }


def error_rate(
    reference: list[Event],
    estimated: list[Event],
    *,
    mode: str = "segment",
    **kwargs: Any,
) -> float:
    """Return just the error rate under the chosen ``mode``.

    A thin convenience over :func:`segment_based_metrics` /
    :func:`event_based_metrics` for callers that only care about the single
    headline number. Extra keyword arguments are forwarded to the underlying
    metric (e.g. ``time_resolution`` or ``t_collar``).
    """
    if mode == "segment":
        return segment_based_metrics(reference, estimated, **kwargs)["error_rate"]
    if mode == "event":
        return event_based_metrics(reference, estimated, **kwargs)["error_rate"]
    raise InvalidParameterError(f"unknown mode {mode!r}; use 'segment' or 'event'")
