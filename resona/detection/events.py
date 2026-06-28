"""Sound events and conversions between event lists and activity rolls.

An *activity roll* is a boolean matrix of shape ``(n_frames, n_labels)``: cell
``(t, k)`` is ``True`` when label ``k`` is active in frame ``t``. Rolls are the
natural interface to frame-level models, while :class:`Event` lists are the
natural interface to humans and annotation files; the helpers here convert
losslessly between the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._typing import BoolArray
from ..exceptions import InvalidParameterError


@dataclass(frozen=True)
class Event:
    """A labelled, scored time interval (seconds)."""

    onset: float
    offset: float
    label: str = "sound"
    score: float = 1.0

    def __post_init__(self) -> None:
        if self.offset < self.onset:
            raise InvalidParameterError(f"event offset {self.offset} precedes onset {self.onset}")

    @property
    def duration(self) -> float:
        """Length of the event in seconds."""
        return self.offset - self.onset


def unique_labels(events: list[Event]) -> list[str]:
    """Sorted list of distinct labels present in ``events``."""
    return sorted({event.label for event in events})


def find_runs(mask: BoolArray) -> list[tuple[int, int]]:
    """Return half-open ``(start, end)`` index ranges of ``True`` runs in ``mask``."""
    flat = np.asarray(mask, dtype=bool)
    if flat.size == 0:
        return []
    padded = np.concatenate(([False], flat, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def events_to_roll(
    events: list[Event],
    *,
    sr: int,
    hop_length: int,
    n_frames: int | None = None,
    labels: list[str] | None = None,
) -> tuple[BoolArray, list[str]]:
    """Rasterise ``events`` into a ``(n_frames, n_labels)`` activity roll.

    Returns the roll together with the (resolved) label ordering so callers can
    map columns back to names.
    """
    if labels is None:
        labels = unique_labels(events)
    index = {label: i for i, label in enumerate(labels)}
    frame_rate = sr / hop_length

    if n_frames is None:
        max_offset = max((event.offset for event in events), default=0.0)
        n_frames = int(np.ceil(max_offset * frame_rate)) + 1

    roll = np.zeros((n_frames, len(labels)), dtype=bool)
    for event in events:
        if event.label not in index:
            continue
        start = max(0, int(np.floor(event.onset * frame_rate)))
        end = min(n_frames, int(np.ceil(event.offset * frame_rate)))
        if end > start:
            roll[start:end, index[event.label]] = True
    return roll, list(labels)


def roll_to_events(roll: BoolArray, *, sr: int, hop_length: int, labels: list[str]) -> list[Event]:
    """Inverse of :func:`events_to_roll`: contiguous active runs become events."""
    frame_duration = hop_length / sr
    events: list[Event] = []
    for column, label in enumerate(labels):
        for start, end in find_runs(roll[:, column]):
            events.append(
                Event(
                    onset=start * frame_duration,
                    offset=end * frame_duration,
                    label=label,
                )
            )
    events.sort(key=lambda event: (event.onset, event.label))
    return events
