"""Sound events and conversions between event lists and activity rolls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    """A single sound event: a labelled, scored time interval (in seconds)."""

    onset: float
    offset: float
    label: str = "sound"
    score: float = 1.0
