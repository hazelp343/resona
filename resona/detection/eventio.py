"""Read and write event lists as JSON or CSV.

Two interchange formats are supported so detections survive a round trip to disk
and interoperate with annotation tools:

* **CSV** -- one event per row with an ``onset,offset,label,score`` header. Both
  comma- and tab-separated files are accepted on read.
* **JSON** -- a list of objects with the same four keys.

The format is inferred from the file extension unless given explicitly.
"""

from __future__ import annotations

import csv
import json
import os

from ..exceptions import AudioIOError, InvalidParameterError
from .events import Event

_FIELDS = ("onset", "offset", "label", "score")


def _infer_format(path: str, fmt: str | None) -> str:
    if fmt is not None:
        if fmt not in ("csv", "json"):
            raise InvalidParameterError(f"unknown format {fmt!r}; use 'csv' or 'json'")
        return fmt
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return "json"
    return "csv"


def _event_to_row(event: Event) -> list[str]:
    return [f"{event.onset:.6f}", f"{event.offset:.6f}", event.label, f"{event.score:.6f}"]


def save_events(path: str, events: list[Event], *, fmt: str | None = None) -> None:
    """Write ``events`` to ``path`` as CSV (default) or JSON."""
    chosen = _infer_format(path, fmt)
    try:
        if chosen == "json":
            payload = [
                {
                    "onset": e.onset,
                    "offset": e.offset,
                    "label": e.label,
                    "score": e.score,
                }
                for e in events
            ]
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        else:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(_FIELDS)
                writer.writerows(_event_to_row(e) for e in events)
    except OSError as exc:
        raise AudioIOError(f"could not write events to {path!r}: {exc}") from exc


def _row_to_event(row: dict[str, str]) -> Event:
    return Event(
        onset=float(row["onset"]),
        offset=float(row["offset"]),
        label=row.get("label", "sound") or "sound",
        score=float(row["score"]) if row.get("score") else 1.0,
    )


def load_events(path: str, *, fmt: str | None = None) -> list[Event]:
    """Read an event list from ``path``."""
    chosen = _infer_format(path, fmt)
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            text = handle.read()
    except OSError as exc:
        raise AudioIOError(f"could not read events from {path!r}: {exc}") from exc

    if chosen == "json":
        records = json.loads(text)
        return [_row_to_event({k: str(v) for k, v in rec.items()}) for rec in records]

    delimiter = "\t" if "\t" in text.splitlines()[0] else "," if text.strip() else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    return [_row_to_event(row) for row in reader]
