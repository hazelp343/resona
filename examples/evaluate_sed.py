"""Score energy-gate detections against the ground truth.

The energy detector emits a single ``active`` class, so the reference events are
relabelled to ``active`` before scoring -- an apples-to-apples activity-detection
evaluation.

Run with::

    python examples/evaluate_sed.py
"""

from __future__ import annotations

from resona import datasets, detect_events, evaluate
from resona.detection.events import Event


def _as_active(events: list[Event]) -> list[Event]:
    return [Event(e.onset, e.offset, "active", e.score) for e in events]


def main() -> None:
    scene = datasets.synthetic_scene(duration=8.0, sr=22050, n_events=6, seed=11)
    reference = _as_active(scene.events)
    estimated = detect_events(
        scene.audio,
        scene.sr,
        threshold=0.3,
        hop_length=512,
        min_duration_on=0.10,
        min_duration_off=0.10,
    )

    for mode in ("segment", "event"):
        scores = evaluate(reference, estimated, mode=mode)
        print(
            f"[{mode:7s}] F={scores['f_measure']:.3f}  ER={scores['error_rate']:.3f}  "
            f"P={scores['precision']:.3f}  R={scores['recall']:.3f}"
        )


if __name__ == "__main__":
    main()
