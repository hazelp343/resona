"""Detect active regions in a synthetic clip and compare with the ground truth.

Run with::

    python examples/detect_events.py
"""

from __future__ import annotations

from resona import datasets, detect_events


def main() -> None:
    scene = datasets.synthetic_scene(duration=6.0, sr=22050, n_events=5, seed=3)
    events = detect_events(
        scene.audio,
        scene.sr,
        threshold=0.3,
        n_fft=2048,
        hop_length=512,
        min_duration_on=0.10,
        min_duration_off=0.05,
    )

    print(f"detected {len(events)} active region(s):")
    for event in events:
        print(f"  {event.onset:6.2f} - {event.offset:6.2f} s  [{event.label}]")

    print("\nground truth:")
    for event in scene.events:
        print(f"  {event.onset:6.2f} - {event.offset:6.2f} s  [{event.label}]")


if __name__ == "__main__":
    main()
