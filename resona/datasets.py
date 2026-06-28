"""Synthetic audio generators.

Deterministic, dependency-free signals for tests, examples, and benchmarks --
including :func:`synthetic_scene`, which drops labelled tone bursts onto a quiet
noise bed and hands back the ground-truth events so detection and metrics can be
exercised end to end without any external data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._typing import FloatArray
from .detection.events import Event
from .exceptions import InvalidParameterError


def _n_samples(duration: float, sr: int) -> int:
    if duration < 0 or sr <= 0:
        raise InvalidParameterError("duration must be >= 0 and sr > 0")
    return int(round(duration * sr))


def tone(
    frequency: float, duration: float, *, sr: int = 22050, amplitude: float = 0.5
) -> FloatArray:
    """A pure sine tone."""
    t = np.arange(_n_samples(duration, sr), dtype=np.float64) / sr
    return amplitude * np.sin(2.0 * np.pi * frequency * t)


def chirp(
    f0: float,
    f1: float,
    duration: float,
    *,
    sr: int = 22050,
    amplitude: float = 0.5,
) -> FloatArray:
    """A linear chirp sweeping from ``f0`` to ``f1`` hertz."""
    n = _n_samples(duration, sr)
    t = np.arange(n, dtype=np.float64) / sr
    rate = (f1 - f0) / duration if duration > 0 else 0.0
    phase = 2.0 * np.pi * (f0 * t + 0.5 * rate * t**2)
    return amplitude * np.sin(phase)


def white_noise(
    duration: float,
    *,
    sr: int = 22050,
    amplitude: float = 0.1,
    seed: int | None = None,
) -> FloatArray:
    """Gaussian white noise."""
    rng = np.random.default_rng(seed)
    return amplitude * rng.standard_normal(_n_samples(duration, sr))


def _apply_fade(signal: FloatArray, sr: int, fade_ms: float = 5.0) -> FloatArray:
    """Apply a short linear fade in/out to avoid click artefacts at the edges."""
    k = int(sr * fade_ms / 1000.0)
    out = signal.copy()
    if k > 0 and 2 * k < out.size:
        ramp = np.linspace(0.0, 1.0, k)
        out[:k] *= ramp
        out[-k:] *= ramp[::-1]
    return out


@dataclass(frozen=True)
class Scene:
    """A synthetic audio clip paired with its ground-truth events."""

    audio: FloatArray
    sr: int
    events: list[Event]


def synthetic_scene(
    *,
    duration: float = 5.0,
    sr: int = 22050,
    n_events: int = 4,
    noise_level: float = 0.02,
    palette: dict[str, float] | None = None,
    seed: int = 0,
) -> Scene:
    """Build a polyphonic test scene of labelled tone bursts over a noise bed."""
    rng = np.random.default_rng(seed)
    n = _n_samples(duration, sr)
    audio = noise_level * rng.standard_normal(n)

    palette = palette or {"beep": 880.0, "buzz": 220.0, "whistle": 1760.0}
    names = list(palette)
    events: list[Event] = []
    for _ in range(n_events):
        label = names[int(rng.integers(len(names)))]
        event_duration = float(rng.uniform(0.2, 0.6))
        onset = float(rng.uniform(0.0, max(1e-3, duration - event_duration)))
        burst = _apply_fade(
            tone(palette[label], event_duration, sr=sr, amplitude=0.4), sr
        )
        start = int(round(onset * sr))
        end = min(n, start + burst.size)
        audio[start:end] += burst[: end - start]
        events.append(Event(onset=onset, offset=onset + event_duration, label=label))

    events.sort(key=lambda event: (event.onset, event.label))
    return Scene(audio=audio, sr=sr, events=events)
