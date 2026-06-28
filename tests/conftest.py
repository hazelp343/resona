"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from resona import datasets


@pytest.fixture
def sr() -> int:
    return 16000


@pytest.fixture
def tone_signal(sr: int) -> np.ndarray:
    """One second of a 440 Hz sine at half amplitude."""
    t = np.arange(sr, dtype=np.float64) / sr
    return 0.5 * np.sin(2.0 * np.pi * 440.0 * t)


@pytest.fixture
def scene() -> datasets.Scene:
    """A small, deterministic synthetic scene with ground-truth events."""
    return datasets.synthetic_scene(duration=4.0, sr=16000, n_events=4, seed=5)
