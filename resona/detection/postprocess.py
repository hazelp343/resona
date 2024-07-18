"""Post-processing for frame-level detector outputs.

Raw activations are noisy: a model may flicker above and below the decision
threshold within a single real event. These helpers smooth activations, turn
them into binary decisions, and clean up the resulting on/off runs.
"""

from __future__ import annotations

import numpy as np

from .._typing import BoolArray, FloatArray
from ..exceptions import InvalidParameterError
from .events import Event, find_runs, roll_to_events


def median_filter(activations: FloatArray, size: int) -> FloatArray:
    """Temporal median filter along the frame axis (axis 0).

    ``size`` must be a positive odd integer; ``size <= 1`` is a no-op. Accepts a
    1-D activation curve or a 2-D ``(n_frames, n_labels)`` matrix and preserves
    the input rank.
    """
    arr = np.asarray(activations, dtype=np.float64)
    if size <= 1:
        return arr
    if size % 2 == 0:
        raise InvalidParameterError("median filter size must be odd")

    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[:, np.newaxis]

    half = size // 2
    padded = np.pad(arr, ((half, half), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size, axis=0)
    smoothed = np.median(windows, axis=-1)
    return smoothed[:, 0] if squeeze else smoothed


def binarize(activations: FloatArray, threshold: float) -> BoolArray:
    """Threshold activations into a boolean activity roll (``>= threshold``)."""
    return np.asarray(activations, dtype=np.float64) >= threshold


def _frames_to_seconds(frames: int, sr: int, hop_length: int) -> float:
    """Convert a frame count to its duration in seconds."""
    return frames * hop_length / sr


def apply_min_duration(
    roll: BoolArray,
    *,
    sr: int,
    hop_length: int,
    min_duration_on: float = 0.0,
    min_duration_off: float = 0.0,
) -> BoolArray:
    """Drop too-short active runs and bridge too-short interior gaps, per label.

    Short gaps are filled first (so a brief dropout does not split one event in
    two), then runs shorter than ``min_duration_on`` are discarded.
    """
    frame_rate = sr / hop_length
    min_on = int(round(min_duration_on * frame_rate))
    min_off = int(round(min_duration_off * frame_rate))

    out = np.array(roll, dtype=bool)
    n_frames = out.shape[0]
    for column in range(out.shape[1]):
        col = out[:, column]
        if min_off > 0:
            for start, end in find_runs(~col):
                # Interior gaps only -- never extend past the signal boundary.
                if start > 0 and end < n_frames and (end - start) < min_off:
                    col[start:end] = True
        if min_on > 0:
            for start, end in find_runs(col):
                if (end - start) < min_on:
                    col[start:end] = False
    return out


def activations_to_events(
    activations: FloatArray,
    labels: list[str],
    *,
    sr: int,
    hop_length: int,
    threshold: float = 0.5,
    min_duration_on: float = 0.0,
    min_duration_off: float = 0.0,
    median_filter_frames: int = 1,
) -> list[Event]:
    """Turn a frame-level activation matrix into a list of events.

    Pipeline: optional median smoothing -> thresholding -> minimum-duration
    cleanup -> run extraction.
    """
    acts = np.asarray(activations, dtype=np.float64)
    if acts.ndim == 1:
        acts = acts[:, np.newaxis]
    if acts.shape[1] != len(labels):
        raise InvalidParameterError(
            f"activations have {acts.shape[1]} columns but {len(labels)} labels given"
        )

    if median_filter_frames > 1:
        acts = median_filter(acts, median_filter_frames)
    roll = binarize(acts, threshold)
    roll = apply_min_duration(
        roll,
        sr=sr,
        hop_length=hop_length,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
    return roll_to_events(roll, sr=sr, hop_length=hop_length, labels=list(labels))
