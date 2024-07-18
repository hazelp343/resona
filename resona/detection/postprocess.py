"""Post-processing for frame-level detector outputs.

Raw activations are noisy: a model may flicker above and below the decision
threshold within a single real event. These helpers smooth activations, turn
them into binary decisions, and clean up the resulting on/off runs.
"""

from __future__ import annotations

import numpy as np

from .._typing import BoolArray, FloatArray
from ..exceptions import InvalidParameterError


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
