"""Windowing and framing utilities.

These are the lowest-level building blocks of the DSP stack: turning a 1-D
signal into a sequence of overlapping, tapered frames that the feature
extractors can then transform one frame at a time.
"""

from __future__ import annotations

import numpy as np

from ._typing import FloatArray
from .exceptions import InvalidParameterError

# Window tapers we know how to build. ``get_window`` is the public entry point;
# the names mirror the ones used by scipy/librosa so call sites stay familiar.
_WINDOW_NAMES = ("rectangular", "hann", "hamming", "blackman")

# TODO: expose a Tukey window once a caller actually needs one.


def get_window(name: str, length: int, *, periodic: bool = True) -> FloatArray:
    """Return a window taper of ``length`` samples.

    Parameters
    ----------
    name:
        One of ``rectangular``/``boxcar``, ``hann``, ``hamming`` or ``blackman``
        (case-insensitive).
    length:
        Number of samples. Must be positive.
    periodic:
        If ``True`` (the default) build a DFT-even window -- the right choice for
        spectral analysis, where the window is assumed to tile periodically. If
        ``False`` build a symmetric window, which is better for filter design.
    """
    if length <= 0:
        raise InvalidParameterError(f"window length must be positive, got {length}")
    if length == 1:
        return np.ones(1, dtype=np.float64)

    key = name.lower()
    denom = float(length if periodic else length - 1)
    n = np.arange(length, dtype=np.float64)

    if key in ("rectangular", "boxcar", "rect", "none"):
        return np.ones(length, dtype=np.float64)
    if key in ("hann", "hanning"):
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / denom)
    if key == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / denom)
    if key == "blackman":
        return (
            0.42
            - 0.5 * np.cos(2.0 * np.pi * n / denom)
            + 0.08 * np.cos(4.0 * np.pi * n / denom)
        )
    raise InvalidParameterError(
        f"unknown window {name!r}; choose one of {', '.join(_WINDOW_NAMES)}"
    )
