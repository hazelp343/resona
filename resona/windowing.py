"""Windowing and framing utilities.

These are the lowest-level building blocks of the DSP stack: turning a 1-D
signal into a sequence of overlapping, tapered frames that the feature
extractors can then transform one frame at a time.
"""

from __future__ import annotations

import numpy as np

from ._typing import FloatArray, IntArray
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


def num_frames(
    n_samples: int, frame_length: int, hop_length: int, *, pad: bool = True
) -> int:
    """Number of frames produced by :func:`frame_signal`.

    With ``pad=False`` only complete frames are counted. With ``pad=True`` the
    signal is conceptually zero-padded at the end so that every sample falls in
    at least one frame.
    """
    if frame_length <= 0 or hop_length <= 0:
        raise InvalidParameterError("frame_length and hop_length must be positive")
    if n_samples <= 0:
        return 0
    if pad:
        return max(1, int(np.ceil((n_samples - frame_length) / hop_length)) + 1)
    if n_samples < frame_length:
        return 0
    return 1 + (n_samples - frame_length) // hop_length


def frame_signal(
    signal: FloatArray,
    frame_length: int,
    hop_length: int,
    *,
    pad: bool = True,
    pad_mode: str = "constant",
) -> FloatArray:
    """Split ``signal`` into overlapping frames.

    Returns a *time-major* array of shape ``(n_frames, frame_length)``. Frame
    ``i`` covers ``signal[i * hop_length : i * hop_length + frame_length]``.
    When ``pad`` is set the signal is padded at the end so the final, otherwise
    partial, frame is preserved.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise InvalidParameterError("signal must be one-dimensional")
    if frame_length <= 0 or hop_length <= 0:
        raise InvalidParameterError("frame_length and hop_length must be positive")

    n = int(signal.shape[0])
    count = num_frames(n, frame_length, hop_length, pad=pad)
    if count == 0:
        return np.empty((0, frame_length), dtype=np.float64)

    needed = (count - 1) * hop_length + frame_length
    if needed > n:
        signal = np.pad(signal, (0, needed - n), mode=pad_mode)

    # Build the frames with a strided view, then copy so callers can mutate.
    strides = (signal.strides[0] * hop_length, signal.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        signal, shape=(count, frame_length), strides=strides
    )
    return np.array(frames, dtype=np.float64)


def frames_to_samples(frames: IntArray | int, hop_length: int) -> IntArray:
    """Map frame indices to the sample offset at which each frame starts."""
    return np.atleast_1d(np.asarray(frames)).astype(np.int_) * int(hop_length)


def samples_to_frames(samples: IntArray | int, hop_length: int) -> IntArray:
    """Map sample offsets to the frame index that contains them."""
    return np.atleast_1d(np.asarray(samples)).astype(np.int_) // int(hop_length)


def frames_to_time(
    frames: IntArray | int, sr: int, hop_length: int
) -> FloatArray:
    """Map frame indices to the start time (seconds) of each frame."""
    return frames_to_samples(frames, hop_length).astype(np.float64) / float(sr)


def time_to_frames(
    times: FloatArray | float, sr: int, hop_length: int
) -> IntArray:
    """Map times (seconds) to the index of the frame that contains them."""
    samples = np.atleast_1d(np.asarray(times, dtype=np.float64)) * float(sr)
    return np.floor(samples / float(hop_length)).astype(np.int_)
