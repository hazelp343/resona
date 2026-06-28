"""Spectral and time-domain features.

Everything here operates on mono ``float64`` signals and returns *time-major*
arrays shaped ``(n_frames, n_features)`` -- the same orientation used by the
embedding and detection layers, so features compose without transposes.
"""

from __future__ import annotations

import numpy as np

from ._typing import ComplexArray, FloatArray
from .exceptions import InvalidParameterError
from .windowing import frame_signal, get_window

#: Default FFT size for spectral analysis (~46 ms at 44.1 kHz).
DEFAULT_N_FFT = 2048

#: Default hop between successive frames (75% overlap at ``DEFAULT_N_FFT``).
DEFAULT_HOP_LENGTH = 512

#: Default number of mel bands for the mel-spectrogram / MFCC front-ends.
DEFAULT_N_MELS = 128


def stft(
    signal: FloatArray,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    window: str = "hann",
    center: bool = True,
) -> ComplexArray:
    """Short-time Fourier transform.

    Returns a complex array of shape ``(n_frames, n_fft // 2 + 1)`` -- one row
    per frame, one column per non-negative frequency bin. With ``center=True``
    the signal is reflect-padded by ``n_fft // 2`` so that frame ``i`` is
    centred on sample ``i * hop_length``.
    """
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim != 1:
        raise InvalidParameterError("stft expects a 1-D signal")
    if n_fft <= 0 or hop_length <= 0:
        raise InvalidParameterError("n_fft and hop_length must be positive")

    if center:
        pad = n_fft // 2
        mode = "reflect" if sig.shape[0] > pad else "constant"
        sig = np.pad(sig, pad, mode=mode)

    win = get_window(window, n_fft, periodic=True)
    frames = frame_signal(sig, n_fft, hop_length, pad=True)
    if frames.shape[0] == 0:
        return np.empty((0, n_fft // 2 + 1), dtype=np.complex128)
    return np.fft.rfft(frames * win, n=n_fft, axis=1).astype(np.complex128)


def magnitude(spectrum: ComplexArray) -> FloatArray:
    """Element-wise magnitude ``|z|`` of a complex spectrum."""
    return np.abs(np.asarray(spectrum)).astype(np.float64)


def spectrogram(
    signal: FloatArray,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    window: str = "hann",
    center: bool = True,
    power: float = 2.0,
) -> FloatArray:
    """Magnitude (``power=1``) or power (``power=2``) spectrogram.

    Shape is ``(n_frames, n_fft // 2 + 1)``.
    """
    if power <= 0:
        raise InvalidParameterError("power must be positive")
    mag = magnitude(
        stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=center,
        )
    )
    if power == 1.0:
        return mag
    return mag**power


def hz_to_mel(frequencies: FloatArray | float) -> FloatArray:
    """Convert frequencies in hertz to the HTK mel scale."""
    freq = np.asarray(frequencies, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_hz(mels: FloatArray | float) -> FloatArray:
    """Inverse of :func:`hz_to_mel` (HTK mel scale)."""
    mel = np.asarray(mels, dtype=np.float64)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
