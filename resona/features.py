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


def mel_filterbank(
    sr: int,
    n_fft: int,
    *,
    n_mels: int = DEFAULT_N_MELS,
    fmin: float = 0.0,
    fmax: float | None = None,
    norm: bool = True,
) -> FloatArray:
    """Build a triangular mel filterbank of shape ``(n_mels, n_fft // 2 + 1)``.

    Each row is a triangular band-pass filter whose centre frequencies are
    equally spaced on the mel scale. With ``norm=True`` filters use Slaney-style
    area normalisation so that wide, high-frequency bands are not over-weighted.
    """
    if fmax is None:
        fmax = sr / 2.0
    if not 0.0 <= fmin < fmax <= sr / 2.0:
        raise InvalidParameterError(
            f"require 0 <= fmin < fmax <= sr/2; got fmin={fmin}, fmax={fmax}, sr={sr}"
        )
    if n_mels <= 0:
        raise InvalidParameterError("n_mels must be positive")

    n_bins = n_fft // 2 + 1
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    mel_edges = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_edges = mel_to_hz(mel_edges)
    fdiff = np.diff(hz_edges)
    ramps = hz_edges[:, np.newaxis] - fft_freqs[np.newaxis, :]

    weights = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        lower = -ramps[m] / fdiff[m]
        upper = ramps[m + 2] / fdiff[m + 1]
        weights[m] = np.maximum(0.0, np.minimum(lower, upper))

    if norm:
        enorm = 2.0 / (hz_edges[2 : n_mels + 2] - hz_edges[:n_mels])
        weights *= enorm[:, np.newaxis]
    return weights


def melspectrogram(
    signal: FloatArray,
    sr: int,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    fmin: float = 0.0,
    fmax: float | None = None,
    power: float = 2.0,
) -> FloatArray:
    """Mel-scaled spectrogram of shape ``(n_frames, n_mels)``."""
    spec = spectrogram(signal, n_fft=n_fft, hop_length=hop_length, power=power)
    fb = mel_filterbank(sr, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return spec @ fb.T


def power_to_db(
    spec: FloatArray,
    *,
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: float | None = 80.0,
) -> FloatArray:
    """Convert a power spectrogram to a decibel (log) scale.

    ``ref`` is the reference power that maps to 0 dB; pass ``float(spec.max())``
    for a peak-normalised result. Values are floored at ``amin`` before the
    logarithm, and -- when ``top_db`` is given -- clipped to at most ``top_db``
    below the per-array peak.
    """
    if amin <= 0:
        raise InvalidParameterError("amin must be positive")
    magnitude = np.abs(np.asarray(spec, dtype=np.float64))
    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref))
    if top_db is not None:
        if top_db < 0:
            raise InvalidParameterError("top_db must be non-negative")
        if log_spec.size:
            log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


def dct_matrix(n_input: int, n_output: int) -> FloatArray:
    """Orthonormal type-II DCT basis of shape ``(n_output, n_input)``.

    Multiplying a feature matrix ``(n_frames, n_input)`` by ``basis.T`` projects
    each frame onto its first ``n_output`` cosine components -- the standard way
    to decorrelate log-mel energies into cepstral coefficients.
    """
    if n_input <= 0 or n_output <= 0:
        raise InvalidParameterError("n_input and n_output must be positive")
    basis = np.empty((n_output, n_input), dtype=np.float64)
    basis[0] = 1.0 / np.sqrt(n_input)
    samples = (2.0 * np.arange(n_input) + 1.0) * np.pi / (2.0 * n_input)
    for k in range(1, n_output):
        basis[k] = np.cos(k * samples) * np.sqrt(2.0 / n_input)
    return basis


def mfcc(
    signal: FloatArray,
    sr: int,
    *,
    n_mfcc: int = 13,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> FloatArray:
    """Mel-frequency cepstral coefficients of shape ``(n_frames, n_mfcc)``."""
    log_mel = power_to_db(
        melspectrogram(
            signal,
            sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
    )
    basis = dct_matrix(n_mels, n_mfcc)
    return log_mel @ basis.T


def delta(features: FloatArray, *, width: int = 9, order: int = 1) -> FloatArray:
    """Local-regression delta (derivative) features along the time axis.

    ``features`` is ``(n_frames, n_dims)``. The estimate at frame ``t`` is the
    slope of a least-squares line fit over a window of ``width`` frames centred
    on ``t``; edges are handled by replicating the boundary frames. Higher
    ``order`` applies the operator repeatedly (delta-delta, ...).
    """
    if width < 3 or width % 2 == 0:
        raise InvalidParameterError("width must be an odd integer >= 3")
    if order < 1:
        raise InvalidParameterError("order must be >= 1")

    out = np.asarray(features, dtype=np.float64)
    if out.ndim != 2:
        raise InvalidParameterError("features must be a 2-D (n_frames, n_dims) array")

    half = width // 2
    denom = 2.0 * sum(n * n for n in range(1, half + 1))
    for _ in range(order):
        padded = np.pad(out, ((half, half), (0, 0)), mode="edge")
        n_frames = out.shape[0]
        result = np.zeros_like(out)
        for n in range(1, half + 1):
            result += n * (
                padded[half + n : half + n + n_frames]
                - padded[half - n : half - n + n_frames]
            )
        out = result / denom
    return out
