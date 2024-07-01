"""Audio input/output.

The only hard dependency is the standard library's :mod:`wave` module, which is
enough to read and write uncompressed PCM WAV files. Anything else (FLAC, OGG,
...) is delegated to :mod:`soundfile` when it happens to be installed.
"""

from __future__ import annotations

import os
import wave

import numpy as np

from ._typing import FloatArray
from .exceptions import AudioIOError, InvalidParameterError

# Sample widths (bytes per sample) the stdlib WAV reader can decode.
_SUPPORTED_SAMPLE_WIDTHS = (1, 2, 3, 4)


def _decode_pcm(raw: bytes, sample_width: int) -> FloatArray:
    """Decode interleaved little-endian PCM bytes to ``float64`` in ``[-1, 1]``."""
    if sample_width == 1:
        # 8-bit WAV is unsigned with a midpoint of 128.
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        return (arr - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    # 24-bit: assemble signed values from three little-endian bytes per sample.
    triples = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    vals = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    vals = np.where(vals >= (1 << 23), vals - (1 << 24), vals)
    return vals.astype(np.float64) / float(1 << 23)


def read_wav(path: str) -> tuple[FloatArray, int]:
    """Read a PCM WAV file.

    Returns ``(data, sample_rate)`` where ``data`` has shape
    ``(n_samples, n_channels)`` and values in ``[-1, 1]``.
    """
    try:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    except (wave.Error, OSError, EOFError) as exc:
        raise AudioIOError(f"could not read WAV file {path!r}: {exc}") from exc

    if sample_width not in _SUPPORTED_SAMPLE_WIDTHS:
        raise AudioIOError(f"unsupported sample width: {sample_width} byte(s)")

    flat = _decode_pcm(raw, sample_width)
    data = flat.reshape(-1, n_channels)
    return data, sample_rate


def to_mono(signal: FloatArray) -> FloatArray:
    """Collapse a multi-channel signal to mono by averaging channels.

    Accepts a 1-D array (returned unchanged) or a 2-D ``(n_samples, n_channels)``
    array with channels in the last axis -- the layout :func:`read_wav` produces.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x.mean(axis=1)
    raise InvalidParameterError("signal must be 1-D or 2-D (n_samples, n_channels)")


def resample(signal: FloatArray, sr_in: int, sr_out: int) -> FloatArray:
    """Resample a mono signal from ``sr_in`` to ``sr_out`` by linear interpolation.

    Linear interpolation is cheap and dependency-free; it is good enough for
    feature extraction, though not for high-fidelity playback. Pass a 1-D array.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise InvalidParameterError("resample expects a 1-D signal; downmix first")
    if sr_in <= 0 or sr_out <= 0:
        raise InvalidParameterError("sample rates must be positive")
    if sr_in == sr_out or x.size == 0:
        return x

    n_out = int(round(x.size * sr_out / sr_in))
    t_old = np.arange(x.size, dtype=np.float64) / float(sr_in)
    t_new = np.arange(n_out, dtype=np.float64) / float(sr_out)
    return np.interp(t_new, t_old, x)


def _read_soundfile(path: str) -> tuple[FloatArray, int]:
    """Read a non-WAV file through the optional ``soundfile`` dependency."""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - exercised only without soundfile
        raise AudioIOError(
            f"reading {path!r} requires the optional 'soundfile' dependency; "
            "install resona[soundfile] or convert the file to WAV first"
        ) from exc
    try:
        data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    except RuntimeError as exc:
        raise AudioIOError(f"could not read audio file {path!r}: {exc}") from exc
    return np.asarray(data, dtype=np.float64), int(sr)


def load_audio(
    path: str, *, sr: int | None = None, mono: bool = True
) -> tuple[FloatArray, int]:
    """Load an audio file, optionally downmixing to mono and resampling.

    WAV files are decoded with the standard library; everything else is routed
    through :mod:`soundfile` if it is installed. Returns ``(signal, sample_rate)``.
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        data, native_sr = read_wav(path)
    else:
        data, native_sr = _read_soundfile(path)

    if mono:
        data = to_mono(data)

    if sr is not None and sr != native_sr:
        if data.ndim == 1:
            data = resample(data, native_sr, sr)
        else:
            channels = [resample(data[:, c], native_sr, sr) for c in range(data.shape[1])]
            data = np.stack(channels, axis=1)
        native_sr = sr

    return data, native_sr
