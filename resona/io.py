"""Audio input/output.

The only hard dependency is the standard library's :mod:`wave` module, which is
enough to read and write uncompressed PCM WAV files. Anything else (FLAC, OGG,
...) is delegated to :mod:`soundfile` when it happens to be installed.
"""

from __future__ import annotations

import wave

import numpy as np

from ._typing import FloatArray
from .exceptions import AudioIOError

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
