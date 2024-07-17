"""Spectral / time-domain feature embedder."""

from __future__ import annotations

import numpy as np

from .._typing import FloatArray
from ..features import (
    DEFAULT_N_FFT,
    rms,
    spectral_bandwidth,
    spectral_centroid,
    spectral_flatness,
    spectral_flux,
    spectral_rolloff,
    zero_crossing_rate,
)
from .base import BaseEmbedder


class SpectralEmbedder(BaseEmbedder):
    """A low-dimensional embedder built from classic descriptors.

    Stacks seven per-frame features -- spectral centroid, bandwidth, roll-off,
    flatness, flux, zero-crossing rate, and RMS energy -- then mean/std pools
    them. The result is a tiny, interpretable 14-D vector that is handy for quick
    visualisation and as a sanity-check baseline against learned embeddings.
    """

    name = "spectral"

    def __init__(
        self,
        *,
        sample_rate: int = 22050,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = 512,
        window_seconds: float = 0.96,
        hop_seconds: float = 0.48,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            hop_length=hop_length,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            stats=("mean", "std"),
        )
        self.n_fft = n_fft

    def _frame_features(self, signal: FloatArray) -> FloatArray:
        sr = self.sample_rate
        n_fft = self.n_fft
        hop = self.hop_length
        columns = [
            spectral_centroid(signal, sr, n_fft=n_fft, hop_length=hop),
            spectral_bandwidth(signal, sr, n_fft=n_fft, hop_length=hop),
            spectral_rolloff(signal, sr, n_fft=n_fft, hop_length=hop),
            spectral_flatness(signal, n_fft=n_fft, hop_length=hop),
            spectral_flux(signal, n_fft=n_fft, hop_length=hop),
            zero_crossing_rate(signal, frame_length=n_fft, hop_length=hop),
            rms(signal, frame_length=n_fft, hop_length=hop),
        ]
        return np.stack(columns, axis=1)
