"""MFCC embedder with optional delta features."""

from __future__ import annotations

import numpy as np

from .._typing import FloatArray
from ..features import DEFAULT_N_FFT, delta, mfcc
from .base import BaseEmbedder


class MFCCEmbedder(BaseEmbedder):
    """Embed audio with mean/std-pooled MFCCs.

    MFCCs are the classic compact timbral descriptor. Enabling ``include_deltas``
    appends first-order temporal derivatives, which captures how the spectral
    envelope moves -- useful for transient, percussive events.
    """

    name = "mfcc"

    def __init__(
        self,
        *,
        sample_rate: int = 22050,
        n_mfcc: int = 20,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = 512,
        n_mels: int = 64,
        fmin: float = 0.0,
        fmax: float | None = None,
        include_deltas: bool = True,
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
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.include_deltas = include_deltas

    def _frame_features(self, signal: FloatArray) -> FloatArray:
        coeffs = mfcc(
            signal,
            self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        if self.include_deltas and coeffs.shape[0] >= 3:
            coeffs = np.concatenate([coeffs, delta(coeffs)], axis=1)
        return coeffs
