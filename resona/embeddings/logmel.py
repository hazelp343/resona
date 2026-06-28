"""Log-mel statistics embedder."""

from __future__ import annotations

from .._typing import FloatArray
from ..features import DEFAULT_N_FFT, melspectrogram, power_to_db
from .base import BaseEmbedder


class LogMelStatsEmbedder(BaseEmbedder):
    """Summarise log-mel energies with per-band statistics.

    Each window's embedding concatenates the mean and standard deviation of the
    log-mel spectrogram over that window. It needs no trained weights yet is a
    strong baseline for audio similarity, clustering, and nearest-neighbour
    retrieval -- the dimensionality is simply ``2 * n_mels``.
    """

    name = "logmel"

    def __init__(
        self,
        *,
        sample_rate: int = 22050,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = 512,
        n_mels: int = 64,
        fmin: float = 0.0,
        fmax: float | None = None,
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
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def _frame_features(self, signal: FloatArray) -> FloatArray:
        mel = melspectrogram(
            signal,
            self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        return power_to_db(mel)
