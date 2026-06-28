"""Embedding container and the embedder base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .._typing import FloatArray
from ..exceptions import InvalidParameterError
from ..io import resample, to_mono


@dataclass(frozen=True)
class Embedding:
    """A time series of embedding vectors extracted from one signal.

    ``vectors`` has shape ``(n_windows, dim)`` and ``timestamps`` gives the start
    time (seconds) of each window. The container is deliberately tiny -- just
    enough structure to keep the vectors, their timing, and their provenance
    together as they flow through the rest of the toolkit.
    """

    vectors: FloatArray
    timestamps: FloatArray
    sample_rate: int
    name: str

    @property
    def n_windows(self) -> int:
        """Number of pooled windows (rows in ``vectors``)."""
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        """Dimensionality of a single embedding vector."""
        return int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0

    def pooled(self, method: str = "mean") -> FloatArray:
        """Collapse the time axis into a single clip-level vector."""
        if self.vectors.shape[0] == 0:
            return np.zeros(self.dim, dtype=np.float64)
        if method == "mean":
            return self.vectors.mean(axis=0)
        if method == "max":
            return self.vectors.max(axis=0)
        if method == "median":
            return np.median(self.vectors, axis=0)
        raise InvalidParameterError(f"unknown pooling method {method!r}")


class BaseEmbedder(ABC):
    """Common machinery for the built-in embedders.

    Subclasses implement :meth:`_frame_features`, returning a frame-level feature
    matrix for a mono signal already at ``self.sample_rate``. The base class
    handles downmixing, resampling, and pooling those frames into fixed-length
    summary vectors over sliding windows.
    """

    name = "base"

    def __init__(
        self,
        *,
        sample_rate: int = 22050,
        hop_length: int = 512,
        window_seconds: float = 0.96,
        hop_seconds: float = 0.48,
        stats: tuple[str, ...] = ("mean", "std"),
    ) -> None:
        if sample_rate <= 0 or hop_length <= 0:
            raise InvalidParameterError("sample_rate and hop_length must be positive")
        if window_seconds <= 0 or hop_seconds <= 0:
            raise InvalidParameterError("window_seconds and hop_seconds must be positive")
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.stats = stats

    @abstractmethod
    def _frame_features(self, signal: FloatArray) -> FloatArray:
        """Frame-level features ``(n_frames, n_dims)`` for a mono signal."""
        raise NotImplementedError

    def _summarise(self, block: FloatArray) -> FloatArray:
        parts = []
        for stat in self.stats:
            if stat == "mean":
                parts.append(block.mean(axis=0))
            elif stat == "std":
                parts.append(block.std(axis=0))
            elif stat == "max":
                parts.append(block.max(axis=0))
            elif stat == "min":
                parts.append(block.min(axis=0))
            else:
                raise InvalidParameterError(f"unknown summary statistic {stat!r}")
        return np.concatenate(parts)

    def _pool_windows(self, features: FloatArray) -> tuple[FloatArray, FloatArray]:
        n_frames, n_dims = features.shape
        out_dim = len(self.stats) * n_dims
        if n_frames == 0:
            return np.zeros((0, out_dim)), np.zeros((0,))

        frame_hop_s = self.hop_length / self.sample_rate
        win = max(1, int(round(self.window_seconds / frame_hop_s)))
        hop = max(1, int(round(self.hop_seconds / frame_hop_s)))
        if n_frames <= win:
            starts = [0]
            win = n_frames
        else:
            starts = list(range(0, n_frames - win + 1, hop))

        vectors = np.stack([self._summarise(features[s : s + win]) for s in starts])
        timestamps = np.array([s * frame_hop_s for s in starts], dtype=np.float64)
        return vectors, timestamps

    def embed(self, signal: FloatArray, sr: int) -> Embedding:
        """Extract pooled embeddings from ``signal`` sampled at ``sr`` Hz."""
        sig = to_mono(np.asarray(signal, dtype=np.float64))
        if sr != self.sample_rate:
            sig = resample(sig, sr, self.sample_rate)
        features = self._frame_features(sig)
        vectors, timestamps = self._pool_windows(features)
        return Embedding(
            vectors=vectors,
            timestamps=timestamps,
            sample_rate=self.sample_rate,
            name=self.name,
        )

    def __call__(self, signal: FloatArray, sr: int) -> Embedding:
        return self.embed(signal, sr)
