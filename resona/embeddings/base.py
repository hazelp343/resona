"""Embedding container and the embedder base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .._typing import FloatArray


@dataclass(frozen=True)
class Embedding:
    """A time series of embedding vectors extracted from one signal."""

    vectors: FloatArray
    timestamps: FloatArray
    sample_rate: int
    name: str


class BaseEmbedder(ABC):
    """Abstract base class for embedders."""

    name = "base"

    @abstractmethod
    def embed(self, signal: FloatArray, sr: int) -> Embedding:
        """Extract embeddings from ``signal`` sampled at ``sr`` Hz."""
        raise NotImplementedError
