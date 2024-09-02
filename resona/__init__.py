"""resona: a NumPy-first toolkit for audio embeddings and sound-event detection.

The package root re-exports the handful of names most programs need; the
submodules (:mod:`resona.features`, :mod:`resona.metrics`, ...) hold the rest.
"""

from __future__ import annotations

from importlib import metadata as _metadata

from . import datasets, features, metrics
from .detection.energy import EnergyDetector
from .detection.events import Event, events_to_roll, roll_to_events
from .detection.threshold import ThresholdDetector
from .embeddings import (
    Embedding,
    available_embedders,
    create_embedder,
    get_embedder,
)
from .embeddings.logmel import LogMelStatsEmbedder
from .embeddings.mfcc import MFCCEmbedder
from .embeddings.spectral import SpectralEmbedder
from .eventio import load_events, save_events
from .exceptions import (
    AudioIOError,
    InvalidParameterError,
    ResonaError,
    UnknownComponentError,
)
from .io import load_audio, read_wav, write_wav
from .metrics import event_based_metrics, segment_based_metrics
from .pipeline import detect_events, evaluate, extract_embedding

try:
    __version__ = _metadata.version("resona")
except _metadata.PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "AudioIOError",
    "EnergyDetector",
    "Embedding",
    "Event",
    "InvalidParameterError",
    "LogMelStatsEmbedder",
    "MFCCEmbedder",
    "ResonaError",
    "SpectralEmbedder",
    "ThresholdDetector",
    "UnknownComponentError",
    "available_embedders",
    "create_embedder",
    "datasets",
    "detect_events",
    "evaluate",
    "event_based_metrics",
    "events_to_roll",
    "extract_embedding",
    "features",
    "get_embedder",
    "load_audio",
    "load_events",
    "metrics",
    "read_wav",
    "roll_to_events",
    "save_events",
    "segment_based_metrics",
    "write_wav",
    "__version__",
]
