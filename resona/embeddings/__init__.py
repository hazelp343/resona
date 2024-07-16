"""Audio embedding models and the embedder registry.

Embedders are looked up by name so callers (and the CLI) can select one without
importing concrete classes. New embedders register themselves here.
"""

from __future__ import annotations

from typing import Any

from ..exceptions import UnknownComponentError
from .base import BaseEmbedder, Embedding

_REGISTRY: dict[str, type[BaseEmbedder]] = {}


def register(cls: type[BaseEmbedder], name: str | None = None) -> type[BaseEmbedder]:
    """Register an embedder class under ``name`` (defaults to ``cls.name``)."""
    _REGISTRY[name or cls.name] = cls
    return cls


def available_embedders() -> list[str]:
    """Names of all registered embedders, sorted."""
    return sorted(_REGISTRY)


def get_embedder(name: str) -> type[BaseEmbedder]:
    """Look up an embedder class by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownComponentError("embedder", name, list(_REGISTRY)) from None


def create_embedder(name: str, **kwargs: Any) -> BaseEmbedder:
    """Instantiate a registered embedder by name."""
    return get_embedder(name)(**kwargs)


from .logmel import LogMelStatsEmbedder  # noqa: E402  (registered below)

register(LogMelStatsEmbedder)


__all__ = [
    "BaseEmbedder",
    "Embedding",
    "LogMelStatsEmbedder",
    "available_embedders",
    "create_embedder",
    "get_embedder",
    "register",
]
