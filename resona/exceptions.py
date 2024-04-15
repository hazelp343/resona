"""Exception hierarchy for resona.

Every error raised by the library derives from :class:`ResonaError`, so callers
can catch the whole family with a single ``except`` clause while still being able
to discriminate between, say, an I/O failure and a bad argument.
"""

from __future__ import annotations


class ResonaError(Exception):
    """Base class for every error raised by resona."""


class AudioIOError(ResonaError):
    """Raised when audio cannot be read from or written to disk."""


class InvalidParameterError(ResonaError, ValueError):
    """Raised when an argument falls outside its valid domain."""


class UnknownComponentError(ResonaError, LookupError):
    """Raised when a name is missing from a registry (embedder, detector, ...)."""

    def __init__(self, kind: str, name: str, available: list[str]) -> None:
        self.kind = kind
        self.name = name
        self.available = sorted(available)
        listing = ", ".join(self.available) or "(none registered)"
        super().__init__(f"unknown {kind} {name!r}; available: {listing}")
