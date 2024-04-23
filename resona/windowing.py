"""Windowing and framing utilities.

These are the lowest-level building blocks of the DSP stack: turning a 1-D
signal into a sequence of overlapping, tapered frames that the feature
extractors can then transform one frame at a time.
"""

from __future__ import annotations

# Window tapers we know how to build. ``get_window`` is the public entry point;
# the names mirror the ones used by scipy/librosa so call sites stay familiar.
_WINDOW_NAMES = ("rectangular", "hann", "hamming", "blackman")

# TODO: expose a Tukey window once a caller actually needs one.
