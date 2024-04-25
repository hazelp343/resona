"""Audio input/output.

The only hard dependency is the standard library's :mod:`wave` module, which is
enough to read and write uncompressed PCM WAV files. Anything else (FLAC, OGG,
...) is delegated to :mod:`soundfile` when it happens to be installed.
"""

from __future__ import annotations

# Sample widths (bytes per sample) the stdlib WAV reader can decode.
_SUPPORTED_SAMPLE_WIDTHS = (1, 2, 3, 4)
