"""Spectral and time-domain features.

Everything here operates on mono ``float64`` signals and returns *time-major*
arrays shaped ``(n_frames, n_features)`` -- the same orientation used by the
embedding and detection layers, so features compose without transposes.
"""

from __future__ import annotations

#: Default FFT size for spectral analysis (~46 ms at 44.1 kHz).
DEFAULT_N_FFT = 2048

#: Default hop between successive frames (75% overlap at ``DEFAULT_N_FFT``).
DEFAULT_HOP_LENGTH = 512

#: Default number of mel bands for the mel-spectrogram / MFCC front-ends.
DEFAULT_N_MELS = 128
