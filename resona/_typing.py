"""Shared type aliases.

resona is NumPy-first: audio, spectra, and features are all dense ``float64``
arrays. Centralising the aliases keeps signatures readable and gives mypy a
single place to reason about array dtypes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: A real-valued array (audio samples, spectrogram, feature matrix, ...).
FloatArray = npt.NDArray[np.float64]

#: A complex-valued array, e.g. the output of an STFT.
ComplexArray = npt.NDArray[np.complex128]

#: An integer array (frame indices, sample offsets, ...).
IntArray = npt.NDArray[np.int_]

#: A boolean array (activity rolls, masks, ...).
BoolArray = npt.NDArray[np.bool_]
