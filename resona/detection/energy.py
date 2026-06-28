"""Energy-gate detector for unsupervised activity detection."""

from __future__ import annotations

from dataclasses import dataclass

from .._typing import FloatArray
from ..features import rms
from .base import BaseDetector
from .events import Event
from .postprocess import activations_to_events


@dataclass
class EnergyDetector(BaseDetector):
    """Detect "sound vs. silence" from short-time energy alone.

    A model-free baseline: it frames the signal, measures RMS energy, and (by
    default) normalises it against the loudest frame so ``threshold`` acts as a
    fraction of the peak. Everything above the gate becomes an active event.
    Useful for trimming silence and for smoke-testing the post-processing chain.
    """

    n_fft: int = 2048
    hop_length: int = 512
    label: str = "active"
    relative: bool = True

    def detect(self, signal: FloatArray, sr: int) -> list[Event]:
        """Detect active regions in ``signal`` sampled at ``sr`` Hz."""
        energy = rms(signal, frame_length=self.n_fft, hop_length=self.hop_length)
        if energy.size == 0:
            return []
        activation = energy
        if self.relative:
            peak = float(energy.max())
            if peak > 0.0:
                activation = energy / peak
        return activations_to_events(
            activation,
            [self.label],
            sr=sr,
            hop_length=self.hop_length,
            threshold=self.threshold,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
            median_filter_frames=self.median_filter_frames,
        )
