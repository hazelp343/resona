import numpy as np
import pytest

from resona.detection import events as ev
from resona.detection import postprocess as pp
from resona.detection.energy import EnergyDetector
from resona.detection.events import Event
from resona.detection.threshold import ThresholdDetector
from resona.exceptions import InvalidParameterError


def test_event_rejects_inverted_interval() -> None:
    with pytest.raises(InvalidParameterError):
        Event(1.0, 0.5)


def test_find_runs() -> None:
    mask = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
    assert ev.find_runs(mask) == [(1, 3), (4, 5), (7, 8)]


def test_events_roll_roundtrip() -> None:
    source = [Event(0.5, 1.0, "dog"), Event(1.5, 2.0, "cat")]
    roll, labels = ev.events_to_roll(source, sr=16000, hop_length=512)
    back = ev.roll_to_events(roll, sr=16000, hop_length=512, labels=labels)
    assert len(back) == 2
    by_label = {event.label: event for event in back}
    assert by_label["dog"].onset == pytest.approx(0.5, abs=0.05)
    assert by_label["cat"].offset == pytest.approx(2.0, abs=0.05)


def test_median_filter_smooths_spike() -> None:
    spike = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    assert np.all(pp.median_filter(spike, 3) == 0.0)


def test_median_filter_requires_odd() -> None:
    with pytest.raises(InvalidParameterError):
        pp.median_filter(np.zeros((5, 1)), 4)


def test_binarize() -> None:
    activations = np.array([0.1, 0.6, 0.5])
    assert list(pp.binarize(activations, 0.5)) == [False, True, True]


def test_apply_min_duration_drops_short_keeps_long() -> None:
    roll = np.zeros((20, 1), dtype=bool)
    roll[5:7, 0] = True  # 2-frame run
    roll[10:18, 0] = True  # 8-frame run
    out = pp.apply_min_duration(roll, sr=100, hop_length=1, min_duration_on=0.05)
    assert not out[5:7, 0].any()
    assert out[10:18, 0].all()


def test_apply_min_duration_bridges_short_gap() -> None:
    roll = np.zeros((20, 1), dtype=bool)
    roll[2:8, 0] = True
    roll[9:15, 0] = True  # one-frame gap at index 8
    out = pp.apply_min_duration(roll, sr=100, hop_length=1, min_duration_off=0.05)
    assert out[2:15, 0].all()


def test_activations_to_events_basic() -> None:
    activations = np.zeros((30, 1))
    activations[10:20, 0] = 0.9
    events = pp.activations_to_events(activations, ["x"], sr=100, hop_length=1, threshold=0.5)
    assert len(events) == 1
    assert events[0].label == "x"


def test_energy_detector_finds_burst(sr: int) -> None:
    t = np.arange(2 * sr, dtype=np.float64) / sr
    signal = np.zeros_like(t)
    burst = slice(sr, int(1.5 * sr))
    signal[burst] = 0.6 * np.sin(2.0 * np.pi * 440.0 * t[burst])
    detector = EnergyDetector(threshold=0.3, n_fft=1024, hop_length=256, min_duration_on=0.05)

    events = detector.detect(signal, sr)
    assert len(events) == 1
    assert events[0].label == "active"
    assert events[0].onset == pytest.approx(1.0, abs=0.1)
    assert events[0].offset == pytest.approx(1.5, abs=0.1)


def test_energy_detector_handles_empty_signal(sr: int) -> None:
    assert EnergyDetector().detect(np.zeros(0), sr) == []


def test_threshold_detector_multilabel() -> None:
    activations = np.zeros((40, 2))
    activations[5:15, 0] = 0.8
    activations[20:30, 1] = 0.9
    detector = ThresholdDetector(threshold=0.5)
    events = detector.detect(activations, labels=["a", "b"], sr=100, hop_length=1)
    assert sorted(event.label for event in events) == ["a", "b"]


def test_short_event_below_min_duration_is_dropped() -> None:
    # Regression: a single-frame blip must not survive a 3-frame minimum, while a
    # 5-frame event clears the same bar.
    activations = np.zeros((30, 1))
    activations[10, 0] = 1.0
    dropped = pp.activations_to_events(
        activations, ["x"], sr=100, hop_length=1, threshold=0.5, min_duration_on=0.03
    )
    assert dropped == []

    activations[20:25, 0] = 1.0
    kept = pp.activations_to_events(
        activations, ["x"], sr=100, hop_length=1, threshold=0.5, min_duration_on=0.03
    )
    assert len(kept) == 1
