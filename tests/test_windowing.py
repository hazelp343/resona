import numpy as np
import pytest
from resona import windowing
from resona.exceptions import InvalidParameterError


def test_get_window_length_and_endpoints() -> None:
    win = windowing.get_window("hann", 64, periodic=False)
    assert win.shape == (64,)
    # A symmetric Hann window touches zero at both ends.
    assert win[0] == pytest.approx(0.0, abs=1e-9)
    assert win[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(win, win[::-1])


def test_get_window_rectangular_is_ones() -> None:
    assert np.all(windowing.get_window("boxcar", 10) == 1.0)


def test_get_window_unknown_raises() -> None:
    with pytest.raises(InvalidParameterError):
        windowing.get_window("triangle-ish", 16)


def test_frame_signal_shapes() -> None:
    signal = np.arange(100, dtype=np.float64)
    frames = windowing.frame_signal(signal, 20, 10, pad=False)
    assert frames.shape == (9, 20)
    # First frame is the head of the signal.
    assert np.array_equal(frames[0], signal[:20])
    assert np.array_equal(frames[1], signal[10:30])


def test_frame_signal_padding_covers_all_samples() -> None:
    signal = np.ones(95, dtype=np.float64)
    no_pad = windowing.frame_signal(signal, 20, 10, pad=False)
    padded = windowing.frame_signal(signal, 20, 10, pad=True)
    assert padded.shape[0] >= no_pad.shape[0]
    assert padded.shape[1] == 20


def test_num_frames_matches_frame_signal() -> None:
    for pad in (True, False):
        count = windowing.num_frames(95, 20, 10, pad=pad)
        frames = windowing.frame_signal(np.zeros(95), 20, 10, pad=pad)
        assert count == frames.shape[0]


def test_time_frame_roundtrip() -> None:
    frames = np.array([0, 5, 10, 20])
    times = windowing.frames_to_time(frames, sr=16000, hop_length=512)
    recovered = windowing.time_to_frames(times, sr=16000, hop_length=512)
    assert np.array_equal(recovered, frames)


def test_frame_signal_rejects_2d() -> None:
    with pytest.raises(InvalidParameterError):
        windowing.frame_signal(np.zeros((4, 4)), 2, 1)
