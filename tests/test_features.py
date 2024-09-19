import numpy as np
import pytest

from resona import features


def test_stft_shape(tone_signal: np.ndarray) -> None:
    spec = features.stft(tone_signal, n_fft=1024, hop_length=256)
    assert spec.shape[1] == 1024 // 2 + 1
    assert np.iscomplexobj(spec)


def test_spectrogram_nonnegative(tone_signal: np.ndarray) -> None:
    spec = features.spectrogram(tone_signal, n_fft=1024, hop_length=256)
    assert np.all(spec >= 0)


def test_mel_to_hz_inverts_hz_to_mel() -> None:
    hz = np.array([0.0, 100.0, 1000.0, 8000.0])
    assert np.allclose(features.mel_to_hz(features.hz_to_mel(hz)), hz, atol=1e-6)


def test_mel_filterbank_shape_and_nonneg(sr: int) -> None:
    fb = features.mel_filterbank(sr, 1024, n_mels=40)
    assert fb.shape == (40, 1024 // 2 + 1)
    assert np.all(fb >= 0)
    assert np.all(fb.sum(axis=1) > 0)


def test_mel_filterbank_rejects_bad_range(sr: int) -> None:
    with pytest.raises(features.InvalidParameterError):
        features.mel_filterbank(sr, 1024, n_mels=40, fmin=4000, fmax=1000)


def test_melspectrogram_shape(tone_signal: np.ndarray, sr: int) -> None:
    mel = features.melspectrogram(tone_signal, sr, n_fft=1024, hop_length=256, n_mels=40)
    assert mel.shape[1] == 40
    assert np.all(mel >= 0)


def test_power_to_db_peak_is_zero_with_max_ref() -> None:
    spec = np.array([[1.0, 10.0, 100.0]])
    db = features.power_to_db(spec, ref=float(spec.max()))
    assert db.max() == pytest.approx(0.0)


def test_spectral_centroid_tracks_tone(tone_signal: np.ndarray, sr: int) -> None:
    centroid = features.spectral_centroid(tone_signal, sr, n_fft=2048, hop_length=512)
    assert np.median(centroid) == pytest.approx(440.0, abs=30.0)
