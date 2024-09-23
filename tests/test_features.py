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


def test_mfcc_shape(tone_signal: np.ndarray, sr: int) -> None:
    coeffs = features.mfcc(tone_signal, sr, n_mfcc=13, n_fft=1024, hop_length=256, n_mels=40)
    assert coeffs.shape[1] == 13


def test_delta_shape_matches_input(tone_signal: np.ndarray, sr: int) -> None:
    coeffs = features.mfcc(tone_signal, sr, n_mfcc=13, n_fft=1024, hop_length=256, n_mels=40)
    assert features.delta(coeffs).shape == coeffs.shape


def test_delta_of_linear_ramp_is_constant() -> None:
    ramp = np.linspace(0.0, 1.0, 50)[:, np.newaxis]
    deltas = features.delta(ramp, width=5)
    # Away from the replicated edges the slope estimate is constant.
    assert np.std(deltas[3:-3]) < 1e-9


def test_delta_rejects_even_width() -> None:
    with pytest.raises(features.InvalidParameterError):
        features.delta(np.zeros((10, 2)), width=8)


def test_spectral_flatness_in_unit_range(tone_signal: np.ndarray) -> None:
    flat = features.spectral_flatness(tone_signal, n_fft=1024, hop_length=256)
    assert np.all(flat >= 0.0)
    assert np.all(flat <= 1.0 + 1e-9)


def test_noise_is_flatter_than_tone(sr: int) -> None:
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(sr)
    t = np.arange(sr, dtype=np.float64) / sr
    tone = np.sin(2.0 * np.pi * 440.0 * t)
    assert np.median(features.spectral_flatness(noise)) > np.median(
        features.spectral_flatness(tone)
    )


def test_rolloff_below_nyquist(tone_signal: np.ndarray, sr: int) -> None:
    rolloff = features.spectral_rolloff(tone_signal, sr, n_fft=1024, hop_length=256)
    assert np.all(rolloff <= sr / 2 + 1e-6)


def test_zcr_and_rms(tone_signal: np.ndarray) -> None:
    zcr = features.zero_crossing_rate(tone_signal, frame_length=1024, hop_length=256)
    level = features.rms(tone_signal, frame_length=1024, hop_length=256)
    assert zcr.ndim == 1
    assert level.ndim == 1
    # RMS of a half-amplitude sine is 0.5 / sqrt(2).
    assert np.median(level) == pytest.approx(0.5 / np.sqrt(2), abs=0.05)
