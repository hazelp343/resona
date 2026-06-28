import numpy as np
import pytest

from resona import io
from resona.exceptions import AudioIOError


def test_wav_roundtrip(tmp_path, tone_signal: np.ndarray, sr: int) -> None:
    path = tmp_path / "tone.wav"
    io.write_wav(str(path), tone_signal, sr)
    data, read_sr = io.read_wav(str(path))
    assert read_sr == sr
    mono = io.to_mono(data)
    assert mono.shape[0] == tone_signal.shape[0]
    # 16-bit quantisation error is ~1/32768.
    assert np.allclose(mono, tone_signal, atol=1e-3)


def test_load_audio_downmix_and_resample(tmp_path, sr: int) -> None:
    t = np.arange(sr, dtype=np.float64) / sr
    stereo = np.stack([0.3 * np.sin(2 * np.pi * 200 * t), 0.3 * np.sin(2 * np.pi * 400 * t)], axis=1)
    path = tmp_path / "stereo.wav"
    io.write_wav(str(path), stereo, sr)

    data, out_sr = io.load_audio(str(path), sr=8000, mono=True)
    assert out_sr == 8000
    assert data.ndim == 1
    assert abs(data.size - sr // 2) <= 2


def test_to_mono_passthrough_for_1d() -> None:
    x = np.arange(10, dtype=np.float64)
    assert np.array_equal(io.to_mono(x), x)


def test_resample_identity_when_rates_match(tone_signal: np.ndarray, sr: int) -> None:
    assert np.array_equal(io.resample(tone_signal, sr, sr), tone_signal)


def test_read_missing_file_raises(tmp_path) -> None:
    with pytest.raises(AudioIOError):
        io.read_wav(str(tmp_path / "does-not-exist.wav"))
