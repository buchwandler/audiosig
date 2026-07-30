from __future__ import annotations

import numpy as np
import pytest

from audiosig import AudioShapeError, InvalidParameterError
from audiosig._spectral import istft, phase_vocoder, stft


@pytest.mark.parametrize("dtype,tolerance", [(np.float32, 1e-5), (np.float64, 1e-8)])
def test_stft_istft_round_trip(dtype: type[np.floating], tolerance: float) -> None:
    rng = np.random.default_rng(42)
    source = rng.normal(size=(2, 5000)).astype(dtype)
    spectrum = stft(source, n_fft=512, hop_length=128)
    assert spectrum.shape[:2] == (2, 257)
    result = istft(spectrum, n_fft=512, hop_length=128, length=5000, dtype=np.dtype(dtype))
    assert result.shape == source.shape
    assert result.dtype == source.dtype
    assert np.max(np.abs(result - source)) <= tolerance


def test_stft_handles_short_signal_and_exact_odd_length() -> None:
    source = np.array([1.0, -0.5, 0.25], dtype=np.float64)
    spectrum = stft(source, n_fft=16, hop_length=4)
    result = istft(spectrum, n_fft=16, hop_length=4, length=3, dtype=np.dtype(np.float64))
    assert result.shape == (3,)
    np.testing.assert_allclose(result, source, atol=1e-8)


def test_spectral_validation_and_phase_vocoder_final_frame() -> None:
    source = np.ones(32, dtype=np.float64)
    with pytest.raises(InvalidParameterError):
        stft(source, n_fft=8, hop_length=9)
    with pytest.raises(AudioShapeError):
        istft(np.ones((4, 2)), n_fft=8, hop_length=2)
    with pytest.raises(AudioShapeError):
        phase_vocoder(np.ones((4, 2)), rate=1.0, hop_length=2, n_fft=8)
    with pytest.raises(AudioShapeError):
        phase_vocoder(np.ones(4), rate=1.0, hop_length=2, n_fft=8)
    spectrum = stft(source, n_fft=8, hop_length=2)
    result = phase_vocoder(spectrum, rate=2.0, hop_length=2, n_fft=8)
    assert result.shape[-1] > 0
    assert stft(np.ones(1), n_fft=8, hop_length=2).shape[-2] == 5
    padded = istft(spectrum, n_fft=8, hop_length=2, length=100, dtype=np.dtype(np.float64))
    assert padded.shape == (100,)
