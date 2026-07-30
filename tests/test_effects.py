from __future__ import annotations

import numpy as np
import pytest

from audiosig import InvalidParameterError, pitch_shift, time_stretch


def sine(frequency: float, sample_rate: int, length: int, dtype: type = np.float64) -> np.ndarray:
    time = np.arange(length) / sample_rate
    return np.sin(2 * np.pi * frequency * time).astype(dtype)


def dominant_frequency(signal: np.ndarray, sample_rate: int) -> float:
    window = np.hanning(signal.size)
    frequencies = np.fft.rfftfreq(signal.size, 1 / sample_rate)
    return float(frequencies[np.argmax(np.abs(np.fft.rfft(signal * window)))])


def test_time_stretch_duration_pitch_and_channels() -> None:
    source = sine(440, 24000, 24000, np.float32)
    stereo = np.stack([source, source * 0.5])
    faster = time_stretch(stereo, 2.0, n_fft=1024, hop_length=256)
    slower = time_stretch(stereo, 0.5, n_fft=1024, hop_length=256)
    assert faster.shape == (2, 12000)
    assert slower.shape == (2, 48000)
    assert faster.dtype == source.dtype
    assert abs(dominant_frequency(faster[0], 24000) - 440) < 20


def test_pitch_shift_preserves_duration_and_moves_frequency() -> None:
    source = sine(440, 24000, 12000, np.float32)
    higher = pitch_shift(source, sample_rate=24000, semitones=12, n_fft=1024, hop_length=256)
    lower = pitch_shift(source, sample_rate=24000, semitones=-12, n_fft=1024, hop_length=256)
    assert higher.shape == lower.shape == source.shape
    assert higher.dtype == source.dtype
    assert 820 < dominant_frequency(higher, 24000) < 940
    assert 180 < dominant_frequency(lower, 24000) < 260


def test_neutral_effects_return_copies() -> None:
    source = sine(220, 8000, 1000, np.float64)
    result = time_stretch(source, 1.0)
    assert result is not source
    np.testing.assert_array_equal(result, source)


def test_effect_parameter_edges() -> None:
    source = np.ones(32, dtype=np.float32)
    with pytest.raises(InvalidParameterError):
        time_stretch(source, 1.1, hop_length=2049)
    with pytest.raises(InvalidParameterError, match="too large"):
        time_stretch(source, 1e-320)
    short = time_stretch(source[:3], 2.0, n_fft=16, hop_length=4)
    assert short.shape == (2,)
    with pytest.raises(InvalidParameterError):
        pitch_shift(source, sample_rate=24000, semitones=np.inf)
    np.testing.assert_array_equal(pitch_shift(source, sample_rate=24000, semitones=0), source)


def test_effects_support_odd_fft_sizes_and_finite_batch_outputs() -> None:
    source = sine(440, 24000, 4000, np.float32)
    batch = np.stack([source, source * 0.5])

    stretched = time_stretch(batch, 1.1, axis=1, n_fft=2049, hop_length=512)
    shifted = pitch_shift(batch, sample_rate=24000, semitones=1, axis=1, n_fft=2049, hop_length=512)

    assert stretched.shape == (2, 3636)
    assert shifted.shape == batch.shape
    assert stretched.dtype == batch.dtype == shifted.dtype
    assert np.isfinite(stretched).all() and np.isfinite(shifted).all()
    np.testing.assert_allclose(stretched[1], stretched[0] * 0.5, atol=2e-3)
