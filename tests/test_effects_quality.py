"""AudioSig behavioral quality tests informed by public DSP invariants.

Some scenarios were selected after reviewing librosa's public test suite, but
this module independently defines AudioSig's contracts and fixtures.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from audiosig import InvalidParameterError, pitch_shift, time_stretch
from tests._quality_helpers import dominant_frequency, normalized_correlation

pytestmark = pytest.mark.quality


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_fft", [256, 257, 512, 513])
@pytest.mark.parametrize("rate", [0.5, 0.8, 1.0, 1.25, 2.0])
def test_time_stretch_length_dtype_and_finiteness(
    dtype: type[np.floating],
    n_fft: int,
    rate: float,
    sine_factory: Callable[..., np.ndarray],
) -> None:
    source = sine_factory(
        440.0,
        sample_rate=16_000,
        length=4096,
        dtype=dtype,
    )

    stretched = time_stretch(source, rate, n_fft=n_fft)

    assert stretched.dtype == source.dtype
    assert stretched.shape == (max(1, round(source.size / rate)),)
    assert np.all(np.isfinite(stretched))
    assert not np.shares_memory(stretched, source)


@pytest.mark.parametrize("rate", [0.5, 0.8, 1.25, 2.0])
def test_time_stretch_approximately_preserves_pitch(
    rate: float,
    sine_factory: Callable[..., np.ndarray],
) -> None:
    sample_rate = 16_000
    expected_frequency = 440.0
    source = sine_factory(
        expected_frequency,
        sample_rate=sample_rate,
        duration=1.0,
        dtype=np.float64,
    )

    stretched = time_stretch(source, rate, n_fft=1024, hop_length=256)
    measured = dominant_frequency(stretched, sample_rate)

    # FFT resolution and phase-vocoder artifacts justify a small tolerance.
    assert measured == pytest.approx(expected_frequency, abs=12.0)


def test_time_stretch_neutral_rate_returns_equal_copy(
    rng: np.random.Generator,
) -> None:
    source = rng.standard_normal((2, 1000)).astype(np.float32)

    result = time_stretch(source, 1.0)

    np.testing.assert_array_equal(result, source)
    assert result.dtype == source.dtype
    assert not np.shares_memory(result, source)


@pytest.mark.parametrize("rate", [0.75, 1.1, 1.5])
def test_time_stretch_channel_independence(
    rate: float,
    sine_factory: Callable[..., np.ndarray],
) -> None:
    sample_rate = 16_000
    left = sine_factory(220.0, sample_rate=sample_rate, length=5000)
    right = sine_factory(
        660.0,
        sample_rate=sample_rate,
        length=5000,
        amplitude=0.4,
    )
    stereo = np.stack([left, right])

    combined = time_stretch(stereo, rate, n_fft=512, hop_length=128)
    left_only = time_stretch(left, rate, n_fft=512, hop_length=128)
    right_only = time_stretch(right, rate, n_fft=512, hop_length=128)

    np.testing.assert_allclose(combined[0], left_only, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(combined[1], right_only, atol=1e-10, rtol=1e-10)
    assert not np.allclose(combined[0], combined[1])


def test_time_stretch_nonfinal_sample_axis(
    sine_factory: Callable[..., np.ndarray],
) -> None:
    source = np.stack(
        [
            sine_factory(220.0, length=3000),
            sine_factory(440.0, length=3000),
        ],
        axis=1,
    )

    stretched = time_stretch(
        source,
        1.25,
        axis=0,
        n_fft=256,
        hop_length=64,
    )

    assert stretched.shape == (round(source.shape[0] / 1.25), 2)
    np.testing.assert_allclose(
        stretched[:, 0],
        time_stretch(source[:, 0], 1.25, n_fft=256, hop_length=64),
        atol=1e-10,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_fft", [512, 513])
@pytest.mark.parametrize("semitones", [-12.0, -3.5, 0.0, 4.0, 12.0])
def test_pitch_shift_length_dtype_and_expected_frequency(
    dtype: type[np.floating],
    n_fft: int,
    semitones: float,
    sine_factory: Callable[..., np.ndarray],
) -> None:
    sample_rate = 16_000
    source_frequency = 330.0
    source = sine_factory(
        source_frequency,
        sample_rate=sample_rate,
        duration=1.0,
        dtype=dtype,
    )

    shifted = pitch_shift(
        source,
        sample_rate=sample_rate,
        semitones=semitones,
        n_fft=n_fft,
        hop_length=128,
    )

    assert shifted.shape == source.shape
    assert shifted.dtype == source.dtype
    assert np.all(np.isfinite(shifted))
    expected_frequency = source_frequency * 2.0 ** (semitones / 12.0)
    assert dominant_frequency(shifted, sample_rate) == pytest.approx(
        expected_frequency,
        abs=18.0,
    )


def test_pitch_shift_zero_returns_equal_copy(
    rng: np.random.Generator,
) -> None:
    source = rng.standard_normal(1000).astype(np.float32)

    shifted = pitch_shift(source, sample_rate=16_000, semitones=0.0)

    np.testing.assert_array_equal(shifted, source)
    assert not np.shares_memory(shifted, source)


@pytest.mark.parametrize("semitones", [-5.0, 3.0, 7.5])
def test_pitch_shift_channel_independence(
    semitones: float,
    sine_factory: Callable[..., np.ndarray],
) -> None:
    sample_rate = 16_000
    left = sine_factory(220.0, sample_rate=sample_rate, length=8000)
    right = sine_factory(
        550.0,
        sample_rate=sample_rate,
        length=8000,
        amplitude=0.5,
    )
    stereo = np.stack([left, right])

    combined = pitch_shift(
        stereo,
        sample_rate=sample_rate,
        semitones=semitones,
        n_fft=512,
        hop_length=128,
    )
    left_only = pitch_shift(
        left,
        sample_rate=sample_rate,
        semitones=semitones,
        n_fft=512,
        hop_length=128,
    )
    right_only = pitch_shift(
        right,
        sample_rate=sample_rate,
        semitones=semitones,
        n_fft=512,
        hop_length=128,
    )

    np.testing.assert_allclose(combined[0], left_only, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(combined[1], right_only, atol=1e-9, rtol=1e-9)
    assert not np.allclose(combined[0], combined[1])


def test_pitch_shift_round_trip_similarity(
    sine_factory: Callable[..., np.ndarray],
) -> None:
    sample_rate = 16_000
    source = sine_factory(
        440.0,
        sample_rate=sample_rate,
        duration=1.0,
        dtype=np.float64,
    )

    upward = pitch_shift(
        source,
        sample_rate=sample_rate,
        semitones=4.0,
        n_fft=1024,
        hop_length=256,
    )
    restored = pitch_shift(
        upward,
        sample_rate=sample_rate,
        semitones=-4.0,
        n_fft=1024,
        hop_length=256,
    )

    # Do not require waveform identity: two phase-vocoder passes are lossy.
    assert dominant_frequency(restored, sample_rate) == pytest.approx(440.0, abs=15.0)
    assert normalized_correlation(source, restored) > 0.25


def test_effect_parameter_validation() -> None:
    source = np.ones(1024, dtype=np.float32)

    for bad_rate in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(InvalidParameterError, match="rate"):
            time_stretch(source, bad_rate)

    with pytest.raises(InvalidParameterError, match="hop_length"):
        time_stretch(source, 1.1, n_fft=256, hop_length=257)

    with pytest.raises(InvalidParameterError, match="sample_rate"):
        pitch_shift(source, sample_rate=0, semitones=1.0)

    with pytest.raises(InvalidParameterError, match="bins_per_octave"):
        pitch_shift(
            source,
            sample_rate=16_000,
            semitones=1.0,
            bins_per_octave=0,
        )

    with pytest.raises(InvalidParameterError, match="semitones"):
        pitch_shift(source, sample_rate=16_000, semitones=np.nan)
