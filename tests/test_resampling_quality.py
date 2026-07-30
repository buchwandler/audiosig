"""AudioSig behavioral quality tests informed by public DSP invariants.

Some scenarios were selected after reviewing librosa's public test suite, but
this module independently defines AudioSig's contracts and fixtures.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from audiosig import InvalidParameterError, resample
from audiosig._resampling import resample_to_length
from tests._quality_helpers import dominant_frequency, scalar_rms, tone_amplitude


pytestmark = pytest.mark.quality


@pytest.mark.parametrize(
    ("input_length", "source_rate", "target_rate", "expected_length"),
    [
        (1, 16_000, 16_000, 1),
        (2, 24_000, 16_000, 1),
        (3, 16_000, 24_000, 4),
        (11, 24_000, 16_000, 7),
        (101, 24_000, 16_000, 67),
        (16_000, 16_000, 24_000, 24_000),
    ],
)
def test_resample_round_length_policy(
    input_length: int,
    source_rate: int,
    target_rate: int,
    expected_length: int,
) -> None:
    source = np.zeros(input_length, dtype=np.float32)

    result = resample(
        source,
        source_rate=source_rate,
        target_rate=target_rate,
    )

    assert result.shape == (expected_length,)


@pytest.mark.parametrize(
    ("input_length", "source_rate", "target_rate", "expected_length"),
    [
        (2, 24_000, 16_000, 2),
        (3, 16_000, 24_000, 5),
        (11, 24_000, 16_000, 8),
        (101, 24_000, 16_000, 68),
    ],
)
def test_resample_ceil_length_policy_when_requested(
    input_length: int,
    source_rate: int,
    target_rate: int,
    expected_length: int,
) -> None:
    source = np.zeros(input_length, dtype=np.float32)

    result = resample(
        source,
        source_rate=source_rate,
        target_rate=target_rate,
        length_mode="ceil",
    )

    assert result.shape == (expected_length,)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_resample_identity_returns_equal_copy(dtype: type[np.floating]) -> None:
    source = np.linspace(-1.0, 1.0, 1000, dtype=dtype)

    result = resample(source, source_rate=16_000, target_rate=16_000)

    np.testing.assert_array_equal(result, source)
    assert result.dtype == source.dtype
    assert not np.shares_memory(result, source)


@pytest.mark.parametrize(
    ("source_rate", "target_rate"),
    [(8_000, 16_000), (16_000, 8_000), (16_000, 24_000), (24_000, 16_000)],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_resample_dc_preservation(
    source_rate: int,
    target_rate: int,
    dtype: type[np.floating],
) -> None:
    source = np.full(source_rate, 0.375, dtype=dtype)

    result = resample(
        source,
        source_rate=source_rate,
        target_rate=target_rate,
    )

    assert result.dtype == source.dtype
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, 0.375, atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize(
    ("source_rate", "target_rate", "tone"),
    [
        (24_000, 16_000, 1_000.0),
        (24_000, 16_000, 6_000.0),
        (16_000, 24_000, 1_000.0),
        (16_000, 24_000, 7_000.0),
    ],
)
def test_resample_passband_frequency_and_amplitude(
    source_rate: int,
    target_rate: int,
    tone: float,
    sine_factory: Callable[..., np.ndarray],
) -> None:
    source = sine_factory(
        tone,
        sample_rate=source_rate,
        duration=1.0,
        amplitude=0.8,
        dtype=np.float64,
    )

    result = resample(
        source,
        source_rate=source_rate,
        target_rate=target_rate,
    )

    measured_frequency = dominant_frequency(result, target_rate)
    measured_amplitude = tone_amplitude(result, target_rate, tone)

    assert measured_frequency == pytest.approx(tone, abs=3.0)
    assert measured_amplitude == pytest.approx(0.8, rel=0.12, abs=0.04)


@pytest.mark.slow
@pytest.mark.parametrize("tone", [9_000.0, 10_000.0, 11_000.0])
def test_downsample_rejects_above_nyquist_tones(
    tone: float,
    sine_factory: Callable[..., np.ndarray],
) -> None:
    source_rate = 24_000
    target_rate = 16_000
    source = sine_factory(
        tone,
        sample_rate=source_rate,
        duration=1.0,
        amplitude=1.0,
        dtype=np.float64,
    )

    result = resample(
        source,
        source_rate=source_rate,
        target_rate=target_rate,
        filter_width=32,
        rolloff=0.945,
    )

    # Use RMS because the aliased frequency depends on the input tone.
    # Target: at least about 40 dB attenuation for the default kernel.
    assert scalar_rms(result[256:-256]) < 0.01


@pytest.mark.slow
def test_downsample_transition_band_is_bounded(
    sine_factory: Callable[..., np.ndarray],
) -> None:
    source_rate = 24_000
    target_rate = 16_000
    source = sine_factory(
        7_000.0,
        sample_rate=source_rate,
        duration=1.0,
        amplitude=1.0,
        dtype=np.float64,
    )

    result = resample(
        source,
        source_rate=source_rate,
        target_rate=target_rate,
    )

    # This is near the default 0.945 * 8 kHz cutoff. Keep a broad bound,
    # but prevent severe attenuation such as the previously observed ~10% loss.
    amplitude = tone_amplitude(result[256:-256], target_rate, 7_000.0)
    assert 0.80 <= amplitude <= 1.05


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_resample_batch_shape_axis_and_independence(axis: int) -> None:
    base = np.stack(
        [
            np.linspace(-1.0, 1.0, 1000),
            np.linspace(0.5, -0.5, 1000),
        ]
    )
    source = base.T if axis == 0 else base
    normalized_axis = axis % source.ndim

    combined = resample(
        source,
        source_rate=1000,
        target_rate=750,
        axis=axis,
    )

    assert combined.shape[normalized_axis] == 750
    channel_axis = 1 - normalized_axis
    for channel in range(source.shape[channel_axis]):
        single = np.take(source, channel, axis=channel_axis)
        expected = resample(single, source_rate=1000, target_rate=750)
        actual = np.take(combined, channel, axis=channel_axis)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("target_length", [1, 7, 1000, 1500])
def test_resample_to_length_is_exact_and_preserves_dtype(
    target_length: int,
    rng: np.random.Generator,
) -> None:
    source = rng.standard_normal(1000).astype(np.float32)

    result = resample_to_length(source, target_length)

    assert result.shape == (target_length,)
    assert result.dtype == source.dtype
    assert np.all(np.isfinite(result))


def test_resample_crosses_internal_chunk_boundary(rng: np.random.Generator) -> None:
    source = rng.standard_normal(4000).astype(np.float32)
    result = resample(source, source_rate=4000, target_rate=3000)
    assert result.shape == (3000,)
    assert np.all(np.isfinite(result))


def test_resample_validation() -> None:
    source = np.ones(100, dtype=np.float32)

    for bad_rate in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(InvalidParameterError):
            resample(source, source_rate=bad_rate, target_rate=16_000)

    with pytest.raises(InvalidParameterError, match="filter_width"):
        resample(
            source,
            source_rate=16_000,
            target_rate=8_000,
            filter_width=0,
        )

    with pytest.raises(InvalidParameterError, match="rolloff"):
        resample(
            source,
            source_rate=16_000,
            target_rate=8_000,
            rolloff=1.1,
        )
