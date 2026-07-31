from __future__ import annotations

import numpy as np
import pytest

from audiosig import (
    AudioShapeError,
    InvalidParameterError,
    resample,
    resample_speed,
    resample_to_length,
)


def test_resample_length_identity_dc_and_batch() -> None:
    source = np.full((2, 3, 1000), 0.25, dtype=np.float32)
    result = resample(source, source_rate=24000, target_rate=16000)
    assert result.shape == (2, 3, 667)
    assert result.dtype == source.dtype
    np.testing.assert_allclose(result, 0.25, atol=1e-3)
    identity = resample(source, source_rate=24000, target_rate=24000)
    assert identity is not source
    np.testing.assert_array_equal(identity, source)


def test_resample_sine_frequency_and_exact_length() -> None:
    source_rate, target_rate = 24000, 16000
    source = np.sin(2 * np.pi * 3000 * np.arange(24000) / source_rate).astype(np.float64)
    result = resample(source, source_rate=source_rate, target_rate=target_rate)
    frequencies = np.fft.rfftfreq(result.size, 1 / target_rate)
    dominant = frequencies[np.argmax(np.abs(np.fft.rfft(result * np.hanning(result.size))))]
    assert result.shape == (16000,)
    assert abs(dominant - 3000) < 20
    assert resample_to_length(source, 1234).shape == (1234,)
    np.testing.assert_array_equal(resample_to_length(source, source.size), source)
    assert resample_to_length(source, 0).shape == (0,)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_resample_to_length_public_empty_axes_and_identity(dtype: type[np.floating]) -> None:
    source = np.arange(12, dtype=dtype).reshape(2, 2, 3)
    identity = resample_to_length(source, 3, axis=-1)
    assert identity is not source
    assert not np.shares_memory(identity, source)
    np.testing.assert_array_equal(identity, source)

    transposed = np.moveaxis(source, -1, 0)
    assert resample_to_length(transposed, 5, axis=0).shape == (5, 2, 2)
    assert resample_to_length(source, 0).shape == (2, 2, 0)

    empty = np.empty((2, 0), dtype=dtype)
    empty_copy = resample_to_length(empty, 0)
    assert empty_copy.shape == empty.shape
    assert not np.shares_memory(empty_copy, empty)
    with pytest.raises(AudioShapeError):
        resample_to_length(empty, 1)


@pytest.mark.parametrize("speed", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
def test_resample_speed_exact_length_and_identity(speed: float) -> None:
    source = np.sin(np.linspace(0, 20, 101, dtype=np.float64))
    result = resample_speed(source, speed)
    assert result.shape == (max(1, round(source.size / speed)),)
    assert result.dtype == source.dtype
    assert np.isfinite(result).all()
    if speed == 1.0:
        assert not np.shares_memory(result, source)
        np.testing.assert_array_equal(result, source)


def test_resample_empty_inputs_and_validation() -> None:
    empty = np.empty((2, 0), dtype=np.float32)
    assert resample(empty, source_rate=24_000, target_rate=16_000).shape == empty.shape
    assert resample_speed(empty, 2.0).shape == empty.shape
    with pytest.raises(InvalidParameterError):
        resample_to_length(np.ones(4, dtype=np.float32), -1)
    with pytest.raises(InvalidParameterError):
        resample_to_length(np.ones(4, dtype=np.float32), 1.5)
    with pytest.raises(InvalidParameterError):
        resample_speed(np.ones(4, dtype=np.float32), 0.0)
    with pytest.raises(InvalidParameterError):
        resample_speed(np.ones(4, dtype=np.float32), np.inf)


@pytest.mark.parametrize(
    ("input_length", "source_rate", "target_rate", "round_length", "ceil_length"),
    [(2, 24000, 16000, 1, 2), (3, 16000, 24000, 4, 5), (11, 24000, 16000, 7, 8)],
)
def test_resample_length_modes_are_explicit(
    input_length: int,
    source_rate: int,
    target_rate: int,
    round_length: int,
    ceil_length: int,
) -> None:
    source = np.ones(input_length, dtype=np.float32)

    assert resample(source, source_rate=source_rate, target_rate=target_rate).size == round_length
    assert (
        resample(
            source,
            source_rate=source_rate,
            target_rate=target_rate,
            length_mode="ceil",
        ).size
        == ceil_length
    )
    with pytest.raises(InvalidParameterError, match="length_mode"):
        resample(source, source_rate=source_rate, target_rate=target_rate, length_mode="floor")
    with pytest.raises(InvalidParameterError, match="too large"):
        resample(source, source_rate=1.0, target_rate=1e308)


def test_resample_quality_and_axis_dtype_invariants() -> None:
    sample_rate, target_rate = 24_000, 16_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    source = np.stack(
        [
            np.full(sample_rate, 0.25, dtype=np.float32),
            np.sin(2 * np.pi * 1000 * time).astype(np.float32),
        ],
        axis=0,
    )

    result = resample(source, source_rate=sample_rate, target_rate=target_rate, axis=1)

    assert result.shape == (2, 16_000)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[0], 0.25, atol=1e-3)
    assert np.sqrt(np.mean(result[1] ** 2)) > 0.5
