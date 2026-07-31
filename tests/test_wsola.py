from __future__ import annotations

import numpy as np
import pytest

from audiosig import InvalidParameterError, time_stretch
from audiosig._wsola import wsola_time_stretch


def source(length: int = 2400, dtype: type = np.float32) -> np.ndarray:
    time = np.arange(length, dtype=np.float64) / 24_000
    return (0.4 * np.sin(2 * np.pi * 180 * time)).astype(dtype)


@pytest.mark.parametrize("rate", [0.5, 0.8, 1.25, 1.5, 2.0])
def test_wsola_exact_length_finiteness_and_dtype(rate: float) -> None:
    audio = source()
    result = time_stretch(audio, rate, sample_rate=24_000, method="wsola")
    assert result.shape == (max(1, round(audio.size / rate)),)
    assert result.dtype == audio.dtype
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(audio, source())


def test_wsola_neutral_rate_is_non_aliasing_copy() -> None:
    audio = source(dtype=np.float64)
    result = time_stretch(audio, 1.0, sample_rate=24_000, method="wsola")
    assert result is not audio
    assert not np.shares_memory(result, audio)
    np.testing.assert_array_equal(result, audio)


@pytest.mark.parametrize("value", [np.zeros(7, dtype=np.float32), np.ones(7, dtype=np.float64)])
def test_wsola_constant_and_silence_are_finite(value: np.ndarray) -> None:
    result = time_stretch(value, 1.25, sample_rate=24_000, method="wsola")
    assert np.isfinite(result).all()
    assert result.dtype == value.dtype
    if np.all(value == 0):
        np.testing.assert_allclose(result, 0.0)


@pytest.mark.parametrize("length", [1, 3, 17, 239])
def test_wsola_short_inputs_and_final_partial_frame(length: int) -> None:
    audio = source(length)
    result = time_stretch(
        audio,
        0.8,
        sample_rate=24_000,
        method="wsola",
        n_fft=32,
        hop_length=8,
    )
    assert result.size == max(1, round(length / 0.8))
    assert np.isfinite(result).all()


def test_wsola_supports_non_final_sample_axis_and_independent_lanes() -> None:
    mono = source()
    batch = np.stack([mono, mono * 0.5])[:, :, None]
    result = time_stretch(
        batch,
        1.2,
        sample_rate=24_000,
        method="wsola",
        axis=1,
    )
    assert result.shape == (2, round(mono.size / 1.2), 1)
    np.testing.assert_allclose(result[1, :, 0], result[0, :, 0] * 0.5, atol=1e-6)


def test_wsola_is_deterministic_and_requires_sample_rate() -> None:
    audio = source()
    first = time_stretch(audio, 1.1, sample_rate=24_000, method="wsola")
    second = time_stretch(audio, 1.1, sample_rate=24_000, method="wsola")
    np.testing.assert_array_equal(first, second)
    with pytest.raises(InvalidParameterError, match="sample_rate"):
        time_stretch(audio, 1.1, method="wsola")


def test_wsola_rejects_invalid_frame_relationships() -> None:
    audio = source()
    with pytest.raises(InvalidParameterError, match="overlap"):
        wsola_time_stretch(
            audio,
            rate=1.1,
            sample_rate=24_000,
            overlap_ms=30.0,
        )
    with pytest.raises(InvalidParameterError, match="search"):
        wsola_time_stretch(
            audio,
            rate=1.1,
            sample_rate=24_000,
            search_ms=-1.0,
        )
