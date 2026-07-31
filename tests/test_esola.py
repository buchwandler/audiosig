from __future__ import annotations

import numpy as np
import pytest

from audiosig import InvalidParameterError, time_stretch
from audiosig._esola import (
    _analysis_shift,
    _centered_moving_average,
    _extract_epochs_lane,
    _positive_zero_crossings,
    _zero_frequency_signal,
)


def speech_like(
    length: int = 3200, sample_rate: int = 16_000, dtype: type = np.float64
) -> np.ndarray:
    time = np.arange(length, dtype=np.float64) / sample_rate
    carrier = 0.35 * np.sin(2.0 * np.pi * 180.0 * time)
    excitation = 0.15 * np.sin(2.0 * np.pi * 90.0 * time) ** 3
    return (carrier + excitation).astype(dtype)


def test_centered_moving_average_uses_edge_padding() -> None:
    values = np.arange(5, dtype=np.float64)
    np.testing.assert_allclose(_centered_moving_average(values, 1), [1 / 3, 1, 2, 3, 11 / 3])


def test_four_cumulative_sums_match_two_zero_frequency_resonators() -> None:
    values = np.array([0.2, -0.4, 0.7, 0.1, -0.2], dtype=np.float64)
    difference = np.empty(values.size, dtype=np.float64)
    difference[0] = values[0]
    difference[1:] = values[1:] - values[:-1]
    cumulative = difference.copy()
    for _ in range(4):
        cumulative = np.cumsum(cumulative, dtype=np.float64)

    first_previous = 0.0
    first_previous_previous = 0.0
    first = np.empty(values.size, dtype=np.float64)
    for index, value in enumerate(difference):
        current = 2.0 * first_previous - first_previous_previous + value
        first[index] = current
        first_previous_previous, first_previous = first_previous, current
    second_previous = 0.0
    second_previous_previous = 0.0
    recurrence = np.empty(values.size, dtype=np.float64)
    for index, value in enumerate(first):
        current = 2.0 * second_previous - second_previous_previous + value
        recurrence[index] = current
        second_previous_previous, second_previous = second_previous, current

    np.testing.assert_allclose(cumulative, recurrence)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([-1.0, 1.0], [1]),
        ([-1.0, 0.0, 0.0, 1.0], [1]),
        ([1.0, -1.0, 1.0, -1.0], [2]),
        ([0.0, 0.0, 1.0], []),
    ],
)
def test_positive_zero_crossings_are_deterministic(
    values: list[float], expected: list[int]
) -> None:
    np.testing.assert_array_equal(_positive_zero_crossings(np.asarray(values)), expected)


def test_epoch_extraction_is_finite_deterministic_and_sample_rate_aware() -> None:
    source = speech_like(16_000, 16_000)
    first = _extract_epochs_lane(source, 16_000)
    second = _extract_epochs_lane(source, 16_000)
    assert first.size > 10
    assert np.all(np.diff(first) > 0)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(_zero_frequency_signal(source, 120)).all()


@pytest.mark.slow
def test_epoch_extraction_remains_finite_for_a_minute_signal() -> None:
    source = speech_like(60 * 16_000, 16_000)
    epochs = _extract_epochs_lane(source, 16_000)
    assert epochs.dtype == np.int64
    assert np.isfinite(_zero_frequency_signal(source, 120)).all()


@pytest.mark.parametrize(
    ("synthesis_epoch", "analysis_epochs", "max_shift", "expected"),
    [
        (None, [10, 20], 40, 0),
        (10, [], 40, 0),
        (10, [8, 20, 30], 40, 10),
        (15, [10, 14], 40, 0),
        (10, [100], 40, 0),
        (10, [50], 40, 40),
    ],
)
def test_analysis_shift_selects_smallest_valid_candidate(
    synthesis_epoch: int | None,
    analysis_epochs: list[int],
    max_shift: int,
    expected: int,
) -> None:
    assert _analysis_shift(synthesis_epoch, np.asarray(analysis_epochs), max_shift) == expected


@pytest.mark.parametrize("rate", [0.5, 0.75, 0.8, 1.2, 1.25, 1.5, 2.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_esola_exact_length_dtype_finite_deterministic_and_non_mutating(
    rate: float, dtype: type
) -> None:
    source = speech_like(dtype=dtype)
    original = source.copy()
    first = time_stretch(source, rate, sample_rate=16_000, method="esola")
    second = time_stretch(source, rate, sample_rate=16_000, method="esola")
    assert first.shape == (max(1, round(source.size / rate)),)
    assert first.dtype == source.dtype
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(source, original)


def test_esola_handles_edge_inputs_and_arbitrary_sample_axis() -> None:
    empty = np.empty((2, 0, 1), dtype=np.float32)
    result = time_stretch(empty, 0.75, sample_rate=16_000, method="esola", axis=1)
    assert result.shape == empty.shape
    assert not np.shares_memory(result, empty)

    constant = np.ones(7, dtype=np.float64)
    stretched = time_stretch(constant, 0.5, sample_rate=16_000, method="esola")
    np.testing.assert_allclose(stretched, 1.0)

    batch = np.stack([speech_like(3200), speech_like(3200) * 0.5])[:, :, None]
    batch_result = time_stretch(batch, 1.25, sample_rate=16_000, method="esola", axis=1)
    assert batch_result.shape == (2, round(3200 / 1.25), 1)
    np.testing.assert_allclose(batch_result[1, :, 0], batch_result[0, :, 0] * 0.5, atol=1e-10)

    one = np.array([0.25], dtype=np.float32)
    np.testing.assert_allclose(time_stretch(one, 0.5, sample_rate=16_000, method="esola"), 0.25)


def test_esola_silence_short_inputs_and_invalid_rates() -> None:
    silence = np.zeros(13, dtype=np.float32)
    stretched = time_stretch(silence, 1.5, sample_rate=16_000, method="esola")
    assert stretched.size == round(silence.size / 1.5)
    np.testing.assert_array_equal(stretched, 0.0)

    short = time_stretch(
        np.array([0.2, -0.1], dtype=np.float64),
        0.8,
        sample_rate=16_000,
        method="esola",
    )
    assert short.size == round(2 / 0.8)
    assert np.isfinite(short).all()

    with pytest.raises(InvalidParameterError, match="sample_rate"):
        time_stretch(silence, 1.2, method="esola")
    with pytest.raises(InvalidParameterError, match="interval"):
        time_stretch(silence, 2.1, sample_rate=16_000, method="esola")
