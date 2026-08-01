from __future__ import annotations

import numpy as np
import pytest

from audiosig import InvalidParameterError, time_stretch
from audiosig._esola import (
    _analysis_shift,
    _centered_moving_average,
    _estimate_trend_window_ms,
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


@pytest.mark.parametrize("frequency", [60.0, 180.0, 400.0])
def test_adaptive_trend_window_tracks_known_pitch_period(frequency: float) -> None:
    sample_rate = 16_000
    source = np.sin(2.0 * np.pi * frequency * np.arange(sample_rate) / sample_rate)
    trend_window_ms = _estimate_trend_window_ms(source, sample_rate)
    expected = 1.5 * 1000.0 / frequency

    assert 4.0 <= trend_window_ms <= 40.0
    assert abs(trend_window_ms - np.clip(expected, 4.0, 40.0)) <= 1.5


def test_two_pass_zfr_is_the_selected_stable_epoch_variant() -> None:
    sample_rate = 16_000
    frequency = 180.0
    source = np.sin(2.0 * np.pi * frequency * np.arange(sample_rate) / sample_rate)
    one_pass = _extract_epochs_lane(source, sample_rate, detrend_passes=1)
    two_pass = _extract_epochs_lane(source, sample_rate, detrend_passes=2)

    assert two_pass.size > 10 * max(1, one_pass.size)
    assert abs(np.median(np.diff(two_pass)) - sample_rate / frequency) <= 2.0


def test_epoch_timing_follows_a_slow_f0_sweep() -> None:
    sample_rate = 16_000
    length = 2 * sample_rate
    time = np.arange(length, dtype=np.float64) / sample_rate
    start_hz, end_hz = 80.0, 300.0
    frequency = start_hz + (end_hz - start_hz) * time / time[-1]
    phase = 2.0 * np.pi * (start_hz * time + (end_hz - start_hz) * time**2 / (2.0 * time[-1]))
    epochs = _extract_epochs_lane(np.sin(phase), sample_rate)
    middle = (epochs[1:] + epochs[:-1]) / 2.0
    expected_period = sample_rate / np.interp(middle, np.arange(length), frequency)
    relative_error = np.abs(np.diff(epochs) - expected_period) / expected_period

    assert epochs.size > 100
    assert np.quantile(relative_error, 0.9) < 0.05


def test_esola_mixed_voiced_unvoiced_boundary_has_no_endpoint_hold() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    source = np.zeros_like(time)
    source[: int(0.35 * sample_rate)] = np.sin(2.0 * np.pi * 180.0 * time[: int(0.35 * sample_rate)])
    source[int(0.35 * sample_rate) : int(0.65 * sample_rate)] = np.random.default_rng(7).normal(
        0.0, 0.08, int(0.3 * sample_rate)
    )
    source[int(0.65 * sample_rate) :] = np.sin(
        2.0 * np.pi * 220.0 * time[int(0.65 * sample_rate) :]
    )
    source[-80:] = np.linspace(0.0, 1.0, 80)
    stretched = time_stretch(source, 0.75, sample_rate=sample_rate, method="esola")

    assert np.isfinite(stretched).all()
    assert np.unique(stretched[-160:]).size > 8


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
