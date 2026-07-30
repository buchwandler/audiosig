"""AudioSig behavioral quality tests informed by public DSP invariants.

Some scenarios were selected after reviewing librosa's public test suite, but
this module independently defines AudioSig's contracts and fixtures.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from audiosig import (
    InvalidParameterError,
    activity_to_intervals,
    frame_rms,
    split,
    trim,
)

pytestmark = pytest.mark.quality


@pytest.mark.parametrize(
    ("activity", "hop_length", "sample_count", "expected"),
    [
        (np.zeros(0, dtype=bool), 4, 0, np.empty((0, 2), dtype=np.int64)),
        (np.zeros(5, dtype=bool), 4, 20, np.empty((0, 2), dtype=np.int64)),
        (
            np.array([True, True, True], dtype=bool),
            4,
            10,
            np.array([[0, 10]], dtype=np.int64),
        ),
        (
            np.array([False, True, True, False, True], dtype=bool),
            4,
            18,
            np.array([[4, 12], [16, 18]], dtype=np.int64),
        ),
        (
            np.array([True, False, True, False], dtype=bool),
            5,
            100,
            np.array([[0, 5], [10, 15]], dtype=np.int64),
        ),
    ],
)
def test_activity_to_intervals_known_masks(
    activity: np.ndarray,
    hop_length: int,
    sample_count: int,
    expected: np.ndarray,
) -> None:
    intervals = activity_to_intervals(
        activity,
        hop_length=hop_length,
        sample_count=sample_count,
    )

    assert intervals.dtype == np.int64
    assert intervals.shape == expected.shape
    np.testing.assert_array_equal(intervals, expected)


def test_activity_to_intervals_returns_sorted_non_overlapping_ranges() -> None:
    activity = np.array(
        [False, True, True, False, True, True, True, False],
        dtype=bool,
    )

    intervals = activity_to_intervals(
        activity,
        hop_length=16,
        sample_count=101,
    )

    assert intervals.ndim == 2
    assert intervals.shape[1] == 2
    assert np.all(intervals[:, 0] < intervals[:, 1])
    assert np.all(intervals[1:, 0] >= intervals[:-1, 1])
    assert np.all((intervals >= 0) & (intervals <= 101))


def test_activity_to_intervals_validation() -> None:
    with pytest.raises(InvalidParameterError, match="one-dimensional"):
        activity_to_intervals(
            np.ones((2, 3), dtype=bool),
            hop_length=4,
            sample_count=12,
        )

    with pytest.raises(InvalidParameterError, match="hop_length"):
        activity_to_intervals(
            np.ones(3, dtype=bool),
            hop_length=0,
            sample_count=12,
        )

    with pytest.raises(InvalidParameterError, match="sample_count"):
        activity_to_intervals(
            np.ones(3, dtype=bool),
            hop_length=4,
            sample_count=-1,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_frame_rms_honors_requested_dtype(dtype: type[np.floating]) -> None:
    source = np.ones(32, dtype=dtype)

    values = frame_rms(
        source,
        frame_length=8,
        hop_length=4,
        center=False,
        dtype=dtype,
    )

    assert values.dtype == np.dtype(dtype)
    np.testing.assert_allclose(values, 1.0, atol=np.finfo(dtype).eps * 8)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_trim_float_threshold_edge_regression(dtype: type[np.floating]) -> None:
    source = np.zeros(1000, dtype=dtype)
    source[333:666] = dtype(0.01)

    trimmed, interval = trim(
        source,
        top_db=60.0,
        ref=np.max,
        frame_length=2,
        hop_length=1,
    )

    np.testing.assert_array_equal(interval, [333, 667])
    np.testing.assert_array_equal(trimmed, source[333:667])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_trim_fixed_reference_exact_threshold(dtype: type[np.floating]) -> None:
    source = np.zeros(512, dtype=dtype)
    source[128:384] = dtype(0.1)

    trimmed, interval = trim(
        source,
        top_db=20.0,
        ref=1.0,
        frame_length=2,
        hop_length=1,
    )

    # The activity predicate is strict: exactly -20 dB is not active.
    np.testing.assert_array_equal(interval, [0, 0])
    assert trimmed.size == 0


@pytest.mark.parametrize(
    ("ref", "expected_interval"),
    [
        (np.max, np.array([0, 1000], dtype=np.int64)),
        (1.0, np.array([0, 0], dtype=np.int64)),
    ],
)
def test_trim_uniform_silence_reference_policy(
    ref: float | Callable[[np.ndarray], float],
    expected_interval: np.ndarray,
) -> None:
    source = np.zeros(1000, dtype=np.float32)

    trimmed, interval = trim(source, ref=ref, frame_length=128, hop_length=32)

    np.testing.assert_array_equal(interval, expected_interval)
    assert trimmed.shape[-1] == expected_interval[1] - expected_interval[0]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("top_db", [20.0, 40.0, 60.0])
def test_trim_slice_and_bounds_invariants(
    dtype: type[np.floating],
    top_db: float,
    alternating_burst_factory: Callable[..., np.ndarray],
) -> None:
    source = np.zeros(5000, dtype=dtype)
    source[1500:3500] = alternating_burst_factory(
        2000,
        amplitude=0.25,
        dtype=dtype,
    )

    trimmed, interval = trim(
        source,
        top_db=top_db,
        frame_length=256,
        hop_length=64,
    )

    start, end = map(int, interval)
    assert 0 <= start <= end <= source.size
    np.testing.assert_array_equal(trimmed, source[start:end])
    assert start <= 1500
    assert end >= 3500
    assert 1500 - start <= 256
    assert end - 3500 <= 256


def test_split_returns_two_known_bursts(
    alternating_burst_factory: Callable[..., np.ndarray],
) -> None:
    source = np.zeros(8192, dtype=np.float32)
    expected = np.array([[1024, 2048], [4096, 5632]], dtype=np.int64)
    source[1024:2048] = alternating_burst_factory(1024)
    source[4096:5632] = alternating_burst_factory(1536)

    intervals = split(
        source,
        top_db=40.0,
        frame_length=256,
        hop_length=64,
    )

    assert intervals.shape == (2, 2)
    assert intervals.dtype == np.int64
    assert np.all(np.abs(intervals - expected) <= 256)


def test_split_fully_active_and_fixed_reference_silence() -> None:
    active = np.ones(2048, dtype=np.float32)
    silence = np.zeros(2048, dtype=np.float32)

    active_intervals = split(active, frame_length=128, hop_length=32)
    silent_intervals = split(
        silence,
        ref=1.0,
        frame_length=128,
        hop_length=32,
    )

    np.testing.assert_array_equal(active_intervals, [[0, active.size]])
    assert silent_intervals.shape == (0, 2)
    assert silent_intervals.dtype == np.int64


def test_split_clips_activity_at_final_sample(
    alternating_burst_factory: Callable[..., np.ndarray],
) -> None:
    source = np.zeros(1001, dtype=np.float32)
    source[800:] = alternating_burst_factory(201)

    intervals = split(
        source,
        top_db=40.0,
        frame_length=128,
        hop_length=32,
    )

    assert intervals.shape == (1, 2)
    assert intervals[0, 0] <= 800
    assert intervals[0, 1] == source.size


def test_split_multichannel_activity_in_one_channel(
    alternating_burst_factory: Callable[..., np.ndarray],
) -> None:
    source = np.zeros((2, 2048), dtype=np.float32)
    source[1, 600:1200] = alternating_burst_factory(600)

    intervals = split(
        source,
        aggregate=np.max,
        frame_length=128,
        hop_length=32,
    )

    assert intervals.shape == (1, 2)
    assert intervals[0, 0] <= 600
    assert intervals[0, 1] >= 1200


def test_split_supports_nonfinal_sample_axis(
    alternating_burst_factory: Callable[..., np.ndarray],
) -> None:
    source = np.zeros((2048, 2), dtype=np.float64)
    source[700:1300, 1] = alternating_burst_factory(
        600,
        dtype=np.float64,
    )

    intervals = split(
        source,
        axis=0,
        aggregate=np.max,
        frame_length=128,
        hop_length=32,
    )

    assert intervals.shape == (1, 2)
    assert intervals[0, 0] <= 700
    assert intervals[0, 1] >= 1300
    assert np.all(intervals <= source.shape[0])


def test_split_empty_input() -> None:
    source = np.zeros(0, dtype=np.float32)

    intervals = split(source)

    assert intervals.shape == (0, 2)
    assert intervals.dtype == np.int64


def test_split_parameter_validation() -> None:
    source = np.ones(64, dtype=np.float32)

    with pytest.raises(InvalidParameterError, match="top_db"):
        split(source, top_db=-1)

    with pytest.raises(InvalidParameterError, match="frame_length"):
        split(source, frame_length=0)

    with pytest.raises(InvalidParameterError, match="hop_length"):
        split(source, hop_length=0)
