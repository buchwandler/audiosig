from __future__ import annotations

import numpy as np
import pytest

from audiosig import (
    InvalidParameterError,
    abs2,
    activity_to_intervals,
    amplitude_to_db,
    energy_based_vad,
    find_speech_bounds,
    find_speech_start,
    frame_rms,
    frame_signal,
    frames_to_samples,
    median_filter_numpy,
    non_silent_frames,
    normalized_energy_vad,
    power_to_db,
    relative_db_vad,
    spectral_flux,
    split,
    trim,
    zero_crossing_rate,
)


def test_trim_matches_centered_reference_boundaries() -> None:
    source = np.zeros(5000, dtype=np.float32)
    source[1500:3500] = 0.5

    trimmed, interval = trim(source, frame_length=256, hop_length=64)

    np.testing.assert_array_equal(interval, [1408, 3648])
    np.testing.assert_array_equal(trimmed, source[1408:3648])


def test_trim_preserves_fully_active_tail() -> None:
    source = np.ones(24_000, dtype=np.float32)

    trimmed, interval = trim(source)

    np.testing.assert_array_equal(interval, [0, source.size])
    np.testing.assert_array_equal(trimmed, source)
    assert trimmed is not source


def test_trim_empty_and_uniform_silence_reference_policies() -> None:
    empty = np.zeros(0, dtype=np.float32)
    trimmed_empty, empty_interval = trim(empty)
    assert trimmed_empty.shape == (0,)
    np.testing.assert_array_equal(empty_interval, [0, 0])

    silence = np.zeros(1000, dtype=np.float32)
    peak_relative, peak_interval = trim(silence)
    fixed_relative, fixed_interval = trim(silence, ref=1.0)

    np.testing.assert_array_equal(peak_interval, [0, silence.size])
    np.testing.assert_array_equal(peak_relative, silence)
    np.testing.assert_array_equal(fixed_interval, [0, 0])
    assert fixed_relative.shape == (0,)


def test_trim_supports_multichannel_and_nonfinal_sample_axis() -> None:
    source = np.zeros((1000, 2), dtype=np.float64)
    source[300:600, 1] = 0.25

    trimmed, interval = trim(
        source,
        frame_length=128,
        hop_length=32,
        axis=0,
        aggregate=np.max,
    )

    assert trimmed.shape == (interval[1] - interval[0], 2)
    assert interval[0] <= 300
    assert interval[1] >= 600


def test_non_silent_frames_supports_custom_aggregation() -> None:
    source = np.zeros((2, 512), dtype=np.float32)
    source[1, 160:320] = 0.5

    activity = non_silent_frames(
        source,
        frame_length=64,
        hop_length=16,
        aggregate=np.max,
    )

    assert activity.ndim == 1
    assert activity.any()


def test_activity_to_intervals_clips_and_merges_active_runs() -> None:
    activity = np.array([False, True, True, False, True, True, True, False])

    intervals = activity_to_intervals(activity, hop_length=4, sample_count=22)

    np.testing.assert_array_equal(intervals, [[4, 12], [16, 22]])
    assert intervals.dtype == np.int64


@pytest.mark.parametrize("activity", [np.array([], dtype=bool), np.zeros(4, dtype=bool)])
def test_activity_to_intervals_empty_and_silent_masks(activity: np.ndarray) -> None:
    intervals = activity_to_intervals(activity, hop_length=4, sample_count=10)

    assert intervals.shape == (0, 2)
    assert intervals.dtype == np.int64


def test_split_finds_multiple_bursts_on_arbitrary_sample_axis() -> None:
    source = np.zeros((1000, 2), dtype=np.float32)
    source[150:300, 0] = 0.5
    source[600:800, 1] = 0.5

    intervals = split(
        source,
        axis=0,
        aggregate=np.max,
        frame_length=64,
        hop_length=16,
    )

    assert intervals.shape[1] == 2
    assert intervals.dtype == np.int64
    assert np.all(intervals[:, 0] < intervals[:, 1])
    assert np.all(intervals[1:, 0] >= intervals[:-1, 1])
    assert intervals[0, 0] <= 150 and intervals[-1, 1] >= 800


def test_split_handles_fully_active_and_empty_input() -> None:
    active = np.ones(128, dtype=np.float32)
    empty = np.zeros(0, dtype=np.float32)

    _, active_interval = trim(active, frame_length=32, hop_length=8)
    intervals = split(active, frame_length=32, hop_length=8)
    empty_intervals = split(empty, frame_length=32, hop_length=8)

    np.testing.assert_array_equal(active_interval, [0, 128])
    np.testing.assert_array_equal(intervals, [[0, 128]])
    assert empty_intervals.shape == (0, 2)


def test_split_noncentered_intervals_include_active_frame_tail() -> None:
    source = np.zeros(32, dtype=np.float32)
    source[8:16] = 0.5

    intervals = split(
        source,
        top_db=20,
        frame_length=8,
        hop_length=4,
        center=False,
    )

    assert intervals[0, 0] <= 8
    assert intervals[-1, 1] >= 16


@pytest.mark.parametrize(
    ("length", "expected_frames"),
    [(0, 0), (1, 1), (9, 1), (10, 1), (11, 1)],
)
def test_frame_signal_millisecond_compatibility(length: int, expected_frames: int) -> None:
    source = np.arange(length, dtype=np.float32)

    frames = frame_signal(
        source,
        sample_rate=1000,
        frame_ms=10,
        hop_ms=5,
    )

    assert frames.shape == (expected_frames, 10)
    if length:
        assert frames[0, 0] == source[0]


def test_frame_signal_center_and_pad_end() -> None:
    source = np.arange(11, dtype=np.float32)

    centered = frame_signal(source, frame_length=4, hop_length=2, center=True)
    padded = frame_signal(source, frame_length=4, hop_length=3, pad_end=True)

    assert centered.shape == (6, 4)
    assert padded.shape == (4, 4)
    assert padded[-1, -1] == 0


def test_frame_rms_returns_centered_frame_values() -> None:
    source = np.ones(8, dtype=np.float32)

    values = frame_rms(source, frame_length=4, hop_length=2, dtype=np.float32)

    assert values.shape == (5,)
    assert np.all(values > 0)
    assert values.dtype == np.float32


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_frame_rms_honors_requested_dtype(dtype: type[np.floating]) -> None:
    source = np.array([0.0, 0.25, -0.5, 1.0], dtype=np.float32)

    values = frame_rms(
        source,
        frame_length=2,
        hop_length=1,
        center=False,
        dtype=dtype,
    )

    assert values.dtype == np.dtype(dtype)


def test_trim_float32_threshold_boundary_does_not_activate_zero_frames() -> None:
    source = np.zeros(1000, dtype=np.float32)
    source[333:666] = np.float32(0.01)

    _, interval = trim(
        source,
        top_db=60,
        ref=np.max,
        frame_length=2,
        hop_length=1,
    )

    np.testing.assert_array_equal(interval, [333, 667])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_non_silent_frames_at_exact_db_threshold_are_inactive(dtype: type[np.floating]) -> None:
    source = np.array([1.0, 0.001], dtype=dtype)

    activity = non_silent_frames(
        source,
        top_db=60.0,
        ref=1.0,
        frame_length=1,
        hop_length=1,
        center=False,
    )

    np.testing.assert_array_equal(activity, [True, False])


@pytest.mark.parametrize(
    ("length", "expected_frames"),
    [(0, 0), (1, 1), (9, 1), (10, 1), (11, 1), (20, 2), (25, 2)],
)
def test_normalized_energy_vad_short_inputs(length: int, expected_frames: int) -> None:
    source = np.ones(length, dtype=np.float32)

    activity = energy_based_vad(
        source,
        1000,
        frame_duration_ms=10,
        energy_threshold=0.5,
    )

    assert activity.shape == (expected_frames,)
    assert not activity.any()


def test_normalized_energy_vad_and_speech_start() -> None:
    source = np.concatenate(
        [
            np.zeros(10, dtype=np.float32),
            np.ones(10, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
        ]
    )

    activity = energy_based_vad(
        source,
        1000,
        frame_duration_ms=10,
        energy_threshold=0.2,
    )

    np.testing.assert_array_equal(activity, [False, True, False])
    assert (
        find_speech_start(
            source,
            1000,
            frame_duration_ms=10,
            energy_threshold=0.2,
        )
        == 10
    )


def test_normalized_energy_vad_can_pad_trailing_partial_speech() -> None:
    source = np.zeros(25, dtype=np.float32)
    source[20:] = 1.0

    compatibility = normalized_energy_vad(
        source,
        1000,
        frame_duration_ms=10,
        energy_threshold=0.2,
        pad_end=False,
    )
    padded = normalized_energy_vad(
        source,
        1000,
        frame_duration_ms=10,
        energy_threshold=0.2,
        pad_end=True,
    )

    np.testing.assert_array_equal(compatibility, [False, False])
    np.testing.assert_array_equal(padded, [False, False, True])
    np.testing.assert_array_equal(
        find_speech_bounds(
            source,
            1000,
            frame_duration_ms=10,
            energy_threshold=0.2,
            pad_end=True,
        ),
        [20, 25],
    )


def test_explicit_vad_entry_points_match_compatibility_dispatch() -> None:
    source = np.zeros(256, dtype=np.float32)
    source[80:180] = 0.5

    np.testing.assert_array_equal(
        relative_db_vad(source, frame_length=32, hop_length=16, threshold_db=30),
        energy_based_vad(source, frame_length=32, hop_length=16, threshold_db=30),
    )
    np.testing.assert_array_equal(
        normalized_energy_vad(source, 1000, frame_duration_ms=10, energy_threshold=0.2),
        energy_based_vad(source, 1000, frame_duration_ms=10, energy_threshold=0.2),
    )


def test_speech_bounds_distinguish_no_speech_from_sample_zero() -> None:
    silence = np.zeros(64, dtype=np.float32)
    speech_at_start = silence.copy()
    speech_at_start[:16] = 0.5

    np.testing.assert_array_equal(
        find_speech_bounds(silence, frame_length=16, hop_length=8),
        [0, 0],
    )
    assert find_speech_bounds(speech_at_start, frame_length=16, hop_length=8)[1] > 0


def test_db_vad_rejects_silence_and_detects_burst() -> None:
    silence = np.zeros(512, dtype=np.float32)
    burst = silence.copy()
    burst[160:320] = 0.5

    silent_mask = energy_based_vad(silence, frame_length=64, hop_length=16)
    burst_mask = energy_based_vad(burst, frame_length=64, hop_length=16)

    assert not silent_mask.any()
    assert burst_mask.any()
    assert 0 <= find_speech_start(burst, frame_length=64, hop_length=16) <= 160


def test_db_vad_accepts_nonzero_signals_below_log_floor() -> None:
    tiny = np.full(128, 1e-12, dtype=np.float32)

    activity = energy_based_vad(tiny, frame_length=32, hop_length=16)

    assert activity.any()


def test_trim_validates_parameters_for_empty_input() -> None:
    empty = np.zeros(0, dtype=np.float32)

    with pytest.raises(InvalidParameterError, match="top_db"):
        trim(empty, top_db=-1)
    with pytest.raises(InvalidParameterError, match="frame_length"):
        trim(empty, frame_length=0)


def test_decibel_helpers_abs2_and_frame_conversion() -> None:
    complex_values = np.array([1 + 2j, 3 + 4j])
    np.testing.assert_array_equal(abs2(complex_values), [5, 25])

    power_db = power_to_db(np.array([1.0, 0.01]), ref=np.max, top_db=10.0)
    amplitude_db = amplitude_to_db(np.array([1.0, 0.1]), ref=np.max)

    np.testing.assert_allclose(power_db, [0.0, -10.0])
    np.testing.assert_allclose(amplitude_db, [0.0, -20.0])
    assert frames_to_samples(3, hop_length=100) == 300
    np.testing.assert_array_equal(
        frames_to_samples(np.array([0, 1, 2]), hop_length=100, n_fft=40),
        [20, 120, 220],
    )


def test_feature_normalization_and_median_modes() -> None:
    source = np.concatenate(
        [
            np.zeros(64, dtype=np.float32),
            np.sin(np.linspace(0, 4 * np.pi, 64)).astype(np.float32),
            np.zeros(64, dtype=np.float32),
        ]
    )

    zcr = zero_crossing_rate(
        source,
        frame_length=64,
        hop_length=64,
        normalize=True,
    )
    flux = spectral_flux(
        source,
        frame_length=64,
        hop_length=64,
        window="hamming",
        normalize=True,
    )
    data = np.array([10.0, 0.0, 0.0, 0.0, 10.0])

    assert np.all((zcr >= 0) & (zcr <= 1))
    assert np.all((flux >= 0) & (flux <= 1))
    assert median_filter_numpy(data, window_size=3, mode="edge").shape == data.shape
    assert median_filter_numpy(data, window_size=3, mode="truncate").shape == data.shape


def test_trim_vad_parameter_validation() -> None:
    source = np.ones(64, dtype=np.float32)

    with pytest.raises(InvalidParameterError, match="top_db"):
        trim(source, top_db=-1)
    with pytest.raises(InvalidParameterError, match="energy_threshold"):
        energy_based_vad(source, 1000, energy_threshold=1.1)
    with pytest.raises(InvalidParameterError, match="sample_rate"):
        energy_based_vad(source, energy_threshold=0.2)
    with pytest.raises(InvalidParameterError, match="frame_ms"):
        frame_signal(source, sample_rate=1000, frame_ms=0)
    with pytest.raises(InvalidParameterError, match="hop_ms"):
        frame_signal(source, sample_rate=1000, frame_ms=10, hop_ms=0)
    with pytest.raises(InvalidParameterError, match="odd"):
        median_filter_numpy(source, size=2)
    with pytest.raises(InvalidParameterError, match="window"):
        spectral_flux(source, frame_length=16, hop_length=8, window="blackman")
