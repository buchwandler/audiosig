"""AudioSig behavioral quality tests informed by public DSP invariants.

Some scenarios were selected after reviewing librosa's public test suite, but
this module independently defines AudioSig's contracts and fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from audiosig import (
    InvalidParameterError,
    energy_based_vad,
    find_speech_bounds,
    find_speech_start,
)


pytestmark = pytest.mark.quality


@pytest.mark.parametrize(
    ("length", "pad_end", "expected_frames"),
    [
        (0, False, 0),
        (0, True, 0),
        (1, False, 1),
        (1, True, 1),
        (10, False, 1),
        (10, True, 1),
        (11, False, 1),
        (11, True, 2),
        (20, False, 2),
        (20, True, 2),
        (25, False, 2),
        (25, True, 3),
    ],
)
def test_normalized_vad_frame_count_policy(
    length: int,
    pad_end: bool,
    expected_frames: int,
) -> None:
    source = np.zeros(length, dtype=np.float32)

    activity = energy_based_vad(
        source,
        1000,
        frame_duration_ms=10.0,
        energy_threshold=0.2,
        pad_end=pad_end,
    )

    assert activity.dtype == np.bool_
    assert activity.shape == (expected_frames,)


def test_normalized_vad_detects_trailing_partial_speech_when_padded() -> None:
    source = np.zeros(25, dtype=np.float32)
    source[20:] = 1.0

    compatibility_mask = energy_based_vad(
        source,
        1000,
        frame_duration_ms=10.0,
        energy_threshold=0.2,
        pad_end=False,
    )
    safe_mask = energy_based_vad(
        source,
        1000,
        frame_duration_ms=10.0,
        energy_threshold=0.2,
        pad_end=True,
    )

    np.testing.assert_array_equal(compatibility_mask, [False, False])
    np.testing.assert_array_equal(safe_mask, [False, False, True])


def test_normalized_vad_constant_energy_is_inactive() -> None:
    source = np.full(40, 0.5, dtype=np.float32)

    activity = energy_based_vad(
        source,
        1000,
        frame_duration_ms=10.0,
        energy_threshold=0.0,
        pad_end=True,
    )

    # Min-max normalization of a constant sequence is all zero.
    assert not np.any(activity)


def test_normalized_vad_multichannel_is_channel_local() -> None:
    source = np.zeros((2, 40), dtype=np.float32)
    source[0, 10:20] = 1.0
    source[1, 20:30] = 1.0

    activity = energy_based_vad(
        source,
        1000,
        frame_duration_ms=10.0,
        energy_threshold=0.2,
        pad_end=True,
    )

    assert activity.shape == (2, 4)
    np.testing.assert_array_equal(activity[0], [False, True, False, False])
    np.testing.assert_array_equal(activity[1], [False, False, True, False])


def test_find_speech_start_propagates_tail_padding() -> None:
    source = np.zeros(25, dtype=np.float32)
    source[20:] = 1.0

    without_padding = find_speech_start(
        source,
        1000,
        frame_duration_ms=10.0,
        energy_threshold=0.2,
        pad_end=False,
    )
    with_padding = find_speech_start(
        source,
        1000,
        frame_duration_ms=10.0,
        energy_threshold=0.2,
        pad_end=True,
    )

    assert without_padding == 0
    assert with_padding == 20


def test_find_speech_bounds_distinguishes_absence_from_sample_zero() -> None:
    absent = np.zeros(30, dtype=np.float32)
    starts_at_zero = np.concatenate(
        [
            np.ones(10, dtype=np.float32),
            np.zeros(20, dtype=np.float32),
        ]
    )

    absent_bounds = find_speech_bounds(
        absent,
        1000,
        frame_duration_ms=10.0,
        energy_threshold=0.2,
        pad_end=True,
    )
    active_bounds = find_speech_bounds(
        starts_at_zero,
        1000,
        frame_duration_ms=10.0,
        energy_threshold=0.2,
        pad_end=True,
    )

    np.testing.assert_array_equal(absent_bounds, [0, 0])
    np.testing.assert_array_equal(active_bounds, [0, 10])
    assert absent_bounds.dtype == np.int64
    assert active_bounds.dtype == np.int64


def test_find_speech_bounds_clips_partial_final_frame() -> None:
    source = np.zeros(25, dtype=np.float32)
    source[20:] = 1.0

    bounds = find_speech_bounds(
        source,
        1000,
        frame_duration_ms=10.0,
        energy_threshold=0.2,
        pad_end=True,
    )

    np.testing.assert_array_equal(bounds, [20, 25])


def test_find_speech_bounds_aggregates_channels() -> None:
    source = np.zeros((2, 50), dtype=np.float32)
    source[0, 10:20] = 1.0
    source[1, 30:50] = 1.0

    bounds = find_speech_bounds(
        source,
        1000,
        frame_duration_ms=10.0,
        energy_threshold=0.2,
        pad_end=True,
    )

    np.testing.assert_array_equal(bounds, [10, 50])


def test_db_vad_detects_burst_relative_to_peak() -> None:
    source = np.zeros(4096, dtype=np.float64)
    source[1024:3072] = 1e-8

    activity = energy_based_vad(
        source,
        frame_length=256,
        hop_length=64,
        threshold_db=20.0,
    )

    assert activity.ndim == 1
    assert np.any(activity)
    active_frames = np.flatnonzero(activity)
    assert active_frames[0] * 64 <= 1024
    assert active_frames[-1] * 64 + 256 >= 3072


def test_db_vad_all_zero_is_inactive() -> None:
    source = np.zeros(4096, dtype=np.float32)

    activity = energy_based_vad(
        source,
        frame_length=256,
        hop_length=64,
        threshold_db=80.0,
    )

    assert activity.dtype == np.bool_
    assert not np.any(activity)


def test_vad_validation_for_tail_policy_and_thresholds() -> None:
    source = np.ones(32, dtype=np.float32)

    with pytest.raises(InvalidParameterError, match="energy_threshold"):
        energy_based_vad(
            source,
            1000,
            frame_duration_ms=10.0,
            energy_threshold=1.1,
            pad_end=True,
        )

    with pytest.raises(InvalidParameterError, match="sample_rate"):
        energy_based_vad(source, energy_threshold=0.2)

    with pytest.raises(InvalidParameterError, match="threshold_db"):
        energy_based_vad(source, threshold_db=-1.0)
