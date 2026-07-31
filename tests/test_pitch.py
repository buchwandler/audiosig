from __future__ import annotations

import numpy as np
import pytest

import audiosig._pitch as pitch
from audiosig import InvalidParameterError
from audiosig._pitch import estimate_pitch_track, estimate_pitch_track_lane


def _tone(sample_rate: int, frequency: float, seconds: float = 0.5) -> np.ndarray:
    time = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * frequency * time)


@pytest.mark.quality
@pytest.mark.parametrize("sample_rate", (8_000, 16_000, 24_000, 48_000))
@pytest.mark.parametrize("frequency", (80.0, 120.0, 220.0, 440.0))
def test_pitch_tracker_recovers_clean_tones(sample_rate: int, frequency: float) -> None:
    track = estimate_pitch_track_lane(
        _tone(sample_rate, frequency),
        sample_rate=sample_rate,
    )
    values = track.frequencies[track.voiced]
    assert values.size >= 3
    assert np.median(values) == pytest.approx(frequency, rel=0.03)
    assert np.all(np.diff(track.pitch_marks) > 0)
    assert np.all((track.pitch_marks >= 0) & (track.pitch_marks < round(sample_rate * 0.5)))


def test_pitch_tracker_is_silent_for_silence_and_mostly_unvoiced_noise() -> None:
    silence = estimate_pitch_track_lane(np.zeros(16_000), sample_rate=16_000)
    assert not np.any(silence.voiced)
    assert silence.pitch_marks.size == 0

    noise = np.random.default_rng(20260731).normal(0.0, 0.1, 16_000)
    noisy = estimate_pitch_track_lane(noise, sample_rate=16_000)
    assert float(np.mean(noisy.voiced)) < 0.10


def test_pitch_tracker_is_amplitude_invariant_and_deterministic() -> None:
    source = _tone(24_000, 220.0) + 0.25 * _tone(24_000, 440.0)
    first = estimate_pitch_track_lane(source, sample_rate=24_000)
    second = estimate_pitch_track_lane(source * 0.1, sample_rate=24_000)
    third = estimate_pitch_track_lane(source, sample_rate=24_000)
    np.testing.assert_array_equal(first.voiced, second.voiced)
    np.testing.assert_array_equal(first.pitch_marks, second.pitch_marks)
    np.testing.assert_array_equal(first.pitch_marks, third.pitch_marks)
    np.testing.assert_array_equal(first.frequencies, third.frequencies)


@pytest.mark.parametrize(
    ("sample_rate", "pitch_floor", "pitch_ceiling"),
    ((0, 60.0, 500.0), (16_000, 0.0, 500.0), (16_000, 500.0, 500.0), (16_000, 60.0, 8_000.0)),
)
def test_pitch_tracker_rejects_invalid_parameters(
    sample_rate: int,
    pitch_floor: float,
    pitch_ceiling: float,
) -> None:
    with pytest.raises(InvalidParameterError):
        estimate_pitch_track_lane(
            np.zeros(256),
            sample_rate=sample_rate,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
        )


def test_pitch_private_edge_helpers_and_empty_lanes() -> None:
    assert pitch._frame_starts(0, 32, 8).size == 0
    np.testing.assert_array_equal(pitch._frame_starts(4, 32, 8), np.array([0]))
    assert pitch._frame_pitch_candidates(np.zeros(4), 16_000, 60.0, 500.0) == []
    assert pitch._local_peak_indices(np.array([1.0, 0.5])).size == 0
    assert pitch._track_pitch_candidates([], np.empty(0))[0].size == 0
    assert pitch._voiced_intervals(np.empty(0, dtype=bool), np.empty(0), 32, 0, 16_000) == []
    assert pitch._correlation_score(np.ones(3), 0, 1, 1) == -1.0
    empty = estimate_pitch_track_lane(np.empty(0), sample_rate=16_000)
    assert empty.pitch_marks.size == 0
    batch = estimate_pitch_track(np.zeros((2, 64)), sample_rate=16_000)
    assert len(batch) == 2
