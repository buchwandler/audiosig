from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import audiosig._pitch as private_pitch
from audiosig import AudioShapeError, InvalidParameterError
from audiosig.pitch import PitchTrack, pitch_track


def _tone(sample_rate: int, frequency: float, seconds: float = 0.5) -> np.ndarray:
    time = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * frequency * time)


@pytest.mark.parametrize("frequency", [80.0, 120.0, 220.0, 440.0])
def test_public_pitch_track_matches_private_defaults(frequency: float) -> None:
    source = _tone(16_000, frequency)
    expected = private_pitch.estimate_pitch_track_lane(source, sample_rate=16_000)
    actual = pitch_track(source, sample_rate=16_000)
    assert isinstance(actual, PitchTrack)
    np.testing.assert_array_equal(actual.frame_times_s, expected.frame_times)
    np.testing.assert_array_equal(actual.f0_hz, expected.frequencies)
    np.testing.assert_array_equal(actual.voiced, expected.voiced)
    np.testing.assert_array_equal(actual.confidence, expected.confidence)
    assert actual.frame_times_s.dtype == actual.f0_hz.dtype == actual.confidence.dtype == np.float64
    assert actual.voiced.dtype == np.bool_
    assert np.median(actual.f0_hz[actual.voiced]) == pytest.approx(frequency, rel=0.03)


def test_public_pitch_track_optional_hop_and_owned_arrays() -> None:
    source = _tone(16_000, 220.0)
    expected = private_pitch.estimate_pitch_track_lane(source, sample_rate=16_000)
    legacy = pitch_track(source, sample_rate=16_000, hop_length=None)
    np.testing.assert_array_equal(legacy.frame_times_s, expected.frame_times)
    np.testing.assert_array_equal(legacy.f0_hz, expected.frequencies)

    hop = 160
    explicit = pitch_track(source, sample_rate=16_000, hop_length=hop)
    assert explicit.frame_times_s.size > 1
    np.testing.assert_allclose(np.diff(explicit.frame_times_s), hop / 16_000)
    assert not np.shares_memory(explicit.frame_times_s, expected.frame_times)
    assert not np.shares_memory(explicit.f0_hz, expected.frequencies)
    assert not np.shares_memory(explicit.voiced, expected.voiced)
    assert not np.shares_memory(explicit.confidence, expected.confidence)


def test_public_pitch_track_silence_is_unvoiced_and_deterministic() -> None:
    silence = np.zeros(8_000, dtype=np.float32)
    first = pitch_track(silence, sample_rate=16_000)
    second = pitch_track(silence, sample_rate=16_000)
    np.testing.assert_array_equal(first.voiced, second.voiced)
    np.testing.assert_array_equal(first.f0_hz, second.f0_hz)
    assert not np.any(first.voiced)
    assert np.all(first.f0_hz == 0.0)


def test_public_pitch_track_requires_mono_finite_float_audio() -> None:
    with pytest.raises(AudioShapeError):
        pitch_track(np.zeros((2, 100), dtype=np.float32), sample_rate=16_000)
    with pytest.raises(AudioShapeError):
        pitch_track(np.zeros(100, dtype=np.int16), sample_rate=16_000)
    with pytest.raises(AudioShapeError):
        pitch_track(np.array([0.0, np.inf]), sample_rate=16_000)


@pytest.mark.parametrize(
    "parameters",
    [
        {"sample_rate": 0},
        {"sample_rate": 16_000, "f0_min": 0.0},
        {"sample_rate": 16_000, "f0_min": 300.0, "f0_max": 200.0},
        {"sample_rate": 16_000, "f0_max": 8_000.0},
        {"sample_rate": 16_000, "hop_length": 0},
        {"sample_rate": 16_000, "hop_length": True},
    ],
)
def test_public_pitch_track_rejects_invalid_parameters(parameters: dict[str, Any]) -> None:
    with pytest.raises(InvalidParameterError):
        pitch_track(np.zeros(512, dtype=np.float64), **parameters)


def test_private_pitch_geometry_default_is_stable_and_hop_is_explicit() -> None:
    rate = 24_000
    floor = 60.0
    default_frame, default_hop = private_pitch._analysis_geometry(rate, floor)
    assert private_pitch._analysis_geometry(rate, floor, None) == (default_frame, default_hop)
    assert private_pitch._analysis_geometry(rate, floor, 240) == (default_frame, 240)
