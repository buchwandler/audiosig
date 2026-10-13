from __future__ import annotations

import numpy as np
import pytest

from audiosig import InvalidParameterError
from audiosig.boundaries import energy_fraction, energy_onset, speech_onset, trim_lead, trim_trail

_SAMPLE_RATE = 1_000


def _speech() -> np.ndarray:
    audio = np.zeros(1_200, dtype=np.float32)
    audio[300:700] = 0.2
    return audio


def test_speech_onset_matches_sustained_window_policy() -> None:
    assert speech_onset(_speech(), sample_rate=_SAMPLE_RATE) == 290
    assert speech_onset(np.zeros(50, dtype=np.float32), sample_rate=_SAMPLE_RATE) is None


def test_energy_onset_and_fraction_match_frame_equations() -> None:
    source = _speech()
    assert energy_onset(source, sample_rate=_SAMPLE_RATE) == 300
    assert energy_onset(np.zeros(20, dtype=np.float32), sample_rate=_SAMPLE_RATE) is None
    assert energy_fraction(source, sample_rate=_SAMPLE_RATE, start=200, end=500) == pytest.approx(
        2 / 3
    )
    assert energy_fraction(source, sample_rate=_SAMPLE_RATE, start=700, end=900) == 0.0


def test_lead_and_trail_trimming_preserve_requested_safety_and_trail() -> None:
    source = _speech()
    lead = trim_lead(source, sample_rate=_SAMPLE_RATE)
    assert lead.shape == (990,)
    np.testing.assert_array_equal(lead, source[210:])

    no_lead = trim_lead(source, sample_rate=_SAMPLE_RATE, lead_seconds=0.0)
    np.testing.assert_array_equal(no_lead, source[270:])
    skipped = trim_lead(source, sample_rate=_SAMPLE_RATE, skip_seconds=0.5)
    np.testing.assert_array_equal(skipped, source[270:])

    trail = trim_trail(source, sample_rate=_SAMPLE_RATE)
    assert trail.shape == (1_000,)
    np.testing.assert_array_equal(trail, source[:1_000])
    silent = np.zeros(200, dtype=np.float32)
    unchanged = trim_trail(silent, sample_rate=_SAMPLE_RATE)
    np.testing.assert_array_equal(unchanged, silent)
    assert not np.shares_memory(unchanged, silent)


def test_boundary_helpers_support_arbitrary_sample_axis() -> None:
    source = np.stack([_speech(), _speech() * 0.5])
    assert speech_onset(source, sample_rate=_SAMPLE_RATE, axis=1) == 290
    assert energy_onset(source, sample_rate=_SAMPLE_RATE, axis=1) == 300
    trimmed = trim_lead(source, sample_rate=_SAMPLE_RATE, axis=1)
    assert trimmed.shape == (2, 990)
    np.testing.assert_array_equal(trimmed, source[:, 210:])
    trailed = trim_trail(source, sample_rate=_SAMPLE_RATE, axis=1)
    assert trailed.shape == (2, 1_000)

    empty_lanes = np.empty((0, 1_200), dtype=np.float32)
    assert speech_onset(empty_lanes, sample_rate=_SAMPLE_RATE, axis=1) is None
    assert energy_onset(empty_lanes, sample_rate=_SAMPLE_RATE, axis=1) is None
    assert energy_fraction(empty_lanes, sample_rate=_SAMPLE_RATE, start=0, end=1_200, axis=1) == 0.0
    assert trim_lead(empty_lanes, sample_rate=_SAMPLE_RATE, axis=1).shape == empty_lanes.shape
    assert trim_trail(empty_lanes, sample_rate=_SAMPLE_RATE, axis=1).shape == empty_lanes.shape


def test_boundary_helpers_validate_parameters_and_sample_ranges() -> None:
    source = _speech()
    with pytest.raises(InvalidParameterError, match="required_frames"):
        speech_onset(source, sample_rate=_SAMPLE_RATE, window_frames=4, required_frames=5)
    with pytest.raises(InvalidParameterError, match="floor_quantile"):
        energy_onset(source, sample_rate=_SAMPLE_RATE, floor_quantile=1.1)
    with pytest.raises(InvalidParameterError, match="end"):
        energy_fraction(source, sample_rate=_SAMPLE_RATE, start=0, end=source.size + 1)
    with pytest.raises(InvalidParameterError, match="non-negative"):
        trim_lead(source, sample_rate=_SAMPLE_RATE, lead_seconds=-0.1)
    with pytest.raises(InvalidParameterError, match="non-negative"):
        trim_trail(source, sample_rate=_SAMPLE_RATE, trail_seconds=-0.1)
