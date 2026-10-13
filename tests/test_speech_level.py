from __future__ import annotations

import numpy as np
import pytest

from audiosig import InvalidParameterError
from audiosig.speech_level import normalize_reference_level, speech_level


def test_speech_level_constant_signal_and_active_duration() -> None:
    source = np.full(2_500, 0.1, dtype=np.float32)
    result = speech_level(source, sample_rate=1_000)
    assert result.level_db == pytest.approx(-20.0, abs=1e-6)
    assert result.active_seconds == pytest.approx(2.48)


def test_speech_level_short_input_uses_whole_waveform_rms() -> None:
    source = np.array([0.03, 0.04], dtype=np.float64)
    result = speech_level(source, sample_rate=1_000)
    expected_rms = np.sqrt((0.03**2 + 0.04**2) / 2.0)
    assert result.level_db == pytest.approx(20.0 * np.log10(expected_rms), abs=1e-12)
    assert result.active_seconds == 0.0


def test_speech_level_silence_uses_floor_and_all_frame_fallback() -> None:
    short = speech_level(np.zeros(4, dtype=np.float32), sample_rate=1_000)
    assert short.level_db == pytest.approx(-120.0)
    assert short.active_seconds == 0.0

    full = speech_level(np.zeros(1_000, dtype=np.float32), sample_rate=1_000)
    assert full.level_db == pytest.approx(-120.0)
    assert full.active_seconds == pytest.approx(0.98)


def test_speech_level_discards_quiet_frames_and_uses_lower_median() -> None:
    source = np.concatenate([np.zeros(500, dtype=np.float64), np.full(500, 0.2, dtype=np.float64)])
    result = speech_level(source, sample_rate=1_000)
    assert result.level_db == pytest.approx(20.0 * np.log10(0.2), abs=1e-12)
    assert result.active_seconds < 0.98
    assert result.active_seconds > 0.0


def test_reference_normalization_boosts_only_and_reports_policy() -> None:
    quiet = np.full(1_000, 0.05, dtype=np.float32)
    quiet_result = normalize_reference_level(quiet, sample_rate=1_000)
    assert quiet_result.measured_db == pytest.approx(-26.0205999, abs=1e-5)
    assert quiet_result.applied_gain_db == pytest.approx(-19.8 - quiet_result.measured_db)
    assert quiet_result.resulting_db == pytest.approx(-19.8)
    assert quiet_result.audio.dtype == quiet.dtype
    assert np.max(np.abs(quiet_result.audio)) < 0.95

    loud_result = normalize_reference_level(np.full(1_000, 0.5), sample_rate=1_000)
    assert loud_result.applied_gain_db == 0.0
    assert loud_result.resulting_db == pytest.approx(loud_result.measured_db)


def test_reference_normalization_obeys_gain_and_peak_limits() -> None:
    gain_limited = normalize_reference_level(
        np.full(1_000, 0.001, dtype=np.float64), sample_rate=1_000
    )
    assert gain_limited.applied_gain_db == 30.0
    assert gain_limited.resulting_db == pytest.approx(-30.0)

    peak_limited_source = np.zeros(1_600, dtype=np.float64)
    peak_limited_source[400] = 0.8
    peak_limited = normalize_reference_level(peak_limited_source, sample_rate=16_000)
    assert peak_limited.applied_gain_db == pytest.approx(20.0 * np.log10(0.95 / 0.8))
    assert np.max(np.abs(peak_limited.audio)) == pytest.approx(0.95)
    assert peak_limited.resulting_db == pytest.approx(
        peak_limited.measured_db + peak_limited.applied_gain_db
    )


def test_reference_normalization_empty_and_validation() -> None:
    empty = np.empty(0, dtype=np.float32)
    result = normalize_reference_level(empty, sample_rate=16_000)
    assert result.audio.shape == empty.shape
    assert result.measured_db == pytest.approx(-120.0)
    assert result.applied_gain_db == 30.0

    with pytest.raises(InvalidParameterError, match="max_gain_db"):
        normalize_reference_level(
            np.ones(10, dtype=np.float32), sample_rate=1_000, max_gain_db=-1.0
        )
    with pytest.raises(InvalidParameterError, match="peak_ceiling"):
        normalize_reference_level(
            np.ones(10, dtype=np.float32), sample_rate=1_000, peak_ceiling=0.0
        )
    with pytest.raises(InvalidParameterError, match="floor_quantile"):
        speech_level(np.ones(10, dtype=np.float32), sample_rate=1_000, floor_quantile=1.1)
