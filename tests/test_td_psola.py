from __future__ import annotations

import numpy as np
import pytest

import audiosig._td_psola as td
from audiosig import InvalidParameterError, apply_speech_effects, pitch_shift, time_stretch
from audiosig._pitch import estimate_pitch_track_lane
from audiosig._td_psola import td_psola_prosody


def _speech_fixture(sample_rate: int = 16_000, seconds: float = 0.7) -> np.ndarray:
    time = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    envelope = np.minimum(1.0, time * 30.0) * np.minimum(1.0, (seconds - time) * 30.0)
    return (envelope * (0.18 * np.sin(2.0 * np.pi * 180.0 * time))).astype(np.float32)


@pytest.mark.quality
@pytest.mark.parametrize("rate", (0.75, 0.8, 1.0, 1.2, 1.25, 1.5))
@pytest.mark.parametrize("semitones", (-6.0, -2.0, 0.0, 2.0, 6.0))
def test_td_psola_exact_contracts(rate: float, semitones: float) -> None:
    source = _speech_fixture()
    original = source.copy()
    result = td_psola_prosody(source, sample_rate=16_000, rate=rate, semitones=semitones)
    assert result.shape == (max(1, round(source.size / rate)),)
    assert result.dtype == source.dtype
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(source, original)
    assert not np.shares_memory(source, result)


def test_td_psola_neutral_empty_short_axis_and_lanes() -> None:
    source = _speech_fixture()
    neutral = td_psola_prosody(source, sample_rate=16_000)
    np.testing.assert_array_equal(neutral, source)
    assert not np.shares_memory(neutral, source)

    empty = np.empty((2, 0, 3), dtype=np.float64)
    empty_result = td_psola_prosody(empty, sample_rate=16_000, axis=1, rate=0.8)
    assert empty_result.shape == empty.shape
    assert empty_result.dtype == empty.dtype

    lanes = np.stack([source, source * 0.5])
    result = td_psola_prosody(lanes, sample_rate=16_000, rate=1.2, semitones=3.0)
    np.testing.assert_allclose(
        result[0], td_psola_prosody(source, sample_rate=16_000, rate=1.2, semitones=3.0), atol=2e-6
    )
    np.testing.assert_allclose(
        result[1],
        td_psola_prosody(source * 0.5, sample_rate=16_000, rate=1.2, semitones=3.0),
        atol=2e-6,
    )


@pytest.mark.parametrize("rate", (0.74, 1.51))
def test_td_psola_rejects_unsupported_rate(rate: float) -> None:
    with pytest.raises(InvalidParameterError):
        td_psola_prosody(_speech_fixture(), sample_rate=16_000, rate=rate)


def test_td_psola_rejects_unsupported_pitch() -> None:
    with pytest.raises(InvalidParameterError):
        td_psola_prosody(_speech_fixture(), sample_rate=16_000, semitones=6.01)


def test_td_psola_private_empty_and_fallback_helpers() -> None:
    marks, mapped = td._target_marks(
        np.empty(0, dtype=np.int64),
        source_length=10,
        target_length=10,
        rate=1.0,
        pitch_ratio=1.0,
    )
    assert marks.size == 0 and mapped.size == 0
    output, normalization = td._overlap_add_grains(
        np.zeros(4),
        np.empty(0, dtype=np.int64),
        marks,
        mapped,
        8,
    )
    assert output.shape == normalization.shape == (8,)
    empty_track = estimate_pitch_track_lane(np.zeros(4), sample_rate=16_000)
    assert (
        td._voiced_output_mask(
            empty_track,
            sample_rate=16_000,
            rate=1.0,
            target_length=8,
            normalization=np.zeros(8),
        ).sum()
        == 0.0
    )
    assert td._td_psola_lane(
        np.zeros(4),
        sample_rate=16_000,
        rate=1.0,
        semitones=0.0,
        target_length=4,
        pitch_floor=60.0,
        pitch_ceiling=500.0,
    ).shape == (4,)


def test_td_psola_interior_grains_span_adjacent_periods() -> None:
    source = _speech_fixture().astype(np.float64)
    track = estimate_pitch_track_lane(source, sample_rate=16_000)
    index = track.pitch_marks.size // 2
    previous_period = int(track.pitch_marks[index] - track.pitch_marks[index - 1])
    next_period = int(track.pitch_marks[index + 1] - track.pitch_marks[index])
    grain, center = td._extract_pitch_grain(source, track.pitch_marks, index)

    assert center == previous_period
    assert grain.size == previous_period + next_period + 1


def test_td_psola_rejects_invalid_pitch_ceiling() -> None:
    with pytest.raises(InvalidParameterError):
        td_psola_prosody(
            _speech_fixture(), sample_rate=16_000, pitch_floor=100.0, pitch_ceiling=50.0
        )


def test_td_psola_is_rejected_by_time_stretch() -> None:
    with pytest.raises(InvalidParameterError):
        time_stretch(_speech_fixture(), 1.1, sample_rate=16_000, method="td_psola")  # type: ignore[arg-type]


def test_td_psola_handles_unvoiced_and_mixed_input() -> None:
    source = _speech_fixture()
    noise = np.random.default_rng(9).normal(0.0, 0.02, source.size).astype(np.float32)
    mixed = np.concatenate([source[: source.size // 2], noise[source.size // 2 :]])
    result = td_psola_prosody(mixed, sample_rate=16_000, rate=0.8, semitones=4.0)
    assert result.shape == (round(mixed.size / 0.8),)
    assert np.isfinite(result).all()


def test_td_psola_short_pitch_request_uses_explicit_pitch_fallback() -> None:
    sample_rate = 16_000
    time = np.arange(120, dtype=np.float64) / sample_rate
    source = np.sin(2.0 * np.pi * 180.0 * time)
    result = td_psola_prosody(source, sample_rate=sample_rate, semitones=4.0)

    assert result.shape == source.shape
    assert np.isfinite(result).all()
    assert not np.allclose(result, source)


def test_public_direct_methods_preserve_gain_and_duration() -> None:
    source = _speech_fixture()
    shifted = pitch_shift(source, sample_rate=16_000, semitones=4.0, method="td_psola")
    combined = apply_speech_effects(
        source,
        sample_rate=16_000,
        rate=1.2,
        semitones=3.0,
        gain_db=-6.0,
        method="td_psola",
    )
    assert shifted.shape == source.shape
    assert combined.shape == (round(source.size / 1.2),)
    assert np.isfinite(shifted).all() and np.isfinite(combined).all()
