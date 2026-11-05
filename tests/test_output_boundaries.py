from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from audiosig import AudioShapeError, InvalidParameterError, speech_end
from audiosig.boundaries import (
    _nonoverlap_frame_rms,
    energy_fraction,
    energy_onset,
    speech_onset,
    trim_lead,
    trim_trail,
)

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


def test_speech_end_returns_confirmed_silence_start_and_conservative_eof() -> None:
    assert speech_end(_speech(), sample_rate=_SAMPLE_RATE) == 700
    assert speech_end(np.zeros(1_000, dtype=np.float32), sample_rate=_SAMPLE_RATE) is None

    active_to_eof = np.zeros(1_000, dtype=np.float32)
    active_to_eof[300:] = 0.2
    assert speech_end(active_to_eof, sample_rate=_SAMPLE_RATE) == active_to_eof.size

    short_silence = np.zeros(750, dtype=np.float32)
    short_silence[300:700] = 0.2
    assert speech_end(short_silence, sample_rate=_SAMPLE_RATE) == short_silence.size


def test_speech_end_preserves_weak_tail_and_cancels_short_internal_pause() -> None:
    weak_tail = np.zeros(1_200, dtype=np.float32)
    weak_tail[300:600] = 0.2
    weak_tail[600:760] = 0.004
    assert speech_end(weak_tail, sample_rate=_SAMPLE_RATE) == 760

    internal_pause = np.zeros(1_200, dtype=np.float32)
    internal_pause[300:600] = 0.2
    internal_pause[640:700] = 0.004
    assert speech_end(internal_pause, sample_rate=_SAMPLE_RATE) == 700


def test_speech_end_ignores_isolated_click_but_accepts_sustained_later_speech() -> None:
    click_after_speech = np.zeros(1_400, dtype=np.float32)
    click_after_speech[300:700] = 0.2
    click_after_speech[1_000:1_010] = 1.0
    assert speech_end(click_after_speech, sample_rate=_SAMPLE_RATE) == 700

    resumed_speech = np.zeros(1_400, dtype=np.float32)
    resumed_speech[300:700] = 0.2
    resumed_speech[850:1_050] = 0.2
    assert speech_end(resumed_speech, sample_rate=_SAMPLE_RATE) == 1_050


def test_trim_trail_applies_trail_after_confirmed_endpoint() -> None:
    click_after_speech = np.zeros(1_400, dtype=np.float32)
    click_after_speech[300:700] = 0.2
    click_after_speech[1_000:1_010] = 1.0
    trimmed_click = trim_trail(click_after_speech, sample_rate=_SAMPLE_RATE)
    assert trimmed_click.shape == (1_000,)
    assert not np.shares_memory(trimmed_click, click_after_speech)

    weak_tail = np.zeros(1_200, dtype=np.float32)
    weak_tail[300:600] = 0.2
    weak_tail[600:760] = 0.004
    assert trim_trail(weak_tail, sample_rate=_SAMPLE_RATE).shape == (1_060,)
    assert trim_trail(weak_tail, sample_rate=_SAMPLE_RATE, trail_seconds=0.0).shape == (760,)


def test_speech_end_analyzes_partial_frame_and_allows_zero_durations() -> None:
    framed = np.concatenate((np.zeros(10), np.ones(2)))
    np.testing.assert_array_equal(
        _nonoverlap_frame_rms(framed, 10, include_partial=True), np.array([0.0, 1.0])
    )
    assert _nonoverlap_frame_rms(framed, 10).shape == (1,)

    partial_silence = np.zeros(805, dtype=np.float32)
    partial_silence[300:800] = 0.2
    assert speech_end(partial_silence, sample_rate=_SAMPLE_RATE, min_silence_seconds=0.005) == 800

    short_silence = np.zeros(705, dtype=np.float32)
    short_silence[300:700] = 0.2
    assert speech_end(short_silence, sample_rate=_SAMPLE_RATE) == short_silence.size
    assert speech_end(short_silence, sample_rate=_SAMPLE_RATE, min_silence_seconds=0.0) == 700
    assert speech_end(_speech(), sample_rate=_SAMPLE_RATE, hysteresis_db=0.0) == 700


def test_speech_end_and_trim_trail_preserve_axis_dtype_and_contiguity() -> None:
    source = np.stack((_speech(), _speech() * 0.5))
    assert speech_end(source, sample_rate=_SAMPLE_RATE, axis=1) == 700
    trimmed = trim_trail(source, sample_rate=_SAMPLE_RATE, axis=1)
    assert trimmed.shape == (2, 1_000)
    assert trimmed.dtype == source.dtype
    assert trimmed.flags.c_contiguous
    assert not np.shares_memory(trimmed, source)


@pytest.mark.parametrize(
    ("invalid_call", "message"),
    [
        (
            lambda: speech_end(_speech(), sample_rate=_SAMPLE_RATE, hysteresis_db=-1.0),
            "hysteresis_db",
        ),
        (
            lambda: speech_end(_speech(), sample_rate=_SAMPLE_RATE, min_silence_seconds=-0.1),
            "min_silence_seconds",
        ),
        (
            lambda: speech_end(
                _speech(), sample_rate=_SAMPLE_RATE, required_frames=7, window_frames=6
            ),
            "required_frames",
        ),
        (lambda: speech_end(_speech(), sample_rate=0), "sample_rate"),
        (
            lambda: speech_end(_speech(), sample_rate=_SAMPLE_RATE, frame_seconds=0.0),
            "frame_seconds",
        ),
        (
            lambda: speech_end(_speech(), sample_rate=_SAMPLE_RATE, floor_quantile=1.1),
            "floor_quantile",
        ),
    ],
)
def test_speech_end_rejects_invalid_parameters(
    invalid_call: Callable[[], int | None], message: str
) -> None:
    with pytest.raises(InvalidParameterError, match=message):
        invalid_call()


def test_speech_end_rejects_invalid_axis() -> None:
    with pytest.raises(AudioShapeError, match="axis"):
        speech_end(_speech(), sample_rate=_SAMPLE_RATE, axis=2)
