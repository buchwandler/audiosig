from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from audiosig import (
    AudioShapeError,
    InvalidParameterError,
    TerminalSilenceAnalysis,
    analyze_terminal_silence,
    speech_end,
    trim_terminal_silence,
)
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


def test_speech_end_preserves_weak_resumed_activity_after_internal_pause() -> None:
    source = np.zeros(1_200, dtype=np.float32)
    source[300:600] = 0.2
    source[750:950] = 0.004

    assert speech_end(source, sample_rate=_SAMPLE_RATE) == 950
    trimmed = trim_trail(source, sample_rate=_SAMPLE_RATE, trail_seconds=0.0)
    np.testing.assert_array_equal(trimmed, source[:950])

    extended = np.concatenate((source, np.zeros(300, dtype=np.float32)))
    assert speech_end(extended, sample_rate=_SAMPLE_RATE) == 950


def test_speech_end_preserves_short_final_phrase_above_minimum_duration() -> None:
    source = np.zeros(1_200, dtype=np.float32)
    source[300:600] = 0.2
    source[750:790] = 0.2

    assert speech_end(source, sample_rate=_SAMPLE_RATE) == 790
    assert speech_end(source, sample_rate=_SAMPLE_RATE, min_activity_seconds=0.05) == 600


def test_speech_end_ignores_one_frame_click_after_internal_pause() -> None:
    source = np.zeros(1_200, dtype=np.float32)
    source[300:600] = 0.2
    source[750:760] = 1.0

    assert speech_end(source, sample_rate=_SAMPLE_RATE) == 600


def test_speech_end_tracks_the_last_of_multiple_activity_islands() -> None:
    source = np.zeros(1_400, dtype=np.float32)
    source[300:600] = 0.2
    source[750:850] = 0.2
    source[950:1_100] = 0.004

    assert speech_end(source, sample_rate=_SAMPLE_RATE) == 1_100


def test_trim_trail_preserves_active_eof_and_exact_terminal_trail_lengths() -> None:
    active_to_eof = np.zeros(1_000, dtype=np.float32)
    active_to_eof[300:] = 0.2
    np.testing.assert_array_equal(
        trim_trail(active_to_eof, sample_rate=_SAMPLE_RATE, trail_seconds=0.0),
        active_to_eof,
    )

    source = _speech()
    no_trail = trim_trail(source, sample_rate=_SAMPLE_RATE, trail_seconds=0.0)
    short_trail = trim_trail(source, sample_rate=_SAMPLE_RATE, trail_seconds=0.05)
    np.testing.assert_array_equal(no_trail, source[:700])
    np.testing.assert_array_equal(short_trail, source[:750])


def test_speech_end_bounds_partial_final_frame_and_multichannel_activity() -> None:
    partial = np.zeros(805, dtype=np.float32)
    partial[300:800] = 0.2
    endpoint = speech_end(partial, sample_rate=_SAMPLE_RATE, min_silence_seconds=0.005)
    assert endpoint == 800
    assert endpoint <= partial.size
    trimmed = trim_trail(
        partial,
        sample_rate=_SAMPLE_RATE,
        trail_seconds=0.0,
        min_silence_seconds=0.005,
    )
    np.testing.assert_array_equal(trimmed, partial[:800])

    multichannel = np.zeros((2, 1_200), dtype=np.float32)
    multichannel[0, 300:600] = 0.2
    multichannel[1, 750:950] = 0.2
    assert speech_end(multichannel, sample_rate=_SAMPLE_RATE, axis=1) == 950


def test_min_activity_seconds_rejects_invalid_values_in_both_public_functions() -> None:
    for duration in (-0.01, np.nan, np.inf):
        with pytest.raises(InvalidParameterError, match="min_activity_seconds"):
            speech_end(_speech(), sample_rate=_SAMPLE_RATE, min_activity_seconds=duration)
        with pytest.raises(InvalidParameterError, match="min_activity_seconds"):
            trim_trail(_speech(), sample_rate=_SAMPLE_RATE, min_activity_seconds=duration)


def test_trim_trail_is_an_independent_prefix_copy_after_resumed_activity() -> None:
    source = np.zeros(1_200, dtype=np.float32)
    source[300:600] = 0.2
    source[750:950] = 0.004

    trimmed = trim_trail(source, sample_rate=_SAMPLE_RATE, trail_seconds=0.05)
    np.testing.assert_array_equal(trimmed, source[:1_000])
    assert trimmed.flags.c_contiguous
    assert not np.shares_memory(trimmed, source)


_TERMINAL_SAMPLE_RATE = 24_000


def _terminal_fixture(
    *, weak_tail_rms: float = 0.0, padding_seconds: float = 0.4
) -> tuple[np.ndarray, int]:
    active = np.full(12_000, 0.2, dtype=np.float32)
    tail = (
        np.random.default_rng(17).normal(0.0, weak_tail_rms, 1_920).astype(np.float32)
        if weak_tail_rms > 0.0
        else np.empty(0, dtype=np.float32)
    )
    padding = np.zeros(round(padding_seconds * _TERMINAL_SAMPLE_RATE), dtype=np.float32)
    return np.concatenate((active, tail, padding)), active.size + tail.size


def test_terminal_silence_preserves_low_power_final_tails() -> None:
    for relative_db in (-40.0, -50.0, -60.0, -70.0):
        tail_rms = 0.2 * 10.0 ** (relative_db / 20.0)
        source, tail_end = _terminal_fixture(weak_tail_rms=tail_rms)
        trimmed, analysis = trim_terminal_silence(source, sample_rate=_TERMINAL_SAMPLE_RATE)

        assert isinstance(analysis, TerminalSilenceAnalysis)
        assert not hasattr(analysis, "__dict__")
        with pytest.raises(AttributeError):
            analysis.confirmed = False
        assert analysis.confirmed
        assert analysis.trim_end_sample >= tail_end
        assert trimmed.shape == (analysis.trim_end_sample,)
        assert trimmed.size >= tail_end
        assert not np.shares_memory(trimmed, source)


@pytest.mark.parametrize("event_ms", (5, 10, 20, 30))
def test_terminal_silence_preserves_short_peak_bearing_final_events(event_ms: int) -> None:
    event_count = event_ms * _TERMINAL_SAMPLE_RATE // 1_000
    event = np.zeros(event_count, dtype=np.float32)
    event[::3] = 0.025
    event[1::3] = -0.025
    source = np.concatenate(
        (np.full(12_000, 0.2, dtype=np.float32), event, np.zeros(9_600, dtype=np.float32))
    )

    trimmed, analysis = trim_terminal_silence(source, sample_rate=_TERMINAL_SAMPLE_RATE)

    assert analysis.confirmed
    assert trimmed.size >= 12_000 + event_count


def test_terminal_silence_trims_clean_zero_and_stationary_noise_padding() -> None:
    clean, active_end = _terminal_fixture()
    clean_trimmed, clean_analysis = trim_terminal_silence(
        clean, sample_rate=_TERMINAL_SAMPLE_RATE, keep_seconds=0.08
    )
    assert clean_analysis.confirmed
    assert clean_trimmed.size == active_end + 0.08 * _TERMINAL_SAMPLE_RATE
    np.testing.assert_array_equal(clean_trimmed, clean[: clean_trimmed.size])

    noise = np.random.default_rng(29).normal(0.0, 0.0005, 9_600).astype(np.float32)
    noisy = np.concatenate((np.full(12_000, 0.2, dtype=np.float32), noise))
    noisy_trimmed, noisy_analysis = trim_terminal_silence(noisy, sample_rate=_TERMINAL_SAMPLE_RATE)
    assert noisy_analysis.confirmed
    assert noisy_trimmed.size < noisy.size


def test_terminal_silence_fails_closed_for_ambiguous_decay_and_active_eof() -> None:
    active = np.full(12_000, 0.2, dtype=np.float32)
    transition = np.linspace(0.02, 0.001, 24_000, dtype=np.float32)
    tail_noise = np.random.default_rng(31).normal(0.0, 0.001, 9_600).astype(np.float32)
    ambiguous = np.concatenate((active, transition, tail_noise))
    result = analyze_terminal_silence(ambiguous, sample_rate=_TERMINAL_SAMPLE_RATE)
    trimmed, trim_result = trim_terminal_silence(ambiguous, sample_rate=_TERMINAL_SAMPLE_RATE)

    assert not result.confirmed
    assert result.trim_end_sample == ambiguous.size
    assert result.removable_samples == 0
    assert not trim_result.confirmed
    np.testing.assert_array_equal(trimmed, ambiguous)

    active_to_eof = np.full(12_000, 0.2, dtype=np.float32)
    eof_result = analyze_terminal_silence(active_to_eof, sample_rate=_TERMINAL_SAMPLE_RATE)
    eof_trimmed, _ = trim_terminal_silence(active_to_eof, sample_rate=_TERMINAL_SAMPLE_RATE)
    assert not eof_result.confirmed
    np.testing.assert_array_equal(eof_trimmed, active_to_eof)


def test_terminal_silence_all_silence_is_unchanged_and_unconfirmed() -> None:
    source = np.zeros(12_345, dtype=np.float32)
    trimmed, analysis = trim_terminal_silence(source, sample_rate=_TERMINAL_SAMPLE_RATE)

    assert not analysis.confirmed
    assert analysis.trim_end_sample == source.size
    assert analysis.removable_samples == 0
    np.testing.assert_array_equal(trimmed, source)


def test_terminal_silence_boundary_is_gain_scale_invariant() -> None:
    source, _ = _terminal_fixture(weak_tail_rms=0.0004)
    boundaries = [
        analyze_terminal_silence(source * scale, sample_rate=_TERMINAL_SAMPLE_RATE).trim_end_sample
        for scale in (0.01, 0.1, 1.0, 2.0)
    ]

    assert len(set(boundaries)) == 1


def test_terminal_silence_handles_partial_frames_lanes_and_sample_axes() -> None:
    one_lane, _ = _terminal_fixture()
    source = np.stack((one_lane, one_lane * 0.5))[:, ::-1][:, ::-1]
    analysis = analyze_terminal_silence(source, sample_rate=_TERMINAL_SAMPLE_RATE, axis=1)
    trimmed, trim_analysis = trim_terminal_silence(
        source, sample_rate=_TERMINAL_SAMPLE_RATE, axis=1, keep_seconds=0.04
    )

    assert analysis.confirmed
    assert 0 <= analysis.trim_end_sample <= source.shape[1]
    assert analysis.removable_samples == source.shape[1] - analysis.trim_end_sample
    assert trimmed.shape == (2, trim_analysis.trim_end_sample)
    assert trimmed.dtype == source.dtype
    assert trimmed.flags.c_contiguous
    assert not np.shares_memory(trimmed, source)

    partial = np.concatenate(
        (np.full(12_000, 0.2, dtype=np.float32), np.zeros(9_607, dtype=np.float32))
    )
    partial_analysis = analyze_terminal_silence(partial, sample_rate=_TERMINAL_SAMPLE_RATE)
    assert partial_analysis.trim_end_sample <= partial.size


def test_terminal_silence_preserves_resumed_activity_and_clicks() -> None:
    resumed = np.zeros(36_000, dtype=np.float32)
    resumed[0:8_000] = 0.2
    resumed[16_000:20_000] = 0.08
    resumed_trimmed, resumed_analysis = trim_terminal_silence(
        resumed, sample_rate=_TERMINAL_SAMPLE_RATE, keep_seconds=0.0
    )
    assert resumed_analysis.confirmed
    assert resumed_trimmed.size >= 20_000

    clean, _ = _terminal_fixture()
    click = clean.copy()
    click[-5_000] = 0.8
    clean_trimmed, _ = trim_terminal_silence(clean, sample_rate=_TERMINAL_SAMPLE_RATE)
    click_trimmed, click_analysis = trim_terminal_silence(click, sample_rate=_TERMINAL_SAMPLE_RATE)
    assert click_analysis.confirmed
    assert click_trimmed.size >= clean_trimmed.size
    assert click_trimmed[click.size - 5_000] == 0.8


def test_terminal_silence_boundary_never_moves_earlier_when_padding_grows() -> None:
    shorter, _ = _terminal_fixture(weak_tail_rms=0.0002, padding_seconds=0.2)
    longer, _ = _terminal_fixture(weak_tail_rms=0.0002, padding_seconds=0.6)
    short_trimmed, short_analysis = trim_terminal_silence(
        shorter, sample_rate=_TERMINAL_SAMPLE_RATE
    )
    long_trimmed, long_analysis = trim_terminal_silence(longer, sample_rate=_TERMINAL_SAMPLE_RATE)

    assert short_analysis.confirmed and long_analysis.confirmed
    assert long_analysis.trim_end_sample >= short_analysis.trim_end_sample
    assert long_trimmed.size >= short_trimmed.size


@pytest.mark.parametrize("duration_ms", (20, 50, 100, 150, 200))
@pytest.mark.parametrize("event_to_floor_ratio", (0.5, 1.0, 2.0, 4.0))
@pytest.mark.parametrize(
    "morphology", ("noise-like", "high-frequency", "sparse-burst", "gradual-decay")
)
def test_terminal_trim_preserves_weak_final_event_boundary(
    duration_ms: int, event_to_floor_ratio: float, morphology: str
) -> None:
    sample_rate = 24_000
    floor_rms = 5e-4
    event_samples = duration_ms * sample_rate // 1_000
    rng = np.random.default_rng(1234)

    if morphology == "noise-like":
        event = rng.normal(0.0, 1.0, event_samples)
    elif morphology == "high-frequency":
        times = np.arange(event_samples, dtype=np.float64) / sample_rate
        event = np.sin(2.0 * np.pi * 8_000.0 * times)
    elif morphology == "sparse-burst":
        event = np.zeros(event_samples, dtype=np.float64)
        event[::8] = 1.0
        event[1::8] = -1.0
    else:
        envelope = np.linspace(1.0, 0.15, event_samples, dtype=np.float64)
        event = rng.normal(0.0, 1.0, event_samples) * envelope

    event *= floor_rms * event_to_floor_ratio / np.sqrt(np.mean(np.square(event)))
    event = event.astype(np.float32)
    active = np.full(round(0.5 * sample_rate), 0.2, dtype=np.float32)
    padding = rng.normal(0.0, floor_rms, round(0.4 * sample_rate)).astype(np.float32)
    source = np.concatenate((active, event, padding))
    protected_end_sample = active.size + event.size

    trimmed, analysis = trim_terminal_silence(source, sample_rate=sample_rate)
    end_error_samples = analysis.trim_end_sample - protected_end_sample

    assert end_error_samples >= 0
    assert trimmed.size == analysis.trim_end_sample


def test_terminal_trim_default_reserve_fixes_near_floor_event_cut() -> None:
    sample_rate = 24_000
    rng = np.random.default_rng(2024)
    active = np.full(round(0.5 * sample_rate), 0.2, dtype=np.float32)
    weak_event = rng.normal(0.0, 5e-4, round(0.1 * sample_rate)).astype(np.float32)
    padding = rng.normal(0.0, 5e-4, round(0.4 * sample_rate)).astype(np.float32)
    source = np.concatenate((active, weak_event, padding))
    protected_end_sample = active.size + weak_event.size

    short_trimmed, short_analysis = trim_terminal_silence(
        source, sample_rate=sample_rate, keep_seconds=0.080
    )
    default_trimmed, default_analysis = trim_terminal_silence(source, sample_rate=sample_rate)

    assert short_analysis.trim_end_sample < protected_end_sample
    assert short_trimmed.size == short_analysis.trim_end_sample
    assert default_analysis.trim_end_sample >= protected_end_sample
    assert default_trimmed.size >= protected_end_sample


def test_terminal_boundary_evaluator_reports_asymmetric_metrics(tmp_path) -> None:
    import wave

    from scripts.evaluate_terminal_boundaries import BoundaryLabel, evaluate

    sample_rate = 24_000
    samples = np.zeros(sample_rate, dtype="<i2")
    samples[2_400:12_000] = 6_553
    labels = [
        BoundaryLabel("preserved.wav", 12_000),
        BoundaryLabel("negative.wav", 19_500),
    ]
    for label in labels:
        with wave.open(str(tmp_path / label.audio_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(samples.tobytes())

    report = evaluate(tmp_path, labels)
    summary = report["summary"]

    assert summary["item_count"] == 2
    assert summary["negative_end_error_counts"] == {
        "0ms": 1,
        "10ms": 1,
        "25ms": 1,
        "50ms": 1,
    }
    assert summary["mean_retained_tail_ms"] == pytest.approx(75.0)
    assert summary["p95_retained_tail_ms"] == pytest.approx(142.5)
