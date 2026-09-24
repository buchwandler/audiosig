from __future__ import annotations

import numpy as np
import pytest

import audiosig
from audiosig import InvalidParameterError
from audiosig._automation import _RateMap
from audiosig._pitch import estimate_pitch_track_lane
from tests._quality_helpers import dominant_frequency


def test_public_envelope_api_matches_static_constant_effects() -> None:
    sample_rate = 24_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    source = np.sin(2.0 * np.pi * 220.0 * time)

    constant_rate = audiosig.apply_speech_effects_envelope(
        source,
        sample_rate=sample_rate,
        rate_points=((0.0, 0.85),),
    )
    static_rate = audiosig.apply_speech_effects(source, sample_rate=sample_rate, rate=0.85)
    np.testing.assert_array_equal(constant_rate, static_rate)

    constant_pitch = audiosig.apply_speech_effects_envelope(
        source,
        sample_rate=sample_rate,
        pitch_points=((0.0, 2.0),),
    )
    static_pitch = audiosig.apply_speech_effects(source, sample_rate=sample_rate, semitones=2.0)
    np.testing.assert_array_equal(constant_pitch, static_pitch)
    identity = audiosig.apply_speech_effects_envelope(
        source, sample_rate=sample_rate, rate_points=((0.0, 1.0),)
    )
    np.testing.assert_array_equal(identity, source)
    assert not np.shares_memory(identity, source)

    combined = audiosig.apply_speech_effects_envelope(
        source,
        sample_rate=sample_rate,
        rate_points=((0.0, 0.85),),
        pitch_points=((0.0, 2.0),),
    )
    static_combined = audiosig.apply_speech_effects(
        source, sample_rate=sample_rate, rate=0.85, semitones=2.0
    )
    np.testing.assert_array_equal(combined, static_combined)


def test_variable_rate_wsola_preserves_landmark_timing_and_exact_length() -> None:
    sample_rate = 4_000
    source = np.random.default_rng(71).normal(0.0, 0.02, 2 * sample_rate)
    marker = np.array([0.0, 5.0, -5.0, 3.0, -3.0, 0.0])
    source_markers = (0.15, 0.55, 1.2, 1.75)
    marker_frames = [round(time * sample_rate) for time in source_markers]
    for frame in marker_frames:
        source[frame : frame + marker.size] += marker
    points = ((0.0, 1.0), (0.4, 0.7), (0.8, 1.15))
    expected_length = audiosig.speech_effects_output_frames(
        source.size, sample_rate=sample_rate, rate_points=points
    )
    result = audiosig.apply_speech_effects_envelope(
        source, sample_rate=sample_rate, rate_points=points
    )
    assert result.size == expected_length

    timing = _RateMap(points)
    for source_frame in marker_frames:
        expected = round(
            timing.output_time_for_source_time(source_frame / sample_rate) * sample_rate
        )
        radius = round(0.1 * sample_rate)
        start = max(0, expected - radius)
        stop = min(result.size, expected + radius + marker.size)
        observed = start + int(np.argmax(np.abs(result[start:stop])))
        assert abs(observed - expected) <= round(0.06 * sample_rate)


def test_variable_pitch_tracks_linear_semitone_envelope_in_output_time() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    source = (
        0.16 * np.sin(2.0 * np.pi * 180.0 * time)
        + 0.08 * np.sin(2.0 * np.pi * 360.0 * time)
        + 0.04 * np.sin(2.0 * np.pi * 540.0 * time)
    ).astype(np.float32)
    result = audiosig.apply_speech_effects_envelope(
        source,
        sample_rate=sample_rate,
        pitch_points=((0.0, 0.0), (0.4, 4.0)),
    )
    assert result.shape == source.shape
    for center in (0.05, 0.15, 0.25, 0.35, 0.5):
        start = round((center - 0.04) * sample_rate)
        end = round((center + 0.04) * sample_rate)
        observed = dominant_frequency(result[start:end], sample_rate)
        semitones = 4.0 * min(center, 0.4) / 0.4
        expected = 180.0 * 2.0 ** (semitones / 12.0)
        assert observed == pytest.approx(expected, rel=0.06)


def test_combined_variable_rate_and_pitch_uses_one_coordinated_call() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    source = (
        0.16 * np.sin(2.0 * np.pi * 180.0 * time)
        + 0.08 * np.sin(2.0 * np.pi * 360.0 * time)
        + 0.04 * np.sin(2.0 * np.pi * 540.0 * time)
    ).astype(np.float32)
    rate_points = ((0.0, 1.0), (0.45, 0.85))
    result = audiosig.apply_speech_effects_envelope(
        source,
        sample_rate=sample_rate,
        rate_points=rate_points,
        pitch_points=((0.0, 0.0), (0.3, 2.0)),
    )
    assert result.size == audiosig.speech_effects_output_frames(
        source.size, sample_rate=sample_rate, rate_points=rate_points
    )
    assert np.isfinite(result).all()
    for center in (0.15, 0.45):
        start = round((center - 0.04) * sample_rate)
        end = round((center + 0.04) * sample_rate)
        observed = dominant_frequency(result[start:end], sample_rate)
        semitones = 2.0 * min(center, 0.3) / 0.3
        expected = 180.0 * 2.0 ** (semitones / 12.0)
        assert observed == pytest.approx(expected, rel=0.07)


def test_combined_envelopes_keep_voiced_to_unvoiced_fallback_on_rate_map() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate // 2, dtype=np.float64) / sample_rate
    voiced = (0.15 * np.sin(2.0 * np.pi * 180.0 * time)).astype(np.float32)
    unvoiced = np.random.default_rng(57).normal(0.0, 0.025, sample_rate // 2)
    marker = np.array([0.0, 2.0, -2.0, 1.5, -1.5, 0.0])
    marker_frame = round(0.8 * sample_rate) - sample_rate // 2
    unvoiced[marker_frame : marker_frame + marker.size] += marker
    source = np.concatenate([voiced, unvoiced]).astype(np.float32)
    rate_points = ((0.0, 1.0), (0.4, 0.95), (0.42, 0.9), (0.45, 0.85))
    result = audiosig.apply_speech_effects_envelope(
        source,
        sample_rate=sample_rate,
        rate_points=rate_points,
        pitch_points=((0.0, 0.0), (0.3, 2.0), (0.32, 2.5)),
    )
    assert result.size == audiosig.speech_effects_output_frames(
        source.size, sample_rate=sample_rate, rate_points=rate_points
    )
    assert np.isfinite(result).all()
    expected_marker = round(_RateMap(rate_points).output_time_for_source_time(0.8) * sample_rate)
    radius = round(0.08 * sample_rate)
    start = max(0, expected_marker - radius)
    stop = min(result.size, expected_marker + radius + marker.size)
    observed_marker = start + int(np.argmax(np.abs(result[start:stop])))
    assert abs(observed_marker - expected_marker) <= radius
    boundary = round(_RateMap(rate_points).output_time_for_source_time(0.5) * sample_rate)
    jumps = np.abs(np.diff(result[boundary - 100 : boundary + 100]))
    assert float(np.max(jumps)) < 0.5


def test_envelope_api_preserves_silence_for_all_automation_modes() -> None:
    sample_rate = 16_000
    silence = np.zeros(round(0.2 * sample_rate), dtype=np.float32)
    curves = (
        {"rate_points": ((0.0, 1.0), (0.3, 0.8))},
        {"pitch_points": ((0.0, 0.0), (0.3, 3.0))},
        {
            "rate_points": ((0.0, 1.0), (0.3, 0.8)),
            "pitch_points": ((0.0, 0.0), (0.3, 3.0)),
        },
    )
    for envelope in curves:
        result = audiosig.apply_speech_effects_envelope(
            silence, sample_rate=sample_rate, **envelope
        )
        assert result.size == audiosig.speech_effects_output_frames(
            silence.size,
            sample_rate=sample_rate,
            rate_points=envelope.get("rate_points", ()),
        )
        np.testing.assert_array_equal(result, 0.0)


def test_short_clip_does_not_compress_a_long_pitch_envelope() -> None:
    sample_rate = 16_000
    time = np.arange(round(0.12 * sample_rate), dtype=np.float64) / sample_rate
    source = (
        0.16 * np.sin(2.0 * np.pi * 180.0 * time)
        + 0.08 * np.sin(2.0 * np.pi * 360.0 * time)
        + 0.04 * np.sin(2.0 * np.pi * 540.0 * time)
    ).astype(np.float32)
    result = audiosig.apply_speech_effects_envelope(
        source, sample_rate=sample_rate, pitch_points=((0.0, 0.0), (0.3, 4.0))
    )
    assert result.shape == source.shape
    track = estimate_pitch_track_lane(result, sample_rate=sample_rate)
    late_voiced = track.voiced & (track.frame_times >= 0.07)
    measured = float(np.median(track.frequencies[late_voiced]))
    measured_time = float(np.median(track.frame_times[late_voiced]))
    expected = 180.0 * 2.0 ** ((4.0 * measured_time / 0.3) / 12.0)
    assert measured == pytest.approx(expected, rel=0.025)


def test_envelope_api_preserves_dtype_axis_and_shared_channel_timing() -> None:
    sample_rate = 16_000
    time = np.arange(round(0.4 * sample_rate), dtype=np.float64) / sample_rate
    first = (0.16 * np.sin(2.0 * np.pi * 180.0 * time)).astype(np.float32)
    second = (0.16 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
    frames_first = np.column_stack((first, second))
    rate_points = ((0.0, 1.0), (0.3, 0.8))
    result = audiosig.apply_speech_effects_envelope(
        frames_first,
        sample_rate=sample_rate,
        rate_points=rate_points,
        axis=0,
    )
    expected_frames = audiosig.speech_effects_output_frames(
        frames_first.shape[0], sample_rate=sample_rate, rate_points=rate_points
    )
    assert result.shape == (expected_frames, 2)
    assert result.dtype == frames_first.dtype
    np.testing.assert_array_equal(
        result,
        audiosig.apply_speech_effects_envelope(
            frames_first,
            sample_rate=sample_rate,
            rate_points=rate_points,
            axis=0,
        ),
    )
    np.testing.assert_array_equal(
        result[:, 0],
        audiosig.apply_speech_effects_envelope(
            first, sample_rate=sample_rate, rate_points=rate_points
        ),
    )
    np.testing.assert_array_equal(
        result[:, 1],
        audiosig.apply_speech_effects_envelope(
            second, sample_rate=sample_rate, rate_points=rate_points
        ),
    )


def test_envelope_api_preserves_empty_shape_and_dtype() -> None:
    empty = np.empty((2, 0, 3), dtype=np.float64)
    result = audiosig.apply_speech_effects_envelope(
        empty,
        sample_rate=16_000,
        rate_points=((0.0, 1.0), (0.3, 0.8)),
        axis=1,
    )
    assert result.shape == empty.shape
    assert result.dtype == empty.dtype
    assert not np.shares_memory(result, empty)


def test_public_output_frame_helper_uses_rate_map() -> None:
    assert (
        audiosig.speech_effects_output_frames(
            24_000,
            sample_rate=24_000,
            rate_points=((0.0, 0.5),),
        )
        == 48_000
    )
    assert audiosig.speech_effects_output_frames(24_000, sample_rate=24_000) == 24_000


def test_public_envelope_api_rejects_missing_curves_and_bad_points() -> None:
    audio = np.ones(32, dtype=np.float32)
    with pytest.raises(InvalidParameterError, match="at least one"):
        audiosig.apply_speech_effects_envelope(audio, sample_rate=24_000)
    with pytest.raises(InvalidParameterError):
        audiosig.apply_speech_effects_envelope(
            audio,
            sample_rate=24_000,
            rate_points=((0.1, 1.0),),
        )
    with pytest.raises(InvalidParameterError):
        audiosig.apply_speech_effects_envelope(
            audio,
            sample_rate=24_000,
            pitch_points=((0.0, np.inf),),
        )


@pytest.mark.parametrize("semitones", [12.0 * 1024.0, -12.0 * 1075.0])
def test_public_api_rejects_unrepresentable_pitch_ratios(semitones: float) -> None:
    with pytest.raises(InvalidParameterError, match="unrepresentable"):
        audiosig.apply_speech_effects_envelope(
            np.ones(32, dtype=np.float32),
            sample_rate=24_000,
            pitch_points=((0.0, semitones),),
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    [("time_base", "source"), ("interpolation", "cubic"), ("method", "esola")],
)
def test_public_envelope_api_rejects_unsupported_options(argument: str, value: str) -> None:
    kwargs: dict[str, object] = {
        "rate_points": ((0.0, 1.0),),
        argument: value,
    }
    with pytest.raises(InvalidParameterError):
        audiosig.apply_speech_effects_envelope(
            np.ones(32, dtype=np.float32), sample_rate=24_000, **kwargs
        )


def test_public_envelope_api_validates_sample_rate_before_empty_return() -> None:
    with pytest.raises(InvalidParameterError):
        audiosig.apply_speech_effects_envelope(
            np.empty(0, dtype=np.float32),
            sample_rate=0,
            pitch_points=((0.0, 0.0),),
        )
