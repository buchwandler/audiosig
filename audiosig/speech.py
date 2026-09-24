"""Composable numeric speech effects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from ._automation import _LinearEnvelope, _RateMap, _validate_points
from ._resampling import resample_to_length
from ._validation import (
    validate_audio,
    validate_boolean,
    validate_choices,
    validate_filter,
    validate_finite,
    validate_gain_db,
    validate_integer,
    validate_positive,
)
from .amplitude import apply_gain_db
from .effects import time_stretch
from .exceptions import InvalidParameterError

SpeechEffectsEnvelopeMethod = Literal["wsola", "td_psola"]
SpeechEffectsMethod = Literal["phase_vocoder", "wsola", "esola", "td_psola"]


def apply_speech_effects(
    audio: np.ndarray,
    *,
    sample_rate: int,
    rate: float = 1.0,
    semitones: float = 0.0,
    gain_db: float = 0.0,
    axis: int = -1,
    clip: bool = False,
    method: SpeechEffectsMethod = "wsola",
    n_fft: int = 2048,
    hop_length: int | None = None,
    filter_width: int = 32,
    rolloff: float = 0.945,
) -> np.ndarray:
    """Apply pitch, pitch-preserving rate, and gain to speech.

    Pitch and rate are planned as one time-scale modification pass followed
    by at most one resample. ``wsola`` is the speech-oriented default, while
    ``phase_vocoder`` and experimental ``esola`` remain available for
    compatibility and comparison. ESOLA supports computed TSM rates from 0.5
    through 2.0.
    This function accepts numeric values only; downstream applications remain
    responsible for parsing SSMD or other user-facing effect syntax.
    """
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    validate_positive(sample_rate, "sample_rate")
    stretch = validate_positive(rate, "rate")
    shift = validate_finite(semitones, "semitones")
    gain = validate_gain_db(gain_db, "gain_db")
    clip_value = validate_boolean(clip, "clip")
    fft_size = validate_integer(n_fft, "n_fft", minimum=2)
    hop = validate_integer(
        hop_length if hop_length is not None else fft_size // 4,
        "hop_length",
    )
    if hop > fft_size:
        raise InvalidParameterError("hop_length must not exceed n_fft")
    width, _ = validate_filter(filter_width, rolloff)
    if method not in ("wsola", "phase_vocoder", "esola", "td_psola"):
        raise InvalidParameterError(
            "method must be one of ('wsola', 'phase_vocoder', 'esola', 'td_psola')"
        )

    result = np.array(source, dtype=source.dtype, copy=True)
    if result.shape[normalized_axis] == 0:
        return result

    if method == "td_psola":
        from ._td_psola import td_psola_prosody

        result = td_psola_prosody(
            result,
            sample_rate=int(sample_rate),
            rate=stretch,
            semitones=shift,
            axis=normalized_axis,
        )
        if gain != 0.0 or gain == -np.inf:
            result = apply_gain_db(result, gain, clip=clip_value)
        elif clip_value:
            result = np.clip(result, -1.0, 1.0).astype(source.dtype, copy=False)
        return np.array(result, dtype=source.dtype, copy=True)

    octaves = shift / 12.0
    max_octaves = np.log2(np.finfo(np.float64).max)
    min_octaves = np.log2(np.nextafter(0.0, 1.0))
    if not min_octaves <= octaves <= max_octaves:
        raise InvalidParameterError("semitones produces an unrepresentable pitch ratio")
    pitch_ratio = float(np.exp2(octaves))
    tsm_rate = stretch / pitch_ratio
    target_length = max(1, round(result.shape[normalized_axis] / stretch))
    if not np.isclose(tsm_rate, 1.0):
        result = time_stretch(
            result,
            tsm_rate,
            sample_rate=int(sample_rate),
            method=method,
            axis=normalized_axis,
            n_fft=fft_size,
            hop_length=hop,
        )
    if result.shape[normalized_axis] != target_length:
        result = resample_to_length(
            result,
            target_length,
            axis=normalized_axis,
            filter_width=width,
            rolloff=rolloff,
        )
    if gain != 0.0 or gain == -np.inf:
        result = apply_gain_db(result, gain, clip=clip_value)
    elif clip_value:
        result = np.clip(result, -1.0, 1.0).astype(source.dtype, copy=False)
    return np.array(result, dtype=source.dtype, copy=True)


def speech_effects_output_frames(
    input_frames: int,
    *,
    sample_rate: int,
    rate_points: Sequence[tuple[float, float]] = (),
) -> int:
    """Return exact rounded output frames for a speech rate envelope.

    Control-point times are measured in output seconds. An omitted rate curve
    means a constant rate of 1.0.
    """
    return _RateMap(rate_points).output_frames_for_input_frames(input_frames, sample_rate)


def apply_speech_effects_envelope(
    audio: np.ndarray,
    *,
    sample_rate: int,
    rate_points: Sequence[tuple[float, float]] = (),
    pitch_points: Sequence[tuple[float, float]] = (),
    time_base: Literal["output"] = "output",
    interpolation: Literal["linear"] = "linear",
    method: SpeechEffectsEnvelopeMethod = "wsola",
    axis: int = -1,
) -> np.ndarray:
    """Apply numeric output-time rate and semitone pitch envelopes to speech.

    Each supplied curve starts at output time zero, interpolates linearly, and
    holds its final value. Rate values are positive playback factors. Pitch
    values are semitone offsets. Downstream applications own semantic policy.
    """
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    sample_hz = validate_integer(sample_rate, "sample_rate")
    validate_choices(time_base, ("output",), "time_base")
    validate_choices(interpolation, ("linear",), "interpolation")
    validate_choices(method, ("wsola", "td_psola"), "method")
    rates = _validate_points(
        rate_points, name="rate_points", positive_values=True, allow_empty=True
    )
    pitches = _validate_points(pitch_points, name="pitch_points", allow_empty=True)
    max_octaves = np.log2(np.finfo(np.float64).max)
    min_octaves = np.log2(np.nextafter(0.0, 1.0))
    if any(not min_octaves <= semitones / 12.0 < max_octaves for _, semitones in pitches):
        raise InvalidParameterError("pitch points produce an unrepresentable pitch ratio")
    if not rates and not pitches:
        raise InvalidParameterError("at least one of rate_points or pitch_points is required")

    rate_envelope = _LinearEnvelope(rates, default=1.0, name="rate_points", positive_values=True)
    pitch_envelope = _LinearEnvelope(pitches, default=0.0, name="pitch_points")
    rate_value = (
        rate_envelope.value_at(0.0)
        if not rates or all(point[1] == rates[0][1] for point in rates)
        else None
    )
    pitch_value = (
        pitch_envelope.value_at(0.0)
        if not pitches or all(point[1] == pitches[0][1] for point in pitches)
        else None
    )
    input_frames = source.shape[normalized_axis]
    rate_map = _RateMap(rates)
    target_frames = rate_map.output_frames_for_input_frames(input_frames, sample_hz)
    if input_frames == 0:
        return np.array(source, copy=True)
    if rate_value is not None and pitch_value is not None:
        return apply_speech_effects(
            source,
            sample_rate=sample_hz,
            rate=rate_value,
            semitones=pitch_value,
            method=method,
            axis=normalized_axis,
        )
    if rate_value is None and pitch_value == 0.0 and method == "wsola":
        from ._wsola import wsola_time_stretch_rate_map

        result = wsola_time_stretch_rate_map(
            source,
            rate_map=rate_map,
            target_length=target_frames,
            sample_rate=sample_hz,
            axis=normalized_axis,
        )
        return np.array(result, dtype=source.dtype, copy=True)
    from ._td_psola import td_psola_prosody_envelope

    return td_psola_prosody_envelope(
        source,
        sample_rate=sample_hz,
        target_length=target_frames,
        rate_map=rate_map,
        pitch_envelope=pitch_envelope,
        axis=normalized_axis,
    )
