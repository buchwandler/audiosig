"""Composable numeric speech effects."""

from __future__ import annotations

import numpy as np

from ._resampling import resample_to_length
from ._validation import (
    validate_audio,
    validate_boolean,
    validate_filter,
    validate_finite,
    validate_gain_db,
    validate_integer,
    validate_positive,
)
from .amplitude import apply_gain_db
from .effects import TimeStretchMethod, time_stretch
from .exceptions import InvalidParameterError


def apply_speech_effects(
    audio: np.ndarray,
    *,
    sample_rate: int,
    rate: float = 1.0,
    semitones: float = 0.0,
    gain_db: float = 0.0,
    axis: int = -1,
    clip: bool = False,
    method: TimeStretchMethod = "wsola",
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
    if method not in ("wsola", "phase_vocoder", "esola"):
        raise InvalidParameterError("method must be one of ('wsola', 'phase_vocoder', 'esola')")

    result = np.array(source, dtype=source.dtype, copy=True)
    if result.shape[normalized_axis] == 0:
        return result

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
