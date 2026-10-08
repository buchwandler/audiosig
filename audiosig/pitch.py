"""Stable public pitch analysis API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._pitch import estimate_pitch_track_lane
from ._validation import validate_audio, validate_integer, validate_positive
from .exceptions import AudioShapeError, InvalidParameterError


@dataclass(frozen=True, slots=True)
class PitchTrack:
    """Public per-frame F0, voicing, confidence, and timing arrays."""

    frame_times_s: np.ndarray
    f0_hz: np.ndarray
    voiced: np.ndarray
    confidence: np.ndarray


def pitch_track(
    audio: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int | None = None,
    f0_min: float = 60.0,
    f0_max: float = 500.0,
) -> PitchTrack:
    """Estimate a mono F0 track using AudioSig's existing private tracker.

    The input must have shape ``(samples,)``. Multichannel callers must choose
    a channel or explicitly downmix before invoking this function.
    """
    source, axis = validate_audio(audio)
    if source.ndim != 1 or axis != 0:
        raise AudioShapeError("pitch_track accepts mono audio shaped (samples,)")
    rate = validate_integer(sample_rate, "sample_rate")
    floor = validate_positive(f0_min, "f0_min")
    ceiling = validate_positive(f0_max, "f0_max")
    if ceiling <= floor:
        raise InvalidParameterError("f0_max must be greater than f0_min")
    if ceiling >= rate / 2.0:
        raise InvalidParameterError("f0_max must be below the Nyquist frequency")
    hop = None if hop_length is None else validate_integer(hop_length, "hop_length")

    track = estimate_pitch_track_lane(
        source,
        sample_rate=rate,
        pitch_floor=floor,
        pitch_ceiling=ceiling,
        hop_length=hop,
    )
    return PitchTrack(
        frame_times_s=np.array(track.frame_times, dtype=np.float64, copy=True),
        f0_hz=np.array(track.frequencies, dtype=np.float64, copy=True),
        voiced=np.array(track.voiced, dtype=bool, copy=True),
        confidence=np.array(track.confidence, dtype=np.float64, copy=True),
    )
