"""Active speech-level analysis and reference gain policy primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validation import validate_audio, validate_finite, validate_integer, validate_positive
from .amplitude import apply_gain_db
from .exceptions import InvalidParameterError


@dataclass(frozen=True)
class SpeechLevel:
    """Active speech level and estimated active duration."""

    level_db: float
    active_seconds: float


@dataclass(frozen=True)
class ReferenceLevelResult:
    """Audio and measurements resulting from reference-level normalization."""

    audio: np.ndarray
    measured_db: float
    applied_gain_db: float
    resulting_db: float


def _frame_rms(audio: np.ndarray, frame_length: int, hop_length: int, floor: float) -> np.ndarray:
    frame_count = 1 + (audio.size - frame_length) // hop_length
    starts = np.arange(frame_count, dtype=np.int64) * hop_length
    result = np.empty(frame_count, dtype=np.float64)
    chunk_size = 1024
    for chunk_start in range(0, frame_count, chunk_size):
        chunk_stop = min(frame_count, chunk_start + chunk_size)
        indices = starts[chunk_start:chunk_stop, None] + np.arange(frame_length)[None, :]
        frames = audio[indices].astype(np.float64, copy=False)
        scales = np.max(np.abs(frames), axis=1)
        rms = np.zeros(scales.shape, dtype=np.float64)
        nonzero = scales > 0.0
        if np.any(nonzero):
            scaled = frames[nonzero] / scales[nonzero, None]
            rms[nonzero] = scales[nonzero] * np.sqrt(np.mean(np.square(scaled), axis=1))
        result[chunk_start:chunk_stop] = np.maximum(rms, floor)
    return result


def speech_level(
    audio: np.ndarray,
    *,
    sample_rate: int,
    frame_seconds: float = 0.025,
    hop_seconds: float = 0.010,
    floor_quantile: float = 0.20,
    activity_multiplier: float = 1.5,
    floor_amplitude: float = 1e-6,
) -> SpeechLevel:
    """Measure active speech level independently of integrated loudness.

    Frames are complete, uncentered windows. RMS is floored before percentile
    selection, active frames exceed ``floor_quantile * activity_multiplier``,
    and the lower median is used to match common tensor median semantics.
    """
    source, _ = validate_audio(audio, allow_empty=True)
    rate = validate_integer(sample_rate, "sample_rate")
    frame_duration = validate_positive(frame_seconds, "frame_seconds")
    hop_duration = validate_positive(hop_seconds, "hop_seconds")
    quantile = validate_finite(floor_quantile, "floor_quantile")
    if not 0.0 <= quantile <= 1.0:
        raise InvalidParameterError("floor_quantile must be in the interval [0, 1]")
    multiplier = validate_positive(activity_multiplier, "activity_multiplier")
    floor = validate_positive(floor_amplitude, "floor_amplitude")
    frame_length = int(rate * frame_duration)
    hop_length = int(rate * hop_duration)
    if frame_length < 1 or hop_length < 1:
        raise InvalidParameterError("frame_seconds and hop_seconds must span at least one sample")

    flat = source.reshape(-1).astype(np.float64, copy=False)
    if flat.size < frame_length:
        if flat.size == 0:
            whole_rms = floor
        else:
            scale = float(np.max(np.abs(flat)))
            whole_rms = (
                0.0 if scale == 0.0 else scale * float(np.sqrt(np.mean(np.square(flat / scale))))
            )
            whole_rms = max(whole_rms, floor)
        return SpeechLevel(level_db=float(20.0 * np.log10(whole_rms)), active_seconds=0.0)

    frame_rms = _frame_rms(flat, frame_length, hop_length, floor)
    threshold = float(np.quantile(frame_rms, quantile)) * multiplier
    active = frame_rms[frame_rms > threshold]
    if active.size == 0:
        active = frame_rms
    median_index = (active.size - 1) // 2
    median_rms = float(np.partition(active, median_index)[median_index])
    level_db = float(20.0 * np.log10(median_rms))
    active_seconds = float(active.size * hop_length / rate)
    return SpeechLevel(level_db=level_db, active_seconds=active_seconds)


def normalize_reference_level(
    audio: np.ndarray,
    *,
    sample_rate: int,
    target_db: float = -19.8,
    max_gain_db: float = 30.0,
    peak_ceiling: float = 0.95,
) -> ReferenceLevelResult:
    """Boost a reference toward ``target_db`` subject to gain and peak limits."""
    source, _ = validate_audio(audio, allow_empty=True)
    target = validate_finite(target_db, "target_db")
    maximum_gain = validate_finite(max_gain_db, "max_gain_db")
    if maximum_gain < 0.0:
        raise InvalidParameterError("max_gain_db must be non-negative")
    ceiling = validate_positive(peak_ceiling, "peak_ceiling")
    measured_db = speech_level(source, sample_rate=sample_rate).level_db
    gain_db = min(max(target - measured_db, 0.0), maximum_gain)
    if source.size:
        peak = float(np.max(np.abs(source)))
        if peak > 0.0:
            peak_gain_db = max(0.0, 20.0 * np.log10(ceiling / peak))
            gain_db = min(gain_db, peak_gain_db)
    output = apply_gain_db(source, gain_db)
    return ReferenceLevelResult(
        audio=output,
        measured_db=measured_db,
        applied_gain_db=float(gain_db),
        resulting_db=float(measured_db + gain_db),
    )
