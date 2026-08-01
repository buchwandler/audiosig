"""Private, experimental time-domain PSOLA speech synthesis."""

from __future__ import annotations

import numpy as np

from ._pitch import PitchTrack, estimate_pitch_track_lane
from ._resampling import resample_to_length
from ._validation import validate_audio, validate_finite, validate_positive
from .exceptions import InvalidParameterError

_MIN_RATE = 0.75
_MAX_RATE = 1.5
_MIN_SEMITONES = -6.0
_MAX_SEMITONES = 6.0
_EPSILON = np.finfo(np.float64).eps


def _padded_slice(signal: np.ndarray, start: int, length: int) -> np.ndarray:
    result = np.zeros(length, dtype=np.float64)
    requested_start = int(start)
    source_start = max(0, min(requested_start, signal.size))
    source_end = min(signal.size, requested_start + length)
    if source_end > source_start:
        result_start = source_start - requested_start
        result[result_start : result_start + source_end - source_start] = signal[
            source_start:source_end
        ]
    return result


def _target_marks(
    source_marks: np.ndarray,
    *,
    source_length: int,
    target_length: int,
    rate: float,
    pitch_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate output pulse locations and their warped source positions."""

    if source_marks.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    first = float(source_marks[0]) / rate
    last = min(float(target_length - 1), float(source_marks[-1]) / rate)
    output_positions: list[int] = []
    source_positions: list[float] = []
    position = first
    while position <= last + 1.0:
        source_position = position * rate
        nearest = int(np.searchsorted(source_marks, source_position, side="left"))
        nearest = min(nearest, source_marks.size - 1)
        if nearest > 0 and abs(source_marks[nearest - 1] - source_position) < abs(
            source_marks[nearest] - source_position
        ):
            nearest -= 1
        if 0 <= source_position < source_length:
            output_positions.append(round(position))
            source_positions.append(float(source_marks[nearest]))
        source_mark_index = nearest
        if source_mark_index + 1 < source_marks.size:
            local_period = float(
                source_marks[source_mark_index + 1] - source_marks[source_mark_index]
            )
        elif source_mark_index > 0:
            local_period = float(
                source_marks[source_mark_index] - source_marks[source_mark_index - 1]
            )
        else:
            local_period = 1.0
        # Duration changes warp where source material is read, but synthesis
        # pulse spacing controls F0 and therefore must not include ``rate``.
        position += max(1.0, local_period / pitch_ratio)
    if not output_positions:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    positions = np.asarray(output_positions, dtype=np.int64)
    source = np.asarray(source_positions, dtype=np.float64)
    keep = np.r_[True, np.diff(positions) > 0]
    return positions[keep], source[keep]


def _extract_pitch_grain(
    signal: np.ndarray, marks: np.ndarray, index: int
) -> tuple[np.ndarray, int]:
    mark = int(marks[index])
    previous_period = (
        marks[index] - marks[index - 1] if index > 0 else marks[min(1, marks.size - 1)] - mark
    )
    next_period = (
        marks[index + 1] - marks[index]
        if index + 1 < marks.size
        else mark - marks[max(0, index - 1)]
    )
    previous_period = max(2, int(previous_period))
    next_period = max(2, int(next_period))
    # Short TD-PSOLA windows need support from both neighboring periods.  A
    # half-period on either side produces only one period total and leaves
    # deterministic gaps when synthesis marks are spread for pitch lowering.
    left = previous_period
    right = next_period
    grain_length = left + right + 1
    return _padded_slice(signal, mark - left, grain_length), left


def _overlap_add_grains(
    signal: np.ndarray,
    source_marks: np.ndarray,
    target_marks: np.ndarray,
    mapped_source_marks: np.ndarray,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros(target_length, dtype=np.float64)
    normalization = np.zeros(target_length, dtype=np.float64)
    if source_marks.size == 0:
        return output, normalization
    source_index = 0
    for target, source_mark in zip(target_marks, mapped_source_marks, strict=True):
        while source_index + 1 < source_marks.size and abs(
            source_marks[source_index + 1] - source_mark
        ) <= abs(source_marks[source_index] - source_mark):
            source_index += 1
        index = source_index
        grain, center = _extract_pitch_grain(signal, source_marks, index)
        window = np.hanning(grain.size)
        start = int(target) - center
        left = max(0, start)
        right = min(target_length, start + grain.size)
        if right <= left:
            continue
        grain_start = left - start
        grain_end = grain_start + right - left
        output[left:right] += grain[grain_start:grain_end] * window[grain_start:grain_end]
        normalization[left:right] += window[grain_start:grain_end]
    return output, normalization


def _voiced_output_mask(
    track: PitchTrack,
    *,
    sample_rate: int,
    rate: float,
    target_length: int,
    normalization: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(target_length, dtype=np.float64)
    if track.voiced.size == 0:
        return mask
    for source_start, source_end in track.voiced_intervals:
        start = max(0, round(source_start / rate))
        end = min(target_length, round(source_end / rate))
        mask[start:end] = 1.0
    # Normalization is only a numerical support check.  Voicing itself comes
    # from the track's actual sample intervals above.
    mask *= normalization > 0.25
    fade = max(1, round(sample_rate * 0.004))
    if fade > 1 and np.any(mask):
        kernel = np.hanning(2 * fade + 1)
        kernel /= np.sum(kernel)
        mask = np.convolve(mask, kernel, mode="same")
        mask = np.clip(mask * 1.8, 0.0, 1.0)
    return mask


def _voiced_td_psola_lane(
    signal: np.ndarray,
    track: PitchTrack,
    *,
    sample_rate: int,
    target_length: int,
    rate: float,
    pitch_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    target_marks, mapped_source_marks = _target_marks(
        track.pitch_marks,
        source_length=signal.size,
        target_length=target_length,
        rate=rate,
        pitch_ratio=pitch_ratio,
    )
    output, normalization = _overlap_add_grains(
        signal,
        track.pitch_marks,
        target_marks,
        mapped_source_marks,
        target_length,
    )
    output /= np.maximum(normalization, _EPSILON)
    mask = _voiced_output_mask(
        track,
        sample_rate=sample_rate,
        rate=rate,
        target_length=target_length,
        normalization=normalization,
    )
    return output, mask


def _duration_fallback(
    signal: np.ndarray,
    *,
    sample_rate: int,
    rate: float,
    target_length: int,
) -> np.ndarray:
    """Apply duration-only WSOLA fallback with exact target length."""
    from ._wsola import wsola_time_stretch

    if rate == 1.0:
        return np.asarray(signal, dtype=np.float64).copy()
    fallback = np.asarray(
        wsola_time_stretch(signal[None, :], rate=rate, sample_rate=sample_rate, axis=-1)[0],
        dtype=np.float64,
    )
    if fallback.size != target_length:
        fallback = np.asarray(resample_to_length(fallback, target_length), dtype=np.float64)
    return fallback


def _pitch_duration_fallback(
    signal: np.ndarray,
    *,
    sample_rate: int,
    rate: float,
    pitch_ratio: float,
    target_length: int,
) -> np.ndarray:
    """Best-effort TSM plus resampling fallback when direct marks are absent."""
    from ._wsola import wsola_time_stretch

    tsm_rate = rate / pitch_ratio
    if tsm_rate == 1.0:
        stretched = np.asarray(signal, dtype=np.float64).copy()
    else:
        stretched = np.asarray(
            wsola_time_stretch(signal[None, :], rate=tsm_rate, sample_rate=sample_rate, axis=-1)[0],
            dtype=np.float64,
        )
    return np.asarray(resample_to_length(stretched, target_length), dtype=np.float64)


def _validate_limits(rate: float, semitones: float) -> tuple[float, float]:
    stretch = validate_positive(rate, "rate")
    shift = validate_finite(semitones, "semitones")
    if not _MIN_RATE <= stretch <= _MAX_RATE:
        raise InvalidParameterError(
            f"td_psola rate must be in the interval [{_MIN_RATE}, {_MAX_RATE}]"
        )
    if not _MIN_SEMITONES <= shift <= _MAX_SEMITONES:
        raise InvalidParameterError(
            f"td_psola semitones must be in the interval [{_MIN_SEMITONES}, {_MAX_SEMITONES}]"
        )
    return stretch, shift


def _td_psola_lane(
    signal: np.ndarray,
    *,
    sample_rate: int,
    rate: float,
    semitones: float,
    target_length: int,
    pitch_floor: float,
    pitch_ceiling: float,
) -> np.ndarray:
    track = estimate_pitch_track_lane(
        signal,
        sample_rate=sample_rate,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
    )
    pitch_ratio = float(np.exp2(semitones / 12.0))
    if track.pitch_marks.size < 3:
        # With duration modification, preserve unvoiced material rather than
        # applying a global resampling pitch shift to noise.  A pure pitch
        # request still receives the explicit best-effort pitch fallback.
        if rate != 1.0:
            return _duration_fallback(
                signal,
                sample_rate=sample_rate,
                rate=rate,
                target_length=target_length,
            )
        return _pitch_duration_fallback(
            signal,
            sample_rate=sample_rate,
            rate=rate,
            pitch_ratio=pitch_ratio,
            target_length=target_length,
        )
    direct, mask = _voiced_td_psola_lane(
        signal,
        track,
        sample_rate=sample_rate,
        target_length=target_length,
        rate=rate,
        pitch_ratio=pitch_ratio,
    )
    fallback = _duration_fallback(
        signal,
        sample_rate=sample_rate,
        rate=rate,
        target_length=target_length,
    )
    return np.asarray(direct * mask + fallback * (1.0 - mask), dtype=np.float64)


def td_psola_prosody(
    audio: np.ndarray,
    *,
    sample_rate: int,
    rate: float = 1.0,
    semitones: float = 0.0,
    axis: int = -1,
    pitch_floor: float = 60.0,
    pitch_ceiling: float = 500.0,
) -> np.ndarray:
    """Apply conservative direct TD-PSOLA pitch and duration modification.

    This private backend is intended for speech experiments.  Unvoiced or
    unreliable material is duration-scaled with WSOLA and is never forced
    through voiced-grain synthesis.
    """

    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    stretch, shift = _validate_limits(rate, semitones)
    sample_hz = int(validate_positive(sample_rate, "sample_rate"))
    floor = validate_positive(pitch_floor, "pitch_floor")
    ceiling = validate_positive(pitch_ceiling, "pitch_ceiling")
    if ceiling <= floor or ceiling >= sample_hz / 2.0:
        raise InvalidParameterError(
            "pitch_ceiling must be greater than pitch_floor and below Nyquist"
        )
    input_length = source.shape[normalized_axis]
    if input_length == 0:
        return np.array(source, dtype=source.dtype, copy=True)
    if stretch == 1.0 and shift == 0.0:
        return np.array(source, dtype=source.dtype, copy=True)
    requested_length = input_length / stretch
    if not np.isfinite(requested_length) or requested_length > np.iinfo(np.intp).max:
        raise InvalidParameterError("requested TD-PSOLA length is too large")
    target_length = max(1, round(requested_length))
    moved = np.moveaxis(source, normalized_axis, -1)
    flat = moved.reshape((-1, input_length))
    lane_count = int(flat.size // input_length)
    output = np.empty((lane_count, target_length), dtype=np.float64)
    for lane_index, lane in enumerate(flat):
        output[lane_index] = _td_psola_lane(
            lane,
            sample_rate=sample_hz,
            rate=stretch,
            semitones=shift,
            target_length=target_length,
            pitch_floor=floor,
            pitch_ceiling=ceiling,
        )
    reshaped = output.reshape((*moved.shape[:-1], target_length))
    return np.moveaxis(reshaped, -1, normalized_axis).astype(source.dtype, copy=False)
