"""NumPy-only epoch-synchronous overlap-add time stretching for speech."""

from __future__ import annotations

import numpy as np

from ._validation import validate_audio, validate_positive
from .exceptions import InvalidParameterError

_DEFAULT_FRAME_MS = 20.0
_DEFAULT_OVERLAP = 0.5
_DEFAULT_TREND_WINDOW_MS = 30.0
_MIN_RATE = 0.5
_MAX_RATE = 2.0


def _centered_moving_average(values: np.ndarray, radius: int) -> np.ndarray:
    """Return an edge-padded centered moving average in O(N) time."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return np.array(values, dtype=np.float64, copy=True)
    padded = np.pad(values, (radius, radius), mode="edge")
    cumulative = np.cumsum(padded, dtype=np.float64)
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), cumulative))
    width = 2 * radius + 1
    return (cumulative[width:] - cumulative[:-width]) / width


def _zero_frequency_signal(values: np.ndarray, trend_radius: int) -> np.ndarray:
    """Generate and detrend the zero-frequency resonator signal."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    difference = np.empty(values.size, dtype=np.float64)
    difference[0] = values[0]
    difference[1:] = values[1:] - values[:-1]
    resonator = difference
    for _ in range(4):
        resonator = np.cumsum(resonator, dtype=np.float64)
    for _ in range(2):
        resonator = resonator - _centered_moving_average(resonator, trend_radius)
    return resonator


def _positive_zero_crossings(values: np.ndarray) -> np.ndarray:
    """Return deterministic indices where a signal crosses from negative to non-negative."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if values.size < 2:
        return np.empty(0, dtype=np.int64)
    crossings = (values[:-1] < 0.0) & (values[1:] >= 0.0)
    return (np.flatnonzero(crossings) + 1).astype(np.int64, copy=False)


def _extract_epochs_lane(
    source: np.ndarray,
    sample_rate: int,
    *,
    trend_window_ms: float = _DEFAULT_TREND_WINDOW_MS,
) -> np.ndarray:
    """Extract positive zero crossings for one speech lane."""
    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("source must be one-dimensional")
    if source.size == 0 or not np.any(source):
        return np.empty(0, dtype=np.int64)
    radius = max(1, round(sample_rate * trend_window_ms / 2000.0))
    return _positive_zero_crossings(_zero_frequency_signal(source, radius))


def _analysis_shift(
    synthesis_epoch_relative: int | None,
    analysis_epoch_relatives: np.ndarray,
    max_shift: int,
) -> int:
    """Choose the smallest valid non-negative epoch alignment shift."""
    if synthesis_epoch_relative is None or max_shift < 0:
        return 0
    candidates = np.asarray(analysis_epoch_relatives, dtype=np.int64) - int(
        synthesis_epoch_relative
    )
    valid = candidates[(candidates >= 0) & (candidates <= max_shift)]
    return int(np.min(valid)) if valid.size else 0


def _padded_slice(source: np.ndarray, start: int, length: int) -> np.ndarray:
    """Return a float64 source slice with deterministic edge padding."""
    result = np.zeros(length, dtype=np.float64)
    if source.size == 0:
        return result
    clipped_start = max(0, min(int(start), source.size))
    available = min(length, source.size - clipped_start)
    if available > 0:
        result[:available] = source[clipped_start : clipped_start + available]
    if available < length:
        result[available:] = source[-1]
    return result


def _raised_cosine_crossfade(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return constant-sum fade-out and fade-in curves."""
    phase = np.linspace(0.0, 1.0, length, endpoint=True, dtype=np.float64)
    fade_in = 0.5 - 0.5 * np.cos(np.pi * phase)
    return 1.0 - fade_in, fade_in


def _esola_lane(source: np.ndarray, rate: float, sample_rate: int) -> np.ndarray:
    """Synthesize one lane using fixed-geometry ESOLA."""
    input_length = source.size
    target_length = max(1, round(input_length / rate))
    frame_length = max(4, round(sample_rate * _DEFAULT_FRAME_MS / 1000.0))
    synthesis_hop = max(1, frame_length // 2)
    overlap_length = frame_length - synthesis_hop
    alpha = target_length / input_length
    analysis_hop = synthesis_hop / alpha
    frame_count = max(1, int(np.ceil(target_length / synthesis_hop)))
    work_length = target_length + frame_length

    epochs = _extract_epochs_lane(source, sample_rate)
    output = np.zeros(work_length, dtype=np.float64)
    output_epochs = np.zeros(work_length, dtype=bool)
    fade_out, fade_in = _raised_cosine_crossfade(overlap_length)

    for frame_index in range(frame_count):
        synthesis_start = frame_index * synthesis_hop
        if synthesis_start >= target_length:
            break
        nominal_analysis_start = round(frame_index * analysis_hop)
        if frame_index == 0:
            shift = 0
        else:
            marked = output_epochs[synthesis_start : synthesis_start + frame_length]
            marked_positions = np.flatnonzero(marked)
            synthesis_epoch = int(marked_positions[0]) if marked_positions.size else None
            analysis_mask = (epochs >= nominal_analysis_start) & (
                epochs < nominal_analysis_start + frame_length
            )
            analysis_relative = epochs[analysis_mask] - nominal_analysis_start
            shift = _analysis_shift(synthesis_epoch, analysis_relative, synthesis_hop)

        shifted_start = nominal_analysis_start + shift
        frame = _padded_slice(source, shifted_start, frame_length)
        output_end = min(work_length, synthesis_start + frame_length)
        frame_size = output_end - synthesis_start
        if frame_index == 0:
            output[synthesis_start:output_end] = frame[:frame_size]
        else:
            overlap_end = min(output_end, synthesis_start + overlap_length)
            overlap_size = overlap_end - synthesis_start
            if overlap_size > 0:
                output[synthesis_start:overlap_end] = (
                    output[synthesis_start:overlap_end] * fade_out[:overlap_size]
                    + frame[:overlap_size] * fade_in[:overlap_size]
                )
            if output_end > synthesis_start + overlap_length:
                tail_start = synthesis_start + overlap_length
                output[tail_start:output_end] = frame[overlap_length:frame_size]

        shifted_epochs = epochs[(epochs >= shifted_start) & (epochs < shifted_start + frame_length)]
        output_epoch_positions = synthesis_start + shifted_epochs - shifted_start
        valid_positions = output_epoch_positions[
            (output_epoch_positions >= 0) & (output_epoch_positions < work_length)
        ]
        output_epochs[valid_positions] = True

    return output[:target_length]


def esola_time_stretch(
    audio: np.ndarray,
    rate: float,
    *,
    sample_rate: int,
    axis: int = -1,
) -> np.ndarray:
    """Stretch speech with clean-room NumPy ESOLA synthesis.

    AudioSig rates above one produce shorter output. The initial implementation
    is intentionally limited to the moderate range documented by the ESOLA
    paper and AudioSig's speech-quality plan.
    """
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    multiplier = validate_positive(rate, "rate")
    sample_hz = validate_positive(sample_rate, "sample_rate")
    if not _MIN_RATE <= multiplier <= _MAX_RATE:
        raise InvalidParameterError(
            f"ESOLA rate must be in the interval [{_MIN_RATE}, {_MAX_RATE}]; "
            f"received backend rate {multiplier:g}. Select WSOLA or phase vocoder "
            "outside this range."
        )
    input_length = source.shape[normalized_axis]
    if input_length == 0:
        return np.array(source, dtype=source.dtype, copy=True)
    requested_length = input_length / multiplier
    if not np.isfinite(requested_length) or requested_length > np.iinfo(np.intp).max:
        raise InvalidParameterError("requested stretched length is too large")
    target_length = max(1, round(requested_length))
    if multiplier == 1.0:
        return np.array(source, dtype=source.dtype, copy=True)
    moved = np.moveaxis(source, normalized_axis, -1)
    flat = moved.reshape((-1, input_length))
    if input_length == 1:
        stretched = np.repeat(flat, target_length, axis=1)
    elif flat.size == 0:
        stretched = np.empty((0, target_length), dtype=np.float64)
    else:
        stretched = np.stack(
            [_esola_lane(lane, multiplier, int(sample_hz)) for lane in flat], axis=0
        )
    reshaped = stretched.reshape((*moved.shape[:-1], target_length))
    return np.moveaxis(reshaped, -1, normalized_axis).astype(source.dtype, copy=False)
