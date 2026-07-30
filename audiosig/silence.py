"""Silence trimming, framing, feature extraction, and lightweight VAD."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from ._framing import frame
from ._validation import (
    validate_audio,
    validate_choices,
    validate_integer,
    validate_positive,
)
from .exceptions import InvalidParameterError

Reference = float | Callable[[np.ndarray], float]
Aggregate = Callable[..., object]


def abs2(values: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Return squared magnitude for real or complex values."""
    array = np.asarray(values)
    if np.iscomplexobj(array):
        result = array.real * array.real + array.imag * array.imag
        return cast(np.ndarray, result.astype(dtype, copy=False) if dtype is not None else result)
    return cast(np.ndarray, np.square(array, dtype=dtype))


def _reference_value(values: np.ndarray, ref: Reference) -> float:
    raw = ref(values) if callable(ref) else abs(float(ref))
    try:
        raw_array = np.asarray(raw)
        if raw_array.ndim != 0:
            raise TypeError
        result = float(raw_array)
    except (TypeError, ValueError):
        raise InvalidParameterError("ref callable must resolve to a scalar") from None
    if not np.isfinite(result) or result < 0:
        raise InvalidParameterError("ref must resolve to a finite non-negative value")
    return result


def power_to_db(
    power: np.ndarray,
    *,
    ref: Reference = 1.0,
    amin: float = 1e-10,
    top_db: float | None = None,
) -> np.ndarray:
    """Convert power values to decibels with optional peak clipping.

    ``ref`` may be a numeric power level or a callable such as ``np.max``.
    When ``top_db`` is provided, values are clipped to that distance below the
    loudest result.
    """
    floor = float(amin)
    if floor <= 0 or not np.isfinite(floor):
        raise InvalidParameterError("amin must be finite and positive")
    values = np.asarray(power)
    magnitude = np.abs(values) if np.iscomplexobj(values) else values
    if magnitude.size == 0:
        return np.asarray(magnitude, dtype=np.float64)
    reference = _reference_value(np.asarray(magnitude), ref)
    log_spec = 10.0 * np.log10(np.maximum(floor, magnitude))
    log_spec -= 10.0 * np.log10(max(floor, reference))
    if top_db is not None:
        threshold = float(top_db)
        if not np.isfinite(threshold) or threshold < 0:
            raise InvalidParameterError("top_db must be finite and non-negative")
        log_spec = np.maximum(log_spec, float(np.max(log_spec)) - threshold)
    return cast(np.ndarray, log_spec)


def amplitude_to_db(
    amplitude: np.ndarray,
    *,
    ref: Reference = 1.0,
    amin: float = 1e-5,
    top_db: float | None = None,
) -> np.ndarray:
    """Convert amplitude values to decibels with optional peak clipping."""
    floor = float(amin)
    if floor <= 0 or not np.isfinite(floor):
        raise InvalidParameterError("amin must be finite and positive")
    magnitude = np.abs(np.asarray(amplitude))
    if magnitude.size == 0:
        return np.asarray(magnitude, dtype=np.float64)
    reference = _reference_value(magnitude, ref)
    return power_to_db(
        abs2(magnitude, dtype=np.float64),
        ref=reference * reference,
        amin=floor * floor,
        top_db=top_db,
    )


def frames_to_samples(
    frames: np.ndarray | int,
    *,
    hop_length: int = 512,
    n_fft: int | None = None,
) -> np.ndarray | int:
    """Convert frame indices to sample indices."""
    hop = validate_integer(hop_length, "hop_length")
    offset = validate_integer(n_fft, "n_fft", minimum=2) // 2 if n_fft is not None else 0
    result = np.asanyarray(frames) * hop + offset
    converted = result.astype(np.int64)
    if np.isscalar(frames):
        return int(converted)
    return cast(np.ndarray, converted)


def rms(audio: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Return root-mean-square amplitude along an axis."""
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    if source.shape[normalized_axis] == 0:
        shape = list(source.shape)
        del shape[normalized_axis]
        return np.zeros(tuple(shape), dtype=np.float64)
    return cast(
        np.ndarray,
        np.sqrt(np.mean(np.square(source, dtype=np.float64), axis=normalized_axis)),
    )


def _frame_lengths(
    *,
    frame_length: int | None,
    hop_length: int | None,
    sample_rate: int | None,
    frame_ms: float,
    hop_ms: float,
) -> tuple[int, int]:
    if frame_length is None:
        if sample_rate is None:
            raise InvalidParameterError("frame_length or sample_rate must be provided")
        rate = validate_positive(sample_rate, "sample_rate")
        frame_duration = validate_positive(frame_ms, "frame_ms")
        length = int(rate * frame_duration / 1000.0)
        if length <= 0:
            raise InvalidParameterError("frame_length must be positive")
    else:
        length = validate_integer(frame_length, "frame_length")

    if hop_length is None:
        if sample_rate is None:
            hop = length
        else:
            rate = validate_positive(sample_rate, "sample_rate")
            hop_duration = validate_positive(hop_ms, "hop_ms")
            hop = int(rate * hop_duration / 1000.0)
            if hop <= 0:
                raise InvalidParameterError("hop_length must be positive")
    else:
        hop = validate_integer(hop_length, "hop_length")
    return length, hop


def frame_signal(
    audio: np.ndarray,
    *,
    frame_length: int | None = None,
    hop_length: int | None = None,
    sample_rate: int | None = None,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    axis: int = -1,
    center: bool = False,
    pad_mode: str = "constant",
    pad_end: bool = False,
) -> np.ndarray:
    """Split audio into ``(..., frames, frame_length)`` arrays.

    Frame sizes may be supplied directly in samples or derived from
    ``sample_rate`` with ``frame_ms`` and ``hop_ms``. Empty inputs produce zero
    frames, and short non-empty inputs are zero-padded to one frame.
    """
    validate_choices(
        pad_mode,
        (
            "constant",
            "edge",
            "linear_ramp",
            "maximum",
            "mean",
            "median",
            "minimum",
            "reflect",
            "symmetric",
            "wrap",
        ),
        "pad_mode",
    )
    length, hop = _frame_lengths(
        frame_length=frame_length,
        hop_length=hop_length,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    moved = np.moveaxis(source, normalized_axis, -1)
    if moved.shape[-1] == 0:
        return np.empty((*moved.shape[:-1], 0, length), dtype=source.dtype)
    if center:
        padding = length // 2
        moved = np.pad(
            moved,
            [(0, 0)] * (moved.ndim - 1) + [(padding, padding)],
            mode=cast(Any, pad_mode),
        )
    if moved.shape[-1] < length:
        moved = np.pad(
            moved,
            [(0, 0)] * (moved.ndim - 1) + [(0, length - moved.shape[-1])],
        )
    elif pad_end:
        remainder = (moved.shape[-1] - length) % hop
        if remainder:
            moved = np.pad(
                moved,
                [(0, 0)] * (moved.ndim - 1) + [(0, hop - remainder)],
            )
    return frame(moved, frame_length=length, hop_length=hop)


def frame_rms(
    audio: np.ndarray,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    axis: int = -1,
    center: bool = True,
    pad_mode: str = "constant",
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Compute RMS amplitude for each analysis frame."""
    frames = frame_signal(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
        axis=axis,
        center=center,
        pad_mode=pad_mode,
    )
    squared = abs2(frames, dtype=dtype)
    energy: np.ndarray = np.mean(squared, axis=-1, dtype=dtype)
    return cast(np.ndarray, np.sqrt(energy).astype(dtype, copy=False))


def short_time_energy(
    audio: np.ndarray,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    axis: int = -1,
    center: bool = False,
) -> np.ndarray:
    """Calculate mean-square energy for each frame."""
    frames = frame_signal(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
        axis=axis,
        center=center,
    )
    return cast(np.ndarray, np.mean(np.square(frames, dtype=np.float64), axis=-1))


def zero_crossing_rate(
    audio: np.ndarray,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    axis: int = -1,
    center: bool = False,
    normalize: bool = False,
) -> np.ndarray:
    """Calculate zero-crossing rates for each frame."""
    frames = frame_signal(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
        axis=axis,
        center=center,
    )
    if frames.shape[-1] < 2:
        result = np.zeros(frames.shape[:-1], dtype=np.float64)
    else:
        signs = np.signbit(frames)
        result = np.mean(signs[..., 1:] != signs[..., :-1], axis=-1)
    return _minmax_normalize(result) if normalize else cast(np.ndarray, result)


def spectral_flux(
    audio: np.ndarray,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    axis: int = -1,
    center: bool = False,
    window: str = "hann",
    normalize: bool = False,
) -> np.ndarray:
    """Calculate positive spectral changes between adjacent frames."""
    frames = frame_signal(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
        axis=axis,
        center=center,
    )
    window_name = validate_choices(window, ("hann", "hamming"), "window")
    taper = np.hanning(frames.shape[-1]) if window_name == "hann" else np.hamming(frames.shape[-1])
    magnitudes = np.abs(np.fft.rfft(frames * taper, axis=-1))
    difference = np.diff(magnitudes, axis=-2, prepend=magnitudes[..., :1, :])
    result = np.sum(np.maximum(difference, 0.0), axis=-1)
    return _minmax_normalize(result) if normalize else cast(np.ndarray, result)


def median_filter_numpy(
    values: np.ndarray,
    size: int = 3,
    *,
    window_size: int | None = None,
    mode: str = "edge",
) -> np.ndarray:
    """Apply an odd-width median filter along the final axis.

    ``mode='edge'`` pads with endpoint values. ``mode='truncate'`` uses smaller
    windows at the boundaries, matching PyKokoro's former local helper.
    """
    width = validate_integer(window_size if window_size is not None else size, "size")
    if width % 2 == 0:
        raise InvalidParameterError("median filter size must be odd")
    filter_mode = validate_choices(mode, ("edge", "truncate"), "mode")
    array = np.asarray(values)
    if array.ndim == 0:
        raise InvalidParameterError("values must have at least one dimension")
    if array.shape[-1] == 0:
        return np.array(array, copy=True)
    radius = width // 2
    if filter_mode == "edge":
        padded = np.pad(
            array,
            [(0, 0)] * (array.ndim - 1) + [(radius, radius)],
            mode="edge",
        )
        windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=-1)
        return cast(np.ndarray, np.median(windows, axis=-1))
    result = np.empty(array.shape, dtype=np.result_type(array.dtype, np.float64))
    for index in range(array.shape[-1]):
        start = max(0, index - radius)
        stop = min(array.shape[-1], index + radius + 1)
        result[..., index] = np.median(array[..., start:stop], axis=-1)
    return result


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1] == 0:
        return array
    minimum = np.min(array, axis=-1, keepdims=True)
    maximum = np.max(array, axis=-1, keepdims=True)
    span = maximum - minimum
    return cast(
        np.ndarray,
        np.divide(array - minimum, span, out=np.zeros_like(array), where=span > 0),
    )


def non_silent_frames(
    audio: np.ndarray,
    *,
    top_db: float = 60.0,
    ref: Reference = np.max,
    frame_length: int = 2048,
    hop_length: int = 512,
    aggregate: Aggregate = np.max,
    axis: int = -1,
    center: bool = True,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Return a one-dimensional mask of frames above the silence threshold."""
    threshold = float(top_db)
    if not np.isfinite(threshold) or threshold < 0:
        raise InvalidParameterError("top_db must be finite and non-negative")
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    if source.shape[normalized_axis] == 0:
        return np.zeros(0, dtype=bool)
    values = frame_rms(
        source,
        frame_length=frame_length,
        hop_length=hop_length,
        axis=normalized_axis,
        center=center,
        pad_mode=pad_mode,
    )
    db = amplitude_to_db(values, ref=ref, top_db=None)
    if db.ndim > 1:
        axes = tuple(range(db.ndim - 1))
        db = np.asarray(aggregate(db, axis=axes))
    # Keep the documented strict threshold while absorbing the small dB
    # rounding error introduced when float32 RMS values are logged.
    rms_dtype = np.asarray(values).dtype
    tolerance = float(np.finfo(rms_dtype).eps * 4) if np.issubdtype(rms_dtype, np.floating) else 0.0
    return cast(np.ndarray, db > -threshold + tolerance)


def activity_to_intervals(
    activity: np.ndarray,
    *,
    hop_length: int,
    sample_count: int,
) -> np.ndarray:
    """Convert a one-dimensional frame mask to clipped sample intervals.

    Returned intervals use half-open ``[start, end)`` boundaries and are
    always sorted, non-overlapping, and represented as ``int64`` values.
    """
    mask = np.asarray(activity, dtype=bool)
    if mask.ndim != 1:
        raise InvalidParameterError("activity must be one-dimensional")
    hop = validate_integer(hop_length, "hop_length")
    count = validate_integer(sample_count, "sample_count", minimum=0)
    if mask.size == 0 or not np.any(mask) or count == 0:
        return np.empty((0, 2), dtype=np.int64)
    padded = np.pad(mask.astype(np.int8), (1, 1))
    frame_edges = np.flatnonzero(np.diff(padded))
    sample_edges = np.minimum(frame_edges.astype(np.int64) * hop, count)
    return cast(np.ndarray, sample_edges.astype(np.int64, copy=False).reshape(-1, 2))


def split(
    audio: np.ndarray,
    *,
    top_db: float = 60.0,
    ref: Reference = np.max,
    frame_length: int = 2048,
    hop_length: int = 512,
    aggregate: Aggregate = np.max,
    axis: int = -1,
    center: bool = True,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Return non-silent sample intervals shaped ``(n_intervals, 2)``.

    Centered analysis uses frame boundaries directly, matching the trim
    boundary convention. With ``center=False``, the final edge is extended by
    the unrepresented portion of the active analysis frame.
    """
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    sample_count = source.shape[normalized_axis]
    if sample_count == 0:
        return np.empty((0, 2), dtype=np.int64)
    activity = non_silent_frames(
        source,
        top_db=top_db,
        ref=ref,
        frame_length=frame_length,
        hop_length=hop_length,
        aggregate=aggregate,
        axis=normalized_axis,
        center=center,
        pad_mode=pad_mode,
    )
    intervals = activity_to_intervals(
        activity,
        hop_length=hop_length,
        sample_count=sample_count,
    )
    if not center and intervals.size:
        extension = validate_integer(frame_length, "frame_length") - validate_integer(
            hop_length, "hop_length"
        )
        intervals[:, 1] = np.minimum(sample_count, intervals[:, 1] + extension)
    return intervals


def normalized_energy_vad(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_duration_ms: float = 5.0,
    energy_threshold: float = 0.02,
    axis: int = -1,
    pad_end: bool = False,
) -> np.ndarray:
    """Return normalized-RMS VAD activity for non-overlapping frames.

    ``pad_end=False`` preserves the compatibility behavior of discarding a
    final partial frame. Set it to ``True`` when tail speech must be analyzed.
    """
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    rate = validate_positive(sample_rate, "sample_rate")
    duration = validate_positive(frame_duration_ms, "frame_duration_ms")
    length = int(rate * duration / 1000.0)
    if length <= 0:
        raise InvalidParameterError("frame_length must be positive")
    threshold = float(energy_threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise InvalidParameterError("energy_threshold must be finite and in [0, 1]")
    frames = frame_signal(
        source,
        frame_length=length,
        hop_length=length,
        axis=normalized_axis,
        pad_end=pad_end,
    )
    frame_energy = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=-1))
    return cast(np.ndarray, _minmax_normalize(frame_energy) > threshold)


def relative_db_vad(
    audio: np.ndarray,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    threshold_db: float = 40.0,
    top_db: float | None = None,
    axis: int = -1,
    pad_end: bool = False,
) -> np.ndarray:
    """Return VAD activity for frames within ``threshold_db`` of the peak."""
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    threshold = float(threshold_db if top_db is None else top_db)
    if not np.isfinite(threshold) or threshold < 0:
        raise InvalidParameterError("threshold_db must be finite and non-negative")
    if pad_end:
        frames = frame_signal(
            source,
            frame_length=frame_length,
            hop_length=hop_length,
            axis=normalized_axis,
            pad_end=True,
        )
        energy = np.mean(np.square(frames, dtype=np.float64), axis=-1)
    else:
        energy = short_time_energy(
            source,
            frame_length=frame_length,
            hop_length=hop_length,
            axis=normalized_axis,
        )
    if energy.shape[-1] == 0:
        return np.zeros(energy.shape, dtype=bool)
    peak = np.max(energy, axis=-1, keepdims=True)
    db = 10.0 * np.log10(np.maximum(energy, 1e-20))
    peak_db = 10.0 * np.log10(np.maximum(peak, 1e-20))
    return cast(np.ndarray, (peak > 0.0) & (db >= peak_db - threshold))


def energy_based_vad(
    audio: np.ndarray,
    sample_rate: int | None = None,
    *,
    frame_duration_ms: float = 5.0,
    energy_threshold: float | None = None,
    frame_length: int = 2048,
    hop_length: int = 512,
    threshold_db: float = 40.0,
    top_db: float | None = None,
    axis: int = -1,
    pad_end: bool = False,
) -> np.ndarray:
    """Return a per-frame boolean voice-activity mask.

    Two threshold models are supported:

    - Passing ``sample_rate`` selects normalized-RMS compatibility mode. Frames
      are non-overlapping, short signals are padded, constant-energy signals are
      inactive, and ``energy_threshold`` is in ``[0, 1]``.
    - Without ``sample_rate``, frame power is compared with the loudest frame in
      decibels using ``threshold_db`` (or the ``top_db`` alias).
    """
    if sample_rate is not None:
        threshold = 0.02 if energy_threshold is None else float(energy_threshold)
        return normalized_energy_vad(
            audio,
            sample_rate,
            frame_duration_ms=frame_duration_ms,
            energy_threshold=threshold,
            axis=axis,
            pad_end=pad_end,
        )
    if energy_threshold is not None:
        raise InvalidParameterError("sample_rate is required with energy_threshold")
    return relative_db_vad(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
        threshold_db=threshold_db,
        top_db=top_db,
        axis=axis,
        pad_end=pad_end,
    )


def find_speech_start(
    audio: np.ndarray,
    sample_rate: int | None = None,
    *,
    frame_duration_ms: float = 5.0,
    energy_threshold: float | None = None,
    frame_length: int = 2048,
    hop_length: int = 512,
    threshold_db: float = 40.0,
    axis: int = -1,
    pad_end: bool = False,
) -> int:
    """Return the first active sample index, or zero when no activity exists."""
    energy_based_vad(
        audio,
        sample_rate,
        frame_duration_ms=frame_duration_ms,
        energy_threshold=energy_threshold,
        frame_length=frame_length,
        hop_length=hop_length,
        threshold_db=threshold_db,
        axis=axis,
        pad_end=pad_end,
    )
    bounds = find_speech_bounds(
        audio,
        sample_rate,
        frame_duration_ms=frame_duration_ms,
        energy_threshold=energy_threshold,
        frame_length=frame_length,
        hop_length=hop_length,
        threshold_db=threshold_db,
        axis=axis,
        pad_end=pad_end,
    )
    return int(bounds[0])


def find_speech_bounds(
    audio: np.ndarray,
    sample_rate: int | None = None,
    *,
    frame_duration_ms: float = 5.0,
    energy_threshold: float | None = None,
    frame_length: int = 2048,
    hop_length: int = 512,
    threshold_db: float = 40.0,
    top_db: float | None = None,
    axis: int = -1,
    pad_end: bool = False,
) -> np.ndarray:
    """Return ``[start, end]`` speech bounds, or ``[0, 0]`` if absent."""
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    mask = energy_based_vad(
        source,
        sample_rate,
        frame_duration_ms=frame_duration_ms,
        energy_threshold=energy_threshold,
        frame_length=frame_length,
        hop_length=hop_length,
        threshold_db=threshold_db,
        top_db=top_db,
        axis=normalized_axis,
        pad_end=pad_end,
    )
    active = np.any(mask, axis=tuple(range(mask.ndim - 1))) if mask.ndim > 1 else mask
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return np.array([0, 0], dtype=np.int64)
    if sample_rate is not None:
        step = int(validate_positive(sample_rate, "sample_rate") * frame_duration_ms / 1000.0)
    else:
        step = validate_integer(hop_length, "hop_length")
    sample_count = source.shape[normalized_axis]
    start = min(sample_count, int(indices[0] * step))
    end = min(sample_count, int((indices[-1] + 1) * step))
    return np.array([start, end], dtype=np.int64)


def trim(
    audio: np.ndarray,
    *,
    top_db: float = 60.0,
    ref: Reference = np.max,
    frame_length: int = 2048,
    hop_length: int = 512,
    aggregate: Aggregate = np.max,
    axis: int = -1,
    center: bool = True,
    pad_mode: str = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """Trim leading and trailing frames below a relative RMS threshold.

    Analysis is centered by default for Librosa/PyKokoro-compatible boundaries.
    Empty input returns an empty copy and ``[0, 0]``. A uniform all-zero signal
    with the default peak reference remains unchanged; use a fixed reference
    such as ``ref=1.0`` to classify it as silence.
    """
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    threshold = float(top_db)
    if not np.isfinite(threshold) or threshold < 0:
        raise InvalidParameterError("top_db must be finite and non-negative")
    length = validate_integer(frame_length, "frame_length")
    hop = validate_integer(hop_length, "hop_length")
    sample_count = source.shape[normalized_axis]
    if sample_count == 0:
        return np.array(source, copy=True), np.array([0, 0], dtype=np.int64)
    activity = non_silent_frames(
        source,
        top_db=top_db,
        ref=ref,
        frame_length=length,
        hop_length=hop,
        aggregate=aggregate,
        axis=normalized_axis,
        center=center,
        pad_mode=pad_mode,
    )
    active_indices = np.flatnonzero(activity)
    if active_indices.size == 0:
        start, end = 0, 0
    else:
        start = int(active_indices[0] * hop)
        if center:
            end = min(sample_count, int((active_indices[-1] + 1) * hop))
        else:
            end = min(sample_count, int(active_indices[-1] * hop + length))
    moved = np.moveaxis(source, normalized_axis, -1)
    trimmed = moved[..., start:end]
    trimmed = np.moveaxis(trimmed, -1, normalized_axis)
    return np.array(trimmed, dtype=source.dtype, copy=True), np.array([start, end], dtype=np.int64)
