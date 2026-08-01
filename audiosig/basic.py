"""Basic NumPy waveform construction and channel-conversion helpers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from ._validation import validate_axis
from .exceptions import AudioShapeError, InvalidParameterError


def _validate_duration(duration: float) -> float:
    """Validate a scalar duration in seconds."""
    if isinstance(duration, (bool, np.bool_, str, bytes)) or not np.isscalar(duration):
        raise InvalidParameterError("duration must be finite and non-negative")
    try:
        value = float(cast(Any, duration))
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidParameterError("duration must be finite and non-negative") from exc
    if not np.isfinite(value) or value < 0:
        raise InvalidParameterError("duration must be finite and non-negative")
    return value


def _validate_sample_rate(sample_rate: int) -> int:
    """Validate a positive integer sample rate."""
    if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(sample_rate, (int, np.integer)):
        raise InvalidParameterError("sample_rate must be a positive integer")
    value = int(sample_rate)
    if value <= 0:
        raise InvalidParameterError("sample_rate must be a positive integer")
    return value


def _validate_float_dtype(dtype: np.dtype[Any] | type[np.floating[Any]]) -> np.dtype[Any]:
    """Resolve a real floating NumPy dtype."""
    try:
        resolved = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError("dtype must be a real floating dtype") from exc
    if not np.issubdtype(resolved, np.floating):
        raise InvalidParameterError("dtype must be a real floating dtype")
    return resolved


def generate_silence(
    duration: float,
    sample_rate: int,
    *,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Return a newly allocated one-dimensional array of digital silence.

    ``duration`` is measured in seconds and the output length is truncated to
    ``int(duration * sample_rate)`` samples. Only real floating NumPy dtypes
    are accepted; the default is ``float32``.
    """
    duration_value = _validate_duration(duration)
    sample_rate_value = _validate_sample_rate(sample_rate)
    resolved_dtype = _validate_float_dtype(dtype)

    try:
        requested_length = duration_value * sample_rate_value
    except OverflowError as exc:
        raise InvalidParameterError("duration * sample_rate must fit in an array") from exc
    max_length = np.iinfo(np.intp).max
    if not np.isfinite(requested_length) or requested_length > max_length:
        raise InvalidParameterError("duration * sample_rate must fit in an array")
    sample_count = int(requested_length)
    return np.zeros(sample_count, dtype=resolved_dtype)


def _validate_audio_for_downmix(audio: np.ndarray) -> None:
    """Validate the shape, dtype, and values of a downmix input."""
    if not isinstance(audio, np.ndarray):
        raise AudioShapeError("audio must be a NumPy array")
    if audio.ndim not in (1, 2):
        raise AudioShapeError("audio must be one- or two-dimensional")
    if not np.issubdtype(audio.dtype, np.floating):
        raise AudioShapeError("audio must have a real floating dtype")
    if not np.all(np.isfinite(audio)):
        raise AudioShapeError("audio must contain only finite values")


def downmix_to_mono(
    audio: np.ndarray,
    *,
    channel_axis: int = -1,
) -> np.ndarray:
    """Convert a finite floating waveform to one channel by arithmetic mean.

    One-dimensional input is treated as mono and copied. For two-dimensional
    input, ``channel_axis`` identifies the channel dimension and is removed by
    a dtype-preserving mean. Results are contiguous and never share storage
    with ``audio``.
    """
    _validate_audio_for_downmix(audio)
    axis = validate_axis(channel_axis, audio.ndim)
    if audio.ndim == 1:
        return np.array(audio, dtype=audio.dtype, copy=True, order="C")

    if audio.shape[axis] == 0:
        raise AudioShapeError("the channel axis cannot be empty")
    frame_axis = 1 - axis
    if audio.shape[frame_axis] == 0:
        return np.empty(0, dtype=audio.dtype)
    mixed = np.mean(audio, axis=axis, dtype=audio.dtype)
    return np.ascontiguousarray(mixed, dtype=audio.dtype)
