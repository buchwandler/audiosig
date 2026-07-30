"""Shared validation helpers for public AudioSig operations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .exceptions import AudioShapeError, InvalidParameterError

FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def validate_audio(
    audio: np.ndarray,
    *,
    axis: int = -1,
    allow_empty: bool = False,
) -> tuple[np.ndarray, int]:
    """Validate an audio array and return it with a normalized axis."""
    if not isinstance(audio, np.ndarray):
        raise AudioShapeError("audio must be a NumPy array")
    if audio.ndim == 0:
        raise AudioShapeError("audio must have at least one dimension")
    if audio.dtype not in FLOAT_DTYPES:
        raise AudioShapeError("audio must have dtype float32 or float64")
    normalized_axis = validate_axis(axis, audio.ndim)
    if audio.shape[normalized_axis] == 0 and not allow_empty:
        raise AudioShapeError("the sample axis cannot be empty")
    if not np.all(np.isfinite(audio)):
        raise AudioShapeError("audio must contain only finite values")
    return audio, normalized_axis


def validate_positive(value: float, name: str) -> float:
    """Return a finite positive parameter."""
    if isinstance(value, (bool, np.bool_)):
        raise InvalidParameterError(f"{name} must be finite and positive")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise InvalidParameterError(f"{name} must be finite and positive")
    return result


def validate_integer(value: int, name: str, *, minimum: int = 1) -> int:
    """Validate an integer-valued parameter without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise InvalidParameterError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise InvalidParameterError(f"{name} must be at least {minimum}")
    return result


def validate_axis(axis: int, ndim: int) -> int:
    """Normalize an axis or raise an AudioShapeError."""
    if not isinstance(axis, (int, np.integer)):
        raise AudioShapeError("axis must be an integer")
    normalized = int(axis)
    if normalized < 0:
        normalized += ndim
    if not 0 <= normalized < ndim:
        raise AudioShapeError(f"axis {axis} is invalid for {ndim}-D audio")
    return normalized


def validate_filter(filter_width: int, rolloff: float) -> tuple[int, float]:
    """Validate resampling kernel settings."""
    width = validate_integer(filter_width, "filter_width", minimum=1)
    cutoff = float(rolloff)
    if not np.isfinite(cutoff) or not 0 < cutoff <= 1:
        raise InvalidParameterError("rolloff must be finite and in the interval (0, 1]")
    return width, cutoff


def validate_choices(value: str, choices: Iterable[str], name: str) -> str:
    """Validate a string enum."""
    if value not in choices:
        raise InvalidParameterError(f"{name} must be one of {tuple(choices)!r}")
    return value
