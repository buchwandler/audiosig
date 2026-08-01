"""Waveform construction helpers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from ._validation import FLOAT_DTYPES, validate_finite, validate_integer
from .exceptions import InvalidParameterError


def _resolve_float_dtype(dtype: np.dtype[Any] | type[np.floating[Any]]) -> np.dtype[Any]:
    """Resolve and validate a public floating-point audio dtype."""
    try:
        resolved = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError("dtype must be float32 or float64") from exc
    if resolved not in FLOAT_DTYPES:
        raise InvalidParameterError("dtype must be float32 or float64")
    return resolved


def generate_silence(
    duration: float,
    sample_rate: int,
    *,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Return a newly allocated mono silence buffer for ``duration`` seconds.

    The sample count is ``int(duration * sample_rate)``. Only float32 and
    float64 output are supported, matching AudioSig's public audio contract.
    """
    if isinstance(duration, (bool, np.bool_, str, bytes)) or not np.isscalar(duration):
        raise InvalidParameterError("duration must be a finite non-negative number")
    try:
        duration_value = validate_finite(float(cast(Any, duration)), "duration")
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidParameterError("duration must be a finite non-negative number") from exc
    if duration_value < 0:
        raise InvalidParameterError("duration must be non-negative")

    rate = validate_integer(sample_rate, "sample_rate", minimum=1)
    resolved_dtype = _resolve_float_dtype(dtype)

    requested_length = duration_value * rate
    if not np.isfinite(requested_length):
        raise InvalidParameterError("duration * sample_rate must be finite")
    sample_count = int(requested_length)
    if sample_count > np.iinfo(np.intp).max:
        raise InvalidParameterError("requested silence length is too large")
    return np.zeros(sample_count, dtype=resolved_dtype)
