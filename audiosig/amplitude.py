"""Simple amplitude-domain transformations."""

from __future__ import annotations

import numpy as np

from ._validation import validate_audio, validate_boolean, validate_gain_db
from .exceptions import InvalidParameterError


def apply_gain_db(audio: np.ndarray, db: float, *, clip: bool = False) -> np.ndarray:
    """Apply a decibel gain without modifying ``audio``."""
    source, _ = validate_audio(audio, allow_empty=True)
    gain_db = validate_gain_db(db)
    clip_value = validate_boolean(clip, "clip")
    if gain_db == -np.inf:
        result = np.zeros_like(source)
    else:
        multiplier = 10.0 ** (gain_db / 20.0)
        result = np.multiply(source, multiplier, dtype=source.dtype)
    if clip_value:
        result = np.clip(result, -1.0, 1.0).astype(source.dtype, copy=False)
    return np.array(result, dtype=source.dtype, copy=True)


def peak_normalize(audio: np.ndarray, *, peak: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """Scale audio so its absolute peak equals ``peak`` when audible."""
    source, _ = validate_audio(audio, allow_empty=True)
    target = float(peak)
    tolerance = float(eps)
    if not np.isfinite(target) or target <= 0:
        raise InvalidParameterError("peak must be finite and positive")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise InvalidParameterError("eps must be finite and non-negative")
    if source.size == 0:
        return np.array(source, dtype=source.dtype, copy=True)
    maximum = float(np.max(np.abs(source)))
    if maximum <= tolerance:
        return np.array(source, dtype=source.dtype, copy=True)
    return np.array(source * (target / maximum), dtype=source.dtype, copy=True)
