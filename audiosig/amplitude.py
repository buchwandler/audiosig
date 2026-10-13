"""Simple amplitude-domain transformations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._validation import (
    validate_audio,
    validate_boolean,
    validate_finite,
    validate_gain_db,
    validate_integer,
)
from .exceptions import AudioShapeError, InvalidParameterError


def apply_gain_db(audio: np.ndarray, db: float, *, clip: bool = False) -> np.ndarray:
    """Apply a decibel gain without modifying ``audio``."""
    source, _ = validate_audio(audio, allow_empty=True)
    gain_db = validate_gain_db(db)
    clip_value = validate_boolean(clip, "clip")
    if gain_db == -np.inf:
        result = np.zeros_like(source)
    else:
        exponent = gain_db / 20.0
        max_exponent = np.log10(np.finfo(np.float64).max)
        if exponent > max_exponent:
            raise InvalidParameterError("db produces an unrepresentable gain")
        multiplier = 10.0**exponent
        if source.size:
            maximum = float(np.max(np.abs(source)))
            dtype_limit = float(np.finfo(source.dtype).max)
            if maximum and multiplier > dtype_limit / maximum:
                raise InvalidParameterError("db produces a non-finite output")
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
    dtype_limit = float(np.finfo(source.dtype).max)
    if target > dtype_limit:
        raise InvalidParameterError(f"peak exceeds the finite range of {source.dtype}")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise InvalidParameterError("eps must be finite and non-negative")
    if source.size == 0:
        return np.array(source, dtype=source.dtype, copy=True)
    maximum = float(np.max(np.abs(source)))
    if maximum <= tolerance:
        return np.array(source, dtype=source.dtype, copy=True)
    result = np.asarray(source * (target / maximum), dtype=source.dtype)
    if not np.isfinite(result).all():
        raise InvalidParameterError("peak produces a non-finite output")
    return np.array(result, dtype=source.dtype, copy=True)


def fade_edges(
    audio: np.ndarray,
    *,
    sample_rate: int,
    fade_in: bool,
    fade_out: bool,
    fade_seconds: float = 0.01,
    axis: int = -1,
) -> np.ndarray:
    """Apply linear fades to selected edges without modifying ``audio``."""
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    rate = validate_integer(sample_rate, "sample_rate")
    fade_in_value = validate_boolean(fade_in, "fade_in")
    fade_out_value = validate_boolean(fade_out, "fade_out")
    duration = validate_finite(fade_seconds, "fade_seconds")
    if duration < 0.0:
        raise InvalidParameterError("fade_seconds must be non-negative")
    requested = duration * rate
    if not np.isfinite(requested) or requested > np.iinfo(np.intp).max:
        raise InvalidParameterError("fade length is too large")
    fade_length = int(requested)
    output = np.array(source, copy=True)
    sample_count = source.shape[normalized_axis]
    if sample_count <= 2 * fade_length or fade_length == 0:
        return output
    moved = np.moveaxis(output, normalized_axis, -1)
    if fade_in_value:
        ramp = np.linspace(0.0, 1.0, fade_length, dtype=np.float64)
        moved[..., :fade_length] = moved[..., :fade_length] * ramp
    if fade_out_value:
        ramp = np.linspace(1.0, 0.0, fade_length, dtype=np.float64)
        moved[..., -fade_length:] = moved[..., -fade_length:] * ramp
    return output


def join_with_edge_fades(
    parts: Sequence[np.ndarray],
    *,
    sample_rate: int,
    fade_seconds: float = 0.01,
    axis: int = -1,
) -> np.ndarray:
    """Concatenate audio parts, fading internal segment edges."""
    if not isinstance(parts, Sequence) or len(parts) == 0:
        raise AudioShapeError("parts must be a non-empty sequence of audio arrays")
    rate = validate_integer(sample_rate, "sample_rate")
    duration = validate_finite(fade_seconds, "fade_seconds")
    if duration < 0.0:
        raise InvalidParameterError("fade_seconds must be non-negative")
    prepared: list[np.ndarray] = []
    reference_shape: tuple[int, ...] | None = None
    normalized_axis: int | None = None
    for index, part in enumerate(parts):
        source, part_axis = validate_audio(part, axis=axis, allow_empty=True)
        moved = np.moveaxis(source, part_axis, -1)
        if reference_shape is None:
            reference_shape = moved.shape[:-1]
            normalized_axis = part_axis
        elif moved.shape[:-1] != reference_shape:
            raise AudioShapeError("all parts must match on non-sample dimensions")
        prepared.append(
            fade_edges(
                moved,
                sample_rate=rate,
                fade_in=index > 0,
                fade_out=index + 1 < len(parts),
                fade_seconds=duration,
                axis=-1,
            )
        )
    assert normalized_axis is not None
    joined = np.concatenate(prepared, axis=-1)
    return np.ascontiguousarray(np.moveaxis(joined, -1, normalized_axis))


def soft_limit(audio: np.ndarray, *, knee: float = 0.9) -> np.ndarray:
    """Apply the smooth tanh limiter above ``knee`` without hard clipping."""
    source, _ = validate_audio(audio, allow_empty=True)
    knee_value = validate_finite(knee, "knee")
    if not 0.0 < knee_value < 1.0:
        raise InvalidParameterError("knee must be in the interval (0, 1)")
    magnitude = np.abs(source).astype(np.float64, copy=False)
    over = knee_value + (1.0 - knee_value) * np.tanh((magnitude - knee_value) / (1.0 - knee_value))
    limited = np.where(magnitude > knee_value, np.sign(source) * over, source)
    return np.array(limited, dtype=source.dtype, copy=True)


def gain_for_level(source_db: float, target_db: float) -> float:
    """Return the linear multiplier that moves ``source_db`` to ``target_db``."""
    source = validate_finite(source_db, "source_db")
    target = validate_finite(target_db, "target_db")
    exponent = (target - source) / 20.0
    if exponent > np.log10(np.finfo(np.float64).max):
        raise InvalidParameterError("level difference produces an unrepresentable gain")
    gain = float(np.power(10.0, exponent))
    if not np.isfinite(gain):
        raise InvalidParameterError("level difference produces an unrepresentable gain")
    return gain
