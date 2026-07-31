"""Chunked windowed-sinc resampling implemented with NumPy only."""

from __future__ import annotations

import numpy as np

from ._validation import (
    validate_audio,
    validate_choices,
    validate_filter,
    validate_integer,
    validate_positive,
)
from .exceptions import InvalidParameterError


def resample(
    audio: np.ndarray,
    *,
    source_rate: float,
    target_rate: float,
    axis: int = -1,
    filter_width: int = 32,
    rolloff: float = 0.945,
    length_mode: str = "round",
) -> np.ndarray:
    """Resample audio with a finite, anti-aliased windowed-sinc kernel.

    ``length_mode='round'`` preserves AudioSig's historical sizing; ``ceil``
    provides librosa-compatible output length semantics.
    """
    source, normalized_axis = validate_audio(audio, axis=axis)
    source_hz = validate_positive(source_rate, "source_rate")
    target_hz = validate_positive(target_rate, "target_rate")
    width, cutoff = validate_filter(filter_width, rolloff)
    mode = validate_choices(length_mode, ("round", "ceil"), "length_mode")
    input_length = source.shape[normalized_axis]
    requested_length = input_length * target_hz / source_hz
    if not np.isfinite(requested_length) or requested_length > np.iinfo(np.intp).max:
        raise InvalidParameterError("requested resampled length is too large")
    output_length = max(
        1, round(requested_length) if mode == "round" else int(np.ceil(requested_length))
    )
    if source_hz == target_hz and output_length == input_length:
        return np.array(source, dtype=source.dtype, copy=True)
    moved = np.moveaxis(source, normalized_axis, -1)
    flat = moved.reshape((-1, input_length))
    ratio = target_hz / source_hz
    effective_cutoff = cutoff * min(1.0, ratio)
    offsets = np.arange(-width, width + 1, dtype=np.int64)
    window = np.kaiser(offsets.size, beta=14.0)[None, :]
    output = np.empty((moved.size // input_length, output_length), dtype=source.dtype)
    chunk_size = 1024
    for start in range(0, output_length, chunk_size):
        stop = min(output_length, start + chunk_size)
        positions = np.arange(start, stop, dtype=np.float64) / ratio
        centers = np.floor(positions).astype(np.int64)
        indices = centers[:, None] + offsets[None, :]
        clipped = np.clip(indices, 0, input_length - 1)
        distance = indices.astype(np.float64) - positions[:, None]
        weights = np.sinc(distance * effective_cutoff) * effective_cutoff * window
        weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), np.finfo(np.float64).eps)
        samples = flat[:, clipped]
        output[:, start:stop] = np.sum(samples * weights[None, :, :], axis=-1, dtype=np.float64)
    reshaped = output.reshape((*moved.shape[:-1], output_length))
    return np.moveaxis(reshaped, -1, normalized_axis).astype(source.dtype, copy=False)


def resample_to_length(
    audio: np.ndarray,
    length: int,
    *,
    axis: int = -1,
    filter_width: int = 32,
    rolloff: float = 0.945,
) -> np.ndarray:
    """Resample audio to an exact number of samples."""
    source, normalized_axis = validate_audio(audio, axis=axis)
    target_length = validate_integer(length, "length")
    if target_length == source.shape[normalized_axis]:
        return np.array(source, dtype=source.dtype, copy=True)
    return resample(
        source,
        source_rate=float(source.shape[normalized_axis]),
        target_rate=float(target_length),
        axis=normalized_axis,
        filter_width=filter_width,
        rolloff=rolloff,
    )
