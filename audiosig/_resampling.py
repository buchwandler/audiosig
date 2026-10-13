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
from .exceptions import AudioShapeError, InvalidParameterError


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
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    source_hz = validate_positive(source_rate, "source_rate")
    target_hz = validate_positive(target_rate, "target_rate")
    width, cutoff = validate_filter(filter_width, rolloff)
    mode = validate_choices(length_mode, ("round", "ceil"), "length_mode")
    input_length = source.shape[normalized_axis]
    if input_length == 0:
        return np.array(source, dtype=source.dtype, copy=True)
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
    """Resample audio to exactly ``length`` samples along ``axis``."""
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    target_length = validate_integer(length, "length", minimum=0)
    validate_filter(filter_width, rolloff)
    input_length = source.shape[normalized_axis]
    if target_length == input_length:
        return np.array(source, dtype=source.dtype, copy=True)
    if target_length == 0:
        shape = list(source.shape)
        shape[normalized_axis] = 0
        return np.empty(shape, dtype=source.dtype)
    if input_length == 0:
        raise AudioShapeError("empty audio cannot be resampled to a non-zero length")
    return resample(
        source,
        source_rate=float(input_length),
        target_rate=float(target_length),
        axis=normalized_axis,
        filter_width=filter_width,
        rolloff=rolloff,
    )


def resample_speed(
    audio: np.ndarray,
    speed: float,
    *,
    axis: int = -1,
    filter_width: int = 32,
    rolloff: float = 0.945,
) -> np.ndarray:
    """Change playback speed by resampling, including its pitch.

    ``speed > 1`` makes audio faster, shorter, and higher pitched. ``speed < 1``
    makes audio slower, longer, and lower pitched. Use :func:`time_stretch`
    when pitch should remain approximately unchanged.
    """
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    multiplier = validate_positive(speed, "speed")
    validate_filter(filter_width, rolloff)
    input_length = source.shape[normalized_axis]
    if input_length == 0:
        return np.array(source, dtype=source.dtype, copy=True)
    requested = input_length / multiplier
    if not np.isfinite(requested) or requested > np.iinfo(np.intp).max:
        raise InvalidParameterError("requested playback-speed length is too large")
    target_length = max(1, round(requested))
    return resample_to_length(
        source,
        target_length,
        axis=normalized_axis,
        filter_width=filter_width,
        rolloff=rolloff,
    )


def resample_windowed_sinc(
    audio: np.ndarray,
    *,
    source_rate: float,
    target_rate: float,
    axis: int = -1,
    radius: int = 16,
    alignment: str = "sample",
    window: str = "kaiser",
    rolloff: float = 1.0,
    antialias: bool = True,
    normalize_weights: bool = True,
    boundary: str = "zero",
    length_mode: str = "round",
) -> np.ndarray:
    """Resample using a configurable, chunked windowed-sinc kernel.

    Unlike :func:`resample`, this compatibility-capable primitive exposes the
    sample alignment and boundary policies. ``radius`` determines a 2*radius
    tap support. ``alignment='half_sample'`` places output centers at
    ``(i + 0.5) * source_rate / target_rate - 0.5``.
    """
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    source_hz = validate_positive(source_rate, "source_rate")
    target_hz = validate_positive(target_rate, "target_rate")
    support = validate_integer(radius, "radius")
    alignment_mode = validate_choices(alignment, ("sample", "half_sample"), "alignment")
    window_mode = validate_choices(window, ("kaiser", "raised_cosine"), "window")
    cutoff_rolloff = validate_filter(1, rolloff)[1]
    antialias_value = _validate_resample_bool(antialias, "antialias")
    normalize_value = _validate_resample_bool(normalize_weights, "normalize_weights")
    boundary_mode = validate_choices(boundary, ("zero", "edge"), "boundary")
    length_policy = validate_choices(length_mode, ("round", "ceil"), "length_mode")

    input_length = source.shape[normalized_axis]
    if input_length == 0:
        return np.array(source, dtype=source.dtype, copy=True, order="C")
    ratio = target_hz / source_hz
    requested_length = input_length * ratio
    if not np.isfinite(requested_length) or requested_length > np.iinfo(np.intp).max:
        raise InvalidParameterError("requested resampled length is too large")
    output_length = max(
        1,
        int(np.floor(requested_length + 0.5))
        if length_policy == "round"
        else int(np.ceil(requested_length)),
    )
    if source_hz == target_hz:
        return np.array(source, dtype=source.dtype, copy=True, order="C")

    moved = np.moveaxis(source, normalized_axis, -1)
    flat = moved.reshape((-1, input_length))
    tap_offsets = np.arange(1 - support, support + 1, dtype=np.int64)
    cutoff = cutoff_rolloff * (min(1.0, ratio) if antialias_value else 1.0)
    output = np.empty((flat.shape[0], output_length), dtype=source.dtype)
    chunk_size = 1024
    epsilon = np.finfo(np.float64).eps

    for start in range(0, output_length, chunk_size):
        stop = min(output_length, start + chunk_size)
        indices_out = np.arange(start, stop, dtype=np.float64)
        if alignment_mode == "half_sample":
            centers = (indices_out + 0.5) / ratio - 0.5
        else:
            centers = indices_out / ratio
        bases = np.floor(centers).astype(np.int64)
        indices = bases[:, None] + tap_offsets[None, :]
        in_bounds = (indices >= 0) & (indices < input_length)
        if boundary_mode == "edge":
            sample_indices = np.clip(indices, 0, input_length - 1)
        else:
            sample_indices = np.clip(indices, 0, input_length - 1)
        distances = centers[:, None] - indices.astype(np.float64)
        scaled_distance = distances / float(support)
        if window_mode == "raised_cosine":
            window_weights = 0.5 + 0.5 * np.cos(np.pi * scaled_distance)
            window_weights = np.where(np.abs(scaled_distance) <= 1.0, window_weights, 0.0)
        else:
            inside = np.maximum(0.0, 1.0 - np.square(scaled_distance))
            window_weights = np.i0(14.0 * np.sqrt(inside)) / np.i0(14.0)
            window_weights = np.where(np.abs(scaled_distance) <= 1.0, window_weights, 0.0)
        weights = np.sinc(distances * cutoff) * cutoff * window_weights
        if boundary_mode == "zero":
            weights *= in_bounds
        if normalize_value:
            weight_sums = np.sum(weights, axis=1, keepdims=True)
            weights = np.divide(
                weights,
                weight_sums,
                out=np.zeros_like(weights),
                where=np.abs(weight_sums) > epsilon,
            )
        samples = flat[:, sample_indices]
        output[:, start:stop] = np.sum(samples * weights[None, :, :], axis=-1, dtype=np.float64)

    reshaped = output.reshape((*moved.shape[:-1], output_length))
    result = np.moveaxis(reshaped, -1, normalized_axis)
    return np.ascontiguousarray(result, dtype=source.dtype)


def _validate_resample_bool(value: bool, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise InvalidParameterError(f"{name} must be a boolean")
    return bool(value)
