"""Waveform boundary selection primitives."""

from __future__ import annotations

import numpy as np

from ._validation import validate_audio, validate_integer
from .exceptions import InvalidParameterError

_LOCAL_RMS_WEIGHT = 0.50
_ENDPOINT_AMPLITUDE_WEIGHT = 0.25
_BOUNDARY_SLOPE_WEIGHT = 0.10
_ANCHOR_DISTANCE_WEIGHT = 0.15


def _normalize_cost(values: np.ndarray) -> np.ndarray:
    """Normalize candidate costs to ``[0, 1]`` without dividing by zero."""
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum == minimum:
        return np.zeros(values.shape, dtype=np.float64)
    return (values - minimum) / (maximum - minimum)


def _aggregate_lanes(values: np.ndarray) -> np.ndarray:
    """Take the conservative maximum over all non-candidate dimensions."""
    if values.ndim == 1:
        return values
    return np.max(values, axis=tuple(range(values.ndim - 1)))


def find_smooth_cut_point(
    audio: np.ndarray,
    *,
    start: int,
    end: int,
    anchor: int | None = None,
    window_length: int = 120,
    axis: int = -1,
) -> int | None:
    """Find a low-disruption waveform boundary in ``[start, end)``.

    Candidates are scored by local RMS, adjacent endpoint amplitude,
    cross-boundary slope, and distance from ``anchor``. The selector always
    returns a legal candidate when the search interval is non-empty. It is not
    a silence detector, so voiced or noisy signals still produce a boundary.

    ``audio`` must be a finite float32 or float64 NumPy array. The sample axis
    may be selected with ``axis``; all other dimensions are treated as lanes
    and aggregated conservatively. ``anchor`` defaults to the middle legal
    sample and may lie outside the search interval. For an empty audio array,
    the explicit empty contract is ``start=0, end=0``, which returns ``None``.
    """
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    sample_count = source.shape[normalized_axis]
    start_index = validate_integer(start, "start", minimum=0)
    end_index = validate_integer(end, "end", minimum=0)
    window = validate_integer(window_length, "window_length", minimum=1)

    if sample_count == 0:
        if start_index == 0 and end_index == 0:
            return None
        raise InvalidParameterError("empty audio only supports start=0 and end=0")
    if start_index >= end_index:
        raise InvalidParameterError("start must be less than end")
    if end_index > sample_count:
        raise InvalidParameterError("end must not exceed the sample count")

    if anchor is None:
        preferred = (start_index + end_index - 1) // 2
    else:
        if isinstance(anchor, (bool, np.bool_)) or not isinstance(anchor, (int, np.integer)):
            raise InvalidParameterError("anchor must be an integer")
        preferred = int(anchor)
    moved = np.moveaxis(source, normalized_axis, -1)
    candidates = np.arange(start_index, end_index, dtype=np.int64)
    half_window = window // 2

    source_start = max(0, start_index - half_window - 1)
    source_end = min(sample_count, end_index + half_window + 1)
    local_source = moved[..., source_start:source_end]
    squared = np.square(local_source, dtype=np.float64)
    prefix = np.concatenate(
        (np.zeros((*squared.shape[:-1], 1), dtype=np.float64), np.cumsum(squared, axis=-1)),
        axis=-1,
    )

    left = np.maximum(0, candidates - half_window)
    right = np.minimum(sample_count, left + window)
    prefix_left = prefix[..., left - source_start]
    prefix_right = prefix[..., right - source_start]
    window_sum = prefix_right - prefix_left
    local_rms = _aggregate_lanes(np.sqrt(window_sum / (right - left)))

    right_samples = moved[..., candidates]
    left_candidates = np.maximum(candidates - 1, 0)
    left_samples = moved[..., left_candidates]
    at_start = candidates == 0
    endpoint_amplitude = np.maximum(np.abs(left_samples), np.abs(right_samples))
    boundary_slope = np.abs(right_samples - left_samples)
    endpoint_amplitude = np.where(at_start, np.abs(right_samples), endpoint_amplitude)
    boundary_slope = np.where(at_start, 0.0, boundary_slope)
    endpoint_cost = _aggregate_lanes(endpoint_amplitude)
    slope_cost = _aggregate_lanes(boundary_slope)

    distance = np.abs(candidates.astype(np.float64) - float(preferred))
    total = (
        _LOCAL_RMS_WEIGHT * _normalize_cost(local_rms)
        + _ENDPOINT_AMPLITUDE_WEIGHT * _normalize_cost(endpoint_cost)
        + _BOUNDARY_SLOPE_WEIGHT * _normalize_cost(slope_cost)
        + _ANCHOR_DISTANCE_WEIGHT * _normalize_cost(distance)
    )

    order = np.lexsort((candidates, distance, total))
    return int(candidates[order[0]])
