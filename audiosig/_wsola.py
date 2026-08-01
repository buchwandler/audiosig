"""NumPy-only waveform-similarity overlap-add time stretching."""

from __future__ import annotations

import numpy as np

from ._validation import validate_audio, validate_finite, validate_positive
from .exceptions import InvalidParameterError

_MAX_SEARCH_CANDIDATES = 4097
_ENERGY_FLOOR = 1e-12


def _analysis_source(source: np.ndarray, frame_length: int) -> np.ndarray:
    """Return a frame-addressable source, padding only truly short inputs.

    Normal signals are returned unchanged, so every frame addressed by the
    caller is complete and comes from the input.  A signal shorter than one
    frame is the one explicit exception: edge padding gives the single frame
    enough support without manufacturing a zero or constant tail on a normal
    speech signal.
    """
    if source.size >= frame_length:
        return source
    return np.pad(source, (0, frame_length - source.size), mode="edge")


def _complete_slice(source: np.ndarray, start: int, length: int) -> np.ndarray:
    """Return a complete source slice from a frame-addressable signal."""
    stop = start + length
    if start < 0 or stop > source.size:
        raise ValueError("WSOLA source slice is outside the analysis buffer")
    return np.asarray(source[start:stop], dtype=np.float64)


def _choose_candidate(
    source: np.ndarray,
    reference: np.ndarray,
    candidates: np.ndarray,
    expected: int,
) -> int:
    """Choose the best candidate, preferring expected-near then earlier starts."""
    reference_centered = reference - np.mean(reference, dtype=np.float64)
    reference_energy = float(np.dot(reference_centered, reference_centered))
    if reference_energy <= _ENERGY_FLOOR:
        return int(min(candidates, key=lambda value: (abs(int(value) - expected), int(value))))

    offsets = np.arange(reference.size, dtype=np.int64)
    indices = candidates[:, None] + offsets[None, :]
    overlaps = np.asarray(source[indices], dtype=np.float64)
    centered = overlaps - np.mean(overlaps, axis=1, keepdims=True, dtype=np.float64)
    energies = np.asarray(np.sum(centered * centered, axis=1, dtype=np.float64))
    scores = np.full(candidates.size, -np.inf, dtype=np.float64)
    valid = energies > _ENERGY_FLOOR
    scores[valid] = (
        centered[valid] @ reference_centered / np.sqrt(reference_energy * energies[valid])
    )

    # lexsort's last key is primary: highest score, then nearest expected,
    # then the earlier source position for a deterministic final tie-break.
    order = np.lexsort(
        (
            candidates.astype(np.int64),
            np.abs(candidates.astype(np.int64) - expected),
            -scores,
        )
    )
    return int(candidates[order[0]])


def wsola_time_stretch(
    audio: np.ndarray,
    *,
    rate: float,
    sample_rate: int,
    axis: int = -1,
    frame_length_ms: float = 30.0,
    overlap_ms: float = 10.0,
    search_ms: float = 10.0,
) -> np.ndarray:
    """Stretch speech with waveform-similarity overlap-add.

    ``rate > 1`` produces shorter output. Frame, overlap, and search settings
    are expressed in milliseconds so the defaults remain useful across common
    speech sample rates. Processing is independent for each lane in ``audio``.
    """
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    multiplier = validate_positive(rate, "rate")
    sample_hz = validate_positive(sample_rate, "sample_rate")
    frame_ms = validate_finite(frame_length_ms, "frame_length_ms")
    overlap_ms_value = validate_finite(overlap_ms, "overlap_ms")
    search_ms_value = validate_finite(search_ms, "search_ms")
    if frame_ms <= 0:
        raise InvalidParameterError("frame_length_ms must be finite and positive")
    if overlap_ms_value <= 0:
        raise InvalidParameterError("overlap_ms must be finite and positive")
    if search_ms_value < 0:
        raise InvalidParameterError("search_ms must be finite and non-negative")

    frame_length = max(2, round(sample_hz * frame_ms / 1000.0))
    overlap_length = max(1, round(sample_hz * overlap_ms_value / 1000.0))
    search_radius = max(0, round(sample_hz * search_ms_value / 1000.0))
    if overlap_length >= frame_length:
        raise InvalidParameterError("overlap_ms must produce less than one frame")

    input_length = source.shape[normalized_axis]
    if input_length == 0:
        return np.array(source, dtype=source.dtype, copy=True)
    requested_length = input_length / multiplier
    if not np.isfinite(requested_length) or requested_length > np.iinfo(np.intp).max:
        raise InvalidParameterError("requested stretched length is too large")
    target_length = max(1, round(requested_length))
    if multiplier == 1.0:
        return np.array(source, dtype=source.dtype, copy=True)

    synthesis_hop = frame_length - overlap_length
    analysis_hop = multiplier * synthesis_hop
    frame_count = max(1, int(np.ceil(max(0, target_length - frame_length) / synthesis_hop)) + 1)
    if search_radius * 2 + 1 > _MAX_SEARCH_CANDIDATES:
        search_radius = (_MAX_SEARCH_CANDIDATES - 1) // 2

    moved = np.moveaxis(source, normalized_axis, -1)
    flat = moved.reshape((-1, input_length))
    lane_count = flat.size // input_length
    output = np.zeros((lane_count, target_length), dtype=np.float64)
    normalization = np.zeros((lane_count, target_length), dtype=np.float64)
    window = np.sin(np.pi * (np.arange(frame_length, dtype=np.float64) + 0.5) / frame_length)

    for lane_index, lane in enumerate(flat):
        analysis_source = _analysis_source(lane, frame_length)
        max_start = analysis_source.size - frame_length
        lane_output = output[lane_index]
        for frame_index in range(frame_count):
            synthesis_start = frame_index * synthesis_hop
            if synthesis_start >= target_length:
                break
            expected_start = round(frame_index * analysis_hop)
            if frame_index == 0:
                chosen_start = int(np.clip(expected_start, 0, max_start))
            else:
                expected_start = int(np.clip(expected_start, 0, max_start))
                search_start = max(0, expected_start - search_radius)
                search_stop = min(max_start, expected_start + search_radius)
                candidates = np.arange(search_start, search_stop + 1, dtype=np.int64)
                reference_end = min(target_length, synthesis_start + overlap_length)
                reference = lane_output[synthesis_start:reference_end]
                chosen_start = _choose_candidate(
                    analysis_source, reference, candidates, expected_start
                )

            frame = _complete_slice(analysis_source, chosen_start, frame_length)
            output_end = min(target_length, synthesis_start + frame_length)
            frame_size = output_end - synthesis_start
            lane_output[synthesis_start:output_end] += frame[:frame_size] * window[:frame_size]
            normalization[lane_index, synthesis_start:output_end] += window[:frame_size]

    output /= np.maximum(normalization, np.finfo(np.float64).eps)
    reshaped = output.reshape((*moved.shape[:-1], target_length))
    return np.moveaxis(reshaped, -1, normalized_axis).astype(source.dtype, copy=False)
