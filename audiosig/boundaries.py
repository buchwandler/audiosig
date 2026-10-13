"""Waveform boundary selection primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ._validation import validate_audio, validate_finite, validate_integer, validate_positive
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
    return np.asarray(np.max(values, axis=tuple(range(values.ndim - 1))), dtype=values.dtype)


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


@dataclass(frozen=True)
class PauseCropConfig:
    """Thresholds and durations for pause-aware reference cropping."""

    frame_seconds: float = 0.025
    hop_seconds: float = 0.010
    floor_quantile: float = 0.10
    quiet_multiplier: float = 4.0
    pause_min_seconds: float = 0.10
    pause_keep_seconds: float = 0.15
    search_forward_seconds: float = 5.0
    search_backward_seconds: float = 5.0
    min_keep_fraction: float = 0.75
    room_tone_seconds: float = 0.25
    fade_seconds: float = 0.02


class RandomSource(Protocol):
    """Minimal interface required for injected room-tone noise sources."""

    def standard_normal(self, size: tuple[int, ...]) -> np.ndarray: ...


_DEFAULT_PAUSE_CROP_CONFIG = PauseCropConfig()


def _validate_pause_crop_config(config: object) -> PauseCropConfig:
    if not isinstance(config, PauseCropConfig):
        raise InvalidParameterError("config must be a PauseCropConfig")
    return config


def _xorshift32_normal(size: tuple[int, ...], seed: int) -> np.ndarray:
    """Generate standard normals using the pinned xorshift32/Box-Muller sequence."""
    mask = 0xFFFFFFFF
    state = seed if seed else 0x6D2B79F5
    count = int(np.prod(size, dtype=np.int64))
    output = np.empty(count, dtype=np.float64)
    spare: float | None = None

    def uniform() -> float:
        nonlocal state
        value = state
        value ^= (value << 13) & mask
        value ^= value >> 17
        value ^= (value << 5) & mask
        state = value & mask
        return (state + 0.5) / 4294967296.0

    def normal() -> float:
        nonlocal spare
        if spare is not None:
            value = spare
            spare = None
            return value
        magnitude = np.sqrt(-2.0 * np.log(max(1.0e-12, uniform())))
        angle = 2.0 * np.pi * uniform()
        spare = float(magnitude * np.sin(angle))
        return float(magnitude * np.cos(angle))

    for index in range(count):
        output[index] = normal()
    return output.reshape(size)


def _pause_runs(quiet: np.ndarray, minimum_frames: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    index = 0
    while index < quiet.size:
        if not bool(quiet[index]):
            index += 1
            continue
        start = index
        while index < quiet.size and bool(quiet[index]):
            index += 1
        if index - start >= minimum_frames:
            runs.append((start, index))
    return runs


def _completed_with_room_tone(
    audio: np.ndarray,
    *,
    sample_rate: int,
    floor: float,
    config: PauseCropConfig,
    room_tone_rng: RandomSource | None,
    room_tone_seed: int | None,
) -> np.ndarray:
    faded = np.array(audio, copy=True)
    fade_count = min(int(config.fade_seconds * sample_rate), faded.shape[-1])
    if fade_count:
        ramp = np.linspace(1.0, 0.0, fade_count, dtype=np.float64)
        faded[..., -fade_count:] *= ramp.astype(faded.dtype, copy=False)
    tone_count = int(config.room_tone_seconds * sample_rate)
    tone_shape = (*audio.shape[:-1], tone_count)
    if tone_count == 0:
        tone = np.empty(tone_shape, dtype=audio.dtype)
    elif room_tone_seed is not None:
        tone = _xorshift32_normal(tone_shape, room_tone_seed)
    elif room_tone_rng is not None:
        tone = np.asarray(room_tone_rng.standard_normal(tone_shape), dtype=np.float64)
        if tone.shape != tone_shape or not np.isfinite(tone).all():
            raise InvalidParameterError(
                "room_tone_rng must return finite values with the requested shape"
            )
    else:
        tone = np.zeros(tone_shape, dtype=audio.dtype)
    tone = (tone.astype(np.float64, copy=False) * float(floor)).astype(audio.dtype)
    return np.concatenate((faded, tone), axis=-1)


def crop_on_pause(
    audio: np.ndarray,
    *,
    target_seconds: float,
    sample_rate: int,
    config: PauseCropConfig = _DEFAULT_PAUSE_CROP_CONFIG,
    room_tone_rng: RandomSource | None = None,
    room_tone_seed: int | None = None,
    axis: int = -1,
) -> np.ndarray:
    """Crop at a nearby quiet pause or complete the tail with room tone.

    If neither an RNG nor a seed is supplied, completion uses zero-valued room
    tone. Supplying ``room_tone_seed`` uses a local xorshift32/Box-Muller source;
    no process-global random state is consulted.
    """
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    config = _validate_pause_crop_config(config)
    rate = validate_integer(sample_rate, "sample_rate")
    target_duration = validate_positive(target_seconds, "target_seconds")
    frame_duration = validate_positive(config.frame_seconds, "frame_seconds")
    hop_duration = validate_positive(config.hop_seconds, "hop_seconds")
    floor_quantile = validate_finite(config.floor_quantile, "floor_quantile")
    if not 0.0 <= floor_quantile <= 1.0:
        raise InvalidParameterError("floor_quantile must be in the interval [0, 1]")
    quiet_multiplier = validate_positive(config.quiet_multiplier, "quiet_multiplier")
    pause_min = validate_finite(config.pause_min_seconds, "pause_min_seconds")
    pause_keep = validate_finite(config.pause_keep_seconds, "pause_keep_seconds")
    forward_search = validate_finite(config.search_forward_seconds, "search_forward_seconds")
    backward_search = validate_finite(config.search_backward_seconds, "search_backward_seconds")
    minimum_fraction = validate_finite(config.min_keep_fraction, "min_keep_fraction")
    room_tone_duration = validate_finite(config.room_tone_seconds, "room_tone_seconds")
    fade_duration = validate_finite(config.fade_seconds, "fade_seconds")
    if (
        min(
            pause_min,
            pause_keep,
            forward_search,
            backward_search,
            room_tone_duration,
            fade_duration,
        )
        < 0.0
    ):
        raise InvalidParameterError(
            "pause, search, room-tone, and fade durations must be non-negative"
        )
    if not 0.0 <= minimum_fraction <= 1.0:
        raise InvalidParameterError("min_keep_fraction must be in the interval [0, 1]")
    if room_tone_rng is not None and not callable(getattr(room_tone_rng, "standard_normal", None)):
        raise InvalidParameterError("room_tone_rng must provide standard_normal(size)")
    seed = None
    if room_tone_seed is not None:
        seed = validate_integer(room_tone_seed, "room_tone_seed", minimum=0)
        if seed > 0xFFFFFFFF:
            raise InvalidParameterError("room_tone_seed must fit in an unsigned 32-bit integer")
        if room_tone_rng is not None:
            raise InvalidParameterError("provide either room_tone_rng or room_tone_seed, not both")

    moved = np.moveaxis(source, normalized_axis, -1)
    total = moved.shape[-1]
    if total == 0 or source.size == 0:
        return np.array(source, copy=True, order="C")
    requested_target = target_duration * rate
    if not np.isfinite(requested_target) or requested_target > np.iinfo(np.intp).max:
        raise InvalidParameterError("target crop length is too large")
    target_samples = int(np.floor(requested_target + 0.5))
    frame_length = int(rate * frame_duration)
    hop_length = int(rate * hop_duration)
    if frame_length < 1 or hop_length < 1:
        raise InvalidParameterError("frame_seconds and hop_seconds must span at least one sample")
    if total < 4 * frame_length:
        return np.array(source, copy=True, order="C")

    flat = moved.reshape((-1, total)).astype(np.float64, copy=False)
    frame_count = 1 + (total - frame_length) // hop_length
    starts = np.arange(frame_count, dtype=np.int64) * hop_length
    rms = np.empty(frame_count, dtype=np.float64)
    sample_offsets = np.arange(frame_length, dtype=np.int64)
    for chunk_start in range(0, frame_count, 1024):
        chunk_stop = min(frame_count, chunk_start + 1024)
        indices = starts[chunk_start:chunk_stop, None] + sample_offsets[None, :]
        frames = flat[:, indices]
        rms[chunk_start:chunk_stop] = np.sqrt(
            np.mean(np.square(frames), axis=(0, 2), dtype=np.float64)
        )
    rms = np.maximum(rms, 1e-6)
    floor = float(np.quantile(rms, floor_quantile))
    minimum_frames = max(1, round(pause_min / hop_duration))
    runs = _pause_runs(rms < floor * quiet_multiplier, minimum_frames)
    keep_frames = round(pause_keep / hop_duration)

    def copy_prefix(sample_count: int) -> np.ndarray:
        return np.array(moved[..., :sample_count], copy=True, order="C")

    def cut_at(run: tuple[int, int]) -> np.ndarray:
        start, stop = run
        kept = min(stop - start, keep_frames)
        return copy_prefix((start + kept) * hop_length + frame_length)

    if total <= target_samples:
        if runs and runs[-1][1] * hop_length + frame_length >= total - hop_length:
            result = moved
        else:
            inside = [run for run in runs if run[0] * hop_length >= int(total * minimum_fraction)]
            if inside:
                result = cut_at(inside[-1])
            else:
                result = _completed_with_room_tone(
                    moved,
                    sample_rate=rate,
                    floor=floor,
                    config=config,
                    room_tone_rng=room_tone_rng,
                    room_tone_seed=seed,
                )
    else:
        forward_limit = target_samples + int(forward_search * rate)
        forward = [run for run in runs if target_samples <= run[0] * hop_length <= forward_limit]
        if forward:
            result = cut_at(forward[0])
        else:
            backward_limit = target_samples - int(backward_search * rate)
            backward = [
                run
                for run in runs
                if run[0] * hop_length < target_samples and run[1] * hop_length >= backward_limit
            ]
            if backward:
                result = cut_at(backward[-1])
            else:
                result = _completed_with_room_tone(
                    moved[..., :target_samples],
                    sample_rate=rate,
                    floor=floor,
                    config=config,
                    room_tone_rng=room_tone_rng,
                    room_tone_seed=seed,
                )
    return np.ascontiguousarray(np.moveaxis(result, -1, normalized_axis), dtype=source.dtype)


def _nonoverlap_frame_rms(audio: np.ndarray, frame_length: int) -> np.ndarray:
    sample_count = audio.shape[-1]
    frame_count = sample_count // frame_length
    if frame_count == 0:
        return np.empty(0, dtype=np.float64)
    if audio.size == 0:
        return np.zeros(frame_count, dtype=np.float64)
    flat = audio.reshape((-1, sample_count)).astype(np.float64, copy=False)
    frames = flat[:, : frame_count * frame_length].reshape(-1, frame_count, frame_length)
    return np.sqrt(np.mean(np.square(frames), axis=(0, 2), dtype=np.float64))


def _db_amplitude(db: float, name: str) -> float:
    with np.errstate(over="ignore"):
        amplitude = float(np.power(10.0, db / 20.0))
    if not np.isfinite(amplitude):
        raise InvalidParameterError(f"{name} produces an unrepresentable amplitude")
    return amplitude


def _speech_onset_threshold(
    rms: np.ndarray, *, absolute_threshold_db: float, floor_quantile: float, over_floor_db: float
) -> float:
    threshold = _db_amplitude(absolute_threshold_db, "absolute_threshold_db")
    if rms.size >= 30:
        floor = float(np.quantile(rms, floor_quantile))
        threshold = max(threshold, floor * _db_amplitude(over_floor_db, "over_floor_db"))
    return threshold


def _energy_threshold(
    rms: np.ndarray, *, absolute_threshold_db: float, floor_quantile: float, over_floor_db: float
) -> float:
    absolute = _db_amplitude(absolute_threshold_db, "absolute_threshold_db")
    floor = float(np.quantile(rms, floor_quantile)) if rms.size else 0.0
    return max(absolute, floor * _db_amplitude(over_floor_db, "over_floor_db"))


def speech_onset(
    audio: np.ndarray,
    *,
    sample_rate: int,
    absolute_threshold_db: float = -45.0,
    floor_quantile: float = 0.10,
    over_floor_db: float = 15.0,
    frame_seconds: float = 0.010,
    window_frames: int = 6,
    required_frames: int = 5,
    axis: int = -1,
) -> int | None:
    """Return the first sample of a window with sustained speech-level energy."""
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    rate = validate_integer(sample_rate, "sample_rate")
    frame_duration = validate_positive(frame_seconds, "frame_seconds")
    absolute_db = validate_finite(absolute_threshold_db, "absolute_threshold_db")
    quantile = validate_finite(floor_quantile, "floor_quantile")
    over_db = validate_finite(over_floor_db, "over_floor_db")
    if not 0.0 <= quantile <= 1.0:
        raise InvalidParameterError("floor_quantile must be in the interval [0, 1]")
    window_count = validate_integer(window_frames, "window_frames")
    required_count = validate_integer(required_frames, "required_frames")
    if required_count > window_count:
        raise InvalidParameterError("required_frames must not exceed window_frames")
    frame_length = int(rate * frame_duration)
    if frame_length < 1:
        raise InvalidParameterError("frame_seconds must span at least one sample")
    moved = np.moveaxis(source, normalized_axis, -1)
    rms = _nonoverlap_frame_rms(moved, frame_length)
    if rms.size < window_count:
        return None
    threshold = _speech_onset_threshold(
        rms,
        absolute_threshold_db=absolute_db,
        floor_quantile=quantile,
        over_floor_db=over_db,
    )
    above = (rms > threshold).astype(np.int64)
    counts = np.convolve(above, np.ones(window_count, dtype=np.int64), mode="valid")
    hits = np.flatnonzero(counts >= required_count)
    return None if hits.size == 0 else int(hits[0] * frame_length)


def energy_onset(
    audio: np.ndarray,
    *,
    sample_rate: int,
    over_floor_db: float = 10.0,
    min_frames: int = 3,
    absolute_threshold_db: float = -45.0,
    floor_quantile: float = 0.10,
    frame_seconds: float = 0.010,
    axis: int = -1,
) -> int | None:
    """Return the start sample of the first run of energetic frames."""
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    rate = validate_integer(sample_rate, "sample_rate")
    frame_duration = validate_positive(frame_seconds, "frame_seconds")
    over_db = validate_finite(over_floor_db, "over_floor_db")
    absolute_db = validate_finite(absolute_threshold_db, "absolute_threshold_db")
    quantile = validate_finite(floor_quantile, "floor_quantile")
    if not 0.0 <= quantile <= 1.0:
        raise InvalidParameterError("floor_quantile must be in the interval [0, 1]")
    minimum_frames = validate_integer(min_frames, "min_frames")
    frame_length = int(rate * frame_duration)
    if frame_length < 1:
        raise InvalidParameterError("frame_seconds must span at least one sample")
    moved = np.moveaxis(source, normalized_axis, -1)
    rms = _nonoverlap_frame_rms(moved, frame_length)
    if rms.size < minimum_frames:
        return None
    threshold = _energy_threshold(
        rms,
        absolute_threshold_db=absolute_db,
        floor_quantile=quantile,
        over_floor_db=over_db,
    )
    above = (rms > threshold).astype(np.int64)
    counts = np.convolve(above, np.ones(minimum_frames, dtype=np.int64), mode="valid")
    hits = np.flatnonzero(counts == minimum_frames)
    return None if hits.size == 0 else int(hits[0] * frame_length)


def energy_fraction(
    audio: np.ndarray,
    *,
    sample_rate: int,
    start: int,
    end: int,
    over_floor_db: float = 10.0,
    absolute_threshold_db: float = -45.0,
    floor_quantile: float = 0.10,
    frame_seconds: float = 0.010,
    axis: int = -1,
) -> float:
    """Return the fraction of segment frames above the global energy threshold."""
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    rate = validate_integer(sample_rate, "sample_rate")
    begin = validate_integer(start, "start", minimum=0)
    stop = validate_integer(end, "end", minimum=0)
    frame_duration = validate_positive(frame_seconds, "frame_seconds")
    over_db = validate_finite(over_floor_db, "over_floor_db")
    absolute_db = validate_finite(absolute_threshold_db, "absolute_threshold_db")
    quantile = validate_finite(floor_quantile, "floor_quantile")
    if not 0.0 <= quantile <= 1.0:
        raise InvalidParameterError("floor_quantile must be in the interval [0, 1]")
    frame_length = int(rate * frame_duration)
    if frame_length < 1:
        raise InvalidParameterError("frame_seconds must span at least one sample")
    moved = np.moveaxis(source, normalized_axis, -1)
    if stop > moved.shape[-1]:
        raise InvalidParameterError("end must not exceed the sample count")
    if stop <= begin or moved.shape[-1] < frame_length:
        return 0.0
    rms = _nonoverlap_frame_rms(moved, frame_length)
    threshold = _energy_threshold(
        rms,
        absolute_threshold_db=absolute_db,
        floor_quantile=quantile,
        over_floor_db=over_db,
    )
    start_frame = begin // frame_length
    end_frame = max(start_frame + 1, stop // frame_length)
    selected = rms[start_frame:end_frame]
    return 0.0 if selected.size == 0 else float(np.mean(selected > threshold))


def trim_lead(
    audio: np.ndarray,
    *,
    sample_rate: int,
    lead_seconds: float = 0.08,
    skip_seconds: float = 0.0,
    absolute_threshold_db: float = -45.0,
    floor_quantile: float = 0.10,
    over_floor_db: float = 15.0,
    frame_seconds: float = 0.010,
    window_frames: int = 6,
    required_frames: int = 5,
    axis: int = -1,
) -> np.ndarray:
    """Trim leading silence while retaining lead and a 20-ms onset safety margin."""
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    rate = validate_integer(sample_rate, "sample_rate")
    lead = validate_finite(lead_seconds, "lead_seconds")
    skip = validate_finite(skip_seconds, "skip_seconds")
    if lead < 0.0 or skip < 0.0:
        raise InvalidParameterError("lead_seconds and skip_seconds must be non-negative")
    onset = speech_onset(
        source,
        sample_rate=rate,
        absolute_threshold_db=absolute_threshold_db,
        floor_quantile=floor_quantile,
        over_floor_db=over_floor_db,
        frame_seconds=frame_seconds,
        window_frames=window_frames,
        required_frames=required_frames,
        axis=normalized_axis,
    )
    moved = np.moveaxis(source, normalized_axis, -1)
    if onset is None:
        return np.array(source, copy=True, order="C")
    cut = max(onset - int(lead * rate), int(skip * rate))
    cut = min(cut, max(0, onset - int(0.020 * rate)))
    return np.ascontiguousarray(np.moveaxis(moved[..., cut:], -1, normalized_axis))


def trim_trail(
    audio: np.ndarray,
    *,
    sample_rate: int,
    trail_seconds: float = 0.30,
    absolute_threshold_db: float = -45.0,
    floor_quantile: float = 0.10,
    over_floor_db: float = 15.0,
    frame_seconds: float = 0.010,
    axis: int = -1,
) -> np.ndarray:
    """Trim trailing silence after the last active frame plus a requested trail."""
    source, normalized_axis = validate_audio(audio, axis=axis, allow_empty=True)
    rate = validate_integer(sample_rate, "sample_rate")
    trail = validate_finite(trail_seconds, "trail_seconds")
    if trail < 0.0:
        raise InvalidParameterError("trail_seconds must be non-negative")
    frame_duration = validate_positive(frame_seconds, "frame_seconds")
    absolute_db = validate_finite(absolute_threshold_db, "absolute_threshold_db")
    quantile = validate_finite(floor_quantile, "floor_quantile")
    over_db = validate_finite(over_floor_db, "over_floor_db")
    if not 0.0 <= quantile <= 1.0:
        raise InvalidParameterError("floor_quantile must be in the interval [0, 1]")
    frame_length = int(rate * frame_duration)
    if frame_length < 1:
        raise InvalidParameterError("frame_seconds must span at least one sample")
    moved = np.moveaxis(source, normalized_axis, -1)
    rms = _nonoverlap_frame_rms(moved, frame_length)
    if rms.size == 0:
        return np.array(source, copy=True, order="C")
    threshold = _speech_onset_threshold(
        rms,
        absolute_threshold_db=absolute_db,
        floor_quantile=quantile,
        over_floor_db=over_db,
    )
    active = np.flatnonzero(rms > threshold)
    if active.size == 0:
        return np.array(source, copy=True, order="C")
    end = min(moved.shape[-1], (int(active[-1]) + 1) * frame_length + int(trail * rate))
    return np.ascontiguousarray(np.moveaxis(moved[..., :end], -1, normalized_axis))
