"""Private NumPy-only F0, voicing, and pitch-mark analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exceptions import InvalidParameterError


@dataclass(frozen=True)
class PitchTrack:
    """Frame and pulse analysis for one audio lane.

    ``frequencies`` contains zero for unvoiced frames.  ``pitch_marks`` are
    integer sample positions retained only for sufficiently coherent voiced
    regions.  The arrays are owned by the track and are safe for callers to
    inspect without affecting later analysis.
    """

    frame_times: np.ndarray
    frequencies: np.ndarray
    voiced: np.ndarray
    confidence: np.ndarray
    pitch_marks: np.ndarray
    voiced_intervals: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class _Candidate:
    frequency: float
    periodicity: float
    emission_cost: float = 0.0


def _validate_pitch_parameters(
    sample_rate: int,
    pitch_floor: float,
    pitch_ceiling: float,
) -> tuple[int, float, float]:
    rate = int(sample_rate)
    floor = float(pitch_floor)
    ceiling = float(pitch_ceiling)
    if rate <= 0:
        raise InvalidParameterError("sample_rate must be positive")
    if not np.isfinite(floor) or floor <= 0.0:
        raise InvalidParameterError("pitch_floor must be finite and positive")
    if not np.isfinite(ceiling) or ceiling <= floor:
        raise InvalidParameterError("pitch_ceiling must be finite and greater than pitch_floor")
    if ceiling >= rate / 2.0:
        raise InvalidParameterError("pitch_ceiling must be below the Nyquist frequency")
    return rate, floor, ceiling


def _analysis_geometry(sample_rate: int, pitch_floor: float) -> tuple[int, int]:
    maximum_period = int(np.ceil(sample_rate / pitch_floor))
    frame_length = max(3 * maximum_period + 1, round(sample_rate * 0.04))
    frame_length = min(frame_length, max(3 * maximum_period + 1, round(sample_rate * 0.12)))
    hop_length = max(1, round(sample_rate * 0.01))
    return frame_length, hop_length


def _frame_starts(length: int, frame_length: int, hop_length: int) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.int64)
    if length <= frame_length:
        return np.array([0], dtype=np.int64)
    count = 1 + int(np.ceil((length - frame_length) / hop_length))
    return np.arange(count, dtype=np.int64) * hop_length


def _normalized_correlations(
    frame: np.ndarray,
    min_lag: int,
    max_lag: int,
) -> np.ndarray:
    correlations = np.zeros(max_lag - min_lag + 1, dtype=np.float64)
    energy = float(np.dot(frame, frame))
    if energy <= np.finfo(np.float64).tiny:
        return correlations
    for offset, lag in enumerate(range(min_lag, max_lag + 1)):
        left = frame[:-lag]
        right = frame[lag:]
        denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
        if denominator > np.finfo(np.float64).tiny:
            correlations[offset] = float(np.dot(left, right) / denominator)
    return correlations


def _local_peak_indices(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return np.empty(0, dtype=np.int64)
    peaks = np.flatnonzero((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])) + 1
    return peaks.astype(np.int64, copy=False)


def _frame_pitch_candidates(
    frame: np.ndarray,
    sample_rate: int,
    pitch_floor: float,
    pitch_ceiling: float,
) -> list[_Candidate]:
    centered = np.asarray(frame, dtype=np.float64) - float(np.mean(frame, dtype=np.float64))
    if centered.size < 8:
        return []
    centered *= np.hanning(centered.size)
    min_lag = max(1, int(np.floor(sample_rate / pitch_ceiling)))
    max_lag = min(centered.size - 2, int(np.ceil(sample_rate / pitch_floor)))
    if max_lag <= min_lag:
        return []
    correlations = _normalized_correlations(centered, min_lag, max_lag)
    peak_indices = _local_peak_indices(correlations)
    if peak_indices.size == 0:
        return []
    # Retain alternatives so the dynamic program can reject a local octave
    # error instead of being forced to accept the strongest isolated peak.
    ranked = peak_indices[np.argsort(correlations[peak_indices])[::-1]]
    candidates: list[_Candidate] = []
    for index in ranked[:6]:
        confidence = float(np.clip(correlations[index], 0.0, 1.0))
        if confidence < 0.18:
            continue
        lag = float(min_lag + int(index))
        if 0 < index < correlations.size - 1:
            left = correlations[index - 1]
            center = correlations[index]
            right = correlations[index + 1]
            curvature = left - 2.0 * center + right
            if curvature < -np.finfo(np.float64).eps:
                lag -= 0.5 * (right - left) / curvature
        frequency = float(sample_rate / lag)
        if pitch_floor <= frequency <= pitch_ceiling and all(
            abs(np.log2(frequency / candidate.frequency)) > 0.035 for candidate in candidates
        ):
            candidates.append(_Candidate(frequency, confidence))
    return candidates


def _transition_cost(previous: _Candidate, current: _Candidate) -> float:
    if previous.frequency == 0.0 and current.frequency == 0.0:
        return 0.0
    if previous.frequency == 0.0 or current.frequency == 0.0:
        return 0.22
    octave_distance = float(abs(np.log2(current.frequency / previous.frequency)))
    return float(min(2.0, 0.95 * octave_distance + 0.12 * octave_distance * octave_distance))


def _track_pitch_candidates(
    candidates: list[list[_Candidate]],
    energies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(candidates)
    if count == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.astype(bool), empty
    states: list[list[_Candidate]] = []
    for frame_candidates, energy in zip(candidates, energies, strict=True):
        # A silence candidate is cheap for quiet frames and deliberately
        # somewhat expensive for energetic frames.
        silence_cost = 0.20 if energy < 0.01 else 0.78
        states.append([_Candidate(0.0, 0.0, silence_cost), *frame_candidates])
    costs = np.full((count, max(map(len, states))), np.inf, dtype=np.float64)
    back = np.zeros_like(costs, dtype=np.int64)

    def emission(candidate: _Candidate) -> float:
        if candidate.frequency == 0.0:
            return candidate.emission_cost
        # Reward coherent periodicity enough to retain low-pitched voices,
        # while weak noise correlations remain more expensive than silence.
        return 0.55 * (1.0 - candidate.periodicity) - 0.18

    first = states[0]
    for index, candidate in enumerate(first):
        costs[0, index] = emission(candidate)
    for frame_index in range(1, count):
        for current_index, current in enumerate(states[frame_index]):
            emission_cost = emission(current)
            previous_costs = costs[frame_index - 1, : len(states[frame_index - 1])]
            transition = np.array(
                [_transition_cost(previous, current) for previous in states[frame_index - 1]],
                dtype=np.float64,
            )
            best = int(np.argmin(previous_costs + transition))
            costs[frame_index, current_index] = (
                previous_costs[best] + transition[best] + emission_cost
            )
            back[frame_index, current_index] = best
    selected = np.zeros(count, dtype=np.int64)
    selected[-1] = int(np.argmin(costs[-1, : len(states[-1])]))
    for frame_index in range(count - 1, 0, -1):
        selected[frame_index - 1] = back[frame_index, selected[frame_index]]
    frequencies = np.array(
        [states[frame_index][selected[frame_index]].frequency for frame_index in range(count)],
        dtype=np.float64,
    )
    confidence = np.array(
        [
            states[frame_index][selected[frame_index]].periodicity
            if frequencies[frame_index] > 0.0
            else 0.0
            for frame_index in range(count)
        ],
        dtype=np.float64,
    )
    voiced = (frequencies > 0.0) & (confidence >= 0.35) & (energies >= 0.01)
    frequencies = np.where(voiced, frequencies, 0.0)
    confidence = np.where(voiced, confidence, 0.0)
    return frequencies, voiced, confidence


def _voiced_intervals(
    voiced: np.ndarray,
    frame_times: np.ndarray,
    frame_length: int,
    sample_count: int,
    sample_rate: int,
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    if voiced.size == 0:
        return intervals
    starts = np.flatnonzero(voiced & np.r_[True, ~voiced[:-1]])
    ends = np.flatnonzero(voiced & np.r_[~voiced[1:], True])
    half_seconds = frame_length / (2.0 * sample_rate)
    for start, end in zip(starts, ends, strict=True):
        if end - start + 1 < 3:
            continue
        left = max(0, round((frame_times[start] - half_seconds) * sample_rate))
        right = min(sample_count, round((frame_times[end] + half_seconds) * sample_rate))
        if right - left > 0:
            intervals.append((left, right))
    return intervals


def _correlation_score(signal: np.ndarray, first: int, second: int, width: int) -> float:
    left_start = max(0, first - width)
    right_start = max(0, second - width)
    available = min(signal.size - left_start, signal.size - right_start, 2 * width + 1)
    if available < 5:
        return -1.0
    left = signal[left_start : left_start + available]
    right = signal[right_start : right_start + available]
    left = left - np.mean(left, dtype=np.float64)
    right = right - np.mean(right, dtype=np.float64)
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else -1.0


def _refine_pitch_marks_lane(
    signal: np.ndarray,
    frame_times: np.ndarray,
    frequencies: np.ndarray,
    voiced: np.ndarray,
    intervals: list[tuple[int, int]],
    sample_rate: int,
    pitch_floor: float,
    pitch_ceiling: float,
) -> np.ndarray:
    marks: list[int] = []
    signal_values = np.asarray(signal, dtype=np.float64).reshape(-1)
    minimum_period = sample_rate / pitch_ceiling
    maximum_period = sample_rate / pitch_floor
    for left, right in intervals:
        frame_indices = np.flatnonzero(
            voiced & (frame_times >= left / sample_rate) & (frame_times <= right / sample_rate)
        )
        if frame_indices.size == 0:
            continue
        frame_positions = frame_times[frame_indices] * sample_rate
        frame_frequencies = frequencies[frame_indices]

        def local_period(
            position: float,
            frame_positions: np.ndarray = frame_positions,
            frame_frequencies: np.ndarray = frame_frequencies,
        ) -> float:
            frequency = float(np.interp(position, frame_positions, frame_frequencies))
            return float(np.clip(sample_rate / frequency, minimum_period, maximum_period))

        seed_period = local_period((left + right) / 2.0)
        seed_left = max(left, round((left + right) / 2.0 - seed_period))
        seed_right = min(right, round((left + right) / 2.0 + seed_period))
        if seed_right <= seed_left:
            continue
        center = seed_left + int(np.argmax(np.abs(signal_values[seed_left:seed_right])))
        local: list[int] = [center]
        for direction in (-1, 1):
            current = center
            propagated: list[int] = []
            while True:
                period = local_period(current)
                position = current + direction * period
                candidate_left = max(left, int(np.floor(position - 0.22 * period)))
                candidate_right = min(right - 1, int(np.ceil(position + 0.22 * period)))
                if candidate_right <= candidate_left:
                    break
                width = max(3, round(0.35 * period))
                candidates = np.arange(candidate_left, candidate_right + 1, dtype=np.int64)
                scores = np.array(
                    [
                        _correlation_score(signal_values, current, int(candidate), width)
                        for candidate in candidates
                    ]
                )
                best_index = int(np.argmax(scores))
                best = int(candidates[best_index])
                distance = abs(best - current)
                if (
                    scores[best_index] < 0.35
                    or not minimum_period * 0.7 <= distance <= maximum_period * 1.3
                ):
                    break
                propagated.append(best)
                current = best
            if direction < 0:
                local = list(reversed(propagated)) + local
            else:
                local.extend(propagated)
        local = sorted({mark for mark in local if left <= mark < right})
        if len(local) >= 3:
            marks.extend(local)
    if not marks:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.asarray(marks, dtype=np.int64))


def estimate_pitch_track_lane(
    audio: np.ndarray,
    *,
    sample_rate: int,
    pitch_floor: float = 60.0,
    pitch_ceiling: float = 500.0,
) -> PitchTrack:
    """Estimate a conservative F0 track and pulse marks for one lane."""

    rate, floor, ceiling = _validate_pitch_parameters(sample_rate, pitch_floor, pitch_ceiling)
    signal = np.asarray(audio, dtype=np.float64).reshape(-1)
    frame_length, hop_length = _analysis_geometry(rate, floor)
    starts = _frame_starts(signal.size, frame_length, hop_length)
    if starts.size == 0:
        empty_float = np.empty(0, dtype=np.float64)
        return PitchTrack(
            empty_float,
            empty_float,
            np.empty(0, dtype=bool),
            empty_float,
            np.empty(0, dtype=np.int64),
            (),
        )
    candidates: list[list[_Candidate]] = []
    energies = np.empty(starts.size, dtype=np.float64)
    for index, start in enumerate(starts):
        frame = np.zeros(frame_length, dtype=np.float64)
        chunk = signal[start : start + frame_length]
        frame[: chunk.size] = chunk
        frame -= np.mean(frame, dtype=np.float64)
        scale = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        energies[index] = scale / (float(np.max(np.abs(signal))) + 1e-12)
        candidates.append(_frame_pitch_candidates(frame, rate, floor, ceiling))
    frame_times = (starts + frame_length / 2.0) / rate
    frequencies, voiced, confidence = _track_pitch_candidates(candidates, energies)
    intervals = _voiced_intervals(voiced, frame_times, frame_length, signal.size, rate)
    pitch_marks = _refine_pitch_marks_lane(
        signal,
        frame_times,
        frequencies,
        voiced,
        intervals,
        rate,
        floor,
        ceiling,
    )
    return PitchTrack(
        np.asarray(frame_times, dtype=np.float64),
        np.asarray(frequencies, dtype=np.float64),
        np.asarray(voiced, dtype=bool),
        np.asarray(confidence, dtype=np.float64),
        np.asarray(pitch_marks, dtype=np.int64),
        tuple(intervals),
    )


def estimate_pitch_track(
    audio: np.ndarray,
    *,
    sample_rate: int,
    axis: int = -1,
    pitch_floor: float = 60.0,
    pitch_ceiling: float = 500.0,
) -> list[PitchTrack]:
    """Estimate independent pitch tracks for all lanes of an array."""

    values = np.moveaxis(np.asarray(audio), axis, -1)
    return [
        estimate_pitch_track_lane(
            lane, sample_rate=sample_rate, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling
        )
        for lane in values.reshape(-1, values.shape[-1])
    ]
