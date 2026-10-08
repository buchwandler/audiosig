"""NumPy-only STOI and ESTOI reference intelligibility metrics."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ._validation import validate_audio, validate_integer
from .exceptions import AudioShapeError, InvalidParameterError
from .metrics import MetricResult

_SAMPLE_RATE = 10_000
_FRAME_LENGTH = 256
_N_FFT = 512
_HOP_LENGTH = _FRAME_LENGTH // 2
_BAND_COUNT = 15
_FIRST_CENTER_HZ = 150.0
_SEGMENT_FRAMES = 30
_DYNAMIC_RANGE_DB = 40.0
_BETA_DB = -15.0
_EPSILON = 1e-12
_WINDOW = np.hanning(_FRAME_LENGTH).astype(np.float64)


def _validate_pair(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None,
    alignment: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    rate = validate_integer(sample_rate, "sample_rate")
    estimate_rate = (
        rate
        if estimate_sample_rate is None
        else validate_integer(estimate_sample_rate, "estimate_sample_rate")
    )
    if rate != _SAMPLE_RATE or estimate_rate != _SAMPLE_RATE:
        raise InvalidParameterError(
            "STOI and ESTOI require both sample rates to be 10000 Hz; "
            "resample explicitly with audiosig.resample()"
        )
    if alignment != "none":
        raise InvalidParameterError("alignment must be 'none'")
    ref, _ = validate_audio(reference)
    est, _ = validate_audio(estimate)
    if ref.ndim != 1 or est.ndim != 1:
        raise AudioShapeError("STOI and ESTOI require mono arrays shaped (samples,)")
    if ref.shape != est.shape:
        raise AudioShapeError("reference and estimate must have equal sample counts")
    return (
        ref.astype(np.float64, copy=False),
        est.astype(np.float64, copy=False),
        rate,
        estimate_rate,
    )


def _complete_frames(signal: np.ndarray) -> np.ndarray:
    if signal.size < _FRAME_LENGTH:
        return np.empty((0, _FRAME_LENGTH), dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(signal, _FRAME_LENGTH)[::_HOP_LENGTH]


def _compress_silent_frames(
    reference: np.ndarray, estimate: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    ref_frames = _complete_frames(reference)
    est_frames = _complete_frames(estimate)
    if ref_frames.shape[0] == 0:
        raise InvalidParameterError("signals are too short for a 256-sample intelligibility frame")
    rms = np.sqrt(np.mean(np.square(ref_frames * _WINDOW), axis=-1, dtype=np.float64))
    peak = float(np.max(rms))
    active = rms > peak * (10.0 ** (-_DYNAMIC_RANGE_DB / 20.0))
    active_count = int(np.count_nonzero(active))
    if active_count == 0:
        raise InvalidParameterError("reference has no active frames for intelligibility analysis")

    target_length = (active_count - 1) * _HOP_LENGTH + _FRAME_LENGTH
    compressed_ref = np.zeros(target_length, dtype=np.float64)
    compressed_est = np.zeros(target_length, dtype=np.float64)
    target_frame = 0
    for ref_frame, est_frame, is_active in zip(ref_frames, est_frames, active, strict=True):
        if not is_active:
            continue
        start = target_frame * _HOP_LENGTH
        end = start + _FRAME_LENGTH
        compressed_ref[start:end] += ref_frame * _WINDOW
        compressed_est[start:end] += est_frame * _WINDOW
        target_frame += 1
    return compressed_ref, compressed_est, active_count


def _short_time_spectrogram(signal: np.ndarray) -> np.ndarray:
    frames = _complete_frames(signal)
    if frames.shape[0] == 0:
        return np.empty((_N_FFT // 2 + 1, 0), dtype=np.complex128)
    spectrum = np.fft.rfft(frames * _WINDOW, n=_N_FFT, axis=-1)
    return np.moveaxis(spectrum, 0, 1)


def _third_octave_matrix() -> np.ndarray:
    frequencies = np.fft.rfftfreq(_N_FFT, d=1.0 / _SAMPLE_RATE)
    bin_width = _SAMPLE_RATE / _N_FFT
    matrix = np.zeros((_BAND_COUNT, frequencies.size), dtype=np.float64)
    for band in range(_BAND_COUNT):
        center = _FIRST_CENTER_HZ * (2.0 ** (band / 3.0))
        lower_edge = center * (2.0 ** (-1.0 / 6.0))
        upper_edge = center * (2.0 ** (1.0 / 6.0))
        lower_bin = int(np.clip(np.rint(lower_edge / bin_width), 0, frequencies.size - 1))
        upper_bin = int(np.clip(np.rint(upper_edge / bin_width), 0, frequencies.size))
        if upper_bin > lower_bin:
            matrix[band, lower_bin:upper_bin] = 1.0
    return matrix


_THIRD_OCTAVE_MATRIX = _third_octave_matrix()


def _band_envelopes(signal: np.ndarray) -> np.ndarray:
    spectrum = _short_time_spectrogram(signal)
    power = np.square(np.abs(spectrum))
    return np.sqrt(_THIRD_OCTAVE_MATRIX @ power)


def _segments(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref_segments = np.lib.stride_tricks.sliding_window_view(reference, _SEGMENT_FRAMES, axis=-1)
    est_segments = np.lib.stride_tricks.sliding_window_view(estimate, _SEGMENT_FRAMES, axis=-1)
    return np.moveaxis(ref_segments, 1, 0), np.moveaxis(est_segments, 1, 0)


def _prepare_segments(
    reference: np.ndarray, estimate: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    compressed_ref, compressed_est, active_frames = _compress_silent_frames(reference, estimate)
    ref_bands = _band_envelopes(compressed_ref)
    est_bands = _band_envelopes(compressed_est)
    if ref_bands.shape != est_bands.shape:
        raise AudioShapeError("equal-length input signals produced different analysis frame counts")
    if ref_bands.shape[-1] < _SEGMENT_FRAMES:
        raise InvalidParameterError(
            "at least 30 active intelligibility frames are required after silence removal"
        )
    ref_segments, est_segments = _segments(ref_bands, est_bands)
    return ref_segments, est_segments, active_frames, ref_bands.shape[-1]


def _stoi_value(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, int, int, int]:
    ref_segments, est_segments, active_frames, total_frames = _prepare_segments(reference, estimate)
    ref_norm = np.linalg.norm(ref_segments, axis=-1, keepdims=True)
    est_norm = np.linalg.norm(est_segments, axis=-1, keepdims=True)
    scale = ref_norm / (est_norm + _EPSILON)
    normalized_estimate = est_segments * scale
    upper_limit = ref_segments * (1.0 + 10.0 ** (-_BETA_DB / 20.0))
    clipped_estimate = np.minimum(normalized_estimate, upper_limit)

    ref_centered = ref_segments - np.mean(ref_segments, axis=-1, keepdims=True)
    est_centered = clipped_estimate - np.mean(clipped_estimate, axis=-1, keepdims=True)
    numerator = np.sum(ref_centered * est_centered, axis=-1, dtype=np.float64)
    denominator = (
        np.linalg.norm(ref_centered, axis=-1) * np.linalg.norm(est_centered, axis=-1) + _EPSILON
    )
    value = float(np.mean(numerator / denominator, dtype=np.float64))
    return value, active_frames, total_frames, ref_segments.shape[0]


def _unit_normalize(values: np.ndarray, axis: int) -> np.ndarray:
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    normalized: np.ndarray = np.divide(
        values, norms, out=np.zeros_like(values), where=norms > _EPSILON
    )
    return normalized


def _estoi_value(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, int, int, int]:
    ref_segments, est_segments, active_frames, total_frames = _prepare_segments(reference, estimate)
    ref_rows = _unit_normalize(
        ref_segments - np.mean(ref_segments, axis=-1, keepdims=True), axis=-1
    )
    est_rows = _unit_normalize(
        est_segments - np.mean(est_segments, axis=-1, keepdims=True), axis=-1
    )
    ref_columns = _unit_normalize(ref_rows - np.mean(ref_rows, axis=1, keepdims=True), axis=1)
    est_columns = _unit_normalize(est_rows - np.mean(est_rows, axis=1, keepdims=True), axis=1)
    per_segment = np.sum(ref_columns * est_columns, axis=(1, 2), dtype=np.float64)
    value = float(np.mean(per_segment / _SEGMENT_FRAMES, dtype=np.float64))
    return value, active_frames, total_frames, ref_segments.shape[0]


def _metric_result(
    name: str,
    value: float,
    reference: np.ndarray,
    *,
    estimate_sample_rate: int,
    active_frames: int,
    total_frames: int,
    segments: int,
) -> MetricResult:
    parameters: dict[str, str | int | float | bool | None] = {
        "sample_rate": _SAMPLE_RATE,
        "estimate_sample_rate": estimate_sample_rate,
        "alignment": "none",
        "frame_length": _FRAME_LENGTH,
        "n_fft": _N_FFT,
        "hop_length": _HOP_LENGTH,
        "n_bands": _BAND_COUNT,
        "first_band_center_hz": _FIRST_CENTER_HZ,
        "segment_frames": _SEGMENT_FRAMES,
        "dynamic_range_db": _DYNAMIC_RANGE_DB,
        "window": "hann_symmetric",
        "center": False,
        "spectral_power": 2.0,
        "band_spacing_octaves": 1.0 / 3.0,
        "band_edge_offset_octaves": 1.0 / 6.0,
        "band_edge_policy": "nearest_fft_bin",
        "segment_hop_frames": 1,
        "epsilon": _EPSILON,
    }
    if name == "stoi":
        parameters["beta_db"] = _BETA_DB
    diagnostics: dict[str, str | int | float | bool | None] = {
        "samples": int(reference.size),
        "channels_or_lanes": 1,
        "active_frames": active_frames,
        "analysis_frames": total_frames,
        "segments": segments,
    }
    return MetricResult(
        name=name,
        value=float(value),
        unit="score",
        higher_is_better=True,
        parameters=dict(parameters),
        diagnostics=dict(diagnostics),
    )


def stoi(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None = None,
    alignment: Literal["none"] = "none",
) -> MetricResult:
    """Compute the original short-time objective intelligibility (STOI) score.

    Input must be explicitly mono, equal-length ``float32``/``float64`` audio at
    10 kHz. Resampling and alignment are caller-owned operations.
    """
    ref, est, _, estimate_rate = _validate_pair(
        reference,
        estimate,
        sample_rate=sample_rate,
        estimate_sample_rate=estimate_sample_rate,
        alignment=alignment,
    )
    value, active_frames, total_frames, segment_count = _stoi_value(ref, est)
    return _metric_result(
        "stoi",
        value,
        ref,
        estimate_sample_rate=estimate_rate,
        active_frames=active_frames,
        total_frames=total_frames,
        segments=segment_count,
    )


def estoi(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None = None,
    alignment: Literal["none"] = "none",
) -> MetricResult:
    """Compute extended STOI (ESTOI) using row/column-normalized segments.

    Input must be explicitly mono, equal-length ``float32``/``float64`` audio
    at 10 kHz. Resampling and alignment are caller-owned operations.
    """
    ref, est, _, estimate_rate = _validate_pair(
        reference,
        estimate,
        sample_rate=sample_rate,
        estimate_sample_rate=estimate_sample_rate,
        alignment=alignment,
    )
    value, active_frames, total_frames, segment_count = _estoi_value(ref, est)
    return _metric_result(
        "estoi",
        value,
        ref,
        estimate_sample_rate=estimate_rate,
        active_frames=active_frames,
        total_frames=total_frames,
        segments=segment_count,
    )
