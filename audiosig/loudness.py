"""Standards-based loudness and peak measurements for NumPy audio."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._resampling import resample
from ._validation import validate_audio, validate_integer
from .exceptions import AudioShapeError

_ABSOLUTE_GATE_LUFS = -70.0
_RELATIVE_GATE_LU = -10.0
_BLOCK_DURATION_SECONDS = 0.400
_HOP_DURATION_SECONDS = 0.100
_LUFS_OFFSET = -0.691


@dataclass(frozen=True, slots=True)
class LoudnessMetrics:
    """Immutable loudness and peak measurements for one audio signal."""

    integrated_lufs: float
    sample_peak_dbfs: float
    true_peak_dbtp: float


def _validate_mono_audio(audio: np.ndarray, axis: int) -> np.ndarray:
    validated, normalized_axis = validate_audio(audio, axis=axis)
    if validated.ndim != 1:
        raise AudioShapeError("loudness measurement supports one-dimensional mono audio only")
    if normalized_axis != 0:
        raise AudioShapeError("the sample axis must be the only axis of mono audio")
    return validated


def _validate_sample_rate(sample_rate: int) -> int:
    return validate_integer(sample_rate, "sample_rate", minimum=1)


def _block_geometry(sample_rate: int) -> tuple[int, int]:
    """Return the BS.1770 400 ms block and 100 ms hop in samples."""
    block_size = round(_BLOCK_DURATION_SECONDS * sample_rate)
    hop_size = round(_HOP_DURATION_SECONDS * sample_rate)
    return block_size, hop_size


def _biquad_coefficients(
    b0: float,
    b1: float,
    b2: float,
    a0: float,
    a1: float,
    a2: float,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((b0 / a0, b1 / a0, b2 / a0), dtype=np.float64),
        np.asarray((1.0, a1 / a0, a2 / a0), dtype=np.float64),
    )


def _high_pass(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Design the BS.1770 pre-filter with a bilinear-transform biquad."""
    frequency = 38.1358
    quality = 0.5
    omega = 2.0 * math.pi * frequency / sample_rate
    sine = math.sin(omega)
    cosine = math.cos(omega)
    alpha = sine / (2.0 * quality)
    return _biquad_coefficients(
        (1.0 + cosine) / 2.0,
        -(1.0 + cosine),
        (1.0 + cosine) / 2.0,
        1.0 + alpha,
        -2.0 * cosine,
        1.0 - alpha,
    )


def _high_shelf(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Design the +4 dB, 1.5 kHz BS.1770 high-shelf biquad."""
    frequency = 1500.0
    quality = 1.0 / math.sqrt(2.0)
    gain_db = 4.0
    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * math.pi * frequency / sample_rate
    sine = math.sin(omega)
    cosine = math.cos(omega)
    alpha = sine / (2.0 * quality)
    beta = 2.0 * math.sqrt(amplitude) * alpha
    return _biquad_coefficients(
        amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine + beta),
        -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cosine),
        amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine - beta),
        (amplitude + 1.0) - (amplitude - 1.0) * cosine + beta,
        2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cosine),
        (amplitude + 1.0) - (amplitude - 1.0) * cosine - beta,
    )


def _apply_biquad(signal: np.ndarray, coefficients: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    numerator, denominator = coefficients
    output = np.empty_like(signal, dtype=np.float64)
    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0
    for index, sample in enumerate(signal):
        current = (
            numerator[0] * sample
            + numerator[1] * x1
            + numerator[2] * x2
            - denominator[1] * y1
            - denominator[2] * y2
        )
        output[index] = current
        x2 = x1
        x1 = float(sample)
        y2 = y1
        y1 = current
    return output


def _k_weight(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    weighted = _apply_biquad(signal, _high_pass(sample_rate))
    return _apply_biquad(weighted, _high_shelf(sample_rate))


def _block_mean_squares(weighted: np.ndarray, sample_rate: int) -> np.ndarray:
    block_size, hop_size = _block_geometry(sample_rate)
    if weighted.size < block_size:
        return np.empty(0, dtype=np.float64)
    block_count = 1 + (weighted.size - block_size) // hop_size
    starts = np.arange(block_count, dtype=np.intp) * hop_size
    blocks = np.lib.stride_tricks.sliding_window_view(weighted, block_size)[starts]
    return np.asarray(np.mean(blocks * blocks, axis=1, dtype=np.float64), dtype=np.float64)


def _energy_to_lufs(energy: float) -> float:
    if energy <= 0.0:
        return -math.inf
    return _LUFS_OFFSET + 10.0 * math.log10(energy)


def integrated_loudness(
    audio: np.ndarray,
    *,
    sample_rate: int,
    axis: int = -1,
) -> float:
    """Measure BS.1770-style gated integrated loudness in LUFS.

    V1 accepts finite one-dimensional mono NumPy audio. Complete 400 ms blocks
    are analyzed with a 100 ms hop; audio shorter than one complete block has
    no reportable integrated loudness and returns ``-math.inf`` without padding.
    """
    signal = _validate_mono_audio(audio, axis)
    rate = _validate_sample_rate(sample_rate)
    weighted = _k_weight(np.asarray(signal, dtype=np.float64), rate)
    energies = _block_mean_squares(weighted, rate)
    if energies.size == 0:
        return -math.inf

    absolute_threshold = 10.0 ** ((_ABSOLUTE_GATE_LUFS - _LUFS_OFFSET) / 10.0)
    absolute = energies >= absolute_threshold
    if not np.any(absolute):
        return -math.inf
    absolute_energy = float(np.mean(energies[absolute], dtype=np.float64))
    relative_threshold = 10.0 ** (
        (_energy_to_lufs(absolute_energy) + _RELATIVE_GATE_LU - _LUFS_OFFSET) / 10.0
    )
    gated = absolute & (energies >= relative_threshold)
    if not np.any(gated):
        return -math.inf
    return float(_energy_to_lufs(float(np.mean(energies[gated], dtype=np.float64))))


def sample_peak_dbfs(audio: np.ndarray, *, axis: int = -1) -> float:
    """Return the largest absolute sample level in dBFS."""
    signal = _validate_mono_audio(audio, axis)
    peak = float(np.max(np.abs(np.asarray(signal, dtype=np.float64))))
    if peak == 0.0:
        return -math.inf
    return float(20.0 * math.log10(peak))


def true_peak_dbtp(
    audio: np.ndarray,
    *,
    sample_rate: int,
    axis: int = -1,
    oversample: int = 4,
) -> float:
    """Estimate inter-sample peak level in dBTP by NumPy oversampling.

    ``oversample=1`` measures the discrete sample peak. Values greater than
    one use AudioSig's windowed-sinc resampler and never alter the input.
    """
    signal = _validate_mono_audio(audio, axis)
    rate = _validate_sample_rate(sample_rate)
    factor = validate_integer(oversample, "oversample", minimum=1)
    sample_peak = float(np.max(np.abs(np.asarray(signal, dtype=np.float64))))
    if sample_peak == 0.0:
        return -math.inf
    if factor == 1:
        return float(20.0 * math.log10(sample_peak))
    oversampled = resample(
        np.asarray(signal, dtype=np.float64),
        source_rate=rate,
        target_rate=rate * factor,
        filter_width=32,
    )
    peak = max(sample_peak, float(np.max(np.abs(np.asarray(oversampled, dtype=np.float64)))))
    return float(20.0 * math.log10(peak))


def measure_loudness(
    audio: np.ndarray,
    *,
    sample_rate: int,
    axis: int = -1,
    true_peak_oversample: int = 4,
) -> LoudnessMetrics:
    """Return integrated loudness, sample peak, and true peak measurements."""
    return LoudnessMetrics(
        integrated_lufs=integrated_loudness(audio, sample_rate=sample_rate, axis=axis),
        sample_peak_dbfs=sample_peak_dbfs(audio, axis=axis),
        true_peak_dbtp=true_peak_dbtp(
            audio,
            sample_rate=sample_rate,
            axis=axis,
            oversample=true_peak_oversample,
        ),
    )
