"""Standards-based loudness and peak measurements for NumPy audio."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._resampling import resample
from ._validation import validate_audio, validate_integer
from .exceptions import AudioShapeError, InvalidParameterError

_ABSOLUTE_GATE_LUFS = -70.0
_RELATIVE_GATE_LU = -10.0
_BLOCK_DURATION_SECONDS = 0.400
_HOP_DURATION_SECONDS = 0.100
_LUFS_OFFSET = -0.691


@dataclass(frozen=True, slots=True)
class LoudnessMetrics:
    """Immutable loudness and peak measurements; unrequested metrics are ``None``."""

    integrated_lufs: float | None
    sample_peak_dbfs: float | None
    true_peak_dbtp: float | None


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
    stops = starts + block_size
    prefix = np.empty(weighted.size + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.multiply(weighted, weighted, out=prefix[1:])
    np.cumsum(prefix[1:], dtype=np.float64, out=prefix[1:])
    return (prefix[stops] - prefix[starts]) / block_size


def _energy_to_lufs(energy: float) -> float:
    if energy <= 0.0:
        return -math.inf
    return _LUFS_OFFSET + 10.0 * math.log10(energy)


def _integrated_loudness_from_energies(energies: np.ndarray) -> float:
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


def integrated_loudness(
    audio: np.ndarray,
    *,
    sample_rate: int,
    axis: int = -1,
) -> float:
    """Measure BS.1770-style gated integrated loudness in LUFS.

    Complete 400 ms blocks are analyzed with a 100 ms hop. Audio shorter than
    one complete block has no reportable integrated loudness and returns
    ``-math.inf`` without padding.
    """
    analysis = analyze_loudness(
        audio,
        sample_rate=sample_rate,
        axis=axis,
        integrated=True,
        sample_peak=False,
        true_peak=False,
    )
    assert analysis.integrated_lufs is not None
    return analysis.integrated_lufs


def sample_peak_dbfs(audio: np.ndarray, *, axis: int = -1) -> float:
    """Return the largest absolute sample level in dBFS."""
    signal = _validate_mono_audio(audio, axis)
    peak = float(np.max(np.abs(np.asarray(signal, dtype=np.float64))))
    if peak == 0.0:
        return -math.inf
    return float(20.0 * math.log10(peak))


_TRUE_PEAK_PHASE_COEFFICIENTS = np.asarray(
    (
        (0.001708984375, -0.0291748046875, -0.0189208984375, -0.00830078125),
        (0.010986328125, 0.029296875, 0.0330810546875, 0.014892578125),
        (-0.0196533203125, -0.0517578125, -0.0582275390625, -0.026611328125),
        (0.033203125, 0.089111328125, 0.1015625, 0.047607421875),
        (-0.0594482421875, -0.16650390625, -0.2003173828125, -0.102294921875),
        (0.1373291015625, 0.465087890625, 0.77978515625, 0.97216796875),
        (0.97216796875, 0.77978515625, 0.465087890625, 0.1373291015625),
        (-0.102294921875, -0.2003173828125, -0.16650390625, -0.0594482421875),
        (0.047607421875, 0.1015625, 0.089111328125, 0.033203125),
        (-0.026611328125, -0.0582275390625, -0.0517578125, -0.0196533203125),
        (0.014892578125, 0.0330810546875, 0.029296875, 0.010986328125),
        (-0.00830078125, -0.0189208984375, -0.0291748046875, 0.001708984375),
    ),
    dtype=np.float64,
)
_TRUE_PEAK_CHUNK_SIZE = 8192


def _stream_true_peak(signal: np.ndarray) -> float:
    """Return a 4-phase FIR peak while retaining only one bounded input chunk."""
    coefficients = _TRUE_PEAK_PHASE_COEFFICIENTS
    history_size = coefficients.shape[0] - 1
    history = np.zeros(history_size, dtype=np.float64)
    peak = 0.0

    def consume(chunk: np.ndarray) -> None:
        nonlocal history, peak
        extended = np.concatenate((history, chunk))
        output_start = history.size
        output_stop = output_start + chunk.size
        for phase in range(coefficients.shape[1]):
            interpolated = np.convolve(extended, coefficients[:, phase], mode="full")
            peak = max(peak, float(np.max(np.abs(interpolated[output_start:output_stop]))))
        history = extended[-history_size:]

    for start in range(0, signal.size, _TRUE_PEAK_CHUNK_SIZE):
        consume(signal[start : start + _TRUE_PEAK_CHUNK_SIZE])
    consume(np.zeros(history_size, dtype=np.float64))
    return peak


def true_peak_dbtp(
    audio: np.ndarray,
    *,
    sample_rate: int,
    axis: int = -1,
    oversample: int = 4,
) -> float:
    """Estimate inter-sample peak level in dBTP without materializing 4x audio.

    The standard 4x path uses the 48-tap, 4-phase FIR example from ITU-R
    BS.1770-5 Annex 2 and streams the input in bounded chunks. Other oversample
    factors retain the historical general-purpose resampler behavior.
    """
    signal = _validate_mono_audio(audio, axis)
    rate = _validate_sample_rate(sample_rate)
    factor = validate_integer(oversample, "oversample", minimum=1)
    values = np.asarray(signal, dtype=np.float64)
    sample_peak = float(np.max(np.abs(values)))
    if sample_peak == 0.0:
        return -math.inf
    if factor == 1:
        return float(20.0 * math.log10(sample_peak))
    if factor == 4:
        peak = max(sample_peak, _stream_true_peak(values))
    else:
        oversampled = resample(
            values,
            source_rate=rate,
            target_rate=rate * factor,
            filter_width=32,
        )
        peak = max(sample_peak, float(np.max(np.abs(np.asarray(oversampled, dtype=np.float64)))))
    return float(20.0 * math.log10(peak))


@dataclass(frozen=True, slots=True)
class LoudnessAnalysis:
    """Reusable measurements and block energies for uniform-gain evaluation."""

    sample_rate: int
    block_mean_squares: np.ndarray | None
    integrated_lufs: float | None
    sample_peak_dbfs: float | None
    true_peak_dbtp: float | None

    def __post_init__(self) -> None:
        if self.block_mean_squares is not None:
            self.block_mean_squares.setflags(write=False)

    def metrics(self) -> LoudnessMetrics:
        """Return the metrics measured during analysis; unselected values are None."""
        return LoudnessMetrics(
            integrated_lufs=self.integrated_lufs,
            sample_peak_dbfs=self.sample_peak_dbfs,
            true_peak_dbtp=self.true_peak_dbtp,
        )

    def metrics_after_gain(self, gain_db: float) -> LoudnessMetrics:
        """Derive metrics after uniform gain and re-run BS.1770 gating from cached blocks."""
        if isinstance(gain_db, bool):
            raise InvalidParameterError("gain_db must be a finite real number")
        try:
            gain = float(gain_db)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidParameterError("gain_db must be a finite real number") from exc
        if not math.isfinite(gain):
            raise InvalidParameterError("gain_db must be a finite real number")
        integrated = self.integrated_lufs
        if self.block_mean_squares is not None:
            gain_factor = 10.0 ** (gain / 10.0)
            integrated = _integrated_loudness_from_energies(self.block_mean_squares * gain_factor)

        def shifted(value: float | None) -> float | None:
            if value is None or not math.isfinite(value):
                return value
            return float(value + gain)

        return LoudnessMetrics(
            integrated_lufs=integrated,
            sample_peak_dbfs=shifted(self.sample_peak_dbfs),
            true_peak_dbtp=shifted(self.true_peak_dbtp),
        )


def analyze_loudness(
    audio: np.ndarray,
    *,
    sample_rate: int,
    axis: int = -1,
    integrated: bool = True,
    sample_peak: bool = True,
    true_peak: bool = True,
    true_peak_oversample: int = 4,
) -> LoudnessAnalysis:
    """Analyze selected loudness metrics and retain only reusable block statistics."""
    for name, selected in (
        ("integrated", integrated),
        ("sample_peak", sample_peak),
        ("true_peak", true_peak),
    ):
        if not isinstance(selected, bool):
            raise InvalidParameterError(f"{name} must be a bool")
    signal = _validate_mono_audio(audio, axis)
    rate = _validate_sample_rate(sample_rate)
    energies = None
    integrated_value = None
    if integrated:
        weighted = _k_weight(np.asarray(signal, dtype=np.float64), rate)
        energies = _block_mean_squares(weighted, rate)
        integrated_value = _integrated_loudness_from_energies(energies)
    values = np.asarray(signal, dtype=np.float64)
    sample_peak_value = None
    if sample_peak:
        peak = float(np.max(np.abs(values)))
        sample_peak_value = -math.inf if peak == 0.0 else float(20.0 * math.log10(peak))
    true_peak_value = None
    if true_peak:
        true_peak_value = true_peak_dbtp(
            signal,
            sample_rate=rate,
            axis=axis,
            oversample=true_peak_oversample,
        )
    return LoudnessAnalysis(
        sample_rate=rate,
        block_mean_squares=energies,
        integrated_lufs=integrated_value,
        sample_peak_dbfs=sample_peak_value,
        true_peak_dbtp=true_peak_value,
    )


def measure_loudness(
    audio: np.ndarray,
    *,
    sample_rate: int,
    axis: int = -1,
    true_peak_oversample: int = 4,
) -> LoudnessMetrics:
    """Return integrated loudness, sample peak, and true peak measurements."""
    return analyze_loudness(
        audio,
        sample_rate=sample_rate,
        axis=axis,
        true_peak_oversample=true_peak_oversample,
    ).metrics()
