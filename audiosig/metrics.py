"""Deterministic NumPy reference metrics for equal-rate audio arrays."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal, TypeAlias

import numpy as np

from ._validation import (
    validate_audio,
    validate_boolean,
    validate_finite,
    validate_integer,
    validate_positive,
)
from .exceptions import AudioShapeError, InvalidParameterError
from .pitch import PitchTrack, pitch_track
from .spectral import match_frames, mel_spectrogram, mfcc, stft

MetricScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class MetricResult:
    """JSON-friendly scalar metric result and reproducibility metadata."""

    name: str
    value: float | None
    unit: str
    higher_is_better: bool
    parameters: dict[str, MetricScalar]
    diagnostics: dict[str, MetricScalar]


def _validate_alignment(alignment: str) -> str:
    if alignment != "none":
        raise InvalidParameterError("alignment must be 'none'")
    return alignment


def _validate_sample_rates(sample_rate: int, estimate_sample_rate: int | None) -> tuple[int, int]:
    rate = validate_integer(sample_rate, "sample_rate")
    estimate_rate = (
        rate
        if estimate_sample_rate is None
        else validate_integer(estimate_sample_rate, "estimate_sample_rate")
    )
    if estimate_rate != rate:
        raise InvalidParameterError(
            "estimate_sample_rate must equal sample_rate; resample explicitly"
        )
    return rate, estimate_rate


def _validate_pair(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None,
    axis: int,
    alignment: str,
    mono: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    rate, estimate_rate = _validate_sample_rates(sample_rate, estimate_sample_rate)
    _validate_alignment(alignment)
    ref, ref_axis = validate_audio(reference, axis=axis)
    est, est_axis = validate_audio(estimate, axis=axis)
    if ref.ndim != est.ndim:
        raise AudioShapeError("reference and estimate must have equal dimensionality")
    if ref_axis != est_axis:
        raise AudioShapeError("reference and estimate must have the same sample axis")
    ref = np.moveaxis(ref, ref_axis, -1)
    est = np.moveaxis(est, est_axis, -1)
    if ref.shape != est.shape:
        raise AudioShapeError("reference and estimate must have identical shapes and sample counts")
    if mono and ref.ndim != 1:
        raise AudioShapeError("pitch metrics require mono audio shaped (samples,)")
    lanes = prod(ref.shape[:-1]) if ref.ndim > 1 else 1
    return ref, est, rate, estimate_rate, lanes


def _metric_parameters(
    *, sample_rate: int, estimate_sample_rate: int, alignment: str, axis: int | None = None
) -> dict[str, MetricScalar]:
    parameters: dict[str, MetricScalar] = {
        "sample_rate": sample_rate,
        "estimate_sample_rate": estimate_sample_rate,
        "alignment": alignment,
    }
    if axis is not None:
        parameters["axis"] = int(axis)
    return parameters


def _result(
    name: str,
    value: float | None,
    unit: str,
    higher_is_better: bool,
    parameters: dict[str, MetricScalar],
    diagnostics: dict[str, MetricScalar],
) -> MetricResult:
    return MetricResult(
        name=name,
        value=None if value is None else float(value),
        unit=unit,
        higher_is_better=higher_is_better,
        parameters=dict(parameters),
        diagnostics=dict(diagnostics),
    )


def _energies(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, float, float]:
    ref64 = reference.astype(np.float64, copy=False)
    est64 = estimate.astype(np.float64, copy=False)
    residual = ref64 - est64
    reference_energy = float(np.sum(np.square(ref64), dtype=np.float64))
    residual_energy = float(np.sum(np.square(residual), dtype=np.float64))
    return reference_energy, residual_energy, float(np.sum(np.square(est64), dtype=np.float64))


def _waveform_metric(
    name: str,
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None,
    axis: int,
    alignment: Literal["none"],
    eps: float,
    estimator: str | None = None,
) -> MetricResult:
    ref, est, rate, estimate_rate, lanes = _validate_pair(
        reference,
        estimate,
        sample_rate=sample_rate,
        estimate_sample_rate=estimate_sample_rate,
        axis=axis,
        alignment=alignment,
    )
    epsilon = validate_positive(eps, "eps")
    reference_energy, residual_energy, _ = _energies(ref, est)
    if reference_energy <= epsilon:
        raise InvalidParameterError(f"{name.upper()} undefined for silent/zero-energy reference")
    value = (
        float("inf")
        if residual_energy == 0.0
        else 10.0 * float(np.log10(reference_energy / residual_energy))
    )
    parameters = _metric_parameters(
        sample_rate=rate, estimate_sample_rate=estimate_rate, alignment=alignment, axis=axis
    )
    parameters["eps"] = epsilon
    if estimator is not None:
        parameters["estimator"] = estimator
        parameters["gain_fit"] = False
    diagnostics: dict[str, MetricScalar] = {
        "samples": ref.shape[-1],
        "channels_or_lanes": lanes,
        "reference_energy": reference_energy,
        "residual_energy": residual_energy,
    }
    return _result(name, value, "dB", True, parameters, diagnostics)


def snr(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None = None,
    axis: int = -1,
    alignment: Literal["none"] = "none",
    eps: float = 1e-12,
) -> MetricResult:
    """Compute global signal-to-noise ratio without gain or alignment fitting."""
    return _waveform_metric(
        "snr",
        reference,
        estimate,
        sample_rate=sample_rate,
        estimate_sample_rate=estimate_sample_rate,
        axis=axis,
        alignment=alignment,
        eps=eps,
    )


def sdr(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None = None,
    axis: int = -1,
    alignment: Literal["none"] = "none",
    eps: float = 1e-12,
) -> MetricResult:
    """Compute the explicitly unscaled reference-residual SDR (equal to SNR)."""
    return _waveform_metric(
        "sdr",
        reference,
        estimate,
        sample_rate=sample_rate,
        estimate_sample_rate=estimate_sample_rate,
        axis=axis,
        alignment=alignment,
        eps=eps,
        estimator="unscaled_reference_residual",
    )


def si_sdr(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None = None,
    axis: int = -1,
    alignment: Literal["none"] = "none",
    zero_mean: bool = True,
    eps: float = 1e-12,
) -> MetricResult:
    """Compute scale-invariant SDR, optionally removing each lane's mean."""
    ref, est, rate, estimate_rate, lanes = _validate_pair(
        reference,
        estimate,
        sample_rate=sample_rate,
        estimate_sample_rate=estimate_sample_rate,
        axis=axis,
        alignment=alignment,
    )
    remove_mean = validate_boolean(zero_mean, "zero_mean")
    epsilon = validate_positive(eps, "eps")
    ref64 = ref.astype(np.float64, copy=False)
    est64 = est.astype(np.float64, copy=False)
    if remove_mean:
        ref64 = ref64 - np.mean(ref64, axis=-1, keepdims=True, dtype=np.float64)
        est64 = est64 - np.mean(est64, axis=-1, keepdims=True, dtype=np.float64)
    ref_flat = ref64.reshape(-1)
    est_flat = est64.reshape(-1)
    reference_energy = float(np.dot(ref_flat, ref_flat))
    if reference_energy <= epsilon:
        raise InvalidParameterError("SI-SDR undefined for zero-energy reference after mean removal")
    alpha = float(np.dot(est_flat, ref_flat) / reference_energy)
    target = alpha * ref_flat
    residual = est_flat - target
    target_energy = float(np.dot(target, target))
    residual_energy = float(np.dot(residual, residual))
    if target_energy <= epsilon:
        value = float("-inf")
    elif residual_energy == 0.0:
        value = float("inf")
    else:
        value = 10.0 * float(np.log10(target_energy / residual_energy))
    parameters = _metric_parameters(
        sample_rate=rate, estimate_sample_rate=estimate_rate, alignment=alignment, axis=axis
    )
    parameters.update({"zero_mean": remove_mean, "eps": epsilon})
    diagnostics: dict[str, MetricScalar] = {
        "samples": ref.shape[-1],
        "channels_or_lanes": lanes,
        "reference_energy": reference_energy,
        "target_energy": target_energy,
        "residual_energy": residual_energy,
    }
    return _result("si_sdr", value, "dB", True, parameters, diagnostics)


def _db_values(values: np.ndarray, floor_db: float, *, magnitude: bool) -> np.ndarray:
    multiplier = 20.0 if magnitude else 10.0
    positive = values > 0.0
    safe_values = np.where(positive, values, 1.0)
    db_values = multiplier * np.log10(safe_values)
    return np.where(positive, np.maximum(db_values, floor_db), floor_db)


def _spectral_pair(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None,
    axis: int,
    alignment: Literal["none"],
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    return _validate_pair(
        reference,
        estimate,
        sample_rate=sample_rate,
        estimate_sample_rate=estimate_sample_rate,
        axis=axis,
        alignment=alignment,
    )


def log_spectral_distance(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None = None,
    axis: int = -1,
    alignment: Literal["none"] = "none",
    n_fft: int = 1024,
    hop_length: int = 256,
    center: bool = True,
    db_floor: float = -120.0,
) -> MetricResult:
    """Compute mean per-frame RMS absolute dB-magnitude spectral distance."""
    ref, est, rate, estimate_rate, lanes = _spectral_pair(
        reference,
        estimate,
        sample_rate=sample_rate,
        estimate_sample_rate=estimate_sample_rate,
        axis=axis,
        alignment=alignment,
    )
    floor = validate_finite(db_floor, "db_floor")
    if floor >= 0.0:
        raise InvalidParameterError("db_floor must be negative")
    ref_spec = stft(ref, n_fft=n_fft, hop_length=hop_length, center=center)
    est_spec = stft(est, n_fft=n_fft, hop_length=hop_length, center=center)
    ref_db = _db_values(np.abs(ref_spec), floor, magnitude=True)
    est_db = _db_values(np.abs(est_spec), floor, magnitude=True)
    per_frame = np.sqrt(np.mean(np.square(ref_db - est_db), axis=-2, dtype=np.float64))
    value = float(np.mean(per_frame, dtype=np.float64))
    parameters = _metric_parameters(
        sample_rate=rate, estimate_sample_rate=estimate_rate, alignment=alignment, axis=axis
    )
    parameters.update(
        {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "center": center,
            "window": "hann_symmetric",
            "db_floor": floor,
        }
    )
    diagnostics: dict[str, MetricScalar] = {
        "samples": ref.shape[-1],
        "channels_or_lanes": lanes,
        "frames": ref_spec.shape[-1],
    }
    return _result("log_spectral_distance", value, "dB", False, parameters, diagnostics)


def mel_spectral_distance(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None = None,
    axis: int = -1,
    alignment: Literal["none"] = "none",
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    center: bool = True,
    f_min: float = 0.0,
    f_max: float | None = None,
    db_floor: float = -120.0,
) -> MetricResult:
    """Compute mean per-frame RMS mel-power dB distance."""
    ref, est, rate, estimate_rate, lanes = _spectral_pair(
        reference,
        estimate,
        sample_rate=sample_rate,
        estimate_sample_rate=estimate_sample_rate,
        axis=axis,
        alignment=alignment,
    )
    floor = validate_finite(db_floor, "db_floor")
    if floor >= 0.0:
        raise InvalidParameterError("db_floor must be negative")
    ref_mel = mel_spectrogram(
        ref,
        sample_rate=rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=center,
        f_min=f_min,
        f_max=f_max,
    )
    est_mel = mel_spectrogram(
        est,
        sample_rate=rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=center,
        f_min=f_min,
        f_max=f_max,
    )
    ref_mel, est_mel = match_frames(ref_mel, est_mel, frame_axis=-1)
    ref_db = _db_values(ref_mel, floor, magnitude=False)
    est_db = _db_values(est_mel, floor, magnitude=False)
    per_frame = np.sqrt(np.mean(np.square(ref_db - est_db), axis=-2, dtype=np.float64))
    value = float(np.mean(per_frame, dtype=np.float64))
    parameters = _metric_parameters(
        sample_rate=rate, estimate_sample_rate=estimate_rate, alignment=alignment, axis=axis
    )
    parameters.update(
        {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
            "center": center,
            "f_min": f_min,
            "f_max": rate / 2.0 if f_max is None else f_max,
            "db_floor": floor,
            "window": "hann_symmetric",
            "mel_scale": "slaney",
            "mel_norm": "slaney",
        }
    )
    diagnostics: dict[str, MetricScalar] = {
        "samples": ref.shape[-1],
        "channels_or_lanes": lanes,
        "frames": ref_mel.shape[-1],
    }
    return _result("mel_spectral_distance", value, "dB", False, parameters, diagnostics)


def log_mel_l1(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None = None,
    axis: int = -1,
    alignment: Literal["none"] = "none",
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    center: bool = True,
    f_min: float = 0.0,
    f_max: float | None = None,
    log_floor: float = 1e-10,
) -> MetricResult:
    """Compute global mean absolute natural-log mel-power difference."""
    ref, est, rate, estimate_rate, lanes = _spectral_pair(
        reference,
        estimate,
        sample_rate=sample_rate,
        estimate_sample_rate=estimate_sample_rate,
        axis=axis,
        alignment=alignment,
    )
    floor = validate_positive(log_floor, "log_floor")
    ref_mel = mel_spectrogram(
        ref,
        sample_rate=rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=center,
        f_min=f_min,
        f_max=f_max,
    )
    est_mel = mel_spectrogram(
        est,
        sample_rate=rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=center,
        f_min=f_min,
        f_max=f_max,
    )
    ref_mel, est_mel = match_frames(ref_mel, est_mel, frame_axis=-1)
    ref_log = np.log(np.maximum(ref_mel, floor))
    est_log = np.log(np.maximum(est_mel, floor))
    value = float(np.mean(np.abs(ref_log - est_log), dtype=np.float64))
    parameters = _metric_parameters(
        sample_rate=rate, estimate_sample_rate=estimate_rate, alignment=alignment, axis=axis
    )
    parameters.update(
        {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
            "center": center,
            "f_min": f_min,
            "f_max": rate / 2.0 if f_max is None else f_max,
            "log_floor": floor,
            "window": "hann_symmetric",
            "mel_scale": "slaney",
            "mel_norm": "slaney",
        }
    )
    diagnostics: dict[str, MetricScalar] = {
        "samples": ref.shape[-1],
        "channels_or_lanes": lanes,
        "frames": ref_mel.shape[-1],
    }
    return _result("log_mel_l1", value, "ln-power", False, parameters, diagnostics)


def mel_cepstral_distortion(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None = None,
    axis: int = -1,
    alignment: Literal["none"] = "none",
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mfcc: int = 13,
    n_mels: int = 40,
    center: bool = True,
    f_min: float = 0.0,
    f_max: float | None = None,
    log_floor: float = 1e-10,
    include_c0: bool = False,
) -> MetricResult:
    """Compute MCD using natural-log orthonormal-DCT-II MFCCs and no DTW."""
    ref, est, rate, estimate_rate, lanes = _spectral_pair(
        reference,
        estimate,
        sample_rate=sample_rate,
        estimate_sample_rate=estimate_sample_rate,
        axis=axis,
        alignment=alignment,
    )
    include_zero = validate_boolean(include_c0, "include_c0")
    if not include_zero and n_mfcc <= 1:
        raise InvalidParameterError(
            "n_mfcc must be greater than one when coefficient 0 is excluded"
        )
    ref_coefficients = mfcc(
        ref,
        sample_rate=rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        center=center,
        f_min=f_min,
        f_max=f_max,
        log_floor=log_floor,
    )
    est_coefficients = mfcc(
        est,
        sample_rate=rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        center=center,
        f_min=f_min,
        f_max=f_max,
        log_floor=log_floor,
    )
    ref_coefficients, est_coefficients = match_frames(
        ref_coefficients, est_coefficients, frame_axis=-1
    )
    start = 0 if include_zero else 1
    difference = ref_coefficients[..., start:, :] - est_coefficients[..., start:, :]
    frame_distances = (10.0 * np.sqrt(2.0) / np.log(10.0)) * np.sqrt(
        np.sum(np.square(difference), axis=-2, dtype=np.float64)
    )
    value = float(np.mean(frame_distances, dtype=np.float64))
    parameters = _metric_parameters(
        sample_rate=rate, estimate_sample_rate=estimate_rate, alignment=alignment, axis=axis
    )
    parameters.update(
        {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mfcc": n_mfcc,
            "n_mels": n_mels,
            "center": center,
            "f_min": f_min,
            "f_max": rate / 2.0 if f_max is None else f_max,
            "log_floor": log_floor,
            "include_c0": include_zero,
            "coefficient_0_included": include_zero,
            "window": "hann_symmetric",
            "mel_scale": "slaney",
            "mel_norm": "slaney",
            "dct_type": 2,
            "dct_norm": "ortho",
        }
    )
    diagnostics: dict[str, MetricScalar] = {
        "samples": ref.shape[-1],
        "channels_or_lanes": lanes,
        "frames": ref_coefficients.shape[-1],
    }
    return _result("mel_cepstral_distortion", value, "dB", False, parameters, diagnostics)


def _pitch_pair(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None,
    alignment: Literal["none"],
    pitch_hop_length: int | None,
    f0_min: float,
    f0_max: float,
) -> tuple[np.ndarray, np.ndarray, int, int, PitchTrack, PitchTrack, int | None, float, float]:
    ref, est, rate, estimate_rate, _ = _validate_pair(
        reference,
        estimate,
        sample_rate=sample_rate,
        estimate_sample_rate=estimate_sample_rate,
        axis=0,
        alignment=alignment,
        mono=True,
    )
    hop = (
        None if pitch_hop_length is None else validate_integer(pitch_hop_length, "pitch_hop_length")
    )
    floor = validate_positive(f0_min, "f0_min")
    ceiling = validate_positive(f0_max, "f0_max")
    ref_track = pitch_track(ref, sample_rate=rate, hop_length=hop, f0_min=floor, f0_max=ceiling)
    est_track = pitch_track(est, sample_rate=rate, hop_length=hop, f0_min=floor, f0_max=ceiling)
    if ref_track.f0_hz.shape != est_track.f0_hz.shape:
        raise AudioShapeError("equal-length inputs must produce identical pitch frame counts")
    return ref, est, rate, estimate_rate, ref_track, est_track, hop, floor, ceiling


def log_f0_rmse(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None = None,
    alignment: Literal["none"] = "none",
    pitch_hop_length: int | None = None,
    f0_min: float = 60.0,
    f0_max: float = 500.0,
) -> MetricResult:
    """Compute natural-log F0 RMSE over jointly voiced frames; undefined if none."""
    _, _, rate, estimate_rate, ref_track, est_track, hop, floor, ceiling = _pitch_pair(
        reference,
        estimate,
        sample_rate=sample_rate,
        estimate_sample_rate=estimate_sample_rate,
        alignment=alignment,
        pitch_hop_length=pitch_hop_length,
        f0_min=f0_min,
        f0_max=f0_max,
    )
    joint = ref_track.voiced & est_track.voiced
    joint_count = int(np.count_nonzero(joint))
    value = (
        None
        if joint_count == 0
        else float(
            np.sqrt(
                np.mean(np.square(np.log(ref_track.f0_hz[joint]) - np.log(est_track.f0_hz[joint])))
            )
        )
    )
    parameters = _metric_parameters(
        sample_rate=rate, estimate_sample_rate=estimate_rate, alignment=alignment
    )
    parameters.update({"pitch_hop_length": hop, "f0_min": floor, "f0_max": ceiling})
    diagnostics: dict[str, MetricScalar] = {
        "reference_voiced_frames": int(np.count_nonzero(ref_track.voiced)),
        "estimate_voiced_frames": int(np.count_nonzero(est_track.voiced)),
        "joint_voiced_frames": joint_count,
        "total_frames": int(ref_track.f0_hz.size),
    }
    return _result("log_f0_rmse", value, "ln-Hz", False, parameters, diagnostics)


def voiced_unvoiced_error(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int,
    estimate_sample_rate: int | None = None,
    alignment: Literal["none"] = "none",
    pitch_hop_length: int | None = None,
    f0_min: float = 60.0,
    f0_max: float = 500.0,
) -> MetricResult:
    """Compute the fraction of pitch frames with different voicing decisions."""
    _, _, rate, estimate_rate, ref_track, est_track, hop, floor, ceiling = _pitch_pair(
        reference,
        estimate,
        sample_rate=sample_rate,
        estimate_sample_rate=estimate_sample_rate,
        alignment=alignment,
        pitch_hop_length=pitch_hop_length,
        f0_min=f0_min,
        f0_max=f0_max,
    )
    mismatch = ref_track.voiced ^ est_track.voiced
    total_frames = int(ref_track.voiced.size)
    mismatch_frames = int(np.count_nonzero(mismatch))
    value = None if total_frames == 0 else mismatch_frames / total_frames
    parameters = _metric_parameters(
        sample_rate=rate, estimate_sample_rate=estimate_rate, alignment=alignment
    )
    parameters.update({"pitch_hop_length": hop, "f0_min": floor, "f0_max": ceiling})
    diagnostics: dict[str, MetricScalar] = {
        "mismatch_frames": mismatch_frames,
        "total_frames": total_frames,
        "reference_voiced_frames": int(np.count_nonzero(ref_track.voiced)),
        "estimate_voiced_frames": int(np.count_nonzero(est_track.voiced)),
    }
    return _result("voiced_unvoiced_error", value, "fraction", False, parameters, diagnostics)
