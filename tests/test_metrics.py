from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import numpy as np
import pytest

from audiosig import AudioShapeError, InvalidParameterError
from audiosig.metrics import (
    MetricResult,
    log_f0_rmse,
    log_mel_l1,
    log_spectral_distance,
    mel_cepstral_distortion,
    mel_spectral_distance,
    sdr,
    si_sdr,
    snr,
    voiced_unvoiced_error,
)
from audiosig.spectral import stft


def _tone(frequency: float, seconds: float = 0.5, sample_rate: int = 16_000) -> np.ndarray:
    time = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * frequency * time)


def test_metric_result_and_waveform_metric_formulas() -> None:
    reference = np.array([1.0, 1.0], dtype=np.float64)
    estimate = np.array([0.9, 1.1], dtype=np.float64)
    result = snr(reference, estimate, sample_rate=24_000)
    assert isinstance(result, MetricResult)
    assert result.name == "snr"
    assert result.unit == "dB"
    assert result.higher_is_better
    assert result.value == pytest.approx(10.0 * np.log10(2.0 / 0.02))
    assert result.parameters["sample_rate"] == 24_000
    assert result.parameters["estimate_sample_rate"] == 24_000
    assert result.parameters["alignment"] == "none"
    assert result.diagnostics["samples"] == 2
    assert result.diagnostics["channels_or_lanes"] == 1

    unscaled = sdr(reference, estimate, sample_rate=24_000)
    assert unscaled.name == "sdr"
    assert unscaled.value == result.value
    assert unscaled.parameters["estimator"] == "unscaled_reference_residual"
    assert unscaled.parameters["gain_fit"] is False
    with pytest.raises(FrozenInstanceError):
        field_name = "name"
        setattr(result, field_name, "changed")


def test_waveform_metrics_infinities_gain_and_zero_estimate() -> None:
    reference = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    exact = snr(reference, reference.copy(), sample_rate=16_000)
    assert exact.value == float("inf")
    assert sdr(reference, reference, sample_rate=16_000).value == float("inf")
    assert si_sdr(reference, reference, sample_rate=16_000).value == float("inf")

    scaled = reference * 2.0
    scaled_snr = snr(reference, scaled, sample_rate=16_000)
    scaled_sdr = sdr(reference, scaled, sample_rate=16_000)
    scaled_si_sdr = si_sdr(reference, scaled, sample_rate=16_000)
    assert scaled_snr.value == pytest.approx(0.0)
    assert scaled_sdr.value == scaled_snr.value
    assert scaled_snr.value is not None
    assert scaled_sdr.value is not None
    assert scaled_si_sdr.value is not None
    assert scaled_si_sdr.value == float("inf") or scaled_si_sdr.value > 100.0

    zero_estimate = snr(reference, np.zeros_like(reference), sample_rate=16_000)
    assert zero_estimate.value == pytest.approx(0.0)


def test_si_sdr_zero_mean_policy_and_silent_reference() -> None:
    reference = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    estimate = reference + 4.0
    without_mean = si_sdr(reference, estimate, sample_rate=16_000, zero_mean=False)
    with_mean = si_sdr(reference, estimate, sample_rate=16_000, zero_mean=True)
    assert without_mean.value is not None
    assert with_mean.value is not None
    assert with_mean.value == float("inf") or with_mean.value > 100.0
    assert without_mean.value < with_mean.value
    assert with_mean.parameters["zero_mean"] is True

    with pytest.raises(InvalidParameterError, match="silent/zero-energy"):
        snr(np.zeros(4), np.ones(4), sample_rate=16_000)
    with pytest.raises(InvalidParameterError, match="zero-energy reference"):
        si_sdr(np.ones(4), np.ones(4), sample_rate=16_000, zero_mean=True)


def test_pairwise_metrics_enforce_rate_shape_dtype_and_alignment() -> None:
    reference = np.ones((2, 32), dtype=np.float32)
    estimate = reference.copy()
    assert snr(reference, estimate, sample_rate=16_000, axis=1).value == float("inf")
    with pytest.raises(InvalidParameterError, match="estimate_sample_rate"):
        snr(reference, estimate, sample_rate=16_000, estimate_sample_rate=8_000)
    with pytest.raises(AudioShapeError):
        snr(reference, np.ones((2, 31), dtype=np.float32), sample_rate=16_000)
    with pytest.raises(AudioShapeError):
        snr(reference, np.ones((3, 32), dtype=np.float32), sample_rate=16_000)
    with pytest.raises(AudioShapeError):
        snr(reference, np.ones((2, 32), dtype=np.int32), sample_rate=16_000)
    with pytest.raises(AudioShapeError):
        snr(np.array([np.nan]), np.array([0.0]), sample_rate=16_000)
    with pytest.raises(AudioShapeError):
        snr(reference, estimate, sample_rate=16_000, axis=2)
    with pytest.raises(InvalidParameterError, match="alignment"):
        snr(reference, estimate, sample_rate=16_000, alignment=cast(Any, "shift"))


def test_waveform_metrics_do_not_mutate_inputs_and_aggregate_lanes() -> None:
    reference = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    estimate = np.array([[0.9, 1.1], [0.9, 1.1]], dtype=np.float32)
    ref_before = reference.copy()
    est_before = estimate.copy()
    result = snr(reference, estimate, sample_rate=16_000)
    assert result.diagnostics["channels_or_lanes"] == 2
    assert result.value == pytest.approx(20.0)
    np.testing.assert_array_equal(reference, ref_before)
    np.testing.assert_array_equal(estimate, est_before)


def test_spectral_distances_are_zero_for_identical_and_respect_gain() -> None:
    source = _tone(440.0)
    identical = source.copy()
    for metric in (
        log_spectral_distance,
        mel_spectral_distance,
        log_mel_l1,
        mel_cepstral_distortion,
    ):
        result = metric(source, identical, sample_rate=16_000)
        assert result.value == pytest.approx(0.0, abs=1e-12), metric.__name__
        assert result.value is not None
        assert isinstance(result.diagnostics["frames"], int) and result.diagnostics["frames"] > 0
        assert result.diagnostics["channels_or_lanes"] == 1
        assert result.parameters["sample_rate"] == 16_000

    silence = np.zeros(2_000, dtype=np.float64)
    assert log_spectral_distance(silence, silence, sample_rate=16_000).value == pytest.approx(0.0)
    assert mel_spectral_distance(silence, silence, sample_rate=16_000).value == pytest.approx(0.0)
    assert (log_spectral_distance(source, source * 2.0, sample_rate=16_000).value or 0.0) > 0.0
    assert (mel_spectral_distance(source, source * 2.0, sample_rate=16_000).value or 0.0) > 0.0


def test_spectral_distance_formulas_and_metadata() -> None:
    reference = _tone(330.0)
    estimate = reference * 0.75
    n_fft, hop, floor = 128, 32, -80.0
    ref_spectrum = np.abs(stft(reference, n_fft=n_fft, hop_length=hop))
    est_spectrum = np.abs(stft(estimate, n_fft=n_fft, hop_length=hop))
    ref_db = np.maximum(20.0 * np.log10(np.maximum(ref_spectrum, 1e-300)), floor)
    est_db = np.maximum(20.0 * np.log10(np.maximum(est_spectrum, 1e-300)), floor)
    expected_lsd = float(np.mean(np.sqrt(np.mean((ref_db - est_db) ** 2, axis=-2))))
    result = log_spectral_distance(
        reference,
        estimate,
        sample_rate=16_000,
        n_fft=n_fft,
        hop_length=hop,
        db_floor=floor,
    )
    assert result.value == pytest.approx(expected_lsd)
    assert result.parameters["db_floor"] == floor
    assert result.parameters["n_fft"] == n_fft
    assert result.parameters["window"] == "hann_symmetric"


def test_log_mel_l1_and_mcd_formulas_and_c0_option() -> None:
    source = _tone(220.0)
    scaled = source * 1.5
    l1 = log_mel_l1(source, scaled, sample_rate=16_000, n_fft=128, hop_length=32, n_mels=12)
    assert l1.value is not None and l1.value > 0.0
    assert l1.unit == "ln-power"
    assert l1.parameters["mel_scale"] == "slaney"
    assert l1.parameters["mel_norm"] == "slaney"

    base = mel_cepstral_distortion(
        source, scaled, sample_rate=16_000, n_fft=128, hop_length=32, n_mfcc=8, n_mels=12
    )
    with_c0 = mel_cepstral_distortion(
        source,
        scaled,
        sample_rate=16_000,
        n_fft=128,
        hop_length=32,
        n_mfcc=8,
        n_mels=12,
        include_c0=True,
    )
    assert base.parameters["coefficient_0_included"] is False
    assert with_c0.parameters["coefficient_0_included"] is True
    assert base.parameters["dct_type"] == 2
    assert base.parameters["dct_norm"] == "ortho"
    assert base.parameters["mel_scale"] == "slaney"
    assert with_c0.value != pytest.approx(base.value)

    identical = mel_cepstral_distortion(
        source, source, sample_rate=16_000, n_fft=128, hop_length=32, n_mfcc=8, n_mels=12
    )
    assert identical.value == pytest.approx(0.0)
    with pytest.raises(InvalidParameterError):
        mel_cepstral_distortion(source, scaled, sample_rate=16_000, n_mfcc=1, include_c0=False)


def test_spectral_metrics_reject_unequal_duration_and_bad_floors() -> None:
    reference = np.ones(1_000, dtype=np.float32)
    with pytest.raises(AudioShapeError):
        log_spectral_distance(reference, reference[:-1], sample_rate=16_000)
    with pytest.raises(InvalidParameterError):
        log_spectral_distance(reference, reference, sample_rate=16_000, db_floor=0.0)
    with pytest.raises(InvalidParameterError):
        log_mel_l1(reference, reference, sample_rate=16_000, log_floor=0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spectral_metric_dtype_stability(dtype: type[np.floating]) -> None:
    source = _tone(440.0).astype(dtype)
    estimate = (source * 0.95).astype(dtype)
    values = [
        log_spectral_distance(source, estimate, sample_rate=16_000).value,
        mel_spectral_distance(source, estimate, sample_rate=16_000).value,
        log_mel_l1(source, estimate, sample_rate=16_000).value,
    ]
    assert all(value is not None and np.isfinite(value) for value in values)


def test_pitch_metrics_joint_voicing_and_error_denominators() -> None:
    reference = _tone(200.0)
    same = _tone(200.0)
    f0_result = log_f0_rmse(reference, same, sample_rate=16_000)
    voiced_result = voiced_unvoiced_error(reference, same, sample_rate=16_000)
    assert f0_result.value == pytest.approx(0.0, abs=1e-12)
    assert f0_result.unit == "ln-Hz"
    assert f0_result.diagnostics["joint_voiced_frames"] == f0_result.diagnostics["total_frames"]
    assert voiced_result.value == 0.0
    assert voiced_result.unit == "fraction"
    assert voiced_result.diagnostics["mismatch_frames"] == 0

    shifted = log_f0_rmse(reference, _tone(220.0), sample_rate=16_000)
    assert shifted.value == pytest.approx(abs(np.log(200.0) - np.log(220.0)), abs=0.03)

    unvoiced = voiced_unvoiced_error(reference, np.zeros_like(reference), sample_rate=16_000)
    assert unvoiced.value is not None and unvoiced.value > 0.5
    no_joint = log_f0_rmse(reference, np.zeros_like(reference), sample_rate=16_000)
    assert no_joint.value is None
    assert no_joint.diagnostics["joint_voiced_frames"] == 0


def test_pitch_metrics_are_mono_and_do_not_align_or_resample() -> None:
    source = _tone(220.0)
    with pytest.raises(AudioShapeError, match="mono"):
        log_f0_rmse(np.stack([source, source]), np.stack([source, source]), sample_rate=16_000)
    with pytest.raises(AudioShapeError):
        voiced_unvoiced_error(source, source[:-1], sample_rate=16_000)
    with pytest.raises(InvalidParameterError, match="estimate_sample_rate"):
        log_f0_rmse(source, source, sample_rate=16_000, estimate_sample_rate=8_000)
