from __future__ import annotations

import time

import numpy as np
import pytest

from audiosig.metrics import log_spectral_distance, mel_cepstral_distortion, si_sdr, snr
from audiosig.spectral import mfcc


@pytest.mark.quality
def test_delays_and_phase_changes_are_not_implicitly_corrected() -> None:
    sample_rate = 16_000
    time_axis = np.arange(sample_rate // 2, dtype=np.float64) / sample_rate
    reference = np.sin(2.0 * np.pi * 220.0 * time_axis)
    delayed = np.roll(reference, 12)
    phase_changed = np.sin(2.0 * np.pi * 220.0 * time_axis + np.pi / 3.0)

    delayed_snr = snr(reference, delayed, sample_rate=sample_rate)
    phase_snr = snr(reference, phase_changed, sample_rate=sample_rate)
    assert delayed_snr.value is not None and np.isfinite(delayed_snr.value)
    assert phase_snr.value is not None and np.isfinite(phase_snr.value)
    assert delayed_snr.value < float("inf")
    assert phase_snr.value < float("inf")
    assert si_sdr(reference, delayed, sample_rate=sample_rate).value != float("inf")


@pytest.mark.quality
def test_stereo_lanes_aggregate_without_downmixing() -> None:
    sample_rate = 16_000
    time_axis = np.arange(sample_rate // 4, dtype=np.float64) / sample_rate
    left = np.sin(2.0 * np.pi * 200.0 * time_axis)
    right = np.sin(2.0 * np.pi * 400.0 * time_axis)
    reference = np.stack([left, right]).astype(np.float32)
    estimate = np.stack([left, right * 0.5]).astype(np.float32)
    result = log_spectral_distance(reference, estimate, sample_rate=sample_rate)
    assert result.value is not None and result.value > 0.0
    assert result.diagnostics["channels_or_lanes"] == 2
    assert result.diagnostics["samples"] == reference.shape[-1]


@pytest.mark.quality
def test_mcd_uses_documented_constant_and_excludes_c0_by_default() -> None:
    sample_rate = 16_000
    time_axis = np.arange(sample_rate // 5, dtype=np.float64) / sample_rate
    reference = np.sin(2.0 * np.pi * 180.0 * time_axis)
    estimate = 0.8 * np.sin(2.0 * np.pi * 180.0 * time_axis + 0.04)
    ref_mfcc = mfcc(
        reference, sample_rate=sample_rate, n_fft=256, hop_length=64, n_mfcc=12, n_mels=24
    )
    est_mfcc = mfcc(
        estimate, sample_rate=sample_rate, n_fft=256, hop_length=64, n_mfcc=12, n_mels=24
    )
    delta = ref_mfcc[1:] - est_mfcc[1:]
    expected = float(
        np.mean((10.0 * np.sqrt(2.0) / np.log(10.0)) * np.sqrt(np.sum(delta**2, axis=0)))
    )
    actual = mel_cepstral_distortion(
        reference,
        estimate,
        sample_rate=sample_rate,
        n_fft=256,
        hop_length=64,
        n_mfcc=12,
        n_mels=24,
    )
    assert actual.value == pytest.approx(expected, rel=1e-12)
    assert actual.parameters["include_c0"] is False


@pytest.mark.quality
def test_speech_like_60_second_waveform_smoke_is_linear_and_finite() -> None:
    sample_rate = 16_000
    sample_count = 60 * sample_rate
    time_axis = np.arange(sample_count, dtype=np.float64) / sample_rate
    envelope = 0.4 + 0.6 * np.sin(2.0 * np.pi * 2.3 * time_axis) ** 2
    reference = (
        envelope
        * (
            0.5 * np.sin(2.0 * np.pi * 137.0 * time_axis)
            + 0.2 * np.sin(2.0 * np.pi * 274.0 * time_axis)
            + 0.1 * np.sin(2.0 * np.pi * 411.0 * time_axis)
        )
    ).astype(np.float32)
    estimate = reference + np.random.default_rng(20261008).normal(
        0.0, 0.01, size=sample_count
    ).astype(np.float32)

    started = time.perf_counter()
    result = snr(reference, estimate, sample_rate=sample_rate)
    elapsed = time.perf_counter() - started
    assert result.value is not None and np.isfinite(result.value)
    assert result.diagnostics["samples"] == sample_count
    assert elapsed < 10.0
