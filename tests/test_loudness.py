"""Tests for AudioSig loudness and peak measurement."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from audiosig import (
    AudioShapeError,
    InvalidParameterError,
    LoudnessMetrics,
    analyze_loudness,
    apply_gain_db,
    integrated_loudness,
    measure_loudness,
    resample,
    sample_peak_dbfs,
    true_peak_dbtp,
)

LOUDNESS_TOLERANCE_LU = 0.05
PEAK_TOLERANCE_DB = 0.02


def _sine(duration: float, sample_rate: int = 24_000, frequency: float = 440.0) -> np.ndarray:
    samples = int(duration * sample_rate)
    time = np.arange(samples, dtype=np.float64) / sample_rate
    return (0.1 * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


@pytest.mark.parametrize("sample_rate", [24_000, 44_100, 48_000])
def test_prefix_block_energy_matches_overlapping_reference(sample_rate: int) -> None:
    from audiosig.loudness import _block_mean_squares

    rng = np.random.default_rng(20260928)
    block_size = round(0.4 * sample_rate)
    hop_size = round(0.1 * sample_rate)
    weighted = rng.normal(size=block_size * 2 + hop_size + 1)
    expected = np.asarray(
        [
            np.mean(weighted[start : start + block_size] ** 2, dtype=np.float64)
            for start in range(0, weighted.size - block_size + 1, hop_size)
        ]
    )

    np.testing.assert_allclose(
        _block_mean_squares(weighted, sample_rate), expected, rtol=1e-12, atol=1e-15
    )
    assert _block_mean_squares(weighted[: block_size - 1], sample_rate).size == 0


@pytest.mark.parametrize("sample_rate", [24_000, 44_100, 48_000])
def test_reusable_analysis_matches_fresh_measurements_after_gain(sample_rate: int) -> None:
    rng = np.random.default_rng(sample_rate)
    audio = (rng.normal(size=sample_rate * 2) * 0.03).astype(np.float32)
    analysis = analyze_loudness(audio, sample_rate=sample_rate)

    for gain_db in (-8.0, 0.0, 5.0):
        cached = analysis.metrics_after_gain(gain_db)
        reference = measure_loudness(apply_gain_db(audio, gain_db), sample_rate=sample_rate)
        assert cached.integrated_lufs == pytest.approx(
            reference.integrated_lufs, abs=LOUDNESS_TOLERANCE_LU
        )
        assert cached.sample_peak_dbfs == pytest.approx(
            reference.sample_peak_dbfs, abs=PEAK_TOLERANCE_DB
        )
        assert cached.true_peak_dbtp == pytest.approx(
            reference.true_peak_dbtp, abs=PEAK_TOLERANCE_DB
        )


def test_cached_analysis_reapplies_absolute_loudness_gate_after_gain() -> None:
    sample_rate = 24_000
    times = np.arange(sample_rate * 4, dtype=np.float64) / sample_rate
    audio = np.empty_like(times)
    audio[: sample_rate * 2] = 0.001 * np.sin(2.0 * np.pi * 1_000.0 * times[: sample_rate * 2])
    audio[sample_rate * 2 :] = 0.0004 * np.sin(2.0 * np.pi * 1_000.0 * times[sample_rate * 2 :])
    audio = audio.astype(np.float32)

    analysis = analyze_loudness(audio, sample_rate=sample_rate, sample_peak=False, true_peak=False)
    cached = analysis.metrics_after_gain(10.0).integrated_lufs
    reference = integrated_loudness(apply_gain_db(audio, 10.0), sample_rate=sample_rate)

    assert cached == pytest.approx(reference, abs=LOUDNESS_TOLERANCE_LU)
    assert analysis.integrated_lufs is not None
    assert abs(cached - (analysis.integrated_lufs + 10.0)) > 0.1


def test_selective_loudness_analysis_skips_unrequested_metrics() -> None:
    result = analyze_loudness(
        _sine(0.5),
        sample_rate=24_000,
        integrated=False,
        sample_peak=False,
        true_peak=False,
    )

    assert result.block_mean_squares is None
    assert result.metrics() == LoudnessMetrics(None, None, None)
    assert result.metrics_after_gain(3.0) == LoudnessMetrics(None, None, None)


def test_silence_returns_negative_infinity() -> None:
    audio = np.zeros(24_000, dtype=np.float32)

    metrics = measure_loudness(audio, sample_rate=24_000)

    assert metrics == LoudnessMetrics(-math.inf, -math.inf, -math.inf)
    assert integrated_loudness(audio, sample_rate=24_000) == -math.inf
    assert sample_peak_dbfs(audio) == -math.inf
    assert true_peak_dbtp(audio, sample_rate=24_000) == -math.inf


def test_integrated_loudness_has_expected_gain_relationship() -> None:
    audio = _sine(2.0)
    before = integrated_loudness(audio, sample_rate=24_000)
    after = integrated_loudness(apply_gain_db(audio, 3.0), sample_rate=24_000)
    quieter = integrated_loudness(apply_gain_db(audio, -6.0), sample_rate=24_000)

    assert after - before == pytest.approx(3.0, abs=LOUDNESS_TOLERANCE_LU)
    assert quieter - before == pytest.approx(-6.0, abs=LOUDNESS_TOLERANCE_LU)


def test_sample_peak_has_expected_gain_relationship() -> None:
    audio = _sine(1.0)

    assert sample_peak_dbfs(apply_gain_db(audio, 6.0)) - sample_peak_dbfs(audio) == pytest.approx(
        6.0, abs=PEAK_TOLERANCE_DB
    )


def test_true_peak_is_not_below_sample_peak() -> None:
    audio = _sine(1.0, frequency=7_000.0)

    sample_peak = sample_peak_dbfs(audio)
    true_peak = true_peak_dbtp(audio, sample_rate=24_000)

    assert true_peak >= sample_peak - PEAK_TOLERANCE_DB
    assert true_peak > sample_peak + 0.01


def test_all_primary_apis_support_common_sample_rates() -> None:
    for sample_rate in (24_000, 44_100, 48_000):
        audio = _sine(0.8, sample_rate=sample_rate)
        metrics = measure_loudness(audio, sample_rate=sample_rate)

        assert math.isfinite(metrics.integrated_lufs)
        assert math.isfinite(metrics.sample_peak_dbfs)
        assert math.isfinite(metrics.true_peak_dbtp)


def test_short_input_behavior_is_deterministic() -> None:
    for duration in (1 / 24_000, 0.010, 0.100, 0.399):
        assert integrated_loudness(_sine(duration), sample_rate=24_000) == -math.inf

    assert math.isfinite(integrated_loudness(_sine(0.400), sample_rate=24_000))
    assert math.isfinite(integrated_loudness(_sine(0.401), sample_rate=24_000))


def test_float32_and_float64_measurements_are_close() -> None:
    float32_audio = _sine(1.0)
    float64_audio = float32_audio.astype(np.float64)

    assert integrated_loudness(float32_audio, sample_rate=24_000) == pytest.approx(
        integrated_loudness(float64_audio, sample_rate=24_000), abs=LOUDNESS_TOLERANCE_LU
    )
    assert sample_peak_dbfs(float32_audio) == pytest.approx(
        sample_peak_dbfs(float64_audio), abs=PEAK_TOLERANCE_DB
    )
    assert true_peak_dbtp(float32_audio, sample_rate=24_000) == pytest.approx(
        true_peak_dbtp(float64_audio, sample_rate=24_000), abs=PEAK_TOLERANCE_DB
    )


def test_axis_zero_is_explicit_for_one_dimensional_mono() -> None:
    audio = _sine(0.5)

    assert integrated_loudness(audio, sample_rate=24_000, axis=0) == pytest.approx(
        integrated_loudness(audio, sample_rate=24_000), abs=LOUDNESS_TOLERANCE_LU
    )


def test_input_is_not_modified() -> None:
    audio = _sine(1.0)
    original = audio.copy()

    measure_loudness(audio, sample_rate=24_000)

    np.testing.assert_array_equal(audio, original)


@pytest.mark.parametrize("sample_rate", [0, -1, 24_000.0, True])
def test_invalid_sample_rate_is_rejected(sample_rate: object) -> None:
    with pytest.raises(InvalidParameterError):
        integrated_loudness(_sine(0.5), sample_rate=cast(int, sample_rate))


@pytest.mark.parametrize("oversample", [0, -1, 3.7, True])
def test_invalid_oversampling_is_rejected(oversample: object) -> None:
    with pytest.raises(InvalidParameterError):
        true_peak_dbtp(_sine(0.5), sample_rate=24_000, oversample=cast(int, oversample))


def test_empty_non_finite_and_multidimensional_inputs_are_rejected() -> None:
    with pytest.raises(AudioShapeError):
        sample_peak_dbfs(np.empty(0, dtype=np.float32))
    with pytest.raises(AudioShapeError):
        sample_peak_dbfs(np.array([0.0, np.nan], dtype=np.float32))
    with pytest.raises(AudioShapeError):
        sample_peak_dbfs(np.array([0.0, np.inf], dtype=np.float32))
    with pytest.raises(AudioShapeError):
        sample_peak_dbfs(np.zeros((2, 100), dtype=np.float32))
    with pytest.raises(AudioShapeError):
        sample_peak_dbfs(_sine(0.5), axis=1)


def test_oversample_one_matches_sample_peak() -> None:
    audio = _sine(0.5)

    assert true_peak_dbtp(audio, sample_rate=24_000, oversample=1) == pytest.approx(
        sample_peak_dbfs(audio), abs=PEAK_TOLERANCE_DB
    )


@pytest.mark.parametrize(
    ("frequency_ratio", "amplitude", "phase_degrees", "expected_dbtp"),
    [
        (1 / 4, 0.50, 0.0, -6.0),
        (1 / 4, 0.50, 45.0, -6.0),
        (1 / 6, 0.50, 60.0, -6.0),
        (1 / 8, 0.50, 67.5, -6.0),
        (1 / 4, 1.41, 45.0, 3.0),
    ],
    ids=["ebu-15", "ebu-16", "ebu-17", "ebu-18", "ebu-19"],
)
def test_ebu_3341_true_peak_cases_15_to_19(
    frequency_ratio: float,
    amplitude: float,
    phase_degrees: float,
    expected_dbtp: float,
) -> None:
    sample_rate = 48_000
    frames = sample_rate
    time = np.arange(frames, dtype=np.float64) / sample_rate
    frequency = sample_rate * frequency_ratio
    signal = amplitude * np.sin(2.0 * np.pi * frequency * time + np.deg2rad(phase_degrees))
    fade_frames = round(0.010 * sample_rate)
    fade = np.minimum(
        1.0, np.minimum(np.arange(frames) / fade_frames, np.arange(frames)[::-1] / fade_frames)
    )
    measured = true_peak_dbtp(signal * fade, sample_rate=sample_rate)

    assert expected_dbtp - 0.4 <= measured <= expected_dbtp + 0.2


@pytest.mark.parametrize("offset", range(4), ids=["ebu-20", "ebu-21", "ebu-22", "ebu-23"])
def test_ebu_3341_true_peak_cases_20_to_23(offset: int) -> None:
    sample_rate = 48_000
    high_rate = sample_rate * 4
    high_frames = high_rate
    high_sample = np.arange(high_frames, dtype=np.float64)
    low_omega = 2.0 * np.pi * (sample_rate / 6.0) / high_rate
    high_omega = 2.0 * np.pi * (sample_rate / 4.0) / high_rate
    burst_start = high_frames // 2
    burst_frames = 16
    burst_end = burst_start + burst_frames
    phase = low_omega * high_sample
    phase[burst_start:burst_end] = low_omega * burst_start + high_omega * (
        high_sample[burst_start:burst_end] - burst_start
    )
    phase[burst_end:] = (
        low_omega * burst_start
        + high_omega * burst_frames
        + low_omega * (high_sample[burst_end:] - burst_end)
    )
    amplitude = np.full(high_frames, 0.5, dtype=np.float64)
    amplitude[burst_start:burst_end] = 1.0
    signal = amplitude * np.sin(phase)
    fade_frames = round(0.002 * high_rate)
    edge = np.minimum(
        1.0,
        np.minimum(
            np.arange(high_frames) / fade_frames,
            np.arange(high_frames)[::-1] / fade_frames,
        ),
    )
    signal *= edge

    # The high-rate synthesis, anti-alias filter, and decimation follow the
    # EBU 4x construction; this general resampler is test-signal generation,
    # not the true-peak measurement path under test.
    downsampled = resample(
        signal[offset:],
        source_rate=high_rate,
        target_rate=sample_rate,
        filter_width=48,
        rolloff=0.98,
    )
    measured = true_peak_dbtp(downsampled, sample_rate=sample_rate)

    assert -0.4 <= measured <= 0.2


def test_four_times_true_peak_meter_does_not_call_general_resampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from audiosig import loudness

    def fail_resampling(*args: object, **kwargs: object) -> np.ndarray:
        pytest.fail("4x true-peak meter called the general resampler")

    monkeypatch.setattr(loudness, "resample", fail_resampling)
    assert math.isfinite(true_peak_dbtp(_sine(0.5, frequency=7_000), sample_rate=24_000))
