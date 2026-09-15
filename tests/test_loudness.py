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
    apply_gain_db,
    integrated_loudness,
    measure_loudness,
    sample_peak_dbfs,
    true_peak_dbtp,
)

LOUDNESS_TOLERANCE_LU = 0.05
PEAK_TOLERANCE_DB = 0.02


def _sine(duration: float, sample_rate: int = 24_000, frequency: float = 440.0) -> np.ndarray:
    samples = int(duration * sample_rate)
    time = np.arange(samples, dtype=np.float64) / sample_rate
    return (0.1 * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


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
