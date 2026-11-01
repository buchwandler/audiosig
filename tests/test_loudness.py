"""Tests for AudioSig loudness and peak measurement."""

from __future__ import annotations

import math
from typing import Literal, cast

import numpy as np
import pytest

from audiosig import (
    AudioShapeError,
    InvalidParameterError,
    LoudnessMetrics,
    LoudnessProgress,
    LoudnessProgressCallback,
    LoudnessProgressPhase,
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


def _legacy_apply_biquad(
    signal: np.ndarray, coefficients: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Test-only copy of the pre-fusion reference biquad loop."""
    numerator, denominator = coefficients
    output = np.empty(signal.size, dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    for index, sample in enumerate(signal):
        current = (
            numerator[0] * sample
            + numerator[1] * x1
            + numerator[2] * x2
            - denominator[1] * y1
            - denominator[2] * y2
        )
        output[index] = current
        x2, x1 = x1, float(sample)
        y2, y1 = y1, current
    return output


def _legacy_k_weight(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    from audiosig import loudness

    high_pass = _legacy_apply_biquad(signal, loudness._high_pass(sample_rate))
    return _legacy_apply_biquad(high_pass, loudness._high_shelf(sample_rate))


@pytest.mark.parametrize("sample_rate", [24_000, 44_100, 48_000])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("kind", ["noise", "sine", "silence", "short"])
def test_fused_k_weight_matches_legacy_cascade(
    sample_rate: int, dtype: type[np.floating], kind: str
) -> None:
    from audiosig.loudness import _k_weight

    rng = np.random.default_rng(sample_rate)
    if kind == "noise":
        audio = rng.normal(0.0, 0.1, 2049).astype(dtype)
    elif kind == "sine":
        time = np.arange(2049, dtype=np.float64) / sample_rate
        audio = (0.1 * np.sin(2.0 * np.pi * 997.0 * time)).astype(dtype)
    elif kind == "silence":
        audio = np.zeros(2049, dtype=dtype)
    else:
        audio = np.asarray([0.25, -0.5, 0.125], dtype=dtype)

    actual = _k_weight(audio, sample_rate)
    expected = _legacy_k_weight(audio, sample_rate)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("chunk_size", [1, 17, 8191, 8192, 8193, 65_537])
def test_fused_k_weight_state_is_continuous_across_chunks(chunk_size: int) -> None:
    from audiosig.loudness import _k_weight, _k_weight_chunk

    rng = np.random.default_rng(3301)
    audio = rng.normal(0.0, 0.1, max(257, chunk_size + 37)).astype(np.float32)
    original = audio.copy()
    expected = _k_weight(audio, 24_000)
    state = None
    pieces = []
    for start in range(0, audio.size, chunk_size):
        weighted, state = _k_weight_chunk(audio[start : start + chunk_size], 24_000, state=state)
        pieces.append(weighted)

    np.testing.assert_array_equal(np.concatenate(pieces), expected)
    np.testing.assert_array_equal(audio, original)


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

    assert cached is not None
    assert reference is not None
    assert analysis.integrated_lufs is not None
    assert cached == pytest.approx(reference, abs=LOUDNESS_TOLERANCE_LU)
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
        assert metrics.integrated_lufs is not None
        assert metrics.sample_peak_dbfs is not None
        assert metrics.true_peak_dbtp is not None
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


@pytest.mark.parametrize("sample_rate", [24_000, 44_100, 48_000])
@pytest.mark.parametrize("chunk_offset", [-7, 3, 257])
def test_block_energy_accumulator_matches_whole_waveform(
    sample_rate: int, chunk_offset: int
) -> None:
    from audiosig.loudness import (
        _block_geometry,
        _block_mean_squares,
        _BlockEnergyAccumulator,
    )

    block_size, hop_size = _block_geometry(sample_rate)
    rng = np.random.default_rng(sample_rate + chunk_offset)
    weighted = rng.normal(size=3 * block_size + 3 * hop_size + 11)
    chunk_size = hop_size + chunk_offset
    accumulator = _BlockEnergyAccumulator(sample_rate)
    for start in range(0, weighted.size, chunk_size):
        accumulator.consume(weighted[start : start + chunk_size])

    actual = accumulator.finish()
    expected = _block_mean_squares(weighted, sample_rate)
    assert actual.size == expected.size
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-15)


@pytest.mark.parametrize("sample_rate", [24_000, 44_100, 48_000])
def test_block_energy_accumulator_preserves_complete_block_boundaries(sample_rate: int) -> None:
    from audiosig.loudness import (
        _block_geometry,
        _block_mean_squares,
        _BlockEnergyAccumulator,
    )

    block_size, hop_size = _block_geometry(sample_rate)
    rng = np.random.default_rng(sample_rate)
    for frame_count in (
        block_size - 1,
        block_size,
        block_size + 1,
        block_size + hop_size - 1,
        block_size + hop_size,
        3 * block_size + 2 * hop_size + 1,
    ):
        weighted = rng.normal(size=frame_count)
        accumulator = _BlockEnergyAccumulator(sample_rate)
        for start in range(0, frame_count, hop_size + 3):
            accumulator.consume(weighted[start : start + hop_size + 3])
        actual = accumulator.finish()
        expected = _block_mean_squares(weighted, sample_rate)
        assert actual.size == expected.size
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-15)


@pytest.mark.parametrize("chunk_size", [1, 17])
def test_block_energy_accumulator_handles_tiny_chunks(chunk_size: int) -> None:
    from audiosig.loudness import _block_mean_squares, _BlockEnergyAccumulator

    weighted = np.linspace(-1.0, 1.0, 1003, dtype=np.float64)
    accumulator = _BlockEnergyAccumulator(24_000)
    for start in range(0, weighted.size, chunk_size):
        accumulator.consume(weighted[start : start + chunk_size])

    np.testing.assert_allclose(
        accumulator.finish(), _block_mean_squares(weighted, 24_000), rtol=2e-12, atol=1e-15
    )


def test_integrated_chunk_pipeline_matches_whole_waveform_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from audiosig import loudness

    rng = np.random.default_rng(20261005)
    monkeypatch.setattr(loudness, "_LOUDNESS_CHUNK_SIZE", 997)
    for sample_rate in (24_000, 44_100, 48_000):
        audio = rng.normal(0.0, 0.05, sample_rate + 37).astype(np.float32)
        weighted = _legacy_k_weight(audio, sample_rate)
        expected = loudness._block_mean_squares(weighted, sample_rate)
        actual = loudness._integrated_block_energies(audio, sample_rate)
        assert actual.size == expected.size
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-15)
        assert loudness._integrated_loudness_from_energies(actual) == pytest.approx(
            loudness._integrated_loudness_from_energies(expected), abs=LOUDNESS_TOLERANCE_LU
        )


def test_block_energy_accumulator_handles_multiminute_synthetic_input() -> None:
    from audiosig.loudness import _block_mean_squares, _BlockEnergyAccumulator

    sample_rate = 24_000
    rng = np.random.default_rng(4460)
    weighted = rng.normal(size=120 * sample_rate)
    accumulator = _BlockEnergyAccumulator(sample_rate)
    for start in range(0, weighted.size, 131_071):
        accumulator.consume(weighted[start : start + 131_071])

    actual = accumulator.finish()
    expected = _block_mean_squares(weighted, sample_rate)
    assert actual.size == expected.size
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-15)


def test_standard_true_peak_converts_only_bounded_input_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from audiosig import loudness

    audio = np.linspace(-0.7, 0.8, 257, dtype=np.float32)
    chunk_size = 31
    history_size = loudness._TRUE_PEAK_PHASE_COEFFICIENTS.shape[0] - 1
    observed: list[tuple[np.dtype, int]] = []
    original_convolve = np.convolve

    def track_convolve(values: np.ndarray, kernel: np.ndarray, mode: Literal["full"]) -> np.ndarray:
        observed.append((values.dtype, values.size))
        return original_convolve(values, kernel, mode=mode)

    monkeypatch.setattr(loudness, "_TRUE_PEAK_CHUNK_SIZE", chunk_size)
    monkeypatch.setattr(loudness.np, "convolve", track_convolve)
    sample_peak, true_peak = loudness._stream_true_peak(audio)

    assert sample_peak == pytest.approx(0.8)
    assert true_peak >= sample_peak
    assert len(observed) > 1
    assert all(dtype == np.dtype(np.float64) for dtype, _ in observed)
    assert all(size <= chunk_size + history_size for _, size in observed)


def test_analyze_loudness_reuses_true_peak_sample_peak_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from audiosig import loudness

    audio = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    seen: list[np.ndarray] = []

    def fake_stream(
        signal: np.ndarray, *, on_progress: LoudnessProgressCallback | None = None
    ) -> tuple[float, float]:
        seen.append(signal)
        return 0.3, 0.4

    def fail_integrated(*args: object, **kwargs: object) -> np.ndarray:
        pytest.fail("selective peak analysis computed integrated loudness")

    monkeypatch.setattr(loudness, "_stream_true_peak", fake_stream)
    monkeypatch.setattr(loudness, "_integrated_block_energies", fail_integrated)
    analysis = analyze_loudness(
        audio,
        sample_rate=24_000,
        integrated=False,
        sample_peak=True,
        true_peak=True,
    )

    assert len(seen) == 1 and seen[0] is audio
    assert analysis.sample_peak_dbfs == pytest.approx(20.0 * math.log10(0.3))
    assert analysis.true_peak_dbtp == pytest.approx(20.0 * math.log10(0.4))


def test_selective_sample_peak_uses_validated_source_without_float64_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from audiosig import loudness

    audio = np.asarray([0.1, -0.6, 0.3], dtype=np.float32)
    seen: list[np.ndarray] = []
    original_peak = loudness._sample_peak_linear

    def track_peak(signal: np.ndarray) -> float:
        seen.append(signal)
        return original_peak(signal)

    monkeypatch.setattr(loudness, "_sample_peak_linear", track_peak)
    analysis = analyze_loudness(
        audio,
        sample_rate=24_000,
        integrated=False,
        sample_peak=True,
        true_peak=False,
    )

    assert len(seen) == 1 and seen[0] is audio
    assert seen[0].dtype == np.float32
    assert analysis.sample_peak_dbfs == pytest.approx(20.0 * math.log10(0.6))


def test_loudness_progress_type_is_immutable_and_public() -> None:
    progress = LoudnessProgress(
        phase="integrated_loudness",
        completed_frames=2,
        total_frames=4,
        phase_fraction=0.5,
    )

    def ignore_progress(event: LoudnessProgress) -> None:
        assert event.total_frames > 0

    callback: LoudnessProgressCallback = ignore_progress
    callback(progress)

    assert progress.phase == "integrated_loudness"
    assert LoudnessProgressPhase.__args__ == ("integrated_loudness", "sample_peak", "true_peak")
    attribute = "phase_fraction"
    with pytest.raises(AttributeError):
        setattr(progress, attribute, 1.0)


def test_integrated_and_analysis_progress_report_selected_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from audiosig import loudness

    audio = _sine(0.6)
    chunk_size = 2000
    monkeypatch.setattr(loudness, "_LOUDNESS_CHUNK_SIZE", chunk_size)
    events: list[LoudnessProgress] = []
    integrated_loudness(audio, sample_rate=24_000, on_progress=events.append)

    expected_completed = [*range(chunk_size, audio.size, chunk_size), audio.size]
    assert [event.phase for event in events] == ["integrated_loudness"] * len(expected_completed)
    assert [event.completed_frames for event in events] == expected_completed
    assert all(event.total_frames == audio.size for event in events)
    assert all(0.0 <= event.phase_fraction <= 1.0 for event in events)
    assert events[-1].phase_fraction == 1.0

    events.clear()
    analysis = analyze_loudness(
        audio,
        sample_rate=24_000,
        integrated=True,
        sample_peak=True,
        true_peak=False,
        on_progress=events.append,
    )
    phases = [event.phase for event in events]
    assert phases == ["integrated_loudness"] * len(expected_completed) + ["sample_peak"]
    assert events[-1].completed_frames == events[-1].total_frames == audio.size
    assert analysis.sample_peak_dbfs is not None


def test_true_peak_progress_tracks_fir_input_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    from audiosig import loudness

    audio = _sine(0.02)
    monkeypatch.setattr(loudness, "_TRUE_PEAK_CHUNK_SIZE", 101)
    events: list[LoudnessProgress] = []
    measured = true_peak_dbtp(audio, sample_rate=24_000, on_progress=events.append)

    assert math.isfinite(measured)
    assert [event.phase for event in events] == ["true_peak"] * 5
    assert [event.completed_frames for event in events] == [101, 202, 303, 404, audio.size]
    assert events[-1].phase_fraction == 1.0
    assert all(event.total_frames == audio.size for event in events)


@pytest.mark.parametrize("oversample", [1, 2])
def test_nonstandard_true_peak_progress_has_terminal_event(oversample: int) -> None:
    events: list[LoudnessProgress] = []
    audio = _sine(0.02)

    true_peak_dbtp(
        audio,
        sample_rate=24_000,
        oversample=oversample,
        on_progress=events.append,
    )

    assert len(events) == 1
    assert events[0].phase == "true_peak"
    assert events[0].completed_frames == events[0].total_frames == audio.size
    assert events[0].phase_fraction == 1.0


def test_measure_loudness_progress_uses_only_integrated_and_true_peak_phases() -> None:
    events: list[LoudnessProgress] = []
    audio = _sine(0.02)

    metrics = measure_loudness(audio, sample_rate=24_000, on_progress=events.append)

    assert metrics.true_peak_dbtp is not None
    assert {event.phase for event in events} == {"integrated_loudness", "true_peak"}
    for phase in ("integrated_loudness", "true_peak"):
        phase_events = [event for event in events if event.phase == phase]
        assert phase_events[-1].completed_frames == phase_events[-1].total_frames == audio.size
        assert phase_events[-1].phase_fraction == 1.0


def test_progress_callback_exceptions_propagate() -> None:
    audio = _sine(0.5)

    def fail(progress: LoudnessProgress) -> None:
        raise RuntimeError(progress.phase)

    with pytest.raises(RuntimeError, match="integrated_loudness"):
        integrated_loudness(audio, sample_rate=24_000, on_progress=fail)
    with pytest.raises(RuntimeError, match="true_peak"):
        true_peak_dbtp(audio, sample_rate=24_000, on_progress=fail)
