from __future__ import annotations

import numpy as np
import pytest

from audiosig._pitch import estimate_pitch_track_lane
from audiosig._td_psola import td_psola_prosody


def _harmonic_source(sample_rate: int = 16_000, seconds: float = 1.0) -> np.ndarray:
    time = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    return (
        0.16 * np.sin(2.0 * np.pi * 180.0 * time)
        + 0.08 * np.sin(2.0 * np.pi * 360.0 * time)
        + 0.04 * np.sin(2.0 * np.pi * 540.0 * time)
    ).astype(np.float32)


def _source_filter_fixture(sample_rate: int = 16_000, seconds: float = 1.0) -> np.ndarray:
    """Impulse train through stable resonators representing F1/F2/F3."""
    samples = round(sample_rate * seconds)
    excitation = np.zeros(samples, dtype=np.float64)
    excitation[:: round(sample_rate / 140.0)] = 1.0
    signal = excitation
    for frequency, bandwidth in ((500.0, 80.0), (1_500.0, 120.0), (2_500.0, 160.0)):
        radius = np.exp(-np.pi * bandwidth / sample_rate)
        coefficient = 2.0 * radius * np.cos(2.0 * np.pi * frequency / sample_rate)
        filtered = np.zeros(samples, dtype=np.float64)
        for index in range(samples):
            filtered[index] = signal[index]
            if index:
                filtered[index] += coefficient * filtered[index - 1]
            if index > 1:
                filtered[index] -= radius * radius * filtered[index - 2]
        signal = filtered
    return (0.3 * signal / np.max(np.abs(signal))).astype(np.float32)


def _envelope_peak_frequencies(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    size = 4096
    windowed = np.asarray(audio[-size:], dtype=np.float64) * np.hanning(size)
    spectrum = np.abs(np.fft.rfft(windowed))
    envelope = np.convolve(spectrum, np.ones(15) / 15.0, mode="same")
    peaks = []
    for low, high in ((300.0, 800.0), (1_000.0, 2_000.0), (2_100.0, 3_200.0)):
        bins = np.arange(round(low * size / sample_rate), round(high * size / sample_rate))
        peaks.append(float(bins[np.argmax(envelope[bins])] * sample_rate / size))
    return np.asarray(peaks)


@pytest.mark.quality
@pytest.mark.parametrize("semitones", (-4.0, -2.0, 2.0, 4.0))
def test_td_psola_hits_target_f0_for_harmonic_source(semitones: float) -> None:
    sample_rate = 16_000
    source = _harmonic_source(sample_rate)
    result = td_psola_prosody(source, sample_rate=sample_rate, semitones=semitones)
    track = estimate_pitch_track_lane(result, sample_rate=sample_rate)
    values = track.frequencies[track.voiced]
    assert values.size > 10
    assert float(np.median(values)) == pytest.approx(180.0 * 2.0 ** (semitones / 12.0), rel=0.05)


@pytest.mark.quality
@pytest.mark.parametrize("semitones", (-6.0, -4.0, -2.0, 2.0, 4.0, 6.0))
def test_td_psola_limits_residual_original_f0_leakage(semitones: float) -> None:
    sample_rate = 16_000
    source_frequency = 180.0
    samples = np.arange(sample_rate, dtype=np.float64)
    source = np.sin(2.0 * np.pi * source_frequency * samples / sample_rate)
    result = td_psola_prosody(source, sample_rate=sample_rate, semitones=semitones)
    window = np.hanning(result.size)
    spectrum = np.abs(np.fft.rfft(np.asarray(result, dtype=np.float64) * window))
    frequencies = np.fft.rfftfreq(result.size, 1.0 / sample_rate)

    def amplitude_at(frequency: float) -> float:
        return float(spectrum[np.argmin(np.abs(frequencies - frequency))])

    target_frequency = source_frequency * 2.0 ** (semitones / 12.0)
    leakage = amplitude_at(source_frequency) / max(amplitude_at(target_frequency), 1e-12)
    assert leakage < 0.1


@pytest.mark.quality
def test_td_psola_preserves_source_filter_formant_envelope_fixture() -> None:
    sample_rate = 16_000
    source = _source_filter_fixture(sample_rate)
    result = td_psola_prosody(source, sample_rate=sample_rate, semitones=4.0)
    source_peaks = _envelope_peak_frequencies(source, sample_rate)
    result_peaks = _envelope_peak_frequencies(result, sample_rate)
    assert np.isfinite(result_peaks).all()
    # This is a bounded regression fixture, not a guarantee for real speech;
    # the listening protocol remains the promotion gate.
    np.testing.assert_allclose(result_peaks, source_peaks, rtol=0.06, atol=25.0)


def test_td_psola_does_not_tonalize_unvoiced_noise() -> None:
    sample_rate = 16_000
    source = np.random.default_rng(123).normal(0.0, 0.1, sample_rate).astype(np.float32)
    result = td_psola_prosody(source, sample_rate=sample_rate, rate=0.8, semitones=6.0)
    track = estimate_pitch_track_lane(result, sample_rate=sample_rate)
    assert float(np.mean(track.voiced)) < 0.15
    assert np.isfinite(result).all()


def test_td_psola_mixed_transition_has_no_uninitialized_holes() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    voiced = 0.2 * np.sin(2.0 * np.pi * 200.0 * time[: sample_rate // 2])
    noise = np.random.default_rng(4).normal(0.0, 0.04, sample_rate // 2)
    source = np.concatenate([voiced, noise]).astype(np.float32)
    result = td_psola_prosody(source, sample_rate=sample_rate, rate=1.25, semitones=-4.0)
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) > 0.0
    assert np.count_nonzero(result) > result.size * 0.9


@pytest.mark.parametrize("sample_rate", (8_000, 16_000, 24_000, 48_000))
def test_td_psola_sample_rate_and_odd_length_matrix(sample_rate: int) -> None:
    length = round(sample_rate * 0.22) + 1
    time = np.arange(length, dtype=np.float64) / sample_rate
    source = (0.18 * np.sin(2.0 * np.pi * 180.0 * time)).astype(np.float32)
    result = td_psola_prosody(source, sample_rate=sample_rate, rate=0.8, semitones=3.0)

    assert result.shape == (round(length / 0.8),)
    assert np.isfinite(result).all()


def test_td_psola_reverberant_voiced_material_remains_finite() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    source = 0.16 * np.sin(2.0 * np.pi * 180.0 * time)
    delay = round(0.035 * sample_rate)
    reverberant = source.copy()
    reverberant[delay:] += 0.35 * source[:-delay]
    result = td_psola_prosody(reverberant.astype(np.float32), sample_rate=sample_rate, semitones=-4.0)

    assert np.isfinite(result).all()
    assert result.size == source.size
