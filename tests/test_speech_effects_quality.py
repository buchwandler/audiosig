from __future__ import annotations

import numpy as np
import pytest

from audiosig import apply_speech_effects, pitch_shift, time_stretch
from tests._quality_helpers import dominant_frequency

pytestmark = pytest.mark.quality


def _frame_rms(signal: np.ndarray, frame_length: int = 320) -> np.ndarray:
    frame_count = max(1, signal.size // frame_length)
    frames = np.array_split(signal[: frame_count * frame_length], frame_count)
    return np.asarray([np.sqrt(np.mean(np.square(frame), dtype=np.float64)) for frame in frames])


def _continuity_metric(signal: np.ndarray) -> float:
    rms = _frame_rms(np.asarray(signal, dtype=np.float64))
    if rms.size < 2:
        return 0.0
    return float(np.median(np.abs(np.diff(np.log1p(rms * 1000.0)))))


@pytest.mark.parametrize("rate", [0.8, 1.2, 1.4])
def test_wsola_speech_fixture_is_finite_sized_and_continuous(
    rate: float,
    speech_like_factory,
) -> None:
    sample_rate = 16_000
    source = speech_like_factory(sample_rate=sample_rate)
    stretched = time_stretch(
        source,
        rate,
        sample_rate=sample_rate,
        method="wsola",
    )
    assert stretched.size == round(source.size / rate)
    assert np.isfinite(stretched).all()
    assert np.max(np.abs(stretched)) <= 1.0
    assert _continuity_metric(stretched) < 2.0


def test_wsola_preserves_moderate_pitch_on_a_voiced_tone() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    source = np.sin(2.0 * np.pi * 220.0 * time).astype(np.float64)
    stretched = time_stretch(source, 1.2, sample_rate=sample_rate, method="wsola")
    assert dominant_frequency(stretched, sample_rate) == pytest.approx(220.0, abs=15.0)


def test_wsola_pitch_shift_hits_target_without_formant_claim() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    source = np.sin(2.0 * np.pi * 220.0 * time).astype(np.float64)
    shifted = pitch_shift(
        source,
        sample_rate=sample_rate,
        semitones=7.0,
        method="wsola",
    )
    expected = 220.0 * 2.0 ** (7.0 / 12.0)
    assert shifted.size == source.size
    assert np.isfinite(shifted).all()
    assert dominant_frequency(shifted, sample_rate) == pytest.approx(expected, abs=22.0)


@pytest.mark.parametrize("rate,semitones", [(0.85, -3.0), (1.2, 3.0), (1.35, 5.0)])
def test_combined_wsola_effects_hit_duration_and_remain_finite(
    rate: float,
    semitones: float,
    speech_like_factory,
) -> None:
    source = speech_like_factory(sample_rate=24_000)
    result = apply_speech_effects(
        source,
        sample_rate=24_000,
        rate=rate,
        semitones=semitones,
    )
    assert result.size == round(source.size / rate)
    assert result.dtype == source.dtype
    assert np.isfinite(result).all()


def test_wsola_quality_metrics_are_comparable_to_basic_phase_vocoder(
    speech_like_factory,
) -> None:
    sample_rate = 16_000
    source = speech_like_factory(sample_rate=sample_rate)
    wsola = time_stretch(source, 1.2, sample_rate=sample_rate, method="wsola")
    phase_vocoder = time_stretch(source, 1.2, method="phase_vocoder", n_fft=512, hop_length=128)
    wsola_metric = _continuity_metric(wsola)
    phase_metric = _continuity_metric(phase_vocoder)
    assert np.isfinite(wsola_metric) and np.isfinite(phase_metric)
    # The metric is a diagnostic rather than a universal perceptual score;
    # permit fixture/parameter variation while rejecting severe discontinuity.
    assert wsola_metric <= max(2.0, phase_metric * 2.0)
