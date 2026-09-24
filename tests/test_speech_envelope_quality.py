from __future__ import annotations

import numpy as np

from audiosig import apply_speech_effects_envelope


def _voiced_tone(sample_rate: int = 16_000, duration: float = 1.2) -> np.ndarray:
    time = np.arange(round(sample_rate * duration), dtype=np.float64) / sample_rate
    signal = (
        0.2 * np.sin(2.0 * np.pi * 180.0 * time)
        + 0.1 * np.sin(2.0 * np.pi * 360.0 * time)
        + 0.05 * np.sin(2.0 * np.pi * 540.0 * time)
    )
    return signal.astype(np.float32)


def test_control_point_jumps_stay_within_local_difference_distribution() -> None:
    sample_rate = 16_000
    result = apply_speech_effects_envelope(
        _voiced_tone(sample_rate),
        sample_rate=sample_rate,
        rate_points=((0.0, 1.0), (0.45, 0.85)),
        pitch_points=((0.0, 0.0), (0.3, 2.0)),
    )
    for point in (0.3, 0.45):
        boundary = round(point * sample_rate)
        local = result[max(0, boundary - 400) : boundary + 400]
        differences = np.abs(np.diff(local))
        typical_jump = float(np.percentile(differences, 95.0))
        boundary_jump = abs(float(result[boundary] - result[boundary - 1]))
        assert boundary_jump <= 8.0 * typical_jump


def test_automation_knots_do_not_create_high_frequency_bursts() -> None:
    sample_rate = 16_000
    result = apply_speech_effects_envelope(
        _voiced_tone(sample_rate),
        sample_rate=sample_rate,
        rate_points=((0.0, 1.0), (0.45, 0.85)),
        pitch_points=((0.0, 0.0), (0.3, 2.0)),
    )
    window_length = 640
    window = np.hanning(window_length)
    frequencies = np.fft.rfftfreq(window_length, d=1.0 / sample_rate)
    high_frequency_bins = frequencies >= 3_000.0
    for point in (0.3, 0.45):
        boundary = round(point * sample_rate)
        frame = np.asarray(result[boundary - window_length // 2 : boundary + window_length // 2])
        power = np.abs(np.fft.rfft(frame * window)) ** 2
        high_frequency_ratio = float(np.sum(power[high_frequency_bins]) / np.sum(power))
        assert high_frequency_ratio < 0.05
