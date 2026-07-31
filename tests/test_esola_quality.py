from __future__ import annotations

import numpy as np

from audiosig import time_stretch


def dominant_frequency(signal: np.ndarray, sample_rate: int) -> float:
    window = np.hanning(signal.size)
    frequencies = np.fft.rfftfreq(signal.size, 1.0 / sample_rate)
    return float(frequencies[np.argmax(np.abs(np.fft.rfft(signal * window)))])


def test_esola_keeps_steady_voiced_frequency_in_moderate_range() -> None:
    sample_rate = 16_000
    source = np.sin(2.0 * np.pi * 220.0 * np.arange(sample_rate) / sample_rate).astype(np.float64)
    for rate in (0.8, 1.2, 1.5, 2.0):
        stretched = time_stretch(source, rate, sample_rate=sample_rate, method="esola")
        assert abs(dominant_frequency(stretched, sample_rate) - 220.0) < 12.0
        assert np.max(np.abs(stretched)) < 1.2


def test_esola_batch_lane_has_no_lane_count_dependent_gain() -> None:
    sample_rate = 16_000
    source = np.sin(2.0 * np.pi * 180.0 * np.arange(3200) / sample_rate).astype(np.float64)
    standalone = time_stretch(source, 1.25, sample_rate=sample_rate, method="esola")
    batch = time_stretch(np.stack([source, source]), 1.25, sample_rate=sample_rate, method="esola")
    np.testing.assert_array_equal(batch[0], standalone)
    np.testing.assert_array_equal(batch[1], standalone)
