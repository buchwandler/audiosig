"""Independent numerical helpers for quality-control tests.

These helpers are deliberately simple and avoid importing anything beyond
NumPy. They are used by multiple test modules for frequency estimation,
amplitude measurement, RMS calculation, and waveform correlation.
"""

from __future__ import annotations

import numpy as np


def dominant_frequency(signal: np.ndarray, sample_rate: int) -> float:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    window = np.hanning(values.size)
    spectrum = np.abs(np.fft.rfft(values * window))
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sample_rate)
    return float(frequencies[int(np.argmax(spectrum))])


def scalar_rms(signal: np.ndarray) -> float:
    values = np.asarray(signal, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    if left64.shape != right64.shape or left64.ndim != 1:
        raise ValueError("inputs must be equal-length one-dimensional arrays")
    denominator = np.linalg.norm(left64) * np.linalg.norm(right64)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left64, right64) / denominator)


def tone_amplitude(signal: np.ndarray, sample_rate: int, frequency: float) -> float:
    values = np.asarray(signal, dtype=np.float64)
    time = np.arange(values.size, dtype=np.float64) / sample_rate
    cosine = np.cos(2.0 * np.pi * frequency * time)
    sine = np.sin(2.0 * np.pi * frequency * time)
    scale = 2.0 / max(1, values.size)
    return float(scale * np.hypot(np.dot(values, cosine), np.dot(values, sine)))
