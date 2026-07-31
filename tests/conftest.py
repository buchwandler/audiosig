"""Shared fixtures for AudioSig quality-control tests.

Provides deterministic generators and synthetic signal factories so every
test can build its own input without external audio fixtures.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Create an independent deterministic generator for each test."""
    return np.random.default_rng(20260730)


@pytest.fixture
def sine_factory() -> Callable[..., np.ndarray]:
    """Generate an exact-length sine wave without external dependencies."""

    def make(
        frequency: float,
        *,
        sample_rate: int = 16_000,
        length: int | None = None,
        duration: float = 1.0,
        amplitude: float = 1.0,
        phase: float = 0.0,
        dtype: type[np.floating] = np.float64,
    ) -> np.ndarray:
        sample_count = length if length is not None else round(sample_rate * duration)
        time = np.arange(sample_count, dtype=np.float64) / sample_rate
        signal = amplitude * np.sin(2.0 * np.pi * frequency * time + phase)
        return signal.astype(dtype)

    return make


@pytest.fixture
def alternating_burst_factory() -> Callable[..., np.ndarray]:
    """Generate a high-energy sign-alternating burst."""

    def make(
        length: int,
        *,
        amplitude: float = 0.5,
        dtype: type[np.floating] = np.float32,
    ) -> np.ndarray:
        values = np.ones(length, dtype=np.float64)
        values[1::2] = -1.0
        return (amplitude * values).astype(dtype)

    return make


@pytest.fixture
def speech_like_factory() -> Callable[..., np.ndarray]:
    """Build a deterministic source-filter-style speech surrogate.

    The fixture combines changing-F0 harmonics, broad formant emphasis,
    amplitude modulation, breath noise, a plosive burst, and a partial tail.
    It is intentionally synthetic and carries no corpus licensing burden.
    """

    def make(
        *,
        sample_rate: int = 16_000,
        length: int | None = None,
        dtype: type[np.floating] = np.float64,
    ) -> np.ndarray:
        sample_count = length if length is not None else round(sample_rate * 1.1)
        time = np.arange(sample_count, dtype=np.float64) / sample_rate
        signal = np.zeros(sample_count, dtype=np.float64)
        f0 = 125.0 + 35.0 * np.sin(2.0 * np.pi * 0.7 * time)
        phase = 2.0 * np.pi * np.cumsum(f0) / sample_rate
        voiced = (time >= 0.12) & (time < 0.78)
        envelope = np.clip(np.sin(np.pi * (time - 0.12) / 0.66), 0.0, 1.0)
        for harmonic in range(1, 16):
            frequency = harmonic * f0
            formant_weight = (
                1.2 * np.exp(-0.5 * ((frequency - 500.0) / 170.0) ** 2)
                + 0.9 * np.exp(-0.5 * ((frequency - 1500.0) / 260.0) ** 2)
                + 0.6 * np.exp(-0.5 * ((frequency - 2500.0) / 420.0) ** 2)
            )
            signal += (formant_weight / harmonic) * np.sin(harmonic * phase)
        signal *= voiced * envelope

        rng = np.random.default_rng(20260730)
        breath = 0.025 * rng.standard_normal(sample_count)
        breath_mask = (time >= 0.78) & (time < 0.98)
        signal += breath * breath_mask
        plosive_center = 0.095
        signal += 0.7 * np.exp(-0.5 * ((time - plosive_center) / 0.0025) ** 2)
        signal += 0.35 * np.exp(-0.5 * ((time - 0.99) / 0.0015) ** 2)
        signal *= 0.25 / max(np.max(np.abs(signal)), np.finfo(np.float64).eps)
        return signal.astype(dtype)

    return make
