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
