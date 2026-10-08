from __future__ import annotations

import numpy as np
import pytest

from audiosig.intelligibility import estoi, stoi


@pytest.mark.quality
def test_intelligibility_scores_degrade_with_added_noise() -> None:
    sample_rate = 10_000
    time_axis = np.arange(40_000, dtype=np.float64) / sample_rate
    f0 = 150.0 + 12.0 * np.sin(2.0 * np.pi * 0.4 * time_axis)
    phase = 2.0 * np.pi * np.cumsum(f0) / sample_rate
    envelope = 0.35 + 0.65 * np.maximum(0.0, np.sin(2.0 * np.pi * 2.2 * time_axis)) ** 0.7
    reference = envelope * (
        0.5 * np.sin(phase) + 0.25 * np.sin(2.0 * phase) + 0.12 * np.sin(3.0 * phase)
    )
    rng = np.random.default_rng(2026)
    lightly_degraded = reference + 0.02 * rng.normal(size=reference.size)
    heavily_degraded = reference + 0.25 * rng.normal(size=reference.size)

    stoi_light = stoi(reference, lightly_degraded, sample_rate=sample_rate)
    stoi_heavy = stoi(reference, heavily_degraded, sample_rate=sample_rate)
    estoi_light = estoi(reference, lightly_degraded, sample_rate=sample_rate)
    estoi_heavy = estoi(reference, heavily_degraded, sample_rate=sample_rate)
    assert stoi_light.value is not None and stoi_heavy.value is not None
    assert estoi_light.value is not None and estoi_heavy.value is not None
    assert stoi_light.value > stoi_heavy.value
    assert estoi_light.value > estoi_heavy.value
