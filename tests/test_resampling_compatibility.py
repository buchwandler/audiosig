from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from audiosig import InvalidParameterError
from audiosig._resampling import resample_windowed_sinc


def _pinned_half_sample_reference(
    audio: np.ndarray, source_rate: float, target_rate: float, radius: int = 16
) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float64)
    length = max(1, int(np.floor(source.size * target_rate / source_rate + 0.5)))
    offsets = np.arange(1 - radius, radius + 1)
    output = np.empty(length, dtype=np.float64)
    for index in range(length):
        center = (index + 0.5) * source_rate / target_rate - 0.5
        base = int(np.floor(center))
        indices = base + offsets
        distance = center - indices
        weights = np.sinc(distance) * (0.5 + 0.5 * np.cos(np.pi * distance / radius))
        valid = (indices >= 0) & (indices < source.size)
        weights = np.where(valid, weights, 0.0)
        output[index] = np.sum(source[np.clip(indices, 0, source.size - 1)] * weights) / np.sum(
            weights
        )
    return output


@pytest.mark.parametrize(
    ("source_rate", "target_rate"),
    [(16_000, 24_000), (24_000, 16_000), (44_100, 24_000), (48_000, 24_000)],
)
def test_sopro_half_sample_resampling_matches_pinned_formula(
    source_rate: int, target_rate: int
) -> None:
    rng = np.random.default_rng(1729)
    source = rng.normal(size=97).astype(np.float32)
    actual = resample_windowed_sinc(
        source,
        source_rate=source_rate,
        target_rate=target_rate,
        alignment="half_sample",
        window="raised_cosine",
        rolloff=1.0,
        antialias=False,
        normalize_weights=True,
        boundary="zero",
        length_mode="round",
    )
    expected = _pinned_half_sample_reference(source, source_rate, target_rate)
    assert actual.dtype == source.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-6)


@pytest.mark.parametrize("length", [1, 2, 7, 31])
def test_compatibility_resampling_handles_very_short_and_constant_inputs(length: int) -> None:
    source = np.full(length, 0.375, dtype=np.float64)
    actual = resample_windowed_sinc(
        source,
        source_rate=16_000,
        target_rate=24_000,
        alignment="half_sample",
        window="raised_cosine",
        antialias=False,
        boundary="zero",
    )
    np.testing.assert_allclose(actual, 0.375, rtol=1e-12, atol=1e-12)


def test_resample_windowed_sinc_supports_arbitrary_axis_and_independent_copy() -> None:
    source = np.arange(2 * 13 * 3, dtype=np.float32).reshape(2, 13, 3)
    actual = resample_windowed_sinc(
        source,
        source_rate=16_000,
        target_rate=24_000,
        axis=1,
        alignment="half_sample",
        window="raised_cosine",
        antialias=False,
    )
    assert actual.shape == (2, 20, 3)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    for row in range(source.shape[0]):
        for channel in range(source.shape[2]):
            expected = resample_windowed_sinc(
                source[row, :, channel],
                source_rate=16_000,
                target_rate=24_000,
                alignment="half_sample",
                window="raised_cosine",
                antialias=False,
            )
            np.testing.assert_array_equal(actual[row, :, channel], expected)

    identity = resample_windowed_sinc(
        source[:, ::2, :], source_rate=16_000, target_rate=16_000, axis=1
    )
    assert identity.flags.c_contiguous
    assert not np.shares_memory(identity, source)
    np.testing.assert_array_equal(identity, source[:, ::2, :])


def test_resample_windowed_sinc_empty_input_returns_independent_copy() -> None:
    source = np.empty((2, 0), dtype=np.float32)
    result = resample_windowed_sinc(source, source_rate=16_000, target_rate=24_000)
    assert result.shape == source.shape
    assert result.flags.c_contiguous
    assert not np.shares_memory(result, source)


@pytest.mark.parametrize(
    "options",
    [
        {"radius": 0},
        {"alignment": "center"},
        {"window": "hann"},
        {"rolloff": 0.0},
        {"antialias": 1},
        {"normalize_weights": "yes"},
        {"boundary": "reflect"},
        {"length_mode": "floor"},
    ],
)
def test_resample_windowed_sinc_validates_configuration(options: dict[str, object]) -> None:
    with pytest.raises(InvalidParameterError):
        resample_windowed_sinc(
            np.ones(12, dtype=np.float32),
            source_rate=16_000,
            target_rate=24_000,
            **cast(Any, options),
        )


def test_resample_windowed_sinc_rejects_invalid_rates() -> None:
    source = np.ones(12, dtype=np.float32)
    with pytest.raises(InvalidParameterError):
        resample_windowed_sinc(source, source_rate=np.inf, target_rate=24_000)
    with pytest.raises(InvalidParameterError):
        resample_windowed_sinc(source, source_rate=16_000, target_rate=0)


def test_pinned_sopro_source_rate_resampling_golden() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "sopro_reference_golden.json"
    golden = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected = golden["tones"]["resampled_16k"]
    indices = np.arange(expected["samples"], dtype=np.float64)
    source = (
        0.08 * np.sin(2.0 * np.pi * 220.0 * indices / expected["sample_rate"])
        + 0.025 * np.sin(2.0 * np.pi * 791.0 * indices / expected["sample_rate"])
    ).astype(np.float32)
    result = resample_windowed_sinc(
        source,
        source_rate=expected["sample_rate"],
        target_rate=24_000,
        radius=16,
        alignment="half_sample",
        window="raised_cosine",
        antialias=False,
        normalize_weights=True,
        boundary="zero",
        length_mode="round",
    )
    assert result.size == expected["resampled_samples"]
    for index, value in expected["resampled"].items():
        np.testing.assert_allclose(result[int(index)], value, rtol=1e-5, atol=1e-7)
