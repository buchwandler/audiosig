from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from audiosig import AudioShapeError, InvalidParameterError
from audiosig.amplitude import fade_edges, gain_for_level, join_with_edge_fades, soft_limit


def test_fade_edges_apply_linear_ramps_without_mutating_input() -> None:
    source = np.ones(100, dtype=np.float32)
    result = fade_edges(
        source,
        sample_rate=1_000,
        fade_in=True,
        fade_out=True,
        fade_seconds=0.01,
    )
    ramp = np.linspace(0.0, 1.0, 10)
    np.testing.assert_allclose(result[:10], ramp, atol=1e-7)
    np.testing.assert_allclose(result[-10:], ramp[::-1], atol=1e-7)
    np.testing.assert_array_equal(result[10:-10], 1.0)
    np.testing.assert_array_equal(source, 1.0)
    assert result.dtype == source.dtype
    assert not np.shares_memory(result, source)


def test_fades_respect_axis_and_skip_short_segments() -> None:
    source = np.ones((2, 100), dtype=np.float64)
    faded = fade_edges(
        source,
        sample_rate=1_000,
        fade_in=True,
        fade_out=False,
        fade_seconds=0.01,
        axis=1,
    )
    np.testing.assert_allclose(faded[:, :10], np.tile(np.linspace(0, 1, 10), (2, 1)))
    assert np.all(faded[:, 10:] == 1.0)

    short = np.ones(20, dtype=np.float32)
    unchanged = fade_edges(
        short,
        sample_rate=1_000,
        fade_in=True,
        fade_out=True,
        fade_seconds=0.01,
    )
    np.testing.assert_array_equal(unchanged, short)
    assert not np.shares_memory(unchanged, short)


def test_join_fades_only_internal_segment_edges() -> None:
    first = np.ones((2, 20), dtype=np.float32)
    second = np.full((2, 20), 0.5, dtype=np.float32)
    joined = join_with_edge_fades([first, second], sample_rate=1_000, fade_seconds=0.004, axis=1)
    assert joined.shape == (2, 40)
    np.testing.assert_array_equal(joined[:, :16], 1.0)
    np.testing.assert_allclose(joined[:, 16:20], np.tile(np.linspace(1, 0, 4), (2, 1)))
    np.testing.assert_allclose(joined[:, 20:24], np.tile(0.5 * np.linspace(0, 1, 4), (2, 1)))
    np.testing.assert_array_equal(joined[:, 24:], 0.5)
    np.testing.assert_array_equal(first, 1.0)
    with pytest.raises(AudioShapeError, match="non-empty"):
        join_with_edge_fades([], sample_rate=1_000)
    with pytest.raises(AudioShapeError, match="non-sample"):
        join_with_edge_fades([np.ones(10), np.ones((2, 10))], sample_rate=1_000)


def test_soft_limiter_and_level_gain_follow_reference_formulas() -> None:
    source = np.array([-1.0, -0.9, 0.0, 0.9, 1.0], dtype=np.float32)
    original = source.copy()
    limited = soft_limit(source)
    expected_peak = 0.9 + 0.1 * np.tanh(1.0)
    np.testing.assert_allclose(limited, [-expected_peak, -0.9, 0.0, 0.9, expected_peak], atol=1e-7)
    assert limited.dtype == source.dtype
    np.testing.assert_array_equal(source, original)
    assert gain_for_level(-20.0, -10.0) == pytest.approx(10**0.5)
    assert gain_for_level(-10.0, -20.0) == pytest.approx(10**-0.5)
    with pytest.raises(InvalidParameterError, match="knee"):
        soft_limit(source, knee=1.0)
    with pytest.raises(InvalidParameterError, match="unrepresentable"):
        gain_for_level(-1e308, 1e308)


def test_output_conditioning_validates_fades() -> None:
    source = np.ones(20, dtype=np.float32)
    with pytest.raises(InvalidParameterError, match="fade_seconds"):
        fade_edges(
            source,
            sample_rate=1_000,
            fade_in=True,
            fade_out=False,
            fade_seconds=-0.1,
        )
    with pytest.raises(InvalidParameterError, match="fade_in"):
        fade_edges(
            source,
            sample_rate=1_000,
            fade_in=cast(bool, 1),
            fade_out=False,
        )
