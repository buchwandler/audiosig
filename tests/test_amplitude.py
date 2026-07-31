from __future__ import annotations

import numpy as np
import pytest

from audiosig import InvalidParameterError, apply_gain_db, peak_normalize


def test_gain_and_clipping_preserve_dtype_and_input() -> None:
    source = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    original = source.copy()
    result = apply_gain_db(source, 6.0205999)
    assert result.dtype == source.dtype
    np.testing.assert_allclose(result, source * 2, rtol=1e-5)
    assert np.array_equal(source, original)
    np.testing.assert_array_equal(apply_gain_db(source, -np.inf), 0)
    np.testing.assert_array_equal(apply_gain_db(source, 20, clip=True), [1, -1, 1])


def test_peak_normalize_and_silence() -> None:
    source = np.array([0.2, -0.4], dtype=np.float64)
    result = peak_normalize(source, peak=0.8)
    assert result.dtype == source.dtype
    np.testing.assert_allclose(np.max(np.abs(result)), 0.8)
    silent = np.zeros(4, dtype=np.float32)
    normalized = peak_normalize(silent)
    assert normalized.dtype == silent.dtype
    np.testing.assert_array_equal(normalized, silent)
    with pytest.raises(InvalidParameterError):
        apply_gain_db(source, np.nan)
    with pytest.raises(InvalidParameterError):
        peak_normalize(source, peak=0)
    with pytest.raises(InvalidParameterError):
        peak_normalize(source, eps=-1)


def test_amplitude_operations_accept_empty_arrays() -> None:
    empty = np.empty((2, 0), dtype=np.float32)
    gain = apply_gain_db(empty, 6.0)
    normalized = peak_normalize(empty)
    assert gain.shape == normalized.shape == empty.shape
    assert gain.dtype == normalized.dtype == empty.dtype
    assert not np.shares_memory(gain, empty)
    assert not np.shares_memory(normalized, empty)
    with pytest.raises(InvalidParameterError):
        apply_gain_db(empty, np.nan)
    with pytest.raises(InvalidParameterError):
        peak_normalize(empty, peak=0)
