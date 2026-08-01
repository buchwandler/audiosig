from __future__ import annotations

import numpy as np
import pytest

from audiosig import InvalidParameterError, generate_silence


def test_default_silence_is_float32_and_zero() -> None:
    result = generate_silence(1.0, 24_000)

    assert result.shape == (24_000,)
    assert result.dtype == np.float32
    assert np.all(result == 0)


def test_silence_supports_float64_and_floor_sample_count() -> None:
    result = generate_silence(1.25, 8_000, dtype=np.float64)

    assert result.shape == (10_000,)
    assert result.dtype == np.float64
    assert np.all(result == 0)
    assert generate_silence(0.999, 1_000).size == 999


def test_zero_and_subsample_duration_return_empty_arrays() -> None:
    zero = generate_silence(0, 24_000)
    short = generate_silence(0.5 / 24_000, 24_000, dtype=np.float64)

    assert zero.shape == (0,)
    assert short.shape == (0,)
    assert zero.dtype == np.float32
    assert short.dtype == np.float64


def test_silence_calls_are_independent_allocations() -> None:
    first = generate_silence(0.1, 100)
    second = generate_silence(0.1, 100)

    assert not np.shares_memory(first, second)
    first[...] = 1
    assert np.all(second == 0)


@pytest.mark.parametrize(
    "duration",
    [-1.0, np.nan, np.inf, -np.inf, True, False, "1.0", None, np.array([1.0])],
)
def test_invalid_duration_raises_typed_error(duration: object) -> None:
    with pytest.raises(InvalidParameterError):
        generate_silence(duration, 24_000)  # type: ignore[arg-type]


@pytest.mark.parametrize("sample_rate", [0, -1, True, False, 24_000.0, "24000"])
def test_invalid_sample_rate_raises_typed_error(sample_rate: object) -> None:
    with pytest.raises(InvalidParameterError):
        generate_silence(1.0, sample_rate)  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [np.int16, np.complex64, object, "int16"])
def test_unsupported_dtype_raises_typed_error(dtype: object) -> None:
    with pytest.raises(InvalidParameterError):
        generate_silence(1.0, 24_000, dtype=dtype)  # type: ignore[arg-type]


def test_overflow_is_rejected_before_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    allocated = False

    def fail_if_allocated(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal allocated
        allocated = True
        raise AssertionError("overflowing silence must not allocate")

    monkeypatch.setattr("audiosig.generation.np.zeros", fail_if_allocated)
    duration = float(np.iinfo(np.intp).max)

    with pytest.raises(InvalidParameterError):
        generate_silence(duration, 2)
    assert not allocated


def test_non_finite_product_is_rejected() -> None:
    with pytest.raises(InvalidParameterError):
        generate_silence(np.finfo(np.float64).max, 2)
