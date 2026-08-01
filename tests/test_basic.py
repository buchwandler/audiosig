from __future__ import annotations

import numpy as np
import pytest

from audiosig import AudioShapeError, InvalidParameterError
from audiosig.basic import downmix_to_mono, generate_silence


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_generate_silence_has_exact_length_zero_content_and_dtype(
    dtype: type[np.floating],
) -> None:
    result = generate_silence(0.125, 24_000, dtype=dtype)

    assert result.shape == (3_000,)
    assert result.dtype == np.dtype(dtype)
    assert result.flags.writeable and result.flags.owndata
    assert np.count_nonzero(result) == 0


def test_generate_silence_truncates_and_zero_duration_is_empty() -> None:
    assert generate_silence(0.999, 1_000).shape == (999,)
    empty = generate_silence(0, 24_000)
    assert empty.shape == (0,)
    assert empty.dtype == np.float32
    assert empty.flags.owndata


def test_generate_silence_allocations_are_independent() -> None:
    first = generate_silence(0.01, 1_000)
    second = generate_silence(0.01, 1_000)

    first[...] = 1
    assert np.count_nonzero(second) == 0
    assert not np.shares_memory(first, second)


@pytest.mark.parametrize(
    "duration",
    [-1.0, np.nan, np.inf, -np.inf, True, False, "1.0", None, np.array([1.0])],
)
def test_generate_silence_rejects_invalid_duration(duration: object) -> None:
    with pytest.raises(InvalidParameterError, match="duration"):
        generate_silence(duration, 24_000)  # type: ignore[arg-type]


@pytest.mark.parametrize("sample_rate", [0, -1, True, False, 24_000.0, "24000"])
def test_generate_silence_rejects_invalid_sample_rate(sample_rate: object) -> None:
    with pytest.raises(InvalidParameterError, match="sample_rate"):
        generate_silence(1.0, sample_rate)  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [np.int16, np.bool_, np.complex64, object, "int16"])
def test_generate_silence_rejects_non_floating_dtype(dtype: object) -> None:
    with pytest.raises(InvalidParameterError, match="dtype"):
        generate_silence(1.0, 24_000, dtype=dtype)  # type: ignore[arg-type]


def test_generate_silence_rejects_oversized_length_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocated = False

    def fail_if_allocated(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal allocated
        allocated = True
        raise AssertionError("oversized silence must not allocate")

    monkeypatch.setattr("audiosig.basic.np.zeros", fail_if_allocated)
    with pytest.raises(InvalidParameterError, match="fit in an array"):
        generate_silence(float(np.iinfo(np.intp).max), 2)
    assert not allocated


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_downmix_mono_is_an_independent_contiguous_copy(dtype: type[np.floating]) -> None:
    source = np.arange(8, dtype=dtype)[::2]

    result = downmix_to_mono(source)

    np.testing.assert_array_equal(result, source)
    assert result.dtype == source.dtype
    assert result.flags.c_contiguous and result.flags.owndata
    assert not np.shares_memory(result, source)
    result[0] = -1
    assert source[0] == 0


@pytest.mark.parametrize("channel_axis", [-1, 1])
def test_downmix_frames_first_averages_channels_in_source_dtype(channel_axis: int) -> None:
    source = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)
    result = downmix_to_mono(source, channel_axis=channel_axis)

    np.testing.assert_array_equal(result, np.array([0.0, 0.5], dtype=np.float32))
    assert result.shape == (2,)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous


def test_downmix_channel_first_and_one_channel_inputs() -> None:
    source = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float64)

    result = downmix_to_mono(source, channel_axis=0)
    one_channel = downmix_to_mono(source[:1], channel_axis=0)

    np.testing.assert_array_equal(result, np.array([2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(one_channel, source[0])
    assert result.dtype == one_channel.dtype == np.float64


def test_downmix_empty_frames_are_empty_and_empty_channels_are_invalid() -> None:
    empty_frames_last = downmix_to_mono(np.empty((0, 2), dtype=np.float32))
    empty_frames_first = downmix_to_mono(np.empty((2, 0), dtype=np.float32), channel_axis=0)

    assert empty_frames_last.shape == empty_frames_first.shape == (0,)
    assert empty_frames_last.dtype == empty_frames_first.dtype == np.float32
    with pytest.raises(AudioShapeError, match="channel"):
        downmix_to_mono(np.empty((2, 0), dtype=np.float32))


@pytest.mark.parametrize(
    "audio",
    [
        [1.0, 2.0],
        np.array(1.0, dtype=np.float32),
        np.ones((2, 2, 2), dtype=np.float32),
        np.ones(4, dtype=np.int16),
        np.ones(4, dtype=np.bool_),
        np.ones(4, dtype=np.complex64),
        np.array([1.0, np.nan], dtype=np.float32),
        np.array([1.0, np.inf], dtype=np.float64),
    ],
)
def test_downmix_rejects_invalid_shape_dtype_and_values(audio: object) -> None:
    with pytest.raises(AudioShapeError):
        downmix_to_mono(audio)  # type: ignore[arg-type]


@pytest.mark.parametrize("axis", [2, -3, 1.5, "0"])
def test_downmix_rejects_invalid_channel_axis(axis: object) -> None:
    with pytest.raises(AudioShapeError, match="axis"):
        downmix_to_mono(np.ones((2, 4), dtype=np.float32), channel_axis=axis)  # type: ignore[arg-type]


def test_downmix_does_not_mutate_source_or_share_result_storage() -> None:
    source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    original = source.copy()
    result = downmix_to_mono(source)

    np.testing.assert_array_equal(source, original)
    result[0] = 9.0
    np.testing.assert_array_equal(source, original)
