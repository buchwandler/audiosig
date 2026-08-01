from __future__ import annotations

import numpy as np
import pytest

from audiosig import AudioShapeError, downmix_to_mono


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_one_dimensional_audio_is_copied_with_dtype(dtype: type[np.floating]) -> None:
    source = np.arange(8, dtype=dtype)

    result = downmix_to_mono(source)

    np.testing.assert_array_equal(result, source)
    assert result.dtype == source.dtype
    assert not np.shares_memory(result, source)
    result[0] = -1
    assert source[0] == 0


def test_channels_first_stereo_is_averaged() -> None:
    source = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32)

    result = downmix_to_mono(source)

    np.testing.assert_array_equal(result, np.array([2, 3, 4], dtype=np.float32))


def test_frames_first_stereo_supports_negative_axis_and_matches_ttsforge() -> None:
    source = np.column_stack([np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32) + 2])
    original = source.copy()

    result = downmix_to_mono(source, channel_axis=-1)
    expected = source.mean(axis=1, dtype=np.float32)

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(source, original)


def test_one_channel_and_three_channel_inputs_reduce_only_channel_axis() -> None:
    one_channel = np.arange(5, dtype=np.float64).reshape(1, 5)
    three_channel = np.arange(24, dtype=np.float64).reshape(3, 2, 4)

    one_result = downmix_to_mono(one_channel)
    three_result = downmix_to_mono(three_channel, channel_axis=0)

    np.testing.assert_array_equal(one_result, one_channel[0])
    np.testing.assert_array_equal(three_result, three_channel.mean(axis=0, dtype=np.float64))
    assert one_result.dtype == three_result.dtype == np.float64


def test_batched_channels_preserve_remaining_axes() -> None:
    source = np.stack(
        [
            np.stack([np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)]),
            np.stack([np.full(4, 2, dtype=np.float32), np.full(4, 4, dtype=np.float32)]),
        ]
    )

    result = downmix_to_mono(source, channel_axis=1)

    assert result.shape == (2, 4)
    np.testing.assert_array_equal(result, np.array([[0.5] * 4, [3.0] * 4], dtype=np.float32))


def test_empty_mono_and_sample_dimensions_are_supported() -> None:
    empty_mono = downmix_to_mono(np.empty(0, dtype=np.float32))
    empty_frames = downmix_to_mono(np.empty((0, 2), dtype=np.float32), channel_axis=1)
    empty_samples = downmix_to_mono(np.empty((2, 0), dtype=np.float32), channel_axis=0)

    assert empty_mono.shape == (0,)
    assert empty_frames.shape == (0,)
    assert empty_samples.shape == (0,)
    assert empty_frames.dtype == empty_samples.dtype == np.float32


def test_empty_channel_axis_is_rejected() -> None:
    with pytest.raises(AudioShapeError):
        downmix_to_mono(np.empty((100, 0), dtype=np.float32), channel_axis=1)
    with pytest.raises(AudioShapeError):
        downmix_to_mono(np.empty((0, 100), dtype=np.float32), channel_axis=0)


@pytest.mark.parametrize(
    "audio",
    [
        [1.0, 2.0],
        np.array(1.0, dtype=np.float32),
        np.ones(4, dtype=np.int16),
        np.ones(4, dtype=np.complex64),
        np.array([1.0, np.nan], dtype=np.float32),
        np.array([1.0, np.inf], dtype=np.float64),
    ],
)
def test_invalid_audio_raises_typed_shape_error(audio: object) -> None:
    with pytest.raises(AudioShapeError):
        downmix_to_mono(audio)  # type: ignore[arg-type]


@pytest.mark.parametrize("axis", [2, -3, 1.5, "0"])
def test_invalid_channel_axis_raises_typed_shape_error(axis: object) -> None:
    with pytest.raises(AudioShapeError):
        downmix_to_mono(np.ones((2, 4), dtype=np.float32), channel_axis=axis)  # type: ignore[arg-type]


def test_result_does_not_share_storage_with_input() -> None:
    source = np.ones((1, 4), dtype=np.float32)
    result = downmix_to_mono(source)

    assert not np.shares_memory(result, source)
    result[0] = 7
    assert source[0, 0] == 1
