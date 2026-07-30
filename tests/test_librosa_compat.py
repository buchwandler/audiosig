"""AudioSig behavioral quality tests informed by public DSP invariants.

Some scenarios were selected after reviewing librosa's public test suite, but
this module independently defines AudioSig's contracts and fixtures.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

import audiosig


librosa = pytest.importorskip("librosa")
pytestmark = pytest.mark.librosa_compat


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("top_db", [20.0, 40.0, 60.0])
@pytest.mark.parametrize("frame_length,hop_length", [(128, 32), (256, 64), (513, 128)])
def test_trim_matches_librosa_for_matching_configuration(
    dtype: type[np.floating],
    top_db: float,
    frame_length: int,
    hop_length: int,
    alternating_burst_factory: Callable[..., np.ndarray],
) -> None:
    source = np.zeros(5000, dtype=dtype)
    source[1333:3666] = alternating_burst_factory(
        2333,
        amplitude=0.25,
        dtype=dtype,
    )

    actual_audio, actual_interval = audiosig.trim(
        source,
        top_db=top_db,
        ref=np.max,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
        pad_mode="constant",
    )
    expected_audio, expected_interval = librosa.effects.trim(
        source,
        top_db=top_db,
        ref=np.max,
        frame_length=frame_length,
        hop_length=hop_length,
        aggregate=np.max,
    )

    np.testing.assert_array_equal(actual_interval, expected_interval)
    np.testing.assert_array_equal(actual_audio, expected_audio)


def test_trim_float32_threshold_regression_matches_librosa() -> None:
    source = np.zeros(1000, dtype=np.float32)
    source[333:666] = 0.01

    _, actual = audiosig.trim(
        source,
        top_db=60.0,
        ref=np.max,
        frame_length=2,
        hop_length=1,
    )
    _, expected = librosa.effects.trim(
        source,
        top_db=60.0,
        ref=np.max,
        frame_length=2,
        hop_length=1,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("case", "expected_count"),
    [("active", 1), ("trailing_silence", 1), ("two_bursts", 2)],
)
def test_split_matches_librosa_with_frame_tolerance(
    case: str,
    expected_count: int,
    alternating_burst_factory: Callable[..., np.ndarray],
) -> None:
    frame_length = 256
    hop_length = 64
    source = np.zeros(8192, dtype=np.float32)

    if case == "active":
        source[:] = alternating_burst_factory(source.size)
    elif case == "trailing_silence":
        source[:5000] = alternating_burst_factory(5000)
    else:
        source[1024:2048] = alternating_burst_factory(1024)
        source[4096:5632] = alternating_burst_factory(1536)

    actual = audiosig.split(
        source,
        top_db=40.0,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    expected = librosa.effects.split(
        source,
        top_db=40.0,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    assert actual.shape[0] == expected_count
    assert expected.shape[0] == expected_count
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("frame_length,hop_length", [(128, 32), (257, 64)])
@pytest.mark.parametrize("center", [False, True])
def test_frame_rms_matches_librosa_values_and_dtype(
    dtype: type[np.floating],
    frame_length: int,
    hop_length: int,
    center: bool,
    rng: np.random.Generator,
) -> None:
    source = rng.standard_normal(4000).astype(dtype)

    actual = audiosig.frame_rms(
        source,
        frame_length=frame_length,
        hop_length=hop_length,
        center=center,
        pad_mode="constant",
        dtype=dtype,
    )
    expected = librosa.feature.rms(
        y=source,
        frame_length=frame_length,
        hop_length=hop_length,
        center=center,
        pad_mode="constant",
        dtype=dtype,
    ).squeeze(axis=0)

    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=5e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_decibel_helpers_match_librosa(dtype: type[np.floating]) -> None:
    amplitude = np.logspace(-6, 0, 128, dtype=dtype)
    power = amplitude * amplitude

    actual_amplitude = audiosig.amplitude_to_db(
        amplitude,
        ref=np.max,
        amin=1e-5,
        top_db=80.0,
    )
    expected_amplitude = librosa.amplitude_to_db(
        amplitude,
        ref=np.max,
        amin=1e-5,
        top_db=80.0,
    )
    actual_power = audiosig.power_to_db(
        power,
        ref=np.max,
        amin=1e-10,
        top_db=80.0,
    )
    expected_power = librosa.power_to_db(
        power,
        ref=np.max,
        amin=1e-10,
        top_db=80.0,
    )

    np.testing.assert_allclose(actual_amplitude, expected_amplitude, atol=2e-5)
    np.testing.assert_allclose(actual_power, expected_power, atol=2e-5)


@pytest.mark.parametrize("frames", [0, 1, 3, np.array([0, 1, 3])])
@pytest.mark.parametrize("hop_length", [64, 512])
@pytest.mark.parametrize("n_fft", [None, 256, 513])
def test_frames_to_samples_matches_librosa(
    frames: int | np.ndarray,
    hop_length: int,
    n_fft: int | None,
) -> None:
    actual = audiosig.frames_to_samples(
        frames,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    expected = librosa.frames_to_samples(
        frames,
        hop_length=hop_length,
        n_fft=n_fft,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("length", "source_rate", "target_rate"),
    [(2, 24_000, 16_000), (3, 16_000, 24_000), (101, 24_000, 16_000)],
)
def test_resample_ceil_mode_matches_librosa_length(
    length: int,
    source_rate: int,
    target_rate: int,
) -> None:
    source = np.zeros(length, dtype=np.float32)

    actual = audiosig.resample(
        source,
        source_rate=source_rate,
        target_rate=target_rate,
        length_mode="ceil",
    )
    expected = librosa.resample(
        source,
        orig_sr=source_rate,
        target_sr=target_rate,
        res_type="soxr_hq",
        fix=True,
        scale=False,
    )

    assert actual.shape == expected.shape
    assert actual.size == math.ceil(length * target_rate / source_rate)


@pytest.mark.parametrize("source_rate,target_rate,tone", [(24_000, 16_000, 1000.0)])
def test_resample_quality_is_bounded_against_soxr_reference(
    source_rate: int,
    target_rate: int,
    tone: float,
    sine_factory: Callable[..., np.ndarray],
) -> None:
    source = sine_factory(
        tone,
        sample_rate=source_rate,
        duration=1.0,
        dtype=np.float64,
    )

    actual = audiosig.resample(
        source,
        source_rate=source_rate,
        target_rate=target_rate,
        length_mode="ceil",
    )
    expected = librosa.resample(
        source,
        orig_sr=source_rate,
        target_sr=target_rate,
        res_type="soxr_hq",
        fix=True,
        scale=False,
    )

    # Compare quality statistics, not exact kernel output.
    assert actual.shape == expected.shape
    actual_rms = float(np.sqrt(np.mean(actual[256:-256] ** 2)))
    expected_rms = float(np.sqrt(np.mean(expected[256:-256] ** 2)))
    assert actual_rms == pytest.approx(expected_rms, rel=0.03, abs=0.01)
